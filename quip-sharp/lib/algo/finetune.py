"""
Utilities for fine tuning
"""
import copy
from operator import attrgetter
import time

import glog
import torch
from torch import nn

from lib import codebook, utils
from lib.linear import *

from . import quip


def finetune_decoder_layer(layer, name, device, train_dl, valid_dl, args):
    layer = layer.to(device)

    susv_params, params = utils.extract_susv_params(layer)
    optim = utils.get_susv_adam(susv_params, params, args)

    best_loss = utils.calculate_mse_loss(layer, valid_dl, device)
    best_sd = copy.deepcopy(layer.state_dict())
    glog.info(f'layer {name} initial loss {best_loss}')
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    worse_ct = 0
    position_ids = None

    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            if position_ids is None:
                position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):
                output = layer(source.to(device), position_ids=position_ids)[0]
                loss = nn.MSELoss()(output, targets.to(device))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_mse_loss(layer, valid_dl, device)
            if test_loss < best_loss:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(layer.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'layer {name} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    del optim, train_dl, valid_dl

    layer.load_state_dict(best_sd)
    utils.clean()
    layer = layer.cpu()

def linear_from_hatw(saved_linear):
    hatW = saved_linear['hatW'].float()
    shapes = saved_linear['shapes']
    scales = saved_linear['scales']
 
    in_dim = shapes[0][1]
    out_dims = [s[0] for s in shapes]
    total_out = sum(out_dims)
    cur = 0
    pieces = []
    for shape, scale in zip(shapes, scales):
        out_dim = shape[0]
        piece = hatW[cur:cur + out_dim] * scale
        pieces.append(piece)
        cur += out_dim
    full_weight = torch.cat(pieces, dim=0)
    assert full_weight.shape == (total_out, in_dim), \
        f"shape mismatch: full_weight {full_weight.shape} vs expected {(total_out, in_dim)}"
 
    if saved_linear['fused']:
        dense_linear = FusedLinear(-1, out_dims, in_dim, total_out, bias=False)
    else:
        dense_linear = nn.Linear(in_dim, total_out, bias=False)
 
    with torch.no_grad():
        dense_linear.weight.copy_(full_weight)
 
    return dense_linear

def quantize_finetune_decoder_layer(mixed_layer, quant_order, idx, cb, args,
                                    device, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(args.num_cpu_threads)

    codebook_id = codebook.get_id(args.codebook)

    mixed_layer = mixed_layer.float()

    train_dl, valid_dl = utils.split_data(pre_orig_emb, orig_emb, args)

    shared_args = (cb.codesz, cb.packsz, cb.pack_out, str(cb.idx_dtype),
                   cb.version)
    shared_kwargs = {
        'rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'resid_scale_override': args.resid_scale_override,
        'bias': False,
        'train_mode': args.ft_train_mode,
        'grad_ckpt': args.ft_grad_ckpt,
    }

    for quant_i, (linear_attr, name) in enumerate(quant_order):
        orig_linear = attrgetter(linear_attr)(mixed_layer)
        if orig_linear.bias is not None:
            # not implemented yet
            raise Exception
        save_path = f'{args.save_path}/{idx}_{name}.pt'
        hessian_path = f'{args.hessian_path}/{idx}_{name}.pt'
        with torch.no_grad():
            if isinstance(orig_linear, FusedLinear):
                weights = torch.split(orig_linear.weight,
                                      orig_linear.fuse_sizes, 0)
            else:
                weights = [orig_linear.weight]
            quip.quantize_linear(weights, save_path, hessian_path, cb, args,
                                 device)
            saved_linear = torch.load(save_path,
                                      map_location=torch.device('cpu'))
            dense_linear = linear_from_hatw(saved_linear)
        split_attr = linear_attr.split('.')
        setattr(
            attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
            dense_linear)
    #         if saved_linear['fused']:
    #             quant_linear = FusedQuantizedLinear(
    #                 -1, [_[0] for _ in saved_linear['shapes']],
    #                 saved_linear['shapes'][0][1],
    #                 sum([_[0] for _ in saved_linear['shapes']]), *shared_args,
    #                 **shared_kwargs)
    #             for i in range(len(saved_linear['scales'])):
    #                 quant_linear.fuse_scales[i].copy_(
    #                     saved_linear['scales'][i])
    #         else:
    #             quant_linear = QuantizedLinear(saved_linear['shapes'][0][1],
    #                                            saved_linear['shapes'][0][0],
    #                                            *shared_args, **shared_kwargs)
    #         utils.unpack_quip(quant_linear, saved_linear, codebook_id,
    #                           cb.codesz)
    #     quant_linear.SU = nn.Parameter(quant_linear.SU.float(),
    #                                    requires_grad=True)
    #     quant_linear.SV = nn.Parameter(quant_linear.SV.float(),
    #                                    requires_grad=True)
    #     split_attr = linear_attr.split('.')
    #     setattr(
    #         attrgetter('.'.join(split_attr[:-1]))(mixed_layer), split_attr[-1],
    #         quant_linear)
    #     if quant_i < len(quant_order) - 1:
    #         finetune_decoder_layer(mixed_layer, f'{idx}_{name}', device,
    #                                train_dl, valid_dl, args)

    if args.sparse_ft_epochs > 0:
        sparse_state = {}
        for linear_attr, name in quant_order:
            save_path = f'{args.save_path}/{idx}_{name}.pt'
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
            state = build_clique_state(saved_linear, device)
            state['codebook'] = cb.to(device)
            sparse_state[name] = state
 
        mixed_layer = sparse_finetune_layer(mixed_layer, quant_order, sparse_state, device, train_dl, valid_dl, args)

        for linear_attr, name in quant_order:
            save_path = f'{args.save_path}/{idx}_{name}.pt'
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
            state = sparse_state[name]
            from lib.algo.quip import incoherence_process
            new_hatW = incoherence_process(
                state['hatWr'], state['SU'].to(device), state['SV'].to(device),
                state.get('scaleWH'), args)
            orig_m, orig_n = state['orig_shape']
            saved_linear['hatW'] = new_hatW[:orig_m, :orig_n].half().cpu()
            saved_linear['hatWr'] = state['hatWr'].cpu()
            # saved_linear['Qidxs_blocks'] = state['Qidxs_blocks'].cpu()
            saved_linear['Qidxs'] = state['Qidxs'].cpu()
            torch.save(saved_linear, save_path)

    # with torch.no_grad():
    #     utils.clean()
    #     for i, (linear_attr, name) in enumerate(quant_order):
    #         utils.save_susv(
    #             attrgetter(linear_attr)(mixed_layer),
    #             f'{args.save_path}/{idx}_{name}.pt')

    mixed_layer = mixed_layer.to(torch.float16).cpu()
    utils.clean()
    torch.set_grad_enabled(False)

def finetune_susv_e2e(model, orig_logits, emb, position_ids, attention_mask,
                      save_fn, args):

    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear) or isinstance(
                module, FusedQuantizedLinear):
            module.SU = nn.Parameter(module.SU.float(), requires_grad=True)
            module.SV = nn.Parameter(module.SV.float(), requires_grad=True)
    model.float()

    train_dl, valid_dl = utils.split_data(emb, orig_logits, args)

    susv_params, params = utils.extract_susv_params(model)
    optim = utils.get_susv_adam(susv_params, params, args)

    best_loss = utils.calculate_ce_loss(model, position_ids, attention_mask,
                                        valid_dl)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(model.state_dict())
    glog.info(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, targets) in enumerate(train_dl):
            with torch.autocast(device_type='cuda',
                                dtype=torch.float16,
                                enabled=True):
                output = model(
                    source,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )[:, :-1].contiguous()
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),
                                             targets.to(0).view(
                                                 -1, targets.shape[-1]))
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(
                    train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = utils.calculate_ce_loss(model, position_ids,
                                                attention_mask, valid_dl)
            if test_loss < best_loss:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER'
                )
                best_loss = test_loss
                best_sd = copy.deepcopy(model.state_dict())
                worse_ct = 0
            else:
                glog.info(
                    f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE'
                )
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    with torch.no_grad():
        model.load_state_dict(best_sd)
        save_fn(model)

def build_clique_state(saved_linear, device):
    if saved_linear['fused']:
        out_dims = [s[0] for s in saved_linear['shapes']]
        in_dim = saved_linear['shapes'][0][1]
        orig_shape = (sum(out_dims), in_dim)
    else:
        orig_shape = saved_linear['shapes'][0]
 
    return {
        'hatWr': saved_linear['hatWr'].to(device).clone(),
        'Qidxs': saved_linear['Qidxs'].to(device).clone(),
        'SU': saved_linear['SU'],
        'SV': saved_linear['SV'],
        'scaleWH': saved_linear.get('scaleWH'),
        'momentum': None,
        'orig_shape': orig_shape,
        'shapes': saved_linear['shapes'],
        'scales': saved_linear['scales'], 
    }

def sparse_finetune_layer(mixed_layer, quant_order, clique_state, device, train_dl, valid_dl, args):
    mixed_layer = mixed_layer.to(device)
    momentum_rate = args.sparse_ft_momentum_rate

    for epoch in range(args.sparse_ft_epochs):
        torch.cuda.synchronize()
        epoch_start = time.time()

        mixed_layer.zero_grad()

        torch.cuda.synchronize()
        t0 = time.time()
        source, target = next(iter(train_dl))
        source = source.to(device).float()
        target = target.to(device).float()
        torch.cuda.synchronize()
        glog.info(f"Data loading: {time.time() - t0:.3f}s")

        t0 = time.time()
        output = mixed_layer(source.to(device), position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
        loss = torch.nn.MSELoss()(output, target)
        torch.cuda.synchronize()
        glog.info(f"Forward pass: {time.time() - t0:.3f}s")

        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        glog.info(f"Backward pass: {time.time() - t0:.3f}s")

        glog.info(f"epoch {epoch}, loss: {loss.item()}")

        for quant_i, (linear_attr, name) in enumerate(quant_order):
            module = attrgetter(linear_attr)(mixed_layer)
            state = clique_state[name]

            t0 = time.time()
            grad_hatW = module.weight.grad.detach().to(module.weight.dtype)
            
            if args.incoh_mode == 'had': # since indices are based on the incoherence processed weights, need to transform gradients too
                grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
            elif args.incoh_mode == 'kron':
                grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
            else:
                raise NotImplementedError
            
            torch.cuda.synchronize()
            glog.info(f"Gradient processing and incoherence transform: {time.time() - t0:.3f}s")

            t0 = time.time()
            m, n = state['Qidxs'].shape
            if grad_Wr.shape[0] < m or grad_Wr.shape[1] < n: # reshaping grad to match Qidxs shape of (m, n//8, 8)
                grad_Wr = torch.nn.functional.pad(grad_Wr, (0, n * 8 - grad_Wr.shape[1], 0, m - grad_Wr.shape[0]), value=0.0)
            grad_coeffs = grad_Wr[:m, :n * 8].reshape(m, n, 8)
            torch.cuda.synchronize()
            glog.info(f"Gradient reshaping and to coefficient processing: {time.time() - t0:.3f}s")

            t0 = time.time()
            if state['momentum'] is None:
                state['momentum'] = torch.zeros_like(grad_coeffs)
            state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs

            neighbors, neighbor_values = state['codebook'].get_neighbors(state['Qidxs']) # (m, n//8, 8), (m, n//8, 8, 8)
            curr_values = state['codebook'].grid[state['Qidxs_blocks'].long()] # (m, n//8, 8)
            torch.cuda.synchronize()
            glog.info(f"Momentum and finding neighbors: {time.time() - t0:.3f}s")

            t0 = time.time()
            scores = torch.einsum('mni,mnbi->mnb', state['momentum'], neighbor_values - curr_values.unsqueeze(2)) # vectorized form of taking inner product of momentum and flipping each bit
            torch.cuda.synchronize()
            glog.info(f"Score calculation: {time.time() - t0:.3f}s")

            t0 = time.time()
            flat_scores = scores.reshape(-1)
            vals, min_i = torch.topk(flat_scores, args.sparse_ft_num_flips, largest=False) # want most negative scores, means change and momentum disagree so we should flip
            bits = min_i % 8
            remainder = min_i // 8
            cols = remainder % (n // 8)
            rows = remainder // (n // 8)
            glog.info(f"Score for chosen flips: {vals}")
            glog.info(f"Finding indices to flip: {time.time() - t0:.3f}s")

            t0 = time.time()
            seen_inds = set() # don't want to make 2 changes within the same clique for a given block
            flip = torch.zeros(rows.shape[0], dtype=torch.bool, device=device)
            for i in range(rows.shape[0]):
                if (rows[i].item(), cols[i].item()) not in seen_inds:
                    seen_inds.add((rows[i].item(), cols[i].item()))
                    flip[i] = True
            rows = rows[flip]
            cols = cols[flip]
            bits = bits[flip]

            state['Qidxs'][rows, cols] = neighbors[rows, cols, bits]
            torch.cuda.synchronize()
            glog.info(f"Removing duplicate indices for flips and flipping: {time.time() - t0:.3f}s")

            t0 = time.time()
            # state['hatWr'] = state['hatWr'].to(module.weight.dtype)
            # modified_blocks = sorted(set(zip(rows.tolist(), cols.tolist())))
            # Wscale = state['SV'].abs().mean()
            # for (r, c) in sorted(modified_blocks):
            #     full_clique_vals = state['codebook'].grid[state['Qidxs_blocks'][r, c].long()].to(module.weight.dtype) # get coeffs for each clique from codebook
            #     all_coeffs_block = torch.zeros(64, dtype=module.weight.dtype, device=device)
            #     for cl in range(8):
            #         all_coeffs_block[state['cliques'][cl]] = full_clique_vals[cl] # unshuffle from the clique order
            #     hat_block = torch.sum(all_coeffs_block.view(64, 1, 1) * state['mats'], dim=0).to(module.weight.dtype) / torch.sqrt(state['norm']) # convert coeffs back into 8x8 block
            #     hat_block = hat_block * Wscale
            #     state['hatWr'][r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = hat_block # put patch in weight matrix of how block was edited
            Wscale = state['SV'].abs().mean()
            new_coeffs = state['codebook'].grid[state['Qidxs'].long()]
            orig_m, orig_n = state['orig_shape']
            for r in sorted(set(rows.tolist())):
                state['hatWr'][r, :n * 8] = new_coeffs[r].reshape(-1) * Wscale
            state['hatWr'] = state['hatWr'][:orig_m, :orig_n]
            torch.cuda.synchronize()
            glog.info(f"Rebuilding and updating hatWr: {time.time() - t0:.3f}s")

            t0 = time.time()
            new_hatW = quip.incoherence_process(state['hatWr'], state['SU'].to(device), state['SV'].to(device), state.get('scaleWH'), args)
            new_hatW = new_hatW.to(module.weight.dtype)
            curr = 0
            pieces = []
            for shape, scale in zip(state['shapes'], state['scales']):
                pieces.append(new_hatW[curr:curr + shape[0]] * scale)
                curr += shape[0]
            with torch.no_grad():
                module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
            torch.cuda.synchronize()
            glog.info(f"Incoherence processing and reshaping at the end: {time.time() - t0:.3f}s")

    mixed_layer = mixed_layer.cpu()
    return mixed_layer
