"""
Utilities for fine tuning
"""
import copy
from operator import attrgetter

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
            state['codebook'] = cb
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
            saved_linear['Qidxs_blocks'] = state['Qidxs_blocks'].cpu()
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
    dtype = saved_linear['hatW'].dtype
    mats = quip.hb_transform(torch.eye(64, dtype=dtype, device=device))
    norm = (mats[0] * mats[0]).sum()
 
    cliques = [
        [0, 2, 11, 25, 33, 39, 47, 57],
        [1, 3, 10, 24, 32, 38, 46, 56],
        [4, 6, 15, 29, 35, 37, 43, 61],
        [5, 7, 14, 28, 34, 36, 42, 60],
        [12, 18, 20, 26, 44, 53, 55, 62],
        [13, 19, 21, 27, 45, 52, 54, 63],
        [8, 16, 22, 30, 40, 49, 51, 58],
        [9, 17, 23, 31, 41, 48, 50, 59]
    ]
 
    if saved_linear['fused']:
        out_dims = [s[0] for s in saved_linear['shapes']]
        in_dim = saved_linear['shapes'][0][1]
        orig_shape = (sum(out_dims), in_dim)
    else:
        orig_shape = saved_linear['shapes'][0]
 
    return {
        'hatWr': saved_linear['hatWr'].to(device).clone(),
        'Qidxs_blocks': saved_linear['Qidxs_blocks'].to(device).clone(),
        'SU': saved_linear['SU'],
        'SV': saved_linear['SV'],
        'scaleWH': saved_linear.get('scaleWH'),
        'mats': mats,
        'cliques': cliques,
        'norm': norm,
        'momentum': None,
        'orig_shape': orig_shape,
    }

def sparse_finetune_layer(mixed_layer, quant_order, clique_state, device, train_dl, valid_dl, args):
    mixed_layer = mixed_layer.to(device)
    momentum_rate = args.sparse_ft_momentum_rate

    for epoch in range(args.sparse_ft_epochs):
        mixed_layer.zero_grad()
        source, target = next(iter(train_dl))
        output = mixed_layer(source.to(device), position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
        loss = torch.nn.MSELoss()(output, target.to(device))
        loss.backward()

        glog.info(f"epoch {epoch}, loss: {loss.item()}")

        for quant_i, (linear_attr, name) in enumerate(quant_order):
            module = attrgetter(linear_attr)(mixed_layer)
            state = clique_state[name]
            grad_hatW = module.weight.grad.detach().to(state['hatWr'].dtype)
            
            if args.incoh_mode == 'had': # since indices are based on the incoherence processed weights, need to transform gradients too
                grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
            elif args.incoh_mode == 'kron':
                grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
            else:
                raise NotImplementedError

            m = state['Qidxs_blocks'].shape[0] * 8
            n = state['Qidxs_blocks'].shape[1] * 8

            if grad_Wr.shape[0] < m or grad_Wr.shape[1] < n:
                grad_Wr = torch.nn.functional.pad(grad_Wr, (0, n - grad_Wr.shape[1], 0, m - grad_Wr.shape[0]), value=0.0)
            
            gm, gn = grad_Wr.shape[0], grad_Wr.shape[1]
            grad_blocks = grad_Wr.reshape(gm // 8, 8, gn // 8, 8).permute(0, 2, 1, 3)
            grad_coeffs = torch.sum(grad_blocks.unsqueeze(2) * state['mats'].view(1, 1, *state['mats'].shape), dim=(-2, -1)) / torch.sqrt(state['norm'])
            grad_clique_coeffs_list = []
            for c in range(8):
                grad_clique_coeffs_list.append(grad_coeffs[:, :, state['cliques'][c]])
            grad_clique_coeffs = torch.stack(grad_clique_coeffs_list, dim=2)

            if state['momentum'] is None:
                state['momentum'] = torch.zeros_like(grad_clique_coeffs)
            state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_clique_coeffs
            momentum = state['momentum']

            neighbors, neighbor_values = state['codebook'].get_neighbors(state['Qidxs_blocks']) # (m // 8, n // 8, 8, 8), (m // 8, n // 8, 8, 8, 8)
            curr_values = state['codebook'].grid[state['Qidxs_blocks'].long()] # (m // 8, n // 8, 8, 8)

            scores = torch.einsum('rcok,rcobk->rcob', momentum, neighbor_values - curr_values.unsqueeze(3)) # vectorized form of taking inner product of momentum and flipping each bit

            if args.sparse_ft_flip_range == "blocks":
                flat_scores = scores.reshape(m * n // 64, 64)
                vals, min_i = torch.topk(flat_scores, args.num_flips, dim=1, largest=False)
                cliques = min_i // 8
                bits = min_i % 8
                blocks = torch.arange(m * n // 64, device=device).unsqueeze(1).expand(-1, args.num_flips)
                rows = blocks // (n // 8)
                cols = blocks % (n // 8)
            else: # flip across entire weight matrix
                flat_scores = scores.reshape(-1)
                vals, min_i = torch.topk(flat_scores, args.num_flips, largest=False) # want most negative scores, means change and momentum disagree so we should flip
                bits = min_i % 8
                remainder = min_i // 8
                cliques = remainder % 8
                remainder = remainder // 8
                cols = remainder % (n // 8)
                rows = remainder // (n // 8)

            bits = bits.reshape(-1)
            cliques = cliques.reshape(-1)
            cols = cols.reshape(-1)
            rows = rows.reshape(-1)

            seen_block_cliques = set() # don't want to make 2 changes within the same clique for a given block
            flip = torch.zeros(rows.shape[0], dtype=torch.bool)
            for i in range(rows.shape[0]):
                if (rows[i].item(), cols[i].item(), cliques[i].item()) not in seen_block_cliques:
                    seen_block_cliques.add((rows[i].item(), cols[i].item(), cliques[i].item()))
                    flip[i] = True
            rows = rows[flip]
            cols = cols[flip]
            cliques = cliques[flip]
            bits = bits[flip]

            state['Qidxs_blocks'][rows, cols, cliques] = neighbors[rows, cols, cliques, bits]

            modified_blocks = sorted(set(zip(rows.tolist(), cols.tolist())))
            for (r, c) in sorted(modified_blocks):
                full_clique_vals = state['codebook'].grid[state['Qidxs_blocks'][r, c].long()] # get coeffs for each clique from codebook
                all_coeffs_block = torch.zeros(64, dtype=full_clique_vals.dtype, device=device)
                for cl in range(8):
                    all_coeffs_block[state['cliques'][cl]] = full_clique_vals[cl] # unshuffle from the clique order
                hat_block = torch.sum(all_coeffs_block.view(64, 1, 1) * state['mats'], dim=0) / torch.sqrt(state['norm']) # convert coeffs back into 8x8 block
                state['hatWr'][r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = hat_block # put patch in weight matrix of how block was edited

            new_hatW = quip.incoherence_process(state['hatWr'], state['SU'].to(device), state['SV'].to(device), state.get('scaleWH'), args)
            orig_m, orig_n = state['orig_shape']
            with torch.no_grad():
                module.weight.data.copy_(new_hatW[:orig_m, :orig_n].to(module.weight.dtype))

    mixed_layer = mixed_layer.cpu()
    return mixed_layer