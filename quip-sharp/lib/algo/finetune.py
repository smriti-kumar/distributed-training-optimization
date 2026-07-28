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
            state['codebook'] = cb.to(device)
            sparse_state[name] = state
 
        mixed_layer = sparse_finetune_layer(mixed_layer, quant_order, sparse_state, device, train_dl, valid_dl, args)

        for linear_attr, name in quant_order:
            save_path = f'{args.save_path}/{idx}_{name}.pt'
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
            state = sparse_state[name]
            m, n = state['orig_shape']
            d = state['d']
            Wscale = state['SV'].abs().mean()
            total_scale = state['Xscale'] * Wscale
            grid = state['codebook'].grid
            coeffs = grid[state['Qidxs'].long()].reshape((m//d) * (n//d), d * d)
            blocks = quip.hbc_transform((coeffs * total_scale)).reshape(m // d, n // d, d, d)
            hatWr = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape'])
            new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
            saved_linear['hatW'] = new_hatW[:m, :n].half().cpu()
            saved_linear['hatWr'] = hatWr.half().cpu()
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

    m, n = orig_shape
    d = 8
    while m % (2 * d) == 0 and n % (2 * d) == 0:
        d *= 2

    neighbors = torch.load('e8_2bit_neighbors.pt', map_location=f'cuda:{device}')

    Qidxs = saved_linear['Qidxs']
    Qidxs = Qidxs.to(device).reshape(m // d, n // d, d * d // 8).clone()
 
    return {
        'hatWr': saved_linear['hatWr'].to(device).clone(),
        'Qidxs': Qidxs,
        'SU': saved_linear['SU'],
        'SV': saved_linear['SV'],
        'scaleWH': saved_linear.get('scaleWH'),
        'momentum': None,
        'orig_shape': orig_shape,
        'shapes': saved_linear['shapes'],
        'scales': saved_linear['scales'], 
        'Xscale': saved_linear['Xscale'].to(device),
        'd': d,
        'neighbors_table': neighbors['table'].to(device),
        'directions': neighbors['directions'].to(device)
    }

def sparse_finetune_layer(mixed_layer, quant_order, clique_state, device, train_dl, valid_dl, args):
    mixed_layer = mixed_layer.to(device)
    momentum_rate = args.sparse_ft_momentum_rate

    for epoch in range(args.sparse_ft_epochs):
        mixed_layer.zero_grad()

        loss = 0
        error = 0
        count = 0
        for source, target in train_dl:
            source = source.to(device).float()
            target = target.to(device).float()
            output = mixed_layer(source, position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
            loss = torch.nn.MSELoss()(output, target)
            loss.backward()
            loss += loss.item()
            error += (output - target).norm().item()
            count += 1
        glog.info(f"epoch {epoch}, loss: {loss / count}, error: {error / count}")

        # for each weight matrix
            # for each block
                # calculate scores for each 8 group which of the 240 neighbors its best to flip to, flip based on best scores
        
        # for the 240 neighbors, need to make a lookup matrix of size 2^16 x 240, represents valid neighbor for each possible point on the lattice
        # should also store for each of the 240 per point on the lattice where it would move to if that was selected
        # have to account for corner cases where we might move off the lattice, store -1 or the same point in that location

        for quant_i, (linear_attr, name) in enumerate(quant_order):
            module = attrgetter(linear_attr)(mixed_layer)
            state = clique_state[name]

            grad_hatW = module.weight.grad.detach().to(module.weight.dtype)

            if args.incoh_mode == 'had': # since indices are based on the incoherence processed weights, need to transform gradients too
                grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
            elif args.incoh_mode == 'kron':
                grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
            else:
                raise NotImplementedError
            
            glog.info(f"Gradient processing and incoherence transform")

            m, n = state['orig_shape']
            d = state['d']
            grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
            grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
            glog.info(f"Gradient to coefficient processing")

            if state['momentum'] is None:
                state['momentum'] = torch.zeros_like(grad_coeffs)
            state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs
            glog.info(f"Momentum update")

            directions = state['directions']
            neighbors_table = state['neighbors_table']
            for i in range(m // d):
                for j in range(n // d):
                    Q_curr = state['Qidxs'][i, j]
                    momentum_curr = state['momentum'][i, j]
                    scores = momentum_curr @ directions.T
                    neighbors = neighbors_table[Q_curr.long()]
                    scores = scores.masked_fill(neighbors < 0, float('inf'))
                    best_scores, best_directions = scores.min(dim=-1)
                    values, inds = torch.topk(best_scores, args.sparse_ft_num_flips, largest=False)
                    inds = inds[values < 0]
                    if inds.numel() == 0: 
                        continue
                    Q_curr[inds] = neighbors_table[Q_curr[inds].long(), best_directions[inds]].long()
            glog.info(f"Updating coodebook indices with new values")

            Wscale = state['SV'].abs().mean()
            total_scale = state['Xscale'] * Wscale
            grid = state['codebook'].grid
            coeffs = grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
            blocks = quip.hbc_transform((coeffs * total_scale)).reshape(m // d, n // d, d, d)
            hatWr = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape'])
            glog.info(f"Rebuilding and updating hatWr")

            new_hatW = quip.incoherence_process(hatWr, state['SU'].to(f'cuda:{device}'), state['SV'].sign().to(f'cuda:{device}'), state.get('scaleWH'), args)
            new_hatW = new_hatW.to(module.weight.dtype)
            curr = 0
            pieces = []
            for shape, scale in zip(state['shapes'], state['scales']):
                pieces.append(new_hatW[curr:curr + shape[0]] * scale)
                curr += shape[0]
            with torch.no_grad():
                module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
            glog.info(f"Updating module weight")

    mixed_layer = mixed_layer.cpu()
    return mixed_layer
