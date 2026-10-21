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
from lib.algo.e8_analytic_kernel import e8_best_valid, set_e8_directions, build_valid_bits

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

        # for linear_attr, name in quant_order:
        #     save_path = f'{args.save_path}/{idx}_{name}.pt'
        #     saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
        #     state = sparse_state[name]
        #     m, n = state['orig_shape']
        #     d = state['d']
        #     Wscale = state['SV'].abs().mean()
        #     total_scale = state['Xscale'] * Wscale
        #     grid = state['codebook'].grid
        #     coeffs = grid[state['Qidxs'].long()].reshape((m//d) * (n//d), d * d)
        #     blocks = quip.hbc_transform((coeffs * total_scale)).reshape(m // d, n // d, d, d)
        #     hatWr = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape'])
        #     new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
        #     saved_linear['hatW'] = new_hatW[:m, :n].half().cpu()
        #     saved_linear['hatWr'] = hatWr.half().cpu()
        #     # saved_linear['Qidxs_blocks'] = state['Qidxs_blocks'].cpu()
        #     saved_linear['Qidxs'] = state['Qidxs'].cpu()
        #     torch.save(saved_linear, save_path)
        for linear_attr, name in quant_order:
            save_path = f'{args.save_path}/{idx}_{name}.pt'
            saved_linear = torch.load(save_path, map_location=torch.device('cpu'))
            state = sparse_state[name]
            m, n = state['orig_shape']
            d = state['d']

            # acc_blocks = state['acc_hatWr'].reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
            # acc_coeffs = quip.ihbc_transform(acc_blocks.reshape((m // d) * (n // d), d, d)).reshape(-1, 8)
            # acc_coeffs = (acc_coeffs / (state['Xscale'] * state['SV'].abs().mean())).to(state['codebook'].grid.dtype)
            # parts = []
            # chunk = 1 << 14
            # for s in range(0, acc_coeffs.shape[0], chunk):
            #     _, idx_part = state['codebook'].quantize(acc_coeffs[s:s+chunk], return_idx=True)
            #     parts.append(idx_part)
            # state['Qidxs'] = torch.cat(parts).reshape(m // d, n // d, d * d // 8)
            # qcoeffs = state['codebook'].grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
            # qblocks = quip.hbc_transform(qcoeffs * state['Xscale'] * state['SV'].abs().mean()).reshape(m // d, n // d, d, d)
            # state['acc_hatWr'] = qblocks.permute(0, 2, 1, 3).reshape(state['orig_shape']).clone()

            hatWr = state['acc_hatWr']
            if args.incoh_mode == 'had_left':
                new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
                if args.rescale_WH:
                    new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
            else:
                new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
            saved_linear['hatW'] = new_hatW[:m, :n].half().cpu()
            saved_linear['hatWr'] = hatWr.half().cpu()
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

    hadK, K = utils.get_hadK(n)
    hadK = hadK.to(device) if hadK is not None else None

    lookup = {}
    keys = (neighbors['directions'] * 2).round().to(torch.long)
    for i in range(keys.shape[0]):
        lookup[tuple(keys[i].tolist())] = i

    powers = (5 ** torch.arange(8, device=device)).long()
    hash_to_idx = torch.full((5 ** 8,), -1, dtype=torch.long, device=device)
    dkeys = ((neighbors['directions'].to(device) * 2).round().long() + 2)
    hashes = (dkeys * powers).sum(-1)
    hash_to_idx[hashes] = torch.arange(neighbors['directions'].shape[0], device=device)

    # e8_best_valid's constant-memory direction table must use the same row
    # ordering as neighbors['table']'s columns -- pass the real table here,
    # not e8_analytic_kernel's own build_e8_directions() enumeration. Cheap
    # and idempotent, so it's fine to call again on every layer even though
    # the file (and therefore this table) never actually changes.
    set_e8_directions(neighbors['directions'].to(device))
    valid_bits = build_valid_bits(neighbors['table'].to(device))

    return {
        'hatWr': None,
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
        'directions': neighbors['directions'].to(device),
        'directions_lookup': lookup,
        'hash_to_idx': hash_to_idx,
        'valid_bits': valid_bits,
        'hadK': hadK,
        'K_had': K
    }

def quaternion_triality_F(dtype=torch.float64, device=None):
    """
    Returns the 16x16 orthogonal order-3 triality operator F.

    Convention:
      - quaternion basis = (1, i, j, k)
      - matrices X in R^{4x4} are flattened row-major
      - F(X) = -(1/2) sum_{a,b} X[a,b] L(e_a) R(e_b)
    """
    H = torch.tensor([
        [-1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1],
        [ 0, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0,-1,  0, 0, 1, 0],
        [ 0, 0, 1, 0,  0, 0, 0, 1,  1, 0, 0, 0,  0,-1, 0, 0],
        [ 0, 0, 0, 1,  0, 0,-1, 0,  0, 1, 0, 0,  1, 0, 0, 0],

        [ 0,-1, 0, 0, -1, 0, 0, 0,  0, 0, 0,-1,  0, 0, 1, 0],
        [-1, 0, 0, 0,  0, 1, 0, 0,  0, 0,-1, 0,  0, 0, 0,-1],
        [ 0, 0, 0,-1,  0, 0, 1, 0,  0, 1, 0, 0,  1, 0, 0, 0],
        [ 0, 0, 1, 0,  0, 0, 0, 1, -1, 0, 0, 0,  0, 1, 0, 0],

        [ 0, 0,-1, 0,  0, 0, 0, 1, -1, 0, 0, 0,  0,-1, 0, 0],
        [ 0, 0, 0, 1,  0, 0, 1, 0,  0, 1, 0, 0, -1, 0, 0, 0],
        [-1, 0, 0, 0,  0,-1, 0, 0,  0, 0, 1, 0,  0, 0, 0,-1],
        [ 0,-1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0],

        [ 0, 0, 0,-1,  0, 0,-1, 0,  0, 1, 0, 0, -1, 0, 0, 0],
        [ 0, 0,-1, 0,  0, 0, 0, 1,  1, 0, 0, 0,  0, 1, 0, 0],
        [ 0, 1, 0, 0, -1, 0, 0, 0,  0, 0, 0, 1,  0, 0, 1, 0],
        [-1, 0, 0, 0,  0,-1, 0, 0,  0, 0,-1, 0,  0, 0, 0, 1],
    ], dtype=dtype, device=device)

    return 0.5 * H

def flipping_algorithm_4x4(module, state, args, device, momentum_rate):
    grad_hatW = module.weight.grad.detach().to(module.weight.dtype)
    if args.incoh_mode == 'had':
        grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
    elif args.incoh_mode == 'kron':
        grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
    elif args.incoh_mode == 'had_left':
        grad_Wr = utils.matmul_hadUt_cuda(grad_hatW * state['SU'].to(device), state['hadK'], state['K_had'])
    else:
        raise NotImplementedError

    m, n = state['orig_shape']
    d = state['d']

    if state['momentum'] is None:
        state['momentum'] = torch.zeros_like(grad_Wr)
    state['momentum'] = (momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_Wr)

    if state.get('F') is None:
        F = quaternion_triality_F(dtype=torch.float64, device=device)
        state['F'] = F
        state['F2'] = F.T # F @ F # change to F.T

    if state.get('acc_hatWr') is None:
        coeffs = state['codebook'].grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
        blocks = quip.hbc_transform(coeffs * (state['Xscale'] * state['SV'].abs().mean())).reshape(m // d, n // d, d, d)
        state['acc_hatWr'] = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape']).clone()
        state['step'] = 0

    r = torch.arange(m // 4, device=device)
    c = (r - (state['step'] % max(m // 4, n // 4))) % max(m // 4, n // 4)
    inbounds = c < n // 4
    r = r[inbounds]
    c = c[inbounds]

    momentum_reshaped = state['momentum'].reshape(m // 4, 4, n // 4, 4).permute(0, 2, 1, 3)
    momentum_diag = momentum_reshaped[r, c].double()

    F, F2 = state['F'], state['F2']

    MF1 = (momentum_diag.reshape(r.numel(), 16) @ F).reshape(r.numel(), 4, 4)
    U, Sigma, V = torch.linalg.svd(MF1)
    u1 = U[:, :, 0] # * Sigma[:, 0].clamp(min=0).sqrt().unsqueeze(-1)
    v1 = V[:, 0, :] # * Sigma[:, 0].clamp(min=0).sqrt().unsqueeze(-1)
    O1 = ((u1.unsqueeze(2) * v1.unsqueeze(1) * 2).reshape(r.numel(), 16) @ F.T).reshape(r.numel(), 4, 4)

    MF2 = (momentum_diag.reshape(r.numel(), 16) @ F2).reshape(r.numel(), 4, 4)
    U, Sigma, V = torch.linalg.svd(MF2)
    u2 = U[:, :, 0] # * Sigma[:, 0].clamp(min=0).sqrt().unsqueeze(-1)
    v2 = V[:, 0, :] # * Sigma[:, 0].clamp(min=0).sqrt().unsqueeze(-1)
    O2 = ((u2.unsqueeze(2) * v2.unsqueeze(1) * 2).reshape(r.numel(), 16) @ F2.T).reshape(r.numel(), 4, 4)

    d1 = (O1 - momentum_diag).reshape(r.numel(), 16).norm(dim=1)
    d2 = (O2 - momentum_diag).reshape(r.numel(), 16).norm(dim=1)
    O = torch.where((d1 <= d2).view(r.numel(), 1, 1), O1, O2)

    lr = getattr(args, 'sparse_ft_new_lr', 0.1)    
    to_add = (-lr) * O

    for i in range(r.numel()):
        a, b = r[i].item() * 4, c[i].item() * 4
        state['acc_hatWr'][a:a + 4, b:b + 4] += to_add[i].to(state['acc_hatWr'].dtype)

    state['step'] += 1

    hatWr = state['acc_hatWr']
    if args.incoh_mode == 'had_left':
        new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
        if args.rescale_WH:
            new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
    else:
        new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
    new_hatW = new_hatW.to(module.weight.dtype)

    curr = 0
    pieces = []
    for shape, scale in zip(state['shapes'], state['scales']):
        pieces.append(new_hatW[curr:curr + shape[0]] * scale)
        curr += shape[0]
    with torch.no_grad():
        module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
        module.weight.grad = None

def flipping_algorithm(module, state, args, device, momentum_rate):
    grad_hatW = module.weight.grad.detach().to(module.weight.dtype)
    if args.incoh_mode == 'had':
        grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
    elif args.incoh_mode == 'kron':
        grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
    elif args.incoh_mode == 'had_left':
        grad_Wr = utils.matmul_hadUt_cuda(grad_hatW * state['SU'].to(device), state['hadK'], state['K_had'])
    else:
        raise NotImplementedError

    m, n = state['orig_shape']
    d = state['d']

    # grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    # grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
    # grad_coeffs = grad_coeffs.to(module.weight.dtype) 

    # if state['momentum'] is None:
    #     state['momentum'] = torch.zeros_like(grad_coeffs)
    # state['momentum'] = (momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs)

    if state['momentum'] is None:
        state['momentum'] = torch.zeros_like(grad_Wr)
    state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_Wr

    if state.get('acc_hatWr') is None:
        coeffs = state['codebook'].grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
        blocks = quip.hbc_transform(coeffs * (state['Xscale'] * state['SV'].abs().mean())).reshape(m // d, n // d, d, d)
        state['acc_hatWr'] = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape']).clone()
        state['step'] = 0

    r = torch.arange(m // 8, device=device)
    c = (r - (state['step'] % max(m // 8, n // 8))) % max(m // 8, n // 8)
    inbounds = c < n // 8
    r = r[inbounds]
    c = c[inbounds]

    # momentum_reshaped = state['momentum'].reshape(m // d, n // d, d // 8, d // 8, 8, 8)
    # blockr = r // (d // 8)
    # blockc = c // (d // 8)
    # subblockr = r % (d // 8)
    # subblockc = c % (d // 8)
    # currblocks_momentum = momentum_reshaped[blockr, blockc, subblockr, subblockc].reshape(r.numel(), 8, 8).float()

    momentum_reshaped = state['momentum'].reshape(m // 8, 8, n // 8, 8).permute(0, 2, 1, 3)
    currblocks_momentum = momentum_reshaped[r, c].float()

    # U, Sigma, V = torch.linalg.svd(currblocks_momentum)
    # u = U[:, :, 0] * Sigma[:, 0].sqrt().unsqueeze(-1)
    # v = V[:, 0, :] * Sigma[:, 0].sqrt().unsqueeze(-1)

    # u_quant = state['codebook'].quantize(u.to(state['codebook'].grid.dtype), return_idx=False).float()
    # v_quant = state['codebook'].quantize(v.to(state['codebook'].grid.dtype), return_idx=False).float()
    # uvT = (u_quant.unsqueeze(2) * v_quant.unsqueeze(1)).reshape(r.numel(), 64)
    # lr = getattr(args, 'sparse_ft_new_lr', 0.1)
    # to_add = quip.hbc_transform(uvT * state['Xscale'] * state['SV'].abs().mean()).reshape(r.numel(), 8, 8) * lr

    U, Sigma, V = torch.linalg.svd(currblocks_momentum)
    O = U @ V
    lr = getattr(args, 'sparse_ft_new_lr', 0.1)
    to_add = (-lr) * O

    for i in range(r.numel()):
        currr = r[i].item() * 8
        currc = c[i].item() * 8
        state['acc_hatWr'][currr:currr + 8, currc:currc + 8] += to_add[i].to(state['acc_hatWr'].dtype)

    state['step'] += 1

    if state['step'] % max(m // 8, n // 8) == 0:
        acc_blocks = state['acc_hatWr'].reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
        acc_coeffs = quip.ihbc_transform(acc_blocks.reshape((m // d) * (n // d), d, d)).reshape(-1, 8)
        acc_coeffs = (acc_coeffs / (state['Xscale'] * state['SV'].abs().mean())).to(state['codebook'].grid.dtype)
        # _, new_quanti = state['codebook'].quantize((acc_coeffs / (state['Xscale'] * state['SV'].abs().mean())).to(state['codebook'].grid.dtype), return_idx=True)
        batched_parts = []
        chunk = 1 << 14
        for i in range(0, acc_coeffs.shape[0], chunk):
            _, part = state['codebook'].quantize(acc_coeffs[i:i + chunk], return_idx=True)
            batched_parts.append(part)
        state['Qidxs'] = torch.cat(batched_parts).reshape(m // d, n // d, d * d // 8)
        # state['Qidxs'] = new_quanti.reshape(m // d, n // d, d * d // 8)
        coeffs = state['codebook'].grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
        blocks = quip.hbc_transform(coeffs * state['Xscale'] * state['SV'].abs().mean()).reshape(m // d, n // d, d, d)
        state['acc_hatWr'] = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape']).clone()

    hatWr = state['acc_hatWr']

    if args.incoh_mode == 'had_left':
        new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
        if args.rescale_WH:
            new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
    else:
        new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
    new_hatW = new_hatW.to(module.weight.dtype)

    curr = 0
    pieces = []
    for shape, scale in zip(state['shapes'], state['scales']):
        pieces.append(new_hatW[curr:curr + shape[0]] * scale)
        curr += shape[0]
    with torch.no_grad():
        module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
        module.weight.grad = None

def flipping_algorithm_8x8(module, state, args, device, momentum_rate):
    grad_hatW = module.weight.grad.detach().to(module.weight.dtype)
    if args.incoh_mode == 'had':
        grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
    elif args.incoh_mode == 'kron':
        grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
    elif args.incoh_mode == 'had_left':
        grad_Wr = utils.matmul_hadUt_cuda(grad_hatW * state['SU'].to(device), state['hadK'], state['K_had'])
    else:
        raise NotImplementedError

    m, n = state['orig_shape']
    d = state['d']

    # grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    # grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
    # grad_coeffs = grad_coeffs.to(module.weight.dtype) 

    # if state['momentum'] is None:
    #     state['momentum'] = torch.zeros_like(grad_coeffs)
    # state['momentum'] = (momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs)

    if state['momentum'] is None:
        state['momentum'] = torch.zeros_like(grad_Wr)
    state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_Wr

    if state.get('acc_hatWr') is None:
        coeffs = state['codebook'].grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
        blocks = quip.hbc_transform(coeffs * (state['Xscale'] * state['SV'].abs().mean())).reshape(m // d, n // d, d, d)
        state['acc_hatWr'] = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape']).clone()
        state['step'] = 0

    r = torch.arange(m // 8, device=device)
    c = (r - (state['step'] % max(m // 8, n // 8))) % max(m // 8, n // 8)
    inbounds = c < n // 8
    r = r[inbounds]
    c = c[inbounds]

    # momentum_reshaped = state['momentum'].reshape(m // d, n // d, d // 8, d // 8, 8, 8)
    # blockr = r // (d // 8)
    # blockc = c // (d // 8)
    # subblockr = r % (d // 8)
    # subblockc = c % (d // 8)
    # currblocks_momentum = momentum_reshaped[blockr, blockc, subblockr, subblockc].reshape(r.numel(), 8, 8).float()

    momentum_reshaped = state['momentum'].reshape(m // 8, 8, n // 8, 8).permute(0, 2, 1, 3)
    currblocks_momentum = momentum_reshaped[r, c].float()

    # U, Sigma, V = torch.linalg.svd(currblocks_momentum)
    # u = U[:, :, 0] * Sigma[:, 0].sqrt().unsqueeze(-1)
    # v = V[:, 0, :] * Sigma[:, 0].sqrt().unsqueeze(-1)

    # u_quant = state['codebook'].quantize(u.to(state['codebook'].grid.dtype), return_idx=False).float()
    # v_quant = state['codebook'].quantize(v.to(state['codebook'].grid.dtype), return_idx=False).float()
    # uvT = (u_quant.unsqueeze(2) * v_quant.unsqueeze(1)).reshape(r.numel(), 64)
    # lr = getattr(args, 'sparse_ft_new_lr', 0.1)
    # to_add = quip.hbc_transform(uvT * state['Xscale'] * state['SV'].abs().mean()).reshape(r.numel(), 8, 8) * lr

    U, Sigma, V = torch.linalg.svd(currblocks_momentum)
    O = U @ V
    lr = getattr(args, 'sparse_ft_new_lr', 0.1)
    to_add = (-lr) * O

    for i in range(r.numel()):
        currr = r[i].item() * 8
        currc = c[i].item() * 8
        state['acc_hatWr'][currr:currr + 8, currc:currc + 8] += to_add[i].to(state['acc_hatWr'].dtype)

    state['step'] += 1

    # if state['step'] % max(m // 8, n // 8) == 0:
    #     acc_blocks = state['acc_hatWr'].reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    #     acc_coeffs = quip.ihbc_transform(acc_blocks.reshape((m // d) * (n // d), d, d)).reshape(-1, 8)
    #     acc_coeffs = (acc_coeffs / (state['Xscale'] * state['SV'].abs().mean())).to(state['codebook'].grid.dtype)
    #     # _, new_quanti = state['codebook'].quantize((acc_coeffs / (state['Xscale'] * state['SV'].abs().mean())).to(state['codebook'].grid.dtype), return_idx=True)
    #     batched_parts = []
    #     chunk = 1 << 14
    #     for i in range(0, acc_coeffs.shape[0], chunk):
    #         _, part = state['codebook'].quantize(acc_coeffs[i:i + chunk], return_idx=True)
    #         batched_parts.append(part)
    #     state['Qidxs'] = torch.cat(batched_parts).reshape(m // d, n // d, d * d // 8)
    #     # state['Qidxs'] = new_quanti.reshape(m // d, n // d, d * d // 8)
    #     coeffs = state['codebook'].grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
    #     blocks = quip.hbc_transform(coeffs * state['Xscale'] * state['SV'].abs().mean()).reshape(m // d, n // d, d, d)
    #     state['acc_hatWr'] = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape']).clone()

    hatWr = state['acc_hatWr']

    if args.incoh_mode == 'had_left':
        new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
        if args.rescale_WH:
            new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
    else:
        new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
    new_hatW = new_hatW.to(module.weight.dtype)

    curr = 0
    pieces = []
    for shape, scale in zip(state['shapes'], state['scales']):
        pieces.append(new_hatW[curr:curr + shape[0]] * scale)
        curr += shape[0]
    with torch.no_grad():
        module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
        module.weight.grad = None

def sparse_finetune_layer(mixed_layer, quant_order, clique_state, device, train_dl, valid_dl, args):
    mixed_layer = mixed_layer.to(device)
    momentum_rate = args.sparse_ft_momentum_rate

    warmup_batches = getattr(args, 'sparse_ft_warmup_batches', 0)
    if warmup_batches > 0:
        it = iter(train_dl)
        for _ in range(warmup_batches):
            source, target = next(it)
            source = source.to(device).float()
            target = target.to(device).float()
            mixed_layer.zero_grad()
            output = mixed_layer(source, position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
            torch.nn.MSELoss()(output, target).backward()
            for linear_attr, name in quant_order:
                module = attrgetter(linear_attr)(mixed_layer)
                state = clique_state[name]

                grad_hatW = module.weight.grad.detach().to(module.weight.dtype)

                if args.incoh_mode == 'had':
                    grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
                elif args.incoh_mode == 'kron':
                    grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
                elif args.incoh_mode == 'had_left':
                    grad_Wr = utils.matmul_hadUt_cuda(grad_hatW * state['SU'].to(device), state['hadK'], state['K_had'])
                else:
                    raise NotImplementedError

                m, n = state['orig_shape']
                d = state['d']
                # grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
                # grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
    
                # if state['momentum'] is None:
                #     state['momentum'] = torch.zeros_like(grad_coeffs)
                # state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs
                if state['momentum'] is None:
                    state['momentum'] = torch.zeros_like(grad_Wr)
                state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_Wr
        mixed_layer.zero_grad()

    for epoch in range(args.sparse_ft_epochs):
        mixed_layer.zero_grad()

        total_loss = 0
        error = 0
        count = 0
        for source, target in train_dl:
            source = source.to(device).float()
            target = target.to(device).float()
            mixed_layer.zero_grad()
            output = mixed_layer(source, position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
            loss = torch.nn.MSELoss()(output, target)
            loss.backward()
            total_loss += loss.item()
            error += (output - target).norm().item()
            count += 1

            for quant_i, (linear_attr, name) in enumerate(quant_order):
                module = attrgetter(linear_attr)(mixed_layer)
                state = clique_state[name]
                flipping_algorithm(module, state, args, device, momentum_rate)
        
        train_loss = total_loss / count
        train_error = error / count
        glog.info(f"epoch {epoch}, loss: {train_loss}, error: {train_error}")

    # for linear_attr, name in quant_order:
    #     state = clique_state[name]
    #     if state.get('acc_hatWr') is None:
    #         continue
    #     m, n = state['orig_shape']
    #     d = state['d']
    #     acc_blocks = state['acc_hatWr'].reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    #     acc_coeffs = quip.ihbc_transform(acc_blocks.reshape((m // d) * (n // d), d, d)).reshape(-1, 8)
    #     acc_coeffs = (acc_coeffs / (state['Xscale'] * state['SV'].abs().mean())).to(state['codebook'].grid.dtype)
    #     # _, new_quanti = state['codebook'].quantize((acc_coeffs), return_idx=True)
    #     batched_parts = []
    #     chunk = 1 << 14
    #     for i in range(0, acc_coeffs.shape[0], chunk):
    #         _, part = state['codebook'].quantize(acc_coeffs[i:i + chunk], return_idx=True)
    #         batched_parts.append(part)
    #     state['Qidxs'] = torch.cat(batched_parts).reshape(m // d, n // d, d * d // 8)
    #     # state['Qidxs'] = new_quanti.reshape(m // d, n // d, d * d // 8)

    mixed_layer = mixed_layer.cpu()
    return mixed_layer

def scale_hook(tuned_scale):
    def hook(module, input, output):
        if isinstance(output, tuple):
            return tuple(i * tuned_scale.to(i.dtype) for i in output) # if fused linear
        return output * tuned_scale.to(output.dtype) # if nn.linear
    return hook

def su_hook(tune_su):
    def hook(module, args):
        return (args[0] * tune_su.to(args[0].dtype),) + args[1:]
    return hook

def sparse_finetune_e2e(model, quant_order, clique_state, train_dl, valid_dl, position_ids, attention_mask, device, args, original_lm_head_w):
    val_loss_sum = 0
    count = 0
    with torch.no_grad():
        for source, target in valid_dl:
            # out = model.model(inputs_embeds=source.to(device).to(torch.bfloat16), position_ids=position_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
            out = model.model(input_ids=source.to(device), use_cache=False).last_hidden_state
            val_loss_sum += torch.nn.MSELoss()(out.float(), target.to(device).float()).item()
            count += 1
    prev_loss = val_loss_sum / count
    prev_Q = {i: state['Qidxs'].clone() for i, state in clique_state.items()}

    other_params = []
    for pname, p in model.named_parameters():
        if pname.endswith('.weight') and ('norm' in pname or 'embed_tokens' in pname or 'lm_head' in pname):
            p.requires_grad_(True)
            other_params.append(p)

    tuned_scales = []
    hooks = []
    for i, linear_attr, name in quant_order:
        module = attrgetter(linear_attr)(model.model.layers[i])
        module.tuned_scale = nn.Parameter(torch.ones((), device=device))
        tuned_scales.append(module.tuned_scale)
        hooks.append(module.register_forward_hook(scale_hook(module.tuned_scale)))

    tuned_su_vals = {}
    hooks_su = []
    for i, linear_attr, name in quant_order:
        module = attrgetter(linear_attr)(model.model.layers[i])
        module.tuned_su = nn.Parameter(torch.ones(module.weight.shape[1], device=device))
        tuned_su_vals[(i, name)] = module.tuned_su
        hooks_su.append(module.register_forward_pre_hook(su_hook(module.tuned_su)))
    other_params = other_params + tuned_scales + list(tuned_su_vals.values())

    adam = torch.optim.AdamW(other_params, lr=5e-5)
    prev_other = [p.detach().clone() for p in other_params]

    for epoch in range(args.sparse_ft_e2e_epochs):
        model.train()
        for source, target in train_dl:
            model.zero_grad()
            # out = model.model(inputs_embeds=source.to(device).to(torch.bfloat16),
            #                   position_ids=position_ids, attention_mask=attention_mask,
            #                   use_cache=False).last_hidden_state
            out = model.model(input_ids=source.to(device), use_cache=False).last_hidden_state
            loss = torch.nn.MSELoss()(out.float(), target.to(device).float())
            curr_unemb = out @ model.lm_head.weight.T
            original_unemb = target.to(device).to(torch.bfloat16) @ original_lm_head_w.T
            loss = loss + torch.nn.MSELoss()(curr_unemb.float(), original_unemb.float())
            loss.backward()
            for i, linear_attr, name in quant_order:
                module = attrgetter(linear_attr)(model.model.layers[i])
                flipping_algorithm(module, clique_state[(i, name)], args, device, args.sparse_ft_momentum_rate)
            adam.step()

        val_loss_sum = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for source, target in valid_dl:
                # out = model.model(inputs_embeds=source.to(device).to(torch.bfloat16), position_ids=position_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
                out = model.model(input_ids=source.to(device), use_cache=False).last_hidden_state
                vloss = torch.nn.MSELoss()(out.float(), target.to(device).float())
                vloss = vloss + torch.nn.MSELoss()((out @ model.lm_head.weight.T).float(), (target.to(device).to(torch.bfloat16) @ original_lm_head_w.T).float())
                val_loss_sum += vloss.item()
                count += 1
        val_loss = val_loss_sum / count
        if val_loss < prev_loss:
            prev_loss = val_loss
            prev_Q = {i: state['Qidxs'].clone() for i, state in clique_state.items()}
            prev_other = [p.detach().clone() for p in other_params]
        glog.info(f"e2e epoch {epoch}: val loss: {val_loss:.6e}, best val loss {prev_loss:.6e}")

    for i, state in clique_state.items():
        state['Qidxs'].copy_(prev_Q[i])
    with torch.no_grad():
        for p, best in zip(other_params, prev_other):
            p.copy_(best)
    for i, linear_attr, name in quant_order:
        module = attrgetter(linear_attr)(model.model.layers[i])
        state = clique_state[(i, name)]
        m, n = state['orig_shape']
        d = state['d']
        module = attrgetter(linear_attr)(model.model.layers[i])
        state = clique_state[(i, name)]
        Wscale = state['SV'].abs().mean()
        total_scale = state['Xscale'] * Wscale
        grid = state['codebook'].grid
        coeffs = grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
        blocks = quip.hbc_transform((coeffs * total_scale)).reshape(m // d, n // d, d, d)
        hatWr = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape'])
    
        if args.incoh_mode == 'had_left':
            new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
            if args.rescale_WH:
                new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
        else:
            new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
        new_hatW = new_hatW.to(module.weight.dtype)
    
        curr = 0
        pieces = []
        for shape, scale in zip(state['shapes'], state['scales']):
            pieces.append(new_hatW[curr:curr + shape[0]] * scale)
            curr += shape[0]
        with torch.no_grad():
            module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
    
    for h in hooks:
        h.remove()

    return model, clique_state

def old_flipping_algorithm(module, state, args, device, momentum_rate):
    grad_hatW = module.weight.grad.detach().to(module.weight.dtype)
    if args.incoh_mode == 'had':
        grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
    elif args.incoh_mode == 'kron':
        grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
    elif args.incoh_mode == 'had_left':
        grad_Wr = utils.matmul_hadUt_cuda(grad_hatW * state['SU'].to(device), state['hadK'], state['K_had'])
    else:
        raise NotImplementedError

    m, n = state['orig_shape']
    d = state['d']
    grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
    grad_coeffs = grad_coeffs.to(module.weight.dtype) 

    if state['momentum'] is None:
        state['momentum'] = torch.zeros_like(grad_coeffs)
    state['momentum'] = (momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs)

    neighbors_table = state['neighbors_table']
    valid_bits = state['valid_bits']

    P, Qd, G, _ = state['momentum'].shape
    mom = state['momentum'].reshape(-1, 8)
    Qflat = state['Qidxs'].reshape(-1)

    best_score, dir_idx = e8_best_valid(mom.float().contiguous(), Qflat.long(), valid_bits)
    landing = neighbors_table[Qflat.long(), dir_idx.clamp(min=0).long()]
    bs = best_score.reshape(P * Qd, G)
    lg = landing.reshape(P * Qd, G)
    Qb = Qflat.reshape(P * Qd, G)
    kf = min(args.sparse_ft_num_flips, G)
    values, gidx = torch.topk(bs, kf, dim=1, largest=False)
    take = values < 0
    if take.any():
        b_ix = torch.arange(P * Qd, device=device).unsqueeze(1).expand(-1, kf)[take]
        g_ix = gidx[take]
        newl = lg[b_ix, g_ix]
        ok = newl >= 0
        Qb[b_ix[ok], g_ix[ok]] = newl[ok].to(Qb.dtype)
    state['Qidxs'] = Qb.reshape(P, Qd, G)

    Wscale = state['SV'].abs().mean()
    total_scale = state['Xscale'] * Wscale
    grid = state['codebook'].grid
    coeffs = grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
    blocks = quip.hbc_transform((coeffs * total_scale)).reshape(m // d, n // d, d, d)
    hatWr = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape'])

    if args.incoh_mode == 'had_left':
        new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
        if args.rescale_WH:
            new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
    else:
        new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
    new_hatW = new_hatW.to(module.weight.dtype)

    curr = 0
    pieces = []
    for shape, scale in zip(state['shapes'], state['scales']):
        pieces.append(new_hatW[curr:curr + shape[0]] * scale)
        curr += shape[0]
    with torch.no_grad():
        module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
        module.weight.grad = None
