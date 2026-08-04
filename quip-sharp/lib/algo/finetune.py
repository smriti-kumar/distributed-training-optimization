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
        'directions': neighbors['directions'].to(device),
        'directions_lookup': lookup,
        'hash_to_idx': hash_to_idx,
        'valid_bits': valid_bits,
        'hadK': hadK,
        'K_had': K
    }

def sparse_finetune_layer(mixed_layer, quant_order, clique_state, device, train_dl, valid_dl, args):
    t_start_func = time.perf_counter()
    mixed_layer = mixed_layer.to(device)
    momentum_rate = args.sparse_ft_momentum_rate

    # t0 = time.perf_counter()
    # val_loss = 0
    # count = 0
    # with torch.no_grad():
    #     for source, target in valid_dl:
    #         source = source.to(device).float()
    #         target = target.to(device).float()
    #         output = mixed_layer(source, position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
    #         val_loss += torch.nn.MSELoss()(output, target).item()
    #         count += 1
    # prev_loss = val_loss / count
    # torch.cuda.synchronize(device)
    # glog.info(f"Initial validation: {time.perf_counter() - t0:.4f}s")

    warmup_batches = getattr(args, 'sparse_ft_warmup_batches', 0)
    if warmup_batches > 0:
        t_warmup_start = time.perf_counter()
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
                grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
                grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
    
                if state['momentum'] is None:
                    state['momentum'] = torch.zeros_like(grad_coeffs)
                state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs
        mixed_layer.zero_grad()
        torch.cuda.synchronize(device)
        t_warmup_total = time.perf_counter() - t_warmup_start
        glog.info(f"Warmup batches: {t_warmup_total:.4f}s")

    total_time_fwd_bwd = 0.0
    total_time_flip_search = 0.0
    # total_time_epoch_val = 0.0
    total_time_hbc = 0.0
    total_time_ihbc = 0.0
    total_time_grad_incoh_proc = 0.0
    total_time_grad_reshaping = 0.0
    total_time_momentum_update = 0.0
    total_time_weight_blocks_reshaping = 0.0
    total_time_weight_incoh_proc = 0.0
    total_time_weight_pieces = 0.0
    total_time_e8_best_valid = 0.0

    for epoch in range(args.sparse_ft_epochs):
        mixed_layer.zero_grad()

        total_loss = 0
        error = 0
        count = 0
        for source, target in train_dl:
            t0 = time.perf_counter()
            source = source.to(device).float()
            target = target.to(device).float()
            mixed_layer.zero_grad()
            output = mixed_layer(source, position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
            loss = torch.nn.MSELoss()(output, target)
            loss.backward()
            torch.cuda.synchronize(device)
            total_time_fwd_bwd += time.perf_counter() - t0
            total_loss += loss.item()
            error += (output - target).norm().item()
            count += 1

            for quant_i, (linear_attr, name) in enumerate(quant_order):
                module = attrgetter(linear_attr)(mixed_layer)
                state = clique_state[name]
    
                grad_hatW = module.weight.grad.detach().to(module.weight.dtype)
                t0 = time.perf_counter()
                if args.incoh_mode == 'had': # since indices are based on the incoherence processed weights, need to transform gradients too
                    grad_Wr = quip.RHT_W(grad_hatW, state['SU'].to(device), state['SV'].to(device))
                elif args.incoh_mode == 'kron':
                    grad_Wr = state['SV'].to(device) @ grad_hatW @ state['SU'].to(device).T
                elif args.incoh_mode == 'had_left':
                    grad_Wr = utils.matmul_hadUt_cuda(grad_hatW * state['SU'].to(device), state['hadK'], state['K_had'])
                else:
                    raise NotImplementedError
                torch.cuda.synchronize(device)
                total_time_grad_incoh_proc += time.perf_counter() - t0

                t_before_grad_reshaping = time.perf_counter()
                m, n = state['orig_shape']
                d = state['d']
                grad_blocks = grad_Wr.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
                torch.cuda.synchronize(device)
                total_time_grad_reshaping += time.perf_counter() - t_before_grad_reshaping
                t_before_ihbc = time.perf_counter()
                grad_coeffs = quip.ihbc_transform(grad_blocks.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d // 8, 8)
                torch.cuda.synchronize(device)
                total_time_ihbc += time.perf_counter() - t_before_ihbc

                t_before_momentum_update = time.perf_counter()
                if state['momentum'] is None:
                    state['momentum'] = torch.zeros_like(grad_coeffs)
                state['momentum'] = momentum_rate * state['momentum'] + (1 - momentum_rate) * grad_coeffs
                torch.cuda.synchronize(device)
                total_time_momentum_update += time.perf_counter() - t_before_momentum_update

                t0 = time.perf_counter()
                neighbors_table = state['neighbors_table']
                valid_bits = state['valid_bits']

                P, Qd, G, _ = state['momentum'].shape
                mom = state['momentum'].reshape(-1, 8)
                Qflat = state['Qidxs'].reshape(-1)

                time_before_e8_best_valid = time.perf_counter()
                best_score, dir_idx = e8_best_valid(mom.contiguous(), Qflat.long(), valid_bits) # kernel, validity baked in
                # dir_idx is -1 (score +inf) wherever no valid direction exists; clamp
                # before gathering so that never-taken row doesn't index out of range.
                landing = neighbors_table[Qflat.long(), dir_idx.clamp(min=0).long()]
                torch.cuda.synchronize(device)
                total_time_e8_best_valid += time.perf_counter() - time_before_e8_best_valid

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

                # for i in range(m // d):
                #     for j in range(n // d):
                #         Q_curr = state['Qidxs'][i, j]
                #         momentum_curr = state['momentum'][i, j]
                #         best_scores = torch.full((momentum_curr.shape[0],), float('inf'), device=device)
                #         best_directions = torch.full((momentum_curr.shape[0],), -1, dtype=Q_curr.dtype, device=device)
                #         for k in range(momentum_curr.shape[0]):
                #             momentum_opposite = torch.where(momentum_curr[k] >= 0, torch.tensor(-1.0, dtype=momentum_curr.dtype, device=momentum_curr.device), torch.tensor( 1.0, dtype=momentum_curr.dtype, device=momentum_curr.device))
                #             top2 = torch.topk(momentum_curr[k].abs(), 2)
                #             proj112 = torch.zeros_like(momentum_curr[k])
                #             proj112[top2.indices[0]] = momentum_opposite[top2.indices[0]]
                #             proj112[top2.indices[1]] = momentum_opposite[top2.indices[1]]
                #             score112 = (momentum_curr[k] * proj112).sum()
                #             signs = momentum_opposite.clone()
                #             num_minus = (signs < 0).sum()
                #             if (num_minus % 2) == 1:
                #                 mins = momentum_curr[k].abs().argmin()
                #                 signs[mins] = -signs[mins]
                #             proj128 = 0.5 * signs
                #             score128 = (momentum_curr[k] * proj128).sum()
                #             if score112 <= score128:
                #                 best_dir = proj112
                #                 best_score = score112
                #             else:
                #                 best_dir = proj128
                #                 best_score = score128
                #             best_dir_val = directions_lookup[tuple((best_dir * 2).round().to(torch.long).tolist())]
                #             neighbor = neighbors_table[Q_curr[k].long(), best_dir_val]
                #             if neighbor.item() < 0:
                #                 scores = momentum_curr[k] @ directions.T
                #                 neighbors = neighbors_table[Q_curr[k].long()]
                #                 scores = scores.masked_fill(neighbors < 0, float('inf'))
                #                 best_score, best_dir = scores.min(dim=-1)
                #                 best_dir = best_dir.item()
                #                 neighbor = neighbors_table[Q_curr[k].long(), best_dir]
                #                 if neighbor.item() < 0:
                #                     continue
                #             best_scores[k]  = best_score
                #             best_directions[k] = neighbor
                #         values, inds = torch.topk(best_scores, args.sparse_ft_num_flips, largest=False)
                #         inds = inds[values < 0]
                #         if inds.numel() == 0:
                #             continue
                #         Q_curr[inds] = best_directions[inds]

                # for i in range(m // d):
                #     for j in range(n // d):
                #         Q_curr = state['Qidxs'][i, j]
                #         momentum_curr = state['momentum'][i, j]
                #         scores = momentum_curr @ directions.T
                #         neighbors = neighbors_table[Q_curr.long()]
                #         scores = scores.masked_fill(neighbors < 0, float('inf'))
                #         best_scores, best_directions = scores.min(dim=-1)
                #         values, inds = torch.topk(best_scores, args.sparse_ft_num_flips, largest=False)
                #         inds = inds[values < 0]
                #         if inds.numel() == 0: 
                #             continue
                #         Q_curr[inds] = neighbors_table[Q_curr[inds].long(), best_directions[inds]].long()
                torch.cuda.synchronize(device)
                total_time_flip_search += time.perf_counter() - t0

                t0 = time.perf_counter()
                Wscale = state['SV'].abs().mean()
                total_scale = state['Xscale'] * Wscale
                grid = state['codebook'].grid
                coeffs = grid[state['Qidxs'].long()].reshape((m // d) * (n // d), d * d)
                t_before_hbc = time.perf_counter()
                blocks = quip.hbc_transform((coeffs * total_scale)).reshape(m // d, n // d, d, d)
                torch.cuda.synchronize(device)
                total_time_hbc += time.perf_counter() - t_before_hbc
                hatWr = blocks.permute(0, 2, 1, 3).reshape(state['orig_shape'])
                torch.cuda.synchronize(device)
                total_time_weight_blocks_reshaping += time.perf_counter() - t0

                t_before_weight_incoh = time.perf_counter()
                # new_hatW = quip.incoherence_process(hatWr, state['SU'].to(f'cuda:{device}'), state['SV'].sign().to(f'cuda:{device}'), state.get('scaleWH'), args)
                if args.incoh_mode == 'had_left':
                    new_hatW = utils.matmul_hadU_cuda(hatWr, state['hadK'], state['K_had']) * state['SU'].to(device)
                    if args.rescale_WH:
                        new_hatW = new_hatW / state['scaleWH'].to(device)[None, :]
                else:
                    new_hatW = quip.incoherence_process(hatWr, state['SU'].to(device), state['SV'].sign().to(device), state.get('scaleWH'), args)
                new_hatW = new_hatW.to(module.weight.dtype)
                torch.cuda.synchronize(device)
                total_time_weight_incoh_proc += time.perf_counter() - t_before_weight_incoh

                t_before_weight_pieces = time.perf_counter()
                curr = 0
                pieces = []
                for shape, scale in zip(state['shapes'], state['scales']):
                    pieces.append(new_hatW[curr:curr + shape[0]] * scale)
                    curr += shape[0]
                with torch.no_grad():
                    module.weight.data.copy_(torch.cat(pieces).to(module.weight.dtype))
                torch.cuda.synchronize(device)
                total_time_weight_pieces += time.perf_counter() - t_before_weight_pieces
        
        train_loss = total_loss / count
        train_error = error / count
        glog.info(f"epoch {epoch}, loss: {train_loss}, error: {train_error}")

        # t0 = time.perf_counter()
        # val_loss = 0
        # count = 0
        # with torch.no_grad():
        #     for source, target in valid_dl:
        #         source = source.to(device).float()
        #         target = target.to(device).float()
        #         output = mixed_layer(source, position_ids=torch.arange(source.shape[1], device=device).unsqueeze(0))[0]
        #         val_loss += torch.nn.MSELoss()(output, target).item()
        #         count += 1
        # torch.cuda.synchronize(device)
        # total_time_epoch_val += time.perf_counter() - t0

        # val_loss = val_loss / count
        # if val_loss < prev_loss:
        #     prev_loss = val_loss
        # glog.info(f"epoch {epoch}: val loss: {val_loss:.6e}, best val loss: {prev_loss:.6e}")

    total_func_time = time.perf_counter() - t_start_func
    glog.info(f"total forward backward time: {total_time_fwd_bwd:.4f}s")
    glog.info(f"total grad incoh proc time: {total_time_grad_incoh_proc:.4f}s")
    glog.info(f"total ihbc time: {total_time_ihbc:.4f}s")
    glog.info(f"total grad reshaping time: {total_time_grad_reshaping:.4f}s")
    glog.info(f"total momentum update time: {total_time_momentum_update:.4f}s")
    glog.info(f"total e8 best valid time: {total_time_e8_best_valid:.4f}s")
    glog.info(f"total flip search time: {total_time_flip_search:.4f}s")
    glog.info(f"total hbc time: {total_time_hbc:.4f}s")
    glog.info(f"total weight blocks reshaping time: {total_time_weight_blocks_reshaping:.4f}s")
    glog.info(f"total weight incoh proc time: {total_time_weight_incoh_proc:.4f}s")
    glog.info(f"total weight pieces time: {total_time_weight_pieces:.4f}s")
    # glog.info(f"total epoch validation time: {total_time_epoch_val:.4f}s")
    glog.info(f"total function time: {total_func_time:.4f}s")

    mixed_layer = mixed_layer.cpu()
    return mixed_layer
