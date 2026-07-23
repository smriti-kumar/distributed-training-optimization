import copy
import os
import math

import glog
import torch
from tqdm import tqdm
from itertools import product

from lib import utils


def RHT_H(H, SU):
    return utils.matmul_hadUt(utils.matmul_hadUt(H * SU).T * SU)


def RHT_W(W, SU, SV):
    return utils.matmul_hadUt(utils.matmul_hadUt(W.T * SV).T * SU)


def incoherence_preprocess(H, W, args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    device = H.device
    (m, n) = W.shape

    def _dump(Hr, Lhr, msg=''):
        torch.save(Hr, f"{args.save_pfx}/Hr_debug_fft.pt")
        torch.save(Lhr, f"{args.save_pfx}/Lhr_debug_fft.pt")
        raise Exception(msg)

    # diagonally rescale W,H to minimize proxy loss
    scaleWH = None
    Wr = W
    Hr = H
    if args.rescale_WH:
        Hr = H / H.abs().max()
        diagH = torch.diag(Hr)
        diagW2 = torch.diag(W.T @ W)
        diagH = torch.clamp(diagH, min=1e-8)
        diagW2 = torch.clamp(diagW2, min=1e-8)
        scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
        scaleWH = scaleWH.clamp(min=1e-8)
        Wr = Wr * scaleWH[None, :]
        Hr = Hr / scaleWH[None, :]
        Hr = Hr / scaleWH[:, None]
        scaleWH = scaleWH.cpu()

    # randomized hadamard transformation on H, W
    if args.incoh_mode == "had":
        SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(dtype_)
        SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(dtype_)
        Hr = RHT_H(Hr, SU)
        Wr = RHT_W(Wr, SU, SV)
    # randomized kronecker product on H, W
    elif args.incoh_mode == "kron":
        SU = utils.rand_ortho_butterfly_noblock(n).to(dtype_).to(device)
        SV = utils.rand_ortho_butterfly_noblock(m).to(dtype_).to(device)
        Hr = SU @ Hr @ SU.T
        Wr = SV @ Wr @ SU.T
    else:
        raise NotImplementedError
    SV = SV.cpu()
    SU = SU.cpu()

    Lhr = torch.linalg.cholesky(Hr)
    if not torch.all(torch.isfinite(Lhr)):
        return None

    Wr = Wr.to(device)

    return Lhr, Hr, Wr, SU, SV, scaleWH


def incoherence_process(hatWr, SU, SV, scaleWH, args):
    device = hatWr.device
    # reverse hadamard transformation
    if args.incoh_mode == 'had':
        hatWr = (utils.matmul_hadU(
            (utils.matmul_hadU(hatWr) * SU.to(device)).T) * SV.to(device)).T
    # reverse kronecker product
    elif args.incoh_mode == 'kron':
        hatWr = SV.T.to(device) @ hatWr @ SU.to(device)
    else:
        raise NotImplementedError

    # reverse rescale W,H
    if args.rescale_WH:
        hatWr /= scaleWH[None, :].to(device)

    assert torch.isfinite(hatWr).all()
    return hatWr


def low_rank_preprocess(Wr, Hr, Lhr, args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    if args.full_svd:
        svdZ = torch.linalg.svd(Wr.to(torch.float64) @ Lhr.to(torch.float64),
                                full_matrices=False)
        Hr -= (Lhr.to(torch.float64) @ svdZ.Vh.T[:, :args.lora_rank] @ \
                   svdZ.Vh[:args.lora_rank] @ Lhr.to(torch.float64).T).to(dtype_)
        Hr += torch.diag(Hr).mean() * args.sigma_reg2 * \
            torch.eye(Hr.shape[0], device=Hr.device, dtype=Hr.dtype)
        Wr -= (svdZ.U[:, :args.lora_rank] @ svdZ.U.T[:args.lora_rank] @ Wr.to(
            torch.float64)).to(dtype_)
    else:
        U_lrz, S_lrz, V_lrz = torch.svd_lowrank(
            Wr.to(torch.float64) @ Lhr.to(torch.float64),
            q=2 * args.lora_rank,
            niter=10)
        U_lrz = U_lrz[:, :args.lora_rank]
        V_lrz = V_lrz[:, :args.lora_rank]
        Hr -= (Lhr.to(torch.float64) @ V_lrz @ V_lrz.T @ Lhr.to(
            torch.float64).T).to(dtype_)
        Hr += torch.diag(Hr).mean() * args.sigma_reg2 * \
            torch.eye(Hr.shape[0], device=Hr.device, dtype=Hr.dtype)
        Wr -= (U_lrz @ U_lrz.T @ Wr.to(torch.float64)).to(dtype_)
    return Wr, Hr


def low_rank_process(Wo, hatWr, Lhr, args):
    # invLhr = torch.linalg.inv(Lhr)
    # assert torch.isfinite(invLhr).all()

    svdRZ = torch.linalg.svd((Wo - hatWr) @ Lhr, full_matrices=False)
    A = svdRZ.U[:, :args.lora_rank]
    # B = torch.diag(svdRZ.S[:args.lora_rank]) @ svdRZ.Vh[:args.lora_rank] @ invLhr
    B = torch.linalg.solve_triangular(
        Lhr,
        torch.diag(svdRZ.S[:args.lora_rank]) @ svdRZ.Vh[:args.lora_rank],
        upper=False,
        left=False)
    assert torch.isfinite(A).all() and torch.isfinite(B).all()

    svdB = torch.linalg.svd(B, full_matrices=False)
    A = (A @ svdB.U @ torch.diag(svdB.S.sqrt())).half()
    B = (torch.diag(svdB.S.sqrt()) @ svdB.Vh).half()

    hatWr = hatWr.to(A.device) + \
        (A @ B).to(torch.float64 if args.use_fp64 else torch.float32)
    return hatWr, A, B


def LDLQ(Wr, Hr, L, D, cb, args):
    '''
    want eta = (Wr - hatWr) @ L
    want hatWr + eta = Wr + (Wr - hatWr) @ (L - I)
    want hatWr = Q( Wr + (Wr - hatWr) @ (L - I) )
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    for k in reversed(range(n // cb.codesz)):
        WXWX = Wr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + \
            (Wr[:, (cb.codesz * (k + 1)):n] - hatWr[:, (cb.codesz * (k + 1)):n]) @ \
            L[(cb.codesz * (k + 1)):n, (cb.codesz * k):(cb.codesz * (k + 1))]
        hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))], Qidxs[:, k] = \
            cb.quantize(WXWX, resid_scale_override=args.resid_scale_override)
    for ie in range(args.quip_tune_iters):
        for k in reversed(range(n // cb.codesz)):
            WXWX = hatWr[:, (cb.codesz * k):(cb.codesz * (k + 1))] + (Wr - hatWr) @ \
                Hr[:, (cb.codesz * k):(cb.codesz * (k + 1))] @ \
                torch.linalg.inv(Hr[(cb.codesz * k):(cb.codesz * (k + 1)),
                                    (cb.codesz * k):(cb.codesz * (k + 1))])
            hatWr[:, (cb.codesz *
                      k):(cb.codesz * (k + 1))], Qidxs[:, k] = cb.quantize(
                          WXWX, resid_scale_override=args.resid_scale_override)

    return hatWr, Qidxs


def LDLQ_buffered(Wr, Hr, L, D, cb, args, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of codesz (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    assert buf_cols % cb.codesz == 0
    assert n % buf_cols == 0
    buf_size = buf_cols // cb.codesz

    hatWr_T = torch.zeros(n, m, dtype=Hr.dtype, device=Hr.device)
    Qidxs_T = torch.zeros(n // cb.codesz,
                          m,
                          dtype=cb.idx_dtype,
                          device=Hr.device)

    device = Wr.device
    Wr = Wr.cpu()
    Hr = Hr.cpu()
    utils.clean()
    Wr_T = Wr.T.contiguous().to(device)
    Hr_T = Hr.T.contiguous().to(device)

    # quip
    prod_cache = torch.zeros(n, m, dtype=Wr_T.dtype, device=Wr_T.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr_T = Wr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col].contiguous()
        b_prod = prod_cache[cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)].T @ \
                (b_Wr_T[cb.codesz * (i + 1):] - b_hatWr_T[cb.codesz * (i + 1):]) + \
                b_prod[cb.codesz * i : cb.codesz * (i + 1)]
            q_out = cb.quantize(WXWX.T,
                                resid_scale_override=args.resid_scale_override)
            b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
            b_Qidxs_T[i] = q_out[1]

        prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)
        hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                cur_col] = b_hatWr_T

    del b_Wr_T, b_hatWr_T, b_L, b_prod, L_offset, prod_cache
    utils.clean()

    # tune
    for ie in range(args.quip_tune_iters):
        # recompute delta to minimize errors
        delta_T = Wr_T - hatWr_T
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr_T = hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Hr_T = Hr_T[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta_T = delta_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                                cur_col]
            b_Qidxs_T = Qidxs_T[cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        torch.linalg.inv(b_Hr_T[cb.codesz * i : cb.codesz * (i + 1), Hr_offset + cb.codesz * i : Hr_offset + cb.codesz * (i + 1)].T).T @ b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                else:
                    WXWX = b_hatWr_T[cb.codesz * i : cb.codesz * (i + 1)] + \
                        (1/b_Hr_T[i, Hr_offset + i]) * b_Hr_T[cb.codesz * i : cb.codesz * (i + 1)] @ delta_T
                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] += b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]

                if ie < args.quip_tune_iters - 1:
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                        WXWX.T,
                        return_idx=False,
                        resid_scale_override=args.resid_scale_override).T
                else:
                    q_out = cb.quantize(
                        WXWX.T, resid_scale_override=args.resid_scale_override)
                    b_hatWr_T[cb.codesz * i:cb.codesz * (i + 1)] = q_out[0].T
                    b_Qidxs_T[i] = q_out[1]

                b_delta_T[cb.codesz * i:cb.codesz *
                          (i + 1)] -= b_hatWr_T[cb.codesz * i:cb.codesz *
                                                (i + 1)]
            hatWr_T[cb.codesz * (cur_col - buf_size):cb.codesz *
                    cur_col] = b_hatWr_T
            Qidxs_T[cur_col - buf_size:cur_col] = b_Qidxs_T

        del delta_T, b_hatWr_T, b_Hr_T, b_delta_T, b_Qidxs_T, Hr_offset
        utils.clean()

    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()


def LDLQ_buffered_lowmem(Wr, Hr, L, D, cb, args, buf_cols=128):
    '''
    reduce overhead of memory r/w
    buffer size is in groups of code_col (4) columns (for D4)
    '''
    (m, n) = Wr.shape
    hatWr = torch.zeros(m, n, dtype=Hr.dtype, device=Hr.device)
    Qidxs = torch.zeros(m,
                        n // cb.codesz,
                        dtype=cb.idx_dtype,
                        device=Hr.device)
    assert n % buf_cols == 0 and buf_cols % cb.codesz == 0
    buf_size = buf_cols // cb.codesz

    # quip
    prod_cache = torch.zeros(m, n, dtype=Wr.dtype, device=Wr.device)
    for cur_col in range(n // cb.codesz, 0, -buf_size):
        b_Wr = Wr[:, cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_hatWr = hatWr[:,
                        cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_L = L[cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
        b_prod = prod_cache[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
        b_Qidxs = Qidxs[:, cur_col - buf_size:cur_col]
        L_offset = cb.codesz * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr[:, cb.codesz * i : cb.codesz * (i + 1)] + \
                (b_Wr[:, cb.codesz * (i + 1):] - b_hatWr[:, cb.codesz * (i + 1):]) @ \
                b_L[cb.codesz * (i + 1):, L_offset + cb.codesz * i : L_offset + cb.codesz * (i + 1)] + \
                b_prod[:, cb.codesz * i : cb.codesz * (i + 1)]
            b_hatWr[:, cb.codesz * i:cb.codesz *
                    (i + 1)], b_Qidxs[:, i] = cb.quantize(
                        WXWX, resid_scale_override=args.resid_scale_override)
        prod_cache += (b_Wr - b_hatWr) @ b_L

    del b_Wr, b_hatWr, b_L, b_prod, L_offset, prod_cache
    utils.clean()

    # tune
    for ie in range(args.quip_tune_iters):
        # recompute delta to minimize errors
        delta = Wr - hatWr
        for cur_col in range(n // cb.codesz, 0, -buf_size):
            b_hatWr = hatWr[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
            b_Hr = Hr[:, cb.codesz * (cur_col - buf_size):cb.codesz * cur_col]
            b_delta = delta[:, cb.codesz * (cur_col - buf_size):cb.codesz *
                            cur_col]
            b_Qidxs = Qidxs[:, cur_col - buf_size:cur_col]
            Hr_offset = cb.codesz * (cur_col - buf_size)
            for i in reversed(range(buf_size)):
                if cb.codesz > 1:
                    inv = torch.linalg.inv(
                        b_Hr[Hr_offset + cb.codesz * i:Hr_offset + cb.codesz *
                             (i + 1), cb.codesz * i:cb.codesz * (i + 1)])
                else:
                    inv = 1 / b_Hr[Hr_offset + i:Hr_offset + i + 1, i:i + 1]

                WXWX = b_hatWr[:, cb.codesz * i : cb.codesz * (i + 1)] + \
                    delta @ b_Hr[:, cb.codesz * i : cb.codesz * (i + 1)] @ inv

                b_delta[:, cb.codesz * i:cb.codesz *
                        (i + 1)] += b_hatWr[:,
                                            cb.codesz * i:cb.codesz * (i + 1)]

                if ie < args.quip_tune_iters - 1:
                    b_hatWr[:,
                            cb.codesz * i:cb.codesz * (i + 1)] = cb.quantize(
                                WXWX,
                                return_idx=False,
                                resid_scale_override=args.resid_scale_override)
                else:
                    b_hatWr[:, cb.codesz * i:cb.codesz *
                            (i + 1)], b_Qidxs[:, i] = cb.quantize(
                                WXWX,
                                resid_scale_override=args.resid_scale_override)

                b_delta[:, cb.codesz * i:cb.codesz *
                        (i + 1)] -= b_hatWr[:,
                                            cb.codesz * i:cb.codesz * (i + 1)]
        del delta, b_hatWr, b_Hr, b_delta, b_Qidxs, Hr_offset
        utils.clean()

    return hatWr, Qidxs

def get_cliques(device):
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
    cperm = torch.tensor(cliques).view(-1).to(device)
    ciperm = cperm.sort().indices.to(device)
    return cperm, ciperm

def hbc_transform(x):
    _, ciperm = get_cliques(x.device)
    mm = x.shape[:-1]
    x = x.view(-1, x.shape[-1])
    m, n = x.shape
    k = 1
    while 4**k < n: k += 1
    assert 4**k == n
    b = torch.tensor([1,1,-1,-1], dtype=x.dtype, device=x.device)
    x = x.reshape(-1,64)[:,ciperm]
    x = x.reshape((m,) + (4,)*k)
    for i in range(k):
        x = x.flip(1+i) + x * b.view((1,)*(i+1)+(4,)+(1,)*(k-1-i))
    x = x.reshape((m,) + (2,2)*k)
    x = x.permute((0,)+tuple(2*i+1 for i in range(k))+tuple(2*i+2 for i in range(k)))
    return x.reshape(mm + (2**k, 2**k)) / (2**(k/2))

def ihbc_transform(x):
    cperm, _ = get_cliques(x.device)
    mm = x.shape[:-2]
    x = x.view(-1, x.shape[-2], x.shape[-1])
    m, n, n_ = x.shape
    assert(n == n_)
    k = 1
    while 2**k < n: k += 1
    assert 2**k == n
    x = x.reshape((m,) + (2,2)*k)
    x = x.permute((0,)+tuple((i>>1)+1+(i&1)*k for i in range(2*k)))
    b = torch.tensor([1,1,-1,-1], dtype=x.dtype, device=x.device)
    x = x.reshape((m,) + (4,)*k)
    for i in range(k):
        x = x.flip(1+i) + x * b.view((1,)*(i+1)+(4,)+(1,)*(k-1-i))
    x = x.reshape(-1,64)[:,cperm]
    return x.reshape(mm + (4**k,)) / (2**(k/2))

def H_multiply(X,H):
    X = hbc_transform(X)
    X = X @ H
    X = ihbc_transform(X)
    return X

def HL_multiply(X,H):
    Xshape = X.shape
    n = X.shape[-1]
    Z = X.clone()
    Z = Z.reshape(-1,n)
    Y = torch.zeros_like(Z)
    for i in range(n):
        Y[:,i] = H_multiply(Z,H)[:,i]
        Z[:,i].zero_()
    return Y.reshape(Xshape)

def decompose_H(H):
    (n, n_) = H.shape
    assert(n_ == n)
    decomp = []
    while n > 8:
        next_n = n // 2
        Ha, Hb = H[:next_n, :next_n], H[:next_n, next_n:]
        Hc, Hd = H[next_n:, :next_n], H[next_n:, next_n:]
        H_decomp = [(Ha+Hd)/2, (Hb+Hc)/2, (Hb-Hc)/2, (Ha-Hd)/2]
        decomp.append(H_decomp)
        H = H_decomp[0]
        n = next_n
    assert(n == 8)
    # lift H
    X = hbc_transform(torch.eye(64, dtype=H.dtype, device=H.device))
    X = X @ H
    X = ihbc_transform(X)
    decomp.append(torch.tril(X))
    return decomp

def fast_HL_multiply(X,H):
    return fast_HL_multiply_sub(X,decompose_H(H))

def fast_HL_multiply_sub(X,DEC):
    if len(DEC) == 1:
        (Y,) = DEC
        return X @ Y
    (H0,H1,H2,H3) = DEC[0]
    Xshape = X.shape
    n = X.shape[-1]
    X = X.reshape(-1,4,n//4)
    Y = fast_HL_multiply_sub(X,DEC[1:])
    Y[:,0,:] += H_multiply(X[:,1,:],H1) + H_multiply(X[:,3,:],H3) - H_multiply(X[:,2,:],H2)
    Y[:,1,:] += H_multiply(X[:,3,:],H2) - H_multiply(X[:,2,:],H3)
    Y[:,2,:] += H_multiply(X[:,3,:],H1)
    return Y.reshape(Xshape)

def ldlq_helper(X, future_error, DEC, cb, Qidxs):
    m, n = X.shape
    if len(DEC) == 1:
        (J,) = DEC
        hatX = torch.zeros_like(X)
        error = torch.zeros_like(X)
        for c in range(7, -1, -1):
            start = 8 * c
            end = 8 * c + 8
            Y = X[:, start:end] + future_error[:, start:end]
            if end < 64:
                Y = Y + error[:, end:] @ J[end:, start:end]
            q_hat, q_idx = cb.quantize(Y.float())
            q_hat = q_hat.to(X.dtype)
            hatX[:, start:end] = q_hat
            error[:, start:end] = X[:, start:end] - q_hat
            Qidxs[:, c] = q_idx.reshape(m)
        return hatX
    (H0, H1, H2, H3) = DEC[0]
    X = X.reshape(m, 4, n // 4)
    future_error = future_error.reshape(m, 4, n // 4)
    X = [X[:, i].contiguous() for i in range(4)]
    future_error = [future_error[:, i].contiguous() for i in range(4)]
    QI = [Qidxs[:, i * (n // 32):(i + 1) * (n // 32)] for i in range(4)]
    res3 = ldlq_helper(X[3], future_error[3], DEC[1:], cb, QI[3])
    error3 = X[3] - res3
    future_error2 = future_error[2] + H_multiply(error3, H1)
    res2 = ldlq_helper(X[2], future_error2, DEC[1:], cb, QI[2])
    error2 = X[2] - res2
    future_error1 = future_error[1] + H_multiply(error3, H2) - H_multiply(error2, H3)
    res1 = ldlq_helper(X[1], future_error1, DEC[1:], cb, QI[1])
    error1 = X[1] - res1
    future_error0 = future_error[0] + H_multiply(error1, H1) - H_multiply(error2, H2) + H_multiply(error3, H3)
    res0 = ldlq_helper(X[0], future_error0, DEC[1:], cb, QI[0])
    return torch.stack([res0, res1, res2, res3], dim=1).reshape(m, n)

def bulk_LDLQ(X, A, cb, H, passes):
    m, n = X.shape
    d = 8
    while m % (2 * d) == 0 and n % (2 * d) == 0:
        d *= 2
    X = X.reshape(m // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    coeffs = ihbc_transform(X.reshape((m // d) * (n // d), d, d)).reshape(m // d, n // d, d * d)
    Xscale = coeffs.square().mean().sqrt() / cb.opt_scale
    coeffs = coeffs / Xscale
    Qidxs = torch.zeros(m // d, n // d, d * d // 8, dtype=cb.idx_dtype, device=coeffs.device)
    A = A.reshape(n // d, d, n // d, d).permute(0, 2, 1, 3).contiguous().clone()
    for i in range(n // d):
        A[:, i] = A[:, i] / A[i, i].diagonal().mean()
    DEC = [decompose_H(A[i, i].contiguous()) for i in range(n // d)]

    hat_coeffs = torch.zeros_like(coeffs)
    error = torch.zeros_like(coeffs)
    for i in range(n // d - 1, -1, -1):
        future_error = torch.zeros_like(coeffs[:, i])
        for j in range(i + 1, n // d):
            future_error += H_multiply(error[:, j], A[j, i])
        hat_coeffs[:, i] = ldlq_helper(coeffs[:, i], future_error, DEC[i], cb, Qidxs[:, i])
        error[:, i] = coeffs[:, i] - hat_coeffs[:, i]
    
    if passes > 0:
        H = H.reshape(n // d, d, n // d, d).permute(0, 2, 1, 3).contiguous().clone()
        for i in range(n // d):
            H[:, i] = H[:, i] / H[i, i].diagonal().mean()
        DEC = [decompose_H(H[i, i].contiguous()) for i in range(n // d)]
        for _ in range(passes):
            total_error = hat_coeffs - coeffs
            hat_coeffs = torch.zeros_like(coeffs)
            error = torch.zeros_like(coeffs)
            for i in range(n // d - 1, -1, -1):
                future_error = torch.zeros_like(coeffs[:, i])
                for j in range(i + 1, n // d):
                    future_error += H_multiply(error[:, j], H[j, i])
                for j in range(0, i):
                    future_error -= H_multiply(total_error[:, j], H[j, i])
                future_error -= (H_multiply(total_error[:, i], H[i, i]) - fast_HL_multiply_sub(total_error[:, i], DEC[i]))
                hat_coeffs[:, i] = ldlq_helper(coeffs[:, i], future_error, DEC[i], cb, Qidxs[:, i])
                error[:, i] = coeffs[:, i] - hat_coeffs[:, i]

    hatW = hbc_transform((hat_coeffs * Xscale).reshape((m // d) * (n // d), d * d)).reshape(m // d, n // d, d, d)
    return hatW.permute(0, 2, 1, 3).reshape(m, n), Qidxs.reshape(m, n // 8)

def bulk_ldlq_wrapper(Wr, Hr, codebook, args, device='cpu'):
    m, n = Wr.shape
    S, V = torch.linalg.eigh(Hr.double())
    H_sqrt = (V @ torch.diag(torch.sqrt(torch.clamp(S, min=0))) @ V.T).to(Wr.dtype)
    del S, V
    utils.clean()
    return bulk_LDLQ(Wr, H_sqrt, codebook, Hr, getattr(args, 'greedy_passes', 0))

def quantize(H_orig, W_orig, rank, codebook_orig, args, device='cpu'):
    glog.info("at the top of quantize")
    orig_device = H_orig.device
    W_orig_dtype = W_orig.dtype
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    (m, n) = W_orig.shape

    H = H_orig.clone().to(dtype_).to(device)
    W = W_orig.clone().to(dtype_).to(device)
    codebook = copy.deepcopy(codebook_orig).to(dtype_)

    assert (m % 8 == 0)
    assert (n % 8 == 0)
    assert (torch.all(torch.isfinite(H.cpu())))
    assert (torch.all(torch.isfinite(W.cpu())))

    # incoherence preprocessing
    if args.incoh_mode != 'none':
        incoh_out = incoherence_preprocess(H, W, args)
        if incoh_out is None:
            if args.use_fp64:
                raise Exception
            new_args = copy.deepcopy(args)
            new_args.use_fp64 = True
            glog.info('incoherence_preprocess failed, recomputing in fp64')
            del H, W, codebook
            utils.clean()
            return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)
        glog.info("done with incoherence processing")

        Lhr, Hr, Wr, SU, SV, scaleWH = incoh_out
        del incoh_out
        utils.clean()
    else:
        SU = torch.ones(n, dtype=dtype_, device=device)
        SV = torch.ones(m, dtype=dtype_, device=device)

        scaleWH = None
        Wr = W
        Hr = H

        if args.rescale_WH:
            Hr = H / H.abs().max()
            diagH = torch.diag(Hr)
            diagW2 = torch.diag(W.T @ W)
            diagH = torch.clamp(diagH, min=1e-8)
            diagW2 = torch.clamp(diagW2, min=1e-8)
            scaleWH = (diagH / diagW2).sqrt().sqrt().to(torch.float32)
            scaleWH = scaleWH.clamp(min=1e-8)
            Wr = Wr * scaleWH[None, :]
            Hr = Hr / scaleWH[None, :]
            Hr = Hr / scaleWH[:, None]
            scaleWH = scaleWH.cpu()

        SV = SV.cpu()
        SU = SU.cpu()

        Lhr = torch.linalg.cholesky(Hr)
        if not torch.all(torch.isfinite(Lhr)):
            return None
        
        Wr = Wr.to(device)

    glog.info(f'mean square of W: {W.square().mean()}')
    glog.info(f'mean square of Wr: {Wr.square().mean()}')
    glog.info(f'difference between Hr and Hr.T: {((Hr - Hr.T).abs().max())}')
    glog.info(f'max abs of Hr: {((Hr.abs().max()))}')
    glog.info(f'min diag of Lhr: {Lhr.diag().min().item()}')

    Wo = Wr.clone()

    # remove low rank components before LDLQ
    if args.lora_rank > 0:
        Wr, Hr = low_rank_preprocess(Wr, Hr, Lhr, args)

    Wscale = Wr.square().mean().sqrt()
    if args.scale_override > 0:
        Wscale /= args.scale_override
    else:
        Wscale /= codebook.opt_scale
    Wr = Wr / Wscale
    glog.info("scaled Wr")
    codebook = codebook.to(device)
    glog.info("created codebook")
    
    glog.info("right before calling orth quantize from quantize")
    # hatWr, Qidxs = clique_quantize(Wr, codebook, device)
    # hatWr, Qidxs, Qidxs_blocks = clique_quantize_rounding(Wr, Hr, codebook, device)
    hatWr, Qidxs = bulk_ldlq_wrapper(Wr, Hr, codebook, args, device)
    glog.info("right after calling orth quantize from quantize")
    
    Wr = Wr.cpu()
    Hr = Hr.cpu()
    utils.clean()
    
    hatWr = hatWr * Wscale

    # low rank correction
    if args.lora_rank > 0:
        hatWr, A, B = low_rank_process(Wo, hatWr, Lhr, args)
        A = A.half().cpu()
        B = B.half().cpu()
    else:
        A, B = None, None

    # reverse incoherence process
    if args.incoh_mode != 'none':
        hatW = incoherence_process(hatWr, SU, SV, scaleWH, args)
        glog.info("reverse incoherence processing done")
    else:
        hatW = hatWr

    Qidxs = codebook.maybe_pack_idxs(Qidxs)

    attr = {
        'Qidxs': Qidxs.to(orig_device),
        'hatW': hatW.half().to(orig_device),
        'A': A,
        'B': B,
        'SU': SU.to(torch.float16).to(orig_device),
        'SV': (SV * Wscale.to(SV.device)).to(
            torch.float16).to(orig_device),  # fuse Wscale into SV
        'scaleWH': scaleWH,
        'hatWr': hatWr.to(orig_device),
        # 'Qidxs_blocks': Qidxs_blocks.to(orig_device),
    }

    utils.clean()

    glog.info("right before returning from quantize")
    return hatW.to(W_orig_dtype).to(orig_device), attr

def original_quantize(H_orig, W_orig, rank, codebook_orig, args, device='cpu'):
    orig_device = H_orig.device
    W_orig_dtype = W_orig.dtype
    dtype_ = torch.float64 if args.use_fp64 else torch.float32
    (m, n) = W_orig.shape

    H = H_orig.clone().to(dtype_).to(device)
    W = W_orig.clone().to(dtype_).to(device)
    codebook = copy.deepcopy(codebook_orig).to(dtype_)

    assert (m % 2 == 0)
    assert (n % 4 == 0)
    assert (torch.all(torch.isfinite(H.cpu())))
    assert (torch.all(torch.isfinite(W.cpu())))

    # incoherence preprocessing
    incoh_out = incoherence_preprocess(H, W, args)
    if incoh_out is None:
        if args.use_fp64:
            raise Exception
        new_args = copy.deepcopy(args)
        new_args.use_fp64 = True
        glog.info('incoherence_preprocess failed, recomputing in fp64')
        del H, W, codebook
        utils.clean()
        return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)

    Lhr, Hr, Wr, SU, SV, scaleWH = incoh_out
    del incoh_out
    utils.clean()

    glog.info(f'mean square of W: {W.square().mean()}')
    glog.info(f'mean square of Wr: {Wr.square().mean()}')
    glog.info(f'difference between Hr and Hr.T: {((Hr - Hr.T).abs().max())}')
    glog.info(f'max abs of Hr: {((Hr.abs().max()))}')
    glog.info(f'min diag of Lhr: {Lhr.diag().min().item()}')

    Wo = Wr.clone()

    # remove low rank components before LDLQ
    if args.lora_rank > 0:
        Wr, Hr = low_rank_preprocess(Wr, Hr, Lhr, args)

    # block LDL
    block_LDL_out = utils.block_LDL(Hr, codebook.codesz)
    if block_LDL_out is None:
        if args.use_fp64:
            raise Exception
        new_args = copy.deepcopy(args)
        new_args.use_fp64 = True
        glog.info('block_LDL failed, recomputing in fp64')
        del H, W, codebook, Lhr, Hr, Wr, SU, SV, scaleWH, Wo
        utils.clean()
        return quantize(H_orig, W_orig, rank, codebook_orig, new_args, device)

    L, D = block_LDL_out
    del block_LDL_out
    del H_orig, W_orig, codebook_orig
    utils.clean()

    # LDLQ
    Wscale = Wr.square().mean().sqrt()
    if args.scale_override > 0:
        Wscale /= args.scale_override
    else:
        Wscale /= codebook.opt_scale
    Wr = Wr / Wscale
    codebook = codebook.to(device)
    if args.no_use_buffered:
        hatWr, Qidxs = LDLQ(Wr, Hr, L, D, codebook, args)
    elif args.lowmem_ldlq or args.use_fp64:
        hatWr, Qidxs = LDLQ_buffered_lowmem(Wr,
                                            Hr,
                                            L,
                                            D,
                                            codebook,
                                            args,
                                            buf_cols=128)
    else:
        hatWr, Qidxs = LDLQ_buffered(Wr,
                                     Hr,
                                     L,
                                     D,
                                     codebook,
                                     args,
                                     buf_cols=128)

    Wr = Wr.cpu()
    Hr = Hr.cpu()
    L = L.cpu()
    D = D.cpu()
    del Wr, Hr, L, D
    utils.clean()

    hatWr = hatWr * Wscale

    # low rank correction
    if args.lora_rank > 0:
        hatWr, A, B = low_rank_process(Wo, hatWr, Lhr, args)
        A = A.half().cpu()
        B = B.half().cpu()
    else:
        A, B = None, None

    # reverse incoherence process
    hatW = incoherence_process(hatWr, SU, SV, scaleWH, args)

    Qidxs = codebook.maybe_pack_idxs(Qidxs)

    attr = {
        'Qidxs': Qidxs.to(orig_device),
        'A': A,
        'B': B,
        'SU': SU.to(torch.float16).to(orig_device),
        'SV': (SV * Wscale.to(SV.device)).to(
            torch.float16).to(orig_device),  # fuse Wscale into SV
        'scaleWH': scaleWH,
    }

    utils.clean()

    return hatW.to(W_orig_dtype).to(orig_device), attr

def quantize_linear(weights, save_path, hessian_path, cb, args, device='cpu'):
    glog.info("top of quantize linear")
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    shapes = [_.shape for _ in weights]
    scales = [_.to(dtype_).square().mean().sqrt() for _ in weights]

    if os.path.exists(save_path):
        return

    H_data = torch.load(hessian_path, map_location=torch.device('cpu'))
    H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    H.add_(mu[None, :] * mu[:, None])
    n = H_data['n']
    # H = torch.eye(weights[0].shape[1], dtype=dtype_, device=device)
    glog.info("set H to identity")
    W = torch.vstack([
        weights[i].to(dtype_) / scales[i] for i in range(len(weights))
    ]).to(dtype_)
    H = utils.regularize_H(H, n, args.sigma_reg)
    glog.info("right before calling quantize from quantize linear")
    hatW, attr = quantize(H, W, args.lora_rank, cb, args, device)
    glog.info("right after calling quantize from quantize linear")
    # if len(scales) == 1:
    #     # fuse single scale into SV too
    #     attr['SV'] *= scales[0]
    #     scales = [1.0]
    attr.update({
        'fused': len(shapes) > 1,
        'shapes': shapes,
        'scales': scales,
    })
    torch.save(attr, save_path)
    glog.info("saved weights to save path")
    # utils.show_metrics(hatW, W, H.cpu().to(dtype_), save_path)
    utils.clean()
