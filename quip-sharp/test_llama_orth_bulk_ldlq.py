import torch
import argparse
from transformers import AutoModelForCausalLM
from lib import codebook, utils
from lib.algo.quip import RHT_H, RHT_W
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, required=True)
parser.add_argument('--hessian_path', type=str, required=True)
parser.add_argument('--layer_i', type=int, default=1)
parser.add_argument('--sublayer', type=str, default='o', choices=['qkv','o','up','down'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def hbc_transform(x):
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

def ldlq_helper(X, future_error, DEC, cb):
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
            quantized = cb.quantize(Y.float())[0].to(torch.float32)
            hatX[:, start:end] = quantized
            error[:, start:end] = X[:, start:end] - quantized
        return hatX
    (H0, H1, H2, H3) = DEC[0]
    X = X.reshape(m, 4, n // 4)
    future_error = future_error.reshape(m, 4, n // 4)
    X = [X[:, i].contiguous() for i in range(4)]
    future_error = [future_error[:, i].contiguous() for i in range(4)]
    res3 = ldlq_helper(X[3], future_error[3], DEC[1:], cb)
    error3 = X[3] - res3
    future_error2 = future_error[2] + H_multiply(error3, H1)
    res2 = ldlq_helper(X[2], future_error2, DEC[1:], cb)
    error2 = X[2] - res2
    future_error1 = future_error[1] + H_multiply(error3, H2) - H_multiply(error2, H3)
    res1 = ldlq_helper(X[1], future_error1, DEC[1:], cb)
    error1 = X[1] - res1
    future_error0 = future_error[0] + H_multiply(error1, H1) - H_multiply(error2, H2) + H_multiply(error3, H3)
    res0 = ldlq_helper(X[0], future_error0, DEC[1:], cb)
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
    A = A.reshape(n // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
    for i in range(n // d):
        A[:, i] = A[:, i] / A[i, i].diagonal().mean()
    DEC = [decompose_H(A[i, i].contiguous()) for i in range(n // d)]

    hat_coeffs = torch.zeros_like(coeffs)
    error = torch.zeros_like(coeffs)
    for i in range(n // d - 1, -1, -1):
        future_error = torch.zeros_like(coeffs[:, i])
        for j in range(i + 1, n // d):
            future_error += H_multiply(error[:, j], A[j, i])
        hat_coeffs[:, i] = ldlq_helper(coeffs[:, i], future_error, DEC[i], cb)
        error[:, i] = coeffs[:, i] - hat_coeffs[:, i]

    H = H.reshape(n // d, d, n // d, d).permute(0, 2, 1, 3).contiguous()
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
            hat_coeffs[:, i] = ldlq_helper(coeffs[:, i], future_error, DEC[i], cb)
            error[:, i] = coeffs[:, i] - hat_coeffs[:, i]

    hatW = hbc_transform((hat_coeffs * Xscale).reshape((m // d) * (n // d), d * d)).reshape(m // d, n // d, d, d)
    return hatW.permute(0, 2, 1, 3).reshape(m, n)

cb = codebook.get_codebook("E8P12")
if hasattr(cb, 'to'):
    cb = cb.to(device)

model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()

layer = model.model.layers[args.layer_i]
wmap = {'qkv': [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight],
         'o': [layer.self_attn.o_proj.weight],
         'up': [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight],
         'down':[layer.mlp.down_proj.weight]}

Ws = [w.detach().to(torch.float32) for w in wmap[args.sublayer]]
n = Ws[0].shape[1]
H_data = torch.load(f"{args.hessian_path}/{args.layer_i}_{args.sublayer}.pt", map_location=device)
H = utils.flat_to_sym(H_data['flatH'], H_data['n']).to(torch.float32)
mu = H_data['mu'].to(torch.float32).to(device) # based on quip#
H = H + mu.unsqueeze(0) * mu.unsqueeze(1) # based on quip#
H = utils.regularize_H(H, H_data['n'], 1e-2)
H = H.to(torch.float32).to(device)
SU = (torch.randn(n, device=device).sign() + 1e-5).sign().to(torch.float32)

Hr = RHT_H(H, SU).to(torch.float32)
S, V = torch.linalg.eigh(Hr)
S_sqrt = torch.sqrt(torch.clamp(S, min=0))
H_sqrt = (V @ torch.diag(S_sqrt) @ V.T).to(device=device, dtype=torch.float32)

for W in Ws:
    m = W.shape[0]
    W = W.to(device)
    SV = (torch.randn(m, device=device).sign() + 1e-5).sign().to(torch.float32)
    Wr = RHT_W(W, SU, SV)
    hatW_ar = bulk_LDLQ(Wr, H_sqrt, cb, Hr, 2)
    hatW_ar = (utils.matmul_hadU((utils.matmul_hadU(hatW_ar) * SU.to(device)).T) * SV.to(device)).T
    E = hatW_ar - W
    print("Adaptive Rounding")
    print(f"Error: {E.square().sum() / W.square().sum()}")
    print(f"Proxy loss: {(E @ H @ E.T).trace()}")

    hatW_nr = bulk_LDLQ(Wr, torch.eye(n, dtype=torch.float32, device=device), cb, Hr, 0)
    hatW_nr = (utils.matmul_hadU((utils.matmul_hadU(hatW_nr) * SU.to(device)).T) * SV.to(device)).T
    E = hatW_nr - W
    print("Non-adaptive Rounding")
    print(f"Error: {E.square().sum() / W.square().sum()}")
    print(f"Proxy loss: {(E @ H @ E.T).trace()}")
