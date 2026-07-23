import argparse, torch
from transformers import AutoModelForCausalLM
from lib import codebook, utils
from lib.algo.quip import RHT_H, RHT_W

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, required=True)
parser.add_argument('--hessian_path', type=str, required=True)
parser.add_argument('--layer_i', type=int, default=1)
parser.add_argument('--sublayer', type=str, default='o', choices=['qkv','o','up','down'])
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hb_transform(x):
    if len(x.shape) == 1:
        x = x.view(1, -1)
    m, n = x.shape
    k = 1
    while 4**k < n:
        k += 1
    assert 4**k == n
    b = torch.tensor([1,1,-1,-1], dtype=x.dtype, device=x.device)
    x = x.reshape((m,) + (4,)*k)
    for i in range(k):
        x = x.flip(1+i) + x * b.view((1,)*(i+1)+(4,)+(1,)*(k-1-i))
    x = x.reshape((m,) + (2,2)*k)
    x = x.permute((0,)+tuple(2*i+1 for i in range(k))+tuple(2*i+2 for i in range(k)))
    return x.reshape(m, 2**k, 2**k) / (2**k)

def calculate_k(n):
    k = 1
    while 4**k < n:
        k += 1
    assert 4**k == n
    return k

def hb_transform_loop(x, k):
    m, n = x.shape
    assert 4**k == n
    b = torch.tensor([1,1,-1,-1], dtype=x.dtype, device=x.device)
    x = x.reshape((m,) + (4,)*k)
    for i in range(k):
        x = x.flip(1+i) + x * b.view((1,)*(i+1)+(4,)+(1,)*(k-1-i))
    return x.reshape((m,) + (2,2)*k)

def hb_transform_reshape(x, k):
    m = x.shape[0]
    x = x.reshape((m,) + (2,)*(2*k))
    fwd = [0]+[2*i+1 for i in range(k)]+[2*i+2 for i in range(k)]
    inv = [0]*(2*k+1)
    for i,p in enumerate(fwd):
        inv[p] = i
    x = x.permute(inv)
    return x.reshape((m,) + (4,)*k)

def bt(M, n):
    k = calculate_k(n*n)
    flat = hb_transform_reshape(M.reshape(1,n,n), k).reshape(1, n*n)
    return (hb_transform_loop(flat, k) / (2**k)).reshape(n*n) * n

def ibt(e, n):
    return hb_transform(e.reshape(1, n*n)).reshape(n, n)

def decompose_H(H_sqrt, n):
    decomp = []
    while n > 8:
        next_n = n // 2
        Ha, Hb = H_sqrt[:next_n, :next_n], H_sqrt[:next_n, next_n:]
        Hc, Hd = H_sqrt[next_n:, :next_n], H_sqrt[next_n:, next_n:]
        H_decomp = [(Ha+Hd)/2, (Hb+Hc)/2, (Hb-Hc)/2, (Ha-Hd)/2]
        decomp.append((H_sqrt, H_decomp))
        H_sqrt = H_decomp[0]
        n = next_n
    decomp.append((H_sqrt, None))
    return decomp

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

cb = codebook.get_codebook("E8P12")
if hasattr(cb, 'to'):
    cb = cb.to(device)

B_2 = (hb_transform(torch.eye(4)) * 2).to(device)
cached_signs = {}
for b in range(4):
    for a in range(b):
        for i in range(4):
            if torch.allclose(B_2[i] @ B_2[a], B_2[b]):
                cached_signs[(a, b)] = (i, 1)
            if torch.allclose(B_2[i] @ B_2[a], -B_2[b]):
                cached_signs[(a, b)] = (i, -1)
basis_mats = (hb_transform(torch.eye(64)) * 8).to(device)

def fast_orth_quant(coeffs, n, hat_coeffs, error, decomp, iter):
    H_sqrt, H_decomp = decomp[iter]
    tr_H_sqrt = torch.diagonal(H_sqrt).sum()
    if n == 8:
        # J = torch.zeros(64, 64, dtype=H_sqrt.dtype, device=H_sqrt.device)
        # for c in range(8):
        #     B_curr = basis_mats[cliques[c]]
        #     for cp in range(c):
        #         B_prior = basis_mats[cliques[cp]]
        #         block = ((B_curr @ H_sqrt).unsqueeze(1) * B_prior.unsqueeze(0)).sum(dim=(-2, -1)) / tr_H_sqrt / 64
        #         for i, r in enumerate(cliques[c]):
        #             for j, cl in enumerate(cliques[cp]):
        #                 J[r, cl] = block[i, j]
        J = ((basis_mats.to(H_sqrt.dtype) @ H_sqrt).unsqueeze(1) * basis_mats.unsqueeze(0)).sum(dim=(-2,-1)) / tr_H_sqrt
        for c in range(8):
            target = coeffs[cliques[c]] + J[cliques[c]] @ error
            hat_coeffs[cliques[c]] = cb.quantize(target.unsqueeze(0).float())[0].squeeze(0).to(coeffs.dtype)
            error[cliques[c]] = coeffs[cliques[c]] - hat_coeffs[cliques[c]]
        return
    next_n = n // 2
    cliques_per_group = (n * n) // 32
    group_size = (n * n) // 4
    clique_vals = torch.stack([torch.tensor(cliques[c % 8], dtype=torch.long, device=coeffs.device) + 64 * (c // 8) for c in range(cliques_per_group)])
    for b in range(4):
        corrections = torch.zeros(cliques_per_group, 8, dtype=coeffs.dtype, device=coeffs.device)
        for a in range(b):
            d, sign = cached_signs[(a, b)]
            e_a = error[a * group_size:(a+1) * group_size]
            f = bt(ibt(e_a, next_n) @ H_decomp[d], next_n)
            corrections += sign * f[clique_vals] / tr_H_sqrt / next_n**2
        coeffs_b = coeffs[b * group_size:(b+1) * group_size].clone()
        for c_ind in range(cliques_per_group):
            coeffs_b[clique_vals[c_ind]] += corrections[c_ind]
        hat_b = hat_coeffs[b * group_size:(b+1) * group_size]
        error_b = error[b * group_size:(b+1) * group_size]
        fast_orth_quant(coeffs_b, next_n, hat_b, error_b, decomp, iter + 1)

model = AutoModelForCausalLM.from_pretrained(args.hf_path, torch_dtype=torch.float64, low_cpu_mem_usage=True)
model.eval()

layer = model.model.layers[args.layer_i]
wmap = {'qkv': [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight],
         'o': [layer.self_attn.o_proj.weight],
         'up': [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight],
         'down':[layer.mlp.down_proj.weight]}
W = torch.cat([w.detach().double() for w in wmap[args.sublayer]], dim=0).to(device)
n = W.shape[0]
k = calculate_k(n * n)
H_data = torch.load(f"{args.hessian_path}/{args.layer_i}_{args.sublayer}.pt", map_location=device)
H = utils.flat_to_sym(H_data['flatH'], H_data['n']).double()
mu = H_data['mu'].double().to(device) # based on quip#
H = H + mu.unsqueeze(0) * mu.unsqueeze(1) # based on quip#
H = utils.regularize_H(H, H_data['n'], 1e-2)
H = H.to(device)
H_nr = torch.eye(n, dtype=torch.float64, device=device)
SU = (torch.randn(n, device=device).sign() + 1e-5).sign().double()
SV = (torch.randn(n, device=device).sign() + 1e-5).sign().double()
Hr = RHT_H(H_nr, SU)
Wr = RHT_W(W, SU, SV)
S, V = torch.linalg.eigh(Hr)
S_sqrt = torch.sqrt(torch.clamp(S, min=0))
H_sqrt = (V @ torch.diag(S_sqrt) @ V.T).to(device=device, dtype=torch.float64)
tr_H_sqrt = torch.diagonal(H_sqrt).sum()
W_flat = Wr.reshape(1, 2**k, 2**k)
W_reshaped = hb_transform_reshape(W_flat, k).reshape(1, n*n)
coeffs = hb_transform_loop(W_reshaped, k) / (2**k)
coeffs = coeffs.reshape(n*n)
norm = 1/n
coeffs = coeffs / norm
Wscale = coeffs.square().mean().sqrt() / cb.opt_scale
coeffs = coeffs / Wscale
hat_coeffs = torch.zeros_like(coeffs, device=device)
error = torch.zeros(n * n, dtype=coeffs.dtype, device=device)
decomp = decompose_H(H_sqrt, n)
fast_orth_quant(coeffs, n, hat_coeffs, error, decomp, 0)
hat_coeffs = hat_coeffs * Wscale
hatW_fast = hb_transform(hat_coeffs.reshape(1, n*n)).reshape(n, n)
hatW_fast = (utils.matmul_hadU((utils.matmul_hadU(hatW_fast) * SU.to(device)).T) * SV.to(device)).T
err = (W - hatW_fast).norm() / W.norm()
print(f"Error: {err}")
proxy_loss = ((hatW_fast - W) @ H @ (hatW_fast - W).T).trace()
print(f"Proxy loss: {proxy_loss}")