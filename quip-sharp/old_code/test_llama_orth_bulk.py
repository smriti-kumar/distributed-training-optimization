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

def bulk_orth_quant(coeffs, H_sqrt, n, iters):
    tr_H_sqrt = H_sqrt.diagonal().sum()
    base_cliques = torch.tensor(cliques, dtype=torch.long, device=coeffs.device)
    num_cliques = n*n // 64
    offsets = torch.arange(num_cliques, dtype=torch.long, device=coeffs.device).view(-1, 1, 1) * 64
    all_cliques = offsets + base_cliques.unsqueeze(0)
    all_cliques = all_cliques.reshape(-1, 8)
    hat_coeffs = torch.zeros_like(coeffs)
    for c in all_cliques:
        hat_coeffs[c] = cb.quantize(coeffs[c].unsqueeze(0).float())[0].squeeze(0).to(coeffs.dtype)
    for _ in range(iters):
        error = coeffs - hat_coeffs
        correction = bt(ibt(error, n) @ H_sqrt, n) / tr_H_sqrt / n**2
        corrected = coeffs + correction
        hat_new = torch.zeros_like(coeffs)
        for c in all_cliques:
            hat_new[c] = cb.quantize(corrected[c].unsqueeze(0).float())[0].squeeze(0).to(coeffs.dtype)
        if torch.allclose(hat_new, hat_coeffs):
            break
        hat_coeffs = hat_new
    return hat_coeffs

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
SU = (torch.randn(n, device=device).sign() + 1e-5).sign().double()
SV = (torch.randn(n, device=device).sign() + 1e-5).sign().double()
Hr = RHT_H(H, SU)
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
hat_coeffs = bulk_orth_quant(coeffs, H_sqrt, n, 20)
hat_coeffs = hat_coeffs * Wscale
hatW_fast = hb_transform(hat_coeffs.reshape(1, n*n)).reshape(n, n)
hatW_fast = (utils.matmul_hadU((utils.matmul_hadU(hatW_fast) * SU.to(device)).T) * SV.to(device)).T
err = (W - hatW_fast).norm() / W.norm()
print(f"Error: {err}")
proxy_loss = ((hatW_fast - W) @ H @ (hatW_fast - W).T).trace()
print(f"Proxy loss: {proxy_loss}")