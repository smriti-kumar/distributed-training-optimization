import argparse, torch, types
from transformers import AutoModelForCausalLM
from lib import utils, codebook as codebook_lib
from lib.algo.quip import original_quantize

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', type=str, required=True)
parser.add_argument('--hessian_path', type=str, required=True)
parser.add_argument('--layer_i', type=int, default=1)
parser.add_argument('--sublayer', type=str, default='o')
args_cli = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = types.SimpleNamespace(
    use_fp64 = True,
    incoh_mode = 'had',
    rescale_WH = False,
    lora_rank = 0,
    scale_override = -1,
    resid_scale_override = -1,
    quip_tune_iters = 0,
    no_use_buffered = False,
    lowmem_ldlq = True,
    sigma_reg = 1e-2,
    save_pfx = '/tmp',
)

model = AutoModelForCausalLM.from_pretrained(args_cli.hf_path, torch_dtype=torch.float64, low_cpu_mem_usage=True)
model.eval()
layer = model.model.layers[args_cli.layer_i]
wmap = {
    'qkv': [layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight],
    'o': [layer.self_attn.o_proj.weight],
    'up': [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight],
    'down': [layer.mlp.down_proj.weight],
}
weights = wmap[args_cli.sublayer]
dtype_ = torch.float64
scales = [w.detach().to(dtype_).square().mean().sqrt() for w in weights]

Ws = [w.detach().double() for w in weights]
n = Ws[0].shape[1]
hessian_file = f"{args_cli.hessian_path}/{args_cli.layer_i}_{args_cli.sublayer}.pt"
H_data = torch.load(hessian_file, map_location='cpu')
H = utils.flat_to_sym(H_data['flatH'], H_data['n']).double()
mu = H_data['mu'].double()
H.add_(mu[None, :] * mu[:, None])
H = utils.regularize_H(H, H_data['n'], args.sigma_reg)
H = H.to(device)
cb = codebook_lib.get_codebook('E8P12').to(dtype_)

for passes in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print(f"Number of passes: {passes}")
    args.quip_tune_iters = passes
    for W in Ws:
        W = W.to(device)
        hatW, _ = original_quantize(H, W, rank=0, codebook_orig=cb, args=args, device=device)
        E = hatW - W
        print(f"Error: {E.square().sum() / W.square().sum()}")
        print(f"Proxy loss: {(E @ H @ E.T).trace()}")