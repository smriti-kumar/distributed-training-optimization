"""Fused CUDA kernel for analytic E8 minimal-vector projection.

For each row g in an (N, 8) float32 momentum tensor, finds the E8 minimal
vector (one of 240) minimizing <g, delta> without ever materializing the
240-way dot product, by exploiting the two closed-form families:

  - 112-family: two nonzero coords, each +-1 (top-2 |g_i| picks the coords).
  - 128-family: all eight coords +-0.5 with even minus-sign parity.

One CUDA thread handles one vector, entirely in registers.

NOTE: this module was written on a machine with no CUDA device available.
The kernel has not been compiled or executed. See the bottom of this file
for how to build and validate it once you have a GPU.
"""

import torch
from torch.utils.cpp_extension import load_inline

cpp_source = r"""
std::vector<torch::Tensor> e8_analytic_score_cuda(torch::Tensor momentum);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void e8_analytic_kernel(
    const float* __restrict__ momentum,
    float* __restrict__ best_score,
    int32_t* __restrict__ dir_hash,
    int64_t N)
{
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float* g = momentum + n * 8;

    float gv[8];
    float absg[8];
    float s[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        gv[i] = g[i];
        absg[i] = fabsf(gv[i]);
        // g_i == 0 must map to s_i = -1, matching (g_i >= 0) -> -1 exactly.
        s[i] = (gv[i] >= 0.0f) ? -1.0f : 1.0f;
    }

    // --- top-2 by magnitude (i0 = largest, i1 = second largest) ---
    int i0, i1;
    float v0, v1;
    if (absg[0] >= absg[1]) { i0 = 0; v0 = absg[0]; i1 = 1; v1 = absg[1]; }
    else                    { i0 = 1; v0 = absg[1]; i1 = 0; v1 = absg[0]; }
    #pragma unroll
    for (int i = 2; i < 8; i++) {
        float v = absg[i];
        if (v > v0) { v1 = v0; i1 = i0; v0 = v; i0 = i; }
        else if (v > v1) { v1 = v; i1 = i; }
    }
    float score112 = -(v0 + v1);

    // --- argmin by magnitude, for the 128-family parity fix ---
    int kmin = 0;
    float vmin = absg[0];
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        if (absg[i] < vmin) { vmin = absg[i]; kmin = i; }
    }

    float signs[8];
    int negcount = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        signs[i] = s[i];
        if (signs[i] < 0.0f) negcount++;
    }
    if (negcount & 1) {
        signs[kmin] = -signs[kmin];
    }

    float sum128 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        sum128 += signs[i] * gv[i];
    }
    float score128 = 0.5f * sum128;

    bool use112 = score112 <= score128;
    float bscore = use112 ? score112 : score128;

    // --- direction hash: digit_i = round(delta_i * 2) + 2, base-5 encoding ---
    int32_t dhash = 0;
    int32_t pow5 = 1;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int digit;
        if (use112) {
            if (i == i0 || i == i1) digit = (int)(s[i] * 2.0f) + 2; // +-1 -> 0 or 4
            else digit = 2;                                        // delta 0
        } else {
            digit = (int)(signs[i]) + 2;                            // +-1 -> 1 or 3
        }
        dhash += digit * pow5;
        pow5 *= 5;
    }

    best_score[n] = bscore;
    dir_hash[n] = dhash;
}

std::vector<torch::Tensor> e8_analytic_score_cuda(torch::Tensor momentum) {
    TORCH_CHECK(momentum.is_cuda(), "momentum must be a CUDA tensor");
    TORCH_CHECK(momentum.scalar_type() == torch::kFloat32, "momentum must be float32");
    TORCH_CHECK(momentum.dim() == 2 && momentum.size(1) == 8, "momentum must be (N, 8)");
    TORCH_CHECK(momentum.is_contiguous(), "momentum must be contiguous");

    int64_t N = momentum.size(0);
    auto best_score = torch::empty({N}, momentum.options().dtype(torch::kFloat32));
    auto dir_hash = torch::empty({N}, momentum.options().dtype(torch::kInt32));

    if (N > 0) {
        const int block = 256;
        const int64_t grid = (N + block - 1) / block;

        e8_analytic_kernel<<<(unsigned int)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            momentum.data_ptr<float>(),
            best_score.data_ptr<float>(),
            dir_hash.data_ptr<int32_t>(),
            N);
    }

    return {best_score, dir_hash};
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="e8_analytic_ext",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["e8_analytic_score_cuda"],
            verbose=True,
            extra_cuda_cflags=["-O3"],
        )
    return _module


def e8_analytic_score(momentum: torch.Tensor):
    """
    momentum: (N, 8) float32 CUDA tensor.
    Returns (best_score: (N,) float32, dir_hash: (N,) int32).
    """
    assert momentum.is_cuda, "momentum must live on a CUDA device"
    assert momentum.dtype == torch.float32, "momentum must be float32"
    assert momentum.dim() == 2 and momentum.shape[1] == 8, "momentum must be (N, 8)"
    momentum = momentum.contiguous()
    mod = _get_module()
    best_score, dir_hash = mod.e8_analytic_score_cuda(momentum)
    return best_score, dir_hash


# ---------------------------------------------------------------------------
# Correctness test (pure-PyTorch reference vs. kernel). Run this on a GPU box:
#   python e8_analytic_kernel.py
# ---------------------------------------------------------------------------
def reference(momentum: torch.Tensor):
    absg = momentum.abs()
    opp = torch.where(momentum >= 0, -1.0, 1.0)
    top2 = torch.topk(absg, 2, dim=-1)
    score112 = -(top2.values[:, 0] + top2.values[:, 1])
    i0, i1 = top2.indices[:, 0], top2.indices[:, 1]
    ar = torch.arange(momentum.shape[0])
    signs = opp.clone()
    odd = ((signs < 0).sum(-1) % 2 == 1)
    kmin = absg.argmin(-1)
    signs[ar, kmin] = torch.where(odd, -signs[ar, kmin], signs[ar, kmin])
    score128 = 0.5 * (signs * momentum).sum(-1)
    use112 = score112 <= score128
    best_score = torch.where(use112, score112, score128)

    k112 = torch.full((momentum.shape[0], 8), 2, dtype=torch.long)
    k112[ar, i0] = (opp[ar, i0] * 2 + 2).long()
    k112[ar, i1] = (opp[ar, i1] * 2 + 2).long()
    k128 = (signs + 2).long()
    k = torch.where(use112.unsqueeze(-1), k112, k128)
    powers = (5 ** torch.arange(8)).long()
    dir_hash = (k * powers).sum(-1).int()
    return best_score, dir_hash


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this test requires a CUDA device"

    torch.manual_seed(0)
    N = 100_000
    g = torch.randn(N, 8, device="cuda", dtype=torch.float32)

    s_ref, h_ref = reference(g.cpu())
    s_k, h_k = e8_analytic_score(g)

    torch.testing.assert_close(s_k.cpu(), s_ref, rtol=1e-4, atol=1e-4)
    assert torch.equal(h_k.cpu(), h_ref), (h_k.cpu()[:10], h_ref[:10])
    print("PASS")
