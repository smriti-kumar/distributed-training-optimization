"""Fused CUDA kernel for analytic E8 minimal-vector projection.

For each row g in an (N, 8) float32 momentum tensor, finds the E8 minimal
vector (one of 240) minimizing <g, delta> without ever materializing the
240-way dot product, by exploiting the two closed-form families:

  - 112-family: two nonzero coords, each +-1 (top-2 |g_i| picks the coords).
  - 128-family: all eight coords +-0.5 with even minus-sign parity.

One CUDA thread handles one vector, entirely in registers.

This module also exposes e8_best_valid(momentum, Qidxs, valid_bits), which
does the same per-group search but restricted to whichever of the 240
directions are valid lattice moves from the group's current codeword. That
constraint can't be expressed analytically (which is why e8_analytic_score
above is "unconstrained best, then fall back to a masked (N,240) search" in
the calling code) so e8_best_valid instead does a masked 240-way loop
per-thread entirely in registers, reading the 240 canonical direction
vectors from __constant__ memory and the per-codeword validity mask from a
packed bit table built by build_valid_bits.

NOTE: this module was written on a machine with no CUDA device available.
The kernels have not been compiled or executed. See the bottom of this file
for how to build and validate them once you have a GPU.
"""

import time

import torch
from torch.utils.cpp_extension import load_inline

cpp_source = r"""
std::vector<torch::Tensor> e8_analytic_score_cuda(torch::Tensor momentum);
std::vector<torch::Tensor> e8_best_valid_cuda(torch::Tensor momentum, torch::Tensor qidxs, torch::Tensor valid_bits);
void set_e8_directions_cuda(torch::Tensor directions);
"""

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>
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

// ---------------------------------------------------------------------------
// e8_best_valid: masked search over the 240 canonical E8 minimal vectors,
// restricted per-group to whichever directions are valid lattice moves from
// that group's current codeword.
// ---------------------------------------------------------------------------

// The 240 direction vectors (112-family +-1 pairs, then 128-family +-0.5
// even-parity patterns), loaded once via set_e8_directions_cuda. Same for
// every thread -> constant memory broadcasts it for free.
__constant__ float c_e8_directions[240 * 8];

__global__ void e8_best_valid_kernel(
    const float* __restrict__ momentum,
    const int64_t* __restrict__ qidxs,
    const uint64_t* __restrict__ valid_bits, // (C, 4)
    float* __restrict__ best_score,
    int32_t* __restrict__ best_dir_idx,
    int64_t N)
{
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float g[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) g[i] = momentum[n * 8 + i];

    int64_t c = qidxs[n];
    const uint64_t* bits = valid_bits + c * 4;
    uint64_t bits_local[4];
    #pragma unroll
    for (int w = 0; w < 4; w++) bits_local[w] = bits[w];

    float best = INFINITY;
    int best_idx = -1;

    // Plain 240x8 register loop, per the spec: correctness first. d is a
    // compile-time constant under full unroll, so word/bit resolve to
    // constants too (no dynamic indexing into bits_local at runtime).
    #pragma unroll
    for (int d = 0; d < 240; d++) {
        int word = d >> 6;
        int bit = d & 63;
        bool valid = (bits_local[word] >> bit) & 1ULL;
        if (valid) {
            const float* dir = c_e8_directions + d * 8;
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i++) dot += g[i] * dir[i];
            if (dot < best) { best = dot; best_idx = d; }
        }
    }

    best_score[n] = best;       // stays +inf if no valid direction existed
    best_dir_idx[n] = best_idx; // stays -1 if no valid direction existed
}

std::vector<torch::Tensor> e8_best_valid_cuda(
    torch::Tensor momentum,
    torch::Tensor qidxs,
    torch::Tensor valid_bits)
{
    TORCH_CHECK(momentum.is_cuda(), "momentum must be a CUDA tensor");
    TORCH_CHECK(momentum.scalar_type() == torch::kFloat32, "momentum must be float32");
    TORCH_CHECK(momentum.dim() == 2 && momentum.size(1) == 8, "momentum must be (N, 8)");
    TORCH_CHECK(momentum.is_contiguous(), "momentum must be contiguous");

    TORCH_CHECK(qidxs.is_cuda(), "qidxs must be a CUDA tensor");
    TORCH_CHECK(qidxs.scalar_type() == torch::kInt64, "qidxs must be int64");
    TORCH_CHECK(qidxs.dim() == 1 && qidxs.size(0) == momentum.size(0), "qidxs must be (N,)");
    TORCH_CHECK(qidxs.is_contiguous(), "qidxs must be contiguous");

    TORCH_CHECK(valid_bits.is_cuda(), "valid_bits must be a CUDA tensor");
    // Stored as int64 rather than a real uint64 dtype (torch's uint64 support
    // is version-gated / experimental); the kernel reinterprets the exact
    // same bit pattern as uint64_t, which is all that matters here since
    // build_valid_bits never treats the packed words as numeric values.
    TORCH_CHECK(valid_bits.scalar_type() == torch::kInt64, "valid_bits must be int64 (bit-packed)");
    TORCH_CHECK(valid_bits.dim() == 2 && valid_bits.size(1) == 4, "valid_bits must be (C, 4)");
    TORCH_CHECK(valid_bits.is_contiguous(), "valid_bits must be contiguous");

    at::cuda::CUDAGuard guard(momentum.device());

    int64_t N = momentum.size(0);
    auto best_score = torch::empty({N}, momentum.options().dtype(torch::kFloat32));
    auto best_dir_idx = torch::empty({N}, momentum.options().dtype(torch::kInt32));

    if (N > 0) {
        const int block = 256;
        const int64_t grid = (N + block - 1) / block;

        e8_best_valid_kernel<<<(unsigned int)grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            momentum.data_ptr<float>(),
            qidxs.data_ptr<int64_t>(),
            reinterpret_cast<const uint64_t*>(valid_bits.data_ptr<int64_t>()),
            best_score.data_ptr<float>(),
            best_dir_idx.data_ptr<int32_t>(),
            N);
    }

    return {best_score, best_dir_idx};
}

void set_e8_directions_cuda(torch::Tensor directions) {
    TORCH_CHECK(directions.dim() == 2 && directions.size(0) == 240 && directions.size(1) == 8,
                "directions must be (240, 8)");
    TORCH_CHECK(directions.scalar_type() == torch::kFloat32, "directions must be float32");

    auto directions_cpu = directions.to(torch::kCPU).contiguous();
    cudaMemcpyToSymbol(c_e8_directions, directions_cpu.data_ptr<float>(), 240 * 8 * sizeof(float));
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
            functions=["e8_analytic_score_cuda", "e8_best_valid_cuda", "set_e8_directions_cuda"],
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
# e8_best_valid: masked search restricted to lattice-valid directions.
# ---------------------------------------------------------------------------

def build_e8_directions() -> torch.Tensor:
    """
    Canonical, fixed enumeration of the 240 E8 minimal vectors (rows 0..239)
    used by e8_best_valid's __constant__ direction table. If your
    neighbors_table / valid_bits were built against a DIFFERENT ordering,
    they will be silently misaligned with this table -- reorder one side to
    match before calling e8_best_valid.

    Order:
      - rows 0..111 (112-family): for each of the 28 unordered coordinate
        pairs (i, j), i < j, in lexicographic order, the 4 sign combinations
        (+,+), (+,-), (-,+), (-,-) in that order, giving +-1 at (i, j) and 0
        elsewhere.
      - rows 112..239 (128-family): all 2**7 = 128 sign patterns over
        coordinates 0..6 (bit b of the pattern index selects the sign of
        coordinate b: 0 -> +0.5, 1 -> -0.5), with coordinate 7's sign fixed
        to whichever value makes the total number of minus signs even.
    """
    dirs = torch.zeros(240, 8, dtype=torch.float32)
    row = 0
    sign_pairs = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))
    for i in range(8):
        for j in range(i + 1, 8):
            for si, sj in sign_pairs:
                dirs[row, i] = si
                dirs[row, j] = sj
                row += 1
    for idx in range(128):
        s = [1.0 if ((idx >> b) & 1) == 0 else -1.0 for b in range(7)]
        neg = sum(1 for x in s if x < 0)
        s.append(-1.0 if (neg % 2 == 1) else 1.0)
        dirs[row] = torch.tensor(s, dtype=torch.float32) * 0.5
        row += 1
    assert row == 240
    return dirs


E8_DIRECTIONS = build_e8_directions()

_directions_set = False


def set_e8_directions(directions: torch.Tensor):
    """
    Loads the 240 direction rows used by e8_best_valid's constant-memory
    table. Call this ONCE (idempotent -- safe, if wasteful, to call again)
    with the SAME directions tensor -- same row ordering -- that your
    neighbors_table / valid_bits were built against. e8_best_valid's
    best_dir_idx output indexes into these rows, so a mismatched ordering
    here silently corrupts the landing-codeword lookup downstream.

    build_e8_directions()'s enumeration is only a self-test convenience; if
    you already have a real (240, 8) directions tensor (e.g. loaded from a
    precomputed neighbors file), pass that instead -- it defines the
    ordering neighbors_table's columns must match, not the other way round.
    """
    global _directions_set
    directions = directions.to(torch.float32).contiguous()
    mod = _get_module()
    mod.set_e8_directions_cuda(directions)
    _directions_set = True


def build_valid_bits(neighbors_table: torch.Tensor) -> torch.Tensor:
    """
    neighbors_table: (C, 240) integer tensor; entry >= 0 means direction d
    (indexing into the same directions ordering passed to
    set_e8_directions) is a valid move from codeword c, -1 means invalid.

    Returns: (C, 4) int64 tensor, 4 x 64-bit words packing the 240-bit
    validity mask per codeword (bit d of the mask set iff
    neighbors_table[c, d] >= 0). Stored as int64 rather than a real uint64
    dtype since torch's uint64 support is version-gated; e8_best_valid's
    kernel reinterprets the identical bit pattern as uint64_t.
    """
    assert neighbors_table.dim() == 2 and neighbors_table.shape[1] == 240
    device = neighbors_table.device
    C = neighbors_table.shape[0]

    valid = neighbors_table >= 0  # (C, 240) bool
    pad = torch.zeros((C, 256 - 240), dtype=torch.bool, device=device)
    valid = torch.cat([valid, pad], dim=1).view(C, 4, 64)  # (C, 4, 64)

    bit_pos = torch.arange(64, dtype=torch.int64, device=device)
    shifts = torch.ones(64, dtype=torch.int64, device=device) << bit_pos  # (64,)

    # Bits within a word are disjoint, so sum == bitwise OR here.
    words = (valid.to(torch.int64) * shifts).sum(dim=-1)  # (C, 4) int64
    return words.contiguous()


def e8_best_valid(momentum: torch.Tensor, qidxs: torch.Tensor, valid_bits: torch.Tensor):
    """
    momentum: (N, 8) float32 CUDA tensor.
    qidxs: (N,) int64 CUDA tensor -- current codeword index per group, in [0, C).
    valid_bits: (C, 4) int64 CUDA tensor from build_valid_bits.

    Returns (best_score: (N,) float32, best_dir_idx: (N,) int32). best_dir_idx
    indexes into whichever (240, 8) directions tensor was last passed to
    set_e8_directions (NOT a landing codeword -- gather the landing from
    your neighbors_table on the Python side). Groups with no valid direction
    get best_score = +inf, best_dir_idx = -1.
    """
    assert momentum.is_cuda, "momentum must live on a CUDA device"
    assert momentum.dtype == torch.float32, "momentum must be float32"
    assert momentum.dim() == 2 and momentum.shape[1] == 8, "momentum must be (N, 8)"
    assert qidxs.is_cuda, "qidxs must live on a CUDA device"
    assert qidxs.dtype == torch.int64, "qidxs must be int64"
    assert qidxs.dim() == 1 and qidxs.shape[0] == momentum.shape[0], "qidxs must be (N,)"
    assert valid_bits.is_cuda, "valid_bits must live on a CUDA device"
    assert valid_bits.dtype == torch.int64, "valid_bits must be int64"
    assert valid_bits.dim() == 2 and valid_bits.shape[1] == 4, "valid_bits must be (C, 4)"

    assert _directions_set, (
        "call set_e8_directions(directions) once before e8_best_valid -- "
        "it loads the 240 direction rows into constant memory and must use "
        "the same ordering as the neighbors_table valid_bits was built from"
    )

    momentum = momentum.contiguous()
    qidxs = qidxs.contiguous()
    valid_bits = valid_bits.contiguous()

    mod = _get_module()
    best_score, best_dir_idx = mod.e8_best_valid_cuda(momentum, qidxs, valid_bits)
    return best_score, best_dir_idx


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


def _test_e8_analytic_score():
    torch.manual_seed(0)
    N = 100_000
    g = torch.randn(N, 8, device="cuda", dtype=torch.float32)

    s_ref, h_ref = reference(g.cpu())
    s_k, h_k = e8_analytic_score(g)

    torch.testing.assert_close(s_k.cpu(), s_ref, rtol=1e-4, atol=1e-4)
    assert torch.equal(h_k.cpu(), h_ref), (h_k.cpu()[:10], h_ref[:10])
    print("e8_analytic_score PASS")


def _test_e8_best_valid():
    torch.manual_seed(0)
    C = 65536
    directions = E8_DIRECTIONS.cuda()
    set_e8_directions(directions)

    # Fabricated neighbors_table: ~40% invalid, rest any nonneg value (only
    # the sign matters for validity). int32 to keep the big timing run's
    # (C,240) gather affordable.
    neighbors_table = torch.randint(0, 1000, (C, 240), device="cuda", dtype=torch.int32)
    invalid_mask = torch.rand(C, 240, device="cuda") < 0.4
    neighbors_table.masked_fill_(invalid_mask, -1)
    # every codeword needs >=1 valid direction so argmin is defined
    all_invalid = (neighbors_table >= 0).sum(dim=-1) == 0
    if all_invalid.any():
        rows = all_invalid.nonzero(as_tuple=True)[0]
        neighbors_table[rows, 0] = 0

    valid_bits = build_valid_bits(neighbors_table)

    # --- correctness, small N ---
    N = 50_000
    momentum = torch.randn(N, 8, device="cuda", dtype=torch.float32)
    qidxs = torch.randint(0, C, (N,), device="cuda", dtype=torch.int64)

    scores = momentum @ directions.T  # (N, 240)
    nb = neighbors_table[qidxs]  # (N, 240)
    scores.masked_fill_(nb < 0, float("inf"))
    ref_score, ref_dir = scores.min(dim=-1)

    kern_score, kern_dir = e8_best_valid(momentum, qidxs, valid_bits)

    torch.testing.assert_close(kern_score.cpu(), ref_score.cpu(), rtol=1e-4, atol=1e-4)
    assert torch.equal(kern_dir.cpu().long(), ref_dir.cpu()), (
        kern_dir.cpu()[:10],
        ref_dir.cpu()[:10],
    )
    print("e8_best_valid PASS")

    # --- timing at N=5,000,000, C=65536 ---
    # NOTE: the pytorch reference materializes a (N, 240) float32 score
    # matrix and a (N, 240) int32 gather from neighbors_table -- at
    # N=5,000,000 that's ~4.8GB each (~10GB total). Shrink N_big if this OOMs
    # on your GPU; the kernel side has no such blowup since it never
    # materializes the (N, 240) matrix.
    N_big = 5_000_000
    momentum_big = torch.randn(N_big, 8, device="cuda", dtype=torch.float32)
    qidxs_big = torch.randint(0, C, (N_big,), device="cuda", dtype=torch.int64)

    for _ in range(3):
        e8_best_valid(momentum_big, qidxs_big, valid_bits)
    torch.cuda.synchronize()

    n_iters = 20
    t0 = time.time()
    for _ in range(n_iters):
        e8_best_valid(momentum_big, qidxs_big, valid_bits)
    torch.cuda.synchronize()
    t_kernel = (time.time() - t0) / n_iters
    print(f"kernel:             {t_kernel * 1e3:8.3f} ms/call  (N={N_big})")

    def _ref_once():
        scores = momentum_big @ directions.T
        nb = neighbors_table[qidxs_big]
        scores.masked_fill_(nb < 0, float("inf"))
        return scores.min(dim=-1)

    for _ in range(2):
        _ref_once()
    torch.cuda.synchronize()

    n_iters_ref = 5
    t0 = time.time()
    for _ in range(n_iters_ref):
        _ref_once()
    torch.cuda.synchronize()
    t_ref = (time.time() - t0) / n_iters_ref
    print(f"pytorch reference:  {t_ref * 1e3:8.3f} ms/call  (N={N_big})")
    print(f"speedup: {t_ref / t_kernel:.1f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "this test requires a CUDA device"

    _test_e8_analytic_score()
    _test_e8_best_valid()
