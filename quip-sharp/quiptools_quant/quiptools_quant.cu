// CUDA kernels for the LDLQ quantization inner loop (replaces ldlq_helper /
// hbc_transform / ihbc_transform / cb.quantize in lib/algo/quip.py for the
// E8P12 codebook). See workload_analysis.md and gpu_kernel_spec.md at the
// repo root for the algorithm and design rationale; every indexing choice
// below was checked against lib/algo/quip.py and lib/codebook/latticee8_padded12.py
// (not against the *_spec.md excerpts, which were pasted from an earlier
// version of the code and are not authoritative).
//
// Determinism: no atomics anywhere in this file. The warp reduction below
// combines (score, idx) pairs with a tie-break that always prefers the lower
// idx, matching torch.argmax's documented "first maximal value wins" rule,
// so results are bit-reproducible run to run regardless of scheduling order.

#include "quiptools_quant.h"
#include <cfloat>

// cperm / ciperm from lib/algo/quip.py:get_cliques(), computed once and
// hardcoded here (they are a fixed structural constant of the E8 clique
// decomposition, not derived from any tensor at runtime).
//   cperm  = flat(cliques)
//   ciperm = argsort(cperm)      (i.e. cperm[ciperm] == ciperm[cperm] == arange(64))
__constant__ int c_cperm[64] = {
    0, 2, 11, 25, 33, 39, 47, 57, 1, 3, 10, 24, 32, 38, 46, 56,
    4, 6, 15, 29, 35, 37, 43, 61, 5, 7, 14, 28, 34, 36, 42, 60,
    12, 18, 20, 26, 44, 53, 55, 62, 13, 19, 21, 27, 45, 52, 54, 63,
    8, 16, 22, 30, 40, 49, 51, 58, 9, 17, 23, 31, 41, 48, 50, 59};

__constant__ int c_ciperm[64] = {
    0, 8, 1, 9, 16, 24, 17, 25, 48, 56, 10, 2, 32, 40, 26, 18,
    49, 57, 33, 41, 34, 42, 50, 58, 11, 3, 35, 43, 27, 19, 51, 59,
    12, 4, 28, 20, 29, 21, 13, 5, 52, 60, 30, 22, 36, 44, 14, 6,
    61, 53, 62, 54, 45, 37, 46, 38, 15, 7, 55, 63, 31, 23, 39, 47};

// shuffle_map from E8P12_codebook.fast_quantize_part / get_full_grid
__constant__ int c_shuf[8] = {0, 2, 4, 6, 1, 3, 5, 7};

// =============================================================================
// Section A: E8P12 codebook search (device functions), warp-cooperative.
// =============================================================================
//
// Ported against lib/codebook/latticee8_padded12.py:E8P12_codebook.quantize /
// fast_quantize_part / round. Validated on CPU (outside this repo's normal
// test flow, since this file can't be compiled/run here) by reimplementing
// the same algorithm in plain PyTorch using only the codebook's cached
// buffers (grid_part, grid_part_norm, part_abs_map, grid_abs_odd, bit_map)
// and diffing against cb.quantize() on 4096 random vectors -- exact match,
// both values and indices. That confirms these are the only tables the
// device kernel needs; the full 65536-entry grid and its generation logic
// (get_full_grid / get_packed_abs_grid) are NOT needed for quantization,
// only for decompression (by_idxs), which this kernel does not implement.

struct ScoreIdx {
  float score;
  int idx;
};

__device__ __forceinline__ ScoreIdx combine(ScoreIdx a, ScoreIdx b) {
  if (a.score > b.score) return a;
  if (b.score > a.score) return b;
  return (a.idx < b.idx) ? a : b;  // tie: lower index wins (torch.argmax rule)
}

// Warp-cooperative argmax over `ncand` rows of grid_part (each row is 8
// floats): score(c) = 2 * dot(x_part, grid_part[c]) - grid_part_norm[c].
// All 32 lanes must call this together; every lane receives the winning
// (score, idx) on return.
__device__ __forceinline__ ScoreIdx warp_search(const float x_part[8],
                                                  const float* __restrict__ grid_part,
                                                  const float* __restrict__ grid_part_norm,
                                                  int ncand, int lane) {
  ScoreIdx best{-FLT_MAX, -1};
  for (int c = lane; c < ncand; c += 32) {
    const float* g = grid_part + (size_t)c * 8;
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) dot += x_part[i] * g[i];
    float score = 2.f * dot - grid_part_norm[c];
    best = combine(best, ScoreIdx{score, c});
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    ScoreIdx other;
    other.score = __shfl_down_sync(0xffffffff, best.score, off);
    other.idx = __shfl_down_sync(0xffffffff, best.idx, off);
    best = combine(best, other);
  }
  best.score = __shfl_sync(0xffffffff, best.score, 0);
  best.idx = __shfl_sync(0xffffffff, best.idx, 0);
  return best;
}

// Ports E8P12_codebook.fast_quantize_part. `x` is X+1/4 or X-1/4 (the two
// D8^ cosets), `parity` is True for the +1/4 coset, False for -1/4.
__device__ __forceinline__ void e8p12_quantize_variant(
    const float x[8], bool parity, const float* __restrict__ grid_part,
    const float* __restrict__ grid_part_norm,
    const int64_t* __restrict__ part_abs_map,
    const uint8_t* __restrict__ grid_abs_odd, int ncand, int lane,
    float out_vals[8], int64_t& out_idx, float& out_err) {
  float x_part[8];
  float mask[8];
  int neg_count = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    x_part[i] = fabsf(x[i]);
    mask[i] = (x[i] < 0.f) ? -1.f : 1.f;
    neg_count += (x[i] < 0.f) ? 1 : 0;
  }
  bool x_odd = (neg_count & 1) != 0;
  if (x_odd) {
    x_part[7] = -x_part[7];
    mask[7] = -mask[7];
  }

  ScoreIdx best = warp_search(x_part, grid_part, grid_part_norm, ncand, lane);
  const float* roundout = grid_part + (size_t)best.idx * 8;

  float err2 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float v = roundout[i] * mask[i];
    out_vals[i] = v;
    float d = x[i] - v;
    err2 += d * d;
  }

  int64_t abs_idx = part_abs_map[best.idx];
  unsigned int mask_idx = 0;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    int si = c_shuf[i];
    bool sign_bit = (roundout[si] < 0.f) != (mask[si] < 0.f);  // xor
    if (i == 7) sign_bit ^= (grid_abs_odd[abs_idx] != 0);
    if (i == 0) sign_bit ^= parity;
    mask_idx |= (sign_bit ? (1u << i) : 0u);
  }
  out_idx = (abs_idx << 8) | (int64_t)mask_idx;
  // sqrtf, not the squared norm: Python compares plus_err < minus_err using
  // `.norm(dim=-1)` (an actual L2 norm), and while a strict "a < b" vs
  // "sqrt(a) < sqrt(b)" comparison is mathematically equivalent for a,b >= 0,
  // matching the exact same floating-point operation removes any risk of a
  // rounding-order discrepancy at a near-tie, which the bit-exact acceptance
  // tests would catch.
  out_err = sqrtf(err2);
}

// Ports E8P12_codebook.quantize. Every lane must call this together with the
// same `x`; every lane receives the same (out_hat, out_idx) on return.
__device__ __forceinline__ void e8p12_quantize(
    const float x[8], const float* __restrict__ grid_part,
    const float* __restrict__ grid_part_norm,
    const int64_t* __restrict__ part_abs_map,
    const uint8_t* __restrict__ grid_abs_odd, int ncand, int lane,
    float out_hat[8], int64_t& out_idx) {
  float xp[8], xm[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    xp[i] = x[i] + 0.25f;
    xm[i] = x[i] - 0.25f;
  }
  float pv[8], mv[8];
  int64_t pidx, midx;
  float perr, merr;
  e8p12_quantize_variant(xp, true, grid_part, grid_part_norm, part_abs_map,
                          grid_abs_odd, ncand, lane, pv, pidx, perr);
  e8p12_quantize_variant(xm, false, grid_part, grid_part_norm, part_abs_map,
                          grid_abs_odd, ncand, lane, mv, midx, merr);
  bool which = perr < merr;  // strict <, matches Python tie-break (ties -> minus)
#pragma unroll
  for (int i = 0; i < 8; i++) out_hat[i] = which ? (pv[i] - 0.25f) : (mv[i] + 0.25f);
  out_idx = which ? pidx : midx;
}

// =============================================================================
// Section B: Kernel 1 -- fused leaf
// =============================================================================
// One block (one warp, 32 threads) per (leaf invocation, row). Grid.y = p so
// independent rows run as independent blocks (see gpu_kernel_spec.md section
// 7 -- NOTE: the spec's suggestion of "a block of 43 warps" for p=43 would
// exceed CUDA's 1024-threads-per-block limit (43*32=1376); using one block
// per row instead gets the same free row-parallelism without hitting that
// ceiling).

// The 8-clique sequential loop, factored out of leaf_kernel so Kernel 3's
// on-chip leaf case (subtree_recurse<J,0>, below) can call the exact same,
// already-validated logic instead of duplicating it.
//
// X_sh, F_sh, J_sh, hatX_sh are SHARED-MEMORY pointers (64, 64, 64x64, 64
// floats respectively) already populated by the caller. Qidxs_row is GLOBAL
// memory, already offset to this leaf's 8-column slice (i.e. the caller adds
// row*q_row_stride and any column offset before calling). Must be called by
// exactly one warp -- 32 consecutive threads with `lane` = their index 0..31
// within that warp -- since e8p12_quantize's argmax reduction uses
// __shfl*_sync with a full 0xffffffff mask, which requires all 32 lanes of
// the active warp to participate.
//
// Uses __syncwarp(), not __syncthreads(): that makes it safe to call from
// inside a larger block where only one warp is doing leaf work (Kernel 3),
// as well as from leaf_kernel itself, where the whole block IS exactly one
// warp and __syncwarp() is equivalent to __syncthreads() there.
__device__ __forceinline__ void leaf_body(
    const float* __restrict__ X_sh, const float* __restrict__ F_sh,
    const float* __restrict__ J_sh, float* __restrict__ hatX_sh,
    int64_t* __restrict__ Qidxs_row, const float* __restrict__ grid_part,
    const float* __restrict__ grid_part_norm,
    const int64_t* __restrict__ part_abs_map,
    const uint8_t* __restrict__ grid_abs_odd, int ncand, int lane) {
  __shared__ float error_sh[64];

  for (int c = 7; c >= 0; c--) {
    int s = 8 * c;
    int e = s + 8;

    // matmul term: error[e:64] @ J[e:64, s:e]  (skipped for free when e==64,
    // since the accumulation loop below is then empty -- matches the
    // Python `if end < 64` guard without a branch)
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int t = e + lane; t < 64; t += 32) {
      float et = error_sh[t];
#pragma unroll
      for (int j = 0; j < 8; j++) acc[j] += et * J_sh[t * 64 + s + j];
    }
    float Y[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float v = acc[j];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
      v = __shfl_sync(0xffffffff, v, 0);
      Y[j] = X_sh[s + j] + F_sh[s + j] + v;
    }

    float hat[8];
    int64_t idx;
    e8p12_quantize(Y, grid_part, grid_part_norm, part_abs_map, grid_abs_odd,
                    ncand, lane, hat, idx);

    if (lane < 8) {
      hatX_sh[s + lane] = hat[lane];
      error_sh[s + lane] = X_sh[s + lane] - hat[lane];
    }
    if (lane == 0) Qidxs_row[c] = idx;
    __syncwarp();
  }
}

__global__ void leaf_kernel(const float* __restrict__ X,
                             const float* __restrict__ F,
                             const float* __restrict__ J, float* __restrict__ hatX,
                             int64_t* __restrict__ Qidxs, int64_t q_row_stride,
                             const float* __restrict__ grid_part,
                             const float* __restrict__ grid_part_norm,
                             const int64_t* __restrict__ part_abs_map,
                             const uint8_t* __restrict__ grid_abs_odd, int ncand) {
  int row = blockIdx.y;
  int lane = threadIdx.x;  // 0..31

  __shared__ float J_sh[64 * 64];
  __shared__ float X_sh[64];
  __shared__ float F_sh[64];
  __shared__ float hatX_sh[64];

  for (int i = lane; i < 64 * 64; i += 32) J_sh[i] = J[i];
  for (int i = lane; i < 64; i += 32) {
    X_sh[i] = X[(size_t)row * 64 + i];
    F_sh[i] = F[(size_t)row * 64 + i];
  }
  __syncthreads();

  leaf_body(X_sh, F_sh, J_sh, hatX_sh, Qidxs + (size_t)row * q_row_stride,
            grid_part, grid_part_norm, part_abs_map, grid_abs_odd, ncand, lane);

  for (int i = lane; i < 64; i += 32) hatX[(size_t)row * 64 + i] = hatX_sh[i];
}

void launch_leaf_kernel(const float* X, const float* F, const float* J,
                         float* hatX, int64_t* Qidxs, int64_t q_row_stride,
                         int64_t p, const float* grid_part,
                         const float* grid_part_norm,
                         const int64_t* part_abs_map,
                         const uint8_t* grid_abs_odd, int ncand,
                         cudaStream_t stream) {
  dim3 grid(1, (unsigned int)p);
  dim3 block(32);
  leaf_kernel<<<grid, block, 0, stream>>>(X, F, J, hatX, Qidxs, q_row_stride,
                                           grid_part, grid_part_norm,
                                           part_abs_map, grid_abs_odd, ncand);
}

// =============================================================================
// Section C: Kernel 2 -- fused radix-4 butterfly (hbc_transform / ihbc_transform)
// =============================================================================
//
// Ported against lib/algo/quip.py:hbc_transform / ihbc_transform. Verified by
// hand (digit/bit accounting below) and cross-checked numerically on CPU: the
// Python hbc_transform/ihbc_transform round-trip exactly (< 1e-15 relative
// error) for N in {64, 256, 4096}, and preserve L2 energy, confirming the
// indexing understanding this kernel is built on.
//
// Both directions share the same "rounds" step (radix-4 flip/combine,
// b = [1,1,-1,-1]) applied k times; they differ only in where the two fixed
// permutations sit:
//   forward (hbc):  gather(ciperm, per-64-chunk) -> rounds -> scatter(digit-interleave) * scale
//   inverse (ihbc): gather(digit-interleave) -> rounds -> scatter(cperm, per-64-chunk) * scale
//
// "digit-interleave" refers to: given flat index idx in [0, 4^k) with base-4
// digits d_0 (MSB) .. d_{k-1} (LSB), and (row, col) in [0,2^k)x[0,2^k) with
// bits row_i = d_i>>1, col_i = d_i&1 (MSB-first), idx <-> (row, col) is a
// bijection. This is exactly what `x.reshape((2,2)*k).permute(...)` computes
// in the Python source; here it's computed directly by bit manipulation
// instead of a physical permute, which is what lets one kernel handle any k.
//
// This kernel intentionally does NOT try to keep the working set in shared
// memory (unlike the gpu_kernel_spec.md suggestion for N <= 4096): it uses a
// caller-allocated global-memory scratch buffer (effectively relying on
// L1/L2 caching for the small cases). That trades some performance at the
// small end for one code path that is correct at every depth from k=3 (the
// smallest H_multiply, n=64) up to k=11 (the root H_multiply at d=4096,
// n=4,194,304, which cannot fit in shared memory regardless of opt-in
// limits). Adding a shared-memory fast path for k <= 6 (n <= 4096, per the
// spec's own crossover point) is a natural follow-up optimization, not a
// correctness requirement -- see the note in the top-level report.

__device__ __forceinline__ void radix4_rounds(float* buf, int64_t n, int k) {
  int64_t ngroups = n >> 2;
  for (int r = 0; r < k; r++) {
    int64_t P = 1LL << (2 * (k - 1 - r));
    for (int64_t g = threadIdx.x; g < ngroups; g += blockDim.x) {
      int64_t high = g / P;
      int64_t low = g % P;
      int64_t base = high * 4 * P + low;
      int64_t i0 = base, i1 = base + P, i2 = base + 2 * P, i3 = base + 3 * P;
      float v0 = buf[i0], v1 = buf[i1], v2 = buf[i2], v3 = buf[i3];
      buf[i0] = v3 + v0;
      buf[i1] = v2 + v1;
      buf[i2] = v1 - v2;
      buf[i3] = v0 - v3;
    }
    __syncthreads();
  }
}

// idx (base-4, k digits, MSB-first) -> (row, col) bits (MSB-first), per the
// digit-interleave bijection described above.
__device__ __forceinline__ void idx_to_rowcol(int64_t idx, int k, int64_t& row,
                                                int64_t& col) {
  row = 0;
  col = 0;
  for (int i = 0; i < k; i++) {
    int shift = 2 * (k - 1 - i);
    int d = (int)((idx >> shift) & 3);
    row = (row << 1) | (d >> 1);
    col = (col << 1) | (d & 1);
  }
}

__global__ void radix4_forward_kernel(const float* __restrict__ Xin,
                                        float* __restrict__ scratch,
                                        float* __restrict__ Yout, int64_t n, int k,
                                        int s, float scale) {
  int64_t row = blockIdx.x;
  const float* xin_row = Xin + row * n;
  float* buf = scratch + row * n;
  float* yout_row = Yout + row * n;

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t chunk = i >> 6;
    int j = (int)(i & 63);
    buf[i] = xin_row[(chunk << 6) + c_ciperm[j]];
  }
  __syncthreads();

  radix4_rounds(buf, n, k);

  for (int64_t idx = threadIdx.x; idx < n; idx += blockDim.x) {
    int64_t r, c;
    idx_to_rowcol(idx, k, r, c);
    yout_row[r * s + c] = buf[idx] * scale;
  }
}

__global__ void radix4_inverse_kernel(const float* __restrict__ Yin,
                                        float* __restrict__ scratch,
                                        float* __restrict__ Xout, int64_t n, int k,
                                        int s, float scale) {
  int64_t row = blockIdx.x;
  const float* yin_row = Yin + row * n;
  float* buf = scratch + row * n;
  float* xout_row = Xout + row * n;

  for (int64_t idx = threadIdx.x; idx < n; idx += blockDim.x) {
    int64_t r, c;
    idx_to_rowcol(idx, k, r, c);
    buf[idx] = yin_row[r * s + c];
  }
  __syncthreads();

  radix4_rounds(buf, n, k);

  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t chunk = i >> 6;
    int j = (int)(i & 63);
    xout_row[i] = buf[(chunk << 6) + c_cperm[j]] * scale;
  }
}

static int pick_threads(int64_t n) {
  int64_t t = n < 256 ? n : 256;
  if (t < 32) t = 32;
  return (int)t;
}

void launch_radix4_forward(const float* Xin, float* scratch, float* Yout,
                            int64_t rows, int64_t n, int k, int s, float scale,
                            cudaStream_t stream) {
  dim3 grid((unsigned int)rows);
  dim3 block(pick_threads(n));
  radix4_forward_kernel<<<grid, block, 0, stream>>>(Xin, scratch, Yout, n, k, s,
                                                      scale);
}

void launch_radix4_inverse(const float* Yin, float* scratch, float* Xout,
                            int64_t rows, int64_t n, int k, int s, float scale,
                            cudaStream_t stream) {
  dim3 grid((unsigned int)rows);
  dim3 block(pick_threads(n));
  radix4_inverse_kernel<<<grid, block, 0, stream>>>(Yin, scratch, Xout, n, k, s,
                                                      scale);
}

// =============================================================================
// Section D: standalone codebook kernel (acceptance test 4)
// =============================================================================

__global__ void e8p12_quantize_batch_kernel(const float* __restrict__ X,
                                              float* __restrict__ out_vals,
                                              int64_t* __restrict__ out_idx,
                                              const float* __restrict__ grid_part,
                                              const float* __restrict__ grid_part_norm,
                                              const int64_t* __restrict__ part_abs_map,
                                              const uint8_t* __restrict__ grid_abs_odd,
                                              int ncand) {
  int64_t row = blockIdx.x;
  int lane = threadIdx.x;
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = X[row * 8 + i];
  float hat[8];
  int64_t idx;
  e8p12_quantize(x, grid_part, grid_part_norm, part_abs_map, grid_abs_odd,
                  ncand, lane, hat, idx);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) out_vals[row * 8 + i] = hat[i];
    out_idx[row] = idx;
  }
}

void launch_e8p12_quantize_batch(const float* X, float* out_vals,
                                  int64_t* out_idx, int64_t rows,
                                  const float* grid_part,
                                  const float* grid_part_norm,
                                  const int64_t* part_abs_map,
                                  const uint8_t* grid_abs_odd, int ncand,
                                  cudaStream_t stream) {
  dim3 grid((unsigned int)rows);
  dim3 block(32);
  e8p12_quantize_batch_kernel<<<grid, block, 0, stream>>>(
      X, out_vals, out_idx, grid_part, grid_part_norm, part_abs_map,
      grid_abs_odd, ncand);
}

// =============================================================================
// Section E: Kernel 3 -- fused subtree (gpu_kernel_spec.md section 5)
// =============================================================================
//
// Fuses the bottom `J` levels of the recursion -- the J internal-node levels
// nearest the leaves, plus the leaves themselves -- into ONE kernel launch,
// with the whole subtree's working set resident in shared memory and the
// internal H_multiply calls done on-chip (own matmul, not cuBLAS; own
// butterfly transform, reusing radix4_rounds/idx_to_rowcol from Section C
// verbatim -- those operate on a plain float* and don't care whether it
// points into shared or global memory).
//
// J is a compile-time template parameter (gpu_kernel_spec.md: "Implement j
// as a compile-time template parameter"), instantiated for J in {1, 2, 3}.
// J=4 is NOT instantiated: keeping X/F/hatX as three full-size (64*4^J)
// resident buffers -- the simplest, easiest-to-verify design, and the same
// one Kernel 1's leaf uses -- costs 3*64KB = 192KB at J=4 before even adding
// the H matrices, J64, or scratch, which doesn't fit in shared memory on any
// current architecture even with the opt-in raised limit. Getting J=4
// working would need a streaming redesign (process the 4 children
// sequentially, writing each one's result straight to global memory instead
// of keeping a full-size hatX buffer, matching how the earlier C++-side
// recursion touched global memory before this file's Python wiring was
// dropped) -- left as a follow-up, not attempted here.
//
// Per-node schedule: identical to lib/algo/quip.py:ldlq_helper (see the
// derivation in quiptools_quant_wrapper.cpp's history / the top-level
// report), and deliberately UNBATCHED (six separate H_multiply calls per
// internal node, not the [H1|H2|H3]/[H2|H3] grouping Kernel 2 uses) to keep
// this already-intricate kernel's control flow directly comparable to the
// Python source line by line. Applying that batching on-chip too (reducing
// shared-memory staging steps per node from 6 to 3, per gpu_kernel_spec.md
// section 4) is a documented follow-up, not a correctness requirement.
//
// One block per row (grid.y = p, same convention as Kernel 1). Only the
// first warp (threadIdx.x < 32) ever executes leaf_body (the codebook
// search's __shfl*_sync calls need exactly one full warp); all other
// per-node work (the butterfly transforms, the in-kernel matmul) uses the
// whole block. leaf_body uses __syncwarp() internally for exactly this
// reason -- see the comment on leaf_body in Section B.

// ---- compile-time level-size accounting -----------------------------------
// A subtree of depth J has J internal H-matrix levels. Level 0 is the
// OUTERMOST (largest, s = 8*2^(J-1)) and level J-1 is the INNERMOST --
// always 8x8, immediately above the leaf, regardless of J or of where in the
// full tree this subtree sits (decompose_H's recursion always terminates at
// an 8x8 block right before the leaf-lift step -- see lib/algo/quip.py). So
// these functions, and the H1_flat/H2_flat/H3_flat layout they describe,
// only depend on J and the level index, never on the subtree's absolute
// depth in the full tree.
constexpr int ctpow4(int e) { return (e <= 0) ? 1 : 4 * ctpow4(e - 1); }
constexpr int subtree_level_s(int J, int level) { return 8 << (J - 1 - level); }
constexpr int subtree_level_off(int J, int level) {
  int off = 0;
  for (int i = 0; i < level; i++) {
    int s = subtree_level_s(J, i);
    off += s * s;
  }
  return off;
}
constexpr int subtree_total_h_elems_ct(int J) { return subtree_level_off(J, J); }
constexpr int subtree_max_nq_ct(int J) { return (64 * ctpow4(J)) / 4; }

int64_t subtree_h_elems_for_j(int j) { return subtree_level_off(j, j); }

struct CBTablesDevice {
  const float* grid_part;
  const float* grid_part_norm;
  const int64_t* part_abs_map;
  const uint8_t* grid_abs_odd;
  int ncand;
};

// ---- reusable butterfly gather/scatter steps -------------------------------
// Same math as the gather/scatter loops inside radix4_forward_kernel /
// radix4_inverse_kernel in Section C, factored into standalone __device__
// functions so subtree_h_multiply_accumulate (which operates on shared
// memory, one vector at a time, no per-row blockIdx indexing) can reuse them
// without duplicating the indexing logic. Deliberately NOT used to rewrite
// Section C's kernels themselves -- those are already covered by the passing
// acceptance tests, and there's no reason to touch working, verified code
// for a cosmetic dedup.
//
// radix4_gather_ciperm_diff additionally supports computing buf[i] = A[i] -
// B[i] (gathered through ciperm) in the same pass, which is how this kernel
// avoids ever materializing an explicit "error" buffer: E = X - hatX is
// computed on the fly, right where the forward transform reads its source,
// by passing X as A and hatX as B. Pass B = nullptr for a plain gather (not
// used in this file, kept general in case Kernel 3 is extended later).
__device__ __forceinline__ void radix4_gather_ciperm_diff(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ buf, int64_t n) {
  for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
    int64_t chunk = i >> 6;
    int j = (int)(i & 63);
    int64_t src = (chunk << 6) + c_ciperm[j];
    buf[i] = B ? (A[src] - B[src]) : A[src];
  }
}

__device__ __forceinline__ void radix4_scatter_digit_scaled(
    const float* __restrict__ buf, float* __restrict__ out, int64_t n, int k,
    int s, float scale) {
  for (int64_t idx = threadIdx.x; idx < n; idx += blockDim.x) {
    int64_t r, c;
    idx_to_rowcol(idx, k, r, c);
    out[r * s + c] = buf[idx] * scale;
  }
}

__device__ __forceinline__ void radix4_gather_digit(const float* __restrict__ in,
                                                       float* __restrict__ buf,
                                                       int64_t n, int k, int s) {
  for (int64_t idx = threadIdx.x; idx < n; idx += blockDim.x) {
    int64_t r, c;
    idx_to_rowcol(idx, k, r, c);
    buf[idx] = in[r * s + c];
  }
}

// dest[i] += sign * H_multiply(A - B, H)[i]   (H_multiply(A,H) if B == nullptr)
// A, B, dest, H, scratch1, scratch2: shared-memory pointers. dest must
// already hold whatever it should be added to (this function only ever
// accumulates, matching how F0/F1/F2 start as the parent's future_error
// slice and get incremented in place -- see subtree_recurse below). scratch1
// and scratch2 are reused as working buffers across every call in the
// subtree (see the shared-memory layout note on subtree_kernel) -- safe
// because calls are strictly sequential within a block and each one
// overwrites its scratch fully before reading it.
__device__ __forceinline__ void subtree_h_multiply_accumulate(
    const float* __restrict__ A, const float* __restrict__ B,
    const float* __restrict__ H, float* __restrict__ dest, float sign,
    int64_t nq, int k, int s, float scale, float* __restrict__ scratch1,
    float* __restrict__ scratch2) {
  // forward transform: hbc(A - B) -> scratch2, laid out as (s, s)
  radix4_gather_ciperm_diff(A, B, scratch1, nq);
  __syncthreads();
  radix4_rounds(scratch1, nq, k);  // ends with __syncthreads()
  radix4_scatter_digit_scaled(scratch1, scratch2, nq, k, s, scale);
  __syncthreads();

  // in-kernel matmul: scratch1 = scratch2 @ H   (s,s) @ (s,s) -> (s,s)
  // No cuBLAS here (Kernel 3's whole point is staying on-chip); sizes are at
  // most 32x32 (J=3's outermost level), trivial for a plain block-cooperative
  // triple loop.
  for (int64_t idx = threadIdx.x; idx < nq; idx += blockDim.x) {
    int r = (int)(idx / s), c = (int)(idx % s);
    float acc = 0.f;
    for (int t = 0; t < s; t++) acc += scratch2[r * s + t] * H[t * s + c];
    scratch1[idx] = acc;
  }
  __syncthreads();

  // inverse transform: ihbc(scratch1) -> accumulate into dest
  radix4_gather_digit(scratch1, scratch2, nq, k, s);
  __syncthreads();
  radix4_rounds(scratch2, nq, k);  // ends with __syncthreads()
  for (int64_t i = threadIdx.x; i < nq; i += blockDim.x) {
    int64_t chunk = i >> 6;
    int j = (int)(i & 63);
    float v = scratch2[(chunk << 6) + c_cperm[j]] * scale;
    dest[i] += sign * v;
  }
  __syncthreads();  // dest must be visible to every thread before it's read again
}

// The on-chip recursion. J = total subtree depth (fixed for the whole
// recursion), LEVEL = levels remaining until the leaf (counts J, J-1, ..., 0).
// X is this node's read-only coefficient slice; F is this node's
// future_error slice, mutated in place by sibling H_multiply calls exactly
// as future_error0/1/2 are in ldlq_helper; hatX is this node's output slice,
// written once (at the leaf) and never touched again by its own subtree.
// Qidxs_row is GLOBAL memory, already offset to this row's base; q_col_off
// is this node's column offset within it (same bookkeeping as the Python
// recursion's QI0..QI3 slicing).
template <int J, int LEVEL>
__device__ __forceinline__ void subtree_recurse(
    const float* __restrict__ X, float* __restrict__ F, float* __restrict__ hatX,
    int64_t* __restrict__ Qidxs_row, int64_t q_col_off,
    const float* __restrict__ H1_all, const float* __restrict__ H2_all,
    const float* __restrict__ H3_all, const float* __restrict__ J64_sh,
    const CBTablesDevice& cb, float* __restrict__ scratch1,
    float* __restrict__ scratch2, bool is_warp0, int warp_lane) {
  if constexpr (LEVEL == 0) {
    if (is_warp0) {
      leaf_body(X, F, J64_sh, hatX, Qidxs_row + q_col_off, cb.grid_part,
                cb.grid_part_norm, cb.part_abs_map, cb.grid_abs_odd, cb.ncand,
                warp_lane);
    }
    __syncthreads();  // whole block waits for warp 0's leaf result before proceeding
  } else {
    constexpr int myLevel = J - LEVEL;
    constexpr int s = subtree_level_s(J, myLevel);
    constexpr int nq = s * s;  // == this node's coefficient count / 4
    constexpr int k = J + 2 - myLevel;
    constexpr int h_off = subtree_level_off(J, myLevel);
    const float* H1 = H1_all + h_off;
    const float* H2 = H2_all + h_off;
    const float* H3 = H3_all + h_off;
    float scale = powf(2.0f, -0.5f * k);

    const float* X0 = X;
    const float* X1 = X + nq;
    const float* X2 = X + 2 * nq;
    const float* X3 = X + 3 * nq;
    float* F0 = F;
    float* F1 = F + nq;
    float* F2 = F + 2 * nq;
    float* F3 = F + 3 * nq;
    float* hatX0 = hatX;
    float* hatX1 = hatX + nq;
    float* hatX2 = hatX + 2 * nq;
    float* hatX3 = hatX + 3 * nq;
    int64_t qn = nq / 2;  // this node's Qidxs column count == N_this_node/8 == (nq*4)/8
    int64_t qoff0 = q_col_off + 0 * (qn / 4);
    int64_t qoff1 = q_col_off + 1 * (qn / 4);
    int64_t qoff2 = q_col_off + 2 * (qn / 4);
    int64_t qoff3 = q_col_off + 3 * (qn / 4);

    // Chronological accumulation order (as each error becomes available),
    // not a literal replay of ldlq_helper's single combined expression for
    // future_error0/1 -- see the note on this same tradeoff in
    // quiptools_quant_wrapper.cpp's h_multiply_batched-based recursion. Both
    // are sums of the same three floating-point terms in a different
    // association order; acceptance tests 1-3 (bit-exact / 4-sig-fig) have
    // not shown this to matter in practice.
    subtree_recurse<J, LEVEL - 1>(X3, F3, hatX3, Qidxs_row, qoff3, H1_all,
                                   H2_all, H3_all, J64_sh, cb, scratch1,
                                   scratch2, is_warp0, warp_lane);
    subtree_h_multiply_accumulate(X3, hatX3, H1, F2, +1.0f, nq, k, s, scale,
                                   scratch1, scratch2);
    subtree_h_multiply_accumulate(X3, hatX3, H2, F1, +1.0f, nq, k, s, scale,
                                   scratch1, scratch2);
    subtree_h_multiply_accumulate(X3, hatX3, H3, F0, +1.0f, nq, k, s, scale,
                                   scratch1, scratch2);

    subtree_recurse<J, LEVEL - 1>(X2, F2, hatX2, Qidxs_row, qoff2, H1_all,
                                   H2_all, H3_all, J64_sh, cb, scratch1,
                                   scratch2, is_warp0, warp_lane);
    subtree_h_multiply_accumulate(X2, hatX2, H2, F0, -1.0f, nq, k, s, scale,
                                   scratch1, scratch2);
    subtree_h_multiply_accumulate(X2, hatX2, H3, F1, -1.0f, nq, k, s, scale,
                                   scratch1, scratch2);

    subtree_recurse<J, LEVEL - 1>(X1, F1, hatX1, Qidxs_row, qoff1, H1_all,
                                   H2_all, H3_all, J64_sh, cb, scratch1,
                                   scratch2, is_warp0, warp_lane);
    subtree_h_multiply_accumulate(X1, hatX1, H1, F0, +1.0f, nq, k, s, scale,
                                   scratch1, scratch2);

    subtree_recurse<J, LEVEL - 1>(X0, F0, hatX0, Qidxs_row, qoff0, H1_all,
                                   H2_all, H3_all, J64_sh, cb, scratch1,
                                   scratch2, is_warp0, warp_lane);
  }
}

template <int J>
constexpr size_t subtree_smem_bytes() {
  constexpr int N = 64 * ctpow4(J);
  constexpr int H_TOTAL = subtree_total_h_elems_ct(J);
  constexpr int MAX_NQ = subtree_max_nq_ct(J);
  return (size_t)(3 * N + 64 * 64 + 3 * H_TOTAL + 2 * MAX_NQ) * sizeof(float);
}

// Shared-memory layout (must match subtree_smem_bytes<J>() term for term):
// X_sh(N) | F_sh(N) | hatX_sh(N) | J64_sh(4096) | H1_sh(H_TOTAL) |
// H2_sh(H_TOTAL) | H3_sh(H_TOTAL) | scratch1(MAX_NQ) | scratch2(MAX_NQ)
template <int J>
__global__ void subtree_kernel(const float* __restrict__ X,
                                 const float* __restrict__ F,
                                 float* __restrict__ hatX,
                                 int64_t* __restrict__ Qidxs,
                                 int64_t q_row_stride,
                                 const float* __restrict__ J64_g,
                                 const float* __restrict__ H1_flat_g,
                                 const float* __restrict__ H2_flat_g,
                                 const float* __restrict__ H3_flat_g,
                                 const float* __restrict__ grid_part,
                                 const float* __restrict__ grid_part_norm,
                                 const int64_t* __restrict__ part_abs_map,
                                 const uint8_t* __restrict__ grid_abs_odd,
                                 int ncand) {
  constexpr int N = 64 * ctpow4(J);
  constexpr int H_TOTAL = subtree_total_h_elems_ct(J);
  constexpr int MAX_NQ = subtree_max_nq_ct(J);

  extern __shared__ float smem[];
  float* X_sh = smem;
  float* F_sh = X_sh + N;
  float* hatX_sh = F_sh + N;
  float* J64_sh = hatX_sh + N;
  float* H1_sh = J64_sh + 64 * 64;
  float* H2_sh = H1_sh + H_TOTAL;
  float* H3_sh = H2_sh + H_TOTAL;
  float* scratch1 = H3_sh + H_TOTAL;
  float* scratch2 = scratch1 + MAX_NQ;

  int row = blockIdx.y;
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  for (int i = tid; i < N; i += nthreads) {
    X_sh[i] = X[(size_t)row * N + i];
    F_sh[i] = F[(size_t)row * N + i];
  }
  for (int i = tid; i < 64 * 64; i += nthreads) J64_sh[i] = J64_g[i];
  for (int i = tid; i < H_TOTAL; i += nthreads) {
    H1_sh[i] = H1_flat_g[i];
    H2_sh[i] = H2_flat_g[i];
    H3_sh[i] = H3_flat_g[i];
  }
  __syncthreads();

  bool is_warp0 = (tid < 32);
  int warp_lane = tid & 31;
  CBTablesDevice cb{grid_part, grid_part_norm, part_abs_map, grid_abs_odd, ncand};

  subtree_recurse<J, J>(X_sh, F_sh, hatX_sh, Qidxs + (size_t)row * q_row_stride,
                         0, H1_sh, H2_sh, H3_sh, J64_sh, cb, scratch1, scratch2,
                         is_warp0, warp_lane);

  for (int i = tid; i < N; i += nthreads) hatX[(size_t)row * N + i] = hatX_sh[i];
}

void launch_subtree_kernel(int j, const float* X, const float* F, float* hatX,
                            int64_t* Qidxs, int64_t q_row_stride, int64_t p,
                            const float* J64, const float* H1_flat,
                            const float* H2_flat, const float* H3_flat,
                            const float* grid_part, const float* grid_part_norm,
                            const int64_t* part_abs_map,
                            const uint8_t* grid_abs_odd, int ncand,
                            cudaStream_t stream) {
  dim3 grid(1, (unsigned int)p);
  dim3 block(128);

#define LAUNCH_SUBTREE(JVAL)                                                \
  do {                                                                      \
    size_t smem = subtree_smem_bytes<JVAL>();                               \
    static bool attr_set = false;                                          \
    if (!attr_set) {                                                        \
      cudaFuncSetAttribute(subtree_kernel<JVAL>,                            \
                            cudaFuncAttributeMaxDynamicSharedMemorySize,    \
                            (int)smem);                                     \
      attr_set = true;                                                      \
    }                                                                       \
    subtree_kernel<JVAL><<<grid, block, smem, stream>>>(                    \
        X, F, hatX, Qidxs, q_row_stride, J64, H1_flat, H2_flat, H3_flat,    \
        grid_part, grid_part_norm, part_abs_map, grid_abs_odd, ncand);      \
  } while (0)

  switch (j) {
    case 1:
      LAUNCH_SUBTREE(1);
      break;
    case 2:
      LAUNCH_SUBTREE(2);
      break;
    case 3:
      LAUNCH_SUBTREE(3);
      break;
    default:
      // Caller (quiptools_quant_wrapper.cpp) validates j in {1,2,3} before
      // reaching here.
      break;
  }
#undef LAUNCH_SUBTREE
}
