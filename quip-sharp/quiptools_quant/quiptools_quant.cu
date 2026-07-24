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
  __shared__ float error_sh[64];
  __shared__ float hatX_sh[64];

  for (int i = lane; i < 64 * 64; i += 32) J_sh[i] = J[i];
  for (int i = lane; i < 64; i += 32) {
    X_sh[i] = X[(size_t)row * 64 + i];
    F_sh[i] = F[(size_t)row * 64 + i];
  }
  __syncthreads();

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
    if (lane == 0) Qidxs[(size_t)row * q_row_stride + c] = idx;
    __syncthreads();
  }

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
