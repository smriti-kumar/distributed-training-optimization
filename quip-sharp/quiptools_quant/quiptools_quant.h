// Kernel launcher declarations shared between quiptools_quant.cu (device code)
// and quiptools_quant_wrapper.cpp (pybind11 / host orchestration).
//
// See workload_analysis.md / gpu_kernel_spec.md at the repo root for the
// algorithm this ports (ldlq_helper in lib/algo/quip.py) and design notes.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel 1: fused leaf (replaces the 8-clique Python loop at len(DEC) == 1)
// ---------------------------------------------------------------------------
// X, F (future_error): (p, 64) row-major float32, tightly packed
// J: (64, 64) row-major float32, lower-triangular, shared by every leaf call
//    in a tile-column
// hatX: (p, 64) row-major float32 output, tightly packed
// Qidxs: (p, 8) int64 output, one E8P12 index per clique. NOT assumed
//    tightly packed -- bulk_LDLQ passes ldlq_helper a column-sliced view of
//    a larger (p, n_total/8) tensor, and this kernel must write through to
//    that same storage ("Qidxs is written in place" per the deliverable
//    contract), so q_row_stride is the caller-supplied element stride
//    between consecutive rows (== Qidxs.stride(0) on the Python/ATen side);
//    the column stride is assumed to be 1 (checked by the caller).
//
// grid_part / grid_part_norm / part_abs_map / grid_abs_odd are the E8P12
// codebook search tables copied from the live `cb` (E8P12_codebook) object
// on the Python side -- see lib/codebook/latticee8_padded12.py. ncand is
// grid_part.size(0) (1366 for the codebook shipped in this repo; passed
// through rather than hardcoded in case the codebook ever changes).
void launch_leaf_kernel(const float* X, const float* F, const float* J,
                         float* hatX, int64_t* Qidxs, int64_t q_row_stride,
                         int64_t p, const float* grid_part,
                         const float* grid_part_norm,
                         const int64_t* part_abs_map,
                         const uint8_t* grid_abs_odd, int ncand,
                         cudaStream_t stream);

// ---------------------------------------------------------------------------
// Kernel 2: fused radix-4 butterfly (hbc_transform / ihbc_transform)
// ---------------------------------------------------------------------------
// Collapses the ~10-20 dispatched ops PyTorch needs per hbc/ihbc call
// (reshape/permute/flip/gather) into a single launch. See section 4 of
// gpu_kernel_spec.md.
//
// Xin: (rows, n) row-major float32, n == 4^k
// scratch: (rows, n) row-major float32 scratch buffer (caller-allocated,
//    contents undefined on entry/exit)
// Yout: (rows, n) row-major float32 output, logically (rows, s, s), s == 2^k
// scale == 2^(-k/2), computed on the host and passed in to avoid powf() in
//    the kernel.
void launch_radix4_forward(const float* Xin, float* scratch, float* Yout,
                            int64_t rows, int64_t n, int k, int s,
                            float scale, cudaStream_t stream);

// Yin: (rows, n) row-major float32, logically (rows, s, s)
// Xout: (rows, n) row-major float32 output
void launch_radix4_inverse(const float* Yin, float* scratch, float* Xout,
                            int64_t rows, int64_t n, int k, int s,
                            float scale, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Standalone codebook kernel, used for acceptance test 4 (codebook port
// isolation) -- feed a batch of 8-vectors through the device search directly,
// independent of the leaf/recursion machinery.
// ---------------------------------------------------------------------------
// X: (rows, 8) row-major float32
// out_vals: (rows, 8) row-major float32
// out_idx: (rows,) int64
void launch_e8p12_quantize_batch(const float* X, float* out_vals,
                                  int64_t* out_idx, int64_t rows,
                                  const float* grid_part,
                                  const float* grid_part_norm,
                                  const int64_t* part_abs_map,
                                  const uint8_t* grid_abs_odd, int ncand,
                                  cudaStream_t stream);
