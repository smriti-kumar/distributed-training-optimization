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
// Kernel 3: fused subtree (gpu_kernel_spec.md section 5)
// ---------------------------------------------------------------------------
// Fuses the bottom `j` levels of the recursion (the j internal-node levels
// closest to the leaves, plus the leaves themselves) into a single launch,
// keeping the whole subtree's working set resident in shared memory and
// doing the internal H_multiply calls on-chip (no cuBLAS, no global memory
// round trips) instead of one launch per node. j is a compile-time template
// parameter (see quiptools_quant.cu); only j in {1, 2, 3} are instantiated --
// j=4 would need X/F/hatX to not be kept as three full-size resident buffers
// (they alone would be 192KB), which this implementation does not attempt.
// See the long comment above subtree_kernel<J> in quiptools_quant.cu for why.
//
// X, F: (p, 64*4^j) row-major float32
// hatX: (p, 64*4^j) row-major float32 output
// Qidxs: (p, 64*4^j/8) int64 output, same row-stride convention as
//    launch_leaf_kernel (q_row_stride, column offset assumed 0 -- the caller
//    is expected to pass a Qidxs view already narrowed to this subtree's
//    column range, the same way bulk_LDLQ narrows Qidxs for ldlq_helper)
// J64: (64, 64) row-major float32, the leaf matrix (same one launch_leaf_kernel uses)
// H1_flat/H2_flat/H3_flat: the subtree's per-level H1/H2/H3 matrices,
//    concatenated in order from the OUTERMOST level of the subtree (matching
//    the LAST j entries of decompose_H's DEC, in DEC order) down to the
//    innermost (always 8x8, immediately above the leaf). Each is a flat 1D
//    array of sum(s_i^2) floats; see subtree_level_off/subtree_level_s in
//    quiptools_quant.cu for the exact per-level offsets, and
//    lib/algo/ldlq_fast.py's _flat_subtree_h for how the Python side builds
//    these from H1_list/H2_list/H3_list.
void launch_subtree_kernel(int j, const float* X, const float* F, float* hatX,
                            int64_t* Qidxs, int64_t q_row_stride, int64_t p,
                            const float* J64, const float* H1_flat,
                            const float* H2_flat, const float* H3_flat,
                            const float* grid_part, const float* grid_part_norm,
                            const int64_t* part_abs_map,
                            const uint8_t* grid_abs_odd, int ncand,
                            cudaStream_t stream);

// Number of floats H1_flat/H2_flat/H3_flat must each have for a given j
// (matches subtree_total_h_elems(j) in quiptools_quant.cu) -- exposed so the
// Python side can size/validate its concatenation without duplicating the
// level-size arithmetic.
int64_t subtree_h_elems_for_j(int j);

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
