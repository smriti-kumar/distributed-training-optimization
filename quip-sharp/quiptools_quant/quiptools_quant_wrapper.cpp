// pybind11 bindings + host-side orchestration for quiptools_quant.
//
// This file implements the tree recursion that lib/algo/quip.py:ldlq_helper
// does in Python, calling into the CUDA kernels in quiptools_quant.cu at the
// leaves and internal nodes. Doing the recursion here (rather than leaving it
// in Python and only swapping in a CUDA leaf, as gpu_kernel_spec.md section 8
// step 2 suggests as an intermediate milestone) additionally removes the
// per-node Python dispatch overhead, which matters given workload_analysis.md
// section 4's point that this workload is ~99.99% dispatch overhead.
//
// The internal-node schedule below (which H_multiply_batched calls happen
// after which recursive call, and with what sign) is copied from
// lib/algo/quip.py:ldlq_helper, not from gpu_kernel_spec.md's pseudocode --
// they agree, but the source is authoritative.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <vector>

#include "quiptools_quant.h"

namespace {

struct CodebookTables {
  torch::Tensor grid_part;       // (ncand, 8) float32
  torch::Tensor grid_part_norm;  // (ncand,) float32
  torch::Tensor part_abs_map;    // (ncand,) int64
  torch::Tensor grid_abs_odd;    // (256,) uint8
  int ncand;
};

int k_from_n(int64_t n) {
  int k = 0;
  while ((int64_t(1) << (2 * k)) < n) k++;
  TORCH_CHECK((int64_t(1) << (2 * k)) == n, "H_multiply input length must be a power of 4, got ", n);
  return k;
}

// H_multiply_batched(X, Hcat, k_out): X is (p, n), Hcat is (s, k_out * s)
// with s = sqrt(n) (Hcat is [H1|H2|H3] for k_out==3, [H2|H3] for k_out==2, or
// H1 alone for k_out==1). Returns a vector of k_out tensors, each (p, n),
// i.e. H_multiply(X, Ha), H_multiply(X, Hb), ... in the same order the
// columns of Hcat were concatenated in.
std::vector<torch::Tensor> h_multiply_batched(const torch::Tensor& X,
                                                const torch::Tensor& Hcat,
                                                int k_out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t p = X.size(0);
  int64_t n = X.size(1);
  int k = k_from_n(n);
  int s = (int)std::llround(std::sqrt((double)n));
  TORCH_CHECK((int64_t)s * (int64_t)s == n, "H_multiply input length is not a perfect square of a power of 2");
  float scale = powf(2.0f, -0.5f * k);

  auto scratch_fwd = torch::empty({p, n}, X.options());
  auto Yfwd = torch::empty({p, n}, X.options());
  launch_radix4_forward(X.data_ptr<float>(), scratch_fwd.data_ptr<float>(),
                         Yfwd.data_ptr<float>(), p, n, k, s, scale, stream);

  // Yfwd is (p, n) == (p, s, s); matmul against Hcat (s, k_out*s) with cuBLAS
  // via ATen. TF32 is explicitly disabled around this call: workload_analysis.md
  // section 5 requires float32 throughout with no TF32/bf16/fp16, since error
  // accumulates along a chain millions of steps long, and PyTorch's global
  // TF32-for-fp32-matmul setting is not something this extension controls.
  bool prev_tf32 = at::globalContext().allowTF32CuBLAS();
  at::globalContext().setAllowTF32CuBLAS(false);
  auto Yfwd_mat = Yfwd.view({p * s, s});
  auto Z = torch::matmul(Yfwd_mat, Hcat);  // (p*s, k_out*s)
  at::globalContext().setAllowTF32CuBLAS(prev_tf32);

  // (p, s, k_out, s) -> (k_out, p, s, s) contiguous, so the k_out results
  // stack into the batch dim for a single inverse-transform launch (spec
  // section 4: "(k, p, N/4) contiguous is the natural choice").
  auto Z4 = Z.view({p, s, k_out, s}).permute({2, 0, 1, 3}).contiguous();
  auto Zflat = Z4.view({(int64_t)k_out * p, n});

  auto scratch_inv = torch::empty({(int64_t)k_out * p, n}, X.options());
  auto Xback = torch::empty({(int64_t)k_out * p, n}, X.options());
  launch_radix4_inverse(Zflat.data_ptr<float>(), scratch_inv.data_ptr<float>(),
                         Xback.data_ptr<float>(), (int64_t)k_out * p, n, k, s,
                         scale, stream);

  auto Xback3 = Xback.view({k_out, p, n});
  std::vector<torch::Tensor> out;
  out.reserve(k_out);
  for (int i = 0; i < k_out; i++) out.push_back(Xback3.select(0, i));
  return out;
}

torch::Tensor leaf_call(const torch::Tensor& X, const torch::Tensor& F,
                          const torch::Tensor& J, const CodebookTables& cbt,
                          torch::Tensor Qidxs_slice) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto hatX = torch::empty_like(X);
  int64_t p = X.size(0);
  // Qidxs_slice is deliberately NOT made contiguous -- bulk_LDLQ passes
  // ldlq_helper a column-sliced view of a larger tensor and expects writes
  // to land in that same storage. slice() along dim 1 never changes
  // stride(0), so this is the same row stride the caller's top-level Qidxs
  // tensor had; only the column count/offset narrows as the recursion goes
  // deeper. Column stride must still be 1 (checked once, at the top of
  // ldlq_fast).
  launch_leaf_kernel(X.data_ptr<float>(), F.data_ptr<float>(), J.data_ptr<float>(),
                      hatX.data_ptr<float>(), Qidxs_slice.data_ptr<int64_t>(),
                      Qidxs_slice.stride(0), p, cbt.grid_part.data_ptr<float>(),
                      cbt.grid_part_norm.data_ptr<float>(),
                      cbt.part_abs_map.data_ptr<int64_t>(),
                      cbt.grid_abs_odd.data_ptr<uint8_t>(), cbt.ncand, stream);
  return hatX;
}

// Mirrors lib/algo/quip.py:ldlq_helper exactly, including the redundancy
// reduction from workload_analysis.md section 3 (each of E3/E2/E1 gets one
// batched H_multiply call instead of being recomputed per destination).
torch::Tensor ldlq_recurse(torch::Tensor X, torch::Tensor F, int depth,
                             const std::vector<torch::Tensor>& Hcat3_list,
                             const std::vector<torch::Tensor>& Hcat2_list,
                             const std::vector<torch::Tensor>& H1_list,
                             const torch::Tensor& J, const CodebookTables& cbt,
                             torch::Tensor Qidxs_slice) {
  int64_t m = X.size(0);
  int64_t n = X.size(1);
  if (n == 64) {
    return leaf_call(X, F, J, cbt, Qidxs_slice);
  }

  auto Xc = X.view({m, 4, n / 4});
  auto Fc = F.view({m, 4, n / 4});
  auto X0 = Xc.select(1, 0).contiguous();
  auto X1 = Xc.select(1, 1).contiguous();
  auto X2 = Xc.select(1, 2).contiguous();
  auto X3 = Xc.select(1, 3).contiguous();
  auto F0 = Fc.select(1, 0).contiguous();
  auto F1 = Fc.select(1, 1).contiguous();
  auto F2 = Fc.select(1, 2).contiguous();
  auto F3 = Fc.select(1, 3).contiguous();

  int64_t qn = Qidxs_slice.size(1);
  auto QI0 = Qidxs_slice.slice(1, 0 * qn / 4, 1 * qn / 4);
  auto QI1 = Qidxs_slice.slice(1, 1 * qn / 4, 2 * qn / 4);
  auto QI2 = Qidxs_slice.slice(1, 2 * qn / 4, 3 * qn / 4);
  auto QI3 = Qidxs_slice.slice(1, 3 * qn / 4, 4 * qn / 4);

  const auto& Hcat3 = Hcat3_list[depth];
  const auto& Hcat2 = Hcat2_list[depth];
  const auto& H1 = H1_list[depth];

  auto r3 = ldlq_recurse(X3, F3, depth + 1, Hcat3_list, Hcat2_list, H1_list, J,
                          cbt, QI3);
  auto E3 = X3 - r3;
  auto A = h_multiply_batched(E3, Hcat3, 3);  // [H_multiply(E3,H1), H_multiply(E3,H2), H_multiply(E3,H3)]
  F2 = F2 + A[0];
  F1 = F1 + A[1];
  F0 = F0 + A[2];

  auto r2 = ldlq_recurse(X2, F2, depth + 1, Hcat3_list, Hcat2_list, H1_list, J,
                          cbt, QI2);
  auto E2 = X2 - r2;
  auto B = h_multiply_batched(E2, Hcat2, 2);  // [H_multiply(E2,H2), H_multiply(E2,H3)]
  F0 = F0 - B[0];
  F1 = F1 - B[1];

  auto r1 = ldlq_recurse(X1, F1, depth + 1, Hcat3_list, Hcat2_list, H1_list, J,
                          cbt, QI1);
  auto E1 = X1 - r1;
  auto C = h_multiply_batched(E1, H1, 1)[0];  // H_multiply(E1,H1)
  F0 = F0 + C;

  auto r0 = ldlq_recurse(X0, F0, depth + 1, Hcat3_list, Hcat2_list, H1_list, J,
                          cbt, QI0);

  return torch::stack({r0, r1, r2, r3}, 1).reshape({m, n});
}

}  // namespace

torch::Tensor ldlq_fast(torch::Tensor X, torch::Tensor future_error,
                          std::vector<torch::Tensor> Hcat3_list,
                          std::vector<torch::Tensor> Hcat2_list,
                          std::vector<torch::Tensor> H1_list, torch::Tensor J,
                          torch::Tensor grid_part, torch::Tensor grid_part_norm,
                          torch::Tensor part_abs_map, torch::Tensor grid_abs_odd,
                          torch::Tensor Qidxs) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32 (this fast path does not support use_fp64)");
  TORCH_CHECK(future_error.sizes() == X.sizes(), "future_error must match X's shape");
  TORCH_CHECK(future_error.scalar_type() == torch::kFloat32, "future_error must be float32");
  TORCH_CHECK(X.dim() == 2, "X must be (p, N)");
  TORCH_CHECK(Qidxs.scalar_type() == torch::kInt64, "Qidxs must be int64 (matches E8P12_codebook.idx_dtype -- NOT int16, see report)");
  TORCH_CHECK(Qidxs.size(0) == X.size(0) && Qidxs.size(1) == X.size(1) / 8,
              "Qidxs must be (p, N/8)");
  TORCH_CHECK(Qidxs.stride(-1) == 1,
              "Qidxs must be contiguous along its last dimension (row stride "
              "may differ from N/8, e.g. when bulk_LDLQ passes a column-sliced "
              "view -- that's fine and handled -- but the column stride must be 1)");
  TORCH_CHECK(Hcat3_list.size() == Hcat2_list.size() && Hcat2_list.size() == H1_list.size(),
              "Hcat3_list/Hcat2_list/H1_list must have one entry per internal tree depth");
  TORCH_CHECK(J.dim() == 2 && J.size(0) == 64 && J.size(1) == 64, "J must be (64, 64)");

  CodebookTables cbt;
  cbt.grid_part = grid_part.contiguous();
  cbt.grid_part_norm = grid_part_norm.contiguous();
  cbt.part_abs_map = part_abs_map.contiguous();
  cbt.grid_abs_odd = grid_abs_odd.contiguous();
  cbt.ncand = (int)grid_part.size(0);
  TORCH_CHECK(cbt.grid_part.scalar_type() == torch::kFloat32, "grid_part must be float32");
  TORCH_CHECK(cbt.part_abs_map.scalar_type() == torch::kInt64, "part_abs_map must be int64");
  TORCH_CHECK(cbt.grid_abs_odd.scalar_type() == torch::kUInt8, "grid_abs_odd must be uint8");

  return ldlq_recurse(X.contiguous(), future_error.contiguous(), 0, Hcat3_list,
                       Hcat2_list, H1_list, J.contiguous(), cbt, Qidxs);
}

std::vector<torch::Tensor> e8p12_quantize_batch(torch::Tensor X,
                                                   torch::Tensor grid_part,
                                                   torch::Tensor grid_part_norm,
                                                   torch::Tensor part_abs_map,
                                                   torch::Tensor grid_abs_odd) {
  TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32, "X must be a float32 CUDA tensor");
  TORCH_CHECK(X.dim() == 2 && X.size(1) == 8, "X must be (rows, 8)");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto Xc = X.contiguous();
  int64_t rows = X.size(0);
  auto out_vals = torch::empty({rows, 8}, X.options());
  auto out_idx = torch::empty({rows}, X.options().dtype(torch::kInt64));
  auto gp = grid_part.contiguous();
  auto gpn = grid_part_norm.contiguous();
  auto pam = part_abs_map.contiguous();
  auto gao = grid_abs_odd.contiguous();
  launch_e8p12_quantize_batch(Xc.data_ptr<float>(), out_vals.data_ptr<float>(),
                               out_idx.data_ptr<int64_t>(), rows,
                               gp.data_ptr<float>(), gpn.data_ptr<float>(),
                               pam.data_ptr<int64_t>(), gao.data_ptr<uint8_t>(),
                               (int)gp.size(0), stream);
  return {out_vals, out_idx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ldlq_fast", &ldlq_fast,
        "Fused CUDA replacement for lib.algo.quip.ldlq_helper");
  m.def("e8p12_quantize_batch", &e8p12_quantize_batch,
        "Standalone E8P12 codebook search, for acceptance-test isolation");
}
