// pybind11 bindings for quiptools_quant.
//
// This file used to also implement the whole-tree recursion that
// lib/algo/quip.py:ldlq_helper does, in C++, calling into the CUDA kernels
// below at the leaves and internal nodes -- reasoning being that it also
// removes per-node Python dispatch overhead, per workload_analysis.md
// section 4's point that this workload is ~99.99% dispatch overhead. That
// recursion has been moved to lib/algo/ldlq_fast.py instead, by request: the
// dispatch-overhead argument is real, but it's more important that the
// recursive control flow live somewhere Python-visible and editable than
// that it be maximally fast. What's left here is a set of primitives --
// leaf_quantize (Kernel 1), h_multiply_batched (Kernel 2), subtree_quantize
// (Kernel 3) -- that the Python recursion calls into.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <vector>

#include "quiptools_quant.h"

namespace {
int k_from_n(int64_t n) {
  int k = 0;
  while ((int64_t(1) << (2 * k)) < n) k++;
  TORCH_CHECK((int64_t(1) << (2 * k)) == n, "H_multiply input length must be a power of 4, got ", n);
  return k;
}
}  // namespace

// Kernel 1: fused leaf. Drop-in for the `len(DEC) == 1` branch of
// lib/algo/quip.py:ldlq_helper -- same signature shape (X, future_error, J,
// cb-tables, Qidxs), just as a standalone call instead of buried in the
// recursion. Qidxs is written in place (see the note on q_row_stride in
// quiptools_quant.h -- it is NOT assumed to be a tightly-packed (p,8)
// tensor, since bulk_LDLQ passes ldlq_helper a column-sliced view of a
// larger tensor and the same holds here).
torch::Tensor leaf_quantize(torch::Tensor X, torch::Tensor F, torch::Tensor J,
                              torch::Tensor grid_part, torch::Tensor grid_part_norm,
                              torch::Tensor part_abs_map, torch::Tensor grid_abs_odd,
                              torch::Tensor Qidxs) {
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32 && F.scalar_type() == torch::kFloat32,
              "X and F must be float32 (this fast path does not support use_fp64)");
  TORCH_CHECK(X.dim() == 2 && X.size(1) == 64, "X must be (p, 64)");
  TORCH_CHECK(F.sizes() == X.sizes(), "F (future_error) must match X's shape");
  TORCH_CHECK(J.dim() == 2 && J.size(0) == 64 && J.size(1) == 64, "J must be (64, 64)");
  TORCH_CHECK(Qidxs.scalar_type() == torch::kInt64,
              "Qidxs must be int64 (matches E8P12_codebook.idx_dtype -- NOT int16, see the top-level report)");
  TORCH_CHECK(Qidxs.size(0) == X.size(0) && Qidxs.size(1) == 8, "Qidxs must be (p, 8)");
  TORCH_CHECK(Qidxs.stride(-1) == 1, "Qidxs must be contiguous along its last dimension");
  TORCH_CHECK(grid_part.scalar_type() == torch::kFloat32, "grid_part must be float32");
  TORCH_CHECK(part_abs_map.scalar_type() == torch::kInt64, "part_abs_map must be int64");
  TORCH_CHECK(grid_abs_odd.scalar_type() == torch::kUInt8, "grid_abs_odd must be uint8");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto Xc = X.contiguous();
  auto Fc = F.contiguous();
  auto Jc = J.contiguous();
  auto gp = grid_part.contiguous();
  auto gpn = grid_part_norm.contiguous();
  auto pam = part_abs_map.contiguous();
  auto gao = grid_abs_odd.contiguous();

  auto hatX = torch::empty_like(Xc);
  int64_t p = Xc.size(0);
  launch_leaf_kernel(Xc.data_ptr<float>(), Fc.data_ptr<float>(), Jc.data_ptr<float>(),
                      hatX.data_ptr<float>(), Qidxs.data_ptr<int64_t>(), Qidxs.stride(0),
                      p, gp.data_ptr<float>(), gpn.data_ptr<float>(),
                      pam.data_ptr<int64_t>(), gao.data_ptr<uint8_t>(), (int)gp.size(0),
                      stream);
  return hatX;
}

// Kernel 2: fused, batched H_multiply. X is (p, n), Hcat is (s, k_out * s)
// with s = sqrt(n) -- Hcat is [H1|H2|H3] for k_out==3, [H2|H3] for k_out==2,
// or a lone H for k_out==1 (i.e. this is also how to call plain
// H_multiply(X, H) -- k_out=1, Hcat=H). Returns a list of k_out tensors,
// each (p, n): H_multiply(X, Ha), H_multiply(X, Hb), ... in the order the
// columns of Hcat were concatenated in.
std::vector<torch::Tensor> h_multiply_batched(torch::Tensor X, torch::Tensor Hcat,
                                                 int64_t k_out) {
  TORCH_CHECK(X.is_cuda() && X.scalar_type() == torch::kFloat32, "X must be a float32 CUDA tensor");
  TORCH_CHECK(X.dim() == 2, "X must be (p, n)");
  TORCH_CHECK(Hcat.scalar_type() == torch::kFloat32, "Hcat must be float32");
  TORCH_CHECK(k_out >= 1 && k_out <= 3, "k_out must be 1, 2, or 3");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto Xc = X.contiguous();
  auto Hc = Hcat.contiguous();
  int64_t p = Xc.size(0);
  int64_t n = Xc.size(1);
  int k = k_from_n(n);
  int s = (int)std::llround(std::sqrt((double)n));
  TORCH_CHECK((int64_t)s * (int64_t)s == n, "H_multiply input length is not a perfect square of a power of 2");
  TORCH_CHECK(Hc.dim() == 2 && Hc.size(0) == s && Hc.size(1) == k_out * s,
              "Hcat must be (s, k_out*s) with s = sqrt(n)");
  float scale = powf(2.0f, -0.5f * k);

  auto scratch_fwd = torch::empty({p, n}, Xc.options());
  auto Yfwd = torch::empty({p, n}, Xc.options());
  launch_radix4_forward(Xc.data_ptr<float>(), scratch_fwd.data_ptr<float>(),
                         Yfwd.data_ptr<float>(), p, n, k, s, scale, stream);

  // Yfwd is (p, n) == (p, s, s); matmul against Hcat (s, k_out*s) with cuBLAS
  // via ATen. TF32 is explicitly disabled around this call: workload_analysis.md
  // section 5 requires float32 throughout with no TF32/bf16/fp16, since error
  // accumulates along a chain millions of steps long, and PyTorch's global
  // TF32-for-fp32-matmul setting is not something this extension controls.
  bool prev_tf32 = at::globalContext().allowTF32CuBLAS();
  at::globalContext().setAllowTF32CuBLAS(false);
  auto Yfwd_mat = Yfwd.view({p * s, s});
  auto Z = torch::matmul(Yfwd_mat, Hc);  // (p*s, k_out*s)
  at::globalContext().setAllowTF32CuBLAS(prev_tf32);

  // (p, s, k_out, s) -> (k_out, p, s, s) contiguous, so the k_out results
  // stack into the batch dim for a single inverse-transform launch (spec
  // section 4: "(k, p, N/4) contiguous is the natural choice").
  auto Z4 = Z.view({p, s, k_out, s}).permute({2, 0, 1, 3}).contiguous();
  auto Zflat = Z4.view({k_out * p, n});

  auto scratch_inv = torch::empty({k_out * p, n}, Xc.options());
  auto Xback = torch::empty({k_out * p, n}, Xc.options());
  launch_radix4_inverse(Zflat.data_ptr<float>(), scratch_inv.data_ptr<float>(),
                         Xback.data_ptr<float>(), k_out * p, n, k, s, scale, stream);

  auto Xback3 = Xback.view({k_out, p, n});
  std::vector<torch::Tensor> out;
  out.reserve(k_out);
  for (int64_t i = 0; i < k_out; i++) out.push_back(Xback3.select(0, i));
  return out;
}

// Kernel 3: fused subtree. Fuses the bottom `j` levels of the recursion
// (j in {1,2,3} -- see quiptools_quant.cu Section E for why j=4 isn't
// available) plus the leaves into a single launch. X, F: (p, 64*4^j).
// H1_flat/H2_flat/H3_flat: each subtree_h_elems_for_j(j) floats, the
// subtree's per-level H1/H2/H3 matrices concatenated outermost-level-first
// (see the header comment on launch_subtree_kernel, and
// lib/algo/ldlq_fast.py's _flat_subtree_h). J64 is the same 64x64
// lower-triangular leaf matrix leaf_quantize takes.
torch::Tensor subtree_quantize(torch::Tensor X, torch::Tensor F, torch::Tensor J64,
                                 torch::Tensor H1_flat, torch::Tensor H2_flat,
                                 torch::Tensor H3_flat, torch::Tensor grid_part,
                                 torch::Tensor grid_part_norm, torch::Tensor part_abs_map,
                                 torch::Tensor grid_abs_odd, torch::Tensor Qidxs,
                                 int64_t j) {
  TORCH_CHECK(j >= 1 && j <= 3,
              "subtree_quantize only supports j in {1, 2, 3} -- see the note on "
              "subtree_kernel<J> in quiptools_quant.cu for why j=4 isn't instantiated");
  TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
  TORCH_CHECK(X.scalar_type() == torch::kFloat32 && F.scalar_type() == torch::kFloat32,
              "X and F must be float32 (this fast path does not support use_fp64)");
  int64_t N = 64;
  for (int64_t i = 0; i < j; i++) N *= 4;
  TORCH_CHECK(X.dim() == 2 && X.size(1) == N, "X's last dimension must be 64*4^j for the given j");
  TORCH_CHECK(F.sizes() == X.sizes(), "F (future_error) must match X's shape");
  TORCH_CHECK(J64.dim() == 2 && J64.size(0) == 64 && J64.size(1) == 64, "J64 must be (64, 64)");
  int64_t expected_h = subtree_h_elems_for_j((int)j);
  TORCH_CHECK(H1_flat.numel() == expected_h && H2_flat.numel() == expected_h &&
                  H3_flat.numel() == expected_h,
              "H1_flat/H2_flat/H3_flat must each have subtree_h_elems_for_j(j) elements, got ",
              H1_flat.numel(), "/", H2_flat.numel(), "/", H3_flat.numel(), " vs expected ",
              expected_h);
  TORCH_CHECK(Qidxs.scalar_type() == torch::kInt64, "Qidxs must be int64");
  TORCH_CHECK(Qidxs.size(0) == X.size(0) && Qidxs.size(1) == N / 8, "Qidxs must be (p, N/8)");
  TORCH_CHECK(Qidxs.stride(-1) == 1, "Qidxs must be contiguous along its last dimension");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto Xc = X.contiguous();
  auto Fc = F.contiguous();
  auto J64c = J64.contiguous();
  auto H1c = H1_flat.contiguous();
  auto H2c = H2_flat.contiguous();
  auto H3c = H3_flat.contiguous();
  auto gp = grid_part.contiguous();
  auto gpn = grid_part_norm.contiguous();
  auto pam = part_abs_map.contiguous();
  auto gao = grid_abs_odd.contiguous();

  auto hatX = torch::empty_like(Xc);
  int64_t p = Xc.size(0);
  launch_subtree_kernel((int)j, Xc.data_ptr<float>(), Fc.data_ptr<float>(),
                         hatX.data_ptr<float>(), Qidxs.data_ptr<int64_t>(),
                         Qidxs.stride(0), p, J64c.data_ptr<float>(), H1c.data_ptr<float>(),
                         H2c.data_ptr<float>(), H3c.data_ptr<float>(), gp.data_ptr<float>(),
                         gpn.data_ptr<float>(), pam.data_ptr<int64_t>(),
                         gao.data_ptr<uint8_t>(), (int)gp.size(0), stream);
  return hatX;
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

int64_t subtree_h_elems(int64_t j) { return subtree_h_elems_for_j((int)j); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("leaf_quantize", &leaf_quantize,
        "Kernel 1: fused leaf (the len(DEC)==1 branch of ldlq_helper)");
  m.def("h_multiply_batched", &h_multiply_batched,
        "Kernel 2: fused, batched H_multiply (k_out=1 for a single H_multiply(X,H))");
  m.def("subtree_quantize", &subtree_quantize,
        "Kernel 3: fused subtree (bottom j levels + leaves in one launch, j in {1,2,3})");
  m.def("e8p12_quantize_batch", &e8p12_quantize_batch,
        "Standalone E8P12 codebook search, for acceptance-test isolation");
  m.def("subtree_h_elems", &subtree_h_elems,
        "Number of floats each of H1_flat/H2_flat/H3_flat must have for subtree_quantize(j=...)");
}
