"""
e8_triton_projector.py

CUDA/Triton handoff implementation for exact m=2 and m=3 shell projection.

What is implemented here
------------------------
The expensive all-candidate *cheap upper-bound* pass is a custom Triton kernel.
The exact B8 and W(D8) survivor DPs reuse the vectorized PyTorch implementation
from e8_matrix_lattice_fast.py, but execute on CUDA.  This keeps the first GPU
version simple and preserves the exact branch-and-bound logic used by the CPU
reference.

The search is exact (up to floating-point tie handling) provided the Triton
upper-bound kernel computes

    min(sum_i max_j |(T^T X)_{ij}|,
        sum_j max_i |(T^T X)_{ij}|)

correctly.  Both terms are upper bounds on max_{D in W(D8)} <T^T X,D>.

This file cannot be runtime-tested in the OpenAI execution environment because
that environment has CPU-only PyTorch and no Triton/CUDA runtime.  The included
GPU validation script compares every GPU result against the exact CPU oracle.

Initial scope
-------------
* float32 CUDA only for the Triton bound pass (the CPU oracle supports float64).
* m=2: 270*135 = 36,450 W(D8) cosets.
* m=3: 2240*135 = 302,400 W(D8) cosets.
* m=4 uses the validated identity S_4 = S_2 S_2.  The 270^2 multiplier-2
  image compositions yield exactly 36,766 distinct multiplier-4 image classes,
  matching the independent E8 similar-sublattice count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from e8_matrix_lattice_fast import E8MatrixLattice

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:  # pragma: no cover - exercised on non-CUDA handoff machines
    triton = None
    tl = None
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.jit
    def _cheap_shell_bound_kernel(
        x_ptr,
        t_ptr,
        out_ptr,
        ntrans: tl.constexpr,
    ):
        """One Triton program per (batch item, transform).

        X is [B,8,8], T is [N,8,8], both contiguous float32.
        We form Y=T^T X entirely in registers and store one scalar upper bound.
        """
        pid = tl.program_id(0)
        b = pid // ntrans
        q = pid - b * ntrans

        ii = tl.arange(0, 8)[:, None]
        jj = tl.arange(0, 8)[None, :]
        y = tl.zeros((8, 8), dtype=tl.float32)
        for p in range(8):
            tv = tl.load(t_ptr + q * 64 + p * 8 + ii)
            xv = tl.load(x_ptr + b * 64 + p * 8 + jj)
            y += tv * xv

        ay = tl.abs(y)
        row_mx = tl.max(ay, axis=1)
        col_mx = tl.max(ay, axis=0)
        row_bound = tl.sum(row_mx, axis=0)
        col_bound = tl.sum(col_mx, axis=0)
        bound = tl.minimum(row_bound, col_bound)
        tl.store(out_ptr + pid, bound)


@dataclass
class TritonShellStats:
    n_transforms: int
    initial_candidates: int
    cheap_bound_survivors: int
    b8_survivors: int


class E8TritonShellProjector:
    """Exact CUDA branch-and-bound projector for S_2 and S_3.

    Parameters
    ----------
    device:
        CUDA device.
    preload_m2 / preload_m3:
        Materialize T=A*C transform tables immediately.  m=3 uses about 77 MB
        in float32 for its 302,400 8x8 matrices.

    Notes
    -----
    The transform table is intentionally materialized in this first kernel
    version.  It makes the Triton kernel very small and robust.  A later fused
    kernel can store only A and C and form A*C on the fly, saving memory.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        preload_m2: bool = True,
        preload_m3: bool = False,
    ):
        if not _HAS_TRITON:
            raise RuntimeError("Triton is not installed. Install a CUDA PyTorch build with triton.")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("E8TritonShellProjector requires a CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")

        self.ref = E8MatrixLattice(
            device=self.device,
            dtype=torch.float32,
            preload_m2=preload_m2,
            preload_m3=preload_m3,
        )
        if preload_m2:
            self.ref._ensure_T2()
        if preload_m3:
            self.ref._ensure_T3()

    def _transforms(self, m: int) -> torch.Tensor:
        if m == 1:
            return self.ref.C1
        if m == 2:
            return self.ref._ensure_T2()
        if m == 3:
            return self.ref._ensure_T3()
        raise ValueError("GPU handoff currently supports m=1,2,3 only")

    @torch.no_grad()
    def cheap_bounds(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Return the safe cheap upper bound for every transform."""
        if X.device.type != "cuda" or T.device.type != "cuda":
            raise ValueError("cheap_bounds expects CUDA tensors")
        if X.dtype != torch.float32 or T.dtype != torch.float32:
            raise ValueError("initial Triton kernel currently supports float32 only")
        if not X.is_contiguous():
            X = X.contiguous()
        if not T.is_contiguous():
            T = T.contiguous()
        bsz = X.shape[0]
        nt = T.shape[0]
        out = torch.empty((bsz, nt), device=X.device, dtype=torch.float32)
        grid = (bsz * nt,)
        _cheap_shell_bound_kernel[grid](X, T, out, ntrans=nt, num_warps=1)
        return out

    @staticmethod
    def _scatter_pair_max(
        best_score: torch.Tensor,
        best_t: torch.Tensor,
        bb: torch.Tensor,
        tt: torch.Tensor,
        ss: torch.Tensor,
        nt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update per-batch best score/candidate from sparse candidate pairs."""
        if ss.numel() == 0:
            return best_score, best_t
        bsz = best_score.numel()
        cand_score = torch.full_like(best_score, float("-inf"))
        cand_score.scatter_reduce_(0, bb, ss, reduce="amax", include_self=True)
        improve = cand_score > best_score
        if not improve.any():
            return best_score, best_t

        # Select one candidate attaining each batch maximum.  Ties are harmless.
        winner = ss == cand_score[bb]
        cand_id = torch.full((bsz,), nt, device=best_t.device, dtype=torch.long)
        cand_id.scatter_reduce_(
            0,
            bb[winner],
            tt[winner],
            reduce="amin",
            include_self=True,
        )
        best_score = torch.where(improve, cand_score, best_score)
        best_t = torch.where(improve, cand_id, best_t)
        return best_score, best_t

    @torch.no_grad()
    def _project_chunk(
        self,
        X: torch.Tensor,
        *,
        m: int,
        initial_candidates: int,
        pair_chunk_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, TritonShellStats]:
        T = self._transforms(m)
        bsz, nt = X.shape[0], T.shape[0]
        k0 = min(int(initial_candidates), nt)

        # Stage 1: custom CUDA kernel over *all* transforms.
        ub = self.cheap_bounds(X, T)

        # Stage 2: obtain a real feasible score from the largest cheap bounds.
        _, topi = torch.topk(ub, k0, dim=-1, largest=True, sorted=False)
        Y0 = torch.matmul(T[topi].transpose(-1, -2), X[:, None, :, :])
        s0 = self.ref._wd8_scores(Y0).reshape(bsz, k0)
        best_score, kk = s0.max(dim=-1)
        ar = torch.arange(bsz, device=X.device)
        best_t = topi[ar, kk]

        seeded = torch.zeros((bsz, nt), device=X.device, dtype=torch.bool)
        seeded.scatter_(1, topi, True)
        tol = self.ref._tol(best_score)[:, None]
        pairs = ((ub + tol > best_score[:, None]) & (~seeded)).nonzero(as_tuple=False)
        cheap_survivors = int(pairs.shape[0])
        del ub, seeded, Y0, s0

        # Stage 3: tighter exact upper bound from B8, then exact W(D8) only for
        # candidates that can still beat the incumbent.
        exact_b = []
        exact_t = []
        if pairs.numel():
            pb, pt = pairs[:, 0], pairs[:, 1]
            for lo in range(0, pb.numel(), pair_chunk_size):
                bb = pb[lo:lo + pair_chunk_size]
                tt = pt[lo:lo + pair_chunk_size]
                YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                bu = self.ref._b8_scores(YY)
                keep = bu + self.ref._tol(best_score[bb]) > best_score[bb]
                if keep.any():
                    exact_b.append(bb[keep])
                    exact_t.append(tt[keep])

        b8_survivors = 0
        if exact_b:
            kb = torch.cat(exact_b)
            kt = torch.cat(exact_t)
            b8_survivors = int(kb.numel())
            for lo in range(0, kb.numel(), pair_chunk_size):
                bb = kb[lo:lo + pair_chunk_size]
                tt = kt[lo:lo + pair_chunk_size]
                YY = torch.bmm(T[tt].transpose(1, 2), X[bb])
                ss = self.ref._wd8_scores(YY)
                best_score, best_t = self._scatter_pair_max(
                    best_score, best_t, bb, tt, ss, nt,
                )

        # Reconstruct only the winning W(D8) element.  This is tiny compared
        # with the all-transform search and can use the tested PyTorch traceback.
        Tbest = T[best_t]
        Ybest = torch.bmm(Tbest.transpose(1, 2), X)
        Dbest, _ = self.ref._wd8_best(Ybest)
        M = torch.bmm(Tbest, Dbest)
        d2 = X.square().sum(dim=(-2, -1)) + float(8 * m) - 2.0 * best_score
        d2 = torch.clamp_min(d2, 0.0)
        stats = TritonShellStats(nt, k0, cheap_survivors, b8_survivors)
        return M, d2, stats

    @torch.no_grad()
    def project_shell(
        self,
        x: torch.Tensor,
        m: int,
        *,
        batch_chunk_size: Optional[int] = None,
        initial_candidates: Optional[int] = None,
        pair_chunk_size: int = 65536,
        return_distance: bool = False,
        return_stats: bool = False,
    ):
        if m == 4:
            if return_stats:
                raise ValueError("return_stats is not yet implemented for recursive m=4")
            return self.project_m4(
                x, batch_chunk_size=1 if batch_chunk_size is None else batch_chunk_size,
                return_distance=return_distance,
            )
        if m not in (1, 2, 3):
            raise ValueError("m must be 1, 2, 3, or 4")
        if x.shape[-2:] != (8, 8):
            raise ValueError("expected x[...,8,8]")
        if x.device.type != "cuda":
            raise ValueError("input must already be on CUDA")
        if x.dtype != torch.float32:
            raise ValueError("initial Triton path supports torch.float32")

        lead = x.shape[:-2]
        X = x.reshape(-1, 8, 8).contiguous()
        n = X.shape[0]

        if initial_candidates is None:
            initial_candidates = {1: 8, 2: 64, 3: 128}[m]
        if batch_chunk_size is None:
            # Dense cheap-bound storage is B_chunk*N.  These defaults use
            # ~37 MB for m=2 and ~39 MB for m=3.
            batch_chunk_size = {1: 1024, 2: 256, 3: 32}[m]

        ys, ds, stats = [], [], []
        for lo in range(0, n, batch_chunk_size):
            M, d2, st = self._project_chunk(
                X[lo:lo + batch_chunk_size],
                m=m,
                initial_candidates=initial_candidates,
                pair_chunk_size=pair_chunk_size,
            )
            ys.append(M)
            ds.append(d2)
            stats.append(st)

        Y = torch.cat(ys, dim=0).reshape(*lead, 8, 8)
        d2 = torch.cat(ds, dim=0).reshape(lead)
        if return_distance and return_stats:
            return Y, d2, stats
        if return_distance:
            return Y, d2
        if return_stats:
            return Y, stats
        return Y

    @torch.no_grad()
    def project_m4(
        self,
        x: torch.Tensor,
        *,
        batch_chunk_size: int = 1,
        outer_a_chunk: int = 1,
        inner_batch_chunk_size: int = 256,
        inner_initial_candidates: int = 64,
        inner_pair_chunk_size: int = 65536,
        return_distance: bool = False,
    ):
        """Exact CUDA m=4 projector via S_4 = S_2 S_2.

        This is deliberately recursive rather than flattening the roughly
        4.96 million W(D8) cosets.  For each of the 270 outer multiplier-2
        representatives A, it projects A^T X onto S_2 and keeps the best.
        It is complete because the 270^2 image compositions cover all 36,766
        multiplier-4 image sublattices.
        """
        if x.shape[-2:] != (8, 8):
            raise ValueError("expected x[...,8,8]")
        if x.device.type != "cuda" or x.dtype != torch.float32:
            raise ValueError("m=4 Triton path expects CUDA float32 input")
        lead=x.shape[:-2]
        X=x.reshape(-1,8,8).contiguous()
        outs=[]; douts=[]
        for blo in range(0,X.shape[0],batch_chunk_size):
            Xb=X[blo:blo+batch_chunk_size]
            bsz=Xb.shape[0]
            best_score=torch.full((bsz,),float("-inf"),device=X.device,dtype=X.dtype)
            best_M=torch.zeros((bsz,8,8),device=X.device,dtype=X.dtype)
            for alo in range(0,270,outer_a_chunk):
                A=self.ref.A2[alo:alo+outer_a_chunk]
                Z=torch.einsum("aji,bjk->baik",A,Xb)
                ss=A.shape[0]
                Zf=Z.reshape(-1,8,8)
                M2=self.project_m2(
                    Zf,
                    batch_chunk_size=min(inner_batch_chunk_size,max(1,Zf.shape[0])),
                    initial_candidates=inner_initial_candidates,
                    pair_chunk_size=inner_pair_chunk_size,
                ).reshape(bsz,ss,8,8)
                score=(Z*M2).sum(dim=(-2,-1))
                cs,ci=score.max(dim=1)
                improve=cs>best_score
                ar=torch.arange(bsz,device=X.device)
                cand=torch.matmul(A[ci],M2[ar,ci])
                best_M=torch.where(improve[:,None,None],cand,best_M)
                best_score=torch.where(improve,cs,best_score)
            d2=Xb.square().sum(dim=(-2,-1))+32.0-2.0*best_score
            outs.append(best_M); douts.append(torch.clamp_min(d2,0.0))
        M=torch.cat(outs,0).reshape(*lead,8,8)
        d2=torch.cat(douts,0).reshape(lead)
        if return_distance:
            return M,d2
        return M

    def project_m1(self, x: torch.Tensor, **kwargs):
        return self.project_shell(x, 1, **kwargs)

    def project_m2(self, x: torch.Tensor, **kwargs):
        return self.project_shell(x, 2, **kwargs)

    def project_m3(self, x: torch.Tensor, **kwargs):
        return self.project_shell(x, 3, **kwargs)
