#!/usr/bin/env python3
"""Run this first on the CUDA machine.

It validates the Triton m=2/m=3 projector against the exact CPU oracle and then
prints a small benchmark.  It is intentionally self-contained so Codex can use
its output as the starting point for kernel tuning.
"""
from __future__ import annotations
import argparse, time, sys
import torch

from e8_matrix_lattice_fast import E8MatrixLattice


def timed_cuda(fn):
    torch.cuda.synchronize()
    t0=time.perf_counter(); out=fn(); torch.cuda.synchronize()
    return out, time.perf_counter()-t0


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--random-tests', type=int, default=8)
    ap.add_argument('--benchmark-batch', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=1234)
    args=ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('CUDA is not available in this PyTorch installation.')
    try:
        from e8_triton_projector import E8TritonShellProjector
    except Exception as e:
        raise SystemExit(f'Could not import Triton projector: {e}')

    torch.manual_seed(args.seed)
    gpu=E8TritonShellProjector(args.device, preload_m2=True, preload_m3=True)
    cpu=E8MatrixLattice('cpu', dtype=torch.float32, preload_m2=True, preload_m3=False)

    print('torch', torch.__version__)
    print('device', torch.cuda.get_device_name(torch.device(args.device)))

    # Unit-test the custom Triton kernel independently of branch-and-bound.
    print('\n=== Triton cheap-bound kernel check ===')
    Xk=torch.randn(3,8,8,device=args.device)
    Tfull=gpu.ref._ensure_T2()
    idx=torch.randperm(Tfull.shape[0],device=args.device)[:1024]
    Tk=Tfull[idx].contiguous()
    uk=gpu.cheap_bounds(Xk,Tk)
    Y=torch.einsum('tji,bjk->btik',Tk,Xk)
    ref_ub=torch.minimum(Y.abs().amax(-1).sum(-1),Y.abs().amax(-2).sum(-1))
    ub_err=(uk-ref_ub).abs().max().item()
    print('max cheap-bound kernel error',ub_err)
    assert ub_err < 2e-4, ub_err

    for m in (2,3):
        print(f'\n=== exactness m={m} ===')
        Xcpu=torch.randn(args.random_tests,8,8,dtype=torch.float32)
        Xgpu=Xcpu.to(args.device)

        (Mgpu,dgpu,stats),tg=timed_cuda(lambda: gpu.project_shell(
            Xgpu,m,return_distance=True,return_stats=True,
        ))
        t0=time.perf_counter()
        Mcpu,dcpu=cpu.project_shell(
            Xcpu,m,return_distance=True,algorithm='branch_bound'
        )
        tc=time.perf_counter()-t0

        Mg=Mgpu.cpu(); dg=dgpu.cpu()
        d_err=(dg-dcpu).abs().max().item()
        # Generic Gaussian inputs almost surely have a unique nearest point.
        m_err=(Mg-Mcpu).abs().max().item()
        shell=(Mg.transpose(-1,-2)@Mg - m*torch.eye(8)).abs().max().item()
        lat=bool(cpu.is_lattice_point(Mg,atol=3e-5).all())
        print('GPU time',tg,'CPU oracle time',tc)
        print('max distance error',d_err)
        print('max matrix error',m_err)
        print('max shell residual',shell,'all lattice points',lat)
        print('chunk stats',stats)
        assert d_err < 2e-3, (m,d_err)
        assert shell < 2e-4, (m,shell)
        assert lat

        # Exact known shell point must project to zero distance.
        A=(cpu.A2[0] if m==2 else cpu.A3[0]).to(args.device)
        A=A.unsqueeze(0)
        Ma,da=gpu.project_shell(A,m,return_distance=True)
        assert float(da.max().cpu()) < 2e-4
        assert float((Ma-A).abs().max().cpu()) < 2e-4

    print('\n=== benchmark ===')
    X=torch.randn(args.benchmark_batch,8,8,device=args.device)
    # warmup
    for m in (2,3):
        gpu.project_shell(X[:min(16,args.benchmark_batch)],m)
    torch.cuda.synchronize()
    for m in (2,3):
        (_,stats),dt=timed_cuda(lambda m=m: gpu.project_shell(X,m,return_stats=True))
        print(f'm={m}: {dt:.6f} s total = {1e6*dt/args.benchmark_batch:.2f} us/matrix')
        cheap=sum(s.cheap_bound_survivors for s in stats)
        b8=sum(s.b8_survivors for s in stats)
        print('    cheap-bound survivors',cheap,'B8 survivors',b8)

if __name__=='__main__':
    main()
