#!/usr/bin/env python3
import argparse,time,torch
from e8_triton_projector import E8TritonShellProjector

ap=argparse.ArgumentParser()
ap.add_argument('--batch',type=int,default=1024)
ap.add_argument('--repeats',type=int,default=5)
ap.add_argument('--device',default='cuda')
a=ap.parse_args()
P=E8TritonShellProjector(a.device,preload_m2=True,preload_m3=True)
X=torch.randn(a.batch,8,8,device=a.device)
for m in (2,3):
    P.project_shell(X[:min(16,a.batch)],m); torch.cuda.synchronize()
    ts=[]
    for _ in range(a.repeats):
        torch.cuda.synchronize(); t=time.perf_counter(); P.project_shell(X,m); torch.cuda.synchronize(); ts.append(time.perf_counter()-t)
    print(f'm={m}: min={min(ts):.6f}s median={sorted(ts)[len(ts)//2]:.6f}s batch={a.batch}')
