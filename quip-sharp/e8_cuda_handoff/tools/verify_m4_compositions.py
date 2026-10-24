#!/usr/bin/env python3
"""Verify the constructive multiplier-4 classification.

Fast mode checks the shipped certificate pairs and the similarity equations.
--full recomputes all 270^2 image Hermite normal forms (requires sympy) and
checks that their union has 36,766 elements.
"""
from __future__ import annotations
import argparse,time
from pathlib import Path
import numpy as np
import torch
from e8_matrix_lattice_fast import E8MatrixLattice


def theoretical_count_p2_r2():
    # Coefficient of Z^2 in the p=2 local E8 similar-sublattice factor from
    # Example 3.4.1 of Zimmermann, Configurations of Sublattices.
    p=2
    w1=p**5+2*p**4+3*p**3+2*p**2+2*p+1
    w2=p**8+2*p**7+2*p**6+3*p**5+2*p**4+p**3
    a=[1,p**3,p**5,p**6,p**6]
    d1=sum(a)
    d2=sum(a[i]*a[j] for i in range(len(a)) for j in range(i,len(a)))
    return w2+w1*d1+d2


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--full',action='store_true'); a=ap.parse_args()
    here=Path(__file__).resolve().parent
    pairs=np.load(here/'m4_composition_pairs.npy')
    assert pairs.shape==(36766,2)
    assert theoretical_count_p2_r2()==36766
    P=E8MatrixLattice(dtype=torch.float64)
    # Every certified representative is a multiplier-4 similarity.
    take=torch.as_tensor(pairs[::max(1,len(pairs)//512)],dtype=torch.long)
    A=P.A2[take[:,0]]@P.A2[take[:,1]]
    err=(A.transpose(-1,-2)@A-4*torch.eye(8,dtype=torch.float64)).abs().max().item()
    print('certificate pairs',pairs.shape,'sample max A^T A-4I error',err)
    assert err<1e-12
    if not a.full:
        print('fast checks passed; use --full to recompute all HNF image classes')
        return

    import sympy as sp
    from sympy.matrices.normalforms import hermite_normal_form
    B=P.B.numpy(); Binv=np.linalg.inv(B)
    K=np.rint(np.einsum('ij,ajk,kl->ail',Binv,P.A2.numpy(),B)).astype(np.int64)
    seen={}
    t=time.time()
    for i in range(270):
        for j in range(270):
            M=K[i]@K[j]
            H=hermite_normal_form(sp.Matrix(M))
            key=tuple(int(x) for x in H)
            if key not in seen: seen[key]=(i,j)
    print('full unique classes',len(seen),'time',time.time()-t)
    assert len(seen)==36766
    cert=set(map(tuple,pairs.tolist()))
    # The chosen representative pair for a given HNF depends on traversal order;
    # pairs.npy is generated in the same lexicographic traversal used above.
    assert set(seen.values())==cert
    print('full m=4 composition verification passed')

if __name__=='__main__': main()
