#!/usr/bin/env python3
"""Regenerate/verify multiplier-2 and multiplier-3 E8 similarity-image reps.

Requires numpy + sympy.  On the CPU used during handoff this takes about 1 s
for m=2 and 8 s for m=3.
"""
from __future__ import annotations
import argparse, base64, zlib
from collections import deque
from pathlib import Path
import numpy as np


def e8_basis():
    roots=[
        np.array([1,-1,-1,-1,-1,-1,-1,1],float)/2,
        np.array([1,1,0,0,0,0,0,0],float),
        np.array([-1,1,0,0,0,0,0,0],float),
        np.array([0,-1,1,0,0,0,0,0],float),
        np.array([0,0,-1,1,0,0,0,0],float),
        np.array([0,0,0,-1,1,0,0,0],float),
        np.array([0,0,0,0,-1,1,0,0],float),
        np.array([0,0,0,0,0,-1,1,0],float),
    ]
    return np.stack(roots,axis=1)


def simple_reflections():
    B=e8_basis()
    return [np.eye(8)-np.outer(B[:,i],B[:,i]) for i in range(8)]


def seed_similarity(m:int):
    I=np.eye(8,dtype=np.int64)
    J2=np.array([[0,-1],[1,0]],dtype=np.int64)
    Z2=np.diag([1,-1]).astype(np.int64)
    J=np.kron(J2,np.eye(4,dtype=np.int64))
    if m==2:
        A=I+J
    elif m==3:
        K=np.kron(Z2,np.kron(J2,np.eye(2,dtype=np.int64)))
        A=I+J+K
    else:
        raise ValueError('m must be 2 or 3')
    assert np.array_equal(A.T@A,m*I)
    return A.astype(float)


def generate(m:int):
    import sympy as sp
    from sympy.matrices.normalforms import hermite_normal_form
    B=e8_basis(); Binv=np.linalg.inv(B)
    G=[]
    for g in simple_reflections():
        M=np.rint(Binv@g@B).astype(np.int64)
        assert np.allclose(B@M,g@B)
        G.append(M)
    K0=np.rint(Binv@seed_similarity(m)@B).astype(np.int64)
    def key(K):
        H=hermite_normal_form(sp.Matrix(K))
        return tuple(int(x) for x in H)
    seen={key(K0):K0}; q=deque([K0])
    while q:
        K=q.popleft()
        for g in G:
            Kn=g@K; k=key(Kn)
            if k not in seen:
                seen[k]=Kn; q.append(Kn)
    expected={2:270,3:2240}[m]
    if len(seen)!=expected:
        raise RuntimeError(f'expected {expected}, got {len(seen)}')
    reps=[]
    for K in seen.values():
        A=B@K@Binv
        if not np.allclose(A.T@A,m*np.eye(8),atol=1e-10):
            raise RuntimeError('bad representative')
        reps.append(A)
    return np.stack(reps)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('m',type=int,choices=[2,3]); ap.add_argument('--out')
    a=ap.parse_args(); A=generate(a.m)
    scale={2:2,3:4}[a.m]
    Q=np.rint(scale*A).astype(np.int8)
    assert np.allclose(Q/scale,A)
    payload=base64.b85encode(zlib.compress(Q.tobytes(),9)).decode('ascii')
    print('shape',A.shape,'scale',scale,'int range',int(Q.min()),int(Q.max()),'payload chars',len(payload))
    if a.out:
        Path(a.out).write_text(payload)
        print('wrote',a.out)

if __name__=='__main__': main()
