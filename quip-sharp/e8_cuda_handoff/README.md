# E8 ⊗ E8 matrix-lattice projector: CPU oracle + CUDA/Triton handoff

This directory is a self-contained handoff for the `8 x 8` realization of
`E8 ⊗ E8` used in our discussion.

The shell convention is

```text
S_m = { M in E8⊗E8 : M^T M = m I_8 },   ||M||_F^2 = 8m.
```

## What is complete here

`e8_matrix_lattice_fast.py` contains CPU-tested exact (up to floating-point
tie-breaking) routines for:

- full Euclidean CVP onto `E8 ⊗ E8`;
- projection onto `S_1`;
- projection onto `S_2`;
- projection onto `S_3`;
- projection onto `S_4`.

The `m=3` table contains all 2,240 multiplier-3 image lattices.  It is embedded
as a compressed int8 table, so normal runtime has no NumPy/SymPy dependency.

For `m=4` we verified a useful constructive fact.  The 270 x 270 products of
multiplier-2 image representatives produce exactly **36,766** distinct
multiplier-4 image sublattices.  This matches the independent E8
similar-sublattice counting formula.  Consequently the shell can be searched
exactly as

```text
S_4 = S_2 S_2.
```

The CPU `project_m4` implementation uses this recursive description rather than
materializing roughly 5 million W(D8) cosets.

## CUDA/Triton status

`e8_triton_projector.py` is the GPU handoff implementation.

For `m=2` and `m=3` it uses a custom Triton kernel for the expensive pass over
all W(D8)-coset transforms.  The kernel computes the safe bound

```text
min( sum_i max_j |Y_ij|, sum_j max_i |Y_ij| ),   Y = T^T X.
```

Only candidates that can still beat the incumbent continue to the tighter B8
assignment bound and then the exact W(D8) DP.  Those survivor DPs currently use
the already-tested PyTorch implementation on CUDA.  This is deliberate: the
first handoff minimizes the amount of untested custom CUDA code.

`m=4` is implemented on CUDA recursively through the exact `m=2` GPU projector.
This is complete but is not expected to be the final optimized `m=4` kernel.

**Important:** this environment has CPU-only PyTorch and no Triton/CUDA runtime.
The CPU mathematics and references were tested here; the Triton kernel itself
has only been syntax/static checked.  Run `validate_gpu.py` first on the GPU
machine before using its results.

## First commands to run on the GPU machine

Use a CUDA-enabled PyTorch build, then install Triton if it is not already
included.

```bash
python validate_gpu.py --device cuda
```

The validation script first compares the custom Triton bound kernel against a
direct PyTorch computation on sampled transforms.  It then compares `m=2` and
`m=3` GPU projections against the exact CPU oracle on random inputs and checks
known shell points.

After validation:

```bash
python benchmark_gpu.py --batch 1024 --repeats 5
```

The default benchmark covers `m=2` and `m=3`.  `m=4` should be benchmarked
separately because the initial recursive implementation is substantially more
expensive.

## Python API

CPU oracle:

```python
import torch
from e8_matrix_lattice_fast import E8MatrixLattice

P = E8MatrixLattice("cpu", dtype=torch.float32, preload_m2=True)
X = torch.randn(1024, 8, 8)

Y  = P.cvp(X)
M1 = P.project_m1(X)
M2 = P.project_m2(X)
M3 = P.project_m3(X[:4])       # CPU reference is slower
M4 = P.project_m4(X[:1])       # recursive correctness oracle
```

CUDA/Triton shell projector:

```python
import torch
from e8_triton_projector import E8TritonShellProjector

P = E8TritonShellProjector("cuda", preload_m2=True, preload_m3=True)
X = torch.randn(1024, 8, 8, device="cuda")

M2 = P.project_m2(X)
M3 = P.project_m3(X)

# m=4 is exact but recursive and currently intended for smaller batches.
M4 = P.project_m4(X[:4])
```

The initial Triton path accepts CUDA `float32`.  The CPU oracle supports both
`float32` and `float64`.

## CPU checks already run

The packaged tests check:

- all 2,240 embedded `m=3` representatives satisfy `A^T A = 3I`;
- random `m=3` outputs are lattice points and have `M^T M = 3I`;
- known `m=3` shell points project to themselves;
- the optimized `m=2` branch-and-bound projector agrees with the old exhaustive
  projector;
- a known `m=4` product projects to itself exactly.

Run them with

```bash
PYTHONPATH=. pytest -q tests/test_cpu_reference.py
```

On the CPU used to prepare this handoff, those five tests pass.

## Regenerating the finite tables

`tools/generate_similarity_reps.py` independently regenerates the `m=2` and
`m=3` image representatives from explicit seed similarities and the W(E8)
orbit of their image sublattices.  It uses exact integer Hermite normal forms.

```bash
python tools/generate_similarity_reps.py 3 --out /tmp/a3.txt
```

`tools/verify_m4_compositions.py` verifies the shipped `m=4` composition
certificate.  The fast mode checks the certificate and similarity equations;
`--full` recomputes all `270^2` Hermite normal forms.

```bash
PYTHONPATH=. python tools/verify_m4_compositions.py
PYTHONPATH=. python tools/verify_m4_compositions.py --full
```

## Where to optimize next

The current GPU design intentionally leaves several high-value targets for the
GPU machine:

1. Profile `cheap_bounds`; tune `num_warps` and batch chunk sizes.
2. Fuse the B8 and W(D8) survivor DPs into Triton/CUDA if PyTorch kernel-launch
   overhead is material.
3. For `m=3`, avoid the ~77 MB materialized transform table by a factorized
   kernel that forms `C^T A^T X` directly from the 2,240 `A` reps and 135 `C`
   reps.
4. For `m=4`, use the 36,766 unique composition pairs and a factorized bound
   kernel instead of recursively searching all 270 outer `m=2` reps.
5. After shell kernels are stable, port the branch-and-bound full
   `E8 ⊗ E8` CVP to CUDA if that operation is also on the critical path.

See `CODEX_HANDOFF.md` for a concrete debugging/tuning checklist.

## Mathematical reference for the m=4 count

The count comes from the local similar-sublattice zeta factor for E8 in
Example 3.4.1 of Marc Christian Zimmermann, *Configurations of Sublattices*.
For multiplier `4 = 2^2`, the relevant coefficient is 36,766.  The included
full composition check independently obtains the same number from the 72,900
products of the 270 multiplier-2 image representatives.
