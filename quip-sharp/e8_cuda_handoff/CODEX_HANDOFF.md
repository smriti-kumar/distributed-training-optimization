# Codex GPU handoff checklist

Start here on the GPU machine.

## 1. Establish environment

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
try:
    import triton
    print('triton', triton.__version__)
except Exception as e:
    print('no triton', e)
PY
```

Do not optimize anything until `validate_gpu.py` passes.

## 2. Validate the only custom kernel first

```bash
python validate_gpu.py --random-tests 4
```

The first section compares `_cheap_shell_bound_kernel` with an equivalent
PyTorch `einsum`.  If this fails, inspect only the Triton `Y=T^T X` indexing and
reductions; do not debug branch-and-bound yet.

## 3. Validate exact projection

The same script compares GPU `m=2` and `m=3` results with the CPU oracle.
For generic Gaussian matrices the nearest shell point should almost surely be
unique, so both distance and matrix should agree to float32 tolerance.

If shell residual/lattice membership pass but matrices differ with equal
objective values, investigate a tie before treating it as a correctness bug.

## 4. Benchmark/profile

```bash
python benchmark_gpu.py --batch 1024 --repeats 5
```

Then profile one call with Nsight Systems/Compute or PyTorch profiler.  Record:

- time in `_cheap_shell_bound_kernel`;
- time in `torch.topk`;
- number of cheap-bound survivors;
- number of B8 survivors;
- time in `_b8_scores` and `_wd8_scores`;
- transform-table memory.

## 5. Likely optimization order

### A. Tune the bound kernel

Try `num_warps` 1, 2, and 4.  It handles a fixed 8x8 product, so register
pressure and occupancy matter more than arithmetic throughput.

### B. Factorized transform kernel

The current m=3 path materializes 302,400 matrices `T=A*C`.  Replace that with
a kernel that computes

```text
Z = A^T X
Y = C^T Z
```

inside each program.  This removes ~77 MB float32 table storage and is the right
basis for a nonrecursive m=4 kernel.

### C. Fuse survivor DPs

Only do this after profiling.  The current B8/W(D8) PyTorch DPs process a small
survivor set.  If launch overhead dominates, fuse them into a CUDA/Triton
kernel.  Keep the CPU DP as the oracle.

### D. Direct m=4

`tools/m4_composition_pairs.npy` contains one `(i,j)` pair for each of the
36,766 unique multiplier-4 image classes, with representative

```python
A4 = A2[i] @ A2[j]
```

A direct m=4 kernel should search these unique image reps, not all 72,900
products.  Do not materialize all `36,766*135` 8x8 transforms (~1.27 GB
float32).  Use the factorized `C^T A4^T X` kernel.

## 6. Exactness invariant

Never prune candidate `t` unless a mathematically valid upper bound `ub[t]`
satisfies

```text
ub[t] + numerical_safety_margin <= current_best_exact_score.
```

The two bounds in the current code are safe because W(D8) is a subset of B8,
and every signed permutation chooses exactly one entry from each row and each
column.

## 7. Useful CPU oracle calls

```python
Pc = E8MatrixLattice('cpu', dtype=torch.float64, preload_m2=True)
M2,d2 = Pc.project_m2(Xcpu, return_distance=True)
M3,d3 = Pc.project_m3(Xcpu[:1], return_distance=True)
M4,d4 = Pc.project_m4(Xcpu[:1], return_distance=True)
```

Use small batches for m=3/m=4 CPU validation.
