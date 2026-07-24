"""Drop-in CUDA-accelerated replacement for lib.algo.quip.ldlq_helper.

This is a NEW file -- lib/algo/quip.py is not modified by this module. See
the top-level report / commit message for the one-line change needed to
actually wire this in (swapping the `ldlq_helper` call inside `bulk_LDLQ`
for `ldlq_fast`), which is deliberately left for you to apply by hand.

Usage:

    from lib.algo import ldlq_fast
    hatX = ldlq_fast.ldlq_fast(X, future_error, DEC, cb, Qidxs)

matches the exact signature/semantics of lib.algo.quip.ldlq_helper (same
inputs, same in-place write to Qidxs, same return value), so it can be used
as a straight substitute anywhere ldlq_helper is called -- including inside
bulk_LDLQ's greedy local-search passes, which call the same function.

Design note: the recursion below is plain Python, deliberately structured to
mirror lib.algo.quip.ldlq_helper line for line (same split order, same
future_error accumulation), with three CUDA calls swapped in for the
expensive parts:
  - quiptools_quant_cuda.leaf_quantize   -- Kernel 1, the len(DEC)==1 branch
  - quiptools_quant_cuda.h_multiply_batched -- Kernel 2, replaces H_multiply
  - quiptools_quant_cuda.subtree_quantize   -- Kernel 3, replaces an entire
    bottom subtree (several levels of this recursion at once) with one launch

An earlier version of this file called a single C++ function
(quiptools_quant_cuda.ldlq_fast) that did the whole tree recursion in C++,
to also cut per-node Python dispatch overhead (workload_analysis.md section
4: this workload is ~99.99% dispatch overhead). That's gone now, by request
-- the recursion lives here instead, in Python, readable and editable,
accepting the extra Python-side dispatch cost. Kernel 3 (subtree_quantize)
claws a meaningful chunk of that back on its own terms: every node it
handles is nodes the Python recursion never has to dispatch into at all.

Constraints (see workload_analysis.md section 5):
  - X, future_error, and the DEC matrices must be float32 CUDA tensors.
    args.use_fp64 runs are NOT supported by this path; the caller should
    keep using ldlq_helper when args.use_fp64 is set.
  - Only the E8P12 codebook is supported (the codebook actually used for
    Llama-2 quantization in this repo -- see quantize_llama/quantize_finetune_llama.py
    and lib/codebook/__init__.py:codebook_id). Passing any other codebook
    raises.
"""

import torch

# NOT `import quiptools_quant` -- the compiled extension's importable name is
# `quiptools_quant_cuda`, deliberately different from the quiptools_quant/
# source directory it's built from (see the note at the top of
# quiptools_quant/setup.py: same name for both can resolve to an empty
# namespace package instead of the compiled .so once the repo root is on
# sys.path).
import quiptools_quant_cuda

_E8P12_CLASS_NAME = 'E8P12_codebook'

# Default for the `subtree_j` parameter of ldlq_fast(). j=3 is the size
# gpu_kernel_spec.md section 5 calls out as "the design worth aiming for"
# (~4,096 launches per tile-column instead of ~800K, at d=4096). Pass
# subtree_j=None to disable Kernel 3 entirely and fall back to pure
# Python recursion + Kernel 1 + Kernel 2 (useful for A/B measurement, or if
# j=3's ~88KB shared-memory footprint doesn't fit on the target GPU -- see
# the comment on subtree_kernel<J> in quiptools_quant.cu).
DEFAULT_SUBTREE_J = 3


def _check_codebook(cb):
    if type(cb).__name__ != _E8P12_CLASS_NAME:
        raise NotImplementedError(
            f'ldlq_fast only supports the E8P12 codebook, got {type(cb).__name__}. '
            'The device quantize kernel is a direct port of E8P12_codebook.quantize '
            'and does not generalize to other codebooks (e.g. the RVQ variants).')
    if cb.codesz != 8:
        raise AssertionError(f'expected cb.codesz == 8, got {cb.codesz}')
    if cb.idx_dtype != torch.int64:
        # This is the thing gpu_kernel_spec.md section 9 explicitly warns to
        # double check rather than assume -- and indeed E8P12_codebook sets
        # idx_dtype = torch.int64, not int16. See the report for detail.
        raise AssertionError(
            f'expected cb.idx_dtype == torch.int64 (the value in the current '
            f'E8P12_codebook), got {cb.idx_dtype}. The Qidxs tensor dtype and '
            f'the CUDA kernel bindings assume int64; if this ever changes, the '
            f'extension needs to change with it.')


class _CodebookTables:
    """Device copies of the E8P12 search tables the CUDA kernels need.

    These come straight from the live `cb` (E8P12_codebook) object's own
    buffers -- grid_part / grid_part_norm / part_abs_map / grid_abs_odd --
    rather than being rederived. That was verified (on CPU, since this repo's
    kernels can only be exercised on a GPU) to be sufficient: reimplementing
    E8P12_codebook.quantize using only these four buffers reproduces
    cb.quantize()'s output and indices exactly on 4096 random probe vectors.
    """

    def __init__(self, cb, device):
        self.grid_part = cb.grid_part.to(device=device, dtype=torch.float32).contiguous()
        self.grid_part_norm = cb.grid_part_norm.to(device=device, dtype=torch.float32).contiguous()
        self.part_abs_map = cb.part_abs_map.to(device=device, dtype=torch.int64).contiguous()
        self.grid_abs_odd = cb.grid_abs_odd.to(device=device, dtype=torch.uint8).contiguous()


class _FlatDEC:
    """DEC, pre-flattened into the device-resident, per-depth tensors the
    kernels want (gpu_kernel_spec.md section 1: "do not walk a Python list
    inside the hot path" -- so this flattening happens once per DEC, not
    once per recursive call).

    DEC (as produced by lib.algo.quip.decompose_H) is a list whose first
    len(DEC)-1 entries are each a 4-tuple (H0, H1, H2, H3) -- H0 unused, see
    lib/algo/quip.py:ldlq_helper -- and whose last entry is the single 64x64
    lower-triangular leaf matrix J. Every recursive call at a given depth
    reuses the identical (H0,H1,H2,H3) for that depth (and every leaf call in
    the whole tree reuses the identical J), so precomputing everything below
    once per depth, rather than once per node, is exact -- not an
    approximation.
    """

    def __init__(self, DEC):
        *levels, J = DEC
        self.H1_list = [H1.contiguous() for (_H0, H1, _H2, _H3) in levels]
        self.H2_list = [H2.contiguous() for (_H0, _H1, H2, _H3) in levels]
        self.H3_list = [H3.contiguous() for (_H0, _H1, _H2, H3) in levels]
        # Hcat3 = [H1|H2|H3] (for the 3-way batched H_multiply on error3),
        # Hcat2 = [H2|H3] (for the 2-way batched H_multiply on error2) -- see
        # quiptools_quant_cuda.h_multiply_batched and workload_analysis.md
        # section 3's redundancy-reduction note.
        self.Hcat3_list = [torch.cat([H1, H2, H3], dim=-1).contiguous() for (_H0, H1, H2, H3) in levels]
        self.Hcat2_list = [torch.cat([H2, H3], dim=-1).contiguous() for (_H0, _H1, H2, H3) in levels]
        self.J = J.contiguous()
        self.depth = len(levels)  # number of internal-node levels above the leaf
        self._subtree_cache = {}

    def to(self, device, dtype=torch.float32):
        self.H1_list = [t.to(device=device, dtype=dtype) for t in self.H1_list]
        self.H2_list = [t.to(device=device, dtype=dtype) for t in self.H2_list]
        self.H3_list = [t.to(device=device, dtype=dtype) for t in self.H3_list]
        self.Hcat3_list = [t.to(device=device, dtype=dtype) for t in self.Hcat3_list]
        self.Hcat2_list = [t.to(device=device, dtype=dtype) for t in self.Hcat2_list]
        self.J = self.J.to(device=device, dtype=dtype)
        self._subtree_cache = {}
        return self

    def subtree_flat(self, j):
        """H1_flat/H2_flat/H3_flat for quiptools_quant_cuda.subtree_quantize(j=...):
        the last `j` levels of H1_list/H2_list/H3_list (the ones nearest the
        leaf), flattened and concatenated outermost-level-first. Must match
        subtree_level_off's layout in quiptools_quant.cu -- see the comment
        on launch_subtree_kernel in quiptools_quant.h.
        """
        cached = self._subtree_cache.get(j)
        if cached is None:
            H1_flat = torch.cat([h.reshape(-1) for h in self.H1_list[-j:]]).contiguous()
            H2_flat = torch.cat([h.reshape(-1) for h in self.H2_list[-j:]]).contiguous()
            H3_flat = torch.cat([h.reshape(-1) for h in self.H3_list[-j:]]).contiguous()
            cached = (H1_flat, H2_flat, H3_flat)
            self._subtree_cache[j] = cached
        return cached


# Caches keyed by id() of the caller's cb / DEC objects. This is safe within
# bulk_LDLQ's normal usage pattern -- the same DEC list object is reused
# across every greedy-search pass for a given tile-column, and the same cb
# object is reused for the whole quantize() call -- but id() can in principle
# be reused after garbage collection if you build and discard many DEC lists
# with the same lifetime pattern. Call clear_cache() if you do something more
# exotic (e.g. constructing fresh DEC objects per call in a tight loop) and
# want to be safe rather than rely on this.
_cb_table_cache = {}
_dec_cache = {}


def clear_cache():
    _cb_table_cache.clear()
    _dec_cache.clear()


def _get_cb_tables(cb, device):
    key = (id(cb), str(device))
    tables = _cb_table_cache.get(key)
    if tables is None:
        _check_codebook(cb)
        tables = _CodebookTables(cb, device)
        _cb_table_cache[key] = tables
    return tables


def _get_flat_dec(DEC, device):
    key = (id(DEC), str(device))
    flat = _dec_cache.get(key)
    if flat is None:
        flat = _FlatDEC(DEC).to(device)
        _dec_cache[key] = flat
    return flat


def _recurse(X, future_error, flat, tables, Qidxs, depth, subtree_j):
    """Mirrors lib.algo.quip.ldlq_helper's structure exactly (same variable
    names, same split/recursion order), with cb.quantize -> leaf_quantize and
    H_multiply -> h_multiply_batched. `depth` counts internal-node levels
    from the root (0-indexed); flat.depth - depth is how many levels remain
    above the leaf at this node, which is what decides whether Kernel 3 can
    take over here.
    """
    m, n = X.shape
    if n == 64:
        return quiptools_quant_cuda.leaf_quantize(
            X, future_error, flat.J, tables.grid_part, tables.grid_part_norm,
            tables.part_abs_map, tables.grid_abs_odd, Qidxs)

    remaining = flat.depth - depth
    if subtree_j is not None and remaining == subtree_j:
        # Kernel 3: fuse the rest of this subtree (subtree_j levels + the
        # leaves under them) into a single launch instead of recursing
        # further in Python.
        H1_flat, H2_flat, H3_flat = flat.subtree_flat(subtree_j)
        return quiptools_quant_cuda.subtree_quantize(
            X, future_error, flat.J, H1_flat, H2_flat, H3_flat,
            tables.grid_part, tables.grid_part_norm, tables.part_abs_map,
            tables.grid_abs_odd, Qidxs, subtree_j)

    Hcat3 = flat.Hcat3_list[depth]
    Hcat2 = flat.Hcat2_list[depth]
    H1 = flat.H1_list[depth]

    X = X.reshape(m, 4, n // 4)
    future_error = future_error.reshape(m, 4, n // 4)
    X = [X[:, i].contiguous() for i in range(4)]
    future_error = [future_error[:, i].contiguous() for i in range(4)]
    QI = [Qidxs[:, i * (n // 32):(i + 1) * (n // 32)] for i in range(4)]

    res3 = _recurse(X[3], future_error[3], flat, tables, QI[3], depth + 1, subtree_j)
    error3 = X[3] - res3
    # [H_multiply(error3,H1), H_multiply(error3,H2), H_multiply(error3,H3)]
    A = quiptools_quant_cuda.h_multiply_batched(error3, Hcat3, 3)
    future_error2 = future_error[2] + A[0]
    future_error1 = future_error[1] + A[1]
    future_error0 = future_error[0] + A[2]

    res2 = _recurse(X[2], future_error2, flat, tables, QI[2], depth + 1, subtree_j)
    error2 = X[2] - res2
    # [H_multiply(error2,H2), H_multiply(error2,H3)]
    B = quiptools_quant_cuda.h_multiply_batched(error2, Hcat2, 2)
    future_error0 = future_error0 - B[0]
    future_error1 = future_error1 - B[1]

    res1 = _recurse(X[1], future_error1, flat, tables, QI[1], depth + 1, subtree_j)
    error1 = X[1] - res1
    C = quiptools_quant_cuda.h_multiply_batched(error1, H1, 1)[0]  # H_multiply(error1,H1)
    future_error0 = future_error0 + C

    res0 = _recurse(X[0], future_error0, flat, tables, QI[0], depth + 1, subtree_j)

    return torch.stack([res0, res1, res2, res3], dim=1).reshape(m, n)


def ldlq_fast(X, future_error, DEC, cb, Qidxs, subtree_j=DEFAULT_SUBTREE_J):
    """Drop-in replacement for lib.algo.quip.ldlq_helper(X, future_error, DEC, cb, Qidxs).

    Same contract: returns hatX with the same shape/dtype as X, and writes
    Qidxs in place.

    subtree_j: which fused-subtree size (Kernel 3) to use for the bottom of
    the tree, one of {1, 2, 3}, or None to disable Kernel 3 and use pure
    Python recursion + Kernel 1 (leaf) + Kernel 2 (H_multiply) only. Trees
    shallower than subtree_j never trigger it (falls through to plain
    recursion down to the leaf) -- this is not an error, just no Kernel-3
    benefit for that call.
    """
    assert X.is_cuda, 'ldlq_fast requires CUDA tensors'
    assert X.dtype == torch.float32 and future_error.dtype == torch.float32, (
        'ldlq_fast is float32-only (see workload_analysis.md section 5); '
        'fall back to ldlq_helper for args.use_fp64 runs')
    assert subtree_j is None or subtree_j in (1, 2, 3), (
        f'subtree_j must be None or one of 1, 2, 3 (Kernel 3 is only instantiated for '
        f'those depths -- see the note on subtree_kernel<J> in quiptools_quant.cu), got {subtree_j}')

    device = X.device
    flat = _get_flat_dec(DEC, device)
    tables = _get_cb_tables(cb, device)

    return _recurse(X, future_error, flat, tables, Qidxs, 0, subtree_j)
