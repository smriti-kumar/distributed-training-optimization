"""Drop-in CUDA replacement for lib.algo.quip.ldlq_helper.

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
    """Device copies of the E8P12 search tables the CUDA kernel needs.

    These come straight from the live `cb` (E8P12_codebook) object's own
    buffers -- grid_part / grid_part_norm / part_abs_map / grid_abs_odd --
    rather than being rederived. That was verified (on CPU, since this repo's
    kernel can only be exercised on a GPU) to be sufficient: reimplementing
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
    kernel wants (gpu_kernel_spec.md section 1: "do not walk a Python list
    inside the hot path").

    DEC (as produced by lib.algo.quip.decompose_H) is a list whose first
    len(DEC)-1 entries are each a 4-tuple (H0, H1, H2, H3) -- H0 unused, see
    lib/algo/quip.py:ldlq_helper -- and whose last entry is the single 64x64
    lower-triangular leaf matrix J. Every recursive call at a given depth
    reuses the identical (H0,H1,H2,H3) for that depth (and every leaf call in
    the whole tree reuses the identical J), so concatenating [H1|H2|H3] and
    [H2|H3] once per depth here, rather than once per node, is exact -- not
    an approximation.
    """

    def __init__(self, DEC):
        *levels, J = DEC
        self.Hcat3_list = [torch.cat([H1, H2, H3], dim=-1).contiguous() for (_H0, H1, H2, H3) in levels]
        self.Hcat2_list = [torch.cat([H2, H3], dim=-1).contiguous() for (_H0, _H1, H2, H3) in levels]
        self.H1_list = [H1.contiguous() for (_H0, H1, _H2, _H3) in levels]
        self.J = J.contiguous()

    def to(self, device, dtype=torch.float32):
        self.Hcat3_list = [t.to(device=device, dtype=dtype) for t in self.Hcat3_list]
        self.Hcat2_list = [t.to(device=device, dtype=dtype) for t in self.Hcat2_list]
        self.H1_list = [t.to(device=device, dtype=dtype) for t in self.H1_list]
        self.J = self.J.to(device=device, dtype=dtype)
        return self


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


def ldlq_fast(X, future_error, DEC, cb, Qidxs):
    """Drop-in replacement for lib.algo.quip.ldlq_helper(X, future_error, DEC, cb, Qidxs).

    Same contract: returns hatX with the same shape/dtype as X, and writes
    Qidxs in place.
    """
    assert X.is_cuda, 'ldlq_fast requires CUDA tensors'
    assert X.dtype == torch.float32 and future_error.dtype == torch.float32, (
        'ldlq_fast is float32-only (see workload_analysis.md section 5); '
        'fall back to ldlq_helper for args.use_fp64 runs')

    device = X.device
    flat = _get_flat_dec(DEC, device)
    tables = _get_cb_tables(cb, device)

    return quiptools_quant_cuda.ldlq_fast(
        X.contiguous(),
        future_error.contiguous(),
        flat.Hcat3_list,
        flat.Hcat2_list,
        flat.H1_list,
        flat.J,
        tables.grid_part,
        tables.grid_part_norm,
        tables.part_abs_map,
        tables.grid_abs_odd,
        Qidxs,
    )
