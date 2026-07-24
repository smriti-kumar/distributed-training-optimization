"""Acceptance tests for quiptools_quant / lib.algo.ldlq_fast.

These implement the acceptance tests from workload_analysis.md section 6.
They require a CUDA GPU and the compiled `quiptools_quant_cuda` extension
(built from this quiptools_quant/ directory -- note the module's import name
is deliberately different from the directory name, see the note at the top
of quiptools_quant/setup.py), so they are written to be run on the cluster --
they were NOT executed while writing this kernel (this development
environment has no GPU). Run with:

    cd quiptools_quant && pip install -e . && cd ..
    pytest quiptools_quant/tests/test_ldlq_fast.py -v

Test 3 (full tile regression) additionally needs a real Llama-2-7B Hessian +
weight checkpoint for layer 1 / sublayer `o`, in the format
quantize_llama/hessian_offline_llama.py produces. Point QUIP_TEST_HESSIAN_PATH
/ QUIP_TEST_WEIGHT_PATH at one, or it is skipped.
"""

import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.algo import quip
from lib.algo import ldlq_fast as ldlq_fast_mod
from lib.codebook import get_codebook

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')

DEVICE = 'cuda'


@pytest.fixture(scope='module')
def cb():
    codebook = get_codebook('E8P12').to(torch.float32).to(DEVICE)
    return codebook


def _random_lower_triangular(n, device, generator):
    A = torch.randn(n, n, device=device, generator=generator, dtype=torch.float32)
    A = A @ A.T + n * torch.eye(n, device=device)  # SPD
    L = torch.linalg.cholesky(A)
    return L / L.diagonal().mean()


def test_codebook_port_isolation(cb):
    """Acceptance test 4: 10^6 random 8-vectors, require 100% identical indices."""
    import quiptools_quant_cuda

    g = torch.Generator(device=DEVICE).manual_seed(0)
    X = (torch.randn(1_000_000, 8, device=DEVICE, generator=g) * 0.6).contiguous()

    ref_vals, ref_idx = cb.quantize(X)

    dev_vals, dev_idx = quiptools_quant_cuda.e8p12_quantize_batch(
        X, cb.grid_part.float(), cb.grid_part_norm.float(),
        cb.part_abs_map.long(), cb.grid_abs_odd.to(torch.uint8))

    mismatches = (ref_idx != dev_idx).sum().item()
    assert mismatches == 0, f'{mismatches} / {X.shape[0]} index mismatches'
    torch.testing.assert_close(ref_vals, dev_vals, atol=0, rtol=0)


def test_bit_exact_leaf(cb):
    """Acceptance test 1: N == 64, random X / future_error / lower-triangular J."""
    g = torch.Generator(device=DEVICE).manual_seed(1)
    p = 5
    N = 64
    X = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32)
    future_error = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32) * 0.1
    J = _random_lower_triangular(N, DEVICE, g)
    J = torch.tril(J)
    DEC = [J]

    Qidxs_ref = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_ref = quip.ldlq_helper(X.clone(), future_error.clone(), DEC, cb, Qidxs_ref)

    Qidxs_fast = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_fast = ldlq_fast_mod.ldlq_fast(X.clone(), future_error.clone(), DEC, cb, Qidxs_fast)

    assert torch.equal(Qidxs_ref, Qidxs_fast), 'Qidxs mismatch'
    torch.testing.assert_close(hatX_ref, hatX_fast, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('d', [16, 64])
def test_bit_exact_small_tree(cb, d):
    """Acceptance test 2: N = 4^4 = 256 (d=16) and N = 4^6 = 4096 (d=64).

    d=16 -> DEC has 1 internal level (below the default subtree_j=3), so
    this exercises pure Python recursion + Kernel 1 + Kernel 2 only.
    d=64 -> DEC has exactly 3 internal levels, so with the default
    subtree_j=3 the *entire* call is handled by one Kernel 3 launch, no
    Python-level recursion at all. See test_subtree_kernel_j_variants and
    test_bit_exact_deep_tree below for the cases in between.
    """
    g = torch.Generator(device=DEVICE).manual_seed(2 + d)
    p = 3
    A = torch.randn(d, d, device=DEVICE, generator=g, dtype=torch.float32)
    A = (A @ A.T + d * torch.eye(d, device=DEVICE)).to(torch.float32)  # symmetric PSD
    DEC = quip.decompose_H(A)
    N = d * d

    X = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32)
    future_error = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32) * 0.1

    Qidxs_ref = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_ref = quip.ldlq_helper(X.clone(), future_error.clone(), DEC, cb, Qidxs_ref)

    Qidxs_fast = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_fast = ldlq_fast_mod.ldlq_fast(X.clone(), future_error.clone(), DEC, cb, Qidxs_fast)

    assert torch.equal(Qidxs_ref, Qidxs_fast), 'Qidxs mismatch'
    torch.testing.assert_close(hatX_ref, hatX_fast, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('subtree_j', [None, 1, 2, 3])
def test_subtree_kernel_j_variants(cb, subtree_j):
    """Acceptance test 2 variant: same d=64 tree (3 internal levels) as
    above, but forcing each subtree_j value explicitly. subtree_j=3 routes
    the whole call through one Kernel 3 launch; subtree_j=1 or 2 make the
    Python recursion run for 2 or 1 levels before handing off to Kernel 3;
    subtree_j=None disables Kernel 3 entirely (pure recursion + Kernel 1/2).
    All four must agree with the reference to the same tolerance.
    """
    d = 64
    g = torch.Generator(device=DEVICE).manual_seed(100 + (subtree_j or 0))
    p = 3
    A = torch.randn(d, d, device=DEVICE, generator=g, dtype=torch.float32)
    A = (A @ A.T + d * torch.eye(d, device=DEVICE)).to(torch.float32)
    DEC = quip.decompose_H(A)
    N = d * d

    X = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32)
    future_error = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32) * 0.1

    Qidxs_ref = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_ref = quip.ldlq_helper(X.clone(), future_error.clone(), DEC, cb, Qidxs_ref)

    Qidxs_fast = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_fast = ldlq_fast_mod.ldlq_fast(X.clone(), future_error.clone(), DEC, cb, Qidxs_fast,
                                          subtree_j=subtree_j)

    assert torch.equal(Qidxs_ref, Qidxs_fast), f'Qidxs mismatch at subtree_j={subtree_j}'
    torch.testing.assert_close(hatX_ref, hatX_fast, rtol=1e-5, atol=1e-5)


def test_bit_exact_deep_tree(cb):
    """d=128 -> N = 4^7 = 16384, DEC has 4 internal levels: one more than
    the default subtree_j=3. This is the case none of the tests above cover
    -- Kernel 3 triggers one level INTO the Python recursion (at depth=1,
    not at the root), so both the plain-Python-recursion path and the
    Kernel-3 handoff are exercised within a single ldlq_fast call.
    """
    d = 128
    g = torch.Generator(device=DEVICE).manual_seed(42)
    p = 2
    A = torch.randn(d, d, device=DEVICE, generator=g, dtype=torch.float32)
    A = (A @ A.T + d * torch.eye(d, device=DEVICE)).to(torch.float32)
    DEC = quip.decompose_H(A)
    assert len(DEC) - 1 == 4, f'expected 4 internal levels for d=128, got {len(DEC) - 1}'
    N = d * d

    X = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32)
    future_error = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32) * 0.1

    Qidxs_ref = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_ref = quip.ldlq_helper(X.clone(), future_error.clone(), DEC, cb, Qidxs_ref)

    Qidxs_fast = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_fast = ldlq_fast_mod.ldlq_fast(X.clone(), future_error.clone(), DEC, cb, Qidxs_fast)

    assert torch.equal(Qidxs_ref, Qidxs_fast), 'Qidxs mismatch'
    torch.testing.assert_close(hatX_ref, hatX_fast, rtol=1e-5, atol=1e-5)


def test_determinism(cb):
    """Acceptance test 5: two runs on identical input must match exactly."""
    g = torch.Generator(device=DEVICE).manual_seed(3)
    d = 16
    A = torch.randn(d, d, device=DEVICE, generator=g, dtype=torch.float32)
    A = (A @ A.T + d * torch.eye(d, device=DEVICE)).to(torch.float32)
    DEC = quip.decompose_H(A)
    N = d * d
    p = 4
    X = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32)
    future_error = torch.randn(p, N, device=DEVICE, generator=g, dtype=torch.float32) * 0.1

    Qidxs_a = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_a = ldlq_fast_mod.ldlq_fast(X.clone(), future_error.clone(), DEC, cb, Qidxs_a)
    Qidxs_b = torch.zeros(p, N // 8, dtype=cb.idx_dtype, device=DEVICE)
    hatX_b = ldlq_fast_mod.ldlq_fast(X.clone(), future_error.clone(), DEC, cb, Qidxs_b)

    assert torch.equal(Qidxs_a, Qidxs_b)
    assert torch.equal(hatX_a, hatX_b)


@pytest.mark.skipif(
    not (os.environ.get('QUIP_TEST_HESSIAN_PATH') and os.environ.get('QUIP_TEST_WEIGHT_PATH')),
    reason='set QUIP_TEST_HESSIAN_PATH / QUIP_TEST_WEIGHT_PATH to a real Llama-2-7B '
           'layer-1 `o` sublayer checkpoint to run the full-tile regression test')
def test_full_tile_regression(cb):
    """Acceptance test 3: bulk_LDLQ on Llama-2-7B layer 1 sublayer `o`, both
    paths. Proxy loss tr(E H E^T) should agree to 4 significant figures
    against the reference value from workload_analysis.md section 6 (~26.26)
    when run with this repo's default args -- adjust if your args differ.
    """
    hessian_path = os.environ['QUIP_TEST_HESSIAN_PATH']
    weight_path = os.environ['QUIP_TEST_WEIGHT_PATH']

    from lib import utils

    H_data = torch.load(hessian_path, map_location='cpu')
    H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
    mu = H_data['mu']
    H.add_(mu[None, :] * mu[:, None])
    H = utils.regularize_H(H, H_data['n'], 1e-2).to(DEVICE).to(torch.float32)

    W = torch.load(weight_path, map_location='cpu').to(DEVICE).to(torch.float32)
    W = W / W.square().mean().sqrt() * cb.opt_scale

    S, V = torch.linalg.eigh(H.double())
    H_sqrt = (V @ torch.diag(torch.sqrt(torch.clamp(S, min=0))) @ V.T).to(torch.float32)

    def run(use_fast):
        orig = quip.ldlq_helper
        if use_fast:
            quip.ldlq_helper = ldlq_fast_mod.ldlq_fast
        try:
            hatW, _ = quip.bulk_LDLQ(W.clone(), H_sqrt.clone(), cb, H.clone(), passes=0)
        finally:
            quip.ldlq_helper = orig
        E = hatW - W
        return torch.trace(E @ H.to(torch.float32) @ E.T).item()

    loss_ref = run(False)
    loss_fast = run(True)
    assert loss_ref == pytest.approx(loss_fast, rel=1e-4)
    assert loss_ref == pytest.approx(26.26, rel=5e-2)
