"""Limiter cost regressions: bit-equality with HEAD + kernel-count budget.

Mission m3262.  Two properties are pinned here, both machine-independent:

1. **Bit-equality.**  ``tensordg/limiters.py`` was rewritten to issue fewer
   kernels (re-using the detector's minmod slope, a cached boundary mask and a
   cached replicate-padding index).  Every value it returns must be identical
   -- ``torch.equal``, not ``allclose`` -- to the reference formulation of HEAD
   (``tests/_limiter_reference.py``), which is the pre-mission code verbatim.

2. **Kernel budget.**  The limiter is element-wise algebra on a few thousand
   doubles, so its runtime is ATen dispatch overhead: the number of kernels per
   call is the cost.  Budgets below are the measured counts plus a small margin;
   they fail if someone reintroduces a per-cell loop, a second detector pass or
   a ``clone``+``where`` write path (all of which cost 1.5x-6x more kernels).
   See ``docs/limiter_performance_verification.md``.
"""
import torch

from tensordg.limiters import (
    _troubled_cell_indicator,
    apply_tvb_limiter,
    apply_weno_limiter,
)

from _limiter_reference import (
    ref_troubled_cell_indicator,
    ref_tvb_limiter,
    ref_weno_limiter,
)


class _KernelCounter(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


def count_kernels(fn):
    counter = _KernelCounter()
    with counter:
        fn()
    return counter.count


def _states():
    """A battery of coefficient tensors: random, physical, smooth, degenerate."""
    torch.manual_seed(0)
    yield "random", torch.randn(40, 3, 3, dtype=torch.float64)
    phys = torch.randn(40, 3, 3, dtype=torch.float64)
    phys[:, 0].abs_().add_(1.0)
    yield "physical", phys
    yield "random_float32", torch.randn(30, 3, 3, dtype=torch.float32)
    yield "degree0", torch.randn(20, 1, 3, dtype=torch.float64)
    yield "degree1", torch.randn(20, 2, 3, dtype=torch.float64)
    x = torch.linspace(0.0, 1.0, 25, dtype=torch.float64)
    smooth = torch.zeros(25, 3, 3, dtype=torch.float64)
    smooth[:, 0] = x[:, None]
    smooth[:, 1] = 0.1
    yield "smooth_linear", smooth
    yield "two_cells", torch.randn(2, 3, 3, dtype=torch.float64)
    yield "one_cell", torch.randn(1, 3, 3, dtype=torch.float64)
    yield "four_vars", torch.randn(30, 3, 4, dtype=torch.float64)


def test_limiters_are_bit_identical_to_head_reference():
    for name, coeff in _states():
        dx = 1.0 / coeff.shape[0]
        for limit_boundaries in (False, True):
            for tvb_constant in (0.0, 1.0, 1e6):
                args = dict(tvb_constant=tvb_constant, limit_boundaries=limit_boundaries)
                got = apply_tvb_limiter(coeff, dx, **args)
                want = ref_tvb_limiter(coeff, dx, **args)
                assert torch.equal(got, want), (name, "tvb", args)
                got = apply_weno_limiter(coeff, dx, **args)
                want = ref_weno_limiter(coeff, dx, **args)
                assert torch.equal(got, want), (name, "weno", args)
                got = _troubled_cell_indicator(coeff, dx, tvb_constant, limit_boundaries)
                want = ref_troubled_cell_indicator(
                    coeff, dx, tvb_constant, limit_boundaries
                )
                assert torch.equal(got, want), (name, "indicator", args)


def test_limiter_is_idempotent_and_preserves_cell_average():
    """Limiting twice must equal limiting once; mode 0 must never move."""
    torch.manual_seed(3)
    coeff = torch.randn(60, 3, 3, dtype=torch.float64)
    coeff[:, 0].abs_().add_(1.0)
    dx = 1.0 / 60
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        once = limiter(coeff, dx, tvb_constant=1.0)
        twice = limiter(once, dx, tvb_constant=1.0)
        assert torch.equal(once[:, 0], coeff[:, 0])
        assert torch.equal(twice, once)


def test_tvb_kernel_budget():
    coeff = torch.randn(200, 3, 3, dtype=torch.float64)
    coeff[:, 0].abs_().add_(1.0)
    dx = 1.0 / 200
    apply_tvb_limiter(coeff, dx, 1.0)  # warm the mask/index caches
    flagged = count_kernels(lambda: apply_tvb_limiter(coeff, dx, 1.0))
    reference = count_kernels(lambda: ref_tvb_limiter(coeff, dx, 1.0))
    assert flagged <= 48, flagged                      # measured 46 (HEAD: 61)
    assert reference > flagged

    smooth = torch.zeros_like(coeff)
    smooth[:, 0] = torch.linspace(0.0, 1.0, 200, dtype=torch.float64)[:, None]
    smooth[:, 1] = 0.25
    apply_tvb_limiter(smooth, dx, 1.0)  # warm
    early_exit = count_kernels(lambda: apply_tvb_limiter(smooth, dx, 1.0))
    assert early_exit <= 30, early_exit                # measured 27 (HEAD: 36)
    assert early_exit < flagged


def test_weno_kernel_budget():
    coeff = torch.randn(200, 3, 3, dtype=torch.float64)
    coeff[:, 0].abs_().add_(1.0)
    dx = 1.0 / 200
    apply_weno_limiter(coeff, dx, 1.0)  # warm the mask/index caches
    flagged = count_kernels(lambda: apply_weno_limiter(coeff, dx, 1.0))
    reference = count_kernels(lambda: ref_weno_limiter(coeff, dx, 1.0))
    assert flagged <= 92, flagged                      # measured 87 (HEAD: 98)
    assert reference > flagged


def test_kernel_cost_is_independent_of_cell_count():
    """The limiter must stay vectorised: doubling the mesh must not double the
    kernel count (that is the signature of a per-cell loop)."""
    counts = {}
    for ncells in (100, 200, 400):
        coeff = torch.randn(ncells, 3, 3, dtype=torch.float64)
        coeff[:, 0].abs_().add_(1.0)
        dx = 1.0 / ncells
        apply_tvb_limiter(coeff, dx, 1.0)  # warm
        counts[ncells] = count_kernels(lambda: apply_tvb_limiter(coeff, dx, 1.0))
    assert max(counts.values()) - min(counts.values()) <= 2, counts
