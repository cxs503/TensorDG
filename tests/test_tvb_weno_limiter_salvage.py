"""Focused pytest for the ``tvb-weno-limiters`` capability (soul-revival pilot).

Three requested axes, each pinned by an *analytically* verifiable property
(no hand-carried reference numbers):

1. **Limiter basic properties** -- two chosen for analytic verifiability:
   * *boundedness* (TVD/minmod): the limited mode-1 coefficient of a troubled
     cell must satisfy ``|c_1| <= 0.5 * max(|d_left|, |d_right|)``, i.e. the
     reconstruction slope never exceeds the larger one-sided cell-average
     increment (the modal ``dx/2`` convention of ``tests/test_limiter_units.py``
     makes this ``|c_1| <= max(|d_left|,|d_right|)/2``);
   * *odd symmetry* (equivariance under ``u -> -u``): both limiters are odd
     maps, ``L(-coeff) == -L(coeff)``, because the minmod slope and every WENO
     candidate slope flip sign while the smoothness indicators (``beta ~ s^2``)
     and the troubled-cell deviation (``|d - mm|``) are sign-invariant.
   * (bonus) *extremum release*: for ``M`` above the data's second-derivative
     bound the TVB indicator is inert on a smooth extremum -- the limiter
     leaves the cell untouched (the ``M``-tunability the scale-consistent
     criterion exists for).

2. **Degenerate inputs** -- P0 no-op, constant/zero data (no troubled cell),
   single-cell mesh, ``dx = 0``, negative ``tvb_constant``, and NaN propagation
   all behave without raising and without corrupting the cell average.

3. **Cross-implementation consistency** -- the ``apply_limiter`` dispatcher
   must be bit-identical to the direct ``apply_tvb_limiter`` /
   ``apply_weno_limiter`` calls on the same input, and the 2-D TVB limiter must
   reduce to the verified 1-D one on y-invariant data (the WENO half of that
   reduction is already pinned in ``tests/test_limiter_2d.py``; the TVB half
   is added here).
"""
from __future__ import annotations

import torch

from tensordg.euler_dg_2d import apply_tvb_limiter_2d
from tensordg.limiters import (
    apply_limiter,
    apply_tvb_limiter,
    apply_weno_limiter,
    _troubled_cell_indicator,
)


def _step_coefficients(ncells: int = 6, dx: float = 0.1):
    """Cell averages with a mild non-monotone slope so the detector fires."""
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor(
        [1.0, 1.0, 1.0, 2.0, 2.6, 2.6], dtype=torch.float64
    )[:ncells]
    return coeff, dx


# ---------------------------------------------------------------------------
# 1. Limiter basic properties (analytically verifiable)
# ---------------------------------------------------------------------------

def test_tvb_limiter_mode1_is_bounded_by_the_larger_one_sided_increment():
    """TVD/minmod boundedness: |c_1| <= max(|d_left|, |d_right|) / 2.

    The modal ``dx/2`` convention (``tests/test_limiter_units.py``) stores a
    physical slope ``s`` as ``c_1 = s * dx / 2``, and the minmod slope is
    ``minmod(d_left, d_right)`` with ``|minmod| <= max(|d_left|, |d_right|)``.
    So in every troubled cell ``|c_1| <= max(|d_left|, |d_right|) / 2`` -- an
    exact bound that holds for *any* data, not just the constructed step.
    """
    torch.manual_seed(7)
    ncells, dx = 25, 0.04
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    avg = torch.cumsum(torch.rand(ncells, dtype=torch.float64), dim=0)
    avg[10:] += 2.0 * torch.rand(ncells - 10, dtype=torch.float64)
    coeff[:, 0, 0] = avg

    padded = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = (avg - padded[:-2])[:, None]
    d_right = (padded[2:] - avg)[:, None]
    bound = 0.5 * torch.maximum(d_left.abs(), d_right.abs())

    limited = apply_tvb_limiter(coeff, dx, tvb_constant=0.0)
    # Every cell (troubled or not) must respect the minmod bound.
    assert bool((limited[:, 1].abs() <= bound + 1e-12).all()), (
        "TVB limiter must never amplify the slope beyond the larger one-sided "
        "increment"
    )


def test_tvb_limiter_is_odd_under_sign_flip():
    """Equivariance under u -> -u: L(-coeff) == -L(coeff).

    The minmod slope is odd in its arguments and the troubled-cell deviation
    ``|d - minmod|`` is sign-invariant, so flipping the field flips the limited
    slope but not the troubled mask -- the limiter is an odd map.
    """
    torch.manual_seed(42)
    coeff = torch.randn(20, 3, 3, dtype=torch.float64)
    dx = 0.05
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=1.0)
    limited_neg = apply_tvb_limiter(-coeff, dx, tvb_constant=1.0)
    assert torch.allclose(limited_neg, -limited, atol=1e-12), (
        "TVB limiter must be odd: L(-u) == -L(u)"
    )


def test_weno_limiter_is_odd_under_sign_flip():
    """WENO limiter is also an odd map (weights are sign-invariant)."""
    torch.manual_seed(42)
    coeff = torch.randn(20, 3, 3, dtype=torch.float64)
    dx = 0.05
    limited = apply_weno_limiter(coeff, dx, tvb_constant=1.0)
    limited_neg = apply_weno_limiter(-coeff, dx, tvb_constant=1.0)
    assert torch.allclose(limited_neg, -limited, atol=1e-12), (
        "WENO limiter must be odd: L(-u) == -L(u)"
    )


def test_tvb_limiter_releases_a_smooth_extremum_for_large_M():
    """M above the second-derivative bound => the indicator is inert.

    For a smooth parabolic extremum the one-sided differences deviate from the
    minmod slope by ``O(dx**2 * u'')``; once ``M`` exceeds that bound the
    troubled-cell test ``deviation > M * dx**2`` never fires and the limiter
    leaves the cell untouched -- the ``M``-tunability the scale-consistent
    criterion exists for.
    """
    ncells, dx = 20, 1.0 / 20
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    # Parabolic profile with a smooth maximum at cell 10: u'' = -2 (constant).
    for i in range(ncells):
        coeff[i, 0, 0] = 1.0 - ((i - 10) * dx) ** 2
    # A large M (>> |u''| = 2) must release the extremum.
    limited = apply_tvb_limiter(coeff, dx, tvb_constant=10.0)
    assert torch.equal(limited, coeff), (
        "a smooth extremum with M >> |u''| must be left unlimited"
    )
    # And the strict M=0 must flag it (the same cell, for contrast).
    troubled = _troubled_cell_indicator(coeff, dx, tvb_constant=0.0)
    assert troubled[10].item() is True


# ---------------------------------------------------------------------------
# 2. Degenerate inputs
# ---------------------------------------------------------------------------

def test_limiter_p0_is_a_noop():
    """P0 (nmodes=1) has nothing to limit: every limiter returns the input."""
    coeff = torch.randn(10, 1, 3, dtype=torch.float64)
    dx = 0.1
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        assert torch.equal(limiter(coeff, dx), coeff)


def test_limiter_on_constant_data_is_a_noop():
    """Constant cell averages have zero deviation: no cell is troubled."""
    ncells, dx = 12, 0.1
    coeff = torch.zeros(ncells, 3, 2, dtype=torch.float64)
    coeff[:, 0] = 1.0  # constant in every variable
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        assert torch.equal(limiter(coeff, dx, tvb_constant=0.0), coeff)


def test_limiter_on_single_cell_mesh():
    """A one-cell mesh has no stencil: the limiter must not raise."""
    coeff = torch.randn(1, 3, 2, dtype=torch.float64)
    dx = 0.1
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        limited = limiter(coeff, dx, tvb_constant=0.0)
        assert limited.shape == coeff.shape
        assert torch.equal(limited[:, 0], coeff[:, 0])  # cell average preserved


def test_limiter_with_zero_dx():
    """dx = 0 makes the threshold 0; the limiter must still run and conserve."""
    ncells, dx = 8, 0.0
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.6, 2.6, 2.6, 2.6])
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        limited = limiter(coeff, dx, tvb_constant=0.0)
        assert torch.equal(limited[:, 0], coeff[:, 0])  # conservation holds


def test_limiter_with_negative_tvb_constant():
    """A negative M is a degenerate (always-flag) policy; must not raise."""
    ncells, dx = 8, 0.1
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.6, 2.6, 2.6, 2.6])
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        limited = limiter(coeff, dx, tvb_constant=-1.0)
        assert torch.equal(limited[:, 0], coeff[:, 0])  # still conservative


def test_limiter_propagates_nan_without_corrupting_cell_average():
    """NaN in a cell average must not raise; mode-0 is still preserved.

    ``torch.equal`` treats NaN as unequal to itself, so the preservation
    check is done on the finite cells and on the NaN cell's mode-0 being
    *still* NaN (i.e. the limiter never rewrites a cell average).
    """
    ncells, dx = 8, 0.1
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.6, 2.6, 2.6, 2.6])
    coeff[4, 0, 0] = float("nan")
    for limiter in (apply_tvb_limiter, apply_weno_limiter):
        limited = limiter(coeff, dx, tvb_constant=0.0)
        # Finite cells: mode-0 preserved exactly.
        assert torch.equal(limited[:, 0][~torch.isnan(coeff[:, 0])],
                           coeff[:, 0][~torch.isnan(coeff[:, 0])])
        # The NaN cell's mode-0 is still NaN (never rewritten).
        assert torch.isnan(limited[4, 0, 0]).item() is True


# ---------------------------------------------------------------------------
# 3. Cross-implementation consistency (multiple implementations, same input)
# ---------------------------------------------------------------------------

def test_dispatcher_is_bit_identical_to_direct_tvb_call():
    """``apply_limiter(kind='tvb')`` must equal ``apply_tvb_limiter`` exactly."""
    torch.manual_seed(42)
    coeff = torch.randn(20, 3, 3, dtype=torch.float64)
    dx = 0.05
    direct = apply_tvb_limiter(coeff, dx, tvb_constant=1.0)
    dispatched = apply_limiter(coeff, dx, kind="tvb", tvb_constant=1.0)
    assert torch.equal(dispatched, direct), (
        "the dispatcher must not re-route or re-scale the TVB limiter"
    )


def test_dispatcher_is_bit_identical_to_direct_weno_call():
    """``apply_limiter(kind='weno')`` must equal ``apply_weno_limiter`` exactly."""
    torch.manual_seed(42)
    coeff = torch.randn(20, 3, 3, dtype=torch.float64)
    dx = 0.05
    direct = apply_weno_limiter(coeff, dx, tvb_constant=1.0)
    dispatched = apply_limiter(coeff, dx, kind="weno", tvb_constant=1.0)
    assert torch.equal(dispatched, direct), (
        "the dispatcher must not re-route or re-scale the WENO limiter"
    )


def test_tvb_and_weno_agree_on_conservation_and_boundedness_for_same_input():
    """Both implementations on the same input: same cell average, same bound.

    The two limiters differ in *how* they reconstruct the slope (minmod vs
    WENO convex combination), but they must agree on the two invariants that
    define a limiter: the cell average is preserved exactly and the
    reconstruction slope never exceeds the larger one-sided increment.
    """
    coeff, dx = _step_coefficients()
    tvb = apply_tvb_limiter(coeff, dx, tvb_constant=1e-9)
    weno = apply_weno_limiter(coeff, dx, tvb_constant=1e-9)
    # Both preserve the cell average.
    assert torch.equal(tvb[:, 0], coeff[:, 0])
    assert torch.equal(weno[:, 0], coeff[:, 0])
    # Both respect the minmod bound on the flagged cell (index 3).
    padded = torch.cat([coeff[:1, 0], coeff[:, 0], coeff[-1:, 0]], dim=0)
    bound = 0.5 * max(
        abs(float(padded[4, 0] - padded[2, 0])),
        abs(float(padded[3, 0] - padded[4, 0])),
    )
    assert tvb[3, 1, 0].abs().item() <= bound + 1e-12
    assert weno[3, 1, 0].abs().item() <= bound + 1e-12


def test_tvb_2d_reduces_to_the_verified_1d_limiter_on_y_invariant_data():
    """The 2-D TVB limiter is the 1-D one on y-invariant data (bit-for-bit).

    The WENO half of this reduction is already pinned in
    ``tests/test_limiter_2d.py``; the TVB half is added here so both
    implementations of the same capability agree on a shared input.
    """
    ncells, degree = 16, 2
    dx = 1.0 / ncells
    x = (torch.arange(ncells, dtype=torch.float64) + 0.5) * dx
    averages = torch.zeros(ncells, 1, 4, dtype=torch.float64)
    averages[..., 0] = torch.where(
        x < 0.5, torch.full_like(x, 1.0), torch.full_like(x, 0.125)
    )[:, None]
    averages[..., 3] = torch.where(
        x < 0.5, torch.full_like(x, 1.0), torch.full_like(x, 0.1)
    )[:, None]
    coeff = torch.zeros(ncells, 1, degree + 1, degree + 1, 4, dtype=torch.float64)
    coeff[..., 0, 0, :] = averages
    coeff[..., 1, 0, :] = 0.4 * averages
    coeff[..., 2, 0, :] = -0.2 * averages

    limited_2d = apply_tvb_limiter_2d(coeff, dx, 0.25, 1.0, (False, True))
    reference = apply_tvb_limiter(
        coeff[:, 0, :, 0, :].clone(), dx, 1.0, limit_boundaries=True
    )
    assert float((limited_2d[:, 0, :, 0, :] - reference).abs().max()) == 0.0, (
        "the 2-D TVB limiter must reduce to the verified 1-D one on "
        "y-invariant data"
    )
