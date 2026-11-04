"""Shared TVB troubled-cell test used by the 1-D and 2-D limiters.

Extracted from ``tensordg/limiters.py`` in mission m2549 so that the
scale-consistent criterion introduced for the 1-D limiter in mission m2500 is
the single implementation used everywhere.  The function is pure tensor algebra
and works for arbitrary leading dimensions (``[ncells, nvar]`` in 1-D,
``[nx, ny, nvar]`` in 2-D).
"""
from __future__ import annotations

import torch


def minmod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise minmod: same-sign selection with smallest magnitude."""
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(a.abs(), b.abs())


def troubled_from_differences(
    d_left: torch.Tensor,
    d_right: torch.Tensor,
    spacing: float,
    tvb_constant: float,
    symmetric_indicator: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale-consistent TVB troubled-cell test and minmod slope.

    ``d_left = u_i - u_{i-1}`` and ``d_right = u_{i+1} - u_i`` are cell-average
    differences along one axis, so both are O(dx * u').  The quantity compared
    against the tolerance ``M dx**2`` is the *deviation of the one-sided
    difference from the minmod slope*,

        |d_left - minmod(d_left, d_right)|,

    which is O(dx * u') at a jump but O(dx**2 * u'') for smooth data (it equals
    ``|d_left - d_right| = |u''| dx**2`` in a smooth stretch and collapses to
    O(dx**2) at a smooth extremum).  Comparing that against ``M dx**2`` is
    dimensionally consistent, so ``M`` keeps its meaning under mesh refinement
    and can be tuned to release smooth extrema.

    The historic form compared ``max(|d_left|, |d_right|)`` -- an O(dx * u')
    measure -- against ``M dx**2``; every smooth extremum was then flagged as
    soon as ``dx > 1/M`` and ``M`` could not be tuned at all
    (docs/limiter_policy_verification.md for 1-D,
    docs/limiter_2d_verification.md for the 2-D consequence).

    Returns
    -------
    troubled : bool tensor with the leading (cell) dimensions of *d_left* and
        the variable axis reduced.
    slope : minmod slope with the same shape as *d_left*.

    Notes
    -----
    Boundary cells are the caller's responsibility: the 1-D limiter never
    flags them, the 2-D limiter replicates edge averages for non-periodic axes.
    """
    mm = minmod(d_left, d_right)
    deviation = (d_left - mm).abs()
    if symmetric_indicator:
        # The historic form measures the deviation of the *left* difference
        # from the minmod slope only.  Under the reflection x -> -x the roles
        # of ``d_left`` and ``d_right`` are exchanged, so
        # |d_left - mm| != |d_right - mm| in general and the flagged set is not
        # mirror-symmetric: on smooth rarefaction-like data 25 vs 19 of 64 cells
        # are flagged for the two orientations (44 mismatched positions), and
        # the resulting Sod / reversed-Sod L2 errors differ by ~2% at P2
        # (docs/riemann_coverage_verification.md).  Taking the larger of the
        # two deviations restores exact reflection symmetry at the price of a
        # somewhat more aggressive indicator for the same M.  It is opt-in so
        # that the validated M policy of docs/limiter_policy_verification.md
        # keeps its calibrated meaning.
        deviation = torch.maximum(deviation, (d_right - mm).abs())
    deviation = deviation.max(dim=-1).values
    threshold = tvb_constant * spacing * spacing
    # Round-off floor: for affine data the deviation is a few ulp of the local
    # differences, which must not be mistaken for a resolved feature.
    scale = torch.maximum(d_left.abs(), d_right.abs()).max(dim=-1).values
    troubled = deviation > threshold + 1e-12 * scale.clamp_min(1.0)
    return troubled, mm