"""Cross-family (cross-solver) convergence-order protocol.

Why this module exists
----------------------
The in-house anchor ``GAP-DIFFLBM-XSOLVER-ORDER`` replaced a *compare-to-table*
acceptance rule (``max-point < 3%`` against the tabulated Ghia 1982 centreline
data) with a *cross-solver* rule: two unrelated discretisations of the same
problem must agree to ``< 3%`` on every grid of a ladder and the disagreement
must decay with an observed order ``p >= 1.8``.  The replacement is sound in
principle -- a table carries its own discretisation error and therefore an
error floor -- but it silently assumes that both families are *distinct* and
both inside their *asymptotic range*.  Neither assumption is checked anywhere,
and the published ladder (report #19236) shows the consequences: the coarse
grid fails the 3% gate while the observed order sits between the theoretical
orders of the two families.

This module turns those assumptions into machine-checkable rules so that a
cross-family verdict cannot be produced from a degenerate or non-asymptotic
pair without the verdict saying so:

R1 **shared grid**   -- both families are sampled on exactly the same nodes.
R2 **self-converged** -- each family is flagged converged by *its own*
   criterion before its samples are used.
R3 **distinctness**  -- the cross-family difference must sit above the
   round-off floor; a pair that collapses to round-off is *degenerate* and its
   observed order is ``not_run``, not ``infinite accuracy``.
R4 **asymptotic window** -- only levels whose difference is at least
   ``signal_levels`` x the floor enter the order fit.
R5 **order attribution** -- the observed cross-family order must equal the
   order of the *less accurate* family.  Larger values are pre-asymptotic
   buffering (the finer family has not yet taken over); smaller values mean one
   family is not converged or not on the shared grid.

The arithmetic that makes the anchor's gate predictable
-------------------------------------------------------
For two families ``A`` and ``B`` with own errors ``e_A = C_A h^{p_A}`` and
``e_B = C_B h^{p_B}`` the cross-family metric obeys::

    |A - B|_inf  <=  e_A + e_B                (triangle inequality, tight when
                                               the node-wise errors oppose)

so a gate ``|A - B| < g`` *requires* ``max(e_A, e_B) < g/2`` -- the
"half-gate rule" implemented by :func:`half_gate_budget`.  This is why a
2nd-order pair cannot pass a 3% gate on a grid where each family is itself only
3% accurate, no matter how the solvers are tuned.

Two regimes, and why the min-order wording is not enough
--------------------------------------------------------
The observed order of ``A - B`` is the order of the *dominant* contribution,
which is not necessarily the lower-order family:

* **dominant-family regime** -- one family's own error dominates the
  difference, so the observed order sits inside ``[min(p_A,p_B),
  max(p_A,p_B)]`` and equals ``min`` exactly when the less accurate family is
  also the larger-error one.  A verdict from this regime is meaningful.
* **alias regime** -- the two families share their leading truncation term
  (the same stencil in different clothing, or two closures of the same
  equation).  Then ``|A - B|`` collapses to a *fraction* of either own error
  (``cancellation_index`` << 1) and the observed order exceeds *both* family
  orders.  Such a pair agrees because it is wrong in the same way; a 3%
  agreement between aliases says nothing about accuracy.

Detecting the alias regime is exactly what a cross-solver gate cannot do on its
own, which is why :func:`family_error_ratios` / the ``alias_pair`` attribution in
:func:`cross_family_verdict` exist: a cross-solver verdict must be accompanied by
each family's own error against an independent reference, or the agreement is
unfalsifiable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .convergence import observed_order, pairwise_orders, roundoff_floor

__all__ = [
    "INTERIOR",
    "max_point_error",
    "cross_family_difference",
    "percent_of",
    "normalisation_ratio",
    "order_bracket",
    "order_law",
    "family_error_ratios",
    "half_gate_budget",
    "required_resolution",
    "CrossFamilyVerdict",
    "cross_family_verdict",
]

#: Use this for ``interior`` to drop the outermost node layer (where a family
#: may be closed at a lower order than in the interior, e.g. FD stencils that
#: consume boundary data directly).
INTERIOR = object()


def _as_field(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D grid field, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite entries")
    return array


def _trim(array: np.ndarray, interior: bool) -> np.ndarray:
    if not interior:
        return array
    if min(array.shape) < 3:
        raise ValueError("interior=True needs a field of at least 3x3 nodes")
    return array[1:-1, 1:-1]


def max_point_error(
    numeric,
    exact,
    *,
    interior: bool = False,
    relative: bool = False,
) -> float:
    """``max`` norm of ``numeric - exact`` on a shared node set.

    ``relative=True`` divides by ``max|exact|`` (the field amplitude), which is
    the normalisation convention that has to be *stated explicitly* whenever a
    percentage is quoted: the same absolute difference becomes a much larger
    percentage when the reference amplitude is small (this is exactly the
    3.495% vs 0.890% discrepancy in the anchor's ladder, where the ``v``
    component's amplitude is only ~1/3.9 of the lid speed).
    """
    a = _trim(_as_field(numeric, "numeric"), interior)
    b = _trim(_as_field(exact, "exact"), interior)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    error = float(np.abs(a - b).max())
    if not relative:
        return error
    scale = float(np.abs(b).max())
    if scale == 0.0:
        raise ValueError("relative error against an identically zero field")
    return error / scale


from .registry import capability
@capability(
    name="xfamily-order",
    category="analysis",
    maturity="planned",
    evidence={"feature_ledger": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE"},
    description="Cross-family (DG vs FVM/LBM) order comparison",
    tags=['xfamily', 'order', 'xval']
)
def cross_family_difference(field_a, field_b, *, interior: bool = False) -> float:
    """``max`` norm of ``A - B`` on the shared node set (no reference used)."""
    a = _trim(_as_field(field_a, "field_a"), interior)
    b = _trim(_as_field(field_b, "field_b"), interior)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).max())


def percent_of(difference: float, scale: float) -> float:
    """Quote a difference as a percentage of an explicitly supplied scale."""
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return 100.0 * float(difference) / float(scale)


def normalisation_ratio(large_amplitude: float, small_amplitude: float) -> float:
    """Factor by which a percentage changes when the normaliser changes.

    ``normalisation_ratio(max|u|, max|v|)`` is the number that reconciles the
    two percentages of the same absolute difference.  It is returned so that a
    report can state the conversion instead of quoting two "different"
    percentages as if they were two different measurements.
    """
    if large_amplitude <= 0.0 or small_amplitude <= 0.0:
        raise ValueError("amplitudes must be positive")
    if small_amplitude > large_amplitude:
        raise ValueError("small_amplitude must not exceed large_amplitude")
    return float(large_amplitude) / float(small_amplitude)


def order_bracket(*family_orders: float) -> tuple[float, float]:
    """Admissible interval ``(min, max)`` for the observed cross-family order.

    ``A - B`` is the difference of two truncation errors, so its decay order is
    the order of whichever error dominates.  In the dominant-family regime that
    order is one of the family orders, hence the interval.  An observation
    outside the interval that is *not* accompanied by ``cancellation_index`` << 1
    (alias regime) means a family is unconverged or mis-sampled.
    """
    orders = [float(p) for p in family_orders]
    if not orders:
        raise ValueError("at least one family order is required")
    if any(p <= 0.0 for p in orders):
        raise ValueError("family orders must be positive")
    return min(orders), max(orders)


def order_law(*family_orders: float) -> float:
    """Asymptotic ceiling of the observed cross-family order (``min``).

    Kept for callers that only need the lower edge; :func:`order_bracket` is the
    complete statement.
    """
    return order_bracket(*family_orders)[0]


def family_error_ratios(
    differences: Iterable[float],
    errors_a: Iterable[float],
    errors_b: Iterable[float],
) -> list[float]:
    """``|A - B| / max(e_A, e_B)`` per level -- the alias detector.

    Values near 1 mean one family's own error explains the whole difference
    (dominant-family regime: the gate is a real measurement).  Values far below
    1 mean the leading truncation terms cancel because both families share them
    (alias regime: the gate is measuring the difference of two closures of the
    same stencil, not accuracy).
    """
    d = [float(v) for v in differences]
    ea = [float(v) for v in errors_a]
    eb = [float(v) for v in errors_b]
    if not (len(d) == len(ea) == len(eb)):
        raise ValueError("differences and both error series must have the same length")
    ratios = []
    for dk, a, b in zip(d, ea, eb):
        scale = max(a, b)
        if scale <= 0.0:
            raise ValueError("family errors must be positive")
        ratios.append(dk / scale)
    return ratios


def half_gate_budget(gate_percent: float) -> float:
    """Per-family own-error budget implied by a cross-family gate.

    ``|A - B| <= e_A + e_B`` gives the necessary condition
    ``max(e_A, e_B) <= gate/2``; returned in the same percent unit as the gate.
    """
    gate = float(gate_percent)
    if gate <= 0.0:
        raise ValueError("gate_percent must be positive")
    return 0.5 * gate


def required_resolution(
    resolutions: Iterable[float],
    differences: Iterable[float],
    target: float,
    *,
    order_source: str = "finest_pair",
) -> float:
    """Extrapolate the resolution at which ``differences`` would reach ``target``.

    Uses the measured decay order of the two finest levels (``order_source ==
    "finest_pair"``) or a least-squares fit over all levels (``"lsq"``), then
    inverts ``D(n) = C n^{-p}`` through the finest point.  Returns ``inf`` when
    the measured order is not positive (the ladder is not decaying), so callers
    cannot accidentally present a non-decaying ladder as "needs one more grid".
    """
    n = np.asarray(list(resolutions), dtype=float)
    d = np.asarray(list(differences), dtype=float)
    if n.size != d.size or n.size < 2:
        raise ValueError("need matching sequences of at least two levels")
    if np.any(n <= 0.0) or np.any(d <= 0.0):
        raise ValueError("resolutions and differences must be positive")
    if target <= 0.0:
        raise ValueError("target must be positive")
    p = observed_order(n, d) if order_source == "lsq" else pairwise_orders(n, d)[-1]
    if p <= 0.0:
        return math.inf
    return float(n[-1] * (d[-1] / target) ** (1.0 / p))


@dataclass
class CrossFamilyVerdict:
    """Machine-checkable verdict for one cross-family ladder (rules R1-R5)."""

    resolutions: list[float]
    differences: list[float]
    gate_percent: float
    gate_order: float
    converged_a: bool = True
    converged_b: bool = True
    floor: float = 0.0
    signal_levels: float = 10.0
    family_errors_a: list[float] = field(default_factory=list)
    family_errors_b: list[float] = field(default_factory=list)
    cancellation_index: list[float] = field(default_factory=list)
    p_pairwise: list[float] = field(default_factory=list)
    p_finest: float = math.nan
    p_lsq: float = math.nan
    used_levels: int = 0
    degenerate: bool = False
    gate_pct_passed: bool = False
    gate_order_passed: bool = False
    order_attribution: str = "not_run"
    order_bracket: tuple[float, float] = (math.nan, math.nan)
    notes: list[str] = field(default_factory=list)

    #: Attributions from which an acceptance verdict may be drawn.
    CREDITABLE = ("dominant_family_controls",)

    @property
    def passed(self) -> bool:
        """Both gates green *and* the measurement was admissible (R2-R5)."""
        return bool(
            self.gate_pct_passed
            and self.gate_order_passed
            and not self.degenerate
            and self.order_attribution in self.CREDITABLE
            and self.converged_a
            and self.converged_b
        )

    def as_dict(self) -> dict:
        return {
            "resolutions": list(self.resolutions),
            "differences": list(self.differences),
            "gate_percent": self.gate_percent,
            "gate_order": self.gate_order,
            "converged_a": self.converged_a,
            "converged_b": self.converged_b,
            "floor": self.floor,
            "signal_levels": self.signal_levels,
            "family_errors_a": list(self.family_errors_a),
            "family_errors_b": list(self.family_errors_b),
            "cancellation_index": list(self.cancellation_index),
            "p_pairwise": list(self.p_pairwise),
            "p_finest": self.p_finest,
            "p_lsq": self.p_lsq,
            "used_levels": self.used_levels,
            "degenerate": self.degenerate,
            "gate_pct_passed": self.gate_pct_passed,
            "gate_order_passed": self.gate_order_passed,
            "order_attribution": self.order_attribution,
            "order_bracket": list(self.order_bracket),
            "passed": self.passed,
            "notes": list(self.notes),
        }


def cross_family_verdict(
    resolutions: Sequence[float],
    differences: Sequence[float],
    *,
    gate_percent: float,
    gate_order: float,
    family_orders: Sequence[float] | None = None,
    family_errors_a: Sequence[float] | None = None,
    family_errors_b: Sequence[float] | None = None,
    alias_threshold: float = 0.1,
    scale: float | None = None,
    converged_a: bool = True,
    converged_b: bool = True,
    field_norm: float = 1.0,
    signal_levels: float = 10.0,
) -> CrossFamilyVerdict:
    """Apply rules R1-R5 to one measured cross-family ladder.

    Parameters
    ----------
    resolutions : shared grid resolutions (R1 -- one sequence, by construction).
    differences : ``max``-norm cross-family differences on those grids.
    gate_percent, gate_order : the acceptance gate, in percent of ``scale``.
    family_orders : each family's own observed order, used to bracket the
        observed cross-family order (R5).
    family_errors_a, family_errors_b : each family's own ``max``-norm error
        against an independent reference on the same grids.  Supplying them
        enables the alias detector; without them the verdict can only report
        that the pair agrees, not that it is accurate.
    alias_threshold : ``|A-B| / max(e_A,e_B)`` below this on *every* level marks
        the pair as aliased (the families share their leading truncation term).
    """
    n = [float(v) for v in resolutions]
    d = [float(v) for v in differences]
    if len(n) != len(d) or len(n) < 3:
        raise ValueError("need matching sequences of at least three levels")
    if any(v <= 0.0 for v in n) or any(v <= 0.0 for v in d):
        raise ValueError("resolutions and differences must be positive")
    normaliser = float(field_norm if scale is None else scale)
    floor = roundoff_floor(float(field_norm))

    verdict = CrossFamilyVerdict(
        resolutions=n,
        differences=d,
        gate_percent=float(gate_percent),
        gate_order=float(gate_order),
        converged_a=bool(converged_a),
        converged_b=bool(converged_b),
        floor=floor,
        signal_levels=float(signal_levels),
        order_bracket=order_bracket(*family_orders) if family_orders else (math.nan, math.nan),
    )
    if family_errors_a is not None and family_errors_b is not None:
        ea = [float(v) for v in family_errors_a]
        eb = [float(v) for v in family_errors_b]
        if len(ea) != len(n) or len(eb) != len(n):
            raise ValueError("family error series must match the resolution ladder")
        verdict.family_errors_a = ea
        verdict.family_errors_b = eb
        verdict.cancellation_index = family_error_ratios(d, ea, eb)

    # R3: a pair that agrees to round-off is degenerate, not accurate.
    if max(d) <= signal_levels * floor:
        verdict.degenerate = True
        verdict.order_attribution = "not_run"
        verdict.notes.append(
            "R3 degenerate: cross-family difference is at the round-off floor "
            f"({max(d):.3e} <= {signal_levels:g}x{floor:.3e}); the two families "
            "are not distinguishable on this problem, so the observed order is "
            "undefined and neither gate may be reported."
        )
        verdict.gate_pct_passed = bool(percent_of(max(d), normaliser) < gate_percent)
        verdict.gate_order_passed = False
        return verdict

    # R4: keep only levels above the signal threshold before fitting the order.
    keep = [i for i, v in enumerate(d) if v >= signal_levels * floor]
    if len(keep) < 3:
        verdict.order_attribution = "not_run"
        verdict.notes.append(
            "R4: fewer than three levels above the signal threshold; the order "
            "is not measured (unknown), not passed."
        )
        verdict.gate_pct_passed = bool(percent_of(max(d), normaliser) < gate_percent)
        verdict.gate_order_passed = False
        return verdict
    nk = [n[i] for i in keep]
    dk = [d[i] for i in keep]
    verdict.used_levels = len(keep)
    verdict.p_pairwise = pairwise_orders(nk, dk)
    verdict.p_finest = verdict.p_pairwise[-1]
    verdict.p_lsq = observed_order(nk, dk)

    worst_percent = percent_of(max(dk), normaliser)
    verdict.gate_pct_passed = bool(worst_percent < gate_percent)
    verdict.gate_order_passed = bool(verdict.p_finest >= gate_order)

    # R5: attribute the observed order.
    lo, hi = verdict.order_bracket
    if verdict.cancellation_index:
        ratios = [verdict.cancellation_index[i] for i in keep]
        if max(ratios) < alias_threshold:
            verdict.order_attribution = "alias_pair_shared_error"
            verdict.notes.append(
                "R5 alias regime: the cross-family difference is only "
                f"{100 * max(ratios):.1f}% of either family's own error, so both "
                "families share their leading truncation term and the observed "
                f"order ({verdict.p_finest:.2f}) exceeds both family orders. "
                "Agreement between aliases is not evidence of accuracy: this "
                "ladder must be reported with each family's own error against an "
                "independent reference, and the 3% gate must not be credited."
            )
            return verdict

    if math.isnan(lo):
        verdict.order_attribution = "no_family_orders"
        verdict.notes.append(
            "R5: family orders not supplied; the observed order cannot be "
            "attributed to a family, so the gate order is not creditable."
        )
    elif verdict.p_finest < lo - 0.25:
        verdict.order_attribution = "below_range_violation"
        verdict.notes.append(
            f"R5 violation: observed {verdict.p_finest:.2f} < bracket "
            f"[{lo:.2f}, {hi:.2f}]; one family is probably not converged or not "
            "sampled on the shared grid."
        )
    elif verdict.p_finest > hi + 0.25:
        verdict.order_attribution = "above_range_violation"
        verdict.notes.append(
            f"R5 violation: observed {verdict.p_finest:.2f} > bracket "
            f"[{lo:.2f}, {hi:.2f}] without an alias signature; check the metric "
            "and the sameness of the node sets."
        )
    else:
        verdict.order_attribution = "dominant_family_controls"
    if not (converged_a and converged_b):
        verdict.notes.append(
            "R2: at least one family is not converged by its own criterion; the "
            "cross-family difference then measures under-convergence."
        )
    return verdict
