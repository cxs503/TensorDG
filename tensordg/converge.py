"""Convergence-table infrastructure for TensorDG.

Implements the deal.II-style convergence-table discipline:
  * h-refinement sequences (≥4 levels by default)
  * L2 / L∞ error norms
  * Observed order of convergence (p_obs) via log2 ratio
  * Richardson extrapolation for asymptotic error estimates
  * Assertion against theoretical order with configurable tolerance
  * GCI (Grid Convergence Index) calculator per Roache (1994)

Usage
-----
>>> from tensordg.converge import ConvergenceTable
>>> tbl = ConvergenceTable(problem="poisson", degree=2)
>>> tbl.run()
>>> tbl.print_table()
>>> tbl.assert_order(tolerance=0.3)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceEntry:
    """One row in a convergence table."""
    level: int
    ncells: int
    h: float
    dofs: int
    error_l2: float
    error_linf: float | None = None
    order_l2: float | None = None
    order_linf: float | None = None


@dataclass
class ConvergenceTable:
    """Collects h-refinement results and computes observed orders.

    Parameters
    ----------
    problem : str
        Identifier for the problem being tested (e.g. "poisson", "advection").
    degree : int
        Polynomial degree of the DG discretisation.
    solver : callable
        Function(ncells, degree, ...) -> (..., error_dict) where error_dict
        contains at least ``"absolute_l2"`` and optionally ``"absolute_linf"``.
    ncells_seq : sequence of int, optional
        Mesh sizes for the h-refinement study.  Defaults to powers of 2
        starting from ``base_ncells``.
    base_ncells : int
        Starting mesh size when *ncells_seq* is not given.
    n_levels : int
        Number of refinement levels (≥4 recommended).
    theoretical_order : float
        Expected asymptotic convergence rate (degree+1 for L2 in SIPG).
    """
    problem: str
    degree: int
    solver: Callable
    ncells_seq: Sequence[int] | None = None
    base_ncells: int = 4
    n_levels: int = 5
    theoretical_order: float | None = None
    entries: list[ConvergenceEntry] = field(default_factory=list)
    richardson_extrapolated: float | None = None

    def __post_init__(self):
        if self.ncells_seq is None:
            self.ncells_seq = [self.base_ncells * (2 ** i) for i in range(self.n_levels)]
        if self.theoretical_order is None:
            # Default: degree+1 for L2 norm of elliptic problems
            self.theoretical_order = self.degree + 1

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def run(self) -> "ConvergenceTable":
        """Execute the h-refinement study and populate *entries*."""
        self.entries.clear()
        prev_l2: float | None = None
        prev_linf: float | None = None
        for level, nc in enumerate(self.ncells_seq, start=1):
            result = self._solve_one(nc)
            err_l2 = result["absolute_l2"]
            err_linf = result.get("absolute_linf")
            dofs = result.get("dofs", nc * (self.degree + 1))
            h = 1.0 / nc if isinstance(nc, int) else 1.0 / max(nc)

            ord_l2 = None
            ord_linf = None
            if prev_l2 is not None and prev_l2 > 0 and err_l2 > 0:
                ord_l2 = math.log2(prev_l2 / err_l2)
            if err_linf is not None and prev_linf is not None and prev_linf > 0 and err_linf > 0:
                ord_linf = math.log2(prev_linf / err_linf)

            entry = ConvergenceEntry(
                level=level, ncells=nc, h=h, dofs=dofs,
                error_l2=err_l2, error_linf=err_linf,
                order_l2=ord_l2, order_linf=ord_linf,
            )
            self.entries.append(entry)
            prev_l2 = err_l2
            prev_linf = err_linf

        # Richardson extrapolation using the last three levels
        self._richardson()
        return self

    def _solve_one(self, ncells: int) -> dict:
        """Call the solver and return a dict with error norms."""
        result = self.solver(ncells, self.degree)
        # Normalise: ensure absolute_l2 is present
        if "absolute_l2" not in result:
            if "relative_l2" in result and "exact_l2" in result:
                result["absolute_l2"] = result["relative_l2"] * result["exact_l2"]
            else:
                raise ValueError("solver must return absolute_l2 or relative_l2+exact_l2")
        return result

    def _richardson(self):
        """Richardson extrapolation using the finest three levels."""
        if len(self.entries) < 3:
            return
        e1, e2, e3 = self.entries[-3:]  # coarse, medium, fine
        r = e2.h / e3.h  # refinement ratio (should be 2 for uniform halving)
        p = self.theoretical_order
        # u_exact ≈ (r^p * u_fine - u_medium) / (r^p - 1)
        # error_extrapolated ≈ |u_fine - u_medium| / (r^p - 1)
        if r ** p != 1.0:
            self.richardson_extrapolated = abs(e3.error_l2 - e2.error_l2) / (r ** p - 1.0)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_table(self) -> str:
        """Return a formatted convergence table string."""
        header = (
            f"{'Level':>5}  {'N':>6}  {'h':>10}  {'DOFs':>8}  "
            f"{'L2 Error':>14}  {'Order':>7}  "
            f"{'L∞ Error':>14}  {'Order':>7}"
        )
        lines = [
            f"Convergence Table: {self.problem}  (degree={self.degree}, "
            f"theoretical order={self.theoretical_order})",
            "=" * len(header),
            header,
            "-" * len(header),
        ]
        for e in self.entries:
            o_l2 = f"{e.order_l2:.4f}" if e.order_l2 is not None else "  ---  "
            o_linf = f"{e.order_linf:.4f}" if e.order_linf is not None else "  ---  "
            elinf = f"{e.error_linf:.6e}" if e.error_linf is not None else "     ---     "
            lines.append(
                f"{e.level:>5}  {e.ncells:>6}  {e.h:>10.6f}  {e.dofs:>8}  "
                f"{e.error_l2:>14.6e}  {o_l2:>7}  "
                f"{elinf:>14}  {o_linf:>7}"
            )
        if self.richardson_extrapolated is not None:
            lines.append(
                f"\nRichardson extrapolated error estimate: "
                f"{self.richardson_extrapolated:.6e}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise the table for JSON / report injection."""
        return {
            "problem": self.problem,
            "degree": self.degree,
            "theoretical_order": self.theoretical_order,
            "richardson_extrapolated": self.richardson_extrapolated,
            "entries": [
                {
                    "level": e.level, "ncells": e.ncells, "h": e.h,
                    "dofs": e.dofs, "error_l2": e.error_l2,
                    "error_linf": e.error_linf,
                    "order_l2": e.order_l2, "order_linf": e.order_linf,
                }
                for e in self.entries
            ],
        }

    # ------------------------------------------------------------------
    # Assertion
    # ------------------------------------------------------------------

    def assert_order(self, tolerance: float = 0.3, norm: str = "l2") -> None:
        """Assert that the observed order on the finest refinement is within
        *tolerance* of the theoretical order.

        Raises AssertionError if the condition is not met.
        """
        if len(self.entries) < 2:
            raise AssertionError("Need at least 2 refinement levels to compute order")
        last = self.entries[-1]
        obs = last.order_l2 if norm == "l2" else last.order_linf
        if obs is None:
            raise AssertionError(f"No observed {norm} order available")
        if obs < self.theoretical_order - tolerance:
            raise AssertionError(
                f"Observed {norm} order {obs:.4f} is below "
                f"theoretical {self.theoretical_order} - tolerance {tolerance} "
                f"= {self.theoretical_order - tolerance:.4f}"
            )

    def assert_all_orders(self, tolerance: float = 0.3, norm: str = "l2",
                          skip_first: int = 1) -> None:
        """Assert that *all* computed orders (after skipping pre-asymptotic
        levels) meet the theoretical order within *tolerance*."""
        for e in self.entries[skip_first:]:
            obs = e.order_l2 if norm == "l2" else e.order_linf
            if obs is None:
                continue
            if obs < self.theoretical_order - tolerance:
                raise AssertionError(
                    f"Level {e.level}: observed {norm} order {obs:.4f} < "
                    f"theoretical {self.theoretical_order} - {tolerance}"
                )


# ---------------------------------------------------------------------------
# GCI (Grid Convergence Index) — Roache (1994) standard
# ---------------------------------------------------------------------------

@dataclass
class GCIResult:
    """Result of a GCI calculation."""
    refinement_ratio: float
    apparent_order: float
    gci_fine: float       # GCI on the fine grid (%)
    gci_coarse: float     # GCI on the coarse grid (%)
    extrapolated_value: float
    fine_value: float
    coarse_value: float
    asymptotic_ratio: float  # GCI_coarse / (r^p * GCI_fine)


def compute_gci(
    fine_value: float,
    coarse_value: float,
    refinement_ratio: float,
    theoretical_order: float | None = None,
    apparent_order: float | None = None,
    safety_factor: float = 1.25,
) -> GCIResult:
    """Compute the Grid Convergence Index per Roache (1994).

    Parameters
    ----------
    fine_value : float
        Solution quantity on the fine grid.
    coarse_value : float
        Solution quantity on the coarse grid.
    refinement_ratio : float
        r = h_coarse / h_fine  (typically 2 for uniform halving).
    theoretical_order : float, optional
        If *apparent_order* is not given, use this as the order.
    apparent_order : float, optional
        Computed observed order from a three-grid study.
    safety_factor : float
        Fs = 1.25 (recommended by Roache for ≥3 grids).

    Returns
    -------
    GCIResult
    """
    if apparent_order is not None:
        p = apparent_order
    elif theoretical_order is not None:
        p = theoretical_order
    else:
        raise ValueError("Must provide apparent_order or theoretical_order")

    if refinement_ratio <= 1.0:
        raise ValueError("refinement_ratio must be > 1")

    eps = abs(fine_value - coarse_value) / abs(fine_value) if fine_value != 0 else abs(fine_value - coarse_value)
    gci_fine = safety_factor * eps / (refinement_ratio ** p - 1.0)
    gci_coarse = safety_factor * eps * (refinement_ratio ** p) / (refinement_ratio ** p - 1.0)

    # Richardson extrapolated value
    extrap = (refinement_ratio ** p * fine_value - coarse_value) / (refinement_ratio ** p - 1.0)

    # Asymptotic ratio: GCI_coarse / (r^p * GCI_fine)
    # For exact power-law convergence with f_exact=0, this equals 1/r^p.
    # For f_exact≠0 in the asymptotic range, this approaches 1.
    denom = (refinement_ratio ** p) * gci_fine
    asymptotic_ratio = gci_coarse / denom if denom != 0 else float("nan")

    return GCIResult(
        refinement_ratio=refinement_ratio,
        apparent_order=p,
        gci_fine=gci_fine,
        gci_coarse=gci_coarse,
        extrapolated_value=extrap,
        fine_value=fine_value,
        coarse_value=coarse_value,
        asymptotic_ratio=asymptotic_ratio,
    )


def compute_apparent_order(
    f1: float, f2: float, f3: float,
    r: float = 2.0,
) -> float:
    """Compute the apparent order p from three solutions on grids with
    constant refinement ratio r.

    f1 = finest, f2 = medium, f3 = coarsest.
    Uses the formula from Celik et al. (2008) / Roache.
    """
    eps21 = f2 - f1
    eps32 = f3 - f2
    if eps21 == 0 or eps32 == 0:
        return float("inf")
    q = abs(eps32 / eps21)
    if q <= 0:
        return float("nan")
    return math.log(q) / math.log(r)


def gci_three_grid(
    f1: float, f2: float, f3: float,
    r: float = 2.0,
    safety_factor: float = 1.25,
) -> dict:
    """Full three-grid GCI analysis.

    Returns a dict with fine/coarse GCI, apparent order, extrapolated value,
    and asymptotic-range check.

    The asymptotic-range check compares GCI_32 / (r^p * GCI_21) against
    the expected value of 1/r^p (for f_exact=0) or 1 (for f_exact≠0).
    We use the more conservative 1/r^p as the reference since our test
    cases typically have f_exact=0.
    """
    p = compute_apparent_order(f1, f2, f3, r)
    gci_21 = compute_gci(f1, f2, r, apparent_order=p, safety_factor=safety_factor)
    gci_32 = compute_gci(f2, f3, r, apparent_order=p, safety_factor=safety_factor)

    # Asymptotic range check:
    # For exact power-law convergence, GCI_32 / (r^p * GCI_21) = 1/r^p
    # We check against this expected value.
    expected_ratio = 1.0 / (r ** p) if p != 0 else float("nan")
    denom = (r ** p) * gci_21.gci_fine
    actual_ratio = gci_32.gci_fine / denom if denom != 0 else float("nan")

    # In asymptotic range: actual_ratio ≈ expected_ratio
    in_asymptotic = (
        abs(actual_ratio - expected_ratio) < 0.1 * expected_ratio
        if expected_ratio != 0 and not math.isnan(expected_ratio)
        else False
    )

    return {
        "apparent_order": p,
        "gci_fine_21": gci_21.gci_fine,
        "gci_coarse_21": gci_21.gci_coarse,
        "gci_fine_32": gci_32.gci_fine,
        "gci_coarse_32": gci_32.gci_coarse,
        "extrapolated_value": gci_21.extrapolated_value,
        "asymptotic_ratio": actual_ratio,
        "expected_asymptotic_ratio": expected_ratio,
        "in_asymptotic_range": in_asymptotic,
    }
