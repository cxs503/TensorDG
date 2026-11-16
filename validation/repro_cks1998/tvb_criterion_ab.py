"""A/B of the *shipped* TVB troubled-cell criterion against the Cockburn--Shu form.

Motivation (anchor ``REPRO-DG-CKS1998``, verification debt)
-----------------------------------------------------------
``tensordg.tvb.troubled_from_differences`` does **not** implement the literal
generalised-minmod test of Cockburn & Shu (Math. Comp. 52 (1989) eq. (2.10);
reviewed in Cockburn & Shu, JCP 141 (1998) §2.2 and SIAM Review 43 (2001)
§2.4).  The paper releases a cell when a *one-sided cell-average difference*
is already ``O(h**2)``, i.e. when the cell straddles a smooth extremum::

    m~(a1, a2) = a1             if |a1| <= M h^2
               = a2             if |a2| <= M h^2
               = minmod(a1,a2)  otherwise,      a1 = u_i - u_{i-1},
                                                a2 = u_{i+1} - u_i

The shipped kernel instead compares the *deviation of each one-sided difference
from the minmod slope* against ``M h^2``.  For smooth data that deviation equals
``|u''| h**2``, so the shipped test reads

    troubled  <=>  |u''| > M          (shipped, smooth-regime reading)

while the paper's test reads

    released  <=>  min(|a1|,|a2|) <= M h^2   (i.e. |u'| h <~ M h^2 near an
                                              extremum, where |u'| = O(h))

The two coincide *in the extremum-release regime* (both switch at
``M ~ |u''|``) but differ in monotone smooth stretches, where the paper's test
still limits and the shipped test does not.  Neither reading had a gate: the
A1 order gate in ``bench_spec`` runs the solver **unlimited**, and every
limiter gate that exists uses ``M = 0`` (strict TVD), i.e. the opposite regime.
This module supplies the missing measurement.

The exact reference is a linear advection problem whose second derivative is
known in closed form: ``u0 = sin(2*pi*x)`` gives ``max|u''| = (2*pi)**2``.
That number, not a tuned value, is the ``M`` used by the certificate below.

Gates produced (see ``GATES``)
-----------------------------
``G1_inert_above_curvature``
    At ``M = max|u''|`` the shipped indicator must (a) flag **zero** cells,
    (b) reproduce the unlimited solution to the last bit, and (c) keep the
    observed order ``>= p + 0.5``.  Non-vacuous: the same assertion fails at
    ``M = max|u''|/4`` (indicator fires) and fails for the paper-form test at
    the same ``M`` (it still limits monotone cells), both measured here.
``G2_no_fire_below_half_curvature``
    At ``M = max|u''|/4`` the shipped indicator *does* fire, so G1 cannot be
    satisfied by a limiter that is silently switched off.

Run ``python -m validation.repro_cks1998.tvb_criterion_ab --profile bench``
to regenerate ``artifacts/tvb_criterion_ab/tvb_criterion_ab.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from tensordg import limiters as _limiters
from tensordg.advection_1d import l2_error, solve_advection
from tensordg.convergence import observed_order
from tensordg.tvb import minmod as _minmod

#: ``max |u''|`` of the reference profile ``sin(2*pi*x)`` -- the only
#: theory-derived TVB constant used by the certificates (no tuning).
SINE_CURVATURE_MAX: float = (2.0 * math.pi) ** 2

#: The public API default of :func:`tensordg.advection_1d.solve_advection`.
DEFAULT_TVB_CONSTANT: float = 1.0

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "tvb_criterion_ab"

PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {"degrees": (1, 2), "cells": (20, 40, 80)},
    "bench": {"degrees": (1, 2, 3), "cells": (20, 40, 80, 160)},
}

#: TVB constants swept, expressed as multiples of ``SINE_CURVATURE_MAX``.
M_MULTIPLES: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)

FINAL_TIME = 0.5
CFL = 0.3
TIME_INTEGRATOR = "ssp_rk3"

Criterion = Callable[..., "tuple[torch.Tensor, torch.Tensor]"]


# --------------------------------------------------------------------------
# Criteria
# --------------------------------------------------------------------------
def paper_troubled_from_differences(
    d_left: torch.Tensor,
    d_right: torch.Tensor,
    spacing: float,
    tvb_constant: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The literal Cockburn--Shu generalised-minmod test (1989 eq. (2.10)).

    A cell is *released* (not troubled) as soon as either one-sided difference
    is ``O(h**2)``, which is exactly the smooth-extremum situation.  Returns the
    same ``(troubled, slope)`` contract as :func:`tensordg.tvb.troubled_from_differences`.
    """
    a_left = d_left.abs()
    a_right = d_right.abs()
    mm = 0.5 * (torch.sign(d_left) + torch.sign(d_right)) * torch.minimum(a_left, a_right)
    threshold = float(tvb_constant) * float(spacing) ** 2
    # released iff min(|a1|, |a2|) <= M h^2; over the variable axis a cell is
    # troubled when *every* variable fails the release test (same any-var
    # convention as the shipped ``.max(dim=-1)`` reduction).
    released = (torch.minimum(a_left, a_right) <= threshold)
    troubled = ~released.all(dim=-1)
    return troubled, mm


@contextlib.contextmanager
def criterion(kind: str = "shipped"):
    """Install ``paper`` (or keep ``shipped``) as the limiter's troubled test."""
    if kind == "shipped":
        yield
        return
    if kind != "paper":
        raise ValueError(f"unknown criterion {kind!r}; choose 'shipped' or 'paper'")
    global _ACTIVE_CRITERION
    saved = _limiters.troubled_from_differences
    saved_kind = _ACTIVE_CRITERION
    _limiters.troubled_from_differences = paper_troubled_from_differences
    _ACTIVE_CRITERION = kind
    try:
        yield
    finally:
        _limiters.troubled_from_differences = saved
        _ACTIVE_CRITERION = saved_kind


def troubled_mask(
    coeff: torch.Tensor,
    tvb_constant: float,
    kind: str = "shipped",
) -> torch.Tensor:
    """Troubled-cell mask of a scalar modal state, periodic wrap-around.

    Mirrors ``limiters._detect``'s stencil (periodic neighbours) so the count is
    the one the solver actually applies, without instrumenting the solver.
    """
    ncells = coeff.shape[0]
    avg = coeff[:, 0]
    avg_pad = torch.cat((avg[-1:], avg, avg[:1]), dim=0)
    d_left = avg - avg_pad[:-2]
    d_right = avg_pad[2:] - avg
    spacing = 1.0 / ncells
    fn: Criterion = (
        paper_troubled_from_differences if kind == "paper"
        else _limiters.troubled_from_differences
    )
    troubled, _slope = fn(d_left.unsqueeze(-1), d_right.unsqueeze(-1), spacing, tvb_constant)
    return troubled


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def _initial_coeff(ncells: int, degree: int) -> torch.Tensor:
    _x, coeff, _meta = solve_advection(ncells=ncells, degree=degree, final_time=0.0)
    return coeff


_SOLVE_CACHE: dict[tuple, torch.Tensor] = {}
#: Which troubled-cell test is installed; part of the cache key because the
#: ``criterion`` context manager swaps the kernel underneath the solver.
_ACTIVE_CRITERION: str = "shipped"


def _solve(ncells: int, degree: int, tvb_constant: float | None) -> torch.Tensor:
    key = (int(ncells), int(degree), tvb_constant, _ACTIVE_CRITERION)
    if key in _SOLVE_CACHE:
        return _SOLVE_CACHE[key]
    limiter = None if tvb_constant is None else "tvb"
    _x, coeff, _meta = solve_advection(
        ncells=ncells, degree=degree, final_time=FINAL_TIME, cfl=CFL,
        limiter=limiter, tvb_constant=(1.0 if tvb_constant is None else tvb_constant),
        time_integrator=TIME_INTEGRATOR,
    )
    _SOLVE_CACHE[key] = coeff
    return coeff


def order_ladder(
    degree: int,
    cells: Sequence[int],
    tvb_constant: float | None,
    kind: str = "shipped",
) -> dict[str, Any]:
    """Observed order, error ladder, and indicator activity for one (p, M)."""
    errs: list[float] = []
    troubled_fracs: list[float] = []
    diffs_vs_unlimited: list[float] = []
    with criterion("shipped" if tvb_constant is None else kind):
        for ncells in cells:
            coeff = _solve(ncells, degree, tvb_constant)
            errs.append(l2_error(_x_centers(ncells), coeff, final_time=FINAL_TIME))
            if tvb_constant is None:
                troubled_fracs.append(0.0)
                diffs_vs_unlimited.append(0.0)
            else:
                mask = troubled_mask(_initial_coeff(ncells, degree), tvb_constant, kind)
                troubled_fracs.append(float(mask.double().mean().item()))
                diffs_vs_unlimited.append(
                    float((coeff - _solve(ncells, degree, None)).abs().max().item())
                )
    order = observed_order(cells, errs) if len(cells) >= 2 else float("nan")
    return {
        "degree": int(degree),
        "criterion": "unlimited" if tvb_constant is None else kind,
        "tvb_constant": None if tvb_constant is None else float(tvb_constant),
        "m_over_curvature": (
            None if tvb_constant is None else float(tvb_constant) / SINE_CURVATURE_MAX
        ),
        "cells": [int(c) for c in cells],
        "l2": [float(e) for e in errs],
        "observed_order": float(order),
        "theory_order": int(degree) + 1,
        "order_floor": float(degree) + 0.5,
        "troubled_fraction_initial": [float(f) for f in troubled_fracs],
        "max_abs_diff_vs_unlimited": [float(d) for d in diffs_vs_unlimited],
    }


def _x_centers(ncells: int) -> torch.Tensor:
    h = 1.0 / ncells
    return (torch.arange(ncells, dtype=torch.float64) + 0.5) * h


def _satisfies(row: dict[str, Any]) -> bool:
    """G1's per-(p, M=max|u''|) predicate, stated once and reused by the gate."""
    return (
        max(row["troubled_fraction_initial"]) == 0.0
        and max(row["max_abs_diff_vs_unlimited"]) == 0.0
        and row["observed_order"] >= row["order_floor"]
    )


# --------------------------------------------------------------------------
# Gate definitions + evaluation
# --------------------------------------------------------------------------
GATES: dict[str, dict[str, Any]] = {
    "G1_inert_above_curvature": {
        "claim": "M = max|u''| => indicator flags 0 cells, solution is bit-identical "
                 "to unlimited, observed order >= p + 1/2",
        "measure": "rows at m_over_curvature == 1.0, criterion == shipped",
    },
    "G2_indicator_fires_below_half_curvature": {
        "claim": "M = max|u''|/4 => indicator flags a non-zero fraction of cells, "
                 "so G1 cannot be met by a switched-off limiter",
        "measure": "rows at m_over_curvature == 0.25, criterion == shipped",
    },
    "D1_public_default_constant": {
        "claim": "the public default tvb_constant=1.0 (``solve_advection``) is an absolute "
                 "M, not tied to the solution's curvature; for sin(2*pi*x) (max|u''|=39.5) "
                 "it flags ~all cells -- behaving exactly like M=0 -- and the observed order "
                 "falls at least a quarter order below the theoretical p+1",
        "measure": "rows with criterion == shipped_default, order vs floor",
    },
    "C1_paper_form_is_stricter": {
        "claim": "the literal Cockburn-Shu test flags at least as many cells as the "
                 "shipped test at the same M (the shipped test is the permissive one)",
        "measure": "paired troubled_fraction_initial, paper vs shipped",
    },
}


def evaluate(profile: str = "bench") -> dict[str, Any]:
    spec = PROFILES[profile]
    degrees: Iterable[int] = spec["degrees"]
    cells: Sequence[int] = spec["cells"]

    rows: list[dict[str, Any]] = []
    for degree in degrees:
        rows.append(order_ladder(degree, cells, None))
        default_row = order_ladder(degree, cells, DEFAULT_TVB_CONSTANT)
        default_row["criterion"] = "shipped_default"
        default_row["m_over_curvature"] = DEFAULT_TVB_CONSTANT / SINE_CURVATURE_MAX
        rows.append(default_row)
        for mult in M_MULTIPLES:
            for kind in ("shipped", "paper"):
                rows.append(order_ladder(degree, cells, mult * SINE_CURVATURE_MAX, kind))

    by_key = {(r["degree"], r["criterion"], r["m_over_curvature"]): r for r in rows}

    g1 = {}
    for degree in degrees:
        row = by_key[(degree, "shipped", 1.0)]
        g1[f"p{degree}"] = {"satisfied": bool(_satisfies(row)),
                           "observed_order": row["observed_order"],
                           "order_floor": row["order_floor"],
                           "troubled_fraction": max(row["troubled_fraction_initial"]),
                           "max_abs_diff": max(row["max_abs_diff_vs_unlimited"])}
    g2 = {}
    for degree in degrees:
        row = by_key[(degree, "shipped", 0.25)]
        g2[f"p{degree}"] = {"satisfied": bool(max(row["troubled_fraction_initial"]) > 0.0),
                            "troubled_fraction": max(row["troubled_fraction_initial"])}
    c1 = {}
    for degree in degrees:
        for mult in M_MULTIPLES[1:]:
            shipped = by_key[(degree, "shipped", mult)]
            paper = by_key[(degree, "paper", mult)]
            s_frac = max(shipped["troubled_fraction_initial"])
            p_frac = max(paper["troubled_fraction_initial"])
            c1[f"p{degree}_m{mult:g}"] = {
                "shipped_fraction": s_frac,
                "paper_fraction": p_frac,
                "satisfied": bool(p_frac >= s_frac),
            }

    d1 = {}
    for degree in degrees:
        row = by_key[(degree, "shipped_default", DEFAULT_TVB_CONSTANT / SINE_CURVATURE_MAX)]
        d1[f"p{degree}"] = {
            # Degradation, not a floor: the claim is that the default M loses at
            # least a quarter order against the theoretical p+1.
            "satisfied": bool(row["observed_order"] < row["theory_order"] - 0.25),
            "observed_order": row["observed_order"],
            "theory_order": row["theory_order"],
            "order_loss": row["theory_order"] - row["observed_order"],
            "troubled_fraction": max(row["troubled_fraction_initial"]),
        }

    gates = {
        "G1_inert_above_curvature": g1,
        "G2_indicator_fires_below_half_curvature": g2,
        "C1_paper_form_is_stricter": c1,
        "D1_public_default_constant": d1,
    }
    all_pass = all(v["satisfied"] for g in gates.values() for v in g.values())

    return {
        "anchor": "REPRO-DG-CKS1998",
        "instrument": "tvb_criterion_ab",
        "profile": profile,
        "sine_curvature_max": SINE_CURVATURE_MAX,
        "final_time": FINAL_TIME,
        "cfl": CFL,
        "time_integrator": TIME_INTEGRATOR,
        "gates": gates,
        "gate_definitions": GATES,
        "all_pass": bool(all_pass),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="bench", choices=sorted(PROFILES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    payload = evaluate(args.profile)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else ARTIFACT_DIR / f"tvb_criterion_ab_{args.profile}.json"
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}")

    header = ("\n p   M/max|u''|  criterion     order  floor  troubled%   max|du|")
    print(header)
    for row in payload["rows"]:
        mult = "-" if row["m_over_curvature"] is None else f"{row['m_over_curvature']:.2f}"
        print(f"{row['degree']:>2} {mult:>10} {row['criterion']:>10} "
              f"{row['observed_order']:>7.3f} {row['order_floor']:>6.1f} "
              f"{100.0 * max(row['troubled_fraction_initial']):>9.3f} "
              f"{max(row['max_abs_diff_vs_unlimited']):>9.2e}")
    for name, rows_ in payload["gates"].items():
        print(f"\n{name}: " + ", ".join(
            f"{k}={'PASS' if v['satisfied'] else 'FAIL'}" for k, v in rows_.items()))
    print(f"\nall_pass = {payload['all_pass']}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())