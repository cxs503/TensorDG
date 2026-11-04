"""Bisection harness: wall heat-flux order and residual of the ``ns_dg_2d``
Dirichlet wall path, per *solver revision* and per ``wall_gradient`` mode.

Why this tool exists (m3139, ratchet channel)
---------------------------------------------
``tests/test_ns_dg_2d_wall_scope.py`` (m2786) pins the wall heat flux at the
*element design order* (P1 -> 1, P2 -> 2, P3 -> 3) and at round-off for P3.
Those numbers were measured against the solver *before* PR #36 (m2920), whose
``+81/-13`` delta in ``tensordg/ns_dg_2d.py`` added the ``wall_gradient="taylor"``
one-sided extrapolation and made it the default.  Four assertions went red on
main afterwards.  This tool answers the only question that matters:

    did the solver lose accuracy (physical regression), or was the m2786
    assertion anchored to a solver behaviour that no longer exists?

It measures the same quantities the red tests assert on, for whichever
``tensordg/ns_dg_2d.py`` is importable, so the *same harness* can be run
against any revision (``PYTHONPATH`` pointing at a checkout of that revision):

``flux``      ``|F_energy,bottom - analytic|`` on ny = 8/16/32/64 for
              degree 1/2/3, and the observed order between successive meshes.
``residual``  ``max|R(e) - R(T)|`` of the exact analytic steady state (the
              quantity behind the m2786 residual assertions).

Both ``wall_gradient`` modes are measured when the revision supports them
(``"taylor"`` only exists after PR #36); an unsupported mode is reported as
``"unsupported"`` instead of silently falling back, so the table never mixes
two different code paths under one label.

Usage::

    PYTHONPATH=. python -m tools.bisect_ns_wall_gradient --label main@3782bf1b
    PYTHONPATH=. python -m tools.bisect_ns_wall_gradient --label pre36 --json out.json
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall import (
    ANALYTIC_E_FLUX,
    ANALYTIC_E_FLUX_BOTTOM,
    ANALYTIC_MX_FLUX,
    KAPPA,
    MU,
    build_state,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "ns_dg_2d_wall"
MESHES = (8, 16, 32, 64)
DEGREES = (1, 2, 3)


def _supports(fn, name):
    return name in inspect.signature(fn).parameters


def _accepts_wall_gradient():
    return _supports(ns._rhs_ns, "wall_gradient")


def _accepts_normal_derivative():
    return _supports(ns._boundary_flux, "normal_derivative")


MODES = ("face", "taylor")
if not _accepts_normal_derivative():
    MODES = ("face",)  # pre-#36 revision: no reconstruction argument exists


def wall_fluxes(ncells, degree, mode="taylor"):
    """Same call the scope tests make, tolerant of the pre-#36 signatures."""
    d = build_state(ncells, degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    h = d["dy"]
    left = ns._face_primitives(d["coeff"], d["basis"], d["derivative"], d["signs"], 1, "left")
    right = ns._face_primitives(d["coeff"], d["basis"], d["derivative"], d["signs"], 1, "right")
    build_recon = getattr(ns, "_wall_normal_derivative", None)
    recon = None if build_recon is None else build_recon(
        d["coeff"], d["basis"], d["signs"], h, 1)
    if mode == "face" or recon is None:
        recon = None
    common = (1, h, MU, KAPPA, penalty, 2.0 / h, 2.0 / d["dx"])
    if _accepts_normal_derivative():
        top = ns._boundary_flux(right, d["boundary"]["ymax"], *common,
                                None if recon is None else recon["right"])
        bottom = ns._boundary_flux(left, d["boundary"]["ymin"], *common,
                                   None if recon is None else recon["left"])
    else:
        assert recon is None
        top = ns._boundary_flux(right, d["boundary"]["ymax"], *common)
        bottom = ns._boundary_flux(left, d["boundary"]["ymin"], *common)
    return dict(
        energy_bottom=float(bottom[0, 0, 0, 3]),
        energy_top=float(top[0, -1, 0, 3]),
        mx_bottom=float(bottom[0, 0, 0, 1]),
    )


def _residual(degree, ny, mode="taylor"):
    d = build_state((2, ny), degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    args = (d["coeff"], d["basis"], d["derivative"], d["weights"], d["signs"],
            d["dx"], d["dy"], (True, False), d["boundary"], MU, KAPPA, penalty)
    rhs = ns._rhs_ns(*args, mode) if _accepts_wall_gradient() else ns._rhs_ns(*args)
    return float(rhs.abs().max()), float(rhs[..., 3].abs().max())


def order(coarse, fine):
    if fine <= 0.0 or coarse <= 0.0:
        return None
    return math.log2(coarse / fine)


def section_flux(mode):
    rows = []
    for degree in DEGREES:
        previous = None
        for ny in MESHES:
            f = wall_fluxes((2, ny), degree, mode)
            row = dict(
                mode=mode, degree=degree, ny=ny, h=1.0 / ny,
                energy_bottom=f["energy_bottom"],
                energy_error_bottom=abs(f["energy_bottom"] - ANALYTIC_E_FLUX_BOTTOM),
                energy_error_top=abs(f["energy_top"] - ANALYTIC_E_FLUX),
                mx_error_bottom=abs(f["mx_bottom"] - ANALYTIC_MX_FLUX),
            )
            row["order_energy_bottom"] = (
                None if previous is None else order(previous, row["energy_error_bottom"])
            )
            previous = row["energy_error_bottom"]
            rows.append(row)
    return rows


def section_residual(mode):
    rows = []
    for degree in DEGREES:
        previous = None
        for ny in MESHES:
            magnitude, energy = _residual(degree, ny, mode)
            row = dict(mode=mode, degree=degree, ny=ny,
                       rhs_max=magnitude, rhs_energy=energy)
            row["order_rhs"] = None if previous is None else order(previous, magnitude)
            previous = magnitude
            rows.append(row)
    return rows


def design_order_summary(rows):
    """Observed order over the 8 -> 32 span, exactly as the red tests compute it."""
    out = {}
    for degree in DEGREES:
        picked = {r["ny"]: r["energy_error_bottom"] for r in rows if r["degree"] == degree}
        err8, err32 = picked[8], picked[32]
        out[f"p{degree}"] = dict(
            error_8=err8, error_32=err32,
            observed_order=math.log2(err8 / err32) / 2.0,
            design_order=float(degree),
        )
    return out


def scope_assertions(flux_rows, residual_rows, mode):
    """The four m2786 assertions that went red on main, evaluated from the table.

    Kept programmatic so the report never hand-transcribes a verdict.
    """
    flux = {(r["degree"], r["ny"]): r["energy_error_bottom"] for r in flux_rows}
    res = {(r["degree"], r["ny"]): r["rhs_max"] for r in residual_rows}
    p1_res = {(r["degree"], r["ny"]): r["rhs_energy"] for r in residual_rows}
    order_p3 = math.log2(flux[(3, 8)] / flux[(3, 32)]) / 2.0
    ratio = {ny: p1_res[(1, ny)] * (1.0 / ny) / flux[(1, ny)] for ny in (16, 32)}
    checks = {
        "wall_heat_flux_converges_at_design_order[p3]": dict(
            observed=order_p3, threshold="> 2.65", passed=order_p3 > 2.65),
        "wall_heat_flux_is_round_off_accurate_at_p3": dict(
            observed=flux[(3, 32)], threshold="< 2e-7", passed=flux[(3, 32)] < 2e-7),
        "exact_state_residual_decays_with_degree": dict(
            p2_ratio=res[(2, 32)] / res[(2, 8)], p3_ratio=res[(3, 32)] / res[(3, 8)],
            p3_over_p1=res[(3, 32)] / res[(1, 8)],
            threshold="p2<0.6, p3<0.25, p3_fine<1e-3*p1",
            passed=(res[(2, 32)] < 0.6 * res[(2, 8)]
                    and res[(3, 32)] < 0.25 * res[(3, 8)]
                    and res[(3, 32)] < 1e-3 * res[(1, 8)])),
        "residual_equals_flux_error_over_h_at_p1": dict(
            ratio_ny16=ratio[16], ratio_ny32=ratio[32], threshold="2.9 < ratio < 3.1",
            passed=all(2.9 < v < 3.1 for v in ratio.values())),
    }
    return checks


def git_rev():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="row label for the table")
    parser.add_argument("--json", default=None, help="write the raw table here")
    args = parser.parse_args(argv)

    payload = dict(label=args.label, rev=git_rev(), modes=list(MODES),
                   suite="tools.bisect_ns_wall_gradient", flux={}, residual={},
                   summary={}, assertions={})
    for mode in MODES:
        payload["flux"][mode] = section_flux(mode)
        payload["residual"][mode] = section_residual(mode)
        payload["summary"][mode] = design_order_summary(payload["flux"][mode])
        payload["assertions"][mode] = scope_assertions(
            payload["flux"][mode], payload["residual"][mode], mode)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n")

    print(f"{args.label}  rev={payload['rev']}  modes={list(MODES)}")
    header = f"{'mode':7s} {'deg':>3s} " + " ".join(f"{'err@'+str(n):>12s}" for n in MESHES) + "   order(8->32)"
    print(header)
    for mode in MODES:
        for degree in DEGREES:
            cells = [r for r in payload["flux"][mode] if r["degree"] == degree]
            print(f"{mode:7s} {degree:3d} " + " ".join(f"{r['energy_error_bottom']:12.4e}" for r in cells)
                  + f"   {payload['summary'][mode][f'p{degree}']['observed_order']:.3f}")
    print()
    print(f"{'mode':7s} {'deg':>3s} " + " ".join(f"{'res@'+str(n):>12s}" for n in MESHES))
    for mode in MODES:
        for degree in DEGREES:
            cells = [r for r in payload["residual"][mode] if r["degree"] == degree]
            print(f"{mode:7s} {degree:3d} " + " ".join(f"{r['rhs_max']:12.4e}" for r in cells))
    print()
    for mode in MODES:
        for name, check in payload["assertions"][mode].items():
            print(f"  [{mode:6s}] {'PASS' if check['passed'] else 'FAIL'} {name}")
    return payload


if __name__ == "__main__":
    main()