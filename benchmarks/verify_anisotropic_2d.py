"""Anisotropic-cell verification of the 2-D compressible Euler DG solver.

Why this exists
---------------
Every convergence claim for ``tensordg/euler_dg_2d.py`` in the repository is
measured on **square** cells: ``validation/euler_dg_2d/design_order.json`` uses
``ncells = n^2`` on ``[-5,5]^2`` for ``p = 1, 2, 3`` and the limiter studies
(``validation/limiter_2d``) inherit the same square mesh.  Three things have
therefore never been falsified:

1. that the design order ``p + 1`` survives *stretched* cells (``dx != dy``),
   where the only mechanism that can silently mix the axes is the ``1/(2dx)`` /
   ``1/(2dy)`` scaling in ``_rhs`` -- an error that is invisible when
   ``dx == dy``;
2. that the per-axis ``spacing`` threading added to the troubled-cell test in
   mission m2549 is *load bearing*, i.e. that using the other axis' spacing
   would actually change the limiter decision;
3. that the m2549 scale consistency ("the flagged set does not grow under
   refinement at fixed ``M``") holds on stretched cells, where ``dy`` and
   ``dx`` differ by a factor of two.

The reference is the analytic isentropic vortex -- an exact solution of the 2-D
Euler equations advected across a periodic box -- so no manufactured source
term and no numerical reference is involved.

Blocks
------
``design_order``      ``p = 1, 2`` on a three-rung 2:1 stretched ladder, with
                      the isotropic ladder of the same ``n_x`` as a control
                      (so a low EOC can be blamed on the pre-asymptotic window
                      rather than on anisotropy) and the transposed ladder
                      (``x``/``y`` roles exchanged) as an invariance check.
``limiter_scale``     the window of TVB constants ``M`` in which a smooth
                      y-profile flips its verdict between the correct spacing
                      ``dy`` and the wrong one ``dx`` (mutation probe), plus the
                      flagged-cell count per axis on a refinement ladder.
``limiter_on_smooth`` limiter-on convergence on the smooth vortex, measured
                      against the no-limiter ladder of the same meshes.
``operator_transpose``exact algebraic ``x <-> y`` invariance of one residual
                      evaluation.
``conservation``      periodic mass/energy drift on stretched meshes.

Run::

    PYTHONPATH=. python -m benchmarks.verify_anisotropic_2d
    PYTHONPATH=. python -m benchmarks.verify_anisotropic_2d --quick

Numbers go to ``validation/anisotropic_2d/verify.json`` (rewritten after every
block, so an interrupted run still leaves usable evidence); the report with
every digit injected from that JSON is produced by
``benchmarks.report_anisotropic_2d``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.verify_euler_dg_2d import _vortex_error
from tensordg.tvb import minmod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "anisotropic_2d" / "verify.json"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
FINAL_TIME = 1.0
CFL = 0.2
TVB = 1.0

# 2:1 stretched ladders: dy = dx / 2 (fine in y).  The isotropic control uses
# the same n_x, so the two ladders share the coarse-direction spacing.
ANISO_MESHES = ((8, 16), (12, 24), (16, 32))
ISO_MESHES = ((8, 8), (12, 12), (16, 16))
MIRROR_MESHES = ((16, 8), (24, 12), (32, 16))
DEGREES = (1, 2)


def _slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def _aspect(nx, ny):
    return (10.0 / nx) / (10.0 / ny)


def _h(nx, ny):
    return math.sqrt((10.0 / nx) * (10.0 / ny))


def _save(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))


def _operators(degree):
    return dg2.legendre_operators(degree, max(degree + 2, 3))


def _vortex_row(degree, nx, ny, limiter=None):
    nodes, weights, basis, _, _ = _operators(degree)
    started = time.time()
    _, _, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=(nx, ny),
        degree=degree,
        final_time=FINAL_TIME,
        cfl=CFL,
        limiter=limiter,
        tvb_constant=TVB,
        domain=DOMAIN,
    )
    errors = _vortex_error(coeff, basis, nodes, DOMAIN, FINAL_TIME)
    row = {
        "degree": degree,
        "cells": [nx, ny],
        "aspect": _aspect(nx, ny),
        "h": _h(nx, ny),
        "dofs": meta["dofs"],
        "steps": meta["steps"],
        "dt_last": meta["dt"],
        "limiter": limiter or "none",
        "errors": {field: errors[field] for field in FIELDS},
        "wall_seconds": time.time() - started,
    }
    return row


def _with_eoc(rows):
    for previous, current in zip(rows, rows[1:]):
        current["eoc"] = {
            field: math.log(previous["errors"][field] / current["errors"][field])
            / math.log(previous["h"] / current["h"])
            for field in FIELDS
        }
    if len(rows) >= 2:
        rows[-1]["eoc_lsq"] = {
            field: _slope([row["h"] for row in rows], [row["errors"][field] for row in rows])
            for field in FIELDS
        }
    return rows


def design_order(payload, degrees, meshes, iso_meshes, mirror_meshes, log):
    blocks = []
    for degree in degrees:
        groups = {}
        for label, ladder in (
            ("anisotropic", meshes),
            ("isotropic_control", iso_meshes),
            ("transposed_control", mirror_meshes),
        ):
            rows = []
            for nx, ny in ladder:
                row = _vortex_row(degree, nx, ny)
                rows.append(row)
                log(f"  {label[:9]:9s} P{degree} {nx:2d}x{ny:2d}: "
                    f"rho {row['errors']['density']:.3e} p {row['errors']['pressure']:.3e} "
                    f"({row['wall_seconds']:.1f}s, {row['steps']} steps)")
                groups[label] = _with_eoc(rows)
                payload["design_order"] = blocks + [
                    {"kind": "anisotropic", "degree": degree,
                     "design_order": degree + 1, "groups": groups}
                ]
                _save(payload)

        transpose = []
        for row, mirror in zip(groups["anisotropic"], groups["transposed_control"]):
            transpose.append(
                {
                    "cells": row["cells"],
                    "transposed_cells": mirror["cells"],
                    "density_rel_diff": abs(
                        row["errors"]["density"] - mirror["errors"]["density"]
                    ) / row["errors"]["density"],
                    "velocity_swap_rel_diff": abs(
                        row["errors"]["velocity_x"] - mirror["errors"]["velocity_y"]
                    ) / row["errors"]["velocity_x"],
                    "steps": [row["steps"], mirror["steps"]],
                }
            )
        groups["transpose_invariance"] = transpose
        blocks.append(
            {"kind": "anisotropic", "degree": degree,
             "design_order": degree + 1, "groups": groups}
        )
        payload["design_order"] = blocks
        _save(payload)
    return blocks


def _flagged_counts(coeff, dx, dy, periodic=(True, True), tvb_constant=TVB):
    """Cells flagged by the troubled-cell test along each axis."""
    averages = coeff[..., 0, 0, :]
    counts = {}
    for name, axis, spacing in (("x", 0, dx), ("y", 1, dy)):
        troubled, _ = dg2._troubled_1d(averages, axis, spacing, tvb_constant, periodic)
        counts[name] = int(troubled.any(dim=-1).sum())
    return counts


def limiter_scale(payload, log):
    """Is the per-axis spacing in the troubled-cell test load bearing?

    A smooth y-profile ``u = 1 + A cos(2 pi y / L)`` on a stretched mesh
    (``dx = 2 dy``) has a cell-average deviation
    ``|d_left - minmod(d_left, d_right)|`` of order ``|u''| dy^2``.  That
    deviation belongs to the data, so the two candidate tolerances
    ``M dy^2`` (correct axis) and ``M dx^2 = 4 M dy^2`` (wrong axis) bracket a
    window of ``M`` in which the verdict flips.  The amplitude is calibrated so
    the demonstration sits inside that window; the window itself is reported, so
    the claim does not rest on the calibration.
    """
    nx, ny = 8, 64                      # dx = 0.3125, dy = 0.15625
    dx, dy = 10.0 / nx, 10.0 / ny
    period = 5.0                        # cos(2 pi (y - 5) / 5)
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    shape = torch.cos(2.0 * math.pi * (y - 5.0) / period)

    def deviation(amplitude):
        """max_cells |d_left - minmod(d_left, d_right)| -- the TVB measure."""
        averages = (1.0 + amplitude * shape).view(1, ny, 1).expand(nx, ny, 1).clone()
        d_left = averages[:, 1:-1, 0] - averages[:, :-2, 0]
        d_right = averages[:, 2:, 0] - averages[:, 1:-1, 0]
        return float((d_left - minmod(d_left, d_right)).abs().max())

    probe_amplitude = 1e-4
    target = 2.0 * TVB * dy * dy
    amplitude = probe_amplitude * target / deviation(probe_amplitude)

    averages = (1.0 + amplitude * shape).view(1, ny, 1).expand(nx, ny, 1).clone()
    correct, _ = dg2._troubled_1d(averages, axis=1, spacing=dy, tvb_constant=TVB, periodic=True)
    wrong, _ = dg2._troubled_1d(averages, axis=1, spacing=dx, tvb_constant=TVB, periodic=True)
    deviation_max = deviation(amplitude)

    # Scale consistency on stretched cells: the flagged set must not grow as
    # the mesh is refined at fixed M (this is exactly what the pre-m2549
    # criterion violated).
    ladder = []
    for cells in ((8, 16), (12, 24), (16, 32)):
        degree = 2
        _, _, _, _, _ = _operators(degree)
        _, _, coeff, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=cells, degree=degree, final_time=FINAL_TIME, cfl=CFL,
            limiter=None, tvb_constant=TVB, domain=DOMAIN,
        )
        ddx, ddy = 10.0 / cells[0], 10.0 / cells[1]
        counts = _flagged_counts(coeff, ddx, ddy)
        limited = dg2.apply_tvb_limiter_2d(coeff, ddx, ddy, TVB, (True, True))
        modified = int((limited - coeff).abs().gt(0.0).any(dim=(2, 3, 4)).sum())
        cell_count = cells[0] * cells[1]
        ladder.append(
            {
                "cells": list(cells),
                "h": _h(*cells),
                "cells_total": cell_count,
                "flagged_x": counts["x"],
                "flagged_y": counts["y"],
                "flagged_fraction_y": counts["y"] / cell_count,
                "limiter_modified": modified,
                "limiter_modified_fraction": modified / cell_count,
            }
        )
        log(f"  flagged cells P2 {cells[0]}x{cells[1]}: x {counts['x']}, y {counts['y']} "
            f"({counts['y'] / cell_count:.1%} of cells); limiter modified {modified}")

    payload["limiter_scale"] = {
        "mutation_probe": {
            "mesh": {"nx": nx, "ny": ny, "dx": dx, "dy": dy, "aspect": dx / dy},
            "amplitude": amplitude,
            "deviation_max": deviation_max,
            "tolerance_correct_axis": TVB * dy * dy,
            "tolerance_wrong_axis": TVB * dx * dx,
            "discriminating_M_window": [
                deviation_max / (dx * dx), deviation_max / (dy * dy)
            ],
            "troubled_cells_correct_axis": int(correct.sum()),
            "troubled_cells_wrong_axis": int(wrong.sum()),
            "verdict_flips": int(correct.sum()) != int(wrong.sum()),
        },
        "refinement_ladder": ladder,
    }
    _save(payload)
    return payload["limiter_scale"]


def limiter_on_smooth(payload, degrees, meshes, iso_meshes, log):
    """Limiter-on convergence on the smooth vortex (policy measurement)."""
    rows = []
    for degree in degrees:
        for label, ladder in (("anisotropic", meshes), ("isotropic_control", iso_meshes)):
            per_limiter = {}
            for limiter in ("tvb", "weno"):
                group = []
                for nx, ny in ladder:
                    row = _vortex_row(degree, nx, ny, limiter=limiter)
                    group.append(row)
                    log(f"  {limiter:4s} {label[:9]:9s} P{degree} {nx:2d}x{ny:2d}: "
                        f"rho {row['errors']['density']:.3e} ({row['wall_seconds']:.1f}s)")
                    _save(payload)
                per_limiter[limiter] = _with_eoc(group)
            rows.append({"degree": degree, "ladder": label, "runs": per_limiter})
    payload["limiter_on_smooth"] = rows
    _save(payload)
    return rows


def operator_transpose(payload, log):
    """Exact algebraic ``x <-> y`` invariance of one residual evaluation."""
    rows = []
    for degree in (1, 2, 3):
        nodes, weights, basis, derivative, signs = _operators(degree)
        nx, ny = 6, 11
        generator = torch.Generator().manual_seed(20261104 + degree)
        coeff = 0.05 * torch.randn(
            nx, ny, degree + 1, degree + 1, 4, dtype=torch.float64, generator=generator
        )
        coeff[..., 0, 0, 0] += 1.0
        coeff[..., 0, 0, 3] += 2.5
        left = dg2._rhs(coeff, basis, derivative, weights, signs,
                        10.0 / nx, 10.0 / ny, (True, True))
        swapped = coeff.permute(1, 0, 3, 2, 4)[..., [0, 2, 1, 3]].contiguous()
        right = dg2._rhs(swapped, basis, derivative, weights, signs,
                         10.0 / ny, 10.0 / nx, (True, True))
        right = right.permute(1, 0, 3, 2, 4)[..., [0, 2, 1, 3]]
        rows.append(
            {
                "degree": degree,
                "relative_error": float((left - right).abs().max())
                / max(float(left.abs().max()), 1e-300),
            }
        )
        log(f"  operator transpose P{degree}: rel {rows[-1]['relative_error']:.2e}")
    payload["operator_transpose"] = rows
    _save(payload)
    return rows


def conservation(payload, log):
    """Periodic mass/energy budgets on stretched meshes (round-off expected)."""
    rows = []
    for cells in ((8, 16), (12, 24)):
        _, _, coeff0, _ = dg2.solve_euler_dg_2d_vortex(
            ncells=cells, degree=2, final_time=0.0, cfl=CFL, domain=DOMAIN
        )
        _, _, coeff1, meta = dg2.solve_euler_dg_2d_vortex(
            ncells=cells, degree=2, final_time=FINAL_TIME, cfl=CFL, domain=DOMAIN
        )
        dx, dy = 10.0 / cells[0], 10.0 / cells[1]
        mass0 = float(coeff0[..., 0, 0, 0].sum()) * dx * dy
        mass1 = float(coeff1[..., 0, 0, 0].sum()) * dx * dy
        energy0 = float(coeff0[..., 0, 0, 3].sum()) * dx * dy
        energy1 = float(coeff1[..., 0, 0, 3].sum()) * dx * dy
        row = {
            "cells": list(cells),
            "steps": meta["steps"],
            "mass_drift": abs(mass1 - mass0) / abs(mass0),
            "energy_drift": abs(energy1 - energy0) / abs(energy0),
        }
        rows.append(row)
        log(f"  conservation {cells[0]}x{cells[1]}: mass {row['mass_drift']:.2e}, "
            f"energy {row['energy_drift']:.2e}")
    payload["conservation"] = rows
    _save(payload)
    return rows


def criteria(payload):
    """Falsifiable verdicts; thresholds fixed before the numbers are seen."""
    checks = []

    def add(name, measured, threshold, operator, what):
        checks.append(
            {"name": name, "measured": measured, "threshold": threshold,
             "operator": operator, "what": what}
        )

    for block in payload["design_order"]:
        degree, order = block["degree"], block["design_order"]
        groups = block["groups"]
        for label in ("anisotropic", "isotropic_control", "transposed_control"):
            lsq = groups[label][-1].get("eoc_lsq", {})
            add(f"P{degree}_{label}_lsq_order",
                min(lsq.get(field, 0.0) for field in FIELDS),
                order - 0.3, ">=",
                f"worst least-squares EOC over (rho,u,v,p), {label} ladder")
        add(f"P{degree}_anisotropic_finest_3pct",
            max(groups["anisotropic"][-1]["errors"][field] for field in FIELDS) * 100.0,
            3.0, "<=",
            "worst relative L2 error on the finest stretched mesh, percent")
        for pair in groups["transpose_invariance"]:
            add(f"P{degree}_transpose_density_{pair['cells'][0]}x{pair['cells'][1]}",
                pair["density_rel_diff"], 0.05, "<=",
                "density error of the stretched ladder vs its transposed twin")

    probe = payload["limiter_scale"]["mutation_probe"]
    add("limiter_wrong_axis_verdict_differs",
        1.0 if probe["verdict_flips"] else 0.0, 1.0, ">=",
        "troubled-cell verdict must depend on the per-axis spacing (mutation probe)")

    ladder = payload["limiter_scale"]["refinement_ladder"]
    add("limiter_flagged_fraction_not_growing",
        ladder[-1]["flagged_fraction_y"], ladder[0]["flagged_fraction_y"], "<=",
        "flagged fraction on the fine axis must not grow under refinement at fixed M "
        "(m2549 scale consistency on stretched cells)")
    add("limiter_flagged_fraction_bounded",
        ladder[-1]["flagged_fraction_y"], 0.5, "<=",
        "at M = 1 the flagged fraction on a resolved smooth flow stays a minority")

    for row in payload.get("limiter_on_smooth", []):
        degree = row["degree"]
        for limiter, group in row["runs"].items():
            lsq = group[-1].get("eoc_lsq", {})
            add(f"limiter_{limiter}_{row['ladder']}_P{degree}_order",
                min(lsq.get(field, 0.0) for field in FIELDS),
                degree + 1 - 0.5, ">=",
                f"worst EOC with the {limiter} limiter active on the smooth vortex "
                f"({row['ladder']}) -- the limiter must not destroy the smooth order")

    for row in payload["operator_transpose"]:
        add(f"operator_transpose_P{row['degree']}", row["relative_error"], 1e-14, "<=",
            "residual invariance under x<->y with u<->v (exact algebraic symmetry)")

    for row in payload["conservation"]:
        cells = f"{row['cells'][0]}x{row['cells'][1]}"
        add(f"conservation_mass_{cells}", row["mass_drift"], 1e-12, "<=",
            "relative mass drift on a periodic stretched mesh (round-off)")
        add(f"conservation_energy_{cells}", row["energy_drift"], 1e-12, "<=",
            "relative energy drift on a periodic stretched mesh (round-off)")

    for check in checks:
        if check["operator"] == ">=":
            check["passed"] = bool(check["measured"] >= check["threshold"])
        else:
            check["passed"] = bool(check["measured"] <= check["threshold"])
    payload["criteria"] = checks
    payload["summary"] = {
        "checks": len(checks),
        "passed": sum(1 for check in checks if check["passed"]),
        "failed": [check["name"] for check in checks if not check["passed"]],
        "overall": "PASS" if all(check["passed"] for check in checks) else "FAIL",
    }
    _save(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two-mesh ladders only")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--degrees", type=int, nargs="+", default=list(DEGREES))
    parser.add_argument("--skip-limiter-on-smooth", action="store_true")
    parser.add_argument("--blocks", nargs="+", default=None,
                        help="re-run only these blocks against the existing JSON "
                             "(design_order limiter_scale limiter_on_smooth "
                             "operator_transpose conservation)")
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)

    meshes = ANISO_MESHES[:2] if args.quick else ANISO_MESHES
    iso = ISO_MESHES[:2] if args.quick else ISO_MESHES
    mirror = MIRROR_MESHES[:2] if args.quick else MIRROR_MESHES

    existing = OUT.exists() and args.blocks is not None
    payload = json.loads(OUT.read_text()) if existing else {
        "benchmark": "euler_dg_2d_anisotropic",
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "domain": DOMAIN,
            "final_time": FINAL_TIME,
            "cfl": CFL,
            "tvb_constant": TVB,
            "anisotropic_meshes": meshes,
            "isotropic_meshes": iso,
            "transposed_meshes": mirror,
            "degrees": args.degrees,
            "threads": args.threads,
        },
    }
    log = lambda message: print(message, flush=True)

    want = set(args.blocks) if args.blocks else {
        "design_order", "limiter_scale", "limiter_on_smooth",
        "operator_transpose", "conservation",
    }
    if args.skip_limiter_on_smooth:
        want.discard("limiter_on_smooth")
    if "design_order" in want:
        log("[design order on stretched cells]")
        design_order(payload, args.degrees, meshes, iso, mirror, log)
    if "limiter_scale" in want:
        log("[per-axis limiter spacing]")
        limiter_scale(payload, log)
    if "limiter_on_smooth" in want:
        log("[limiter on the smooth vortex]")
        limiter_on_smooth(payload, args.degrees, meshes, iso, log)
    if "operator_transpose" in want:
        log("[operator transpose invariance]")
        operator_transpose(payload, log)
    if "conservation" in want:
        log("[conservation on stretched cells]")
        conservation(payload, log)
    payload.setdefault("config", {})["last_blocks"] = sorted(want)
    payload["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    criteria(payload)
    log(f"summary: {payload['summary']['passed']}/{payload['summary']['checks']} "
        f"passed -> {payload['summary']['overall']}")
    if payload["summary"]["failed"]:
        log("failed: " + ", ".join(payload["summary"]["failed"]))
    return payload["summary"]["overall"] == "PASS"


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)