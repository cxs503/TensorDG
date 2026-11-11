"""Adjudication of the 2-D WENO limiter on a smooth flow (m14921).

Why this exists
---------------
``docs/anisotropic_2d_verification.md`` records eight honest FAIL criteria: with
``limiter="tvb"`` or ``"weno"`` active at the default ``tvb_constant = 1`` the
isentropic vortex does **not** converge at design order (EOC 0.06 .. 0.46 for
P1, -0.61 .. 0.16 for P2).  ``docs/tvb_constant_adjudication.md`` adjudicated the
TVB half of that record: the criterion is scale consistent, ``M`` is a
*data-dependent* absolute bound on the cell-average second derivative, so at
``M = 1`` the limiter legitimately flags resolved smooth cells.  That
adjudication explicitly stopped short of saying the same for ``weno``
("未覆盖 ``limiter="weno"`` 的同类裁决"), and mission m14777 (WENO mixed-mode
write defect) left two items honestly open:

* ``not_run``: the order of a *forced* limited representation was never
  measured ("forcing the limiter and checking O(h^2)");
* the cause of the limiter-on error plateau was left open.

This benchmark closes both, for the WENO path, with the same analytic reference
(isentropic vortex = exact solution of the 2-D Euler equations on a periodic
box, so no manufactured source and no numerical reference):

``sweep``     limiter-on error, flagged fraction and EOC vs ``M`` on iso and 2:1
              stretched ladders, P1 and P2, against the *unlimited* baseline of
              the same meshes.  Falsifiable prediction of the "policy, not a
              WENO-specific defect" hypothesis: as soon as ``M`` exceeds the
              data's second-derivative bound the limiter is *inert* and the
              limiter-on trajectory becomes **bit-identical** to the unlimited
              one, so the design order returns exactly.
``forced``    the missing ``not_run`` measurement: project the exact vortex
              initial state, apply the WENO limiter once with ``M = 0`` (every
              cell with a non-vanishing Cockburn-Shu deviation flagged), and
              measure the L2 error of the *limited representation itself*.  A
              limiter that falls back to a linear reconstruction in flagged
              cells can only deliver second order, so EOC -> 2 is the ceiling;
              measuring it turns "the limiter destroys the smooth order" into a
              quantitative statement about what a limited representation can
              represent at all.
``cross``     TVB vs WENO at identical ``M`` on the same mesh: the flagged sets
              share one indicator, so a plateau shared by both limiters is a
              *policy* property, not a WENO implementation defect.

Run::

    PYTHONPATH=. python -m benchmarks.weno_limiter_smooth_vortex            # full
    PYTHONPATH=. python -m benchmarks.weno_limiter_smooth_vortex --quick    # smoke

Numbers go to ``validation/weno_limiter_policy/results.json``; the report with
every digit injected from it is ``docs/weno_limiter_smooth_flow_policy.md``.
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
from tensordg.euler_2d import vortex_primitive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation" / "weno_limiter_policy" / "results.json"

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
FINAL_TIME = 1.0
CFL = 0.2

ISO_MESHES = ((8, 8), (12, 12), (16, 16))
ANISO_MESHES = ((8, 16), (12, 24), (16, 32))
M_VALUES = (1.0, 2.0, 10.0, 1.0e6)
FORCED_MESHES = ((8, 8), (16, 16), (32, 32), (64, 64))


def _slope(xs, ys):
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


def _h(nx, ny):
    return math.sqrt((10.0 / nx) * (10.0 / ny))


def _save(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))


def _operators(degree):
    return dg2.legendre_operators(degree, max(degree + 2, 3))


def _flag_stats(coeff, dx, dy, m, periodic=(True, True)):
    """Flagged fraction of the final modal state, per axis and on either axis.

    Uses the shipping troubled-cell indicator (``dg2._troubled_1d``) on the
    cell averages, i.e. the same decision the limiter itself made.
    """
    averages = coeff[..., 0, 0, :]
    troubled_x, _ = dg2._troubled_1d(averages, 0, dx, m, periodic[0])
    troubled_y, _ = dg2._troubled_1d(averages, 1, dy, m, periodic[1])
    either = troubled_x | troubled_y
    total = either.numel()
    return {
        "flagged_x": int(troubled_x.sum().item()),
        "flagged_y": int(troubled_y.sum().item()),
        "flagged_either": int(either.sum().item()),
        "flagged_fraction_x": float(troubled_x.sum().item()) / total,
        "flagged_fraction_y": float(troubled_y.sum().item()) / total,
        "flagged_fraction_either": float(either.sum().item()) / total,
    }


def _vortex_run(degree, nx, ny, limiter, m):
    nodes, weights, basis, _, _ = _operators(degree)
    started = time.time()
    _, _, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=(nx, ny),
        degree=degree,
        final_time=FINAL_TIME,
        cfl=CFL,
        limiter=limiter,
        tvb_constant=m,
        domain=DOMAIN,
    )
    errors = _vortex_error(coeff, basis, nodes, DOMAIN, FINAL_TIME)
    dx, dy = 10.0 / nx, 10.0 / ny
    row = {
        "degree": degree,
        "cells": [nx, ny],
        "h": _h(nx, ny),
        "limiter": limiter or "none",
        "tvb_constant": m,
        "steps": int(meta["steps"]),
        "errors": {field: errors[field] for field in FIELDS},
        "wall_seconds": time.time() - started,
    }
    row.update(_flag_stats(coeff, dx, dy, m))
    return row, coeff, errors


def block_sweep(quick=False):
    meshes = {"isotropic": ISO_MESHES, "anisotropic": ANISO_MESHES}
    degrees = (1, 2)
    m_values = (1.0, 1.0e6) if quick else M_VALUES
    if quick:
        meshes = {"isotropic": ISO_MESHES[:2], "anisotropic": ANISO_MESHES[:2]}
    rows = []
    for ladder, mesh_set in meshes.items():
        for degree in degrees:
            unlimited = {}
            for nx, ny in mesh_set:
                row, coeff, _ = _vortex_run(degree, nx, ny, None, 1.0)
                row["ladder"] = ladder
                row["bitwise_identical_to_unlimited"] = None
                unlimited[(nx, ny)] = (coeff, row)
            for m in m_values:
                for nx, ny in mesh_set:
                    row, coeff, _ = _vortex_run(degree, nx, ny, "weno", m)
                    row["ladder"] = ladder
                    ref_coeff, ref_row = unlimited[(nx, ny)]
                    row["unlimited_errors"] = ref_row["errors"]
                    row["error_ratio_vs_unlimited"] = {
                        field: row["errors"][field] / ref_row["errors"][field]
                        for field in FIELDS
                    }
                    row["bitwise_identical_to_unlimited"] = bool(
                        torch.equal(coeff, ref_coeff)
                    )
                    rows.append(row)
                    print(
                        f"[sweep] {ladder:11s} p{degree} {nx}x{ny} M={m:g} "
                        f"rho={row['errors']['density']:.4e} "
                        f"ratio={row['error_ratio_vs_unlimited']['density']:.4g} "
                        f"flag={row['flagged_fraction_either']:.3f} "
                        f"bitwise={row['bitwise_identical_to_unlimited']} "
                        f"{row['wall_seconds']:.1f}s",
                        flush=True,
                    )
            for nx, ny in mesh_set:
                row = unlimited[(nx, ny)][1]
                rows.append(row)
    # per-M order estimates, computed from the rows above
    orders = []
    for ladder in meshes:
        for degree in degrees:
            for m in m_values:
                ladder_rows = [
                    r for r in rows
                    if r.get("ladder") == ladder and r["degree"] == degree
                    and r["limiter"] == "weno" and r["tvb_constant"] == m
                ]
                ladder_rows.sort(key=lambda r: r["cells"][0])
                if len(ladder_rows) < 2:
                    continue
                hs = [r["h"] for r in ladder_rows]
                entry = {"ladder": ladder, "degree": degree, "tvb_constant": m}
                for field in FIELDS:
                    entry[f"eoc_{field}"] = _slope(
                        hs, [r["errors"][field] for r in ladder_rows]
                    )
                    entry[f"worst_ratio_{field}"] = max(
                        r["error_ratio_vs_unlimited"][field] for r in ladder_rows
                    )
                entry["design_order"] = degree + 1
                entry["bitwise_identical_all"] = all(
                    r["bitwise_identical_to_unlimited"] for r in ladder_rows
                )
                orders.append(entry)
    return rows, orders


def _project_vortex(degree, nx, ny):
    nodes, weights, basis, _, _ = _operators(degree)
    dx, dy = 10.0 / nx, 10.0 / ny
    x = -5.0 + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = -5.0 + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = dg2._initial_coefficients(
        lambda xq, yq: vortex_primitive(xq, yq, 0.0, 5.0, (1.0, 1.0)),
        x, y, dx, dy, nodes, weights, basis,
        "cpu", torch.float64,
    )
    return coeff, nodes, basis, dx, dy


def block_forced(quick=False):
    meshes = FORCED_MESHES[:3] if quick else FORCED_MESHES
    rows = []
    for degree in (1, 2, 3):
        for nx, ny in meshes:
            coeff, nodes, basis, dx, dy = _project_vortex(degree, nx, ny)
            unlimited_error = _vortex_error(coeff, basis, nodes, DOMAIN, 0.0)
            limited = dg2.apply_weno_limiter_2d(coeff, dx, dy, 0.0, (True, True))
            limited_error = _vortex_error(limited, basis, nodes, DOMAIN, 0.0)
            mode0_changed = int(
                (coeff[..., 0, 0, :] != limited[..., 0, 0, :]).sum().item()
            )
            row = {
                "degree": degree,
                "cells": [nx, ny],
                "h": _h(nx, ny),
                "unlimited_errors": {f: unlimited_error[f] for f in FIELDS},
                "limited_errors": {f: limited_error[f] for f in FIELDS},
                "mode0_changed_components": mode0_changed,
            }
            row.update(_flag_stats(coeff, dx, dy, 0.0))
            rows.append(row)
            print(
                f"[forced] p{degree} {nx}x{ny} unlimited={unlimited_error['density']:.3e} "
                f"limited={limited_error['density']:.3e} "
                f"flag={row['flagged_fraction_either']:.3f}",
                flush=True,
            )
    orders = []
    for degree in (1, 2, 3):
        group = sorted(
            [r for r in rows if r["degree"] == degree], key=lambda r: r["cells"][0]
        )
        if len(group) < 2:
            continue
        hs = [r["h"] for r in group]
        entry = {"degree": degree}
        for field in FIELDS:
            entry[f"eoc_limited_{field}"] = _slope(
                hs, [r["limited_errors"][field] for r in group]
            )
            entry[f"eoc_unlimited_{field}"] = _slope(
                hs, [r["unlimited_errors"][field] for r in group]
            )
        entry["worst_last_ratio"] = {
            field: group[-1]["limited_errors"][field]
            / group[-1]["unlimited_errors"][field]
            for field in FIELDS
        }
        orders.append(entry)
    return rows, orders


def block_cross(quick=False):
    meshes = ((8, 8), (12, 12)) if quick else ((8, 8), (16, 16))
    m_values = (1.0, 10.0) if quick else (1.0, 10.0)
    rows = []
    for degree in (2,):
        for m in m_values:
            for limiter in ("tvb", "weno"):
                for nx, ny in meshes:
                    row, _, _ = _vortex_run(degree, nx, ny, limiter, m)
                    row["ladder"] = "isotropic"
                    rows.append(row)
                    print(
                        f"[cross] p{degree} {nx}x{ny} M={m:g} {limiter} "
                        f"rho={row['errors']['density']:.4e} "
                        f"flag={row['flagged_fraction_either']:.3f}",
                        flush=True,
                    )
    shared = []
    for m in m_values:
        for nx, ny in meshes:
            tvb = [r for r in rows if r["limiter"] == "tvb"
                   and r["tvb_constant"] == m and r["cells"] == [nx, ny]]
            weno = [r for r in rows if r["limiter"] == "weno"
                    and r["tvb_constant"] == m and r["cells"] == [nx, ny]]
            if tvb and weno:
                shared.append({
                    "cells": [nx, ny],
                    "tvb_constant": m,
                    "flagged_identical": tvb[0]["flagged_either"]
                    == weno[0]["flagged_either"],
                    "flagged_either": tvb[0]["flagged_either"],
                    "tvb_ratio_error_density": None,  # filled by the report from rows
                    "error_density_tvb": tvb[0]["errors"]["density"],
                    "error_density_weno": weno[0]["errors"]["density"],
                })
    return rows, shared


def _criteria(sweep_rows, orders, forced_rows, forced_orders):
    criteria = {}

    def add(key, value, ok, criterion, note=""):
        criteria[key] = {
            "value": value,
            "criterion": criterion,
            "verdict": "PASS" if ok else "FAIL",
            "note": note,
        }

    inert = [o for o in orders if o["tvb_constant"] >= 1.0e6]
    add(
        "inert_M_bitwise_matches_unlimited",
        all(o["bitwise_identical_all"] for o in inert) if inert else False,
        bool(inert) and all(o["bitwise_identical_all"] for o in inert),
        "at M=1e6 the WENO-limiter trajectory is bit-identical to the unlimited one",
    )
    flagged = [r for r in sweep_rows if r["limiter"] == "weno"]
    add(
        "flagged_fraction_bounded_at_M1",
        max(r["flagged_fraction_either"] for r in flagged
            if r["tvb_constant"] == 1.0),
        max(r["flagged_fraction_either"] for r in flagged
            if r["tvb_constant"] == 1.0) <= 0.5,
        "at M=1 the flagged fraction on a resolved smooth vortex is a minority",
    )
    monotone = True
    for ladder in {r["ladder"] for r in flagged}:
        for degree in {r["degree"] for r in flagged}:
            for cells in {tuple(r["cells"]) for r in flagged}:
                series = sorted(
                    [r for r in flagged if r["ladder"] == ladder
                     and r["degree"] == degree and tuple(r["cells"]) == cells],
                    key=lambda r: r["tvb_constant"],
                )
                counts = [r["flagged_either"] for r in series]
                monotone = monotone and all(
                    a >= b for a, b in zip(counts, counts[1:])
                )
    add(
        "flagged_count_monotone_in_M", monotone, monotone,
        "flagged cell count is non-increasing in M on every mesh/degree",
    )
    high = [o for o in orders if o["tvb_constant"] >= 10.0 and o["tvb_constant"] < 1.0e6]
    recovered = all(
        o["eoc_density"] >= o["design_order"] * 0.75 for o in high
    ) if high else False
    add(
        "design_order_recovered_when_M_above_data_bound",
        {f"{o['ladder']}-p{o['degree']}-M{o['tvb_constant']:g}": round(o["eoc_density"], 3)
         for o in high},
        recovered,
        "with M >= 10 the density EOC returns to >= 75% of the design order",
    )
    worst_forced = max(
        o[f"eoc_limited_{f}"] for o in forced_orders for f in FIELDS
    )
    add(
        "forced_limiter_representation_order",
        {f"p{o['degree']}": {f: round(o[f"eoc_limited_{f}"], 3) for f in FIELDS}
         for o in forced_orders},
        all(o[f"eoc_limited_{f}"] >= 1.9 for o in forced_orders for f in FIELDS),
        "a forced (M=0) WENO-limited representation converges at >= 1.9 in every field",
        f"worst measured EOC = {worst_forced:.3f}",
    )
    add(
        "forced_limiter_preserves_cell_average",
        max(r["mode0_changed_components"] for r in forced_rows),
        max(r["mode0_changed_components"] for r in forced_rows) == 0,
        "the limiter never changes a cell average (exact conservation)",
    )
    add(
        "forced_limiter_flags_all_cells",
        min(r["flagged_fraction_either"] for r in forced_rows),
        min(r["flagged_fraction_either"] for r in forced_rows) == 1.0,
        "M=0 flags every cell of a resolved smooth vortex",
    )
    return criteria


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    sweep_rows, orders = block_sweep(args.quick)
    forced_rows, forced_orders = block_forced(args.quick)
    cross_rows, cross_shared = block_cross(args.quick)
    criteria = _criteria(sweep_rows, orders, forced_rows, forced_orders)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "domain": DOMAIN,
            "final_time": FINAL_TIME,
            "cfl": CFL,
            "iso_meshes": ISO_MESHES,
            "aniso_meshes": ANISO_MESHES,
            "m_values": M_VALUES,
            "forced_meshes": FORCED_MESHES,
            "quick": args.quick,
            "threads": args.threads,
        },
        "sweep": sweep_rows,
        "orders": orders,
        "forced": forced_rows,
        "forced_orders": forced_orders,
        "cross": cross_rows,
        "cross_shared": cross_shared,
        "criteria": criteria,
        "wall_seconds": time.time() - started,
    }
    _save(payload)
    print(json.dumps({k: v["verdict"] for k, v in criteria.items()}, indent=1))
    print(f"wall {payload['wall_seconds']:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()