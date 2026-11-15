"""Embedded-wall vs boundary-wall parity for the CKS1998 shock-reflection gate.

The Woodward--Colella / CKS1998 forward-facing step needs *interior* solid walls,
which ``tensordg.euler_dg_2d`` implements with a boolean ``solid`` cell mask
(``docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md``).  That document leaves gate G3
("a shock reflecting off an embedded wall matches the exact two-shock solution")
at **WITHHELD/FAIL** because the plateau behind the reflected shock missed the
closed-form value, and it lists two suspects that the earlier runs could not
separate:

1. the embedded-wall numerical flux is wrong, or
2. P1/P2 DG simply smears the two shocks on the tested meshes.

This module settles the question with a *matched* A/B: the same shock
reflection, the same cell size ``dx``, the same limiter, the same wall position,
run twice --

* ``solid``    : domain ``[0, 1.2]``, cells ``i >= i_wall`` frozen as solid;
* ``boundary`` : domain truncated at the wall ``[0, 1.0]``, ``bcs right="wall"``.

``nx_boundary = nx_solid * x_wall / xmax`` makes the two meshes share ``dx``
exactly, so cell ``i`` occupies the same interval in both arms and the cell
averages are directly comparable.  Any difference between the arms is then
attributable to the wall treatment alone; any error *common* to both arms is
shock capturing error.

Outputs (``--outdir``): ``parity.json`` with every number below and
``parity.png`` with the profiles, the error-vs-h ladder and the arm difference.

Usage::

    python -m benchmarks.cks1998_embedded_wall_parity                  # default ladder
    python -m benchmarks.cks1998_embedded_wall_parity --resolutions 120 240
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.euler_dg_2d import solve_euler_dg_2d

from benchmarks.forward_facing_step import (
    WallReflection,
    _plateau_mask,
    _pressure,
    _snap_to_face,
    wall_solid_mask,
)

XMAX = 1.2        # solid arm: [0, 1.2]; the boundary arm ends at the wall
X_WALL = 1.0
X_SHOCK = 0.4
YMAX = 0.1
MACH = 2.0
CFL = 0.12
# Time marched after the incident shock reaches the wall.  The gate default
# (0.08) leaves the reflected plateau only ``|u_reflected| * 0.08 / dx`` cells
# wide -- fewer than the 4 buffer cells below nx = 200, so every coarse run
# would report ``not_run``.  0.15 leaves >= 4 samples from nx = 120 up and is
# the only difference from the gate configuration.
EXTRA_TIME = 0.15
TVB = 1.0


def _initial(wr, x_shock_face):
    pre_rho, pre_u, pre_p = wr.pre
    post_rho, post_u, post_p = wr.post

    def initial(x, y):
        ahead = x > x_shock_face
        return (
            torch.where(ahead, torch.full_like(x, pre_rho), torch.full_like(x, post_rho)),
            torch.where(ahead, torch.full_like(x, pre_u), torch.full_like(x, post_u)),
            torch.zeros_like(x),
            torch.where(ahead, torch.full_like(x, pre_p), torch.full_like(x, post_p)),
        )

    return initial


def run_arm(nx, degree, wall_mode, cfl=CFL, extra_time=EXTRA_TIME, tvb=TVB):
    """One shock-reflection run; returns profiles, error norms and metadata."""
    if wall_mode not in ("solid", "boundary"):
        raise ValueError("wall_mode must be 'solid' or 'boundary'")
    ny = 1
    xmax = XMAX if wall_mode == "solid" else X_WALL
    dx = xmax / nx
    x_wall_face, wall_offset = _snap_to_face(X_WALL, dx)
    x_shock_face, shock_offset = _snap_to_face(X_SHOCK, dx)
    wr = WallReflection(x_shock=x_shock_face, x_wall=x_wall_face, incident_mach=MACH)
    solid = wall_solid_mask(nx, ny, x_wall_face, dx) if wall_mode == "solid" else None
    t_end = wr.wall_time + extra_time

    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        _initial(wr, x_shock_face), ncells=(nx, ny), degree=degree, final_time=t_end,
        cfl=cfl, domain=((0.0, xmax), (0.0, YMAX)), periodic=(False, False),
        solid=solid, limiter=None, tvb_constant=tvb, limiter_stage="final",
        track_boundary_flux=True,
        bcs={"left": "outflow", "right": "wall", "bottom": "wall", "top": "wall"},
    )
    elapsed = time.time() - started

    avg = coeff[..., 0, 0, :]
    rho = avg[:, 0, 0].cpu().numpy()
    pressure = _pressure(avg)[:, 0].cpu().numpy()
    centres = x.cpu().numpy()
    exact_rho, exact_u, exact_p = wr.state(x, t_end)
    fluid = np.ones(nx, dtype=bool) if solid is None else (~solid[:, 0].cpu().numpy())

    norms = {}
    for name, got, want in (("density", rho, exact_rho), ("pressure", pressure, exact_p)):
        diff = np.abs((got - want)[fluid])
        scale = float(np.abs(want[fluid]).mean())
        norms[name] = {"l1_relative": float(diff.mean() / scale),
                       "linf_relative": float(diff.max() / scale)}

    front = wr.front(t_end)
    mask = _plateau_mask(torch.as_tensor(centres), torch.as_tensor(fluid), front,
                         x_wall_face, dx, front_buffer=3, wall_buffer=1).numpy()
    rho5, _, p5 = wr.reflected
    plateau = {
        "cells": int(mask.sum()),
        "available_cells": float((x_wall_face - front) / dx),
        "mean_rho_error_percent": (100.0 * (rho[mask].mean() / rho5 - 1.0)
                                   if mask.any() else None),
        "mean_p_error_percent": (100.0 * (pressure[mask].mean() / p5 - 1.0)
                                 if mask.any() else None),
        "worst_error_percent": (float(max(np.abs(rho[mask] / rho5 - 1.0).max(),
                                          np.abs(pressure[mask] / p5 - 1.0).max()) * 100.0)
                                if mask.any() else None),
    }
    return {
        "wall_mode": wall_mode, "nx": nx, "degree": degree, "dx": dx,
        "steps": meta["steps"], "seconds": elapsed,
        "x_wall_face": x_wall_face, "x_shock_face": x_shock_face,
        "wall_offset_cells": wall_offset, "shock_offset_cells": shock_offset,
        "final_time": t_end, "front_exact": front,
        "solid_cells": int(meta.get("solid_cells", 0) or 0),
        "mass_relative_residual": abs(
            meta["conservation_residual_total"]["mass"]["relative"]),
        "norms": norms, "plateau": plateau,
        "profiles": {"x": centres.tolist(), "rho": rho.tolist(),
                     "pressure": pressure.tolist(),
                     "exact_rho": np.asarray(exact_rho).tolist(),
                     "exact_pressure": np.asarray(exact_p).tolist()},
        "plateau_mask": mask.tolist(),
    }


def parity(solid, boundary):
    """Difference between the two arms on the cells they share.

    The boundary arm is the common sub-domain ``[0, x_wall]``, so cell ``i`` of
    both arrays is the same physical cell (the meshes are built to share ``dx``).
    """
    n = boundary["nx"]
    if not math.isclose(solid["dx"], boundary["dx"], rel_tol=1e-12):
        raise ValueError("arms do not share dx")
    rho_s = np.asarray(solid["profiles"]["rho"][:n])
    rho_b = np.asarray(boundary["profiles"]["rho"][:n])
    p_s = np.asarray(solid["profiles"]["pressure"][:n])
    p_b = np.asarray(boundary["profiles"]["pressure"][:n])
    exact_rho = np.asarray(solid["profiles"]["exact_rho"][:n])
    fluid = np.asarray(solid["plateau_mask"])[:n]  # plateau band only
    scale_rho = float(np.abs(exact_rho).mean())
    gap = np.abs(rho_s - rho_b)
    out = {
        "shared_cells": int(n), "dx": solid["dx"],
        "whole_domain_l1_relative": float(gap.mean() / scale_rho),
        "plateau_max_relative": float(gap[fluid].max() / scale_rho) if fluid.any() else None,
        "plateau_max_percent": (float(100.0 * gap[fluid].max() / scale_rho)
                               if fluid.any() else None),
        "pressure_plateau_max_percent": (float(
            100.0 * np.abs(p_s - p_b)[fluid].max()
            / float(np.abs(np.asarray(solid["profiles"]["exact_pressure"])[:n]).mean()))
            if fluid.any() else None),
        "solid_error_plateau_percent": solid["plateau"]["worst_error_percent"],
        "boundary_error_plateau_percent": boundary["plateau"]["worst_error_percent"],
    }
    return out


def plan(resolutions, degrees):
    """Arm list, as ``mode:nx:degree`` strings, with ``nx`` matched to share dx."""
    arms = []
    for degree in degrees:
        for nx in resolutions:
            nx_b = int(round(nx * X_WALL / XMAX))
            if not math.isclose(nx_b * XMAX, nx * X_WALL, rel_tol=1e-12):
                raise ValueError(f"nx={nx} does not split into equal dx at x_wall")
            arms.append(f"boundary:{nx_b}:{degree}")
            arms.append(f"solid:{nx}:{degree}")
    return arms


def arm_key(mode, nx, degree):
    return f"arm_{mode}_nx{nx}_p{degree}"


def run_single(spec, outdir, log=print):
    """Run one arm (``mode:nx:degree``) and write ``outdir/arm_*.json``."""
    mode, nx, degree = spec.split(":")
    nx, degree = int(nx), int(degree)
    log(f"  {mode:8s} nx={nx:4d} P{degree} ...")
    result = run_arm(nx, degree, mode)
    path = Path(outdir) / f"{arm_key(mode, nx, degree)}.json"
    path.write_text(json.dumps(result, indent=2))
    worst = result["plateau"]["worst_error_percent"]
    log(f"    steps={result['steps']} t={result['seconds']:.0f}s "
        f"plateau_worst={worst} L1_rho={100 * result['norms']['density']['l1_relative']:.3f}%")
    return result


def load_arms(outdir):
    arms = {}
    for path in sorted(Path(outdir).glob("arm_*.json")):
        arm = json.loads(path.read_text())
        arms[arm_key(arm["wall_mode"], arm["nx"], arm["degree"])] = arm
    return arms


def summaries(arms):
    """Rows (one per arm) and solid-vs-boundary parities (one per grid/degree)."""
    rows = []
    for arm in sorted(arms.values(), key=lambda a: (a["degree"], a["nx"], a["wall_mode"])):
        rows.append({
            "wall_mode": arm["wall_mode"], "nx": arm["nx"], "degree": arm["degree"],
            "dx": arm["dx"], "steps": arm["steps"], "seconds": round(arm["seconds"], 1),
            "plateau_cells": arm["plateau"]["cells"],
            "plateau_worst_percent": arm["plateau"]["worst_error_percent"],
            "plateau_mean_rho_percent": arm["plateau"]["mean_rho_error_percent"],
            "plateau_mean_p_percent": arm["plateau"]["mean_p_error_percent"],
            "l1_rho_percent": 100.0 * arm["norms"]["density"]["l1_relative"],
            "linf_rho_percent": 100.0 * arm["norms"]["density"]["linf_relative"],
            "l1_p_percent": 100.0 * arm["norms"]["pressure"]["l1_relative"],
            "pass_3pct": bool(arm["plateau"]["worst_error_percent"] is not None
                              and arm["plateau"]["worst_error_percent"] < 3.0),
        })
    parities = []
    for degree in sorted({a["degree"] for a in arms.values()}):
        solid_arms = [a for a in arms.values()
                      if a["wall_mode"] == "solid" and a["degree"] == degree]
        for arm in sorted(solid_arms, key=lambda a: a["nx"]):
            nx_b = int(round(arm["nx"] * X_WALL / XMAX))
            mate = arms.get(arm_key("boundary", nx_b, degree))
            if mate is None:
                continue
            entry = parity(arm, mate)
            entry["degree"] = degree
            entry["nx_solid"] = arm["nx"]
            entry["nx_boundary"] = nx_b
            parities.append(entry)
    return rows, parities


def _order(values, dxs):
    """Observed order between consecutive (dx, error) pairs."""
    out = []
    for (d0, e0), (d1, e1) in zip(zip(dxs, values), zip(dxs[1:], values[1:])):
        out.append(float(math.log(e0 / e1) / math.log(d0 / d1))
                   if e0 > 0 and e1 > 0 else None)
    return out


def figure(path, rows, parities, arms, degree=2):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    fine = max(r["nx"] for r in rows if r["degree"] == degree)
    b_nx = int(round(fine * X_WALL / XMAX))
    s = arms[arm_key("solid", fine, degree)]
    b = arms[arm_key("boundary", b_nx, degree)]
    xs = np.asarray(s["profiles"]["x"])
    axes[0].plot(xs, s["profiles"]["exact_rho"], "k-", lw=1.6, label="exact two-shock")
    axes[0].plot(xs, s["profiles"]["rho"], "o", ms=3, label=f"solid wall ({fine} cells)")
    xb = np.asarray(b["profiles"]["x"])
    axes[0].plot(xb, b["profiles"]["rho"], "x", ms=3.4, label="boundary wall (same dx)")
    axes[0].axvline(X_WALL, color="0.6", ls="--", lw=1)
    axes[0].set_xlabel("x"); axes[0].set_ylabel(r"density $\rho$")
    axes[0].set_title(f"Mach {MACH:g} shock reflecting off a wall, P{degree}")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for mode, style in (("solid", "o-"), ("boundary", "s--")):
        sub = sorted([r for r in rows if r["degree"] == degree and r["wall_mode"] == mode],
                     key=lambda r: r["dx"], reverse=True)
        dxs = [r["dx"] for r in sub]
        errs = [r["l1_rho_percent"] for r in sub]
        axes[1].loglog(dxs, errs, style, label=f"{mode} wall L1(rho)")
        pl = [(r["dx"], r["plateau_worst_percent"]) for r in sub
              if r["plateau_worst_percent"] is not None]
        if pl:
            axes[1].loglog([a for a, _ in pl], [c for _, c in pl], "^:", alpha=0.8,
                           label=f"{mode} plateau worst")
    axes[1].axhline(3.0, color="r", ls="--", lw=1, label="3 % gate")
    axes[1].set_xlabel("dx"); axes[1].set_ylabel("relative error [%]")
    axes[1].set_title("error ladder"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3,
                                                                               which="both")

    sub = sorted([p for p in parities if p["degree"] == degree],
                 key=lambda p: p["dx"], reverse=True)
    axes[2].loglog([p["dx"] for p in sub], [100 * p["whole_domain_l1_relative"] for p in sub],
                   "o-", label="whole-domain L1 gap")
    axes[2].loglog([p["dx"] for p in sub], [p["plateau_max_percent"] for p in sub],
                   "s--", label="plateau max gap")
    axes[2].set_xlabel("dx"); axes[2].set_ylabel("solid vs boundary [%]")
    axes[2].set_title("embedded wall vs boundary wall (matched dx)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[30, 60, 120, 240])
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--outdir", type=Path,
                        default=Path("docs/benchmarks/results/cks1998-embedded-wall"))
    parser.add_argument("--plan", action="store_true",
                        help="print the mode:nx:degree arms and exit (for xargs -P)")
    parser.add_argument("--single", metavar="MODE:NX:DEGREE",
                        help="run one arm and write its JSON (parallel driver)")
    parser.add_argument("--aggregate", action="store_true",
                        help="combine the per-arm JSONs into parity.json + parity.png")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.plan:
        for spec in plan(tuple(args.resolutions), tuple(args.degrees)):
            print(spec)
        return None
    if args.single:
        run_single(args.single, args.outdir)
        return None
    if args.aggregate:
        arms = load_arms(args.outdir)
    else:
        if (args.outdir / "arm_solid_nx1_p0.json").exists():  # pragma: no cover
            raise SystemExit("outdir already holds arms; use --aggregate")
        arms = {}
        for spec in plan(tuple(args.resolutions), tuple(args.degrees)):
            mode, nx, degree = spec.split(":")
            arms[arm_key(mode, int(nx), int(degree))] = run_single(spec, args.outdir)

    rows, parities = summaries(arms)
    orders = {}
    for degree in sorted({a["degree"] for a in arms.values()}):
        for mode in ("solid", "boundary"):
            sub = sorted([r for r in rows if r["degree"] == degree and r["wall_mode"] == mode],
                         key=lambda r: r["dx"], reverse=True)
            orders[f"{mode}_p{degree}_l1_rho_order"] = _order(
                [r["l1_rho_percent"] for r in sub], [r["dx"] for r in sub])

    payload = {
        "case": "REPRO-DG-CKS1998 embedded-wall shock reflection: solid vs boundary arm",
        "reference": "Cockburn & Shu JCP 141 (1998); exact two-shock solution (Rankine-Hugoniot)",
        "settings": {"incident_mach": MACH, "cfl": CFL, "tvb_constant": TVB,
                     "extra_time": EXTRA_TIME, "x_shock": X_SHOCK, "x_wall": X_WALL,
                     "xmax": XMAX, "tolerance_percent": 3.0},
        "rows": rows, "parity": parities, "orders": orders,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform()},
    }
    (args.outdir / "parity.json").write_text(json.dumps(payload, indent=2))
    degrees = sorted({a["degree"] for a in arms.values()})
    figure(args.outdir / "parity.png", rows, parities, arms, degree=max(degrees))

    print("\n| mode | nx | P | dx | steps | plateau cells | plateau worst % | L1 rho % | pass(<3%) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        worst = "-" if r["plateau_worst_percent"] is None else f"{r['plateau_worst_percent']:.2f}"
        print(f"| {r['wall_mode']} | {r['nx']} | {r['degree']} | {r['dx']:.5f} | {r['steps']} "
              f"| {r['plateau_cells']} | {worst} | {r['l1_rho_percent']:.3f} | {r['pass_3pct']} |")
    print("\n| P | nx solid | dx | solid-vs-boundary L1 % | plateau max % |")
    print("|---|---|---|---|---|")
    for p in parities:
        print(f"| {p['degree']} | {p['nx_solid']} | {p['dx']:.5f} "
              f"| {100 * p['whole_domain_l1_relative']:.3f} | {p['plateau_max_percent']} |")
    print("\nobserved orders (L1 rho):", json.dumps(orders, indent=2))
    return payload


if __name__ == "__main__":
    main()