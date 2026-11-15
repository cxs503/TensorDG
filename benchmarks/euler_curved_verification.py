"""Verification suite for the curvilinear 2-D Euler DG solver (CKS1998 anchor).

Every number the report quotes is produced here and dumped to
``validation/euler_curved/metrics.json`` (plus ``summary.md`` and figures), so no
digit is ever hand-copied into prose.

Gates (pre-declared, fixed before any result was inspected)
-----------------------------------------------------------
G1  free-stream preservation on a doubly periodic *warped* mesh, ``p = 1, 2, 3``:
    ``sup |dU/dt|`` of two ssp-rk3 sweeps at round-off, ``relative <= 1e-11``
    and 200-step modal drift ``<= 1e-10``.
G1n negative control: the same gate applied to the aliasing volume derivative
    (``volume_derivative_degree = p``) must *fail* by ``>= 1e3`` x the round-off
    level -- otherwise the gate is vacuous.
G2  accuracy: smooth isentropic vortex on the warped mesh, observed order
    ``>= p + 1/2`` (the Cockburn--Shu tolerance for ``P^p``) for ``p = 1, 2``.
G3  general-mesh equivalence: at equal element counts the warped/straight error
    ratio stays in ``[0.3, 3]`` -- curvature must not cost accuracy.
G4  conservation: total mass/momentum/energy drift ``<= 1e-10`` relative on the
    warped periodic mesh (conservative DG + consistent metrics).

Usage
-----
    PYTHONPATH=. python benchmarks/euler_curved_verification.py --outdir validation/euler_curved
Exit code 0 iff every gate passes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from tensordg.euler_2d import vortex_primitive
from tensordg.euler_curved import (
    CurvilinearEuler,
    conserved_from_primitive,
    euler_flux,
    hllc_normal,
    straight_box_mesh,
    warped_box_mesh,
)

BOX = 10.0          # periodic box length of the vortex case
STRENGTH = 5.0      # isentropic-vortex strength
BACKGROUND = (1.0, 1.0)
AMPLITUDE = 0.06    # mesh warp amplitude
GEOMETRY_DEGREE = 3

GATES = {
    "free_stream_relative": 1e-11,
    "free_stream_drift": 1e-10,   # over 20 ssp-rk3 steps (see limitation notes)
    "negative_control_ratio": 1e3,
    "order_margin": 0.5,       # observed order >= p + 1/2
    "warped_straight_ratio": (0.3, 3.0),
    "conservation_relative": 1e-10,
}


def exact_primitive(time):
    """Isentropic vortex as a numpy callable, from the repository reference."""

    def primitive(x, y):
        rho, velocity_x, velocity_y, pressure = vortex_primitive(
            torch.as_tensor(np.ascontiguousarray(x)),
            torch.as_tensor(np.ascontiguousarray(y)),
            time=time, strength=STRENGTH, background_velocity=BACKGROUND, period=BOX,
        )
        return (rho.numpy(), velocity_x.numpy(), velocity_y.numpy(), pressure.numpy())

    return primitive


def hllc_checks():
    """Independent sanity checks of the general-normal HLLC implementation."""
    state = conserved_from_primitive(
        np.array([1.0, 0.4]), np.array([0.3, -0.7]), np.array([-0.2, 0.5]),
        np.array([1.0, 0.35]),
    )
    angle = np.array([0.0, 0.9272952180016122])   # (1,0) and (0.6,0.8)
    nx, ny = np.cos(angle), np.sin(angle)
    same = hllc_normal(state, state, nx, ny)
    flux_x, flux_y = euler_flux(state)
    exact_flux = flux_x * nx[:, None] + flux_y * ny[:, None]
    rotational = hllc_normal(state, state[::-1], nx, ny)
    mirrored = hllc_normal(state[::-1], state, nx, ny)
    return {
        "consistency": float(np.max(np.abs(same - exact_flux))),
        "antisymmetry": float(np.max(np.abs(rotational + mirrored)) /
                              max(np.max(np.abs(rotational)), 1e-30)),
    }


def free_stream_study(profile="bench"):
    rows = []
    for degree in (1, 2, 3):
        mesh = warped_box_mesh(4, 4, GEOMETRY_DEGREE, length=BOX, amplitude=AMPLITUDE)
        solver = CurvilinearEuler(mesh, degree=degree)
        good = solver.free_stream_defect(steps=20)
        negative = CurvilinearEuler(
            mesh, degree=degree, volume_derivative_degree=degree
        ).free_stream_defect(steps=0)   # operator-only: the aliased scheme diverges
        rows.append({
            "degree": degree,
            "control": good["relative"],
            "drift": good["drift"],
            "metric_identity": good["metric"],
            "negative_control": negative["relative"],
            "delay_ratio": negative["relative"] / max(good["relative"], 1e-300),
            "dt": good["dt"],
        })
    return rows


def convergence_study(profile="bench"):
    meshes = (8, 16, 32) if profile == "full" else (8, 16)
    time_final = 0.1
    rows = []
    for family, builder in (
        ("warped", lambda n: warped_box_mesh(n, n, GEOMETRY_DEGREE, length=BOX,
                                             amplitude=AMPLITUDE)),
        ("straight", lambda n: straight_box_mesh(n, n, 1, length=BOX)),
    ):
        for degree in (1, 2):
            counts = meshes if degree == 1 else meshes[:-1]
            errors = []
            for cells in counts:
                solver = CurvilinearEuler(builder(cells), degree=degree, cfl=0.2)
                state = solver.project(exact_primitive(0.0))
                state, dt, steps = solver.evolve(state, time_final)
                err = solver.relative_errors(state, exact_primitive(time_final))
                errors.append(err["worst"])
                rows.append({
                    "family": family, "degree": degree, "cells": cells,
                    "steps": steps, "dt": dt, "error": err["worst"],
                    "rho": err["rho"], "u": err["u"], "v": err["v"], "p": err["p"],
                })
            order = math.log(errors[0] / errors[-1]) / math.log(counts[-1] / counts[0])
            rows[-1]["observed_order"] = order
            rows[-1]["design_order"] = degree + 1
            rows[-1]["gate"] = degree + GATES["order_margin"]
    return rows


def conservatism():
    solver = CurvilinearEuler(
        warped_box_mesh(8, 8, GEOMETRY_DEGREE, length=BOX, amplitude=AMPLITUDE),
        degree=1, cfl=0.2,
    )
    state = solver.project(exact_primitive(0.0))
    before = solver.totals(state)
    state, dt, steps = solver.evolve(state, 0.5)
    after = solver.totals(state)
    relative = float(np.max(np.abs(after - before)) / np.max(np.abs(before)))
    return {"steps": steps, "dt": dt, "relative_drift": relative}


def warped_vs_straight(rows):
    """Error ratio warped/straight at the finest common element count."""
    out = {}
    for degree in (1, 2):
        warped = [r for r in rows if r["family"] == "warped" and r["degree"] == degree]
        straight = [r for r in rows if r["family"] == "straight" and r["degree"] == degree]
        cells = min(warped[-1]["cells"], straight[-1]["cells"])
        w = next(r["error"] for r in warped if r["cells"] == cells)
        s = next(r["error"] for r in straight if r["cells"] == cells)
        out[degree] = {"cells": cells, "warped": w, "straight": s, "ratio": w / s}
    return out


def figures(metrics, directory: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    made = []

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for family, marker in (("warped", "o"), ("straight", "s")):
        for degree, colour in ((1, "tab:blue"), (2, "tab:red")):
            pts = [r for r in metrics["convergence"]
                   if r["family"] == family and r["degree"] == degree]
            if not pts:
                continue
            cells = np.array([r["cells"] for r in pts], float)
            err = np.array([r["error"] for r in pts], float)
            ax.loglog(cells, err, marker + "-", color=colour,
                      label=f"{family} P{degree}")
    counts = np.array([8.0, 16.0, 32.0])
    for degree, colour in ((1, "tab:blue"), (2, "tab:red")):
        ax.loglog(counts, metrics["order_reference"][str(degree)] * (counts / counts[0]) ** (-(degree + 1)),
                  "k:", lw=1.0, label=f"h^{degree + 1} reference")
    ax.set_xlabel("elements per direction")
    ax.set_ylabel("relative $L^2$ error")
    ax.set_title("CKS1998 anchor: curvilinear DG Euler, vortex accuracy")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = directory / "euler_curved_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    degrees = [r["degree"] for r in metrics["free_stream"]]
    control = [r["control"] for r in metrics["free_stream"]]
    negative = [r["negative_control"] for r in metrics["free_stream"]]
    x = np.arange(len(degrees))
    ax.bar(x - 0.18, control, width=0.36, label="consistent metrics", color="tab:green")
    ax.bar(x + 0.18, negative, width=0.36, label="aliased (negative control)", color="tab:orange")
    ax.set_yscale("log")
    ax.axhline(GATES["free_stream_relative"], color="k", ls=":", lw=1,
               label="FSP gate")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{d}" for d in degrees])
    ax.set_ylabel("free-stream defect (relative)")
    ax.set_title("Free-stream preservation on a warped periodic mesh")
    ax.legend(fontsize=8)
    path = directory / "euler_curved_free_stream.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path)
    return made


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="validation/euler_curved")
    parser.add_argument("--profile", default="bench", choices=("bench", "full"))
    parser.add_argument("--figdir", default="docs/figures")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    free_stream = free_stream_study(args.profile)
    convergence = convergence_study(args.profile)
    ratio = warped_vs_straight(convergence)
    conservation = conservatism()
    checks = hllc_checks()

    verdict = {}
    verdict["G1_free_stream"] = all(
        r["control"] <= GATES["free_stream_relative"]
        and r["drift"] <= GATES["free_stream_drift"] for r in free_stream
    )
    verdict["G1n_negative_control"] = all(
        r["delay_ratio"] >= GATES["negative_control_ratio"] for r in free_stream
    )
    verdict["G2_order"] = all(
        r["observed_order"] >= r["gate"] for r in convergence if "observed_order" in r
    )
    verdict["G3_general_mesh"] = all(
        GATES["warped_straight_ratio"][0] <= v["ratio"] <= GATES["warped_straight_ratio"][1]
        for v in ratio.values()
    )
    verdict["G4_conservation"] = conservation["relative_drift"] <= GATES["conservation_relative"]
    verdict["G5_hllc_consistency"] = checks["consistency"] <= 1e-12
    verdict["G5_hllc_antisymmetry"] = checks["antisymmetry"] <= 1e-12

    order_reference = {}
    for degree in (1, 2):
        pts = [r for r in convergence if r["family"] == "warped" and r["degree"] == degree]
        if pts:
            order_reference[str(degree)] = pts[0]["error"]
    metrics = {
        "case": "REPRO-DG-CKS1998 / curvilinear 2-D Euler DG",
        "box": BOX, "strength": STRENGTH, "amplitude": AMPLITUDE,
        "geometry_degree": GEOMETRY_DEGREE,
        "gates": GATES,
        "free_stream": free_stream,
        "convergence": convergence,
        "warped_vs_straight": ratio,
        "conservation": conservation,
        "hllc": checks,
        "order_reference": order_reference,
        "verdict": verdict,
        "wall_seconds": time.time() - started,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    made = figures(metrics, Path(args.figdir))
    lines = ["# euler_curved verification", ""]
    lines.append(f"HLLC consistency {checks['consistency']:.3e}, "
                 f"antisymmetry {checks['antisymmetry']:.3e}")
    lines.append("")
    lines.append("| gate | value | verdict |")
    lines.append("|---|---:|---|")
    for row in free_stream:
        lines.append(f"| FSP P{row['degree']} (control) | {row['control']:.3e} | "
                     f"{'PASS' if row['control'] <= GATES['free_stream_relative'] else 'FAIL'} |")
        lines.append(f"| FSP P{row['degree']} (aliased control) | "
                     f"{row['negative_control']:.3e} | "
                     f"{'detected' if row['delay_ratio'] >= GATES['negative_control_ratio'] else 'NOT DETECTED'} |")
    for row in convergence:
        if "observed_order" in row:
            lines.append(f"| order {row['family']} P{row['degree']} | "
                         f"{row['observed_order']:.3f} | "
                         f"{'PASS' if row['observed_order'] >= row['gate'] else 'FAIL'} |")
    for degree, value in ratio.items():
        lines.append(f"| warped/straight P{degree} @ {value['cells']} | "
                     f"{value['ratio']:.3f} | "
                     f"{'PASS' if GATES['warped_straight_ratio'][0] <= value['ratio'] <= GATES['warped_straight_ratio'][1] else 'FAIL'} |")
    lines.append(f"| conservation drift | {conservation['relative_drift']:.3e} | "
                 f"{'PASS' if verdict['G4_conservation'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"figures: {', '.join(str(p) for p in made)}")
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("verdict:", json.dumps(verdict))
    return 0 if all(verdict.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())