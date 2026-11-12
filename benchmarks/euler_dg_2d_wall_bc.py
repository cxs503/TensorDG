"""Verification of the 2-D DG wall / general ghost-state boundaries (m16803).

Until this mission ``tensordg.euler_dg_2d.solve_euler_dg_2d`` only knew periodic
faces and a zero-gradient outer face, so neither a reflecting wall nor a tracked
inflow existed and the Woodward--Colella double Mach reflection (the 2-D system
benchmark left open by ``REPRO-DG-CKS1998``) could not be posed.  The solver now
accepts ``bcs`` (see the module docstring).  This driver verifies it against
closed-form solutions, none of which comes from the solver:

**G1 free-stream preservation.**  A uniform flow parallel to a wall must stay
uniform: the discrete wall flux it sees is exactly ``(0, p, 0, 0)``, so the
result cannot be "almost" conservative.  Gate: max relative deviation ``< 1e-12``.

**G2/G3 shock reflecting off a wall** (the classical wall-reflection problem).
A right-moving normal shock of Mach number ``M1`` is initialised *exactly* from
the Rankine--Hugoniot relations, so before it reaches the wall the solution is
the initial data simply translated.  After the reflection an exact two-shock
solution exists: a reflected shock runs back into the post-shock state with a
speed obtained from the shock jump conditions alone (bisection on the reflected
shock Mach number, ``rhs_reflected`` below -- no solver code involved).  The
plateau behind the reflected shock and the reflected front position are compared
with the solver.

**G4 closed box.**  Four walls, no inflow: total mass and total energy must be
conserved to round-off (the wall flux carries no mass and only pressure work,
which integrates to zero over a closed surface).

Usage::

    python -m benchmarks.euler_dg_2d_wall_bc            # write JSON + PNG
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordg.euler_1d import GAMMA
from tensordg.euler_dg_2d import cell_fields, conservative_state, solve_euler_dg_2d

RESULT_DIR = Path("docs/benchmarks/results/euler-dg-2d-wall-bc")


# ---------------------------------------------------------------------------
# Closed-form references (independent of the solver)
# ---------------------------------------------------------------------------

def post_normal_shock(pre, mach):
    """Post-shock state of a right-moving normal shock of Mach ``mach``."""
    rho, u, p = pre
    sound = math.sqrt(GAMMA * p / rho)
    speed = u + mach * sound
    ratio = (GAMMA + 1.0) * mach ** 2 / ((GAMMA - 1.0) * mach ** 2 + 2.0)
    rho2 = rho * ratio
    p2 = p * (2.0 * GAMMA * mach ** 2 - (GAMMA - 1.0)) / (GAMMA + 1.0)
    u2 = speed * (1.0 - rho / rho2)
    return (rho2, u2, p2), speed


def rhs_reflected(state2, tolerance=1e-14):
    """Reflected shock inside ``state2`` leaving the gas at rest.

    A shock of Mach number ``M_r`` (measured with the post-incident-shock sound
    speed) runs *left* into ``state2``; the jump conditions give
    ``u5 = u2 - M_r c2 (1 - rho2/rho5)`` and the reflected plateau has ``u5 = 0``.
    ``u5(M_r)`` is monotone, so bisection gives the exact reflected state.
    """
    rho2, u2, p2 = state2
    sound2 = math.sqrt(GAMMA * p2 / rho2)

    def residual(mach):
        ratio = (GAMMA + 1.0) * mach ** 2 / ((GAMMA - 1.0) * mach ** 2 + 2.0)
        return u2 - mach * sound2 * (1.0 - 1.0 / ratio)

    low, high = 1.0 + 1e-12, 2.0
    # ``residual > 0`` means the reflected shock is too weak to stop the gas,
    # so the root is bracketed by [low, high] with opposite signs.
    if not residual(low) > 0.0:  # pragma: no cover - residual(1) == u2 > 0
        raise ValueError("reflected shock Mach number below unity; no reflection")
    while residual(high) > 0.0:
        high *= 2.0
        if high > 1e6:  # pragma: no cover
            raise ValueError("no reflected shock root found")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
        if high - low < tolerance * high:
            break
    mach = 0.5 * (low + high)
    ratio = (GAMMA + 1.0) * mach ** 2 / ((GAMMA - 1.0) * mach ** 2 + 2.0)
    rho5 = rho2 * ratio
    p5 = p2 * (2.0 * GAMMA * mach ** 2 - (GAMMA - 1.0)) / (GAMMA + 1.0)
    speed = u2 - mach * sound2
    return (rho5, 0.0, p5), mach, speed


class WallReflection:
    """Exact solution of a normal shock reflecting off a solid wall."""

    def __init__(self, x_shock=0.5, x_wall=1.0, incident_mach=2.0,
                 pre_shock=(1.0, 0.0, 1.0)):
        self.x_zero = x_shock
        self.x_wall = x_wall
        self.pre = pre_shock
        self.post, self.incident_speed = post_normal_shock(pre_shock, incident_mach)
        self.reflected, self.reflected_mach, self.reflected_speed = rhs_reflected(self.post)
        self.wall_time = (x_wall - x_shock) / self.incident_speed

    def front(self, time):
        """Position of the active discontinuity at ``time``."""
        if time <= self.wall_time:
            return self.x_zero + self.incident_speed * time
        return self.x_wall + self.reflected_speed * (time - self.wall_time)

    def state(self, x, time):
        """Exact primitive ``(rho, u, p)`` at ``x`` (``u`` along ``x``)."""
        front = self.front(time)
        behind = self.post if time <= self.wall_time else self.reflected
        return tuple(np.where(x < front, value, value0)
                     for value, value0 in zip(behind, self.pre))


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def free_stream_deviation(ncells=(24, 8), degree=2, final_time=0.2, cfl=0.2):
    """Uniform flow parallel to the walls: the solution must not move."""
    uniform = (1.0, 0.7, 0.0, 1.0)  # rho, u, v, p -- v = 0 at both walls

    def initial(x, y):
        return tuple(torch.full_like(x, value) for value in uniform)

    x, y, coeff, meta = solve_euler_dg_2d(
        initial, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        periodic=(True, False), bcs={"bottom": "wall", "top": "wall"},
        limiter=None,
    )
    reference = conservative_state(*initial(x[:, None], y[None, :]))
    scale = reference.abs().amax().item()
    deviation = float((coeff - reference[:, :, None, None, :]).abs().amax().item()) / scale
    return {"max_relative_deviation": deviation, "metadata": meta}


def wall_reflection(ncells=(200, 4), degree=2, final_time=0.4, cfl=0.12,
                    profile_band=None):
    """Shock reflecting off a wall vs the closed-form two-shock solution."""
    case = WallReflection()
    rho2, u2, p2 = case.post
    rho1, u1, p1 = case.pre

    def initial(x, y):
        before = x < case.x_zero
        zeros = torch.zeros_like(x)
        return (
            torch.where(before, torch.full_like(x, rho2), torch.full_like(x, rho1)),
            torch.where(before, torch.full_like(x, u2), torch.full_like(x, u1)),
            zeros,
            torch.where(before, torch.full_like(x, p2), torch.full_like(x, p1)),
        )

    started = time.time()
    x, y, coeff, meta = solve_euler_dg_2d(
        initial, ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        periodic=(False, True), bcs={"left": "outflow", "right": "wall"},
        limiter="tvb", tvb_constant=0.0, limiter_stage="final",
    )
    wall = time.time() - started
    fields = cell_fields(coeff)
    rho = fields["density"][:, 0].cpu().numpy()
    velocity = fields["velocity_x"][:, 0].cpu().numpy()
    pressure = fields["pressure"][:, 0].cpu().numpy()
    centers = x.cpu().numpy()

    exact_rho, exact_u, exact_p = case.state(centers, final_time)
    l2 = {}
    for name, got, want in (("density", rho, exact_rho),
                            ("velocity", velocity, exact_u),
                            ("pressure", pressure, exact_p)):
        denominator = float(np.sqrt(np.mean(want ** 2)))
        if denominator < 1e-12:  # exact state at rest: normalise by the measured field
            denominator = max(float(np.sqrt(np.mean(got ** 2))), 1e-12)
        l2[name] = float(np.sqrt(np.mean((got - want) ** 2)) / denominator)

    # plateau behind the reflected shock: it lies *between* the reflected front
    # and the wall, so the band is fixed by the exact front, not by hand
    front_exact = case.front(final_time)
    low, high = profile_band or (front_exact + 2.0 * (centers[1] - centers[0]), 0.98)
    band = (centers > low) & (centers < high)
    rho5, u5, p5 = case.reflected
    plateau = {
        "density": float(np.mean(rho[band]) - rho5) / rho5,
        "pressure": float(np.mean(pressure[band]) - p5) / p5,
        "velocity_over_sound": float(np.mean(velocity[band])) /
                               math.sqrt(GAMMA * p5 / rho5),
    }

    gradient = np.abs(np.gradient(rho, centers))
    index = int(np.argmax(gradient))
    front_numeric = float(centers[index])
    if 0 < index < len(centers) - 1:
        slope = gradient[index - 1] - 2.0 * gradient[index] + gradient[index + 1]
        if slope != 0.0:
            offset = 0.5 * (gradient[index - 1] - gradient[index + 1]) / slope
            front_numeric += offset * (centers[1] - centers[0])
    cell_width = (centers[-1] - centers[0]) / (len(centers) - 1)

    return {
        "case": {
            "incident_mach": 2.0,
            "wall_time": case.wall_time,
            "reflected_mach": case.reflected_mach,
            "reflected_speed": case.reflected_speed,
            "post_shock": case.post,
            "reflected_state": case.reflected,
            "final_time": final_time,
            "ncells": list(ncells),
            "degree": degree,
        },
        "l2_relative": l2,
        "plateau_relative": plateau,
        "front": {
            "exact": front_exact,
            "numeric": front_numeric,
            "cells_off": abs(front_numeric - front_exact) / cell_width,
        },
        "wall_seconds": wall,
        "steps": meta["steps"],
        "profiles": {
            "centers": centers.tolist(),
            "density": rho.tolist(),
            "exact_density": exact_rho.tolist(),
            "pressure": pressure.tolist(),
            "exact_pressure": exact_p.tolist(),
            "velocity": velocity.tolist(),
            "exact_velocity": exact_u.tolist(),
        },
    }


def closed_box(ncells=(16, 16), degree=2, final_time=0.15, cfl=0.15):
    """Four walls, an initial pressure pulse: mass and energy must not drift."""
    def initial(x, y):
        rho = torch.ones_like(x)
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        bubble = torch.exp(-40.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))
        return rho + 0.2 * bubble, u, v, 1.0 + 0.4 * bubble

    walls = {"left": "wall", "right": "wall", "bottom": "wall", "top": "wall"}
    common = dict(ncells=ncells, degree=degree, cfl=cfl, periodic=(False, False),
                  bcs=walls, limiter=None)

    def totals(coefficient):
        averages = coefficient[..., 0, 0, :]
        mass = float(averages[..., 0].sum().item())
        energy = float(averages[..., 3].sum().item())
        return mass, energy

    _, _, start, _ = solve_euler_dg_2d(initial, final_time=0.0, **common)
    _, _, end, meta = solve_euler_dg_2d(initial, final_time=final_time, **common)
    mass0, energy0 = totals(start)
    mass1, energy1 = totals(end)
    return {
        "mass_relative_drift": abs(mass1 - mass0) / abs(mass0),
        "energy_relative_drift": abs(energy1 - energy0) / abs(energy0),
        "steps": meta["steps"],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

GATES = {
    "free_stream": 1e-12,
    "plateau_density": 0.01,
    "plateau_pressure": 0.01,
    "front_cells": 2.0,
    "l2_relative": 0.05,
    "closed_box": 1e-12,
}


def evaluate(results):
    checks = {
        "free_stream": (results["free_stream"]["max_relative_deviation"], 1e-12),
        "plateau_density": (abs(results["reflection"]["plateau_relative"]["density"]), 0.01),
        "plateau_pressure": (abs(results["reflection"]["plateau_relative"]["pressure"]), 0.01),
        "front_cells": (results["reflection"]["front"]["cells_off"], 2.0),
        "l2_relative": (max(results["reflection"]["l2_relative"].values()), 0.05),
        "closed_box": (max(results["closed_box"]["mass_relative_drift"],
                           results["closed_box"]["energy_relative_drift"]), 1e-12),
    }
    verdicts = {
        name: {"measured": value, "threshold": limit, "pass": value < limit}
        for name, (value, limit) in checks.items()
    }
    return verdicts, all(item["pass"] for item in verdicts.values())


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def make_figure(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    profiles = results["reflection"]["profiles"]
    centers = np.asarray(profiles["centers"])
    exact = np.asarray(profiles["exact_density"])
    numeric = np.asarray(profiles["density"])
    left.plot(centers, exact, "k--", label="closed-form two-shock solution")
    left.plot(centers, numeric, "C3-", label="DG / HLLC (wall BC)")
    left.set_xlabel("x")
    left.set_ylabel(r"$\rho$")
    left.set_title("shock reflecting off a wall, t = %.2f" % results["case"]["final_time"])
    left.legend()
    verdicts = results["gates"]
    names = list(verdicts)
    values = [verdicts[name]["measured"] for name in names]
    right.barh(names, values, color=["C2" if verdicts[n]["pass"] else "C1" for n in names])
    right.set_xscale("log")
    right.set_xlabel("measured (log scale)")
    right.set_title("gates: green = pass")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="coarser grids (still gates, faster)")
    args = parser.parse_args(argv)

    started = time.time()
    results = {
        "free_stream": free_stream_deviation(),
        "reflection": wall_reflection(
            ncells=(100, 3) if args.quick else (200, 4),
            degree=2, final_time=0.4,
        ),
        "closed_box": closed_box(ncells=(12, 12) if args.quick else (16, 16)),
    }
    verdicts, passed = evaluate(results)
    results["gates"] = verdicts
    results["verdict"] = "PASS" if passed else "FAIL"
    results["wall_seconds"] = time.time() - started

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    figure_path = RESULT_DIR / "wall_reflection.png"
    make_figure(results, figure_path)
    results["figures"] = [str(figure_path)]

    payload = {
        "verdict": results["verdict"],
        "gates": results["gates"],
        "case": results["reflection"]["case"],
        "l2_relative": results["reflection"]["l2_relative"],
        "plateau_relative": results["reflection"]["plateau_relative"],
        "front": results["reflection"]["front"],
        "free_stream": {key: value for key, value in results["free_stream"].items()
                        if key != "metadata"},
        "closed_box": results["closed_box"],
        "steps": results["reflection"]["steps"],
        "wall_seconds": results["wall_seconds"],
        "profiles": results["reflection"]["profiles"],
    }
    json_path = RESULT_DIR / "results.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    provenance = {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "arch": platform.machine(),
        "command": "python -m benchmarks.euler_dg_2d_wall_bc",
        "artifact_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (json_path, figure_path) if path.exists()
        },
    }
    (RESULT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())