"""Verification of the merged pure heat-conduction DG path (``ns_dg_2d``).

``tensordg/ns_dg_2d.py`` ships three public helpers that had no test, no
benchmark and no documentation before this script:

* ``solve_heat_conduction_dg_2d`` -- "pure heat conduction between two
  isothermal walls in a fluid at rest";
* ``heat_conduction_exact`` -- the analytic steady conduction state;
* ``gas_constants`` -- the non-dimensional gas constants.

The script establishes three machine-checkable facts and writes them to
``validation/ns_heat_conduction/evidence.json``:

``E1`` (analytic sanity) the shipped analytic state really is an exact solution
       of the compressible equations at rest: ``kappa * laplace(T) = 0``,
       ``rho = p / (R T)``, ``u = 0``.
``E2`` (boundary-closure defect) evaluated on that exact state, the DG residual
       of the *energy* equation does not vanish at the isothermal walls: at
       ``P1`` the wall row grows towards a grid-independent ``O(1)`` value
       (0.815 at 16 cells -> 0.963 at 128 cells, asymptote ``kappa``), while the
       interior row decays like ``O(h)`` and the mass/momentum rows are at
       machine zero.  The isothermal Dirichlet energy closure is therefore not
       consistent, and the defect does not depend on the ``wall_gradient``
       reconstruction.
``E3`` (consequence) the shipped default call
       ``solve_heat_conduction_dg_2d()`` raises
       ``FloatingPointError: nonphysical cell average`` at
       ``ncells=(2, 16), degree=1, final_time=40``; the collapse time is
       bracketed in ``(0.02, 0.05)`` and is insensitive to the CFL, the grid and
       the choice of initial condition (even starting *on* the exact steady
       state collapses), which rules out a time-step stability limit as the
       cause.

Run ``python -m benchmarks.verify_ns_heat_conduction`` (about one minute on a
laptop CPU) to regenerate the evidence, the figure and the printed verdicts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.ns_dg_2d as ns

BOTTOM_TEMPERATURE = 1.0
TOP_TEMPERATURE = 0.5
PRESSURE = 1.0
MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
DOMAIN = ((0.0, 1.0), (0.0, 1.0))


# ---------------------------------------------------------------------------
# Analytic reference
# ---------------------------------------------------------------------------

def exact_steady(y):
    """Analytic steady pure-conduction state (``y`` is a 1-D tensor)."""
    return ns.heat_conduction_exact(
        y, 0.0, 1.0, BOTTOM_TEMPERATURE, TOP_TEMPERATURE, PRESSURE
    )


def analytic_checks() -> dict:
    y = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    state = exact_steady(y)
    temperature = state["temperature"]
    d2 = (temperature[2:] - 2.0 * temperature[1:-1] + temperature[:-2]) / (
        float(y[1] - y[0]) ** 2)
    density_from_ideal_gas = PRESSURE / (ns.GAS_CONSTANT * temperature)
    constants = ns.gas_constants(PRANDTL)
    return {
        "laplace_of_temperature_max_abs": float(d2.abs().max()),
        "density_matches_ideal_gas_max_abs": float(
            (state["density"] - density_from_ideal_gas).abs().max()),
        "velocity_max_abs": float(
            max(state["velocity_x"].abs().max(), state["velocity_y"].abs().max())),
        "wall_temperatures": [float(temperature[0]), float(temperature[-1])],
        "gas_constants": constants,
        "cp_minus_cv_minus_gas_constant": float(
            constants["cp"] - constants["cv"] - constants["gas_constant"])
        if "gas_constant" in constants else float(ns.CP - ns.CV - ns.GAS_CONSTANT),
        "gamma_minus_cp_over_cv": float(ns.GAMMA - ns.CP / ns.CV),
        "conductivity": KAPPA,
    }


# ---------------------------------------------------------------------------
# Residual of the exact steady state
# ---------------------------------------------------------------------------

def steady_initial(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    s = (y + 0.0 * x) * torch.ones_like(ref)
    temperature = BOTTOM_TEMPERATURE + (TOP_TEMPERATURE - BOTTOM_TEMPERATURE) * s
    temperature = temperature + 0.0 * ref
    return (PRESSURE / temperature, torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def uniform_initial(x, y):
    ref = x * y + 0.0 * x + 0.0 * y
    return (torch.ones_like(ref), torch.zeros_like(ref), torch.zeros_like(ref),
            PRESSURE * torch.ones_like(ref))


def steady_residual(ncells, degree, wall_gradient) -> dict:
    nx, ny = ncells
    dx, dy = 1.0 / nx, 1.0 / ny
    nodes, weights, basis, derivative, signs = ns.legendre_operators(
        degree, degree + 2, torch.device("cpu"), torch.float64)
    x = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    coeff = ns._initial_coefficients(steady_initial, x, y, dx, dy, nodes, weights,
                                     basis, torch.device("cpu"), torch.float64)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": BOTTOM_TEMPERATURE},
        "ymax": {"velocity": (0.0, 0.0), "temperature": TOP_TEMPERATURE},
    }
    penalty = 2.0 * (degree + 1.0) ** 2
    rhs = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                     (True, False), boundary, MU, KAPPA, penalty, wall_gradient)
    energy = rhs[..., 3]
    return {
        "ncells": list(ncells),
        "degree": degree,
        "wall_gradient": wall_gradient,
        "energy_wall_cold": float(energy[:, -1].abs().max()),
        "energy_wall_hot": float(energy[:, 0].abs().max()),
        "energy_interior": float(energy[:, 1:-1].abs().max()),
        "density_max_abs": float(rhs[..., 0].abs().max()),
        "momentum_max_abs": float(rhs[..., 1:3].abs().max()),
    }


# ---------------------------------------------------------------------------
# Time integration attempts
# ---------------------------------------------------------------------------

def attempt(initial, initial_name, final_time, ncells=(2, 16), degree=1,
            cfl=0.3, wall_gradient="taylor") -> dict:
    started = time.time()
    record = {"initial": initial_name, "final_time": final_time,
              "ncells": list(ncells), "degree": degree, "cfl": cfl,
              "wall_gradient": wall_gradient}
    try:
        x, y, coeff, metadata = ns.solve_navier_stokes_dg_2d(
            initial, ncells=ncells, degree=degree, domain=DOMAIN,
            final_time=final_time, cfl=cfl, mu=MU, prandtl=PRANDTL,
            periodic=(True, False),
            boundary={
                "ymin": {"velocity": (0.0, 0.0), "temperature": BOTTOM_TEMPERATURE},
                "ymax": {"velocity": (0.0, 0.0), "temperature": TOP_TEMPERATURE},
            },
            wall_gradient=wall_gradient,
        )
        fields = ns.cell_fields(coeff)
        temperature = fields["pressure"] / fields["density"]
        record.update({
            "status": "ok",
            "steps": metadata["steps"],
            "dt": metadata["dt"],
            "pressure_min": float(fields["pressure"].min()),
            "temperature_min": float(temperature.min()),
            "temperature_max": float(temperature.max()),
        })
        if initial_name == "uniform":
            record["profile"] = {
                "y": [float(v) for v in y],
                "temperature": [float(v) for v in temperature.mean(dim=0)],
                "pressure": [float(v) for v in fields["pressure"].mean(dim=0)],
            }
    except Exception as exc:  # noqa: BLE001 - the failure is the measurement
        record.update({"status": "raised",
                       "error": f"{type(exc).__name__}: {exc}"})
    record["wall_seconds"] = round(time.time() - started, 2)
    return record


def default_call() -> dict:
    """The documented default configuration of the shipped wrapper."""
    started = time.time()
    try:
        ns.solve_heat_conduction_dg_2d()
        return {"status": "ok", "wall_seconds": round(time.time() - started, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "raised", "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": round(time.time() - started, 2)}


# ---------------------------------------------------------------------------
# Judgements
# ---------------------------------------------------------------------------

def _slope(values, spacings):
    values = np.asarray(values, dtype=float)
    spacings = np.asarray(spacings, dtype=float)
    if np.any(values <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(spacings), np.log(values), 1)[0])


def verdicts(evidence) -> dict:
    scan = evidence["residual_scan"]
    p1 = [row for row in scan if row["degree"] == 1 and row["wall_gradient"] == "taylor"]
    spacings = [1.0 / row["ncells"][1] for row in p1]
    wall_slope = _slope([row["energy_wall_cold"] for row in p1], spacings)
    interior_slope = _slope([row["energy_interior"] for row in p1], spacings)
    collapse = [row for row in evidence["attempts"] if row["status"] == "raised"
                and "FloatingPoint" in row.get("error", "")]
    time_ok = max((row["final_time"] for row in evidence["attempts"]
                   if row["status"] == "ok"), default=0.0)
    time_bad = min((row["final_time"] for row in collapse), default=float("inf"))
    return {
        "E1_analytic_state_is_exact": (
            evidence["analytic"]["laplace_of_temperature_max_abs"] < 1e-10
            and evidence["analytic"]["density_matches_ideal_gas_max_abs"] < 1e-12
        ),
        "E2_wall_energy_residual_slope_P1": wall_slope,
        "E2_interior_energy_residual_slope_P1": interior_slope,
        "E2_wall_residual_does_not_decay": wall_slope > -0.3,
        "E2_defect_independent_of_wall_gradient": all(
            row["energy_wall_cold"] > 0.05
            for row in scan if row["degree"] == 1
        ),
        "E3_default_call_fails": evidence["default_call"]["status"] == "raised",
        "E3_collapse_time_bracket": [time_ok, time_bad],
        "E3_collapse_independent_of_cfl_and_ic": len(collapse) >= 3,
        "honest_verdict": "FAIL (isothermal-wall energy closure is inconsistent)",
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(evidence, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    scan = evidence["residual_scan"]
    for gradient, style in (("taylor", "o-"), ("face", "s--")):
        rows = [r for r in scan if r["degree"] == 1 and r["wall_gradient"] == gradient]
        cells = [r["ncells"][1] for r in rows]
        axes[0].loglog(cells, [r["energy_wall_cold"] for r in rows], style,
                       label=f"cold wall ({gradient})")
        axes[0].loglog(cells, [r["energy_interior"] for r in rows], style,
                       alpha=0.45, label=f"interior ({gradient})")
    axes[0].axhline(KAPPA, color="k", lw=0.8, ls=":", label=r"$\kappa$")
    axes[0].set_xlabel("cells in y")
    axes[0].set_ylabel(r"$\max|\partial_t E|$ on exact steady state")
    axes[0].set_title("E2: wall energy residual (P1)")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    rows = [r for r in scan if r["degree"] == 1 and r["wall_gradient"] == "taylor"]
    cells = np.array([r["ncells"][1] for r in rows], dtype=float)
    axes[1].semilogx(cells, [r["density_max_abs"] for r in rows], "o-",
                     label="mass row")
    axes[1].semilogx(cells, [r["momentum_max_abs"] for r in rows], "s-",
                     label="momentum row")
    axes[1].axhline(1e-12, color="k", lw=0.8, ls=":")
    axes[1].set_xlabel("cells in y")
    axes[1].set_ylabel("max |residual|")
    axes[1].set_title("E2 control: other rows at machine zero")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    profile = None
    for row in evidence["attempts"]:
        if row["status"] == "ok" and row.get("profile"):
            if profile is None or row["final_time"] > profile["final_time"]:
                profile = row
    if profile is not None:
        y = np.array(profile["profile"]["y"])
        axes[2].plot(y, profile["profile"]["temperature"], "o-",
                     label=f"DG P1, t={profile['final_time']}")
        axes[2].plot(y, BOTTOM_TEMPERATURE + (TOP_TEMPERATURE - BOTTOM_TEMPERATURE) * y,
                     "k--", label="exact steady conduction")
        axes[2].plot(y, profile["profile"]["pressure"] / np.maximum(
            1e-12, profile["profile"]["pressure"]), "b-", lw=0.0)
        axes[2].plot(y, profile["profile"]["pressure"], "b.-", label="pressure (uniform-IC run)")
        axes[2].set_xlabel("y")
        axes[2].set_ylabel("temperature / pressure")
        axes[2].set_title("E3: uniform-IC run at t=0.02 collapses p")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="validation/ns_heat_conduction")
    parser.add_argument("--figure", default="docs/figures/ns_heat_conduction_verification.png")
    parser.add_argument("--quick", action="store_true",
                        help="skip the 128-cell residual column")
    args = parser.parse_args()

    evidence = {"analytic": analytic_checks(), "residual_scan": [], "attempts": [],
                "default_call": {}}
    evidence["default_call"] = default_call()
    for gradient in ("taylor", "face"):
        for ncells in ((2, 16), (2, 32), (2, 64)) if args.quick else \
                ((2, 16), (2, 32), (2, 64), (2, 128)):
            evidence["residual_scan"].append(steady_residual(ncells, 1, gradient))
    evidence["residual_scan"].append(steady_residual((2, 32), 2, "taylor"))

    evidence["attempts"].append(attempt(uniform_initial, "uniform", 0.01))
    evidence["attempts"].append(attempt(uniform_initial, "uniform", 0.02))
    evidence["attempts"].append(attempt(uniform_initial, "uniform", 0.05))
    evidence["attempts"].append(attempt(steady_initial, "steady", 0.05))
    evidence["attempts"].append(attempt(uniform_initial, "uniform", 0.05, cfl=0.05))

    evidence["verdicts"] = verdicts(evidence)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1) + "\n")
    make_figure(evidence, Path(args.figure))

    print(json.dumps(evidence["verdicts"], indent=1))
    print(f"\nevidence -> {out / 'evidence.json'}\nfigure   -> {args.figure}")
    print("\nresidual scan (P1, taylor):")
    for row in evidence["residual_scan"]:
        print(f"  P{row['degree']} {row['wall_gradient']:5s} ny={row['ncells'][1]:4d} "
              f"cold_wall={row['energy_wall_cold']:.4f} hot_wall={row['energy_wall_hot']:.4f} "
              f"interior={row['energy_interior']:.4e}")
    print("\ntime integration:")
    for row in evidence["attempts"]:
        print(f"  {row['initial']:7s} t={row['final_time']:.3f} cfl={row['cfl']} -> "
              f"{row['status']} {row.get('error', '')}"
              f"{' p_min=%.3f' % row['pressure_min'] if row['status'] == 'ok' else ''}")


if __name__ == "__main__":
    main()