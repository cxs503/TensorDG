"""Wave-configuration coverage audit of the exact Riemann baseline and the DG solver.

Motivation (verification debt)
------------------------------
``docs/entropy_verification.md`` §5 honestly records that the analytic baseline
used by the TensorDG verification suite only covers the *right-shock* family
(``p* > p_R``), of which Sod and Lax are members, and that *left-shock* and
*double-rarefaction* configurations were never exercised.  Likewise the flux
kernel audit (``docs/euler_flux_kernel_verification.md``) only samples the two
textbook cases.  This tool closes that debt: it re-verifies
``benchmarks/riemann_exact.py`` and the modal DG/HLLC/SSP-RK3 solver on all four
wave patterns a 1-D Riemann problem can produce.

The five configurations used here are Toro's standard test set (Toro, *Riemann
Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009, ch. 4), each
selected for one wave pattern:

======  ==================  ==================================  ===============
name    left | right        wave pattern                        family
======  ==================  ==================================  ===============
sod     (1,0,1) | (.125,0,.1)   left rarefaction + right shock  right shock
revsod  (.125,0,.1) | (1,0,1)   left shock + right rarefaction  **left shock**
123     (1,-2,.4) | (1,2,.4)    two rarefactions, near vacuum   **double rarefaction**
coll    (1,0,1000) | (1,0,.01) two shocks (strong collision)    **two shocks**
cont    (1,2,1) | (.5,2,1)       isolated contact discontinuity   contact only
======  ==================  ==================================  ===============

Evidence levels produced
------------------------
A. **Analytic invariants of the exact solution** -- falsifiable without any
   reference solution and independent of the Newton iteration inside
   ``star_state``:
   * the integral form of the Euler equations (initial data plus the
     analytic boundary-flux budget) for mass, momentum and energy,
   * Rankine-Hugoniot residuals (mass/momentum/energy) across every detected
     shock,
   * isentropic + Riemann-invariant constancy inside every rarefaction fan,
   * wave-confinement (no characteristic leaves the computational domain
     before ``final_time``, so periodic/transmissive ends cannot pollute the
     comparison).
B. **Literature star states** (Toro 2009, ch. 4) for the right-shock case (Sod)
   and the near-vacuum double rarefaction (123).
C. **Functionality of the DG solver on the newly covered configurations**:
   relative L2 errors against the exact solution, positivity of the polynomial
   state, and grid convergence of the error.
D. **Reflection invariance**: reflecting ``x -> 1-x`` maps configuration
   ``(L, R)`` onto ``(R, L)``.  Density and pressure are even under that map, so
   a correct scheme must return identical error norms for ``sod`` and
   ``revsod``.  Any asymmetry (a common limiter/boundary bug) shows up here.

Run:  PYTHONPATH=. python tools/verify_riemann_coverage.py
Writes ``runs/riemann_coverage.json`` and ``runs/riemann_coverage.png``.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

GAMMA = 1.4
DOMAIN = (0.0, 1.0)
DIAPHRAGM = 0.5

CONFIGS = {
    "sod": {
        "left": (1.0, 0.0, 1.0), "right": (0.125, 0.0, 0.1), "final_time": 0.13,
        "pattern": "left rarefaction + right shock",
        "family": "right shock (previously covered)",
    },
    "revsod": {
        "left": (0.125, 0.0, 0.1), "right": (1.0, 0.0, 1.0), "final_time": 0.13,
        "pattern": "left shock + right rarefaction",
        "family": "left shock (NEW)",
    },
    "123": {
        "left": (1.0, -2.0, 0.4), "right": (1.0, 2.0, 0.4), "final_time": 0.15,
        "pattern": "two rarefactions (near vacuum)",
        "family": "double rarefaction (NEW)",
    },
    "coll": {
        "left": (1.0, 0.0, 1000.0), "right": (1.0, 0.0, 0.01), "final_time": 0.008,
        "pattern": "two shocks (strong collision)",
        "family": "two shocks (NEW)",
    },
    "cont": {
        "left": (1.0, 2.0, 1.0), "right": (0.5, 2.0, 1.0), "final_time": 0.10,
        "pattern": "isolated contact discontinuity",
        "family": "contact only (NEW)",
    },
}

# Toro (2009) tabulated star states for the two cases with a published value.
# Only 3-5 significant digits are quoted in the book, so the tolerance below is
# a citation-accuracy tolerance, not a solver tolerance.
# Cost-bounded solver matrix (each 1-D run costs ~0.5 min on a loaded host).
# ``no_limiter`` and ``weno`` are carried only where they answer a question:
# reflection invariance for the soda pair, and near-vacuum robustness for 123.
DG_MATRIX = {
    "sod": (None, "tvb", "weno"),
    "revsod": (None, "tvb", "weno"),
    "123": (None, "tvb"),
    "coll": ("tvb",),
    "cont": ("tvb",),
}

LITERATURE = {
    "sod": {"pressure": 0.30313, "velocity": 0.92745, "tol": 5e-4},
    "123": {"pressure": 0.00189, "velocity": 0.0, "tol": 1e-2},
}


# --------------------------------------------------------------------------- #
# exact-solution helpers (independent of tensordg)
# --------------------------------------------------------------------------- #
def primitive(state):
    """(rho, u, p) -> conserved (rho, rho u, E)."""
    rho, u, p = state
    return np.array([rho, rho * u, p / (GAMMA - 1.0) + 0.5 * rho * u * u])


def integrate_exact(field, npts=400001, domain=None):
    """Mid-point integration of a sampled exact field over ``DOMAIN``."""
    lower, upper = DOMAIN if domain is None else domain
    x = np.linspace(lower, upper, npts, endpoint=False) + 0.5 * (upper - lower) / npts
    return x, field(x)


def integral_balance(sample, left, right, final_time, domain=None, npts=400001):
    """Integral form of the Euler equations for the sampled exact solution.

    The naive criterion "the space integral of the conserved variables equals
    its initial value" is **wrong** for a Riemann problem: the ends are open, so
    the momentum integral grows by ``(p_L - p_R) * t`` and the mass/energy
    integrals by the corresponding boundary fluxes.  For Sod at ``t = 0.13``
    that is a 0.117 momentum gain -- an order-one effect that a "conservation"
    test would report as a catastrophic error of a perfectly correct exact
    solution.  The budgeted statement is

        int U(x, T) dx = int U(x, 0) dx + T * (F(U_L) - F(U_R)),

    which holds exactly because the ends stay in their constant states (checked
    separately by :func:`wave_confinement`), and which does test the branch
    logic of :func:`sample`: any wrong region/polarity perturbs the integral at
    the O(1) relative level, far above the quadrature error.

    The residual is normalised by ``int |U| dx`` (the transported magnitude),
    not by the initial integral, so zero-momentum initial states do not produce
    a division by zero.
    """
    lower, upper = DOMAIN if domain is None else domain
    x, fields = integrate_exact(
        lambda pts: sample(pts, final_time, left, right, DIAPHRAGM), npts,
        domain=(lower, upper),
    )
    rho, u, p = fields["density"], fields["velocity"], fields["pressure"]
    momentum = rho * u
    energy = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    dx = (upper - lower) / npts
    numeric = np.array([rho.sum(), momentum.sum(), energy.sum()]) * dx

    def flux(state):
        r, v, q = state
        return np.array([
            r * v,
            r * v * v + q,
            v * (q / (GAMMA - 1.0) + 0.5 * r * v * v + q),
        ])

    initial = 0.5 * (primitive(left) + primitive(right)) * (upper - lower)
    budget = final_time * (flux(left) - flux(right))
    predicted = initial + budget
    magnitude = np.array([
        np.abs(rho).sum(), np.abs(momentum).sum(), np.abs(energy).sum(),
    ]) * dx
    residual = numeric - predicted
    relative = np.abs(residual) / np.maximum(magnitude, 1e-30)
    return {
        "numeric": numeric.tolist(),
        "initial": initial.tolist(),
        "boundary_budget": budget.tolist(),
        "predicted": predicted.tolist(),
        "residual": residual.tolist(),
        "scale": magnitude.tolist(),
        "relative_error": relative.tolist(),
        "max_relative_error": float(relative.max()),
    }


def _find_jumps(values, threshold_ratio=0.05):
    """Return indices where adjacent samples differ by more than a fraction of the range."""
    span = values.max() - values.min()
    if span <= 0.0:
        return []
    jump = np.abs(np.diff(values))
    return [int(i) for i in np.flatnonzero(jump > threshold_ratio * span)]


def shock_rankine_hugoniot(x, fields, left, right, final_time, threshold_ratio=0.02):
    """Rankine-Hugoniot residuals across every detected shock.

    A genuine discontinuity satisfies all three jump relations simultaneously
    with the *same* speed ``sigma`` estimated from the mass balance.  A fan
    edge or a contact discontinuity violates the momentum/energy relations, so
    the detector below only reports jumps whose mass-derived speed also closes
    momentum and energy -- those are shocks, and the residuals are the
    verification number.
    """
    rho, u, p = fields["density"], fields["velocity"], fields["pressure"]
    momentum = rho * u
    energy = p / (GAMMA - 1.0) + 0.5 * rho * u * u
    total = p + rho * u * u
    indices = _find_jumps(rho, threshold_ratio)
    shocks = []
    for i in indices:
        a, b = i, i + 1
        mass_jump = rho[b] - rho[a]
        if abs(mass_jump) < 1e-12:
            continue
        sigma = (momentum[b] - momentum[a]) / mass_jump
        mass_res = sigma * mass_jump - (momentum[b] - momentum[a])
        mom_res = sigma * (momentum[b] - momentum[a]) - (total[b] - total[a])
        eng_res = sigma * (energy[b] - energy[a]) - (
            u[b] * (energy[b] + p[b]) - u[a] * (energy[a] + p[a])
        )
        scale = max(abs(sigma * mass_jump), abs(total[b] - total[a]),
                    abs(total[a]), abs(total[b]), 1e-30)
        closure = max(abs(mom_res), abs(eng_res)) / scale
        shocks.append({
            "cell_index": int(a),
            "position": float(x[a]),
            "speed": float(sigma),
            "mass_residual": float(mass_res),
            "momentum_residual": float(mom_res),
            "energy_residual": float(eng_res),
            "normalised_closure": float(closure),
        })
    genuine = [s for s in shocks if s["normalised_closure"] < 1e-2]
    return {"candidates": shocks, "shocks": genuine}


def fan_invariants(x, fields, left, right):
    """Entropy and Riemann-invariant constancy inside rarefaction fans.

    A simple (centred) rarefaction carries constant entropy ``ln(p/rho**gamma)``
    and constant ``u + 2a/(gamma-1)`` (right-facing) or ``u - 2a/(gamma-1)``
    (left-facing).  Smoothly varying regions that are *not* fans (shock
    structure, contact smearing) show large deviations, so this doubles as a
    discriminator between fan samples and the rest.
    """
    rho, u, p = fields["density"], fields["velocity"], fields["pressure"]
    sound = np.sqrt(GAMMA * p / rho)
    entropy = np.log(p / rho ** GAMMA)
    invariants = {
        "left_facing": u - 2.0 * sound / (GAMMA - 1.0),
        "right_facing": u + 2.0 * sound / (GAMMA - 1.0),
    }
    gradient = np.abs(np.gradient(entropy, x))
    smooth = gradient < 1e-6
    report = {}
    for name, values in invariants.items():
        grad = np.abs(np.gradient(values, x))
        # fan samples: entropy-gradient free and invariant-gradient free
        fan = smooth & (grad < 1e-8)
        count = int(fan.sum())
        if count < 10:
            report[name] = {"samples": count, "spread": None}
            continue
        report[name] = {
            "samples": count,
            "spread": float(values[fan].max() - values[fan].min()),
            "level": float(np.median(values[fan])),
        }
    return report


def wave_confinement(left, right, final_time):
    """Check that no characteristic reaches an end of ``DOMAIN`` before ``final_time``."""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = math.sqrt(GAMMA * p_l / rho_l)
    a_r = math.sqrt(GAMMA * p_r / rho_r)
    return {
        "leftmost_speed": float(u_l - a_l),
        "rightmost_speed": float(u_r + a_r),
        "left_reach": float(DIAPHRAGM - (u_l - a_l) * final_time),
        "right_reach": float(DIAPHRAGM + (u_r + a_r) * final_time),
        "confined": bool(
            DIAPHRAGM - (u_l - a_l) * final_time > DOMAIN[0]
            and DIAPHRAGM + (u_r + a_r) * final_time < DOMAIN[1]
        ),
    }


# --------------------------------------------------------------------------- #
# solver runs
# --------------------------------------------------------------------------- #
def dg_run(config, ncells=200, degree=2, cfl=0.1, limiter="tvb", tvb_constant=1.0,
           limiter_stage="every"):
    import torch

    from benchmarks.riemann_dg_error import riemann_dg_errors
    from benchmarks.riemann_exact import sample
    from tensordg.euler_dg_1d import solve_euler_dg_1d

    attempt = {
        "ncells": ncells, "degree": degree, "cfl": cfl, "limiter": limiter,
        "limiter_stage": limiter_stage, "tvb_constant": tvb_constant,
        "final_time": config["final_time"],
    }
    try:
        x, coeff, meta = solve_euler_dg_1d(
            config["left"], config["right"], ncells=ncells, degree=degree,
            final_time=config["final_time"], diaphragm=DIAPHRAGM, domain=DOMAIN,
            cfl=cfl, limiter=limiter, tvb_constant=tvb_constant,
            limiter_stage=limiter_stage,
        )
    except FloatingPointError as exc:
        attempt.update({"status": "NONPHYSICAL", "error": str(exc)})
        return attempt
    errors = riemann_dg_errors(
        x, coeff, config["final_time"], config["left"], config["right"], DIAPHRAGM,
    )
    values = coeff.detach().cpu().numpy()
    average = values[:, 0]
    pressure_avg = (GAMMA - 1.0) * (average[:, 2] - 0.5 * average[:, 1] ** 2 / average[:, 0])
    attempt.update({
        "status": "ok",
        "steps": meta["steps"],
        "dt": meta["dt"],
        "relative_l2_percent": {
            key: errors[key]["relative_l2_percent"] for key in errors
        },
        "min_cell_density": float(average[:, 0].min()),
        "min_cell_pressure": float(pressure_avg.min()),
        "star_state": {
            "pressure": float(sample(np.array([DIAPHRAGM]), config["final_time"],
                                     config["left"], config["right"], DIAPHRAGM)["pressure"][0]),
        },
    })
    return attempt


def literature_check(report, config_meta):
    p_star, u_star = _star_state(config_meta["left"], config_meta["right"])
    report["computed_pressure"] = float(p_star)
    report["computed_velocity"] = float(u_star)
    report["pressure_relative_deviation"] = float(
        abs(p_star - report["literature_pressure"]) / report["literature_pressure"]
    )
    velocity_scale = max(abs(report["literature_velocity"]), 1e-3)
    report["velocity_absolute_deviation"] = float(
        abs(u_star - report["literature_velocity"]) / velocity_scale
    )
    report["pass"] = bool(
        report["pressure_relative_deviation"] <= report["tolerance"]
        and report["velocity_absolute_deviation"] <= report["tolerance"]
    )
    return report


def _star_state(left, right):
    from benchmarks.riemann_exact import star_state
    return star_state(left, right)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main():
    from benchmarks.riemann_exact import sample

    os.makedirs("runs", exist_ok=True)
    output = {"gamma": GAMMA, "domain": DOMAIN, "diaphragm": DIAPHRAGM, "configs": {}}

    for name, config in CONFIGS.items():
        print(f"[config] {name}: {config['pattern']}", flush=True)
        x = np.linspace(DOMAIN[0], DOMAIN[1], 200001)
        fields = sample(x, config["final_time"], config["left"], config["right"], DIAPHRAGM)
        entry = {
            "pattern": config["pattern"],
            "family": config["family"],
            "left": config["left"],
            "right": config["right"],
            "final_time": config["final_time"],
            "wave_confinement": wave_confinement(
                config["left"], config["right"], config["final_time"],
            ),
            "integral_balance": integral_balance(
                sample, config["left"], config["right"], config["final_time"],
            ),
            "rankine_hugoniot": shock_rankine_hugoniot(
                x, fields, config["left"], config["right"], config["final_time"],
            ),
            "fans": fan_invariants(x, fields, config["left"], config["right"]),
        }
        if name in LITERATURE:
            literature = LITERATURE[name]
            entry["literature"] = literature_check(
                {
                    "literature_pressure": literature["pressure"],
                    "literature_velocity": literature["velocity"],
                    "tolerance": literature["tol"],
                },
                config,
            )
        limiters = DG_MATRIX[name]
        entry["dg"] = {}
        for limiter in limiters:
            key = "no_limiter" if limiter is None else limiter
            entry["dg"][key] = dg_run(config, limiter=limiter)
            result = entry["dg"][key]
            detail = result.get("relative_l2_percent")
            message = (
                "err_rho={:.4f}%".format(detail["density"]) if detail else result["status"]
            )
            print(f"  [dg] {name:7s} {key:10s} {message}", flush=True)
        output["configs"][name] = entry

    # reflection invariance: sod <-> revsod must give identical error norms
    mirror = {}
    for key in ("no_limiter", "tvb", "weno"):
        sod = output["configs"]["sod"]["dg"].get(key)
        rev = output["configs"]["revsod"]["dg"].get(key)
        if sod is None or rev is None:
            continue
        if sod["status"] != "ok" or rev["status"] != "ok":
            mirror[key] = {"status": "unavailable"}
            continue
        differences = {}
        for field in ("density", "pressure"):
            a = sod["relative_l2_percent"][field]
            b = rev["relative_l2_percent"][field]
            differences[field] = {
                "sod": a, "revsod": b,
                "absolute_difference": abs(a - b),
                "relative_asymmetry": abs(a - b) / max(a, b, 1e-30),
            }
        mirror[key] = differences
        # velocity is odd under reflection: compare its absolute L2 magnitude
        mirror[key]["velocity"] = {
            "sod_l2": sod["relative_l2_percent"]["velocity"],
            "revsod_l2": rev["relative_l2_percent"]["velocity"],
        }
    output["reflection_invariance"] = mirror

    # grid convergence of the newly covered configurations (tvb limiter, P2)
    convergence = {}
    for name in ("revsod", "123", "coll", "cont"):
        print(f"[convergence] {name}", flush=True)
        config = CONFIGS[name]
        series = []
        for ncells in (100, 200):
            run = dg_run(config, ncells=ncells, limiter="tvb")
            series.append({"ncells": ncells, **{
                key: run.get(key) for key in
                ("status", "relative_l2_percent", "min_cell_density", "min_cell_pressure")
            }})
        finite = [s for s in series if s["status"] == "ok"]
        trends = {}
        for field in ("density", "velocity", "pressure"):
            values = [s["relative_l2_percent"][field] for s in finite]
            trends[field] = {
                "values": values,
                "ratio_fine_over_coarse": (values[-1] / values[0]) if len(values) == 2 else None,
            }
        convergence[name] = {"series": series, "trends": trends}
    output["convergence"] = convergence

    with open("runs/riemann_coverage.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(_table(output))
    return output


# --------------------------------------------------------------------------- #
# troubled-cell indicator symmetry
# --------------------------------------------------------------------------- #
def indicator_flags(cell_averages, spacing, tvb_constant=1.0, symmetric=False):
    """Troubled-cell flags of ``tensordg.tvb`` for one scalar cell-average field."""
    import torch

    from tensordg.tvb import troubled_from_differences

    average = torch.as_tensor(np.ascontiguousarray(cell_averages), dtype=torch.float64)[:, None]
    padded = torch.cat([average[:1], average, average[-1:]], dim=0)
    d_left = average - padded[:-2]
    d_right = padded[2:] - average
    troubled, _ = troubled_from_differences(
        d_left, d_right, spacing, tvb_constant, symmetric_indicator=symmetric,
    )
    return troubled.numpy()


def indicator_mirror_probe(ncells=64, tvb_constant=1.0):
    """Mirror symmetry of the TVB troubled-cell indicator.

    Reflection ``i -> N-1-i`` maps ``d_left`` onto ``-d_right``.  The historic
    deviation ``|d_left - minmod|`` is *not* invariant under that map, so the
    flagged set is not mirror-symmetric; the symmetric variant uses
    ``max(|d_left - mm|, |d_right - mm|)`` and is invariant by construction.
    Two probes: a smooth rarefaction-like profile (where the indicator is
    marginal) and a single jump (where it is not).
    """
    x = (np.arange(ncells) + 0.5) / ncells
    spacing = 1.0 / ncells
    probes = {
        "smooth": 1.0 - 0.35 * np.tanh((x - 0.35) / 0.15) + 0.05 * np.sin(4.0 * x),
        "jump": np.where(x < 0.5, 1.0, 0.125),
    }
    report = {}
    for name, values in probes.items():
        entry = {}
        for variant, symmetric in (("historic", False), ("symmetric", True)):
            forward = indicator_flags(values, spacing, tvb_constant, symmetric)
            backward = indicator_flags(values[::-1], spacing, tvb_constant, symmetric)
            entry[variant] = {
                "flagged_forward": int(forward.sum()),
                "flagged_mirrored": int(backward.sum()),
                "mirror_mismatches": int((forward != backward[::-1]).sum()),
            }
        report[name] = entry
    return report


def indicator_symmetry_effect(config_pair=("sod", "revsod"), ncells=100,
                              final_time=0.05, cfl=0.1):
    """Does opting into the symmetric indicator restore solver mirror symmetry?

    Both configurations are integrated with the same parameters; only the
    indicator variant differs.  Density and pressure are even under reflection,
    so their relative L2 errors must coincide; the historic indicator leaves a
    residual asymmetry that quantifies the modelling bias.
    """
    import torch

    from benchmarks.riemann_dg_error import riemann_dg_errors
    from tensordg.euler_dg_1d import solve_euler_dg_1d
    from tensordg.limiters import apply_limiter

    results = {}
    for variant, symmetric in (("historic", False), ("symmetric", True)):
        errors = {}
        for name in config_pair:
            config = CONFIGS[name]
            original = apply_limiter

            def patched(coeff, dx, kind=None, **kwargs):
                if kind in (None, "none"):
                    return original(coeff, dx, kind=kind, **kwargs)
                kwargs["symmetric_indicator"] = symmetric
                return original(coeff, dx, kind=kind, **kwargs)

            import tensordg.euler_dg_1d as solver_module
            saved = solver_module.apply_limiter
            solver_module.apply_limiter = patched
            try:
                _, coeff, _ = solve_euler_dg_1d(
                    config["left"], config["right"], ncells=ncells, degree=2,
                    final_time=final_time, diaphragm=DIAPHRAGM, domain=DOMAIN,
                    cfl=cfl, limiter="tvb", tvb_constant=1.0,
                )
            finally:
                solver_module.apply_limiter = saved
            errors[name] = riemann_dg_errors(
                torch.arange(ncells) / ncells + 0.5 / ncells, coeff, final_time,
                config["left"], config["right"], DIAPHRAGM,
            )
        entry = {}
        for field in ("density", "pressure", "velocity"):
            a = errors[config_pair[0]][field]["relative_l2_percent"]
            b = errors[config_pair[1]][field]["relative_l2_percent"]
            entry[field] = {
                config_pair[0]: a, config_pair[1]: b,
                "absolute_difference": abs(a - b),
                "relative_asymmetry": abs(a - b) / max(a, b, 1e-30),
            }
        results[variant] = entry
    return results


def _table(output):
    lines = []
    header = f"{'config':8s} {'family':34s} {'max|dU|/|U0|':>13s} {'shocks':>7s} {'P2 err %rho':>12s} {'min rho':>9s}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, entry in output["configs"].items():
        dg = entry["dg"]["tvb"]
        err = dg.get("relative_l2_percent", {}).get("density")
        err_text = "{:12.4f}".format(err) if err is not None else "{:>12s}".format(dg["status"])
        min_rho = dg.get("min_cell_density", float("nan"))
        lines.append(
            "{:8s} {:34s} {:13.3e} {:7d} {:12s} {:9.4f}".format(
                name, entry["family"],
                entry["integral_balance"]["max_relative_error"],
                len(entry["rankine_hugoniot"]["shocks"]),
                err_text, min_rho,
            )
        )
    return "\n".join(lines)


def supplement():
    """Add the indicator-symmetry evidence to an existing ``runs/riemann_coverage.json``."""
    path = "runs/riemann_coverage.json"
    with open(path, "r", encoding="utf-8") as handle:
        output = json.load(handle)
    output["indicator_symmetry"] = indicator_mirror_probe()
    print("[probe] indicator symmetry done", flush=True)
    output["indicator_symmetry"]["solver_effect"] = indicator_symmetry_effect()
    print("[probe] solver effect done", flush=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    return output


if __name__ == "__main__":
    if "--supplement" in sys.argv:
        supplement()
        sys.exit(0)
    sys.exit(0 if main() is not None else 1)