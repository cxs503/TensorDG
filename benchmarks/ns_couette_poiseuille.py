"""Verification of the body-force (``source``) pathway of ``tensordg.ns_dg_2d``.

Case
----
Plane Poiseuille channel flow driven by a constant volumetric force ``S`` in
``+x``, periodic in ``x``, no-slip isothermal walls at ``y = 0`` and ``y = H``.
The exact steady solution is closed-form (:class:`tensordg.ns_poiseuille.
BodyForcePoiseuille`): a parabolic velocity profile plus the Brinkman quartic
temperature profile produced by ``k T'' = S u - mu u'^2``.  It is *not* read from
literature: every number is generated from the polynomial double integration in
``tensordg/ns_poiseuille.py``, so the comparison is a genuine PDE verification.

Why this case
-------------
``solve_navier_stokes_dg_2d`` grew a ``source`` hook in m2789 but no test or
benchmark exercised it (``grep -rn "source=" tests/test_ns_*.py`` was empty
before this mission), so the sign, the units and the quadrature placement of the
source projection were unverified.  The case also pins down the ``u tau`` work
term of the viscous energy flux, because the temperature profile only comes out
right when the force work ``S u`` and the dissipation ``mu u'^2`` balance as in
the continuous equations.

Usage
-----
``PYTHONPATH=. python -m benchmarks.ns_couette_poiseuille [--quick]``

Writes ``validation/ns_couette_poiseuille/metrics.json`` and
``docs/figures/ns_couette_poiseuille.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import torch

from tensordg import ns_dg_2d as ns
from tensordg.ns_poiseuille import BodyForcePoiseuille

ROOT = pathlib.Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "validation" / "ns_couette_poiseuille" / "metrics.json"
FIGURE_PATH = ROOT / "docs" / "figures" / "ns_couette_poiseuille.png"

CASE = dict(force=0.2, mu=0.1, prandtl=0.71, pressure=1.0, wall_temperature=1.0,
            ymin=0.0, ymax=1.0)
DOMAIN_X = (0.0, 0.5)          # two cells; the case is x-independent
NX = 2


def rest_initial(pressure):
    def initial(x, y):
        return (torch.ones_like(x), torch.zeros_like(x), torch.zeros_like(x),
                pressure * torch.ones_like(x))

    return initial


def quadrature_profile(coeff, case, ncells, degree):
    """Primitive fields at the Gauss points, plus their ``y`` coordinates.

    The mode layout of ``coeff`` is ``(nx, ny, k_x, k_y, variable)``, so the
    third tensor axis is the ``x`` quadrature node and the fourth is the ``y``
    node (as in ``ns_dg_2d``'s own ``x_quad``/``y_quad`` construction).
    """
    order = max(degree + 2, 3)
    nodes, weights, basis, _, _ = ns.legendre_operators(degree, order, "cpu", torch.float64)
    nx, ny = ncells
    dy = case.height / ny
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho = values[..., 0].clamp_min(1e-30)
    energy = values[..., 3]
    pressure = (ns.GAMMA - 1.0) * (
        energy - 0.5 * (values[..., 1] ** 2 + values[..., 2] ** 2) / rho
    )
    y = case.ymin + (torch.arange(ny, dtype=torch.float64).view(1, ny, 1, 1)
                     + 0.5 * (1.0 + nodes.view(1, 1, 1, -1))) * dy
    y = y.expand(nx, ny, nodes.numel(), nodes.numel())
    return {
        "velocity_x": values[..., 1] / rho,
        "velocity_y": values[..., 2] / rho,
        "pressure": pressure,
        "temperature": pressure / rho,
        "density": rho,
        "y": y,
        "weights": weights,
    }


def relative_errors(profile, case):
    """Weighted relative L2 and L-infinity errors of the three fields."""
    w = profile["weights"].view(1, 1, -1, 1) * profile["weights"].view(1, 1, 1, -1)
    out = {}
    for key, exact in (
        ("velocity_x", case.velocity(profile["y"].numpy())),
        ("temperature", case.temperature(profile["y"].numpy())),
        ("density", case.density(profile["y"].numpy())),
    ):
        numeric = profile[key]
        exact = torch.as_tensor(exact, dtype=numeric.dtype)
        weight = w.expand_as(numeric)
        scale = math.sqrt(float((weight * exact**2).sum()) / float(weight.sum()))
        l2 = math.sqrt(float((weight * (numeric - exact) ** 2).sum()) / float(weight.sum()))
        linf = float((numeric - exact).abs().max()) / float(exact.abs().max())
        out[key] = {"relative_l2": l2 / scale, "relative_linf": linf}
    return out


def wall_shear(coeff, case, ncells, degree):
    """Second-order one-sided wall shear from cell-centre values (both walls).

    Uses the same parabola stencil as the solver's ``fd2`` wall closure,
    ``(-8/3 u_w + 3 u_1 - u_2/3)/h`` with ``u_w = 0`` at a no-slip wall, taken
    along the *inward* normal so both walls give the same (positive) value for
    a force in ``+x``; the reported shear is the best estimate the
    discretisation itself offers.
    """
    _, ny = ncells
    h = case.height / ny
    centre = ns._cell_centre_primitives(coeff)[0][0, :]  # (ny,) u_x at cell centres
    bottom = (3.0 * float(centre[0]) - float(centre[1]) / 3.0) / h
    top = (3.0 * float(centre[-1]) - float(centre[-2]) / 3.0) / h
    return bottom, top


def run_case(name, ncells, degree, final_time, initial="exact", cfl_viscous=3.0,
             case_kwargs=None):
    kwargs = dict(CASE)
    kwargs.update(case_kwargs or {})
    case = BodyForcePoiseuille(**kwargs)
    if initial == "exact":
        start = case.initial
    elif initial == "rest":
        start = rest_initial(case.pressure)
    else:
        raise ValueError("initial must be 'exact' or 'rest'")
    boundary = {
        "ymin": {"velocity": (0.0, 0.0), "temperature": case.wall_temperature},
        "ymax": {"velocity": (0.0, 0.0), "temperature": case.wall_temperature},
    }
    started = time.time()
    _, y, coeff, meta = ns.solve_navier_stokes_dg_2d(
        start,
        ncells=ncells,
        degree=degree,
        domain=(DOMAIN_X, (case.ymin, case.ymax)),
        final_time=final_time,
        cfl=0.3,
        cfl_viscous=cfl_viscous,
        mu=case.mu,
        prandtl=case.prandtl,
        periodic=(True, False),
        boundary=boundary,
        source=case.source,
    )
    wall = time.time() - started
    profile = quadrature_profile(coeff, case, ncells, degree)
    errors = relative_errors(profile, case)
    bottom, top = wall_shear(coeff, case, ncells, degree)
    exact_shear = case.wall_slope
    # global force balance: total body force vs the shear on both walls
    force_total = case.force * case.height
    shear_total = case.mu * (bottom + top)
    result = {
        "name": name,
        "ncells": list(ncells),
        "degree": degree,
        "initial": initial,
        "final_time": final_time,
        "steps": meta["steps"],
        "dt": meta["dt"],
        "diffusive_steps": meta["diffusive_steps"],
        "wall_seconds": wall,
        "errors": errors,
        "wall_shear": {
            "bottom_numeric": bottom,
            "top_numeric": top,
            "exact": exact_shear,
            "relative_error_bottom": abs(bottom - exact_shear) / exact_shear,
            "relative_error_top": abs(top - exact_shear) / exact_shear,
        },
        "force_balance": {
            "body_force_total": force_total,
            "wall_shear_total_numeric": shear_total,
            "relative_error": abs(shear_total - force_total) / force_total,
        },
        "mean_velocity": {
            "numeric": float(ns._cell_centre_primitives(coeff)[0][0, :].mean()),
            "exact": 2.0 * case.u_max / 3.0,
        },
        "temperature_drop": {
            "numeric": float(case.wall_temperature - profile["temperature"].min()),
            "exact": float(case.wall_temperature - case.temperature(
                torch.tensor([case.ymin + 0.5 * case.height])).min()),
        },
        "finite": bool(torch.isfinite(coeff).all()),
    }
    return result, case, profile


def _verdict(case, results):
    """3 % gate on the finest P1 mesh and on the P2 run."""
    checks = []
    for result in results:
        worst = max(v["relative_l2"] for v in result["errors"].values())
        checks.append({
            "case": result["name"],
            "worst_relative_l2": worst,
            "below_three_percent": worst < 0.03,
            "finite": result["finite"],
        })
    return checks


def convergence_order(results):
    """P1 order estimate from the mesh-refinement runs of the exact-init set."""
    rows = [r for r in results if r["initial"] == "exact" and r["degree"] == 1]
    rows.sort(key=lambda r: r["ncells"][1])
    table = []
    for coarse, fine in zip(rows, rows[1:]):
        h_coarse = 1.0 / coarse["ncells"][1]
        h_fine = 1.0 / fine["ncells"][1]
        entry = {"coarse": coarse["name"], "fine": fine["name"]}
        for field in ("velocity_x", "temperature"):
            e_c = coarse["errors"][field]["relative_l2"]
            e_f = fine["errors"][field]["relative_l2"]
            entry[field] = math.log(e_c / e_f) / math.log(h_coarse / h_fine) if e_f > 0 else float("nan")
        table.append(entry)
    return table


def plot(results, case):
    """Velocity and temperature profiles at the finest P1 mesh, plus the exact lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finest = max((r for r in results if r["degree"] == 1 and r["initial"] == "exact"),
                 key=lambda r: r["ncells"][1])
    profile = finest["_profile"]
    # fields are (nx, ny, node_x, node_y); average over the (unused) x direction
    y = profile["y"][0, :, 0, :].reshape(-1).numpy()
    u = profile["velocity_x"][0].mean(dim=2).reshape(-1).numpy()
    t = profile["temperature"][0].mean(dim=2).reshape(-1).numpy()
    order = y.size
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(u, y, "o", ms=4.5,
                 label="DG P%d, ny=%d (max err %.2f%%)" % (
                     finest["degree"], finest["ncells"][1],
                     100.0 * finest["errors"]["velocity_x"]["relative_linf"]))
    yy = torch.linspace(case.ymin, case.ymax, 200).numpy()
    axes[0].plot(case.velocity(yy), yy, "-", lw=1.5, label="exact $u=S\,y(H-y)/(2\mu)$")
    axes[0].set_xlabel("$u_x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_title("plane Poiseuille velocity, t = %.1f" % finest["final_time"])
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, y, "s", ms=4.5, color="tab:red",
                 label="DG (max err %.3f%%)" % (
                     100.0 * finest["errors"]["temperature"]["relative_linf"]))
    axes[1].plot(case.temperature(yy), yy, "-", lw=1.5, color="tab:orange",
                 label="exact Brinkman quartic")
    axes[1].set_xlabel("$T$")
    axes[1].set_ylabel("$y$")
    axes[1].set_title("viscous heating: $k T'' = S u - \mu u'^2$")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=140)
    plt.close(figure)
    print("figure ->", FIGURE_PATH.relative_to(ROOT), "(%d points)" % order)
    return FIGURE_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="single short case (test scale)")
    args = parser.parse_args(argv)

    if args.quick:
        specs = [("exact_p1_ny8", (NX, 8), 1, 1.0, "exact")]
    else:
        specs = [
            # from rest: exercises the whole operator (source, walls, fluxes)
            ("rest_p1_ny8", (NX, 8), 1, 8.0, "rest"),
            # from the exact state: the same fixed point is reached without the
            # long diffusive transient, which is what the mesh study needs
            ("exact_p1_ny8", (NX, 8), 1, 0.5, "exact"),
            ("exact_p1_ny12", (NX, 12), 1, 0.5, "exact"),
            ("exact_p1_ny16", (NX, 16), 1, 0.5, "exact"),
            ("exact_p2_ny8", (NX, 8), 2, 0.5, "exact"),
        ]

    results = []
    case = None
    for name, ncells, degree, final_time, initial in specs:
        result, case, profile = run_case(name, ncells, degree, final_time, initial)
        result["profile"] = None
        results.append(result)
        _emit(result)
        if case is not None:
            result["_profile"] = profile

    checks = _verdict(case, results)
    orders = convergence_order(results)
    # does the from-rest march land on the same fixed point as the from-exact run?
    agreement = None
    by_name = {r["name"]: r for r in results}
    if "rest_p1_ny8" in by_name and "exact_p1_ny8" in by_name:
        rest_run = by_name["rest_p1_ny8"]
        exact_run = by_name["exact_p1_ny8"]
        agreement = {
            key: abs(rest_run["errors"][key]["relative_l2"] - exact_run["errors"][key]["relative_l2"])
            for key in rest_run["errors"]
        }

    payload = {
        "mission": "m6454",
        "case": case.case_summary(),
        "domain_x": list(DOMAIN_X),
        "results": [{k: v for k, v in r.items() if k != "_profile"} for r in results],
        "convergence": orders,
        "fixed_point_agreement_rest_vs_exact_ny8": agreement,
        "checks": checks,
        "verdict": {
            "finest_p1_worst_relative_l2": min(
                (c["worst_relative_l2"] for c in checks
                 if "p2" not in c["case"] and "ny16" in c["case"]), default=None),
            "all_below_three_percent": all(c["below_three_percent"] for c in checks),
        },
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2))
    if not args.quick:
        figure = plot(results, case)
        payload["figure"] = str(figure.relative_to(ROOT))
    print("\nconvergence (P1):")
    for row in orders:
        print("  %-14s -> %-14s  u:%5.2f  T:%5.2f" % (
            row["coarse"], row["fine"], row["velocity_x"], row["temperature"]))
    print("\n3%% gate:")
    for check in checks:
        print("  %-14s worst relative L2 = %.4f%%  %s" % (
            check["case"], 100.0 * check["worst_relative_l2"],
            "PASS" if check["below_three_percent"] else "FAIL"))
    print("\nverdict:", payload["verdict"])
    return payload


def _emit(result):
    print("%-14s ncells=%-9s degree=%d steps=%6d dt=%.2e wall=%6.1fs" % (
        result["name"], result["ncells"], result["degree"], result["steps"],
        result["dt"], result["wall_seconds"]))
    for field, value in result["errors"].items():
        print("    %-12s relative L2 = %9.3e (%.4f%%)  Linf = %.4f%%" % (
            field, value["relative_l2"], 100.0 * value["relative_l2"],
            100.0 * value["relative_linf"]))
    print("    wall shear rel. err bottom/top = %.4f%% / %.4f%%; force balance = %.4f%%" % (
        100.0 * result["wall_shear"]["relative_error_bottom"],
        100.0 * result["wall_shear"]["relative_error_top"],
        100.0 * result["force_balance"]["relative_error"]))


if __name__ == "__main__":
    main()