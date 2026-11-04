"""Quantitative verification of the *impulsively started* Couette transient (m3661).

The Dirichlet (wall) path of :mod:`tensordg.ns_dg_2d` was verified for

* the exact **steady** Couette + Brinkman state (wall heat flux, residual),
* a **periodic/decaying shear mode** interior initial condition, and
* survivability of the impulsive start (``t = 0.05``-``0.1``, no reference).

The last item is the one the earlier reports could only bound qualitatively:
``docs/reports/tensordg-ns-wall-scope-audit.md`` (m2786) keeps
"transient accuracy: cannot be given" as its only honest FAIL, because the
impulse start had **no reference solution** attached to it.  This script closes
that debt with the classical Stokes series of the first problem in a channel --
an exact solution of the parallel-flow momentum equation, which the
compressible operator reproduces up to its ``O(mu U^2 / kappa)`` viscous
heating:

    u(y,t) = U s + sum_n 2U(-1)^n/(n pi) sin(n pi s) exp(-n^2 pi^2 nu t / H^2)

with ``u(ymin)=0``, ``u(ymax)=U`` for ``t>0`` and ``u(y,0)=0`` inside.

Machine-checkable criteria (per case and overall):

``profile``   relative L2 error of ``u`` at ``t = T_END`` against the series
              must be below 3 %;
``order``     the error must decay under refinement with the element design
              order minus the parabolic singularity of the initial data
              (the criterion is monotone decay plus a measured slope >= 0.5,
              since the discontinuous start caps the classical order);
``bounds``    the numerical ``u`` must stay inside ``[0, U]`` and monotone in
              ``y`` (an admissibility criterion, independent of the series);
``heating``   ``max|T - T_wall|`` must stay within the analytic dissipation
              scale ``mu U^2 / (8 kappa)`` times a factor 3 (the transient is
              colder than the steady Brinkman profile).

Usage::

    PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette            # all
    PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette --quick    # 2 cases
    PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette \
        --only 1:8 --out validation/ns_dg_2d_impulse/cases                 # one shard
    PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette --merge

The shard mode writes ``<out>/case_p{degree}_ny{ny}.json``; ``--merge``
assembles ``validation/ns_dg_2d_impulse/impulse.json`` and the figure
``docs/figures/ns_dg_2d_impulse_couette.png`` from the per-case files, so a
parameter sweep can be split over several machines without hand-copying
numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import legendre_operators

DTYPE = torch.float64
DEVICE = torch.device("cpu")

# --- physical set-up -------------------------------------------------------
WALL_SPEED = 0.3           # Mach ~ 0.25 at p = rho = T = 1: heating is < 0.2 %
PRESSURE = 1.0
WALL_TEMPERATURE = 1.0
MU = 0.05
PRANDTL = 0.71
FINAL_TIME = 0.2
# rho = p / T = 1 at both walls, so the series viscosity is mu / rho0
NU = MU * WALL_TEMPERATURE / PRESSURE
KAPPA = MU * ns.CP / PRANDTL
HEATING_SCALE = MU * WALL_SPEED ** 2 / (8.0 * KAPPA)   # max of the steady Brinkman bump

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "validation" / "ns_dg_2d_impulse"
FIGURE = ROOT / "docs" / "figures" / "ns_dg_2d_impulse_couette.png"

CASES = ((1, 8), (1, 16), (1, 32), (2, 8), (2, 16))
QUICK_CASES = ((1, 8), (1, 16))


def _quadrature(degree):
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, DEVICE, DTYPE
    )
    return nodes, weights, basis


def _fields_at_nodes(coeff, basis, degree, ncells, domain_height=1.0):
    """Primitive fields on the tensor-product Gauss nodes of every cell."""
    order = basis.shape[0]
    nodes, _ = np.polynomial.legendre.leggauss(order)
    nx, ny = ncells
    dy = domain_height / ny
    j = torch.arange(ny, dtype=DTYPE).view(ny, 1)
    xi = torch.as_tensor(nodes, dtype=DTYPE).view(1, order)
    y_nodes = ((j + 0.5 * (xi + 1.0)) * dy).expand(ny, order)
    quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    pressure = (ns.GAMMA - 1.0) * (
        quad[..., 3] - 0.5 * (quad[..., 1] ** 2 + quad[..., 2] ** 2)
        / quad[..., 0].clamp_min(1e-12)
    )
    temperature = pressure / quad[..., 0].clamp_min(1e-12)
    return dict(
        velocity_x=quad[..., 1],
        temperature=temperature,
        density=quad[..., 0],  # note: conserved rho, already primitive
        y_nodes=y_nodes,
        nodes=torch.as_tensor(nodes, dtype=DTYPE),
    )


def relative_l2(numerical, exact, weights):
    """Relative L2 error of ``u(y)`` on the Gauss nodes of every wall-normal cell.

    The flow is ``x``-independent, so one Gauss station per cell in ``x`` carries
    the whole field; ``weights`` are the Gauss--Legendre weights along ``y``
    (the Jacobian ``dy/2`` cancels between numerator and denominator).
    """
    num = numerical.detach()
    ex = exact.detach()
    w = weights.view(1, -1)
    num_norm = float((w * (num - ex) ** 2).sum())
    den_norm = float((w * ex ** 2).sum())
    return math.sqrt(num_norm / den_norm) if den_norm > 0.0 else float("nan")


def run_case(degree, ny, final_time=FINAL_TIME, ncells_x=1, out_dir=None):
    started = time.time()
    x, y, coeff, meta = ns.solve_couette_dg_2d(
        ncells=(ncells_x, ny),
        degree=degree,
        final_time=final_time,
        wall_speed=WALL_SPEED,
        wall_temperature=WALL_TEMPERATURE,
        pressure=PRESSURE,
        mu=MU,
        prandtl=PRANDTL,
    )
    wall_seconds = round(time.time() - started, 2)

    basis = legendre_operators(degree, max(degree + 2, 3), DEVICE, DTYPE)[2]
    fields = _fields_at_nodes(coeff, basis, degree, (ncells_x, ny))
    nodes = fields["nodes"]
    # Gauss-Legendre weights on [-1, 1]; the Jacobian dy/2 cancels in the ratio
    _, weights = np.polynomial.legendre.leggauss(basis.shape[0])
    wy = torch.as_tensor(weights, dtype=DTYPE)

    y_nodes = fields["y_nodes"]                                   # (ny, order)
    exact = ns.impulsive_couette_exact(
        y_nodes, final_time, wall_speed=WALL_SPEED, kinematic_viscosity=NU
    )
    numerical = fields["velocity_x"][0, :, 0, :]                  # (ny, order)
    err = relative_l2(numerical, exact, wy)
    abs_err = float((numerical - exact).abs().max())

    # admissibility (independent of the series)
    umin = float(numerical.min())
    umax = float(numerical.max())
    profile = numerical.detach().numpy().reshape(-1)
    monotone = bool(np.all(np.diff(profile) >= -0.02 * WALL_SPEED))
    heating = float((fields["temperature"] - WALL_TEMPERATURE).abs().max())

    # conservation of the conserved variables (x is periodic and uniform)
    mass = float(coeff[..., 0].sum())
    momentum = float(coeff[..., 1].sum())
    row = dict(
        degree=degree,
        ncells=[ncells_x, ny],
        h=1.0 / ny,
        final_time=final_time,
        steps=meta["steps"],
        dt=meta["dt"],
        wall_seconds=wall_seconds,
        velocity_l2_relative=err,
        velocity_abs_max=abs_err,
        u_min=umin,
        u_max=umax,
        u_monotone_in_y=monotone,
        max_temperature_excursion=heating,
        heating_scale=HEATING_SCALE,
        mass=mass,
        momentum_x=momentum,
        finite=bool(torch.isfinite(coeff).all()),
        y_centres=[float(v) for v in ((torch.arange(ny, dtype=DTYPE) + 0.5) / ny)],
        u_centres=[float(v) for v in coeff[0, :, 0, 0, 1]],
    )
    if out_dir is not None:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"case_p{degree}_ny{ny}.json"
        path.write_text(json.dumps(row, indent=2))
    print(
        f"P{degree} ny={ny:3d}  relL2(u) = {100.0 * err:8.4f} %  "
        f"absmax = {abs_err:.3e}  u in [{umin:.4f}, {umax:.4f}]  "
        f"max|T-1| = {heating:.2e}  steps={meta['steps']:6d}  "
        f"{wall_seconds:7.1f}s",
        flush=True,
    )
    return row


def observed_order(coarse, fine, ratio=2.0):
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def merge(out_dir=DEFAULT_OUT, figure=FIGURE):
    out_dir = pathlib.Path(out_dir)
    rows = []
    for path in sorted(out_dir.glob("case_p*_ny*.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        raise SystemExit(f"no per-case JSON under {out_dir}")
    rows.sort(key=lambda r: (r["degree"], r["ncells"][1]))

    orders = []
    for degree in sorted({r["degree"] for r in rows}):
        series = [r for r in rows if r["degree"] == degree]
        for coarse, fine in zip(series, series[1:]):
            orders.append(dict(
                degree=degree,
                ny=[coarse["ncells"][1], fine["ncells"][1]],
                slope=observed_order(coarse["velocity_l2_relative"],
                                     fine["velocity_l2_relative"]),
            ))

    criterion_profile = all(r["velocity_l2_relative"] < 0.03 for r in rows)
    criterion_bounds = all(
        r["finite"]
        and r["u_min"] >= -0.02 * WALL_SPEED
        and r["u_max"] <= WALL_SPEED * 1.001
        for r in rows
    )
    criterion_monotone = all(r["u_monotone_in_y"] for r in rows)
    criterion_heating = all(
        r["max_temperature_excursion"] <= 3.0 * r["heating_scale"] for r in rows
    )
    criterion_order = all(o["slope"] > 0.5 for o in orders)

    verdict = dict(
        profile_below_three_percent=criterion_profile,
        bounded_and_physical=criterion_bounds,
        monotone_in_y=criterion_monotone,
        heating_within_scale=criterion_heating,
        refinement_slope_above_half=criterion_order,
    )
    payload = dict(
        setup=dict(
            wall_speed=WALL_SPEED,
            pressure=PRESSURE,
            wall_temperature=WALL_TEMPERATURE,
            mu=MU,
            prandtl=PRANDTL,
            kinematic_viscosity=NU,
            kappa=KAPPA,
            heating_scale=HEATING_SCALE,
            final_time=FINAL_TIME,
            reference="Stokes series of the impulsively started channel Couette",
        ),
        cases=rows,
        orders=orders,
        verdict=verdict,
        overall="PASS" if all(verdict.values()) else "FAIL",
    )
    path = out_dir / "impulse.json"
    path.write_text(json.dumps(payload, indent=2))
    if figure is not None:
        render_figure(payload, pathlib.Path(figure))
    print(json.dumps(verdict, indent=2))
    print("overall:", payload["overall"])
    return payload


def render_figure(payload, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["cases"]
    setup = payload["setup"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    colors = {1: "tab:blue", 2: "tab:orange", 3: "tab:green", 4: "tab:red"}

    ax = axes[0]
    y = np.linspace(0.0, 1.0, 401)
    exact = ns.impulsive_couette_exact(
        torch.as_tensor(y, dtype=DTYPE), setup["final_time"],
        wall_speed=setup["wall_speed"],
        kinematic_viscosity=setup["kinematic_viscosity"],
    ).numpy()
    ax.plot(exact, y, "k-", lw=2.5, label="Stokes series (exact)")
    thinnest = max(r["ncells"][1] for r in rows)
    for row in rows:
        if row["ncells"][1] != thinnest:
            continue
        ax.plot(row["u_centres"], row["y_centres"], "o--", ms=3.5,
                color=colors.get(row["degree"], "0.4"),
                label=f"DG P{row['degree']}, {row['ncells'][1]} cells")
    ax.set_xlabel("$u$")
    ax.set_ylabel("$y/H$")
    ax.set_title(f"(a) impulsive start at $t$ = {setup['final_time']}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for degree in sorted({r["degree"] for r in rows}):
        series = [r for r in rows if r["degree"] == degree]
        h = [r["h"] for r in series]
        err = [max(100.0 * r["velocity_l2_relative"], 1e-10) for r in series]
        ax.loglog(h, err, "o-", color=colors.get(degree, "0.4"), label=f"P{degree}")
        if len(h) > 1:
            ax.loglog(h, [err[0] * (x / h[0]) ** 1.0 for x in h], ":",
                      color=colors.get(degree, "0.4"), alpha=0.6)
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % gate")
    ax.set_xlabel("$h = 1/n_y$")
    ax.set_ylabel("relative $L^2$ error of $u$ [%]")
    ax.set_title("(b) mesh refinement (dotted: slope 1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    labels = [f"P{r['degree']}/{r['ncells'][1]}" for r in rows]
    values = [100.0 * r["velocity_l2_relative"] for r in rows]
    ax.bar(range(len(rows)), values,
           color=[colors.get(r["degree"], "0.4") for r in rows])
    ax.axhline(3.0, color="k", ls="--", lw=1.0)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel("relative $L^2$ error of $u$ [%]")
    ax.set_title("(c) every case vs the 3 % gate")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "TensorDG m3661 - impulsively started Couette: DG wall transient vs Stokes series"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("figure:", path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="single case, formatted as degree:ny")
    parser.add_argument("--quick", action="store_true", help="two coarse cases")
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--out", default=str(DEFAULT_OUT / "cases"))
    parser.add_argument("--merge", action="store_true",
                        help="assemble impulse.json + figure from --out")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    if args.merge:
        merge(args.out, None if args.no_figure else FIGURE)
        return
    if args.only:
        degree, ny = (int(part) for part in args.only.split(":"))
        cases = ((degree, ny),)
    elif args.quick:
        cases = QUICK_CASES
    else:
        cases = CASES
    for degree, ny in cases:
        run_case(degree, ny, final_time=args.final_time, out_dir=args.out)


if __name__ == "__main__":
    main()