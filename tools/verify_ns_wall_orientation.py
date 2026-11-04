"""Reproduce the Dirichlet-penalty orientation fix of :mod:`tensordg.ns_dg_2d`.

Writes ``validation/ns_dg_2d_wall/orientation_fix.json`` and
``docs/figures/ns_wall_orientation_fix.png``.  Three machine-checked blocks:

1. **Spectral stability** -- the eigenvalues of the semi-discrete Jacobian at the
   impulsive-start state, for the pre-fix variant (penalty ``+1`` on both faces,
   forced through the wrapper) and the fixed one (``outward_sign=-1`` on the
   ``min`` face; in this chain the mechanism is m2782's ``outward_sign``).
   Re-anchored from m2876's ``orientation`` keyword.  A positive real part means
   exponential growth of the wall mode, i.e. the divergence m2496/m2589 saw.
2. **Wall heat flux** -- relative error of the Dirichlet Fourier flux against the
   Brinkman parabola on both walls, with the observed order in ``h``.
3. **Transient** -- the impulsive-start Couette march, physical time reached.

Run: ``PYTHONPATH=. python -m tools.verify_ns_wall_orientation``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import _initial_coefficients, legendre_operators

torch.set_num_threads(2)
MU = 0.1
KAPPA = MU * ns.CP / 0.71
DEVICE = torch.device("cpu")
DTYPE = torch.float64
ROOT = Path(__file__).resolve().parents[1]


def _operators(degree: int, ncells):
    nx, ny = ncells
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, max(degree + 2, 3), DEVICE, DTYPE
    )
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=DTYPE) + 0.5) * dx
    y = (torch.arange(ny, dtype=DTYPE) + 0.5) * dy
    return dict(nodes=nodes, weights=weights, basis=basis, derivative=derivative,
                signs=signs, dx=dx, dy=dy, x=x, y=y)


def _couette_state(ncells=(2, 8), degree=1, wall_speed=1.0):
    op = _operators(degree, ncells)
    amplitude = MU * wall_speed**2 / (2.0 * KAPPA)

    def initial(x, y):
        u = wall_speed * y
        zeros = torch.zeros_like(u)
        temperature = 1.0 + amplitude * y * (1.0 - y)
        return 1.0 / temperature, u, zeros, torch.ones_like(u)

    coeff = _initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                  op["nodes"], op["weights"], op["basis"], DEVICE, DTYPE)
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (wall_speed, 0.0), "temperature": 1.0}}
    return op, coeff, boundary


def spectrum(ncells=(2, 4), degree=1, flip_min=True):
    """max Re(eig(J)) of the wall-bounded residual at the impulsive-start state."""
    op, coeff, boundary = _couette_state(ncells, degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    args = (op["basis"], op["derivative"], op["weights"], op["signs"], op["dx"],
            op["dy"], (True, False), boundary, MU, KAPPA, penalty)
    original = ns._boundary_flux
    if not flip_min:  # recreate the shipped variant: same sign on both faces
        def shipped(side, wall, axis, h, mu, kappa, pen, normal_scale,
                    tangential_scale, normal_derivative=None, neighbour=None,
                    recovered=None, centre=None, side_name="right",
                    outward_sign=1.0):
            # force the pre-fix variant: same penalty sign on both faces
            return original(side, wall, axis, h, mu, kappa, pen, normal_scale,
                            tangential_scale, normal_derivative, neighbour,
                            recovered, centre, side_name, 1.0)
        ns._boundary_flux = shipped
    try:
        flat = coeff.reshape(-1).clone()
        n = flat.numel()
        jac = np.zeros((n, n))
        eps = 1e-6
        base = ns._rhs_ns(coeff, *args).reshape(-1).numpy().astype(float)
        for i in range(n):
            probe = flat.clone()
            probe[i] += eps
            jac[:, i] = (
                ns._rhs_ns(probe.reshape(coeff.shape), *args).reshape(-1).numpy().astype(float)
                - base
            ) / eps
        eigenvalues = np.linalg.eigvals(jac)
    finally:
        ns._boundary_flux = original
    return {
        "ncells": list(ncells),
        "degree": degree,
        "dofs": int(eigenvalues.size),
        "max_real": float(eigenvalues.real.max()),
        "max_abs": float(np.abs(eigenvalues).max()),
        "positive": int((eigenvalues.real > 1e-8).sum()),
    }


def wall_fluxes(ncells=(2, 8), degree=1, wall_speed=1.0):
    op, coeff, boundary = _couette_state(ncells, degree, wall_speed)
    penalty = 2.0 * (degree + 1.0) ** 2
    h = op["dy"]
    left = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "left")
    right = ns._face_primitives(coeff, op["basis"], op["derivative"], op["signs"], 1, "right")
    top = ns._boundary_flux(right, boundary["ymax"], 1, h, MU, KAPPA, penalty,
                            2.0 / h, 2.0 / op["dx"])
    bottom = ns._boundary_flux(left, boundary["ymin"], 1, h, MU, KAPPA, penalty,
                               2.0 / h, 2.0 / op["dx"],
                               side_name="left", outward_sign=-1.0)
    return {"energy_top": float(top[0, -1, 0, 3]),
            "energy_bottom": float(bottom[0, 0, 0, 3]),
            "mx_top": float(top[0, -1, 0, 1]),
            "mx_bottom": float(bottom[0, 0, 0, 1])}


def analytic_residual(ncells, degree=1):
    op, coeff, boundary = _couette_state(ncells, degree)
    penalty = 2.0 * (degree + 1.0) ** 2
    rhs = ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"], op["signs"],
                     op["dx"], op["dy"], (True, False), boundary, MU, KAPPA, penalty)
    return float(rhs.abs().max())


def transient(final_time=0.1):
    try:
        _, _, _, meta = ns.solve_couette_dg_2d(ncells=(2, 8), degree=1, mu=MU,
                                               final_time=final_time)
        return {"status": "physical", "time": final_time,
                "steps": meta["steps"], "dt": meta["dt"]}
    except FloatingPointError as exc:  # pragma: no cover - regression guard
        return {"status": "nonphysical", "error": str(exc)}


def main():
    # Fourier flux magnitude kappa * A, A = mu U^2 / (2 kappa); equals mu U^2 / 2
    amplitude = KAPPA * (MU * 1.0**2 / (2.0 * KAPPA))
    fixed = {"spectrum": spectrum(flip_min=True),
             "residual": [], "flux": [], "transient": transient()}
    shipped = {"spectrum": spectrum(flip_min=False)}
    for ny in (8, 16, 32, 64):
        f = wall_fluxes((2, ny))
        fixed["flux"].append({
            "ny": ny,
            "energy_top": f["energy_top"], "energy_bottom": f["energy_bottom"],
            "error_top": abs(f["energy_top"] + amplitude) / amplitude,
            "error_bottom": abs(f["energy_bottom"] + amplitude) / amplitude,
            "error_mx": abs(abs(f["mx_top"]) - MU * 1.0) / (MU * 1.0),
        })
    for ny in (8, 16, 32, 64):
        fixed["residual"].append({"ny": ny, "max_abs_rhs": analytic_residual((2, ny))})
    orders = {}
    for key in ("error_top", "error_bottom"):
        errs = [row[key] for row in fixed["flux"]]
        orders[key] = float(math.log2(errs[0] / errs[-1]) / 2.0)
    res = [row["max_abs_rhs"] for row in fixed["residual"]]
    orders["residual"] = float(math.log2(res[0] / res[-1]) / 2.0) if res[-1] > 0 else 0.0
    fixed["orders"] = orders
    payload = {"mu": MU, "kappa": KAPPA, "shipped": shipped, "fixed": fixed,
               "note": "pre-fix = penalty with +1.0 sign on both faces; "
                       "fixed = outward-normal sign (-1 on the min face, "
                       "outward_sign keyword)"}
    out = ROOT / "validation" / "ns_dg_2d_wall" / "orientation_fix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({"shipped_max_real": shipped["spectrum"]["max_real"],
                      "fixed_max_real": fixed["spectrum"]["max_real"],
                      "fixed_positive": fixed["spectrum"]["positive"],
                      "orders": orders,
                      "transient": fixed["transient"]}, indent=2))
    return payload


def figure(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    labels = ["shipped\n(+1 both faces)", "fixed\n(outward normal)"]
    values = [payload["shipped"]["spectrum"]["max_real"],
              payload["fixed"]["spectrum"]["max_real"]]
    ax.bar(labels, values, color=["#c0392b", "#27ae60"], width=0.55)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_ylabel(r"$\max\,\mathrm{Re}\,\lambda$ of $\partial R/\partial U$")
    ax.set_title("Semi-discrete stability (2x4 P1, impulsive start)")
    for i, v in enumerate(values):
        ax.annotate(f"{v:+.2f}" if abs(v) > 1e-6 else "0 (machine zero)",
                    (i, v), textcoords="offset points", xytext=(0, 6 if v >= 0 else -14),
                    ha="center", fontsize=9)
    ax = axes[1]
    ny = [row["ny"] for row in payload["fixed"]["flux"]]
    ax.loglog(ny, [row["error_top"] for row in payload["fixed"]["flux"]], "o-",
              label=r"top wall, order %.2f" % payload["fixed"]["orders"]["error_top"])
    ax.loglog(ny, [row["error_bottom"] for row in payload["fixed"]["flux"]], "s-",
              label=r"bottom wall, order %.2f" % payload["fixed"]["orders"]["error_bottom"])
    ax.loglog(ny, [2.0 / n for n in ny], "k--", lw=1, label=r"$O(h)$ reference")
    ax.axhline(0.03, color="gray", ls=":", lw=1, label="3% gate")
    ax.set_xlabel("cells in y")
    ax.set_ylabel("relative error of the wall heat flux")
    ax.set_title("Dirichlet heat flux vs Brinkman parabola")
    ax.legend(fontsize=8)
    fig.tight_layout()
    target = ROOT / "docs" / "figures" / "ns_wall_orientation_fix.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    print("figure ->", target)


if __name__ == "__main__":
    figure(main())
