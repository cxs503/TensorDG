"""Linear-stability spectrum of the ``ns_dg_2d`` Dirichlet-wall operator.

The m2496 delivery documented a wall-bounded *transient* failure (impulsively
started Couette dies near ``t = 0.033``) and the m2589 audit localised it to the
temperature channel and to a residual that does not decay with ``h``.  This tool
answers the next question, which neither delivery could: **is the semi-discrete
wall operator linearly stable at all?**  It assembles the Jacobian of
``_rhs_ns`` by central differences on a small (2 x 8, P1) state and reports the
eigenvalue with the largest real part.

What the numbers show (see ``docs/reports/tensordg-ns-wall-spectrum-m2731.md``):

* the wall operator is *linearly unstable*, with ``max Re(lambda)`` proportional
  to the SIPG penalty ``sigma``;
* the instability is **sign-symmetric in the penalty** -- flipping the sign of
  the wall penalty term changes ``max Re(lambda)`` by less than 1 %, and the
  channel splits add up -- which is the fingerprint of a *non-symmetric rank-1*
  boundary term (a Nitsche form with its adjoint term missing), not of a
  too-large or too-small penalty;
* with the penalty removed the operator is exactly *neutral*
  (``|max Re| < 1e-8``): every bit of wall dissipation comes from that term.

Usage
-----
``PYTHONPATH=. python3 tools/diag_ns_wall_spectrum.py`` writes
``validation/ns_dg_2d_wall/spectrum_m2731.json`` and
``docs/figures/ns_dg_2d_wall_spectrum_m2731.png``.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import legendre_operators

DT = torch.float64
DEV = torch.device("cpu")
MU = 0.1
PRANDTL = 0.71
KAPPA = MU * ns.CP / PRANDTL
ORIGINAL_WALL_FLUX = ns._boundary_flux


# ---------------------------------------------------------------------------
# Grid and base states
# ---------------------------------------------------------------------------
def operators(degree, ncells):
    nx, ny = ncells
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(degree, order, DEV, DT)
    dx, dy = 1.0 / nx, 1.0 / ny
    x = (torch.arange(nx, dtype=DT) + 0.5) * dx
    y = (torch.arange(ny, dtype=DT) + 0.5) * dy
    return {"nodes": nodes, "weights": weights, "basis": basis,
            "derivative": derivative, "signs": signs, "dx": dx, "dy": dy, "x": x, "y": y}


def _project(op, initial):
    return ns._initial_coefficients(initial, op["x"], op["y"], op["dx"], op["dy"],
                                    op["nodes"], op["weights"], op["basis"], DEV, DT)


def static_state(ncells=(1, 8), degree=1):
    """Fluid at rest, uniform (T = 1), isothermal no-slip walls: zero residual."""
    op = operators(degree, ncells)
    def initial(x, y):
        shape = (x * y).shape
        ones = torch.ones(shape, dtype=DT)
        return ones.clone(), torch.zeros_like(ones), torch.zeros_like(ones), ones.clone()
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (0.0, 0.0), "temperature": 1.0}}
    return op, _project(op, initial), boundary


def couette_state(ncells=(2, 8), degree=1, wall_speed=1.0):
    """Analytic steady viscous Couette state projected on the DG space."""
    op = operators(degree, ncells)
    def initial(x, y):
        profile = ns.couette_exact(y, 0.0, 1.0, wall_speed, 1.0, 1.0, MU, KAPPA)
        shape = (x * y).shape
        return (profile["density"].expand(shape).clone(),
                profile["velocity_x"].expand(shape).clone(),
                profile["velocity_y"].expand(shape).clone(),
                profile["pressure"].expand(shape).clone())
    boundary = {"xmin": None, "xmax": None,
                "ymin": {"velocity": (0.0, 0.0), "temperature": 1.0},
                "ymax": {"velocity": (wall_speed, 0.0), "temperature": 1.0}}
    return op, _project(op, initial), boundary


# ---------------------------------------------------------------------------
# Wall-flux variants
# ---------------------------------------------------------------------------
def wall_flux_variant(coefficients=(1.0, 1.0, 1.0)):
    """Wall flux with the penalty scaled per channel ``(u, v, T)``.

    ``coefficients=(1, 1, 1)`` reproduces the shipped code; ``(0, 0, 0)`` removes
    the wall penalty entirely; negative values flip its sign.
    """
    def wall_flux(side, wall, axis, h, mu, kappa, penalty, normal_scale,
                  tangential_scale, normal_derivative=None, recovered=None,
                  wall_gradient="taylor", wall_penalty=None,
                  centre=None, side_name="right", outward_sign=1.0):
        # the kwargs are accepted for signature compatibility with the current
        # ``_boundary_flux``; this reimplementation deliberately models the
        # pre-m2782 (sign-blind) wall operator that mission m2731 diagnosed
        if wall is None:
            return ORIGINAL_WALL_FLUX(side, wall, axis, h, mu, kappa, penalty,
                                      normal_scale, tangential_scale,
                                      normal_derivative, recovered, centre,
                                      side_name, outward_sign)
        cu, cv, cT = coefficients
        inner_u, inner_v = side["u"], side["v"]
        velocity = wall.get("velocity", (0.0, 0.0))
        face_u = torch.as_tensor(velocity[0], dtype=inner_u.dtype).expand_as(inner_u)
        face_v = torch.as_tensor(velocity[1], dtype=inner_v.dtype).expand_as(inner_v)
        temperature = wall.get("temperature", None)
        face_T = (side["temperature"] if temperature is None
                  else torch.as_tensor(temperature, dtype=inner_u.dtype).expand_as(inner_u))
        normal_u = side["u_n"] * normal_scale - cu * (penalty / h) * (inner_u - face_u)
        normal_v = side["v_n"] * normal_scale - cv * (penalty / h) * (inner_v - face_v)
        normal_T = (side["T_n"] * normal_scale
                    - cT * (penalty / h) * (side["temperature"] - face_T))
        zeros = torch.zeros_like(inner_u)
        pressure = side["pressure"]
        inviscid = torch.stack([zeros,
                                pressure if axis == 0 else zeros,
                                zeros if axis == 0 else pressure,
                                zeros], dim=-1)
        face = {"u": face_u, "v": face_v}
        if axis == 0:
            face.update(u_x=normal_u, v_x=normal_v, T_x=normal_T,
                        u_y=side["u_t"] * tangential_scale,
                        v_y=side["v_t"] * tangential_scale,
                        T_y=side["T_t"] * tangential_scale)
        else:
            face.update(u_y=normal_u, v_y=normal_v, T_y=normal_T,
                        u_x=side["u_t"] * tangential_scale,
                        v_x=side["v_t"] * tangential_scale,
                        T_x=side["T_t"] * tangential_scale)
        return inviscid - ns._viscous_flux(face, axis, mu, kappa)
    return wall_flux


def rhs(op, coeff, boundary, penalty):
    return ns._rhs_ns(coeff, op["basis"], op["derivative"], op["weights"], op["signs"],
                      op["dx"], op["dy"], (True, False), boundary, MU, KAPPA, penalty)


def jacobian(op, coeff, boundary, penalty, eps=1e-7):
    base = rhs(op, coeff, boundary, penalty)
    n = coeff.numel()
    flat = coeff.reshape(-1)
    matrix = np.zeros((n, n))
    for i in range(n):
        perturbed = flat.clone()
        perturbed[i] += eps
        matrix[:, i] = ((rhs(op, perturbed.reshape(coeff.shape), boundary, penalty) - base)
                        .reshape(-1).numpy() / eps)
    return matrix, base


def spectrum(op, coeff, boundary, penalty):
    matrix, base = jacobian(op, coeff, boundary, penalty)
    eigenvalues = np.linalg.eigvals(matrix)
    return {"max_real": float(eigenvalues.real.max()),
            "min_real": float(eigenvalues.real.min()),
            "abs_max": float(np.abs(eigenvalues).max()),
            "rhs_max": float(base.abs().max()),
            "eigenvalues": eigenvalues, "matrix": matrix}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run(out_json="validation/ns_dg_2d_wall/spectrum.json",
        out_png="docs/figures/ns_dg_2d_wall_spectrum.png"):
    summary = {"mission": "m2731", "mu": MU, "prandtl": PRANDTL, "kappa": KAPPA,
               "grid": {"static": [1, 8], "couette": [2, 8]}, "degree": 1,
               "penalty_sigma_sweep": {}, "channel_split": {}, "sign_symmetry": {},
               "dominant_mode": {}, "reference": {}}

    op_s, coeff_s, bc_s = static_state()
    op_c, coeff_c, bc_c = couette_state()

    # (a) sigma sweep on both base states (the shipped penalty is 2 (p+1)^2 = 8)
    for sigma in (0.0, 2.0, 4.0, 8.0, 16.0):
        row = {}
        for name, (op, coeff, bc, base_penalty) in {
            "static": (op_s, coeff_s, bc_s, sigma),
            "couette": (op_c, coeff_c, bc_c, sigma),
        }.items():
            res = spectrum(op, coeff, bc, sigma)
            row[name] = {key: res[key] for key in
                         ("max_real", "min_real", "abs_max", "rhs_max")}
        summary["penalty_sigma_sweep"][f"{sigma:g}"] = row

    # shipped penalty = 2 (degree+1)^2 = 8 with the standard 2 (p+1)^2 convention
    sigma_shipped = 8.0
    reference = spectrum(op_s, coeff_s, bc_s, sigma_shipped)
    summary["reference"] = {key: reference[key] for key in
                            ("max_real", "min_real", "abs_max", "rhs_max")}

    # (b) channel split at the shipped penalty
    ns._boundary_flux = ORIGINAL_WALL_FLUX
    for label, coefficients in {
        "all_channels": (1.0, 1.0, 1.0),
        "momentum_only": (1.0, 1.0, 0.0),
        "temperature_only": (0.0, 0.0, 1.0),
        "no_penalty": (0.0, 0.0, 0.0),
    }.items():
        ns._boundary_flux = wall_flux_variant(coefficients)
        res = spectrum(op_s, coeff_s, bc_s, sigma_shipped)
        summary["channel_split"][label] = {
            "max_real": res["max_real"], "abs_max": res["abs_max"]}
    ns._boundary_flux = ORIGINAL_WALL_FLUX

    # (c) sign symmetry of the penalty coefficient
    for value in (1.0, 2.0, 4.0, 8.0, 16.0):
        row = {}
        for label, coefficients in {
            "positive": (1.0, 1.0, 1.0), "negative": (-1.0, -1.0, -1.0)}.items():
            ns._boundary_flux = wall_flux_variant(coefficients)
            res = spectrum(op_s, coeff_s, bc_s, value)
            row[label] = res["max_real"]
        row["relative_gap"] = abs(row["positive"] - row["negative"]) / max(
            abs(row["positive"]), 1e-30)
        summary["sign_symmetry"][f"{value:g}"] = row
        ns._boundary_flux = ORIGINAL_WALL_FLUX

    # (d) structure of the dominant eigenvector
    matrix = reference["matrix"]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(eigenvalues.real))
    vector = np.abs(eigenvectors[:, index]).reshape(coeff_s.shape)
    summary["dominant_mode"] = {
        "eigenvalue": [float(eigenvalues[index].real), float(eigenvalues[index].imag)],
        "by_variable": vector.sum(axis=(0, 1, 2, 3)).tolist(),
        "by_row": vector.sum(axis=(0, 2, 3, 4)).tolist(),
        "by_y_mode": vector.sum(axis=(0, 1, 4, 2)).tolist(),
        "variable_labels": ["density", "x-momentum", "y-momentum", "energy"],
    }

    pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_json).write_text(json.dumps(summary, indent=2))

    _figure(summary, out_png)
    return summary


def _figure(summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    sigmas = [float(key) for key in summary["penalty_sigma_sweep"]]
    for name, style in (("static", "o-"), ("couette", "s--")):
        values = [summary["penalty_sigma_sweep"][f"{s:g}"][name]["max_real"]
                  for s in sigmas]
        axes[0].plot(sigmas, values, style, label=name)
    axes[0].set_xlabel("SIPG penalty $\\sigma$ (code: $2(p+1)^2$)")
    axes[0].set_ylabel("max Re$\\,\\lambda$  [1/time]")
    axes[0].set_title("(a) wall operator is unstable, growth $\\propto\\sigma$")
    axes[0].axhline(0.0, color="k", lw=0.6)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    values = [float(key) for key in summary["sign_symmetry"]]
    pos = [summary["sign_symmetry"][f"{v:g}"]["positive"] for v in values]
    neg = [summary["sign_symmetry"][f"{v:g}"]["negative"] for v in values]
    axes[1].loglog(values, pos, "o-", label="penalty as shipped ($-\\sigma/h$)")
    axes[1].loglog(values, neg, "x--", label="penalty sign flipped ($+\\sigma/h$)")
    axes[1].set_xlabel("$|$penalty coefficient$|$")
    axes[1].set_ylabel("max Re$\\,\\lambda$  [1/time]")
    axes[1].set_title("(b) growth depends on $|\\sigma|$ only")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    rows = summary["dominant_mode"]["by_row"]
    axes[2].semilogy(range(len(rows)), np.maximum(rows, 1e-16), "o-")
    axes[2].set_xlabel("y-cell row (0 = bottom wall)")
    axes[2].set_ylabel("dominant eigenvector magnitude")
    axes[2].set_title("(c) unstable mode lives on the wall rows")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in
                      ("penalty_sigma_sweep", "channel_split", "sign_symmetry")},
                     indent=2)[:4000])
    mode = result["dominant_mode"]
    print("dominant eigenvalue", np.round(mode["eigenvalue"], 3))
    print("by variable", np.round(mode["by_variable"], 4))
    print("by row", np.round(mode["by_row"], 5))