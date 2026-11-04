"""Spectrum of the ``ns_dg_2d`` semi-discrete operator around the wall (Dirichlet) path.

Context: ``docs/reports/tensordg-ns-wall-verification.md`` (m2589) localised the
wall-bounded transient failure to an O(1) wall residual and guessed a *missing
second-order wall gradient* as the seed.  This tool tests that guess directly:
the residual is differentiable, so the Jacobian of ``_rhs_ns`` at the analytic
steady Couette state is exact and its spectrum decides stability without
marching anything.

Findings (see ``docs/reports/tensordg-ns-wall-spectrum.md``):

* fully periodic operator: ``max Re(lambda) = 0`` (marginal, no growth);
* wall operator: ``max Re(lambda) > 0`` -- a genuine spatial instability, and the
  unstable eigenvector lives in the **energy** channel of the **wall cell**;
* switching the wall *penalty* off leaves ``max Re`` practically unchanged,
  switching the wall *gradient* (the diffusive flux) off removes it; the penalty
  alone contributes O(1) to ``max Re`` while the gradient contributes O(10^2);
* a second-order one-sided wall gradient (``wall_gradient="taylor"`` in
  ``tensordg.ns_dg_2d``) halves the wall flux error but leaves ``max Re``
  unchanged -- so the one-sided gradient is *not* the destabiliser;
* ``max Re`` grows linearly with the SIPG penalty ``sigma``, i.e. the boundary
  term is anti-dissipative: the wall lacks the adjoint (Nitsche) boundary term
  that pairs with the penalty, and no penalty magnitude can compensate for it.

Usage::

    PYTHONPATH=. python -m tools.spectrum_ns_wall            # full study (~2 min)
    PYTHONPATH=. python -m tools.spectrum_ns_wall spectra    # one section
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall import KAPPA, MU, WALL_SPEED, WALL_T, build_state

DTYPE = torch.float64
OUT_JSON = Path("validation/ns_dg_2d_wall/spectrum.json")
OUT_FIG = Path("docs/figures/ns_dg_2d_wall_spectrum.png")

VARIANTS = ("default", "inviscid", "grad", "penalty")


def _patched_flux(variant: str):
    """``ns._boundary_flux`` with one wall term switched off (A/B knob)."""
    original = ns._boundary_flux

    def patched(side, wall, axis, h, mu, kappa, pen, normal_scale, tangential_scale,
                normal_derivative=None):
        if wall is None or variant == "default":
            return original(side, wall, axis, h, mu, kappa, pen, normal_scale,
                            tangential_scale, normal_derivative)
        inner_u, inner_v = side["u"], side["v"]
        wall_velocity = wall.get("velocity", (0.0, 0.0))
        face_u = torch.as_tensor(wall_velocity[0], dtype=inner_u.dtype).expand_as(inner_u)
        face_v = torch.as_tensor(wall_velocity[1], dtype=inner_v.dtype).expand_as(inner_v)
        wall_temperature = wall.get("temperature", None)
        face_T = (side["temperature"] if wall_temperature is None else
                  torch.as_tensor(wall_temperature, dtype=inner_u.dtype).expand_as(inner_u))
        base_u = side["u_n"] * normal_scale
        base_v = side["v_n"] * normal_scale
        base_T = side["T_n"] * normal_scale
        if normal_derivative is not None:
            base_u = normal_derivative["u_n"]
            base_v = normal_derivative["v_n"]
            base_T = normal_derivative["T_n"]
        if variant == "grad":       # keep only the Dirichlet penalty
            base_u = torch.zeros_like(base_u)
            base_v = torch.zeros_like(base_v)
            base_T = torch.zeros_like(base_T)
        normal_u = base_u - (pen / h) * (inner_u - face_u)
        normal_v = base_v - (pen / h) * (inner_v - face_v)
        normal_T = base_T - (pen / h) * (side["temperature"] - face_T)
        if variant == "penalty":    # keep only the one-sided gradient
            normal_u, normal_v, normal_T = base_u, base_v, base_T
        zeros = torch.zeros_like(inner_u)
        pressure = side["pressure"]
        inviscid = (torch.stack([zeros, pressure, zeros, zeros], dim=-1) if axis == 0
                    else torch.stack([zeros, zeros, pressure, zeros], dim=-1))
        if variant == "inviscid":   # adiabatic / free-slip wall
            return inviscid
        face = {"u": face_u, "v": face_v}
        tu = side["u_t"] * tangential_scale
        tv = side["v_t"] * tangential_scale
        tT = side["T_t"] * tangential_scale
        if axis == 0:
            face.update(u_x=normal_u, v_x=normal_v, T_x=normal_T, u_y=tu, v_y=tv, T_y=tT)
        else:
            face.update(u_y=normal_u, v_y=normal_v, T_y=normal_T, u_x=tu, v_x=tv, T_x=tT)
        return inviscid - ns._viscous_flux(face, axis, mu, kappa)

    return original, patched


def _jacobian(coeff, state, boundary, periodic, penalty, wall_gradient, variant="default"):
    original, patched = _patched_flux(variant)
    ns._boundary_flux = patched
    try:
        x = coeff.clone().requires_grad_(True)

        def residual(z):
            return ns._rhs_ns(z, state["basis"], state["derivative"], state["weights"],
                              state["signs"], state["dx"], state["dy"], periodic,
                              boundary, MU, KAPPA, penalty, wall_gradient)

        flat = residual(x).reshape(-1)
        rows = []
        for i in range(flat.numel()):
            gradient = torch.autograd.grad(flat[i], x, retain_graph=True)[0]
            rows.append(gradient.reshape(-1).detach().numpy())
        return np.array(rows)
    finally:
        ns._boundary_flux = original


def _spectrum(coeff, state, boundary, periodic, penalty, wall_gradient, variant="default"):
    """Leading eigenvalues plus the channel breakdown of the fastest mode."""
    matrix = _jacobian(coeff, state, boundary, periodic, penalty, wall_gradient, variant)
    values, vectors = np.linalg.eig(matrix)
    real = values.real
    index = int(np.argmax(real))
    vector = np.abs(vectors[:, index])
    shape = coeff.shape
    channel = np.zeros(shape[-1])
    last = np.unravel_index(np.arange(vector.size), shape)[-1]
    for slot, weight in zip(last, vector):
        channel[slot] += weight
    channel = channel / channel.sum() if channel.sum() else channel
    peak = np.unravel_index(int(np.argmax(vector)), shape)
    return {
        "max_real": float(real.max()),
        "min_real": float(real.min()),
        "growth_rate": float(real.max()),
        "dominant_channel": ["rho", "rho_u", "rho_v", "E"][int(np.argmax(channel))],
        "channel_weights": [float(c) for c in channel],
        "peak_dof": [int(p) for p in peak],
    }


def section_spectra():
    d = build_state((2, 8), 1)
    walls = d["boundary"]
    free = {"xmin": None, "xmax": None, "ymin": None, "ymax": None}
    penalty = 8.0
    rows = []

    def add(tag, boundary, periodic, wall_gradient="taylor", variant="default", pen=penalty):
        result = _spectrum(d["coeff"], d, boundary, periodic, pen, wall_gradient, variant)
        rows.append(dict(case=tag, wall_gradient=wall_gradient, variant=variant,
                         penalty=pen, **result))
        print(f"{tag:42s} maxRe={result['max_real']:+9.3f} "
              f"minRe={result['min_real']:10.2f} "
              f"chan={result['dominant_channel']} peak={tuple(result['peak_dof'])}")
        return result

    add("periodic (no wall)", free, (True, True))
    add("wall, legacy one-sided face", walls, (True, False), wall_gradient="face")
    add("wall, Taylor one-sided", walls, (True, False))
    add("wall, inviscid only (adiabatic)", walls, (True, False), variant="inviscid")
    add("wall, gradient only (no penalty)", walls, (True, False), variant="grad")
    add("wall, penalty only (no gradient)", walls, (True, False), variant="penalty")
    return rows


def section_penalty_scaling():
    d = build_state((2, 8), 1)
    rows = []
    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        penalty = sigma * 4.0
        result = _spectrum(d["coeff"], d, d["boundary"], (True, False), penalty, "taylor")
        rows.append(dict(sigma=sigma, penalty=penalty, **result))
        print(f"sigma={sigma:5.2f} penalty={penalty:6.1f} maxRe={result['max_real']:+9.3f}")
    for row in rows:
        row["growth_per_penalty"] = row["max_real"] / row["penalty"]
    return rows


def make_figure(spectra, scaling):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=160)

    labels = [row["case"].replace(", ", "\n") for row in spectra]
    rates = [row["max_real"] for row in spectra]
    colors = ["#2e7d32" if value <= 0.0 else "#c62828" for value in rates]
    left.barh(range(len(rates)), rates, color=colors)
    left.set_yticks(range(len(rates)))
    left.set_yticklabels(labels, fontsize=6.5)
    left.axvline(0.0, color="k", lw=0.8)
    left.set_xlabel(r"max Re$(\lambda)$ of $\partial \dot{U}/\partial U$")
    left.set_title("(a) which wall term destabilises the operator", fontsize=8)
    left.grid(axis="x", alpha=0.3)

    sigmas = [row["sigma"] for row in scaling]
    rates = [row["max_real"] for row in scaling]
    right.plot(sigmas, rates, "o-", color="#c62828", label="max Re")
    fit = np.polyfit(np.log(sigmas), np.log(rates), 1)
    right.plot(sigmas, np.exp(np.polyval(fit, np.log(sigmas))), "--",
               color="k", lw=0.8,
               label=f"power law, exponent {fit[0]:.2f}")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel(r"SIPG penalty multiplier $\sigma$")
    right.set_ylabel(r"max Re$(\lambda)$")
    right.set_title("(b) growth rate grows with the penalty", fontsize=8)
    right.legend(fontsize=7)
    right.grid(alpha=0.3, which="both")

    figure.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT_FIG)
    plt.close(figure)
    return {"figure": str(OUT_FIG), "penalty_exponent": float(fit[0])}


def main(argv):
    sections = {"spectra": section_spectra, "scaling": section_penalty_scaling}
    wanted = argv or list(sections)
    payload = {}
    for name in wanted:
        print(f"--- {name} ---")
        payload[name] = sections[name]()
    if {"spectra", "scaling"} <= set(payload):
        payload["figure"] = make_figure(payload["spectra"], payload["scaling"])
    payload["meta"] = {"ncells": [2, 8], "degree": 1, "mu": MU, "kappa": KAPPA,
                       "wall_speed": WALL_SPEED, "wall_temperature": WALL_T}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))