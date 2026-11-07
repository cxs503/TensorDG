"""Verification study for the TensorDG shallow-water solvers (1-D / 2-D).

Produces ``metrics.json`` (all numbers quoted in docs/swe_verification.md are
read back from here - nothing is transcribed by hand) and the figures used in
the report.

Run with::

    PYTHONPATH=. python validation/swe_dam_break/study.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

import tensordg.swe_1d as S1
import tensordg.swe_2d as S2
from tensordg.swe_1d import reconstruct_fields, solve_swe_1d
from tensordg.swe_exact import dam_break_cell_averages, stoker_star_state, stoker_exact

torch.set_num_threads(2)

OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(OUT, "figs")
os.makedirs(FIGS, exist_ok=True)

RESULT: dict = {"env": {"torch": torch.__version__, "numpy": np.__version__}}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dam_break_case(h_left: float, h_right: float, ncells: int, degree: int,
                   final_time: float = 0.5, domain=(-3.0, 3.0), cfl: float = 0.3,
                   flux_kind: str = "hll", limiter=None):
    """Solve the 1-D dam break and return numpy profiles + metadata."""
    x, coeff, zb_faces, meta = solve_swe_1d(
        lambda xx: torch.where(xx < 0.0, torch.full_like(xx, float(h_left)),
                               torch.full_like(xx, float(h_right))),
        lambda xx: torch.zeros_like(xx),
        lambda xx: torch.zeros_like(xx),
        ncells=ncells, degree=degree, final_time=final_time, domain=domain,
        cfl=cfl, flux_kind=flux_kind, limiter=limiter)
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    return (x.numpy(), h.numpy(), coeff[:, 0, 1].numpy(), meta)


def dam_break_metrics(h_left: float, h_right: float, x, h, hu, final_time: float):
    """Compare against the exact Stoker/Ritter solution (cell averages)."""
    faces = np.linspace(x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[1] - x[0]), len(x) + 1)
    h_ex, hu_ex = dam_break_cell_averages(faces, final_time, h_left, h_right)
    star = stoker_star_state(h_left, h_right)

    dx = float(faces[1] - faces[0])
    mass0 = h_left * abs(faces[0]) + h_right * abs(faces[-1])
    mass = float(h.sum() * dx)

    m = {"mass_err_rel": abs(mass - mass0) / mass0,
         "rel_l1_h": float(np.abs(h - h_ex).sum() / h_ex.sum()),
         "linf_h": float(np.abs(h - h_ex).max()),
         "rel_l1_hu": float(np.abs(hu - hu_ex).sum() / max(np.abs(hu_ex).sum(), 1e-30)),
         "h_min": float(h.min())}

    # star plateau: cells strictly inside the constant star region
    tail_x = final_time * star["wave_tail"]
    shock_x = final_time * star["shock_speed"]
    band = (x > tail_x + 3 * dx) & (x < shock_x - 3 * dx)
    if star["h_star"] > 0 and band.sum() >= 2:
        h_star_num = float(h[band].mean())
        u_star_num = float((hu[band] / h[band]).mean())
        m["h_star_num"] = h_star_num
        m["h_star_exact"] = star["h_star"]
        m["h_star_err_rel"] = abs(h_star_num - star["h_star"]) / star["h_star"]
        m["u_star_num"] = u_star_num
        m["u_star_exact"] = star["u_star"]
        m["u_star_err_rel"] = abs(u_star_num - star["u_star"]) / abs(star["u_star"])

    # rarefaction head: first cell whose depth drops below h_left
    below = np.where(h < h_left - 1e-9)[0]
    if len(below):
        m["head_num"] = float(x[below[0]])
        m["head_exact"] = final_time * star["wave_head"]
    # right front: last cell whose depth is above the still-water level
    above = np.where(h > h_right + 1e-9)[0]
    if len(above):
        m["front_num"] = float(x[above[-1]])
        m["front_exact"] = final_time * star["shock_speed"]
    return m


def lake_at_rest_1d(ncells=64, final_time=1.0, degree=1, legacy_source=False):
    eta0 = 2.0
    zb_f = lambda xx: 0.4 * torch.sin(math.pi * xx / 2.0)
    if legacy_source:
        S1._rhs = _legacy_rhs_1d
    try:
        x, coeff, zb_faces, meta = solve_swe_1d(
            lambda xx: eta0 - zb_f(xx), lambda xx: torch.zeros_like(xx), zb_f,
            ncells=ncells, degree=degree, final_time=final_time,
            domain=(-2.0, 2.0), cfl=0.3)
    finally:
        S1._rhs = _current_rhs_1d
    h, u, eta = reconstruct_fields(coeff, zb_faces)
    return {"max_eta_drift": float((eta - eta0).abs().max()),
            "max_hu": float(coeff[:, 0, 1].abs().max()),
            "steps": meta["steps"]}


_current_rhs_1d = S1._rhs


def _legacy_rhs_1d(coeff, dx, weights, phi, dphi, phi_left, phi_right, zb_jump,
                   flux_kind="hll", periodic=False):
    """Pre-fix residual (source term carrying the spurious 2/(2k+1) factor)."""
    at_quad = torch.einsum("cmv,qm->cqv", coeff, phi)
    volume = torch.einsum("cqv,q,qk->ckv", S1.physical_flux(at_quad), weights, dphi)
    right_trace = torch.einsum("cmv,m->cv", coeff, phi_right)
    left_trace = torch.einsum("cmv,m->cv", coeff, phi_left)
    right_neighbour = torch.roll(left_trace, shifts=-1, dims=0)
    if not periodic:
        right_neighbour[-1] = right_trace[-1]
    interface_flux = S1.hll_flux(right_trace, right_neighbour, flux_kind)
    left_interface_flux = torch.roll(interface_flux, shifts=1, dims=0)
    if not periodic:
        left_interface_flux[0] = S1.hll_flux(left_trace[:1], left_trace[:1], flux_kind)
    dbdx = zb_jump / dx
    out = torch.empty_like(coeff)
    for k in range(coeff.shape[1]):
        rhs_k = ((2 * k + 1) / dx) * (volume[:, k] - interface_flux * phi_right[k]
                                      + left_interface_flux * phi_left[k])
        rhs_k[:, 1] += -S1.GRAVITY * dbdx * (2.0 / (2 * k + 1)) * coeff[:, k, 0]
        out[:, k] = rhs_k
    return out


def lake_at_rest_2d(nx=16, final_time=0.5, degree=1):
    eta0 = 2.0
    flat = lambda X, Y: torch.zeros_like(X + Y)
    x, y, coeff, zb_faces, meta = S2.solve_swe_2d(
        lambda X, Y: eta0 - flat(X, Y), lambda X, Y: torch.zeros_like(X + Y),
        lambda X, Y: torch.zeros_like(X + Y), flat,
        nx=nx, ny=nx, degree=degree, final_time=final_time,
        domain=((-2.0, 2.0), (-2.0, 2.0)), cfl=0.2)
    eta = coeff[:, :, 0, 0, 0] + 0.5 * (zb_faces["x"][0] + zb_faces["x"][1])
    return {"max_eta_drift": float((eta - eta0).abs().max()),
            "max_hu": float(coeff[:, :, :, :, 1].abs().max()),
            "steps": meta["steps"]}


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    # ---- 1-D dam break (Stoker wet bed, Ritter dry bed) -------------------
    RESULT["stoker"] = {}
    profiles = {}
    for h_right in (0.1, 0.5, 0.0):
        key = f"hR_{h_right}"
        RESULT["stoker"][key] = {"config": {"h_left": 1.0, "h_right": h_right,
                                            "final_time": 0.5, "domain": [-3.0, 3.0],
                                            "cfl": 0.3, "flux": "hll"}}
        RESULT["stoker"][key]["exact_star"] = stoker_star_state(1.0, h_right)
        rows = []
        for degree in (1, 2):
            for ncells in (100, 200, 400):
                if degree == 2 and ncells == 400:
                    continue
                x, h, hu, meta = dam_break_case(1.0, h_right, ncells, degree)
                m = dam_break_metrics(1.0, h_right, x, h, hu, 0.5)
                m.update({"degree": degree, "ncells": ncells, "steps": meta["steps"]})
                rows.append(m)
                print(f"  stoker hR={h_right} deg={degree} n={ncells} "
                      f"relL1h={m['rel_l1_h']*100:.4f}%", flush=True)
                if degree == 1 and ncells == 400:
                    profiles[key] = (x, h, hu)
        RESULT["stoker"][key]["rows"] = rows

    # ---- 1-D lake at rest ------------------------------------------------
    RESULT["lake_rest_1d"] = {
        "flat": lake_at_rest_1d(final_time=1.0),
        "sloped_fixed": lake_at_rest_1d(final_time=1.0),
        "sloped_legacy_source": lake_at_rest_1d(final_time=1.0, legacy_source=True),
    }

    # ---- 2-D lake at rest (flat bed) ------------------------------------
    RESULT["lake_rest_2d"] = {"flat_16": lake_at_rest_2d(16, 0.5),
                              "flat_32": lake_at_rest_2d(32, 0.2)}
    RESULT["wall_clock_s"] = time.time() - t_start

    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(RESULT, fh, indent=2, sort_keys=True)

    _make_figures(profiles)
    print(f"done in {RESULT['wall_clock_s']:.1f}s", flush=True)


def _make_figures(profiles) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib unavailable: {exc}")
        return

    # 1) Stoker / Ritter profiles vs exact solution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    titles = {"hR_0.1": "Stoker, $h_R/h_L=0.1$", "hR_0.0": "Ritter (dry bed)"}
    for ax, key in zip(axes, ("hR_0.1", "hR_0.0")):
        x, h, hu = profiles[key]
        h_right = 0.1 if key == "hR_0.1" else 0.0
        xs = np.linspace(-3.0, 3.0, 4001)
        h_ex, u_ex, _ = stoker_exact(xs, 0.5, 1.0, h_right)
        ax.plot(xs, h_ex, "-", color="k", lw=1.4, label="exact (Ritter/Stoker)")
        ax.plot(x, h, "o", ms=2.2, color="tab:red", label="TensorDG P1, 400 cells")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$h$ [m]")
        ax.set_title(titles[key] + "   $t=0.5$ s")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dam_break_profiles.png"), dpi=140)
    plt.close(fig)

    # 2) convergence
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for h_right, style in ((0.1, "o-"), (0.5, "s-")):
        rows = [r for r in RESULT["stoker"][f"hR_{h_right}"]["rows"] if r["degree"] == 1]
        n = [r["ncells"] for r in rows]
        e = [r["rel_l1_h"] for r in rows]
        ax.loglog(n, e, style, label=f"Stoker $h_R={h_right}$")
    ax.loglog([100, 200, 400], [3e-3 * (100 / np.array([100, 200, 400])) ** 1.0][0] * np.ones(3),
              "k--", lw=1.0, label=r"slope $-1$")
    ax.set_xlabel("cells")
    ax.set_ylabel(r"relative $L_1$ error of $h$")
    ax.set_title("dam-break convergence (P1, HLL)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "dam_break_convergence.png"), dpi=140)
    plt.close(fig)

    # 3) well-balancedness: eta drift vs time, legacy vs fixed source
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    eta0 = 2.0
    zb_f = lambda xx: 0.4 * torch.sin(math.pi * xx / 2.0)
    for label, legacy, style in (("pre-fix source (2/(2k+1) factor)", True, "r--"),
                                 ("fixed source", False, "b-")):
        S1._rhs = _legacy_rhs_1d if legacy else _current_rhs_1d
        try:
            x, coeff, zb_faces, meta = solve_swe_1d(
                lambda xx: eta0 - zb_f(xx), lambda xx: torch.zeros_like(xx), zb_f,
                ncells=64, degree=1, final_time=0.5, domain=(-2.0, 2.0), cfl=0.3)
        finally:
            S1._rhs = _current_rhs_1d
        h, u, eta = reconstruct_fields(coeff, zb_faces)
        drift = (eta - eta0).abs()
        ax.semilogy(x.numpy(), np.maximum(drift.numpy(), 1e-18), style, lw=1.3, label=label)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel(r"$|\eta-\eta_0|$ [m] after $t=0.5$ s")
    ax.set_title("lake at rest on a sloping bed (64 cells, P1)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "lake_at_rest_drift.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()