"""Stability *envelope* of the shipped ``ns_dg_2d`` Dirichlet-wall operator.

Why this tool exists
--------------------
Two earlier deliveries each produced one spectral point and one claim:

* ``docs/reports/tensordg-ns-wall-spectrum.md`` (m2731/m2920) measured the
  *pre-m2782* wall operator at the shipped penalty (``sigma = 8`` for P1) and
  found ``max Re(lambda) > +350`` growing **linearly with the SIPG penalty**
  (``sigma = 0.5 -> 101.6 ... sigma = 16 -> 2960.9``, slope 1.00), concluding
  that "no penalty magnitude can repair the wall" -- the verdict was an honest
  FAIL of the wall-bounded transient.
* ``tests/test_ns_dg_2d_wall.py::test_wall_operator_has_no_growing_eigenmode``
  retires that limitation after the m2782 ``outward_sign`` fix, but checks a
  **single point** of the option space (``sigma = 8``, one gradient closure, the
  adjoint term implicitly off).
* The m2972 symmetric Nitsche adjoint (``wall_adjoint=True``) was verified on
  transients and on an analytic wall-bounded standing wave, but its *spectrum*
  was never measured, so "the adjoint adds dissipation" was an inference, not a
  measurement.

This tool measures the whole **stability envelope**: penalty sweep x wall
gradient closure x adjoint on/off x polynomial degree, at the analytic steady
Couette state (with the resting state as the residual-free control), and keeps
the pre-fix sign-blind wall flux as a **negative control** so the gate cannot
pass vacuously.

Usage
-----
``PYTHONPATH=. python3 tools/spectrum_ns_wall_envelope.py`` writes
``validation/ns_dg_2d_wall/stability_envelope.json`` and
``docs/figures/ns_dg_2d_wall_stability_envelope.png`` (~3 min, single core).
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall import build_state
from tools.diag_ns_wall_spectrum import wall_flux_variant
from tools.spectrum_ns_wall import _spectrum

REFERENCE_WALL_FLUX = ns._boundary_flux
# Byte-for-byte the wall flux shipped before the m2782 ``outward_sign`` fix.
PREFIX_WALL_FLUX = wall_flux_variant((1.0, 1.0, 1.0))

SIGMAS = (2.0, 8.0, 32.0, 128.0)
GRADIENTS = ("face", "fd2", "taylor")


def _penalty(degree):
    return 2.0 * (degree + 1.0) ** 2


def _case(state, sigma, gradient, adjoint, pre_fix=False):
    """One spectrum; ``pre_fix`` swaps in the sign-blind (m2731-era) wall flux."""
    ns._boundary_flux = PREFIX_WALL_FLUX if pre_fix else REFERENCE_WALL_FLUX
    try:
        t0 = time.time()
        result = _spectrum(state["coeff"], state, state["boundary"], (True, False),
                           sigma, gradient, "default",
                           wall_adjoint=adjoint)
    finally:
        ns._boundary_flux = REFERENCE_WALL_FLUX
    result["sigma"] = sigma
    result["wall_gradient"] = gradient
    result["wall_adjoint"] = adjoint
    result["pre_fix_wall_flux"] = pre_fix
    result["seconds"] = time.time() - t0
    return result


def _residual(state, sigma, gradient, adjoint):
    """Base residual of the analytic state (a discrete steady state would be 0)."""
    ns._boundary_flux = REFERENCE_WALL_FLUX
    rhs = ns._rhs_ns(state["coeff"], state["basis"], state["derivative"],
                     state["weights"], state["signs"], state["dx"], state["dy"],
                     (True, False), state["boundary"], 0.1,
                     0.1 * ns.CP / 0.71, sigma, gradient,
                     wall_adjoint=adjoint)
    return float(rhs.abs().max())


def _residual_peak(state, sigma, gradient, adjoint):
    """Location ``(nx_index, ny_index, k, l, variable)`` of the largest residual."""
    ns._boundary_flux = REFERENCE_WALL_FLUX
    rhs = ns._rhs_ns(state["coeff"], state["basis"], state["derivative"],
                     state["weights"], state["signs"], state["dx"], state["dy"],
                     (True, False), state["boundary"], 0.1,
                     0.1 * ns.CP / 0.71, sigma, gradient, wall_adjoint=adjoint)
    flat = int(rhs.reshape(-1).abs().argmax())
    return [int(i) for i in np.unravel_index(flat, rhs.shape)]


def section_residual_refinement(ny_list=(8, 16, 32, 64), degree=1,
                               configurations=(("face", False), ("face", True),
                                               ("fd2", False), ("fd2", True),
                                               ("taylor", False))):
    """Is the analytic steady Couette state a discrete steady state?

    A *consistent* stabilisation must leave a residual that vanishes under grid
    refinement.  This measures ``max|_rhs_ns|`` on the projected analytic state
    for every closure / adjoint combination, so the m2972 adjoint term -- which
    is proportional to the Dirichlet deficit -- is judged by its residual order,
    not only by its spectrum.
    """
    sigma = _penalty(degree)
    rows = {}
    for gradient, adjoint in configurations:
        tag = f"{gradient}|adjoint={'on' if adjoint else 'off'}"
        values, peaks = [], []
        for ny in ny_list:
            state = build_state((2, ny), degree)
            values.append(_residual(state, sigma, gradient, adjoint))
            peaks.append(_residual_peak(state, sigma, gradient, adjoint))
        orders = [float(np.log2(values[i] / values[i + 1])) for i in range(len(values) - 1)]
        rows[tag] = {"cells_y": list(ny_list), "residual": values,
                     "observed_order": orders,
                     "peak_dof_coarsest": peaks[0], "peak_dof_finest": peaks[-1],
                     "state_shape": list(build_state((2, ny_list[0]), degree)["coeff"].shape)}
    return rows


def main(out_json="validation/ns_dg_2d_wall/stability_envelope.json",
         out_png="docs/figures/ns_dg_2d_wall_stability_envelope.png"):
    p1 = build_state((2, 8), 1)
    p2 = build_state((2, 8), 2)
    sigma_p1, sigma_p2 = _penalty(1), _penalty(2)

    summary = {
        "mission": "m3595",
        "question": "for which (penalty, gradient closure, adjoint, degree) is the "
                    "shipped Dirichlet-wall operator free of a growing eigenmode?",
        "grid": {"couette_p1": [2, 8], "couette_p2": [2, 8]},
        "shipped_penalty": {"p1": sigma_p1, "p2": sigma_p2},
        "sigma_sweep": [], "gradient_matrix": [], "degree_matrix": [],
        "negative_control": [], "residual": {}, "meta": {},
    }

    # (a) penalty sweep at the default closure, adjoint off and on
    for sigma in SIGMAS:
        for adjoint in (False, True):
            summary["sigma_sweep"].append(
                _case(p1, sigma, ns.WALL_GRADIENT, adjoint))

    # (b) gradient closures at the shipped penalty, adjoint off and on
    for gradient in GRADIENTS:
        for adjoint in (False, True):
            summary["gradient_matrix"].append(
                _case(p1, sigma_p1, gradient, adjoint))

    # (c) degree 2 at its own shipped penalty
    for adjoint in (False, True):
        summary["degree_matrix"].append(_case(p2, sigma_p2, ns.WALL_GRADIENT, adjoint))

    # (d) negative control: the same sweep on the pre-m2782 sign-blind flux
    for sigma in SIGMAS:
        summary["negative_control"].append(_case(p1, sigma, ns.WALL_GRADIENT, False,
                                                 pre_fix=True))

    # (e) residual of the analytic steady state (context for the spectra)
    for tag, (state, sigma) in {"p1": (p1, sigma_p1), "p2": (p2, sigma_p2)}.items():
        summary["residual"][tag] = {
            "adjoint_off": _residual(state, sigma, ns.WALL_GRADIENT, False),
            "adjoint_on": _residual(state, sigma, ns.WALL_GRADIENT, True),
        }

    # (f) residual order of the analytic state (consistency of each closure)
    summary["residual_refinement"] = section_residual_refinement()

    summary["meta"] = {
        "shipped_wall_gradient": ns.WALL_GRADIENT,
        "wall_adjoint_factor": ns.WALL_ADJOINT_FACTOR,
        "mu": 0.1, "prandtl": 0.71, "wall_speed": 1.0,
        "spectrum_scale": {"p1": float(np.abs([c["min_real"] for c in
                                               summary["sigma_sweep"]]).max())},
    }

    pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_json).write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(summary, out_png)
    return summary


def make_figure(summary, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_top, ax_bottom, ax_res) = plt.subplots(1, 3, figsize=(16.0, 4.2))

    sweep = [c for c in summary["sigma_sweep"] if not c["wall_adjoint"]]
    adjoint = [c for c in summary["sigma_sweep"] if c["wall_adjoint"]]
    control = summary["negative_control"]
    sig = [c["sigma"] for c in sweep]
    floor = 1e-12

    ax_top.semilogx(sig, [max(c["max_real"], floor) for c in sweep], "o-",
                    color="#1f77b4", label="shipped (adjoint off)")
    ax_top.semilogx(sig, [max(c["max_real"], floor) for c in adjoint], "s-",
                    color="#2ca02c", label="shipped (adjoint on, m2972)")
    ax_top.semilogx(sig, [c["max_real"] for c in control], "^--", color="#d62728",
                    label="negative control: pre-m2782 sign-blind flux")
    ax_top.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax_top.set_xlabel(r"SIPG wall penalty $\sigma$")
    ax_top.set_ylabel(r"$\max\,\mathrm{Re}(\lambda)$")
    ax_top.set_title("growing eigenmode vs penalty (P1, 2x8, Couette)")
    ax_top.grid(alpha=0.3, which="both")
    ax_top.legend(fontsize=8, loc="lower right")

    labels, values, colors = [], [], []
    for c in summary["gradient_matrix"]:
        labels.append(f"{c['wall_gradient']}\nadj={'on' if c['wall_adjoint'] else 'off'}")
        values.append(c["min_real"])
        colors.append("#2ca02c" if c["wall_adjoint"] else "#1f77b4")
    ax_bottom.bar(range(len(values)), values, color=colors)
    ax_bottom.set_xticks(range(len(values)))
    ax_bottom.set_xticklabels(labels, fontsize=7)
    ax_bottom.set_ylabel(r"$\min\,\mathrm{Re}(\lambda)$  (wall dissipation)")
    ax_bottom.set_title(r"wall dissipation at the shipped $\sigma$ = 8")
    ax_bottom.grid(alpha=0.3, axis="y")

    styles = {"face|adjoint=off": ("o-", "#1f77b4"), "face|adjoint=on": ("s--", "#d62728"),
              "fd2|adjoint=off": ("d-.", "#2ca02c"), "fd2|adjoint=on": ("^:", "#9467bd"),
              "taylor|adjoint=off": ("v-", "#8c564b")}
    for tag, row in summary["residual_refinement"].items():
        style, color = styles.get(tag, ("-", "grey"))
        ax_res.loglog(row["cells_y"], row["residual"], style, color=color, label=tag)
    ax_res.set_xlabel(r"cells in $y$ (P1, $\sigma$ = 8)")
    ax_res.set_ylabel(r"$\max\,|\mathcal{R}(U_{\rm exact})|$")
    ax_res.set_title("residual of the analytic steady state")
    ax_res.grid(alpha=0.3, which="both")
    ax_res.legend(fontsize=8)

    fig.tight_layout()
    pathlib.Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    data = main()
    for block in ("sigma_sweep", "gradient_matrix", "degree_matrix", "negative_control"):
        print(f"--- {block}")
        for row in data[block]:
            print(f"  sigma={row['sigma']:6.1f} grad={row['wall_gradient']:5s} "
                  f"adj={row['wall_adjoint']!s:5s} prefix={row['pre_fix_wall_flux']!s:5s} "
                  f"maxRe={row['max_real']:+.4e} minRe={row['min_real']:+.4e}")
    print("--- residual", data["residual"])
    print("--- residual_refinement")
    for tag, row in data["residual_refinement"].items():
        print(f"  {tag:20s} cells={row['cells_y']} res="
              f"{['%.3e' % v for v in row['residual']]} order="
              f"{['%.2f' % v for v in row['observed_order']]} "
              f"peak={row['peak_dof_coarsest']}")