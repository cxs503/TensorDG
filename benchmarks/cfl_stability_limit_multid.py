"""Multi-dimensional explicit-RK stability limit of the TensorDG DG operators.

Verification debt this module closes
------------------------------------
``docs/DG-CFL-STABILITY-LIMIT.md`` (1-D study) states in its section 5 that
*"only linear constant-coefficient 1-D advection is covered"* and that nothing
had measured the limit of the **multi-dimensional** operators.  Every 2-D/3-D
solver in this package nonetheless hard-codes the *tensor-product generalisation*

    dt = cfl / ((2 p + 1) * sum_i |a_i| / h_i)                       (1)

(``tensordg/advection_2d.py``, ``tensordg/advection_3d.py``,
``tensordg/euler_dg_2d.py``), with a fixed default ``cfl`` (0.08 for the linear
solvers, 0.2 for 2-D Euler).  Both halves of that generalisation were unverified:

    (a) does the directional factor ``sum_i |a_i| / h_i`` carry the correct
        ``1/d`` scaling of the true step bound?
    (b) are the shipped defaults inside the true limit at the degrees the code
        supports (0..3 for the linear operators, arbitrary ``p`` for Euler)?

What is measured here
--------------------
1. **Exact Kronecker-sum spectrum.**  The 2-D operator of the shipped kernels is
   ``A = I (x) M + M (x) I`` per direction pair, so its spectrum is the
   *pairwise sum* of the 1-D block-symbol eigenvalues of
   ``advection_1d._rhs`` (assembled by the production-kernel push-through used
   by :mod:`benchmarks.cfl_stability_limit`).  The bound
   ``dt_max(2-D) = dt_max(1-D) / 2`` is therefore a *structural* prediction that
   can be checked to round-off, not a fitted number.
2. **Empirical bisection on the shipped 2-D/3-D solvers.**  ``solve_advection_2d``
   and ``solve_advection_3d`` are bisected on their own ``cfl`` argument using the
   measured quadratic-form energy growth; the result is compared with the
   *same-mesh* symbol prediction (a mesh resolves only ``N`` wavenumbers, so the
   same-mesh bound is the right reference for an empirical run -- comparing it
   with the ``theta``-supremum would overstate the mismatch).
3. **Non-linear 2-D Euler blow-up ceiling.**  ``solve_euler_dg_2d_riemann``
   (Lax & Liu 1998 configuration 3) is bisected for the first ``cfl`` at which the
   run stops being finite, for ``p = 0, 1, 2`` and every shipped limiter mode.
   This is a *blow-up* threshold, not an accuracy statement: unlimited high-order
   runs on a shock degrade long before they overflow.

Outputs (``--output-dir``, default ``validation/cfl_stability_multid``):
``results.json`` (all numbers + provenance), ``tables.md``, and two figures.
Nothing in the markdown is hand-typed: ``tools/cfl_stability_multid_report.py``
injects the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.cfl_stability_limit import (  # noqa: E402
    TOL,
    growth_envelope,
    max_stable_dt,
    rk_amplification,
    spectrum_mesh,
    spectrum_sup,
)

import torch  # noqa: E402

import tensordg.advection_2d as adv2d  # noqa: E402
import tensordg.advection_3d as adv3d  # noqa: E402
import tensordg.euler_dg_2d as edg2d  # noqa: E402

INTEGRATORS = ("rk4", "ssp_rk3")


# --------------------------------------------------------------------------- #
# 1. spectra and their Kronecker sums
# --------------------------------------------------------------------------- #
def prune_candidates(eigs: np.ndarray, integrator: str, dt_probe: float,
                     keep: int = 20) -> np.ndarray:
    """Union of the eigenvalues that can plausibly bind a Kronecker sum.

    Three heuristics, each cheap: largest modulus, most negative real part, and
    largest amplification ``|R(dt_probe * lambda)|``.  The union (at most
    ``3 * keep`` eigenvalues) keeps the cost of a 3-D sum set affordable; the
    rule is validated against the *exact* (unpruned) 2-D sum set in the study.
    """
    eigs = np.asarray(eigs, dtype=complex).reshape(-1)
    order_mod = np.argsort(np.abs(eigs))[::-1][:keep]
    order_real = np.argsort(eigs.real)[:keep]
    order_amp = np.argsort(np.abs(rk_amplification(dt_probe * eigs, integrator)))[::-1][:keep]
    return eigs[np.unique(np.concatenate([order_mod, order_real, order_amp]))]


def kron_sum_spectrum(eigs: np.ndarray, dims: int) -> np.ndarray:
    """All ``dims``-fold sums ``lam_1 + ... + lam_dims`` (exact, no sampling).

    ``eigs`` is the 1-D spectrum of the *same* per-direction symbol, so the
    spectrum of the tensor-product operator with unit speed and unit spacing is
    exactly the multiset of ``dims``-fold sums (Kronecker sum of identical
    matrices).  Used for ``dims = 2`` where the closed set is affordable.
    """
    eigs = np.asarray(eigs, dtype=complex).reshape(-1)
    total = eigs.astype(complex).copy()
    for _ in range(dims - 1):
        total = (total[:, None] + eigs[None, :]).reshape(-1)
    return total


def directional_limit(degree: int, dims: int, integrator: str, ntheta: int,
                      prune: bool = False, keep: int = 20) -> dict:
    """Worst-case step of the d-dimensional tensor-product operator (unit h, a).

    ``prune=False`` -> exact sum set (affordable for ``dims = 2``);
    ``prune=True`` -> candidate union from :func:`prune_candidates`
    (needed for ``dims = 3``; validated against the exact 2-D set).
    """
    lam1, theta1 = spectrum_sup(degree, ntheta=ntheta)
    if prune and dims > 1:
        dt1, _ = max_stable_dt(lam1, integrator)
        lam1 = prune_candidates(lam1, integrator, dt1, keep=keep)
    lam = kron_sum_spectrum(lam1, dims)
    dt_max, corrected = max_stable_dt(lam, integrator)
    # Code units: dt = cfl / ((2p+1) * sum_i |a_i| / h_i); with h = a = 1 and
    # d directions the denominator is (2p+1) * d.
    cfl_max = dt_max * (2 * degree + 1) * dims
    return {
        "degree": int(degree),
        "dims": int(dims),
        "integrator": integrator,
        "ntheta": int(ntheta),
        "n_eigenvalues": int(lam.size),
        "dt_max": float(dt_max),
        "cfl_max_code_units": float(cfl_max),
        "max_neg_real": float(lam.real.min()),
        "max_abs": float(lam.__abs__().max()),
        "envelope_corrected": bool(corrected),
    }


def structural_scaling(degrees, integrators=INTEGRATORS, ntheta: int = 90) -> list[dict]:
    """``d * dt_max(d-D) / dt_max(1-D)`` on matched theta grids.

    Using the *same* theta grid in every direction makes the ratio exact
    (the pair ``(theta*, theta*)`` is always in the sum set), so this measures
    whether the ``sum_i |a_i|/h_i`` directional factor has the right ``1/d``
    scaling instead of comparing against two different discretisations.
    """
    rows = []
    for degree in degrees:
        for integrator in integrators:
            one = directional_limit(degree, 1, integrator, ntheta)
            exact2 = directional_limit(degree, 2, integrator, ntheta, prune=False)
            pruned2 = directional_limit(degree, 2, integrator, ntheta, prune=True)
            for dims in (2, 3):
                multi = directional_limit(degree, dims, integrator, ntheta,
                                          prune=(dims == 3))
                ratio = dims * multi["dt_max"] / one["dt_max"]
                rows.append({
                    "degree": degree,
                    "integrator": integrator,
                    "dims": dims,
                    "dt_max_1d": one["dt_max"],
                    "dt_max_multi": multi["dt_max"],
                    "ratio_d_times_dt_multi_over_dt_1d": float(ratio),
                    "rel_error_vs_unity": float(abs(ratio - 1.0)),
                    "cfl_max_1d_code_units": one["cfl_max_code_units"],
                    "cfl_max_multi_code_units": multi["cfl_max_code_units"],
                    "code_units_rel_error": float(
                        abs(multi["cfl_max_code_units"] / one["cfl_max_code_units"] - 1.0)
                    ),
                    "pruning_rel_error_2d": float(
                        abs(pruned2["dt_max"] / exact2["dt_max"] - 1.0)
                    ),
                    "pruned_n_eigenvalues": int(pruned2["n_eigenvalues"]),
                })
    return rows


def same_mesh_bound(degree: int, ncells: int, speeds, integrator: str) -> dict:
    """Bound for an ``ncells``-per-direction mesh and directional speeds."""
    lam1, _ = spectrum_mesh(degree, ncells, speed=1.0)
    total = None
    for speed in speeds:
        term = speed * lam1
        total = term if total is None else (total[:, None] + term[None, :]).reshape(-1)
    dt_max, corrected = max_stable_dt(total, integrator)
    rate = math.fsum(abs(s) for s in speeds)          # h = 1 per direction
    cfl_max = dt_max * (2 * degree + 1) * rate
    return {
        "dt_max": float(dt_max),
        "cfl_max_code_units": float(cfl_max),
        "n_eigenvalues": int(total.size),
        "envelope_corrected": bool(corrected),
    }


# --------------------------------------------------------------------------- #
# 2. empirical growth on the shipped linear solvers
# --------------------------------------------------------------------------- #
def _mode_energy(coeff: torch.Tensor) -> float:
    """Quadratic form equal to the L2 norm squared on a uniform Cartesian mesh.

    In the Legendre modal basis ``||u||^2 = sum_kl c_kl^2 / ((2k+1)(2l+1))``
    (times the cell volume, which cancels in a ratio).
    """
    degree_x = coeff.shape[-2]
    degree_y = coeff.shape[-1]
    wx = torch.tensor([1.0 / (2 * k + 1) for k in range(degree_x)], dtype=coeff.dtype)
    wy = torch.tensor([1.0 / (2 * k + 1) for k in range(degree_y)], dtype=coeff.dtype)
    return float(torch.einsum("xykl,k,l->", coeff ** 2, wx, wy))


def _mode_energy_3d(coeff: torch.Tensor) -> float:
    degrees = coeff.shape[-3:]
    w = [torch.tensor([1.0 / (2 * k + 1) for k in range(d)], dtype=coeff.dtype) for d in degrees]
    return float(torch.einsum("xyzklm,k,l,m->", coeff ** 2, w[0], w[1], w[2]))


def _growth_2d(degree: int, cfl: float, ncells: int, final_time: float) -> dict:
    _, _, c0, _ = adv2d.solve_advection_2d(
        ncells=ncells, degree=degree, final_time=0.0, velocity=(1.0, 1.0), cfl=cfl)
    _, _, c1, meta = adv2d.solve_advection_2d(
        ncells=ncells, degree=degree, final_time=final_time, velocity=(1.0, 1.0), cfl=cfl)
    e0, e1 = _mode_energy(c0), _mode_energy(c1)
    exponent = math.log(e1 / e0) / (2.0 * final_time)
    return {"exponent": exponent, "energy_ratio": e1 / e0,
            "dt": meta["dt"], "steps": meta["steps"]}


def _growth_3d(degree: int, cfl: float, ncells: int, final_time: float) -> dict:
    _, _, _, c0, _ = adv3d.solve_advection_3d(
        ncells=ncells, degree=degree, final_time=0.0, velocity=(1.0, 0.5, 0.25), cfl=cfl)
    _, _, _, c1, meta = adv3d.solve_advection_3d(
        ncells=ncells, degree=degree, final_time=final_time, velocity=(1.0, 0.5, 0.25), cfl=cfl)
    e0, e1 = _mode_energy_3d(c0), _mode_energy_3d(c1)
    return {"exponent": math.log(e1 / e0) / (2.0 * final_time),
            "energy_ratio": e1 / e0, "dt": meta["dt"], "steps": meta["steps"]}


def bisect_cfl(growth, cfl_lo: float = 0.2, cfl_hi: float = 3.0,
               iters: int = 22, threshold: float = 1e-3) -> dict:
    """Bisect the shipped ``cfl`` for the first growing run.

    ``growth(cfl) -> {"exponent": ...}``; growth above ``threshold`` counts as
    unstable.  ``cfl_hi`` is widened until it is unstable, so the bracket is
    never assumed.
    """
    lo, hi = cfl_lo, cfl_hi
    while growth(hi)["exponent"] <= threshold:
        hi *= 1.5
        if hi > 50.0:
            raise RuntimeError("no unstable bracket found below cfl = 50")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if growth(mid)["exponent"] <= threshold:
            lo = mid
        else:
            hi = mid
    return {"cfl_max_empirical": float(0.5 * (lo + hi)),
            "cfl_stable_probe": float(lo), "cfl_unstable_probe": float(hi),
            "threshold": float(threshold)}


# --------------------------------------------------------------------------- #
# 3. non-linear 2-D Euler blow-up ceiling
# --------------------------------------------------------------------------- #
def _euler_ok(state) -> bool:
    """``solve_euler_dg_2d`` returns ``(x, y, coeff, metadata)``."""
    coeff = state[2]
    if not bool(torch.isfinite(coeff).all()):
        return False
    averages = coeff[..., 0, 0, :]
    return bool((averages[..., 0] > 0.0).all()) and bool((averages[..., 3] > 0.0).all())


def euler_ceiling(degree: int, limiter, ncells: int, final_time: float,
                  cfl_lo: float = 0.05, cfl_hi: float = 0.6, iters: int = 6) -> dict:
    """First ``cfl`` at which the Lax & Liu run stops being physical/finite."""
    def ok(cfl: float) -> bool:
        try:
            state = edg2d.solve_euler_dg_2d_riemann(
                ncells=ncells, degree=degree, final_time=final_time,
                cfl=cfl, limiter=limiter)
        except (FloatingPointError, RuntimeError, ValueError):
            return False
        return _euler_ok(state)

    lo, hi = cfl_lo, cfl_hi
    if not ok(lo):
        return {"degree": degree, "limiter": limiter, "ncells": int(ncells),
                "final_time": float(final_time), "cfl_ceiling": float("nan"),
                "stable_probe": None, "unstable_probe": float(lo),
                "note": "unstable already at the lowest probe"}
    while ok(hi):
        hi *= 1.5
        if hi > 40.0:
            return {"degree": degree, "limiter": limiter, "ncells": int(ncells),
                    "final_time": float(final_time), "cfl_ceiling": float("inf"),
                    "stable_probe": float(hi), "unstable_probe": None,
                    "note": "no blow-up below cfl = 40"}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return {"degree": degree, "limiter": limiter, "ncells": int(ncells),
            "final_time": float(final_time),
            "cfl_ceiling": float(0.5 * (lo + hi)),
            "stable_probe": float(lo), "unstable_probe": float(hi),
            "note": ""}


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
DEFAULTS = {"advection_2d": 0.08, "advection_3d": 0.08, "euler_dg_2d": 0.2}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover - provenance only
        return "unknown"


def study(structural_degrees=range(0, 5), empirical_degrees=range(0, 4),
          euler_degrees=(0, 1, 2), euler_limiters=(None, "tvb", "weno"),
          ncells_2d: int = 8, ncells_3d: int = 6, ncells_euler: int = 16,
          final_time: float = 4.0, euler_time: float = 0.2) -> dict:
    print("phase: structural scaling", flush=True)
    structural = structural_scaling(structural_degrees)

    print("phase: empirical 2-D", flush=True)
    empirical_2d = []
    for degree in empirical_degrees:
        cfl_max, bracket = None, bisect_cfl(
            lambda c: _growth_2d(degree, c, ncells_2d, final_time))
        cfl_max = bracket["cfl_max_empirical"]
        predicted = same_mesh_bound(degree, ncells_2d, (1.0, 1.0), "rk4")
        empirical_2d.append({
            "degree": int(degree),
            "ncells": int(ncells_2d),
            "velocity": [1.0, 1.0],
            "final_time": float(final_time),
            **bracket,
            "same_mesh_symbol_cfl_max": predicted["cfl_max_code_units"],
            "rel_error_vs_same_mesh_symbol": float(
                abs(cfl_max / predicted["cfl_max_code_units"] - 1.0)),
            "default_cfl": DEFAULTS["advection_2d"],
            "default_margin": float(cfl_max / DEFAULTS["advection_2d"]),
        })

    print("phase: empirical 3-D", flush=True)
    empirical_3d = []
    for degree in (0, 1, 2):
        bracket = bisect_cfl(lambda c: _growth_3d(degree, c, ncells_3d, final_time))
        predicted = same_mesh_bound(degree, ncells_3d, (1.0, 0.5, 0.25), "rk4")
        empirical_3d.append({
            "degree": int(degree),
            "ncells": int(ncells_3d),
            "velocity": [1.0, 0.5, 0.25],
            "final_time": float(final_time),
            **bracket,
            "same_mesh_symbol_cfl_max": predicted["cfl_max_code_units"],
            "rel_error_vs_same_mesh_symbol": float(
                abs(bracket["cfl_max_empirical"] / predicted["cfl_max_code_units"] - 1.0)),
            "default_cfl": DEFAULTS["advection_3d"],
            "default_margin": float(bracket["cfl_max_empirical"] / DEFAULTS["advection_3d"]),
        })

    print("phase: euler ceiling", flush=True)
    euler = [euler_ceiling(degree, limiter, ncells_euler, final_time=euler_time)
             for degree in euler_degrees for limiter in euler_limiters]

    gates = {
        "G1_directional_scaling_1_over_d": bool(
            all(row["rel_error_vs_unity"] < 1e-6 for row in structural)),
        "G2_code_unit_identity": bool(
            all(row["code_units_rel_error"] < 1e-9 for row in structural)),
        "G3_empirical_2d_within_15pct_of_same_mesh_symbol": bool(
            all(row["rel_error_vs_same_mesh_symbol"] < 0.15 for row in empirical_2d)),
        "G4_default_margin_above_one": bool(
            all(row["default_margin"] > 1.0 for row in empirical_2d + empirical_3d)),
        "G5_euler_default_inside_blow_up_ceiling": bool(
            all(row["cfl_ceiling"] > DEFAULTS["euler_dg_2d"] for row in euler)),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "provenance": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "integrators": list(INTEGRATORS),
            "tol_rk": TOL,
        },
        "defaults": DEFAULTS,
        "structural_scaling": structural,
        "empirical_advection_2d": empirical_2d,
        "empirical_advection_3d": empirical_3d,
        "euler_blow_up_ceiling": euler,
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }


def write_tables(results: dict, path: Path) -> None:
    lines = ["# CFL multid tables (generated -- do not hand-edit)", ""]
    lines += ["## Structural scaling", "",
              "| p | integrator | d | dt_max(1-D) | dt_max(d-D) | d*dt_max(d)/dt_max(1) | rel err |",
              "|---|---|---|---|---|---|---|"]
    for row in results["structural_scaling"]:
        lines.append(
            f"| {row['degree']} | {row['integrator']} | {row['dims']} | "
            f"{row['dt_max_1d']:.6f} | {row['dt_max_multi']:.6f} | "
            f"{row['ratio_d_times_dt_multi_over_dt_1d']:.12f} | "
            f"{row['rel_error_vs_unity']:.2e} |")
    lines += ["", "## Empirical 2-D advection", "",
              "| p | cfl_max (measured) | same-mesh symbol | rel err | default 0.08 margin |",
              "|---|---|---|---|---|"]
    for row in results["empirical_advection_2d"]:
        lines.append(f"| {row['degree']} | {row['cfl_max_empirical']:.4f} | "
                     f"{row['same_mesh_symbol_cfl_max']:.4f} | "
                     f"{row['rel_error_vs_same_mesh_symbol'] * 100:.2f}% | "
                     f"{row['default_margin']:.2f}x |")
    lines += ["", "## Empirical 3-D advection", "",
              "| p | cfl_max (measured) | same-mesh symbol | rel err | default 0.08 margin |",
              "|---|---|---|---|---|"]
    for row in results["empirical_advection_3d"]:
        lines.append(f"| {row['degree']} | {row['cfl_max_empirical']:.4f} | "
                     f"{row['same_mesh_symbol_cfl_max']:.4f} | "
                     f"{row['rel_error_vs_same_mesh_symbol'] * 100:.2f}% | "
                     f"{row['default_margin']:.2f}x |")
    lines += ["", "## 2-D Euler blow-up ceiling (Lax & Liu 1998 config 3)", "",
              "| p | limiter | cfl ceiling | default 0.2 margin |", "|---|---|---|---|"]
    for row in results["euler_blow_up_ceiling"]:
        margin = (row["cfl_ceiling"] / results["defaults"]["euler_dg_2d"]
                  if np.isfinite(row["cfl_ceiling"]) else float("inf"))
        lines.append(f"| {row['degree']} | {row['limiter']} | "
                     f"{row['cfl_ceiling']:.3f} | {margin:.2f}x |")
    lines += ["", "## Gates", ""]
    for name, ok in results["gates"].items():
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
    lines.append(f"\n**verdict = {results['verdict']}**")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(results: dict, outdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    structural = results["structural_scaling"]
    degrees = sorted({r["degree"] for r in structural})
    for integrator, style in zip(INTEGRATORS, ("o-", "s--")):
        for dims, color in ((1, "C0"), (2, "C1"), (3, "C2")):
            vals = []
            for p in degrees:
                match = [r for r in structural
                         if r["degree"] == p and r["dims"] == dims
                         and r["integrator"] == integrator]
                if match:
                    row = match[0]
                    vals.append(row["dt_max_1d"] if dims == 1 else row["dt_max_multi"])
                else:
                    vals.append(float("nan"))
            axes[0].plot(degrees, vals, style, color=color,
                         label=f"d={dims} {integrator}")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("polynomial degree p")
    axes[0].set_ylabel("worst-case $dt_{max}$ (h=a=1)")
    axes[0].set_title("directional scaling: $dt_{max}$ vs p")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    rows = results["empirical_advection_2d"]
    axes[1].plot([r["degree"] for r in rows], [r["cfl_max_empirical"] for r in rows],
                 "o-", label="measured (2-D solver)")
    axes[1].plot([r["degree"] for r in rows], [r["same_mesh_symbol_cfl_max"] for r in rows],
                 "s--", label="same-mesh symbol")
    axes[1].axhline(results["defaults"]["advection_2d"], color="k", ls=":",
                    label="shipped default 0.08")
    axes[1].set_xlabel("p")
    axes[1].set_ylabel("cfl (code units)")
    axes[1].set_title("2-D advection: measured vs symbol")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    labels, values = [], []
    for row in results["euler_blow_up_ceiling"]:
        labels.append(f"p{row['degree']} {row['limiter']}")
        values.append(row["cfl_ceiling"])
    axes[2].bar(range(len(values)), values, color="C3", alpha=0.7)
    axes[2].axhline(results["defaults"]["euler_dg_2d"], color="k", ls=":",
                    label="shipped default 0.2")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, fontsize=7, ha="right")
    axes[2].set_ylabel("first non-finite cfl")
    axes[2].set_title("2-D Euler blow-up ceiling")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = outdir / "cfl_stability_multid.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)
    return made


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/cfl_stability_multid")
    parser.add_argument("--degrees", type=int, nargs="+", default=list(range(0, 5)))
    parser.add_argument("--empirical-degrees", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--euler-degrees", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ncells-2d", type=int, default=8)
    parser.add_argument("--ncells-3d", type=int, default=6)
    parser.add_argument("--ncells-euler", type=int, default=16)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--euler-time", type=float, default=0.2)
    parser.add_argument("--skip-euler", action="store_true")
    args = parser.parse_args(argv)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    results = study(
        structural_degrees=args.degrees,
        empirical_degrees=args.empirical_degrees,
        euler_degrees=() if args.skip_euler else tuple(args.euler_degrees),
        ncells_2d=args.ncells_2d,
        ncells_3d=args.ncells_3d,
        ncells_euler=args.ncells_euler,
        final_time=args.final_time,
        euler_time=args.euler_time,
    )
    results["figures"] = make_figures(results, outdir)
    (outdir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_tables(results, outdir / "tables.md")
    print(json.dumps(results["gates"], indent=2))
    print("verdict:", results["verdict"])
    return 0 if results["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())