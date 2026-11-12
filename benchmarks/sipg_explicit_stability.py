"""Measured explicit-Runge--Kutta stability limit of the SIPG diffusion operator.

Motivation (verification debt)
------------------------------
Two production modules limit their explicit time step with a *hand-set constant*
that was never measured on the operators they actually use:

* ``tensordg/advection_diffusion.py`` (RK4, ``nu > 0``)::

      # spectral radius of the SIPG operator measures ~50*sigma*nu/h^2 for
      # p in [0,3] (power iteration, see the benchmark report)
      dt_diff = 0.5 * 2.78 * min(hx, hy)**2 / (50.0 * sigma * nu)

* ``tensordg/ns_dg_2d.py`` (SSP-RK3, ``mu > 0``)::

      # S = dt * diffusivity * (p+1)**4 / h**2 was measured by bisection on the
      # shear-wave and Couette cases (S_max ~ 0.3-0.5); VISCOUS_STABILITY = 0.1
      rate_viscous = diffusivity * (degree + 1.0)**4 / (VISCOUS_STABILITY * h_min**2)

Neither the constant ``50`` nor the range ``0.3-0.5`` has a reproducible artifact,
and neither statement says how they vary with the polynomial degree.  This
benchmark measures the quantity both of them are proxy-ing for:

.. math::

   \\lambda_{\\max}\\big(M^{-1}K\\big) \\;=\\; c(p,\\alpha)\\,\\alpha\\,\\nu\\,/\\,h^{2},
   \\qquad \\alpha = \\text{penalty}\\,(p+1)^{2}

where :math:`M du/dt = -K u` is the SIPG semi-discrete diffusion system and
:math:`\\alpha` is the penalty coefficient the code uses (``penalty*(p+1)**2``).
With every eigenvalue of :math:`M^{-1}K` real and non-negative (checked here),
the explicit RK4 limit is :math:`dt_{\\max} = z_{\\rm RK4}/\\lambda_{\\max}` with
:math:`z_{\\rm RK4}` the negative-real-axis stability interval of the integrator.

Four independent measurement channels are reported:

1. **1-D exact** -- ``heat_1d.assemble_sipg_1d`` (production SIPG assembly) gives
   dense ``M`` and ``K``; the generalised eigenvalues of ``(K, M)`` are computed
   exactly via the symmetric transform ``M^{-1/2} K M^{-1/2}``.
2. **2-D exact** -- ``sipg_sparse.assemble_poisson_2d_sparse`` (production sparse
   assembly, interior faces identical to ``advection_diffusion._diffusion``);
   extreme eigenvalues by Lanczos on the symmetrised operator.
3. **2-D production closure** -- power iteration on the *actual*
   ``advection_diffusion.rhs`` with ``velocity = 0``, so the measured spectral
   radius includes that module's own boundary/face bookkeeping.
4. **Empirical bisection** -- RK4 stepping of ``advection_diffusion.rhs`` at a
   user-chosen ``dt``; the limit is where the modal-norm growth exponent changes
   sign.  This validates the whole chain (assembly -> spectrum -> RK
   amplification -> observed stability) end to end.

References for the anchors
--------------------------
* Classical RK4 negative-real-axis stability interval 2.785293563405282
  (Hairer, Norsett & Wanner 1993, *Solving ODEs I*, §IV.2 / Table 4.1).
* SSP-RK3 (Shu & Osher 1988) stability polynomial ``1 + z + z^2/2 + z^3/6``;
  negative-real-axis interval 2.5127... (recomputed here, not typed).
* SIPG / symmetric interior penalty: Arnold, Brezzi, Cockburn & Marini (2002),
  SIAM J. Numer. Anal. 39(5), 1749-1779 (coercivity requires a penalty of the
  order ``(p+1)^2``; the constants of the resulting spectral radius are not
  tabulated there, which is why they are measured here).
* Eigenvalue scaling of DG diffusion operators like ``h^{-2}(p+1)^{4}``:
  Krivodonova & Flaherty (2003), Appl. Numer. Math. 47, 415-431 (§3 spectral
  radius of the DG diffusion operator); also used as the ``(p+1)^4`` heuristic
  inside ``ns_dg_2d``.

Usage
-----
::

    python -m benchmarks.sipg_explicit_stability --output-dir validation/sipg_explicit_stability
    python tools/sipg_explicit_stability_report.py     # regenerate the markdown report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from tensordg import advection_diffusion as ad
from tensordg import sipg_sparse
from tensordg.heat_1d import assemble_sipg_1d

#: Classical RK4 stability interval on the negative real axis (literature anchor).
RK4_REAL_AXIS_LITERATURE = 2.785293563405282

#: Degree range ``advection_diffusion.solve`` accepts.
AD_DEGREES = (0, 1, 2, 3)

#: Constants that the production code hard-codes (kept in sync by the tests).
CODE_AD_DIFF_CONSTANT = 50.0
CODE_AD_DIFF_SAFETY = 0.5
CODE_NS_VISCOUS_STABILITY = 0.1


# --------------------------------------------------------------------------- #
# RK stability functions / intervals
# --------------------------------------------------------------------------- #
def rk4_amplification(z: complex) -> complex:
    """Classical RK4 amplification factor ``R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24``."""
    return 1.0 + z + z * z / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0


def ssp_rk3_amplification(z: complex) -> complex:
    """SSP-RK3 (Shu--Osher) amplification factor ``1 + z + z^2/2 + z^3/6``."""
    return 1.0 + z + z * z / 2.0 + z ** 3 / 6.0


AMPLIFIERS: dict[str, Callable[[complex], complex]] = {
    "rk4": rk4_amplification,
    "ssp_rk3": ssp_rk3_amplification,
}


def stability_interval(integrator: str, hi: float = 10.0,
                       tol: float = 1e-14) -> float:
    """Largest ``x > 0`` with ``|R(-x)| <= 1`` (negative-real-axis interval).

    Bisection on the crossing of ``|R(-x)| = 1``; ``|R(-x)|`` is monotone past
    the crossing for both integrators in :data:`AMPLIFIERS`.
    """
    R = AMPLIFIERS[integrator]
    lo = 0.0
    while abs(R(complex(-hi, 0.0))) <= 1.0:
        hi *= 2.0
        if hi > 1e4:
            raise RuntimeError("no crossing found")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if abs(R(complex(-mid, 0.0))) <= 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# 1-D exact spectrum (production SIPG assembly)
# --------------------------------------------------------------------------- #
def spectrum_1d(degree: int, ncells: int, penalty: float,
                dtype: torch.dtype = torch.float64) -> dict:
    """Exact extreme generalised eigenvalues of ``(K, M)`` for the 1-D SIPG assembly.

    ``heat_1d.assemble_sipg_1d`` returns ``M`` (SPD, block diagonal) and ``K``
    (SIPG stiffness) of ``M du/dt = -K u``.  The symmetry transform
    ``A = M^{-1/2} K M^{-1/2}`` has the same spectrum as ``M^{-1}K``.
    """
    data = assemble_sipg_1d(ncells=ncells, degree=degree, penalty=penalty,
                            device="cpu", dtype=dtype)
    mass, stiff = data["mass_matrix"], data["diffusion"]
    scale = torch.diag(mass).clamp_min(0.0).sqrt()
    sym = stiff / scale[:, None] / scale[None, :]
    sym = 0.5 * (sym + sym.transpose(0, 1))
    eig = torch.linalg.eigvalsh(sym).cpu().numpy()
    h = data["h"]
    alpha = penalty * (degree + 1) ** 2
    return {
        "degree": degree,
        "ncells": ncells,
        "penalty": penalty,
        "alpha": alpha,
        "h": h,
        "dofs": int(eig.size),
        "lam_max": float(eig[-1]),
        "lam_min": float(eig[0]),
        # dimensionless spectral constant: lam_max = c * alpha * nu / h^2 with nu = 1
        "c": float(eig[-1]) * h * h / alpha,
        "min_dimensionless": float(eig[0]) * h * h / alpha,
    }


# --------------------------------------------------------------------------- #
# 2-D: assembled spectrum + production-operator power iteration
# --------------------------------------------------------------------------- #
def _mass_diagonal_2d(nx: int, ny: int, degree: int) -> np.ndarray:
    """Diagonal 2-D modal mass matrix (product basis, one diagonal per cell)."""
    modes = degree + 1
    hx, hy = 1.0 / nx, 1.0 / ny
    k = torch.arange(modes, dtype=torch.float64)
    local = torch.outer(hx / (2 * k + 1.0), hy / (2 * k + 1.0)).reshape(-1).numpy()
    return np.tile(local, nx * ny)


def spectrum_2d(nx: int, ny: int, degree: int, penalty: float) -> dict:
    """Extreme eigenvalues of ``M^{-1}K`` for the production 2-D SIPG assembly."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh

    stiff, _ = sipg_sparse.assemble_poisson_2d_sparse(
        nx, ny, degree, penalty, torch.device("cpu"), torch.float64)
    scale = np.sqrt(_mass_diagonal_2d(nx, ny, degree))
    dinv = diags(1.0 / scale)
    sym = (dinv @ stiff @ dinv).tocsr()
    sym = ((sym + sym.T) * 0.5).tocsc()
    lam_max = float(eigsh(sym, k=1, which="LA", return_eigenvectors=False,
                          tol=1e-12)[0])
    lam_min = float(eigsh(sym, k=1, which="SA", return_eigenvectors=False,
                          tol=1e-12)[0])
    h = min(1.0 / nx, 1.0 / ny)
    alpha = penalty * (degree + 1) ** 2
    return {
        "nx": nx, "ny": ny, "degree": degree, "penalty": penalty, "alpha": alpha,
        "h": h, "dofs": nx * ny * (degree + 1) ** 2,
        "lam_max": lam_max, "lam_min": lam_min,
        "c": lam_max * h * h / alpha,
        "min_dimensionless": lam_min * h * h / alpha,
    }


def _zero_velocity(t, x, y):
    return torch.zeros_like(x), torch.zeros_like(y)


def production_lanczos_2d(nx: int, ny: int, degree: int, penalty: float,
                           lanczos_steps: int = 40) -> dict:
    """Spectral radius of the *actual* ``advection_diffusion`` diffusion operator.

    ``nu = 1`` with identically zero velocity makes ``rhs`` the linear map
    ``L = -M^{-1}K`` for the production face bookkeeping (``boundary =
    "extrapolate"``).  ``L`` is symmetric in the mass inner product, so a
    ``lanczos_steps``-step Lanczos process with ``<u, v>_M`` yields accurate
    extreme Ritz values from black-box mat-vecs only.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    ops = ad.Operators(degree, device, dtype)
    hx, hy, xc, yc, xf, yf = ad._mesh_arrays(nx, ny, ad.UNIT_SQUARE, device, dtype)
    geo = (xc, yc, xf, yf)
    modes = degree + 1
    k = torch.arange(modes, dtype=dtype)
    local_mass = torch.outer(hx / (2 * k + 1.0), hy / (2 * k + 1.0))
    mass = local_mass.reshape(1, 1, modes, modes).expand(nx, ny, modes, modes)

    def apply(state):
        return ad.rhs(state, 0.0, _zero_velocity, 1.0, hx, hy, ops,
                      "extrapolate", penalty, geo)

    def inner(u, v):
        return float((u * v * mass).sum())

    gen = torch.Generator().manual_seed(20261112)
    q = torch.randn((nx, ny, modes, modes), generator=gen, dtype=dtype)
    q = q / math.sqrt(inner(q, q))
    q_prev = torch.zeros_like(q)
    beta_prev = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    for step in range(lanczos_steps):
        w = apply(q)
        alpha = inner(w, q)
        w = w - alpha * q - beta_prev * q_prev
        # full reorthogonalisation is unnecessary for extreme Ritz values at 40 steps
        beta = math.sqrt(max(inner(w, w), 0.0))
        alphas.append(alpha)
        if step + 1 == lanczos_steps or beta <= 1e-14 * abs(alpha):
            break
        betas.append(beta)
        q_prev, q = q, w / beta
        beta_prev = beta
    from scipy.linalg import eigvalsh_tridiagonal
    ritz = eigvalsh_tridiagonal(np.array(alphas), np.array(betas)) if betas else np.array(alphas)
    h = min(hx, hy)
    alpha_pen = penalty * (degree + 1) ** 2
    lo, hi = float(ritz.min()), float(ritz.max())
    abs_max = max(abs(lo), abs(hi))
    return {
        "nx": nx, "ny": ny, "degree": degree, "penalty": penalty, "alpha": alpha_pen,
        "h": h,
        # ``L = -M^{-1}K`` is negative semidefinite; the stability bound uses |lambda|max
        "lam_abs_max": abs_max,
        # NB: Lanczos assumes symmetry; the production operator is symmetric in the
        # mass inner product only up to O(5%) (see ``production_dense_2d``), so the
        # positive-side Ritz values here are artefacts, not growth modes.
        "ritz_max": hi,
        "ritz_min": lo,
        "c": abs_max * h * h / alpha_pen,
        "lanczos_steps": len(alphas),
    }


def production_dense_2d(nx: int, degree: int, penalty: float,
                        boundary: str = "extrapolate") -> dict:
    """Exact spectrum of the *actual* ``advection_diffusion.rhs`` diffusion operator.

    Builds the dense matrix of ``L = -M^{-1}K`` by applying ``rhs`` (``nu = 1``,
    zero velocity) to every unit modal vector, so no assumption about symmetry or
    structure enters.  Cost is ``O(D)`` residual evaluations for ``D = nx*ny*(p+1)^2``
    degrees of freedom, hence the small meshes.
    """
    device = torch.device("cpu")
    dtype = torch.float64
    ops = ad.Operators(degree, device, dtype)
    hx, hy, xc, yc, xf, yf = ad._mesh_arrays(nx, nx, ad.UNIT_SQUARE, device, dtype)
    geo = (xc, yc, xf, yf)
    modes = degree + 1
    dofs = nx * nx * modes * modes
    dense = np.zeros((dofs, dofs))
    for j in range(dofs):
        vec = torch.zeros((nx, nx, modes, modes), dtype=dtype)
        vec.reshape(-1)[j] = 1.0
        out = ad.rhs(vec, 0.0, _zero_velocity, 1.0, hx, hy, ops, boundary, penalty, geo)
        dense[:, j] = out.reshape(-1).numpy()
    eig = np.linalg.eigvals(dense)
    k = torch.arange(modes, dtype=dtype)
    local_mass = torch.outer(hx / (2 * k + 1.0), hy / (2 * k + 1.0)).reshape(-1).numpy()
    mass = np.tile(local_mass, nx * nx)
    root = np.sqrt(mass)
    # symmetry transform of L = -M^{-1}K :  M^{1/2} L M^{-1/2} = -M^{-1/2} K M^{-1/2}
    sym = (root[:, None] * dense) / root[None, :]
    asymmetry = float(np.abs(sym - sym.T).max() / np.abs(sym).max())
    h = min(hx, hy)
    alpha = penalty * (degree + 1) ** 2
    lam_abs = float(np.abs(eig).max())
    return {
        "nx": nx, "degree": degree, "penalty": penalty, "alpha": alpha, "h": h,
        "boundary": boundary, "dofs": dofs,
        "lam_abs_max": lam_abs,
        "re_max": float(eig.real.max()),
        "im_max": float(np.abs(eig.imag).max()),
        "re_max_relative": float(eig.real.max()) / lam_abs,
        "symmetry_violation": asymmetry,
        "c": lam_abs * h * h / alpha,
    }


# --------------------------------------------------------------------------- #
# empirical bisection on the production RK4 path
# --------------------------------------------------------------------------- #
def empirical_dt_max_2d(degree: int, nx: int, penalty: float,
                        final_time_factor: float = 120.0,
                        iterations: int = 14) -> dict:
    """Bisect the RK4 step size on ``advection_diffusion.rhs`` (production operator).

    ``dt_max`` is the largest ``dt`` for which the modal-norm growth exponent
    ``log(||u(T)||/||u(0)||)/T`` stays non-positive for
    ``T = final_time_factor * dt`` (fixed number of steps, so the test isolates
    ``dt`` rather than the integration window).
    """
    device = torch.device("cpu")
    dtype = torch.float64
    ops = ad.Operators(degree, device, dtype)
    hx, hy, xc, yc, xf, yf = ad._mesh_arrays(nx, nx, ad.UNIT_SQUARE, device, dtype)
    geo = (xc, yc, xf, yf)
    gen = torch.Generator().manual_seed(20261112)
    state0 = torch.randn((nx, nx, degree + 1, degree + 1), generator=gen,
                         dtype=dtype)

    def f(state):
        return ad.rhs(state, 0.0, _zero_velocity, 1.0, hx, hy, ops,
                      "extrapolate", penalty, geo)

    def growth(dt: float) -> float:
        steps = int(round(final_time_factor))
        state = state0.clone()
        for _ in range(steps):
            k1 = f(state)
            k2 = f(state + 0.5 * dt * k1)
            k3 = f(state + 0.5 * dt * k2)
            k4 = f(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            scale = float(state.norm())
            if not math.isfinite(scale) or scale > 1e12:
                return 1e3
        T = steps * dt
        return math.log(float(state.norm()) / float(state0.norm())) / T

    lo, hi = 1e-9, 1e-1
    # bracket: lo stable, hi unstable
    while growth(lo) > 0.0:
        lo *= 0.1
    while growth(hi) <= 0.0:
        hi *= 10.0
        if hi > 1e3:
            raise RuntimeError("could not bracket dt_max")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if growth(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return {
        "degree": degree, "nx": nx, "penalty": penalty,
        "dt_max_empirical": lo, "dt_above": hi,
        "growth_below": growth(lo), "growth_above": growth(hi),
        "steps": int(round(final_time_factor)),
    }


# --------------------------------------------------------------------------- #
# consequences for the two production heuristics
# --------------------------------------------------------------------------- #
def ad_dt_diff(degree: int, penalty: float, h: float, nu: float = 1.0) -> float:
    """Exactly the ``advection_diffusion.solve`` diffusion step-size rule."""
    sigma = penalty * (degree + 1) ** 2
    return CODE_AD_DIFF_SAFETY * 2.78 * h ** 2 / (CODE_AD_DIFF_CONSTANT * sigma * nu)


def ns_S_max(degree: int, penalty: float, c: float, z: float) -> float:
    """``S_max = dt_max * D * (p+1)^4 / h^2`` for the ``ns_dg_2d`` step-size group."""
    return z * (degree + 1) ** 4 / (c * penalty * (degree + 1) ** 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            text=True).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def collect(results: dict) -> dict:
    z_rk4 = stability_interval("rk4")
    z_ssp = stability_interval("ssp_rk3")
    results["rk_intervals"] = {
        "rk4": z_rk4,
        "rk4_literature": RK4_REAL_AXIS_LITERATURE,
        "rk4_relative_error": abs(z_rk4 - RK4_REAL_AXIS_LITERATURE) / RK4_REAL_AXIS_LITERATURE,
        "ssp_rk3": z_ssp,
    }

    pen_1d, pen_2d = 2.0, 2.0
    results["spectrum_1d"] = [
        spectrum_1d(p, n, pen_1d)
        for p in (1, 2, 3, 4, 5)
        for n in (4, 8, 16, 32)
    ]
    results["spectrum_1d_penalty20"] = [
        spectrum_1d(p, n, 20.0)
        for p in (1, 2, 3)
        for n in (8, 16, 32)
    ]
    results["spectrum_2d"] = [
        spectrum_2d(n, n, p, pen_2d)
        for p in AD_DEGREES
        for n in (4, 8, 12)
    ]
    results["production_lanczos_2d"] = [
        production_lanczos_2d(n, n, p, pen_2d)
        for p in AD_DEGREES
        for n in (4, 8, 12, 16)
    ]
    results["production_dense_2d"] = [
        production_dense_2d(4, p, pen_2d) for p in (0, 1, 2)
    ]
    results["production_dense_2d_periodic"] = [
        production_dense_2d(4, 2, pen_2d, boundary="periodic")
    ]
    results["empirical_2d"] = [
        empirical_dt_max_2d(p, n, pen_2d)
        for p, n in ((0, 4), (0, 8), (1, 8), (2, 8))
    ]

    # consequences: production operator (advection_diffusion default BC, finest mesh)
    fine = {row["degree"]: row for row in results["production_lanczos_2d"] if row["nx"] == 16}
    conv = {}
    for p, row in sorted(fine.items()):
        c = row["c"]
        h = row["h"]
        dt_true = z_rk4 / row["lam_abs_max"]
        dt_code = ad_dt_diff(p, pen_2d, h)
        conv[p] = {
            "degree": p,
            "c": c,
            "lam_abs_max": row["lam_abs_max"],
            "h": h,
            "dt_true_rk4": dt_true,
            "dt_code_advection_diffusion": dt_code,
            "safety_ratio_code_over_true": dt_code / dt_true,
            "code_constant_relative_error": (c - CODE_AD_DIFF_CONSTANT) / CODE_AD_DIFF_CONSTANT,
            "ns_S_max": ns_S_max(p, pen_2d, c, z_ssp),
            "ns_assumed_S": CODE_NS_VISCOUS_STABILITY,
            "ns_margin": ns_S_max(p, pen_2d, c, z_ssp) / CODE_NS_VISCOUS_STABILITY,
        }
    results["consequences"] = [conv[p] for p in sorted(conv)]

    # empirical vs predicted (same (p, nx))
    pred = {(row["degree"], row["nx"]): row for row in results["production_lanczos_2d"]}
    checks = []
    for row in results["empirical_2d"]:
        key = (row["degree"], row["nx"])
        if key not in pred:
            continue
        predicted = z_rk4 / pred[key]["lam_abs_max"]
        checks.append({
            "degree": row["degree"], "nx": row["nx"],
            "dt_max_empirical": row["dt_max_empirical"],
            "dt_max_predicted": predicted,
            "relative_difference": abs(row["dt_max_empirical"] - predicted) / predicted,
            "growth_below": row["growth_below"],
            "growth_above": row["growth_above"],
        })
    results["empirical_vs_predicted"] = checks
    return results


def plot(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    c1d = {}
    for row in results["spectrum_1d"]:
        if row["ncells"] == 32:
            c1d[row["degree"]] = row["c"]
    c2d = {row["degree"]: row["c"] for row in results["spectrum_2d"] if row["nx"] == 12}
    cprod = {row["degree"]: row["c"] for row in results["production_lanczos_2d"] if row["nx"] == 16}
    ax.plot(sorted(c1d), [c1d[p] for p in sorted(c1d)], "o-", label=r"1-D exact ($N=32$)")
    ax.plot(sorted(c2d), [c2d[p] for p in sorted(c2d)], "s-", label=r"2-D exact ($12\times12$)")
    ax.plot(sorted(cprod), [cprod[p] for p in sorted(cprod)], "^-",
            label=r"2-D production op. ($16\times16$)")
    shared = [p for p in sorted(c2d) if p in c1d]
    ax.plot(shared, [2.0 * c1d[p] for p in shared], "k--",
            label=r"$2\times$1-D (expectation)")
    ax.axhline(CODE_AD_DIFF_CONSTANT, color="crimson", ls=":",
               label="code constant 50")
    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel(r"$c(p)=\lambda_{\max}h^2/(\alpha\nu)$")
    ax.set_yscale("log")
    ax.set_title("measured spectral constant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    degs = [row["degree"] for row in results["consequences"]]
    ax.bar(degs, [row["safety_ratio_code_over_true"] for row in results["consequences"]],
           color="steelblue", label=r"$dt_{\rm code}/dt_{\max}$")
    ax.axhline(1.0, color="crimson", ls="--", label="stability boundary")
    for row in results["consequences"]:
        ax.annotate(f"{row['safety_ratio_code_over_true']:.2f}",
                    (row["degree"], row["safety_ratio_code_over_true"]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xlabel("polynomial degree p")
    ax.set_ylabel("safety factor")
    ax.set_title("advection_diffusion dt rule vs measured limit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    for row in results["empirical_vs_predicted"]:
        ax.loglog(row["dt_max_predicted"], row["dt_max_empirical"], "o",
                  label=f"p={row['degree']}, N={row['nx']}")
    lo = min(r["dt_max_predicted"] for r in results["empirical_vs_predicted"])
    hi = max(r["dt_max_empirical"] for r in results["empirical_vs_predicted"])
    ax.loglog([lo, hi], [lo, hi], "k--", label="1:1")
    ax.set_xlabel(r"predicted $z_{\rm RK4}/\lambda_{\max}$")
    ax.set_ylabel(r"empirical $dt_{\max}$ (bisection)")
    ax.set_title("end-to-end closure on production rhs")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/sipg_explicit_stability")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "case": "sipg-explicit-stability",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "code_constants": {
            "advection_diffusion_dt_diff_constant": CODE_AD_DIFF_CONSTANT,
            "advection_diffusion_safety": CODE_AD_DIFF_SAFETY,
            "ns_viscous_stability": CODE_NS_VISCOUS_STABILITY,
        },
    }
    collect(results)
    json_path = out / "results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    figure = out / "sipg_explicit_stability.png"
    plot(results, figure)
    provenance = {
        "results_sha256": _sha256(json_path),
        "figure_sha256": _sha256(figure),
        "figure": figure.name,
        "git_commit": results["git_commit"],
        "repro": "python -m benchmarks.sipg_explicit_stability --output-dir " + str(out),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if not args.quiet:
        print(json.dumps({
            "rk4_interval": results["rk_intervals"]["rk4"],
            "consequences": results["consequences"],
            "closure": results["empirical_vs_predicted"],
            "artifacts": [str(json_path), str(figure)],
        }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())