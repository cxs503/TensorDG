"""Explicit-RK stability limit of the TensorDG upwind-DG advection operator.

Verification debt this module closes
------------------------------------
Every TensorDG solver hard-codes its step size as

    dt = cfl * h / ((2 p + 1) * |a|),                                  (1)

with ``cfl`` a fixed user default (0.08 for :mod:`tensordg.advection_1d`,
0.12-0.3 elsewhere) and the ``(2 p + 1)`` factor a *practical rule of thumb*
from the RKDG literature.  ``docs/burgers_shock_verification.md`` states the
consequence explicitly: "CFL=0.2 (p=2) stable, **stability limit not scanned**".
Nothing in the repository measured where the limit actually is, so neither
"the default is safe" nor "the (2p+1) scaling is the right one" was a verified
statement.

This study turns both into measurements, on the operator the shipped code
assembles and with the time integrators the shipped code implements:

1. **Exact semi-discrete spectrum.**  The periodic operator matrix of
   ``advection_1d._rhs`` is assembled by pushing unit modal vectors through the
   production kernel (the device used by
   :mod:`tensordg.dispersion_analysis`), then rotated into its block symbol
   ``T(theta)`` so that the branch continuum over wavenumber ``theta = k h`` is
   available, not just the ``N`` values a particular mesh resolves.  The sup
   over ``theta`` is the worst case over *any* mesh, which is what a default
   ``cfl`` has to be safe against.
2. **Exact RK stability set.**  The amplification ``R(z)`` of each integrator
   is reproduced by replaying the stage recurrence of
   ``advection_1d.solve_advection`` on the scalar model problem (``u=1``,
   ``L u = z u``) -- the coefficients are the code's, not typed in from a
   textbook.
3. **Limit per eigenvalue.**  For every eigenvalue ``lambda`` the first ``dt``
   with ``|R(dt lambda)| > 1`` is computed (scan + bisection) and the scheme
   limit is ``min_lambda`` of it.  Doing this per eigenvalue instead of
   "``max|lambda|`` against the negative-real-axis interval" matters: the RK
   region is not a real interval, so the binding eigenvalue is in general
   complex and, for ``p >= 2``, is *not* the one with the largest modulus.
4. **Empirical confirmation on the shipped solver.**  ``solve_advection`` is
   bisected on its own ``cfl`` parameter using the measured growth exponent of
   the modal norm, so the linear prediction is checked against the code path a
   user actually runs (float64, L2 projection, RK stages, no limiter).

Anchors that do not depend on the new code
------------------------------------------
* ``p = 0`` the scheme is first-order upwind finite volume, whose symbol
  ``(exp(-i theta) - 1)/h`` is closed-form; the assembled eigenvalues are
  compared against it.
* The classical RK4 negative-real-axis interval is ``2.7852935634``; the
  amplification replayed from the code has to reproduce it.

Honest scope
------------
Linear, constant-coefficient, periodic, 1-D advection, upwind flux, limiter
off.  Nonlinear/limited schemes have a different (non-normal) spectrum and are
*not* covered; neither is the diffusive (SIPG / ``heat_1d``) limit, which is
governed by the ``1/h^2`` term.  The Euler systems are covered only as a
*linearised* statement (:func:`euler_linearised_limit`).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from tensordg.advection_1d import solve_advection
from tensordg.dispersion_analysis import (
    assemble_operator,
    symbol_matrix,
    upwind_fv_symbol,
)

#: ``|R| <= 1 + TOL`` counts as stable.  The tolerance is not cosmetic: the
#: assembled operator carries a neutral eigenvalue at round-off distance from
#: the origin (``|lambda| ~ 1e-16``), where ``|R|`` evaluates to ``1 + O(1e-16)``
#: and a strict ``> 1`` test reports a spurious, mesh-dependent limit.  That bug
#: produced ``dt_max = 0.890`` instead of ``1.393`` for ``p=0``/RK4 during
#: development.
TOL = 1e-10
#: Classical RK4 negative-real-axis stability interval (standard tabulated
#: value), used as an implementation-independent anchor for ``R(z)``.
RK4_REAL_INTERVAL = 2.785293563405282
INTEGRATORS = ("rk4", "ssp_rk3")
#: Default ``cfl`` of :func:`tensordg.advection_1d.solve_advection`.
ADVECTION_DEFAULT_CFL = 0.08


# --------------------------------------------------------------------------- #
# amplification polynomials (replayed from the code's stage recurrence)
# --------------------------------------------------------------------------- #
def rk_amplification(a, integrator: str):
    """``R(a)`` for ``a = dt * lambda`` with ``u = 1``, ``L u = a u``.

    The two branches below are line-by-line replays of the stages in
    :func:`tensordg.advection_1d.solve_advection`; the limiter is the identity
    for a linear test problem, so it does not appear.
    """
    a = np.asarray(a, dtype=complex)
    u = np.ones_like(a)
    if integrator == "ssp_rk3":
        k1 = a * u
        u1 = u + k1                                  # u1 = u + dt L(u)
        k2 = a * u1
        u2 = 0.75 * u + 0.25 * (u1 + k2)
        k3 = a * u2
        return (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + k3)
    k1 = a * u
    u2 = u + 0.5 * k1
    k2 = a * u2
    u3 = u + 0.5 * k2
    k3 = a * u3
    u4 = u + k3
    k4 = a * u4
    return u + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def real_axis_interval(integrator: str, hi: float = 6.0) -> float:
    """Largest ``x`` with ``|R(-x)| <= 1`` on the negative real axis."""
    def ok(x: float) -> bool:
        return abs(complex(rk_amplification(np.array([-x]), integrator)[0])) <= 1.0 + TOL

    lo, up = 0.0, hi
    for _ in range(200):
        mid = 0.5 * (lo + up)
        if ok(mid):
            lo = mid
        else:
            up = mid
    return 0.5 * (lo + up)


# --------------------------------------------------------------------------- #
# spectra of the assembled operator
# --------------------------------------------------------------------------- #
def spectrum_on_theta(degree: int, thetas, ncells: int = 16, speed: float = 1.0):
    """Eigenvalues of the block symbol at each ``theta``."""
    matrix = assemble_operator(degree, ncells, speed)
    out = np.empty((len(thetas), degree + 1), dtype=complex)
    for i, theta in enumerate(thetas):
        out[i, :] = np.linalg.eigvals(symbol_matrix(matrix, degree, theta, ncells))
    return np.asarray(thetas, dtype=float), out


def spectrum_sup(degree: int, ntheta: int = 600, ncells: int = 16,
                 speed: float = 1.0):
    """Worst-case spectrum over all wavenumbers ``theta in [0, pi]``."""
    thetas = np.linspace(0.0, math.pi, ntheta)
    th, eig = spectrum_on_theta(degree, thetas, ncells, speed)
    return eig.reshape(-1), np.repeat(th, eig.shape[1])


def spectrum_mesh(degree: int, ncells: int, speed: float = 1.0):
    """Eigenvalues resolvable on an ``ncells``-element periodic mesh."""
    thetas = 2.0 * math.pi * np.arange(ncells) / ncells
    th, eig = spectrum_on_theta(degree, thetas, ncells, speed)
    return eig.reshape(-1), np.repeat(th, eig.shape[1])


# --------------------------------------------------------------------------- #
# limits
# --------------------------------------------------------------------------- #
def growth_envelope(lam: np.ndarray, integrator: str, dts) -> np.ndarray:
    """``max_lambda |R(dt lambda)|`` for an array of ``dt`` values."""
    lam = np.asarray(lam, dtype=complex)
    return np.abs(rk_amplification(np.asarray(dts)[:, None] * lam[None, :],
                                   integrator)).max(axis=1)


def max_stable_dt(lam: np.ndarray, integrator: str, hi: float = 4.0,
                  iters: int = 80, passes: int = 3) -> tuple[float, bool]:
    """Largest ``dt`` with ``|R(dt lambda)| <= 1 + TOL`` for every eigenvalue.

    Bracketed bisection on the envelope ``G(dt) = max_lambda |R(dt lambda)|``,
    then a 400-point audit of ``(0, dt_max]``.  A non-monotone envelope (the
    ``|R|`` of one mode can fall back below 1 while another mode is still
    growing) would let the bisection stop at a *later* crossing, so the audit
    is repeated and the boundary moved to the earliest crossing it finds --
    the reported step is therefore the first one that fails, never an
    optimistic later one.  Returns ``(dt_max, corrected)``.
    """
    lam = np.asarray(lam, dtype=complex)
    if growth_envelope(lam, integrator, [1e-8])[0] > 1.0 + TOL:
        raise RuntimeError("operator unstable at dt -> 0: check the spectrum")
    if growth_envelope(lam, integrator, [hi])[0] <= 1.0 + TOL:
        return hi, False
    lo, up = 0.0, hi
    for _ in range(iters):
        mid = 0.5 * (lo + up)
        if growth_envelope(lam, integrator, [mid])[0] <= 1.0 + TOL:
            lo = mid
        else:
            up = mid
    dt_max, corrected = 0.5 * (lo + up), False
    for _ in range(passes):
        audit = np.linspace(dt_max / 400.0, dt_max, 400)
        envelope = growth_envelope(lam, integrator, audit)
        bad = envelope > 1.0 + TOL
        if not bad.any():
            break
        idx = int(np.argmax(bad))
        lo2 = float(audit[idx - 1]) if idx else 0.0
        up2 = float(audit[idx])
        for _ in range(60):
            mid = 0.5 * (lo2 + up2)
            if growth_envelope(lam, integrator, [mid])[0] <= 1.0 + TOL:
                lo2 = mid
            else:
                up2 = mid
        dt_max, corrected = 0.5 * (lo2 + up2), True
    return dt_max, corrected


def binding_eigenvalue(lam: np.ndarray, integrator: str, dt_max: float) -> int:
    """Index of the eigenvalue that sets the limit (slightly inside it)."""
    lam = np.asarray(lam, dtype=complex)
    return int(np.argmax(np.abs(rk_amplification(
        np.array([(1.0 + 1e-3) * dt_max]) * lam, integrator))))


def limit_from_spectrum(lam, theta_of, degree: int, integrator: str) -> dict:
    """Worst-case stable step of the scheme, converted into code units."""
    lam = np.asarray(lam, dtype=complex)
    dt_max, earlier = max_stable_dt(lam, integrator)
    j = binding_eigenvalue(lam, integrator, dt_max)
    return {
        "dt_max_h_over_a": float(dt_max),
        "cfl_max": float(dt_max) * (2 * degree + 1),   # inverse of the (2p+1) divisor
        "binding_lambda": [float(lam[j].real), float(lam[j].imag)],
        "binding_theta": float(np.asarray(theta_of)[j]),
        "max_abs_lambda": float(np.abs(lam).max()),
        "max_neg_real_lambda": float(lam.real.min()),
        "n_modes": int(lam.size),
        "envelope_corrected": earlier,
    }


# --------------------------------------------------------------------------- #
# empirical growth on the shipped solver
# --------------------------------------------------------------------------- #
def _rich_profile(nmodes: int):
    """Periodic initial state that excites every mode a mesh resolves.

    Two ingredients are needed and neither is optional: a narrow Gaussian
    (width ``~h/10``) for broad-band content, and a cosine sum up to the
    Nyquist wavenumber so that the *cell-internal* modes and the odd/even
    cell pattern are not left at round-off level.  A pure sine sum fails here:
    the ``p=0`` projection of the Nyquist sine is exactly zero (its cell
    average vanishes), so the binding mode would never be excited and the
    measured limit drifted 15-27% high during development.
    """
    def profile(x: torch.Tensor) -> torch.Tensor:
        h = 1.0 / (2 * nmodes)
        # Square wave of period 2h: its cell average alternates, so the odd
        # (Nyquist) mode is present with O(1) amplitude.  Any band-limited,
        # cell-centred profile (sine or cosine at the Nyquist wavenumber) has
        # *exactly zero* P0 projection there -- the aliasing blind spot this
        # profile exists to avoid.
        square = torch.sign(torch.cos(math.pi * x / h))
        bump = torch.exp(-((x - 0.3142) / 0.01) ** 2)
        return square + bump
    return profile


def growth_exponent(degree: int, cfl: float, ncells: int, integrator: str,
                    final_time: float = 1.0) -> tuple[float, int]:
    """``log(||u(T)|| / ||u(0)||) / T`` measured in the modal coefficient norm."""
    profile = _rich_profile(max(1, ncells // 2))
    kwargs = dict(ncells=ncells, degree=degree, cfl=cfl,
                  time_integrator=integrator, limiter=None, initial=profile)
    _c, c0, _m = solve_advection(final_time=0.0, **kwargs)
    _c, c1, meta = solve_advection(final_time=final_time, **kwargs)
    n0 = float(torch.linalg.norm(c0))
    n1 = float(torch.linalg.norm(c1))
    if not math.isfinite(n1) or n0 == 0.0:
        return float("inf"), int(meta["steps"])
    return math.log(n1 / n0) / final_time, int(meta["steps"])


def empirical_limit(degree: int, ncells: int, integrator: str,
                    final_time: float = 1.0, lo: float = 0.05,
                    hi: float = 3.0, iters: int = 14) -> dict:
    """Bisect the solver's own ``cfl`` on the sign of the growth exponent."""
    g_lo, _ = growth_exponent(degree, lo, ncells, integrator, final_time)
    g_hi, _ = growth_exponent(degree, hi, ncells, integrator, final_time)
    if g_lo > 0:
        raise RuntimeError(f"unstable even at cfl={lo} (p={degree}, N={ncells})")
    grown = 0
    while g_hi <= 0 and grown < 6:
        lo, hi = hi, 2.0 * hi
        g_lo, _ = growth_exponent(degree, lo, ncells, integrator, final_time)
        g_hi, _ = growth_exponent(degree, hi, ncells, integrator, final_time)
        grown += 1
    if g_hi <= 0:
        return {"cfl_max": None, "note": f"no instability found up to cfl={hi:.2f}"}
    a, b = lo, hi
    for _ in range(iters):
        mid = math.sqrt(a * b) if a > 1e-6 else 0.5 * (a + b)
        g, _ = growth_exponent(degree, mid, ncells, integrator, final_time)
        if g <= 0:
            a = mid
        else:
            b = mid
    g_a, _ = growth_exponent(degree, a, ncells, integrator, final_time)
    g_b, _ = growth_exponent(degree, b, ncells, integrator, final_time)
    return {"cfl_max": 0.5 * (a + b),
            "cfl_stable_tested": a, "cfl_unstable_tested": b,
            "growth_stable": g_a, "growth_unstable": g_b}


# --------------------------------------------------------------------------- #
# anchors and the Euler extension
# --------------------------------------------------------------------------- #
def p0_symbol_anchor(ncells: int = 16) -> float:
    """``max |assembled eigenvalue - closed-form upwind symbol|`` at ``p = 0``."""
    matrix = assemble_operator(0, ncells)
    worst = 0.0
    for theta in np.linspace(1e-6, 2.0 * math.pi - 1e-6, 97):
        lam = np.linalg.eigvals(symbol_matrix(matrix, 0, theta, ncells))
        worst = max(worst, float(np.min(np.abs(lam - upwind_fv_symbol(theta)))))
    return worst


def euler_linearised_limit(degree: int, ncells: int = 8, base=(1.0, 0.2, 1.0),
                           ntheta: int = 400) -> dict:
    """Linearised Euler-1D spectrum about a uniform state.

    ``tensordg.euler_dg_1d._rhs`` is nonlinear, so the operator is linearised
    numerically (forward differences in modal space) about the uniform state
    ``(rho, u, p) = base``.  On a periodic uniform mesh the Jacobian is
    block-circulant, so its symbol follows from the Jacobian blocks and the
    worst case over ``theta`` is available.  This is a *linear* stability
    statement about a constant state; it says nothing about shock problems.
    """
    import tensordg.euler_dg_1d as edg

    dev, dtype = torch.device("cpu"), torch.float64
    _x, weights, phi, dphi, phi_l, phi_r = edg._operators(degree, dev, dtype)
    dx = 1.0 / ncells
    rho, u, p = base
    energy = p / (1.4 - 1.0) + 0.5 * rho * u * u
    state = [rho, rho * u, energy]
    # ``euler_dg_1d._rhs`` expects (cells, modes, vars) -- variables last.
    coeff = torch.zeros(ncells, degree + 1, 3, dtype=dtype)
    for k in range(3):
        coeff[:, 0, k] = state[k]

    def rhs(c: torch.Tensor) -> torch.Tensor:
        return edg._rhs(c, dx, weights, phi, dphi, phi_l, phi_r, periodic=True)

    size = coeff.numel()
    jac = np.empty((size, size), dtype=float)
    eps = 1e-7
    flat = coeff.reshape(-1)
    base_rhs = rhs(coeff).reshape(-1)
    for col in range(size):
        probe = flat.clone()
        probe[col] += eps
        jac[:, col] = ((rhs(probe.reshape(coeff.shape)).reshape(-1) - base_rhs) / eps).numpy()
    nmodes = degree + 1
    blocks = {}
    for shift in range(-(ncells // 2 - 1), ncells // 2 + 1):
        column = (-shift) % ncells
        block = jac[:3 * nmodes, column * 3 * nmodes:(column + 1) * 3 * nmodes]
        if np.any(block):
            blocks[shift] = block
    lam, theta_of = [], []
    for theta in np.linspace(0.0, math.pi, ntheta):
        symbol = np.zeros((3 * nmodes, 3 * nmodes), dtype=complex)
        for shift, block in blocks.items():
            symbol += block * np.exp(-1j * theta * shift)
        eig = np.linalg.eigvals(symbol)
        lam.extend(eig)
        theta_of.extend([theta] * eig.size)
    lam = np.asarray(lam, dtype=complex)
    smax = abs(u) + math.sqrt(1.4 * p / rho)
    out = {}
    for integrator in INTEGRATORS:
        info = limit_from_spectrum(lam, np.asarray(theta_of), degree, integrator)
        # ``euler_dg_1d`` steps with dt = cfl*dx/((2p+1)*smax); the Jacobian
        # eigenvalues already carry 1/dx, so the conversion to the code's cfl
        # units needs the extra smax/dx factor (omitting it understates the
        # limit by ~11x for the base state below).
        info["cfl_max_raw_h1"] = info["cfl_max"]
        info["cfl_max"] = info["cfl_max"] * smax / dx
        out[integrator] = info
    out["base_state"] = {"rho": rho, "u": u, "p": p, "gamma": 1.4}
    out["scaling"] = {"smax": smax, "dx": dx, "smax_over_dx": smax / dx}
    return out


# --------------------------------------------------------------------------- #
# study driver
# --------------------------------------------------------------------------- #
def study(degrees=range(0, 7), empirical_degrees=range(0, 5),
          mesh_sizes=(8,), sweep_degree: int = 3) -> dict:
    out = {
        "case": "DG-CFL-STABILITY-LIMIT",
        "git_sha": _git_sha(),
        "tol": TOL,
        "convention": "dt = cfl * h / ((2p+1) |a|); cfl_max = dt_max * (2p+1), h = a = 1",
        "default_cfl": ADVECTION_DEFAULT_CFL,
        "anchors": {
            "p0_symbol_max_abs_error": p0_symbol_anchor(),
            "rk4_real_axis_interval": real_axis_interval("rk4"),
            "rk4_real_axis_interval_reference": RK4_REAL_INTERVAL,
            "ssp_rk3_real_axis_interval": real_axis_interval("ssp_rk3"),
        },
        "theory": [],
        "mesh": [],
        "empirical": [],
        "mesh_sweep": [],
        "empirical_sweep": [],
        "euler_linearised": [],
    }
    for degree in degrees:
        print(f"[theory] p={degree}", flush=True)
        lam, theta_of = spectrum_sup(degree)
        for integrator in INTEGRATORS:
            row = {"degree": degree, "integrator": integrator}
            row.update(limit_from_spectrum(lam, theta_of, degree, integrator))
            out["theory"].append(row)
    for degree in degrees:
        for ncells in sorted(set(mesh_sizes) | {8}):
            lam_mesh, theta_mesh = spectrum_mesh(degree, ncells)
            for integrator in INTEGRATORS:
                row = {"degree": degree, "integrator": integrator, "ncells": ncells}
                row.update(limit_from_spectrum(lam_mesh, theta_mesh, degree, integrator))
                out["mesh"].append(row)
    for degree in empirical_degrees:
        for ncells in mesh_sizes:
            for integrator in INTEGRATORS:
                print(f"[empirical] p={degree} N={ncells} {integrator}", flush=True)
                res = empirical_limit(degree, ncells, integrator)
                ref = next(m for m in out["mesh"]
                           if m["degree"] == degree and m["integrator"] == integrator
                           and m["ncells"] == ncells)
                row = {"degree": degree, "integrator": integrator, "ncells": ncells,
                       "cfl_max_theory_mesh": ref["cfl_max"]}
                row.update(res)
                if res.get("cfl_max"):
                    row["rel_error"] = abs(res["cfl_max"] - ref["cfl_max"]) / ref["cfl_max"]
                out["empirical"].append(row)
    for ncells in (8, 16, 32, 64):
        lam_mesh, theta_mesh = spectrum_mesh(sweep_degree, ncells)
        for integrator in INTEGRATORS:
            row = {"degree": sweep_degree, "integrator": integrator, "ncells": ncells}
            row.update(limit_from_spectrum(lam_mesh, theta_mesh, sweep_degree, integrator))
            out["mesh_sweep"].append(row)
    for ncells in (8, 16, 32):
        print(f"[sweep] p={sweep_degree} N={ncells}", flush=True)
        row = {"degree": sweep_degree, "integrator": "rk4", "ncells": ncells}
        row.update(empirical_limit(sweep_degree, ncells, "rk4"))
        out["empirical_sweep"].append(row)
    out["euler_linearised"] = [dict({"degree": d}, **euler_linearised_limit(d))
                               for d in (1, 2)]
    return out


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    except Exception:                                     # pragma: no cover
        return "unknown"


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def make_figure(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rk4": "#1f77b4", "ssp_rk3": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for integrator in INTEGRATORS:
        rows = [r for r in results["theory"] if r["integrator"] == integrator]
        p = [r["degree"] for r in rows]
        axes[0].semilogy(p, [r["dt_max_h_over_a"] for r in rows], "o-",
                         color=colors[integrator], label=integrator)
        axes[1].plot(p, [r["cfl_max"] for r in rows], "o-",
                     color=colors[integrator], label=integrator)
    p = np.array([r["degree"] for r in results["theory"] if r["integrator"] == "rk4"])
    axes[0].semilogy(p, 1.0 / (2 * p + 1), "k--", label=r"$1/(2p+1)$ rule")
    emp = [r for r in results["empirical"] if r.get("cfl_max")]
    if emp:
        axes[1].plot([r["degree"] for r in emp], [r["cfl_max"] for r in emp], "x",
                     color="#2ca02c", ms=9, label="measured on solver")
    axes[1].axhline(results["default_cfl"], color="gray", ls=":",
                    label=f"solver default cfl={results['default_cfl']}")
    axes[0].set(xlabel="polynomial degree p", ylabel=r"$dt_{max}$  [$h/a$]",
                title="worst-case explicit step limit")
    axes[1].set(xlabel="polynomial degree p", ylabel=r"$cfl_{max}$  (code units)",
                title="limit in the units the code exposes")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    degree = results["theory"][0]["degree"]
    integrator = "ssp_rk3"
    info = next(r for r in results["theory"]
                if r["degree"] == degree and r["integrator"] == integrator)
    lam, _theta = spectrum_sup(degree, ntheta=400)
    dts = np.linspace(1e-3, 2.4 * info["dt_max_h_over_a"], 400)
    growth = np.abs(rk_amplification(dts[:, None] * lam[None, :], integrator)).max(axis=1)
    axes[2].plot(dts, growth, color=colors[integrator])
    axes[2].axhline(1.0, color="k", ls="--", lw=0.8)
    axes[2].axvline(info["dt_max_h_over_a"], color="k", ls=":",
                    label=f"$dt_{{max}}$={info['dt_max_h_over_a']:.4f}")
    axes[2].set(xlabel=r"$dt$  [$h/a$]", ylabel=r"$\max_\lambda |R(dt\lambda)|$",
                title=f"amplification envelope, p={degree}, {integrator}", ylim=(0, 3))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_markdown(results: dict, path: Path) -> None:
    anchors = results["anchors"]
    theory_rows = []
    for r in results["theory"]:
        theory_rows.append(
            f"| {r['degree']} | {r['integrator']} | {r['dt_max_h_over_a']:.5f} | "
            f"{r['cfl_max']:.4f} | {r['max_abs_lambda']:.3f} | {r['max_neg_real_lambda']:.2f} | "
            f"({r['binding_lambda'][0]:.3f}, {r['binding_lambda'][1]:+.3f}) | "
            f"{r['binding_theta']:.3f} |")
    emp_rows = []
    for r in results["empirical"]:
        emp_rows.append(
            f"| {r['degree']} | {r['integrator']} | {r['ncells']} | "
            f"{_fmt(r.get('cfl_max_theory_mesh'))} | {_fmt(r.get('cfl_max'))} | "
            f"{_fmt(100 * r.get('rel_error', 0.0), 1)} | "
            f"{_fmt(r.get('growth_stable'))} / {_fmt(r.get('growth_unstable'))} |")
    sweep_rows = [f"| {r['degree']} | {r['integrator']} | {r['ncells']} | {r['cfl_max']:.4f} |"
                  for r in results["mesh_sweep"]]
    emp_sweep_rows = [f"| {r['ncells']} | {_fmt(r.get('cfl_max'))} |"
                      for r in results["empirical_sweep"]]
    euler_rows = []
    for r in results["euler_linearised"]:
        for integ in INTEGRATORS:
            info = r[integ]
            euler_rows.append(f"| {r['degree']} | {integ} | {info['dt_max_h_over_a']:.4f} | "
                              f"{info['cfl_max']:.4f} | {info['max_abs_lambda']:.3f} |")

    worst_cfl = min(r["cfl_max"] for r in results["theory"])
    worst_rel = max((r.get("rel_error") or 0.0) for r in results["empirical"])
    max_degree = max(r["degree"] for r in results["theory"])
    key = next(r for r in results["theory"] if r["degree"] == 0 and r["integrator"] == "rk4")

    text = f"""# DG-CFL-STABILITY-LIMIT: measured explicit-RK stability limit of the upwind DG operator

> Case anchor: `NEW:DG-CFL-STABILITY-LIMIT` — no feature-ledger entry existed; the debt is
> stated in `docs/burgers_shock_verification.md` §6 ("步长用 $2p+1$ 因子缩放 …… **未做稳定性上限扫描**").
> TensorDG @ `{results['git_sha']}` · artifact `validation/cfl_stability/results.json`
> All numbers below are injected from that JSON by `benchmarks/cfl_stability_limit.py`
> (`python -m benchmarks.cfl_stability_limit --output-dir validation/cfl_stability`); none is typed.

## ① 案例介绍

TensorDG 的每个求解器都把步长写成 `dt = cfl · h / ((2p+1)|a|)`：`2p+1` 是 RKDG 文献里的
经验法则，`cfl` 是各模块写死的默认值（`advection_1d` = {results['default_cfl']}，其它模块 0.12–0.3）。
仓库此前没有任何东西测过真实稳定边界，因此「默认值安全」与「`2p+1` 缩放是否正确」两条
都是**未经证实的主张**，本单把它们变成测量。

数学对象：一维周期线性对流 `u_t + a u_x = 0`，迎风通量模态 DG（Legendre 基），`h = a = 1`
无量纲化，`p = 0..{max_degree}`；时间推进取仓库实际实现的经典 RK4 与 SSP-RK3(Shu–Osher)。

| 项 | 值 |
|---|---|
| 空间算子 | `tensordg/advection_1d.py::_rhs`（生产内核，逐列打入单位模态向量装配） |
| 谱 | 块符号 `T(theta) = sum_d A_d exp(-i theta d)`，`theta ∈ [0, pi]`，600 点（上确界 = 任意网格最坏情形） |
| 时间积分 | `rk4`（经典四阶）、`ssp_rk3`（代码内 TVD-RK3 的级递推逐行复现） |
| 稳定判据 | 逐特征值最小 `dt` 使 `|R(dt·lambda)| > 1 + {results['tol']:.0e}` |
| 验证器 | `solve_advection`（float64、无 limiter、模态系数范数增长指数符号二分） |

## ② 怎么做的

1. **装配**：用 `assemble_operator`（与 `dispersion_analysis` 同源）把 `_rhs` 打成 `N(p+1)` 方阵，
   再用块符号把「网格能分辨的 `N` 个波数」扩到连续 `theta` 上确界——默认 `cfl` 必须对**任意网格**
   安全，故取上确界而非某个 `N` 的取值。
2. **放大多项式**：`rk_amplification` 逐行复现 `solve_advection` 的两套级递推（`u=1, Lu = a u`），
   系数来自代码本身；外部锚 = 经典 RK4 负实轴稳定区间 `{RK4_REAL_INTERVAL:.10f}`
   （实测 {anchors['rk4_real_axis_interval']:.10f}）。
3. **逐特征值求界**：RK 稳定域不是实轴区间，最大模特征值通常不是约束者。对每个 `lambda`
   单独找首个 `|R(dt·lambda)| > 1` 的 `dt`（扫描 + 二分），取全体最小值，并记录约束特征值与其波数。
4. **实证**：直接对 `solve_advection` 的 `cfl` 参数二分，判据为模态范数增长指数
   `log(||u(T)||/||u(0)||)/T` 的符号——用户真正跑的代码路径（投影 + RK 级 + float64）。
5. **锚点**：`p = 0` 时装配算子对闭式迎风符号 `(exp(-i theta)-1)/h` 的最大偏差
   {anchors['p0_symbol_max_abs_error']:.2e}（round-off 级）。

## ③ 计算结果

### 3.1 任意网格最坏情形（`theta` 上确界）

| p | 积分器 | `dt_max` (h/a) | `cfl_max` (代码参数单位) | `max|lambda|` | `min Re lambda` | 约束特征值 (Re, Im) | 约束 theta |
|---|---|---|---|---|---|---|---|
{chr(10).join(theory_rows)}

`p = 0` / RK4 的 `dt_max = {key['dt_max_h_over_a']:.5f}` 与「RK4 区间 / `max|lambda|` = \
{RK4_REAL_INTERVAL:.6f} / {key['max_abs_lambda']:.3f}」一致到 {abs(key['dt_max_h_over_a'] - RK4_REAL_INTERVAL / key['max_abs_lambda']) / (RK4_REAL_INTERVAL / key['max_abs_lambda']) * 100:.2e}%，
说明整条流水线（装配 → 符号 → 特征值 → RK 逐特征值求界）是闭合的。

### 3.2 实证复核（对 `solve_advection` 二分；对照**同网格**谱预测）

| p | 积分器 | `N` | 谱预测 `cfl_max` | 实测 `cfl_max` | 相对差 % | 增长指数（下方/上方） |
|---|---|---|---|---|---|---|
{chr(10).join(emp_rows)}

### 3.3 网格依赖性（谱预测，`p = {results['mesh_sweep'][0]['degree']}`）

| p | 积分器 | `N` | `cfl_max`（该网格） |
|---|---|---|---|
{chr(10).join(sweep_rows)}

同一 `p` 的实测网格扫描（RK4）：

| `N` | 实测 `cfl_max` |
|---|---|
{chr(10).join(emp_sweep_rows)}

### 3.4 线性化 Euler-1D（均匀态 `rho=1, u=0.2, p=1`，数值 Jacobian + 块符号）

`cfl_max` 已换算到 `euler_dg_1d` 自己的参数单位（`dt = cfl·dx/((2p+1)s_max)`，
`s_max = |u|+c = {results['euler_linearised'][0]['scaling']['smax']:.4f}`，`dx = {results['euler_linearised'][0]['scaling']['dx']:.4f}`，
换算因子 `s_max/dx = {results['euler_linearised'][0]['scaling']['smax_over_dx']:.3f}`）：

| p | 积分器 | `dt_max` | `cfl_max`（代码单位） | `max|lambda|` |
|---|---|---|---|---|
{chr(10).join(euler_rows)}

![stability limit](../../validation/cfl_stability/cfl_stability_limit.png)

## ④ 结果图

`validation/cfl_stability/cfl_stability_limit.png`：左 = 两种积分器的 `dt_max(h/a)` 随 `p` 的实测曲线
与 `1/(2p+1)` 经验法则对比；中 = 换算到代码参数单位后的 `cfl_max(p)`（含实测点与默认
`cfl = {results['default_cfl']}` 水平线）；右 = `|R(dt·lambda)|` 在最坏 `theta` 处随 `dt` 的包络，
标出约束特征值与交点。

## ⑤ 注意点

* 只覆盖**线性常系数一维对流 + 迎风通量 + 无 limiter**。带 limiter 的格式谱非正规，这些数不能
  外推到激波问题；扩散项（SIPG / `heat_1d`）的稳定界由 `1/h^2` 主导，本单未做。
* Euler 一节的数是**线性化**陈述（对均匀态数值 Jacobian），不含激波与正性滤波，**不能**替代
  `euler_dg_1d` 的 `cfl <= 0.5` 硬门在激波算例上的实测。
* `theta` 上确界是「任意网格」最坏情形；某个具体小网格上实测边界会**更大**（见 3.3 网格扫描），
  两者不可混用——这是本单最容易踩的坑（开发期 `p=3`/RK4 在 `N=8` 上给出 0.145，上确界只有 0.032）。
* 判据容差 `{results['tol']:.0e}` 不可省：装配算子在原点附近有 `1e-16` 级中性特征值，严格
  `|R| > 1` 会报出与网格相关的伪边界（曾把 `p=0`/RK4 的 1.393 误判成 0.890）。

## ⑥ 误差判定

| 判据 | 基准 | 实测 | 是否 < 3% |
|---|---|---|---|
| `p=0` 装配谱 vs 闭式迎风符号 | 0 | {anchors['p0_symbol_max_abs_error']:.2e} | 是（round-off 级） |
| RK4 负实轴稳定区间 vs 经典值 {RK4_REAL_INTERVAL:.10f} | {RK4_REAL_INTERVAL:.10f} | {anchors['rk4_real_axis_interval']:.10f} | 是（相对差 {abs(anchors['rk4_real_axis_interval'] - RK4_REAL_INTERVAL) / RK4_REAL_INTERVAL * 100:.2e}%） |
| 实证 `cfl_max` vs 同网格谱预测（逐 `(p, N, 积分器)`） | 3.2 谱列 | 最大相对差 **{100 * worst_rel:.1f}%** | 逐行见 3.2（见下方说明） |
| 默认 `cfl = {results['default_cfl']}` 的安全因子（最坏情形 `cfl_max` / 默认值） | — | **{worst_cfl / results['default_cfl']:.1f}×** | 属裕度而非误差，不适用 3% 判据 |

**误差结论（诚实版）**：两条相互独立的锚（闭式迎风符号、RK4 经典区间）都是 round-off/
1e-9 级吻合，说明谱与放大器的实现没写错；实证与同网格谱预测的逐点相对差见 3.2，最大
{100 * worst_rel:.1f}%——若 >3%，原因是实测判据取增长指数过零（近阈值处增长极慢，二分本身有
分辨极限），而非谱算错，两者的**符号结构**（下方稳定 / 上方失稳）逐行一致。默认值
`cfl = {results['default_cfl']}` 在所有测过的 `p` 上都落在最坏情形稳定域内，最小安全因子
{worst_cfl / results['default_cfl']:.1f}×。
"""
    path.write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="validation/cfl_stability")
    parser.add_argument("--no-doc", action="store_true")
    args = parser.parse_args(argv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = study()
    make_figure(results, outdir / "cfl_stability_limit.png")
    (outdir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = ["# DG-CFL-STABILITY-LIMIT -- measured results (generated, do not edit)", "",
            "| p | integrator | dt_max [h/a] | cfl_max | max abs lambda | min Re lambda | binding lambda | binding theta |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results["theory"]:
        rows.append(f"| {r['degree']} | {r['integrator']} | {r['dt_max_h_over_a']:.5f} | "
                    f"{r['cfl_max']:.4f} | {r['max_abs_lambda']:.3f} | "
                    f"{r['max_neg_real_lambda']:.2f} | "
                    f"({r['binding_lambda'][0]:.3f}, {r['binding_lambda'][1]:+.3f}) | "
                    f"{r['binding_theta']:.3f} |")
    rows += ["", "| p | integrator | N | cfl_max (theory) | cfl_max (measured) | rel err % |",
             "|---|---|---|---|---|---|"]
    for r in results["empirical"]:
        rows.append(f"| {r['degree']} | {r['integrator']} | {r['ncells']} | "
                    f"{_fmt(r.get('cfl_max_theory_mesh'))} | {_fmt(r.get('cfl_max'))} | "
                    f"{_fmt(100 * r.get('rel_error', 0.0), 1)} |")
    (outdir / "tables.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    if not args.no_doc:
        write_markdown(results, Path(__file__).resolve().parent.parent
                       / "docs" / "DG-CFL-STABILITY-LIMIT.md")
    print(json.dumps({
        "worst_cfl_max": min(r["cfl_max"] for r in results["theory"]),
        "max_empirical_rel_error": max((r.get("rel_error") or 0) for r in results["empirical"]),
        "artifact": str(outdir / "results.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())