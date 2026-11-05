"""Semi-discrete dispersion / dissipation analysis of the production DG advection operators.

The analysis is *black-box* with respect to the solver: the space-time operator is
recovered by applying the production right-hand sides (``tensordg.advection_1d._rhs``
and ``tensordg.advection_2d._rhs``) to the identity, so whatever assembly the solver
uses (modal Legendre basis, upwind numerical flux, element Jacobian scaling) is
exactly what gets analysed.  Nothing in ``tensordg/`` is modified.

Anchors
-------
1. ``degree == 0``: piecewise-constant DG with an upwind flux is *identical* to
   first-order upwind finite volume, whose symbol is known in closed form,
   ``lambda(theta) = (a/h) (exp(-i theta) - 1)``.  This pins the eigenvalue sign
   convention and validates the Bloch decomposition to round-off.
2. ``theta -> 0``: the physical branch must reproduce the exact continuum
   dispersion ``lambda -> -i a k``, giving a measurable phase-error order
   (the spatial order of the DG wave operator, distinct from the L2 order).
3. Energy stability: ``Re lambda <= 0`` for every mode (upwind DG is dissipative),
   so the only instability risk is the implicit time integrator.
4. Time-step rule ``dt = cfl*h/((2p+1)|a|)`` used by the solver: the spectral radius
   ``max|lambda| h/a`` fixes the exact RK4 stability boundary, so the heuristic
   ``2p+1`` divisor can be checked for safety (and for conservativeness).

The physical (acoustic-like) branch is followed by eigenvector continuation from
``theta -> 0`` instead of nearest-eigenvalue matching, because the ``p+1`` branches
of upwind DG cross at coarse resolution.

Outputs: ``dispersion_report.json``, ``dispersion_dissipation.png`` and a markdown
table fragment consumed by ``tools/render_dispersion_report.py``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensordg.advection_1d import _operators, _rhs

# Explicit RK4 stability interval on the negative real axis.
RK4_REAL_LIMIT = 2.785293563405282


def semidiscrete_matrix(ncells: int, degree: int, speed: float = 1.0) -> tuple[np.ndarray, float]:
    """Return the exact semi-discrete operator ``A`` (``du/dt = A u``) and ``h``."""
    dtype = torch.float64
    device = torch.device("cpu")
    h = 1.0 / ncells
    _x, _w, _phi, volume, phi_l, phi_r = _operators(degree, device, dtype)
    ndof = ncells * (degree + 1)
    A = np.zeros((ndof, ndof))
    for col in range(ndof):
        basis = torch.zeros(ncells, degree + 1, dtype=dtype)
        basis[col // (degree + 1), col % (degree + 1)] = 1.0
        A[:, col] = _rhs(basis, h, speed, volume, phi_l, phi_r).reshape(-1).numpy()
    return A, h


def semidiscrete_matrix_2d(ncells: int, degree: int,
                           velocity: tuple[float, float] = (1.0, 0.5)) -> tuple[np.ndarray, float, float]:
    """Same construction for the 2-D tensor-product operator of ``advection_2d``."""
    from tensordg.advection_2d import _operators as ops2d
    from tensordg.advection_2d import _rhs as rhs2d

    dtype = torch.float64
    device = torch.device("cpu")
    hx = hy = 1.0 / ncells
    nodes, weights, phi, volume, phi_left, phi_right = ops2d(degree, device, dtype)
    p1 = degree + 1
    ndof = ncells * ncells * p1 * p1
    A = np.zeros((ndof, ndof))
    for col in range(ndof):
        basis = torch.zeros(ncells, ncells, p1, p1, dtype=dtype)
        idx = np.unravel_index(col, (ncells, ncells, p1, p1))
        basis[idx] = 1.0
        A[:, col] = rhs2d(basis, hx, hy, velocity, volume, phi_left, phi_right).reshape(-1).numpy()
    return A, hx, hy


def eigenvalue_set_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Max-over-theta minimal matching distance between two eigenvalue sets.

    Plain sorting is not a valid multiset comparison here: upwind-DG spectra contain
    conjugate pairs with *equal* real parts, so round-off reorders them.
    """
    import itertools
    worst = 0.0
    for m in range(a.shape[0]):
        best = min(max(abs(a[m, i] - b[m, j]) for i, j in enumerate(perm))
                   for perm in itertools.permutations(range(b.shape[1])))
        worst = max(worst, best)
    return float(worst)


def symbol_blocks(A: np.ndarray, ncells: int, degree: int) -> dict[int, np.ndarray]:
    """Extract the (p+1)x(p+1) block matrices of the block-circulant operator.

    ``A`` is circulant in the cell index, so ``A[(j,a),(j+r,b)]`` depends on ``r``
    only.  The ``r`` far enough to be structurally zero are dropped, which turns the
    Bloch symbol into an explicit ``(p+1)``-dimensional matrix for *any* angle.
    """
    p1 = degree + 1
    blocks: dict[int, np.ndarray] = {}
    for r in range(ncells):
        rr = r if r <= ncells // 2 else r - ncells
        blk = A[0:p1, (rr % ncells) * p1:(rr % ncells) * p1 + p1]
        if np.abs(blk).max() > 0.0:
            blocks[rr] = blk
    return blocks


def symbol_eigen(blocks: dict[int, np.ndarray], theta: np.ndarray):
    """Eigenpairs of ``S(theta) = sum_r A_r exp(i theta r)``; any ``theta`` is allowed."""
    p1 = next(iter(blocks.values())).shape[0]
    lam = np.empty((theta.size, p1), dtype=complex)
    vec = np.empty((theta.size, p1, p1), dtype=complex)
    for m in range(theta.size):
        S = np.zeros((p1, p1), dtype=complex)
        for r, blk in blocks.items():
            S += blk * np.exp(1j * theta[m] * r)
        w, Q = np.linalg.eig(S)
        lam[m] = w
        vec[m] = Q
    return lam, vec


def bloch_eigen(degree: int, A: np.ndarray, ncells: int, theta: np.ndarray):
    """Eigenpairs of the block-circulant symbol matrix at each Bloch angle.

    The operator commutes with cell shifts, so the ``(p+1)``-dimensional subspace
    ``v[j, alpha] = exp(i theta j) delta_{alpha beta}`` is invariant; projecting there
    gives ``p+1`` semi-discrete frequencies per wavenumber.
    """
    p1 = degree + 1
    ndof = A.shape[0]
    lam = np.empty((theta.size, p1), dtype=complex)
    vec = np.empty((theta.size, p1, p1), dtype=complex)
    eye = np.eye(p1, dtype=complex)
    phase = np.exp(1j * np.outer(np.arange(ncells), theta))
    for m in range(theta.size):
        V = np.zeros((ndof, p1), dtype=complex)
        for j in range(ncells):
            V[j * p1:(j + 1) * p1, :] = phase[j, m] * eye
        w, Q = np.linalg.eig(V.conj().T @ (A @ V) / ncells)
        lam[m] = w
        vec[m] = Q
    return lam, vec


def track_physical_branch(lam: np.ndarray, vec: np.ndarray, theta: np.ndarray,
                          speed: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Follow the physical branch by eigenvector continuation from ``theta -> 0``.

    Returns ``(index_per_wavenumber, eigenvalues)``.  Ties/continuation failures are
    reported through a boolean array so the caller can flag ambiguous modes.
    """
    k = theta / h
    exact = -1j * speed * k
    n = theta.size
    idx = np.empty(n, dtype=int)
    tracked = np.empty(n, dtype=complex)
    ambiguous = np.zeros(n, dtype=bool)
    prev = None
    for m in range(n):
        if prev is None:
            j = int(np.argmin(np.abs(lam[m] - exact[m])))
        else:
            overlap = np.abs(vec[m].conj().T @ prev)
            j = int(np.argmax(overlap))
            sorted_ov = np.sort(overlap)[::-1]
            ambiguous[m] = bool(len(sorted_ov) > 1 and
                                (sorted_ov[0] - sorted_ov[1]) < 1e-3)
        idx[m] = j
        tracked[m] = lam[m, j]
        prev = vec[m][:, j]
    return idx, tracked, ambiguous


def dispersion_curve(degree: int, ncells: int, speed: float = 1.0) -> dict:
    """Phase error, dissipation and branch-tracking diagnostics on a periodic grid."""
    A, h = semidiscrete_matrix(ncells, degree, speed)
    blocks = symbol_blocks(A, ncells, degree)
    theta = theta_grid(ncells)
    lam, vec = symbol_eigen(blocks, theta)
    idx, tracked, ambiguous = track_physical_branch(lam, vec, theta, speed, h)
    k = theta / h
    c_num = -tracked.imag / k
    return {
        "theta": theta,
        "phase_err": np.abs(c_num - speed) / speed,
        # amplitude decay per wavelength, dimensionless
        "diss_per_wavelength": -tracked.real * h / speed * (2.0 * math.pi / theta),
        "max_real_part_all_modes": float(lam.real.max()),
        "max_abs_lambda": float(np.abs(lam).max()),
        "spectral_radius_h_over_a": float(np.abs(lam).max() * h / speed),
        "ambiguous_modes": int(ambiguous.sum()),
    }


def theta_grid(ncells: int, modes: int | None = None) -> np.ndarray:
    """Positive Bloch angles of a periodic grid of ``ncells`` cells."""
    mmax = ncells // 2 if modes is None else modes
    return 2.0 * math.pi * np.arange(1, mmax + 1) / ncells


def refine_order(degree: int, ncells_list, speed: float = 1.0) -> dict:
    """Phase-error order at the fixed physical wavenumber ``k = 2*pi``.

    ``theta = 2*pi/N`` shrinks with the grid, so very fine grids hit the eigen-solver
    round-off floor; the reported order therefore uses the finest adjacent pair whose
    errors both stay above ``FLOOR`` (flagged so the choice is auditable).
    """
    FLOOR = 1e-10
    errs = []
    for n in ncells_list:
        # The symbol scales like a/h, so it must be built on the grid it is read on.
        blocks = symbol_blocks(semidiscrete_matrix(n, degree, speed)[0], n, degree)
        theta = np.array([2.0 * math.pi / n])
        lam, vec = symbol_eigen(blocks, theta)
        _i, tracked, _a = track_physical_branch(lam, vec, theta, speed, 1.0 / n)
        errs.append(abs(-tracked.imag[0] / (theta[0] * n) - speed) / speed)
    pairs = []
    for i in range(len(errs) - 1):
        pairs.append(math.log(errs[i] / errs[i + 1]) / math.log(ncells_list[i + 1] / ncells_list[i]))
    usable = [i for i in range(len(pairs)) if errs[i] > FLOOR and errs[i + 1] > FLOOR]
    if usable:
        i = usable[-1]
        order, pair = pairs[i], f"N={ncells_list[i]}->{ncells_list[i + 1]}"
    else:  # everything at round-off; report the coarsest pair
        order, pair = pairs[0], f"N={ncells_list[0]}->{ncells_list[1]}"
    return {"ncells": list(ncells_list), "phase_err": errs, "orders_all": pairs,
            "observed_order": order, "order_pair": pair, "roundoff_floor": FLOOR}


def ppw_for_tolerance(degree: int, tol: float, speed: float = 1.0) -> dict:
    """Points per wavelength at which phase error *or* per-wavelength amplitude decay
    first exceeds ``tol`` (both criteria must hold).  Grid-independent: the branch is
    a function of ``theta`` only, evaluated on a logarithmic sweep down to 1e-4.
    """
    A, h = semidiscrete_matrix(16, degree, speed)
    blocks = symbol_blocks(A, 16, degree)
    theta = np.geomspace(1e-4, math.pi, 600)
    lam, vec = symbol_eigen(blocks, theta)
    _i, tracked, ambiguous = track_physical_branch(lam, vec, theta, speed, h)
    k = theta / h
    phase = np.abs(-tracked.imag / k - speed) / speed
    diss = -tracked.real * h / speed * (2.0 * math.pi / theta)
    ok = (phase <= tol) & (diss <= tol)
    if not ok.any():
        return {"ppw": float("inf"), "theta_max": 0.0}
    th_max = float(theta[ok].max())
    near_nyquist = bool(th_max > 0.9 * math.pi)
    return {"ppw": 2.0 * math.pi / th_max, "theta_max": th_max,
            "phase_err_at_theta_max": float(np.interp(th_max, theta, phase)),
            "diss_at_theta_max": float(np.interp(th_max, theta, diss)),
            "at_nyquist_limit": near_nyquist,
            "ambiguous_modes_in_sweep": int(ambiguous.sum())}


def one_step_faithfulness(ncells: int, degree: int, speed: float = 1.0, cfl: float = 0.08) -> float:
    """Relative mismatch between an RK4 step of ``A`` and one production solver step."""
    from tensordg.advection_1d import solve_advection

    h = 1.0 / ncells
    dt = cfl * h / (abs(speed) * (2 * degree + 1))
    centers, coeff_prod, meta = solve_advection(ncells=ncells, degree=degree, final_time=dt,
                                                speed=speed, cfl=cfl)
    assert meta["steps"] == 1, meta
    x_np, w_np = np.polynomial.legendre.leggauss(max(2 * degree + 3, 4))
    _x, _w, phi, _volume, _phi_l, _phi_r = _operators(degree, torch.device("cpu"), torch.float64)
    x_phys = centers[:, None] + 0.5 * h * torch.as_tensor(x_np)[None, :]
    u0 = torch.sin(2.0 * math.pi * x_phys)
    coeff0 = torch.empty((ncells, degree + 1), dtype=torch.float64)
    wq = torch.as_tensor(w_np)
    for k in range(degree + 1):
        coeff0[:, k] = (2 * k + 1) * 0.5 * torch.sum(wq[None, :] * u0 * phi[:, k][None, :], dim=1)
    A, _ = semidiscrete_matrix(ncells, degree, speed)

    def f(u):
        return A @ u.reshape(-1)

    u = coeff0.numpy().reshape(-1)
    k1 = f(u); k2 = f(u + 0.5 * dt * k1); k3 = f(u + 0.5 * dt * k2); k4 = f(u + dt * k3)
    u_ref = u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(np.abs(u_ref - coeff_prod.numpy().reshape(-1)).max() / np.abs(u).max())


def negative_control(ncells: int = 24, degree: int = 1, final_time: float = 0.2,
                     cfls=(0.08, 1.0, 2.0, 4.0)) -> list[dict]:
    """Integrate with the production solver, varying only CFL, to confirm the spectral
    bound is real: steps above ``cfl_max`` must actually destabilise."""
    from tensordg.advection_1d import l2_error, solve_advection

    rows = []
    for cfl in cfls:
        centers, coeff, meta = solve_advection(ncells=ncells, degree=degree,
                                               final_time=final_time, cfl=cfl)
        rows.append({"cfl": cfl, "dt": meta["dt"], "steps": meta["steps"],
                     "l2": l2_error(centers, coeff, final_time=final_time)})
    return rows


def analyse(degrees=(0, 1, 2, 3), ncells=(8, 16, 32), refine=(8, 16, 32, 64),
            speed: float = 1.0, ncells_2d: int = 6) -> dict:
    report: dict = {"speed": speed, "anchors": {}, "spectra": {}, "orders": {},
                    "ppw": {}, "cfl": {}}

    # Anchor 1: p = 0 against the closed-form first-order upwind symbol.
    A, h = semidiscrete_matrix(16, 0, speed)
    theta = theta_grid(16)
    lam, _vec = bloch_eigen(0, A, 16, theta)
    exact = (speed / h) * (np.exp(-1j * theta) - 1.0)
    report["anchors"]["p0_symbol_max_abs_err"] = float(np.abs(lam[:, 0] - exact).max())

    for p in degrees:
        rows = []
        curve = dispersion_curve(p, ncells[-1], speed)
        for n in ncells:
            c = dispersion_curve(p, n, speed)
            rows.append({"ncells": n, "ndofs": n * (p + 1),
                         "spectral_radius_h_over_a": c["spectral_radius_h_over_a"],
                         "max_real_part": c["max_real_part_all_modes"]})
        report["spectra"][f"p{p}"] = rows
        report["anchors"][f"p{p}_max_real_part"] = curve["max_real_part_all_modes"]
        report["anchors"][f"p{p}_ambiguous_modes_n{ncells[-1]}"] = curve["ambiguous_modes"]

        radius = max(r["spectral_radius_h_over_a"] for r in rows)
        report["cfl"][f"p{p}"] = {
            "spectral_radius_h_over_a": radius,
            "heuristic_factor_2p1": 2 * p + 1,
            "radius_over_heuristic": radius / (2 * p + 1),
            "cfl_max_rk4_with_heuristic": RK4_REAL_LIMIT * (2 * p + 1) / radius,
            "cfl_default_used_0.08": 0.08,
            "safety_margin": (RK4_REAL_LIMIT * (2 * p + 1) / radius) / 0.08,
        }

        report["orders"][f"p{p}"] = refine_order(p, refine, speed)

        report["ppw"][f"p{p}"] = {"tol_1pct": ppw_for_tolerance(p, 1e-2, speed),
                                  "tol_0p1pct": ppw_for_tolerance(p, 1e-3, speed)}

    # Cross-check: explicit block symbol vs. projection of the full operator.
    for p_ in degrees:
        A_, h_ = semidiscrete_matrix(16, p_, speed)
        th_ = theta_grid(16)
        lam_blocks, _ = symbol_eigen(symbol_blocks(A_, 16, p_), th_)
        lam_proj, _ = bloch_eigen(p_, A_, 16, th_)
        report["anchors"][f"p{p_}_symbol_vs_projection_max_abs_err"] = (
            eigenvalue_set_distance(lam_blocks, lam_proj))

    report["negative_control"] = negative_control()

    report["anchors"]["rk4_step_max_rel_mismatch"] = {
        f"p{p}_n32": one_step_faithfulness(32, p, speed) for p in degrees
    }

    # 2-D: the dt rule uses wave_rate = |ax|/hx + |ay|/hy; verify the same
    # 2p+1 divisor is a safe spectral-radius bound for the tensor-product operator.
    vel = (1.0, 0.5)
    for p in degrees:
        A2, hx, hy = semidiscrete_matrix_2d(ncells_2d, p, vel)
        radius = float(np.abs(np.linalg.eigvals(A2)).max())
        wave_rate = abs(vel[0]) / hx + abs(vel[1]) / hy
        report["cfl_2d"] = report.get("cfl_2d", {})
        report["cfl_2d"][f"p{p}"] = {
            "ncells": ncells_2d,
            "ndofs": int(A2.shape[0]),
            "spectral_radius": radius,
            "spectral_radius_over_wave_rate": radius / wave_rate,
            "heuristic_factor_2p1": 2 * p + 1,
            "cfl_max_rk4_with_heuristic": RK4_REAL_LIMIT * (2 * p + 1) / (radius / wave_rate),
        }
    return report


def figure(degrees=(0, 1, 2, 3), ncells: int = 64, speed: float = 1.0,
           outdir: Path = Path(".")) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = theta_grid(ncells)
    ppw = 2.0 * math.pi / theta
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for p in degrees:
        c = dispersion_curve(p, ncells, speed)
        axes[0].loglog(ppw, c["phase_err"], marker="o", ms=3.0, lw=1.2, label=f"DG-P{p}")
        axes[1].loglog(ppw, np.maximum(c["diss_per_wavelength"], 1e-17), marker="o", ms=3.0,
                       lw=1.2, label=f"DG-P{p}")
    axes[0].set_xlabel("points per wavelength"); axes[0].set_ylabel(r"$|c_{num}-a|/a$")
    axes[0].set_title("phase (dispersion) error, upwind DG"); axes[0].set_xlim(2, 64)
    axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("points per wavelength")
    axes[1].set_ylabel("amplitude decay per wavelength")
    axes[1].set_title("numerical dissipation, upwind DG"); axes[1].set_xlim(2, 64)
    axes[1].grid(True, which="both", alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "dispersion_dissipation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def table_main(report: dict) -> str:
    lines = ["| degree | phase-err order (pair) | ppw @1% | ppw @0.1% | "
             r"$\|\lambda\|_{max}h/a$" " | $(2p+1)$ | $cfl_{max}$ | $cfl_{max}/0.08$ |",
             "|---|---|---|---|---|---|---|---|"]
    for key in sorted(report["orders"], key=lambda s: int(s[1:])):
        p = int(key[1:]); o = report["orders"][key]; c = report["cfl"][key]
        pp = report["ppw"][key]
        nyq = " (Nyquist)" if pp["tol_1pct"]["at_nyquist_limit"] else ""
        lines.append(f"| {p} | {o['observed_order']:.2f} ({o['order_pair']}) | "
                     f"{pp['tol_1pct']['ppw']:.2f}{nyq} | "
                     f"{pp['tol_0p1pct']['ppw']:.2f} | {c['spectral_radius_h_over_a']:.3f} | "
                     f"{c['heuristic_factor_2p1']} | {c['cfl_max_rk4_with_heuristic']:.3f} | "
                     f"{c['safety_margin']:.1f} |")
    return "\n".join(lines)


def table_anchors(report: dict) -> str:
    a = report["anchors"]
    lines = ["| anchor | value |", "|---|---|",
             f"| $p=0$ symbol vs closed-form upwind FTBS, "
             r"$\max|\lambda-\lambda_{exact}|$" f" | {a['p0_symbol_max_abs_err']:.3e} |"]
    for key in sorted((k for k in a if k.startswith("p") and k.endswith("max_real_part")),
                      key=lambda s: int(s.split('_')[0][1:])):
        lines.append("| " + r"$\max \mathrm{Re}\lambda$" + f" ({key.split('_')[0]}) "
                     f"| {a[key]:.3e} |")
    for key in sorted((k for k in a if "ambiguous" in k),
                      key=lambda s: int(s.split('_')[0][1:])):
        lines.append(f"| branch-tracking ambiguities ({key.split('_')[0]}, N=32) "
                     f"| {a[key]} |")
    for key in sorted((k for k in a if "symbol_vs_projection" in k),
                      key=lambda s: int(s.split('_')[0][1:])):
        lines.append("| block symbol vs full-operator projection ("
                     f"{key.split('_')[0]}) | {a[key]:.3e} |")
    mis = a["rk4_step_max_rel_mismatch"]
    for key in sorted(mis, key=lambda s: int(s.split('_')[0][1:])):
        lines.append(f"| RK4 step: spectral operator vs production solver ({key}) "
                     f"| {mis[key]:.3e} |")
    return "\n".join(lines)


def table_2d(report: dict) -> str:
    lines = ["| degree | 2-D ndofs | " + r"$\|\lambda\|_{max}$"
             + " | " + r"$\|\lambda\|_{max}/\mathrm{wave\_rate}$"
             + " | $(2p+1)$ | $cfl_{max}$ |", "|---|---|---|---|---|---|"]
    for key in sorted(report["cfl_2d"], key=lambda s: int(s[1:])):
        c = report["cfl_2d"][key]
        lines.append(f"| {key[1:]} | {c['ndofs']} | {c['spectral_radius']:.3f} | "
                     f"{c['spectral_radius_over_wave_rate']:.3f} | {c['heuristic_factor_2p1']} | "
                     f"{c['cfl_max_rk4_with_heuristic']:.3f} |")
    return "\n".join(lines)


def table_negctl(report: dict) -> str:
    lines = ["| CFL | dt | steps | terminal $L_2$ error |", "|---|---|---|---|"]
    for row in report["negative_control"]:
        lines.append(f"| {row['cfl']} | {row['dt']:.3e} | {row['steps']} | {row['l2']:.3e} |")
    return "\n".join(lines)


def markdown_tables(report: dict) -> str:
    return "\n\n".join([table_main(report), table_anchors(report),
                         table_2d(report), table_negctl(report)])


def render(template: Path, out: Path, report: dict) -> Path:
    body = template.read_text()
    for marker, table in (("<!--TABLE:main-->", table_main(report)),
                          ("<!--TABLE:anchors-->", table_anchors(report)),
                          ("<!--TABLE:2d-->", table_2d(report)),
                          ("<!--TABLE:negctl-->", table_negctl(report))):
        if marker not in body:
            raise SystemExit(f"template missing marker {marker}")
        body = body.replace(marker, table)
    out.write_text(body)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="validation/dispersion")
    ap.add_argument("--template", default="docs/validation/_dg-dispersion-dissipation.template.md")
    ap.add_argument("--doc", default="docs/validation/dg-dispersion-dissipation.md")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    report = analyse()
    fig = figure(outdir=outdir)
    report["figure"] = fig.name
    (outdir / "dispersion_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (outdir / "tables.md").write_text(markdown_tables(report) + "\n")
    doc = render(Path(args.template), Path(args.doc), report)
    print(table_main(report))
    print(f"\nwrote {outdir/'dispersion_report.json'}, {fig}, {doc}")


if __name__ == "__main__":
    main()
