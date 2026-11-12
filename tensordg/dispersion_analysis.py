"""Eigenvalue (dispersion/dissipation) analysis of the TensorDG advection operator.

Verification debt this module closes
------------------------------------
Every ``tensordg`` verification so far is an *error norm* study: L2 error versus
mesh size, or a profile comparison against an exact solution.  None of them
answers the question a wave-propagation user actually asks -- **how many points
per wavelength does DG need, and what does it do to the wave amplitude?**  The
existing ``tensordg/dispersive.py`` is *physics* dispersion (water waves); it
says nothing about the dispersion of the *scheme*.

This module computes the exact spectrum of the semi-discrete operator that
:mod:`tensordg.advection_1d` actually assembles, and derives from it

* the **dissipation** (amplitude decay rate) and **phase-speed error** of every
  polynomial degree ``p = 0..4`` as a function of the non-dimensional
  wavenumber ``theta = k h`` (``h`` = element size),
* the **super-convergence exponent** of the dissipative error, measured from the
  computed spectrum (not quoted),
* the **points-per-wavelength (PPW)** needed for 1 % phase / 1 % amplitude
  error -- the practical "resolution budget" of the scheme.

Method (no hand copies, no analytic black boxes)
-----------------------------------------------
1. **Assembly.** The operator is assembled *from the production kernel*
   ``advection_1d._rhs``: unit modal vectors are pushed through ``_rhs`` on a
   periodic mesh of ``N`` elements with ``h = 1``, giving the exact
   ``(N (p+1)) x (N (p+1))`` matrix ``S`` of the code under test.  Nothing is
   re-derived by hand, so the spectrum below describes the shipped code, not an
   idealized scheme.
2. **Block symbol.** ``S`` is block-circulant (translation invariance on a
   uniform periodic mesh), so a single ``(p+1) x (p+1)`` *symbol matrix*
   ``T(theta) = sum_j S[0, j] exp(-i theta j)`` carries the whole spectrum,
   with ``theta = 2 pi m / N``.  The identity

       spec(S) = union over m of spec(T(2 pi m / N))

   is asserted numerically (``symbol_union_error``) -- that is the structural
   check that "symbol matrix" is the right language and that the two independent
   eigenvalue computations agree.
3. **Closed-form anchor at p = 0.** For ``p = 0`` the scheme is the classic
   1st-order upwind finite-volume operator, whose symbol is known exactly:

       lambda(theta) = (exp(-i theta) - 1) / h

   (or its complex conjugate, depending on the sign convention of the upwind
   flux).  The assembled eigenvalue is compared against this closed form to
   roundoff -- an implementation-independent anchor for the whole pipeline.
4. **Branch tracking.** The *physical* mode is the least-damped eigenvalue
   branch: starting at ``theta = 0`` from the zero eigenvalue and continuing in
   ``theta`` by nearest-neighbour tracking.  Exact advection has
   ``lambda = -i theta``, so the numerical phase speed is ``c_num = -Im(lambda)
   / theta`` and the amplitude loss is ``-Re(lambda) / theta``.

Honest scope statements
-----------------------
* ``Re(lambda) <= 0`` for every degree and every ``theta`` is asserted; this is
  the discrete energy-stability property of the upwind DG operator and it is
  checked on the assembled matrix, not assumed.
* The classical literature result is that the upwind DG *dissipation* error
  super-converges at a rate faster than the formal order (Hu, Hussaini &
  Rasetarinera, J. Comput. Phys. 156 (1999); Ainsworth, J. Comput. Phys. 198
  (2004)).  The **paper text was not readable in this sandbox**, so no number
  is copied from it: the exponents below are *measured* from the assembled
  operator and are reported as measurements.  The citation is context only.
* Linear, constant-coefficient, periodic, one-dimensional advection with
  upwind flux and no limiter; nonlinear/limited schemes have a different (and
  non-normal) spectrum, so these numbers must not be transferred to shock
  problems.

Everything is deterministic (fixed grids, no random numbers).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from .advection_1d import _operators, _rhs

MAX_ANALYSED_DEGREE = 4
#: Number of elements in the periodic mesh used for the assembly.
ASSEMBLY_CELLS = 16
#: Normalised wavenumbers ``theta = k h`` used for the branch/error curves.
THETA = np.linspace(1e-3, math.pi, 400)
#: Window for the super-convergence fit.  The lower end is limited by the
#: round-off floor of the eigenvalue solver (~1e-16 absolute): for ``p = 4`` the
#: dissipative error hits that floor already at ``theta ~ 1e-2``, so points whose
#: decay falls below ``FIT_FLOOR`` are dropped instead of being fitted (a wrong
#: slope from round-off noise was the bug this constant prevents).
FIT_THETA = np.logspace(-1.4, -0.3, 60)
FIT_FLOOR = 1e-12


# --------------------------------------------------------------------------- #
# assembled operator and its block symbol
# --------------------------------------------------------------------------- #
def assemble_operator(degree: int, ncells: int = ASSEMBLY_CELLS,
                      speed: float = 1.0) -> torch.Tensor:
    """Assemble the exact periodic operator matrix of ``advection_1d._rhs``."""
    _x, _w, _phi, volume, phi_l, phi_r = _operators(
        degree, torch.device("cpu"), torch.float64)
    nmodes = degree + 1
    size = ncells * nmodes
    matrix = torch.zeros(size, size, dtype=torch.float64)
    for index in range(size):
        probe = torch.zeros(ncells, nmodes, dtype=torch.float64)
        probe[index // nmodes, index % nmodes] = 1.0
        # h = 1: the operator is homogeneous in h, lambda(h) = lambda(1) / h.
        matrix[:, index] = _rhs(probe, 1.0, speed, volume, phi_l, phi_r).reshape(-1)
    return matrix


def symbol_matrix(matrix: torch.Tensor, degree: int, theta: float,
                  ncells: int = ASSEMBLY_CELLS) -> np.ndarray:
    """Block symbol ``T(theta) = sum_d A[d] exp(-i theta d)``.

    ``A[d]`` is the block mapping the cell ``d`` positions to the *left* of a
    cell into that cell: ``A[d] = S[0, (-d) mod N]``.  Summing over the wrapped
    column index instead of the stencil shift ``d`` is a silent, sign-free bug:
    for the two-point ``p = 0`` stencil the wrapped column is ``j = N - 1`` while
    the physical shift is ``d = +1``, so the naive sum evaluates
    ``exp(+i (N-1) theta)`` and the spectrum comes out rotated by a factor that
    happens to vanish at the grid frequencies ``theta = 2 pi m / N`` -- i.e. the
    union identity below still passes at grid points while every off-grid
    wavenumber (the whole dispersion curve) is wrong.
    """
    nmodes = degree + 1
    theta = float(theta)
    symbol = np.zeros((nmodes, nmodes), dtype=complex)
    for shift in range(-(ncells // 2 - 1), ncells // 2 + 1):
        column = (-shift) % ncells
        block = matrix[:nmodes, column * nmodes:(column + 1) * nmodes]
        if not torch.any(block):
            continue
        symbol += block.numpy().astype(complex) * np.exp(-1j * theta * shift)
    return symbol


def symbol_union_error(degree: int, ncells: int = ASSEMBLY_CELLS) -> float:
    """``spec(S)`` vs ``union_m spec(T(2 pi m / N))`` -- structural identity."""
    matrix = assemble_operator(degree, ncells)
    full = np.linalg.eigvals(matrix.numpy().astype(complex))
    union = np.concatenate([
        np.linalg.eigvals(symbol_matrix(matrix, degree, 2.0 * math.pi * m / ncells, ncells))
        for m in range(ncells)
    ])
    if full.shape != union.shape:  # pragma: no cover - defensive
        return float("inf")
    # Nearest-neighbour matching: sorting complex eigenvalues is not a stable
    # identification (near-degenerate pairs swap under round-off).
    return float(np.max([np.min(np.abs(union - value)) for value in full]))


def upwind_fv_symbol(theta: float) -> complex:
    """Exact ``p = 0`` upwind symbol ``(exp(-i theta) - 1) / h`` with ``h = 1``."""
    return complex(np.exp(-1j * theta) - 1.0)


# --------------------------------------------------------------------------- #
# physical branch, dissipation and phase error
# --------------------------------------------------------------------------- #
def _pick_continuation(values: np.ndarray, target: complex, anchor: int) -> complex:
    """Choose the eigenvalue continuing the *right-going* physical branch.

    Nearest-neighbour tracking alone is ambiguous whenever two candidates are
    equidistant from the predicted target.  That is not a corner case: at the
    Nyquist wavenumber ``theta = pi`` the symbol matrix is real, so its spectrum
    is closed under complex conjugation and the physical pair
    ``Re(lambda) +- i Im(lambda)`` is *exactly* degenerate in distance.  Before
    this anchor existed the tie was broken by whatever order ``eigvals``
    returned, and a point query ``physical_branch(3, [pi])`` reported
    ``c_num = -1.0006`` -- a right-going wave described as left-going -- while
    the full sweep from below happened to keep the correct member.  A
    time-domain measurement of the shipped solver (see
    ``tools/dispersion_time_domain.py``) gives ``+1.0006``, so the *sign* was
    the only wrong quantity.

    The anchor is the sign of ``Im(lambda)`` already established along the
    tracked branch (``anchor = 0`` before the first non-real pick, where the
    right-going mode has ``Im(lambda) <= 0`` because ``c_num = -Im(lambda) /
    theta > 0``).  Ties are therefore resolved physically instead of
    numerically.  Near-ties within the round-off of the symbol matrix
    (``1e-9`` relative) are treated as exact ties: a genuine, well-separated
    second branch is never masked, because its distance differs by more than
    that.
    """
    distance = np.abs(values - target)
    best = float(distance.min())
    candidates = np.flatnonzero(distance <= best + 1e-9 + 1e-9 * best)
    if candidates.size > 1:
        if anchor == 0:
            keep = [int(i) for i in candidates if values[i].imag <= 0.0]
        else:
            keep = [int(i) for i in candidates if anchor * values[i].imag >= 0.0]
        if keep:
            candidates = np.asarray(keep)
    return complex(values[candidates[int(np.argmin(distance[candidates]))]])


def physical_branch(degree: int, thetas: np.ndarray | None = None) -> np.ndarray:
    """Least-damped right-going eigenvalue branch, tracked from ``theta = 0``.

    The branch is *sign anchored*: degenerate (conjugate) candidates are
    resolved with the ``Im(lambda) <= 0`` continuation instead of by
    eigval ordering -- see :func:`_pick_continuation` for the defect this
    prevents and the time-domain measurement that adjudicates it.
    """
    thetas = THETA if thetas is None else np.atleast_1d(np.asarray(thetas, dtype=float))
    matrix = assemble_operator(degree)
    branch = []
    previous = 0.0 + 0.0j  # theta = 0 eigenvalue of the constant mode
    slope = 0.0 + 0.0j
    anchor = 0
    last = None
    for theta in thetas:
        values = np.linalg.eigvals(symbol_matrix(matrix, degree, float(theta)))
        target = previous + slope * (theta - last) if last is not None else previous
        pick = _pick_continuation(values, target, anchor)
        if pick.imag != 0.0:
            anchor = int(np.sign(pick.imag))
        if last is not None:
            slope = (pick - previous) / (theta - last)
        branch.append(pick)
        previous, last = pick, theta
    return np.asarray(branch, dtype=complex)


def _fit_window(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick the widest small-``theta`` window whose decay stays above round-off.

    The dissipative error of degree ``p`` behaves like ``theta^(2p+2)``, so for
    ``p = 4`` it drops below the ~1e-16 eigenvalue round-off floor already at
    ``theta ~ 2e-2``.  Fitting a fixed window therefore returns noise (a
    garbage slope of 46 was the first result of this module); the window is
    chosen per degree instead.
    """
    for lo, hi in ((-1.4, -0.3), (-1.2, -0.1), (-1.0, 0.1), (-0.8, 0.3), (-0.5, 0.5)):
        grid = np.logspace(lo, hi, 60)
        branch = physical_branch(degree, grid)
        decay = -branch.real
        keep = decay > FIT_FLOOR
        if keep.sum() >= 8:
            return grid, keep
    grid = np.logspace(-0.5, 0.5, 60)
    return grid, -physical_branch(degree, grid).real > 0


def _fit_slopes(degree: int) -> dict:
    """Measured slopes of ``log(-Re lambda)`` and ``log|dphase|`` vs ``log theta``."""
    FIT_THETA, _ = _fit_window(degree)
    branch = physical_branch(degree, FIT_THETA)
    decay = -branch.real
    phase = np.abs(-branch.imag / FIT_THETA - 1.0)
    keep = decay > FIT_FLOOR
    if keep.sum() < 6:  # pragma: no cover - only for very high degrees
        return {"dissipation_exponent": float("nan"),
                "phase_exponent": float("nan"), "points_used": int(keep.sum())}
    diss = float(np.polyfit(np.log(FIT_THETA[keep]), np.log(decay[keep]), 1)[0])
    pkeep = phase[keep] > 1e-14
    phase_exp = (float(np.polyfit(np.log(FIT_THETA[keep][pkeep]),
                                  np.log(phase[keep][pkeep]), 1)[0])
                 if pkeep.sum() >= 6 else float("nan"))
    return {"dissipation_exponent": diss, "phase_exponent": phase_exp,
            "points_used": int(keep.sum()),
            "fit_theta_range": [float(FIT_THETA[keep][0]), float(FIT_THETA[keep][-1])]}


def resolution_table(degree: int, tolerance: float = 1e-2) -> dict:
    """Smallest points-per-wavelength meeting a phase and an amplitude tolerance."""
    branch = physical_branch(degree)
    c_num = -branch.imag / THETA
    phase_error = np.abs(c_num - 1.0)
    amplitude_error = -branch.real / THETA

    def first_ppw(error: np.ndarray) -> float | None:
        ok = np.where(error <= tolerance)[0]
        if ok.size == 0:
            return None
        # theta decreases towards the left; the resolution requirement is the
        # *largest* theta (fewest points per wavelength) that still meets tolerance.
        return float(2.0 * math.pi / THETA[ok[-1]])

    return {
        "degree": degree,
        "tolerance": tolerance,
        "ppw_phase": first_ppw(phase_error),
        "ppw_amplitude": first_ppw(amplitude_error),
        "max_phase_error_at_pi": float(abs(c_num[-1] - 1.0)),
    }


def analyse(degrees: range | None = None) -> dict:
    degrees = range(0, MAX_ANALYSED_DEGREE + 1) if degrees is None else degrees
    records = []
    for degree in degrees:
        branch = physical_branch(degree)
        c_num = -branch.imag / THETA
        records.append({
            "degree": degree,
            "theta": THETA.tolist(),
            "eigenvalue_real": branch.real.tolist(),
            "eigenvalue_imag": branch.imag.tolist(),
            "numerical_phase_speed": c_num.tolist(),
            "phase_error": np.abs(c_num - 1.0).tolist(),
            "amplitude_decay": (-branch.real / THETA).tolist(),
            **_fit_slopes(degree),
            "resolution": resolution_table(degree),
            "samples": {
                f"{t:.3f}": {
                    "theta": t,
                    "numerical_phase_speed": float(-branch[np.argmin(np.abs(THETA - t))].imag / t),
                    "amplitude_decay": float(-branch[np.argmin(np.abs(THETA - t))].real / t),
                }
                for t in (0.1, math.pi / 4, math.pi / 2, 2.5, math.pi)
            },
            "design_order": degree + 1,
            "max_eigenvalue_real": float(np.max(branch.real)),
            "symbol_union_error": symbol_union_error(degree),
            "max_stability_violation": float(np.max(
                np.linalg.eigvals(assemble_operator(degree).numpy().astype(complex)).real)),
        })
    anchor = {
        "degree": 0,
        "closed_form": "lambda = (exp(-i theta) - 1) / h, h = 1",
        "max_abs_error": float(np.max(np.abs(
            physical_branch(0)[:50] - np.array([upwind_fv_symbol(t) for t in THETA[:50]])))),
        "max_abs_error_conjugate": float(np.max(np.abs(
            physical_branch(0)[:50] - np.array([np.conj(upwind_fv_symbol(t)) for t in THETA[:50]])))),
    }
    return {
        "config": {
            "assembly_cells": ASSEMBLY_CELLS,
            "degrees": list(degrees),
            "theta_points": int(THETA.size),
            "fit_theta": [float(FIT_THETA[0]), float(FIT_THETA[-1])],
            "operator": "tensordg.advection_1d._rhs, upwind flux, h=1, speed=1",
        },
        "p0_closed_form_anchor": anchor,
        "records": records,
        "provenance": {
            "git_sha": _git_sha(),
            "arch": __import__("platform").machine(),
            "torch": torch.__version__,
        },
    }


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover - best effort provenance
        return "unknown"


def main() -> int:
    out_dir = Path("validation/dispersion_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = analyse()
    (out_dir / "results.json").write_text(json.dumps(result, indent=1))
    print(f"p0 anchor: max|lambda - closed form| = "
          f"{result['p0_closed_form_anchor']['max_abs_error']:.3e} "
          f"(conjugate branch {result['p0_closed_form_anchor']['max_abs_error_conjugate']:.3e})")
    for record in result["records"]:
        res = record["resolution"]
        print(f"p={record['degree']}  diss.exp={record['dissipation_exponent']:.2f}  "
              f"phase.exp={record['phase_exponent']:.2f}  "
              f"PPW(phase 1%)={res['ppw_phase']}  PPW(amp 1%)={res['ppw_amplitude']}  "
              f"spec(S)vs symbol={record['symbol_union_error']:.2e}  "
              f"max Re lambda={record['max_stability_violation']:.2e}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())