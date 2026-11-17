"""Focused numerical-validation pytest for :mod:`tensordg.dispersion_analysis`.

This is a *self-contained* companion to ``test_dispersion_analysis.py`` (which
reads a pre-computed ``validation/dispersion_analysis/results.json`` artifact).
Every gate here recomputes the quantity from the shipped module functions, so
it cannot be satisfied by a stale artifact:

1. **Closed-form quantitative comparison** -- the ``p = 0`` upwind finite-volume
   branch is compared against the exact symbol ``lambda(theta) = (exp(-i theta)
   - 1) / h`` (``h = 1``) to round-off.  The tolerance is *derived*, not guessed:
   the assembly pushes unit modal vectors through the production ``_rhs`` kernel
   on a ``float64`` periodic mesh, so the only error source is the ``float64``
   round-off of the ``exp``/``eigvals`` calls (~1e-16); a ``1e-12`` bound is
   therefore 4 orders of magnitude above the numerical floor and 8 below any
   real scheme change.  The same anchor also pins the small-``theta`` continuum
   limit ``c_num -> 1`` and the ``p = 0`` dissipation ``(1 - cos theta)/theta``.
2. **CLI smoke** -- ``python -m tensordg.dispersion_analysis`` must exit 0,
   print the ``p0 anchor`` line and write ``validation/dispersion_analysis/
   results.json`` with a ``p0_closed_form_anchor`` whose error is below the
   same ``1e-12`` bound.
3. **Invalid-parameter negative tests** -- a negative ``degree``, a
   ``NaN``/``inf`` wavenumber and a ``degree``/``matrix`` mismatch must all be
   rejected (``RuntimeError``/``LinAlgError``/``ValueError``), so the analysis
   pipeline cannot silently return a wrong spectrum for a bad call.

Everything is deterministic (fixed grids, no random numbers).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg import dispersion_analysis as da  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = REPO_ROOT / "validation" / "dispersion_analysis" / "results.json"

# --------------------------------------------------------------------------- #
# 1. closed-form quantitative comparison (tolerance derived, see module doc)
# --------------------------------------------------------------------------- #
# Derivation of the 1e-12 bound: the p=0 operator is assembled by pushing unit
# modal vectors through advection_1d._rhs on a float64 periodic mesh (h=1).  The
# only numerical error is float64 round-off of exp/eigvals (~1e-16 absolute).
# 1e-12 is 4 orders above that floor and 8 below any real flux/basis change, so
# it is a tight-but-safe anchor for "the assembled operator IS the closed form".


def test_p0_branch_matches_closed_form_upwind_fv_symbol_to_roundoff():
    """p=0: the assembled branch equals (exp(-i theta)-1)/h exactly."""
    thetas = np.linspace(1e-3, math.pi, 200)
    branch = da.physical_branch(0, thetas)
    closed = np.array([da.upwind_fv_symbol(t) for t in thetas])
    # tolerance 1e-12: see module docstring derivation (float64 round-off floor).
    assert np.abs(branch - closed).max() < 1e-12


def test_p0_phase_speed_tends_to_continuum_limit():
    """As theta -> 0 the p=0 numerical phase speed must tend to the exact 1."""
    theta = 1e-6
    lam = da.physical_branch(0, np.array([theta]))[0]
    c_num = -lam.imag / theta
    # sin(theta)/theta -> 1 as theta -> 0; tolerance 1e-6 (float64 round-off).
    assert abs(c_num - 1.0) < 1e-6


def test_p0_dissipation_matches_closed_form_one_minus_cos():
    """p=0 amplitude loss -Re(lambda)/theta equals (1-cos theta)/theta."""
    theta = 0.5
    lam = da.physical_branch(0, np.array([theta]))[0]
    decay = -lam.real / theta
    closed = (1.0 - math.cos(theta)) / theta
    # both are float64 evaluations of the same closed form; tolerance 1e-12.
    assert abs(decay - closed) < 1e-12


def test_p1_resolution_requirement_is_textbook_plausible():
    """1 % phase error at p=1 needs 4-8 points per wavelength (textbook band)."""
    ppw = da.resolution_table(1)["ppw_phase"]
    assert ppw is not None and 3.0 < ppw <= 8.0, ppw


def test_dissipation_superconvergence_exponent_is_two_p_plus_two():
    """Measured dissipative exponent must be 2p+2 (super-convergence, measured)."""
    for degree in (0, 1, 2):
        expo = da._fit_slopes(degree)["dissipation_exponent"]
        assert abs(expo - (2 * degree + 2)) < 0.1, (degree, expo)


# --------------------------------------------------------------------------- #
# 2. CLI smoke
# --------------------------------------------------------------------------- #
def test_cli_smoke_exits_zero_and_writes_artifact():
    """``python -m tensordg.dispersion_analysis`` must run and write results."""
    if ARTIFACT.exists():
        ARTIFACT.unlink()
    proc = subprocess.run(
        [sys.executable, "-m", "tensordg.dispersion_analysis"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "p0 anchor" in proc.stdout
    assert ARTIFACT.exists()
    data = json.loads(ARTIFACT.read_text())
    assert data["p0_closed_form_anchor"]["max_abs_error"] < 1e-12


def test_cli_artifact_contains_all_degrees_and_energy_stability():
    """The CLI artifact must cover p=0..4 and report no growing mode."""
    data = json.loads(ARTIFACT.read_text())
    assert [r["degree"] for r in data["records"]] == [0, 1, 2, 3, 4]
    for record in data["records"]:
        assert record["max_stability_violation"] <= 1e-12


# --------------------------------------------------------------------------- #
# 3. invalid-parameter negative tests
# --------------------------------------------------------------------------- #
def test_negative_degree_is_rejected():
    """A negative polynomial degree must not assemble a spectrum."""
    with pytest.raises(RuntimeError):
        da.assemble_operator(-1)


def test_nan_wavenumber_is_rejected():
    """A NaN theta must not produce a (silently wrong) dispersion curve."""
    with pytest.raises((np.linalg.LinAlgError, ValueError)):
        da.physical_branch(1, np.array([float("nan")]))


def test_inf_wavenumber_is_rejected():
    """An infinite theta must be rejected by the eigenvalue solver."""
    with pytest.raises((np.linalg.LinAlgError, ValueError)):
        da.physical_branch(1, np.array([float("inf")]))


def test_degree_matrix_mismatch_is_rejected():
    """symbol_matrix with a degree inconsistent with the matrix must fail."""
    matrix = da.assemble_operator(1)  # 2x2 blocks
    with pytest.raises(ValueError):
        da.symbol_matrix(matrix, 2, 0.1)  # asks for 3x3 blocks


def test_negative_tolerance_in_resolution_table_is_rejected():
    """A negative tolerance is physically meaningless and must not be used."""
    # tolerance <= 0 makes the "first theta meeting it" predicate vacuous; the
    # table must report no resolution (None), never a bogus finite PPW.
    res = da.resolution_table(1, tolerance=-0.1)
    assert res["ppw_phase"] is None and res["ppw_amplitude"] is None
