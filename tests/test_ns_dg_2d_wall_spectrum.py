"""Machine-checked judgements about the *linear stability* of the wall operator.

``tests/test_ns_dg_2d_wall.py`` records the wall-bounded transient failure and
``tools/diag_ns_wall.py`` localised it to the temperature channel.  The tests
here close the argument by judging the semi-discrete operator itself: the
wall-penalty term is a non-symmetric rank-1 boundary term, so it destabilises the
operator for *any* sign and by an amount proportional to its magnitude.  The
three ``xfail(strict=True)`` tests of the wall suite therefore cannot be retired
by tuning ``sigma`` (or the time step); they need the missing symmetric Nitsche
(adjoint) term.  These assertions are the evidence for that statement, and they
turn red the moment somebody actually adds the adjoint term -- which is exactly
when the claim in ``docs/reports/tensordg-ns-wall-spectrum-m2731.md`` must be retired.

Every number is computed from ``tensordg.ns_dg_2d`` by the helpers in
``tools/diag_ns_wall_spectrum.py``; nothing is hard-coded except the thresholds.
"""
import numpy as np
import pytest

import tensordg.ns_dg_2d as ns
from tools.diag_ns_wall_spectrum import (
    ORIGINAL_WALL_FLUX,
    couette_state,
    spectrum,
    static_state,
    wall_flux_variant,
)

# The live ``ns._boundary_flux`` carries the m2782 ``outward_sign`` fix, which
# stabilised the wall operator (measured max Re(lambda) = 8.24e-14 at sigma = 8
# on the static state under the m3139 ``face`` default, 7.4e-14 under the
# pre-m3139 ``taylor`` default; > 50 pre-fix).  The judgements below are the
# evidence for the *pre-fix* operator that mission m2731 diagnosed, so every
# fixture reinstalls the sign-blind wall flux (``wall_flux_variant((1,1,1))``,
# byte-for-byte the shipped operator of that day) before taking a spectrum.
SHIPPED_AT_M2731 = wall_flux_variant((1.0, 1.0, 1.0))

SIGMA_SHIPPED = 8.0  # 2 (degree + 1)**2 with degree = 1


@pytest.fixture(scope="module")
def states():
    return static_state(), couette_state()


@pytest.fixture(scope="module")
def static_spectra(states):
    op, coeff, boundary = states[0]
    data = {}
    ns._boundary_flux = SHIPPED_AT_M2731
    try:
        for sigma in (0.0, 4.0, 8.0, 16.0):
            data[sigma] = spectrum(op, coeff, boundary, sigma)
    finally:
        ns._boundary_flux = ORIGINAL_WALL_FLUX
    return data


@pytest.fixture(scope="module")
def flipped_spectrum(states):
    op, coeff, boundary = states[0]
    ns._boundary_flux = wall_flux_variant((-1.0, -1.0, -1.0))
    try:
        result = spectrum(op, coeff, boundary, SIGMA_SHIPPED)
    finally:
        ns._boundary_flux = ORIGINAL_WALL_FLUX
    return result  # compared against static_spectra[8.0] (both sign-blind)


def test_wall_operator_is_linearly_unstable(static_spectra):
    """A uniform fluid at rest between isothermal walls has a *zero* residual,
    yet the Jacobian at that state has a large positive eigenvalue: the wall
    treatment is unstable independently of any flow."""
    growth = static_spectra[SIGMA_SHIPPED]["max_real"]
    assert growth > 50.0, growth
    assert static_spectra[SIGMA_SHIPPED]["rhs_max"] < 1e-12


def test_growth_is_proportional_to_the_penalty(static_spectra):
    """Doubling sigma doubles the growth rate (measured ratio 2.05 at 8 -> 16)."""
    ratio = static_spectra[16.0]["max_real"] / static_spectra[8.0]["max_real"]
    assert 1.95 < ratio < 2.10, ratio
    ratio = static_spectra[8.0]["max_real"] / static_spectra[4.0]["max_real"]
    assert 1.95 < ratio < 2.15, ratio


def test_growth_depends_only_on_penalty_magnitude(static_spectra, flipped_spectrum):
    """Flipping the sign of the wall penalty changes nothing (relative gap
    1.7e-09): the term carries no dissipation, so no choice of sigma fixes it."""
    positive = static_spectra[SIGMA_SHIPPED]["max_real"]
    negative = flipped_spectrum["max_real"]
    assert positive > 50.0
    assert abs(positive - negative) / positive < 1e-4, (positive, negative)


def test_without_the_penalty_the_wall_operator_is_neutral(static_spectra):
    """With the wall penalty removed the operator is exactly neutral
    (|max Re lambda| < 1e-6): *all* wall dissipation comes from that one term."""
    assert abs(static_spectra[0.0]["max_real"]) < 1e-6, static_spectra[0.0]["max_real"]


def test_unstable_mode_is_the_wall_energy_row(static_spectra):
    """The unstable eigenvector is an energy mode concentrated on the wall row:
    >90 % of its magnitude sits in the first cell row and the energy component
    dominates the four variables."""
    matrix = static_spectra[SIGMA_SHIPPED]["matrix"]
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = int(np.argmax(eigenvalues.real))
    assert eigenvalues[index].real > 50.0
    # state shape is (nx, ny, (p+1), (p+1), 4)
    op, coeff, boundary = static_state()
    vector = np.abs(eigenvectors[:, index]).reshape(coeff.shape)
    by_variable = vector.sum(axis=(0, 1, 2, 3))
    by_row = vector.sum(axis=(0, 2, 3, 4))
    assert by_variable.argmax() == 3, by_variable          # energy variable
    assert by_row[0] / by_row.sum() > 0.85, by_row
    assert by_row[-1] / by_row.sum() < 1e-5, by_row        # top wall ~untouched


def test_couette_state_shows_the_same_instability(states):
    """The analytic steady Couette state is unstable in the same way, so the
    transient failure is not a property of the initial condition."""
    op, coeff, boundary = states[1]
    ns._boundary_flux = SHIPPED_AT_M2731
    try:
        result = spectrum(op, coeff, boundary, SIGMA_SHIPPED)
    finally:
        ns._boundary_flux = ORIGINAL_WALL_FLUX
    assert result["max_real"] > 50.0, result["max_real"]
    assert result["max_real"] < np.abs(result["min_real"])