"""Verification tests for Burgers shock capturing (module burgers_shock_1d).

The acceptance criteria of the IN16-K7 campaign node are encoded here:

* smooth-regime design order ``>= p + 0.9`` for P1/P2/P3 and both interface
  fluxes (Rusanov/Lax--Friedrichs and Roe);
* the captured shock sits at the Rankine--Hugoniot location, and the
  wave-front position error decreases under refinement;
* the Gibbs overshoot is bounded by shock capturing: the TVB-limited solution
  stays several times below the unlimited one, while the unlimited overshoot
  does not decay with the mesh (that is what makes it a Gibbs oscillation);
* negative cases are rejected: a negative density state, rarefaction data
  treated as a shock, an unknown flux and an unsupported degree all raise.
"""
import math

import pytest
import torch

from tensordg.burgers_1d import burgers_error, initial_profile as smooth_profile
from tensordg.burgers_shock_1d import (
    RiemannData,
    exact_solution,
    gibbs_overshoot,
    lax_friedrichs_flux,
    positivity_filter,
    roe_flux,
    shock_expected_position,
    solve_burgers_shock,
    solve_burgers_tracer,
    total_variation,
)
from tensordg.burgers_shock_1d import _operators

DATA = RiemannData(u_left=1.0, u_right=0.0, x0=0.1)
FINAL_TIME = 0.2
SHOCK = 0.2  # x0 + 0.5 * FINAL_TIME
PHI = _operators(2, torch.device("cpu"), torch.float64)[2]

_SHOCK_CACHE: dict = {}


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    """Small tensors: one thread is faster and deterministic (kernels dominate)."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def shock_run(cells: int, flux: str = "lax_friedrichs", limiter=None):
    """Memoised shock solve -- the tests below share the same configurations."""
    key = (cells, flux, limiter)
    if key not in _SHOCK_CACHE:
        _SHOCK_CACHE[key] = solve_burgers_shock(
            ncells=cells, degree=2, final_time=FINAL_TIME, cfl=0.2,
            flux=flux, limiter=limiter, data=DATA,
        )
    return _SHOCK_CACHE[key]


# ---------------------------------------------------------------------------
# 1. Smooth-regime design order
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("flux", ["lax_friedrichs", "roe"])
def test_smooth_region_keeps_design_order(degree, flux):
    """Pre-shock sine data: observed order must stay at or above p + 0.9."""
    meshes = [32, 64, 128]
    errors = []
    for cells in meshes:
        x, coeff, _ = solve_burgers_shock(
            ncells=cells, degree=degree, final_time=0.1, cfl=0.1,
            flux=flux, periodic=True, initial=smooth_profile,
        )
        errors.append(burgers_error(x, coeff, 0.1)["relative_l2"])
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    # The acceptance order is the asymptotic (finest-pair) value; every pair is
    # reported so a pre-asymptotic drop cannot hide behind an average.
    assert orders[-1] >= degree + 0.9, (degree, flux, orders)
    assert min(orders) >= degree + 0.8, (degree, flux, orders)


def test_smooth_regime_matches_verified_p2_benchmark():
    """Cross-check against the already verified burgers_1d P2 baseline."""
    x_new, coeff_new, _ = solve_burgers_shock(
        ncells=64, degree=2, final_time=0.1, cfl=0.1, flux="roe",
        periodic=True, initial=smooth_profile,
    )
    error_new = burgers_error(x_new, coeff_new, 0.1)["relative_l2"]
    reference = 1.82024e-05  # README: smooth Burgers P2, 0.001820 %
    assert abs(error_new - reference) / reference < 5e-3, (error_new, reference)


# ---------------------------------------------------------------------------
# 2. Shock position and its convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flux", ["lax_friedrichs", "roe"])
@pytest.mark.parametrize("limiter", [None, "tvb"])
def test_shock_sits_at_rankine_hugoniot_position(flux, limiter):
    """The discrete mass implies the analytic shock location to round-off."""
    for cells in (50, 100):
        _, coeff, meta = shock_run(cells, flux, limiter)
        assert meta["shock_position_exact"] == pytest.approx(SHOCK, abs=0.0)
        assert meta["shock_position_from_mass_error"] < 1e-8, (cells, flux, limiter)
        assert abs(meta["conservation_defect"]) < 1e-8
        assert meta["shock_position"] == pytest.approx(SHOCK, abs=5e-3)
        # discrete mass == exact mass of the entropy solution (single shock)
        mass = coeff[:, 0].sum().item() * meta["dx"]
        assert mass == pytest.approx(
            DATA.u_left * SHOCK + DATA.u_right * (1.0 - SHOCK), abs=1e-12)


def test_shock_position_error_decays_with_refinement():
    """Wave-front centroid (and level crossing) errors decrease under refinement."""
    centroids, crossings = [], []
    for cells in (50, 100, 200):
        _, _, meta = shock_run(cells)
        centroids.append(meta["shock_position_moment_error"])
        crossings.append(meta["shock_position_error"])
    assert centroids[0] > centroids[1] > centroids[2]
    assert crossings[0] > crossings[1] > crossings[2]
    assert centroids[-1] < 3e-3 and crossings[-1] < 1e-5
    # a captured shock is O(dx) wide, so the centroid converges at first order
    order = math.log(centroids[0] / centroids[2]) / math.log(4.0)
    assert 0.7 < order < 1.3, order


# ---------------------------------------------------------------------------
# 3. Gibbs oscillation bound, before and after limiting
# ---------------------------------------------------------------------------

def test_gibbs_oscillation_is_bounded_by_limiter():
    phi = PHI
    unlimited, limited = {}, {}
    for limiter, store in ((None, unlimited), ("tvb", limited)):
        for cells in (50, 200):
            _, coeff, _ = shock_run(cells, "roe", limiter)
            store[cells] = gibbs_overshoot(coeff, phi, DATA.u_right, DATA.u_left)
    # the exact solution is bounded by [u_right, u_left], so any excursion is Gibbs
    assert min(unlimited.values()) > 0.15
    assert max(limited.values()) < 0.07
    assert min(unlimited.values()) > 3.0 * max(limited.values())
    # unlimited overshoot is mesh independent: that is the Gibbs phenomenon
    assert abs(unlimited[50] - unlimited[200]) / unlimited[50] < 0.05
    # the unlimited solution also carries more total variation than the exact jump
    _, coeff, meta = shock_run(100, "roe", None)
    _, coeff_limited, meta_limited = shock_run(100, "roe", "tvb")
    tv_unlimited = total_variation(coeff, phi, meta["dx"])
    tv_limited = total_variation(coeff_limited, phi, meta_limited["dx"])
    assert tv_limited < tv_unlimited < 3.0


def test_roe_and_rusanov_agree_in_smooth_regime():
    """Both interface fluxes are consistent: same smooth solution, same mass."""
    solutions = {}
    for flux in ("lax_friedrichs", "roe"):
        x, coeff, meta = solve_burgers_shock(
            ncells=64, degree=2, final_time=0.1, cfl=0.1, flux=flux,
            periodic=True, initial=smooth_profile,
        )
        solutions[flux] = coeff
        assert meta["steps"] > 0
        # periodic smooth case: the mean of a cell average is the period mean
        assert abs(coeff[:, 0].mean().item() - 0.2) < 1e-10
    difference = (solutions["roe"] - solutions["lax_friedrichs"]).abs().max().item()
    assert difference < 1e-5, difference


def test_scalar_fluxes_are_consistent_and_upwind_at_a_shock():
    for value in (1.0, -0.4, 0.7):
        state = torch.tensor([[value]], dtype=torch.float64)
        # consistency: both fluxes collapse to the physical flux for equal states
        assert lax_friedrichs_flux(state, state).item() == pytest.approx(
            0.5 * value ** 2, abs=1e-14)
        assert roe_flux(state, state).item() == pytest.approx(
            0.5 * value ** 2, abs=1e-14)
    left = torch.tensor([[1.0]], dtype=torch.float64)
    right = torch.tensor([[0.0]], dtype=torch.float64)
    # scalar Roe is pure upwinding, so for u_left > u_right it returns f(u_left)
    assert roe_flux(left, right).item() == pytest.approx(0.5, abs=1e-14)
    # Lax--Friedrichs adds dissipation on top of the central flux (0.25 here)
    assert lax_friedrichs_flux(left, right).item() > 0.5


# ---------------------------------------------------------------------------
# 4. Negative cases: inadmissible states and invalid configurations are rejected
# ---------------------------------------------------------------------------

def test_negative_density_state_is_rejected():
    with pytest.raises(ValueError, match="negative density state rejected"):
        solve_burgers_tracer(
            ncells=64, degree=2, final_time=0.1, cfl=0.1, data=DATA,
            density=lambda x: x - 0.5,  # negative on the left half
        )
    with pytest.raises(ValueError, match="non-finite"):
        solve_burgers_tracer(
            ncells=64, degree=2, final_time=0.1, cfl=0.1, data=DATA,
            density=lambda x: torch.full_like(x, float("nan")),
        )


def test_positivity_filter_reverts_only_inadmissible_cells():
    phi = PHI
    # Gauss nodes plus the two traces phi_k(-1) = (-1)^k, phi_k(1) = 1
    check_points = torch.cat((phi, torch.tensor([[1.0, -1.0, 1.0]]),
                              torch.ones(1, 3)), dim=0)
    coeff = torch.zeros(2, 3, 1, dtype=torch.float64)
    coeff[0, 0, 0] = 1.0
    coeff[1, 0, 0] = 1.0
    coeff[1, 1, 0] = -3.0  # drives the right cell negative at some nodes
    filtered = positivity_filter(coeff, check_points)
    assert torch.equal(filtered[0], coeff[0])          # admissible cell untouched
    assert filtered[1, 0, 0].item() == pytest.approx(1.0)  # average kept
    assert filtered[1, 1:, 0].abs().max().item() == 0.0    # higher modes removed
    assert torch.einsum("cmv,qm->cqv", filtered, check_points).min().item() > 0.0


def test_rarefaction_data_has_no_shock_and_is_rejected():
    rarefaction = RiemannData(u_left=0.0, u_right=1.0, x0=0.5)
    with pytest.raises(ValueError, match="no shock"):
        shock_expected_position(rarefaction, 0.5)
    # the exact solver still answers, with the correct fan
    grid = torch.tensor([0.5, 0.6, 0.85], dtype=torch.float64)
    values = exact_solution(grid, 0.5, rarefaction)
    assert values.tolist() == pytest.approx([0.0, 0.2, 0.7], abs=1e-12)


@pytest.mark.parametrize("kwargs, match", [
    ({"flux": "hll"}, "unknown flux"),
    ({"degree": 5}, "degree must be"),
    ({"ncells": 1}, "ncells must be"),
])
def test_invalid_configurations_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve_burgers_shock(final_time=0.01, **kwargs)


# ---------------------------------------------------------------------------
# 5. Passive density: conservation and positivity
# ---------------------------------------------------------------------------

def test_tracer_density_is_conserved_and_positive():
    """Periodic smooth velocity: the passive density must be conserved exactly."""
    _, _, density, meta = solve_burgers_tracer(
        ncells=100, degree=2, final_time=0.1, cfl=0.1, flux="roe",
        data=DATA, periodic=True, initial=smooth_profile,
        density=lambda x: 1.0 + 0.5 * torch.sin(2.0 * math.pi * x),
    )
    assert meta["density_min"] > 0.0
    assert abs(meta["density_mass_final"] - meta["density_mass_initial"]) < 1e-12
    assert density[:, 0].mean().item() == pytest.approx(1.0002257, abs=2e-3)