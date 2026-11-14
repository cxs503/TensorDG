"""Conservation / positivity gates for the CKS1998 2-D Riemann case.

Anchor: ``REPRO-DG-CKS1998``.  These tests defend the *replacement* gates that
closed the two honest FAILs of ``docs/CASE-DG-CKS1998-2D-RIEMANN.md``:

* the 25.16 % "mass conservation error" was the physical flux through the
  transmissive boundaries of the case, not a limiter defect -- so the gate is
  now the discrete **flux-balance identity** ``Q(t) - Q(0) == boundary flux``;
* the "self-convergence" metric compared total masses of two meshes (a scalar),
  so it measured nothing -- the gate is now a relative L1 distance between the
  coarse mesh and the block-averaged fine mesh.

Every test here is written so that it *fails* if the property it names is
broken: the identities are compared against measured numbers, and the gates are
probed for vacuity (``NOT_RUN`` must not read as ``PASS``).
"""
import numpy as np
import torch

from benchmarks.riemann_2d_cks1998 import (
    GATES,
    block_average,
    judge,
    run_riemann_2d,
    self_convergence_l1,
)
from tensordg.euler_dg_2d import (
    _admissible_cells,
    _positivity_filter,
    apply_tvb_limiter_2d,
    legendre_operators,
)

CONSERVED = ("mass", "x_momentum", "y_momentum", "energy")


def _operators(degree=1):
    return legendre_operators(degree, max(degree + 2, 3))


def _random_state(ncells=6, degree=2, seed=0):
    rng = np.random.default_rng(seed)
    coeff = torch.zeros(
        ncells, ncells, degree + 1, degree + 1, 4, dtype=torch.float64
    )
    for index, (low, high) in enumerate(((1.0, 1.2), (-0.1, 0.1), (-0.1, 0.1), (2.0, 2.4))):
        coeff[..., 0, 0, index] = torch.as_tensor(rng.uniform(low, high, (ncells, ncells)))
    for k in range(degree + 1):
        for l in range(degree + 1):
            if k + l == 0:
                continue
            coeff[..., k, l, :] = torch.as_tensor(
                rng.normal(0.0, 0.05, (ncells, ncells, 4))
            )
    return coeff


# ---------------------------------------------------------------------------
# 1. Where the conservation error does NOT come from
# ---------------------------------------------------------------------------
def test_tvb_limiter_preserves_cell_averages():
    """The limiter must never touch mode (0,0): that is what makes it conservative.

    A discontinuous state guarantees that the troubled-cell branch actually
    fires (otherwise the test would pass vacuously on an already-limited state).
    """
    coeff = _random_state()
    coeff[2:, :, 0, 0, 0] += 1.0            # jump in density between columns 1 and 2
    means_before = coeff[..., 0, 0, :].clone()

    limited = apply_tvb_limiter_2d(coeff, dx=1.0 / 6, dy=1.0 / 6, tvb_constant=0.0)

    assert not torch.equal(limited, coeff), "limiter was inert; test would be vacuous"
    assert torch.equal(limited[..., 0, 0, :], means_before)


def test_positivity_filter_preserves_cell_averages_and_restores_admissibility():
    """Zhang-Shu scaling ``c <- mean + theta (c - mean)`` keeps the average exactly."""
    nodes, _weights, basis, _derivative, signs = _operators()
    coeff = _random_state(degree=1)
    coeff[..., 1, 0, 0] -= 0.8              # negative density at the quadrature nodes
    assert not bool(_admissible_cells(coeff, basis, signs).all()), "state must start inadmissible"

    means_before = coeff[..., 0, 0, :].clone()
    repaired = _positivity_filter(coeff, basis, signs)

    assert torch.equal(repaired[..., 0, 0, :], means_before)
    assert bool(_admissible_cells(repaired, basis, signs).all())


# ---------------------------------------------------------------------------
# 2. The two conservation measurements
# ---------------------------------------------------------------------------
def test_periodic_run_conserves_to_roundoff():
    """No boundary at all: mass, momentum and energy must be conserved exactly."""
    record = run_riemann_2d(
        ncells=8, degree=1, final_time=0.02, cfl=0.1, periodic=(True, True)
    )
    for name in CONSERVED:
        row = record["conservation"][name]
        scale = max(abs(record["conservation"]["mass"]["change"]), 1.0)
        assert row["boundary_flux"] == 0.0, row
        assert abs(row["change"]) / 1.0 < 1e-12, (name, row)


def test_transmissive_flux_balance_identity_closes():
    """``Q(t) - Q(0)`` must equal the time-integrated boundary flux to round-off.

    This is the well-posed replacement for the old "mass change < 1 %" gate: it
    holds for *any* boundary condition, and it is only true because every
    interior interface contributes ``+F_i`` and ``-F_i`` to neighbouring cells.
    """
    record = run_riemann_2d(
        ncells=8, degree=1, final_time=0.02, cfl=0.1, periodic=(False, False)
    )
    mass_change = record["conservation"]["mass"]["change"]
    assert abs(mass_change) > 1e-3, "boundary flux too small: the identity would be vacuous"
    for name in CONSERVED:
        row = record["conservation"][name]
        assert abs(row["change"] - row["boundary_flux"]) < 1e-11 * max(
            abs(record["total_mass"]), abs(row["boundary_flux"]), 1e-3
        ), (name, row)
        assert row["relative_residual"] < GATES["C4_flux_balance_rel"]


def test_boundary_flux_tracking_does_not_change_the_solution():
    """The audit hook must be observation-only (bit-identical trajectory)."""
    base = run_riemann_2d(ncells=8, degree=1, final_time=0.02, cfl=0.1)
    from tensordg.euler_dg_2d import solve_euler_dg_2d
    from benchmarks.riemann_2d_cks1998 import initial_condition

    _x, _y, coeff, _meta = solve_euler_dg_2d(
        initial_condition, ncells=8, degree=1, domain=((0.0, 1.0), (0.0, 1.0)),
        final_time=0.02, cfl=0.1, limiter="tvb", tvb_constant=1.0,
        periodic=(False, False), initial_projection="l2", track_boundary_flux=False,
    )
    assert torch.equal(base["coeff"], coeff)


# ---------------------------------------------------------------------------
# 3. Self-convergence metric
# ---------------------------------------------------------------------------
def test_block_average_and_self_convergence_metric():
    """The metric must be 0 for identical data and > 0 when the meshes differ."""
    coarse = torch.ones(4, 4, dtype=torch.float64)
    fine = torch.ones(8, 8, dtype=torch.float64)
    folded = block_average(fine, 2)
    assert folded.shape == (4, 4)
    assert torch.allclose(folded, coarse)
    assert self_convergence_l1(coarse, fine) == 0.0

    fine = fine.clone()
    fine[0, 0] = 3.0                       # localised blob in the fine mesh only
    assert self_convergence_l1(coarse, fine) > 1e-3


# ---------------------------------------------------------------------------
# 4. Gate non-vacuity
# ---------------------------------------------------------------------------
def _fake_row(**overrides):
    row = {
        "ncells": 32,
        "degree": 1,
        "periodic": (False, False),
        "total_mass": 0.6757,
        "max_flux_balance_residual": 1e-15,
        "finite": True,
        "positivity": {
            "rho_min": 0.1, "p_min": 0.0 + 1e-3,
            "rho_positive": True, "p_positive": True,
            "rho_positive_mean": True, "p_positive_mean": True,
        },
        "conservation": {
            name: {"change": 0.0, "boundary_flux": 0.0, "residual": 0.0,
                   "relative_residual": 1e-15}
            for name in CONSERVED
        },
    }
    row.update(overrides)
    return row


def test_judge_fails_on_a_broken_flux_balance():
    """A row whose flux-balance residual is O(1) must fail C4 (no hardcoded PASS)."""
    bad = _fake_row(max_flux_balance_residual=1.0)
    verdicts = judge([bad], convergence=None)
    assert verdicts["C4_flux_balance"]["passed"] is False
    good = _fake_row()
    assert judge([good], convergence=None)["C4_flux_balance"]["passed"] is True


def test_judge_fails_on_negative_pressure():
    bad = _fake_row(positivity={
        "rho_min": 0.1, "p_min": -1.0,
        "rho_positive": True, "p_positive": False,
        "rho_positive_mean": True, "p_positive_mean": True,
    })
    assert judge([bad])["C1_positivity"]["passed"] is False


def test_judge_reports_missing_checks_as_not_run_not_pass():
    """Absent evidence must never be scored as a pass (gate-vacuity policy)."""
    # no non-periodic row -> the flux-balance gate has nothing to measure
    only_periodic = judge([_fake_row(periodic=(True, True))], convergence=None)
    assert only_periodic["C4_flux_balance"]["check_state"] == "not_run"
    assert only_periodic["C4_flux_balance"]["passed"] is False

    # no periodic row -> the conservation control has nothing to measure
    only_open = judge([_fake_row()], convergence=None)
    assert only_open["C5_periodic_drift"]["check_state"] == "not_run"
    assert only_open["C5_periodic_drift"]["passed"] is False

    # single mesh -> self-convergence has nothing to measure
    assert only_open["C2_self_convergence"]["check_state"] == "not_run"
    assert only_open["C2_self_convergence"]["passed"] is False


def test_judge_scores_self_convergence_against_its_tolerance():
    passing = judge([_fake_row()], convergence={"32_vs_64": 0.01})
    failing = judge([_fake_row()], convergence={"32_vs_64": 0.5})
    assert passing["C2_self_convergence"]["passed"] is True
    assert failing["C2_self_convergence"]["passed"] is False