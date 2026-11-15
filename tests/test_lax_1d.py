import torch

from benchmarks.lax_1d import LEFT, RIGHT
from benchmarks.lax_a0_metric_catalogue import nodal_fields_from_coefficients
from benchmarks.lax_a0_quadrature_audit import measure
from benchmarks.riemann_dg_error import riemann_dg_errors
from benchmarks.riemann_exact import star_state
from tensordg.euler_dg_1d import solve_euler_dg_1d

# Ledger bar for CASE-LAX-DG, read from the registered case module.
TOLERANCE = 0.03
# Converged density reading at the pinned leg p3.N400 (see
# docs/CASE-LAX-DG-a0-metric-catalogue.md and docs/CASE-LAX-DG-a0-quadrature-audit.md).
PINNED_LEG_CONVERGED = 3.0251e-02


def test_lax_star_state_and_short_time_positivity():
    pressure, velocity = star_state(LEFT, RIGHT)
    assert abs(pressure - 2.4660979192073564) < 1e-12
    assert abs(velocity - 1.528723026632884) < 1e-12
    _, coeff, _ = solve_euler_dg_1d(LEFT, RIGHT, ncells=40, degree=2, final_time=0.005)
    assert torch.isfinite(coeff).all()
    average = coeff[:, 0]
    pressure_avg = 0.4 * (average[:, 2] - 0.5 * average[:, 1] ** 2 / average[:, 0])
    assert (average[:, 0] > 0).all()
    assert (pressure_avg > 0).all()


def test_documented_lax_case_is_an_honest_fail_against_the_three_percent_bar():
    """The ledger arm measures the whole-field relative L2 at element Gauss nodes.

    The exact Lax solution has kinks *inside* cells, so an ``order``-point Gauss
    rule is not exact there: ``benchmarks.riemann_dg_error`` (8 nodes) reads
    ``2.906e-02`` -- below the bar -- while the breakpoint-aligned converged
    value is ``3.0251e-02``, i.e. **above** it.  This test therefore asserts the
    converged number and records the honest FAIL instead of certifying a PASS
    that only exists at an under-resolved quadrature
    (``docs/CASE-LAX-DG-a0-metric-catalogue.md``).
    """
    x, coeff, _ = solve_euler_dg_1d(
        LEFT, RIGHT, ncells=400, degree=3, final_time=0.13, cfl=0.35,
    )
    unconverged = riemann_dg_errors(x, coeff, 0.13, LEFT, RIGHT)["density"]["relative_l2"]
    fields = nodal_fields_from_coefficients(x, coeff)
    converged = measure(fields, "aligned", order=20)["l2"]

    # (a) the honest, converged verdict: the bar is missed
    assert converged["density"] > TOLERANCE, converged
    # (b) the number is the one recorded in the case-pool audit (band, not a copy)
    assert abs(converged["density"] - PINNED_LEG_CONVERGED) / PINNED_LEG_CONVERGED < 2e-2
    # (c) velocity / pressure do clear the bar -- only density fails
    assert converged["velocity"] < TOLERANCE
    assert converged["pressure"] < TOLERANCE
    # (d) the under-resolved reading is lower, and that difference is what made
    #     the old "within three percent" assertion vacuous rather than meaningful
    assert unconverged < TOLERANCE
    assert unconverged < converged["density"]
