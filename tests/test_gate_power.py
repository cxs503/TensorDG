"""Negative controls for the documented 3% tolerances (gate power).

``benchmarks/audit_gate_power.py`` measures how much slack each documented
``< 0.03`` gate carries and writes the whole table to
``validation/gate_power_audit.json``.  The full sweep is deliberately *not* run
here (the documented Sod/Lax cases alone cost ~2 minutes); these tests keep the
CI-cheap core of the argument, i.e. they prove the gates and the error metrics
can fail:

  * the 1-D advection gate breaks when the same case is under-resolved by 8x;
  * the SIPG (heat) gate breaks when the penalty is forced below the stability
    threshold -- an implementation defect, not a resolution effect;
  * the error metric is locked to the compared time: judging a solution at
    t = 0.25 against the exact solution at t = 0.375 costs O(1) error, so a
    passing gate is evidence about the right problem and not about any problem.

The numbers asserted here are the ones recorded in the audit artifacts, and the
tolerances are one-sided (``>``) so that they hold under any BLAS/hardware
round-off.  The sabotage control treats every failure mode of a non-positive
discrete operator (garbage blow-up, ``inf``/``nan``, or a raised linear-algebra
error) as "broken", because which of them a given torch/BLAS build reports is
an implementation detail of the solver stack, not of the audit.
"""
import math

import torch

from benchmarks.audit_gate_power import GATE, run_advection_1d, run_heat_1d

# A sabotaged operator must lose at least ten orders of magnitude of accuracy.
# The documented run measures ~1e-5, so this bound is 15 orders below the
# measurement and only a "the control has no teeth" outcome can fall under it.
SABOTAGE_FLOOR = 1e10


def _sabotaged_heat_error(cells=16, penalty=0.2):
    """Relative L2 error of the sabotaged SIPG path, solver-failure tolerant.

    Forcing the penalty below the stability threshold makes the discrete
    operator lose positivity.  How that surfaces is backend dependent: on this
    machine the solve amplifies round-off to ~1e57, other BLAS builds return
    ``inf``/``nan``, and builds that detect the singularity raise
    ``torch._C._LinAlgError`` (a ``RuntimeError``).  All of these mean the same
    thing for the audit -- the gate was broken -- so they are normalised to
    ``inf`` instead of turning a genuine negative control into an error.
    """
    try:
        value = float(run_heat_1d(cells, penalty=penalty)["u"])
    except (RuntimeError, ArithmeticError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def test_one_dimensional_advection_gate_breaks_when_under_resolved():
    """Resolution negative control: 40 cells pass, 5 cells (8x coarser) fail."""
    documented = run_advection_1d(40)["u"]
    coarsened = run_advection_1d(5)["u"]
    assert documented < GATE, documented
    assert coarsened > GATE, coarsened
    # The 3% gate tolerates a 20x error regression at the documented resolution,
    # which is exactly the slack this audit quantifies.
    assert GATE / documented > 10.0, documented


def test_sipg_penalty_below_stability_threshold_breaks_the_heat_gate():
    """Implementation negative control: penalty sabotage on the Crank-Nicolson SIPG path."""
    healthy = run_heat_1d(16, penalty=20.0)["u"]
    degraded = run_heat_1d(16, penalty=1.0)["u"]
    sabotaged = _sabotaged_heat_error()
    assert healthy < GATE, healthy
    # A 20x penalty reduction is still stable (the gate is not penalty-brittle)...
    assert degraded < GATE, degraded
    # ...but a 100x reduction loses positivity of the discrete operator and the
    # linear solve blows up (or diverges to inf/nan, or is rejected by the
    # solver) by many orders of magnitude.
    assert sabotaged >= SABOTAGE_FLOOR, sabotaged
    assert sabotaged > healthy, sabotaged


def test_advection_error_metric_is_locked_to_the_compared_time():
    """Metric negative control: comparing against the wrong time must not pass."""
    locked = run_advection_1d(40, final_time=0.25, compare_time=0.25)["u"]
    shifted = run_advection_1d(40, final_time=0.25, compare_time=0.375)["u"]
    assert locked < GATE, locked
    assert shifted > 100.0 * locked, (shifted, locked)
    assert shifted > GATE, shifted


def test_audit_registry_documented_resolutions_match_the_committed_gates():
    """Guard against drift: the registry must reproduce the committed settings.

    ``tests/test_advection_2d.py`` pins 24 cells / degree 1 / t = 1.0 and
    ``tests/test_poisson_2d.py`` pins 16 cells / degree 1; if someone retunes a
    documented case, the audit would silently measure a different gate.
    """
    from benchmarks.audit_gate_power import CASES

    assert CASES["advection_2d_p1"][1] == 24
    assert CASES["advection_2d_p1"][3].get("degree", 1) == 1
    assert CASES["poisson_2d_p1"][1] == 16
    assert CASES["heat_1d_p2"][1] == 16
    assert CASES["lax_p3"][1] == 400
    assert CASES["sod_p0"][1] == 3200
    assert CASES["vortex_p0"][1] == 192
    assert CASES["advection_1d_p1"][1] == 40
    assert GATE == 0.03
    assert torch.is_tensor(torch.zeros(1))  # torch import is used by the runners