"""Time-step metadata contract shared by the time-marching solvers.

Why this module exists
----------------------
Most solvers here advance with an *adaptive* SSP-RK step

    dt = min(cfl * dx / ((2p+1) * max_speed), final_time - time)

so the ``dt`` stored in the returned ``metadata`` is the **last step actually
taken**, which is very often the truncated remainder step
``final_time - time_previous`` rather than the representative step of the run.
That value is not a run step size: it does not shrink monotonically under CFL
or mesh refinement (it is bounded above by the remainder, which is essentially
uniformly distributed in ``(0, dt_max]``), so any consumer that uses it as a
step — a time axis reconstructed as ``step_index * dt``, a ``dt``-vs-error
curve, a "CFL halving must halve dt" invariant — silently reads a numerical
residue.

The contract implemented here is **additive** (no semantic change of the
existing ``dt`` key, so old consumers keep their old numbers):

============  =============================================================
key           meaning
============  =============================================================
``dt``        unchanged: the last step taken (legacy, may be a remainder)
``dt_last``   the same quantity under an honest name
``dt_mean``   ``final_time / steps``: the representative step of the run
============  =============================================================

``dt_mean`` is the field to use whenever a *step size* is meant.  For a run
with ``steps == 0`` (``final_time == 0``) all three fields are ``0.0``.

``tests/test_time_step_metadata_contract.py`` pins these invariants and
machine-checks which solver modules carry the trio.
"""
from __future__ import annotations


def step_metadata(final_time: float, steps: int, last_step: float) -> dict:
    """Return the ``dt`` / ``dt_last`` / ``dt_mean`` fields for a solver run.

    Parameters
    ----------
    final_time:
        Requested end time of the run (the solver stops at or after it).
    steps:
        Number of time steps actually taken.
    last_step:
        Value of the local ``dt`` after the last iteration (the legacy
        ``meta["dt"]`` value).  Ignored when ``steps == 0``.
    """
    steps = int(steps)
    dt_last = float(last_step) if steps > 0 else 0.0
    dt_mean = float(final_time) / steps if steps > 0 else 0.0
    return {"dt": float(last_step), "dt_last": dt_last, "dt_mean": dt_mean}