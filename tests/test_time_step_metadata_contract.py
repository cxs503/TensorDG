"""The time-step metadata contract of the time-marching solvers.

Anchor: ``CASE-LAX-DG`` — the verification debt recorded in the m18266 report
(``docs/CASE-LAX-DG.md`` §⑤.1): the adaptive solvers return in ``metadata``

    dt = min(cfl * dx / ((2p+1) * max_speed), final_time - time)

evaluated at the **last** iteration, i.e. usually the truncated remainder step.
That number is *not* a step size: it does not shrink monotonically under CFL or
mesh refinement, so `dt`-vs-error curves, time axes reconstructed as
``step_index * dt`` and "CFL halving must halve dt" invariants all read a
residue.

The contract (``tensordg/stepping.py``) is additive:

* ``dt``      — unchanged legacy value = last step taken
* ``dt_last`` — same quantity, honest name
* ``dt_mean`` — ``final_time / steps``, the representative step of the run

These tests pin (a) the helper arithmetic, (b) which solver modules carry the
trio (source scan, so a new solver cannot silently miss it), (c) the runtime
invariants on every solver that carries it, and (d) a measured counterexample
showing the legacy field cannot be used as a step size.
"""
from __future__ import annotations

import importlib
import inspect
import math
import re

import pytest
import torch

from tensordg.stepping import step_metadata

#: The clamp that turns the local ``dt`` into a remainder of ``final_time``.
RESIDUAL_CLAMP = re.compile(r"final_time\s*-\s*time")

#: Modules whose stepping loop carries that clamp (source scan below re-derives
#: this set, so a new solver cannot be added without updating the registry).
RESIDUAL_MODULES = (
    "burgers_1d",
    "burgers_shock_1d",
    "euler_1d",
    "euler_2d",
    "euler_dg_1d",
    "euler_dg_2d",
    "ns_dg_2d",
    "swe_1d",
    "swe_2d",
)

#: Residual modules that do **not** carry the trio yet.  Each entry needs a
#: reason and an owner; the set is allowed to shrink, never to grow.
PENDING = {
    "euler_dg_1d": (
        "fix in flight on branch agent/m18266 (commit 9227c42) touching the same "
        "file; duplicating it here would create a merge conflict"
    ),
}


def _source(module_name: str) -> str:
    return inspect.getsource(importlib.import_module(f"tensordg.{module_name}"))


def _carries_trio(module_name: str) -> bool:
    return "step_metadata(" in _source(module_name)


def _meta(out):
    """Last dict in a solver return tuple (earlier dicts are face data)."""
    dicts = [item for item in out if isinstance(item, dict)] if isinstance(out, tuple) else []
    assert dicts, "solver returned no metadata dict"
    return dicts[-1]


# ---------------------------------------------------------------------------
# (a) helper arithmetic
# ---------------------------------------------------------------------------
def test_helper_reports_mean_and_last_step():
    fields = step_metadata(final_time=0.13, steps=10, last_step=0.0031)
    assert fields["dt"] == 0.0031  # legacy value untouched
    assert fields["dt_last"] == 0.0031
    assert math.isclose(fields["dt_mean"], 0.013, rel_tol=1e-15)


def test_helper_zero_steps_returns_three_zeros():
    fields = step_metadata(final_time=0.0, steps=0, last_step=0.0)
    assert fields == {"dt": 0.0, "dt_last": 0.0, "dt_mean": 0.0}


def test_helper_ignores_a_stale_last_step_when_no_step_was_taken():
    fields = step_metadata(final_time=0.0, steps=0, last_step=1e-3)
    assert fields["dt_last"] == 0.0 and fields["dt_mean"] == 0.0


# ---------------------------------------------------------------------------
# (b) source-level coverage (drift guard)
# ---------------------------------------------------------------------------
def test_residual_module_registry_matches_the_source():
    """Every module with the ``final_time - time`` clamp must be registered."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tensordg"
    found = set()
    for path in sorted(root.glob("*.py")):
        text = path.read_text()
        # ``stepping`` *defines* the keys but never takes a step; modules such
        # as ``kovasznay`` clamp a step but do not publish a ``dt`` field.
        if path.stem == "stepping":
            continue
        publishes_dt = '"dt":' in text or "step_metadata(" in text
        if publishes_dt and RESIDUAL_CLAMP.search(text):
            found.add(path.stem)
    assert found == set(RESIDUAL_MODULES), (
        "residual-dt solver registry is stale; found="
        f"{sorted(found)} registered={sorted(RESIDUAL_MODULES)}"
    )


def test_every_registered_residual_module_carries_the_trio_unless_pending():
    missing = [
        name for name in RESIDUAL_MODULES
        if name not in PENDING and not _carries_trio(name)
    ]
    assert not missing, f"residual solvers without dt_last/dt_mean: {missing}"


def test_pending_set_is_a_subset_of_the_registry():
    """The pending list may shrink; it must not name modules that do not exist."""
    assert set(PENDING) <= set(RESIDUAL_MODULES)


# ---------------------------------------------------------------------------
# (c) runtime invariants for every solver that carries the trio
# ---------------------------------------------------------------------------
def _smoke_burgers():
    from tensordg.burgers_1d import solve_burgers
    return solve_burgers(ncells=16, degree=1, final_time=0.05, cfl=0.2), 0.05


def _smoke_burgers_shock():
    from tensordg.burgers_shock_1d import solve_burgers_shock
    return solve_burgers_shock(ncells=32, degree=1, final_time=0.05), 0.05


def _smoke_burgers_tracer():
    from tensordg.burgers_shock_1d import solve_burgers_tracer
    return solve_burgers_tracer(ncells=32, degree=1, final_time=0.05), 0.05


def _smoke_euler_1d():
    from tensordg.euler_1d import solve_riemann
    return solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=40,
                         final_time=0.05, cfl=0.3), 0.05


def _smoke_euler_2d():
    from tensordg.euler_2d import solve_euler_2d
    out = solve_euler_2d(
        lambda x, y: (1.0 + 0.05 * torch.sin(2 * torch.pi * x), 0.0 * x, 0.0 * x,
                      1.0 + 0.0 * x),
        ncells=8, final_time=0.05, cfl=0.3,
        domain=((0.0, 1.0), (0.0, 1.0)), quadrature=3,
    )
    return out, 0.05


def _smoke_euler_dg_2d():
    from tensordg.euler_dg_2d import solve_euler_dg_2d_sod
    return solve_euler_dg_2d_sod(ncells=16, degree=1, final_time=0.02), 0.02


def _smoke_ns_dg_2d():
    from tensordg.ns_dg_2d import solve_navier_stokes_dg_2d
    out = solve_navier_stokes_dg_2d(
        lambda x, y: (1.0 + 0.0 * x, 0.0 * x, 0.0 * x, 1.0 + 0.0 * x),
        ncells=6, degree=1, final_time=0.02, mu=0.01, cfl=0.05,
    )
    return out, 0.02


def _smoke_swe_1d():
    from tensordg.swe_1d import solve_swe_1d
    out = solve_swe_1d(
        lambda x: 1.0 + 0.1 * torch.sin(2 * torch.pi * x),
        lambda x: torch.zeros_like(x), lambda x: torch.zeros_like(x),
        ncells=16, degree=1, final_time=0.02,
    )
    return out, 0.02


def _smoke_swe_2d():
    from tensordg.swe_2d import solve_swe_2d
    out = solve_swe_2d(
        lambda x, y: 1.0 + 0.05 * torch.sin(2 * torch.pi * x),
        lambda x, y: torch.zeros_like(x), lambda x, y: torch.zeros_like(x),
        lambda x, y: torch.zeros_like(x),
        nx=8, ny=8, degree=1, final_time=0.02,
    )
    return out, 0.02


SOLVER_SMOKE = {
    "burgers_1d": _smoke_burgers,
    "burgers_shock_1d": _smoke_burgers_shock,
    "burgers_shock_1d.tracer": _smoke_burgers_tracer,
    "euler_1d": _smoke_euler_1d,
    "euler_2d": _smoke_euler_2d,
    "euler_dg_2d": _smoke_euler_dg_2d,
    "ns_dg_2d": _smoke_ns_dg_2d,
    "swe_1d": _smoke_swe_1d,
    "swe_2d": _smoke_swe_2d,
}


@pytest.mark.parametrize("name", sorted(SOLVER_SMOKE))
def test_solver_metadata_satisfies_the_contract(name):
    out, final_time = SOLVER_SMOKE[name]()
    meta = _meta(out)
    for key in ("dt", "dt_last", "dt_mean"):
        assert key in meta, f"{name}: metadata is missing {key!r}"

    steps = meta["steps"]
    assert steps >= 1, f"{name}: smoke run took no step"
    assert meta["dt"] == pytest.approx(meta["dt_last"], rel=1e-15, abs=0.0)
    assert meta["dt_last"] <= meta["dt_mean"] * (1.0 + 1e-12), (
        f"{name}: last step {meta['dt_last']} exceeds the mean step {meta['dt_mean']}"
    )
    assert meta["dt_mean"] > 0.0
    # dt_mean is *defined* as final_time / steps: the identity must hold.
    assert meta["dt_mean"] == pytest.approx(final_time / steps, rel=1e-12)
    # the run must reach the requested time, otherwise the identity is vacuous
    assert meta["dt_mean"] * steps == pytest.approx(final_time, rel=1e-12)


def test_zero_final_time_returns_three_zeros():
    from tensordg.burgers_1d import solve_burgers
    _, _, meta = solve_burgers(ncells=8, degree=1, final_time=0.0)
    assert meta["steps"] == 0
    assert (meta["dt"], meta["dt_last"], meta["dt_mean"]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# (d) the measured counterexample: legacy dt is a residue, dt_mean is the step
# ---------------------------------------------------------------------------
def test_cfl_ladder_halves_dt_mean_but_not_the_last_step():
    """On a CFL ladder the representative step halves; the last step does not.

    ``dt_mean`` must track the ladder (ratio ~2 per halving).  ``dt_last`` is a
    truncated remainder, so at least one pair of rungs must jump by a factor
    far from 2 — that divergence is exactly why the legacy field is unusable as
    a step size.
    """
    from tensordg.euler_1d import solve_riemann

    cf_ladder = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    means, lasts = [], []
    for cfl in cf_ladder:
        _, _, meta = solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=100,
                                   final_time=0.05, cfl=cfl)
        means.append(meta["dt_mean"])
        lasts.append(meta["dt_last"])

    step_ratios = [a / b for a, b in zip(means, means[1:])]
    assert all(1.8 <= ratio <= 2.2 for ratio in step_ratios), step_ratios

    last_ratios = [a / b for a, b in zip(lasts, lasts[1:])]
    worst = max(abs(math.log2(ratio)) for ratio in last_ratios)
    assert worst > 2.0, (
        "the legacy dt behaved like a step size on every rung; "
        f"ratios={last_ratios} — recheck the contract documentation"
    )