"""Tests for CASE-DG-EULER-VORTEX-ERR (error conservation under transport).

The point of these tests is not "the script ran" but the *invariants* the case
claims:

* the exact vortex really is isentropic and momentum-consistent, so the
  entropy functional ``int p rho^-gamma`` has the closed value ``box area``;
* the discrete mass / momentum / energy of both families survive transport at
  round-off (periodic interface fluxes telescope) -- a hard, falsifiable
  identity, not a tolerance chosen to fit the output;
* the C2 gate *discriminates*: a synthetic error history that accumulates like
  ``t^1`` must fail it while a conserved one passes (anti-vacuity check on the
  gate logic itself).
"""
from __future__ import annotations

import math

import pytest

from benchmarks import case_euler_vortex_errconserve as ec


# ---------------------------------------------------------------------------
# exact solution
# ---------------------------------------------------------------------------

def test_exact_vortex_is_isentropic_and_carries_uniform_momentum():
    exact = ec.exact_invariants(points=240)
    # isentropic -> S = int p rho^-gamma = area of the box
    assert exact["entropy"] == pytest.approx(ec.BOX_AREA, rel=1e-11)
    # density deficit: mass below the uniform-flow mass (rho_inf * area)
    assert exact["mass"] < ec.BOX_AREA
    # the swirl has zero net circulation, so <u> = u_inf = 1 exactly
    assert exact["momentum_x"] / exact["mass"] == pytest.approx(1.0, rel=1e-11)
    assert exact["momentum_y"] / exact["mass"] == pytest.approx(1.0, rel=1e-11)
    # the sampled grid minimum must agree with an independent fine radial sample
    assert exact["density_min"] == pytest.approx(ec.exact_core_density(), rel=2e-3)


def test_exact_invariants_converge_with_quadrature():
    coarse = ec.exact_invariants(points=80)
    fine = ec.exact_invariants(points=320)
    for name in ("mass", "energy", "entropy"):
        assert abs(coarse[name] - fine[name]) / abs(fine[name]) < 1e-12


# ---------------------------------------------------------------------------
# discrete conservation (C3) on cheap meshes
# ---------------------------------------------------------------------------

def test_dg_transport_conserves_mass_momentum_energy_at_round_off():
    record = ec.run_task({"task": "dg", "degree": 1, "cells": 8,
                          "final_time": 0.2, "cfl": 0.2})
    for name in ("mass", "momentum_x", "momentum_y", "energy"):
        assert record["invariant_drift"][name] < 1e-12
    # and the transport error itself is already far below the case tolerance
    assert record["max_relative_l2"] < ec.TOLERANCE


def test_p0_transport_conserves_mass_momentum_energy_at_round_off():
    record = ec.run_task({"task": "p0", "cells": 16,
                          "final_time": 0.2, "cfl": 0.4})
    for name in ("mass", "momentum_x", "momentum_y", "energy"):
        assert record["invariant_drift"][name] < 1e-12
    # dissipation is visible in the entropy budget of the low-order family
    assert record["entropy_drift"] > 0.0


def test_invariant_reference_is_the_discrete_initial_state():
    """Both families must reference their *own* t=0 discrete state, not the mesh."""
    dg_record = ec.run_task({"task": "dg", "degree": 2, "cells": 8,
                             "final_time": 0.2, "cfl": 0.2})
    p0_record = ec.run_task({"task": "p0", "cells": 16,
                             "final_time": 0.2, "cfl": 0.4})
    # the L2 projection and the cell-average projection differ on the same field,
    # so the two masses must not be identical; each is self-consistent.
    assert dg_record["initial"]["invariants"]["mass"] != \
        p0_record["initial"]["invariants"]["mass"]
    for record in (dg_record, p0_record):
        change = record["final"]["invariants"]["mass"] - \
            record["initial"]["invariants"]["mass"]
        assert abs(change) < 1e-12 * abs(record["initial"]["invariants"]["mass"])


# ---------------------------------------------------------------------------
# gate logic (C1/C2/C3/C4) -- must reject bad evidence
# ---------------------------------------------------------------------------

def _synthetic_record(family, time, error, drift=1e-15, entropy=1e-6):
    return {
        "task_key": f"{family}_t{time:g}",
        "family": family,
        "config": {},
        "final_time": time,
        "cfl": 0.2,
        "steps": 1,
        "errors": {"density": error, "velocity_x": error * 0.5,
                   "velocity_y": error * 0.5, "pressure": error * 0.5},
        "max_relative_l2": error,
        "max_percent": 100.0 * error,
        "invariant_drift": {"mass": drift, "momentum_x": drift,
                            "momentum_y": drift, "energy": drift, "entropy": entropy},
        "entropy_drift": entropy,
        "entropy_production": -entropy,
        "density_min_relative_error": 0.0,
        "gate_pass": error < ec.TOLERANCE,
        "wall_seconds": 0.0,
    }


def test_c2_gate_rejects_secular_error_growth():
    exact = ec.exact_invariants(points=80)
    accumulating = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
                    for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(accumulating, exact)["gates"]
    assert gates["C2_error_conservation"]["exponent"] == pytest.approx(1.0, abs=1e-6)
    assert gates["C2_error_conservation"]["passed"] is False
    assert gates["all_passed"] is False


def test_c2_gate_accepts_conserved_error():
    exact = ec.exact_invariants(points=80)
    conserved = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * (1.0 + 0.01 * math.log(t)))
                 for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(conserved, exact)["gates"]
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["C3_discrete_conservation"]["passed"] is True
    assert gates["C1_tolerance"]["passed"] is True
    assert gates["C4_entropy_budget"]["passed"] is True
    assert gates["all_passed"] is True


def test_c3_gate_rejects_a_leaking_scheme():
    exact = ec.exact_invariants(points=80)
    leaking = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4, drift=1e-3)
               for t in (1.0, 2.0)]
    gates = ec.evaluate_gates(leaking, exact)["gates"]
    assert gates["C3_discrete_conservation"]["passed"] is False


def test_c1_gate_rejects_a_checkpoint_above_tolerance():
    exact = ec.exact_invariants(points=80)
    bad = [_synthetic_record(ec.GATE_FAMILY, 1.0, 1e-4),
           _synthetic_record(ec.GATE_FAMILY, 20.0, 0.10)]
    gates = ec.evaluate_gates(bad, exact)["gates"]
    assert gates["C1_tolerance"]["passed"] is False


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def test_shards_partition_the_ladder():
    for total in (1, 3, 5, len(ec.LADDER)):
        shards = [ec.select_shard(index, total) for index in range(total)]
        flat = [config for shard in shards for config in shard]
        assert len(flat) == len(ec.LADDER)
        assert sorted(map(str, flat)) == sorted(map(str, ec.LADDER))


def test_aggregate_writes_artifacts(tmp_path):
    records = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
               for t in (1.0, 2.0, 5.0)]
    exact = ec.exact_invariants(points=80)
    metrics = ec.aggregate(records, exact, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "error_conservation.csv").exists()
    assert (tmp_path / "summary.md").exists()
    text = (tmp_path / "summary.md").read_text()
    assert "C2 误差守恒" in text and "未通过" in text
    assert metrics["gates"]["C2_error_conservation"]["passed"] is False
    assert math.isnan(metrics["gates"]["C4_entropy_budget"]["measured"]) is False


def test_figures_are_produced(tmp_path):
    records = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
               for t in (1.0, 2.0, 5.0)]
    figures = ec.make_figures(records, tmp_path)
    assert {f["name"] for f in figures} == {"fig_error_vs_time.png",
                                            "fig_invariants_vs_time.png"}
    for figure in figures:
        assert (tmp_path / figure["name"]).stat().st_size > 2000