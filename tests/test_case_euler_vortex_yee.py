"""Guards for the CASE-DG-EULER-VORTEX-ERR harness (fast, deterministic)."""
from __future__ import annotations

import json

import pytest

from benchmarks.case_euler_vortex_yee import (
    CONFIGS,
    GROWTH_LIMIT,
    PERIOD,
    TOLERANCE,
    aggregate,
    run_task,
)


def test_contract_constants():
    assert PERIOD == pytest.approx(10.0)          # L / u_inf with L = 10, u_inf = 1
    assert TOLERANCE == pytest.approx(0.06)       # case-pool tolerance for this round
    assert GROWTH_LIMIT >= 1.0
    ids = [cfg["id"] for cfg in CONFIGS]
    assert len(ids) == len(set(ids)), "configuration ids must be unique"
    assert any(cfg.get("role") == "gate" for cfg in CONFIGS)


def test_small_dg_run_is_accepted():
    """A cheap P1 run must still land inside the 6 % tolerance."""
    record = run_task({"task": "dg", "degree": 1, "cells": 8,
                       "final_time": 0.5, "cfl": 0.2})
    assert record["max_relative_l2"] < TOLERANCE
    assert record["steps"] > 0
    for field in ("density", "velocity_x", "velocity_y", "pressure"):
        assert record["errors"][field] >= 0.0
    assert set(record["errors"]) == {"density", "velocity_x", "velocity_y", "pressure"}


def _fake(role, degree, cells, final_time, cfl, error, config_id):
    return {
        "config_id": config_id, "role": role, "degree": degree, "cells": cells,
        "final_time": final_time, "cfl": cfl, "steps": 10, "dt": 1e-3,
        "wall_seconds": 0.1,
        "errors": {"density": error, "velocity_x": error, "velocity_y": error,
                   "pressure": error},
        "max_relative_l2": error, "max_percent": 100.0 * error,
        "gate_pass": error < TOLERANCE,
    }


def test_aggregate_gate_and_growth(tmp_path):
    """Aggregation must recompute the gate and the growth factor from records."""
    results = tmp_path / "results"
    results.mkdir(parents=True)
    records = [
        _fake("gate", 2, 32, PERIOD, 0.2, 0.001, "dg_d2_n32_t10_cfl0.2"),
        _fake("history", 2, 32, 2.5, 0.2, 0.0005, "dg_d2_n32_t2.5_cfl0.2"),
        _fake("history", 2, 32, 1.0, 0.2, 0.0005, "dg_d2_n32_t1_cfl0.2"),
        _fake("ladder", 2, 16, PERIOD, 0.2, 0.008, "dg_d2_n16_t10_cfl0.2"),
        _fake("cfl", 2, 32, PERIOD, 0.1, 0.0011, "dg_d2_n32_t10_cfl0.1"),
        _fake("baseline", 0, 64, PERIOD, 0.4, 0.05, "p0_n64_t10_cfl0.4"),
    ]
    for record in records:
        (results / f"{record['config_id']}.json").write_text(json.dumps(record))
    docs = tmp_path / "docs"
    assert aggregate(tmp_path, docs) == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["gate_pass"] is True
    assert metrics["growth_reference_time"] == pytest.approx(1.0)
    assert metrics["growth_factor"] == pytest.approx(2.0)
    assert metrics["growth_pass"] is True
    assert metrics["overall_passed"] is True
    order = metrics["observed_orders"]["2"][0]
    assert order["from_cells"] == 16 and order["to_cells"] == 32
    assert order["order"] == pytest.approx(3.0, abs=0.05)
    report = (docs / "CASE-DG-EULER-VORTEX-ERR.md").read_text()
    assert "0.1000%" in report and "6%" in report


def test_aggregate_flags_failure(tmp_path):
    """An config above the tolerance must be reported as a miss, not hidden."""
    results = tmp_path / "results"
    results.mkdir(parents=True)
    records = [
        _fake("gate", 2, 32, PERIOD, 0.2, 0.09, "dg_d2_n32_t10_cfl0.2"),
        _fake("history", 2, 32, 1.0, 0.2, 0.0005, "dg_d2_n32_t1_cfl0.2"),
    ]
    for record in records:
        (results / f"{record['config_id']}.json").write_text(json.dumps(record))
    assert aggregate(tmp_path, tmp_path / "docs") == 0
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["gate_pass"] is False
    assert metrics["overall_passed"] is False
    assert metrics["growth_factor"] > GROWTH_LIMIT
    assert metrics["growth_pass"] is False