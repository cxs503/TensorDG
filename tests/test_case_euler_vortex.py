"""CASE-EULER-VORTEX conformance harness: reference validity and gate teeth.

The gate is only meaningful if (a) the analytic profile really is an exact
solution of the Euler equations and (b) the gate actually rejects a
configuration that is not converged.  Both are tested here, plus a smoke test
of the record schema and of the aggregation step.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks import case_euler_vortex as cev
from benchmarks import _case_euler_vortex_report as report
from tensordg.euler_1d import GAMMA
from tensordg.euler_2d import vortex_primitive


def test_reference_solution_is_exact():
    """The analytic vortex must satisfy the Euler equations to round-off."""
    record = cev.run_residual()
    assert record["steady_frame_residual"] < 1e-12
    assert record["spacetime_residual"] < 1e-8
    assert max(record["solution_properties"].values()) < 1e-8
    assert record["gate_pass"]


def test_steady_residual_detects_wrong_pressure_law():
    """The residual check must be able to fail: p = rho is not an Euler solution."""
    dtype = torch.float64
    x = torch.linspace(-3.0, 3.0, 65, dtype=dtype).requires_grad_(True)
    y = torch.linspace(-3.0, 3.0, 65, dtype=dtype).requires_grad_(True)
    rho, u, v, _ = vortex_primitive(x, y, 0.0, cev.STRENGTH, (0.0, 0.0))
    p = rho  # deliberately wrong barotropic law
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    flux_x = (rho * u, rho * u ** 2 + p, rho * u * v, u * (energy + p))
    flux_y = (rho * v, rho * u * v, rho * v ** 2 + p, v * (energy + p))
    worst = 0.0
    for component in range(4):
        d_dx = torch.autograd.grad(flux_x[component].sum(), x, retain_graph=True)[0]
        d_dy = torch.autograd.grad(flux_y[component].sum(), y, retain_graph=True)[0]
        worst = max(worst, float((d_dx + d_dy).abs().max().item())
                    / float((d_dx.abs() + d_dy.abs()).max().item()))
    assert worst > 1e-2


def test_gate_rejects_unconverged_p0_mesh_and_accepts_finer_one():
    """Low-order P0 at 32^2 misses 5 %, at 64^2 meets it: the gate discriminates."""
    coarse = cev.run_task({"task": "p0", "cells": 32, "final_time": 1.0, "cfl": 0.4,
                           "strength": cev.STRENGTH})
    fine = cev.run_task({"task": "p0", "cells": 64, "final_time": 1.0, "cfl": 0.4,
                         "strength": cev.STRENGTH})
    assert coarse["max_relative_l2"] > cev.TOLERANCE
    assert not coarse["gate_pass"]
    assert fine["max_relative_l2"] < cev.TOLERANCE
    assert fine["gate_pass"]
    assert fine["max_relative_l2"] < coarse["max_relative_l2"]


def test_dg_record_schema_and_monotone_error():
    """A tiny high-order run must produce the documented record and converge."""
    config = {"task": "dg", "degree": 2, "cells": 8, "final_time": 1.0, "cfl": 0.2,
              "strength": cev.STRENGTH}
    record = cev.run_task(config)
    for key in ("errors", "max_relative_l2", "gate_pass", "steps", "dt", "dofs"):
        assert key in record
    assert set(record["errors"]) == set(cev.FIELDS)
    assert all(value >= 0.0 for value in record["errors"].values())
    assert record["max_relative_l2"] == pytest.approx(max(record["errors"].values()))
    assert record["gate_pass"]
    assert record["steps"] > 0
    # second order: doubling the mesh must not increase the error
    finer = cev.run_task({**config, "cells": 16})
    assert finer["max_relative_l2"] < record["max_relative_l2"]


def test_task_key_is_unique_per_configuration():
    """One key per distinct configuration; identical configs share one key."""
    configs = [{"strength": cev.STRENGTH, **config}
               for config in cev.LADDER + cev.SENSITIVITY + tuple(cev.GATE_SET)]
    signatures = {}
    for config in configs:
        key = cev.task_key(config)
        signature = tuple(sorted((name, value) for name, value in config.items()
                                 if name != "strength"))
        signatures.setdefault(key, set()).add(signature)
    # a key must never stand for two different configurations
    assert all(len(group) == 1 for group in signatures.values())
    # and two different configurations must never collide on one key
    distinct = [next(iter(group)) for group in signatures.values()]
    assert len(set(distinct)) == len(distinct)


def _synthetic_record(task, cells, degree, max_error, final_time=1.0, cfl=0.2):
    errors = {name: max_error / math.sqrt(2.0) for name in cev.FIELDS}
    return {
        "task": task, "cells": cells, "degree": degree, "dx": 10.0 / cells,
        "steps": 10, "dt": 0.01, "errors": errors, "final_time": final_time, "cfl": cfl,
        "strength": cev.STRENGTH, "max_relative_l2": max_error,
        "max_percent": 100.0 * max_error, "gate_pass": max_error < cev.TOLERANCE,
        "wall_seconds": 0.1, "dofs": cells * cells * (degree + 1) ** 2,
    }


def test_aggregate_builds_tables_and_gate(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    fixtures = {
        "dg_d2_n16_t1_cfl0.2": _synthetic_record("dg", 16, 2, 3.0e-4),
        "dg_d2_n32_t1_cfl0.2": _synthetic_record("dg", 32, 2, 3.7e-5),
        "p0_n128_t1_cfl0.4": _synthetic_record("p0", 128, 0, 2.8e-2, cfl=0.4),
        "p0_n32_t1_cfl0.4": _synthetic_record("p0", 32, 0, 7.8e-2, cfl=0.4),
    }
    for key, payload in fixtures.items():
        (results / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    (results / "exact_solution_residual.json").write_text(json.dumps({
        "task": "residual", "steady_frame_residual": 3.2e-16,
        "spacetime_residual": 1.3e-10,
        "solution_properties": {"isentropic_law": 0.0, "divergence_free": 0.0,
                                "radial_balance": 1.1e-15},
        "gate_pass": True,
    }), encoding="utf-8")

    exit_code = report.aggregate(tmp_path, docs_dir=None)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    markdown = (tmp_path / "summary.md").read_text()
    # the two gate configurations are synthetic and pass; the design-order gate
    # is the only remaining requirement
    assert payload["p0_coarse_percent"] == pytest.approx(7.8)
    assert "CASE-EULER-VORTEX" in markdown
    assert "2.8000%" in markdown
    assert exit_code in (0, 1)
    assert payload["gate"][1]["present"]
    # numeric values in the report are injected, never typed
    assert f"{payload['peak_tangential_mach']:.4f}" in markdown