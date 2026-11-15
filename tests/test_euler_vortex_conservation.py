"""CASE-DG-EULER-VORTEX-ERR: Conservation verification for the isentropic vortex.

For the Euler equations on a periodic domain, a conservative numerical flux
(HLLC for DG, Rusanov for P0) must preserve the domain-integral of the four
conserved quantities (mass, x-momentum, y-momentum, energy) to machine
precision.  This test verifies that both the high-order DG path and the P0
baseline satisfy this invariant.

Reference: the conservation property follows from the divergence theorem:
  d/dt int_Omega U dV = - int_dOmega F·n dS = 0  (periodic BCs)
so the discrete scheme must satisfy sum_i U_i |Omega_i| = const.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from benchmarks.case_euler_vortex import (
    CONSERVATION_NAMES,
    CONSERVATION_TOL,
    conservation_drift_from_totals,
    conservation_summary,
    conservation_totals,
    run_dg,
    run_p0,
)

# Machine-precision tolerance for conservation drift (float64).
# The DG scheme uses HLLC (exactly conservative) and the P0 scheme uses
# Rusanov (also exactly conservative); with periodic BCs the net boundary
# flux is zero, so drift should be at the level of round-off accumulation.
# The value itself lives in the driver (benchmarks/case_euler_vortex.py) so the
# tests and the published study cannot quote two different tolerances.


@pytest.fixture(autouse=True, scope="module")
def _single_threaded():
    """One BLAS/ATen thread per process.

    The study is a ladder of small meshes; multi-threading oversubscribes and
    costs more than it saves (measured: DG P2 32x32 111 s -> 39 s), and a fixed
    thread count also keeps the drift comparison free of reduction-order
    variation between runs.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _conservation_pass(record: dict) -> tuple[bool, dict]:
    """Check whether all conservation drifts are below tolerance."""
    drift = record.get("conservation_drift", {})
    results = {}
    for name in CONSERVATION_NAMES:
        value = drift.get(name, float("nan"))
        results[name] = value < CONSERVATION_TOL if not math.isnan(value) else False
    all_pass = all(results.values())
    return all_pass, results


class TestDGConservation:
    """High-order DG (HLLC flux) must conserve to machine precision."""

    @pytest.mark.parametrize("degree,cells", [(1, 8), (2, 8), (3, 6)])
    def test_dg_conservation_periodic(self, degree, cells):
        config = {
            "task": "dg",
            "degree": degree,
            "cells": cells,
            "final_time": 1.0,
            "cfl": 0.2,
        }
        record = run_dg(config)
        passed, details = _conservation_pass(record)
        assert passed, (
            f"DG degree={degree} cells={cells} conservation drift exceeds "
            f"{CONSERVATION_TOL:.0e}: {details}"
        )

    def test_dg_conservation_with_limiter(self):
        """Even with a limiter (which preserves cell averages), conservation holds."""
        from tensordg.euler_dg_2d import solve_euler_dg_2d_vortex

        x, y, coeff, meta = solve_euler_dg_2d_vortex(
            ncells=8, degree=2, final_time=0.5, cfl=0.2, limiter="tvb", tvb_constant=10.0
        )
        drift = meta.get("conservation_drift", {})
        for name in CONSERVATION_NAMES:
            value = drift.get(name, float("nan"))
            assert not math.isnan(value), f"Missing conservation drift for {name}"
            assert value < CONSERVATION_TOL, (
                f"DG+limiter conservation drift for {name} = {value:.2e} > {CONSERVATION_TOL:.0e}"
            )


class TestP0Conservation:
    """P0 finite volume (Rusanov flux) must conserve to machine precision."""

    @pytest.mark.parametrize("cells", [16, 32])
    def test_p0_conservation_periodic(self, cells):
        config = {
            "task": "p0",
            "cells": cells,
            "final_time": 1.0,
            "cfl": 0.4,
        }
        record = run_p0(config)
        passed, details = _conservation_pass(record)
        assert passed, (
            f"P0 cells={cells} conservation drift exceeds "
            f"{CONSERVATION_TOL:.0e}: {details}"
        )


class TestRecordSchema:
    """Both study entry points must expose the documented record fields.

    Regression: ``max_relative_l2`` was only attached by ``run_task``, so the
    conservation tests (which call ``run_p0`` / ``run_dg`` directly) died with
    ``KeyError`` on the coarse-mesh control even though the scheme conserved.
    """

    @pytest.mark.parametrize("runner,config", [
        (run_p0, {"task": "p0", "cells": 16, "final_time": 0.5, "cfl": 0.4}),
        (run_dg, {"task": "dg", "degree": 1, "cells": 8, "final_time": 0.5, "cfl": 0.2}),
    ])
    def test_direct_runner_record_is_complete(self, runner, config):
        record = runner(config)
        for key in ("max_relative_l2", "max_percent", "gate_pass",
                    "conservation_drift", "max_conservation_drift",
                    "conservation_tolerance", "conservation_pass",
                    "max_step_drift"):
            assert key in record, f"{key} missing from {runner.__name__} record"
        assert record["max_relative_l2"] == max(record["errors"].values())
        assert math.isclose(record["max_percent"], 100.0 * record["max_relative_l2"])
        assert record["conservation_tolerance"] == CONSERVATION_TOL


class TestPerStepCeiling:
    """The audit must bound the drift at *every* step, not only at the end.

    An endpoint-only measurement cannot separate "conserved throughout" from
    "excursion that decayed before the final time".  The P0 path samples the
    domain integrals once per step, so the ceiling is available there.
    """

    def test_p0_step_ceiling_is_machine_precision(self):
        record = run_p0({"task": "p0", "cells": 16, "final_time": 1.0, "cfl": 0.4})
        ceiling = record["max_step_drift"]
        assert not math.isnan(ceiling), "per-step ceiling not recorded"
        assert ceiling < CONSERVATION_TOL, (
            f"per-step drift ceiling {ceiling:.2e} exceeds {CONSERVATION_TOL:.0e}"
        )
        # The ceiling is a maximum over a superset that contains the endpoint,
        # so it can never be smaller than the endpoint drift.
        assert ceiling >= record["max_conservation_drift"] * (1.0 - 1e-12)
        assert record["max_step_drift_step"] >= 1

    def test_dg_path_declares_missing_ceiling(self):
        """No per-step hook exists on the DG path: say NaN, do not fake it."""
        record = run_dg({"task": "dg", "degree": 1, "cells": 8, "final_time": 0.5, "cfl": 0.2})
        assert math.isnan(record["max_step_drift"])
        assert record["max_step_drift_step"] == -1


class TestAuditResolution:
    """Negative control: does the audit actually *detect* a violation?

    A verification suite that only ever observes ``1e-15`` cannot distinguish
    "the scheme conserves" from "the metric is blind".  These tests inject a
    known relative perturbation into one cell of an otherwise uniform state and
    require the audit to report the injected value back (within a factor 2) and
    to fail its own tolerance verdict.
    """

    @staticmethod
    def _uniform_state(cells: int = 8) -> torch.Tensor:
        """Cell averages with unit density/momentum/energy in every cell."""
        return torch.ones(cells, cells, 4, dtype=torch.float64)

    @pytest.mark.parametrize("delta", [1e-3, 1e-6, 1e-10])
    def test_injected_mass_leak_is_measured(self, delta):
        cells = 8
        state = self._uniform_state(cells)
        dx = dy = 1.0
        initial = conservation_totals(state, dx, dy)
        leaking = state.clone()
        # Perturb *one* cell of cells^2 so the perturbation of the domain
        # integral is exactly ``delta``: localised leaks are the ones a global
        # audit is most likely to miss.
        leaking[0, 0, 0] *= 1.0 + delta * cells * cells
        final = conservation_totals(leaking, dx, dy)
        drift = conservation_drift_from_totals(initial, final)
        assert 0.5 * delta <= drift["mass"] <= 2.0 * delta, (
            f"audit reported {drift['mass']:.3e} for an injected {delta:.0e} leak"
        )
        # A mass-only perturbation leaves the other three invariants untouched.
        for name in CONSERVATION_NAMES[1:]:
            assert drift[name] < 1e-18, f"{name} reacted to a mass-only leak"
        summary = conservation_summary(drift)
        assert summary["conservation_pass"] is (delta < CONSERVATION_TOL)

    def test_tolerance_line_separates_roundoff_from_leaks(self):
        """1e-13 passes, 1e-11 does not: the threshold is not vacuous."""
        cells = 8
        drift_clean = conservation_summary({name: 1e-15 for name in CONSERVATION_NAMES})
        drift_noise = conservation_summary({name: 1e-13 for name in CONSERVATION_NAMES})
        drift_leak = conservation_summary({name: 1e-11 for name in CONSERVATION_NAMES})
        assert drift_clean["conservation_pass"] and drift_noise["conservation_pass"]
        assert not drift_leak["conservation_pass"]
        assert math.isclose(drift_leak["max_conservation_drift"], 1e-11)

    def test_nan_drift_fails_the_verdict(self):
        summary = conservation_summary({name: float("nan") for name in CONSERVATION_NAMES})
        assert summary["conservation_pass"] is False
        assert math.isnan(summary["max_conservation_drift"])


class TestPublishedArtifacts:
    """The artifacts behind the report must be strict JSON and agree with it.

    The report is generated from these files, so the two are only allowed to
    disagree if someone typed a number into the markdown by hand.
    """

    ROOT = Path(__file__).resolve().parents[1]

    @staticmethod
    def _strict_loads(text: str):
        def reject(const):
            raise ValueError(f"non-JSON constant {const!r} in artifact")
        return json.loads(text, parse_constant=reject)

    @pytest.mark.parametrize("rel", [
        "case_euler_vortex/conservation_results.json",
        "case_euler_vortex/audit_resolution.json",
    ])
    def test_artifact_is_strict_json(self, rel):
        path = self.ROOT / rel
        assert path.is_file(), f"missing published artifact {rel}"
        payload = self._strict_loads(path.read_text())
        assert payload, f"{rel} is empty"

    def test_report_numbers_come_from_the_artifact(self):
        records = self._strict_loads(
            (self.ROOT / "case_euler_vortex/conservation_results.json").read_text()
        )
        doc = (self.ROOT / "docs/case_dg_euler_vortex_conservation.md").read_text()
        for record in records:
            field = f"{record['max_conservation_drift']:.2e}"
            assert field in doc, (
                f"{record['config_label']}: drift {field} from the artifact is not in the report"
            )
            assert f"{record['max_percent']:.4f}%" in doc, (
                f"{record['config_label']}: solution error missing from the report"
            )

    def test_negative_control_is_sensitive_at_one_permille(self):
        rows = self._strict_loads(
            (self.ROOT / "case_euler_vortex/audit_resolution.json").read_text()
        )
        for row in rows:
            delta = row["injected_relative_leak"]
            assert 0.5 * delta < row["measured_mass_drift"] < 2.0 * delta
            assert row["conservation_pass"] is (row["measured_mass_drift"] < CONSERVATION_TOL)


class TestConservationVsSolutionError:
    """Conservation drift is independent of solution error magnitude."""

    def test_coarse_mesh_conservation_still_exact(self):
        """Even on a very coarse mesh where solution error is large,
        conservation must still hold to machine precision."""
        # P0 at 16x16 has ~8% solution error but must conserve exactly
        record = run_p0({"task": "p0", "cells": 16, "final_time": 1.0, "cfl": 0.4})
        # Solution error is large (expected)
        assert record["max_relative_l2"] > 0.01, "Solution error should be significant on coarse mesh"
        # But conservation is exact
        passed, details = _conservation_pass(record)
        assert passed, f"Coarse mesh conservation failed: {details}"

    def test_fine_mesh_conservation_still_exact(self):
        """On a fine mesh where solution error is tiny, conservation is still exact."""
        record = run_dg({"task": "dg", "degree": 2, "cells": 32, "final_time": 1.0, "cfl": 0.2})
        # Solution error is tiny
        assert record["max_relative_l2"] < 0.001, "Solution error should be tiny on fine mesh"
        # Conservation is still exact
        passed, details = _conservation_pass(record)
        assert passed, f"Fine mesh conservation failed: {details}"
