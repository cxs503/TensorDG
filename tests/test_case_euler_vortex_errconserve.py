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


def _p0_ladder(times=(1.0, 5.0)):
    """Synthetic first-order cross-check ladder (n=64/128/256)."""
    return [_synthetic_record(f"p0_n{n}", t, 0.4 / n * (1.0 + 0.01 * math.log(t)))
            for n in (64, 128, 256) for t in times]


def test_c2_gate_rejects_superlinear_error_growth():
    exact = ec.exact_invariants(points=80)
    accumulating = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t * t)
                    for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(accumulating, exact)["gates"]
    assert gates["C2_error_conservation"]["exponent"] == pytest.approx(2.0, abs=1e-6)
    assert gates["C2_error_conservation"]["passed"] is False
    assert gates["C2_error_conservation"]["status"] == "failed"
    assert gates["all_passed"] is False


def test_c2_linear_accumulation_is_the_boundary_and_passes():
    """alpha == 1 is the physical bound (error ∝ elapsed time) -> accepted.

    The stricter ``alpha <= 0.5`` reading has no literature source (see
    docs/AUDIT-m20486-CASE-DG-EULER-VORTEX-ERR.md) and is reported only.
    """
    exact = ec.exact_invariants(points=80)
    linear = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
              for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(linear, exact)["gates"]
    assert gates["C2_error_conservation"]["exponent"] == pytest.approx(1.0, abs=1e-6)
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["C2_error_conservation"]["strict_target"]["passed"] is False


def test_c2_gate_is_decidable_only_with_three_checkpoints():
    exact = ec.exact_invariants(points=80)
    two_points = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t) for t in (1.0, 2.0)]
    gate = ec.evaluate_gates(two_points, exact)["gates"]["C2_error_conservation"]
    assert gate["status"] == "not_evaluated"
    assert math.isnan(gate["exponent"])


def test_c2_gate_accepts_conserved_error():
    exact = ec.exact_invariants(points=80)
    conserved = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * (1.0 + 0.01 * math.log(t)))
                 for t in (1.0, 2.0, 5.0, 10.0, 20.0)] + _p0_ladder()
    gates = ec.evaluate_gates(conserved, exact)["gates"]
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["C3_discrete_conservation"]["passed"] is True
    assert gates["C1_tolerance"]["passed"] is True
    assert gates["C4_entropy_budget"]["passed"] is True
    assert gates["C5_crosscheck_order"]["passed"] is True
    assert gates["all_passed"] is True


def test_c3_gate_rejects_a_leaking_scheme():
    exact = ec.exact_invariants(points=80)
    leaking = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4, drift=1e-3)
               for t in (1.0, 2.0)]
    gates = ec.evaluate_gates(leaking, exact)["gates"]
    assert gates["C3_discrete_conservation"]["passed"] is False


def test_c1_gate_rejects_a_checkpoint_above_tolerance():
    """The 6 % case-pool tolerance is a snapshot tolerance at t = 1."""
    exact = ec.exact_invariants(points=80)
    bad = [_synthetic_record(ec.GATE_FAMILY, 1.0, 0.10),
           _synthetic_record(ec.GATE_FAMILY, 20.0, 1e-4)]
    gates = ec.evaluate_gates(bad, exact)["gates"]
    assert gates["C1_tolerance"]["passed"] is False
    assert gates["C1_tolerance"]["worst_at"].endswith("t1")


def test_long_time_checkpoints_exceed_tolerance_but_stay_visible():
    """Scope correction, guarded: a t=20 violation must NOT vanish.

    It leaves C1 (snapshot) untouched -- that is the corrected scope -- but the
    number must still be reported verbatim in C1x, otherwise the fix would be
    evidence laundering.
    """
    exact = ec.exact_invariants(points=80)
    records = [_synthetic_record(ec.GATE_FAMILY, 1.0, 1e-4),
               _synthetic_record(ec.GATE_FAMILY, 20.0, 0.10)] + _p0_ladder()
    gates = ec.evaluate_gates(records, exact)["gates"]
    assert gates["C1_tolerance"]["passed"] is True
    assert gates["C1x_crosscheck_raw"]["measured"] == pytest.approx(0.10)
    assert gates["C1x_crosscheck_raw"]["worst_at"].endswith("t20")
    assert "C1x_crosscheck_raw" in gates["informational"]


def test_c5_crosscheck_order_flags_a_non_converging_second_path():
    exact = ec.exact_invariants(points=80)
    flat = [_synthetic_record(f"p0_n{n}", 5.0, 3e-3) for n in (64, 128, 256)]
    flat += [_synthetic_record(f"p0_n{n}", 1.0, 1e-3) for n in (64, 128, 256)]
    gates = ec.evaluate_gates(flat, exact)["gates"]
    assert gates["C5_crosscheck_order"]["status"] == "failed"
    assert gates["verdict"] == "FAIL"


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
    records = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t * t)
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

# ---------------------------------------------------------------------------
# reference model: the exact solution must be *periodic* on the periodic box
# ---------------------------------------------------------------------------
# The m20486 release measured every long-time checkpoint against the free-space
# vortex.  On a periodic box that reference stops being the solution once the
# centre leaves the domain, so the reported "super-linear error accumulation"
# (alpha = 3.3 ... 4.4) was a defect of the *reference*, not of the scheme.  The
# tests below pin the invariant: the reference must be L-periodic and the metric
# must return ~0 when it is fed the exact periodic state.

def _grid(lo=ec.DOMAIN[0][0], hi=ec.DOMAIN[0][1], points=37):
    import torch
    axis = torch.linspace(lo, hi, points, dtype=torch.float64)
    return torch.meshgrid(axis, axis, indexing="ij")


def test_periodic_reference_is_invariant_under_a_box_translation():
    import torch
    grid_x, grid_y = _grid(-4.3, 4.1)
    shifted_x, shifted_y = _grid(5.7, 14.1)  # same points, +10 in both axes
    base = ec.vortex_primitive(grid_x, grid_y, 5.0, ec.STRENGTH, ec.BACKGROUND,
                               period=ec.PERIOD)
    moved = ec.vortex_primitive(shifted_x, shifted_y, 5.0, ec.STRENGTH,
                                ec.BACKGROUND, period=ec.PERIOD)
    for name, a, b in zip(ec.FIELDS, base, moved):
        assert torch.allclose(a, b, atol=1e-12), f"{name} is not periodic"


def test_free_space_reference_still_matches_while_the_centre_is_inside():
    """The wrap must be a no-op at t=0 and tiny early -- it is not a retune.

    Measured on the open box ``[-4.5, 4.5]^2`` (the closed-box edge is the fold's
    own discontinuity, where the free-space tail is locally 1e-3 by construction).
    """
    import torch
    grid_x, grid_y = _grid(-4.5, 4.5, points=51)
    for time, bound in ((0.0, 0.0), (1.0, 1e-3), (2.0, 5e-3)):
        free = ec.vortex_primitive(grid_x, grid_y, time, ec.STRENGTH,
                                   ec.BACKGROUND)
        periodic = ec.vortex_primitive(grid_x, grid_y, time, ec.STRENGTH,
                                       ec.BACKGROUND, period=ec.PERIOD)
        for name, a, b in zip(ec.FIELDS, free, periodic):
            relative = float(torch.sqrt(torch.mean((a - b) ** 2))
                             / torch.sqrt(torch.mean(b ** 2)))
            assert relative <= bound, f"{name} at t={time}: {relative:.3e}"


def test_free_space_reference_is_the_wrong_solution_once_the_vortex_wraps():
    """Quantifies the defect the case used to absorb into its error metric."""
    sweep = ec.reference_defect_sweep([1.0, 2.0, 5.0, 10.0])
    # early checkpoints: the two references are indistinguishable
    assert sweep["1"]["velocity_x"] < 0.02
    # wrapped checkpoints: the free-space reference is off by more than the 6 %
    # case tolerance all by itself, i.e. it can no longer measure the scheme
    assert sweep["5"]["velocity_x"] > ec.TOLERANCE * 100.0
    assert sweep["10"]["velocity_x"] > 15.0
    # density only feels the image through its radial profile, velocities through
    # the signed displacement -> the velocity defect must dominate
    assert sweep["5"]["velocity_x"] > sweep["5"]["density"]


def test_p0_metric_reports_zero_for_the_exact_periodic_state():
    import torch
    cells = 32
    x = torch.linspace(ec.DOMAIN[0][0], ec.DOMAIN[0][1], cells + 1, dtype=torch.float64)
    x = 0.5 * (x[1:] + x[:-1])
    c = x.clone()
    dx = (ec.DOMAIN[0][1] - ec.DOMAIN[0][0]) / cells
    exact_periodic = lambda xx, yy: ec.vortex_primitive(
        xx, yy, 5.0, ec.STRENGTH, ec.BACKGROUND, period=ec.PERIOD)
    average = ec._cell_averages(exact_periodic, x, c, dx, dx, ec.P0_QUADRATURE,
                                torch.device("cpu"), torch.float64)
    good = ec.vortex_error(x, c, average, 5.0, ec.STRENGTH, ec.BACKGROUND,
                           period=ec.PERIOD)
    bad = ec.vortex_error(x, c, average, 5.0, ec.STRENGTH, ec.BACKGROUND)
    # density: the cell average is reproduced exactly.  velocity/pressure keep an
    # O(h^2) residual of the primitive-from-averages nonlinearity
    # (avg(rho u)/avg(rho) != avg(u)), 3.6e-4 here -- four orders below the
    # reference defect it has to discriminate against.
    assert good["density"]["relative_l2"] < 1e-12
    for name in ("velocity_x", "velocity_y", "pressure"):
        assert good[name]["relative_l2"] < 5e-3, f"{name}: metric is not consistent"
    # the free-space metric reports the reference defect, not the field error
    assert bad["velocity_x"]["relative_l2"] > 0.1


def test_dg_metric_reports_projection_error_only_for_the_exact_periodic_state():
    """Same invariant on the DG path: exact state in -> (projection) error out."""
    import torch
    degree, cells = 2, 16
    nodes, weights, basis, _derivative, signs = ec._operators(degree)
    dx = (ec.DOMAIN[0][1] - ec.DOMAIN[0][0]) / cells
    centre = ec.DOMAIN[0][0] + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    exact_periodic = lambda xx, yy: ec.vortex_primitive(
        xx, yy, 5.0, ec.STRENGTH, ec.BACKGROUND, period=ec.PERIOD)
    coeff = ec.dg2._initial_coefficients(exact_periodic, centre, centre, dx, dx,
                                         nodes, weights, basis,
                                         torch.device("cpu"), torch.float64)
    coeff = ec.dg2._positivity_filter(coeff, basis, signs)
    good = ec.dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, 5.0, ec.STRENGTH,
                              ec.BACKGROUND, period=ec.PERIOD)
    bad = ec.dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, 5.0, ec.STRENGTH,
                             ec.BACKGROUND)
    assert max(good[name] for name in ec.FIELDS) < 1e-2   # projection error only
    assert bad["velocity_x"] > 0.1                        # reference defect
