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


def _ladder_record(config, error, drift=1e-15, entropy=1e-6):
    """A synthetic record that carries the *declared* task key of ``config``.

    C6 reconciles declared checkpoints with records, so gate tests must speak the
    ladder's own vocabulary instead of inventing keys the module never declares.
    """
    family = (f"dg_d{config['degree']}_n{config['cells']}_cfl{config['cfl']:g}"
              if config["task"] == "dg" else f"p0_n{config['cells']}")
    record = _synthetic_record(family, config["final_time"], error,
                               drift=drift, entropy=entropy)
    record["task_key"] = ec.task_key(config)
    record["config"] = dict(config)
    # Real result files carry the configuration at top level, and C8/C9 read it
    # there (degree / cells); a synthetic bundle missing it would grade as nan.
    record["task"] = config["task"]
    record["cells"] = config["cells"]
    record["cfl"] = config["cfl"]
    if config["task"] == "dg":
        record["degree"] = config["degree"]
    return record


def _full_ladder_records(dg_error=None, p0_error=None, drop=()):
    """Synthetic evidence bundle covering every declared checkpoint (C6-complete).

    ``dg_error`` is a function of ``(degree, cells, final_time)``: C8 and C9 grade
    the *degree* and the *mesh refinement* of a DG family, so a synthetic error
    that depends on time alone would (rightly) read as zero spatial order.
    """
    dg_error = dg_error or (
        lambda degree, cells, t: 0.02 / cells ** (degree + 1)
        * (1.0 + 0.01 * math.log(t)))
    p0_error = p0_error or (lambda n, t: 0.4 / n * (1.0 + 0.01 * math.log(t)))
    records = []
    for config in ec.LADDER:
        if ec.task_key(config) in drop:
            continue
        error = (dg_error(config["degree"], config["cells"], config["final_time"])
                 if config["task"] == "dg"
                 else p0_error(config["cells"], config["final_time"]))
        records.append(_ladder_record(config, error))
    return records


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
    conserved = _full_ladder_records()
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


def test_c6_gate_flags_a_declared_checkpoint_with_no_result():
    """The defect C6 exists for: 11 of 12 declared checkpoints graded as PASS."""
    exact = ec.exact_invariants(points=80)
    records = _full_ladder_records(drop={"dg_d2_n32_t10_cfl0.2"})
    gates = ec.evaluate_gates(records, exact)["gates"]
    c6 = gates["C6_ladder_completeness"]
    assert c6["declared"] == len(ec.LADDER)
    assert c6["present"] == len(ec.LADDER) - 1
    assert c6["missing"] == ["dg_d2_n32_t10_cfl0.2"]
    assert c6["passed"] is False and c6["status"] == "failed"
    assert gates["all_passed"] is False and gates["verdict"] == "FAIL"


def test_c6_gate_passes_on_a_complete_ladder():
    exact = ec.exact_invariants(points=80)
    gates = ec.evaluate_gates(_full_ladder_records(), exact)["gates"]
    assert gates["C6_ladder_completeness"]["missing"] == []
    assert gates["C6_ladder_completeness"]["passed"] is True
    assert gates["all_passed"] is True


def test_c6_reports_unexpected_keys_without_failing():
    exact = ec.exact_invariants(points=80)
    records = _full_ladder_records() + [_synthetic_record("p0_n512", 1.0, 1e-4)]
    gates = ec.evaluate_gates(records, exact)["gates"]
    assert gates["C6_ladder_completeness"]["unexpected"] == ["p0_n512_t1"]
    assert gates["C6_ladder_completeness"]["passed"] is True


def test_committed_evidence_covers_every_declared_checkpoint():
    """Guards the shipped bundle: every declared checkpoint must be on disk."""
    import pathlib
    import json as _json
    results = pathlib.Path(__file__).resolve().parents[1] / \
        "case_euler_vortex_errconserve" / "results"
    records = [_json.loads(p.read_text()) for p in sorted(results.glob("*.json"))]
    assert records, f"no records under {results}"
    completeness = ec.ladder_completeness(records)
    # Every declared checkpoint must be on disk *or* be a run explicitly
    # registered in PENDING_RUNS with the job that is producing it.
    assert set(completeness["missing"]) == set(completeness["pending"])
    assert completeness["declared"] == len(ec.LADDER)


# ---------------------------------------------------------------------------
# C7: design-space invariance
# ---------------------------------------------------------------------------

def test_c7_passes_when_every_design_point_conserves_error():
    exact = ec.exact_invariants(points=80)
    gates = ec.evaluate_gates(_full_ladder_records(), exact)["gates"]
    c7 = gates["C7_design_space_invariance"]
    assert len(c7["evaluated_families"]) >= ec.MIN_DECIDABLE_DESIGN_FAMILIES
    assert "dg_d2_n32_cfl0.2" in c7["evaluated_families"]
    assert "dg_d2_n32_cfl0.4" in c7["evaluated_families"]   # CFL axis
    assert "dg_d2_n64_cfl0.2" in c7["evaluated_families"]   # mesh axis
    assert c7["offenders"] == {} and c7["passed"] is True


def test_c7_rejects_conservation_that_holds_only_on_the_design_point():
    """The defect C7 exists for: alpha <= 1 on 32^2 / CFL 0.2 and nowhere else."""
    exact = ec.exact_invariants(points=80)

    def error(config):
        time = config["final_time"]
        if config["task"] != "dg":
            return 0.4 / config["cells"]
        # super-linear accumulation on the fine mesh only
        return 1e-4 * time ** (2.0 if config["cells"] == 64 else 0.2)

    records = _full_ladder_records(dg_error=lambda d, c, t: 1e-4 * t,
                                   p0_error=lambda n, t: 0.4 / n)
    for config in ec.LADDER:
        # only the degree-2 refined mesh: the p = 1 pair is a different family
        if (config["task"] == "dg" and config["cells"] == 64
                and config["degree"] == 2):
            for record in records:
                if record["task_key"] == ec.task_key(config):
                    record["max_relative_l2"] = error(config)
                    record["max_percent"] = 100.0 * record["max_relative_l2"]
    gates = ec.evaluate_gates(records, exact)["gates"]
    c7 = gates["C7_design_space_invariance"]
    assert list(c7["offenders"]) == ["dg_d2_n64_cfl0.2"]
    assert c7["offenders"]["dg_d2_n64_cfl0.2"] > 1.0
    assert c7["passed"] is False and c7["status"] == "failed"
    # C2 (design point) still reads as passed: the gate that catches this is C7
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["verdict"] == "FAIL"


def test_c7_is_not_evaluated_on_a_single_design_point():
    """A one-point design space must read not_evaluated, never passed."""
    exact = ec.exact_invariants(points=80)
    records = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * (1.0 + 0.01 * math.log(t)))
               for t in (1.0, 2.0, 5.0, 10.0)]
    gates = ec.evaluate_gates(records, exact)["gates"]
    c7 = gates["C7_design_space_invariance"]
    assert c7["status"] == "not_evaluated" and c7["passed"] is False
    assert "C7_design_space_invariance" in gates["not_evaluated"]
    # not-evaluated is never folded into PASS: the case cannot certify C7 here
    assert gates["all_passed"] is False


def test_ladder_covers_both_design_axes():
    """Guard against a silent revert to the single-point design space."""
    dg = [c for c in ec.LADDER if c["task"] == "dg"]
    meshes = {c["cells"] for c in dg if abs(c["cfl"] - 0.2) < 1e-12}
    cfls = {c["cfl"] for c in dg if c["cells"] == 32}
    assert len(meshes) >= 2 and len(cfls) >= 2
    # and every DG family must offer >= 3 checkpoints, otherwise alpha is nan
    families = {}
    for config in dg:
        families.setdefault(
            f"d{config['degree']}_n{config['cells']}_cfl{config['cfl']:g}",
            []).append(config["final_time"])
    for key, times in families.items():
        assert len(set(times)) >= 3, key
    # the degree axis itself must stay covered: C8 is vacuous without it
    assert {c["degree"] for c in dg} >= set(ec.DEGREE_AXIS)


def test_c7x_amplification_prediction_is_falsifiable():
    """The refinement prediction must be able to read False, and to read None."""
    record = {"family": "dg_d2_n32_cfl0.2", "cfl": 0.2, "final_time": 10.0,
              "max_relative_l2": 1e-3, "max_percent": 0.1,
              "config": {"task": "dg", "degree": 2, "cells": 32,
                         "final_time": 10.0, "cfl": 0.2}}
    refined = dict(record, family="dg_d2_n64_cfl0.2",
                   config=dict(record["config"], cells=64))
    amp = ec.amplification([record, refined])
    assert amp["supported"] is not None
    assert "dg_d2_n32_cfl0.2" in amp["values"] and "dg_d2_n64_cfl0.2" in amp["values"]
    assert amp["values"]["dg_d2_n32_cfl0.2"]["amplification"] > 1.0
    # same numbers on both meshes -> the refined mesh amplifies *more* -> falsified
    assert amp["supported"] is False
    # and the criterion is not evaluable with the design point alone
    assert ec.amplification([record])["supported"] is None


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
                                            "fig_invariants_vs_time.png",
                                            "fig_design_space.png"}
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


def test_reference_defect_mimics_the_velocity_only_mechanism_signature():
    """The "u jumps, rho/p stay flat" signature is the *reference*, not a scheme.

    The superseded mechanism module (``benchmarks/case_euler_vortex_err_mechanism``,
    M1) read that signature as a divergence-free alias mode and planned a
    Helmholtz split to confirm it.  The defect sweep shows the confound: at the
    very checkpoint where the jump appears (t = 2) the free-space reference is
    already wrong by more than the jump itself in velocity, while its density
    error is three orders smaller -- velocities feel the signed displacement,
    density and pressure only feel the radial profile.  So M1's evidence cannot
    discriminate a scheme defect from a reference defect, and no mechanism claim
    may be built on the pre-fix numbers.
    """
    sweep = ec.reference_defect_sweep([2.0])
    velocity, density = sweep["2"]["velocity_x"], sweep["2"]["density"]
    # >> the 19.5x u/rho ratio the old numbers were read as
    assert velocity / density > 100.0
    # large enough on its own to cover the 0.2467 % "jump" the old metric reported
    assert velocity > 0.25


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


# ---------------------------------------------------------------------------
# C8: degree invariance
# ---------------------------------------------------------------------------

def test_c8_rejects_conservation_that_holds_only_at_the_design_degree():
    """The defect C8 exists for: alpha <= 1 at p = 2 and nowhere else.

    C7 moves the mesh and the CFL, so before C8 a "conserved error" certified on
    p = 2 alone was still a statement about one polynomial degree.  Here the
    p = 1 and p = 3 families accumulate super-linearly while the design degree
    stays clean: C2 (design point) must still read passed, C8 must fail.
    """
    exact = ec.exact_invariants(points=80)
    base = _full_ladder_records()
    records = []
    for record in base:
        if record["task"] == "dg" and record["degree"] != 2 and record["cfl"] == 0.2:
            record["max_relative_l2"] = 1e-6 * record["final_time"] ** 2
            record["max_percent"] = 100.0 * record["max_relative_l2"]
        records.append(record)
    gates = ec.evaluate_gates(records, exact)["gates"]
    c8 = gates["C8_degree_invariance"]
    assert set(c8["offenders"]) == {1, 3}
    assert c8["passed"] is False and c8["status"] == "failed"
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["verdict"] == "FAIL"


def test_c8_is_not_evaluated_when_only_one_degree_is_decidable():
    """One degree is the anecdote C7 was written against; C8 says so out loud."""
    exact = ec.exact_invariants(points=80)
    # only the design degree p = 2 is kept: one degree is not an invariance
    keep = {2}
    records = [record for record in _full_ladder_records()
               if record["task"] != "dg" or record.get("degree") in keep]
    gates = ec.evaluate_gates(records, exact)["gates"]
    c8 = gates["C8_degree_invariance"]
    assert c8["status"] == "not_evaluated" and c8["passed"] is False
    assert "C8_degree_invariance" in gates["not_evaluated"]
    assert gates["all_passed"] is False


def test_c8_degree_axis_is_actually_declared_in_the_ladder():
    """Guard: a C8 with no degree families to grade is a vacuous pass."""
    degrees = {config["degree"] for config in ec.LADDER if config["task"] == "dg"}
    assert set(ec.DEGREE_AXIS) <= degrees
    assert ec.MIN_DECIDABLE_DEGREES <= len(ec.DEGREE_AXIS)


# ---------------------------------------------------------------------------
# C9: the long-time error must still fall at the design order
# ---------------------------------------------------------------------------

def test_c9_measures_design_order_at_both_checkpoints():
    exact = ec.exact_invariants(points=80)
    gates = ec.evaluate_gates(_full_ladder_records(), exact)["gates"]
    c9 = gates["C9_design_order"]
    assert c9["status"] == "passed"
    for degree in ec.ORDER_FAMILIES:
        for checkpoint in ec.ORDER_CHECKPOINTS:
            row = c9["per_case"][f"dg_d{degree}_t{checkpoint:g}"]
            assert row["order"] == pytest.approx(degree + 1, abs=0.05), row
            assert row["target"] == pytest.approx(ec.ORDER_SLACK * (degree + 1))


def test_c9_rejects_a_long_time_residue_that_ignores_the_mesh():
    """A grid-independent residue after the box crossing kills the order.

    This is the failure mode C2's exponent alone cannot see: an error that stops
    improving with h still grows slowly in time, so alpha stays small while the
    "the error is a grid property" claim is already false.
    """
    exact = ec.exact_invariants(points=80)
    base = _full_ladder_records()
    records = []
    for record in base:
        if (record["task"] == "dg" and record["degree"] == 2
                and record["cfl"] == 0.2
                and abs(record["final_time"] - 10.0) < 1e-12):
            record["max_relative_l2"] = 1e-5    # same on n = 32 and n = 64
            record["max_percent"] = 1e-3
        records.append(record)
    gates = ec.evaluate_gates(records, exact)["gates"]
    c9 = gates["C9_design_order"]
    assert list(c9["failures"]) == ["dg_d2_t10"]
    assert c9["failures"]["dg_d2_t10"] == pytest.approx(0.0, abs=1e-9)
    assert c9["passed"] is False and c9["status"] == "failed"
    # the t = 1 snapshot is untouched: C9 is specific, not a blanket failure
    assert c9["per_case"]["dg_d2_t1"]["passed"] is True


def test_c9_is_not_evaluated_when_a_degree_has_no_refinement():
    exact = ec.exact_invariants(points=80)
    records = [record for record in _full_ladder_records()
               if not (record["task"] == "dg" and record["degree"] == 1
                       and record["cells"] == 64)]
    gates = ec.evaluate_gates(records, exact)["gates"]
    c9 = gates["C9_design_order"]
    assert c9["status"] == "not_evaluated" and c9["passed"] is False
    assert "C9_design_order" in gates["not_evaluated"]
    assert gates["all_passed"] is False
