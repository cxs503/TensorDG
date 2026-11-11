"""Gates for the Shu--Osher case-pool artifacts (analytic anchors + independent solver).

The expensive part of the case (the DG ladder and the FV-WENO5 reference) lives
in ``benchmarks/shu_osher_case_pool.py`` and its artifacts under
``runs/shu_osher_case_pool/``.  These tests only re-check the *anchors* that make
those artifacts trustworthy, so they stay fast:

* the Rankine--Hugoniot machinery reproduces the case's left state from a
  Mach-3 shock (i.e. the case definition is self-consistent, no transcribed
  numbers),
* the exact mass budget and the exact unshocked sine hold on the committed
  reference profile,
* the independent FV-WENO5 solver is a genuine 5th-order reconstruction and
  leaves a uniform flow untouched,
* when the ladder artifacts exist, every configuration stays admissible and the
  tabulated errors agree with the 5% acceptance line recorded in ``metrics.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.fv_weno5_1d import solve_fv_weno5, weno5_reconstruct
from benchmarks.shu_osher_1d import LEFT, initial_state
from benchmarks.shu_osher_case_pool import (
    exact_mass_budget,
    profile_metrics,
    rankine_hugoniot_anchor,
)

RUN = Path("runs/shu_osher_case_pool")
COMMITTED_REFERENCE = Path("benchmarks/data/shu_osher_reference.npz")


def test_rankine_hugoniot_reproduces_the_case_left_state():
    """The left state must be the post-shock state of a Mach-3 shock into (1, 1)."""
    anchor = rankine_hugoniot_anchor()
    rel = anchor["hugoniot_relative_error"]
    assert rel["rho"] < 1e-6
    assert rel["u"] < 1e-6
    assert rel["p"] < 1e-5
    # ... and the shock speed obtained from the jump conditions must equal 3 c_1
    assert abs(anchor["shock_speed_analytic"] - anchor["shock_speed_from_mach3"]) / \
        anchor["shock_speed_from_mach3"] < 1e-6


def test_analytic_positions_are_ordered_as_the_physics_requires():
    anchor = rankine_hugoniot_anchor()
    assert anchor["quiet_zone_head_analytic"] < anchor["contact_position_analytic"]
    assert anchor["contact_position_analytic"] < anchor["shock_position_analytic"]
    assert anchor["quiet_zone_head_analytic"] < -4.0 + 1.8 * anchor["contact_speed_analytic"]


def test_exact_mass_budget_closure():
    budget = exact_mass_budget()
    assert budget["mass_final_predicted"] == pytest.approx(
        budget["mass_initial"] + budget["mass_inflow"], rel=1e-12)
    assert budget["mass_inflow"] > 0.0  # inflow, not outflow: the left boundary is upstream


@pytest.mark.skipif(not COMMITTED_REFERENCE.exists(), reason="committed reference missing")
def test_committed_reference_satisfies_the_exact_anchors():
    """The 300-cell P3 profile shipped with the repo must obey the closed-form anchors."""
    blob = np.load(COMMITTED_REFERENCE, allow_pickle=False)
    x, rho = blob["x"], blob["rho"]
    metrics = profile_metrics(x, rho)
    budget = exact_mass_budget()
    anchor = rankine_hugoniot_anchor()
    # quiet zone: exact constant left state
    assert metrics["plateau_rho_dev_percent"] < 0.5
    # unshocked region: exact initial sine
    assert metrics["preshock_sine_rel_l1_percent"] < 0.2
    # global mass: exact identity (normalised by the initial mass)
    assert abs(metrics["mass"] - budget["mass_final_predicted"]) / budget["mass_initial"] < 1e-3
    # shock position from Rankine-Hugoniot
    assert abs(metrics["shock_position"] - anchor["shock_position_analytic"]) / \
        anchor["shock_position_analytic"] < 0.02


def test_weno5_reconstruction_is_fifth_order_on_a_smooth_wave():
    """Refining by 2 must cut the reconstruction error by about 2**5."""
    errors = []
    for ncells in (32, 64):
        # exact *cell averages* of sin(2*pi*x) (point values would cap the test at
        # second order, since the projection onto a cell average is O(h^2))
        edges = np.arange(ncells + 1) / ncells
        exact = (np.cos(2.0 * np.pi * edges[:-1]) - np.cos(2.0 * np.pi * edges[1:])) / (
            2.0 * np.pi / ncells)
        exact_faces = np.sin(2.0 * np.pi * edges)
        u = torch.tensor(np.stack((exact, exact, exact), axis=-1), dtype=torch.float64)
        u_left, _ = weno5_reconstruct(u)
        # interior faces only: the transmissive ghost cells are first-order by
        # construction, which would mask the 5th-order interior behaviour
        errors.append(float(np.max(np.abs(u_left[3:-3, 0].numpy() - exact_faces[3:-3]))))
    order = np.log2(errors[0] / errors[1])
    assert order > 4.0, f"observed WENO5 reconstruction order {order:.2f}"


def test_fv_weno5_leaves_a_uniform_flow_untouched():
    """A uniform state is an exact steady solution for FV-WENO5/HLLC."""
    def uniform(x):
        return torch.ones_like(x), 0.5 * torch.ones_like(x), torch.ones_like(x)

    _, u, meta = solve_fv_weno5(uniform, ncells=64, final_time=0.1, cfl=0.3)
    assert np.max(np.abs(u[:, 0] - 1.0)) < 1e-12
    assert meta["density_min"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.skipif(not (RUN / "metrics.json").exists(), reason="case artefacts not generated")
def test_ladder_artifacts_are_admissible_and_report_the_5pct_gate():
    metrics = json.loads((RUN / "metrics.json").read_text())
    rows = metrics["ladder"]
    assert rows, "ladder table is empty"
    for row in rows:
        assert row["finite"], row
        assert row["density_min"] if "density_min" in row else True
        assert row["rho_min"] > 0.0
        # the recorded verdict must be derivable from the recorded error
        assert row["passes_5pct"] == (row["rel_l1_percent"] < 5.0)
    # at least the finest P3 run is expected inside the 5% acceptance line
    finest = max(rows, key=lambda r: (r["degree"], r["ncells"]))
    assert finest["rel_l1_percent"] < 5.0, finest