"""Tests for the projection-error baseline and the field-snapshot figure.

The baseline answers "what is the irreducible error the scheme starts from?" --
without it, an exponent ``alpha`` cannot be turned into an amplification per box
crossing, and "error conservation" stays an unquantified word.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks import vortex_err_field_figure as fig
from benchmarks import vortex_err_projection_baseline as pb
from benchmarks.case_euler_vortex_errconserve import TOLERANCE

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "case_euler_vortex_errconserve" / "results"


@pytest.fixture(scope="module")
def dg_base():
    torch.set_num_threads(1)
    return pb.dg_baseline(2, 32)


def test_dg_baseline_is_the_irreducible_projection_error(dg_base):
    """e(0) is a projection artefact: strictly positive, and far below tolerance."""
    assert 0.0 < dg_base["max_relative_l2"] < 0.1 * TOLERANCE
    assert set(dg_base["errors"]) == {"density", "velocity_x", "velocity_y", "pressure"}


def test_p0_cell_average_baseline_improves_with_refinement():
    torch.set_num_threads(1)
    coarse, fine = pb.p0_baseline(32), pb.p0_baseline(64)
    assert 0.0 < fine["max_relative_l2"] < coarse["max_relative_l2"]


def test_amplification_is_one_when_the_error_does_not_grow(dg_base):
    """A truly conserved error must read A = 1 by construction."""
    base = dg_base["max_relative_l2"]
    records = [{"task_key": f"dg_d2_n32_t{t:g}", "task": "dg", "degree": 2,
                "cells": 32, "final_time": t, "max_relative_l2": base}
               for t in (1.0, 5.0)]
    rows = pb.amplification(dg_base, records)
    assert [r["final_time"] for r in rows] == [1.0, 5.0]
    assert all(r["amplification"] == pytest.approx(1.0) for r in rows)


def test_amplification_ignores_other_families_and_the_initial_state(dg_base):
    records = [{"task_key": "p0_n128_t1", "task": "p0", "degree": None,
                "cells": 128, "final_time": 1.0, "max_relative_l2": 0.028},
               {"task_key": "dg_d2_n32_t0", "task": "dg", "degree": 2,
                "cells": 32, "final_time": 0.0,
                "max_relative_l2": dg_base["max_relative_l2"]}]
    assert pb.amplification(dg_base, records) == []


def test_measured_amplification_separates_conserved_from_accumulating():
    """The committed evidence bundle must show A = O(1) for DG, A >> 1 for P0."""
    if not RESULTS.exists() or not any(RESULTS.glob("*.json")):
        pytest.skip("no committed evidence bundle")
    records = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    torch.set_num_threads(1)
    dg = pb.dg_baseline(2, 32)
    rows = {r["final_time"]: r for r in pb.amplification(dg, records)}
    assert max(rows) >= 5.0, "DG family needs a long-time checkpoint"
    assert rows[max(rows)]["amplification"] < 3.0, (
        "DG error amplification over the transport window must stay order one")
    p0 = pb.p0_baseline(128)
    p0_rows = {r["final_time"]: r for r in pb.amplification(p0, records)}
    assert p0_rows[1.0]["amplification"] > 100.0, (
        "the first-order family accumulates: e(1)/e(0) must be large")


def test_field_snapshot_error_field_is_small_and_consistent():
    """Cheap small-mesh snapshot: the error field must be finite and sub-tolerance."""
    torch.set_num_threads(1)
    data = fig.snapshot(degree=2, cells=16, final_time=0.5)
    drho = data["rho"] - data["rho_exact"]
    assert np.all(np.isfinite(drho))
    peak = float(np.max(np.abs(drho)))
    # absolute peak on this coarse mesh, not a relative tolerance: density is
    # bounded below by the vortex core (0.4943), so 5e-2 is ~5 % of the field
    assert 0.0 < peak < 5e-2
    # the volume-normalised error of the snapshot must agree with the case
    # tolerance: the figure shows the *same* solution the metrics measure
    rel = float(np.sqrt((drho ** 2).mean()) / np.sqrt((data["rho_exact"] ** 2).mean()))
    assert 0.0 < rel < TOLERANCE


def test_field_figure_is_written(tmp_path):
    torch.set_num_threads(1)
    data = fig.snapshot(degree=2, cells=8, final_time=0.5)
    png = tmp_path / "fig.png"
    figure = fig.make_figure(data, png)
    assert png.exists() and png.stat().st_size > 1000
    assert figure["name"] == "fig.png"
    assert "t=0.5" in figure["caption"]