"""Tests for the EULER-VORTEX-ERR mechanism harness (m20598)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from benchmarks.case_euler_vortex_err_mechanism import (
    configs,
    evaluate,
    main,
    run_config,
)
from benchmarks.verify_euler_dg_2d import _vortex_error
import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))


def test_config_matrix_is_indexable_and_unique():
    rows = configs()
    assert [r["index"] for r in rows] == list(range(len(rows)))
    assert len({r["id"] for r in rows}) == len(rows)
    # the mechanism question needs the baseline, a de-aliased arm and a CFL arm
    ids = {r["id"] for r in rows}
    assert {"base_t1", "base_t2", "base_t5", "q6_t2", "q8_t2", "cfl01_t2"} <= ids


@pytest.fixture(scope="module")
def small_record():
    # degree=2/quad=4 makes the harness quadrature coincide with the solver
    # default (max(degree+2, 3) = 4), so the metric test compares like with like
    cfg = {"id": "tiny", "index": 0, "role": "test", "t": 0.05, "quad": 4,
           "cfl": 0.2, "degree": 2, "cells": 8}
    return cfg, run_config(cfg)


def test_record_fields_and_internal_consistency(small_record):
    cfg, rec = small_record
    assert rec["steps"] > 0 and rec["t"] == cfg["t"]
    for field in ("density", "velocity_x", "velocity_y", "pressure"):
        assert 0.0 < rec["rel"][field] < 1.0
    assert 0.0 <= rec["solenoidal_fraction"] <= 1.0
    assert 0.0 <= rec["gridband_fraction"] <= 1.0
    # grid band + smooth band must reconstruct the total velocity-error energy
    total = rec["vel_error_l2"] ** 2
    parts = rec["gridband_l2"] ** 2 + rec["smooth_l2"] ** 2
    assert abs(total - parts) <= 1e-10 * max(total, 1.0)
    # solenoidal + dilatational likewise
    # the Helmholtz split is orthogonal in Fourier space; the residual below is
    # FFT round-off (measured 3e-6 relative), not a modelling gap
    assert abs(rec["solenoidal_l2"] ** 2 + rec["dilatational_l2"] ** 2 - total) \
        <= 1e-4 * total


def test_metric_matches_committed_harness(small_record):
    """The reported relative L2 must be the committed C1 metric, not a variant."""
    cfg, rec = small_record
    nodes, _, basis, _, _ = dg2.legendre_operators(cfg["degree"], 4)

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, 5.0, (1.0, 1.0))

    x = torch.arange(cfg["cells"], dtype=torch.float64)
    _x, _y, coeff, _meta = dg2.solve_euler_dg_2d(
        initial, ncells=cfg["cells"], degree=cfg["degree"], domain=DOMAIN,
        final_time=cfg["t"], cfl=cfg["cfl"], periodic=(True, True),
        initial_projection="l2", quadrature=4)
    ref = _vortex_error(coeff, basis, nodes, DOMAIN, cfg["t"], 5.0, (1.0, 1.0))
    for name in ("density", "velocity_x", "pressure"):
        assert rec["rel"][name] == pytest.approx(float(ref[name]), rel=1e-12)


def _synthetic(ident, t, max_rel, sol, grid, u_over_rho=1.0, steps=10):
    return {"id": ident, "index": 0, "role": "synthetic", "t": t, "quad": 4,
            "cfl": 0.2, "degree": 2, "cells": 32, "steps": steps,
            "rel": {"density": max_rel / 20.0, "velocity_x": max_rel,
                    "velocity_y": max_rel * 0.9, "pressure": max_rel / 20.0},
            "max_rel": max_rel, "u_over_rho": u_over_rho,
            "solenoidal_fraction": sol, "gridband_fraction": grid,
            "vel_err_peak_in_cells": 1.0, "gridband_l2": 1.0, "smooth_l2": 0.1,
            "solenoidal_l2": 1.0, "dilatational_l2": 0.1,
            "vel_error_l2": 1.0, "rho_error_l2": 1e-3, "p_error_l2": 1e-3}


def test_gates_are_decidable_and_directional():
    base = [_synthetic("base_t1", 1.0, 1.0e-4, 0.8, 0.7, u_over_rho=2.5),
            _synthetic("base_t2", 2.0, 3.0e-3, 0.97, 0.95, u_over_rho=48.0),
            _synthetic("base_t5", 5.0, 1.0e-2, 0.99, 0.97, u_over_rho=50.0),
            _synthetic("q6_t2", 2.0, 1.0e-3, 0.6, 0.5),
            _synthetic("q8_t2", 2.0, 5.0e-4, 0.5, 0.4),
            _synthetic("q8_t5", 5.0, 2.0e-3, 0.5, 0.4),
            _synthetic("cfl01_t2", 2.0, 3.1e-3, 0.97, 0.95)]
    metrics = evaluate(base)
    gates = metrics["gates"]
    assert gates["M1_signature"]["passed"] is True          # alias signature
    assert gates["M2_dealias"]["passed"] is True            # quadrature-limited
    assert gates["M3_cfl"]["passed"] is True                 # CFL-insensitive
    assert gates["M1_signature"]["status"] == "evaluated"

    # a smooth, dispersion-like error must NOT be classified as an alias
    smooth = [_synthetic("base_t1", 1.0, 1.0e-4, 0.5, 0.05, u_over_rho=2.5),
              _synthetic("base_t2", 2.0, 2.0e-4, 0.5, 0.05, u_over_rho=2.6),
              _synthetic("cfl01_t2", 2.0, 2.0e-4, 0.5, 0.05)]
    assert evaluate(smooth)["gates"]["M1_signature"]["passed"] is False


def test_missing_records_are_not_evaluated_not_passed():
    metrics = evaluate([_synthetic("base_t2", 2.0, 3.0e-3, 0.97, 0.95, 48.0)])
    assert metrics["gates"]["M2_dealias"]["status"] == "not_evaluated"
    assert metrics["gates"]["M2_dealias"]["passed"] is False


def test_aggregate_writes_artifacts(tmp_path):
    src = tmp_path / "records"
    src.mkdir()
    rows = [("base_t1", 1.0, 1.0e-4, 0.8, 0.7, 2.5),
            ("base_t2", 2.0, 3.0e-3, 0.97, 0.95, 48.0),
            ("base_t5", 5.0, 1.0e-2, 0.99, 0.97, 50.0),
            ("q6_t2", 2.0, 1.0e-3, 0.6, 0.5, 20.0),
            ("q8_t2", 2.0, 5.0e-4, 0.5, 0.4, 10.0),
            ("q8_t5", 5.0, 2.0e-3, 0.5, 0.4, 10.0),
            ("cfl01_t2", 2.0, 3.1e-3, 0.97, 0.95, 48.0)]
    for i, (ident, t, e, sol, grid, ratio) in enumerate(rows):
        rec = _synthetic(ident, t, e, sol, grid, ratio)
        rec["index"] = i
        (src / f"{ident}.json").write_text(json.dumps(rec))
    out = tmp_path / "out"
    assert main(["--aggregate", "--records", str(src), "--output-dir", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics["gates"]) == {"M1_signature", "M2_dealias", "M3_cfl"}
    text = (out / "summary.md").read_text()
    assert "nan" not in text
    assert (out / "fig_error_vs_quadrature.png").exists()
    assert (out / "fig_error_structure.png").exists()