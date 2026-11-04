"""Pins for the temporal/spatial error split of the 2-D Euler DG solver.

Production runs use ``cfl = 0.2`` with SSP-RK3, so at a *fixed* CFL the time step
scales as ``dt ~ h`` and the third-order temporal error ``O(dt^3) = O(h^3)``
competes with the fourth-order spatial error of P3.  ``benchmarks/
temporal_asymptotics_euler_dg_2d.py`` measures that split by refining ``dt`` at
fixed ``h`` (``e = e0 + C dt^q``) and extrapolating to ``dt -> 0``.

The tests here pin the cheap corner of that study:

* the extrapolation machinery is exact on synthetic data,
* the measured temporal order at a fixed mesh is the SSP-RK3 order (3), and
* the ``dt -> 0`` limit still converges at ``p + 1`` for P2.

The full P3 ladder is a benchmark, not a unit test; its evidence file is pinned
by ``test_p3_limit_reaches_the_design_order`` when it is present.
"""
import json
import math
from pathlib import Path

import pytest

from benchmarks.temporal_asymptotics_euler_dg_2d import (
    _fit_three_parameter,
    _richardson,
    temporal_block,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "validation" / "euler_dg_2d" / "temporal_asymptotics.json"
CFLS = (0.2, 0.1, 0.05)


def test_richardson_is_exact_for_a_pure_power_law():
    """``e(dt) = e0 + C dt^q``: Richardson with the right q returns e0 exactly."""
    e0, c, q, ratio = 1.0e-6, 3.0e-3, 3.0, 2.0
    dt_fine = 1.0e-2
    e_fine = e0 + c * dt_fine ** q
    e_coarse = e0 + c * (ratio * dt_fine) ** q
    assert _richardson(e_coarse, e_fine, ratio, q) == pytest.approx(e0, rel=1e-12)


def test_three_parameter_fit_recovers_the_limit():
    dts = [2.0e-2, 1.0e-2, 5.0e-3]
    es = [1.0e-6 + 3.0e-3 * dt ** 3 for dt in dts]
    e0, c = _fit_three_parameter(dts, es, 3.0)
    assert e0 == pytest.approx(1.0e-6, rel=1e-9)
    assert c == pytest.approx(3.0e-3, rel=1e-9)


def test_temporal_block_needs_three_levels():
    with pytest.raises(ValueError):
        temporal_block(2, 8, [0.2, 0.1])


def test_temporal_order_at_fixed_mesh_is_the_rk3_order():
    """Refining dt at fixed h must show ``q ~ 3`` (SSP-RK3), not ``q ~ p + 1``."""
    block = temporal_block(2, 8, CFLS, log=lambda *_a, **_k: None)
    q = block["temporal_order"]["density"]["q"]
    assert 2.4 < q < 3.6, (q, block["temporal_order"]["density"]["adjacent"])


def test_dt_limit_keeps_the_design_order_for_p2():
    """P2: after removing the RK3 error the spatial order must stay ~3."""
    fine = temporal_block(2, 12, CFLS, log=lambda *_a, **_k: None)
    coarse = temporal_block(2, 8, CFLS, log=lambda *_a, **_k: None)
    for field in ("density", "pressure"):
        e_fine = fine["spatial_limit"][field]["richardson"]
        e_coarse = coarse["spatial_limit"][field]["richardson"]
        eoc = math.log(e_coarse / e_fine) / math.log(coarse["h"] / fine["h"])
        assert 2.7 < eoc < 3.3, (field, eoc)
        # the limit can never be negative and must not exceed the raw error
        assert 0.0 < e_fine <= fine["rows"][-1]["errors"][field]


@pytest.mark.skipif(not EVIDENCE.exists(), reason="run benchmarks.temporal_asymptotics_euler_dg_2d")
def test_p3_limit_reaches_the_design_order():
    """The deliverable of the study: P3 spatial order from the dt -> 0 limits."""
    data = json.loads(EVIDENCE.read_text())
    entry = data["spatial_orders"].get("3")
    assert entry is not None, "no P3 block in the evidence file"
    assert entry["design_order"] == 4
    for field in ("density", "pressure"):
        slope = entry["fit_limit"][field]
        assert slope > 3.9, (field, slope, entry["eoc_limit"][field])