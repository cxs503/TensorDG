"""Failure tests for the DG-side time axis (D0-D3) of the round-2 anchor.

The axis claims a *negative* result -- the shipped DG legs' time step is not what
their error measures -- so the tests must show the machinery can say the opposite
when the physics does: a synthetic tree where the time error dominates has to be
reported as polluted and has to flip a decision.  Both directions are asserted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_certified as xval  # noqa: E402
from benchmarks import xval_dg_round2_time_pollution as tp  # noqa: E402


# ------------------------------------------------------------------ fixtures
def _write_level(raw: Path, ncells: int, cfl: float, field) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-5.0, 5.0, int(ncells), endpoint=False) + 5.0 / int(ncells)
    np.savez(raw / f"fv_cfl{tp.cfl_tag(cfl)}_n{int(ncells)}.npz",
             x=x, rho=np.asarray(field, dtype=float),
             rho_u=np.zeros(int(ncells)), rho_e=np.zeros(int(ncells)),
             meta=json.dumps({"synthetic": True}))


def _coarsen_once(field) -> np.ndarray:
    f = np.asarray(field, dtype=float)
    return 0.5 * (f[0::2] + f[1::2])


def _power_law_tree(tmp_path: Path, *, temporal_coeff: float) -> Path:
    """A tree whose DG legs have a *known* temporal error of the chosen size.

    The reference ladder is a pure power law ``q + C h^p`` with ``q`` piecewise
    smooth, so the certified reference is exact and every leg error is known in
    closed form.  The three DG legs are
    ``ref + C_leg h_leg^2 + temporal_coeff * cfl**2`` on the leg's own grid;
    ``temporal_coeff`` therefore sets the time/pollution ratio directly, and no
    CFD is run.
    """
    raw = tmp_path / "raw"
    base = np.linspace(0.0, 1.0, 6400, endpoint=False)
    q = 1.0 + 0.2 * np.sin(4.0 * np.pi * base)
    levels = {}
    for n in (800, 1600, 3200, 6400):
        idx = np.linspace(0, base.size - 1, n).astype(int)
        field = q[idx] + 0.5 * (10.0 / n) ** 1.5
        levels[n] = field
        _write_level(raw, n, 0.3, field)
    # certified reference: Richardson on the 3200/6400 pair
    ref = tp.certify([tp.load_level(raw, n, 0.3) for n in (800, 1600, 3200, 6400)],
                     tolerance_percent=tp.GATE_PERCENT)
    reference = ref["reference"]          # on the 3200 grid
    for cfl in tp.DG_TIME_CFLS:
        leg = reference + 0.02 + temporal_coeff * float(cfl) ** 2
        np.savez(raw / xval.leg_filename(tp.DG_TIME_DEGREE, tp.DG_TIME_GRID, cfl),
                 x=np.linspace(-5.0, 5.0, reference.size, endpoint=False),
                 rho=np.asarray(leg, dtype=float))
    return raw


# ------------------------------------------------------------------- tests
def test_axis_is_clean_when_the_time_step_does_not_move_the_leg(tmp_path):
    raw = _power_law_tree(tmp_path, temporal_coeff=0.0)
    axis = tp.dg_temporal_axis(raw)
    assert axis["state"] == tp.DG_AXIS_CLEAN, axis["criteria"]
    assert axis["accepted"] is True
    # a leg whose CFL is irrelevant must show *bit-identical* levels
    assert axis["criteria"]["D1"]["ratio"] == pytest.approx(0.0, abs=1e-12)
    assert axis["criteria"]["D2"]["pass"] is True


def test_axis_is_polluted_when_the_time_error_dominates(tmp_path):
    """Negative direction: the same machinery must refuse a time-polluted leg."""
    raw = _power_law_tree(tmp_path, temporal_coeff=0.5)
    axis = tp.dg_temporal_axis(raw)
    assert axis["criteria"]["D1"]["state"] == "POLLUTED", axis["criteria"]["D1"]
    assert axis["state"] == tp.DG_AXIS_POLLUTED
    assert axis["accepted"] is False


def test_axis_reports_a_decision_flip_instead_of_hiding_it(tmp_path):
    """D2 must fire when the coarse-CFL leg crosses the gate the fine one clears."""
    raw = _power_law_tree(tmp_path, temporal_coeff=0.0)
    # overwrite the CFL=0.4 leg with an offset large enough to fail a 3 % gate
    reference = tp.certify([tp.load_level(raw, n, 0.3)
                            for n in (*tp.BASE_LEVELS, *tp.EXTRA_RATIO_LEVELS)],
                           tolerance_percent=tp.GATE_PERCENT)["reference"]
    name = xval.leg_filename(tp.DG_TIME_DEGREE, tp.DG_TIME_GRID, 0.4)
    np.savez(raw / name, x=np.linspace(-5.0, 5.0, reference.size, endpoint=False),
             rho=np.asarray(reference * 1.05, dtype=float))
    axis = tp.dg_temporal_axis(raw, tolerance_percent=3.0)
    assert axis["criteria"]["D2"]["pass"] is False
    assert len(set(axis["criteria"]["D2"]["decisions"].values())) > 1
    assert axis["state"] == tp.DG_AXIS_POLLUTED


def test_axis_refuses_a_missing_leg_loudly(tmp_path):
    raw = _power_law_tree(tmp_path, temporal_coeff=0.0)
    (raw / xval.leg_filename(tp.DG_TIME_DEGREE, tp.DG_TIME_GRID, 0.2)).unlink()
    with pytest.raises(FileNotFoundError, match="DG time axis"):
        tp.dg_temporal_axis(raw)


def test_withheld_audit_still_reports_the_dg_axis(tmp_path):
    """The FV-side claim may be withheld without withholding a measurement that
    exists: the withheld record must carry the DG axis, and must still claim no
    T0-T5 criteria."""
    raw = _power_law_tree(tmp_path, temporal_coeff=0.0)
    # this tree carries only CFL=0.3 FV levels, so the FV-side T0-T5 audit must
    # withhold -- and the DG axis, which reads committed profiles only, must not
    res2 = tp.audit_or_withheld(raw)
    assert res2["verdict"] == tp.WITHHELD_MISSING_INPUTS
    assert "criteria" not in res2
    assert res2["dg_temporal_axis"]["state"] == tp.DG_AXIS_CLEAN
    assert "DG legs" in tp.markdown_withheld(res2)


@pytest.mark.parametrize("name", ["band_percent", "reference_ncells"])
def test_axis_is_bound_to_the_certified_reference(tmp_path, name):
    raw = _power_law_tree(tmp_path, temporal_coeff=0.0)
    axis = tp.dg_temporal_axis(raw)
    assert axis["reference"][name] is not None


# ------------------------------------------------------- committed real data
def _has_committed_legs() -> bool:
    raw = Path(ROOT) / tp.DEFAULT_RAW
    return all((raw / xval.leg_filename(tp.DG_TIME_DEGREE, tp.DG_TIME_GRID, c)).exists()
               for c in tp.DG_TIME_CFLS)


@pytest.mark.skipif(not _has_committed_legs(), reason="committed DG legs absent")
def test_committed_axis_is_clean_and_below_the_band():
    """The published claim: the shipped CFL=0.4 leg error is a *space* error."""
    axis = tp.dg_temporal_axis(ROOT / tp.DEFAULT_RAW)
    assert axis["state"] == tp.DG_AXIS_CLEAN, axis["criteria"]
    d0 = axis["criteria"]["D0"]
    # the time step change must be indistinguishable against the reference band
    assert d0["delta_t_percent"] < d0["band_percent"]
    assert d0["resolvable"] is False
    # and it must be a rounding error against the leg error it is quoted for
    assert axis["criteria"]["D1"]["ratio"] <= tp.POLLUTION_MAX
    assert axis["criteria"]["D2"]["pass"] is True


@pytest.mark.skipif(not _has_committed_legs(), reason="committed DG legs absent")
def test_committed_axis_artifact_is_not_stale():
    """A committed artifact must equal a fresh run (no hand-written numbers)."""
    path = Path(ROOT) / "validation" / "xval_dg_round2" / "artifacts" / "dg_time_axis.json"
    if not path.exists():
        pytest.skip("artifact not generated yet")
    stored = json.loads(path.read_text())
    fresh = tp._jsonable(tp.dg_temporal_axis(ROOT / tp.DEFAULT_RAW))
    assert stored["state"] == fresh["state"]
    for a, b in zip(stored["legs"], fresh["legs"]):
        assert a["rel_l1_percent"] == pytest.approx(b["rel_l1_percent"], rel=1e-12)
    assert (stored["criteria"]["D1"]["ratio"]
            == pytest.approx(fresh["criteria"]["D1"]["ratio"], rel=1e-12))