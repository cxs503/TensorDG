"""Tests for :mod:`benchmarks.xval_dg_round2_time_pollution`.

What is asserted, and why in this order
--------------------------------------
* **estimator unit tests** first: a synthetic fixed-grid CFL series whose
  temporal error is an *exact* power law ``u(cfl) = u_exact + c * cfl**p`` (the
  Richardson step is then exact, not approximate), so the temporal order
  recovery, the correction and the pollution ratio are verified before the
  machinery touches CFD output;
* **refusal tests**: a temporal ladder that has bottomed out (two identical
  profiles, or a *growing* difference) carries no order information and must be
  refused, not extrapolated;
* **decision-rule tests**: the ``CLEAN/MARGINAL/POLLUTED`` boundaries are pinned
  at the predeclared constants, so a later re-tuning of a threshold cannot
  silently change a published verdict;
* **committed-data tests** (skipped when the extra CFL profiles are absent): the
  audit runs end to end on the shipped raw profiles, and the numbers it reports
  for the reference grid are exactly the ones the Markdown artifact carries --
  the document is rendered from the audit dict, never hand-written.

Run: ``pytest -q tests/test_xval_dg_round2_time_pollution.py``
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks import xval_dg_round2_time_pollution as tp

RAW = Path(tp.DEFAULT_RAW)


# ------------------------------------------------------------ synthetic series
def _series(order: float, const: float, cfls=tp.CFL_LEVELS, ncells: int = 64):
    """A fixed-grid CFL series with an exact temporal power law.

    ``u(cfl) = exact + const * cfl**order``; cell averages of a linear profile
    are sampled at midpoints, so there is no spatial component at all and any
    difference between two members is purely temporal.
    """
    x = (np.arange(ncells) + 0.5) * (10.0 / ncells) - 5.0
    exact = 1.0 + 0.2 * x
    return [{"ncells": ncells, "cfl": float(c), "x": x,
             "field": exact + const * float(c) ** order, "meta": {}, "file": f"synth_{c}"}
            for c in sorted(cfls, reverse=True)]


def test_cfl_tag_filenames():
    assert tp.cfl_tag(0.3) == "0p3"
    assert tp.cfl_tag(0.15) == "0p15"
    assert tp.cfl_tag(0.075) == "0p075"
    assert tp.cfl_tag(1.0) == "1"
    assert tp.level_path("raw", 1600, 0.15).name == "fv_cfl0p15_n1600.npz"


@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_temporal_order_recovered_exactly(order):
    rep = tp.temporal_report(_series(order=order, const=0.01))
    assert rep["temporal_order"] == pytest.approx(order, rel=1e-12)
    assert rep["monotone"] is True
    assert rep["ncells"] == 64
    # differences are strictly decreasing, as a contracting temporal ladder must be
    d = [row["rel_l1"] for row in rep["differences"]]
    assert d[0] > d[1] > 0.0


def test_temporal_difference_is_pure_time_error():
    """``delta_t`` equals the analytic change of the temporal term alone.

    The series' differences share one denominator (the finest member's L1 norm),
    so the arithmetic is checkable in closed form rather than only self-consistently.
    """
    const, order = 0.02, 2.0
    series = _series(order=order, const=const)
    rep = tp.temporal_report(series)
    x = (np.arange(64) + 0.5) * (10.0 / 64) - 5.0
    norm = float(np.sum(np.abs(1.0 + 0.2 * x + const * 0.075 ** order)))
    expected = const * (0.3 ** order - 0.15 ** order) * 64 / norm
    assert rep["delta_t_percent"] == pytest.approx(100.0 * expected, rel=1e-10)
    # the order is exact because the two ratios share that denominator
    assert rep["temporal_order"] == pytest.approx(order, rel=1e-14)


def test_time_correction_kills_the_temporal_error():
    """Richardson on a pure power law recovers the ``CFL -> 0`` limit exactly."""
    series = _series(order=2.0, const=0.05)
    rep = tp.temporal_report(series)
    corrected = tp.time_corrected_field(series, rep["temporal_order"])
    x = (np.arange(64) + 0.5) * (10.0 / 64) - 5.0
    assert np.allclose(corrected, 1.0 + 0.2 * x, rtol=0, atol=1e-13)
    # ... and the correction is a real move, not a no-op
    assert tp.rel_l1(corrected, series[-1]["field"]) > 0.0


def test_time_correction_requires_a_single_grid():
    series = _series(order=2.0, const=0.05)
    series[-1] = {**series[-1], "field": np.resize(series[-1]["field"], 128),
                  "ncells": 128}
    with pytest.raises(ValueError, match="one grid|same grid|fixed grid"):
        tp.temporal_report(series)


def test_bottomed_out_temporal_series_is_refused():
    """Two identical profiles carry no order; the correction must refuse."""
    series = _series(order=2.0, const=0.0)  # const = 0 -> all members identical
    rep = tp.temporal_report(series)
    assert rep["temporal_order"] is None
    with pytest.raises(ValueError):
        tp.time_corrected_field(series, rep["temporal_order"])


def test_growing_temporal_difference_is_flagged_not_monotone():
    """A temporal ladder whose increments *grow* is not asymptotic."""
    series = _series(order=-1.0, const=0.01)  # error grows as CFL decreases
    rep = tp.temporal_report(series)
    assert rep["monotone"] is False
    assert rep["differences"][1]["rel_l1"] > rep["differences"][0]["rel_l1"]


# ------------------------------------------------------------- decision rule
@pytest.mark.parametrize("ratio,state", [
    (0.0, "CLEAN"), (0.05, "CLEAN"), (tp.POLLUTION_MAX, "CLEAN"),
    (tp.POLLUTION_MAX * 1.0001, "MARGINAL"), (tp.POLLUTION_MARGINAL, "MARGINAL"),
    (tp.POLLUTION_MARGINAL * 1.0001, "POLLUTED"), (5.0, "POLLUTED"),
    (None, "WITHHELD"),
])
def test_classify_pollution_boundaries(ratio, state):
    assert tp.classify_pollution(ratio) == state


def test_thresholds_are_the_predeclared_constants():
    """Guards against a later, result-driven re-tuning of the thresholds."""
    assert tp.POLLUTION_MAX == 0.10
    assert tp.POLLUTION_MARGINAL == 0.25
    assert tp.P_MIN_T == 0.5
    assert tp.BAND_SHIFT_MAX == 0.25
    assert tp.LEG_STABILITY_MAX_PP == 0.5
    assert tp.CFL_LEVELS == (0.3, 0.15, 0.075)
    assert tp.REFERENCE_GRID == 3200


# ------------------------------------------------------------- committed data
_REQUIRED = [tp.level_path(RAW, n, c) for n in tp.BASE_LEVELS for c in tp.CFL_LEVELS]
_REQUIRED.append(tp.level_path(RAW, tp.EXTRA_RATIO_LEVELS[0], 0.15))

requires_cfl_profiles = pytest.mark.skipif(
    not all(p.exists() for p in _REQUIRED),
    reason="CFL-refined FV profiles not generated yet "
           "(validation/xval_dg_round2/gen_fv_level.py --cfl 0.15/0.075)")


@requires_cfl_profiles
def test_committed_audit_runs_end_to_end():
    res = tp.audit(RAW)
    assert res["verdict"] in ("TIME_CLEAN", "TIME_MARGINAL", "TIME_POLLUTED")
    assert set(res["criteria"]) == {"T0", "T1", "T2", "T3", "T4"}
    # every grid in the CFL series must report a real temporal difference
    for row in res["grids"]:
        assert row["delta_t"] > 0.0 and np.isfinite(row["delta_t"])
        if row["ncells"] in tp.BASE_LEVELS:
            assert row["temporal_order"] is not None
    # the headline ratio is the reference-grid ratio, not some other grid's
    ref = [r for r in res["grids"] if r["ncells"] == tp.REFERENCE_GRID][0]
    assert res["headline"]["pollution_ratio"] == pytest.approx(ref["pollution_ratio"])


@requires_cfl_profiles
def test_markdown_is_rendered_from_the_audit_not_hand_written():
    res = tp.audit(RAW)
    md = tp.markdown(res)
    assert res["verdict"] in md
    assert f"N={tp.REFERENCE_GRID}" in md or str(tp.REFERENCE_GRID) in md
    for row in res["grids"]:
        assert str(row["ncells"]) in md
    for leg in res["criteria"]["T4"]["legs_raw"]:
        assert leg["configuration"].replace("DG ", "") in md


@requires_cfl_profiles
def test_time_corrected_ladder_is_built_from_the_same_grids():
    levels, used = tp.time_corrected_levels(RAW, tp.BASE_LEVELS)
    assert [lv["ncells"] for lv in levels] == list(tp.BASE_LEVELS)
    for lv, n in zip(levels, tp.BASE_LEVELS):
        assert lv["field"].size == n
        assert np.all(np.isfinite(lv["field"]))
        assert lv["field"].min() > 0.0
    assert set(used) == set(tp.BASE_LEVELS)


@requires_cfl_profiles
def test_artifacts_write_json_and_markdown(tmp_path):
    res = tp.audit(RAW)
    out = tp.write_artifacts(res, tmp_path)
    import json
    blob = json.loads(Path(out["json"]).read_text())
    assert blob["verdict"] == res["verdict"]
    assert Path(out["md"]).read_text().strip().startswith("#")

# ------------------------------------------- end-to-end on a synthetic raw tree
BLOCK_GRID = 200  #: coarsest grid; every profile is constant on each 200-cell block


def _block_field(blocks: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Sample a per-block constant field on ``grid`` cells (exact coarsening)."""
    n = grid.size
    return blocks[(np.arange(n) * BLOCK_GRID // n)]


def _synthetic_rawdir(root: Path, k_time: float, *, p_time: float = 2.0,
                      a_space: float = 0.1, p_space: float = 1.2) -> Path:
    """A raw directory obeying ``spatial ~ N^-p_s`` and ``time ~ N^-p_t CFL^p_t``.

    Block-constant structure makes every cell average exact under coarsening, so
    the ladder differences are pure power laws and the verdict is decidable in
    closed form -- the pipeline is then checked against ground truth instead of
    against itself.
    """
    root.mkdir(parents=True, exist_ok=True)
    idx = np.arange(BLOCK_GRID)
    w = 1.0 + 0.3 * np.sin(2.0 * np.pi * idx / BLOCK_GRID)

    def field(n: int, cfl: float) -> np.ndarray:
        x = (np.arange(n) + 0.5) * (10.0 / n) - 5.0
        base = 1.0 + 0.2 * x
        wsel = _block_field(w, np.arange(n))
        ratio = n / BLOCK_GRID
        return (base + a_space * ratio ** (-p_space) * wsel
                + k_time * ratio ** (-p_time) * float(cfl) ** p_time * wsel)

    for n in (400, 800, 1600, 3200, 6400):
        for cfl in tp.CFL_LEVELS:
            f = field(n, cfl)
            np.savez(tp.level_path(root, n, cfl), x=np.arange(n, dtype=float),
                     rho=f, rho_u=f, rho_e=f, meta="{}")
    # DG legs: same block structure, half the spatial error, on the shipped grids
    for degree, ncells, cfl in tp.xval.DG_LEGS:
        wsel = _block_field(w, np.arange(ncells))
        x = (np.arange(ncells) + 0.5) * (10.0 / ncells) - 5.0
        f = 1.0 + 0.2 * x + 0.5 * a_space * (ncells / BLOCK_GRID) ** (-p_space) * wsel
        np.savez(Path(root) / tp.xval.leg_filename(degree, ncells, cfl),
                 x=x, rho=f, rho_u=f, rho_e=f, meta="{}")
    return root


def test_audit_end_to_end_clean(tmp_path):
    """A tree whose time leg is ~1 % of the spatial leg must report TIME_CLEAN."""
    raw = _synthetic_rawdir(tmp_path / "raw", k_time=1.0)
    res = tp.audit(raw)
    ref = [g for g in res["grids"] if g["ncells"] == tp.REFERENCE_GRID][0]
    assert ref["pollution_ratio"] < tp.POLLUTION_MAX
    assert res["criteria"]["T0"]["pass"] is True
    assert res["criteria"]["T1"]["state"] == "CLEAN"
    # the synthetic temporal law is an exact power law in CFL at fixed N
    assert res["criteria"]["T2"]["temporal_order"] == pytest.approx(2.0, rel=1e-8)
    assert res["verdict"] == "TIME_CLEAN", res["headline"]
    assert res["accepted"] is True


def test_audit_end_to_end_polluted(tmp_path):
    """A tree whose time error rivals the spatial error must NOT be accepted."""
    raw = _synthetic_rawdir(tmp_path / "raw", k_time=2000.0)
    res = tp.audit(raw)
    assert res["criteria"]["T1"]["state"] == "POLLUTED"
    assert res["criteria"]["T1"]["pass"] is False
    assert res["verdict"] == "TIME_POLLUTED"
    assert res["accepted"] is False


def test_audit_artifacts_and_figures_render(tmp_path):
    raw = _synthetic_rawdir(tmp_path / "raw", k_time=1.0)
    res = tp.audit(raw)
    out = tp.write_artifacts(res, tmp_path / "art")
    figs = tp.figures(res, raw, tmp_path / "art")
    assert len(figs) == 3
    for f in figs:
        p = Path(f)
        assert p.exists() and p.stat().st_size > 1000, f
    md = Path(out["md"]).read_text()
    assert res["verdict"] in md and "T0" in md and "DG P1 N=200 CFL=0.4" in md


# ------------------------------------------------- partial CFL series (2-level)
def _write_partial_tree(root: Path, ncells: int, cfls) -> Path:
    """Materialise a fixed-grid series with only the given CFL members."""
    x = (np.arange(ncells) + 0.5) * (10.0 / ncells) - 5.0
    for c in cfls:
        f = 1.0 + 0.2 * x + 0.05 * float(c) ** 2.0
        np.savez(tp.level_path(root, ncells, c), x=x, rho=f, rho_u=f, rho_e=f, meta="{}")
    return root


def test_two_level_series_yields_the_ratio_but_no_order(tmp_path):
    """The extra ratio grid ships two CFL members: R is computable, p_t is not.

    This is the documented shape of the finest ladder grid (``EXTRA_RATIO_LEVELS``
    carries CFL 0.3/0.15 only -- a third level costs ~70 CPU-minutes).  Before
    this contract was pinned, ``temporal_series`` demanded the third level and
    the whole audit died with ``FileNotFoundError`` on a dataset its own
    ``_REQUIRED`` list called complete.
    """
    root = _write_partial_tree(tmp_path, 6400, (0.3, 0.15))
    series = tp.temporal_series(root, 6400, require=2)
    assert [s["cfl"] for s in series] == [0.3, 0.15]
    rep = tp.temporal_report(series)
    assert rep["levels_present"] == 2
    assert rep["delta_t"] > 0.0          # the leading-order temporal error
    assert rep["temporal_order"] is None  # one difference carries no order
    assert rep["monotone"] is True        # nothing to contradict monotonicity
    with pytest.raises(ValueError):
        tp.time_corrected_field(series, rep["temporal_order"])


def test_base_grid_still_requires_the_full_series(tmp_path):
    """A corrected (BASE) grid must not fall back to a two-member series."""
    root = _write_partial_tree(tmp_path, 3200, (0.3, 0.15))
    with pytest.raises(FileNotFoundError, match="0p075"):
        tp.temporal_series(root, 3200, require=len(tp.CFL_LEVELS))


def test_single_level_is_refused_loudly(tmp_path):
    root = _write_partial_tree(tmp_path, 800, (0.3,))
    with pytest.raises(FileNotFoundError):
        tp.temporal_series(root, 800)


def test_present_levels_are_used_in_coarse_to_fine_order(tmp_path):
    """An incomplete ladder mid-grid is still ordered, never silently reindexed."""
    root = _write_partial_tree(tmp_path, 1600, (0.3, 0.075))
    series = tp.temporal_series(root, 1600)
    assert [s["cfl"] for s in series] == [0.3, 0.075]
    rep = tp.temporal_report(series)
    assert rep["differences"][0]["ratio"] == pytest.approx(4.0)
