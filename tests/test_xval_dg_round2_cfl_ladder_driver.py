"""Invariants of the CFL-ladder producer (``gen_cfl_ladder.py``).

The time-pollution audit (``benchmarks.xval_dg_round2_time_pollution``) reads its
real-data branch from files with a *hard-coded* name pattern
(``fv_cfl<cfl tag>_n<ncells>.npz``) and refuses to invent anything when a level
is absent.  A producer that names its output differently, or that skips one of
the levels the audit needs, therefore does not fail loudly -- it silently leaves
the whole audit at ``WITHHELD``.  These tests pin the contract between the two:

1. the ``(ncells, cfl)`` table is *exactly* the cross product the audit consumes;
2. the file-name spelling of both modules agrees (two independent ``cfl_tag``
   implementations, one file name -- they must not drift);
3. usability gating rejects missing / corrupt / inadmissible profiles, so a
   truncated cluster artifact cannot be mistaken for a level;
4. ``--shard i --nshards n`` partitions the table (no case run twice, no case
   dropped) and is idempotent (a present level is not re-solved).

No CFD is run here: the solver call is monkeypatched.
"""
from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

from benchmarks import xval_dg_round2_time_pollution as tp

ROOT = Path(__file__).resolve().parents[1]


def _load_driver():
    path = ROOT / "validation" / "xval_dg_round2" / "gen_cfl_ladder.py"
    spec = importlib.util.spec_from_file_location("gen_cfl_ladder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drv():
    return _load_driver()


def _write_level(path: Path, rho):
    rho = np.asarray(rho, dtype=float)
    np.savez(path, x=np.linspace(-5.0, 5.0, rho.size), rho=rho,
             rho_u=np.zeros_like(rho), rho_e=np.ones_like(rho),
             meta='{"ncells": %d}' % rho.size)
    return path


# ---------------------------------------------------------------- case table
def test_case_table_is_the_cross_product_the_audit_consumes(drv):
    """Every level the audit needs has a producer, and no producer is idle.

    The audit's requirement is not uniform: the grids inside the corrected
    spatial ladder (``BASE_LEVELS``) need all three CFL members because the
    temporal order comes from *two* successive differences, while the extra
    ratio grid only needs the coarsest pair to report the pollution ratio R.
    """
    needed = {(int(n), float(c)) for n in tp.BASE_LEVELS for c in tp.CFL_LEVELS}
    needed |= {(int(n), float(c))
               for n in tp.EXTRA_RATIO_LEVELS
               for c in sorted(tp.CFL_LEVELS, reverse=True)[:2]}
    cases = {(int(n), float(c)) for n, c in drv.CASES}
    # levels already committed in the repository need no producer
    shipped = {(n, c) for n, c in needed
               if drv.level_is_usable(drv.target_path(n, c))}
    assert cases <= needed, "the driver produces levels the audit never reads"
    assert needed <= cases | shipped, "a level the audit needs has no producer"


def test_case_table_has_no_duplicates_and_is_ordered_coarse_first(drv):
    keys = [(int(n), float(c)) for n, c in drv.CASES]
    assert len(keys) == len(set(keys))
    assert [n for n, _ in keys] == sorted(n for n, _ in keys), \
        "a partial run must leave the coarse levels behind, not the fine ones"


@pytest.mark.parametrize("cfl", [0.3, 0.15, 0.075, 0.05, 0.0125])
def test_file_name_spelling_agrees_with_the_audit(drv, cfl):
    """Two independent spellings of the CFL tag must produce one file name."""
    assert drv.cfl_tag(cfl) == tp.cfl_tag(cfl)
    assert (drv.target_path(3200, cfl).name
            == tp.level_path(tp.DEFAULT_RAW, 3200, cfl).name)


# ------------------------------------------------------------------ usability
def test_usable_level_round_trips_through_the_audit_loader(drv, tmp_path):
    path = _write_level(tmp_path / "fv_cfl0p15_n800.npz", np.linspace(0.8, 4.6, 800))
    assert drv.level_is_usable(path)
    loaded = tp.load_level(tmp_path, 800, 0.15)
    assert loaded["field"].size == 800 and np.all(loaded["field"] > 0.0)


def test_missing_level_is_not_usable(drv, tmp_path):
    assert not drv.level_is_usable(tmp_path / "fv_cfl0p15_n800.npz")


def test_corrupt_level_is_not_usable(drv, tmp_path):
    path = tmp_path / "fv_cfl0p15_n800.npz"
    path.write_bytes(b"not a numpy archive")
    assert not drv.level_is_usable(path)


@pytest.mark.parametrize("rho", [[0.8, -0.1, 1.0], [0.8, np.nan, 1.0],
                                 [0.8, np.inf, 1.0]])
def test_inadmissible_level_is_not_usable(drv, tmp_path, rho):
    """A negative/NaN density is a blown-up run, not a level."""
    path = _write_level(tmp_path / "fv_cfl0p15_n800.npz", rho)
    assert not drv.level_is_usable(path)


# ------------------------------------------------------------------- sharding
def test_shards_partition_the_case_table(drv, monkeypatch, tmp_path):
    calls = []

    def fake_run_one(ncells, cfl, out):
        calls.append((int(ncells), float(cfl)))
        return {"out": str(out)}

    monkeypatch.setattr(drv, "run_one", fake_run_one)
    monkeypatch.setattr(drv, "level_is_usable", lambda p: False)
    for shard in range(len(drv.CASES)):
        drv.main(["--shard", str(shard), "--nshards", str(len(drv.CASES)),
                  "--rawdir", str(tmp_path)])
    assert sorted(calls) == sorted((int(n), float(c)) for n, c in drv.CASES)
    assert len(calls) == len(set(calls))


def test_present_levels_are_not_re_solved(drv, monkeypatch, tmp_path):
    """Idempotence: re-dispatching a shard whose output landed must not redo CFD."""
    calls = []

    def fake_run_one(ncells, cfl, out):
        calls.append((ncells, cfl))
        return {"out": str(out)}

    def fake_usable(path):
        return Path(path).name == drv.target_path(3200, 0.075).name

    monkeypatch.setattr(drv, "run_one", fake_run_one)
    monkeypatch.setattr(drv, "level_is_usable", fake_usable)
    for shard in range(len(drv.CASES)):
        drv.main(["--shard", str(shard), "--nshards", str(len(drv.CASES)),
                  "--rawdir", str(tmp_path)])
    assert (3200, 0.075) not in calls
    assert len(calls) == len(drv.CASES) - 1


def test_list_reports_presence_without_solving(drv, monkeypatch, tmp_path, capsys):
    import json

    def boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("--list must not invoke the solver")

    monkeypatch.setattr(drv, "run_one", boom)
    assert drv.main(["--list", "--rawdir", str(tmp_path)]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == len(drv.CASES)
    assert all(row["present"] is False for row in rows)


def test_shard_with_more_shards_than_cases_still_covers_everything(drv, monkeypatch,
                                                                  tmp_path):
    calls = []
    monkeypatch.setattr(drv, "run_one",
                        lambda n, c, o: calls.append((n, c)) or {"out": str(o)})
    monkeypatch.setattr(drv, "level_is_usable", lambda p: False)
    for shard in range(len(drv.CASES) + 3):
        drv.main(["--shard", str(shard), "--nshards", str(len(drv.CASES) + 3),
                  "--rawdir", str(tmp_path)])
    assert sorted(calls) == sorted((int(n), float(c)) for n, c in drv.CASES)


def test_every_case_lives_in_exactly_one_shard(drv):
    n = len(drv.CASES)
    for i, (ncells, cfl) in enumerate(drv.CASES):
        owners = [s for s in range(n) if i % n == s]
        assert owners == [i]
    assert len(set(itertools.chain.from_iterable([[i % n] for i in range(n)]))) == n