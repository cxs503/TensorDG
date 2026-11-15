"""Tests for the sharded replay driver ``tools/provenance_certify.py``.

The pointed question these tests answer is the one the C11 gate depends on:
when the outstanding replays are split across parallel jobs, is the split a
*partition* of the outstanding set?  If two shards replay the same checkpoint the
work is wasted, and if a shard silently drops one the gate keeps reporting
``unverified`` -- which is exactly the failure mode rounds 10-12 kept hitting
("the number is queued" is not "the number landed").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import provenance_certify as pc  # noqa: E402


def record(key, cells=32, degree=2, final_time=1.0, cfl=0.2, legacy=True):
    return {"task_key": key,
            "reference_model": None if legacy else "min_image",
            "config": {"task": "dg", "cells": cells, "degree": degree,
                       "final_time": final_time, "cfl": cfl, "strength": 5.0},
            "errors": {}, "max_percent": 0.0}


KEYS = ["dg_d2_n32_t1_cfl0.2", "dg_d2_n32_t5_cfl0.2", "dg_d2_n32_t10_cfl0.2",
        "dg_d2_n64_t1_cfl0.2", "dg_d2_n64_t2_cfl0.2", "dg_d2_n64_t10_cfl0.2"]
RECORDS = {
    "dg_d2_n32_t1_cfl0.2": record(KEYS[0]),
    "dg_d2_n32_t5_cfl0.2": record(KEYS[1], final_time=5.0),
    "dg_d2_n32_t10_cfl0.2": record(KEYS[2], final_time=10.0),
    "dg_d2_n64_t1_cfl0.2": record(KEYS[3], cells=64),
    "dg_d2_n64_t2_cfl0.2": record(KEYS[4], cells=64, final_time=2.0),
    "dg_d2_n64_t10_cfl0.2": record(KEYS[5], cells=64, final_time=10.0),
}


def shards(total):
    return [pc.partition(KEYS, RECORDS, index, total) for index in range(total)]


@pytest.mark.parametrize("total", [1, 2, 3, 6, 8])
def test_shards_partition_the_outstanding_set(total):
    """Union = the outstanding set, pairwise disjoint: no checkpoint lost or doubled."""
    buckets = shards(total)
    flat = [key for bucket in buckets for key in bucket]
    assert sorted(flat) == sorted(KEYS), f"shards {buckets} are not a partition"
    assert len(flat) == len(set(flat)), "a checkpoint was assigned to two shards"


def test_shard_index_is_one_based_and_validated():
    """``--shard 0/N`` and ``i > n`` must be rejected, not silently rounded."""
    with pytest.raises(ValueError):
        pc.partition(KEYS, RECORDS, -1, 3)
    with pytest.raises(ValueError):
        pc.partition(KEYS, RECORDS, 3, 3)
    with pytest.raises(ValueError):
        pc.partition(KEYS, RECORDS, 0, 0)


def test_shards_are_cost_balanced():
    """The expensive n64_t10 checkpoint must not be stacked with another big one."""
    buckets = shards(2)
    loads = [sum(pc.cost_proxy(RECORDS[key]) for key in bucket) for bucket in buckets]
    assert max(loads) / min(loads) <= 2.0, f"unbalanced shards: {loads}"


def test_cost_proxy_orders_by_work():
    """Sanity: work grows with cells and with final_time, falls with cfl."""
    base = pc.cost_proxy(RECORDS["dg_d2_n32_t1_cfl0.2"])
    assert pc.cost_proxy(RECORDS["dg_d2_n64_t1_cfl0.2"]) > base
    assert pc.cost_proxy(RECORDS["dg_d2_n32_t10_cfl0.2"]) > base
    loose = record("x", cfl=0.4)
    assert pc.cost_proxy(loose) < base


def test_legacy_keys_ignores_records_that_know_their_reference():
    records = [RECORDS["dg_d2_n32_t1_cfl0.2"], record("modern", legacy=False)]
    assert pc.legacy_keys(records) == ["dg_d2_n32_t1_cfl0.2"]


def test_remaining_skips_already_certified(tmp_path, monkeypatch):
    """``--remaining`` must not re-pay for landed certificates."""
    output_dir = tmp_path / "case"
    results = output_dir / "results"
    results.mkdir(parents=True)
    for key in KEYS:
        (results / f"{key}.json").write_text(json.dumps(RECORDS[key]))
    certified = KEYS[:4]
    (output_dir / "provenance_certificates.json").write_text(
        json.dumps({"certificates": {key: {"replay_delta": 0.0} for key in certified},
                    "reference_model": "min_image"}))

    attempted = []

    def fake_replay(rec):
        attempted.append(rec["task_key"])
        return {"task_key": rec["task_key"], "reference_model": "min_image",
                "config": rec["config"], "stored_max_percent": 0.0,
                "replayed_max_percent": 0.0,
                "absolute_delta": {}, "replay_delta": 0.0,
                "wall_seconds": 0.0, "steps": 1, "certified_at": "now",
                "tolerance": 1e-9}

    monkeypatch.setattr(pc, "replay", fake_replay)
    rc = pc.main(["--output-dir", str(output_dir), "--remaining",
                  "--max-seconds", "100000"])
    assert rc == 0
    # order follows the sorted result files, so compare as a set
    assert set(attempted) == set(KEYS[4:]), f"replayed {attempted}, expected only the uncertified"
    bundle = json.loads((output_dir / "provenance_certificates.json").read_text())
    assert set(bundle["certificates"]) == set(KEYS)