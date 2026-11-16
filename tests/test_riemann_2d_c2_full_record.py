"""Bind the merged ``full`` 2-D Riemann record (C2 ladder) to its own leg artifacts.

Why this test exists
--------------------
``docs/CASE-DG-CKS1998-2D-RIEMANN.md`` declared gate C2 (mesh self-convergence:
relative L1 between 64x64 and the block-averaged 32x32 density) and then recorded
it as ``NOT_RUN`` for as long as the case existed, because the ``full`` profile was
one ~1 h command that never fit a runnable budget.  The ladder is now sharded into
per-leg jobs (``--leg D N``) whose artifacts are ingested with checksums and
merged by ``--merge-legs``; this module makes sure the merged record cannot
*assert* convergence that its leg files do not support:

1. **Leg arrays -> convergence.**  ``self_convergence_l1`` is recomputed from
   ``legs/<stem>.npz`` cell-mean arrays and must equal the recorded number for
   every pair (P1 and P2 separately).  A hand-typed L1 value, or a record merged
   from a different mesh, fails here.
2. **Rows -> verdicts.**  ``judge(runs, convergence)`` must reproduce every
   committed verdict, so C2 cannot be PASS while its own numbers say otherwise.
3. **C2 is scored.**  ``check_state != 'not_run'``, the tolerance is the
   pre-declared ``GATES['C2_self_convergence_l1']``, ``passed`` is exactly
   ``worst < tolerance``, and worst really is the max over both ladders (the
   inherited code path let P2 overwrite P1 under one shared key).
4. **Record -> document.**  The document's ``DOC-NUMBERS-FULL`` block equals the
   rendered block byte for byte.
5. **Merge refuses a partial ladder.**  ``merge_legs`` on a directory holding one
   leg raises ``FileNotFoundError`` naming the missing stems instead of scoring
   "worst over whatever happened to exist".
6. **Ingest refuses a corrupted artifact.**  A base64 payload whose sha256 does
   not match its declared digest is rejected, so a truncated cluster capture
   cannot become the record's evidence.

Run: ``python -m pytest tests/test_riemann_2d_c2_full_record.py -q``
"""
from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.riemann_2d_cks1998 import (
    DOC_BLOCK_FULL_BEGIN,
    FULL_RECORD_NAME,
    GATES,
    judge,
    leg_stem,
    merge_legs,
    render_doc_numbers_block_full,
    self_convergence_l1,
)
from scripts.c2_leg_ingest import _payloads, ingest

RESULTS = Path("docs/benchmarks/results/riemann_2d_cks1998")
RECORD_PATH = RESULTS / FULL_RECORD_NAME
DOC_PATH = Path("docs/CASE-DG-CKS1998-2D-RIEMANN.md")
FINAL_TIME = 0.25
CFL = 0.2


@pytest.fixture(scope="module")
def record():
    if not RECORD_PATH.is_file():
        pytest.skip(f"{RECORD_PATH} not produced yet (needs --merge-legs)")
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def _cell_means(stem):
    """Stored cell-mean density of one leg (skip rather than fake a value)."""
    path = RESULTS / "legs" / f"{stem}.npz"
    if not path.is_file():
        pytest.skip(f"{path} missing")
    return torch.from_numpy(np.load(path)["cell_means_rho"])


def test_convergence_is_recomputed_from_leg_arrays(record):
    """Every recorded L1 self-distance must fall out of the stored leg arrays."""
    pairs = record["self_convergence_by_degree"]
    assert set(pairs) == {"p1", "p2"}, "both degree ladders must be recorded"
    for degree, ladder in pairs.items():
        for pair, value in ladder.items():
            coarse, fine = (int(x) for x in pair.split("_vs_"))
            recomputed = self_convergence_l1(
                _cell_means(f"{degree}_{coarse}_t{FINAL_TIME:g}_cfl{CFL:g}"),
                _cell_means(f"{degree}_{fine}_t{FINAL_TIME:g}_cfl{CFL:g}"),
            )
            assert recomputed == pytest.approx(value, rel=1e-12, abs=0.0), (
                f"{degree} {pair}: recorded {value!r} but legs give {recomputed!r}")
            assert record["convergence"][f"{degree}_{pair}"] == pytest.approx(value)


def test_verdicts_follow_from_rows_and_convergence(record):
    """``judge`` must reproduce the committed verdict block from the rows."""
    verdicts = judge(record["runs"], convergence=record["convergence"],
                     gates=record["gates"])
    assert set(verdicts) == set(record["verdicts"])
    for gate, row in verdicts.items():
        committed = record["verdicts"][gate]
        assert committed.get("check_state") == row.get("check_state"), gate
        assert committed["passed"] == row["passed"], gate
        for key, value in row.items():
            if isinstance(value, float):
                assert committed[key] == pytest.approx(value, rel=1e-12), (gate, key)
            elif isinstance(value, dict):
                assert committed[key] == value, (gate, key)
            elif key != "detail":
                assert committed[key] == value, (gate, key)


def test_c2_is_scored_not_not_run(record):
    """The gate that used to be NOT_RUN is scored, over both ladders, vs the gate."""
    c2 = record["verdicts"]["C2_self_convergence"]
    assert c2.get("check_state") != "not_run"
    assert c2["tolerance"] == GATES["C2_self_convergence_l1"]
    worst = c2["worst_relative_l1"]
    assert c2["passed"] is (worst < c2["tolerance"])
    assert worst == pytest.approx(max(record["convergence"].values()), rel=1e-15)
    assert set(c2["per_pair"]) == {"p1_32_vs_64", "p2_32_vs_64"}, (
        "both ladders must enter the worst-case score (the inherited code keyed "
        "them identically and dropped one)")
    assert record["degrees"] == [1, 2]
    assert record["meshes"] == [32, 64]
    assert all(int(v) > 0 for v in record["steps_by_leg"].values())
    assert record["final_time"] == pytest.approx(FINAL_TIME)
    assert record["cfl"] == pytest.approx(CFL)
    # Honest scope: the full profile has no periodic row, so C5 stays NOT_RUN and
    # must say so rather than inherit the quick profile's PASS.
    assert record["verdicts"]["C5_periodic_drift"]["check_state"] == "not_run"


def test_document_quotes_the_rendered_full_block(record):
    text = DOC_PATH.read_text(encoding="utf-8")
    block = render_doc_numbers_block_full(record)
    assert DOC_BLOCK_FULL_BEGIN in text
    assert block in text, ("DOC-NUMBERS-FULL block drifted from "
                          f"{RECORD_PATH}; re-run --merge-legs --sync-doc")


def test_leg_files_named_in_provenance_exist_and_match(record):
    for leg in record["provenance"]["legs"]:
        stem = leg["stem"]
        assert (RESULTS / "legs" / f"{stem}.json").is_file(), stem
        assert (RESULTS / "legs" / f"{stem}.npz").is_file(), stem
        payload = json.loads((RESULTS / "legs" / f"{stem}.json").read_text())
        assert payload["record"]["final_time"] == pytest.approx(FINAL_TIME)
        assert payload["record"]["cfl"] == pytest.approx(CFL)
        assert payload["arrays"] == f"{stem}.npz"


def test_merge_refuses_a_partial_ladder(tmp_path):
    """One leg present is not a ladder: the merge must name what is missing."""
    if not (RESULTS / "legs").is_dir():
        pytest.skip("committed legs not available")
    one = sorted((RESULTS / "legs").glob(f"p1_32_t{FINAL_TIME:g}_cfl{CFL:g}.*"))
    if len(one) != 2:
        pytest.skip("committed legs not available")
    for path in one:
        shutil.copy(path, tmp_path / path.name)
    with pytest.raises(FileNotFoundError) as error:
        merge_legs(tmp_path, final_time=FINAL_TIME, cfl=CFL)
    message = str(error.value)
    for missing in (leg_stem(1, 64, FINAL_TIME, CFL), leg_stem(2, 32, FINAL_TIME, CFL),
                    leg_stem(2, 64, FINAL_TIME, CFL)):
        assert missing in message


def test_ingest_rejects_a_corrupted_payload(tmp_path):
    """A truncated capture must fail the checksum, not silently become evidence."""
    blob = b'{"record": {}}'
    digest = hashlib.sha256(blob).hexdigest()
    lines = [f"LEG_JSON:p1_32_t0.25_cfl0.2:{digest}",
             base64.b64encode(blob).decode("ascii")]
    assert list(_payloads(lines)), "control: an intact block must parse"
    log = tmp_path / "capture.log"
    log.write_text("\n".join([lines[0], base64.b64encode(
        b'{"record": {"x": 1}}').decode("ascii")]), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ingest([log], tmp_path / "out", verbose=False)
    log.write_text("\n".join(lines), encoding="utf-8")
    written = ingest([log], tmp_path / "out", verbose=False)
    assert (tmp_path / "out" / "legs" / "p1_32_t0.25_cfl0.2.json").read_bytes() == blob
    assert len(written) == 1