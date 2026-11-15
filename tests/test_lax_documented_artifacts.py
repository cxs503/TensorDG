"""Contract tests for the ``CASE-LAX-DG`` section (4) artifact inventory.

The canonical report claims "工件的 sha256 由 ``sha256sum`` 直读，重跑可逐位校验".
That claim was false: the table cited ``runs/lax_case_pool/...`` -- a directory the
merge gate never carries into ``main`` -- with the byte counts and hashes of an
earlier figure generation (m16585), while the committed record lives under
``docs/benchmarks/results/lax_case_pool/`` and the figures were re-rendered
since.  A reader could not re-check a single row.

The layers below protect the repair:

1. **row layer** -- every row's byte count and ``sha256[:16]`` must match the
   file on disk, so the table cannot be hand-edited into agreement;
2. **coverage layer** -- the table lists exactly the committed artifacts (no
   stale row, no silently dropped figure);
3. **path layer** -- no ``runs/`` citation survives (untracked path);
4. **non-vacuity layer** -- tampering with a byte count or a path must make the
   report gate fail, i.e. the gate bites.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from scripts.sync_lax_report_numbers import (
    ARTIFACT_DIR, _FIGURE_BLOCK, _INVENTORY_BLOCK, committed_artifacts,
    discrepancies, figure_block, inventory_table,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "CASE-LAX-DG.md"
METRICS = ARTIFACT_DIR / "metrics.json"


def _rows() -> dict[str, tuple[str, int, str]]:
    match = _INVENTORY_BLOCK.search(REPORT.read_text())
    assert match is not None, "section (4) artifact inventory table is missing"
    rows: dict[str, tuple[str, int, str]] = {}
    for line in match.group(0).splitlines()[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows[cells[0].strip("`")] = (cells[1], int(cells[2]), cells[3])
    return rows


# ------------------------------------------------------------------- row layer
def test_every_row_matches_the_bytes_and_sha256_on_disk():
    rows = _rows()
    assert rows, "inventory table has no rows"
    for rel, (_purpose, size, digest) in rows.items():
        path = ROOT / rel
        assert path.is_file(), f"inventory row points at a missing file: {rel}"
        assert path.stat().st_size == size, (
            f"{rel}: table says {size} bytes, disk has {path.stat().st_size}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        assert actual == digest, f"{rel}: table says {digest}, disk has {actual}"


# -------------------------------------------------------------- coverage layer
def test_the_table_lists_exactly_the_committed_artifacts():
    prefix = ARTIFACT_DIR.relative_to(ROOT).as_posix() + "/"
    listed = {rel[len(prefix):] for rel in _rows() if rel.startswith(prefix)}
    assert listed == committed_artifacts(), (
        "inventory must list every committed artifact of the pool record and "
        "nothing else; re-run `python scripts/sync_lax_report_numbers.py`")


def test_figure_list_is_the_committed_figure_set():
    text = REPORT.read_text()
    match = _FIGURE_BLOCK.search(text)
    assert match is not None, "section (4) picture list is missing"
    cited = set(re.findall(r"`([^`]+\.png)`", match.group(0)))
    committed = {p.relative_to(ROOT).as_posix()
                 for p in sorted((ARTIFACT_DIR / "figures").glob("*.png"))}
    assert cited == committed, f"figures cited {sorted(cited)} != {sorted(committed)}"
    count = int(re.match(r"\D*(\d+) 张图", match.group(0)).group(1))
    assert count == len(committed), "the lead sentence miscounts the figures"


def test_recorded_figures_agree_with_the_committed_files():
    metrics = json.loads(METRICS.read_text())
    recorded = set(metrics["figures"])
    on_disk = {p.name for p in (ARTIFACT_DIR / "figures").glob("*.png")}
    assert recorded == on_disk, (
        f"metrics.json records {sorted(recorded)}, committed {sorted(on_disk)}")


# ------------------------------------------------------------------ path layer
def test_no_untracked_runs_path_survives_in_the_inventory():
    for rel in _rows():
        assert not rel.startswith("runs/"), f"untracked path in inventory: {rel}"
    match_fig = _FIGURE_BLOCK.search(REPORT.read_text())
    match_inv = _INVENTORY_BLOCK.search(REPORT.read_text())
    for block in (match_fig.group(0), match_inv.group(0)):
        assert "`runs/" not in block, "untracked `runs/` path in section (4)"
    assert "runs/" not in figure_block() and "runs/" not in inventory_table()


# ----------------------------------------------------------- non-vacuity layer
def test_tampered_byte_count_is_caught():
    text = REPORT.read_text()
    match = _INVENTORY_BLOCK.search(text)
    tampered = text[:match.start()] + match.group(0).replace(
        "| 225846 |", "| 225847 |", 1) + text[match.end():]
    metrics = json.loads(METRICS.read_text())
    assert any("artifact inventory" in p for p in discrepancies(tampered, metrics))


def test_untracked_path_is_caught():
    text = REPORT.read_text()
    tampered = text.replace("docs/benchmarks/results/lax_case_pool/figures/lax_profiles.png",
                            "runs/lax_case_pool/figures/lax_profiles.png", 1)
    metrics = json.loads(METRICS.read_text())
    problems = discrepancies(tampered, metrics)
    assert any("`runs/`" in p or "artifact inventory" in p or "figure block" in p
               for p in problems), problems


def test_report_and_inventory_are_clean():
    metrics = json.loads(METRICS.read_text())
    assert discrepancies(REPORT.read_text(), metrics) == []
    result = subprocess.run(
        [sys.executable, "scripts/sync_lax_report_numbers.py", "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr