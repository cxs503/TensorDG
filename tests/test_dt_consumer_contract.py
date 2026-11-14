"""Gate: no in-repo consumer may treat the legacy ``meta["dt"]`` as a step.

Background — ``docs/CASE-LAX-DG-dt-metadata-contract.md`` §⑤.1 handed off one
item on purpose: "另有 ≥6 处基准/测试把 ``meta["dt"]`` 当步长用 …… 逐个审计是
后续单的独立工作项".  ``scripts/audit_dt_consumers.py`` closes it mechanically;
these tests make the closure a regression gate instead of a one-off report.

The tests assert *invariants*, not merely that the script runs:

* every ``["dt"]`` read in ``tensordg/ benchmarks/ tests/ scripts/ tools/`` is
  classified into one of the six usage kinds (no silent unclassified growth);
* no read that carries **step semantics** sits in a file that calls a
  *residual-step* solver, unless the line is waived with a written reason
  (waivers are pinned by file **and** line number);
* the producer classification is cross-checked against the running solvers:
  in the residual family ``steps * dt_mean == final_time`` to machine
  precision while ``dt`` does not, i.e. the audit is measuring the real defect;
* the committed document and its figure are regenerated from the same code
  path (no hand-edited numbers).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_dt_consumers as audit  # noqa: E402

KINDS = {"PUBLISH", "REPORT", "STEP_ARITHMETIC", "STEP_ARGUMENT", "INVARIANT", "OTHER"}
STEP_SEMANTICS = {"STEP_ARITHMETIC", "STEP_ARGUMENT", "INVARIANT"}


@pytest.fixture(scope="module")
def sites():
    return audit.consumer_sites()


def test_every_read_is_classified(sites):
    assert sites, "the scanner found no consumer at all -- the scan itself broke"
    unclassified = [s for s in sites if s["kind"] not in KINDS]
    assert not unclassified, unclassified
    # the scan must actually see step-semantics reads, otherwise the gate below
    # would be vacuously green
    kinds = {s["kind"] for s in sites}
    assert STEP_SEMANTICS <= kinds, f"step-semantics reads disappeared: {sorted(kinds)}"


def test_no_unwaived_step_semantics_read_of_a_residual_dt(sites):
    violations = [s for s in sites if s["violation"]]
    assert not violations, "\n".join(
        f"{s['file']}:{s['line']} [{s['kind']}] {s['text']}"
        f"   (residual producers: {s['residual_producers']})" for s in violations
    )


def test_waivers_are_pinned_to_the_line_they_were_reviewed_on(sites):
    waived = {f"{s['file']}:{s['line']}" for s in sites if s["waived"]}
    declared = {f"{path}:{line}" for path, (line, _reason) in audit.WAIVERS.items()}
    # an entry whose line moved (code drifted) must fail loudly, not silently pass
    assert declared <= waived, f"stale waiver lines: {sorted(declared - waived)}"
    for path, (line, reason) in audit.WAIVERS.items():
        assert reason and len(reason) > 20, f"waiver {path}:{line} has no usable reason"


def test_producer_families_are_measured_not_asserted():
    table = {row["module"]: row for row in audit.producer_source_table()}
    assert table["euler_1d"]["semantics"] == "residual"
    assert table["swe_1d"]["semantics"] == "residual"
    assert table["elastic_1d"]["semantics"] == "fixed/average"

    meta = audit._meta(audit.solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
                                          ncells=100, final_time=0.05, cfl=0.4))
    final_time, steps = 0.05, int(meta["steps"])
    # the honest field is exactly the run step ...
    assert meta["dt_mean"] == pytest.approx(final_time / steps, rel=0, abs=0)
    # ... and the legacy field is not (this is the defect the audit measures)
    residual_share = abs(meta["dt"] - meta["dt_mean"]) / meta["dt_mean"]
    assert residual_share > 0.01, residual_share
    assert meta["dt"] == meta["dt_last"]


def test_cfl_halving_halves_the_mean_step_but_not_the_legacy_field():
    rows = audit.cfl_ladder()
    ratios = [r["ratio_mean"] for r in rows if "ratio_mean" in r]
    assert ratios and all(1.8 <= r <= 2.2 for r in ratios), ratios
    legacy = [r["ratio_last"] for r in rows if "ratio_last" in r]
    assert any(abs(r - 2.0) > 0.5 for r in legacy), legacy


def test_gate_mode_exits_zero_on_the_current_tree():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_dt_consumers.py"), "--check",
         "--skip-measure"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT),
             "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "MPLBACKEND": "Agg"},
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]


def test_committed_document_matches_the_audit_artifact():
    doc = ROOT / "docs" / "CASE-LAX-DG-dt-consumer-audit.md"
    art = ROOT / "artifacts" / "dt_consumer_audit" / "audit.json"
    assert doc.exists() and art.exists()
    payload = json.loads(art.read_text())
    text = doc.read_text()
    assert f"{payload['counts']['sites']} 个读取点" in text
    assert payload["counts"]["violations"] == 0
    # Figure is embedded as data-URI, so the document is self-contained.
    assert "data:image/png;base64," in text
    fig = ROOT / "docs" / "benchmarks" / "results" / "dt_consumer_audit" / "fig_dt_residual_share.png"
    assert fig.exists() and fig.stat().st_size > 10_000