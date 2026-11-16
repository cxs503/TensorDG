"""Guards for the CASE-DG-ADVECT-ORDER round-2 evidence bundle.

The round-2 report (``docs/CASE-DG-ADVECT-ORDER-round2.md``) is *generated* from
``validation/advection_order_matched_rk/metrics.json`` by
``publish_case_dg_advect_order_round2.py``.  That arrangement only stays honest
if the generated file cannot drift away from the evidence and if the renderer
itself keeps running on the interpreter the repository supports
(``requires-python = ">=3.10"``).  Both have silently broken before: the
documented verdict was a hand-written constant that said "M2 = FAIL /
M5 = NOT RUN" long after the checks passed, and the renderer contained an
f-string that reused its own quote character (a syntax error before Python
3.12, i.e. the module could not even be imported).

The tests below pin the three properties that make the report trustworthy:

* the renderer compiles and renders under this interpreter;
* the committed document is exactly the renderer's output for the committed
  metrics (no hand editing, no stale verdict);
* the verdict claims follow ``metrics["checks"]`` and the *design* time order
  follows ``dt_order_probes[*].design_order`` (not ``degree + 1``, which is
  wrong at ``p = 4``: the stage-matched scheme is 5-stage fourth order).
"""
from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
METRICS = ROOT / "validation/advection_order_matched_rk/metrics.json"
DOC = ROOT / "docs/CASE-DG-ADVECT-ORDER-round2.md"


@pytest.fixture(scope="module")
def publisher():
    """Import the publisher with the platform credentials stubbed out."""
    saved = {k: os.environ.get(k) for k in ("TENSOR_AGENT_USER", "TENSOR_AGENT_PASS")}
    os.environ.setdefault("TENSOR_AGENT_USER", "test")
    os.environ.setdefault("TENSOR_AGENT_PASS", "test")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        yield importlib.import_module("publish_case_dg_advect_order_round2")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text(encoding="utf-8"))


def test_renderer_compiles_on_supported_python():
    """A nested same-quote f-string made this module unimportable before 3.12."""
    source = (ROOT / "publish_case_dg_advect_order_round2.py").read_text(encoding="utf-8")
    compile(source, "publish_case_dg_advect_order_round2.py", "exec")


def test_committed_doc_is_the_rendered_evidence(publisher, metrics):
    rendered = publisher.render(metrics)
    committed = DOC.read_text(encoding="utf-8")
    assert rendered == committed, (
        "docs/CASE-DG-ADVECT-ORDER-round2.md is not the renderer output for "
        "validation/advection_order_matched_rk/metrics.json -- regenerate it "
        "with `PYTHONPATH=. python publish_case_dg_advect_order_round2.py`")


def test_verdict_line_follows_the_checks(publisher, metrics):
    md = publisher.render(metrics)
    assert f"**总判定** **{metrics['verdict']}**" in md
    verdict_line = next(ln for ln in md.splitlines() if "**总判定**" in ln)
    for name in metrics["failed"]:
        assert name in verdict_line, name


def test_claims_are_not_hand_written_constants(publisher, metrics):
    md = publisher.render(metrics)
    checks = metrics["checks"]
    if checks.get("M2_stages") is True:
        assert "M2 = PASS" in md and "M2 = FAIL" not in md
        assert "Butcher 6 级 5 阶" not in md
    if checks.get("M5_dt_order") is True:
        assert "M5 = PASS" in md and "M5 = NOT RUN" not in md


def test_design_order_is_read_from_the_probes(publisher, metrics):
    md = publisher.render(metrics)
    for probe in metrics["dt_order_probes"]:
        design = probe["design_order"]
        shown = "n/a" if probe["fitted_time_order"] is None else f"{probe['fitted_time_order']:.3f}"
        assert f"| P{probe['degree']} | {probe['integrator']} | {probe['design_stages']} | " \
               f"{design} | {shown} | " in md, probe["degree"]


def test_render_flags_a_wrong_design_order(publisher, metrics):
    """A `degree + 1` bug would have printed ❌ for a correct p=4 run."""
    m = json.loads(json.dumps(metrics))          # deep copy
    probe = next(p for p in m["dt_order_probes"] if p["degree"] == 4)
    probe["integrator"], probe["design_order"], probe["design_stages"] = "ssp_rk54", 4, 5
    probe["fitted_time_order"] = 3.99            # within 0.2 of the *design* order 4
    md = publisher.render(m)
    row = next(line for line in md.splitlines()
               if line.startswith("| P4 | ssp_rk54 | 5 |"))
    assert row.rstrip().endswith("✅ |"), row