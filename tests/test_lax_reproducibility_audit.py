"""Contract tests for the CASE-LAX-DG reproducibility audit.

Two kinds of assertion live here, deliberately separated:

* **artifact invariants** — the audit artifact is internally consistent and its
  verdicts are the ones the rendered document publishes (no hand transcription);
* **red-line regressions** — the tracked evidence artifacts still reproduce from
  the current solver.  These are cheap (JSON only, no solve) and are meant to go
  red when a later merge silently changes the numerics behind a published number.

Nothing here re-runs the DG solver: the expensive re-execution lives in
``benchmarks.lax_reproducibility_audit`` and its output is what gets asserted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "docs" / "benchmarks" / "results" / "lax_reproducibility_audit"
METRICS = ART / "metrics.json"
DOC = ROOT / "docs" / "CASE-LAX-DG-reproducibility-audit.md"
FIG_DIR = ART / "figures"

LEDGER_BAR = 0.03
FIELDS = ("density", "velocity", "pressure")
REGISTERED_ARTIFACTS = (
    "lax_case_pool",
    "lax_resolution_requirement",
    "lax_certified_configuration",
    "lax_asymptotic_probe",
    "lax_resolution_closure",
)


@pytest.fixture(scope="module")
def metrics():
    if not METRICS.exists():  # pragma: no cover - artifact is tracked
        pytest.skip("audit artifact not generated")
    return json.loads(METRICS.read_text())


# ---------------------------------------------------------------------------
# artifact invariants
# ---------------------------------------------------------------------------

class TestArtifactIntegrity:
    def test_digest_self_consistent(self, metrics):
        """metrics_sha256 covers the payload without the digest field."""
        payload = dict(metrics)
        declared = payload.pop("metrics_sha256")
        recomputed = hashlib.sha256(
            json.dumps(payload, indent=2, sort_keys=True).encode()).hexdigest()
        assert declared == recomputed

    def test_registered_legs_present(self, metrics):
        legs = {(run["cells"], run["degree"]) for run in metrics["runs"]}
        for cells, degree in ((400, 1), (200, 3), (400, 3), (800, 3), (800, 2)):
            assert (cells, degree) in legs, f"registered leg p{degree}.N{cells} missing"

    def test_every_artifact_has_verdict(self, metrics):
        reported = metrics["reproduction"]["by_artifact"]
        assert set(reported) == set(REGISTERED_ARTIFACTS)
        for name, payload in reported.items():
            assert payload["verdict"] in ("REPRODUCIBLE", "STALE", "NOT_COMPARED")
            if payload["shared_readings"] == 0:
                assert payload["verdict"] == "NOT_COMPARED", name
            else:
                assert payload["rows"], name

    def test_quadrature_table_is_not_vacuous(self, metrics):
        """R4 must cover every leg x field, and the bias must actually be measured."""
        rows = metrics["quadrature"]["rows"]
        assert len(rows) == len(metrics["runs"]) * len(FIELDS)
        for row in rows:
            assert row["order_10"] > 0.0 and row["order_20"] > 0.0
            assert row["quadrature_bias"] == pytest.approx(
                abs(row["order_10"] - row["order_20"]) / max(row["order_20"], 1e-300))
        assert metrics["quadrature"]["worst_quadrature_bias"] > 0.0

    def test_environment_provenance_recorded(self, metrics):
        env = metrics["environment"]
        for key in ("git_head", "python", "numpy", "torch", "platform"):
            assert env.get(key)


# ---------------------------------------------------------------------------
# red-line regressions: do the tracked readings still come out of this code?
# ---------------------------------------------------------------------------

class TestReproductionRedLines:
    def test_determinism_gate(self, metrics):
        """Same configuration twice (incl. a pinned thread count) -> same coefficients."""
        determinism = metrics["determinism"]
        assert determinism["available"], "the audit did not run a repeated leg"
        assert determinism["verdict"] == "PASS", determinism["rows"]

    def test_case_pool_still_reproduces(self, metrics):
        """The A0 production-leg readings must still come out of the current solver."""
        payload = metrics["reproduction"]["by_artifact"]["lax_case_pool"]
        assert payload["verdict"] == "REPRODUCIBLE", {
            "worst": payload["worst_relative_difference"],
            "rows": [r for r in payload["rows"] if r["relative_difference"] > 1e-6],
        }

    def test_resolution_requirement_still_reproduces(self, metrics):
        payload = metrics["reproduction"]["by_artifact"]["lax_resolution_requirement"]
        assert payload["verdict"] == "REPRODUCIBLE", payload["worst_relative_difference"]

    def test_certified_configuration_still_reproduces(self, metrics):
        payload = metrics["reproduction"]["by_artifact"]["lax_certified_configuration"]
        assert payload["verdict"] == "REPRODUCIBLE", payload["worst_relative_difference"]

    def test_ledger_verdict_survives_the_rerun(self, metrics):
        """honest FAIL at p3.N400 and PASS at p3.N800/p2.N800, measured afresh."""
        assert metrics["materiality"]["verdict"] == "STABLE"
        assert not metrics["materiality"]["flips"]
        runs = {(run["cells"], run["degree"]): run for run in metrics["runs"]}
        assert runs[(400, 3)]["l2_errors"]["density"] > LEDGER_BAR
        assert runs[(800, 3)]["l2_errors"]["density"] < LEDGER_BAR
        assert runs[(800, 2)]["l2_errors"]["density"] < LEDGER_BAR


# ---------------------------------------------------------------------------
# document binding
# ---------------------------------------------------------------------------

class TestDocumentBinding:
    def test_document_exists(self):
        assert DOC.exists()

    def test_document_matches_artifact(self, metrics):
        """Rendered document must equal a fresh render of the artifact."""
        from scripts.render_lax_reproducibility_doc import render
        assert DOC.read_text() == render(metrics)

    def test_document_publishes_the_digest(self, metrics):
        assert metrics["metrics_sha256"] in DOC.read_text()

    def test_stale_artifacts_are_disclosed(self, metrics):
        """Every STALE artifact must be named in the document (honest reporting)."""
        text = DOC.read_text()
        for name in metrics["stale_artifacts"]:
            assert name in text, f"stale artifact {name} is not disclosed"

    def test_figures_present(self):
        for name in ("lax_reproducibility_drift.png", "lax_quadrature_bias.png"):
            path = FIG_DIR / name
            assert path.exists() and path.stat().st_size > 0, name