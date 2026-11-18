"""Tests for KBM extraction pipeline and card schemas.

Covers:
- Card type validation (V1)
- Quote verification (V2)
- Number verification (V3)
- Required fields check (V4)
- ID format check (V5)
- Duplicate ID check (V6)
- Schema validation
- Pipeline end-to-end
- Determinism
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensordg.ke.kbm_extract import (
    CARD_TYPES,
    REQUIRED_BY_CARD,
    build_claim_record,
    build_provenance,
    load_source,
    locate_quote,
    make_card_id,
    run_pipeline,
    sha256_file,
    verify_numbers,
    verify_quotes,
)
from tensordg.ke.kbm_validate import (
    CARD_SCHEMAS,
    KE_BASE_FIELDS,
    load_schema,
    validate_card,
    validate_package,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_report(tmp_path):
    """A synthetic report with extractable content."""
    p = tmp_path / "report.md"
    text = """# Synthetic Ship Resistance Report

## 1. Hull Geometry
The test case uses a Wigley hull with the following parameters:
- Length L = 2.0 m
- Beam B = 0.3 m
- Draft T = 0.125 m
- Block coefficient Cb = 0.467

## 2. Method
The resistance is computed using the Holtrop-Mennen method (Holtrop & Mennen, 1982).
The total resistance coefficient is Ct = 3.25e-3 at Fr = 0.25.

## 3. Results
At Froude number Fr = 0.25, the computed resistance coefficient is Ct = 3.28e-3.
The reference value from the original paper is Ct_ref = 3.25e-3.
The relative error is 0.92%.

## 4. Known Issues
When the Froude number exceeds 0.45, the Holtrop-Mennen method tends to
overpredict resistance by 5-10%. This is a known limitation of the empirical
formulation for high-speed displacement hulls.

## 5. Code Reference
The implementation is in TensorFVM, module resistance/holtrop_mennen.py,
function compute_ct().
"""
    p.write_text(text)
    return p
@pytest.fixture
def sample_llm_output():
    """Synthetic LLM output with five card types.
    Quotes must match the sample_report exactly.
    """
    return {
        "cards": [
            {
                "card_type": "method",
                "fields": {
                    "problem": {"description": "Ship resistance prediction for displacement hulls"},
                    "method": {"name": "Holtrop-Mennen", "description": "Empirical resistance prediction method", "family": "other"},
                    "equations": [{"formula": "Ct = f(L, B, T, Cb, Fr)", "source": "Holtrop & Mennen (1982)", "source_type": "paper"}],
                    "applicable_domain": {"domain": "ship-resistance", "limitations": ["valid for Fr < 0.45"]}
                },
                "source_quotes": [
                    {"quote": "The resistance is computed using the Holtrop-Mennen method", "path": "report.md"}
                ],
                "numbers": [
                    {"value": 0.00325, "needle": "Ct = 3.25e-3"}
                ]
            },
            {
                "card_type": "parameter",
                "fields": {
                    "parameter": {
                        "symbol": "Cb",
                        "name": "Block coefficient",
                        "definition": "Ratio of displaced volume to L*B*T",
                        "formula": "Cb = V / (L * B * T)",
                        "typical_value": 0.467
                    }
                },
                "source_quotes": [
                    {"quote": "Block coefficient Cb = 0.467", "path": "report.md"}
                ],
                "numbers": [
                    {"value": 0.467, "needle": "Cb = 0.467"}
                ]
            },
            {
                "card_type": "failure",
                "fields": {
                    "failure_mode": {"description": "Overprediction at high Froude numbers", "symptom": "Resistance exceeds reference by 5-10%", "severity": "major"},
                    "root_cause": {"description": "Empirical formulation limitations", "category": "model"}
                },
                "source_quotes": [
                    {"quote": "When the Froude number exceeds 0.45, the Holtrop-Mennen method tends to overpredict resistance by 5-10%", "path": "report.md"}
                ],
                "numbers": [
                    {"value": 0.45, "needle": "Froude number exceeds 0.45"}
                ]
            },
            {
                "card_type": "benchmark",
                "fields": {
                    "benchmark_case": {"case_id": "wigley-fr025", "description": "Wigley hull at Fr=0.25"},
                    "reference": {"value": 0.00325, "source": "Holtrop & Mennen (1982)", "tolerance_pct": 5.0}
                },
                "source_quotes": [
                    {"quote": "Computed Ct = 3.28e-3", "path": "report.md"}
                ],
                "numbers": [
                    {"value": 0.00328, "needle": "Ct = 3.28e-3"}
                ]
            },
            {
                "card_type": "code-pointer",
                "fields": {
                    "code_location": {"repo": "TensorFVM", "path": "resistance/holtrop_mennen.py", "symbol": "compute_ct"},
                    "implements": {"description": "Holtrop-Mennen resistance computation"}
                },
                "source_quotes": [
                    {"quote": "The implementation is in TensorFVM, module resistance/holtrop_mennen.py", "path": "report.md"}
                ],
                "numbers": []
            }
        ]
    }


@pytest.fixture
def valid_method_card():
    """A valid method card for schema validation."""
    return {
        "id": "tkc:test:method-holtrop-mennen",
        "card_type": "method",
        "problem": {"description": "Ship resistance prediction"},
        "method": {"name": "Holtrop-Mennen", "description": "Empirical method"},
        "equations": [{"formula": "Ct = Cf + Cr", "source": "Holtrop & Mennen (1982)"}],
        "applicable_domain": {"domain": "ship-resistance"},
        "created_at": "2026-11-15T00:00:00Z",
        "version": "1.0.0",
    }


@pytest.fixture
def valid_parameter_card():
    """A valid parameter card for schema validation."""
    return {
        "id": "tkc:test:parameter-fr",
        "card_type": "parameter",
        "parameter": {"symbol": "Fr", "name": "Froude number", "definition": "V / sqrt(gL)"},
        "created_at": "2026-11-15T00:00:00Z",
        "version": "1.0.0",
    }


@pytest.fixture
def valid_failure_card():
    """A valid failure card for schema validation."""
    return {
        "id": "tkc:test:failure-overprediction",
        "card_type": "failure",
        "failure_mode": {"description": "Overprediction", "symptom": "High resistance"},
        "root_cause": {"description": "Model limitation"},
        "created_at": "2026-11-15T00:00:00Z",
        "version": "1.0.0",
    }


@pytest.fixture
def valid_benchmark_card():
    """A valid benchmark card for schema validation."""
    return {
        "id": "tkc:test:benchmark-wigley",
        "card_type": "benchmark",
        "benchmark_case": {"case_id": "wigley-fr025", "description": "Wigley hull"},
        "reference": {"value": 0.00325, "source": "Holtrop & Mennen (1982)"},
        "created_at": "2026-11-15T00:00:00Z",
        "version": "1.0.0",
    }


@pytest.fixture
def valid_code_pointer_card():
    """A valid code-pointer card for schema validation."""
    return {
        "id": "tkc:test:code-tensorfvm-resistance",
        "card_type": "code-pointer",
        "code_location": {"repo": "TensorFVM", "path": "resistance/holtrop_mennen.py"},
        "implements": {"description": "Holtrop-Mennen resistance computation"},
        "created_at": "2026-11-15T00:00:00Z",
        "version": "1.0.0",
    }


# --------------------------------------------------------------------------- #
# Card type validation (V1)
# --------------------------------------------------------------------------- #
class TestCardTypeValidation:
    """Test V1: card_type is one of the five allowed types."""

    def test_all_card_types_defined(self):
        assert CARD_TYPES == {"method", "parameter", "failure", "benchmark", "code-pointer"}

    def test_all_schemas_defined(self):
        assert set(CARD_SCHEMAS.keys()) == CARD_TYPES

    def test_invalid_card_type_rejected(self, sample_report, tmp_path):
        """Card with invalid card_type should be discarded."""
        llm_output = tmp_path / "llm.json"
        llm_output.write_text(json.dumps({"cards": [{"card_type": "invalid_type", "fields": {}}]}))
        manifest = run_pipeline(sample_report, llm_output, tmp_path / "out", tmp_path / "run")
        assert manifest["metrics"]["cards_extracted"] == 1
        assert manifest["metrics"]["cards_verified"] == 0
        assert manifest["metrics"]["cards_discarded"] == 1


# --------------------------------------------------------------------------- #
# Quote verification (V2)
# --------------------------------------------------------------------------- #
class TestQuoteVerification:
    """Test V2: every source_quote occurs exactly once in the source report."""

    def test_quote_found(self, sample_report):
        text = load_source(sample_report)
        result = verify_quotes(text, [{"quote": "Block coefficient Cb = 0.467"}])
        assert result["verified"] == 1
        assert result["failed"] == 0

    def test_quote_not_found(self, sample_report):
        text = load_source(sample_report)
        result = verify_quotes(text, [{"quote": "This text does not exist in the report"}])
        assert result["verified"] == 0
        assert result["failed"] == 1

    def test_empty_quote(self, sample_report):
        text = load_source(sample_report)
        result = verify_quotes(text, [{"quote": ""}])
        assert result["verified"] == 0
        assert result["failed"] == 1

    def test_multiple_quotes(self, sample_report):
        text = load_source(sample_report)
        result = verify_quotes(text, [
            {"quote": "Block coefficient Cb = 0.467"},
            {"quote": "Ct = 3.25e-3"},
        ])
        assert result["verified"] == 2
        assert result["failed"] == 0


# --------------------------------------------------------------------------- #
# Number verification (V3)
# --------------------------------------------------------------------------- #
class TestNumberVerification:
    """Test V3: every number's needle occurs in the source and matches."""

    def test_number_match(self, sample_report):
        text = load_source(sample_report)
        result = verify_numbers(text, [{"value": 0.467, "needle": "Cb = 0.467"}])
        assert result["verified"] == 1
        assert result["failed"] == 0

    def test_number_mismatch(self, sample_report):
        text = load_source(sample_report)
        result = verify_numbers(text, [{"value": 0.999, "needle": "Cb = 0.467"}])
        assert result["verified"] == 0
        assert result["failed"] == 1

    def test_number_not_found(self, sample_report):
        text = load_source(sample_report)
        result = verify_numbers(text, [{"value": 0.467, "needle": "nonexistent = 0.467"}])
        assert result["verified"] == 0
        assert result["failed"] == 1

    def test_no_needle(self, sample_report):
        text = load_source(sample_report)
        result = verify_numbers(text, [{"value": 0.467}])
        assert result["verified"] == 0
        assert result["failed"] == 1

    def test_scientific_notation(self, sample_report):
        text = load_source(sample_report)
        result = verify_numbers(text, [{"value": 0.00325, "needle": "Ct = 3.25e-3"}])
        assert result["verified"] == 1
        assert result["failed"] == 0


# --------------------------------------------------------------------------- #
# Required fields check (V4)
# --------------------------------------------------------------------------- #
class TestRequiredFields:
    """Test V4: required fields per card_type are present."""

    def test_method_missing_equations(self):
        card = {"card_type": "method", "fields": {"problem": {"description": "test"}, "method": {"name": "test", "description": "test"}}}
        errors = validate_card(card, "method")
        assert any("equations" in e for e in errors)

    def test_parameter_missing_definition(self):
        card = {"card_type": "parameter", "parameter": {"symbol": "Fr", "name": "Froude"}, "id": "tkc:test:parameter-fr", "created_at": "2026-11-15T00:00:00Z", "version": "1.0.0"}
        errors = validate_card(card, "parameter")
        assert any("definition" in e for e in errors)

    def test_failure_missing_symptom(self):
        card = {"card_type": "failure", "failure_mode": {"description": "test"}, "root_cause": {"description": "test"}, "id": "tkc:test:failure-test", "created_at": "2026-11-15T00:00:00Z", "version": "1.0.0"}
        errors = validate_card(card, "failure")
        assert any("symptom" in e for e in errors)

    def test_benchmark_missing_reference(self):
        card = {"card_type": "benchmark", "fields": {"benchmark_case": {"case_id": "test", "description": "test"}}}
        errors = validate_card(card, "benchmark")
        assert any("reference" in e for e in errors)

    def test_code_pointer_missing_repo(self):
        card = {"card_type": "code-pointer", "code_location": {"path": "test.py"}, "implements": {"description": "test"}, "id": "tkc:test:code-test", "created_at": "2026-11-15T00:00:00Z", "version": "1.0.0"}
        errors = validate_card(card, "code-pointer")
        assert any("repo" in e for e in errors)


# --------------------------------------------------------------------------- #
# ID format check (V5)
# --------------------------------------------------------------------------- #
class TestIdFormat:
    """Test V5: id follows tkc: namespace: slug pattern."""

    def test_valid_id(self):
        assert make_card_id("test", "method", "holtrop-mennen") == "tkc:test:method-holtrop-mennen"

    def test_id_pattern(self):
        card_id = make_card_id("test", "method", "holtrop-mennen")
        assert card_id.startswith("tkc:")
        assert ":" in card_id

    def test_id_with_special_chars(self):
        card_id = make_card_id("test-ns", "parameter", "fr-number")
        assert card_id == "tkc:test-ns:parameter-fr-number"


# --------------------------------------------------------------------------- #
# Schema validation
# --------------------------------------------------------------------------- #
class TestSchemaValidation:
    """Test schema validation for all card types."""

    def test_method_card_valid(self, valid_method_card):
        errors = validate_card(valid_method_card, "method")
        assert len(errors) == 0

    def test_parameter_card_valid(self, valid_parameter_card):
        errors = validate_card(valid_parameter_card, "parameter")
        assert len(errors) == 0

    def test_failure_card_valid(self, valid_failure_card):
        errors = validate_card(valid_failure_card, "failure")
        assert len(errors) == 0

    def test_benchmark_card_valid(self, valid_benchmark_card):
        errors = validate_card(valid_benchmark_card, "benchmark")
        assert len(errors) == 0

    def test_code_pointer_card_valid(self):
        card = {
            "id": "tkc:test:code-tensorfvm-resistance",
            "card_type": "code-pointer",
            "code_location": {"repo": "TensorFVM", "path": "resistance/holtrop_mennen.py"},
            "implements": {"description": "Holtrop-Mennen resistance computation"},
            "created_at": "2026-11-15T00:00:00Z",
            "version": "1.0.0",
        }
        errors = validate_card(card, "code-pointer")
        assert len(errors) == 0

    def test_invalid_card_type(self):
        card = {"id": "tkc:test:invalid", "card_type": "invalid", "fields": {}}
        errors = validate_card(card, "invalid")
        assert len(errors) > 0


# --------------------------------------------------------------------------- #
# Pipeline end-to-end
# --------------------------------------------------------------------------- #
class TestPipeline:
    """Test the full extraction pipeline."""

    def test_pipeline_runs(self, sample_report, sample_llm_output, tmp_path):
        llm_output = tmp_path / "llm.json"
        llm_output.write_text(json.dumps(sample_llm_output))
        manifest = run_pipeline(
            source_report=sample_report,
            llm_output=llm_output,
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
        )
        assert manifest["metrics"]["cards_extracted"] == 5
        assert manifest["metrics"]["cards_verified"] >= 3  # At least 3 should pass

    def test_pipeline_outputs_created(self, sample_report, sample_llm_output, tmp_path):
        llm_output = tmp_path / "llm.json"
        llm_output.write_text(json.dumps(sample_llm_output))
        run_pipeline(
            source_report=sample_report,
            llm_output=llm_output,
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
        )
        assert (tmp_path / "out" / "claims").is_dir()
        assert (tmp_path / "out" / "evidence").is_dir()
        assert (tmp_path / "out" / "provenance").is_dir()
        assert (tmp_path / "out" / "INDEX.md").is_file()
        assert (tmp_path / "run" / "run_manifest.json").is_file()

    def test_pipeline_with_invalid_cards(self, sample_report, tmp_path):
        """Pipeline should discard invalid cards gracefully."""
        llm_output = tmp_path / "llm.json"
        llm_output.write_text(json.dumps({
            "cards": [
                {"card_type": "invalid_type", "fields": {}},
                {"card_type": "method", "fields": {"problem": {"description": "test"}}},  # missing required fields
                {"card_type": "method", "fields": {"problem": {"description": "test"}, "method": {"name": "test", "description": "test"}, "equations": [{"formula": "x", "source": "y"}], "applicable_domain": {"domain": "z"}}},
            ]
        }))
        manifest = run_pipeline(
            source_report=sample_report,
            llm_output=llm_output,
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
        )
        assert manifest["metrics"]["cards_extracted"] == 3
        assert manifest["metrics"]["cards_discarded"] >= 2

    def test_pipeline_determinism(self, sample_report, sample_llm_output, tmp_path):
        """Running the pipeline twice should produce identical results."""
        llm_output = tmp_path / "llm.json"
        llm_output.write_text(json.dumps(sample_llm_output))

        out1 = tmp_path / "out1"
        run1 = tmp_path / "run1"
        run_pipeline(source_report=sample_report, llm_output=llm_output, out_dir=out1, run_dir=run1)

        out2 = tmp_path / "out2"
        run2 = tmp_path / "run2"
        run_pipeline(source_report=sample_report, llm_output=llm_output, out_dir=out2, run_dir=run2)

        # Compare claims (excluding timestamps)
        claims1 = sorted([(f.name, json.loads(f.read_text())) for f in (out1 / "claims").glob("*.json")])
        claims2 = sorted([(f.name, json.loads(f.read_text())) for f in (out2 / "claims").glob("*.json")])
        for name1, c1 in claims1:
            assert name1 in [n for n, _ in claims2]
            c1_clean = {k: v for k, v in c1.items() if k not in ("created_at",)}
            c2_clean = {k: v for k, v in dict(claims2).get(name1, {}).items() if k not in ("created_at",)}
            assert c1_clean == c2_clean, f"Mismatch in {name1}"

    def test_pipeline_manifest_metrics(self, sample_report, sample_llm_output, tmp_path):
        llm_output = tmp_path / "llm.json"
        llm_output.write_text(json.dumps(sample_llm_output))
        manifest = run_pipeline(
            source_report=sample_report,
            llm_output=llm_output,
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
        )
        m = manifest["metrics"]
        assert m["cards_extracted"] == 5
        assert m["cards_verified"] >= 0
        assert m["cards_discarded"] >= 0
        assert m["cards_extracted"] == m["cards_verified"] + m["cards_discarded"]
        assert 0 <= m["verification_rate"] <= 1.0

    def test_pipeline_with_demo_sample(self, tmp_path):
        """Test with the actual demo sample report."""
        demo_report = ROOT / "knowledge" / "kbm" / "demo-sample" / "report.md"
        demo_llm = ROOT / "knowledge" / "kbm" / "demo-sample" / "llm_output.json"
        if not demo_report.is_file() or not demo_llm.is_file():
            pytest.skip("Demo sample not available")
        manifest = run_pipeline(
            source_report=demo_report,
            llm_output=demo_llm,
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
        )
        m = manifest["metrics"]
        assert m["cards_extracted"] == 6
        # At least the method and parameter cards should pass (they have matching quotes)
        assert m["cards_verified"] >= 2

    def test_pipeline_consistency_rate(self, tmp_path):
        """Test that consistency rate meets the ≥80% threshold."""
        demo_report = ROOT / "knowledge" / "kbm" / "demo-sample" / "report.md"
        demo_llm = ROOT / "knowledge" / "kbm" / "demo-sample" / "llm_output.json"
        if not demo_report.is_file() or not demo_llm.is_file():
            pytest.skip("Demo sample not available")
        manifest = run_pipeline(
            source_report=demo_report,
            llm_output=demo_llm,
            out_dir=tmp_path / "out",
            run_dir=tmp_path / "run",
        )
        rate = manifest["metrics"]["verification_rate"]
        assert rate >= 0.8, f"Verification rate {rate:.1%} < 80% threshold"


# --------------------------------------------------------------------------- #
# Schema loading
# --------------------------------------------------------------------------- #
class TestSchemaLoading:
    """Test schema loading and validation infrastructure."""

    def test_all_schemas_loadable(self):
        for card_type in CARD_TYPES:
            schema = load_schema(card_type)
            assert "$schema" in schema
            assert "title" in schema
            assert "required" in schema

    def test_ke_base_fields_stripped(self):
        card = {
            "id": "tkc:test:method-test",
            "card_type": "method",
            "problem": {"description": "test"},
            "method": {"name": "test", "description": "test"},
            "equations": [{"formula": "x", "source": "y"}],
            "applicable_domain": {"domain": "z"},
            "created_at": "2026-11-15T00:00:00Z",
            "version": "1.0.0",
            "statement": "test statement",  # KE base field
            "claim_type": "methodological",  # KE base field
        }
        errors = validate_card(card, "method")
        assert len(errors) == 0

    def test_validate_package_empty(self, tmp_path):
        result = validate_package(tmp_path)
        assert result == {"total": 0, "valid": 0, "invalid": 0, "errors_by_file": {}}

    def test_validate_package_with_files(self, tmp_path):
        claims_dir = tmp_path / "claims"
        claims_dir.mkdir()
        # Write a valid card
        card = {
            "id": "tkc:test:method-test",
            "card_type": "method",
            "problem": {"description": "test"},
            "method": {"name": "test", "description": "test"},
            "equations": [{"formula": "x", "source": "y"}],
            "applicable_domain": {"domain": "z"},
            "created_at": "2026-11-15T00:00:00Z",
            "version": "1.0.0",
        }
        (claims_dir / "tkc_test_method_test.json").write_text(json.dumps(card))
        result = validate_package(tmp_path)
        assert result["total"] == 1
        assert result["valid"] == 1
        assert result["invalid"] == 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class TestCLI:
    """Test the CLI entry point."""

    def test_cli_help(self):
        from tensordg.ke.kbm_extract import main
        try:
            main(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_cli_missing_args(self):
        from tensordg.ke.kbm_extract import main
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2  # argparse returns 2 for missing required args

    def test_cli_file_not_found(self, tmp_path):
        from tensordg.ke.kbm_extract import main
        result = main([
            "--source-report", "/nonexistent/report.md",
            "--llm-output", str(tmp_path / "llm.json"),
            "--out", str(tmp_path / "out"),
            "--run-dir", str(tmp_path / "run"),
        ])
        assert result == 1
