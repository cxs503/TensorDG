"""KBM card schema validation.

Validates the five card types against their JSON schemas.
Used both in the extraction pipeline and as a standalone checker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMAS_DIR = Path(__file__).resolve().parents[1] / "schemas"

CARD_SCHEMAS = {
    "method": "method-card.schema.json",
    "parameter": "parameter-card.schema.json",
    "failure": "failure-card.schema.json",
    "benchmark": "benchmark-card.schema.json",
    "code-pointer": "code-pointer-card.schema.json",
}

# Fields that are KE base fields AND NOT required by any card schema.
# These are stripped before schema validation.
KE_BASE_FIELDS = {
    "statement", "claim_type", "epistemic_status", "evidence_refs",
    "scope", "modality", "created_by", "supersedes",
    "notes", "verification",
}


def load_schema(card_type: str) -> dict:
    """Load the JSON schema for a card type."""
    schema_file = SCHEMAS_DIR / CARD_SCHEMAS[card_type]
    if not schema_file.is_file():
        raise FileNotFoundError(f"schema not found: {schema_file}")
    return json.loads(schema_file.read_text(encoding="utf-8"))


def _strip_ke_fields(card: dict) -> dict:
    """Remove KE base fields so the card validates against its specific schema."""
    return {k: v for k, v in card.items() if k not in KE_BASE_FIELDS}


def validate_card(card: dict, card_type: str) -> list[str]:
    """Validate a card dict against its schema. Return list of error strings."""
    errors = []
    
    # Check card_type
    if card_type not in {"method", "parameter", "failure", "benchmark", "code-pointer"}:
        errors.append(f"unknown card_type: {card_type!r}")
        return errors
    
    # Check required fields in card schema
    try:
        import jsonschema
        schema = load_schema(card_type)
        # Strip KE base fields before validation
        card_for_schema = _strip_ke_fields(card)
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(card_for_schema), key=lambda e: list(e.absolute_path))
        return [f"at {list(e.absolute_path)}: {e.message}" for e in errors]
    except ImportError:
        # Fallback: basic validation without jsonschema
        pass
    
    # Fallback: check required fields manually
    try:
        schema = load_schema(card_type)
        required_fields = schema.get("required", [])
        fields = card.get("fields", {})
        
        for req_field in required_fields:
            if req_field not in fields:
                errors.append(f"missing required field: {req_field}")
        
        # Also check nested required fields
        if card_type == "failure":
            fm = fields.get("failure_mode", {})
            if "symptom" not in fm:
                errors.append("missing required field: failure_mode.symptom")
        elif card_type == "code-pointer":
            cl = fields.get("code_location", {})
            if "repo" not in cl:
                errors.append("missing required field: code_location.repo")
        elif card_type == "parameter":
            param = fields.get("parameter", {})
            if "definition" not in param:
                errors.append("missing required field: parameter.definition")
    except Exception:
        pass
    
    return errors


def validate_package(pkg_dir: Path) -> dict:
    """Validate all cards in a KBM package directory.

    Returns {total, valid, invalid, errors_by_file}.
    """
    claims_dir = pkg_dir / "claims"
    if not claims_dir.is_dir():
        return {"total": 0, "valid": 0, "invalid": 0, "errors_by_file": {}}

    total = 0
    valid = 0
    invalid = 0
    errors_by_file: dict[str, list[str]] = {}

    for path in sorted(claims_dir.glob("*.json")):
        total += 1
        card = json.loads(path.read_text(encoding="utf-8"))
        card_type = card.get("card_type", "")
        if card_type not in CARD_SCHEMAS:
            errors_by_file[str(path)] = [f"unknown card_type: {card_type!r}"]
            invalid += 1
            continue
        errors = validate_card(card, card_type)
        if errors:
            errors_by_file[str(path)] = errors
            invalid += 1
        else:
            valid += 1

    return {"total": total, "valid": valid, "invalid": invalid, "errors_by_file": errors_by_file}
