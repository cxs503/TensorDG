"""KBM (Knowledge Base Mining) module for TensorDG.

Provides extraction pipeline and schema validation for five knowledge card types:
- method-card: problem → method → equation → applicable domain
- parameter-card: dimensionless parameters with values and sources
- failure-card: negative knowledge (failure mode + root cause + workaround)
- benchmark-card: case + reference value + tolerance + history
- code-pointer-card: repo/module → implementation → artifacts
"""
from __future__ import annotations

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

__all__ = [
    "CARD_TYPES",
    "CARD_SCHEMAS",
    "KE_BASE_FIELDS",
    "REQUIRED_BY_CARD",
    "build_claim_record",
    "build_provenance",
    "load_schema",
    "load_source",
    "locate_quote",
    "make_card_id",
    "run_pipeline",
    "sha256_file",
    "validate_card",
    "validate_package",
    "verify_numbers",
    "verify_quotes",
]
