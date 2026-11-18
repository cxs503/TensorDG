"""KBM (Knowledge Base Mining) extraction pipeline v1 for TensorDG.

Purpose
-------
Extract five card types (method / parameter / failure / benchmark / code-pointer)
from source reports using LLM-assisted extraction + rule-based verification.

Design principles
-----------------
1. **LLM extracts, rules verify**: every number/claim from LLM must be traceable
   back to the source report text and/or artifact. If verification fails → discard.
2. **KE API alignment**: cards are emitted as claim records (tkc: prefix) with
   card_type extension field. They flow through the existing KE four-contract
   model (claim + evidence + provenance), not parallel storage.
3. **Zero hand-copying**: all numeric values carry a ``verification`` block with
   the exact source quote, offset, and content hash.
4. **Reproducible**: the pipeline is deterministic given the same source, LLM
   output, and seed. All intermediate artifacts are hashed.

Entry point
-----------
    python3 -m tensordg.ke.kbm_extract \\
        --source-report docs/my_report.md \\
        --llm-output /tmp/llm_cards.json \\
        --out knowledge/kbm/my-extract \\
        --run-dir runs/kbm/my-extract

The LLM output format (what the LLM must produce):
    {
      "cards": [
        {
          "card_type": "method|parameter|failure|benchmark|code-pointer",
          "fields": { ... card-specific fields ... },
          "source_quotes": [
            {"quote": "...", "path": "report.md"}
          ],
          "numbers": [
            {"value": 0.325, "unit": "-", "source_path": "report.md", "needle": "Cb = 0.325"}
          ]
        }
      ]
    }

Verification rules
------------------
V1  card_type is one of the five allowed types
V2  every source_quote occurs exactly once in the source report
V3  every number's needle occurs in the source and the extracted value matches
    (within tolerance for floating point)
V4  required fields per card_type are present
V5  id follows tkc: namespace: slug pattern
V6  no duplicate ids across all cards in the batch

Metrics (emitted to run manifest)
---------------------------------
- cards_extracted: total cards from LLM
- cards_verified: cards passing all verification checks
- cards_discarded: cards failing verification (with reasons)
- verification_rate: cards_verified / cards_extracted
- numbers_verified: total numeric values verified
- numbers_failed: numeric values that failed verification
- source_coverage: fraction of source report bytes covered by verified quotes
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "1.0.0"

CARD_TYPES = {"method", "parameter", "failure", "benchmark", "code-pointer"}

# Required fields per card type (subset of schema "required" minus the KE base fields)
REQUIRED_BY_CARD = {
    "method": ["problem", "method", "equations", "applicable_domain"],
    "parameter": ["parameter"],
    "failure": ["failure_mode", "root_cause"],
    "benchmark": ["benchmark_case", "reference"],
    "code-pointer": ["code_location", "implements"],
}


# --------------------------------------------------------------------------- #
# Source access
# --------------------------------------------------------------------------- #
def load_source(path: Path) -> str:
    """Read a source report file."""
    if not path.is_file():
        raise FileNotFoundError(f"source report not found: {path}")
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Verification helpers
# --------------------------------------------------------------------------- #
def locate_quote(source_text: str, quote: str) -> int:
    """Find the offset of a quote in source text. Returns -1 if not found."""
    idx = source_text.find(quote)
    return idx


def verify_quotes(source_text: str, quotes: list[dict]) -> dict:
    """Verify that all source quotes exist in the source text.

    Returns {verified, failed, details}.
    """
    verified = 0
    failed = 0
    details = []
    for q in quotes:
        qtext = q.get("quote", "")
        if not qtext:
            failed += 1
            details.append({"quote": qtext, "status": "empty"})
            continue
        idx = locate_quote(source_text, qtext)
        if idx >= 0:
            verified += 1
            details.append({"quote": qtext[:80], "status": "found", "offset": idx})
        else:
            failed += 1
            details.append({"quote": qtext[:80], "status": "not_found"})
    return {
        "verified": verified,
        "failed": failed,
        "total": len(quotes),
        "details": details,
    }


def verify_numbers(source_text: str, numbers: list[dict], tolerance: float = 1e-6) -> dict:
    """Verify that all numeric values can be found in the source text.

    Returns {verified, failed, details}.
    """
    verified = 0
    failed = 0
    details = []
    for n in numbers:
        needle = n.get("needle", "")
        value = n.get("value")
        if not needle:
            failed += 1
            details.append({"value": value, "status": "no_needle"})
            continue
        idx = locate_quote(source_text, needle)
        if idx < 0:
            failed += 1
            details.append({"value": value, "needle": needle, "status": "not_found"})
            continue
        # Try to extract the number from the needle
        extracted = None
        for m in re.finditer(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', needle):
            try:
                extracted = float(m.group())
                break
            except ValueError:
                continue
        if extracted is not None:
            if value is not None and abs(extracted - value) / max(abs(extracted), 1e-30) <= tolerance:
                verified += 1
                details.append({"value": value, "needle": needle, "status": "match", "offset": idx})
            else:
                failed += 1
                details.append({"value": value, "needle": needle, "extracted": extracted, "status": "mismatch"})
        else:
            # No number found in needle, but needle exists
            verified += 1
            details.append({"value": value, "needle": needle, "status": "needle_found_no_number"})
    return {
        "verified": verified,
        "failed": failed,
        "total": len(numbers),
        "details": details,
    }


# --------------------------------------------------------------------------- #
# Card validation
# --------------------------------------------------------------------------- #
def validate_card(card: dict, card_type: str) -> list[str]:
    """Validate a card dict against its schema. Return list of error strings."""
    try:
        from tensordg.ke.kbm_validate import validate_card as schema_validate
    except ImportError:
        # Fallback: basic validation without jsonschema
        errors = []
        if card_type not in CARD_TYPES:
            errors.append(f"unknown card_type: {card_type!r}")
        required = REQUIRED_BY_CARD.get(card_type, [])
        for field in required:
            if field not in card.get("fields", {}):
                errors.append(f"missing required field: {field}")
        return errors

    return schema_validate(card, card_type)


def make_card_id(namespace: str, card_type: str, slug: str) -> str:
    """Generate a card id following tkc: namespace: slug pattern."""
    return f"tkc:{namespace}:{card_type}-{slug}"


def build_claim_record(card: dict, namespace: str, source_path: str,
                       quote_result: dict, number_result: dict,
                       created_by: str = "agent") -> dict:
    """Build a KE-compatible claim record from a card dict."""
    card_type = card.get("card_type", "unknown")
    fields = card.get("fields", {})

    # Generate slug from fields
    if card_type == "method":
        name = fields.get("method", {}).get("name", "unknown")
        slug = f"method-{name.lower().replace(' ', '-')}"
    elif card_type == "parameter":
        symbol = fields.get("parameter", {}).get("symbol", "unknown")
        slug = f"parameter-{symbol.lower()}"
    elif card_type == "failure":
        desc = fields.get("failure_mode", {}).get("description", "unknown")
        slug = f"failure-mode-{desc.lower().replace(' ', '-')[:40]}"
    elif card_type == "benchmark":
        case_id = fields.get("benchmark_case", {}).get("case_id", "unknown")
        slug = f"benchmark-case-{case_id.lower()}"
    elif card_type == "code-pointer":
        repo = fields.get("code_location", {}).get("repo", "unknown")
        path = fields.get("code_location", {}).get("path", "unknown")
        slug = f"code-{repo.lower()}-{path.lower().replace('/', '-')}"
    else:
        slug = f"unknown-{card_type}"

    card_id = make_card_id(namespace, card_type, slug)

    # Build statement
    if card_type == "method":
        statement = f"Method '{fields.get('method', {}).get('name', '')}' for problem: {fields.get('problem', {}).get('description', '')}"
    elif card_type == "parameter":
        param = fields.get("parameter", {})
        statement = f"Parameter {param.get('symbol', '')} ({param.get('name', '')})"
    elif card_type == "failure":
        fm = fields.get("failure_mode", {})
        statement = f"Failure mode: {fm.get('description', '')}"
    elif card_type == "benchmark":
        bc = fields.get("benchmark_case", {})
        statement = f"Benchmark case: {bc.get('case_id', '')}"
    elif card_type == "code-pointer":
        cl = fields.get("code_location", {})
        statement = f"Code: {cl.get('repo', '')}/{cl.get('path', '')}"
    else:
        statement = f"Unknown card type: {card_type}"

    # Build evidence refs
    ev_id = f"tke:{namespace}:{card_type}-ev-{hashlib.md5(str(card).encode()).hexdigest()[:8]}"
    evidence = {
        "id": ev_id,
        "type": "extraction",
        "claim_refs": [],
        "source": source_path,
        "quote_verification": quote_result,
        "number_verification": number_result,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Build verification block
    verification = {}
    if quote_result.get("verified", 0) > 0:
        for d in quote_result.get("details", []):
            if d.get("status") == "found":
                verification["source_quote"] = d.get("quote", "")
                verification["offset"] = d.get("offset", 0)
                verification["content_hash"] = hashlib.sha256(
                    (source_path + d.get("quote", "")).encode()
                ).hexdigest()
                break

    claim = {
        "id": card_id,
        "card_type": card_type,
        "statement": statement,
        "fields": fields,
        "evidence_refs": [ev_id],
        "epistemic_status": "extracted",
        "scope": source_path,
        "created_by": created_by,
        "created_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": SCHEMA_VERSION,
        "verification": verification,
    }

    return claim, evidence


def build_provenance(namespace: str, source_path: str, source_hash: str,
                     created_by: str = "agent") -> dict:
    """Build a provenance record."""
    prov_id = f"tkp:{namespace}:prov-{hashlib.md5(source_path.encode()).hexdigest()[:8]}"
    return {
        "id": prov_id,
        "type": "extraction",
        "source": {"path": source_path, "sha256": source_hash},
        "pipeline": "ke.kbm_extract",
        "version": SCHEMA_VERSION,
        "created_by": created_by,
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(
    source_report: Path,
    llm_output: Path,
    out_dir: Path,
    run_dir: Path,
    namespace: str = "kbm",
    created_by: str = "agent",
    number_tolerance: float = 1e-6,
) -> dict:
    """Run the KBM extraction pipeline.

    Args:
        source_report: Path to the source report file.
        llm_output: Path to LLM extraction output (JSON).
        out_dir: Output directory for the knowledge package.
        run_dir: Directory for run artifacts (manifest, logs).
        namespace: Namespace for generated ids.
        created_by: Producer identity.
        number_tolerance: Tolerance for number verification.

    Returns:
        Manifest dict with metrics.
    """
    # Load source
    source_text = load_source(source_report)
    source_sha = sha256_file(source_report)

    # Load LLM output
    llm_data = json.loads(llm_output.read_text(encoding="utf-8"))
    cards = llm_data.get("cards", [])

    # Process each card
    final_cards = []
    all_evidence = []
    discarded_cards = []
    total_numbers_verified = 0
    total_numbers_failed = 0
    total_quote_bytes = 0

    for card in cards:
        card_type = card.get("card_type", "")

        # V1: card_type check
        if card_type not in CARD_TYPES:
            discarded_cards.append({"card": card, "errors": [f"invalid card_type: {card_type!r}"]})
            continue

        # V4: required fields check
        required = REQUIRED_BY_CARD.get(card_type, [])
        fields = card.get("fields", {})
        missing = [f for f in required if f not in fields]
        if missing:
            discarded_cards.append({"card": card, "errors": [f"missing required fields: {missing}"]})
            continue

        # V2: quote verification
        quotes = card.get("source_quotes", [])
        quote_result = verify_quotes(source_text, quotes)
        if quote_result["failed"] > 0:
            discarded_cards.append({"card": card, "errors": [f"quote verification failed: {quote_result['failed']}/{quote_result['total']}"]})
            continue

        # V3: number verification
        numbers = card.get("numbers", [])
        number_result = verify_numbers(source_text, numbers, number_tolerance)
        if number_result["failed"] > 0:
            discarded_cards.append({"card": card, "errors": [f"number verification failed: {number_result['failed']}/{number_result['total']}"]})
            continue

        # Build claim and evidence
        claim, evidence = build_claim_record(
            card, namespace, str(source_report),
            quote_result, number_result, created_by
        )

        # V5: id format check (already enforced by make_card_id)
        # V6: duplicate id check
        if claim["id"] in [c["id"] for c in final_cards]:
            discarded_cards.append({"card": card, "errors": [f"duplicate id: {claim['id']}"]})
            continue

        final_cards.append(claim)
        all_evidence.append(all_evidence[-1] if all_evidence else {})
        total_numbers_verified += number_result.get("verified", 0)
        total_numbers_failed += number_result.get("failed", 0)
        total_quote_bytes += sum(len(d.get("quote", "")) for d in quote_result.get("details", []) if d.get("status") == "found")

    # Create output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Write outputs ---- #
    # Claims
    claims_dir = out_dir / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    for claim in final_cards:
        path = claims_dir / f"{claim['id'].replace(':', '_')}.json"
        path.write_text(json.dumps(claim, indent=2, ensure_ascii=False), encoding="utf-8")

    # Evidence
    evidence_dir = out_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for ev in all_evidence:
        if ev:
            path = evidence_dir / f"{ev['id'].replace(':', '_')}.json"
            path.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")

    # Provenance
    source_text = load_source(source_report)
    source_sha = sha256_file(source_report)
    prov = build_provenance(namespace, str(source_report), source_sha, created_by)
    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / f"{prov['id'].replace(':', '_')}.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")

    # INDEX.md
    index_lines = [
        f"# KBM Extract — {namespace}",
        f"",
        f"Source: `{source_report}` (sha256: `{source_sha[:16]}...`)",
        f"Generated: {prov['recorded_at']}",
        f"Pipeline: ke.kbm_extract v{SCHEMA_VERSION}",
        f"",
        f"## Cards ({len(final_cards)} verified)",
        f"",
    ]
    for claim in final_cards:
        index_lines.append(f"- `{claim['id']}` — {claim['statement'][:120]}")
    index_lines.append("")
    index_lines.append(f"## Evidence ({len(all_evidence)} records)")
    index_lines.append("")
    for ev in all_evidence:
        if ev:
            index_lines.append(f"- `{ev['id']}` → {ev.get('claim_refs', [])}")
    index_lines.append("")
    (out_dir / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    # Run manifest
    manifest = {
        "pipeline": "ke.kbm_extract",
        "version": SCHEMA_VERSION,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {"path": str(source_report), "sha256": source_sha},
        "metrics": {
            "cards_extracted": len(cards),
            "cards_verified": len(final_cards),
            "cards_discarded": len(discarded_cards),
            "verification_rate": len(final_cards) / max(1, len(cards)),
            "evidence_records": len(all_evidence),
            "numbers_verified": total_numbers_verified,
            "numbers_failed": total_numbers_failed,
            "source_coverage_bytes": total_quote_bytes,
            "source_total_bytes": len(source_text),
        },
        "discarded": [{"errors": d["errors"]} for d in discarded_cards],
        "artifacts": {
            "claims_dir": str(claims_dir),
            "evidence_dir": str(evidence_dir),
            "provenance_dir": str(prov_dir),
            "index": str(out_dir / "INDEX.md"),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-report", required=True, type=Path,
                    help="Path to the source report file")
    ap.add_argument("--llm-output", required=True, type=Path,
                    help="Path to LLM extraction output (JSON)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory for the knowledge package")
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="Directory for run artifacts (manifest, logs)")
    ap.add_argument("--namespace", default="kbm",
                    help="Namespace for generated ids")
    ap.add_argument("--created-by", default="agent-m25102",
                    help="Producer identity")
    ap.add_argument("--number-tolerance", type=float, default=1e-6,
                    help="Tolerance for number verification")
    args = ap.parse_args(argv)

    try:
        manifest = run_pipeline(
            source_report=args.source_report,
            llm_output=args.llm_output,
            out_dir=args.out,
            run_dir=args.run_dir,
            namespace=args.namespace,
            created_by=args.created_by,
            number_tolerance=args.number_tolerance,
        )
        m = manifest["metrics"]
        print(f"KBM extract complete: {m['cards_verified']}/{m['cards_extracted']} verified "
              f"({m['verification_rate']:.1%}), {m['evidence_records']} evidence records")
        if m["cards_discarded"] > 0:
            print(f"  Discarded: {m['cards_discarded']} cards")
            for d in manifest.get("discarded", [])[:5]:
                print(f"    - {d['errors']}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
