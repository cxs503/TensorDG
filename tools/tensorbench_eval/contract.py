"""TensorBench contract parser / validator / metric evaluator.

This module implements the platform-side contract that spec/result/workload
documents must satisfy for ``POST /tensor/bench/results`` ingestion.

Public API
----------
* ``ContractError`` — raised on schema / contract violations.
* ``parse_spec(spec)`` — validate and parse a benchmark spec document.
* ``validate_result(result, parsed)`` — validate a result document against a
  parsed spec; returns a metadata dict (``benchmark_id``, etc.).
* ``extract_document(stdout)`` — parse the workload stdout JSON document.
* ``evaluate_metrics(parsed, doc)`` — resolve JSON pointers in the stdout
  document and evaluate each metric against its tolerance.
"""
from __future__ import annotations

import json
from typing import Any

# ------------------------------------------------------------------ exceptions
class ContractError(Exception):
    """Raised when a spec or result document violates the TensorBench contract."""


# ------------------------------------------------------------------ constants
#: Top-level keys allowed in a spec document.
SPEC_KEYS = frozenset({"benchmark", "workload", "implementation",
                       "execution", "evaluation", "notes"})
#: Required top-level keys in a result document (schema v1.0.0).
RESULT_REQUIRED = frozenset({"schema_version", "run_id", "benchmark",
                             "implementation", "workload", "execution",
                             "metrics", "validation"})
#: Allowed keys inside each metric entry of the spec evaluation block.
METRIC_KEYS = frozenset({"name", "got_pointer", "ref_pointer",
                         "tolerance_rel", "unit", "better"})


# ------------------------------------------------------------------ parse_spec
def parse_spec(spec: dict) -> dict:
    """Validate and parse a benchmark spec document.

    Raises ``ContractError`` if the spec has forbidden top-level keys or
    malformed metric entries.
    """
    if not isinstance(spec, dict):
        raise ContractError("spec must be a dict")
    extra = set(spec) - SPEC_KEYS
    if extra:
        raise ContractError(f"spec has forbidden top-level keys: {sorted(extra)}")
    # Validate evaluation.metrics
    eval_block = spec.get("evaluation", {})
    if not isinstance(eval_block, dict):
        raise ContractError("spec.evaluation must be a dict")
    metrics = eval_block.get("metrics", [])
    if not isinstance(metrics, list):
        raise ContractError("spec.evaluation.metrics must be a list")
    for i, m in enumerate(metrics):
        if not isinstance(m, dict):
            raise ContractError(f"spec.evaluation.metrics[{i}] must be a dict")
        bad = set(m) - METRIC_KEYS
        if bad:
            raise ContractError(
                f"spec.evaluation.metrics[{i}] has forbidden keys: {sorted(bad)}")
        for key in ("name", "got_pointer", "ref_pointer", "tolerance_rel"):
            if key not in m:
                raise ContractError(
                    f"spec.evaluation.metrics[{i}] missing required key '{key}'")
    return {
        "benchmark_id": spec.get("benchmark", {}).get("benchmark_id"),
        "workload_id": spec.get("workload", {}).get("workload_id"),
        "metrics": metrics,
    }


# ---------------------------------------------------------------- validate_result
def validate_result(result: dict, parsed: dict) -> dict:
    """Validate a result document against a parsed spec.

    Raises ``ContractError`` if required keys are missing or the schema
    version is unsupported.
    """
    if not isinstance(result, dict):
        raise ContractError("result must be a dict")
    missing = RESULT_REQUIRED - set(result)
    if missing:
        raise ContractError(f"result missing required keys: {sorted(missing)}")
    if result.get("schema_version") != "1.0.0":
        raise ContractError(
            f"unsupported schema_version: {result.get('schema_version')!r}")
    # Cross-check benchmark_id
    result_bench = result.get("benchmark", {})
    if parsed.get("benchmark_id") and result_bench.get("benchmark_id"):
        if parsed["benchmark_id"] != result_bench["benchmark_id"]:
            raise ContractError(
                f"benchmark_id mismatch: spec={parsed['benchmark_id']!r} "
                f"vs result={result_bench['benchmark_id']!r}")
    return {
        "benchmark_id": parsed.get("benchmark_id") or result_bench.get("benchmark_id"),
        "workload_id": parsed.get("workload_id") or result.get("workload", {}).get("workload_id"),
        "schema_version": result["schema_version"],
        "run_id": result["run_id"],
    }


# --------------------------------------------------------------- extract_document
def extract_document(stdout: str) -> dict:
    """Parse the workload stdout JSON document."""
    try:
        doc = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ContractError(f"workload stdout is not valid JSON: {e}")
    if not isinstance(doc, dict):
        raise ContractError("workload stdout must be a JSON object")
    return doc


# ------------------------------------------------------------ evaluate_metrics
def _resolve_pointer(doc: dict, pointer: str) -> Any:
    """Resolve a dot-separated JSON pointer in a document."""
    node = doc
    for part in pointer.split("."):
        if not isinstance(node, dict):
            raise ContractError(
                f"cannot resolve pointer '{pointer}': "
                f"intermediate node is not a dict at '{part}'")
        if part not in node:
            raise ContractError(
                f"pointer '{pointer}' not found: key '{part}' missing")
        node = node[part]
    return node


def evaluate_metrics(parsed: dict, doc: dict) -> list[dict]:
    """Resolve JSON pointers in the stdout document and evaluate each metric.

    Returns a list of dicts with keys: ``metric``, ``got``, ``ref``,
    ``rel_error``, ``tolerance_rel``, ``passed``.
    """
    rows = []
    for m in parsed["metrics"]:
        got = _resolve_pointer(doc, m["got_pointer"])
        ref = _resolve_pointer(doc, m["ref_pointer"])
        if not isinstance(got, (int, float)):
            raise ContractError(
                f"metric '{m['name']}': got value is not numeric: {got!r}")
        if not isinstance(ref, (int, float)):
            raise ContractError(
                f"metric '{m['name']}': ref value is not numeric: {ref!r}")
        if ref == 0:
            rel_error = float("inf") if got != 0 else 0.0
        else:
            rel_error = abs(got - ref) / abs(ref)
        tol = m["tolerance_rel"]
        rows.append({
            "metric": m["name"],
            "got": float(got),
            "ref": float(ref),
            "rel_error": rel_error,
            "tolerance_rel": tol,
            "passed": bool(rel_error <= tol),
        })
    return rows
