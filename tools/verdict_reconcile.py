#!/usr/bin/env python3
"""Single-source-of-truth adjudicator for case-pool verdicts.

Two measured defects motivate this module
-----------------------------------------
**D1 -- two artefacts, one slug, opposite verdicts, same commit.**
Ledger row ``CASE-DG-EULER-VORTEX-ERR`` was covered by

=======================================  ==========  ==================================
artefact                                 gate        published verdict
=======================================  ==========  ==================================
``case_euler_vortex/metrics_yee.json``    16.4433 %   ``overall_passed: false``
``case_euler_vortex_yee/metrics.json``     0.01516 %  ``overall_passed: true``
=======================================  ==========  ==================================

Both carry the *same* ``reference`` string and the same ``config_id``
(``dg_d2_n32_t10_cfl0.2``), so nothing inside either file tells a consumer which
one is current.  Reading the wrong file flips the anchor.

**D2 -- "not run" is published as "failed".**
Five study artefacts name the same slug and expose ``all(gates.passed)``.  Most
of their gates were never evaluated in that output directory (``measured: nan``,
``passed: None``, because the checkpoints live in a *different* tree), so the
boolean is ``False`` -- indistinguishable, to any consumer, from a physics
failure.  A coverage fact was being published as a scientific result.

What this tool does
-------------------
1. enumerates every tracked artefact that asserts a verdict for a slug;
2. reduces each to a **tri-state** verdict -- ``PASS`` / ``FAIL`` / ``INCOMPLETE``
   -- where a gate that was not evaluated can only ever yield ``INCOMPLETE``;
3. requires every verdict-bearing artefact to **declare its scope** (``anchor``,
   ``study:<name>``, ``superseded``); an undeclared artefact is itself a defect,
   so a future tree cannot silently publish a competing verdict;
4. adjudicates the ledger row from the single ``anchor``-scoped artefact and
   refuses to certify while any *explained-less* ``PASS``/``FAIL`` clash or
   undeclared scope remains.

The free-space reference certificate
------------------------------------
Before the periodic (min-image) reference landed the vortex error was measured
against the free-space analytic profile.  On ``[-5, 5]^2`` with
``u_inf = (1, 1)`` the centre sits on ``(t, t)``, so at ``t = 10`` (one
translation period) the free-space core is outside the box while the periodic
solution has returned to its initial state.  ``certificate()`` computes that
discrepancy exactly; it reproduces the stale 16.4433 % to ~2e-5 relative, which
is what licenses calling that artefact ``SUPERSEDED`` rather than ``FAIL``.

Usage
-----
::

    PYTHONPATH=. python3 tools/verdict_reconcile.py --check
    PYTHONPATH=. python3 tools/verdict_reconcile.py --mark --report artifacts/verdict_reconciliation.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Ledger slug under adjudication by default.
DEFAULT_SLUG = "CASE-DG-EULER-VORTEX-ERR"

PASS, FAIL, INCOMPLETE = "PASS", "FAIL", "INCOMPLETE"

#: Scope declaration contract.  Every verdict-bearing artefact for the slug must
#: appear here; ``anchor`` competes for the ledger row, ``study:*`` is supporting
#: evidence judged on its own criteria, ``superseded`` is historical.
SCOPE_REGISTRY = {
    # -- the row itself -----------------------------------------------------
    "case_euler_vortex_yee/metrics.json": "anchor",
    # -- historical, measured against the pre-fix free-space reference ------
    "case_euler_vortex/metrics_yee.json": "superseded",
    # -- supporting studies, each with its own acceptance criteria ----------
    "case_euler_vortex_errconserve/metrics.json": "study:error-conservation-ladder",
    "case_euler_vortex_errconserve/instrument_certification.json": "study:instrument-certification",
    "case_euler_vortex_errconserve/spatial_order.json": "study:spatial-order",
    "case_euler_vortex_errconserve_lattice/metrics.json": "study:lattice-reference",
    "case_euler_vortex_errconserve_lattice/lattice_adjudication.json": "study:lattice-adjudication",
    "case_euler_vortex_errconserve_longtime/horizon.json": "study:longtime-horizon",
    "case_euler_vortex_errconserve_r6/reproduction.json": "study:reproduction-r6",
    "case_euler_vortex_errmechanism/metrics.json": "study:error-mechanism",
}

#: Why a superseded artefact is superseded.  Stated once, checked by
#: :func:`certificate_match`.
SUPERSEDED_REASON = {
    "case_euler_vortex/metrics_yee.json": (
        "pre-min-image aggregation left behind when the case moved to "
        "case_euler_vortex_yee/; no tracked script writes this path, and every "
        "one of its numbers reproduces the free-space vs periodic reference "
        "discrepancy (see certificate) rather than a scheme error"
    ),
}

#: Scripts that must *name* an artefact without producing it.  Without this the
#: adjudicator (and its report builder) count as producers of every path they
#: mention, and a frozen artefact looks refreshed.
PRODUCER_EXCLUDE = {
    "tools/verdict_reconcile.py",
    "publish_case_dg_euler_vortex_err_verdict_reconciliation.py",
}

#: Gate keys that are reported for transparency and never block a verdict.
INFORMATIONAL_GATES = ("C1x_crosscheck_raw",)

#: Quadrature of the certificate: the gate configuration's own lattice
#: (4x4 Gauss-Legendre, 32x32 cells) so the numbers are directly comparable.
CERT_CELLS = 32
CERT_QUADRATURE = 4
CERT_DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
CERT_STRENGTH = 5.0
CERT_BACKGROUND = (1.0, 1.0)
CERT_PERIOD = 10.0

#: Certificate match band on the *relative* difference.  Two independent ways of
#: evaluating the same discrepancy differ by ~2e-5 here, while a genuine scheme
#: error is off by orders of magnitude, so 1 % separates the two cleanly.
CERT_MATCH_BAND = 0.01


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def tracked_files(patterns=("*.json",)) -> list[str]:
    """Tracked paths at HEAD, relative to ``ROOT``.

    Only tracked files count: an untracked scratch file cannot mislead a
    consumer of the repository, and the adjudicator must speak about the same
    evidence a reader can check out.
    """
    try:
        out = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT,
                             check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:      # pragma: no cover
        raise SystemExit(f"git ls-files failed in {ROOT}: {exc}")
    return [line for line in out.splitlines() if line.strip()]


def _slug_of(payload: dict) -> str | None:
    for key in ("ledger_slug", "case", "slug"):
        value = payload.get(key)
        if isinstance(value, str) and value.upper().startswith("CASE-"):
            return value.upper()
    return None


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# --------------------------------------------------------------------------
# tri-state reduction
# --------------------------------------------------------------------------
def gate_state(gate: dict) -> str:
    """Reduce one gate to PASS / FAIL / INCOMPLETE.

    Two failure modes this rule has to separate, both measured in this repository:

    * D2 -- ``passed: false`` with ``measured: null``/``nan`` because the gate was
      never evaluated in that output directory.  That is ``INCOMPLETE``, and
      publishing it as ``FAIL`` invents a scientific result out of a coverage fact.
    * the converse -- ``C6_ladder_completeness`` reports ``status: "failed"`` with
      ``measured: null`` because its criterion is a *count* (records present vs
      declared), not a number.  Demoting that to ``INCOMPLETE`` would hide a real
      failure.  An explicit ``status`` is therefore the strongest evidence and is
      consulted first.
    """
    status = gate.get("status")
    if isinstance(status, str):
        state = status.strip().lower()
        if state in {"not_evaluated", "not_run", "not-run", "pending", "skipped", "missing"}:
            return INCOMPLETE
        if state in {"failed", "fail"}:
            return FAIL
        if state in {"passed", "pass", "informational", "info"}:
            return PASS
    measured = gate.get("measured")
    passed = gate.get("passed")
    if passed is True:
        return PASS
    if not _finite(measured):
        return INCOMPLETE
    if passed is False:
        return FAIL
    return INCOMPLETE


def tri_state(states) -> str:
    states = list(states)
    if any(s == FAIL for s in states):
        return FAIL
    if any(s == INCOMPLETE for s in states) or not states:
        return INCOMPLETE
    return PASS


def extract(path: str, payload: dict) -> dict | None:
    """Describe one verdict-bearing artefact (tri-state), or ``None``."""
    record = {"path": path, "slug": _slug_of(payload), "shape": None,
              "reference_model": None, "verdict": None, "gate_percent": None,
              "gate_config": None, "tolerance": None, "not_run_gates": [],
              "failed_gates": [], "reference_string": None}
    reference = payload.get("reference")
    if isinstance(reference, str):
        record["reference_string"] = reference[:180]

    # shape A -- aggregation over per-configuration records
    if "configs" in payload and ("overall_passed" in payload or "gate_percent" in payload):
        record["shape"] = "aggregate"
        overall = payload.get("overall_passed")
        record["verdict"] = PASS if overall is True else (FAIL if overall is False else INCOMPLETE)
        if payload.get("gate_pass") is None and overall is None:
            record["verdict"] = INCOMPLETE
        record["gate_percent"] = payload.get("gate_percent")
        record["gate_config"] = payload.get("gate_config")
        record["tolerance"] = payload.get("tolerance") or payload.get("tolerance_relative_l2")
        # an aggregation that folded the box records its period
        record["reference_model"] = "min_image" if payload.get("period") else None
        return record

    # shape B -- case-pool record with a gate dictionary
    if isinstance(payload.get("gates"), dict):
        record["shape"] = "gates"
        states, measured = {}, []
        for name, gate in payload["gates"].items():
            if not isinstance(gate, dict) or "passed" not in gate:
                continue                                     # data block, not a gate
            if name in INFORMATIONAL_GATES:
                continue
            states[name] = gate_state(gate)
            if states[name] == FAIL and _finite(gate.get("measured")):
                measured.append(abs(gate["measured"]))
            if states[name] == FAIL:
                record["failed_gates"].append(name)
            if states[name] == INCOMPLETE:
                record["not_run_gates"].append(name)
        record["verdict"] = tri_state(states.values())
        # what ``all(gate.passed for gate in gates)`` would have published --
        # the D2 defect, kept as a number so reports need not re-derive it
        record["raw_all_passed_boolean"] = bool(
            all(gate.get("passed") for name, gate in payload["gates"].items()
                if isinstance(gate, dict) and "passed" in gate
                and name not in INFORMATIONAL_GATES))
        record["gate_percent"] = max(measured) if measured else None
        record["gate_states"] = states
        record["tolerance"] = payload.get("tolerance_relative_l2")
        model = payload.get("reference_model")
        if isinstance(model, dict):
            record["reference_model"] = model.get("model")
        elif isinstance(model, str):
            record["reference_model"] = model
        return record

    # shape C -- reproduction / certification bundle
    verdict = payload.get("verdict")
    if isinstance(verdict, dict) and "status" in verdict:
        record["shape"] = "reproduction"
        status = str(verdict["status"]).upper()
        record["verdict"] = (PASS if status in {"CERTIFIED", "PASS", "PASSED"}
                             else INCOMPLETE if status in {"INCOMPLETE", "PARTIAL", "NOT_RUN"}
                             else FAIL)
        record["reference_model"] = "min_image"
        record["status_string"] = status
        return record

    # shape D -- a single configuration record (ladder evidence, not a verdict)
    if "config_id" in payload and "gate_pass" in payload:
        record["shape"] = "config"
        record["verdict"] = PASS if payload.get("gate_pass") else FAIL
        record["gate_percent"] = payload.get("max_percent")
        record["gate_config"] = payload.get("config_id")
        record["tolerance"] = payload.get("tolerance")
        record["reference_model"] = "min_image" if payload.get("period") else None
        return record

    return None


def collect(slug: str) -> list[dict]:
    found = []
    for path in tracked_files():
        try:
            payload = json.loads((ROOT / path).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or _slug_of(payload) != slug:
            continue
        record = extract(path, payload)
        if record is not None:
            found.append(record)
    return found


# --------------------------------------------------------------------------
# producer scan
# --------------------------------------------------------------------------
def producer_scan(paths: list[str]) -> dict[str, list[str]]:
    """Which tracked scripts name each artefact's filename?

    An artefact no script names is frozen: nothing in this checkout will ever
    refresh its verdict.
    """
    blobs = {}
    for source in tracked_files(("*.py",)):
        if source.startswith("tests/") or source in PRODUCER_EXCLUDE:
            continue
        try:
            blobs[source] = (ROOT / source).read_text(errors="replace")
        except OSError:                                   # pragma: no cover
            continue
    return {path: [source for source, text in blobs.items() if Path(path).name in text]
            for path in paths}


# --------------------------------------------------------------------------
# certificate
# --------------------------------------------------------------------------
def certificate(times=(1.0, 2.5, 5.0, 10.0)) -> dict:
    """Free-space vs periodic exact-vortex discrepancy on the gate lattice.

    Both fields are the *analytical* solution; only the reference model differs,
    so the number isolates the artefact D1 inherited.  The norm mirrors
    :func:`benchmarks.verify_euler_dg_2d._vortex_error` -- unweighted sum of
    squares over volume-quadrature nodes, per primitive field, max over fields --
    because that is the metric that produced the published numbers.
    """
    import numpy as np
    import torch

    from tensordg.euler_2d import vortex_primitive

    (xmin, xmax), (ymin, ymax) = CERT_DOMAIN
    nx = ny = CERT_CELLS
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    nodes_np, _ = np.polynomial.legendre.leggauss(CERT_QUADRATURE)
    nodes = torch.tensor(nodes_np, dtype=torch.float64)
    x = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]

    fields = ("density", "velocity_x", "velocity_y", "pressure")
    out = {"metric": "unweighted volume-quadrature relative L2, max over primitive fields",
           "cells": CERT_CELLS, "quadrature": CERT_QUADRATURE,
           "domain": [list(CERT_DOMAIN[0]), list(CERT_DOMAIN[1])],
           "period": CERT_PERIOD, "records": {}}
    for t in times:
        periodic = vortex_primitive(xq, yq, t, CERT_STRENGTH, CERT_BACKGROUND,
                                    period=CERT_PERIOD)
        free = vortex_primitive(xq, yq, t, CERT_STRENGTH, CERT_BACKGROUND)
        per_field = {}
        for index, name in enumerate(fields):
            difference = (periodic[index] - free[index]) ** 2
            per_field[name] = float(torch.sqrt(difference.sum() / (free[index] ** 2).sum()))
        out["records"][f"{t:g}"] = {"max_percent": 100.0 * max(per_field.values()),
                                    "per_field_percent": {k: 100.0 * v
                                                          for k, v in per_field.items()}}
    return out


def certificate_match(observed_percent: float, cert: dict) -> dict:
    """Nearest certificate time to an observed stale number."""
    best = None
    for key, record in cert["records"].items():
        predicted = record["max_percent"]
        if predicted <= 0.0:
            continue
        relative = abs(observed_percent - predicted) / predicted
        if best is None or relative < best["relative_difference"]:
            best = {"certificate_time": float(key), "predicted_percent": predicted,
                    "observed_percent": observed_percent,
                    "relative_difference": relative}
    if best is None:                                       # pragma: no cover
        return {"matched": False}
    best["matched"] = best["relative_difference"] < CERT_MATCH_BAND
    return best


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------
def adjudicate(slug: str = DEFAULT_SLUG) -> dict:
    artefacts = collect(slug)
    producers = producer_scan([a["path"] for a in artefacts])
    cert = certificate()

    problems = []
    rows = [a for a in artefacts if a["shape"] != "config"]

    for record in rows:
        record["producers"] = producers.get(record["path"], [])
        scope = SCOPE_REGISTRY.get(record["path"])
        record["scope"] = scope
        if scope is None:
            record["classification"] = "UNDECLARED"
            record["reason"] = ("no scope declared in tools/verdict_reconcile.py "
                                "SCOPE_REGISTRY; a verdict nobody scoped cannot be adjudicated")
            problems.append({"kind": "undeclared_scope", "path": record["path"],
                             "verdict": record["verdict"]})
            continue
        if scope == "anchor":
            record["classification"] = "AUTHORITATIVE"
            record["reason"] = "the artefact the ledger row is adjudicated on"
        elif scope == "superseded":
            record["classification"] = "SUPERSEDED"
            record["reason"] = SUPERSEDED_REASON.get(record["path"], "superseded")
            if record["gate_percent"] is not None:
                record["certificate"] = certificate_match(record["gate_percent"], cert)
                if not record["certificate"].get("matched"):
                    problems.append({"kind": "unexplained_superseded",
                                     "path": record["path"],
                                     "detail": "the superseded number is not reproduced "
                                               "by the free-space certificate",
                                     "certificate": record["certificate"]})
            if record["producers"]:
                problems.append({"kind": "superseded_still_produced",
                                 "path": record["path"],
                                 "detail": f"still written by {record['producers']}"})
        else:
            record["classification"] = "STUDY"
            record["reason"] = f"supporting study judged on its own criteria ({scope})"
            # D2 guard: a study may not publish a FAIL it did not measure
            if record["verdict"] == FAIL and record["not_run_gates"]:
                problems.append({
                    "kind": "fail_carries_not_run_gates",
                    "path": record["path"],
                    "detail": ("tri-state says FAIL, so at least one gate really failed; "
                               "listed for review that the not-run gates are not the cause"),
                    "not_run_gates": record["not_run_gates"],
                    "failed_gates": record["failed_gates"],
                })

    authority = next((a for a in rows if a["scope"] == "anchor"), None)
    if authority is None:
        problems.append({"kind": "missing_authority",
                         "detail": "no artefact declares scope=anchor for this slug"})
    anchors = [a for a in rows if a["scope"] == "anchor"]
    if len(anchors) > 1:
        problems.append({"kind": "multiple_authorities",
                         "paths": [a["path"] for a in anchors]})

    # a superseded or study artefact must never carry the anchor verdict either
    if authority is not None:
        for record in rows:
            if record["classification"] in {"AUTHORITATIVE", "UNDECLARED"}:
                continue
            if record["verdict"] == FAIL and record["classification"] == "STUDY":
                continue          # its own criteria, reported separately
            if (record["classification"] == "SUPERSEDED"
                    and record["verdict"] != authority["verdict"]
                    and not record.get("certificate", {}).get("matched")):
                problems.append({"kind": "unexplained_contradiction",
                                 "path": record["path"],
                                 "path_verdict": record["verdict"],
                                 "authority_verdict": authority["verdict"]})

    blocking = [p for p in problems
                if p["kind"] in {"undeclared_scope", "missing_authority",
                                 "multiple_authorities", "unexplained_superseded",
                                 "unexplained_contradiction", "superseded_still_produced"}]

    return {
        "slug": slug,
        "authoritative_path": None if authority is None else authority["path"],
        "authoritative_verdict": None if authority is None else authority["verdict"],
        "authoritative_percent": None if authority is None else authority["gate_percent"],
        "tolerance": None if authority is None else authority["tolerance"],
        "artefacts": rows,
        "config_records": len(artefacts) - len(rows),
        "evidence_trees": sorted({str(Path(a["path"]).parent) for a in rows}),
        "certificate": cert,
        "problems": problems,
        "blocking": blocking,
        "adjudicated_pass": (authority is not None
                             and authority["verdict"] == PASS
                             and not blocking),
    }


def mark_superseded(report: dict) -> dict:
    """Drop a machine-readable marker next to each superseded artefact."""
    written = []
    for record in report["artefacts"]:
        if record["classification"] != "SUPERSEDED":
            continue
        target = ROOT / Path(record["path"]).parent / "SUPERSEDED.json"
        marker = {
            "superseded_artefact": Path(record["path"]).name,
            "slug": report["slug"],
            "current_artefact": report["authoritative_path"],
            "current_percent": report["authoritative_percent"],
            "current_verdict": report["authoritative_verdict"],
            "superseded_percent": record["gate_percent"],
            "superseded_verdict": record["verdict"],
            "reference_model_of_superseded": "free_space (pre min-image fix)",
            "reason": record["reason"],
            "certificate": record.get("certificate"),
            "producers": record["producers"],
            "adjudicator": "tools/verdict_reconcile.py",
            "note": ("kept in the tree as historical evidence; NOT the current verdict. "
                     "Read the authoritative artefact for the ledger value."),
        }
        target.write_text(json.dumps(marker, indent=1) + "\n")
        written.append(str(target.relative_to(ROOT)))
    return {"written": written}


def render(report: dict) -> str:
    lines = [f"# verdict reconciliation: {report['slug']}", ""]
    lines.append(f"authoritative artefact : `{report['authoritative_path']}`")
    lines.append(f"authoritative verdict  : {report['authoritative_verdict']} "
                 f"({report['authoritative_percent']} % vs tolerance {report['tolerance']})")
    lines.append(f"evidence trees         : {', '.join(report['evidence_trees'])}")
    lines.append(f"config-level records   : {report['config_records']}")
    lines.append("")
    lines.append("| artefact | scope | class | verdict | gate % | producers | note |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in report["artefacts"]:
        percent = "--" if record["gate_percent"] is None else f"{record['gate_percent']:.6g}"
        note = ""
        cert = record.get("certificate")
        if cert and "certificate_time" in cert:
            note = (f"cert t={cert['certificate_time']:g} predicts "
                    f"{cert['predicted_percent']:.4f} % (rel.diff "
                    f"{cert['relative_difference']:.2e})")
        elif record.get("not_run_gates"):
            note = f"{len(record['not_run_gates'])} gate(s) not run"
        lines.append(f"| `{record['path']}` | {record['scope']} | "
                     f"{record['classification']} | {record['verdict']} | {percent} | "
                     f"{len(record['producers'])} | {note} |")
    lines.append("")
    if report["blocking"]:
        lines.append(f"## {len(report['blocking'])} blocking problem(s)")
        for item in report["blocking"]:
            lines.append(f"- **{item['kind']}** `{item.get('path', '')}` -- {item.get('detail', '')}")
    else:
        lines.append("## no blocking problems")
    notes = [p for p in report["problems"] if p not in report["blocking"]]
    if notes:
        lines.append("")
        lines.append(f"## {len(notes)} non-blocking note(s)")
        for item in notes:
            lines.append(f"- {item['kind']} `{item.get('path', '')}`: "
                         f"{item.get('detail', '')}")
    lines.append("")
    lines.append(f"**adjudicated verdict: {'PASS' if report['adjudicated_pass'] else 'FAIL'}**")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", default=DEFAULT_SLUG)
    parser.add_argument("--report", default=None, help="write the reconciliation JSON here")
    parser.add_argument("--mark", action="store_true",
                        help="drop SUPERSEDED.json next to superseded artefacts")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero when a blocking problem remains")
    parser.add_argument("--quiet", action="store_true", help="suppress the text report")
    args = parser.parse_args(argv)

    report = adjudicate(args.slug)
    if not args.quiet:
        print(render(report))
    if args.report:
        target = ROOT / args.report
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {target.relative_to(ROOT)}")
    if args.mark:
        print(json.dumps(mark_superseded(report), indent=1))
    if args.check and report["blocking"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())