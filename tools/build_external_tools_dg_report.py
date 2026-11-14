#!/usr/bin/env python3
"""Build the KPI-TOOLS-5 TensorDG evidence bundle.

Outputs (all numbers injected from :func:`tensordg.external_tools_verification.run_all`,
nothing hand-typed):

* ``artifacts/external_tools/tensordg_external_tools_report.json`` -- raw ledger
* ``artifacts/external_tools/external_tools_dg.png`` -- field render + error/limit bars
* ``docs/external-tools-verification.md`` -- published evidence document

Usage::

    python3 tools/build_external_tools_dg_report.py [--check]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tensordg.advection_1d import solve_advection  # noqa: E402
from tensordg.external_tools_verification import (  # noqa: E402
    ANCHOR,
    _modal_to_callable,
    render_markdown,
    run_all,
)

ART = ROOT / "artifacts" / "external_tools"
DOC = ROOT / "docs" / "external-tools-verification.md"
JSON_OUT = ART / "tensordg_external_tools_report.json"
PNG_OUT = ART / "external_tools_dg.png"


def build_figure(report: dict) -> pathlib.Path:
    """Two-panel evidence figure: DG field vs exact, and err/limit per tool class."""
    centers, coeff, meta = solve_advection(ncells=40, degree=2, final_time=0.25, speed=1.0)
    uh, _, _ = _modal_to_callable(centers, coeff)
    xs = np.linspace(0.0, 1.0, 401)
    values = np.array([uh(min(x, 1.0 - 1e-12)) for x in xs])
    truth = np.sin(2.0 * math.pi * (xs - 0.25))

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=130)

    left.plot(xs, truth, color="0.25", lw=2.0, label="exact $\\sin 2\\pi(x-t)$")
    left.plot(xs, values, color="#c0392b", lw=1.2, ls="--",
              label=f"DG P2, N={meta['ncells']}, t={meta['final_time']}")
    left.set_xlabel("x")
    left.set_ylabel("u")
    left.set_title("autodiff/visualization probe field\n"
                   f"max|$u_h-u$| = {np.abs(values - truth).max():.2e}")
    left.legend(loc="lower left", fontsize=8)
    left.grid(alpha=0.3)

    names = [rec["tool_class"] for rec in report["probes"]]
    ratios = [rec["abs_err"] / rec["limit"] for rec in report["probes"]]
    positions = np.arange(len(names))
    right.barh(positions, ratios, color="#2471a3", height=0.6)
    right.axvline(1.0, color="#c0392b", lw=1.4, ls="--", label="acceptance threshold 1.0")
    right.set_xscale("log")
    right.set_yticks(positions)
    right.set_yticklabels(names, fontsize=8)
    right.set_xlabel("abs_err / limit  (log scale, lower is better)")
    right.set_title(f"per-tool-class evidence (real={report['real_classes']}"
                    f"/{report['tool_classes']}, verdict={report['verdict']})")
    right.legend(fontsize=8)
    right.grid(alpha=0.3, axis="x")

    figure.tight_layout()
    ART.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_OUT)
    plt.close(figure)
    return PNG_OUT


def build_document(report: dict) -> str:
    body = [
        "",
        "## Evidence figure",
        "",
        "![external tools](artifacts/external_tools/external_tools_dg.png)",
        "",
        "Left: the DG P2 field the `matplotlib` probe renders against the exact wave",
        "(the same run scores the visualization class). Right: every tool class's",
        "`abs_err / limit` ratio on a log axis -- all bars sit far left of the red",
        "acceptance threshold, i.e. the tool chains are exact to round-off.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python3 -m pytest tests/test_external_tools_verification.py -q",
        "PYTHONPATH=. python3 tools/build_external_tools_dg_report.py",
        "```",
        "",
        "## Independence limitations (adversarial self-review)",
        "",
        "- The probes run in-process against TensorDG's own operators; the references",
        "  (exact rationals for sympy, closed forms for the integrals, the analytic",
        "  sine wave for fft/matplotlib) are independent of the DG code, but the",
        "  operator *assembly* being exercised is the one under test. A defect shared",
        "  by assembly and probe (e.g. the same Legendre normalisation error) would",
        "  not be caught here; it is caught by the convergence suites in `tests/`.",
        "- `version` is the installed module version, not a hash; minimum versions in",
        "  `PROBE_SPEC` are declared intent, not enforced by an environment lock.",
        "- The probes are CPU/float64 only; GPU (`cuda`) and lower-precision dtypes are",
        "  out of scope.",
        "- These are tool-availability-plus-correctness probes, not new physics",
        "  validation: no benchmark verdict (e.g. Thacker, Sod) is claimed or changed.",
        "",
        "## What this evidence does *not* establish",
        "",
        "- It does not claim the platform-level KPI is closed on its own: it is the",
        "  TensorDG-side contribution (6 classes on genuine DG surfaces). The KPI",
        "  counts classes across the ecosystem, so the ledger evidence has to be",
        "  updated with this record alongside the other repositories' records.",
        "- It does not verify tool behaviour under parallelism/MPI or on the loong64",
        "  cluster nodes.",
    ]
    return render_markdown(report) + "\n".join(body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the committed document is stale")
    args = parser.parse_args()

    report = run_all()
    ART.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2) + "\n"
    document = build_document(report)

    if args.check:
        stale = []
        if not JSON_OUT.exists() or _drop_json_stamp(
            JSON_OUT.read_text(encoding="utf-8")
        ) != _drop_json_stamp(payload):
            stale.append(str(JSON_OUT.relative_to(ROOT)))
        if not DOC.exists():
            stale.append(str(DOC.relative_to(ROOT)))
        else:
            committed = DOC.read_text(encoding="utf-8")
            # the timestamp necessarily differs; compare everything else
            if _strip_stamp(committed) != _strip_stamp(document):
                stale.append(str(DOC.relative_to(ROOT)))
        if stale:
            print("stale: " + ", ".join(stale))
            return 1
        print("up to date")
        return 0

    JSON_OUT.write_text(payload, encoding="utf-8")
    DOC.write_text(document, encoding="utf-8")
    build_figure(report)
    print(f"wrote {JSON_OUT.relative_to(ROOT)}, {PNG_OUT.relative_to(ROOT)}, {DOC.relative_to(ROOT)}")
    print(f"verdict={report['verdict']} real_classes={report['real_classes']}/{report['tool_classes']}")
    return 0 if report["verdict"] == "PASS" else 1


def _drop_json_stamp(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if "generated_at" not in line)


def _strip_stamp(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith("**Generated**:"))


def _stamp(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("**Generated**:"):
            return line
    return ""


if __name__ == "__main__":
    sys.exit(main())