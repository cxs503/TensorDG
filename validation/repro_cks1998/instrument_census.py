"""Census of the reference *instrument* across TensorDG: who still uses the fixed rule?

Why this module exists
----------------------
The anchor ``REPRO-DG-CKS1998`` publishes "< 3 %" numbers, and every one of them is a
ratio: a numerical cell average divided by a **reference** cell average.  Two
instruments produce that reference in this repository:

``benchmarks.riemann_exact.cell_averages`` (re-exported as
``benchmarks.sod_exact.cell_averages``)
    a **fixed** 8-point Gauss rule per cell.  Its own docstring says it is not a
    converged reference: a cell that contains a wave is integrated inexactly with an
    error of size ``rho_q * jump`` (``rho_q ~ 1e-2``, a constant of the rule), so the
    bias does not shrink under refinement -- it only moves into fewer cells.

``benchmarks.shock_reference_quadrature.cell_averages``
    breakpoint-split exact integration; converges to machine precision in the rule
    order once the cell is cut at the analytic waves, i.e. it has no knob.

The same docstring states the policy: use the second one "when the number is compared
against a tolerance".  ``docs/REFERENCE-QUADRATURE-INSTRUMENT-AUDIT.md`` measured what
that policy buys on the three Sod gates; ``docs/REPRO-DG-CKS1998-E2-STATION-REFERENCE.md``
switched the anchor's **station** metrics and left the repo-wide census undeclared
(its §5-6: "``benchmarks.riemann_exact.cell_averages`` ... 本单只修本锚点路径，
未做全仓普查").  This module is that census, and it is machine-checked:

* :func:`discover` walks the whole package with :mod:`ast` and returns every call site
  of the fixed rule (including aliased and attribute-form imports);
* :data:`REGISTRY` states, for every one of them, **what the number is used for**
  (``gating`` / ``plot`` / ``diagnostic`` / ``instrument_measurement``) and, where the
  fixed rule is still in a gating path, the measured **margin** and the measured
  **bias** that eats it;
* :func:`census` fails closed: a new call site that nobody declared raises
  :class:`UndeclaredInstrumentSite`, so the instrument policy cannot silently decay
  while the numbers keep passing.

Nothing here changes a tolerance, a mesh, a station or a registered metric.

Usage::

    PYTHONPATH=. python -m validation.repro_cks1998.instrument_census            # table
    PYTHONPATH=. python -m validation.repro_cks1998.instrument_census --json \
        validation/cks1998_instrument_census/census.json
    PYTHONPATH=. python -m validation.repro_cks1998.instrument_census --check    # exit 1 on drift
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks import shock_reference_quadrature as srq  # noqa: E402
from benchmarks.sod_exact import LEFT as SOD_LEFT, RIGHT as SOD_RIGHT  # noqa: E402

__all__ = [
    "FIXED_RULE",
    "SWITCHED_BY_THIS_CENSUS",
    "EXACT_RULE",
    "REGISTRY",
    "SOD_T",
    "CensusDrift",
    "UndeclaredInstrumentSite",
    "discover",
    "census",
    "bias_table",
    "wave_cells",
]

#: Canonical names of the two instruments.
FIXED_RULE = "benchmarks.riemann_exact.cell_averages"
EXACT_RULE = "benchmarks.shock_reference_quadrature.cell_averages"

#: Sod test bed of the anchor (single source: ``benchmarks.sod_exact``).
SOD_T = 0.2

#: Mesh ladder on which the instrument bias is tabulated.  It contains every mesh the
#: anchor's own tools score, plus the two the older audit left as ``not_run``.
BIAS_MESHES: Tuple[int, ...] = (48, 100, 200, 300, 400, 600, 800, 3200)

#: Scan roots (everything that can publish or gate a number).
SCAN_ROOTS: Tuple[str, ...] = ("benchmarks", "validation", "tools", "tensordg",
                               "tests", "examples", "scripts")

#: Files that *define* an instrument, or that are this census itself: never sites.
SCAN_SKIP_FILES: Tuple[str, ...] = (
    "benchmarks/riemann_exact.py",
    "benchmarks/sod_exact.py",
    "benchmarks/shock_reference_quadrature.py",
    "benchmarks/shock_reference_quadrature_audit.py",
    "validation/repro_cks1998/instrument_census.py",
    "tests/test_cks1998_instrument_census.py",
)


class UndeclaredInstrumentSite(RuntimeError):
    """A call of the fixed-rule reference exists that :data:`REGISTRY` does not cover."""


class CensusDrift(RuntimeError):
    """A declared site no longer exists, or the recorded bias/margin changed."""


# --------------------------------------------------------------------------- #
# static discovery                                                            #
# --------------------------------------------------------------------------- #
def _scan_files() -> List[Path]:
    files: List[Path] = []
    skip = {REPO / p for p in SCAN_SKIP_FILES}
    for root in SCAN_ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        files.extend(sorted(p for p in base.rglob("*.py") if p not in skip))
    return files


def _imports(tree: ast.AST) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Resolve the module-level names that denote the fixed rule.

    Returns ``(names, module_aliases)``:

    * ``names`` maps a local identifier to the canonical symbol it denotes
      (``from benchmarks.sod_exact import cell_averages as ca`` -> ``{"ca": FIXED_RULE}``);
    * ``module_aliases`` maps a local identifier to a module path whose
      ``cell_averages`` attribute is the fixed rule
      (``from benchmarks import sod_exact`` -> ``{"sod_exact": "benchmarks.sod_exact"}``).
    """
    names: Dict[str, str] = {}
    module_aliases: Dict[str, str] = {}
    fixed_modules = {"benchmarks.sod_exact", "benchmarks.riemann_exact"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            where = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                if where in fixed_modules and alias.name == "cell_averages":
                    names[local] = FIXED_RULE
                elif where == "benchmarks" and alias.name in ("sod_exact", "riemann_exact"):
                    module_aliases[local] = f"benchmarks.{alias.name}"
                elif where == "benchmarks.shock_reference_quadrature" \
                        and alias.name == "fixed_rule_cell_averages":
                    # the same legacy rule, reached through the new module
                    names[local] = "shock_reference_quadrature.fixed_rule_cell_averages"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in fixed_modules:
                    local = alias.asname or alias.name.split(".")[-1]
                    module_aliases[local] = alias.name
                if alias.name in ("benchmarks", "benchmarks.sod_exact",
                                  "benchmarks.riemann_exact"):
                    local = alias.asname or "benchmarks"
                    module_aliases[local] = alias.name
    return names, module_aliases


def _canonical(func: ast.AST, names: Mapping[str, str],
               module_aliases: Mapping[str, str]) -> str | None:
    """Canonical symbol of a call target, or ``None`` when it is not the fixed rule."""
    if isinstance(func, ast.Name):
        return names.get(func.id)
    if isinstance(func, ast.Attribute) and func.attr == "cell_averages":
        base = func.value
        if isinstance(base, ast.Name) and base.id in module_aliases:
            module = module_aliases[base.id]
            if module in ("benchmarks.sod_exact", "benchmarks.riemann_exact"):
                return FIXED_RULE
        if isinstance(base, ast.Attribute) and base.attr in ("sod_exact", "riemann_exact"):
            inner = base.value
            if isinstance(inner, ast.Name) and inner.id in ("benchmarks",):
                return FIXED_RULE
    return None


def discover() -> List[Dict[str, Any]]:
    """Every call site of the fixed-rule reference in the scanned roots."""
    sites: List[Dict[str, Any]] = []
    for path in _scan_files():
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:                                # pragma: no cover
            continue
        names, module_aliases = _imports(tree)
        if not names and not module_aliases:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            symbol = _canonical(node.func, names, module_aliases)
            if symbol is None:
                continue
            rel = str(path.relative_to(REPO)).replace("\\", "/")
            sites.append({"path": rel, "line": int(node.lineno), "symbol": symbol})
    return sorted(sites, key=lambda s: (s["path"], s["line"]))


# --------------------------------------------------------------------------- #
# the declared registry                                                       #
# --------------------------------------------------------------------------- #
#: ``role`` vocabulary
#:   ``gating``                 the number is compared against a tolerance / decides a verdict
#:   ``plot``                   the number is only drawn
#:   ``diagnostic``             printed or explored, no verdict attached
#:   ``instrument_measurement`` deliberately measures the legacy rule *against* the exact one
#:
#: ``status`` says what this census did about the site:
#:   ``switched_to_exact``  the site now scores against the exact instrument (this change)
#:   ``at_risk_legacy``     still on the fixed rule; the measured bias is a bound on how much
#:                          of its gate margin the instrument can eat (see :func:`gate_risk`)
#:   ``n/a``                the site publishes no verdict, so the instrument does not gate
#:
#: Roles are read off the module's own tolerance constants (``TOLERANCE``/``GATE``/assert
#: thresholds) by static inspection; the notes say so where that is all that was done.
G, D, P, M, N = "gating", "diagnostic", "plot", "instrument_measurement", "n/a"
S, R = "switched_to_exact", "at_risk_legacy"

#: ``(path, role, status, note)`` -- one entry per file that calls the fixed rule.
_SITES: Tuple[Tuple[str, str, str, str], ...] = (
    ("validation/repro_cks1998/diagnose_sod_tvd.py", D, N,
     "TVD diagnosis print-out, no tolerance attached"),
    ("validation/repro_cks1998/station_reference.py", M, N,
     "the E2 station instrument: keeps the legacy rule on purpose to measure its bias"),
    # ---- repo-wide: declared and measured here, not changed (out of this task) ----
    ("benchmarks/artificial_viscosity_verification.py", G, R,
     "AV verification compares relative L2 against a tolerance"),
    ("benchmarks/audit_gate_power.py", G, R,
     "gate-power audit: the gate under audit is a 3 % Sod gate"),
    ("benchmarks/av_independent_check.py", D, N, "independent check, prints both sides"),
    ("benchmarks/certified_reference_sod.py", M, N,
     "builds the certified Sod reference: uses the exact instrument and this one to compare"),
    ("benchmarks/char_limiter_2d.py", G, R, "2-D characteristic limiter gate on Sod"),
    ("benchmarks/compare_euler_sod_1d.py", P, N, "comparison figure only"),
    ("benchmarks/in31_hp_ladder.py", G, R, "hp ladder error norms feed observed orders"),
    ("benchmarks/lax_reference_crosscheck.py", G, R, "Lax (Toro test 2) cross-check vs tolerance"),
    ("benchmarks/report_euler_dg_2d_verification.py", P, N, "report figure assembly"),
    ("benchmarks/riemann_vacuum_1d.py", G, R, "vacuum Riemann case (its waves differ)"),
    ("benchmarks/sod_dg_benchmark.py", G, R, "TOLERANCE = 3 % on max relative error"),
    ("benchmarks/validate_limiter_reconstruction.py", D, N,
     "limiter-reconstruction study; NOTE the import aliases the *fixed* rule to a name "
     "that says 'exact' (line 44: `cell_averages as exact_cell_averages`) -- a naming "
     "defect this census records but does not rename"),
    ("benchmarks/verification_ladder.py", G, R,
     "order / shock-position / monotonicity criteria are asserted against tolerances"),
    ("benchmarks/verify_euler_2d_p0.py", G, R, "GATE = 3 %"),
    ("benchmarks/verify_euler_dg_2d.py", G, R, "THREE_PERCENT = 3 %"),
    ("benchmarks/verify_euler_dg_2d_limiter.py", D, N, "limiter comparison rows, no verdict"),
    ("benchmarks/verify_limiter_2d.py", D, N, "limiter probe, no verdict"),
    ("benchmarks/verify_limiter_2d_m2596.py", D, N, "extruded Sod probe, no verdict"),
    ("benchmarks/verify_sod_density_peak.py", G, R, "L2_TOLERANCE = 3 %, position 4 %"),
    ("tests/test_euler_2d_p0_verification.py", G, R,
     "extruded 2-D P0 Sod gate: the tightest recorded margin of the family (~0.5 pp)"),
    ("tests/test_euler_dg_2d.py", G, R, "2-D vs 1-D DG agreement asserted at 2.5e-3"),
    ("tests/test_euler_sod_1d.py", G, R, "1-D P0 Sod gate"),
    ("tests/test_euler_verification_chain.py", G, R,
     "per-field pinned regression guards (recorded value x 1.10)"),
    ("tests/test_lax_reference_crosscheck.py", G, R, "Lax cross-check gate"),
    ("tests/test_limiter_2d.py", G, R,
     "`test_weno_limited_squeezed_sod_meets_the_three_percent_gate`"),
    ("tests/test_repro_cks1998.py", G, R, "the anchor's own DG P2 N=400 Sod gate"),
    ("tests/test_sod_l1_ladder.py", G, R, "re-asserts the pre-registered ladder criteria"),
    ("tests/test_verification_ladder.py", G, R, "order criterion asserted at 15 %"),
    ("validation/verification_chain/shock_tube_refinement.py", G, R,
     "relative-L2 refinement ladder: the non-convergent reference is in the order"),
)

#: Sites this census **moved** off the fixed rule: on the anchor's own measurement path
#: the reference decides a 3 % verdict, so the exact instrument is required by the
#: repository's own policy.  They are not call sites of the fixed rule any more, and the
#: tests assert exactly that (a regression would put them back in :func:`discover`).
SWITCHED_BY_THIS_CENSUS: Tuple[Dict[str, str], ...] = (
    {"path": "validation/repro_cks1998/sod_gate_margin.py",
     "was": "E2 gate-margin study: its docstring already promised 'exact Riemann cell "
            "averages' while the code called the fixed rule",
     "now": "gate numbers from station_reference.exact_cell_averages; the legacy rule is "
            "still evaluated on the same solution and recorded as *_legacy, so every row "
            "keeps its own before/after and nothing can flip unobserved"},
    {"path": "validation/repro_cks1998/repro_cks1998.py",
     "was": "E2 Sod section: per-degree relative L2 and the station table, both scored at 3 %",
     "now": "station_reference.exact_cell_averages"},
    {"path": "validation/repro_cks1998/sod_l1_ladder.py",
     "was": "the gated quantity is the observed L1 *order*; a reference that does not "
            "converge contaminates the order itself, not just one error value",
     "now": "station_reference.exact_cell_averages for the cell-average L1 leg"},
)


REGISTRY: Tuple[Dict[str, Any], ...] = tuple(
    {"path": path, "role": role, "status": status, "note": note}
    for path, role, status, note in _SITES
)

#: Gate margins this repository already recorded elsewhere, and where.  ``None`` means
#: nobody has measured the margin, so the instrument risk at that site is unbounded and
#: :func:`gate_risk` says so instead of guessing.  Nothing here is re-measured by the
#: census -- the source string is the evidence.
RECORDED_MARGINS: Dict[str, Dict[str, Any]] = {
    "tests/test_repro_cks1998.py": {
        "margin_pp": 100.0 * (0.03 - 0.02368),
        "source": "docs/CASE-REPRO-DG-CKS1998-... gate margin study: DG P2 N=400 Sod "
                  "reproduces 2.368 % against the 3 % bar",
    },
    "validation/repro_cks1998/sod_gate_margin.py": {
        "margin_pp": 100.0 * (0.03 - 0.02368),
        "source": "same gate configuration (limiter=tvb, M=1, final stage, N=400, CFL 0.12)",
    },
    "tests/test_euler_2d_p0_verification.py": {
        "margin_pp": 0.5,
        "source": "benchmarks/shock_reference_quadrature.py docstring: 'the limiting field "
                  "of the extruded-2D gate sits at 2.5 % of a 3 % bar'",
    },
}


def _dedup(registry: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for entry in registry:
        key = (entry["path"], entry.get("symbol", FIXED_RULE))
        if key in seen:
            continue
        seen[key] = dict(entry)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# the census proper                                                           #
# --------------------------------------------------------------------------- #
def census(registry: Iterable[Mapping[str, Any]] = REGISTRY) -> Dict[str, Any]:
    """Discovered sites joined with the declared roles; fails closed on any gap."""
    declared = {(e["path"], e.get("symbol", FIXED_RULE)): e for e in _dedup(registry)}
    sites = discover()
    found = {(s["path"], s["symbol"]) for s in sites}
    undeclared = [s for s in sites if (s["path"], s["symbol"]) not in declared]
    if undeclared:
        listing = "\n".join(f"  {s['path']}:{s['line']}  {s['symbol']}" for s in undeclared)
        raise UndeclaredInstrumentSite(
            "fixed-rule reference instrument used without a declared role "
            f"({len(undeclared)} site(s)):\n{listing}\n"
            "Declare each in validation/repro_cks1998/instrument_census.REGISTRY with a "
            "role, or switch it to benchmarks.shock_reference_quadrature.cell_averages."
        )
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for site in sites:
        by_path.setdefault(site["path"], []).append(site)
    rows = []
    for path, occurrences in sorted(by_path.items()):
        entry = declared[(path, FIXED_RULE)]
        margin = RECORDED_MARGINS.get(path)
        rows.append({
            "path": path,
            "lines": [s["line"] for s in occurrences],
            "calls": len(occurrences),
            "symbol": FIXED_RULE,
            "role": entry["role"],
            "status": entry["status"],
            "note": entry.get("note"),
            "recorded_margin_pp": None if margin is None else margin["margin_pp"],
            "margin_source": None if margin is None else margin["source"],
        })
    roles = [r["role"] for r in rows]
    return {
        "fixed_rule": FIXED_RULE,
        "exact_rule": EXACT_RULE,
        "roots": list(SCAN_ROOTS),
        "files": len(rows),
        "calls": sum(r["calls"] for r in rows),
        "by_role": {role: roles.count(role) for role in sorted(set(roles))},
        "switched": [dict(e) for e in SWITCHED_BY_THIS_CENSUS],
        "gating_at_risk_legacy": sorted(r["path"] for r in rows
                                        if r["status"] == "at_risk_legacy"),
        # a declared entry with no call site left is drift in the other direction: the
        # registry would then be describing code that no longer exists
        "declared_without_site": sorted(
            f"{path}" for (path, symbol) in declared
            if (path, symbol) not in found
        ),
        "sites": rows,
    }


# --------------------------------------------------------------------------- #
# the measured bias of the legacy instrument                                  #
# --------------------------------------------------------------------------- #
def wave_cells(ncells: int, final_time: float = SOD_T) -> Dict[str, Any]:
    """Cells of the Sod mesh that contain an exact breakpoint (where the rule lies)."""
    breaks = np.asarray(srq.breakpoints(SOD_LEFT, SOD_RIGHT, final_time, 1.4, 0.5, (0.0, 1.0)),
                        dtype=float)
    dx = 1.0 / ncells
    idx = np.clip(np.floor(breaks / dx).astype(int), 0, ncells - 1)
    cells = sorted({int(i) for i in idx})
    return {"ncells": int(ncells), "breakpoints": [float(b) for b in breaks],
            "cells": cells, "count": len(cells), "fraction": len(cells) / ncells}


def bias_table(meshes: Iterable[int] = BIAS_MESHES,
               final_time: float = SOD_T) -> List[Dict[str, Any]]:
    """How far the fixed-rule reference sits from the exact one, per mesh and field."""
    rows = []
    for ncells in meshes:
        _, exact = srq.cell_averages(int(ncells), final_time, SOD_LEFT, SOD_RIGHT)
        _, fixed = srq.fixed_rule_cell_averages(int(ncells), final_time, SOD_LEFT, SOD_RIGHT)
        fields = {}
        for key in srq.FIELDS:
            ref = np.asarray(exact[key])
            diff = np.asarray(fixed[key]) - ref
            rel_cell = np.abs(diff) / np.maximum(np.abs(ref), 1e-300)
            fields[key] = {
                "rel_l2": float(np.sqrt(np.mean(diff ** 2))
                                / np.sqrt(np.mean(ref ** 2))),
                "max_abs": float(np.max(np.abs(diff))),
                "max_rel_cell": float(np.max(rel_cell)),
            }
        worst = max(fields, key=lambda k: fields[k]["rel_l2"])
        rows.append({
            "ncells": int(ncells),
            "cells_with_wave": wave_cells(int(ncells), final_time)["count"],
            "fields": fields,
            "worst_field": worst,
            "worst_rel_l2": fields[worst]["rel_l2"],
            # the legacy reference is the denominator, so a reference bias of this size
            # moves a reported relative error by about this many percentage points
            "worst_shift_pp": 100.0 * fields[worst]["rel_l2"],
        })
    return rows


def gate_risk(margin_pp: float | None, bias_pp: float) -> Dict[str, Any]:
    """Can a measured gate margin survive the instrument, or does the bias eat it?"""
    if margin_pp is None:
        return {"margin_pp": None, "bias_pp": float(bias_pp),
                "margin_with_exact_pp": None, "at_risk": None,
                "bias_share_of_margin": None,
                "verdict": "unknown_margin"}
    return {
        "margin_pp": float(margin_pp),
        "bias_pp": float(bias_pp),
        "margin_with_exact_pp": float(margin_pp - bias_pp),
        "at_risk": bool(margin_pp - bias_pp <= 0.0),
        "bias_share_of_margin": float(bias_pp / margin_pp) if margin_pp > 0 else float("inf"),
        "verdict": "cannot_flip" if margin_pp - bias_pp > 0 else "may_flip",
    }


def gating_bound(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """The one-line statement the census can actually support.

    A 3 %-gate whose exact-instrument margin exceeds the largest measured reference
    bias cannot be flipped by the instrument, whatever its field or mesh.  Anything
    tighter has to be run, and :func:`gate_risk` marks which recorded margins are.
    """
    bias = max((row["worst_rel_l2"] for row in payload["bias"]), default=0.0)
    per_site = {}
    for site in payload["sites"]:
        if site["role"] != "gating":
            continue
        per_site[site["path"]] = {
            "status": site["status"],
            "risk": gate_risk(site["recorded_margin_pp"], 100.0 * bias),
            "margin_source": site["margin_source"],
        }
    at_risk = sorted(p for p, v in per_site.items()
                     if v["risk"]["at_risk"] is True)
    unknown = sorted(p for p, v in per_site.items()
                     if v["risk"]["verdict"] == "unknown_margin")
    return {
        "tolerance_3pct_pp": 3.0,
        "worst_bias_pp": 100.0 * bias,
        "claim": (f"any 3 % gate with a margin > {100.0 * bias:.2f} pp cannot be flipped by "
                  "the reference instrument; tighter margins must be run"),
        "gating_sites": per_site,
        "at_risk": at_risk,
        "unknown_margin": unknown,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def build_payload() -> Dict[str, Any]:
    payload = census()
    payload["bias"] = bias_table()
    payload["bias_meshes"] = list(BIAS_MESHES)
    payload["sod"] = {"left": list(SOD_LEFT), "right": list(SOD_RIGHT), "final_time": SOD_T}
    payload["gating_bound"] = gating_bound(payload)
    return payload


def render(payload: Mapping[str, Any]) -> str:
    lines = [f"fixed rule : {payload['fixed_rule']}",
             f"exact rule : {payload['exact_rule']}",
             f"files {payload['files']} / calls {payload['calls']} / roles {payload['by_role']}",
             ""]
    for site in payload["sites"]:
        margin = site["recorded_margin_pp"]
        margin_s = "" if margin is None else f"{margin:.2f}pp"
        lines.append(f"{site['path']:<56}{site['calls']:>3}  "
                     f"{site['role']:<21}{site['status']:<19}{margin_s}")
    lines += ["", f"{'ncells':>7}{'wave cells':>12}{'rho %':>9}{'u %':>9}{'p %':>9}"
                  f"{'shift pp':>10}"]
    for row in payload["bias"]:
        f = row["fields"]
        lines.append(f"{row['ncells']:>7}{row['cells_with_wave']:>12}"
                     f"{100 * f['density']['rel_l2']:>9.4f}"
                     f"{100 * f['velocity']['rel_l2']:>9.4f}"
                     f"{100 * f['pressure']['rel_l2']:>9.4f}"
                     f"{row['worst_shift_pp']:>10.4f}")
    bound = payload["gating_bound"]
    lines += ["", f"gating bound: {bound['claim']}",
              f"  at risk  : {bound['at_risk'] or 'none'}",
              f"  unknown  : {len(bound['unknown_margin'])} of {len(bound['gating_sites'])} "
              f"gating sites have no recorded margin"]
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fixed-rule reference instrument census")
    parser.add_argument("--json", type=Path, help="write the census payload here")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if a fixed-rule site is undeclared")
    args = parser.parse_args(argv)
    try:
        payload = build_payload()
    except UndeclaredInstrumentSite as error:
        print(str(error), file=sys.stderr)
        return 1
    print(render(payload))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
