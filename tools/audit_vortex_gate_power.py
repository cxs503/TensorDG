#!/usr/bin/env python3
"""Failure-injection (mutation) audit of the ``CASE-DG-EULER-VORTEX-ERR`` gate.

Why this exists
---------------
Rounds 1-19 of the moving-vortex case turned eleven criteria (C1-C11) green on
the *published* bundle.  A green verdict only says the gate accepted the real
evidence; it does **not** say the gate would have refused the evidence a real
defect produces.  Ten of the eleven criteria carry a hand-written
counter-example test (``tests/test_case_euler_vortex_errconserve*.py``); C4
(entropy budget) did not, so its discriminating power was an assumption.

This tool closes that gap mechanically instead of by inspection.  It loads the
**committed** evidence bundle (40 result records + 19 replay certificates, no
synthetic fixtures), confirms the published verdict reproduces, then injects one
plausible defect at a time and records the *full* status vector of the gate.
A criterion is "killed" when a defect aimed at it turns it into ``failed`` (or
``not_evaluated``, which blocks the verdict just as hard).

Two things are measured:

``--power`` (default with the table)
    killing power -- one defect per criterion, plus two control mutations that
    change fields no criterion is about (wall clock, iteration counters).  A
    control that flips anything means the gate reacts to noise.

``--calibrate``
    threshold calibration -- for every criterion whose threshold is a named
    constant, one injected defect just *inside* the threshold (must stay
    ``passed``) and one just *outside* (must read ``failed``).  This is the
    text/code agreement check: it proves the number the criterion *states* is
    the number the code *applies*.

Exit status: ``0`` when every acceptance criterion is killed by at least one
mutation, both controls are inert and every calibration pair brackets its
constant, ``2`` otherwise (that is a hole in the gate, and the report names it).

Usage::

    PYTHONPATH=. python3 tools/audit_vortex_gate_power.py --dry-run
    PYTHONPATH=. python3 tools/audit_vortex_gate_power.py --calibrate
    PYTHONPATH=. python3 tools/audit_vortex_gate_power.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import case_euler_vortex_errconserve as ec  # noqa: E402

BUNDLE = ROOT / "case_euler_vortex_errconserve"

#: every criterion that gates the verdict (``gates["informational"]`` excluded)
ACCEPTANCE = (
    "C1_tolerance",
    "C2_error_conservation",
    "C3_discrete_conservation",
    "C4_entropy_budget",
    "C5_crosscheck_order",
    "C6_ladder_completeness",
    "C7_design_space_invariance",
    "C8_degree_invariance",
    "C9_design_order",
    "C10_direction_invariance",
    "C11_provenance_integrity",
    "C12_growth_statistic_robustness",
)


# ---------------------------------------------------------------------------
# bundle access
# ---------------------------------------------------------------------------

def load_bundle(bundle: Path = BUNDLE):
    """The committed evidence bundle, exactly as the aggregator reads it."""
    records = [json.loads(path.read_text())
               for path in sorted((bundle / "results").glob("*.json"))]
    if not records:
        raise SystemExit(f"no records under {bundle / 'results'}")
    exact = ec.exact_invariants()
    exact["core_density"] = ec.exact_core_density()
    certificates = ec.load_provenance_certificates(bundle)
    return records, exact, certificates


def evaluate(records, exact, certificates):
    return ec.evaluate_gates(copy.deepcopy(records), exact,
                             copy.deepcopy(certificates))["gates"]


def statuses(gates) -> dict:
    return {name: gates[name]["status"] for name in ACCEPTANCE}


# ---------------------------------------------------------------------------
# mutation primitives -- each one mimics a defect a real bug would produce
# ---------------------------------------------------------------------------

def record_by_key(records, key):
    for record in records:
        if record["task_key"] == key:
            return record
    raise KeyError(key)


def set_errors(record, value: float) -> None:
    """Overwrite every field error (the shape of a wrong-reference or blown run)."""
    record["errors"] = {name: value for name in ec.FIELDS}
    record["max_relative_l2"] = value
    record["max_percent"] = 100.0 * value
    record["gate_pass"] = value < ec.TOLERANCE


def scale_errors(record, factor: float) -> None:
    record["errors"] = {name: value * factor for name, value in record["errors"].items()}
    record["max_relative_l2"] = max(record["errors"].values())
    record["max_percent"] = 100.0 * record["max_relative_l2"]
    record["gate_pass"] = record["max_relative_l2"] < ec.TOLERANCE


def _published_exponent(records, family: str) -> float:
    """Least-squares exponent of the family as published (so the mutation sets it)."""
    rows = [r for r in records if ec.family_key(r) == family]
    if len(rows) < 2:
        return 0.0
    times = [float(r["final_time"]) for r in rows]
    errors = [float(r["max_relative_l2"]) for r in rows]
    return float(ec._fit_exponent(times, errors, t_min=0.0))


def growth_on_family(records, family: str, exponent: float) -> list:
    """Re-shape one family's errors so that the *fitted* exponent is ``exponent``.

    This is what a scheme with super-linear error accumulation reports, i.e. the
    exact defect C2/C7/C8/C10 exist to refuse.
    """
    shift = exponent - _published_exponent(records, family)
    touched = []
    for record in records:
        if ec.family_key(record) != family:
            continue
        scale_errors(record, float(record["final_time"]) ** shift)
        touched.append(record["task_key"])
    return touched


def set_entropy_drift(records, value: float, key="dg_d2_n32_t10_cfl0.2") -> str:
    record = record_by_key(records, key)
    record["entropy_production"] = value
    record["entropy_drift"] = abs(value)
    return key


def set_mass_drift(records, drift: float, key="dg_d2_n32_t1_cfl0.2") -> str:
    record = record_by_key(records, key)
    record["final"]["invariants"]["mass"] = (
        record["initial"]["invariants"]["mass"] * (1.0 + drift))
    return key


def set_certificate_deviation(certificates, relative: float,
                              key="dg_d2_n32_t1_cfl0.2") -> str:
    certificate = certificates[key]
    certificate["stored_max_percent"] = certificate["stored_max_percent"] * (1.0 + relative)
    return key


def set_p0_order(records, order: float, anchor_error: float = 1.0e-3) -> str:
    """Force the P0 cross-check errors to ``e(n) = C n^-order`` (fitted order = order)."""
    anchor_cells = 64
    for cells in (64, 128, 256):
        set_errors(record_by_key(records, f"p0_n{cells}_t1_cfl0.4"),
                   anchor_error * (cells / anchor_cells) ** (-order))
    return f"P0 t=1 errors ~ n^-{order:g}"


def set_dg_order(records, order: float, checkpoint: float,
                 degree: int = 2, cells=(32, 64)) -> str:
    """Force one DG refinement pair to show ``order`` (C9's measured quantity)."""
    coarse = f"dg_d{degree}_n{cells[0]}_t{checkpoint:g}_cfl0.2"
    fine = f"dg_d{degree}_n{cells[1]}_t{checkpoint:g}_cfl0.2"
    target = float(record_by_key(records, coarse)["max_relative_l2"]) / 2.0 ** order
    set_errors(record_by_key(records, fine), target)
    return f"{fine} set to {target:.3e} ({order:g}-th order against n={cells[0]})"


# ---------------------------------------------------------------------------
# the mutation catalogue: one plausible defect per criterion
# ---------------------------------------------------------------------------

def mutation_c1(records, certificates):
    """A checkpoint measured against a wrong reference: 0.0094 % -> 7 %."""
    set_errors(record_by_key(records, "dg_d2_n32_t1_cfl0.2"), 7.0e-2)
    return "dg_d2_n32_t1_cfl0.2 errors 0.0094% -> 7.00% (above the 6% acceptance)"


def mutation_c2(records, certificates):
    """Super-linear accumulation on the *design* family."""
    return ("design family errors reshaped to e ~ t^3: "
            + ", ".join(growth_on_family(records, ec.GATE_FAMILY, 3.0)))


def mutation_c3(records, certificates):
    """A leaky discretisation: mass drifts 1% over t = 1."""
    return f"{set_mass_drift(records, 1e-2)} final mass +1%"


def mutation_c4(records, certificates):
    """An entropy-unstable scheme: |dS|/S = 2e-3 on the design family."""
    key = set_entropy_drift(records, 2.0e-3)
    return f"{key} |dS|/S 3e-7 -> 2e-3 (2x the 1e-3 budget)"


def mutation_c5(records, certificates):
    """A cross-check path that does not converge: observed order -> 0."""
    worst = max(float(r["max_relative_l2"]) for r in records
                if ec.family_key(r) == "p0_n64")
    for key in ("p0_n256_t1_cfl0.4", "p0_n128_t1_cfl0.4", "p0_n64_t1_cfl0.4"):
        set_errors(record_by_key(records, key), worst)
    return "P0 n=64/128/256 at t=1 all set to the n=64 error (observed order 0)"


def mutation_c6(records, certificates):
    """A declared checkpoint whose evidence silently vanished."""
    dropped = "dg_d2_n64_t10_cfl0.2"
    records[:] = [r for r in records if r["task_key"] != dropped]
    return f"{dropped} removed from the bundle"


def mutation_c7(records, certificates):
    """Conservation that holds only on the design point (refined mesh)."""
    return ("non-design point dg_d2_n64_cfl0.2 errors reshaped to e ~ t^3: "
            + ", ".join(growth_on_family(records, "dg_d2_n64_cfl0.2", 3.0)))


def mutation_c8(records, certificates):
    """Conservation that holds only at the design degree (p=2 collapses at p=3)."""
    return ("degree p=3 family errors reshaped to e ~ t^3: "
            + ", ".join(growth_on_family(records, "dg_d3_n32_cfl0.2", 3.0)))


def mutation_c9(records, certificates):
    """Mesh refinement stops buying accuracy at the long-time checkpoint."""
    coarse = float(record_by_key(records, "dg_d2_n32_t10_cfl0.2")["max_relative_l2"])
    set_errors(record_by_key(records, "dg_d2_n64_t10_cfl0.2"), coarse)
    return "dg_d2_n64_t10 error forced to the n=32 value (observed order 0)"


def mutation_c10(records, certificates):
    """Anisotropy the direction axis exists to catch: growth on a rotated leg."""
    family = "dg_d2_n32_cfl0.2_dirobl30"
    return (f"rotated-direction family {family} errors reshaped to e ~ t^3: "
            + ", ".join(growth_on_family(records, family, 3.0)))


def mutation_c11_stale(records, certificates):
    """A certificate that replayed a different number than the record claims."""
    key = set_certificate_deviation(certificates, 0.5)
    return f"{key} certificate stored_max_percent inflated 50% (replay no longer reproduces)"


def mutation_c11_unverified(records, certificates):
    """A legacy record with no certificate at all: unproven provenance."""
    dropped = "dg_d2_n32_t10_cfl0.2"
    certificates.pop(dropped, None)
    return f"{dropped} certificate removed (evidence provenance unknown)"


#: the five checkpoints of the design family, in transport order
DESIGN_FAMILY_KEYS = ("dg_d2_n32_t1_cfl0.2", "dg_d2_n32_t2_cfl0.2",
                      "dg_d2_n32_t5_cfl0.2", "dg_d2_n32_t10_cfl0.2",
                      "dg_d2_n32_t20_cfl0.2")


def late_onset_rows(records):
    rows = [record_by_key(records, key) for key in DESIGN_FAMILY_KEYS]
    rows.sort(key=lambda r: r["final_time"])
    return rows


def robust_statistic_of(rows) -> float:
    """The C12 reading a given set of design-family rows would produce."""
    rows = sorted(rows, key=lambda r: r["final_time"])
    variant = ec.exponent_variants([float(r["final_time"]) for r in rows],
                                   [float(r["max_relative_l2"]) for r in rows])
    return float(variant["robust"])


def apply_late_onset(records, exponent: float):
    """Flat early checkpoints, then the last one at ``e(1) * (t_last / 1) ** exponent``.

    This is the *late-onset* defect: the error sits at its projection floor for the
    first box crossing and only then starts to grow.  The least-squares slope over
    the whole window is diluted by the flat part, so C2/C7/C8 stay green; the
    endpoint (and leave-one-out) reading sees it.  It is exactly the shape C12 was
    written for -- a run that is steady early and unstable late is the one a
    whole-window regression hides.
    """
    rows = late_onset_rows(records)
    base = float(rows[0]["max_relative_l2"])
    last_time = float(rows[-1]["final_time"])
    for row in rows[:-1]:
        set_errors(row, base)
    set_errors(rows[-1], base * last_time ** exponent)
    return rows


def solve_late_onset(records, magnitude: float) -> float:
    """Late-onset exponent whose C12 reading reaches ``magnitude`` (bisection).

    The probe uses the *full* bootstrap, so the number this returns is the number
    ``evaluate`` reports: a calibration probe built on a cheaper statistic would
    bracket one quantity and grade another (which is exactly how a threshold
    calibration can look right and be wrong).
    """
    low, high = 0.0, 3.0
    for _ in range(18):
        middle = 0.5 * (low + high)
        if robust_statistic_of(apply_late_onset(records, middle)) < magnitude:
            low = middle
        else:
            high = middle
    return high


def mutation_c12(records, certificates):
    rows = apply_late_onset(records, 1.3)
    variant = ec.exponent_variants([float(r["final_time"]) for r in rows],
                                   [float(r["max_relative_l2"]) for r in rows])
    ols = float(ec._fit_exponent([float(r["final_time"]) for r in rows],
                                 [float(r["max_relative_l2"]) for r in rows]))
    return (f"late-onset growth on the design family, flat to t=10 then unstable "
            f"(endpoint "
            f"{variant['endpoint']:.3f}, OLS {ols:.3f}, robust "
            f"{variant['robust']:.3f}); the whole-window regression still reads "
            f"alpha={ols:.3f} <= {ec.MAX_GROWTH_EXPONENT}")


def cal_c12(magnitude):
    def apply(records, certificates):
        exponent = solve_late_onset(records, magnitude)
        achieved = robust_statistic_of(late_onset_rows(records))
        return (f"late-onset exponent {exponent:.6f} -> robust statistic "
                f"{achieved:.6f}")
    return apply


def control_wall_clock(records, certificates):
    """Control: timing metadata only -- no criterion may react."""
    for record in records:
        record["wall_seconds"] = float(record.get("wall_seconds") or 0.0) * 10.0
    return "every wall_seconds x10"


def control_iteration_counters(records, certificates):
    """Control: iteration bookkeeping only -- no criterion may react."""
    for record in records:
        record["steps"] = int(record.get("steps") or 1) * 3
    return "every steps x3"


#: (id, target criterion or ``None`` for a control, description, mutator)
MUTATIONS = (
    ("m01", "C1_tolerance", "checkpoint above the 6% acceptance", mutation_c1),
    ("m02", "C2_error_conservation", "super-linear accumulation, design family", mutation_c2),
    ("m03", "C3_discrete_conservation", "mass leak of 1%", mutation_c3),
    ("m04", "C4_entropy_budget", "entropy budget overrun", mutation_c4),
    ("m05", "C5_crosscheck_order", "cross-check path stops converging", mutation_c5),
    ("m06", "C6_ladder_completeness", "checkpoint evidence missing", mutation_c6),
    ("m07", "C7_design_space_invariance", "growth off the design point", mutation_c7),
    ("m08", "C8_degree_invariance", "growth at a non-design degree", mutation_c8),
    ("m09", "C9_design_order", "refinement stops buying accuracy", mutation_c9),
    ("m10", "C10_direction_invariance", "growth on a rotated direction", mutation_c10),
    ("m11", "C11_provenance_integrity", "certificate no longer reproduces", mutation_c11_stale),
    ("m12", "C11_provenance_integrity", "legacy record left unverified",
     mutation_c11_unverified),
    ("m13", "C12_growth_statistic_robustness",
     "late-onset growth: flat for one box crossing, then unstable "
     "(invisible to the whole-window least-squares slope)",
     mutation_c12),
    ("c01", None, "control: timing metadata", control_wall_clock),
    ("c02", None, "control: iteration counters", control_iteration_counters),
)


# ---------------------------------------------------------------------------
# threshold calibration: does the number the criterion *states* get applied?
# ---------------------------------------------------------------------------

def cal_c1(magnitude):
    def apply(records, certificates):
        set_errors(record_by_key(records, "dg_d2_n32_t1_cfl0.2"), magnitude)
        return f"snapshot error {magnitude:.4%} at t=1"
    return apply


def cal_c2(magnitude):
    def apply(records, certificates):
        return "design family exponent -> " + ", ".join(
            growth_on_family(records, ec.GATE_FAMILY, magnitude))
    return apply


def cal_c3(magnitude):
    def apply(records, certificates):
        return f"mass drift {magnitude:.3e}"
    return apply


def cal_c4(magnitude):
    def apply(records, certificates):
        return "entropy drift " + f"{magnitude:.3e} on " + set_entropy_drift(records, magnitude)
    return apply


def cal_c5(magnitude):
    def apply(records, certificates):
        return set_p0_order(records, magnitude)
    return apply


def cal_c9(magnitude):
    def apply(records, certificates):
        return "; ".join(set_dg_order(records, magnitude, time) for time in (1.0, 10.0))
    return apply


def cal_c11(magnitude):
    def apply(records, certificates):
        return "certificate deviation " + set_certificate_deviation(certificates, magnitude)
    return apply


#: (id, criterion, stated threshold, threshold source, direction, builder)
#: direction ``+1`` means "fails when the injected magnitude exceeds the
#: threshold", ``-1`` means "fails when it falls below it".
CALIBRATIONS = (
    ("k01", "C1_tolerance", ec.TOLERANCE, "TOLERANCE", +1, cal_c1),
    ("k02", "C2_error_conservation", ec.MAX_GROWTH_EXPONENT,
     "MAX_GROWTH_EXPONENT", +1, cal_c2),
    ("k03", "C3_discrete_conservation", ec.INVARIANT_TOLERANCE,
     "INVARIANT_TOLERANCE", +1, cal_c3),
    ("k04", "C4_entropy_budget", ec.ENTROPY_TOLERANCE, "ENTROPY_TOLERANCE", +1, cal_c4),
    ("k05", "C5_crosscheck_order", 0.8, "0.8 (literal in evaluate_gates)", -1, cal_c5),
    ("k09", "C9_design_order", ec.ORDER_SLACK * 3.0, "ORDER_SLACK*(p+1), p=2", -1, cal_c9),
    ("k12", "C12_growth_statistic_robustness", ec.MAX_GROWTH_EXPONENT,
     "MAX_GROWTH_EXPONENT (conservative statistic)", +1, cal_c12),
    ("k11", "C11_provenance_integrity", ec.PROVENANCE_REPLAY_TOLERANCE,
     "PROVENANCE_REPLAY_TOLERANCE", +1, cal_c11),
)

#: multiplicative distance kept from a threshold so that the probe is decided by
#: the comparison and not by floating-point noise
CALIBRATION_SLACK = 1.0e-3


def calibrate(bundle: Path = BUNDLE) -> dict:
    records, exact, certificates = load_bundle(bundle)
    rows = []
    for ident, criterion, threshold, source, direction, builder in CALIBRATIONS:
        inner = threshold * (1.0 - CALIBRATION_SLACK if direction > 0
                             else 1.0 + CALIBRATION_SLACK)
        outer = threshold * (1.0 + CALIBRATION_SLACK if direction > 0
                             else 1.0 - CALIBRATION_SLACK)
        readings = {}
        for label, magnitude in (("inside", inner), ("outside", outer)):
            working_records = copy.deepcopy(records)
            working_certificates = copy.deepcopy(certificates)
            apply = builder(magnitude) if criterion != "C3_discrete_conservation" else None
            if criterion == "C3_discrete_conservation":
                detail = f"mass drift {magnitude:.3e}"
                set_mass_drift(working_records, magnitude)
            else:
                detail = apply(working_records, working_certificates)
            gates = evaluate(working_records, exact, working_certificates)
            readings[label] = {
                "magnitude": magnitude,
                "detail": detail,
                "status": gates[criterion]["status"],
                "verdict": gates["verdict"],
            }
        ok = (readings["inside"]["status"] == "passed"
              and readings["outside"]["status"] == "failed")
        rows.append({
            "id": ident,
            "criterion": criterion,
            "threshold": threshold,
            "threshold_source": source,
            "direction": ("fails above" if direction > 0 else "fails below"),
            "readings": readings,
            "brackets": ok,
        })
    return {
        "case": "EULER-VORTEX-ERR",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "slack": CALIBRATION_SLACK,
        "rows": rows,
        "calibrated": [row["criterion"] for row in rows if row["brackets"]],
        "miscalibrated": [row["criterion"] for row in rows if not row["brackets"]],
        "all_bracketed": all(row["brackets"] for row in rows),
    }


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------

def audit(bundle: Path = BUNDLE) -> dict:
    records, exact, certificates = load_bundle(bundle)
    baseline = evaluate(records, exact, certificates)
    base_status = statuses(baseline)
    base_verdict = baseline["verdict"]

    rows = []
    for ident, target, description, mutator in MUTATIONS:
        working_records = copy.deepcopy(records)
        working_certificates = copy.deepcopy(certificates)
        try:
            detail = mutator(working_records, working_certificates)
            gates = evaluate(working_records, exact, working_certificates)
            after = statuses(gates)
            verdict = gates["verdict"]
            error = None
        except Exception as exc:                     # pragma: no cover - reported
            detail, gates, after, verdict = "", {}, {}, None
            error = f"{type(exc).__name__}: {exc}"
        changed = sorted(name for name in ACCEPTANCE
                         if after.get(name) != base_status.get(name))
        killed = (target is not None
                  and after.get(target) in ("failed", "not_evaluated"))
        rows.append({
            "id": ident,
            "target": target,
            "kind": ("mutation" if target else "control"),
            "description": description,
            "detail": detail,
            "verdict": verdict,
            "statuses": after,
            "changed": changed,
            "killed": killed,
            "collateral": sorted(name for name in changed if name != target),
            "error": error,
        })

    killed_by = {name: [row["id"] for row in rows
                        if row["target"] == name and row["killed"]]
                 for name in ACCEPTANCE}
    survivors = [name for name in ACCEPTANCE if not killed_by[name]]
    controls = [row for row in rows if row["kind"] == "control"]
    control_inert = all(not row["changed"] and row["verdict"] == base_verdict
                        for row in controls)
    return {
        "case": "EULER-VORTEX-ERR",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "bundle": str(bundle),
        "record_count": len(records),
        "certificate_count": len(certificates),
        "tolerance_relative_l2": ec.TOLERANCE,
        "baseline": {"verdict": base_verdict, "statuses": base_status},
        "rows": rows,
        "acceptance": list(ACCEPTANCE),
        "killed_by": killed_by,
        "survivors": survivors,
        "control_inert": control_inert,
        "power": f"{len(ACCEPTANCE) - len(survivors)}/{len(ACCEPTANCE)}",
        "sound": bool(not survivors and control_inert
                      and all(row["error"] is None for row in rows)),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_markdown(result: dict, calibration: dict | None = None) -> str:
    base = result["baseline"]["statuses"]
    lines = [
        "| 变异 | 目标判据 | 注入的缺陷 | 裁决 | 目标判据读数 | 被杀 | 附带翻转 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        target = row["target"] or "—（对照）"
        reading = row["statuses"].get(row["target"], "—") if row["target"] else "—"
        verdict = row["verdict"] or "（异常）"
        kill = "是" if row["killed"] else ("—" if not row["target"] else "**否**")
        lines.append(
            f"| `{row['id']}` | {target} | {row['description']} | {verdict} | "
            f"{reading} | {kill} | {', '.join(row['collateral']) or '无'} |")
    matrix = ["", "### 全判据 x 全变异状态矩阵", ""]
    header = "| 判据 | 基线 | " + " | ".join(r["id"] for r in result["rows"]) + " |"
    matrix += [header, "|---" * (len(result["rows"]) + 2) + "|"]
    for name in result["acceptance"]:
        cells = [result["rows"][i]["statuses"].get(name, "—")
                 for i in range(len(result["rows"]))]
        matrix.append(f"| {name} | {base[name]} | " + " | ".join(cells) + " |")
    out = lines + matrix
    if calibration is not None:
        out += ["", "### 门限标定（阈值两侧各一针）", "",
                "| 标定 | 判据 | 声明阈值 | 来源 | 失败方向 | 内侧读数 | 外侧读数 | 咬合 |",
                "|---|---|---|---|---|---|---|---|"]
        for row in calibration["rows"]:
            inner, outer = row["readings"]["inside"], row["readings"]["outside"]
            out.append(
                f"| `{row['id']}` | {row['criterion']} | {row['threshold']:.3e} | "
                f"`{row['threshold_source']}` | {row['direction']} | "
                f"{inner['magnitude']:.3e} → {inner['status']} | "
                f"{outer['magnitude']:.3e} → {outer['status']} | "
                f"{'是' if row['brackets'] else '**否**'} |")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", default=str(BUNDLE))
    parser.add_argument("--json", default=str(BUNDLE / "gate_power_matrix.json"))
    parser.add_argument("--calibration-json",
                        default=str(BUNDLE / "gate_threshold_calibration.json"))
    parser.add_argument("--markdown", default=None,
                        help="write the human-readable matrices here")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the summary only, write nothing")
    parser.add_argument("--skip-calibration", action="store_true")
    args = parser.parse_args(argv)

    result = audit(Path(args.bundle))
    print(f"baseline verdict: {result['baseline']['verdict']} "
          f"({result['record_count']} records, {result['certificate_count']} certificates)")
    for row in result["rows"]:
        mark = "kill" if row["killed"] else ("inert" if row["kind"] == "control" else "SURVIVE")
        print(f"  {row['id']} -> {row['verdict']!s:10s} {mark:7s} "
              f"target={row['target'] or '-'} changed={row['changed']}")
    print(f"gate power: {result['power']}  survivors={result['survivors']}  "
          f"controls_inert={result['control_inert']}")

    calibration = None
    if not args.skip_calibration:
        calibration = calibrate(Path(args.bundle))
        for row in calibration["rows"]:
            inner, outer = row["readings"]["inside"], row["readings"]["outside"]
            print(f"  {row['id']} {row['criterion']:26s} {row['threshold']:.3e} "
                  f"inside={inner['status']:8s} outside={outer['status']:8s} "
                  f"{'ok' if row['brackets'] else 'MISCALIBRATED'}")
        print(f"threshold calibration: {len(calibration['calibrated'])}/{len(calibration['rows'])}"
              f"  miscalibrated={calibration['miscalibrated']}")

    sound = result["sound"] and (calibration is None or calibration["all_bracketed"])
    if args.dry_run:
        return 0 if sound else 2
    Path(args.json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if calibration is not None:
        Path(args.calibration_json).write_text(
            json.dumps(calibration, indent=2, ensure_ascii=False))
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(result, calibration))
    return 0 if sound else 2


if __name__ == "__main__":
    raise SystemExit(main())