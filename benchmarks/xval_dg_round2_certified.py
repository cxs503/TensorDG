"""Round-2 cross-validation of the certified reference: oracle, legs, platform block.

Why this file exists
--------------------
``tensordg.certified_reference`` answers one question -- *how big is the
reference's own discretisation error?* -- and answers it with a Roache/GCI band
plus three refusal rules.  That is necessary but not sufficient for a
verification claim, because a certified reference can still be **wrong in the
same way as the code it judges** if the only evidence for it is the same
`ladder -> Richardson -> band` machinery that the candidate legs come from.

This module is the missing second leg of the round-2 case
``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` (mission ``CASE-SHU-OSHER``): it takes the
certified reference *as data* and cross-validates it against **sources that do
not share its discretisation**:

* closed-form Rankine--Hugoniot values for the post-shock plateau density
  (``rho_L`` of the Mach-3 state, 3.857143) and for the transmitted-shock
  position (``x_L + W t``, ``W`` from the Hugoniot jump conditions);
* the exact global mass budget at ``t = 1.8`` (only the ``x = -5`` inflow
  contributes, so ``M(0) + rho_L u_L t`` is analytic);
* the **untouched** entropy wave downstream of the shock, where the solution is
  ``1 + 0.2 sin(5x)`` and therefore known analytically.

All three oracles are re-used from :mod:`tensordg.cases.shock_entropy` (the case
module, not the engine) -- so a defect in the engine's extrapolation cannot
silently move the oracle with it.

What is decided (thresholds predeclared, unchanged after looking at results)
---------------------------------------------------------------------------
``D1``  the reference is *usable*: FV ladder verdict ``CERTIFIED`` and its band
        ``U`` is at most ``1/3`` of the gate it is used at (gate non-vacuity).
``D2``  every shipped DG leg passes the strict gate ``rel L1 < 3 %`` -- the
        precision standard of the mission, not the loose 5 % case-pool gate.
``D3``  every oracle comparison on the certified reference is ``< 1 %``.
``D4``  no DG ladder may certify *itself*: all three DG families must be
        refused (``WITHHELD_ORDER_TOO_LOW``), otherwise the "reference" would
        be the code's own extrapolation and D1-D3 would be self-referential.

``accepted`` is the conjunction; the CLI exits non-zero when it is false, so a
red run cannot be mistaken for a green one.

Platform block
--------------
:func:`platform_block` turns the physics into the ``|got-ref|/|ref|`` form the
TensorBench contract wants (``tests``-side schema: ``benchmark`` / ``workload``
/ ``metrics`` with ``got_pointer`` / ``ref_pointer`` / ``tolerance_rel``, see
``validation/repro_cks1998/bench_spec.py``).  A relative L1 *error* of a leg is
deliberately **not** encoded as a metric: ``|err - ref|/|ref|`` with a non-zero
``ref`` would be a fabricated number (the same reasoning recorded in
``validation/repro_shu_osher/tensorbench_ingest.py``).  Leg errors stay in the
audit artifact and in the repo-side gate ``D2``.

Usage::

    python -m benchmarks.xval_dg_round2_certified                 # audit + artifacts
    python -m benchmarks.xval_dg_round2_certified --figures       # + 2 PNGs
    python -m benchmarks.xval_dg_round2_certified --json-stdout   # machine-readable
    python -m benchmarks.xval_dg_round2_certified --require-accepted   # CI gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python benchmarks/xval_dg_round2_certified.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.certified_reference import (
    CERTIFIED,
    U_SHARE_MAX,
    WITHHELD_ORDER_TOO_LOW,
    certify,
    judge,
)
from tensordg.cases.shock_entropy import (
    DOMAIN,
    FINAL_TIME,
    WAVE_AMPLITUDE,
    WAVE_NUMBER,
    exact_mass_budget,
    rankine_hugoniot_anchor,
    relative_l1_error,
)

__all__ = [
    "BENCH_ID",
    "BENCH_VER",
    "WORKLOAD_ID",
    "GATE_PERCENT",
    "STRICT_PERCENT",
    "ANCHOR_TOL_PERCENT",
    "FV_LEVELS",
    "DG_LEGS",
    "CASE_KEY",
    "leg_filename",
    "load_ladder",
    "certified_reference",
    "measure_metrics",
    "oracle_values",
    "cross_validate",
    "platform_block",
    "write_artifacts",
    "main",
]

# ---------------------------------------------------------------- identity
CASE_KEY = "shu_osher_shock_entropy_wave"
BENCH_ID = "XVAL-DG-ROUND2-CERTIFIED-REFERENCE"
BENCH_VER = "2026.1"
WORKLOAD_ID = "xval-dg-round2-certified"
WORKLOAD_VER = "1"

# ---------------------------------------------------------------- predeclared
#: Reference ladder tolerance at which the band/gate rule is checked (case pool).
GATE_PERCENT = 5.0
#: The mission's precision standard: every DG leg must be better than this.
STRICT_PERCENT = 3.0
#: Oracle tolerance on the certified reference (declared before the run).
ANCHOR_TOL_PERCENT = 1.0
#: Gate-validity share, restated so a reader does not have to trust the engine.
U_SHARE_MAX_DECLARED = U_SHARE_MAX

FV_LEVELS = (800, 1600, 3200)
#: (degree, ncells, cfl) -- the eleven shipped round-2 DG legs, in audit order.
DG_LEGS = (
    (1, 200, 0.4), (1, 400, 0.4), (1, 800, 0.4),
    (2, 200, 0.4), (2, 400, 0.4), (2, 800, 0.4),
    (3, 200, 0.4), (3, 400, 0.4), (3, 800, 0.4),
    (2, 400, 0.1), (2, 400, 0.2),
)

DEFAULT_RAW = "validation/repro_shu_osher/artifacts/raw"
DEFAULT_OUT = "validation/xval_dg_round2/artifacts"

#: Oracle metric table.  ``kind="relative"`` compares ``got`` with a non-zero
#: closed-form value (``|got-ref|/|ref|``, usable as a TensorBench metric).
#: ``kind="bounded"`` compares ``got`` with an *absolute* upper bound: the
#: deviation from the untouched entropy wave is already a percentage, its
#: reference value is 0, and encoding it as ``|got-0|/|0|`` would be a
#: fabricated metric (same reasoning as
#: ``validation/repro_shu_osher/tensorbench_ingest.py``).
ORACLE_METRICS = (
    {"name": "plateau_rho", "kind": "relative", "unit": "-",
     "description": "post-shock plateau density rho_L (Hugoniot, Mach 3)"},
    {"name": "shock_position", "kind": "relative", "unit": "-",
     "description": "transmitted-shock position x_L + W t (Hugoniot speed)"},
    {"name": "mass_integral", "kind": "relative", "unit": "-",
     "description": "global mass M(0) + rho_L u_L t (exact budget)"},
    {"name": "preshock_sine_linf_percent", "kind": "bounded", "unit": "%",
     "description": "untouched entropy wave 1 + 0.2 sin(5x), x >= 2.6 (max |err| %)"},
)


# ---------------------------------------------------------------- profiles
def leg_filename(degree: int, ncells: int, cfl: float) -> str:
    """Committed raw-profile name for one DG leg (no hand-typed names in callers)."""
    tag = f"{cfl:g}".replace(".", "p")
    return f"dg_cfl{tag}_limweno_n{ncells}_p{degree}.npz"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return np.asarray(d["x"], dtype=float), np.asarray(d["rho"], dtype=float)


def load_ladder(rawdir: Path | str = DEFAULT_RAW) -> dict:
    """Load the committed round-2 ladder: three FV levels + eleven DG legs."""
    raw = Path(rawdir)
    fv = []
    for n in FV_LEVELS:
        x, rho = _load(raw / f"fv_cfl0p3_n{n}.npz")
        fv.append({"ncells": n, "x": x, "field": rho})
    dg = []
    for degree, ncells, cfl in DG_LEGS:
        name = leg_filename(degree, ncells, cfl)
        x, rho = _load(raw / name)
        dg.append({"configuration": f"DG P{degree} N={ncells} CFL={cfl}",
                   "degree": degree, "ncells": ncells, "cfl": cfl,
                   "file": name, "x": x, "field": rho})
    return {"fv": fv, "dg": dg, "rawdir": str(raw)}


def certified_reference(rawdir: Path | str = DEFAULT_RAW,
                        tolerance_percent: float = GATE_PERCENT) -> dict:
    """Certify the FV reference and return it on its own (second-finest) grid."""
    fv = load_ladder(rawdir)["fv"]
    cert = certify([{"ncells": lv["ncells"], "field": lv["field"]} for lv in fv],
                   tolerance_percent=tolerance_percent)
    if cert["reference"] is None:
        return {"certification": cert, "x": None, "rho": None}
    rho = np.asarray(cert["reference"], dtype=float)
    n = rho.size
    x = DOMAIN[0] + (np.arange(n) + 0.5) * (DOMAIN[1] - DOMAIN[0]) / n
    return {"certification": cert, "x": x, "rho": rho}


# ---------------------------------------------------------------- measurements
def shock_position(x: np.ndarray, rho: np.ndarray, level: float = 2.0) -> float | None:
    """Rightmost upward crossing of ``rho = level`` -- the transmitted shock.

    Measured here independently of :func:`tensordg.cases.shock_entropy.gate_report`
    on purpose: the cross-validation must not inherit the case module's
    measurement when the case module's *closed form* is the oracle.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    for i in range(rho.size - 1, 0, -1):
        if rho[i - 1] > level >= rho[i]:
            return float(x[i])
    return None


def measure_metrics(x: np.ndarray, rho: np.ndarray) -> dict:
    """The four oracle-comparable physical quantities of one density profile."""
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    dx = (DOMAIN[1] - DOMAIN[0]) / rho.size
    quiet = x >= 2.6
    sine = 1.0 + WAVE_AMPLITUDE * np.sin(WAVE_NUMBER * x[quiet])
    sine_err = relative_l1_error(rho[quiet], sine)
    return {
        "ncells": int(rho.size),
        "plateau_rho": float(rho[x <= -3.0].mean()),
        "shock_position": shock_position(x, rho),
        "mass_integral": float(rho.sum() * dx),
        "preshock_sine_rel_l1_percent": sine_err["rel_l1_percent"],
        "preshock_sine_linf_percent": sine_err["linf_percent"],
        "wave_amplitude": float(0.5 * (rho[quiet].max() - rho[quiet].min())),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "rho_positive": bool(rho.min() > 0.0),
    }


def oracle_values() -> dict:
    """Closed-form / case-module values for the same quantities (the ``ref`` side)."""
    rh = rankine_hugoniot_anchor()
    budget = exact_mass_budget()
    return {
        "plateau_rho": float(rh["post_shock_plateau"]["rho"]),
        "shock_position": float(rh["shock_position_analytic"]),
        "mass_integral": float(budget["mass_final_predicted"]),
        # bounded metric: the reference value is 0 by construction, so it is
        # deliberately absent (see ORACLE_METRICS)
        "preshock_sine_linf_percent": None,
        "shock_speed_analytic": float(rh["shock_speed_analytic"]),
        "contact_position_analytic": float(rh["contact_position_analytic"]),
        "quiet_zone_head_analytic": float(rh["quiet_zone_head_analytic"]),
        "mass_initial": float(budget["mass_initial"]),
        "mass_inflow": float(budget["mass_inflow"]),
        "wave_amplitude": float(WAVE_AMPLITUDE),
        "final_time": float(FINAL_TIME),
        "source": "tensordg.cases.shock_entropy closed forms (Hugoniot + mass budget)",
    }


def _rel_percent(got, ref) -> float | None:
    if got is None or ref is None or ref == 0.0:
        return None
    return 100.0 * abs(float(got) - float(ref)) / abs(float(ref))


# ---------------------------------------------------------------- decisions
def cross_validate(rawdir: Path | str = DEFAULT_RAW,
                   tolerance_percent: float = GATE_PERCENT,
                   strict_tolerance_percent: float = STRICT_PERCENT,
                   anchor_tolerance_percent: float = ANCHOR_TOL_PERCENT) -> dict:
    """Run D1-D4 and return the whole audit (no side effects, deterministic)."""
    ladder = load_ladder(rawdir)
    ref = certified_reference(rawdir, tolerance_percent)
    cert = ref["certification"]
    oracle = oracle_values()

    # ---- D1: reference usable within the gate-validity rule
    band = cert.get("band_percent")
    band_share = None if band is None else band / tolerance_percent
    d1 = {
        "verdict": cert.get("verdict"),
        "classification": cert.get("classification"),
        "observed_order": cert.get("observed_order_used"),
        "band_percent": band,
        "band_share_of_gate": band_share,
        "u_share_max": U_SHARE_MAX_DECLARED,
        "reference_ncells": cert.get("reference_ncells"),
        "ladder": cert.get("ladder"),
        "differences_percent": [d["rel_l1_percent"] for d in cert.get("differences", [])],
    }
    d1["pass"] = bool(cert.get("verdict") == CERTIFIED
                      and band_share is not None and band_share <= U_SHARE_MAX_DECLARED)

    # ---- D3: certified reference vs closed-form oracle
    anchor_rows = []
    if ref["rho"] is not None:
        metrics = measure_metrics(ref["x"], ref["rho"])
        for spec in ORACLE_METRICS:
            name = spec["name"]
            got, want = metrics[name], oracle[name]
            if spec["kind"] == "relative":
                err = _rel_percent(got, want)
                ok = err is not None and err < anchor_tolerance_percent
            else:  # bounded: an absolute percentage must stay under the bound
                err = float(got)
                ok = err < anchor_tolerance_percent
            anchor_rows.append({
                "name": name, "kind": spec["kind"], "description": spec["description"],
                "unit": spec["unit"], "got": got, "ref": want,
                "rel_error_percent": err, "tolerance_percent": anchor_tolerance_percent,
                "pass": bool(ok),
            })
    d3 = {"tolerance_percent": anchor_tolerance_percent,
          "rows": anchor_rows,
          "worst_rel_error_percent": max((r["rel_error_percent"] or 0.0) for r in anchor_rows)
          if anchor_rows else None,
          "pass": bool(anchor_rows) and all(r["pass"] for r in anchor_rows)}

    # ---- D2: every DG leg under the strict gate; keep 5 % decision as well
    legs = []
    for leg in ladder["dg"]:
        res5 = judge(leg["field"], cert, tolerance_percent=tolerance_percent)
        res3 = judge(leg["field"], cert, tolerance_percent=strict_tolerance_percent)
        leg_metrics = measure_metrics(leg["x"], leg["field"])
        legs.append({
            "configuration": leg["configuration"],
            "degree": leg["degree"], "ncells": leg["ncells"], "cfl": leg["cfl"],
            "file": leg["file"],
            "rel_l1_percent": res5.get("rel_l1_percent"),
            "rel_l2_percent": res5.get("rel_l2_percent"),
            "band_percent": res5.get("band_percent"),
            "decision_5pct": res5["decision"],
            "decision_3pct": res3["decision"],
            "rel_l1_percent_3pct_gate": res3.get("rel_l1_percent"),
            "plateau_rho": leg_metrics["plateau_rho"],
            "plateau_dev_percent": _rel_percent(leg_metrics["plateau_rho"],
                                                oracle["plateau_rho"]),
            "shock_position": leg_metrics["shock_position"],
            "shock_position_dev_percent": _rel_percent(leg_metrics["shock_position"],
                                                       oracle["shock_position"]),
            "mass_dev_percent_of_initial": 100.0 * abs(
                leg_metrics["mass_integral"] - oracle["mass_integral"]) / oracle["mass_initial"],
            "rho_positive": leg_metrics["rho_positive"],
        })
    worst = max(legs, key=lambda lg: lg["rel_l1_percent"])
    counts5, counts3 = {}, {}
    for lg in legs:
        counts5[lg["decision_5pct"]] = counts5.get(lg["decision_5pct"], 0) + 1
        counts3[lg["decision_3pct"]] = counts3.get(lg["decision_3pct"], 0) + 1
    d2 = {
        "gate_percent": tolerance_percent,
        "strict_percent": strict_tolerance_percent,
        "n_legs": len(legs),
        "worst_leg": worst["configuration"],
        "worst_rel_l1_percent": worst["rel_l1_percent"],
        "decision_counts_5pct": counts5,
        "decision_counts_3pct": counts3,
        "pass": bool(legs) and all((lg["rel_l1_percent"] or 0.0) < strict_tolerance_percent
                                   and lg["decision_5pct"] not in ("FAIL", "WITHHELD")
                                   for lg in legs),
    }

    # ---- D4: no DG family may certify itself
    self_cert = {"ladders": {}, "pass": True}
    for degree in (1, 2, 3):
        lv = [{"ncells": leg["ncells"], "field": leg["field"]}
              for leg in ladder["dg"] if leg["degree"] == degree and leg["cfl"] == 0.4]
        verdict = certify(lv, tolerance_percent=tolerance_percent)["verdict"]
        self_cert["ladders"][f"dg_p{degree}"] = verdict
        if verdict != WITHHELD_ORDER_TOO_LOW:
            self_cert["pass"] = False
    self_cert["required_verdict"] = WITHHELD_ORDER_TOO_LOW

    accepted = bool(d1["pass"] and d2["pass"] and d3["pass"] and self_cert["pass"])
    return {
        "case": CASE_KEY,
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "workload": {"workload_id": WORKLOAD_ID, "version": WORKLOAD_VER},
        "rawdir": str(Path(rawdir)),
        "thresholds": {
            "gate_percent": tolerance_percent,
            "strict_percent": strict_tolerance_percent,
            "anchor_tolerance_percent": anchor_tolerance_percent,
            "u_share_max": U_SHARE_MAX_DECLARED,
        },
        "oracle": oracle,
        "D1_reference_usable": d1,
        "D2_legs_under_strict_gate": d2,
        "D3_reference_vs_oracle": d3,
        "D4_no_self_certification": self_cert,
        "legs": legs,
        "accepted": accepted,
    }


# ---------------------------------------------------------------- platform block
def platform_block(audit: dict) -> dict:
    """Wrap the oracle rows in the TensorBench spec/result shape (no fabrication).

    Only quantities with an *independent* ``ref`` become metrics.  The per-leg
    relative L1 error is an inequality gate, not ``|got-ref|/|ref|``, so it is
    reported under ``leg_gates`` and kept out of ``metrics``.
    """
    stdout_doc = {
        "reference": {r["name"]: r["got"] for r in audit["D3_reference_vs_oracle"]["rows"]},
        "oracle": {r["name"]: r["ref"] for r in audit["D3_reference_vs_oracle"]["rows"]},
        "metadata": {
            "case": audit["case"],
            "reference_ncells": audit["D1_reference_usable"]["reference_ncells"],
            "band_percent": audit["D1_reference_usable"]["band_percent"],
        },
    }
    metrics, bounds = [], []
    for row in audit["D3_reference_vs_oracle"]["rows"]:
        if row["kind"] == "relative":
            metrics.append({
                "name": row["name"],
                "description": row["description"],
                "got_pointer": f"reference.{row['name']}",
                "ref_pointer": f"oracle.{row['name']}",
                "tolerance_rel": row["tolerance_percent"] / 100.0,
                "unit": row["unit"],
            })
        else:
            bounds.append({
                "name": row["name"],
                "description": row["description"],
                "got_pointer": f"reference.{row['name']}",
                "bound_percent": row["tolerance_percent"],
                "unit": row["unit"],
                "reason": "reference value is 0 by construction; an inequality bound, "
                          "not a |got-ref|/|ref| metric",
            })
    return {
        "benchmark": {"benchmark_id": audit["benchmark"]["benchmark_id"],
                      "version": audit["benchmark"]["version"]},
        "workload": {"workload_id": audit["workload"]["workload_id"],
                     "version": audit["workload"]["version"],
                     "command": ["python", "-m", "benchmarks.xval_dg_round2_certified",
                                 "--json-stdout"],
                     "configuration": {
                         "case": audit["case"],
                         "reference_ladder": list(FV_LEVELS),
                         "reference_scheme": "FV-WENO5-JS/HLLC/SSP-RK3",
                         "reference_ncells": audit["D1_reference_usable"]["reference_ncells"],
                         "gate_percent": audit["thresholds"]["gate_percent"],
                     }},
        "metrics": metrics,
        "bounds": bounds,
        "stdout_document": stdout_doc,
        "leg_gates": {
            "gate_percent": audit["thresholds"]["gate_percent"],
            "strict_percent": audit["thresholds"]["strict_percent"],
            "worst_leg": audit["D2_legs_under_strict_gate"]["worst_leg"],
            "worst_rel_l1_percent": audit["D2_legs_under_strict_gate"]["worst_rel_l1_percent"],
            "pass": audit["D2_legs_under_strict_gate"]["pass"],
        },
    }


# ---------------------------------------------------------------- artifacts
def _markdown(audit: dict) -> str:
    th = audit["thresholds"]
    d1, d2, d3 = audit["D1_reference_usable"], audit["D2_legs_under_strict_gate"], audit["D3_reference_vs_oracle"]
    lines = [
        "# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — cross-validation audit",
        "",
        "Generated by `python -m benchmarks.xval_dg_round2_certified` from the committed raw",
        f"profiles (`{audit['rawdir']}`); no number in this file is hand-typed.",
        "",
        f"Predeclared thresholds: reference gate **{th['gate_percent']:g} %**, strict leg gate "
        f"**{th['strict_percent']:g} %**, oracle tolerance **{th['anchor_tolerance_percent']:g} %**, "
        f"band share max **{th['u_share_max']:.4f}**.",
        "",
        "## D1 — is the reference usable?",
        "",
        f"verdict **{d1['verdict']}** (classification `{d1['classification']}`), "
        f"observed order {d1['observed_order']:.6f}, "
        f"band U = {d1['band_percent']:.6f} %, U / gate = {d1['band_share_of_gate']:.4f} "
        f"<= {th['u_share_max']:.4f} → **{'PASS' if d1['pass'] else 'FAIL'}**.",
        "",
        "## D3 — certified reference vs closed-form oracle",
        "",
        "| quantity | kind | got (certified, N=%s) | ref (closed form) | err %% | limit %g %% "
        "|" % (d1["reference_ncells"], th["anchor_tolerance_percent"]),
        "|---|---|---|---|---|---|",
    ]
    for row in d3["rows"]:
        got = "n/a" if row["got"] is None else f"{row['got']:.10g}"
        ref = "bound" if row["ref"] is None else f"{row['ref']:.10g}"
        err = "n/a" if row["rel_error_percent"] is None else f"{row['rel_error_percent']:.3e}"
        lines.append(f"| `{row['name']}` | {row['kind']} | {got} | {ref} | {err} | "
                     f"{'yes' if row['pass'] else 'NO'} |")
    lines += ["", "## D2 — DG legs against the certified reference", "",
              f"{d2['n_legs']} legs; worst `{d2['worst_leg']}` = "
              f"**{d2['worst_rel_l1_percent']:.6f} %** (< {th['strict_percent']:g} % → "
              f"**{'PASS' if d2['pass'] else 'FAIL'}**).", "",
              "| leg | rel L1 % | +U % | 5 % gate | 3 % gate | plateau dev % | shock pos dev % |",
              "|---|---|---|---|---|---|---|"]
    for lg in audit["legs"]:
        lines.append(
            f"| {lg['configuration']} | {lg['rel_l1_percent']:.6f} | "
            f"{lg['rel_l1_percent'] + lg['band_percent']:.6f} | {lg['decision_5pct']} | "
            f"{lg['decision_3pct']} | {lg['plateau_dev_percent']:.3e} | "
            f"{lg['shock_position_dev_percent']:.4f} |")
    sc = audit["D4_no_self_certification"]
    lines += ["", "## D4 — no family may certify itself", "",
              "| ladder | verdict | required |", "|---|---|---|"]
    for name, verdict in sc["ladders"].items():
        lines.append(f"| {name} | {verdict} | {sc['required_verdict']} |")
    lines += ["", f"**accepted = {audit['accepted']}** "
                  f"(D1 {d1['pass']}, D2 {d2['pass']}, D3 {d3['pass']}, D4 {sc['pass']})", ""]
    return "\n".join(lines)


def write_artifacts(audit: dict, outdir: Path | str = DEFAULT_OUT) -> dict:
    """Write ``xval_round2_audit.json`` and ``.md``; return their paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / "xval_round2_audit.json"
    md = out / "xval_round2_audit.md"
    js.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    md.write_text(_markdown(audit))
    return {"json": str(js), "markdown": str(md)}


def figures(audit: dict, rawdir: Path | str = DEFAULT_RAW,
            outdir: Path | str = DEFAULT_OUT) -> dict:
    """Two PNGs: profile cross-validation and per-leg gate margins."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ladder = load_ladder(rawdir)
    ref = certified_reference(rawdir, audit["thresholds"]["gate_percent"])
    oracle = audit["oracle"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(ref["x"], ref["rho"], "k-", lw=1.6,
               label=f"certified reference (N={ref['rho'].size})")
    for leg in ladder["dg"]:
        if leg["degree"] == 2 and leg["cfl"] == 0.4:
            ax[0].plot(leg["x"], leg["field"], lw=0.9, alpha=0.8,
                       label=f"DG P2 N={leg['ncells']}")
    ax[0].set_xlabel("x"), ax[0].set_ylabel(r"$\rho$")
    ax[0].set_title("Shu-Osher t=1.8: certified reference vs DG legs")
    ax[0].legend(fontsize=8), ax[0].grid(alpha=0.3)

    labels, errs, dev = [], [], []
    for lg in audit["legs"]:
        labels.append(f"P{lg['degree']} N{lg['ncells']} c{lg['cfl']:g}")
        errs.append(lg["rel_l1_percent"])
    xpos = np.arange(len(labels))
    ax[1].bar(xpos, errs, color="#3b6ea5", label="rel L1 vs certified ref")
    ax[1].axhline(audit["thresholds"]["strict_percent"], color="crimson", ls="--",
                  label=f"strict gate {audit['thresholds']['strict_percent']:g} %")
    ax[1].axhline(audit["thresholds"]["gate_percent"], color="darkorange", ls=":",
                  label=f"case gate {audit['thresholds']['gate_percent']:g} %")
    ax[1].set_xticks(xpos), ax[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("relative L1 error [%]")
    ax[1].set_title("Per-leg error and gate margins")
    ax[1].legend(fontsize=8), ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = out / "fig1_xval_profiles.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    devs = [lg["plateau_dev_percent"] for lg in audit["legs"]]
    ax.semilogy(xpos, np.maximum(devs, 1e-14), "o-", label="plateau $\\rho_L$")
    ax.semilogy(xpos, np.maximum([lg["shock_position_dev_percent"] for lg in audit["legs"]],
                                 1e-14), "s--", label="shock position")
    ax.axhline(audit["thresholds"]["anchor_tolerance_percent"], color="crimson", ls="--",
               label=f"oracle tol {audit['thresholds']['anchor_tolerance_percent']:g} %")
    ax.set_xticks(xpos), ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("deviation from closed form [%]")
    ax.set_title("DG legs vs closed-form anchors")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = out / "fig2_leg_oracle_deviation.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    return {"fig1": str(p1), "fig2": str(p2)}


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rawdir", default=DEFAULT_RAW)
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    ap.add_argument("--gate", type=float, default=GATE_PERCENT)
    ap.add_argument("--strict", type=float, default=STRICT_PERCENT)
    ap.add_argument("--anchor-tolerance", type=float, default=ANCHOR_TOL_PERCENT)
    ap.add_argument("--figures", action="store_true", help="also write the two PNGs")
    ap.add_argument("--json-stdout", action="store_true",
                    help="print the platform block (stdout document) instead of the tables")
    ap.add_argument("--no-write", action="store_true", help="do not write artifacts")
    ap.add_argument("--require-accepted", action="store_true",
                    help="exit 1 unless all four decisions pass")
    args = ap.parse_args(argv)

    audit = cross_validate(args.rawdir, args.gate, args.strict, args.anchor_tolerance)
    if args.json_stdout:
        print(json.dumps(platform_block(audit)["stdout_document"], indent=2, sort_keys=True))
    else:
        print(_markdown(audit))
    if not args.no_write:
        paths = write_artifacts(audit, args.outdir)
        print(f"wrote {paths['json']} and {paths['markdown']}", file=sys.stderr)
        if args.figures:
            figs = figures(audit, args.rawdir, args.outdir)
            print(f"wrote {figs['fig1']} and {figs['fig2']}", file=sys.stderr)
    if args.require_accepted and not audit["accepted"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())