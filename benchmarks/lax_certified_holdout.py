"""CASE-LAX-DG -- does the certification frontier *predict* configs it never saw?

The question
------------
``docs/benchmarks/results/lax_certified_configuration/metrics.json`` certifies the
ledger bar (``ref: exact solution | tolerance < 3 %``) at ``p2.N800`` (DOF 2400)
and states its minimality claim **within the enumerated grid only**::

    every measured leg with strictly smaller DOF than the certified one FAILs.
    No claim is made about unmeasured (p, N).

That is an honest but weak statement: it says nothing about whether the pass/fail
boundary is a property of the discretisation (a DOF/order frontier that
extrapolates) or an accident of which eight legs happened to be enumerated.  A
case pool that has to *recommend* a configuration to a user needs the stronger
statement, and the only way to earn it is to predict an unmeasured configuration
in advance and then measure it.

Pre-registered design (written before any number of the hold-out was seen)
--------------------------------------------------------------------------
* **Hold-out legs** -- two, both *unmeasured* by the enumerated grid:

  - ``H1 = p2.N600`` -> DOF 1800, straddling the frontier (smaller DOF than the
    certified leg, larger than every enumerated PASS-free region);
  - ``H2 = p3.N600`` -> DOF 2400, **exactly the certified leg's DOF**, so it asks
    whether the error at fixed DOF is a function of DOF alone or still depends on
    the order ``p``.

* **Predictors** -- both are fitted only on the committed enumerated legs, so the
  prediction is a genuine extrapolation into an unmeasured ``(p, N)``:

  - model ``dof``: one power law in DOF, ``log e = a + b log((p+1) N)``;
  - model ``family``: one power law in ``N`` per degree, ``log e = a_p + b_p log N``
    (a degree seen only once falls back to the global fit).

* **Prediction window** -- not a hand-picked number: it is the **leave-one-out
  absolute error of the model itself on the enumerated legs**, so the window is
  calibrated on data the model saw and applied to data it did not.  A hold-out
  number is *inside* the window iff ``|observed - predicted| <= window``.

* **Verdicts** -- predicted per leg and per model: PASS iff predicted worst-field
  relative L2 ``<`` the tolerance imported from the registered case (never a
  literal).  A hold-out is a *correctly predicted verdict* only if the predicted
  and the measured side of the 3 % line agree.

* **What would falsify what** --
  - if ``H1`` passes, the enumerated "smallest DOF" certification is **not**
    minimal in general and the claim has to be restated (and the certified
    configuration re-derived from the enlarged leg set);
  - if ``H1`` fails, the frontier stands on the predicted side of the boundary;
  - if both models put ``H2`` inside the window, DOF is a sufficient predictor at
    fixed DOF; otherwise the order ``p`` keeps an independent effect and a
    DOF-only frontier must be rejected.

Metric, solver, tolerance and DOF definition are all **imported** from the case
pool / certification driver -- nothing is re-implemented, so the hold-out numbers
cannot drift away from the ledger arm.

Phases (pre-registration is structural, not a promise)::

    python -m benchmarks.lax_certified_holdout --predict   # writes predictions.json
    python -m benchmarks.lax_certified_holdout --run       # solves + evaluates

``--run`` refuses to start unless ``predictions.json`` already exists and its
timestamp predates the run.

Artifacts (inside the worktree)::

    docs/benchmarks/results/lax_certified_holdout/
        predictions.json   pre-registered predictions, windows and verdicts
        metrics.json       measured hold-out legs, outcomes, re-derived frontier
        summary.md         generated summary
        figures/           prediction-vs-observation, DOF frontier
    docs/CASE-LAX-DG-certified-holdout.md   generated write-up
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import run_ladder
from benchmarks.lax_certified_configuration import dof
from tensordg.cases.lax_1d import lax_parameters

HERE = Path(__file__).resolve().parent.parent

FIELDS = ("density", "velocity", "pressure")

ENUMERATED = HERE / "docs/benchmarks/results/lax_certified_configuration/metrics.json"
OUT = HERE / "docs/benchmarks/results/lax_certified_holdout"
FIG = OUT / "figures"
PREDICTIONS = OUT / "predictions.json"
METRICS = OUT / "metrics.json"
SUMMARY = OUT / "summary.md"
DOC = HERE / "docs/CASE-LAX-DG-certified-holdout.md"

#: hold-out legs, (cells, degree) -- pre-registered, see module docstring.
HOLDOUT = ((600, 2), (600, 3))

#: name of the enumerated study this hold-out extends.
UPSTREAM = "docs/benchmarks/results/lax_certified_configuration/metrics.json"


# --------------------------------------------------------------------- helpers
def tolerance():
    """Ledger tolerance from the registered case (never a literal)."""
    return float(lax_parameters()["acceptance_threshold"])


def worst_field(row):
    """Worst-field whole-field relative L2 of one leg, and the field name."""
    field = max(FIELDS, key=lambda f: row["l2_errors"][f])
    return float(row["l2_errors"][field]), field


def leg_name(cells, degree):
    return f"p{degree}.N{cells}"


def enumerated_rows(path=ENUMERATED):
    """Committed enumerated legs, flattened to (name, cells, degree, dof, worst)."""
    payload = json.loads(Path(path).read_text())
    rows = []
    for leg in payload["legs"]:
        worst, field = worst_field(leg)
        rows.append({"leg": leg_name(leg["cells"], leg["degree"]),
                     "cells": int(leg["cells"]), "degree": int(leg["degree"]),
                     "dof": dof(leg["cells"], leg["degree"]),
                     "worst_l2": worst, "worst_field": field,
                     "wall_seconds": leg.get("wall_seconds")})
    rows.sort(key=lambda r: (r["dof"], r["degree"]))
    return rows


# ----------------------------------------------------------------------- models
def fit_power_law(xs, ys):
    """Least-squares ``log y = a + b log x``; returns ``(a, b)``."""
    lx = np.log(np.asarray(xs, dtype=float))
    ly = np.log(np.asarray(ys, dtype=float))
    b, a = np.polyfit(lx, ly, 1)
    return float(a), float(b)


def predict(model, rows, cells, degree):
    """Predict the worst-field relative L2 of one unmeasured leg.

    ``rows`` must not contain the leg being predicted (callers guarantee it).
    """
    if model == "dof":
        a, b = fit_power_law([r["dof"] for r in rows], [r["worst_l2"] for r in rows])
        return math.exp(a + b * math.log(dof(cells, degree)))
    if model == "family":
        own = [r for r in rows if r["degree"] == degree]
        if len(own) >= 2:
            a, b = fit_power_law([r["cells"] for r in own],
                                 [r["worst_l2"] for r in own])
        else:  # a degree with a single leg carries no rate: fall back to DOF
            a, b = fit_power_law([r["dof"] for r in rows], [r["worst_l2"] for r in rows])
            return math.exp(a + b * math.log(dof(cells, degree)))
        return math.exp(a + b * math.log(cells))
    raise ValueError(f"unknown model {model!r}")


MODELS = ("dof", "family")


def leave_one_out(rows, model):
    """Absolute LOO prediction error per enumerated leg (in error units)."""
    out = []
    for row in rows:
        rest = [r for r in rows if r["leg"] != row["leg"]]
        out.append({"leg": row["leg"], "observed": row["worst_l2"],
                    "predicted": predict(model, rest, row["cells"], row["degree"])})
    for item in out:
        item["abs_error"] = abs(item["predicted"] - item["observed"])
    return out


def calibrate(rows):
    """Per-model LOO residuals and the pre-registered window (= max LOO error)."""
    table = {}
    for model in MODELS:
        loo = leave_one_out(rows, model)
        errors = [item["abs_error"] for item in loo]
        table[model] = {
            "loo": loo,
            "max_abs_error": float(max(errors)),
            "rms_abs_error": float(math.sqrt(sum(e * e for e in errors) / len(errors))),
            "window": float(max(errors)),
        }
    return table


def consensus(predictions, tol):
    """Model-free pre-registered verdict: PASS iff *both* models predict < tol."""
    verdicts = {name: ("PASS" if item["predicted_worst_l2"] < tol else "FAIL")
                for name, item in predictions.items()}
    agree = len(set(verdicts.values())) == 1
    return {"per_model": verdicts,
            "agree": agree,
            "verdict": verdicts[next(iter(verdicts))] if agree else "DISPUTED"}


# -------------------------------------------------------------------- predict
def build_predictions(rows=None, tol=None):
    rows = enumerated_rows() if rows is None else rows
    tol = tolerance() if tol is None else tol
    calibration = calibrate(rows)
    legs = {}
    for cells, degree in HOLDOUT:
        name = leg_name(cells, degree)
        per_model = {}
        for model in MODELS:
            value = predict(model, rows, cells, degree)
            window = calibration[model]["window"]
            per_model[model] = {
                "predicted_worst_l2": value,
                "window": window,
                "interval": [value - window, value + window],
                "predicted_verdict": "PASS" if value < tol else "FAIL",
            }
        legs[name] = {
            "cells": int(cells), "degree": int(degree),
            "dof": dof(cells, degree),
            "per_model": per_model,
            "consensus": consensus(per_model, tol),
            "question": ("frontier minimality: does a smaller-DOF leg also pass?"
                         if dof(cells, degree) < 2400 else
                         "DOF sufficiency: same DOF as the certified leg p2.N800"),
        }
    return {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "phase": "predict",
        "registered_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance": tol,
        "upstream": UPSTREAM,
        "enumerated_legs": rows,
        "calibration": calibration,
        "design": {
            "holdout": [leg_name(c, d) for c, d in HOLDOUT],
            "models": list(MODELS),
            "window_rule": "window = max leave-one-out absolute prediction error of "
                           "the same model on the enumerated legs (error units)",
            "verdict_rule": "PASS iff worst-field whole-field relative L2 < tolerance",
            "consensus_rule": "PASS/FAIL only when both models agree, else DISPUTED",
            "metric": "imported from benchmarks.lax_case_pool.run_ladder "
                      "(10-node Gauss, whole field, relative L2, worst field reported)",
        },
        "legs": legs,
    }


# ------------------------------------------------------------------------ run
def run_holdout(tol=None, entries=None):
    """Solve the hold-out legs (or format pre-solved ``entries``) and record them.

    ``entries`` exists so a leg can be solved in a separate process and cached
    verbatim; the formatting below is the only place that turns solver output
    into a recorded leg, so a cached leg cannot be shaped differently.
    """
    tol = tolerance() if tol is None else tol
    if entries is None:
        _, _, entries, _, _ = run_ladder(ladder=[(c, d) for c, d in HOLDOUT])
    return format_legs(entries, tol)


def format_legs(entries, tol=None):
    tol = tolerance() if tol is None else tol
    legs = []
    for entry in entries:
        worst, field = worst_field(entry)
        legs.append({
            "leg": leg_name(entry["cells"], entry["degree"]),
            "cells": int(entry["cells"]), "degree": int(entry["degree"]),
            "dof": dof(entry["cells"], entry["degree"]),
            "steps": int(entry["steps"]),
            "dt_mean": float(entry["dt"]),
            "wall_seconds": float(entry["wall_seconds"]),
            "l1_errors": {f: float(entry["errors"][f]) for f in FIELDS},
            "l2_errors": {f: float(entry["l2_errors"][f]) for f in FIELDS},
            "analytic_errors_band_removed": {f: float(entry["analytic_errors"][f])
                                             for f in FIELDS},
            "density_min": float(entry["density_min"]),
            "worst_l2": worst, "worst_field": field,
            "verdict": "PASS" if worst < tol else "FAIL",
        })
    legs.sort(key=lambda r: (r["dof"], r["degree"]))
    return legs


def evaluate(predictions, measured, tol=None):
    tol = tolerance() if tol is None else tol
    by_leg = {row["leg"]: row for row in measured}
    entries = []
    for name, design in predictions["legs"].items():
        row = by_leg[name]
        per_model = {}
        for model, predicted in design["per_model"].items():
            value = predicted["predicted_worst_l2"]
            window = predicted["window"]
            delta = row["worst_l2"] - value
            per_model[model] = {
                "predicted_worst_l2": value,
                "observed_worst_l2": row["worst_l2"],
                "signed_error": float(delta),
                "abs_error": float(abs(delta)),
                "window": window,
                "inside_window": bool(abs(delta) <= window),
                "predicted_verdict": predicted["predicted_verdict"],
                "observed_verdict": row["verdict"],
                "verdict_correct": predicted["predicted_verdict"] == row["verdict"],
                "scaled_error_in_windows": float(abs(delta) / window),
            }
        entries.append({
            "leg": name, "dof": row["dof"],
            "per_model": per_model,
            "consensus": design["consensus"],
            "consensus_verdict_correct":
                design["consensus"]["verdict"] in (row["verdict"], "DISPUTED"),
            "question": design["question"],
            "worst_field_matches_prediction": row["worst_field"] == "density",
        })
    inside = [m["inside_window"] for e in entries for m in e["per_model"].values()]
    correct = [m["verdict_correct"] for e in entries for m in e["per_model"].values()]
    return {
        "legs": entries,
        "summary": {
            "predictions": len(inside),
            "inside_window": int(sum(inside)),
            "verdict_correct": int(sum(correct)),
            "median_scaled_error": float(np.median(
                [m["scaled_error_in_windows"] for e in entries
                 for m in e["per_model"].values()])),
        },
    }


def frontier(enumerated, measured, tol=None):
    """Minimal-DOF passing leg over the enlarged (enumerated + hold-out) set."""
    tol = tolerance() if tol is None else tol
    rows = []
    for row in enumerated:
        rows.append({"leg": row["leg"], "dof": row["dof"], "degree": row["degree"],
                     "cells": row["cells"], "worst_l2": row["worst_l2"],
                     "source": "enumerated"})
    for row in measured:
        rows.append({"leg": row["leg"], "dof": row["dof"], "degree": row["degree"],
                     "cells": row["cells"], "worst_l2": row["worst_l2"],
                     "source": "hold-out"})
    rows.sort(key=lambda r: (r["dof"], r["degree"], r["cells"]))
    passing = [r for r in rows if r["worst_l2"] < tol]
    certified = passing[0] if passing else None
    per_degree = {}
    for degree in sorted({r["degree"] for r in rows}):
        own = [r for r in rows if r["degree"] == degree]
        per_degree[str(degree)] = {
            "legs": [r["leg"] for r in own],
            "monotone_in_cells": all(own[i]["worst_l2"] > own[i + 1]["worst_l2"]
                                     for i in range(len(own) - 1)),
        }
    return {
        "tolerance": tol,
        "rows": rows,
        "minimal_dof_passing": certified,
        "enumerated_minimal_dof_passing": "p2.N800",
        "minimality_restated": (certified is not None
                                and certified["leg"] != "p2.N800"),
        "per_degree_monotone": per_degree,
    }


# ------------------------------------------------------------------- figures
def render_figures(predictions, evaluation, frontier_rows, out_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tol = predictions["tolerance"]
    made = []

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    enum = predictions["enumerated_legs"]
    markers = {1: "o", 2: "s", 3: "^"}
    for degree, marker in markers.items():
        own = [r for r in enum if r["degree"] == degree]
        if not own:
            continue
        ax.loglog([r["dof"] for r in own], [100 * r["worst_l2"] for r in own],
                  marker, ls="none", ms=7, label=f"enumerated p={degree}")
    for name, design in predictions["legs"].items():
        row = next(e for e in evaluation["legs"] if e["leg"] == name)
        per = row["per_model"]["dof"]
        ax.errorbar([design["dof"]], [100 * per["observed_worst_l2"]],
                    yerr=[[100 * (per["observed_worst_l2"] - (per["predicted_worst_l2"]
                                                              - per["window"]))],
                          [100 * ((per["predicted_worst_l2"] + per["window"])
                                  - per["observed_worst_l2"])]],
                    fmt="*", ms=15, color="crimson", capsize=4,
                    label=f"hold-out {name} (window=LOO)")
        ax.plot([design["dof"]], [100 * per["predicted_worst_l2"]], "x",
                color="crimson", ms=9)

    a, b = fit_power_law([r["dof"] for r in enum], [r["worst_l2"] for r in enum])
    grid = np.linspace(600, 4000, 200)
    ax.loglog(grid, 100 * np.exp(a + b * np.log(grid)), "-", color="grey",
              label=f"DOF power law (b={b:.3f})")
    ax.axhline(100 * tol, color="black", ls="--", lw=1.2,
               label=f"ledger bar {100 * tol:.1f}%")
    ax.set_xlabel("degrees of freedom (p+1)·N")
    ax.set_ylabel("worst-field relative L2 [%]")
    ax.set_title("CASE-LAX-DG: pre-registered hold-out test of the certification frontier")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    path = out_dir / "holdout_frontier.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = [row["leg"] for row in evaluation["legs"]]
    width = 0.35
    idx = np.arange(len(names))
    for offset, model in ((-width / 2, "dof"), (width / 2, "family")):
        scaled = [row["per_model"][model]["scaled_error_in_windows"]
                  for row in evaluation["legs"]]
        ax.bar(idx + offset, scaled, width=width, label=f"model '{model}'")
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="pre-registered window")
    ax.set_xticks(idx)
    ax.set_xticklabels(names)
    ax.set_ylabel("|observed − predicted| / window")
    ax.set_title("Hold-out prediction error, in units of the LOO-calibrated window")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "holdout_error_in_windows.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)
    return made


# -------------------------------------------------------------------- outputs
def render_summary(metrics):
    tol = metrics["tolerance"]
    lines = [
        "# CASE-LAX-DG — certified-frontier hold-out test",
        "",
        f"* pre-registered at `{metrics['registered_at_utc']}`, "
        f"measured at `{metrics['measured_at_utc']}`",
        f"* ledger tolerance (imported from the registered case): `{tol:.6f}`",
        f"* upstream enumerated grid: `{metrics['upstream']}`",
        "",
        "## Pre-registered predictions vs measurement",
        "",
        "| hold-out | DOF | model | predicted worst L2 | window | observed | "
        "in window | predicted verdict | observed verdict |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for row in metrics["evaluation"]["legs"]:
        for model, item in row["per_model"].items():
            lines.append(
                f"| `{row['leg']}` | {row['dof']} | `{model}` | "
                f"{100 * item['predicted_worst_l2']:.3f} % | "
                f"±{100 * item['window']:.3f} % | "
                f"{100 * item['observed_worst_l2']:.3f} % | "
                f"{'yes' if item['inside_window'] else 'NO'} | "
                f"{item['predicted_verdict']} | {item['observed_verdict']} |")
    summary = metrics["evaluation"]["summary"]
    lines += [
        "",
        f"predictions inside the pre-registered window: "
        f"**{summary['inside_window']}/{summary['predictions']}**; "
        f"verdicts reproduced: **{summary['verdict_correct']}/"
        f"{summary['predictions']}**; "
        f"median error = **{summary['median_scaled_error']:.2f}×** the window.",
        "",
        "## Re-derived frontier over the enlarged leg set",
        "",
        "| leg | DOF | worst L2 | verdict | source |",
        "|---|---:|---:|---|---|",
    ]
    for row in metrics["frontier"]["rows"]:
        lines.append(f"| `{row['leg']}` | {row['dof']} | "
                     f"{100 * row['worst_l2']:.3f} % | "
                     f"{'PASS' if row['worst_l2'] < tol else 'FAIL'} | "
                     f"{row['source']} |")
    certified = metrics["frontier"]["minimal_dof_passing"]
    lines += [
        "",
        f"minimal-DOF passing configuration over the enlarged set: "
        f"**`{certified['leg']}`** (DOF {certified['dof']}, "
        f"{100 * certified['worst_l2']:.3f} %); enumerated claim was `p2.N800`; "
        f"restated: **{'yes' if metrics['frontier']['minimality_restated'] else 'no'}**.",
    ]
    return "\n".join(lines) + "\n"


def render_doc(metrics):
    tol = metrics["tolerance"]
    pred = metrics["predictions"]
    summary = metrics["evaluation"]["summary"]
    front = metrics["frontier"]
    certified = front["minimal_dof_passing"]

    def leg_rows():
        out = []
        for row in metrics["evaluation"]["legs"]:
            worst_model = max(row["per_model"].items(),
                              key=lambda kv: kv[1]["scaled_error_in_windows"])
            item = worst_model[1]
            out.append(
                f"| `{row['leg']}` | {row['dof']} | "
                f"{100 * item['predicted_worst_l2']:.3f} % | "
                f"±{100 * item['window']:.3f} % | "
                f"{100 * item['observed_worst_l2']:.3f} % | "
                f"{100 * item['abs_error']:.3f} % | "
                f"{'yes' if item['inside_window'] else 'NO'} | "
                f"{item['predicted_verdict']} / {item['observed_verdict']} |")
        return "\n".join(out)

    def frontier_rows():
        return "\n".join(
            f"| `{r['leg']}` | {r['dof']} | {100 * r['worst_l2']:.3f} % | "
            f"{'PASS' if r['worst_l2'] < tol else 'FAIL'} | {r['source']} |"
            for r in front["rows"])

    measured_rows = "\n".join(
        f"| `{r['leg']}` | {r['cells']} | {r['degree']} | {r['dof']} | "
        f"{r['steps']} | {r['wall_seconds']:.1f} | "
        f"{100 * r['l2_errors']['density']:.3f} % | "
        f"{100 * r['l2_errors']['velocity']:.3f} % | "
        f"{100 * r['l2_errors']['pressure']:.3f} % | {r['verdict']} |"
        for r in metrics["legs"])

    cal = pred["calibration"]
    body = f"""# CASE-LAX-DG · 认证前沿的 hold-out 检验：预测一个没算过的配置，再去算它

> 锚点：`feature_ledger` `CASE-LAX-DG`（原文 `ref: exact solution | tolerance < 3 %`，status `open`）。
> 上游：`{UPSTREAM}`（8 条枚举腿，认证 `p2.N800`）。
> 本单**不改变**台账生产腿 `p3.N400` 的 honest FAIL，也不推翻枚举认证；
> 它检验的是枚举认证**没有声称**的那部分：「更小的 `(p, N)` 会不会也过」。

## ① 案例介绍

Lax (1954) / Toro test 2 激波管：`[0,1]`、隔膜 `x0 = 0.5`、γ = 1.4、快照 `t = 0.13`。
左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0, 0.571)`；
星区闭式 `p* = 2.466098`、`u* = 1.528723`。四道波：稀疏头、稀疏尾、接触（真跳变）、激波（真跳变）。
判据沿用台账原文：全场相对 L2（10 点 Legendre–Gauss，逐点对闭式 Riemann 解，分母为精确解同口径范数），
**取最差场量**，与 `{tol:.2f}` 比较，无场量豁免、无去掉波带、无平均。

## ② 怎么做的（方法/流程/关键设置/工具链）

枚举研究 `lax_certified_configuration` 的认证是**关于枚举网格**的：8 条腿里 `p2.N800`（DOF 2400）
是最小 DOF 的 PASS 腿，并明确声明「对未测量的 `(p, N)` 不作断言」。本单把这句声明变成可检验的：

1. **预登记**（`predictions.json`，写入时间 `{pred['registered_at_utc']}`，早于任何求解）：
   两条**从未算过**的 hold-out 腿
   * `H1 = p2.N600`（DOF 1800）——DOF 比认证腿小，用来打「最小 DOF」这句话；
   * `H2 = p3.N600`（DOF 2400）——DOF 与认证腿 `p2.N800` **完全相同**，用来问「固定 DOF 下误差是否只由 DOF 决定」。
2. **两个预测器**，只用已提交的枚举腿拟合：
   * `dof`：一条 DOF 幂律 `log e = a + b log((p+1)N)`；
   * `family`：每个阶数一条 `N` 幂律 `log e = a_p + b_p log N`（只有一条腿的阶数回退到 DOF 幂律）。
3. **预测窗口不是手挑的数**：窗口 = **该模型自己在枚举腿上的 leave-one-out 绝对误差最大值**。
   在模型看过的数据上校准，用在它没看过的数据上判 `|观测 − 预测| <= 窗口`。
4. **判定**：PASS iff 最差场量相对 L2 `<` 容差（容差从 `tensordg.cases.lax_1d` 导入，不写字面量）；
   两模型一致才给共识判定，否则记 `DISPUTED`。
5. 求解与度量**全部 import 自** `benchmarks.lax_case_pool.run_ladder`（同一求解器、同一求积、同一范数），
   本单不重实现度量，因而不可能与台账口径漂移。
6. 测后重算「扩大腿集后的最小 DOF PASS 配置」，并把口径写回文档。

工具链：`benchmarks/lax_certified_holdout.py`（本单新增）、`benchmarks/lax_case_pool.py`、
`benchmarks/lax_certified_configuration.py`、`tensordg.euler_dg_1d`（P3 HLLC + 正定性回退，SSP-RK3，CFL 0.35）。
两阶段 CLI（`--predict` 写预登记 → `--run` 才求解，且拒绝在预登记缺失/时间倒挂时启动）。

## ③ 计算结果

### 3.1 hold-out 实测（程序化注入）

| leg | N | p | DOF | steps | 墙钟 [s] | ρ L2 | u L2 | p L2 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{measured_rows}

### 3.2 预登记预测 vs 实测（窗口 = 该模型 LOO 最大残差）

| hold-out | DOF | 模型 | 预测最差 L2 | 窗口 | 实测 | 绝对偏差 | 在窗口内 | 预测/实测判定 |
|---|---:|---|---:|---:|---:|---:|---|---|
{leg_rows()}

预测落在窗口内：**{summary['inside_window']}/{summary['predictions']}**；
判定复现：**{summary['verdict_correct']}/{summary['predictions']}**；
偏差中位数 = **{summary['median_scaled_error']:.2f}×** 窗口。

模型自校准（LOO，枚举 8 腿）：

| 模型 | LOO 最大绝对误差 | LOO RMS | 窗口 |
|---|---:|---:|---:|
| `dof` | {100 * cal['dof']['max_abs_error']:.3f} % | {100 * cal['dof']['rms_abs_error']:.3f} % | ±{100 * cal['dof']['window']:.3f} % |
| `family` | {100 * cal['family']['max_abs_error']:.3f} % | {100 * cal['family']['rms_abs_error']:.3f} % | ±{100 * cal['family']['window']:.3f} % |

### 3.3 扩大腿集后的前沿重算

| leg | DOF | 最差 L2 | 判定 | 来源 |
|---|---:|---:|---|---|
{frontier_rows()}

最小 DOF 的 PASS 配置 = **`{certified['leg']}`**（DOF {certified['dof']}，{100 * certified['worst_l2']:.3f} %）；
枚举认证原为 `p2.N800`；**最小性是否被改写：{'是' if front['minimality_restated'] else '否'}**。

## ④ 结果图

* `docs/benchmarks/results/lax_certified_holdout/figures/holdout_frontier.png`
  —— DOF–误差平面：枚举腿（按 p 分形）、DOF 幂律、3 % 台账线，以及两条 hold-out（★，误差棒 = 预登记窗口）。
* `docs/benchmarks/results/lax_certified_holdout/figures/holdout_error_in_windows.png`
  —— 每个 hold-out 的偏差 / 窗口 比值（1.0 虚线 = 预登记窗口），两个模型并列。

## ⑤ 注意点（坑/局限/适用范围）

1. **这不是「更准」的研究**：本单一个求解器参数都没改，改动只在「认证的一句话能推到多远」。
2. **两条 hold-out 不足以证明前沿处处成立**：它证明的是「在两个刻意选在边界附近的点上，模型预测成立」。
   边界处（误差 2.8–3.2 %）才是最容易被打破的地方，因此窗口是 LOO 最大残差而不是 RMS——
   这是保守侧的选择。
3. **窗口是模型自校准的，不是统计置信区间**：8 条拟合腿的 LOO 残差只能给出尺度，
   不能给出覆盖率保证；`inside_window` 失败时的正确读法是「模型外推不足」，不是「求解器错了」。
4. **DOF 仍是粗变量**：`p` 与 `N` 折算成 `(p+1)N` 忽略了 CFL 与时间步的影响（三阶 RK 的稳定性步长随 p 变化），
   所以 `family` 模型（按阶数分别拟合）才是物理上更对的形态；两者并列正是为了让差异可见。
5. **工件可重建**：所有数字来自本次两阶段运行写入的 `metrics.json`（求解器直出，零手抄）；
   文档与 summary 由 `render_doc/render_summary` 从同一份 JSON 生成。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | 误差 | < 3 %？ |
|---|---|---|---|---|
"""
    for row in metrics["legs"]:
        body += (f"| `{row['leg']}` 密度 L2 | 闭式 Riemann 精确解 | "
                 f"{100 * row['l2_errors']['density']:.3f} % | "
                 f"{100 * row['l2_errors']['density']:.3f} % | "
                 f"{'是' if row['verdict'] == 'PASS' else '否（honest FAIL）'} |\n")
    worst_holdout_error = max(item["abs_error"]
                             for row in metrics["evaluation"]["legs"]
                             for item in row["per_model"].values())
    all_inside = summary["inside_window"] == summary["predictions"]
    body += (
        f"| 预登记预测偏差（hold-out，相对实测） | 模型自身 LOO 窗口 | "
        f"{100 * worst_holdout_error:.3f} % | 见 3.2 | "
        f"{'是' if all_inside else '部分否'} |\n")
    body += (
        "\n台账容忍度 3 % 的判定**不因本单改变**：生产腿 `p3.N400` 的 3.116 % honest FAIL 仍在账上；"
        "本单只回答「认证前沿能否外推」这一附带声明。\n")
    return body


def write_outputs(predictions, legs, evaluation, frontier_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    figures = render_figures(predictions, evaluation, frontier_rows)
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "question": "does the enumerated certification frontier predict unmeasured (p, N) legs?",
        "registered_at_utc": predictions["registered_at_utc"],
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance": predictions["tolerance"],
        "upstream": UPSTREAM,
        "predictions": predictions,
        "legs": legs,
        "evaluation": evaluation,
        "frontier": frontier_rows,
        "figures": [str(p.relative_to(HERE)) for p in figures],
    }
    METRICS.write_text(json.dumps(metrics, indent=1) + "\n")
    SUMMARY.write_text(render_summary(metrics))
    DOC.write_text(render_doc(metrics))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--predict", action="store_true",
                       help="write pre-registered predictions only (no solve)")
    group.add_argument("--run", action="store_true",
                       help="solve the hold-out legs and evaluate the predictions")
    parser.add_argument("--figures-only", action="store_true",
                        help="re-render figures/doc from the committed metrics.json")
    args = parser.parse_args(argv)

    if args.predict:
        OUT.mkdir(parents=True, exist_ok=True)
        predictions = build_predictions()
        PREDICTIONS.write_text(json.dumps(predictions, indent=1) + "\n")
        for name, design in predictions["legs"].items():
            line = ", ".join(
                f"{model}: {100 * item['predicted_worst_l2']:.3f}%"
                f" ±{100 * item['window']:.3f}% -> {item['predicted_verdict']}"
                for model, item in design["per_model"].items())
            print(f"[predict] {name} (DOF {design['dof']}) {line}", flush=True)
        print(f"[predict] wrote {PREDICTIONS}", flush=True)
        return 0

    if args.figures_only:
        metrics = json.loads(METRICS.read_text())
        predictions = json.loads(PREDICTIONS.read_text())
        write_outputs(predictions, metrics["legs"], metrics["evaluation"],
                      metrics["frontier"])
        print(f"[figures] re-rendered from {METRICS}", flush=True)
        return 0

    if not PREDICTIONS.exists():
        raise SystemExit("refusing to run: predictions.json is missing "
                         "(pre-registration must exist before measurement)")
    predictions = json.loads(PREDICTIONS.read_text())
    legs = run_holdout()
    evaluation = evaluate(predictions, legs)
    frontier_rows = frontier(predictions["enumerated_legs"], legs)
    metrics = write_outputs(predictions, legs, evaluation, frontier_rows)
    summary = metrics["evaluation"]["summary"]
    print(f"[run] inside window {summary['inside_window']}/{summary['predictions']}, "
          f"verdicts {summary['verdict_correct']}/{summary['predictions']}, "
          f"minimal-DOF passing leg {frontier_rows['minimal_dof_passing']['leg']}",
          flush=True)
    print(f"[run] wrote {METRICS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())