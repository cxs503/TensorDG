#!/usr/bin/env python3
"""Long-time horizon of the isentropic-vortex error-conservation claim.

Anchor: ``CASE-DG-EULER-VORTEX-ERR`` (ref: Yee et al. 2000, tolerance 6 %).

The published archive of this case stops at ``t = 20`` for one family and at
``t = 10`` for every other family, so the claim it makes -- "the error is a grid
property, not a time property" -- is a statement about *one or two box crossings*
of transport.  This tool does two things, in this order, and never the reverse:

``--predict``
    Read the archived checkpoints (``final_time <= 20`` only; the tool refuses a
    record with a later time) and freeze an out-of-sample prediction of the
    ``t = 40`` reading under four extrapolation models.  The prediction is
    written to ``predictions.json`` together with a digest of the archive it was
    derived from, so it cannot be quietly re-fitted after the data arrives.

``--analyze``
    Read the new ``t = 40`` checkpoints, score every model prediction against the
    measurement, re-fit the accumulation exponent over the extended window
    ``[1, 40]`` with the same robust estimators the C2 gate uses, measure the
    reference defect of the *folded* (``min_image``) reference at ``t = 40``
    (that is why the long-time runs use the lattice reference), and extrapolate
    the time at which the design family would cross the 6 % case tolerance.

The point of the exercise is falsification, not confirmation: the four models
disagree by a factor ~4 at ``t = 40`` (a saturating fit predicts ~2.5 %, a
sustained tail slope predicts ~4.5 %), so the measurement arbitrates between
"error conservation" and "secular accumulation" with real power.

Usage::

    PYTHONPATH=. python3 tools/vortex_longtime_horizon.py --predict
    PYTHONPATH=. python3 tools/vortex_longtime_horizon.py --analyze --figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND,
    DOMAIN,
    PERIOD,
    STRENGTH,
    TOLERANCE,
    exponent_variants,
    family_key,
)
from tensordg.euler_2d import vortex_primitive

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "case_euler_vortex_errconserve" / "results"
OUTPUT_DIR = ROOT / "case_euler_vortex_errconserve_longtime"
PREDICTIONS = OUTPUT_DIR / "predictions.json"

#: Families whose long-time behaviour decides the anchor's central sentence.
#: ``dg_d2_n32_cfl0.2`` is the C2 gate family; ``p0_n64_cfl0.4`` is the
#: first-order control (its error is *expected* to saturate, not to conserve).
TARGETS = {
    "dg_d2_n32_cfl0.2": "design family (C2 gate family, p=2, 32^2)",
    "dg_d1_n32_cfl0.2": "degree axis p=1 (32^2)",
    "dg_d3_n32_cfl0.2": "degree axis p=3 (32^2)",
    "p0_n64": "first-order cross-check control (64^2)",
}

#: Extra checkpoint that the four models are asked to predict.
PREDICT_TIME = 40.0
#: No archived record later than this may enter a prediction (leak guard).
ARCHIVE_TIME_CAP = 20.0


def _record_family(record: dict) -> str:
    """Family identity recomputed from the record's own configuration.

    The stored ``family`` field is **not** trusted: four archived records carry
    a label from before the CFL entered the family identity (``dg_d2_n32``
    instead of ``dg_d2_n32_cfl0.2``), and grouping on the stored label silently
    splits the C2 gate family in two.
    """
    try:
        return family_key(record)
    except Exception:  # pragma: no cover - only for hand-written fixtures
        return record.get("family", "")


def _series(records, family):
    """(times, errors-in-percent) of one family, sorted, archive-only."""
    picked = [(r["final_time"], r["max_percent"]) for r in records
              if _record_family(r) == family]
    picked.sort()
    return [p[0] for p in picked], [p[1] for p in picked]


def _local_slope(times, errors) -> float:
    """Slope of the last two checkpoints: the tail's *instantaneous* exponent."""
    if len(times) < 2:
        return float("nan")
    return math.log(errors[-1] / errors[-2]) / math.log(times[-1] / times[-2])


def _model_alpha(times, errors, archive_final=None) -> dict:
    """The four extrapolation models, each reduced to one exponent."""
    variants = exponent_variants(times, errors)
    window = [(t, e) for t, e in zip(times, errors)
              if t >= 1.0 and (archive_final is None or e > 0.0)]
    if len(window) < 3:
        raise SystemExit(f"need >= 3 checkpoints for a fit, got {len(window)}")
    lt = np.log(np.array([p[0] for p in window]))
    le = np.log(np.array([p[1] for p in window]))
    ols = float(np.polyfit(lt, le, 1)[0])
    return {
        # a fit through the whole window says the early plateau carries weight
        "ols_window": {
            "alpha": ols,
            "note": "least squares over the whole archived window",
        },
        # the gate's own conservative reading (worst of the robust variants)
        "robust_window": {
            "alpha": float(variants["robust"]),
            "note": f"worst robust variant ({variants['robust_at']}), the number C2 gates on",
        },
        # the tail slope: what the error is doing *now*, not on average
        "tail_local": {
            "alpha": _local_slope(times, errors),
            "note": "two-point slope of the last archived pair",
        },
        # the physical upper bound the C2 threshold encodes: dissipative e ~ t
        "linear_bound": {
            "alpha": 1.0,
            "note": "e proportional to elapsed time (the alpha = 1 bound C2 permits)",
        },
    }


def _digest(paths) -> str:
    sha = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        sha.update(path.name.encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def _relpath(path: Path) -> str:
    """Path relative to the repo when possible (tmp dirs live outside it)."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def reference_fold_defect(times, points: int = 80) -> dict:
    """Relative L2 gap between the folded (``min_image``) and the lattice reference.

    This is the *instrument* question at ``t = 40``: the published archive was
    measured against the min-image fold, which drops the neighbouring image
    shells.  As the displacement grows the nearest folded image sits closer to
    the box, so the dropped tail grows -- exactly why the long-time runs use the
    lattice reference.  No solver is involved: both references are analytic.
    """
    import torch

    axis = torch.linspace(DOMAIN[0][0], DOMAIN[0][1], points, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    travel = max(abs(BACKGROUND[0]), abs(BACKGROUND[1]))
    out = {}
    for time in times:
        folded = vortex_primitive(grid_x, grid_y, float(time), STRENGTH,
                                  BACKGROUND, period=PERIOD)
        half_box = 0.5 * max(abs(DOMAIN[0][1] - DOMAIN[0][0]),
                             abs(DOMAIN[1][1] - DOMAIN[1][0]))
        images = 2 + int(math.ceil(travel * float(time) / min(PERIOD)))
        lattice = vortex_primitive(grid_x, grid_y, float(time), STRENGTH,
                                   BACKGROUND, period=PERIOD, images=images)
        worst = 0.0
        for i in range(len(folded)):
            den = float(torch.sqrt(torch.mean(lattice[i] ** 2)))
            diff = float(torch.sqrt(torch.mean((folded[i] - lattice[i]) ** 2)))
            worst = max(worst, 100.0 * diff / den if den > 0 else 0.0)
        out[f"{float(time):g}"] = worst
        del half_box
    return out


def predict(archive_dir: Path = ARCHIVE, output_dir: Path = OUTPUT_DIR,
            targets=None, predict_time: float = PREDICT_TIME) -> dict:
    """Freeze the out-of-sample prediction of the ``t = 40`` readings."""
    targets = targets or TARGETS
    paths = sorted(archive_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no archived records in {archive_dir}")
    records = [json.loads(p.read_text()) for p in paths]

    late = [(r.get("task_key"), r.get("final_time")) for r in records
            if (r.get("final_time") or 0.0) > ARCHIVE_TIME_CAP + 1e-12]
    if late:
        raise SystemExit(
            "prediction leak guard: archived records with final_time > "
            f"{ARCHIVE_TIME_CAP}: {late}.  A prediction that has seen the "
            "answer is not a prediction.")

    families = {}
    for family, role in targets.items():
        times, errors = _series(records, family)
        if len(times) < 3:
            raise SystemExit(f"{family}: {len(times)} checkpoints, need >= 3")
        models = _model_alpha(times, errors)
        anchor_t, anchor_e = times[-1], errors[-1]
        for name, model in models.items():
            model["predicted_percent"] = anchor_e * (predict_time / anchor_t) ** model["alpha"]
            model["predicted_growth_ratio_vs_anchor"] = (predict_time / anchor_t) ** model["alpha"]
        spread = [m["predicted_percent"] for m in models.values()]
        families[family] = {
            "role": role,
            "archive_times": times,
            "archive_errors_percent": errors,
            "anchor_time": anchor_t,
            "anchor_percent": anchor_e,
            "models": models,
            "prediction_spread_percent": [min(spread), max(spread)],
            "tolerance_percent": 100.0 * TOLERANCE,
        }

    doc = {
        "case": "CASE-DG-EULER-VORTEX-ERR",
        "predict_time": predict_time,
        "archive_time_cap": ARCHIVE_TIME_CAP,
        "archive_dir": str(_relpath(archive_dir)),
        "archive_digest_sha256": _digest(paths),
        "archive_files": [p.name for p in paths],
        "frozen_before_runs": True,
        "families": families,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(json.dumps(doc, indent=2))

    lines = ["# 盲预测（t = 40，冻结于跑数之前）", "",
             f"档案输入 `{doc['archive_dir']}`，digest `{doc['archive_digest_sha256'][:16]}`，"
             f"全部检查点 `final_time <= {ARCHIVE_TIME_CAP}`。", "",
             "| 族 | 锚点 t | 锚点误差 % | OLS α | robust α | 尾段 α | 线性 α=1 | 预测区间 % |",
             "|---|---|---|---|---|---|---|---|"]
    for family, entry in families.items():
        m = entry["models"]
        lines.append(
            f"| `{family}` | {entry['anchor_time']:g} | {entry['anchor_percent']:.4f} | "
            f"{m['ols_window']['alpha']:.3f} → {m['ols_window']['predicted_percent']:.4f} | "
            f"{m['robust_window']['alpha']:.3f} → {m['robust_window']['predicted_percent']:.4f} | "
            f"{m['tail_local']['alpha']:.3f} → {m['tail_local']['predicted_percent']:.4f} | "
            f"{m['linear_bound']['predicted_percent']:.4f} | "
            f"{entry['prediction_spread_percent'][0]:.4f} … "
            f"{entry['prediction_spread_percent'][1]:.4f} |")
    (output_dir / "predictions.md").write_text("\n".join(lines) + "\n")
    return doc


# ---------------------------------------------------------------------------
# analysis (runs after the t = 40 checkpoints land)
# ---------------------------------------------------------------------------

def _load_runs(run_dirs) -> list:
    records = []
    for directory in run_dirs:
        for path in sorted(Path(directory).glob("*.json")):
            records.append(json.loads(path.read_text()))
    return records


def _horizon_time(t_last: float, e_last: float, alpha: float,
                  tolerance_percent: float) -> float:
    """Extrapolated time at which ``e`` reaches the case tolerance."""
    if not (alpha > 0.0) or e_last <= 0.0:
        return float("inf")
    if e_last >= tolerance_percent:
        return t_last
    return t_last * (tolerance_percent / e_last) ** (1.0 / alpha)


def analyze(predictions_path: Path = PREDICTIONS, run_dirs=None,
            output_dir: Path = OUTPUT_DIR) -> dict:
    predictions = json.loads(Path(predictions_path).read_text())
    run_dirs = run_dirs or [output_dir / "results"]
    records = _load_runs(run_dirs)
    # no records at all is *absence of evidence*: the verdict falls to
    # INCOMPLETE below instead of crashing or declaring anything.

    by_family = {}
    for record in records:
        by_family.setdefault(_record_family(record), []).append(record)

    tolerances = 100.0 * TOLERANCE
    families = {}
    for family, entry in predictions["families"].items():
        new = sorted(by_family.get(family, []), key=lambda r: r["final_time"])
        if not new:
            families[family] = {"status": "missing", "role": entry["role"]}
            continue
        target_time = predictions["predict_time"]
        hit = [r for r in new if abs(r["final_time"] - target_time) < 1e-9]
        if not hit:
            families[family] = {
                "status": "missing",
                "role": entry["role"],
                "landed_times": [r["final_time"] for r in new],
            }
            continue
        record = hit[0]
        measured = record["max_percent"]
        scores = {}
        for name, model in entry["models"].items():
            predicted = model["predicted_percent"]
            scores[name] = {
                "alpha": model["alpha"],
                "predicted_percent": predicted,
                "relative_error_percent": 100.0 * (measured - predicted) / measured,
                "within_10_percent": abs(measured - predicted) <= 0.10 * measured,
            }
        best = min(scores, key=lambda k: abs(scores[k]["relative_error_percent"]))

        times = list(entry["archive_times"]) + [record["final_time"]]
        errors = list(entry["archive_errors_percent"]) + [measured]
        extended = exponent_variants(times, errors)
        tail = _local_slope(times, errors)
        families[family] = {
            "status": "measured",
            "role": entry["role"],
            "times": times,
            "errors_percent": errors,
            "measured_percent": measured,
            "reference": record["config"].get("reference"),
            "steps": record.get("steps"),
            "wall_seconds": record.get("wall_seconds"),
            "scores": scores,
            "best_model": best,
            "prediction_confirmed": bool(
                entry["prediction_spread_percent"][0] * 0.9 <= measured
                <= entry["prediction_spread_percent"][1] * 1.1),
            "extended": {
                "window": [times[0], times[-1]],
                "ols": float(extended["ols"]),
                "robust": float(extended["robust"]),
                "robust_at": extended["robust_at"],
                "tail_local": tail,
                "gate_bound": 1.0,
                "gate_holds": bool(extended["robust"] <= 1.0 + 1e-9),
            },
            "growth_ratio_window": errors[-1] / errors[0],
            "tolerance_percent": tolerances,
            "within_tolerance": bool(measured < tolerances),
            "horizon_to_tolerance": {
                f"alpha={name}": _horizon_time(times[-1], measured, model_alpha,
                                               tolerances)
                for name, model_alpha in (
                    ("extended_ols", float(extended["ols"])),
                    ("extended_robust", float(extended["robust"])),
                    ("tail_local", tail),
                    ("linear 1", 1.0),
                )
            },
        }

    # worst field per checkpoint time is what decides instrument adequacy
    defect = reference_fold_defect([1.0, 10.0, 20.0, predictions["predict_time"]])
    verdict = _verdict(families, defect)
    doc = {
        "case": predictions["case"],
        "predict_time": predictions["predict_time"],
        "archive_digest_sha256": predictions["archive_digest_sha256"],
        "reference_defect_percent": defect,
        "families": families,
        "verdict": verdict,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "horizon.json").write_text(json.dumps(doc, indent=2))
    (output_dir / "horizon.md").write_text(_markdown(doc))
    return doc


def _verdict(families, defect) -> dict:
    design = families.get("dg_d2_n32_cfl0.2", {})
    checks = {}
    if design.get("status") != "measured":
        checks["design_measured"] = False
        return {"status": "INCOMPLETE", "checks": checks,
                "reason": "design family t = 40 checkpoint has not landed"}
    checks["design_measured"] = True
    checks["extended_alpha_within_bound"] = design["extended"]["gate_holds"]
    checks["design_within_tolerance"] = design["within_tolerance"]
    # a reference whose own defect is a large fraction of the reading cannot
    # certify anything at that time; the lattice reference is the instrument.
    worst_defect = max(defect.values()) if defect else 0.0
    checks["reference_defect_negligible"] = bool(
        worst_defect <= 0.10 * design["measured_percent"])
    failed = [k for k, ok in checks.items() if not ok]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "worst_reference_defect_percent": worst_defect,
        "design_measured_percent": design["measured_percent"],
        "design_extended_alpha": design["extended"]["robust"],
    }


def _markdown(doc: dict) -> str:
    lines = ["# 误差守恒的时间地平线：t = 40 延伸读数", "",
             f"预测冻结档案 digest `{doc['archive_digest_sha256'][:16]}`；"
             f"预测时刻 t = {doc['predict_time']:g}。", "",
             "| 族 | 实测 % | 最佳模型 | 该模型偏差 | 区间内 | 扩展窗口 α (robust) | α ≤ 1 | 容差内 |",
             "|---|---|---|---|---|---|---|---|"]
    for family, entry in doc["families"].items():
        if entry.get("status") != "measured":
            lines.append(f"| `{family}` | 未落盘 | — | — | — | — | — | — |")
            continue
        best = entry["best_model"]
        scores = entry["scores"][best]
        lines.append(
            f"| `{family}` | {entry['measured_percent']:.4f} | {best} | "
            f"{scores['relative_error_percent']:+.2f}% | "
            f"{'yes' if entry['prediction_confirmed'] else 'no'} | "
            f"{entry['extended']['robust']:.3f} | "
            f"{'yes' if entry['extended']['gate_holds'] else 'NO'} | "
            f"{'yes' if entry['within_tolerance'] else 'NO'} |")
    lines += ["", "## 参考解截断缺陷：min_image 折叠 vs 格点求和（最坏场，随位移增长）", "",
              "| t | 缺陷 % |", "|---|---|"]
    for t, value in sorted(doc["reference_defect_percent"].items(),
                          key=lambda kv: float(kv[0])):
        lines.append(f"| {float(t):g} | {value:.6g} |")
    lines += ["", f"## 裁决：**{doc['verdict']['status']}**", "",
              "```json", json.dumps(doc["verdict"], indent=2), "```", ""]
    return "\n".join(lines)


def figures(doc: dict, output_dir: Path = OUTPUT_DIR) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    measured = {k: v for k, v in doc["families"].items() if v.get("status") == "measured"}
    if not measured:
        return made

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for family, entry in measured.items():
        t = np.array(entry["times"])
        e = np.array(entry["errors_percent"])
        line, = axes[0].loglog(t, e, "o-", label=family)
        axes[0].loglog([t[-1]], [e[-1]], "o", color=line.get_color(),
                       markerfacecolor="none", markersize=9)
        for name, model in doc["families"][family].get("scores", {}).items():
            axes[0].loglog([entry["times"][-1]], [model["predicted_percent"]], "x",
                           color=line.get_color(), alpha=0.7)
        local_t, local_a = [], []
        for i in range(2, len(t)):
            local_t.append(math.sqrt(t[i] * t[i - 1]))
            local_a.append(math.log(e[i] / e[i - 1]) / math.log(t[i] / t[i - 1]))
        axes[1].semilogx(local_t, local_a, "o-", color=line.get_color(), label=family)
    axes[0].axhline(100.0 * TOLERANCE, color="crimson", ls="--", lw=1.2,
                    label=f"{100 * TOLERANCE:g} % case tolerance")
    axes[0].set_xlabel("t (transport time)")
    axes[0].set_ylabel("max-field relative L2 error (%)")
    axes[0].set_title("error vs time; × = model predictions at t = 40")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].axhline(1.0, color="crimson", ls="--", lw=1.2, label="C2 bound alpha = 1")
    axes[1].set_xlabel("t (midpoint of the segment)")
    axes[1].set_ylabel("local exponent  alpha = d log e / d log t")
    axes[1].set_title("local accumulation exponent")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    path = output_dir / "fig_longtime_horizon.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    made.append(path.name)

    defect = doc["reference_defect_percent"]
    if defect:
        ts = sorted(float(k) for k in defect)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.loglog(ts, [defect[f"{t:g}"] for t in ts],
                  "s-", color="darkorange", label="min_image fold vs lattice sum (worst field)")
        design = measured.get("dg_d2_n32_cfl0.2")
        if design:
            ax.axhline(design["measured_percent"], color="navy", ls="--",
                       label="measured design-family error at t = 40")
            ax.axhline(0.10 * design["measured_percent"], color="crimson", ls=":",
                       label="10 % of the reading (instrument budget)")
        ax.set_xlabel("t")
        ax.set_ylabel("reference defect (%)")
        ax.set_title("why the long-time runs need the lattice reference")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        path = output_dir / "fig_longtime_reference_defect.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path.name)
    return made


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--archive-dir", default=str(ARCHIVE))
    parser.add_argument("--results-dir", action="append", default=None,
                        help="t = 40 result directory (repeatable)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if args.predict:
        doc = predict(Path(args.archive_dir), output_dir)
        print(f"frozen predictions for {len(doc['families'])} families -> "
              f"{output_dir / 'predictions.json'}")
        print((output_dir / "predictions.md").read_text())
    if args.analyze:
        runs = [Path(p) for p in (args.results_dir or [])] or None
        doc = analyze(output_dir / "predictions.json", runs, output_dir)
        print((output_dir / "horizon.md").read_text())
        if args.figures:
            print("figures:", figures(doc, output_dir))
        return 0 if doc["verdict"]["status"] == "PASS" else 1
    if not (args.predict or args.analyze):
        parser.error("nothing to do: pass --predict and/or --analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())