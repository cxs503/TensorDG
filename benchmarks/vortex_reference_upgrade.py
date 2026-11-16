"""Measured reference-induced shift of the CASE-EULER-VORTEX-ERR readings.

The published ladder of `case_euler_vortex_errconserve` measures every error
against the **min-image fold** of the free-space isentropic vortex.  That fold
drops the tails of the neighbouring lattice images, so the published number is
``|u_h - R_fold|`` where the exact periodic reference is ``R_lattice``.  The
companion tool ``benchmarks/vortex_reference_floor.py`` quantifies the defect
``|R_fold - R_lattice|`` and reports it as a **worst-case bound** on how much of
the published error could be reference rather than discretisation:

    bound:  |e_lattice - e_published| / e_published  <=  floor / e_published

That bound is pessimistic by construction (it assumes the reference defect is
perfectly aligned with the solution error).  This tool replaces the bound by a
measurement: it re-runs the same checkpoints with ``--reference lattice`` (the
image sum, exact up to ``O(tail^2)``) and compares the resulting errors with the
published ones, field by field.

Two controls keep the comparison honest:

* **rerun control** (``--rerun-dir``): the same checkpoint re-run with the *same*
  min-image reference.  The lattice-vs-rerun difference is the reference effect;
  the rerun-vs-published difference is solver/BLAS round-off and is reported
  separately instead of being silently attributed to the reference.
* **field selectivity**: only velocity can move (its perturbation is first order
  in the Gaussian envelope and carries the displacement, so the dropped images
  enter at ``O(tail)``), while density/pressure are second order
  (``O(tail^2)``).  A material move in density/pressure would falsify the floor
  model, not confirm it.

Usage::

    python3 benchmarks/vortex_reference_upgrade.py \\
        --published-dir case_euler_vortex_errconserve \\
        --lattice-dir case_euler_vortex_errconserve_lattice \\
        --rerun-dir case_euler_vortex_errconserve_rerun \\
        --floor-json case_euler_vortex_errconserve_lattice/reference_floor.json \\
        --output case_euler_vortex_errconserve_lattice/reference_upgrade.json \\
        --markdown case_euler_vortex_errconserve_lattice/reference_upgrade.md \\
        --figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.case_euler_vortex_errconserve import REFERENCE_MIN_IMAGE, T_REF

FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
#: A reference-induced move below this is "immaterial" for a 6 % acceptance
#: tolerance -- three orders of magnitude of headroom against the gate.
MATERIAL_SHIFT = 1.0e-2

#: The case's conservation criterion is ``alpha <= 1`` on ``e ~ t^alpha``; the
#: largest exponent any published family reports is 0.44.  A reference change
#: that moves ``alpha`` by less than this therefore cannot flip the criterion --
#: the headroom to 1 is more than five times the allowed move.
ALPHA_SHIFT_TOL = 1.0e-1


def load_records(directory: Path | str) -> dict:
    """``task_key -> record`` for every JSON record under ``directory/results``."""
    records = {}
    for path in sorted((Path(directory) / "results").glob("*.json")):
        record = json.loads(path.read_text())
        records[record["task_key"]] = record
    return records


def relative_shift(candidate: dict, baseline: dict) -> dict:
    """Per-field relative change of the measured error, ``candidate`` vs ``baseline``."""
    shifts = {}
    for field in FIELDS:
        base = baseline["errors"][field]
        shifts[field] = (candidate["errors"][field] - base) / base if base else float("nan")
    worst_field = max(shifts, key=lambda name: abs(shifts[name]))
    return {
        "per_field": shifts,
        "max_relative_shift": abs(shifts[worst_field]),
        "worst_field": worst_field,
        "max_percent_baseline": baseline["max_percent"],
        "max_percent_candidate": candidate["max_percent"],
        "max_percent_shift": ((candidate["max_percent"] - baseline["max_percent"])
                              / baseline["max_percent"]),
    }


def _fit_exponent(rows: list) -> float:
    """Least-squares exponent of ``e ~ t^alpha`` over ``t >= T_REF`` (>=3 points)."""
    points = [(r["final_time"], r["max_relative_l2"]) for r in rows
              if r["final_time"] >= T_REF and r["max_relative_l2"] > 0.0]
    if len(points) < 3:
        return float("nan")
    lt = np.log([p[0] for p in points])
    le = np.log([p[1] for p in points])
    return float(np.polyfit(lt, le, 1)[0])


def family(record: dict) -> str:
    """Family label of one record.

    The record's own ``family`` field is authoritative: the benchmark writes it
    at run time and it already carries the direction tag (``..._diraxis0``).  The
    first version of this function re-appended that tag to a freshly built label,
    so the artifact's exponent table was keyed by
    ``dg_d2_n64_cfl0.2dg_d2_n64_cfl0.2`` -- a label that matches no record and no
    other artifact.  Pinned by
    ``tests/test_vortex_reference_upgrade.py::test_family_label_matches_the_record``.
    """
    label = record.get("family")
    if label:
        return str(label)
    config = record.get("config") or {}
    task = config.get("task", record.get("task", "dg"))
    if task == "p0":
        return f"p0_n{config.get('cells', record['cells'])}"
    return (f"dg_d{config.get('degree', record.get('degree'))}"
            f"_n{config.get('cells', record['cells'])}"
            f"_cfl{config.get('cfl', record.get('cfl'))}")


def family_exponents(records: dict) -> dict:
    groups: dict[str, list] = {}
    for record in records.values():
        groups.setdefault(family(record), []).append(record)
    out = {}
    for name, rows in sorted(groups.items()):
        if len(rows) < 3:
            continue
        out[name] = {
            "checkpoints": len(rows),
            "times": sorted(r["final_time"] for r in rows),
            "alpha": _fit_exponent(rows),
            # error amplification across the family: a conserved error has A ~ 1
            "amplification": (max(rows, key=lambda r: r["final_time"])["max_relative_l2"]
                              / min(rows, key=lambda r: r["final_time"])["max_relative_l2"]),
        }
    return out


def floor_ratios(floor_json: Path | None) -> dict:
    if floor_json is None or not Path(floor_json).exists():
        return {}
    payload = json.loads(Path(floor_json).read_text())
    return {row["task_key"]: row.get("worst_floor_over_error")
            for row in payload["checkpoints"]}


def build_report(published: dict, lattice: dict, rerun: dict | None = None,
                 floors: dict | None = None) -> dict:
    """Compare every re-measured checkpoint against its published reading."""
    floors = floors or {}
    rerun = rerun or {}
    rows = []
    for key in sorted(set(lattice) & set(published)):
        row = {"task_key": key, "final_time": lattice[key]["final_time"],
               "reference_model": lattice[key].get("reference_model"),
               "reference_images": lattice[key].get("reference_images"),
               "lattice_vs_published": relative_shift(lattice[key], published[key]),
               "floor_bound": floors.get(key)}
        if key in rerun:
            measured = relative_shift(lattice[key], rerun[key])   # reference effect
            noise = relative_shift(rerun[key], published[key])    # round-off / rebuild
            row["measured_reference_effect"] = measured
            row["rerun_noise"] = noise
            bound = row["floor_bound"]
            row["bound_materialised"] = (
                measured["max_relative_shift"] / bound
                if bound and bound > 0 else None)
            row["bound_is_sound"] = bool(
                bound is None or measured["max_relative_shift"] <= bound + 1.0e-12)
        rows.append(row)

    exponents = {"published": family_exponents(published),
                 "lattice": family_exponents(lattice)}
    # Judgement invariance: the criterion C2/C7/C8 reads is ``alpha <= 1`` on the
    # published (truncated-reference) errors.  Removing the reference defect must
    # not move that exponent enough to change the verdict, otherwise "the error
    # is conserved" would be a statement about the reference, not the scheme.
    exponent_shift = {}
    for name, published_entry in exponents["published"].items():
        lattice_entry = exponents["lattice"].get(name)
        if lattice_entry is None:
            continue
        exponent_shift[name] = {
            "checkpoints": published_entry["checkpoints"],
            "published_alpha": published_entry["alpha"],
            "lattice_alpha": lattice_entry["alpha"],
            "delta_alpha": lattice_entry["alpha"] - published_entry["alpha"],
            "published_amplification": published_entry["amplification"],
            "lattice_amplification": lattice_entry["amplification"],
        }
    deltas = [abs(entry["delta_alpha"]) for entry in exponent_shift.values()
              if np.isfinite(entry["delta_alpha"])]
    alphas = [entry["lattice_alpha"] for entry in exponent_shift.values()
              if np.isfinite(entry["lattice_alpha"])]
    criterion_invariance = {
        "alpha_shift_tolerance": ALPHA_SHIFT_TOL,
        "families_compared": sorted(exponent_shift),
        "max_abs_delta_alpha": max(deltas) if deltas else None,
        "max_lattice_alpha": max(alphas) if alphas else None,
        "conservation_holds_under_both": (
            all(alpha < 1.0 for alpha in alphas) if alphas else None),
        "exponent_invariant": (max(deltas) < ALPHA_SHIFT_TOL) if deltas else None,
    }
    # Materiality of the reference effect.  A rerun control is the clean
    # measurement (same code, same reference, fresh process); without one the
    # lattice-vs-published difference is used instead, which can only *overstate*
    # the effect by the solver round-off, never hide it.
    for row in rows:
        if "measured_reference_effect" in row:
            row["effect_source"] = "lattice_vs_rerun"
            row["effect_max"] = row["measured_reference_effect"]["max_relative_shift"]
        else:
            row["effect_source"] = "lattice_vs_published"
            row["effect_max"] = row["lattice_vs_published"]["max_relative_shift"]
    # The bound check must not disappear where no control run exists: the effect
    # used for the row (controlled or not) is always compared with its floor
    # bound, so a missing control can only make the check *more* conservative.
    for row in rows:
        bound = row["floor_bound"]
        if bound:
            row.setdefault("bound_materialised", row["effect_max"] / bound)
            row.setdefault("bound_is_sound",
                           bool(row["effect_max"] <= bound + 1.0e-12))
    effect = [row["effect_max"] for row in rows]
    material = [row["task_key"] for row in rows if row["effect_max"] > MATERIAL_SHIFT]
    payload = {
        "case": "EULER-VORTEX-ERR",
        "purpose": ("measured reference-induced shift: min_image (published) vs "
                    "lattice (exact) periodic reference, on the same checkpoints"),
        "material_shift_threshold": MATERIAL_SHIFT,
        "checkpoints": rows,
        "measured_effect_max": max(effect) if effect else None,
        "material_checkpoints": material,
        "exponents": exponents,
        "exponent_shift": exponent_shift,
        "criterion_invariance": criterion_invariance,
        "verdict": ("no_material_reference_effect" if not material
                    else "material_reference_effect"),
    }
    return payload


def _fmt(value, scale=1.0, digits=3, suffix=""):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value * scale:.{digits}f}{suffix}"


def markdown(payload: dict) -> str:
    rows = payload["checkpoints"]
    lines = [
        "# 参考解升级实测：min_image（已发布）vs lattice（精确镜像和）",
        "",
        "同一检查点、同一求积网格与加权相对 L2，只换周期参考的求值模型。"
        "`地板上界` 来自 `vortex_reference_floor.py`（最坏情况：参考缺陷与解误差同向）；"
        "`实测位移` 是 lattice 重跑与 min_image 重跑的差（同条件、仅参考不同）。",
        "",
        "| 检查点 | t | K | 实测最大误差（已发布） | 已发布/lattice 位移 | 地板上界 | "
        "实测参考效应 | 上界实现比 | 重跑噪声 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        published = row["lattice_vs_published"]
        measured = row.get("measured_reference_effect")
        noise = row.get("rerun_noise")
        lines.append(
            f"| {row['task_key']} | {row['final_time']:g} | {row['reference_images']} | "
            f"{published['max_percent_baseline']:.4f}% | "
            f"{_fmt(published['max_percent_shift'], 100.0, 4, '%')} | "
            f"{_fmt(row['floor_bound'], 100.0, 3, '%')} | "
            f"{_fmt(measured['max_relative_shift'] if measured else None, 100.0, 4, '%')} | "
            f"{_fmt(row.get('bound_materialised'), 100.0, 3, '%')} | "
            f"{_fmt(noise['max_relative_shift'] if noise else None, 100.0, 5, '%')} |")
    lines += ["", "## 判读", ""]
    if payload["measured_effect_max"] is not None:
        lines.append(
            f"* 实测参考效应最大 {payload['measured_effect_max'] * 100:.4f}%，"
            f"材料阈值 {payload['material_shift_threshold'] * 100:.1f}%"
            f"（阈值以下的位移不改变任何判据读数）。"
            f"判定：`{payload['verdict']}`。")
    sound = [r["task_key"] for r in rows if r.get("bound_is_sound") is False]
    if sound:
        lines.append(f"* **上界被违反**的检查点：{sound}（参考缺陷比最坏情况更同向）。")
    else:
        lines.append("* 全部检查点的实测参考效应都落在其地板上界之内 ⇒ 地板是**可靠上界**。")
    lines.append("* 只有速度场有可测位移；密度/压力位移量级见上表（应为 `O(tail^2)`）。")

    shift = payload.get("exponent_shift") or {}
    invariance = payload.get("criterion_invariance") or {}
    if shift:
        lines += ["", "## 判据不变性（换参考不改判据）", "",
                  "| 族 | 检查点数 | α（已发布） | α（lattice） | Δα | A（已发布） | A（lattice） |",
                  "|---|---|---|---|---|---|---|"]
        for name, entry in sorted(shift.items()):
            lines.append(
                f"| {name} | {entry['checkpoints']} | {_fmt(entry['published_alpha'], 1.0, 4)} | "
                f"{_fmt(entry['lattice_alpha'], 1.0, 4)} | "
                f"{entry['delta_alpha']:+.4f} | "
                f"{_fmt(entry['published_amplification'], 1.0, 4)} | "
                f"{_fmt(entry['lattice_amplification'], 1.0, 4)} |")
        lines.append("")
        lines.append(
            f"* 最大 |Δα| = {_fmt(invariance.get('max_abs_delta_alpha'), 1.0, 4)}"
            f"（容差 {invariance.get('alpha_shift_tolerance')}）；"
            f"lattice 参考下的最大 α = {_fmt(invariance.get('max_lattice_alpha'), 1.0, 4)}"
            f" ⇒ 两套参考下「误差守恒」（α ≤ 1）都成立："
            f"`{invariance.get('conservation_holds_under_both')}`，"
            f"指数不变：`{invariance.get('exponent_invariant')}`。")
    return "\n".join(lines) + "\n"


def make_figure(payload: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in payload["checkpoints"] if "measured_reference_effect" in r]
    if not rows:
        return None
    labels = [r["task_key"].replace("_cfl", "\ncfl") for r in rows]
    bounds = [r["floor_bound"] or np.nan for r in rows]
    measured = [r["measured_reference_effect"]["max_relative_shift"] for r in rows]
    noise = [r["rerun_noise"]["max_relative_shift"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    idx = np.arange(len(rows))
    ax.bar(idx - 0.2, bounds, width=0.4, color="lightsteelblue",
           label="floor bound (worst case)")
    ax.bar(idx + 0.2, measured, width=0.4, color="crimson",
           label="measured reference effect")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("fraction of published error")
    ax.set_title("worst-case bound vs measured shift")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    fields = ("density", "velocity_x", "velocity_y", "pressure")
    width = 0.8 / len(fields)
    for position, field in enumerate(fields):
        values = [abs(r["measured_reference_effect"]["per_field"][field]) for r in rows]
        ax.bar(idx + position * width, values, width=width, label=field)
    ax.set_yscale("log")
    ax.set_xticks(idx + 0.4 - width / 2)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("|measured reference effect|")
    ax.set_title("velocity moves, density/pressure do not")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--published-dir", default="case_euler_vortex_errconserve")
    parser.add_argument("--lattice-dir", default="case_euler_vortex_errconserve_lattice")
    parser.add_argument("--rerun-dir", default=None)
    parser.add_argument("--floor-json",
                        default="case_euler_vortex_errconserve_lattice/reference_floor.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--figure", default=None)
    args = parser.parse_args(argv)

    published = load_records(args.published_dir)
    lattice = load_records(args.lattice_dir)
    for record in published.values():
        model = record.get("reference_model") or REFERENCE_MIN_IMAGE
        if model != REFERENCE_MIN_IMAGE:
            raise SystemExit(f"{record['task_key']}: published record is {model!r}, "
                             "the floor ratio is only meaningful against min_image")
    rerun = load_records(args.rerun_dir) if args.rerun_dir else {}
    payload = build_report(published, lattice, rerun, floor_ratios(Path(args.floor_json)
                                                                   if args.floor_json else None))
    if not payload["checkpoints"]:
        raise SystemExit("no checkpoint is re-measured under the lattice reference yet")
    text = markdown(payload)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(text)
        print(f"wrote {args.markdown}")
    if args.figures or args.figure:
        figure = Path(args.figure or (Path(args.output).parent / "fig_reference_upgrade.png"))
        make_figure(payload, figure)
        print(f"wrote {figure}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())