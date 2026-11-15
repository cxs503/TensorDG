"""Truncation floor of the min-image periodic reference (verification debt).

The long-time errors of CASE-EULER-VORTEX-ERR are measured against
``vortex_primitive(..., period=10)``, i.e. against the *min-image fold* of the
free-space isentropic vortex.  That fold keeps only the nearest lattice image and
therefore drops the tails of the neighbouring images, which the exact solution of
the periodic problem does carry.  This tool measures that floor **in the same
metric as the case gate** (same quadrature grid, same weighted relative L2) and
compares it with the published per-checkpoint errors, so that the following
question is answered by measurement instead of assertion:

    is the min-image reference accurate enough that the reported discretisation
    error is not a property of the reference?

For the case at hand the floor is second order in the Gaussian envelope for
density/pressure (``~5e-13``) but first order for velocity (``~1.5e-6``, the
envelope is multiplied by the displacement), so only the velocity fields can be
reference-limited -- and only on the checkpoints whose discretisation error
approaches ``1e-6``.

Usage::

    python3 benchmarks/vortex_reference_floor.py \\
        --results-dir case_euler_vortex_errconserve \\
        --output case_euler_vortex_errconserve_lattice/reference_floor.json \\
        --figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND, DOMAIN, LADDER, P0_QUADRATURE, PERIOD, REFERENCE_LATTICE,
    REFERENCE_MIN_IMAGE, STRENGTH, T_REF, reference_images, task_key,
)
from tensordg.euler_2d import vortex_primitive

FIELDS = ("density", "velocity_x", "velocity_y", "pressure")


def quadrature_grid(config: dict):
    """The quadrature grid the case gate uses for this checkpoint."""
    (xmin, xmax), (ymin, ymax) = DOMAIN
    cells = config["cells"]
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    x = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    if config.get("task") == "p0":
        nodes_np, weights_np = np.polynomial.legendre.leggauss(P0_QUADRATURE)
        order = P0_QUADRATURE
        weights = torch.as_tensor(weights_np, dtype=torch.float64)
    else:
        nodes_np, _, _, _, _ = dg2.legendre_operators(
            config.get("degree", 2), max(config.get("degree", 2) + 2, 3))
        nodes_np = nodes_np.numpy() if isinstance(nodes_np, torch.Tensor) else nodes_np
        order = len(nodes_np)
        weights = torch.as_tensor(
            np.polynomial.legendre.leggauss(order)[1], dtype=torch.float64)
    nodes = torch.as_tensor(nodes_np, dtype=torch.float64)
    xq = x[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = y[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    weight = weights[:, None] * weights[None, :]
    return xq, yq, weight


def reference_fields(xq, yq, final_time, images, strength=STRENGTH, background=BACKGROUND):
    return vortex_primitive(xq, yq, final_time, strength, background,
                            period=PERIOD, images=images)


def weighted_relative_l2(a, b, weight):
    """``||a - b|| / ||b||`` in the case gate's weighted L2 (b is the reference)."""
    difference = ((a - b) ** 2 * weight).sum()
    denominator = (b ** 2 * weight).sum()
    return float(torch.sqrt(difference / denominator))


def floor_at(config: dict, images=None) -> dict:
    """min-image vs lattice reference difference for one checkpoint."""
    final_time = config["final_time"]
    if images is None:
        images = reference_images(final_time)
    xq, yq, weight = quadrature_grid(config)
    fold = reference_fields(xq, yq, final_time, None)
    lattice = reference_fields(xq, yq, final_time, images)
    per_field = {name: weighted_relative_l2(fold[i], lattice[i], weight)
                 for i, name in enumerate(FIELDS)}
    return {
        "task_key": task_key(config),
        "task": config.get("task", "dg"),
        "degree": config.get("degree", 0 if config.get("task") == "p0" else 2),
        "cells": config["cells"],
        "final_time": final_time,
        "cfl": config.get("cfl"),
        "images": images,
        "floor_relative_l2": per_field,
        "floor_percent": {name: 100.0 * value for name, value in per_field.items()},
        "max_floor_percent": 100.0 * max(per_field.values()),
        "metric": ("same quadrature grid and weighted relative L2 as the case "
                   "gate; denominator = lattice reference"),
    }


def compare_with_published(floors: list, results_dir: Path) -> list:
    """Attach the published min-image error and the floor/error ratio."""
    rows = []
    for floor in floors:
        path = results_dir / "results" / f"{floor['task_key']}.json"
        row = dict(floor)
        if path.exists():
            published = json.loads(path.read_text())
            published_model = published.get("reference_model", REFERENCE_MIN_IMAGE)
            if published_model != REFERENCE_MIN_IMAGE:
                raise SystemExit(
                    f"{path} was measured against {published_model!r}, the floor "
                    "ratio is only meaningful against the min-image reading")
            row["published_errors"] = published["errors"]
            row["published_max_percent"] = published["max_percent"]
            row["floor_over_published"] = {
                name: floor["floor_relative_l2"][name] / published["errors"][name]
                for name in FIELDS}
            row["worst_floor_over_error"] = max(row["floor_over_published"].values())
            row["worst_floor_over_error_field"] = max(
                row["floor_over_published"], key=row["floor_over_published"].get)
        else:
            row["published_errors"] = None
            row["floor_over_published"] = None
            row["worst_floor_over_error"] = None
            row["worst_floor_over_error_field"] = None
        rows.append(row)
    return rows


def markdown(rows: list, results_dir: Path) -> str:
    measured = [r for r in rows if r["published_errors"] is not None]
    worst = max(measured, key=lambda r: r["worst_floor_over_error"]) if measured else None
    lines = [
        "# 最小镜像参考解的截断地板（程序化生成）",
        "",
        f"逐检查点比较 `min_image` 折返参考与 `lattice` 镜像求和参考，"
        f"用**与算例门禁相同的求积网格与加权相对 L2**（分母 = lattice 参考）。"
        f"结果目录：`{results_dir}`。",
        "",
        "| 检查点 | t | 像半宽 K | 密度地板 | u 地板 | v 地板 | p 地板 | "
        "实测最大误差 | 地板/误差 | 主导场 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        f = row["floor_percent"]
        if row["published_errors"] is None:
            tail = "（无实测记录） | - | - |"
        else:
            tail = (f"{row['published_max_percent']:.4f}% | "
                    f"{row['worst_floor_over_error']:.3f} | "
                    f"{row['worst_floor_over_error_field']} |")
        lines.append(
            f"| {row['task_key']} | {row['final_time']:g} | {row['images']} | "
            f"{f['density']:.3e}% | {f['velocity_x']:.3e}% | {f['velocity_y']:.3e}% | "
            f"{f['pressure']:.3e}% | " + tail)
    lines += ["", "## 判读", ""]
    if worst is not None:
        lines += [
            f"* 最不利检查点：`{worst['task_key']}`，地板/实测 = "
            f"{worst['worst_floor_over_error']:.3f}（{worst['worst_floor_over_error_field']}）。",
            "* 密度与压力场的地板在机器精度量级：自由空间等熵涡的密度扰动是包络的"
            "**二阶**量，邻居镜像尾巴贡献 `O(tail^2)`；速度扰动是包络的**一阶**量并"
            "乘以位移，故速度是唯一可能被参考解限住的场。",
            "* 因此「误差守恒」的判定在密度/压力上无参考解怀疑，在速度上存在一个"
            "可量化的上限（见上表最不利比值）：该比值 < 1 时实测误差仍由离散误差主导。",
        ]
    return "\n".join(lines) + "\n"


def make_figure(rows: list, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    measured = [r for r in rows if r["published_errors"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for field, style in (("velocity_x", "o-"), ("density", "s--")):
        xs = [r["final_time"] for r in rows]
        ys = [max(r["floor_relative_l2"][field], 1e-18) for r in rows]
        ax.semilogy(xs, ys, style, label=field, alpha=0.8)
    ax.set_xlabel("time t")
    ax.set_ylabel("min-image vs lattice reference (relative L2)")
    ax.set_title("truncation floor of the min-image reference")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    if measured:
        labels = [r["task_key"].replace("_cfl", "\ncfl") for r in measured]
        ratios = [r["worst_floor_over_error"] for r in measured]
        ax.bar(range(len(measured)), ratios, color="steelblue")
        ax.set_xticks(range(len(measured)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.axhline(1.0, color="crimson", ls="--", label="floor = measured error")
        ax.set_ylabel("floor / measured discretisation error")
        ax.set_title("is the gate reference-limited?")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="case_euler_vortex_errconserve")
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    results_dir = Path(args.results_dir)
    floors = [floor_at(dict(config)) for config in LADDER]
    rows = compare_with_published(floors, results_dir)
    payload = {
        "case": "EULER-VORTEX-ERR",
        "purpose": ("truncation floor of the min-image periodic reference, measured "
                    "in the case gate's own metric"),
        "reference_a": REFERENCE_MIN_IMAGE,
        "reference_b": REFERENCE_LATTICE,
        "period": list(PERIOD),
        "strength": STRENGTH,
        "background_velocity": list(BACKGROUND),
        "t_ref": T_REF,
        "checkpoints": rows,
        "worst": (min(rows, key=lambda r: -(r["worst_floor_over_error"] or 0.0))["task_key"]
                  if rows else None),
        "max_floor_over_error": max((r["worst_floor_over_error"] or 0.0) for r in rows),
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
    text = markdown(rows, results_dir)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(text)
        print(f"wrote {args.markdown}")
    if args.figures or args.figure:
        figure = Path(args.figure or (Path(args.output).parent / "fig_reference_floor.png")
                      if args.output else "fig_reference_floor.png")
        make_figure(rows, figure)
        print(f"wrote {figure}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())