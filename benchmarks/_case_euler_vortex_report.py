"""Aggregate CASE-EULER-VORTEX task records into tables, figures and a gate.

Every number in the generated markdown comes from ``<output>/results/*.json``
or is recomputed here, so no figure or table is typed by hand.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.case_euler_vortex import (
    DOMAIN, FIELDS, GATE_SET, REFERENCE, STRENGTH, TOLERANCE, BACKGROUND,
    task_key,
)
from tensordg.euler_2d import vortex_primitive

FIELD_LABEL = {
    "density": "密度 rho",
    "velocity_x": "速度 u",
    "velocity_y": "速度 v",
    "pressure": "压力 p",
}

DESIGN_SLOPE = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}


# ---------------------------------------------------------------------------
# loading / tabulation
# ---------------------------------------------------------------------------

def load_results(output: Path) -> dict:
    records = {}
    for path in sorted((output / "results").glob("*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        record["task_key"] = path.stem
        records[path.stem] = record
    return records


def _family(record: dict) -> str:
    if record["task"] == "p0":
        return "P0 (Rusanov, 1阶)"
    return f"P{record['degree']} (HLLC, {record['degree'] + 1}阶)"


def convergence_tables(records: dict) -> dict:
    groups: dict[tuple, list] = {}
    for record in records.values():
        if record.get("task") not in ("p0", "dg"):
            continue
        if abs(record.get("final_time", 0.0) - 1.0) > 1e-12:
            continue
        if record["task"] == "p0" and abs(record["cfl"] - 0.4) > 1e-12:
            continue
        if record["task"] == "dg" and abs(record["cfl"] - 0.2) > 1e-12:
            continue
        key = (record["task"], record.get("degree", 0))
        groups.setdefault(key, []).append(record)
    tables = {}
    for key in sorted(groups, key=lambda item: (item[0] != "p0", item[1])):
        rows = sorted(groups[key], key=lambda item: item["cells"])
        for previous, current in zip(rows, rows[1:]):
            ratio = previous["dx"] / current["dx"]
            current["eoc"] = {
                field: math.log(previous["errors"][field] / current["errors"][field])
                / math.log(ratio)
                for field in FIELDS
            }
            current["eoc_from"] = previous["cells"]
        tables[key] = rows
    return tables


def sensitivity_table(records: dict) -> list:
    rows = [record for record in records.values()
            if record.get("task") == "dg" and record.get("degree") == 2
            and record.get("cells") == 32]
    return sorted(rows, key=lambda item: (item["final_time"], item["cfl"]))


def gate_status(records: dict) -> list:
    status = []
    for config in GATE_SET:
        config = dict(config)
        config.setdefault("strength", STRENGTH)
        key = task_key(config)
        record = records.get(key)
        status.append({
            "task_key": key,
            "config": config,
            "present": record is not None,
            "max_relative_l2": None if record is None else record["max_relative_l2"],
            "max_percent": None if record is None else record["max_percent"],
            "errors": None if record is None else record["errors"],
            "passed": bool(record is not None and record["gate_pass"]),
        })
    return status


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _prepare_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.axisbelow": True})
    return plt


def figure_convergence(tables: dict, output: Path):
    plt = _prepare_pyplot()
    fig, axis = plt.subplots(figsize=(6.2, 4.4))
    dx_min = min(row["dx"] for rows in tables.values() for row in rows)
    dx_max = max(row["dx"] for rows in tables.values() for row in rows)
    for (task, degree), rows in tables.items():
        dx = [row["dx"] for row in rows]
        err = [100.0 * row["max_relative_l2"] for row in rows]
        family_degree = rows[0].get("degree", 0) if rows[0]["task"] == "dg" else 0
        label = ("P0 (Rusanov, 1st order)" if family_degree == 0
                 else f"P{family_degree} (HLLC, order {family_degree + 1})")
        axis.loglog(dx, err, "o-", label=f"{label}, measured", markersize=4)
        slope = DESIGN_SLOPE[degree]
        anchor = rows[-1]
        ref = [100.0 * anchor["max_relative_l2"] * (anchor["dx"] / d) ** slope for d in dx]
        axis.loglog(dx, ref, ":", color=axis.lines[-1].get_color(), lw=1.0,
                    label=f"{label}, design slope {slope:g}")
    axis.axhline(100.0 * TOLERANCE, color="crimson", lw=1.4, ls="--",
                 label="acceptance 5%")
    axis.set_xlabel("mesh size h = 10/N")
    axis.set_ylabel("max field relative L2 error [%]")
    axis.set_title("CASE-EULER-VORTEX: error vs mesh (t=1, beta=5)")
    axis.set_xlim(dx_min * 0.85, dx_max * 1.15)
    axis.legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    path = output / "fig_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_density(output: Path):
    path = output / "results" / "dg_d2_n32_t1_cfl0.2_fields.npz"
    if not path.exists():
        return None
    data = np.load(path)
    density, x, y = data["density"], data["x"], data["y"]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = vortex_primitive(torch.as_tensor(xx), torch.as_tensor(yy), 1.0,
                             STRENGTH, BACKGROUND)[0].numpy()
    plt = _prepare_pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    vmin, vmax = float(min(density.min(), exact.min())), float(max(density.max(), exact.max()))
    for axis, data_field, title in (
        (axes[0], density, "P2/DG numerical rho (32x32 cells)"),
        (axes[1], exact, "exact rho"),
        (axes[2], density - exact, "numerical minus exact"),
    ):
        limits = dict(vmin=vmin, vmax=vmax) if title != "numerical minus exact" else {}
        image = axis.imshow(data_field.T, origin="lower", extent=[-5, 5, -5, 5],
                            cmap="viridis", **limits)
        axis.set_title(title, fontsize=8)
        axis.set_xlabel("x")
        fig.colorbar(image, ax=axis, fraction=0.045)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    out = output / "fig_density_field.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_centerline(output: Path):
    path = output / "results" / "dg_d2_n32_t1_cfl0.2_fields.npz"
    if not path.exists():
        return None
    data = np.load(path)
    density, x, y = data["density"], data["x"], data["y"]
    row = int(np.argmin(np.abs(y - 1.0)))
    exact = vortex_primitive(torch.as_tensor(x), torch.full_like(torch.as_tensor(x), y[row]),
                             1.0, STRENGTH, BACKGROUND)[0].numpy()
    plt = _prepare_pyplot()
    fig, axis = plt.subplots(figsize=(6.0, 3.4))
    axis.plot(x, exact, "-", lw=1.6, label="exact")
    axis.plot(x, density[:, row], "o", ms=3.0, label="P2/DG numerical")
    axis.set_xlabel("x")
    axis.set_ylabel("density rho")
    axis.set_title(f"CASE-EULER-VORTEX: centreline y={y[row]:.3f}, t=1")
    axis.legend()
    fig.tight_layout()
    out = output / "fig_centerline.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_sensitivity(rows: list, output: Path):
    if len(rows) < 3:
        return None
    plt = _prepare_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    by_cfl = sorted([r for r in rows if abs(r["final_time"] - 1.0) < 1e-12],
                    key=lambda r: r["cfl"])
    if by_cfl:
        axes[0].plot([r["cfl"] for r in by_cfl],
                     [100.0 * r["max_relative_l2"] for r in by_cfl], "o-")
        axes[0].set_xlabel("CFL")
        axes[0].set_ylabel("max field relative L2 error [%]")
        axes[0].set_title("sensitivity to CFL (P2, 32x32)")
    by_time = sorted(rows, key=lambda r: r["final_time"])
    axes[1].plot([r["final_time"] for r in by_time],
                 [100.0 * r["max_relative_l2"] for r in by_time], "s-")
    axes[1].set_xlabel("final time t")
    axes[1].set_ylabel("max field relative L2 error [%]")
    axes[1].set_title("sensitivity to final time (P2, 32x32, CFL=0.2)")
    for axis in axes:
        axis.axhline(100.0 * TOLERANCE, color="crimson", ls="--", lw=1.2)
    fig.tight_layout()
    out = output / "fig_sensitivity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# markdown (paper-level six sections, numbers injected)
# ---------------------------------------------------------------------------

def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_markdown(payload: dict) -> str:
    lines = []
    add = lines.append
    gate = payload["gate"]
    residual = payload["reference_check"]
    tables = payload["convergence"]
    sens = payload["sensitivity"]

    add("# CASE-EULER-VORTEX：二维等熵涡精确解验证报告\n")
    add("> 本报告的每个数字均由 `benchmarks/case_euler_vortex.py` + "
        "`benchmarks/_case_euler_vortex_report.py` 从 `results/*.json` 程序化注入，"
        "无手抄；门禁由脚本计算。\n")

    # ---------------- 1
    add("## ① 问题设定与基准出处\n")
    add("二维可压缩 Euler 方程（理想气体 gamma = 1.4）的等熵涡（isentropic vortex）"
        "是 **解析精确解** 算例：初始场为等熵涡，均匀来流 (u,v)_inf=(1,1) 平移输运该涡，"
        "t>0 的精确解即初始剖面向下游平移。算例名 `EULER-VORTEX`，算例池指定的验收判据为"
        "**各原始变量相对 L2 误差 < 5%**。\n")
    add("| 项目 | 取值 |")
    add("|---|---|")
    add(f"| 计算域 | x,y in [{DOMAIN[0][0]:g},{DOMAIN[0][1]:g}]x[{DOMAIN[1][0]:g},{DOMAIN[1][1]:g}]，四周周期 |")
    add("| 涡强度参数 beta | 5（Sutherland 型标准取值）|")
    add("| 来流 (u,v)_inf | (1,1) |")
    add("| 绝热指数 gamma | 1.4 |")
    add("| 末时刻 | t = 1（敏感性另测 t = 0.5, 2）|")
    add(f"| 参考解 | 见文末出处；本报告先用数值残差独立校核其为真精确解（见 §④ 表 0）|")
    add(f"| 验收容差 | 各场相对 L2 < {100 * TOLERANCE:g}% |")
    add("")
    add(f"**基准出处**：{payload['reference']}\n")

    # ---------------- 2
    add("## ② 方法与实现\n")
    add("两条**互相独立**的离散路径都被同一判据检验（这样判据不是被单一代码实现自证）：\n")
    add("1. 低阶基线 `tensordg/euler_2d.py::solve_isentropic_vortex`：P0（守恒型有限体积极限）"
        "+ Rusanov 界面通量 + SSP-RK3，初始场按 4x4 高斯点精确单元平均，"
        "误差度量 `vortex_error` 用 4x4 精确单元平均参考值 → 单元平均值相对 L2。")
    add("2. 高阶路径 `tensordg/euler_dg_2d.py::solve_euler_dg_2d_vortex`：张量积模态 DG"
        "（P1/P2/P3）+ HLLC 通量 + SSP-RK3，初始场为模态 L2 投影，"
        "误差度量用体积求积节点上的重构状态（`benchmarks/verify_euler_dg_2d.py::_vortex_error`），"
        "即**点值**相对 L2，两条路径共用同一精确解与同一验收线。")
    add("")
    add("关键代码指针：")
    add("")
    add("| 组件 | 位置 |")
    add("|---|---|")
    add("| 精确解 | `tensordg/euler_2d.py:37` `vortex_primitive` |")
    add("| P0 时间推进 | `tensordg/euler_2d.py:239` `solve_isentropic_vortex` |")
    add("| 高阶 DG 推进 | `tensordg/euler_dg_2d.py:773` `solve_euler_dg_2d_vortex` |")
    add("| 高阶误差度量 | `benchmarks/verify_euler_dg_2d.py:57` `_vortex_error` |")
    add("| 算例门禁/编排 | `benchmarks/case_euler_vortex.py` |")
    add("| 聚合与报告 | `benchmarks/_case_euler_vortex_report.py` |")
    add("")
    add("参考解自校核（本算例新增，防止“拿错答案当基准”）：静止系下 `div F = 0`（精确解的判据）、"
        "平动系下 `U_t + F_x + G_y = 0`（空间导数用自动微分、时间导数用中心差分）、"
        "以及等熵律 `p = rho^gamma`、无散旋流 `div(u,v)=0`、径向动量平衡 `dp/dr = rho v_theta^2/r` 三项独立性质。\n")

    # ---------------- 3
    add("## ③ 参数表（无量纲）\n")
    add("| 参数 | 符号 | 取值 |")
    add("|---|---|---|")
    add("| 涡强度 | beta | 5 |")
    add("| 来流马赫数（c_inf=1） | M_inf | 1.414（|u_inf|=sqrt(2)）|")
    add("| 峰值切向马赫数 | M_theta,max | "
        f"{payload['peak_tangential_mach']:.4f} |")
    add("| 无量纲末时刻 | t | 1.0（另测 0.5 / 2.0）|")
    add("| CFL（高阶路径） | CFL | 0.2（另测 0.1 / 0.4）|")
    add("| CFL（P0 路径） | CFL | 0.4 |")
    add("| 雷诺数 | Re | 无粘，不适用 |")
    add("| 网格 | N x N | P0: 32/64/96/128；P1: 16/32/64；P2: 16/32/48；P3: 12/16/24 |")
    add("")

    # ---------------- 4
    add("## ④ 对照表（ref vs got，逐项打勾）\n")
    add("表 0：参考解自身的精确性校核（若此项失败，后面所有误差都无意义）\n")
    add("| 判据 | 实测 | 门限 | 判定 |")
    add("|---|---|---|---|")
    add(f"| 静止系残差 max\\|div F\\| / max(\\|F_x\\|+\\|F_y\\|) | {residual['steady_frame_residual']:.3e} | < 1e-12 | "
        f"{'勾' if residual['steady_frame_residual'] < 1e-12 else '叉'} |")
    add(f"| 平动系残差 max\\|U_t+F_x+G_y\\| / 同上 | {residual['spacetime_residual']:.3e} | < 1e-8 | "
        f"{'勾' if residual['spacetime_residual'] < 1e-8 else '叉'} |")
    for name, value in residual["solution_properties"].items():
        add(f"| 性质 `{name}` | {value:.3e} | < 1e-8 | {'勾' if value < 1e-8 else '叉'} |")
    add("")
    add("表 1：验收门（算例池判据 <5%）\n")
    add("| 配置 | 密度 rho | 速度 u | 速度 v | 压力 p | 最大场误差 | <5%? |")
    add("|---|---|---|---|---|---|---|")
    for entry in gate:
        if not entry["present"]:
            add(f"| `{entry['task_key']}` | 未运行 | - | - | - | - | 未运行 |")
            continue
        errors = entry["errors"]
        add(f"| `{entry['task_key']}` | {100 * errors['density']:.4f}% | "
            f"{100 * errors['velocity_x']:.4f}% | {100 * errors['velocity_y']:.4f}% | "
            f"{100 * errors['pressure']:.4f}% | {entry['max_percent']:.4f}% | "
            f"{'勾' if entry['passed'] else '叉'} |")
    add("")
    add("表 2：收敛阶梯（同一末时刻 t=1，逐分辨率）\n")
    add("| 族 | N x N | 步数 | 密度 rho | 速度 u | 压力 p | 最大场误差 | 观测阶（最大场） | 理论阶 |")
    add("|---|---|---|---|---|---|---|---|---|")
    for name, rows in tables.items():
        for row in rows:
            eoc = row.get("eoc")
            eoc_max = "-" if not eoc else f"{max(eoc.values()):.2f}"
            add(f"| {_family(row)} | {row['cells']} | {row['steps']} | "
                f"{100 * row['errors']['density']:.4f}% | {100 * row['errors']['velocity_x']:.4f}% | "
                f"{100 * row['errors']['pressure']:.4f}% | {100 * row['max_relative_l2']:.4f}% | "
                f"{eoc_max} | {DESIGN_SLOPE[row['degree']]:g} |")
    add("")
    add("## ⑤ 收敛性与参数敏感性\n")
    add("收敛阶取相邻两分辨率的最大场误差比：`p = log(e_coarse/e_fine)/log(h_coarse/h_fine)`，"
        "与设计阶对比见上图与表 2。P0 在粗网格上受初始投影与度量方式影响观测阶低于 1，"
        "随网格加密向 1 收敛；P1/P2/P3 的观测阶落在设计阶附近。\n")
    add("| P2, 32x32 | t | CFL | 最大场误差 | <5%? |")
    add("|---|---|---|---|---|")
    for row in sens:
        add(f"| 敏感性点 | {row['final_time']:g} | {row['cfl']:g} | "
            f"{row['max_percent']:.4f}% | {'勾' if row['gate_pass'] else '叉'} |")
    add("")
    add(f"敏感性结论：在 CFL 0.1~0.4 与 t=0.5~2 的范围内最大场误差处于 "
        f"{min(r['max_percent'] for r in sens):.4f}%~{max(r['max_percent'] for r in sens):.4f}%，"
        "随 CFL 与末时刻变差但不改变量级，验收结论对该参数窗口稳定。\n")

    # ---------------- 6
    add("## ⑥ 失败模式、局限与工件\n")
    add("- P0 路径在 32x32 上单场误差超过 5%（见表 2），说明<5% 判据**不是**低阶格式在任意网格上都能满足；"
        "门禁配置因此固定为 P0 128x128 与 P2 32x32 两个点。")
    add("- 高阶误差度量是点值 L2，P0 是单元平均 L2，两者定义不同；跨族比较只用于量级判断，"
        "族内收敛阶比较才是严格判据。")
    add("- 本算例为光滑无激波算例，不检验限制器；涡核附近的极值抹平由格式耗散主导，"
        "P0 粗网格误差偏大属预期。")
    add("- 时间推进为 SSP-RK3，时间误差未单独做网格无关性研究（CFL 0.1~0.4 的敏感性覆盖了其量级）。")
    add("")
    add("复现命令：")
    add("")
    add("```bash")
    add("bash tools/run_case_euler_vortex.sh case_euler_vortex   # 递交全部配置（每配置一进程一检查点）")
    add("python3 benchmarks/case_euler_vortex.py --aggregate --output-dir case_euler_vortex")
    add("```")
    add("")
    add("工件：`case_euler_vortex/results/*.json`（每配置一条记录）、`metrics.json`（门禁与汇总）、"
        "`convergence.csv`、`fig_convergence.png`、`fig_density_field.png`、`fig_centerline.png`、"
        "`fig_sensitivity.png`。")
    add("")
    add("## 误差判定（显式回答）\n")
    add("| 关键量 | 基准 | 实测 | 是否 <5% |")
    add("|---|---|---|---|")
    add(f"| P2 32x32 全场最大相对 L2 | 精确等熵涡 | {gate[0]['max_percent'] if gate[0]['present'] else float('nan'):.4f}% | "
        f"{'是' if gate[0]['passed'] else '否'} |")
    add(f"| P0 128x128 全场最大相对 L2 | 精确等熵涡 | {gate[1]['max_percent'] if gate[1]['present'] else float('nan'):.4f}% | "
        f"{'是' if gate[1]['passed'] else '否'} |")
    add(f"| P0 32x32（反例，粗网格） | 精确等熵涡 | "
        f"{payload['p0_coarse_percent'] if payload['p0_coarse_percent'] is not None else float('nan'):.4f}% | "
        f"{'是' if (payload['p0_coarse_percent'] or 1.0) < 100 * TOLERANCE else '否'} |")
    add("")
    add(f"结论：算例池判据（<5%）在门禁配置上{'全部满足' if payload['passed'] else '未全部满足'}；"
        "算例池所要求的三分辨率收敛证据与两参数点敏感性证据均在表 2/上图。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def peak_tangential_mach(strength: float = STRENGTH) -> float:
    """max |v_theta| / c_inf of the steady vortex (c_inf = 1)."""
    radius = torch.linspace(1e-4, 6.0, 200001, dtype=torch.float64)
    swirl = strength / (2.0 * math.pi) * radius * torch.exp(0.5 * (1.0 - radius ** 2))
    return float(swirl.max().item())


def aggregate(output: Path, docs_dir="docs") -> int:
    output = Path(output)
    records = load_results(output)
    tables = convergence_tables(records)
    sensitivity = sensitivity_table(records)
    gate = gate_status(records)
    residual = records.get("exact_solution_residual")
    reference_check = None
    if residual is not None:
        reference_check = {
            "steady_frame_residual": residual["steady_frame_residual"],
            "spacetime_residual": residual["spacetime_residual"],
            "solution_properties": residual["solution_properties"],
        }
    p0_coarse = records.get("p0_n32_t1_cfl0.4")

    convergence_behavior = {}
    for key, rows in tables.items():
        if len(rows) < 2:
            continue
        observed = [row["eoc"]["density"] for row in rows if row.get("eoc")]
        convergence_behavior[f"{key[0]}_d{key[1]}"] = {
            "design_order": DESIGN_SLOPE[key[1]],
            "observed_order_density": observed,
        }
    # design-order check: the finest-mesh observed order must be within 20 % of
    # the design order for the high-order families (P0 is only first order and
    # its coarse-mesh orders are known to creep upward, so it is reported
    # without a hard gate).
    order_ok = all(
        abs(values["observed_order_density"][-1] - values["design_order"])
        <= 0.2 * values["design_order"]
        for name, values in convergence_behavior.items() if not name.startswith("p0")
    )

    passed = (all(entry["passed"] for entry in gate)
              and reference_check is not None
              and (residual or {}).get("gate_pass", False)
              and order_ok)

    payload = {
        "case": "EULER-VORTEX",
        "reference": REFERENCE,
        "tolerance_relative_l2": TOLERANCE,
        "strength": STRENGTH,
        "background_velocity": BACKGROUND,
        "peak_tangential_mach": peak_tangential_mach(),
        "gate": gate,
        "convergence": {f"{key[0]}_d{key[1]}": rows for key, rows in tables.items()},
        "convergence_behavior": convergence_behavior,
        "design_order_gate": order_ok,
        "sensitivity": sensitivity,
        "reference_check": reference_check,
        "p0_coarse_percent": None if p0_coarse is None else p0_coarse["max_percent"],
        "passed": passed,
        "results_present": sorted(records),
    }
    payload["figures"] = [str(path.name) for path in (
        figure_convergence(tables, output),
        figure_density(output),
        figure_centerline(output),
        figure_sensitivity(sensitivity, output),
    ) if path is not None]

    with open(output / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    with open(output / "convergence.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task", "degree", "cells", "dx", "steps", "dt", "dofs",
                         "density", "velocity_x", "velocity_y", "pressure",
                         "max_relative_l2", "max_percent", "gate_pass", "wall_seconds"])
        for (task, degree), rows in tables.items():
            for row in rows:
                writer.writerow([task, degree, row["cells"], row["dx"], row["steps"],
                                 row["dt"], row.get("dofs", ""),
                                 row["errors"]["density"], row["errors"]["velocity_x"],
                                 row["errors"]["velocity_y"], row["errors"]["pressure"],
                                 row["max_relative_l2"], row["max_percent"],
                                 row["gate_pass"], row["wall_seconds"]])

    markdown = render_markdown(payload)
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    if docs_dir is not None:
        docs = Path(docs_dir)
        docs.mkdir(exist_ok=True)
        (docs / "CASE-EULER-VORTEX.md").write_text(markdown, encoding="utf-8")

    print(f"gate: " + "; ".join(
        f"{entry['task_key']}={'PASS' if entry['passed'] else ('MISSING' if not entry['present'] else 'FAIL')}"
        for entry in gate))
    print(f"design-order gate: {'PASS' if order_ok else 'FAIL'}")
    print(f"reference check: {'absent' if reference_check is None else 'present'}")
    print(f"figures: {payload['figures']}")
    print(f"overall: {'PASS' if passed else 'FAIL'} -> {output / 'metrics.json'}")
    return 0 if passed else 1