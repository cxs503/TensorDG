"""One-off patch: data-driven entropy report (finest-mesh gate, new profile metric).

Kept for provenance; run ``python3 tools/patch_entropy_report.py``.
"""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "benchmarks/report_entropy_budget.py"
text = PATH.read_text(encoding="utf-8")


def replace(old: str, new: str) -> None:
    global text
    if new in text:
        print("already applied")
        return
    if old not in text:
        raise SystemExit(f"anchor missing:\n{old[:160]}")
    text = text.replace(old, new, 1)


# --------------------------------------------------------------------------- #
# table rows carry a ``label`` that encodes the limiter stage
# --------------------------------------------------------------------------- #
replace(
    '''def _grid_table(rows, analytic):
    lines = [
        "| limiter | cells | steps | $\\\\Delta\\\\int\\\\rho s\\\\,dx$ | analytic $P\\\\,T$ | relative error | second law |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['limiter']} | {row['cells']} | {row['steps']} "''',
    '''def _grid_table(rows, analytic):
    lines = [
        "| limiter (stage) | cells | steps | $\\\\Delta\\\\int\\\\rho s\\\\,dx$ | analytic $P\\\\,T$ | relative error | second law |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('label', row['limiter'])} | {row['cells']} | {row['steps']} "''',
)

replace(
    '''    limiter_cell = ", ".join(limiter_names) + ' (tvb_constant = 1.0, limiter_stage = "every")'
''',
    '''    labels = sorted({row.get("label", row["limiter"]) for row in sod["rows"]})
    stages = sorted({row.get("limiter_stage", "every") for row in sod["rows"]})
    limiter_cell = ", ".join(labels) + " (tvb_constant = 1.0, limiter_stage = " + \\
        "/".join(f'"{stage}"' for stage in stages) + ")"
''',
)

# --------------------------------------------------------------------------- #
# C2 table: coarsest AND finest mesh + measured order of the budget error
# --------------------------------------------------------------------------- #
replace(
    '''    c2_rows = verdicts["C2_relative_error_at_finest_mesh"]
    c2_ok = verdicts["C2_below_3_percent"]
    c2_lines = ["| run | relative error at finest mesh | < 3 %? |", "|---|---:|---|"]
    for key in sorted(c2_rows):
        c2_lines.append(f"| {key} | {100.0 * c2_rows[key]:.3f}% | {'yes' if c2_ok[key] else '**no**'} |")
''',
    '''    c2_rows = verdicts["C2_relative_error_at_finest_mesh"]
    c2_coarse = verdicts.get("C2_relative_error_at_coarsest_mesh", {})
    c2_orders = verdicts.get("C2_error_observed_orders", {})
    c2_ok = verdicts["C2_below_3_percent"]
    c2_lines = [
        "| run | rel. error @ coarsest mesh | rel. error @ **finest** mesh | observed order (last pair) | < 3 %? |",
        "|---|---:|---:|---:|---|",
    ]
    for key in sorted(c2_rows):
        order = c2_orders.get(key)
        order_text = "n/a" if not order or order[-1] is None else f"{order[-1]:.2f}"
        coarse_text = f"{100.0 * c2_coarse[key]:.3f}%" if key in c2_coarse else "n/a"
        c2_lines.append(
            f"| {key} | {coarse_text} | {100.0 * c2_rows[key]:.3f}% | {order_text} "
            f"| {'yes' if c2_ok[key] else '**no**'} |")
    finest_ok = sorted(key for key, ok in c2_ok.items() if ok)
    failing = sorted(key for key, ok in c2_ok.items() if not ok)
''',
)

# --------------------------------------------------------------------------- #
# profile table: interpretable metric instead of ||e||/||rho s||
# --------------------------------------------------------------------------- #
replace(
    '''    profile_lines = ["| limiter | relative $L^2$ error of $\\\\rho s(x)$ |", "|---|---:|"]
    for key, value in profile["relative_l2_entropy_density"].items():
        profile_lines.append(f"| {key} | {100.0 * value:.3f}% |")
''',
    '''    profile_metrics = profile.get("metrics", {})
    jump_scale = profile.get("entropy_jump_scale")
    profile_lines = [
        "| limiter | RMS error of $\\\\rho s$ | RMS / $|[s]|$ | deprecated $\\\\|e\\\\|_2 / \\\\|\\\\rho s\\\\|_2$ |",
        "|---|---:|---:|---:|",
    ]
    for key, value in profile_metrics.items():
        profile_lines.append(
            f"| {key} | {value['rms']:.4e} | {value['rms_over_entropy_jump']:.4f} "
            f"| {100.0 * value['legacy_relative_l2']:.1f}% |")
''',
)

# --------------------------------------------------------------------------- #
# limitations: stage coverage is now complete, and the gate is the finest mesh
# --------------------------------------------------------------------------- #
replace(
    '''5. 限制器设置固定为 `tvb_constant = 1.0`、`limiter_stage = "every"`（仓库历史默认）；
   `"final"` 分支的熵收支未在本轮覆盖。
6. 运行环境为多席并发的高负载机（load15 > 120），因此研究脚本默认 `torch.set_num_threads(1)`
   并用 CFL = {cfl} 控制步数；矩阵规模很小，结论与线程数无关。''',
    '''5. 限制器设置固定为 `tvb_constant = 1.0`；`limiter_stage` 的**两个分支**
   （`"every"` 与 `"final"`）都已纳入熵收支（表中 `tvb@final` 行），
   这补上了旧版明写的"未覆盖"缺口。
6. **门判定以最细网格为准**（`finest_row`）。旧版 `C2_relative_error_at_finest_mesh`
   用 `min(rows, key=cells)` 取了**最粗**网格，Sod/none 因此被记成 11.651 %（64 cells），
   而 256 cells 的真实值是 3.126 %——旧版的 honest-FAIL 有一部分是**标签缺陷**而非求解器结果。
   该缺陷已修，并由 `tests/test_entropy_budget_gate.py` 钉住。
7. 运行环境为多席并发的高负载机（load15 > 120），因此研究脚本默认 `torch.set_num_threads(1)`
   并用 CFL = {cfl} 控制步数；矩阵规模很小，结论与线程数无关。''',
)

# --------------------------------------------------------------------------- #
# section 7: conclusions computed from the JSON, never typed by hand
# --------------------------------------------------------------------------- #
start = text.index('## ⑦ 结论与建议（本轮 honest 判定）')
end = text.index('## 复现')

new_section = '''## ⑦ 结论与建议（本轮 honest 判定）

{chr(10).join(conclusion_lines)}

'''

text = text[:start] + new_section + text[end:]

# build the conclusion lines just before the f-string is returned
replace(
    '''    ladder = selftest["entropy_integral_ladder"]

    return f"""''',
    '''    ladder = selftest["entropy_integral_ladder"]

    def _trend(rows):
        rows = sorted(rows, key=lambda row: row["cells"])
        return " → ".join(f"{100.0 * row['relative_error']:.3f}%" for row in rows)

    conclusion_lines = [
        f"1. **第二定律（C1）成立**：{len(sod['rows']) + len(lax['rows'])} 次激波运行"
        "（含 `tvb` 与 `tvb@final`）全部 $\\\\Delta\\\\int\\\\rho s\\\\,dx > 0$，"
        "格式没有做出\\"熵减少\\"这一违反物理的解。",
        f"2. **C2 定量收支随网格收敛**：Sod/none 的相对误差 {_trend([r for r in sod['rows'] if r['label'] == 'none'])}"
        f"，观测阶 {c2_orders.get('sod/none', [None])[-1]:.2f}；"
        f"Sod/tvb 的 {_trend([r for r in sod['rows'] if r['label'] == 'tvb'])}"
        f"（观测阶 {c2_orders.get('sod/tvb', [None])[-1]:.2f}）。"
        f"3 % 门在最细网格上{'**通过**：' + ', '.join(finest_ok) if finest_ok else '**仍未通过**'}；"
        f"{'未通过：' + ', '.join(failing) if failing else '无未通过项'}。",
        f"3. **限制器阶段分支**：`tvb@final`（只在 RK 末段限幅）在 Sod 上的相对误差 "
        f"{_trend([r for r in sod['rows'] if r['label'] == 'tvb@final'])}，"
        f"与 `tvb`（每个子步都限幅）相比{'更小' if min(r['relative_error'] for r in sod['rows'] if r['label'] == 'tvb@final') < min(r['relative_error'] for r in sod['rows'] if r['label'] == 'tvb') else '并不更小'}："
        "限幅次数越少，人为耗散产生的熵越少，但激波稳定性由 C1 单独把关。",
        "4. **两条诚实结论**：①熵产生 = 物理熵产生 + 数值耗散，粗网格上后者远大于前者，"
        "所以该判据在粗网格上只能当\\"上界检查\\"，定量门必须看收敛后的最细网格；"
        "②熵密度剖面的相对 $L^2$ 误差不是可解释的量（$\\\\rho s$ 变号且 $s=\\\\ln(p/\\\\rho^\\\\gamma)$ 在 $p\\\\to0$ 时发散），"
        "本轮改用 RMS 误差 / 激波熵跃 $|[s]|$ 的归一化度量。",
        "5. **建议（下一位接单者）**：Lax 强激波（$\\\\sigma \\\\approx 2.48$，误差仍在数百 %）"
        "需要更高阶格式或更细网格才能谈定量门；`weno` 限制器的熵收支与二维 $\\\\rho s$ 收支仍未覆盖。",
    ]

    return f"""''',
)

PATH.write_text(text, encoding="utf-8")
print("report generator patched")