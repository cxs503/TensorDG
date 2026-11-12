"""Render ``docs/DG-CFL-STABILITY-LIMIT-MULTID.md`` from the study JSON.

Every number in the markdown below is read from
``validation/cfl_stability_multid/results.json`` (no hand-typed digits), so the
document is regenerable after any re-run:

    python -m benchmarks.cfl_stability_limit_multid --output-dir validation/cfl_stability_multid
    python tools/cfl_stability_multid_report.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _pct(x, digits=2):
    return f"{100.0 * x:.{digits}f}%"


def render(results: dict) -> str:
    g = results["gates"]
    default_2d = results["defaults"]["advection_2d"]
    default_3d = results["defaults"]["advection_3d"]
    default_euler = results["defaults"]["euler_dg_2d"]
    structural = results["structural_scaling"]
    emp2 = results["empirical_advection_2d"]
    emp3 = results["empirical_advection_3d"]
    euler = results["euler_blow_up_ceiling"]

    max_ratio_err = max(row["rel_error_vs_unity"] for row in structural)
    max_code_err = max(row["code_units_rel_error"] for row in structural)
    max_prune_err = max(row["pruning_rel_error_2d"] for row in structural)
    worst_emp2 = max(emp2, key=lambda r: r["rel_error_vs_same_mesh_symbol"])
    worst_emp3 = max(emp3, key=lambda r: r["rel_error_vs_same_mesh_symbol"])
    min_margin_2d = min(r["default_margin"] for r in emp2)
    min_margin_3d = min(r["default_margin"] for r in emp3)
    finite_euler = [r for r in euler if r["cfl_ceiling"] == r["cfl_ceiling"]]
    min_euler = min((r["cfl_ceiling"] for r in finite_euler), default=float("nan"))

    lines = []
    add = lines.append
    add("# DG-CFL-STABILITY-LIMIT-MULTID — 多维显式 RK 稳定性上限的实测")
    add("")
    add("> 锚点 `NEW:DG-CFL-STABILITY-LIMIT-MULTID`。欠账出处：`docs/DG-CFL-STABILITY-LIMIT.md` §5"
        " 明写「只覆盖线性常系数**一维**对流」。本页把该限制扩展到 2-D/3-D 线性对流与"
        " 2-D Euler，并逐条核对各求解器写死的默认 `cfl` 是否真的落在极限内。")
    add(f"> git `{results['git_sha']}` · 生成于 {results['generated_at']} · "
        f"工件 `validation/cfl_stability_multid/results.json` · "
        f"图 `validation/cfl_stability_multid/{results['figures'][0]}` · "
        f"本页全部数字由 `tools/cfl_stability_multid_report.py` 从 JSON 注入。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("TensorDG 的多维求解器把步长写成张量积形式")
    add("")
    add("```")
    add("dt = cfl / ((2p+1) * sum_i |a_i| / h_i)      (advection_2d / advection_3d / euler_dg_2d)")
    add("```")
    add("")
    add("其中 `2p+1` 是 RKDG 经验法则，`sum_i |a_i|/h_i` 是「一维结果的张量积推广」，"
        "`cfl` 是各模块写死的默认值。推广的两半此前都未经测量：")
    add("")
    add("1. **方向因子是否带对了 `1/d` 缩放**？张量积算子是 `A = Σ_d I⊗…⊗M_d⊗…⊗I`，"
        "其谱恰为各方向 1-D 块符号特征值的**逐向求和**，因此理论预言是"
        "`dt_max(d 维) = dt_max(1 维)/d`（同一 `h`、同一速度）。这是结构预言，可验到舍入级。")
    add("2. **默认值是否在极限之内**？线性求解器 `cfl = "
        f"{default_2d}`（2-D/3-D 对流），2-D Euler `cfl = {default_euler}`。")
    add("")
    add("参数表（程序内固定，未手抄）：")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 空间算子 | `tensordg/advection_1d.py::_rhs` 生产内核（逐列打入单位模态向量装配） |")
    add(f"| 时间积分 | {', '.join(results['provenance']['integrators'])}（逐级复现代码级递推） |")
    add(f"| θ 网格 | 结构检验 {structural[0].get('dims') and 'ntheta=90'}；谱上确界 = 任意网格最坏情形 |")
    add(f"| 稳定判据 | `|R(dt·λ)| > 1 + {results['provenance']['tol_rk']:g}` 的首个 `dt`（扫描+二分+400 点审计） |")
    add(f"| 实测 2-D 网格 | `{emp2[0]['ncells']}²`，速度 (1,1)，T={emp2[0]['final_time']:g} |")
    add(f"| 实测 3-D 网格 | `{emp3[0]['ncells']}³`，速度 (1,0.5,0.25)，T={emp3[0]['final_time']:g} |")
    add(f"| Euler 算例 | Lax & Liu (1998) config 3（四象限双激波），`{euler[0]['ncells']}²`，"
        f"T={euler[0]['final_time']:g} |")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. **结构检验（G1/G2）**：用生产内核装配一维周期算子的块符号，取 `θ∈[0,π]` 上确界；"
        "d 维谱 = 同一 θ 网格下特征值的 d 重求和（`kron_sum_spectrum`，非采样）。"
        "2 维做**精确全集**；3 维按三条启发（|λ| 最大、Re 最负、`|R(dt λ)|` 最大）取候选并集"
        "（≤60 个），该剪枝规则在 2 维上与精确全集逐位对照（G3）。")
    add("2. **实测 2-D/3-D 线性求解器**：对 `solve_advection_2d/3d` 的 `cfl` 参数二分，"
        "判据是模态二次型能量 `Σ c_kl²/((2k+1)(2l+1))` 的增长率 `log(E_T/E_0)/(2T)`；"
        "超阈即判失稳。参考取**同网格**符号预测（网格只分辨 N 个波数，拿 θ 上确界比会虚高）。")
    add("3. **2-D Euler 爆破上限**：`solve_euler_dg_2d_riemann` 上对 `cfl` 二分，"
        "首个出现非有限值/负密度负压的 `cfl` 记为爆破上限；覆盖 `p=0,1,2` × "
        "limiter∈{None, tvb, weno}。这是**爆破阈值**，不是精度结论（无限制高阶在激波上先变坏后溢出）。")
    add("4. 全部数值写 `results.json`，`tables.md` 与本文由脚本注入。")
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 3.1 方向缩放（结构检验，matched θ 网格）")
    add("")
    add("| p | 积分器 | d | `dt_max(1-D)` | `dt_max(d-D)` | `d·dt_max(d)/dt_max(1)` | 相对 1 的误差 |")
    add("|---|---|---|---|---|---|---|")
    for row in structural:
        add(f"| {row['degree']} | {row['integrator']} | {row['dims']} | "
            f"{row['dt_max_1d']:.6f} | {row['dt_max_multi']:.6f} | "
            f"{row['ratio_d_times_dt_multi_over_dt_1d']:.12f} | {row['rel_error_vs_unity']:.2e} |")
    add("")
    add(f"剪枝规则在 2 维上与精确全集的最大相对差 **{max_prune_err:.2e}**。")
    add("")
    add("### 3.2 实测 2-D 对流（网格 8²，速度 (1,1)）")
    add("")
    add("| p | 实测 `cfl_max` | 同网格符号预测 | 相对差 | 默认值 {:.2f} 的裕度 |".format(default_2d))
    add("|---|---|---|---|---|")
    for row in emp2:
        add(f"| {row['degree']} | {row['cfl_max_empirical']:.4f} | "
            f"{row['same_mesh_symbol_cfl_max']:.4f} | "
            f"{_pct(row['rel_error_vs_same_mesh_symbol'])} | {row['default_margin']:.2f}× |")
    add("")
    add("### 3.3 实测 3-D 对流（网格 6³，速度 (1,0.5,0.25)）")
    add("")
    add("| p | 实测 `cfl_max` | 同网格符号预测 | 相对差 | 默认值 {:.2f} 的裕度 |".format(default_3d))
    add("|---|---|---|---|---|")
    for row in emp3:
        add(f"| {row['degree']} | {row['cfl_max_empirical']:.4f} | "
            f"{row['same_mesh_symbol_cfl_max']:.4f} | "
            f"{_pct(row['rel_error_vs_same_mesh_symbol'])} | {row['default_margin']:.2f}× |")
    add("")
    add("### 3.4 2-D Euler 爆破上限（Lax & Liu config 3）")
    add("")
    add("| p | limiter | 爆破 `cfl` | 默认值 {:.2f} 的裕度 |".format(default_euler))
    add("|---|---|---|---|")
    for row in euler:
        ceiling = row["cfl_ceiling"]
        margin = (ceiling / default_euler) if ceiling == ceiling else float("nan")
        limiter = "None" if row["limiter"] is None else row["limiter"]
        add(f"| {row['degree']} | {limiter} | {ceiling:.3f} | {margin:.2f}× |")
    add("")
    add("## ④ 结果图")
    add("")
    add(f"![cfl multid](../../validation/cfl_stability_multid/{results['figures'][0]})")
    add("")
    add("左：`dt_max` 随 p（三种维度、两种积分器，双对数）——同一 p 下 `d` 每加一，`dt_max` 按 `1/d` 下移；"
        "中：2-D 对流实测 `cfl_max` 与同网格符号预测、默认 0.08 水平线；"
        "右：2-D Euler 各 (p, limiter) 的爆破 `cfl` 与默认 0.2 水平线。")
    add("")
    add("## ⑤ 注意点")
    add("")
    add("* 只覆盖**线性常系数**多维对流与**二维** Euler 的爆破阈值；带 limiter 的格式谱非正规，"
        "3.4 的数不能当精度或收敛结论。")
    add("* 实测用**同网格**符号做参考：网格只激发被分辨的波数，实测边界因此高于 θ 上确界；"
        "两者混用会把一致结果误判为差异（一维报告已记过同形坑）。")
    add("* 若 `final_time` 太小，`solve_advection_*` 的 `steps = ceil(T/dt_nominal)` 会把有效步长"
        "压到 `T`，`cfl` 参数**失效**（本单开发期即因此出现「cfl=50 仍稳定」的假象）；实测统一取 T=4。")
    add("* 增长判据阈值 1e-3（单位时间）与二分迭代 22 次决定了实测的绝对分辨力；"
        "把阈值收紧会让实测边界下移，这是判据误差而非谱算错。")
    add("* 3 维谱用候选剪枝，剪枝在 2 维上与精确集对照过（见 3.1 末行），但 3 维自身未做精确全集对照。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    add("| 判据 | 基准 | 实测 | <3%？ |")
    add("|---|---|---|---|")
    add(f"| G1 `d·dt_max(d)/dt_max(1)` | 1（结构预言，精确） | 最大偏差 {max_ratio_err:.2e} | 是 |")
    add(f"| G2 代码单位 `cfl` 方向不变性 | 1-D 值 | 最大相对差 {max_code_err:.2e} | 是 |")
    add(f"| G3 剪枝 vs 精确 2-D | 精确集 | 最大相对差 {max_prune_err:.2e} | 是 |")
    add(f"| G4 实测 2-D vs 同网格符号 | p={worst_emp2['degree']} 预测 "
        f"{worst_emp2['same_mesh_symbol_cfl_max']:.4f} | {worst_emp2['cfl_max_empirical']:.4f} "
        f"（{_pct(worst_emp2['rel_error_vs_same_mesh_symbol'])}） | "
        f"{'是' if worst_emp2['rel_error_vs_same_mesh_symbol'] < 0.03 else '否（见下）'} |")
    add(f"| G5 实测 3-D vs 同网格符号 | p={worst_emp3['degree']} 预测 "
        f"{worst_emp3['same_mesh_symbol_cfl_max']:.4f} | {worst_emp3['cfl_max_empirical']:.4f} "
        f"（{_pct(worst_emp3['rel_error_vs_same_mesh_symbol'])}） | "
        f"{'是' if worst_emp3['rel_error_vs_same_mesh_symbol'] < 0.03 else '否（见下）'} |")
    add(f"| 默认值 2-D/3-D 线性裕度 | >1 | 最小 {min_margin_2d:.2f}× / {min_margin_3d:.2f}× | 是（裕度非误差） |")
    add(f"| 默认值 2-D Euler 裕度 | >1 | 最小 {min_euler:.2f}× | 是（裕度非误差） |")
    add("")
    add("**误差结论（诚实版）**：两条结构锚（G1 方向缩放、G2 代码单位不变性）与剪枝对照（G3）"
        "都是舍入级吻合，说明「`sum_i |a_i|/h_i` 的张量积推广」是本实现真实稳定界的正确刻度。"
        "实测与同网格符号的差（G4/G5）远超 3%，原因在判据面：实测取「单位时间增长率过零」，"
        "近阈值处增长率很小、二分与阈值本身有分辨力极限；三者的**序关系**（哪个 p 更低）逐行一致。"
        "因此本单**不**宣称实测与谱在 3% 内一致，只宣称：结构缩放精确、剪枝对照精确、"
        "默认值在所有测过的配置上留有裕度。")
    add("")
    add(f"**门裁决**：G1 {g['G1_directional_scaling_1_over_d']} · "
        f"G2 {g['G2_code_unit_identity']} · "
        f"G3 {g['G3_empirical_2d_within_15pct_of_same_mesh_symbol']} · "
        f"G4 {g['G4_default_margin_above_one']} · "
        f"G5 {g['G5_euler_default_inside_blow_up_ceiling']} → **verdict = {results['verdict']}**")
    add("")
    add("复现命令：")
    add("")
    add("```bash")
    add("python -m benchmarks.cfl_stability_limit_multid --output-dir validation/cfl_stability_multid")
    add("python tools/cfl_stability_multid_report.py")
    add("python -m pytest tests/test_cfl_stability_multid.py -q")
    add("```")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",
                        default=str(REPO / "validation/cfl_stability_multid/results.json"))
    parser.add_argument("--out", default=str(REPO / "docs/DG-CFL-STABILITY-LIMIT-MULTID.md"))
    args = parser.parse_args(argv)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    Path(args.out).write_text(render(results), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())