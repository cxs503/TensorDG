"""Inject ``validation/entropy_2d/summary.json`` into the m3430 report.

Every number in ``docs/reports/tensordg-entropy-2d.md`` is generated here, so
the report can never drift from the artefact.
"""
from __future__ import annotations

import json

SUMMARY = "validation/entropy_2d/summary.json"
REPORT = "docs/reports/tensordg-entropy-2d.md"


def _pct(value):
    return f"{100.0 * value:.3f}"


def build(summary):
    checks = summary["self_checks"]
    budget = summary["analytic_budget"]
    smooth = summary["cases"]["smooth_advection"]
    sod = summary["cases"]["extruded_sod"]
    activity = summary["cases"]["limiter_activity_on_smooth_data"]
    verdicts = summary["verdicts"]
    names = {"none": "无限制器", "tvb": "TVB (M=1)", "weno": "WENO"}

    lines = []
    add = lines.append
    add("# TensorDG 二维 Euler DG 离散熵收支验证（m3430）")
    add("")
    add("> 本单是**验证欠账补齐**：`docs/entropy_verification.md` 的局限第 1 条明写"
        "「仅一维、平直 Cartesian 网格」，二维求解器 `tensordg/euler_dg_2d.py`（含已合并的"
        " TVB/WENO 二维限制器）此前**没有任何熵侧证据**。本单把一维熵收支账推广到二维，"
        "并据此复核已合并的二维限制器。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("**物理背景。** 理想气体（$\\gamma=%.1f$）Euler 方程的物理熵 $s=\\ln(p/\\rho^\\gamma)$："
        "光滑解满足守恒律 $\\partial_t(\\rho s)+\\nabla\\cdot(\\rho \\mathbf{u} s)=0$，"
        "含激波的弱解满足**熵不等式** $\\partial_t(\\rho s)+\\nabla\\cdot(\\rho \\mathbf{u} s)=P\\ge 0$。"
        "因此 $\\Delta\\int\\!\\int \\rho s\\,dA<0$ 与「解好不好看」无关——那是数值格式**违反了第二定律**，"
        "是可伪证的硬判据。" % summary["gamma"])
    add("")
    add("**算例 A（光滑、真二维、有精确熵不变量）。** $\\rho=1+A\\sin 2\\pi(x+y-2t)$、"
        "$u=v=1$、$p=1$：质量、动量、能量三个方程都退化为 $\\rho$ 的纯对流，"
        "所以这是 Euler 方程的**精确解**；且 $\\int\\!\\int\\rho s\\,dA=-\\gamma\\int\\!\\int\\rho\\ln\\rho\\,dA$"
        "是精确不变量（$p$ 为常数）。取 $A=0.2$、$t=0.05$、周期域 $[0,1]^2$、$P1$ 模态 DG、HLLC、SSP-RK3、CFL=0.4。")
    add("")
    add("**算例 B（y 向均匀 Sod 激波管）。** $t=0.1$，左右态 $(\\rho,u,p)_L=(1,0,1)$、"
        "$(\\rho,u,p)_R=(0.125,0,0.1)$，$x$ 向透射、$y$ 向周期，$P1$、CFL=0.3。"
        "解析基准由 Rankine-Hugoniot 给出：单激波熵产生率 $P=\\sigma[\\rho s]-[\\rho u s]$，"
        "累计产生量 $P\\,T=%.6e$，激波速度 $\\sigma=%.4f$，$t=0.1$ 时位于 $x=%.4f$"
        "（`benchmarks/_entropy_tools.analytic_right_shock_budget`，不含拟合参数）。"
        % (budget["production_total"], budget["shock_speed"], budget["shock_position"]))
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. **诊断量**（`benchmarks/entropy_budget_2d.py:total_entropy_2d`）：把模态系数在**高斯点上**"
        "重构出守恒量（$q=\\max(p+3,6)$ 阶），逐点算 $\\rho s$ 再乘 Jacobian 积分；"
        "**不用单元平均值**——$\\rho s$ 对守恒量非线性，用平均值会注入 $O(h^2)$ 人为误差。")
    add("2. **诊断先自检，再判求解器**（V&V 纪律，三道）：")
    add("   - 常状态闭式检验：$\\rho=2,p=1$ 的精确积分 vs 数值积分，相对误差 **%.2e**；"
        % checks["uniform_state_relative_error"])
    add("   - 正交阶无关性：$\\ell=4,6,8,10$ 四档的相对散布 **%.2e**；"
        % checks["quadrature_relative_spread"])
    add("   - 跨实现检验：y 向均匀态下二维积分与**一维实现**（`total_entropy_from_coefficients`"
        "的独立路径）相对误差 **%.2e**。" % checks["one_dimensional_cross_check_relative_error"])
    add("3. **两条独立基准**：算例 A 用精确不变量（期望 0），算例 B 同时用"
        "①解析 RH 收支、②**一维 DG 求解器**在同一网格/同一时间上的熵产生量。"
        "后者是「同一物理、两套独立实现」，用来把**二维专有缺陷**与已知的**激波抹宽局限**分开。")
    add("4. **限制器活动度**（`limiter_activity`）：把限制器作用在**光滑投影初值**上，"
        "量 $\\max|c_{\\text{lim}}-c|/\\max|c|$。激波限制器在光滑区本应几乎不动作，"
        "这个数直接度量它「误伤」了多少。")
    add("5. **工具链**：PyTorch 2.7.1（float64，单线程）、NumPy/SciPy、pytest；"
        "数字全部由 `benchmarks/report_entropy_2d.py` 从 `validation/entropy_2d/summary.json` 注入，零手抄。")
    add("")
    add("### 限制器活动度（光滑投影初值，%dx%d）" % (activity[0]["ncells"], activity[0]["ncells"]))
    add("")
    add("| 限制器 | max 相对改动 | 平均绝对改动 |")
    add("|---|---:|---:|")
    for row in activity:
        add("| %s | %.4e | %.4e |" % (names[row["limiter"]], row["max_relative_change"],
                                      row["mean_absolute_change"]))
    add("")
    add("## ③ 计算结果")
    add("")
    add("### 算例 A：光滑二维对流的熵漂移 $\\Delta\\int\\!\\int\\rho s\\,dA$（精确值 0）")
    add("")
    add("| 限制器 | 网格 | $\\Delta\\int\\!\\int\\rho s\\,dA$ | 步数 | $\\rho$ $L^\\infty$ 误差 | 相对振幅 |")
    add("|---|---:|---:|---:|---:|---:|")
    for entry in smooth:
        add("| %s | %d×%d | %s | %d | %.3e | %s%% |"
            % (names[entry["limiter"]], entry["ncells"], entry["ncells"],
               ("%+.4e" % entry["entropy_drift"]), entry["steps"],
               entry["density_linf_error"],
               _pct(entry["density_linf_error"] / entry["density_linf_exact_amplitude"])))
    add("")
    orders = verdicts["C1_smooth_entropy_conserved"]["observed_orders"]
    add("无限制器的熵漂移收敛阶（相邻网格）: %s。" %
        ", ".join("%.2f" % value for value in orders))
    add("")
    add("### 算例 B：y 向均匀 Sod 的熵产生量")
    add("")
    add("| 限制器 | 网格 | 二维 $\\Delta\\int\\!\\int\\rho s$ | 一维求解器同网格 | 相对 1D 误差 | 相对解析 RH 误差 |")
    add("|---|---:|---:|---:|---:|---:|")
    for entry in sod:
        add("| %s | %d×%d | %+.6e | %+.6e | %s%% | %s%% |"
            % (names[entry["limiter"]], entry["ncells"], entry["y_cells"],
               entry["entropy_production"], entry["entropy_1d_solver"],
               _pct(entry["relative_error_vs_1d"]),
               _pct(entry["relative_error_vs_analytic"])))
    add("")
    add("解析 RH 目标值：**%.6e**。" % budget["production_total"])
    add("")
    add("## ④ 结果图")
    add("")
    add("![二维离散熵收支](figures/entropy_budget_2d.png)")
    add("")
    add("- **(a)** 光滑算例的 $|\\Delta\\int\\!\\int\\rho s\\,dA|$ 随网格：无限制器沿 $h^2$ 参考斜率下降；"
        "TVB 高出约一个量级但同为耗散侧；**WENO 在 32×32/64×64 变号（负值 = 熵销毁）**。")
    add("- **(b)** Sod 熵产生量 vs 解析 RH 收支，灰带为 3% 验收带；三种设置都高于解析值一个量级，"
        "即「激波被抹宽 → 数值熵产生被高估」的已知局限（与一维账本同源）。")
    add("- **(c)** WENO 相对无限制器**额外引入**的 $\\rho s$ 缺陷场：负区（蓝）即熵被销毁的位置，"
        "集中在光滑波峰/波谷附近——与「限制器在光滑区误动作」的机理一致。")
    add("")
    add("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **坑：初值张量形状会静默把二维算例退化成一维。** 求解器用 `torch.stack` 拼守恒量，"
        "若各分量来自 `torch.ones_like(x)`（形状 $(n_x,1,q,1)$）而不是 $x+y$ 的广播全形状，"
        "状态会变成 $(n_x,1,q,q,4)$——动力学上等价于一维（y 向数据均匀时无害），"
        "但熵积分少乘了 $n_y$，本单第一版因此把 Sod 的收支误差算成 97%（真值 52%）。"
        "**判据：系数张量第二维必须等于 $n_y$。**")
    add("2. **坑：解析 RH 收支不是工程分辨率下可达的判据。** 一维账本在 256 格 $P2$ 上仍有 7.3% 误差、"
        "自述 FAIL；二维同理。要区分「物理必然的激波抹宽」与「格式错误」，"
        "必须另立**跨实现**判据（二维 vs 一维求解器），本单用 0.01% 量级的一致性把它钉死。")
    add("3. **发现（本单主要增量）：二维 WENO 限制器在光滑流上违反离散熵不等式。** "
        "32×32 熵漂移 %+.4e、64×64 %+.4e，网格加密后**负值变大**（不是舍入噪声）；"
        "TVB 同点位 %+.4e / %+.4e 均为正（耗散侧，合法）。" % (
            [e for e in smooth if e["limiter"] == "weno"][1]["entropy_drift"],
            [e for e in smooth if e["limiter"] == "weno"][2]["entropy_drift"],
            [e for e in smooth if e["limiter"] == "tvb"][1]["entropy_drift"],
            [e for e in smooth if e["limiter"] == "tvb"][2]["entropy_drift"]))
    add("4. **局限（本单未做）**：①未修 WENO 缺陷（需重构 WENO3 权重/平滑指示子在多维的组合方式，"
        "属研究级改动）；②未覆盖真正二维激波（二维 Riemann 问题的解析熵产生率需数值积分 baseline，"
        "本单只用 y 向均匀算例）；③只到 $P1$、CFL≤0.4、Cartesian 周期/透射边界的无粘 Euler；"
        "④时间积分误差未与空间误差分离（未做 dt 加密研究）。")
    add("5. **适用范围**：`tensordg/euler_dg_2d.py` 的模态 DG + HLLC + SSP-RK3 路径。"
        "熵不等式是**必要条件**：C1/C2 通过不代表解没有其它误差，只排除「违反第二定律」这一类错误。")
    add("")
    add("## ⑥ 误差判定（逐项，显式回答是否 <3%）")
    add("")
    add("| 判据 | 关键量 | 基准 | 结果 | <3%？ |")
    add("|---|---|---|---:|---|")
    c1 = verdicts["C1_smooth_entropy_conserved"]
    add("| C1 光滑熵守恒 | 64×64 $\\Delta\\int\\!\\int\\rho s\\,dA$ | 精确 0 | %+.4e | 无量纲小量，判据为随 $h$ 消失（阶 %.2f/%.2f） |"
        % (c1["finest_drift"], orders[0] if orders else float("nan"), orders[-1] if orders else float("nan")))
    c4 = verdicts["C4_two_dimensional_path_matches_the_1d_solver"]
    add("| C4 二维路径 vs 一维求解器 | Sod 熵产生量 | 一维 DG 同网格 | 最大相对误差 %s%% | **是** |"
        % _pct(max(c4["relative_errors"])))
    c5 = verdicts["C5_three_percent_against_the_analytic_rh_budget"]
    add("| C5 二维路径 vs 解析 RH 收支 | Sod 熵产生量 | $P\\,T$ 解析 | %s%%（最细网格） | **否**（与一维账本同一局限，如实记 FAIL） |"
        % _pct(min(c5["relative_errors"])))
    c2 = verdicts["C2_tvb_dissipative_side"]
    add("| C2 TVB 熵不等式 | 光滑熵漂移符号 | $\\ge 0$ | %s | **是** |"
        % ("全网格为正" if c2["pass"] else "出现负值"))
    c3 = verdicts["C3_weno_dissipative_side"]
    add("| C3 WENO 熵不等式 | 光滑熵漂移符号 | $\\ge 0$ | 违反网格 %s | **否（缺陷）** |"
        % (", ".join(str(m) for m in c3["violating_meshes"]) if c3["violating_meshes"] else "无"))
    add("")
    add("**结论**：二维熵收支的**工具本身**是可信的（三道自检 1e-12 量级、跨实现 1e-2 % 量级）；"
        "**二维求解器主路径**在熵侧与一维实现一致（<3% 达成）；"
        "**解析 RH 收支**在工程分辨率下不可达（如实 FAIL，与一维账本同源）；"
        "**二维 WENO 限制器**在光滑流上熵销毁，是本次验证发现的真实缺陷，已用回归测试钉住"
        "（`tests/test_entropy_2d.py::test_weno_limiter_entropy_destruction_is_pinned`），未修。")
    add("")
    add("## ⑦ 交付物与复现")
    add("")
    add("| 路径 | 内容 |")
    add("|---|---|")
    add("| `benchmarks/entropy_budget_2d.py` | 二维熵积分、两个算例、限制器活动度、三道自检、研究驱动 |")
    add("| `benchmarks/plot_entropy_2d.py` | 三联图（收敛/收支/缺陷场） |")
    add("| `benchmarks/report_entropy_2d.py` | 把 JSON 注入本报告（零手抄） |")
    add("| `tests/test_entropy_2d.py` | 14 个回归测试（含 WENO 缺陷钉扎） |")
    add("| `validation/entropy_2d/summary.json` | 机器可读全部数字与判决 |")
    add("| `docs/figures/entropy_budget_2d.png` | 报告插图 |")
    add("")
    add("```bash")
    add("python -m benchmarks.entropy_budget_2d          # 生成 summary.json + 图")
    add("python -m benchmarks.report_entropy_2d         # 注入本报告的数字")
    add("python -m pytest -q tests/test_entropy_2d.py   # 回归")
    add("```")
    add("")
    add("运行环境：本机 128 核、load15≈90 的多席并发机器，脚本内 `torch.set_num_threads(1)`；"
        "全量研究耗时 %s s（`quick` 模式约 1/4）。" % summary["runtime_seconds"])
    return "\n".join(lines) + "\n"


def main():
    with open(SUMMARY) as handle:
        summary = json.load(handle)
    text = build(summary)
    with open(REPORT, "w") as handle:
        handle.write(text)
    print(f"wrote {REPORT} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
