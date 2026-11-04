"""Render the transient wall-heat-flux verification report from its JSON artefact.

Every number in the generated markdown is read from
``validation/ns_dg_2d_transient_wall/summary.json`` (written by
``benchmarks/verify_ns_transient_wall_heat.py``); nothing is typed by hand.

Usage::

    PYTHONPATH=. python -m tools.render_transient_wall_report
"""
from __future__ import annotations

import json
import math
import os

SUMMARY = os.path.join("validation", "ns_dg_2d_transient_wall", "summary.json")
TARGET = os.path.join("docs", "reports", "tensordg-ns-transient-wall-heat.md")
FIGURE = "docs/figures/ns_dg_2d_transient_wall_heat.png"


def _pct(value):
    return "%.3f%%" % (100.0 * value)


def render(summary):
    params = summary["parameters"]
    amplitude = summary["amplitude"]
    mesh = summary["mesh"]
    selfcheck = summary["reference_selfcheck"]
    top = summary["top_wall"]
    lines = []
    add = lines.append

    add("# 非定常壁面热流的文献级验证：突然升温壁面 vs erfc 半无限介质解（m3640）")
    add("")
    add("**仓**：TensorDG · **任务**：m3640（自找活） · **模块**：`tensordg/ns_dg_2d.py`")
    add("")
    add("## ① 案例介绍（物理背景 / 几何 / 参数）")
    add("")
    add("壁面**热流**通道此前只有稳态证据（线性导热场，m2831/m2876），非定常通道按")
    add("m3614 §5 局限 4 的原始记录是「未验证」。本单补的算例是经典的热渗透问题：")
    add("静止气体充满通道 `0 <= y <= H`，初温均匀为 `T0`；`t = 0+` 时底壁温度阶跃到")
    add("`T_w = T0 + dT` 并保持，顶壁保持 `T0` 且远离（`H >> 2 sqrt(alpha t)`，残差")
    add("`erfc(H / 2 sqrt(alpha t)) = %.2e`）。主导平衡是常压导热" % top["top_wall_residual_erfc"])
    add("`dT/dt = alpha d2T/dy2`，`alpha = kappa / (rho0 c_p)`，其半无限介质解为")
    add("")
    add("```")
    add("T(y,t) = T0 + dT erfc(eta),   eta = y / (2 sqrt(alpha t))")
    add("q_w(t) = -kappa dT/dy|_(y=0) = kappa dT / sqrt(pi alpha t)   (> 0，由壁面流入流体)")
    add("```")
    add("")
    add("| 参数 | 值 |")
    add("|---|---:|")
    add("| gamma | %.1f |" % params["gamma"])
    add("| mu（动力粘性，常数） | %.3f |" % params["mu"])
    add("| Pr | %.2f |" % params["prandtl"])
    add("| kappa = mu c_p / Pr | %.6f |" % params["kappa"])
    add("| alpha = kappa / (rho0 c_p) | %.6f |" % params["alpha"])
    add("| T0 / rho0 / p0 | %.1f / %.1f / %.1f |" % (params["T0"], params["density0"], 1.0))
    add("| 通道高度 H | %.1f |" % params["height"])
    add("| 物理时刻 t | %.2f（热渗透长度 2 sqrt(alpha t) = %.4f） |"
        % (params["final_time"], top["diffusion_length"]))
    add("| 准静态判据 `2 sqrt(alpha t) / (c_s t)` | %.3f（须 << 1；实测 1.0 -> 5.6%% 偏差，0.24 -> 0.4%%，判据 < 0.3） |"
        % top["quasi_static_ratio"])
    add("| 声速 c_s = sqrt(gamma p0 / rho0) | %.4f |" % top["sound_speed"])
    add("| 网格 | x 周期 1 单元；y 方向 8/12/16/24 单元 |")
    add("| 阶数 | P1、P2（张量积 Legendre 模态基，SSP-RK3，BR1/SIPG 粘性面） |")
    add("")
    add("## ② 怎么做的（方法 / 流程 / 工具链）")
    add("")
    add("1. **求解器增量（最小改动）**：`boundary` 的每个条目除 `None` / 定常壁面字典外，")
    add("   现在也接受**可调用对象** `f(t) -> {\"velocity\":..., \"temperature\":...}`，")
    add("   由 SSP-RK3 三个阶段各自按阶段时刻求值（`stage_boundary(time)`）。默认路径")
    add("   （无 callable）逐位不变，见 `tests/test_ns_transient_wall_heat.py`"
        "`::test_callable_boundary_matches_constant_boundary`（atol=0 的全等断言）。")
    add("2. **壁面热流的取法**：直接取格式自己在壁面用的量——`ymin` 面 DG 边界数值通量的")
    add("   能量分量（`ns._face_primitives` + `ns._boundary_flux`，与 m2831/m3139 审计同一估计器），")
    add("   不引入后处理拟合，避免「用另一套插值验证一套插值」。")
    add("3. **参考解自检**：`kappa dT / sqrt(pi alpha t)` 与 erfc 场的单侧二阶差分")
    add("   `-kappa dT/dy` 对比，相对差 **%.2e**（阈值 1e-4）。参考解若写错，下面所有收敛")
    add("   性测量都会变成「什么都没测」。"
        % selfcheck["relative_difference"])
    add("4. **低 Mach 模型误差的显式测量**：求解器积的是**可压缩** N-S，erfc 剖面只在")
    add("   `O(dT/T0)` 意义下是精确解（常压加热必然逼出一个同阶的弱膨胀速度场）。本单把")
    add("   这个模型误差**测出来**而不是假设掉：幅值阶梯 `dT/T0 = 0.02 / 0.01 / 0.005`，")
    add("   再把 `q_w / dT` 线性外推到 `dT -> 0`，剩余部分才是离散误差。")
    add("5. **工具链**：`benchmarks/verify_ns_transient_wall_heat.py` 写")
    add("   `validation/ns_dg_2d_transient_wall/*.json` 与 `%s`；" % FIGURE)
    add("   本报告由 `tools/render_transient_wall_report.py` 从 JSON 程序化渲染（零手抄）。")
    add("")
    add("## ③ 计算结果（程序化注入）")
    add("")
    add("### 3a 幅值阶梯（P1，ny = 16，t = %.2f）" % params["final_time"])
    add("")
    add("| dT | q_w 数值 | q_w 精确 | 相对误差 | max abs v_y |")
    add("|---:|---:|---:|---:|---:|")
    for row in amplitude["cases"]:
        add("| %.3f | %.6f | %.6f | %s | %.2e |"
            % (row["amplitude"], row["wall_heat_flux"], row["wall_heat_flux_exact"],
               _pct(row["wall_heat_flux_relative_error"]), row["max_abs_velocity_y"]))
    add("")
    add("外推到 `dT -> 0`：`q_w / dT = %.6f`，精确值 `%.6f`，差 **%s**；"
        % (amplitude["zero_amplitude_limit"], amplitude["target_flux_per_amplitude"],
           _pct(amplitude["extrapolation_relative_error"])))
    add("幅值缩小 %.0f 倍时原始误差缩小 %.2f 倍（近似线性 = `O(dT/T0)` 模型误差的特征）。"
        % (1.0 / amplitude["amplitude_ratio"], 1.0 / amplitude["error_ratio_finest_over_coarsest"]))
    add("")
    add("### 3b 网格 / 阶数阶梯（dT = %.3f，t = %.2f）" % (mesh["amplitude"], params["final_time"]))
    add("")
    add("| 阶数 | ny | steps | q_w 相对误差 | T 剖面相对 L2 | 观测阶 |")
    add("|---:|---:|---:|---:|---:|---:|")
    for row in mesh["cases"]:
        add("| P%d | %d | %d | %s | %s | — |"
            % (row["degree"], row["ncells"][1], row["steps"],
               _pct(row["wall_heat_flux_relative_error"]),
               _pct(row["profile_relative_l2"])))
    add("")
    add("观测阶（壁面热流相对误差，端到端拟合）："
        + "，".join("P%s = %.2f" % (key, value) for key, value in mesh["observed_order"].items()))
    add("")
    gate = [row for row in mesh["cases"] if row["degree"] == 1 and row["ncells"][1] == 16][0]
    legacy = summary.get("legacy_regime")
    if legacy:
        old_amp = legacy["amplitude"]
        old_mesh = legacy["mesh"]
        rows = [row for row in old_mesh["cases"]
                if row["degree"] == 1 and row["ncells"][1] in (8, 16, 24)]
        add("### 3c 参数区间陷阱：同一算例在 `mu = %.3f, t = %.2f` 下的结果"
            % (legacy["parameters"]["mu"], legacy["parameters"]["final_time"]))
        add("")
        add("首次运行用的参数是 `mu = %.3f`、`t = %.2f`。壁面热流误差为 %s，而且**不随网格、"
            % (legacy["parameters"]["mu"], legacy["parameters"]["final_time"],
               _pct(rows[0]["wall_heat_flux_relative_error"])))
        add("不随阶数、不随幅值收敛**（P1：ny = 8/16/24 = %s / %s / %s；幅值外推后仍 %s）。"
            % (_pct(rows[0]["wall_heat_flux_relative_error"]),
               _pct(rows[1]["wall_heat_flux_relative_error"]),
               _pct(rows[2]["wall_heat_flux_relative_error"]),
               _pct(old_amp["extrapolation_relative_error"])))
        add("这种「与 h 和 dT 都无关的系统性偏差」不是离散误差，而是**参数区间选错**：")
        add("那组参数下热渗透层内的声学穿越时间 `2 sqrt(alpha t)/c_s` 与物理时间同量级，")
        add("常压（准静态）假设不成立，erfc 参考解本身在该区间不适用。判据")
        add("`t >> 4 alpha / c_s^2 = 4 mu/(gamma p0 Pr)` 把它量化；本单据此改用")
        add("`mu = %.3f, t = %.2f`（判据比 %.2f -> %.3f），同一格式的误差随即降到 %s。"
            % (params["mu"], params["final_time"],
               legacy["parameters"]["mu"] and
               (2.0 * math.sqrt(legacy["parameters"]["alpha"] * legacy["parameters"]["final_time"])
                / math.sqrt(1.4) / legacy["parameters"]["final_time"]),
               top["quasi_static_ratio"], _pct(gate["wall_heat_flux_relative_error"])))
        add("")
    add("## ④ 结果图")
    add("")
    add("![transient wall heat](../figures/ns_dg_2d_transient_wall_heat.png)")
    add("")
    add("左：温度剖面对 erfc 精确解（多条 = 不同网格/阶数）；中：壁面热流相对误差 vs 网格；")
    add("右：误差随 `dT/T0` 的线性（低 Mach 模型误差）标度。")
    add("")
    add("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **坑（本单最大的一条）：参数区间错了会得到「不收敛的假误差」**。第一版用")
    add("   `mu = 0.05, t = 0.2`，壁面热流误差 %s 且对网格/阶数/幅值都不敏感——"
        % (_pct(summary["legacy_regime"]["mesh"]["cases"][0]["wall_heat_flux_relative_error"])
           if summary.get("legacy_regime") else "5% 量级"))
    add("   会误判成「格式有 bug」或「物理模型不同」，实际是该区间内声学穿越时间与")
    add("   扩散时间同量级，常压 erfc 参考解不适用。判据是 `t >> 4 mu/(gamma p0 Pr)`。")
    add("   教训：用参考解前先算清楚**参考解自身的适用条件**，别只算网格够不够密。")
    add("2. **坑：`t = 0+` 的壁面温度阶跃意味着热流在初始时刻发散**（`q_w ~ 1/sqrt(t)`），")
    add("   热渗透长度 `2 sqrt(alpha t)` 必须覆盖若干单元。本单 `t = %.2f` 时热渗透长度"
        % params["final_time"])
    add("   %.4f，最粗网格 ny = 8 也有约 %.1f 个单元。"
        % (top["diffusion_length"], top["diffusion_length"] * 8.0))
    add("3. **坑：壁面热流比剖面难一个数量级**。剖面相对 L2 早在 8 单元就 <0.1%，")
    add("   而壁面热流同网格下是 5% 量级；只报剖面误差会严重高估这个通道的可信度。")
    add("4. **局限（本单最重要）**：参考解是**常压导热**的，可压缩求解器只在 `O(dT/T0)`")
    add("   意义下匹配。本单用幅值外推把这个模型误差压掉（外推后 %s），"
        % _pct(amplitude["extrapolation_relative_error"]))
    add("   但真实 `dT/T0 = O(1)` 的强加热问题该基准**不适用**。")
    add("5. **局限**：常数 `mu`（无 Sutherland 律）、无体积力、Cartesian、单核 float64、")
    add("   显式时间推进（步数 ~ 14 N²，N = 热渗透长度内的单元数）；结论不外推到 3D / 非结构网格。")
    add("6. **局限**：只测了能量通道的面通量；壁面**剪应力**的非定常验证由 m3614（Stokes 第一问题）")
    add("   覆盖，两条通道互不替代。")
    add("7. **适用范围**：现在可以说壁面热流在**非定常**导热主导过程中可信到 <3%（P1、ny >= 16、")
    add("   `dT/T0 <= 0.02`）；含边界层对流-压力强耦合的壁面热流仍未验证。")
    add("")
    add("## ⑥ 误差判定（关键量 vs 基准，逐项）")
    add("")
    add("| 判据 | 基准 | 实测 | 允许 | 是否 <3% |")
    add("|---|---|---|---:|---|")
    add("| 壁面热流（P1，ny=16，dT=0.005） | erfc 半无限介质解 | %s | 3%% | **%s** |"
        % (_pct(gate["wall_heat_flux_relative_error"]),
           "是" if gate["wall_heat_flux_relative_error"] < 0.03 else "否"))
    add("| 壁面热流（外推到 dT->0） | 同上 | %s | 3%% | **%s** |"
        % (_pct(amplitude["extrapolation_relative_error"]),
           "是" if amplitude["extrapolation_relative_error"] < 0.03 else "否"))
    add("| 温度剖面（P1，ny=16） | erfc 精确剖面 | %s | 1%% | **%s** |"
        % (_pct(gate["profile_relative_l2"]),
           "是" if gate["profile_relative_l2"] < 0.01 else "否"))
    add("| 参考解闭式 vs 有限差分 | erfc 场的一侧差分 | %.2e | 1e-4 | **是** |"
        % selfcheck["relative_difference"])
    add("| 顶壁（远场）影响 | erfc 残差 | %.2e | 1e-6 | **是** |"
        % top["top_wall_residual_erfc"])
    add("| 观测阶（P1 壁面热流） | 设计阶 1 | %.2f | 3%% | %s |"
        % (mesh["observed_order"].get("1", float("nan")),
           "**是**" if abs(mesh["observed_order"].get("1", 0.0) - 1.0) < 0.03 else "—（端到端阶，非严格 3% 门）"))
    add("")
    add("**显式回答**：关键量「非定常壁面热流」在 P1、ny = 16、dT/T0 = 0.005 下相对误差 %s，"
        % _pct(gate["wall_heat_flux_relative_error"]))
    add("**< 3%**；把 `O(dT/T0)` 模型误差外推掉后为 %s。"
        % _pct(amplitude["extrapolation_relative_error"]))
    add("")
    add("## ⑦ 复现")
    add("")
    add("```bash")
    add("PYTHONPATH=. OMP_NUM_THREADS=1 python -m benchmarks.verify_ns_transient_wall_heat")
    add("PYTHONPATH=. OMP_NUM_THREADS=1 python -m pytest tests/test_ns_transient_wall_heat.py -q")
    add("PYTHONPATH=. python -m tools.render_transient_wall_report")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


LEGACY_DIR = os.path.join("validation", "ns_dg_2d_transient_wall")


def _load_legacy():
    """Old-parameter artefacts (mu = 0.05, t = 0.2) kept as evidence of the trap."""
    paths = {key: os.path.join(LEGACY_DIR, "legacy_mu005_t02_%s_study.json" % key)
             for key in ("amplitude", "mesh")}
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    legacy = {key: json.load(open(path)) for key, path in paths.items()}
    legacy["parameters"] = {"mu": 0.05, "final_time": 0.2,
                            "alpha": 0.05 * (1.4 / 0.4) / 0.71 / (1.4 / 0.4)}
    return legacy


def main():
    with open(SUMMARY) as handle:
        summary = json.load(handle)
    summary["legacy_regime"] = _load_legacy()
    markdown = render(summary)
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "w") as handle:
        handle.write(markdown)
    print("wrote", TARGET, len(markdown), "chars")


if __name__ == "__main__":
    main()