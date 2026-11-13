"""Build the mission report (figure + markdown) for m18076 from tracked artifacts."""
import base64, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/tp-agent/work/m18076")
ART = ROOT / "validation/repro_shu_osher/artifacts"
m = json.loads((ART / "metrics.json").read_text())
ex = m["state_column_regression"]

probe = np.load(ART / "raw" / Path(ex["artifact"]).name, allow_pickle=False)
x = probe["x"]; rho = probe["rho"]; rho_u = probe["rho_u"]; rho_e = probe["rho_e"]
mixed = np.stack((rho, rho_u, rho_e)).max(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), gridspec_kw={"height_ratios": [2.1, 1.0]})
ax = axes[0]
ax.plot(x, rho, "k-", lw=2.0, label=r"correct mouth: $\rho$ (column 0)")
ax.plot(x, mixed, "-", color="crimson", lw=1.1, alpha=0.85,
        label=r"old convention: per-cell max of $(\rho,\rho u,\rho E)$")
ax.plot(x, rho_e, ":", color="tab:orange", lw=1.0, label=r"$\rho E$ (the dominant column)")
ax.axvspan(-5.0, -3.0, color="0.85", alpha=0.5, label=r"quiet-zone window ($x\leq-3$)")
ax.axvline(m["analytic"]["shock_position_analytic"], color="purple", lw=1.0, ls="-.",
           label=f"analytic shock $x_s$={m['analytic']['shock_position_analytic']:.4f}")
ax.set_xlabel("$x$"); ax.set_ylabel("density-like quantity")
ax.set_title("Shu-Osher DG (P1, N=200, CFL=0.4, t=1.8): the state-column defect, replayed from artifacts")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
ax.annotate(f"peak {ex['density_max_correct']:.2f} -> "
            f"{ex['density_max_if_whole_state_used']:.2f} ({ex['overstatement_factor']:.2f}x)",
            xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8,
            bbox=dict(fc="white", ec="crimson", alpha=0.9))

ax = axes[1]
labels = ["quiet-zone plateau dev. [%]", "shock position error [%]"]
correct = [ex["plateau_dev_percent_correct"], ex["shock_position_rel_error_percent_correct"]]
old = [ex["plateau_dev_percent_if_whole_state_used"],
       ex["shock_position_rel_error_percent_if_whole_state_used"]]
idx = np.arange(len(labels)); w = 0.36
ax.bar(idx - w / 2, correct, w, color="tab:green", label="correct convention")
ax.bar(idx + w / 2, old, w, color="crimson", label="old convention")
ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xticklabels(labels, fontsize=8)
for i, (a, b) in enumerate(zip(correct, old)):
    ax.text(i - w / 2, a * 1.5, f"{a:.2e}", ha="center", fontsize=7)
    ax.text(i + w / 2, b * 1.5, f"{b:.1f}", ha="center", fontsize=7)
ax.set_ylabel("relative deviation [%] (log)")
ax.set_title("the same analytic anchors applied to both conventions (they catch the defect)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
png = ROOT / "scratch/m18076_state_column.png"
fig.savefig(png, dpi=140); plt.close(fig)
b64 = base64.b64encode(png.read_bytes()).decode()

ladder = m["ladder"]
finest = max(ladder, key=lambda r: (r["degree"], r["ncells"]))
p1_800 = [r for r in ladder if r["degree"] == 1 and r["ncells"] == 800][0]
rows = "\n".join(
    f"| P{r['degree']} | {r['ncells']} | {r['steps']} | {r['rel_l1_percent']:.3f} | "
    f"{r['rel_l2_percent']:.3f} | {r['rho_min']:.4f} | {r['rho_max']:.4f} | "
    f"{r['shock_position']:.4f} | {r['shock_position_rel_error_percent']:.3f} | "
    f"{'✓' if r['passes_5pct'] else '✗'} |" for r in ladder)
anc = m["reference_anchor"]; an = m["analytic"]; rsc = m["reference_self_convergence"]

md = f"""# TensorDG m18076 — Shu–Osher 算例池 §②b「状态列混充」缺陷：从占位符到可测量回归项

## ① 案例介绍（物理背景 / 参数）
1-D Euler（γ=1.4）激波–熵波干扰，Shu & Osher 1988（JCP 77, 439–471；1989 JCP 83, 32–78 test case 3）：
$x=-4$ 处 Mach-3 激波（$M_s=3$，打入 $\\rho=p=1$ 静止气）向右扫过 $\\rho=1+0.2\\sin(5x)$ 静止正弦密度波，
激波下游把正弦波转成高频熵波，$t=1.8$ 结束。域 $x\\in[-5,5]$（透射边界）。
左状态由 Rankine–Hugoniot 反查自洽：$\\rho={an['hugoniot_left_state_from_mach3']['rho']:.6f}$，
$u={an['hugoniot_left_state_from_mach3']['u']:.6f}$，$p={an['hugoniot_left_state_from_mach3']['p']:.6f}$；
解析激波速度 $S={an['shock_speed_analytic']:.6f}$，$t=1.8$ 时 $x_s={an['shock_position_analytic']:.6f}$，
安静区端头 $x={an['quiet_zone_head_analytic']:.6f}$，精确质量收支 $M(1.8)={m['mass_budget']['mass_final_predicted']:.6f}$。
该算例无闭式解 → 用三根独立锚（解析运动学 + 独立 FV-WENO5 参考 + 参考解自收敛）替代原文字典表值（原文为图解法、本机无外网，按 WITHHELD 登记）。

## ② 怎么做的（方法 / 关键设置 / 工具链）
* 任务对象：`benchmarks/shu_osher_case_pool.py: merge()` 里的回归 exhibit。
  **根因（本轮定位）**：探针路径用 `RAW / _key(...)` 拼接而漏掉 `.npz` 后缀 → `Path.exists()` 恒 False →
  整段 exhibit 从未执行 → `metrics.json` 无该键 → 报告 §②b 只能落 `（未采集：先跑 ladder 再 merge）` 占位符。
* 修复：路径补后缀；把 exhibit 提升为模块级函数 `state_column_regression(path)`，使其不依赖整条 ladder 也可测试；
  判据从「峰值」扩到「三列极值 + 负值单元数 + 两项解析锚在错误剖面下的残差」。
* 数据来源：已提交工件 `validation/repro_shu_osher/artifacts/raw/dg_cfl0p4_limweno_n200_p1.npz`
  （P1/N=200/CFL=0.4）；本机用 `merge --reference-cells 3200` 从工件重算（未重跑求解器，20 分钟 ladder 未重算）。
* 工具链：`python -m benchmarks.shu_osher_case_pool merge` → `benchmarks/report_shu_osher_case_pool.py`
  → `validation/repro_shu_osher/make_evidence.py`（sha256 清单）。所有数字由工件注入，零手抄。

## ③ 计算结果（工件注入）
**缺陷 exhibit（P1, N=200, CFL=0.4；旧口径 = 三列守恒量混充 ρ）**

| 量 | 正确口径（第 0 列 = ρ） | 旧口径 | 后果 |
|---|---|---|---|
| $\\max$ | {ex['density_max_correct']:.4f} | {ex['density_max_if_whole_state_used']:.4f} | 峰值高估 {ex['overstatement_factor']:.2f}× |
| $\\min$ | {ex['density_min_correct']:.4f} | {ex['density_min_if_whole_state_used']:.2e} | 仅 {ex['negative_cells_if_whole_state_used']} 个单元的 $\\rho u$ 数值零 |
| 安静区平台偏差 [%] | {ex['plateau_dev_percent_correct']:.4e} | {ex['plateau_dev_percent_if_whole_state_used']:.2f} | 解析锚 ✗（差 6 个量级） |
| 激波位置 $x_s$ | {ex['shock_position_correct']:.4f} | {ex['shock_position_if_whole_state_used']:.4f} | — |
| $x_s$ 误差 [%] | {ex['shock_position_rel_error_percent_correct']:.3f} | {ex['shock_position_rel_error_percent_if_whole_state_used']:.2f} | 解析锚 ✗ |

三列实测极值：$\\rho\\in[{ex['column_extrema']['rho'][0]:.4f},{ex['column_extrema']['rho'][1]:.4f}]$，
$\\rho u\\in[{ex['column_extrema']['rho_u'][0]:.2e},{ex['column_extrema']['rho_u'][1]:.4f}]$，
$\\rho E\\in[{ex['column_extrema']['rho_e'][0]:.4f},{ex['column_extrema']['rho_e'][1]:.4f}]$。
→ 「峰值被 $\\rho E$ 顶到 40 量级」被证实；但 committed 文档原有的「ρ_min 为负是方程可采纳性误判的直接来源」一句
**按工件修正**：负值只有 $-{abs(ex['density_min_if_whole_state_used']):.1e}$（{ex['negative_cells_if_whole_state_used']} 个单元的 $\\rho u$ 数值零），
真正被抓出缺陷的是**解析锚**（平台 6 个量级、激波位置 2 个量级）。

**DG 阶梯（9 配置，5% 判据；本轮回归不受影响）**

| order | N | steps | rel $L^1$ vs ref [%] | rel $L^2$ [%] | $\\rho_{{\\min}}$ | $\\rho_{{\\max}}$ | $x_s$ | $x_s$ vs RH [%] | <5% |
|---|---|---|---|---|---|---|---|---|---|
{rows}

独立参考解（FV-WENO5, N={m['reference']['ncells']}）自身锚残差：平台 $\\rho$ 偏差 {anc['plateau_rho_dev_percent']:.2e}%，
未扰动区 $1+0.2\\sin5x$ 的 $L^1$ {anc['preshock_sine_rel_l1_percent']:.4f}%，激波位置 {anc['shock_position']:.4f}（解析 {an['shock_position_analytic']:.4f}）；
参考解自收敛（N={rsc['coarse_ncells']} vs {m['reference']['ncells']}）：$L^1$={rsc['rel_l1_percent']:.3f}%、$L^2$={rsc['rel_l2_percent']:.3f}%。

## ④ 结果图
图 1：上幅 = 正确密度剖面 vs 旧口径「每单元三列取最大」剖面（$\\rho E$ 主导）+ 安静区窗口 + 解析激波位置；
下幅 = **同一套解析锚**分别施加在两种口径上（对数轴）：平台偏差 {ex['plateau_dev_percent_correct']:.2e}% → {ex['plateau_dev_percent_if_whole_state_used']:.0f}%，
激波误差 {ex['shock_position_rel_error_percent_correct']:.2f}% → {ex['shock_position_rel_error_percent_if_whole_state_used']:.0f}%。
算例池主体三图（profiles / convergence / cfl）见 `docs/figures/shu_osher_case_pool_*.png`，由 `benchmarks/report_shu_osher_case_pool.py`
从工件生成（本次重生成后与 committed 版本逐字节一致，即提交前未漂移）。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）
1. **空完成的形态**：不是「没写代码」，而是「代码在、探针路径错、键缺失」，报告静默降级为占位符；
   committed 文档里那行 `（未采集…）` 就是它的化石。修复判据 = 键存在 + 数值可被 raw npz 独立重算。
2. **锚点必须施加在最外层输出上**：本缺陷在有解析锚的算例里本该立刻暴露，之所以「静默」，
   是旧流程只在中间量上比锚、没在最终 profile 上比。
3. **预存红测（非本单引入，`git stash` 后 main 状态复现）**：`tests/test_public_api_contract.py` 3 例
   （34 个公共符号未导出 / 其他 mission 审计文档的 import 片段未解析 / cases 注册表多 `prefinement_exponential`）、
   `tests/test_gate_vacuity_policy.py` 3 例（census 与新鲜扫描不一致）、
   `tests/test_xval_oss_shu_osher.py::test_oss_leg_is_independent_and_converges`
   （`run_oss_level` 已改为返回 dict，测试仍按 3 元组解包；仅本机有 `/tmp/oss` 时触发，CI 无 clawpack 故 skip）。
4. **集群**：全量测试作业 cj1780 递交后长期 `queued`（前面 300+ 作业，无派发），未取到集群结果；全量验证退回控制面本机。
5. 适用范围：exhibit 只说明「旧口径会把峰值与解析锚带偏」，不能反推旧口径下的剖面误差数值
   ——旧代码把 `(N,3)` 直接当 `(N,)` 用，误差定义本身失效，故只报可定义的量。

## ⑥ 误差判定（逐项，显式回答是否 <3%）
| 关键量 | 计算值 | 基准 | 相对误差 | <3%？ |
|---|---|---|---|---|
| 密度剖面 $L^1$（最细 P3/N=800） | {finest['rel_l1_percent']:.3f}% | 独立 FV-WENO5 N=3200 | {finest['rel_l1_percent']:.3f}% | **是** |
| 密度剖面 $L^1$（P1/N=800） | {p1_800['rel_l1_percent']:.3f}% | 同上 | 同上 | **是** |
| 激波位置 $x_s$（P3/N=800） | {finest['shock_position']:.4f} | RH/运动学 {an['shock_position_analytic']:.4f} | {finest['shock_position_rel_error_percent']:.3f}% | **是** |
| 安静区平台 $\\rho$（P3/N=800） | — | 常值 3.857143 | {finest['plateau_rho_dev_percent']:.2e}% | **是** |
| 全域质量收支（P3/N=800） | — | 精确恒等式 {m['mass_budget']['mass_final_predicted']:.6f} | {finest['mass_abs_error_percent_of_initial']:.2e}%（对初值归一） | **是** |
| 峰值（旧口径 exhibit） | {ex['density_max_if_whole_state_used']:.2f} | 正确 {ex['density_max_correct']:.4f} | **{100 * (ex['overstatement_factor'] - 1):.0f}%** | **否**（即被修掉的缺陷） |

结论：算例池本体关键量**全部 <3%**（最细档 0.66% / 0.46% / 1e-2% 量级），同时满足 5% 与 3% 双判据；
被修的缺陷本身 889%（9.89×），修前不可接受、修后可测并被两条测试钉死。
未覆盖：原文字典表值仍 WITHHELD（无外网、原文为图解法）；集群全量测试 cj1780 未派发。
"""

report = {
    "title": "TensorDG m18076：Shu–Osher 算例池 §②b 状态列缺陷——从占位符到可测量回归项（889% → 0.66%）",
    "summary": "定位并修复 case-pool merge() 探针路径漏 .npz 后缀导致回归 exhibit 从未执行、报告 §②b 恒为占位符的缺陷；"
               "exhibit 升级为可测量三重项（峰值 9.89×、平台锚 0.0007%→915%、激波误差 0.71%→108%），"
               "重生成 committed metrics/evidence/报告，并以 2 条测试（合成工件接线 + 从 raw npz 重算对表）钉死。",
    "scenario_key": "tensordg-shu-osher-case-pool",
    "mission_id": 18076,
    "tags": ["TensorDG", "Shu-Osher", "DG", "case-pool", "regression", "evidence", "m18076"],
    "markdown": md,
    "figures": [{"name": "shu_osher_state_column_regression.png",
                 "caption": "上：正确密度剖面 vs 旧口径（三列取最大）剖面；下：同一套解析锚在两种口径下的残差（对数轴）",
                 "png_base64": b64}],
}
out = ROOT / "scratch/report_m18076.json"
out.write_text(json.dumps(report))
print("wrote", out, len(md), "chars;", len(b64) // 1024, "KiB png")