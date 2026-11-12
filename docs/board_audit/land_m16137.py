"""Land the m16195 board-audit verdict + publish the paper-grade report.

Run once:  python docs/board_audit/land_m16137.py
Writes: tensor_maturity_verdicts row (repo=TensorDG, scope='m16137 ...'),
        artifacts/audit_m16137/verdict.json, and POSTs /tensor/reports.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MISSION = 16195           # 本审计单
TARGET = 16137            # 被审单
REPO = "TensorDG"
SCOPE = "m16137 CASE-DG-DALEMBERT-ELASTIC-1D"
LEVEL = 2
WT = "/root/tp-agent/work/m16137"              # 交付工作树（文件真实存在）
AUD = "/root/tp-agent/work/m16195"             # 本审计工作树
ART = f"{AUD}/artifacts/audit_m16137"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")

indep = json.loads(Path(f"{ART}/independent_recompute.json").read_text())
c = indep["checks"]

EVIDENCE = {
    "impl_files": [
        f"{WT}/tensordg/elastic_1d.py (456 行：_operators:63-93 Legendre 基/求积；"
        f"_numerical_flux:120-124 精确 Riemann upwind；_rhs:127-152 模态弱形式；"
        f"dalembert_exact:230-296 闭式解；solve_elastic_rod:298-400 RK4/SSP-RK3)",
        f"{WT}/tensordg/cases/elastic_rod_1d.py (361 行：题面/参数表/探针表/反射/能量/门禁)",
        f"{WT}/benchmarks/elastic_rod_dalembert.py (288 行：七段研究 + 4 图)",
        f"{WT}/docs/CASE-DG-DALEMBERT-ELASTIC-1D.md (168 行，六段齐全)",
    ],
    "entry": f"{WT}/benchmarks/elastic_rod_dalembert.py::main:212-240 "
             f"(resolutions 25/50/100/200 × degrees 1-4; --quick 副本本单实跑 83.8 s)",
    "run_artifacts": [
        f"{WT}/runs/elastic_rod_dalembert/results.json (gate/convergence/probes/reflection/energy/flux/peaks)",
        f"{WT}/runs/elastic_rod_dalembert/fig1_profiles.png",
        f"{WT}/runs/elastic_rod_dalembert/fig2_convergence.png",
        f"{WT}/runs/elastic_rod_dalembert/fig3_reflection.png",
        f"{WT}/runs/elastic_rod_dalembert/fig4_flux_dispersion.png",
        f"{ART}/independent_recompute.json (本单独立复算：C1/C3/C4/C5 + C2 四档阶)",
        f"{ART}/independent_recompute.log",
        f"{ART}/pytest_rerun.log (15 passed in 44.84 s)",
        f"{ART}/bench_quick/results.json + fig1..fig4 (repro_cmd 全链重跑 83.8 s)",
        f"{ART}/l4_verdict.json (平台 L4 裁判原文)",
    ],
    "tests": [
        "tests/test_elastic_rod_1d.py::test_exact_solution_solves_the_system (解析解自身 PDE 残差)",
        "tests/test_elastic_rod_1d.py::test_invariant_is_constant_along_characteristic",
        "tests/test_elastic_rod_1d.py::test_convergence_design_order[1..4] (断言 order > p+0.6)",
        "tests/test_elastic_rod_1d.py::test_headline_case_meets_two_percent_criterion (判据 + 自加 <1e-3)",
        "tests/test_elastic_rod_1d.py::test_reflection_wall_stress_ratio (fixed 2.0 / free 0)",
        "tests/test_elastic_rod_1d.py::test_energy_conserved_periodic / test_upwind_beats_central_flux_on_smooth_wave",
        f"本单复跑：python -m pytest tests/test_elastic_rod_1d.py -q → 15 passed in 44.84 s ({ART}/pytest_rerun.log)",
    ],
    "integration_callers": (
        "NONE in main：审计检出 /root/tp-agent/work/m16195 (main cc1c2df) 无 tensordg/elastic_1d.py，"
        "git log -- tensordg/elastic_1d.py 空；tensordg/__init__.py 未导出（对照 main :1-40 已导出 "
        "ns_dg_2d/poisson_2d 等 20+ 求解器）；tensordg/cases/__init__.py 仍只导出 shock_entropy；"
        "git grep elastic_1d 3240063 仅命中自身 bench/test/case → 不达 L3"
    ),
    "benchmark": {
        "criterion_source": "tensor_case_pool id=96 tol_policy '<2% | 指标: 应力波剖面 sigma(x,t) vs d'Alembert 双行波解析'",
        "gate_config": "double-wave, P3, N=200, CFL=0.3, t=0.4/c",
        "ref": "闭式 d'Alembert 双行波（无物理弥散）；本单以 Fourier 谱传播子独立交叉验证：max|Δ|/A = 5.5e-15",
        "got": c["C1_gate_p3_n200"]["relative_l2_sigma_auditor"],
        "claimed": c["C1_gate_p3_n200"]["claimed_in_delivery_results_json"],
        "auditor_vs_claimed_reldiff": abs(c["C1_gate_p3_n200"]["relative_l2_sigma_auditor"]
                                          - c["C1_gate_p3_n200"]["claimed_in_delivery_results_json"])
                                         / c["C1_gate_p3_n200"]["claimed_in_delivery_results_json"],
        "within_tolerance": "PASS（<2%，余量 ×31107；<3% 判据同 PASS）",
        "convergence_auditor": {k: c["C2_convergence"][k]["observed_order"] for k in ("p1", "p2", "p3", "p4")},
        "design_order": "p+1 = 2/3/4/5；偏差 ≤0.42，无隐性降阶",
        "reflection_auditor": {k: c["C5_reflection"][k]["ratio"] for k in ("fixed", "free")},
        "energy_drift_auditor": c["C4_energy"]["relative_drift_auditor"],
        "registry_gap": "tensor_benchmark_cases WHERE anchor_slug='CASE-DG-DALEMBERT-ELASTIC-1D' → 0 行；"
                        "tensorbench_results WHERE mission_id=16137 → 0 行；tensor_case_pool id=96 仍 dispatched/未填 linked_anchor",
        "platform_judge": "GET /tensor/bench/l4-verdict/16137 → CLAIMED_NOT_PROVEN, rows=0, "
                          "failed_checks=[execution_success,source_cluster_job,cluster_job_bound,"
                          "ref_got_error_present,within_tolerance]",
    },
    "repro_cmd": (
        f"cd {WT} && python -m benchmarks.elastic_rod_dalembert   # 全量 225 s（交付跑）\n"
        f"cd /tmp/audit_m16137 && PYTHONPATH=. python3 -m benchmarks.elastic_rod_dalembert --quick "
        f"--outdir {ART}/bench_quick   # 本单实测 83.8 s，gate 6.429e-07 PASS\n"
        f"cd /tmp/audit_m16137 && PYTHONPATH=. python3 -m pytest tests/test_elastic_rod_1d.py -q "
        f"# 本单实测 15 passed 44.84 s\n"
        f"cd {AUD} && python3 docs/board_audit/audit_m16137.py --repo /tmp/audit_m16137 --outdir {ART}"
    ),
    "external_anchor": (
        "Graff 1975《Wave Motion in Elastic Solids》Ch.1（d'Alembert 表示 + 自由/固定端反射律）作公式出处；"
        "正文未取得 → 文献表格数字 WITHHELD（交付 ⑥ 已自述零转录）。唯一独立文献级量：钢细杆 c=sqrt(E/rho)=5172.19 m/s，"
        "落在手册 5100-5200 m/s 带内（本单复核一致）；反射律 wall σ=2σ_max(fixed)/0(free) 由本单独立复算命中 2.2e-7"
    ),
    "level_rationale": (
        "L2（本单独立复跑确认）：真求解器输入→计算→输出，15 测试从 git archive 干净副本通过，"
        "自写参考解与自写 L2 范数复算 6.429e-07（与交付差 4e-12），收敛阶 2.328/3.414/4.051/5.011。"
        "物理/数值实质已达 L4 强度（完整算例+逐项误差+收敛+反射/能量锚点+可复现），"
        "但 L3 未达（main 无文件、包未导出、无外部调用方）故不越级；"
        "L4 机器门自行否决（CLAIMED_NOT_PROVEN, rows=0, 无集群作业号）。附百页报告不改变级别，级别=功能实质。"
    ),
    "findings": {
        "F1_registry_not_backfilled": "tensor_benchmark_cases 无该锚点行、tensor_case_pool id=96 仍 dispatched/linked_anchor NULL → 平台基准判据缺位，L4 无法机器确认。",
        "F2_not_merged": "commit 3240063 仅在 agent/m16137；main cc1c2df 无 tensordg/elastic_1d.py；__init__ 未导出 → 交付能力目前对库用户不可调用。",
        "F3_local_not_cluster": "交付跑在本机 225 s 无 cluster_job_id → l4-verdict 的 source_cluster_job/cluster_job_bound 必红；本单 cj1264 已递交，递交时队列 435 位、4.2 min 仍排队，未取回，不计数。",
        "F4_report_attribution": "交付宣称 POST /reports → id=report_c222e0c255b4，但 tensor_reports WHERE mission_id=16137 → 0 行（同名报告挂在 16149/16158），mission.report_id 为空 → 溯源链断。",
        "F5_criterion_is_resolution_specific": "2% 只在 P3/N=200 成立；交付自披露 P1/N=25 = 38%（每波长 5 单元）。建议 tolerance 行写分辨率前提，避免粗网格复现者误判。",
        "F6_no_fabrication_signal": "四项独立交叉（自写解析解/自写 L2/Fourier 谱解/干净副本复跑）全部与交付逐位一致，未见造假迹象。",
    },
    "verdict": {"level": LEVEL,
                "means": "独立功能实现（真代码），数值实质达 L4 强度但因未合入 main/未登记基准而停在 L2",
                "remediation": ["merge agent/m16137 → main 并导出 solve_elastic_rod/solve_case/gate_report",
                                "tensor_benchmark_cases 落行 + case_pool id=96 置 passed 且填 linked_anchor",
                                "集群复跑 benchmark → ingest tensorbench_results 一行(ref/got/error)"]},
}

FIGDIR = Path(f"{ART}/figs")
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- independent figure: auditor's convergence + gate margin
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
for p, color in zip(("p1", "p2", "p3", "p4"), ("#888", "#3b7", "#37c", "#c33")):
    rows = c["C2_convergence"][p]["rows"]
    n = [r["ncells"] for r in rows]
    e = [r["relative_l2"] for r in rows]
    ax.loglog(n, e, "o-", color=color, label=f"p={p[1]} (order {c['C2_convergence'][p]['observed_order']:.2f})")
nn = np.array(n, dtype=float)
ax.loglog(nn, e[-1] * (nn / nn[-1]) ** -4.0, "k--", lw=0.8, label="slope -4")
ax.set_xlabel("cells N"); ax.set_ylabel(r"relative $L_2$ error of $\sigma$")
ax.set_title("(a) auditor's independent convergence (d'Alembert ref)")
ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8)
ax = axes[1]
labels = ["gate\nP3 N=200", "criterion\n<2%", "strict\n<3%"]
vals = [c["C1_gate_p3_n200"]["relative_l2_sigma_auditor"], 0.02, 0.03]
ax.bar(labels, vals, color=["#37c", "#bbb", "#ddd"])
ax.set_yscale("log"); ax.set_ylabel("relative L2 error")
ax.set_title(f"(b) gate margin x{c['C1_gate_p3_n200']['margin_vs_tolerance']:.0f}")
for i, v in enumerate(vals):
    ax.text(i, v * 1.4, f"{v:.1e}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIGDIR / "audit_convergence.png", dpi=160)
plt.close(fig)

FIGS = [
    {"name": "audit_convergence.png",
     "caption": "审计方独立复算（不调用交付侧验证函数）：(a) 自写 d'Alembert 参考 + 自写 L2 范数得到的 h/p 收敛，"
                "观测阶 2.33/3.41/4.05/5.01；(b) 主判据 P3/N=200 的误差与 2%/3% 容差对比，余量 ×31107。"},
    {"name": "fig1_profiles.png",
     "caption": "交付场量对比（取自交付工件 runs/elastic_rod_dalembert/fig1_profiles.png）：σ(x) 数值剖面 vs 精确解，"
                "P1/P2/P3 (N=50) 及误差曲线，可见 P3 已与解析解重叠。"},
]
for f in FIGS[1:]:
    src = Path(f"{WT}/runs/elastic_rod_dalembert") / f["name"]
    (FIGDIR / f["name"]).write_bytes(src.read_bytes())

MARKDOWN = f"""# 交差审计裁定：m16137 CASE-DG-DALEMBERT-ELASTIC-1D（TensorDG 一维弹性杆 d'Alembert 波）

**裁定 = L{LEVEL}**（独立功能实现）。物理与数值实质已达 L4 强度，但"合入 main / 基准登记"两梯缺位，不越级。

## ① 案例介绍
均质线弹性细杆 L=1 m，速度-应力形式一阶对称双曲系统
$\\sigma_t=E v_x,\\ v_t=\\sigma_x/\\rho$，波速 $c=\\sqrt{{E/\\rho}}$。
材料钢：E=210 GPa、ρ=7850 kg/m³、Z=4.0602e7 kg/(m²·s)、**c=5172.19 m/s**（手册带 5100–5200 内）。
初值 double-wave：σ₀(x)=1 MPa·exp(−((x−0.5)/0.08)²)·sin(2π(x−0.5)/0.20)，v₀=0；
按 d'Alembert 分裂为左右两个半幅行波，t_end=0.4/c（各走 0.4 m）。周期域（留 0.5 L 余量）。
判据（tensor_case_pool id=96）：σ 相对 L2 **<2%**。

## ② 怎么做的
- 被测实现：模态 Legendre DG（每单元 p+1 模态），**精确 Riemann upwind 通量**
  σ*=½(σ⁻+σ⁺)+½ρc(v⁻−v⁺)、v*=½(v⁻+v⁺)+(σ⁻−σ⁺)/(2ρc)，RK4 时间推进（dt=cfl·h/(c(2p+1))）。
  边界：periodic / free(σ=0) / fixed(v=0) 用镜像 ghost。
- 参考解：闭式 d'Alembert 双行波（无物理弥散）。
- **本审计的独立性**：只 import 被测求解器与题面初值；参考解、模态重构、L2 范数、能量积分、
  以及用于判别参考解的 **Fourier 谱传播子**全部由审计方重写，不调用 `dalembert_exact/evaluate/relative_error/gate_report`。
- 复现：`git archive 3240063` → 干净副本 → pytest + benchmark(--quick, 83.8 s) + 审计脚本。

## ③ 计算结果（程序化注入，零手抄）

| 项 | ref | got | 误差 | 判 |
|---|---|---|---|---|
| 主判据 σ 相对 L2（P3 N=200 CFL=0.3） | d'Alembert 闭式 | {c['C1_gate_p3_n200']['relative_l2_sigma_auditor']:.4e} | 交付值 {c['C1_gate_p3_n200']['claimed_in_delivery_results_json']:.4e}，相对差 {abs(c['C1_gate_p3_n200']['relative_l2_sigma_auditor']-c['C1_gate_p3_n200']['claimed_in_delivery_results_json'])/c['C1_gate_p3_n200']['claimed_in_delivery_results_json']:.1e} | **PASS** |
| 收敛阶 P1/P2/P3/P4 | p+1 = 2/3/4/5 | {c['C2_convergence']['p1']['observed_order']:.3f} / {c['C2_convergence']['p2']['observed_order']:.3f} / {c['C2_convergence']['p3']['observed_order']:.3f} / {c['C2_convergence']['p4']['observed_order']:.3f} | ≤0.42 | PASS |
| 反射壁应力 fixed | 2.0 | {c['C5_reflection']['fixed']['ratio']:.8f} | 2.2e-7 | PASS |
| 反射壁应力 free | 0.0 | {c['C5_reflection']['free']['ratio']:.3e} | 2.2e-7 | PASS |
| 弹性能漂移（P3 N=200，0.4 L） | E0={c['C4_energy']['energy_initial_auditor']:.8f} | E1={c['C4_energy']['energy_final_auditor']:.8f} | {c['C4_energy']['relative_drift_auditor']:.2e} | PASS |
| 参考解判别性（vs Fourier 谱 PDE 解） | 6 探针点 | max\\|Δ\\|/A = {c['C3_reference_vs_spectral']['max_abs_diff_rel_to_amplitude']:.1e} | — | PASS |

## ④ 结果图
见 figures：审计方独立收敛曲线 + 交付场量剖面。

## ⑤ 注意点
1. **判据是工况特定的**：2% 只在 P3/N=200 成立；交付自披露 P1/N=25 = 38%（每波长仅 5 单元），不是失败而是分辨率不足。
   建议把分辨率前提写进 `tensor_benchmark_cases.tolerance`。
2. **爬梯缺位（非物理问题）**：main(`cc1c2df`) 无 `tensordg/elastic_1d.py`、`tensordg/__init__.py` 未导出、无外部调用方 → L3；
   `tensor_benchmark_cases` 0 行 / `tensorbench_results` 0 行 / `l4-verdict=CLAIMED_NOT_PROVEN` → L4。
3. 交付跑在本机（225 s）无集群作业号，`source_cluster_job` 判据天然红。本单一度递交 `cj1264`（干净检出，pytest+benchmark+独立复算），
   递交时队列 435 位、4.2 min 仍在排队，**未取回结果，不计入证据**（not_run）。
4. 报告溯源链断：交付宣称 `POST /reports → report_c222e0c255b4`，但 `tensor_reports WHERE mission_id=16137` 为 0 行
   （同名报告挂在 16149/16158），`mission.report_id` 空。
5. 文献表格数字 WITHHELD：Graff 1975 正文不可达，交付已自述零转录；仅用 c 的手册带与反射律做独立锚点。

## ⑥ 误差判定（逐项回答"是否 <3%"）
- 主判据 σ 相对 L2：**是**，{c['C1_gate_p3_n200']['relative_l2_sigma_auditor']:.3e} = 0.0000643%，比 3% 低 5 个数量级（余量 ×{c['C1_gate_p3_n200']['margin_vs_tolerance']:.0f} 对 2% 判据）。
- 反射壁应力：fixed 相对误差 1.1e-7、free 绝对误差 2.2e-7·A → **是**。
- 能量守恒：漂移 {c['C4_energy']['relative_drift_auditor']:.1e} → **是**。
- 收敛阶：与设计阶偏差 ≤0.42（P1 超收敛）→ **是**（无隐性降阶）。
- 反面如实：P1/N=25 误差 38% **不 <3%**，属粗网格工况，不构成判据失败但必须记录。
"""


def land_verdict() -> str:
    """Insert the verdict through psql (this sandbox ships a psycopg2 stub only)."""
    payload = json.dumps(EVIDENCE, ensure_ascii=False)
    sql = (
        "INSERT INTO tensor_maturity_verdicts "
        "(repo, scope, mission_id, level, evidence, verdict_by) VALUES "
        f"('{REPO}', '{SCOPE}', {MISSION}, {LEVEL}, $json${payload}$json$, 'board-audit') "
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
        "mission_id=EXCLUDED.mission_id, level=EXCLUDED.level, evidence=EXCLUDED.evidence "
        "RETURNING id, coalesce(gate_state,'pending');"
    )
    out = subprocess.run(["psql", "-U", "postgres", "-d",
                          os.environ.get("TP_PG_DB", "tensorplatform"),
                          "-tA", "-c", sql],
                         capture_output=True, text=True, check=True).stdout.strip()
    vid, state = out.split("|")
    Path(ART, "verdict.json").write_text(json.dumps(
        {"verdict_id": int(vid), "gate_state": state, "repo": REPO, "scope": SCOPE,
         "mission_id": MISSION, "level": LEVEL, "evidence": EVIDENCE},
        indent=2, ensure_ascii=False) + "\n")
    return f"verdict id={vid} gate_state={state}"


def publish() -> str:
    def call(method, path, payload=None, token=None):
        req = urllib.request.Request(BASE + path, method=method)
        req.add_header("Content-Type", "application/json")
        if token:
            req.add_header("Authorization", "Bearer " + token)
        data = json.dumps(payload).encode() if payload is not None else None
        try:
            with urllib.request.urlopen(req, data=data, timeout=180) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode("utf-8", "ignore")[:1500]}

    code, d = call("POST", "/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                           "password": os.environ["TENSOR_AGENT_PASS"]})
    tok = d["access_token"]
    figs = [{"name": f["name"], "caption": f["caption"],
             "png_base64": base64.b64encode((FIGDIR / f["name"]).read_bytes()).decode()}
            for f in FIGS]
    body = {"title": "交差审计裁定 L2：m16137 CASE-DG-DALEMBERT-ELASTIC-1D（TensorDG 一维弹性杆 d'Alembert 波）",
            "summary": ("独立复算确认真求解器与真解析解（σ 相对 L2 6.429e-07，判据 <2% 余量 ×31107，<3% 判据低 5 个数量级；"
                        "收敛阶 2.33/3.41/4.05/5.01；反射律 2.0/0.0；能量漂移 5.5e-10），"
                        "但 main 未合入/包未导出（L3 缺）且 tensor_benchmark_cases 0 行、l4-verdict=CLAIMED_NOT_PROVEN（L4 缺）→ 判 L2。"),
            "scenario_key": "board_audit_m16137", "mission_id": MISSION,
            "tags": ["board-audit", "TensorDG", "CASE-DG-DALEMBERT-ELASTIC-1D", "maturity-L2",
                     "dAlembert", "elastic-wave", "DG"],
            "markdown": MARKDOWN, "figures": figs}
    code, r = call("POST", "/tensor/reports", body, tok)
    Path(ART, "report_post.json").write_text(json.dumps({"status": code, "resp": r},
                                                        indent=2, ensure_ascii=False)[:2000])
    return f"report POST {code} id={r.get('id') or r}"


if __name__ == "__main__":
    print(land_verdict())
    print(publish())