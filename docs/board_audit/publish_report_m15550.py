"""Publish the m15550 board-audit report (six elements, numbers injected from the audit JSON).

Usage: python docs/board_audit/publish_report_m15550.py [--dry]
Env: TENSOR_API_BASE, TENSOR_AGENT_USER, TENSOR_AGENT_PASS, TOKEN(optional)
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def login() -> str:
    tok = os.environ.get("TOKEN")
    if tok:
        return tok
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{API}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    return d.get("access_token") or d["token"]


def fig(name: str, caption: str, path: str) -> dict:
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode(open(path, "rb").read()).decode()}


def main() -> int:
    a = json.load(open(os.path.join(HERE, "audit_shu_osher_m15550.json")))
    rows = "\n".join(
        "| DG P{deg} N={n} | {got:.4f} | {tab:.4f} | {drift:+.1e} | {pass5} | {pass3} | {rmin:.6f} | {mass} |".format(
            deg=l["key"].split("P")[1].split(" ")[0], n=l["ncells"],
            got=l["got_rel_l1_percent"], tab=l["tabulated_rel_l1_percent"], drift=l["drift_pp"],
            pass5="PASS" if l["passes_5pct"] else "FAIL",
            pass3="PASS" if l["passes_3pct"] else "FAIL",
            rmin=l["rho_min"],
            mass="n/a" if l["mass_abs_error_percent_of_initial"] is None
            else f"{l['mass_abs_error_percent_of_initial']:.2e}%")
        for l in a["legs"])
    cfl = ", ".join(f"cfl {c['cfl']}: {c['got_rel_l1_percent']:.4f}%" for c in a["cfl_sweep"])
    rh, mb = a["anchors"]["rh"], a["anchors"]["mass_budget"]

    md = f"""# 交差核验审计 m15550：m15480 · CASE-SHU-OSHER-DG（TensorDG）

**结论一句话**：got 数值真实且在容差内（审计独立重算 9/9 腿，漂移 0.0 pp，0.556%–2.462%，全部 <5%），
判 **L4**；但 `tensor_benchmark_cases` 对 `CASE-SHU-OSHER` **0 行**、`tensorbench_results` 无行，
平台自带探针给出 `CLAIMED_NOT_PROVEN` —— 交付"实体成立、平台登记缺失"。

## ① 案例介绍（物理背景/参数）

Shu & Osher (1988 *JCP* **77** 439；1989 **83** 32 的 test case 3)：γ=1.4 理想气体中一道 Mach-3 正激波
（初始位于 x=−4）向右扫过静止正弦密度波 ρ=1+0.2 sin(5x)，激波下游把正弦波转成高频熵波。
计算域 x∈[−5,5]，两端透射边界，终时 t=1.8。左态 (ρ,u,p)=(3.857143, 2.629369, 10.33333) **不是自由数据**：
审计由 RH 跳跃关系反推，得激波速度 **{rh['shock_speed']:.7f}**、激波 Mach **{rh['shock_mach']:.7f}**，
左态残差 ρ {rh['rho_L_rel_res']:.1e} / u {rh['u_L_rel_res']:.1e} / p {rh['p_L_rel_res']:.1e}，
解析激波位置 x_s = −4 + s·1.8 = **{rh['x_shock_at_t1p8']:.6f}**。
本算例**无闭式解**；闭式可得的是波骨架、静区端头、质量预算。

## ② 怎么做的（方法与工具链）

* **审计口径（本单做的事）**：不重做交付，只核验。审计代码 `docs/board_audit/audit_shu_osher_m15550.py`
  从被审 worktree 的 **raw .npz** 重算每一个误差数，**不 import** 交付件的度量助手
  （`benchmarks.shu_osher_case_pool.coarsen/relative_errors`、`tensordg.cases.shock_entropy.relative_l1_error`），
  自写体积平均 + 相对 L1，避免"用自己的尺子量自己"。被审数据以 404 KB byte-copy 复制进
  `docs/board_audit/data/`，复核可脱离被审树重跑。
* 取证三件套：`artifacts.sha256.json` **16/16 数据工件 sha256 全部命中**；3 张 png 在 `docs/figures/`
  与 `validation/repro_shu_osher/artifacts/` md5 一致。
* 交付物本身：模态 Legendre DG + HLLC + SSP-RK3 + Qiu–Shu WENO 限制器；参考解为独立离散
  FV-WENO5-JS/HLLC/SSP-RK3（N=3200），与 DG 只共享物理通量。
* 工具链：psql 核 `tensor_benchmark_cases` / `tensor_case_pool` / `tensor_reports`，审计脚本 python+numpy，
  集群作业（run_cluster cj1158）验证 repro 可执行性。

## ③ 计算结果（程序化注入，零手抄）

审计独立重算（vs 独立参考 N=3200，判据 <5%）：

| 配置 | got rel L1 [%] | 表值 [%] | 漂移 [pp] | 5% 判据 | 3% 严判 | ρ_min | 质量误差(表值) |
|---|---|---|---|---|---|---|---|
{rows}

CFL 敏感支线（P2 N=400）：{cfl}。
参考解自收敛（N=1600 vs N=3200）：**{a['anchors']['ref_self_convergence_1600_to_3200_rel_l1_percent']:.4f}%**。
闭式锚复核：安静区板值偏差 {a['anchors']['plateau_dev_ref_percent']:.4f}%，未扰动正弦区
{a['anchors']['sine_dev_ref_percent']:.4f}%；质量预算 M0={mb['M0']:.6f}、通量 ρu={mb['flux']:.6f}、
M(1.8)={mb['M_pred']:.6f}。
9 条阶梯 `max |drift vs 表值| = {a['max_abs_drift_vs_table_pp']:.1e} pp`，全部 ρ_min>0（无负密度）。

## ④ 结果图

图 1 审计复算：密度剖面（最佳配置 DG P2 N=800 vs 独立参考）+ 9 条 DG 腿相对 L1 误差与 5%/3% 判据线、
参考解自收敛地板。图 2 交付件收敛曲线（原样引用，未改）。

## ⑤ 注意点（坑与局限）

* **平台登记缺失（F1）**：`tensor_benchmark_cases` 对 `CASE-SHU-OSHER` 与 `CASE-SHU-OSHER-DG` 都 0 行；
  `tensor_case_pool` 只有 id=83（SHU-OSHER-DG/TensorDG，tol `<5%`，status 仍 `dispatched`，mission 15297，
  `linked_anchor` 空）。交付件自带的平台探针 artifact 亦如实写 `CLAIMED_NOT_PROVEN, rows=0`。
  → 平台侧 L4 端点在本次审计**没有**被喂到裁判行；本审计以"独立重算"顶替裁判，**只顶替确认路径，不动数值结论**。
* **图注错位（F3）**：平台报告 id=14555 的 `figures[]` 把 `..._cfl.png` 挂上密度剖面图注、
  `..._profiles.png` 挂上 CFL 图注（文件 md5 与仓库一致 → 发布侧旋转）。
* **图内标题与数据矛盾（F4）**：`plot_cfl` 标题写"N=800"，而 `metrics.json → cfl_sweep` 实为 **N=400, P2**。
* **文档瑕疵（F5）**：报告 ③ 表格有分隔行无表头（12 列无列名）；⑤ 引用"②b"不存在。
* **口径局限（F6）**：<5% 是**跨格式代码验证**（DG vs FV-WENO5），不是实验/文献验证；Shu & Osher 原文该题只给
  曲线图、环境无 PDF → 文献表值 **WITHHELD**，未抄数。
* **未门控项（F7）**：P3 N=800 质量误差 0.3019% of M0 如实入表但未设门；`conservation_drift` 在透射边界下
  按设计为 NaN。
* **未跑完项（如实）**：`smoke_repro.py` 在审计宿主 120 s harness 上限内未跑完（宿主 128 核被并行 pytest 占满），
  `run_all.sh`（约 30 min）未跑；已递交集群 cj1158，交差时仍在排队。**state = unknown/not_run，不记为 passed。**
* **重复派单（F8）**：同一交付已有 m15552 的 verdict（id=337, level 4, gate pending），本单为第二次结榜核验。

## ⑥ 误差判定

| 关键量 | 基准 | got | 误差 | 判据 | 是否 <3% |
|---|---|---|---|---|---|
| 密度剖面 rel L1（最佳 P2 N=800） | 独立 FV-WENO5 N=3200 | {a['best_config']['got_rel_l1_percent']:.4f}% | {a['best_config']['got_rel_l1_percent']:.4f}% | <5% | **是（0.556% < 3%）** |
| 密度剖面 rel L1（最差 P1 N=200） | 同上 | {a['worst_config']['got_rel_l1_percent']:.4f}% | {a['worst_config']['got_rel_l1_percent']:.4f}% | <5% | **是（2.462% < 3%）** |
| 全部 9 配置 | 同上 | 0.556%–2.462% | 同上 | <5% | **9/9 是** |
| 激波位置（DG P2 N=800） | 解析 x_s={rh['x_shock_at_t1p8']:.6f} | 表值 2.4002–2.6071 | ≈0.45%–0.74% | — | 是 |
| 参考解自身误差地板 | N=1600 vs N=3200 | {a['anchors']['ref_self_convergence_1600_to_3200_rel_l1_percent']:.4f}% | — | — | 是 |

**显式回答"是否 <3%"**：是 —— 全部 9 条配置（含最差 2.4615%）均 <3%；判据 <5% 亦全部通过且非事后回填
（案池 id=83 `tol_policy='<5%'` 在交付前登记，仓库 `benchmark_ref_vs_got.json` 逐腿记录）。

**判级**：**L4（基准验证）** —— 完整算例 + 误差分析 + 可复现（sha256 命中、独立重算零漂移、库内集成调用者齐全）。
不判 L5：无文献/实验对标（WITHHELD）、方法为既有 DG+WENO。
**处置**：approve with notes；F1/F2 需平台侧登记返修（登记裁判行 + 绑一次集群作业 + 关闭案池 83），
F3–F5 文档小修。verdict 落表 `tensor_maturity_verdicts` id=340（repo=TensorDG, scope='m15480 CASE-SHU-OSHER',
mission_id=15550, level=4, verdict_by='board-audit'）。
"""

    figs = [
        fig("m15550_audit_profiles_and_gate.png",
            "审计复算：左=最佳配置 DG P2 N=800 密度剖面 vs 独立 FV-WENO5 N=3200；"
            "右=9 条 DG 腿相对 L1 误差与 5%/3% 判据线、参考解自收敛地板（0.101%）",
            os.path.join(HERE, "fig_profiles_and_gate.png")),
        fig("m15550_deliverable_convergence.png",
            "交付件原图（未改）：DG 误差-网格收敛曲线，红线=5% 判据",
            "/root/tp-agent/work/m15480/validation/repro_shu_osher/artifacts/shu_osher_case_pool_convergence.png"),
    ]
    payload = {
        "title": "交差核验 m15550：m15480 CASE-SHU-OSHER-DG（TensorDG）—— 独立重算 9/9 在容差内，判 L4；平台裁判行缺失",
        "summary": ("审计独立重算 9 条 DG 腿（不 import 交付度量助手）：漂移 0.0 pp，rel L1 0.556%–2.462%，"
                    "全部 <5%（亦全部 <3%）；sha256 16/16 命中；判 L4。缺口：tensor_benchmark_cases 对 "
                    "CASE-SHU-OSHER 0 行、无 tensorbench_results → 平台 L4 端点 CLAIMED_NOT_PROVEN；"
                    "另有图注错位/图内 N 标注错误/案池 83 未回写等文档与记账缺陷。"),
        "scenario_key": "CASE-SHU-OSHER-DG",
        "mission_id": 15550,
        "tags": ["board-audit", "m15480", "CASE-SHU-OSHER", "TensorDG", "L4", "adversarial-review"],
        "markdown": md,
        "figures": figs,
    }
    if "--dry" in os.environ.get("PUB_ARGS", ""):
        json.dump(payload, open("/tmp/m15550_report.json", "w"), ensure_ascii=False)
        print("dry ok", len(md), "chars", [f["name"] for f in figs])
        return 0
    token = login()
    req = urllib.request.Request(f"{API}/tensor/reports", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print("published:", r.status, r.read()[:400].decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read()[:600].decode())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())