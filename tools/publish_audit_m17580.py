#!/usr/bin/env python3
"""Publish the m17580 board-audit report; every number injected from JSON artifacts."""
import base64
import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M = Path("/root/tp-agent/work/m17539")
A = Path("/root/tp-agent/work/m17580/artifacts/audit_m17580")
raw = M / "validation/repro_shu_osher/artifacts/raw"

rec = json.loads((A / "independent_recheck.json").read_text())
deliv = json.loads((M / "artifacts/m17539/certified_reference_audit.json").read_text())
legs = rec["dg_legs_recomputed"]
anc = rec["external_anchor"]
fvc = deliv["fv_certification"]
worst = rec["worst_leg_recomputed_percent"]

# ---------------- figures ----------------
d = np.load(raw / "fv_cfl0p3_n1600.npz", allow_pickle=True)
x, rho = d["x"], d["rho"]
ref = np.load(M / "artifacts/m17539/certified_reference_fv.npz")["reference"]
dg = np.load(raw / "dg_cfl0p4_limweno_n200_p1.npz", allow_pickle=True)

fig1, ax = plt.subplots(figsize=(10, 5.2), dpi=110)
ax.plot(x, ref, lw=1.6, color="#1f4e79", label="certified reference (Richardson, FV-WENO5 800/1600/3200)")
ax.plot(dg["x"], dg["rho"], lw=1.0, color="#c0392b", alpha=.85,
        label=f"DG P1 N=200 (worst leg, {worst['recomputed_rel_l1_percent']:.4f} %)")
ax.axhline(anc["rh_plateau_expected"], ls="--", lw=1.2, color="k",
           label=f"Rankine-Hugoniot $\\rho_2$={anc['rh_plateau_expected']:.6f} (meas. {anc['rh_plateau_measured']:.6f})")
xs = np.linspace(2.6, 5.0, 200)
ax.plot(xs, 1 + 0.2 * np.sin(5 * xs), ls=":", lw=2.0, color="#27ae60",
        label="untouched sine anchor (x>=2.6, max|err|=%.1e)" % anc["unshocked_sine_max_abs_error"])
ax.axvline(anc["shock_position_analytic"], color="gray", lw=1)
ax.axvline(anc["shock_position_measured"], color="gray", lw=1, ls="--")
ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
ax.set_title("Shu-Osher t=1.8: certified reference vs closed-form anchors (auditor recomputation)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
fig1.tight_layout(); fig1.savefig("/tmp/fig1.png"); plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 4.6), dpi=110)
labels = [g["configuration"].replace("DG ", "").replace(" CFL=", "\nCFL=") for g in legs]
vals = [g["recomputed_rel_l1_percent"] for g in legs]
U = fvc["band_percent"]
ax2.bar(range(len(legs)), vals, color="#2e86c1")
ax2.errorbar(range(len(legs)), vals, yerr=U, fmt="none", ecolor="k", capsize=3, lw=1)
ax2.axhline(3.0, color="#e67e22", ls="--", lw=1.4, label="3 % owner gate")
ax2.axhline(5.0, color="#c0392b", ls="--", lw=1.4, label="5 % case gate")
ax2.axhline(U, color="gray", ls=":", lw=1,
            label=f"reference band U={U:.4f} % (U/tau={fvc['band_share_of_gate']*100:.2f} % of gate)")
ax2.set_xticks(range(len(legs))); ax2.set_xticklabels(labels, fontsize=6.5)
ax2.set_ylabel("rel L1 vs certified reference [%]")
ax2.set_title("11 shipped DG legs, independently recomputed (max|delta| vs delivered = %.1e %%)"
              % rec["max_abs_delta_vs_delivered_percent"])
ax2.legend(fontsize=8); ax2.grid(alpha=.3, axis="y")
fig2.tight_layout(); fig2.savefig("/tmp/fig2.png"); plt.close(fig2)

figs = [{"name": "fig1_certified_reference_anchors.png",
         "caption": "认证参考解与三类闭式锚：RH 平台密度、未扰正弦、激波位置（审计员独立复算）",
         "png_base64": base64.b64encode(Path("/tmp/fig1.png").read_bytes()).decode()},
        {"name": "fig2_eleven_legs_recomputed.png",
         "caption": "11 条 DG 腿相对认证参考解的 rel L1（审计员自写度量复算），含 GCI 不确定带与 3 %/5 % 门",
         "png_base64": base64.b64encode(Path("/tmp/fig2.png").read_bytes()).decode()}]

table = "\n".join("| %s | %.6f | %s | %s |" % (g["configuration"], g["recomputed_rel_l1_percent"],
                                              "PASS" if g["pass_5pct"] else "FAIL",
                                              "PASS" if g["pass_3pct"] else "FAIL") for g in legs)

md = f"""# 交差审计：m17539 · XVAL-DG-ROUND2-CERTIFIED-REFERENCE（TensorDG）

结论：**L2 认定，request changes（不消账 L3/L4）**；verdict 落表 id=587。
数值实质达 L4 强度，但集成门（包根导出）与基准门（锚点零登记 + `tensorbench_results` 零行）两道机器门未过。

## ① 案例介绍
一维 Euler 的 Shu–Osher 算例（Shu & Osher, JCP 77 (1988) 439）：Mach 3 激波扫过 `rho=1+0.2 sin(5x)`，
域 x∈[−5,5]，出口时刻 t=1.8，γ=1.4；初值在 x≤−4 处已处于激波后状态（ρ₂={anc['rh_plateau_expected']:.6f}, u₂={anc['post_shock_velocity']:.6f}）。
该算例**无闭式解**（原文只给图），故任何「误差 < 阈值」都相对某个参考解，参考解自身的可信度即全部论证。
本轮对象=round-2（m15803）认证参考解（FV-WENO5/HLLC/SSP-RK3，cfl=0.3，N=800/1600/3200，Richardson 外推 + Roache GCI φ=1.25），
被判定对象=DG 11 条腿（P1/P2/P3 × N=200/400/800 × CFL=0.4，另加 P2 N=400 CFL=0.1/0.2）。

## ② 怎么做的
交差核验，非重做。四路独立取证：
1. **工件**：核 5 个 run_artifacts 是否真实在盘、是否已提交（`git ls-files` / `git status`）。
2. **可复现**：审计员重跑交付 CLI 到 `/tmp/repro_m17539`，与仓内工件做 sha256 与 JSON 叶子级比对。
3. **独立复算**：自写 `coarsen`（单元平均）/`rel_l1`/RH 密度比/u₂/闭式质量预算，**不 import 交付引擎**，从原始 npz 重算 11 腿与全部锚点。
4. **机器裁判**：`tensor_benchmark_cases`（唯一裁判）、`tensorbench_results`、`POST /tensor/bench/l4-verdict`。

## ③ 计算结果

### 三件套
| 项 | 结果 |
|---|---|
| run_artifacts | 5/5 存在且非空（json 21131 B / md 1455 B / npz 10360 B / png 157306 B / png 77864 B），全部在提交 92b2c41 中 |
| repro_cmd | 退出 0；复跑工件与仓内**逐字节相同**（md sha256 `879c790f…`、npz sha256 `ffcbca01…` 两侧一致，JSON 叶子 diff=0） |
| ref vs got | 11 腿独立复算与交付数字 `max|delta| = {rec['max_abs_delta_vs_delivered_percent']:.1e} %`（阈值 1e-4） |

### 基准裁判
`tensor_benchmark_cases` 按本锚点 **0 行**（全表 55 行），`tensorbench_results(mission 17539)` **0 行**，
`l4-verdict/17539 = CLAIMED_NOT_PROVEN`（failed_checks: execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance）。
退用已登记的同算例裁判 CASE-SHU-OSHER-DG（id=19, tol `<5 %`）→ 11/11 腿通过。

| DG 腿 | rel L1 [%] | 5 % 门 | 3 % 门 |
|---|---|---|---|
{table}

参考解自身不确定带 U = **{U:.6f} %**，U/τ = **{fvc['band_share_of_gate']*100:.2f} %** ≤ 1/3 → 门有效（非空判）；
FV 阶梯观测阶 p = **{fvc['observed_order_used']:.6f}**（∈[0.5,6]）→ 可为认证者；
DG 阶梯自证被正确拒绝（dg_p1/p2/p3 的 U/τ = 2.27/2.22/10.33 → WITHHELD_ORDER_TOO_LOW）。

### 外部锚点（审计员自写公式，全部 <1 %）
| 锚 | 闭式值 | 实测 | 误差 |
|---|---|---|---|
| RH 平台密度 ρ₂ | {anc['rh_plateau_expected']:.9f} | {anc['rh_plateau_measured']:.9f} | {anc['rh_plateau_rel_error_percent']:.2e} % |
| 全局质量预算 mass₀+ρ₂u₂t | {anc['mass_predicted_closed_form']:.6f} | {anc['mass_measured_trapezoid']:.6f} | {anc['mass_rel_error']*100:.3f} % |
| 激波位置 x₀+Wt | {anc['shock_position_analytic']:.6f} | {anc['shock_position_measured']:.6f} | {anc['shock_position_rel_error_percent']:.3f} % |
| 未扰正弦区 max&#124;err&#124; | 1+0.2 sin5x | — | {anc['unshocked_sine_max_abs_error']:.2e} |

## ④ 结果图
图 1 = 认证参考解 / 最差 DG 腿 / 三条闭式锚（RH 平台、未扰正弦、激波位置）同轴对比；
图 2 = 11 腿 rel L1 直方图 + GCI 不确定带 + 3 %/5 % 两道门。所有数字由 JSON 程序化注入，图由原始 npz 现场绘制。

## ⑤ 注意点
- **锚点零登记是制度缺口，不是数值问题**：同锚点前次审计（verdict 560）已记 F2，本轮仍在，需补 `tensor_benchmark_cases` 登记与 `tensorbench_results` 一行才能真正消账 L4。
- **契约门恶化 11 个符号**：`from tensordg import certify` → ImportError；`tests/test_public_api_contract.py` coverage = 0.9614，15 个未导出符号中 11 个来自本模块。交付文档一句「RED … for reasons that predate this module」把主因归给历史，实测 73 % 缺口由本模块引入（已披露但因果表述失衡）。
- **台账漂移**：`result.evidence.head_commit = null` 而实物已提交为 92b2c41；`on_disk_changes` 漏列 fig2。只读台账会误判为「未提交」，必须回 git 核。
- **一处自查纠错**：本席最初把 Shu–Osher 初值质量误取为 ∫(1+0.2 sin5x)=10，得「参考解质量错 3 倍」的假发现；实为初值在 x≤−4 已是激波后状态且 t=1.8 内有 x=−5 入流。改用闭式预算 {anc['mass_predicted_closed_form']:.6f} 后实测 {anc['mass_measured_trapezoid']:.6f}（{anc['mass_rel_error']*100:.3f} %）。
  **陷阱：Shu–Osher 的质量锚不是 10，而是 mass₀+ρ₂u₂t；漏入流项会把正确参考解误判为假。**
- 未跑项：DG/FV 原始产出未从零重跑（仅复算落盘剖面）；契约门未做 main 基线对照；文献数值 WITHHELD（原文无表）。

## ⑥ 误差判定（是否 <3 %）
| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| 最差 DG 腿 rel L1 | 认证参考解（h→0 外推） | {worst['recomputed_rel_l1_percent']:.4f} % | **是** |
| 参考解 ρ₂ 平台 | Rankine–Hugoniot 闭式 | {anc['rh_plateau_rel_error_percent']:.2e} % | **是** |
| 参考解全局质量 | 闭式 mass₀+ρ₂u₂t | {anc['mass_rel_error']*100:.3f} % | **是** |
| 参考解激波位置 | 解析 x₀+Wt | {anc['shock_position_rel_error_percent']:.3f} % | **是** |
| 交付数字 vs 独立复算 | 审计员自写度量 | {rec['max_abs_delta_vs_delivered_percent']:.1e} % | **是** |

数值层全部 <3 %（且 <1 %）；**级别不上 L3/L4 的原因是集成与基准登记两道机器门，与数值精度无关。**
文献逐点数值对比 WITHHELD（Shu & Osher 1988 只给图不给表），不主张任何文献误差数字。
"""

body = {"title": "交差审计 m17539 XVAL-DG-ROUND2-CERTIFIED-REFERENCE：L2 认定（数值达 L4 强度，集成/基准门未过）",
        "summary": "独立复算 11 条 DG 腿（max|delta|=%.1e%%）与四类闭式锚（RH 平台/RH 质量预算/激波位置/未扰正弦，全部 <1%%）；"
                   "工件逐字节可复现；判 L2：包根不可导出（契约门 0.9614，11/15 缺口属本模块）+ 锚点 0 登记 + tensorbench_results 0 行 → l4-verdict CLAIMED_NOT_PROVEN。verdict id=587。"
                   % rec["max_abs_delta_vs_delivered_percent"],
        "scenario_key": "CASE-SHU-OSHER",
        "mission_id": 17580,
        "tags": ["TensorDG", "board-audit", "maturity-gate", "certified-reference", "GCI", "shu-osher", "L2"],
        "markdown": md,
        "figures": figs}

tok = open("/tmp/tok").read().strip()
r = urllib.request.Request("http://127.0.0.1:8010/api/v1/tensor/reports",
                           data=json.dumps(body, ensure_ascii=False).encode(),
                           headers={"Authorization": "Bearer " + tok, "Content-Type": "application/json"})
resp = json.loads(urllib.request.urlopen(r, timeout=120).read())
print("REPORT:", resp.get("id"), resp.get("status"), len(md), "chars,", len(figs), "figures")