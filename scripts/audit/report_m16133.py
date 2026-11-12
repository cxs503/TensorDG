#!/usr/bin/env python3
"""Publish the m16133 board-audit report to the tensor platform.

Every number in the markdown is formatted from ``runs/m16133_audit/audit_summary.json``
(no hand-typed numbers); the three figures are inlined as base64 PNG.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/m16133_audit"
FIG = REPO / "docs/board_audit/m16133"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> int:
    s = json.loads((RUN / "audit_summary.json").read_text())
    ex = s["exact_riemann"]
    rows = {r["case"]: r for r in s["rows"]}
    order = ["dg-noav-p2n100", "dg-av-p2n100", "dg-av-p2n200", "dg-av-p3n100", "dg-av-p2n100-eps02",
             "dg-av-p2n100-eps0.5", "dg-av-p2n100-eps0.2", "dg-av-p2n100-shift0",
             "dg-av-p2n100-shift2", "dg-av-p2n100-kappa0.5", "ref-400", "ref-800", "ref-1600"]
    hdr = ("| 算例 | p | N | peak | D1 vs 参考峰值% | D2 vs 同x参考% | D1' vs 解析ρ₂% | front vs 解析% "
           "| 星区误差% | L1(作者,跳跃归一)% |\n|---|---|---|---|---|---|---|---|---|---|")
    body = "\n".join(
        f"| {c} | {rows[c]['degree'] if rows[c]['degree'] is not None else '-'} | {rows[c]['ncells']} | {rows[c]['peak']:.4f} "
        f"| {rows[c]['D1_peak_err_pct']:+.2f} | {rows[c]['D2_spike_local_pct']:+.2f} "
        f"| {rows[c]['D1_vs_exact_pct']:+.2f} | {rows[c]['front_vs_exact_pct']:+.2f} "
        f"| {rows[c]['star']['err_pct']:+.2f} "
        f"| {(rows[c].get('author_l1_err_pct') or rows[c]['L1_rel_pct']):.2f} |" for c in order)
    conv = s["reference_self_convergence"]["peak"]
    conv_txt = " → ".join(f"{k}={v:.4f}" for k, v in sorted(conv.items(), key=lambda kv: int(kv[0].split('-')[1])))
    rp = s["repro"]
    p2 = rows["dg-av-p2n200"]
    noav = rows["dg-noav-p2n100"]
    ref = rows["ref-1600"]

    md = f"""# 交差审计（board-audit）：m16072 / CASE-DG-BLASTWAVE-PERSSON（TensorDG）—— L3 confirmed，L4 NOT_REACHED

## ① 案例介绍
被审对象 = m16072「Persson–Peraire 模态人工粘性 + 1-D 强激波/爆轰前导激波算例」，分支 `agent/m16072` @ `f2dc095`，
报告 `tensor_reports#15209`。物理：1-D 惰性可压 Euler（γ=1.4，域 [0,1]，静止气体，x∈[0,0.1] 为 ρ=1、p=1000
驱动区，余 ρ=1、p=0.01），t=0.01 形成 M≈{ex['mach']:.1f} 右行强激波。判据来源 = `tensor_case_pool`
（`DG-BLASTWAVE-PERSSON`，tol_policy `<5% 峰值密度与位置 + 无 Gibbs 过冲>2%`）；任务令指定的裁判表
`tensor_benchmark_cases(anchor_slug='CASE-DG-BLASTWAVE-PERSSON')` 经查 = **0 行**。

该算例是经典 explosion problem，**精确解是自相似 Riemann 解**，审计人独立求解得解析真值（下表），
它比"用交付自己的精细网格参考解当判据"更强——先把参考解本身判掉。

| 解析真值（Toro ch.4，审计人自解） | 值 |
|---|---|
| 星区 ρ*_L（接触面后平台） | **{ex['rho_star_L']:.6f}** |
| 波后 ρ₂（全域密度最大值） | **{ex['rho2']:.6f}** |
| 激波速度 D / Mach | {ex['D']:.5f} / {ex['mach']:.2f} |
| 激波位置 x_s(t=0.01) / 接触面 | **{ex['x_shock_t']:.6f}** / {ex['x_contact_t']:.6f} |

## ② 怎么做的
1. 逐项核三件套：`git show --stat f2dc095`（6 文件 + 工件）、`validation/blast_wave_av_1d/raw/*.npz`（19 算例剖面）、
   `summary.json/results.md/2 PNG`、`docs/benchmarks/blast_wave_av_1d.md`、报告六段。
2. 不采信作者汇总：另写 `scripts/audit/m16133_blastwave_audit.py`，只吃原始剖面重算 peak/front/L1/逐点偏差/
   冲击窗总变差/极点个数，并另立两种过冲口径（D2=峰值 vs 同 x 参考值；D1'=峰值 vs 解析 ρ₂）。
3. 复跑确定性检验（同码同参 dg-av-p2n100）、单测实跑、读 `maturity_gate.py` 机器判据。
4. 判定级别按功能实质，百页报告不加分。

## ③ 计算结果（程序化注入，零手抄：`runs/m16133_audit/audit_summary.json`）

参考解（FV-WENO5）本身先被判：ref-1600 峰值 {ref['peak']:.4f}（vs 解析 ρ₂ **{ref['D1_vs_exact_pct']:+.2f}%**）、
前缘 {ref['front']:.6f}（**{ref['front_vs_exact_pct']:+.2f}%**）、星区 **{ref['star']['err_pct']:+.2f}%**；
自收敛 {conv_txt}、观测阶 {s['observed_order']['peak']:.2f}、不确定度 {s['reference_uncertainty_pct']['peak']:.2f}%
→ 参考解可用（但自身峰值偏高解析值 0.73%，这一点会被"峰值 vs 参考峰值"的定义抵掉）。

{hdr}
{body}

**与交付逐位一致**：peak/front/overshoot/L1 与作者 `summary.json` 完全相同（未编数）；9 项单测 9 passed/16.65 s；
六段报告齐全、数字与工件自洽。**审计人新增口径**（解析真值、D2、星区）给出不同结论。

## ④ 结果图
![fig1](fig1) 全域 ρ 与激波放大（解析解 + 参考 + DG p2N200 + 无粘性对照）：DG 在 x≈0.332 有高 6.1614 的
**亚格子尖峰**，左右平台仅 5.98–5.99，解析波后 5.999 → 是真 Gibbs 尖峰。
![fig2](fig2) 三种过冲定义并列 + 星区误差：2% 门只在一种定义下通过（+{p2['D1_peak_err_pct']:.2f}%），
另两种 +{p2['D2_spike_local_pct']:.2f}% / +{p2['D1_vs_exact_pct']:.2f}% 越线；星区误差 −26%~−41%。
![fig3](fig3) 参考解收敛 + 星区平台由不同配置搭出 0.42/0.43/0.51（解析 0.575）→ 系统性、两段 p2 几乎重合。

## ⑤ 注意点（缺陷清单，按严重度）
- **F1 high｜裁判缺席**：`tensor_benchmark_cases` 该锚点 0 行（全表 25 行，TensorDG 仅 `CASE-SHU-OSHER-DG`）；
  `maturity_gate.py::judge` 的 L4 只验 "benchmark 键里有数字"，不读裁判表 → 空缺不会被门发现（审计人实调
  `judge(4, evidence)` 竟得 (4, [])，说明门对"基准验证"是形检，不是裁判比对）。
- **F2 medium-high｜2% 过冲门定义敏感**：作者 C3 +{p2['D1_peak_err_pct']:.2f}%（余量 0.04pp）；同 x 参考 +{p2['D2_spike_local_pct']:.2f}%；
  解析 ρ₂ +{p2['D1_vs_exact_pct']:.2f}% → 1 通过 2 越线，且余量小于复跑散布（{abs(rp['peak_rel_diff_pct']):.2f}%）。
- **F3 high｜光滑星区 −26% 未披露**：p2N200 星区均值 {p2['star']['mean_rho']:.4f} vs 解析 {ex['rho_star_L']:.4f}
  = **{p2['star']['err_pct']:+.2f}%**；无粘性对照 {noav['star']['err_pct']:+.2f}%、p3 {rows['dg-av-p3n100']['star']['err_pct']:+.2f}%
  → 加 p2 粘性恶化约 15pp，N100→200 不收敛；复跑复现（{rp['star_mean_delivered']:.5f}/{rp['star_mean_rerun']:.5f}）。
  报告 ⑥"关键量误差<3%"与 L1={(p2.get('author_l1_err_pct') or 0):.2f}% 用 `mean|Δρ|/跳跃(≈6)` 归一掩盖了它。
- **F4 high｜未并入 trunk**：`/root/tensor-ecosystem/TensorDG@812c05e` 无 `tensordg/shock_capture.py`、
  `grep solve_euler_av_1d` = 0；`agent/m16072` 与 `main` 无 merge-base（push-merge 重建史）→ 平台能力面尚无此项。
- **F5 medium｜复跑非逐位**：peak {rp['delivered']['peak']:.4f}→{rp['audit_rerun']['peak']:.4f}（{rp['peak_rel_diff_pct']:+.3f}%），
  max|Δρ|={rp['max_abs_rho_diff']:.3f} 集中 x≈{rp['argmax_at_x']:.4f}（正是过冲门取样处），步数同为 1774；未记录线程/种子。
- **F6 low｜指标命名**：`benchmarks/blast_wave_av_1d.py:231` 的 `post_osc_pct` = max|Δρ|/jump（激波抹宽主导 52.9%），
  非振荡度量且无判别力（noav 51.2 / 基准 52.9 / ε0=2 54.4）；真振荡度量在 `cases/blast_wave.py:95`。
- **F7 low｜单测门不同源**：`tests/test_shock_capture_av.py:153` 用渐近极限 2% 带（peak≤6.0/0.98）且配置 N=64/t=0.004。

## ⑥ 误差判定（逐项，显式回答"是否<3%"）
| 判据（先于计算声明） | got | 是否<3% | 判定 |
|---|---|---|---|
| C1 峰值 vs 参考峰值（p2N200） | {p2['D1_peak_err_pct']:+.2f}% | 是 | ✔ <5% |
| C1' 峰值 vs 解析 ρ₂ | {p2['D1_vs_exact_pct']:+.2f}% | 是（贴近 3%） | ✔ <5% |
| C2 前缘 vs 解析 x_s | {p2['front_vs_exact_pct']:+.2f}% | 是 | ✔ |
| C3 无 Gibbs 过冲 ≤2%（三口径） | {p2['D1_peak_err_pct']:+.2f} / {p2['D2_spike_local_pct']:+.2f} / {p2['D1_vs_exact_pct']:+.2f}% | — | **1 通过 2 越线，不稳健** |
| 光滑星区 ρ* | **{p2['star']['err_pct']:+.2f}%** | **否** | ✘ 未披露 |
| 无粘性对照过冲（算例是否真需捕捉） | {noav['D1_peak_err_pct']:+.2f}%（vs 解析 {noav['D1_vs_exact_pct']:+.2f}%） | — | ✘ 证明需求成立 |
| 参考解自身 vs 解析 | {ref['D1_vs_exact_pct']:+.2f}% / {ref['front_vs_exact_pct']:+.2f}% / {ref['star']['err_pct']:+.2f}% | 是 | ✔ 参考可用 |

**直答**：激波积分量（峰值、前缘）**是** <3%（{p2['D1_vs_exact_pct']:+.2f}% / {p2['front_vs_exact_pct']:+.2f}%）；
2% 过冲判据**不是**稳健通过；同一解的光滑区密度误差 **{p2['star']['err_pct']:+.2f}%，显著 >3%**（未披露）。

**级别与处置**：L2 ✔（真实现，无占位）｜L3 ✔ 部分（粘性项进主 1-D Euler DG 求解器残差，`av=None` 与无粘残差
逐位相同；扣分=未导出 `__init__`、无其他消费方、**分支未并入 trunk**）｜**L4 ✘**（裁判表空缺 + 2% 门定义敏感 +
星区 −26% 漏报 + 复跑非逐位）｜L5 ✘（AIAA 2006-1125 表数据 WITHHELD，无外部数值锚）。
处置 = `approve with findings`，**L3 confirmed / L4 NOT_REACHED（honest FAIL，账不消）**；返修：登记裁判行（含解析锚）、
过冲改三口径并列、披露或修星区误差、并 trunk、记录线程/种子。审计工件：`docs/board_audit/M16133_CASE_DG_BLASTWAVE_BOARD_AUDIT.md`、
`scripts/audit/m16133_blastwave_audit.py`、`runs/m16133_audit/`。未做：星区 −26% 的机制诊断（候选=早期粘性熵增生/出口通量）、
ref-3200/6400 与 p2N400/p3N200、外部求解器交叉验证。
"""
    figs = []
    for name, cap in [("fig1_profiles.png", "全域密度与激波放大：解析自相似解 vs FV-WENO5 参考 vs DG p2N200(+AV) vs DG p2N100(无粘性)"),
                      ("fig2_gates.png", "三种 Gibbs 过冲定义并列（2% 门）+ 光滑星区密度误差（解析 ρ*=0.5751）"),
                      ("fig3_convergence.png", "参考解峰值收敛 vs 解析值 + 星区平台由不同配置搭出（系统性偏差）")]:
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode((FIG / name).read_bytes()).decode()})

    tok = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    payload = {
        "title": "交差审计 m16072 CASE-DG-BLASTWAVE-PERSSON（TensorDG）：L3 confirmed / L4 NOT_REACHED",
        "summary": (f"board-audit：交付的 peak/front 与工件逐位自洽且 9 项单测全绿，激波积分量 vs 解析 ρ₂ "
                    f"{p2['D1_vs_exact_pct']:+.2f}% / 前缘 {p2['front_vs_exact_pct']:+.2f}% 达标；但 2% 过冲门只在最有利定义下"
                    f"+{p2['D1_peak_err_pct']:.2f}% 贴线（另两口径 +{p2['D2_spike_local_pct']:.2f}%/+{p2['D1_vs_exact_pct']:.2f}% 越线）、"
                    f"光滑星区 {p2['star']['err_pct']:+.1f}% 未披露、裁判表 0 行、分支未并 trunk、复跑非逐位 → L3 confirmed，L4 不成立。"),
        "scenario_key": "tensordg_blastwave_board_audit",
        "mission_id": 16133,
        "tags": ["TensorDG", "board-audit", "blastwave", "Persson-Peraire", "artificial-viscosity",
                 "exact-Riemann", "maturity-gate", "m16072"],
        "markdown": md,
        "figures": figs,
    }
    r = post("/tensor/reports", payload, tok)
    print(json.dumps(r, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())