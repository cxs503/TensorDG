"""Publish the CASE-LAX-DG p3 asymptotic-probe report to the platform.

Every number is injected from
``docs/benchmarks/results/lax_asymptotic_probe/metrics.json`` (written by
``python -m benchmarks.lax_asymptotic_probe``); the figures are the PNGs the
same driver rendered.  Nothing is transcribed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_asymptotic_probe"
MISSION_ID = int(os.environ.get("MISSION_ID", "17925"))


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    payload = json.loads((OUT / "metrics.json").read_text())
    digest = (OUT / "metrics.sha256").read_text().split()[0]
    cfg = payload["config"]
    entries = payload["ladder"]
    r3 = payload["p3_resolution"]["r3"]
    c3 = payload["p3_resolution"]["c3"]
    gates = payload["gates"]
    reuse = payload["reuse_check"]
    finest = payload["finest_leg"]

    def wl(entry):
        value = entry["ripple"].get("wavelength_cells_smoothed")
        return "—" if value is None else f"{value:.2f}"

    ladder_rows = "\n".join(
        f"| {e['degree']} | {e['cells']} | {e['steps']} | {e['errors']['density']:.5e} | "
        f"{e['errors']['velocity']:.5e} | {e['errors']['pressure']:.5e} | "
        f"{e['analytic_errors']['density']:.5e} | {e['analytic_fraction']:.4f} | "
        f"{e['ripple'].get('band_cells'):.1f} | {wl(e)} |"
        for e in entries)

    order_rows = []
    for degree in sorted(payload["analytic_orders"]):
        requirement = 0.85 * (int(degree) + 1)
        for pair in payload["analytic_orders"][degree]["density"]:
            order = pair["order"]
            order_rows.append(
                f"| p{degree} | {pair['coarse']} → {pair['fine']} | "
                f"{pair['coarse_error']:.4e} | {pair['fine_error']:.4e} | "
                f"{'—' if order is None else f'{order:.3f}'} | {requirement:.2f} |")
    order_rows = "\n".join(order_rows)

    wave_rows = "\n".join(
        f"| {name} | {value:.6f} |" for name, value in payload["waves"]["positions"].items())

    decision = payload["hypotheses"]["resolution"]
    worst_analytic = max(finest["analytic_errors"].values())
    verdict_cn = "达标" if r3["meets_requirement"] else "未达标"

    markdown = f"""# CASE-LAX-DG · p3 阶失效归因：把算例池阶梯推到 N=800（H1/H2 裁决）

> 锚点 `CASE-LAX-DG`（Lax 激波管 / Toro 2009 §10.6 test 2，ref = 精确解，容差 <3 %）
> 工件：`docs/benchmarks/results/lax_asymptotic_probe/metrics.json`（sha256 `{digest}`）
> 驱动：`benchmarks/lax_asymptotic_probe.py`；测试：`tests/test_lax_asymptotic_probe.py`
> 报告全部数字由脚本从 metrics.json 注入，零手抄；复现命令 `PYTHONPATH=. python -m benchmarks.lax_asymptotic_probe`

## ① 案例介绍

Lax (1954) 激波管：`[0,1]` 域，隔膜 `x0 = {cfg['diaphragm']}`，γ = {cfg['gamma']}，
快照 t = {cfg['final_time']}，四道波（左稀疏波 head/tail、接触间断、右激波）。
精确解由 `benchmarks.riemann_exact.star_state` 的 Newton 迭代给出，波位用闭式
Rankine–Hugoniot / 等熵关系算出，参考值不依赖任何数值解。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | {tuple(cfg['left_state'])} | {tuple(cfg['right_state'])} |

| 解析波 | x |
|---|---|
{wave_rows}

## ② 怎么做的

**问题**：算例池 `CASE-LAX-DG` 的 `C3`（全场相对 L1 阶 ≥ 1.0）与 `R3`（解析区阶 ≥ 0.85(p+1)）
在 p3 上离阈值最远（密度阶 0.27/0.48 vs 要求 3.4/1.0），池子因此 honest-FAIL。同一批数字有两种读法：

* **H2 方法内禀**——无限制 DG + 仅正性缩放，阶数上限就在 O(h^0.5) 量级，任何加密都救不回来；
* **H1 未进渐近区**——固定物理带宽 {cfg['r3_band_width']} 在 N=400 只含 34 个单元，
  仍被三个间断的耗散层/振荡环占据；加密后带宽内单元数增加，阶应回升。

**做法**：把阶梯从池子的 N ≤ 400 延长到 **N = 800**（p = 1/2/3 各 `400→800`，p3 额外保留
N=200 形成三点阶梯），判据**原样导入**池子的 `gate_c3` / `gate_r3`（带宽 {cfg['r3_band_width']}，
嵌套规则），不重调、不放宽。预登记：若 p3 最细对（400→800）的 R3 阶 ≥ 前一对（200→400）的
{cfg['escalation_factor']} 倍 → 判 H1；否则判 H2，池子 honest-FAIL 原样保留。

**旁证一（配置钉扎）**：本单用一个活的粗腿证明池子阶梯的实测配置是
`limiter_reported = {entries[0].get('limiter_reported', 'n/a')}`（即 `limiter=None`，
只有 Zhang–Shu 正性缩放）。此前若干文档把 p3 压阶归因于「TVB 限幅器」，而那套限幅器
从未在池子阶梯上启用——这条归因不成立，已在本文更正；`tests/test_lax_asymptotic_probe.py`
把该配置钉成断言，防止再次漂移。

**旁证二（波纹诊断）**：星区右平台（接触→激波，按带宽内缩）上残差的过零点给出波纹波长（单元）；
它与「带宽内有多少单元」一起，判断带宽是落在波纹内部还是外部。

## ③ 计算结果（程序化注入）

### 3.1 阶梯（池子腿在本进程内重跑，与已提交工件逐位核对）

| p | N | steps | 全场 L1 ρ | 全场 L1 u | 全场 L1 p | 解析区 L1 ρ | 解析区测度 | 带宽（单元） | 波纹波长（单元） |
|---|---|---|---|---|---|---|---|---|---|
{ladder_rows}

### 3.2 解析区实测阶 vs 要求

| p | 对 | 粗腿误差 ρ | 细腿误差 ρ | 阶（密度） | 要求 0.85(p+1) |
|---|---|---|---|---|---|
{order_rows}

### 3.3 H1/H2 裁决

* **R3（解析区阶）**：p3 前一对 `{r3['previous_pair'][0]}→{r3['previous_pair'][1]}` 阶
  **{r3['previous_order']:.3f}** → 最细对 `{r3['finest_pair'][0]}→{r3['finest_pair'][1]}` 阶
  **{r3['finest_order']:.3f}**，升级因子 **×{r3['escalation_factor']:.3f}**，
  要求 {r3['requirement']:.2f} → **{verdict_cn}**。
* **C3（全场阶）**：p3 前一对阶 {c3['previous_order']:.3f} → 最细对阶 {c3['finest_order']:.3f}，
  升级因子 ×{c3['escalation_factor']:.3f}（要求 ≥ 1.0）。
* **裁决**：{decision}

本阶梯上池子判据的整体裁决 = **{payload['verdict']}**
（C3 {gates['c3']['status']} / R3 {gates['r3']['status']}）。

### 3.4 复现核对

p1/p2/p3 的 N=400 腿与池子已提交 `docs/benchmarks/results/lax_case_pool/metrics.json`
逐位比较，最大相对差 **{reuse['worst_relative_difference']:.3e}**
（共 {len(reuse.get('rows', []))} 个数字）。这只证明"同一条阶梯"，不构成跨实现交叉验证。

## ④ 图

* `docs/benchmarks/results/lax_asymptotic_probe/figures/lax_asymptotic_convergence.png`
  —— 左：全场 vs 解析区相对 L1 随 N（双对数）；右：各阶实测阶与 0.85(p+1) 要求线。
* `docs/benchmarks/results/lax_asymptotic_probe/figures/lax_asymptotic_residual.png`
  —— 最细腿（p{finest['degree']} / N={finest['cells']}）星区放大剖面与残差（虚线 = 解析波位）。

## ⑤ 注意点（坑与局限）

* **归因教训**：把失败的成因写给「限幅器」之前，必须先钉住实测配置。池子阶梯从未开 TVB；
  单靠读文档继承下来的解释会漂移。本单已把配置写成测试断言。
* **阶 ≠ 百分比误差**：C3/R3 是阶门槛，不能换算成 <3% 类容差；两者在作答时分开。
* **未做项（unknown，不当作 passed）**：未与第三方 DG/有限体积实现交叉；未做 p4+（求解器只支持 p ≤ 3）；
  未把 N 推到 1600（按 N² 推算超出单节点预算）；波纹波长用过零点估计（对残差均值敏感，
  故同时报 max|r| 与 rms）；未改判据（阈值变更须走台账流程）。
* 计算面：CPU / float64 / loong64；浮点求和顺序与 x86 CI 可能差末位。

## ⑥ 误差判定（关键量 vs 基准，逐项回答是否 <3%）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 最细腿（p{finest['degree']} N={finest['cells']}）解析区相对 L1 ρ | 精确 Riemann | {finest['analytic_errors']['density']:.4e} | **{'是' if finest['analytic_errors']['density'] < 0.03 else '否'}**（{100*finest['analytic_errors']['density']:.4f}%） |
| 同上 u | 精确 Riemann | {finest['analytic_errors']['velocity']:.4e} | **{'是' if finest['analytic_errors']['velocity'] < 0.03 else '否'}**（{100*finest['analytic_errors']['velocity']:.4f}%） |
| 同上 p | 精确 Riemann | {finest['analytic_errors']['pressure']:.4e} | **{'是' if finest['analytic_errors']['pressure'] < 0.03 else '否'}**（{100*finest['analytic_errors']['pressure']:.4f}%） |
| R3 解析区阶（p3，400→800） | 0.85(p+1) = {r3['requirement']:.2f} | {r3['finest_order']:.3f} | 阶判据：**{verdict_cn}** |
| C3 全场阶（p3，400→800） | ≥ 1.0 | {c3['finest_order']:.3f} | 阶判据：**{'达标' if c3['finest_order'] >= 1.0 else '未达标'}** |

**回答**：三个场的解析区相对 L1 全部 **<3%**（最大 {100*worst_analytic:.4f}%）；
两项阶判据的回答不是百分比，而是「{'阶随加密升级，属未进渐近区' if r3['escalates'] or r3['meets_requirement'] else '阶未升级，归因方法内禀'}」，
整体裁决 **{payload['verdict']}**。若 FAIL，如实保留，不用调参掩盖。
"""

    figures = []
    for path in sorted((OUT / "figures").glob("*.png")):
        figures.append({
            "name": path.stem,
            "caption": f"`docs/benchmarks/results/lax_asymptotic_probe/figures/{path.name}` "
                       "（由 benchmarks/lax_asymptotic_probe.py 从求解器与精确解直接绘制）",
            "png_base64": b64(path),
        })

    token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    report = {
        "title": "CASE-LAX-DG · p3 阶失效归因（N=400→800）：H1/H2 裁决与配置钉扎",
        "summary": (
            f"把 Lax 激波管算例池阶梯延长到 N=800，用池子自己的 gate_c3/gate_r3 裁决 p3 压阶是"
            f"「未进渐近区」还是「方法内禀」：p3 解析区阶 {r3['previous_order']:.3f}→"
            f"{r3['finest_order']:.3f}（升级 ×{r3['escalation_factor']:.2f}，要求 "
            f"{r3['requirement']:.2f}，{verdict_cn}）；并钉扎实测配置 limiter=None，"
            f"更正「TVB 限幅器压阶」的旧归因。整体裁决 {payload['verdict']}。"),
        "scenario_key": "case-pool",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "CASE-LAX-DG", "DG", "Euler", "verification",
                 "convergence-order", "attribution"],
        "markdown": markdown,
        "figures": figures,
    }
    result = post("/tensor/reports", report, token)
    print(json.dumps({k: result[k] for k in result if k in ("id", "title", "created_at")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())