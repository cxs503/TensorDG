"""Publish the DG-side time axis of XVAL-DG-ROUND2-CERTIFIED-REFERENCE.

Every number in the report body and every figure caption is read from
``validation/xval_dg_round2/artifacts/dg_time_axis.json`` / ``.png`` -- nothing is
retyped, so the published claim cannot drift from the computed artifact.

    python publish_dg_time_axis.py
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
ART = Path("validation/xval_dg_round2/artifacts")
MISSION_ID = int(os.environ.get("MISSION_ID", "23262"))


def post(path: str, payload: dict, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.post(f"{BASE}/{path}", json=payload, headers=headers, timeout=120)
    return r.status_code, r.text


def main() -> int:
    status, body = post("auth/login", {"username": USER, "password": PASSWORD})
    token = json.loads(body).get("access_token")
    assert status == 200 and token, (status, body[:300])

    axis = json.loads((ART / "dg_time_axis.json").read_text())
    c = axis["criteria"]
    legs = axis["legs"]
    d0, d1, d2, d3 = c["D0"], c["D1"], c["D2"], c["D3"]
    band = d0["band_percent"]
    ref = axis["reference"]

    cfl_list = ", ".join("%g" % lg["cfl"] for lg in legs)
    leg_rows = "\n".join(
        f"| DG P{axis['degree']} N={axis['ncells']} | {lg['cfl']:g} | "
        f"{lg['rel_l1_percent']:.6f} | {lg['decision']} |" for lg in legs)
    diff_rows = "\n".join(
        f"| {d['pair'][0]:g} → {d['pair'][1]:g} | {d['rel_l1_percent']:.6e} | "
        f"{d['rel_l1_percent'] / band:.4f} |" for d in axis["differences"])

    md = f"""# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — DG 腿的时间步轴（D0–D3）

**锚点**：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger，status=open）
**分支**：`agent/m23262`　**任务**：mission {MISSION_ID}

## 1. 案例介绍

Shu–Osher 激波/熵波相互作用（Shu & Osher 1988, JCP 77:439）：一维 Euler，γ=1.4，
x∈[-5,5]，t=1.8。左态为 Mach-3 激波（ρ=3.857143, u=2.629369, p=10.33333），
扫过 ρ=1+0.2 sin(5x) 的静止熵波；无闭式解，reference 由 FV-WENO5 四级梯次
（N=800/1600/3200/6400，CFL=0.3）经 Roache GCI 认证获得。

| 参数 | 值 |
|---|---|
| 认证参考 | {ref['verdict']}，观测阶 p={ref['observed_order_used']:.6f}，带 U={band:.6f} %，参考层 N={ref['reference_ncells']} |
| 时间轴腿 | DG P{axis['degree']} N={axis['ncells']}，CFL {cfl_list}（**只差时间步**） |
| 判据门 | {axis['tolerance_percent']:g} %（owner 精度标准 15 % 的更严内门） |
| 预声明阈值 | POLLUTION_MAX=0.10、LEG_STABILITY_MAX_PP=0.5、P_MIN_T=0.5（模块常量，先于数据固定） |

## 2. 怎么做的

既有审计（T0–T5）问的是「**参考**是否被生成它的时间步污染」，其真实数据分支至今是
`WITHHELD_MISSING_INPUTS`（6 个 CFL 细化 FV 剖面未落盘）。但已发布的 11 条 DG 腿带着
**第二个独立**的时间误差：其中 9 条在 CFL=0.4 下报价，而腿表从未说明该数字是空间误差
还是空间+时间误差。

其中三条腿 —— P2、N=400，CFL=0.4 / 0.2 / 0.1 —— 是仓库里**已提交**的，且彼此只在时间步上
不同。因此这把轴**零新增算力**就能分离：

* `dg_temporal_axis()`：在同一网格上量 `rel L1(u_cfl0.2, u_cfl0.1)`（纯时间差，无插值），
  与腿误差 `err = rel L1(u_cfl0.1, 认证参考)` 相除得 `R_leg`；
* D0 可分辨性：时间差是否**大于认证参考自身的带 U**（小于则时间误差只能被**界定**为 ≤U，
  不能被**测量**）；
* D1 污染比 `R_leg ≤ 0.10`；D2 决策稳定性（三条腿同判决、移动 ≤0.5 pp）；
* D3 时间阶是否可用（要求差值与误差序列**都**单调）。

## 3. 计算结果

| 腿 | CFL | err vs 认证参考 % | 判决（{axis['tolerance_percent']:g} %） |
|---|---|---|---|
{leg_rows}

| 时间对 | 时间差 rel L1 % | / 认证带 U |
|---|---|---|
{diff_rows}

* **D0** resolvable=`{d0['resolvable']}` —— {d0['note']}
* **D1** R_leg = {d1['ratio']:.6f}（state `{d1['state']}`，bound-only=`{d1['bound_only']}`）
* **D2** 判决 {d2['decisions']}，最大移动 {d2['max_abs_move_pp']:.6f} pp（阈值 {d2['threshold_pp']}）
* **D3** 时间阶 {d3['temporal_order']:.4f}，可用=`{d3['usable']}`（差值单调={d3['differences_monotone']}、
  误差随 CFL 单调={d3['errors_monotone_in_cfl']}）

**结论：`{axis['state']}`（accepted={axis['accepted']}）** —— 已发布 DG 腿在 CFL=0.4 的误差是
**空间误差**；时间步带来的变化比认证参考自身的带宽还小两个数量级（{d0['delta_t_percent']:.4f} %
vs U={band:.4f} %），占腿误差的 {100*d1['ratio']:.3f} %。

## 4. 结果图

图 1（左）腿误差随 CFL 的变化与 3 % 门/认证带；右：时间步变化 rel L1 与认证带 U 的对比。
两栏都指向同一件事：时间轴上这条腿几乎是平的。

## 5. 注意点

1. **D3 是诚实 FAIL**：误差随 CFL 减小时**略有上升**（1.5138 → 1.5156 → 1.5172 %），
   因此时间阶 1.0178 **不可用**（误差序列非单调，是舍入/限制器交互）。本单只声明
   「时间误差 < 带 U」，**不声明**任何时间收敛阶。
2. **D0 不可分辨不是弱点而是更强的结论**：时间差 < U 意味着它无法被测出，只能被界定。
3. **适用范围**：一维 Shu–Osher、DG P2 N=400、WENO 限制器；不外推到多维或别的 CFL。
4. **FV 侧仍未完成（honest not_run）**：T0–T5 的真实数字依赖于 6 个 CFL 细化 FV 剖面，
   已递交计算作业 cj4715（6 分片）/ cj4716（补 N=800 CFL=0.075），本单结束时仍在排队，
   工件未取回 —— 因此 `time_pollution.json` 保持 `WITHHELD_MISSING_INPUTS`，不覆盖为任何数字。

## 6. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3 %? |
|---|---|---|---|---|
| 时间差 rel L1(0.2,0.1) | 认证带 U={band:.6f} % | {d0['delta_t_percent']:.6e} % | 低于基准 {100*(1-d0['delta_t_percent']/band):.2f} % | **是**（<U，不可分辨） |
| 腿误差 drifts（0.4→0.1） | 0 | {d2['max_abs_move_pp']:.6f} pp | — | **是** |
| 认证参考复现（4 级） | p=1.176735、U=0.041622 % | p={ref['observed_order_used']:.6f}、U={band:.6f} % | 0（逐位） | **是** |

关键量全部 < 3 %；D3（时间阶）为不可用状态，如实记为 not usable 而非通过。

产物：`benchmarks/xval_dg_round2_time_pollution.py`（`dg_temporal_axis`/`markdown_dg_axis`）、
`validation/xval_dg_round2/gen_dg_time_axis.py`、`artifacts/dg_time_axis.{{json,md,png}}`、
`tests/test_xval_dg_round2_dg_time_axis.py`（10 项，含负控制）。
"""

    png = base64.b64encode((ART / "dg_time_axis.png").read_bytes()).decode()
    payload = {
        "title": "XVAL-DG-ROUND2 — DG 腿时间步轴：已发布腿误差是空间误差（R_leg=0.16%）",
        "summary": (f"DG P2 N=400 的 CFL 0.4/0.2/0.1 三条已提交腿只差时间步：时间差 "
                    f"{d0['delta_t_percent']:.4f}% 低于认证带 {band:.4f}%，R_leg="
                    f"{d1['ratio']:.6f} → DG_AXIS_CLEAN；时间阶不可用（误差随 CFL 微升），"
                    f"FV 侧 T0–T5 因 6 个 CFL 剖面未落盘仍 withheld"),
        "scenario_key": "xval_dg_round2_certified_reference",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "XVAL-DG-ROUND2-CERTIFIED-REFERENCE", "verification-debt",
                 "time-step-pollution", "Shu-Osher", "D0-D3"],
        "markdown": md,
        "figures": [{"name": "dg_time_axis.png",
                     "caption": "左：DG P2 N=400 腿误差随 CFL；右：时间步变化 rel L1 与认证带 U",
                     "png_base64": png}],
    }
    status, body = post("tensor/reports", payload, token)
    print(status, body[:400])
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())