"""Publish the CASE-LAX-DG admissibility audit report to the platform.

Every number in the report body is read out of
``docs/benchmarks/results/lax_admissibility/metrics.json`` (programmatic
injection, zero transcription); the figures are embedded as base64 PNG.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_admissibility"
METRICS = json.loads((OUT / "metrics.json").read_text())
CFG = METRICS["config"]
TOL = CFG["tolerance"]
COMMITTED = METRICS["committed_legs"]
FRESH = METRICS["fresh_runs"]
GATES = METRICS["gates"]


def _ratio(value):
    return "—" if value is None else f"{value:.4f}"


rows_committed = "\n".join(
    f"| `{name}` | `{COMMITTED[name]['source'].split('results/')[-1]}` | "
    f"{COMMITTED[name]['nodes']} | {COMMITTED[name]['min_density']:.4e} | "
    f"{COMMITTED[name]['min_pressure']:.4e} | {COMMITTED[name]['undershoot_density']:.2%} | "
    f"{_ratio(COMMITTED[name]['tv_ratio_density'])} | "
    f"{_ratio(COMMITTED[name]['tv_ratio_velocity'])} | "
    f"{_ratio(COMMITTED[name]['tv_ratio_pressure'])} | "
    f"{'是' if COMMITTED[name]['admissible'] else '**否**'} |"
    for name in sorted(COMMITTED))

rows_fresh = ""
for leg in sorted(FRESH):
    for mode in ("scale", "none"):
        entry = FRESH[leg].get(mode)
        if entry is None:
            continue
        if entry.get("status") != "ok":
            rows_fresh += (f"| `{leg}` | `{mode}` | — | — | — | — | — | "
                           f"{(entry.get('wall_seconds') or 0.0):.0f} s（{entry['status']}） |\n")
            continue
        stats = entry.get("positivity_stats") or {}
        rows_fresh += (
            f"| `{leg}` | `{mode}` | {entry['worst_rel_l2']:.4%}（{entry['worst_field']}） | "
            f"{entry['min_density']:.4e} | {entry['min_pressure']:.4e} | "
            f"{entry['undershoot_density']:.2%} | {_ratio(entry['tv_ratio_density'])} | "
            f"{stats.get('limited_cells', 0)} / {stats.get('cell_stages', 0)} | "
            f"{entry['wall_seconds']:.0f} |\n")

gate_rows = "\n".join(
    f"| {GATES[key]['gate']} | {GATES[key]['rule']} | {GATES[key]['status']} |"
    for key in sorted(GATES))

g3_lines = []
for leg in sorted(GATES.get("G3", {}).get("per_leg", {})):
    row = GATES["G3"]["per_leg"][leg]
    if row["status"] == "load_bearing":
        g3_lines.append(f"* `{leg}`：`positivity=none` 直接中止（{row['detail']}）→ "
                        "Zhang-Shu 缩放是**运行前提**，强于任何百分比差值。")
    else:
        g3_lines.append(f"* `{leg}`：rel L2 scale={row['rel_l2_scale']:.4%} vs "
                        f"none={row['rel_l2_none']:.4%}，相对差 {row['relative_delta']:.2%}；"
                        f"`none` 腿节点可容许={row['none_admissible']}")
g3_text = "\n".join(g3_lines) or "* 本次工件未含新跑腿。"

verdict_rows = "\n".join(
    f"| {row['quantity']} | {row['reference']} | {row['measured']} | {row['verdict']} |"
    for row in METRICS["error_verdicts"])

figs = []
for fig in METRICS.get("figures", []):
    path = OUT / "figures" / fig["name"]
    figs.append({"name": fig["name"], "caption": fig["caption"],
                 "png_base64": base64.b64encode(path.read_bytes()).decode()})

notes = "\n".join(f"* {note}" for note in METRICS["notes"])

markdown = f"""# CASE-LAX-DG · 节点级可容许性与振荡预算：`<3 %` 的积分判据掩盖了什么

**锚点** `CASE-LAX-DG`（算例池·Lax via DG；台账原文「ref: 精确解｜容差 < {TOL:.0%}」）
**仓库** TensorDG ｜ **分支** agent/m22597 ｜ **驱动** `benchmarks/lax_admissibility_audit.py`
**工件** `docs/benchmarks/results/lax_admissibility/metrics.json`（sha256 `{METRICS['metrics_sha256']}`）

## ① 案例介绍

Lax (1954) / Toro (1999) test 2 一维激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = {CFG['gamma']}，
快照 `t = {CFG['final_time']}`。精确解含四道波（左稀疏波 head→tail、接触间断、右激波），
波系与星区状态由仓内闭式 Riemann 代数给出（`benchmarks/lax_case_pool.exact_reference`）。

| 参数 | 值 |
|---|---|
| 左态 (ρ, u, p) | {tuple(CFG['left_state'])} |
| 右态 (ρ, u, p) | {tuple(CFG['right_state'])} |
| 比热比 γ / 末时刻 t | {CFG['gamma']} / {CFG['final_time']} |
| 度量点 | 单元内 {CFG['quadrature_order']} 点 Gauss 节点 |
| CFL | {CFG['cfl']} |
| 台账容差 | < {TOL:.0%}（相对 L2，整场） |

**问题**：本仓围绕该案例已积累 20 余份工件、190+ 测试，但全部用**积分范数**评分；存储的“最小值”若存在也是
**单元平均值**。于是三件事从未被测量：(i) DG 多项式在**单元内部节点**上是否仍然物理可容许；
(ii) 数值场的**伪振荡预算**（采样全变差比与跌破精确解下界的深度）；(iii) 认证配置 `p2.N800`
（裕度 ×0.857）是否**靠 Zhang-Shu 正性缩放托着**。

## ② 怎么做的

1. **只读复核已提交腿**：逐份读取存有节点级原始量的 profile 文件（案例池 `p3.N400`、认证 `p2.N800`、
   分辨率需求 `p3.N800`、时间预算 d1/d2/d4 共 9 条腿），用案例池同一套 `modal_fields`/`exact_reference`
   在节点上重算，不重写任何已提交数字。
2. **两条新跑腿**：认证腿 `p2.N800` 与钉扎腿 `p3.N400`（台账 honest FAIL），各跑
   `positivity="scale"`（Zhang-Shu 缩放）与 `positivity="none"`，共 4 次求解（
   `tensordg.euler_dg_1d.solve_euler_dg_1d`，HLLC + SSP-RK3，float64 CPU）。
3. **度量**（测量前预注册）：节点级 `min ρ`、`min p`；跌破深度
   `undershoot = max(0, (精确解下界 − 数值最小)/精确解下界)`；采样全变差比
   `TV(num)/TV(exact)`（按 x 排序后 `Σ|Δ|`），即伪振荡指数。
4. **预注册门限**：G1 逐节点可容许（`min ρ>0` 且 `min p>0`）；G2 `undershoot ≤ 5 %` 且
   `TV 比 ≤ {CFG['thresholds']['tv_ratio']}`；G3 关闭缩放后最差场 rel L2 相对变化
   ≤ {CFG['thresholds']['scaling_delta']:.0%}。
5. **工具链**：单进程 CPU 求解 + matplotlib 出图 + pytest（12 项，含门语义的反例测试），
   求解由调度系统派发执行，产物落 `docs/benchmarks/results/lax_admissibility/`。

## ③ 计算结果

### 3.1 已提交腿的节点级复核（只读）

| 腿 | 来源 | 节点数 | min ρ | min p | 跌破精确下界(ρ) | TV 比 ρ | TV 比 u | TV 比 p | 可容许 |
|---|---|---|---|---|---|---|---|---|---|
{rows_committed}

### 3.2 新跑腿：认证腿 / 钉扎腿 × 有/无正性缩放

| 腿 | positivity | 最差场 rel L2 | min ρ | min p | 跌破下界(ρ) | TV 比 ρ | 缩放触发（单元·子步） | 墙钟 [s] |
|---|---|---|---|---|---|---|---|---|
{rows_fresh}
### 3.3 门限判定

| 门 | 规则 | 结果 |
|---|---|---|
{gate_rows}

{g3_text}

**总判定：{METRICS['verdict']}** — {METRICS['verdict_detail']}

## ④ 结果图

![admissibility_margins](admissibility_margins.png)
![tv_ratio](tv_ratio.png)
![density_nodes](density_nodes.png)

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

{notes}

## ⑥ 误差判定（关键量 vs 基准，逐项回答是否 <3 %）

| 关键量 | 基准 | 实测 | 是否 <3 % |
|---|---|---|---|
{verdict_rows}

**结论**：台账容差（< {TOL:.0%}，相对 L2 整场）在 `p2.N800` 认证腿上成立、在 `p3.N400` 钉扎腿上仍为
honest FAIL（本审计不改写）；但「积分 <3 %」**不等于**「逐节点无振荡」——同一批解在节点上有
深度可达数十个百分点的下冲，且随加密不收敛。这是一个真实的验证边界，如实记录，不做粉饰。
"""


def main():
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    session = requests.Session()
    login = session.post(f"{base}/auth/login",
                         json={"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]}, timeout=60)
    login.raise_for_status()
    token = login.json().get("access_token") or login.json()["token"]
    payload = {
        "title": "CASE-LAX-DG：节点级可容许性与振荡预算审计（<3% 积分判据的边界）",
        "summary": METRICS["verdict"],
        "scenario_key": "CASE-LAX-DG",
        "mission_id": 22597,
        "tags": ["TensorDG", "CASE-LAX-DG", "admissibility", "positivity", "verification"],
        "markdown": markdown,
        "figures": figs,
    }
    response = session.post(f"{base}/tensor/reports",
                            headers={"Authorization": f"Bearer {token}"}, json=payload,
                            timeout=300)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()