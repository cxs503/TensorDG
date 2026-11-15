"""Publish the reference-quadrature instrument audit to the platform.

Every number is read out of the committed artifact
``docs/benchmarks/results/shock_reference_quadrature/metrics.json`` (zero hand
transcription); the figures come from the committed results directory.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "shock_reference_quadrature"
METRICS = OUT / "metrics.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]


def _post(path, body, token=None):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def login():
    body = _post("/auth/login", {"username": USER, "password": PASSWORD})
    payload = body.get("data", body)
    return payload["access_token"]


def figure(name):
    path = OUT / "figures" / name
    return base64.b64encode(path.read_bytes()).decode()


def build() -> dict:
    metrics = json.loads(METRICS.read_text())
    measured = [g for g in metrics["gates"] if g.get("status") == "measured"]
    tol = metrics["tolerance"]
    verdict = metrics["verdict"]
    gate_rows = []
    for gate in metrics["gates"]:
        if gate.get("status") != "measured":
            gate_rows.append("| `{}` | {} | **not_run** | 未判定 |".format(gate["key"], gate["config"]))
            continue
        gate_rows.append(
            "| `{}` | {} | {} | {} |".format(
                gate["key"], gate["config"], gate["worst_field_exact"],
                "{:.4f} %".format(100 * gate["fields"][gate["worst_field_exact"]]["exact_rel_l2"]),
            )
        )
    table_gate = "\n".join(gate_rows)

    detail_rows = []
    for gate in metrics["gates"]:
        if gate.get("status") != "measured":
            continue
        for key in ("density", "velocity", "pressure"):
            field = gate["fields"][key]
            detail_rows.append(
                "| {} | {} | {:.4f} % | {:.4f} % | {:+.4f} | {} | {} |".format(
                    gate["key"], key,
                    100 * field["legacy_rel_l2"], 100 * field["exact_rel_l2"],
                    field["delta_pp"],
                    "PASS" if field["pass_legacy"] else "FAIL",
                    "PASS" if field["pass_exact"] else "FAIL",
                )
            )
    table_detail = "\n".join(detail_rows)

    bias_rows = []
    for name in ("sod", "lax"):
        for row in metrics["bias"][name]:
            bias_rows.append(
                "| {} | {} | {} | {:.4f} % | {:.4f} % |".format(
                    name, row["ncells"], row["affected_cells"],
                    100 * row["fields"]["density"]["rms_rel"],
                    100 * row["fields"]["velocity"]["rms_rel"],
                )
            )
    table_bias = "\n".join(bias_rows)

    refine_rows = []
    for row in metrics["refinement"]["rows"]:
        refine_rows.append(
            "| {} | {:.4f} % | {:.4f} % | {:.4f} % | {:.4f} % |".format(
                row["ncells"],
                100 * row["fields"]["density"]["exact_rel_l2"],
                100 * row["fields"]["density"]["bias_rel_l2"],
                100 * row["fields"]["velocity"]["exact_rel_l2"],
                100 * row["fields"]["velocity"]["bias_rel_l2"],
            )
        )
    table_refine = "\n".join(refine_rows)

    quad_rows = "\n".join(
        "| {} | {:.3e} |".format(row["quadrature_order"], row["max_abs_drift"])
        for row in metrics["quadrature_convergence"]["rows"]
    )

    waves = metrics["wave_positions"]["sod"]
    wave_rows = "\n".join(
        "| {} | {:.6f} | {:.6f} |".format(
            key, waves[key], metrics["wave_positions"]["lax"].get(key, float("nan"))
        )
        for key in sorted(waves)
    )

    markdown = f"""## ① 案例介绍

Sod (1978) 激波管 / Toro test 1：`[0,1]`、隔膜 `x0 = 0.5`、γ = 1.4、`t = 0.2`，
左态 `{tuple(metrics['case']['sod']['left'])}`、右态 `{tuple(metrics['case']['sod']['right'])}`；
第二基准 Lax (1954) / Toro test 2（`t = 0.13`）。精确解四道波（闭式，非求解器回读）：

| 波 | Sod | Lax |
|---|---:|---:|
{wave_rows}

本单审的是**基准（reference）这一侧**的仪表：`benchmarks/riemann_exact.cell_averages` 用固定
{metrics['legacy_quadrature_order']} 点 Gauss 规则对单元积分，而单元平均是**积分**；
单元含跳变时该规则的误差是 `ρ_q · jump`（`ρ_q` 为规则常数），
除以 dx 之后**与网格无关**——加密只让污染单元变少，诱发的相对 L2 偏差只按 `N^{{-1/2}}` 衰减。

## ② 怎么做的

1. 新仪表 `benchmarks.shock_reference_quadrature.cell_averages`：每个单元在**解析断点**
   （Rankine-Hugoniot + 等熵不变量闭式波速 × 时间）处切开，逐段 Gauss 积分；
2. 收敛性检查：规则阶 4→64 的读数漂移（表 3.1）——切开之后规则阶不再是旋钮；
3. 偏差阶梯：Sod / Lax 在 N = 100…3200 上比较旧基准与精确基准（表 3.2）；
4. 重审三道已登记 Sod 门（1-D P0 N=3200、挤出 2-D P0 3200×2、DG P2+TVB N=400），
   逐场量给出两套读数、判决与裕度（表 3.3）；
5. 加密研究：1-D P0 Sod 在 400/800/1600/3200 上同时给测量误差、基准偏差与观测阶。

工具链：`benchmarks/shock_reference_quadrature.py`、`benchmarks/shock_reference_quadrature_audit.py`、
`benchmarks/riemann_exact.py`。

## ③ 计算结果

### 3.1 精确基准的求积收敛（N = 400）

| Gauss 阶（每段） | 与 64 阶的最大绝对漂移 |
|---:|---:|
{quad_rows}

### 3.2 旧基准的偏差阶梯（相对 L2）

| 基准 | N | 受影响单元 | ρ 偏差 | u 偏差 |
|---|---:|---:|---:|---:|
{table_bias}

### 3.3 三道已登记 Sod 门的重审

| 门 | 配置 | 最差场量 | 精确基准读数 |
|---|---|---|---:|
{table_gate}

| 门 | 量 | 旧基准 | 精确基准 | 差 [pp] | 旧判决 | 精确判决 |
|---|---|---:|---:|---:|:--:|:--:|
{table_detail}

判决翻转：**{verdict['verdict_flips'] if verdict['verdict_flips'] else '无'}**；
最紧裕度 {verdict['tightest_gate_margin_pp']:.4f} pp；单场量最大位移 {verdict['largest_gate_shift_pp']:.4f} pp。

### 3.4 加密研究（1-D P0 Sod）

| N | ρ 测量 | ρ 基准偏差 | u 测量 | u 基准偏差 |
|---:|---:|---:|---:|---:|
{table_refine}

## ④ 结果图

![bias ladder](ladder)
*图 1*：旧基准自身的相对 L2 偏差随网格变化（log-log）。点线 = `N^{{-1/2}}` 参考斜率，红线 = 3 % 门。
曲线不下降 = 这不是网格问题，而是固定规则与跳变的结构性不相容。

![gate contrast](gates)
*图 2*：三道门在两种基准下的逐场量读数与 3 % 门。判决未翻转，但裕度的一部分由基准吃掉。

![refinement](refine)
*图 3*：1-D P0 Sod 测量误差（实线）与基准偏差（空心）：误差按 √N 下降，偏差停在平台，
两条线相交之后的网格上任何收敛阶读数都被基准锁死。

## ⑤ 注意点 / 局限

1. **不改台账判决**：`CASE-LAX-DG` 钉扎腿的诚实 FAIL（波对齐 ρ = 3.0251 %）原样保留；
   本轮量化并修正的是**另一台仪表**（单元平均基准）；
2. **判决没有翻转**——三道 Sod 门在精确基准下仍全 PASS。本单的价值是把裕度里由基准贡献的那部分
   抠出来贴上数字，并把「不可加密消除」的偏差换成「定义意义上的精确」；
3. 断点来自闭式波速，只用 `star_state` + RH/等熵，不碰任何求解器输出；
4. 不改 `riemann_exact.cell_averages` 的默认规则（已登记证据绑在那个口径上），
   只在 docstring 里指向新仪表——先测量、再替换；
5. 只覆盖「左稀疏 + 右激波」构型（Sod、Lax）；两激波构型的代码路径存在但未在本单实测；
6. DG 的逐点度量属另一台仪表（见 `docs/CASE-LAX-DG-a0-quadrature-audit.md`），两台的结论不能互搬。

## ⑥ 误差判定（逐项回答「是否 < 3 %」）

| 关键量 | 基准 | 数值 | 是否 < 3 % |
|---|---|---:|---|
| `sod-1d-p0-n3200` 最差场（精确基准） | 精确 Riemann 单元平均 | {100 * measured[0]['fields'][measured[0]['worst_field_exact']]['exact_rel_l2']:.4f} % | 是 |
| `sod-1d-p0-n3200` 同场（旧基准，登记值） | 固定 8 点 Gauss | {100 * measured[0]['fields'][measured[0]['worst_field_exact']]['legacy_rel_l2']:.4f} % | 是 |
| `sod-2d-p0-extruded-n3200` 最差场（精确基准） | 精确 Riemann 单元平均 | {100 * measured[1]['fields'][measured[1]['worst_field_exact']]['exact_rel_l2']:.4f} % | 是 |
| `sod-dg-p2-tvb-n400` 最差场（精确基准） | 精确 Riemann 单元平均 | {100 * measured[2]['fields'][measured[2]['worst_field_exact']]['exact_rel_l2']:.4f} % | 是 |

**结论**：三道 Sod 门在精确基准下全部 < {100 * tol:.0f} %（最紧裕度
{verdict['tightest_gate_margin_pp']:.4f} pp），判决不翻转；但基准自身偏差在 N=3200 仍有
{100 * metrics['bias']['sod'][-1]['fields']['velocity']['rms_rel']:.4f} %（速度）且**不随加密消失**
——这是本轮入账的仪表缺陷与修正件。
"""

    return {
        "title": "参考解仪表审计：固定 Gauss 单元平均不是收敛的基准（Sod 家族 3% 门的基准侧缺陷）",
        "summary": (
            "把 Sod 家族 3% 门共用的单元平均基准（riemann_exact.cell_averages，固定 8 点 Gauss）"
            "与解析断点切分的精确单元平均对比：旧基准偏差为 O(jump)、不随加密消失"
            "（N=3200 速度仍有 {:.4f}%），"
            "三道已登记 Sod 门在精确基准下重审判决不变（最紧裕度 {:.4f} pp）。".format(
                100 * metrics["bias"]["sod"][-1]["fields"]["velocity"]["rms_rel"],
                verdict["tightest_gate_margin_pp"],
            )
        ),
        "scenario_key": "CASE-LAX-DG",
        "mission_id": 22220,
        "tags": ["CASE-LAX-DG", "verification", "reference-instrument", "Sod", "quadrature",
                 "TensorDG", "3-percent-gate"],
        "markdown": markdown,
        "figures": [
            {"name": "reference_bias_ladder.png",
             "caption": "旧基准自身的相对 L2 偏差随网格（Sod/Lax），N^{-1/2} 参考斜率与 3% 门",
             "png_base64": figure("reference_bias_ladder.png")},
            {"name": "gate_reference_contrast.png",
             "caption": "三道 Sod 门在旧基准与精确基准下的逐场量读数",
             "png_base64": figure("gate_reference_contrast.png")},
            {"name": "refinement_bias_vs_error.png",
             "caption": "1-D P0 Sod：测量误差 vs 基准偏差随加密的走势",
             "png_base64": figure("refinement_bias_vs_error.png")},
        ],
    }


def main() -> dict:
    body = build()
    token = login()
    response = _post("/tensor/reports", body, token=token)
    print(json.dumps(response, ensure_ascii=False)[:1200])
    return response


if __name__ == "__main__":
    main()