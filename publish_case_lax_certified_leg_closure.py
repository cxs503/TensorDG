"""Publish the CASE-LAX-DG certified-leg closure study to the platform.

Every number is read out of the committed artifact
``docs/benchmarks/results/lax_certified_leg_closure/metrics.json`` (zero hand
transcription); figures come from the committed results directory.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_leg_closure"
METRICS = OUT / "metrics.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 22051
FIELDS = ("density", "velocity", "pressure")
ORDER = "20"


def _post(path, body, token=None):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def login():
    body = _post("/auth/login", {"username": USER, "password": PASSWORD})
    payload = body.get("data", body)
    return payload["access_token"]


def figure(name):
    return base64.b64encode((OUT / "figures" / name).read_bytes()).decode()


def leg(metrics, name):
    return next(item for item in metrics["legs"] if item["leg"] == name)


def build():
    metrics = json.loads(METRICS.read_text())
    cfg = metrics["config"]
    checks = metrics["checks"]
    ledger = checks["ledger"]
    tol = cfg["tolerance"]
    cert = leg(metrics, "p2.N800")
    cheap = leg(metrics, "p2.N400")
    pinned = leg(metrics, "p3.N400")

    level_rows = "\n".join(
        f"| {row['leg']} | {level['cfl']:.5f} | {level['steps']} | "
        f"{level['dt_mean']:.4e} | {level['uniform']['l2']['density']:.5e} | "
        f"{level['aligned'][ORDER]['l2']['density']:.5e} | "
        f"{max(level['aligned'][ORDER]['l2'].values()):.5e} | "
        f"{'PASS' if max(level['aligned'][ORDER]['l2'].values()) < tol else 'FAIL'} |"
        for row in metrics["legs"] for level in row["levels"])

    bias_rows = "\n".join(
        f"| {row['leg']} | {field} | {row['levels'][0]['uniform']['l2'][field]:.5e} | "
        f"{row['levels'][0]['aligned'][ORDER]['l2'][field]:.5e} | "
        f"{100.0 * (row['levels'][0]['uniform']['l2'][field] - row['levels'][0]['aligned'][ORDER]['l2'][field]) / row['levels'][0]['aligned'][ORDER]['l2'][field]:+.2f} % |"
        for row in metrics["legs"] for field in FIELDS)

    time_rows = "\n".join(
        f"| {row['leg']} | {field} | "
        f"{checks['time_budget']['per_leg'][row['leg']][field]['aligned_series'][0]:.5e} | "
        f"{checks['time_budget']['per_leg'][row['leg']][field]['aligned_series'][2]:.5e} | "
        f"{checks['time_budget']['per_leg'][row['leg']][field]['temporal_order_aligned']} | "
        f"{checks['time_budget']['per_leg'][row['leg']][field]['reading']} "
        f"{100.0 * checks['time_budget']['per_leg'][row['leg']][field]['value']:.3f} % |"
        for row in metrics["legs"] for field in FIELDS)

    cert_rows = "\n".join(
        f"| {level['cfl']:.5f} | {level['aligned'][ORDER]['l2']['density']:.5e} | "
        f"{level['aligned'][ORDER]['l2']['velocity']:.5e} | "
        f"{level['aligned'][ORDER]['l2']['pressure']:.5e} | "
        f"{level['aligned'][ORDER]['l2']['density'] * 100:.3f} % | "
        f"{'是' if level['aligned'][ORDER]['l2']['density'] < tol else '否'} |"
        for level in cert["levels"])

    pinned_baseline = pinned["levels"][0]["aligned"][ORDER]["l2"]
    cert_baseline = cert["levels"][0]["aligned"][ORDER]["l2"]
    worst_share = checks["time_budget"]["worst"]
    margin = cert_baseline["density"] / tol

    markdown = f"""## ① 案例介绍

**锚点**：`feature_ledger` `CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 < 3 %」，
status `open`）。**本单要还的验证欠账**：这条 3 % 判据落在三条记录上，但它们从未在同一次测量里
同时闭合「空间求积」与「时间步」两个维度。

Lax (1954) / Toro (2009) Test 2 激波管：域 `[0,1]`，隔膜 `x0 = {0.5}`，γ = {cfg['gamma']}，快照
`t = {cfg['final_time']}`；左态 `(ρ,u,p) = {tuple(cfg['left_state'])}`，右态
`(ρ,u,p) = {tuple(cfg['right_state'])}`。精确解闭式可得（左稀疏波 head/tail、接触间断、右激波），
参考值不依赖任何第三方求解器。

| 记录 | 用途 | 已知弱处 |
|---|---|---|
| `p3.N400` | 钉扎生产腿（honest FAIL） | 求积维由 `lax_a0_quadrature_audit` 闭合、时间维由 `lax_time_budget_certified` 闭合，但分属两支不同驱动 |
| `p2.N800` | **认证配置**（闭合建议要登记的腿） | 只在单一 CFL = {cfg['cfl']}、单一 10 点 Gauss 口径下测过；认证裕度与求积偏差同量级 |
| `p2.N400` | 认证最小性的对照腿（必须 FAIL） | 同上，且从未做 dt 细化 |

## ② 怎么做的

1. **同一份求解器输出，两套测量**（模态 Legendre DG + HLLC + SSP-RK3，float64，单元 10 点 Gauss 取场）：
   * `uniform`：每单元一个 10 点 Gauss 规则 = 已提交口径，必须逐位复现已提交工件（C0）；
   * `aligned`：每个单元按**解析折点**切段（稀疏头 {cfg['breakpoints'][0][1]:.6f}、尾
     {cfg['breakpoints'][1][1]:.6f}、接触 {cfg['breakpoints'][2][1]:.6f}、激波
     {cfg['breakpoints'][3][1]:.6f}，闭式给出、不读数值场），每段 `order` 点 Gauss，
     order ∈ {list(cfg['aligned_orders'])}；`16 vs 20` 是收敛闸，`20` 是收敛读数。
2. **时间维**：网格固定，只把 dt 按 CFL/{list(cfg['divisors'])} 精确对折（沿用池子
   `TIME_CFL_DIVISORS`），于是「时间项占基线收敛误差的份额」可算（`temporal_share`），
   时间阶不可辨识时退回实测 4×dt 上界（`temporal_bound`）。
3. **预登记规则**（跑之前写死在驱动 `benchmarks/lax_certified_leg_closure.py` 里，事后未调）：
   C0 复现已提交工件 ≤1e-9；C0b 收敛读数复现两支独立驱动（a0 aligned-20 ≤1e-9、
   reference-floor `total_exact` ≤1e-6）；C1 aligned 16 vs 20 ≤1e-10；C2 `p2.N800`
   每个 CFL 都 < 3 %；C3 `p2.N400` 每个 CFL 都 ≥ 3 %；C4 时间份额/上界 < 5 %；
   C5 `p3.N400` 仍 FAIL；C6 steps 与 `dt_mean` 随 CFL 除数对折（1.9–2.1）。
4. **不改任何已提交数字**：本驱动只在独立工件里追加二维陈述，池子与 a0 的读数原样保留。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 逐腿 × 逐 CFL（全场相对 L2）

| leg | CFL | steps | `dt_mean` | uniform ρ | converged ρ | converged 最差 | 对 3 % |
|---|---|---|---|---|---|---|---|
{level_rows}

### 3.2 认证腿 `p2.N800` 的三量读数（收敛口径）

| CFL | ρ | u | p | ρ 相对误差 | < 3 % ? |
|---|---|---|---|---|---|
{cert_rows}

### 3.3 已提交口径的求积偏差（基线 CFL）

| leg | 场 | committed | converged | 偏差 |
|---|---|---|---|---|
{bias_rows}

### 3.4 时间误差预算（收敛口径序列，CFL/1 → CFL/4）

| leg | 场 | rel L2 @ 粗 dt | @ 细 dt | 时间阶 | 份额/上界 |
|---|---|---|---|---|---|
{time_rows}

| 闸 | 结果 |
|---|---|
| C0 复现已提交工件（最差相对差 {checks['reuse']['worst_relative_difference']:.2e} ≤ {checks['reuse']['tolerance']:.0e}） | **{checks['reuse']['status']}** |
| C0b 复现两支独立驱动（a0 {checks['cross_driver']['worst_a0_relative_difference']:.2e}、floor {checks['cross_driver']['worst_floor_relative_difference']:.2e}） | **{checks['cross_driver']['status']}** |
| C1 aligned 16 vs 20（最差漂移 {checks['convergence']['worst_relative_drift']:.2e}） | **{checks['convergence']['status']}** |
| C2 `p2.N800` 每个 CFL 都 PASS | **{ledger['C2_certified_pass_is_quadrature_and_dt_robust']}** |
| C3 `p2.N400` 每个 CFL 都 FAIL | **{ledger['C3_cheaper_leg_fail_is_quadrature_and_dt_robust']}** |
| C4 时间份额 < 5 %（最坏 {100.0 * worst_share:.3f} %） | **{checks['time_budget']['status']}** |
| C5 `p3.N400` 仍 FAIL | **{ledger['C5_pinned_honest_fail_survives']}** |
| C6 dt 对折律 | **{checks['step_law']['status']}** |
| **Overall** | **{metrics['verdict']}** |

## ④ 结果图

*图 1*：三条腿「最差场量全场相对 L2」随 CFL 变化，实线 = 收敛（断点对齐）求积、虚线 = 已提交
10 点 Gauss 口径、点线 = 3 % 判据。判决在这一族曲线上位置不变 —— 既不是求积假象也不是 dt 假象。

*图 2*：已提交口径相对收敛口径的偏差（正值 = 高报误差）。偏差量级与认证裕度同阶。

*图 3*：时间项在基线收敛误差中的份额/上界，点线 = 5 % 预算。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **只动测量与 dt，不动格式**：空间格式、CFL 基线、终止时刻全部沿用已提交配置，未引入任何新格式参数。
2. **收敛读数是「断点对齐」而不是「加密」**：简单等分细分对折函数只以 `O(1/M)` 收敛
   （见 `lax_a0_quadrature_audit`），不是收敛手段；只有把折点对齐，读数才与折点位置无关。
3. **`temporal_share` 需要可辨识的时间阶**：三档 dt 序列不单调时退回 `temporal_bound`
   （实测 4×dt 敏感度），后者是**上界**而非份额，表中如实标注，不把「测不出」写成「等于零」。
4. **不移动台账生产腿**：本文不把 `p2.N800` 登记为生产腿，只给出「若按闭合建议登记该腿，
   其 PASS 在求积维与时间维上是否稳」的证据；`p3.N400` 的 honest FAIL 原样保留。
5. **算力**：三条腿 × 三档 dt 共 9 次求解，最细档 p2.N800 需 {cert['levels'][-1]['steps']} 步，
   属重计算，递交集群调度执行（单命令作业）。

## ⑥ 误差判定（关键量 vs 基准，逐项回答是否 < 3 %）

| 关键量 | 基准 | 已提交口径 | 收敛口径 | < 3 % ? |
|---|---|---|---|---|
| `p2.N800` ρ rel L2（基线 CFL） | 精确 Riemann 解 | {cert['levels'][0]['uniform']['l2']['density']:.5e} | {cert_baseline['density']:.5e} | {'是' if cert_baseline['density'] < tol else '否'} |
| `p2.N800` u rel L2（基线 CFL） | 精确 Riemann 解 | {cert['levels'][0]['uniform']['l2']['velocity']:.5e} | {cert_baseline['velocity']:.5e} | {'是' if cert_baseline['velocity'] < tol else '否'} |
| `p2.N800` p rel L2（基线 CFL） | 精确 Riemann 解 | {cert['levels'][0]['uniform']['l2']['pressure']:.5e} | {cert_baseline['pressure']:.5e} | {'是' if cert_baseline['pressure'] < tol else '否'} |
| `p2.N400` ρ rel L2（对照腿，必须 FAIL） | 精确 Riemann 解 | {cheap['levels'][0]['uniform']['l2']['density']:.5e} | {cheap['levels'][0]['aligned'][ORDER]['l2']['density']:.5e} | {'是' if cheap['levels'][0]['aligned'][ORDER]['l2']['density'] < tol else '否（如实 FAIL）'} |
| `p3.N400` ρ rel L2（钉扎腿，未改） | 精确 Riemann 解 | {pinned['levels'][0]['uniform']['l2']['density']:.5e} | {pinned_baseline['density']:.5e} | {'是' if pinned_baseline['density'] < tol else '否（honest FAIL）'} |
| 时间份额（最坏，含上界读法） | 池子注册门 5 % | — | {100.0 * worst_share:.3f} % | {'是' if worst_share < cfg['time_error_share'] else '否'} |

**结论（如实，不粉饰）**：认证配置 `p2.N800` 的 PASS 在**收敛求积**与**三次 dt 对折**下都成立，
收敛口径 ρ 相对误差 {cert_baseline['density'] * 100:.3f} %，对 3 % 的判定余量
×{margin:.3f}（越接近 1 越紧）；`p2.N400` 与钉扎腿 `p3.N400` 在同样两种口径下都仍 FAIL，
属方法层面的 honest FAIL（间断主导的全场误差在有限分辨率下的如实结果），本报告不改变它，
也不把「换口径后更好看」当作达标。
"""

    figures = [
        {"name": "closure_error_vs_cfl.png",
         "caption": "三条腿「最差场量全场相对 L2」vs CFL：实线=收敛（断点对齐）求积、"
                    "虚线=已提交 10 点 Gauss 口径、点线=登记 3 % 判据",
         "png_base64": figure("closure_error_vs_cfl.png")},
        {"name": "closure_quadrature_bias.png",
         "caption": "已提交 10 点 Gauss 口径相对收敛口径的偏差（逐腿逐量，基线 CFL）",
         "png_base64": figure("closure_quadrature_bias.png")},
        {"name": "closure_time_share.png",
         "caption": "时间项占基线收敛误差的份额/上界（三条腿 × 三量）vs 5 % 注册预算",
         "png_base64": figure("closure_time_share.png")},
    ]

    return {
        "title": ("CASE-LAX-DG · 认证腿的二维闭合：p2.N800 的 PASS 既不是求积假象也不是 dt 假象，"
                  "p3.N400 仍是 honest FAIL"),
        "summary": (
            "在认证配置 p2.N800、对照腿 p2.N400 与钉扎腿 p3.N400 上同轮做两件事："
            "①用断点对齐收敛求积（order 20，16 vs 20 漂移 "
            f"{checks['convergence']['worst_relative_drift']:.1e}）复核读数；"
            f"②CFL {cfg['cfl']}/2/4 三次精确 dt 对折分离时间项。"
            f"认证腿三档 CFL 全部 < 3 %（ρ 收敛口径 {cert_baseline['density'] * 100:.3f} %，"
            f"余量 ×{margin:.3f}）；时间份额最坏 {100.0 * worst_share:.3f} %（门 5 %）；"
            f"p3.N400 在两种口径下都仍 FAIL；复现已提交工件最差相对差 "
            f"{checks['reuse']['worst_relative_difference']:.1e}，收敛读数复现两支独立驱动。"
        ),
        "scenario_key": "CASE-LAX-DG",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "CASE-LAX-DG", "verification-debt", "quadrature-convergence",
                 "temporal-budget", "certified-configuration", "adversarial-review"],
        "markdown": markdown,
        "figures": figures,
    }


def publish():
    token = login()
    body = build()
    return _post("/tensor/reports", body, token)


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False)[:400])