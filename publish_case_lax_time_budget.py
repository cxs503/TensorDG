"""Publish the CASE-LAX-DG temporal-budget study to the platform.

Every number is read out of the committed artifact
``docs/benchmarks/results/lax_time_budget_certified/metrics.json`` (zero hand
transcription); figures come from the committed results directory.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_time_budget_certified"
METRICS = OUT / "metrics.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 21380
FIELDS = ("density", "velocity", "pressure")


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
    tol = metrics["config"]["tolerance"]
    share_tol = metrics["config"]["time_error_share"]
    n400, n800 = leg(metrics, "p3.N400"), leg(metrics, "p3.N800")
    checks = metrics["checks"]
    t1 = checks["temporal_budget"]
    ledger = checks["ledger"]
    reuse = checks["reuse"]

    level_rows = "\n".join(
        "| {} | {:.5f} | {} | {:.4e} | {:.5e} | {:.5e} | {:.5e} | {:.5e} (`{}`) | {} |".format(
            row["leg"], level["cfl"], level["steps"], level["dt_mean"],
            level["l2_errors"]["density"], level["l2_errors"]["velocity"],
            level["l2_errors"]["pressure"], max(level["l2_errors"].values()),
            max(level["l2_errors"], key=level["l2_errors"].get),
            "PASS" if max(level["l2_errors"].values()) < tol else "FAIL")
        for row in metrics["legs"] for level in row["levels"])

    field_rows = []
    for row in metrics["legs"]:
        for field in FIELDS:
            data = row["fields"][field]
            reading = (f"share {data['temporal_share']:.3e}"
                       if data["temporal_share"] is not None
                       else f"bound {data['temporal_bound']:.3e}")
            order = "null（不可辨识）" if data["temporal_order"] is None else f"{data['temporal_order']:.3f}"
            field_rows.append(
                f"| {row['leg']} | {field} | {data['relative_l2'][0]:.6e} | "
                f"{data['relative_l2'][-1]:.6e} | {order} | {reading} |")
    field_table = "\n".join(field_rows)

    verdict_rows = "\n".join(
        f"| {name} | {key} | {level['worst_l2']:.5e} | {level['worst_field']} | "
        f"{level['margin_of_tolerance']:.3f} | {level['status']} |"
        for name, entry in ledger["per_leg"].items()
        for key, level in entry["per_level"].items())

    dt_rows = "\n".join(
        f"| {name} | {entry['steps']} | "
        f"{', '.join(f'{r:.4f}' for r in entry['step_ratios'])} | "
        f"{', '.join(f'{r:.4f}' for r in entry['dt_mean_ratios'])} | {entry['status']} |"
        for name, entry in checks["dt_refinement"]["per_leg"].items())

    fine800 = n800["levels"][-1]["l2_errors"]
    base400 = n400["levels"][0]["l2_errors"]
    budget_worst = max(t1["per_leg"].values(), key=lambda b: b["worst"])

    markdown = f"""# CASE-LAX-DG · 认证腿的时间误差预算（P3, N = 400 / 800）

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管，`[0,1]`，γ = {metrics['config']['gamma']}，
隔膜 `x0 = 0.5`，左态 `(ρ,u,p) = {tuple(metrics['config']['left_state'])}`，
右态 `{tuple(metrics['config']['right_state'])}`，快照 `t = {metrics['config']['final_time']}`。
参考值 = 精确 Riemann 解（闭式，不依赖第三方求解器）。

算例池里两条方向相反的关键读数——钉扎腿 `p3.N400` 密度 rel L2 = `{base400['density']:.5e}`（未达 3 %）、
分辨率声明腿 `p3.N800` = `{n800['levels'][0]['l2_errors']['density']:.5e}`（达标）——**都取在单一
CFL = {metrics['config']['cfl']} 下**。池子的时间门（注册门 {share_tol:.0%}）此前只在廉价腿
`N ≤ 200` 上、且只在 L1 范数上评过。也就是说：**认证分辨率上的两个相反判定都没有时间误差预算**。

## ② 怎么做的

1. 网格固定（P3, N = 400 / 800），只动 `dt`：CFL ∈ {{ {', '.join(f"{metrics['config']['cfl']/d:.5f}" for d in metrics['config']['divisors'])} }}，
   即三次精确 2× 细化（与池子 `TIME_CFL_DIVISORS=(1,2,4)` 同口径）。
2. 误差口径完全沿用池子：单元 Gauss 节点（求积阶 {metrics['config']['quadrature_order']}）、Legendre 求积加权、
   全场、不挖波带、不给任何量豁免；参考解由 `benchmarks.lax_case_pool.exact_reference` 供给。
3. 时间诊断复用池子注册实现：`temporal_order`（三次对分的闭式阶）/ `temporal_share`（可辨识时的模型份额）/
   `temporal_bound`（序列不可辨识时的 4× dt 实测敏感度，作诚实上界）。
4. **预登记规则**：{metrics['rule']}
5. 工具链：PyTorch float64（`tensordg.euler_dg_1d`，HLLC + SSP-RK3）、集群调度递交、pytest 绑闸。
   不移动 A0 生产腿、不改任何已提交数字。

## ③ 计算结果（程序化注入）

### 3.1 全部测量点（rel L2，台账度量）

| leg | CFL | steps | dt_mean | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 对 {tol:.0%} 判据 |
|---|---|---|---|---|---|---|---|---|
{level_rows}

### 3.2 时间误差预算（T1）

| leg | 量 | rel L2 @ 最粗 dt | rel L2 @ 最细 dt | 观测时间阶 | 时间份额/上界 |
|---|---|---|---|---|---|
{field_table}

两腿最坏读数 `{budget_worst['worst']:.3e}`（注册门 {share_tol:.0%}）→ **T1 = {t1['status']}**。

### 3.3 台账判定的 dt 稳健性（T2 / T3）

| leg | 层级 | 最差 rel L2 | 最差量 | 容差裕度（×） | verdict |
|---|---|---|---|---|---|
{verdict_rows}

* T2「N = 800 达标不是 dt 假象」= **{ledger['T2_resolution_pass_is_dt_robust']}**
  （三档 CFL 判定一致：`{ledger['per_leg']['p3.N800']['uniform_status']}`）；
* T3「N = 400 不达标不是时间步问题」= **{ledger['T3_pinned_fail_is_not_dt_artifact']}**
  （`{ledger['per_leg']['p3.N400']['uniform_status']}`）。

### 3.4 复用检查（T4）与细化有效性（T5）

基线 CFL 读数 vs 已提交 `lax_resolution_requirement` 工件，最差相对差
`{reuse['worst_relative_difference']:.3e}`（容差 {reuse['tolerance']:.0e}）→ **T4 = {reuse['status']}**。

| leg | steps | 步数比 | dt_mean 比 | T5 |
|---|---|---|---|---|
{dt_rows}

## ④ 结果图

* `lax_time_budget_l2_vs_cfl.png`：两腿 rel L2（ρ/u/p）随 dt 细化 + 3 % 容差线；
* `lax_time_budget_share.png`：各腿各量时间份额 vs 5 % 注册门；
* `lax_time_budget_profiles.png`：基线 CFL 与 4× 细化 dt 的密度剖面叠加（vs 精确解）。

## ⑤ 注意点

1. 只回答时间维度：不重判空间阶、不移动 A0 生产腿；`p3.N400` 的 FAIL 原样保留。
2. 激波主导的范数在 dt 细化下常是平的 → `temporal_order = null`，此时只能报 4× dt 实测敏感度作上界；
   本文按池子注册读法取上界，**不把「测不出」写成「等于零」**。
3. 口径敏感性：换单元平均 / L∞ 会给出不同数字；口径写在工件 `config` 里供重裁。
4. 算力：N = 800 最细档需 {n800['levels'][-1]['steps']} 步，属重计算，递交集群调度执行。

## ⑥ 误差判定（是否 < {tol:.0%}）

| 关键量 | 基准 | 数值 | 相对误差 | < {tol:.0%} ? |
|---|---|---|---|---|
| rel L2 ρ @ p3.N800, 最细 dt | 精确 Riemann 解 | {fine800['density']:.5e} | {fine800['density']*100:.3f} % | {'是' if fine800['density'] < tol else '否'} |
| rel L2 u @ p3.N800, 最细 dt | 精确 Riemann 解 | {fine800['velocity']:.5e} | {fine800['velocity']*100:.3f} % | {'是' if fine800['velocity'] < tol else '否'} |
| rel L2 p @ p3.N800, 最细 dt | 精确 Riemann 解 | {fine800['pressure']:.5e} | {fine800['pressure']*100:.3f} % | {'是' if fine800['pressure'] < tol else '否'} |
| rel L2 ρ @ p3.N400（钉扎，未改） | 精确 Riemann 解 | {base400['density']:.5e} | {base400['density']*100:.3f} % | {'是' if base400['density'] < tol else '否'} |
| 时间误差份额（最坏字段） | 池子注册门 {share_tol:.0%} | {budget_worst['worst']:.3e} | {budget_worst['worst']*100:.3f} % | {'是' if budget_worst['worst'] < share_tol else '否'} |

**结论**：`p3.N800` 在三档 dt 下三量 rel L2 全部
{'达标（< 3 %）' if max(fine800.values()) < tol else '未达标'}，时间误差份额远低于注册门
（{budget_worst['worst']:.2e} vs {share_tol:.0%}），因此该空间分辨率结论不是 `dt` 误差注水的；
`p3.N400` 的 FAIL 在三档 dt 下都成立，即它不是时间步问题、而是分辨率问题——与分辨率研究一致。
总体判定：**{metrics['verdict']}**。
"""

    figures = [
        {"name": "lax_time_budget_l2_vs_cfl.png",
         "caption": "认证两腿（P3 N=400/800）的 rel L2（ρ/u/p）随 dt 细化；虚线 = 台账 3 % 容差",
         "png_base64": figure("lax_time_budget_l2_vs_cfl.png")},
        {"name": "lax_time_budget_share.png",
         "caption": "各腿各量的时间误差份额 vs 池子注册的 5 % 门（symlog，看不清=极小于门）",
         "png_base64": figure("lax_time_budget_share.png")},
        {"name": "lax_time_budget_profiles.png",
         "caption": "基线 CFL 与 4× 细化 dt 的密度剖面叠加 + 精确解：细化 dt 几乎不移动场，误差由空间分辨率主导",
         "png_base64": figure("lax_time_budget_profiles.png")},
    ]

    return {
        "title": "CASE-LAX-DG · 认证腿的时间误差预算：N=800 达标不是 dt 假象、N=400 不达标也不是",
        "summary": (
            "在钉扎腿 p3.N400 与分辨率声明腿 p3.N800 上做三次精确 dt 对分（CFL "
            f"{metrics['config']['cfl']}/2/4），用池子自身的时间门语义复核："
            f"时间误差份额最坏 {budget_worst['worst']:.2e}（注册门 {share_tol:.0%}）；"
            f"p3.N800 三档 CFL 全部 < 3 %（最细 {fine800['density']*100:.3f} %），"
            f"p3.N400 三档全部 FAIL——两个相反判定对 dt 稳健。"
        ),
        "scenario_key": "CASE-LAX-DG",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "CASE-LAX-DG", "temporal-budget", "verification-debt",
                 "adversarial-review", "dt-refinement"],
        "markdown": markdown,
        "figures": figures,
    }


def publish():
    token = login()
    body = build()
    return _post("/tensor/reports", body, token)


if __name__ == "__main__":
    print(json.dumps(publish(), ensure_ascii=False)[:400])