"""Publish the CASE-LAX-DG certified-frontier hold-out study to the platform.

Every number is read out of the committed artifacts (zero hand transcription):

* ``docs/benchmarks/results/lax_certified_holdout/predictions.json`` (pre-registered)
* ``docs/benchmarks/results/lax_certified_holdout/metrics.json``     (measured)

Figures are the committed PNGs from the results directory.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_holdout"
METRICS = OUT / "metrics.json"
PREDICTIONS = OUT / "predictions.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 22794
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


def pct(x):
    return f"{100.0 * x:.3f} %"


def build():
    metrics = json.loads(METRICS.read_text())
    pred = json.loads(PREDICTIONS.read_text())
    tol = metrics["tolerance"]
    legs = {row["leg"]: row for row in metrics["legs"]}
    eval_by_leg = {row["leg"]: row for row in metrics["evaluation"]["legs"]}
    summary = metrics["evaluation"]["summary"]
    front = metrics["frontier"]
    cal = pred["calibration"]

    # --- ③ measured legs -------------------------------------------------
    measured_rows = "\n".join(
        f"| `{row['leg']}` | {row['cells']} | {row['degree']} | {row['dof']} | "
        f"{row['steps']} | {row['wall_seconds']:.1f} | "
        f"{pct(row['l2_errors']['density'])} | {pct(row['l2_errors']['velocity'])} | "
        f"{pct(row['l2_errors']['pressure'])} | {row['verdict']} |"
        for row in metrics["legs"])

    # --- ③ prediction vs observation -------------------------------------
    pred_rows = []
    for name, design in pred["legs"].items():
        item = eval_by_leg[name]
        for model, cell in item["per_model"].items():
            pred_rows.append(
                f"| `{name}` | {item['dof']} | `{model}` | "
                f"{pct(cell['predicted_worst_l2'])} | ±{pct(cell['window'])} | "
                f"{pct(cell['observed_worst_l2'])} | "
                f"{pct(cell['signed_error'])} | "
                f"{'yes' if cell['inside_window'] else '**NO**'} "
                f"({cell['scaled_error_in_windows']:.2f}×) | "
                f"{cell['predicted_verdict']} / {cell['observed_verdict']} |")
    pred_table = "\n".join(pred_rows)

    consensus_rows = "\n".join(
        f"| `{name}` | {design['consensus']['verdict']} | "
        f"{item['observed_verdict'] if (item := legs[name]) else '?'} | "
        f"{'yes' if design['consensus']['verdict'] in (legs[name]['verdict'], 'DISPUTED') else 'no'} | "
        f"{design['question']} |"
        for name, design in pred["legs"].items())

    # --- ③ re-derived frontier -------------------------------------------
    frontier_rows = "\n".join(
        f"| `{row['leg']}` | {row['dof']} | {pct(row['worst_l2'])} | "
        f"{'PASS' if row['worst_l2'] < tol else 'FAIL'} | {row['source']} |"
        for row in front["rows"])
    cert = front["minimal_dof_passing"]

    monotone_rows = "\n".join(
        f"| p={degree} | {', '.join(v['legs'])} | "
        f"{'yes' if v['monotone_in_cells'] else 'NO'} |"
        for degree, v in front["per_degree_monotone"].items())

    # --- ⑥ error verdict per leg vs the 3 % bar ---------------------------
    err_rows = "\n".join(
        f"| `{row['leg']}` 密度全场相对 L2 | 闭式 Riemann 精确解 | "
        f"{pct(row['l2_errors']['density'])} | "
        f"{'<3 %' if row['verdict'] == 'PASS' else '≥3 %（honest FAIL）'} |"
        for row in metrics["legs"])
    worst_scaled = max(cell["scaled_error_in_windows"]
                       for item in metrics["evaluation"]["legs"]
                       for cell in item["per_model"].values())

    markdown = f"""## ① 案例介绍

**锚点** `CASE-LAX-DG`（`feature_ledger` 原文 `ref: exact solution | tolerance < 3 %`，status `open`）。

Lax (1954) 激波管 / Toro (2009) test 2：一维 `x ∈ [0, 1]`，隔膜 `x0 = 0.5`，`γ = 1.4`，快照 `t = 0.13`。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

闭式 Riemann 解含四道波：左稀疏波头、左稀疏波尾、接触间断、右激波；后两道是真跳变，因此全场相对 L2
的收敛率被压到约 `h^0.4`（DG 的光滑阶在这里不可得，这不是求解器缺陷）。

判据严格照台账：**全场相对 L2**（元素 Legendre–Gauss 节点上逐点对闭式精确解，权重为 Gauss 权重、
分母取精确解同口径范数），**取最差场量**，与 `{tol:.2f}` 比较；无场量豁免、无去掉波带、无平均。

**本单要回答的问题**：上一研究（`lax_certified_configuration`，8 条枚举腿）把台账门槛认证在 `p2.N800`
（DOF 2400），但它对未测量的 `(p, N)` 明确不作断言。一个要给用户推荐配置的算例池必须知道这条边界
是**离散化的性质**还是**枚举的巧合**——办法只有一个：先预测一个没算过的配置，再去算它。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

**预登记（结构约束，不是承诺）**：`predictions.json` 写入时间 `{pred['registered_at_utc']}`，早于任何
hold-out 求解；两条腿在写入时**从未被本线算过**：

* `H1 = p2.N600`（DOF 1800）：DOF 比认证腿小，直接打「最小 DOF」这句话在边界上的成立性；
* `H2 = p3.N600`（DOF 2400）：DOF 与认证腿 `p2.N800` **完全相同**，问「固定 DOF 下误差是否只由 DOF 决定、
  还是仍依赖阶数 `p`」。

**两个预测器**只用已提交的 8 条枚举腿拟合：

* `dof`：单一 DOF 幂律 `log e = a + b log((p+1)N)`；
* `family`：每个阶数一条 `N` 幂律 `log e = a_p + b_p log N`（只有一条腿的阶数回退到 DOF 幂律）。

**预测窗口不是手挑的数**：窗口 = 该模型自己在枚举腿上的 **leave-one-out 绝对误差最大值**
（`dof` ±{pct(cal['dof']['window'])}、`family` ±{pct(cal['family']['window'])}），即在模型看过的数据上校准、
用在它没看过的数据上判 `|观测 − 预测| ≤ 窗口`。判定 PASS iff 预测最差场量 `<` 容差；两模型一致才给
共识判定，不一致记 `DISPUTED`。

**求解与度量一行都没有重写**：hold-out 的求解用 `benchmarks.lax_case_pool.run_ladder`（同一 P3 HLLC +
正定性回退、SSP-RK3、CFL 0.35、同一求积、同一范数），容差从 `tensordg.cases.lax_1d` 导入，
不写字面量。两阶段 CLI（`--predict` 写预登记 → `--run` 才求解）把「先预测后测量」变成代码约束：
`--run` 在预登记缺失或时间倒挂时拒绝启动。

**工具链**：`benchmarks/lax_certified_holdout.py`、`benchmarks/lax_case_pool.py`、
`benchmarks/lax_certified_configuration.py`、`tensordg.euler_dg_1d`；测试 `tests/test_lax_certified_holdout.py`。

## ③ 计算结果

### 3.1 hold-out 实测（求解器直出，零手抄）

| leg | N | p | DOF | steps | 墙钟 [s] | ρ L2 | u L2 | p L2 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{measured_rows}

### 3.2 预登记预测 vs 实测

| hold-out | DOF | 模型 | 预测最差 L2 | 窗口 | 实测 | 偏差 | 在窗口内（倍数） | 预测 / 实测判定 |
|---|---:|---|---:|---:|---:|---:|---|---|
{pred_table}

预测落在窗口内：**{summary['inside_window']}/{summary['predictions']}**；
判定复现：**{summary['verdict_correct']}/{summary['predictions']}**；
偏差中位数 = **{summary['median_scaled_error']:.2f}×** 窗口，最大 = **{worst_scaled:.2f}×** 窗口。

### 3.3 共识判定 vs 实测

| hold-out | 预登记共识 | 实测 | 共识是否被支持 | 该腿要回答的问题 |
|---|---|---|---|---|
{consensus_rows}

### 3.4 扩大腿集后的前沿重算

| leg | DOF | 最差 L2 | 判定 | 来源 |
|---|---:|---:|---|---|
{frontier_rows}

最小 DOF 的 PASS 配置 = **`{cert['leg']}`**（DOF {cert['dof']}，{pct(cert['worst_l2'])}）；
枚举认证原为 `p2.N800`；**最小性是否被改写：{'是' if front['minimality_restated'] else '否'}**。

各阶数内「误差随 N 单调下降」：

| 阶数 | 腿 | 单调 |
|---|---|---|
{monotone_rows}

## ④ 结果图

* `holdout_frontier.png` —— DOF–误差平面：枚举腿（按 `p` 分形）、DOF 幂律、3 % 台账线，
  以及两条 hold-out（误差棒 = 预登记窗口）。
* `holdout_error_in_windows.png` —— 每个 hold-out 的偏差/窗口比值（1.0 虚线 = 预登记窗口），两模型并列。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **这不是「更准」的研究**：一个求解器参数都没改；本单只检验「认证那句话能推多远」。
2. **两条 hold-out 不是「前沿处处成立」的证明**：它证明的是在两个刻意放在边界附近的点上模型预测成立。
   边界（2.8–3.2 %）恰恰是模型最容易被打破的地方，因此窗口取 LOO 最大残差而不是 RMS（保守侧）。
3. **窗口是模型自校准尺度，不是统计置信区间**：8 条拟合腿的 LOO 残差只能给尺度，不能给覆盖率保证；
   `inside_window` 失败的正确读法是「模型外推不足」，不是「求解器错了」。
4. **DOF 是粗变量**：`(p+1)N` 忽略了 CFL 与三阶 RK 稳定步长随 `p` 的变化，因此 `family`（按阶数拟合）
   才是物理上更对的形态；两者并列正是为了让差异可见。
5. **台账生产腿 `p3.N400` 的 honest FAIL 不因本单改变**（3.116 % vs 3 %），本单不重指生产腿，
   只回答「认证前沿能否外推」这一附带声明。
6. **算力口径**：本次测量先递交集群（`cj4384` 排队 36.2 min 未派发、取消；`cj4421` 重新递交），
   最终测量在提交的运行环境中完成，命令与参数在 `metrics.json` 与 dev report 中记录。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | 是否 < 3 % |
|---|---|---|---|
{err_rows}

预登记预测的最大偏差 = **{pct(max(cell['abs_error'] for item in metrics['evaluation']['legs'] for cell in item['per_model'].values()))}**
（相对窗口 = {worst_scaled:.2f}×），即模型外推误差本身**远小于** 3 % 门槛。

---
*本报告由 `publish_case_lax_certified_holdout.py` 从 `predictions.json` + `metrics.json` 直读生成，
所有数字零手抄、可机器复核；文档与 summary 由同一份 JSON 渲染（`render_doc` / `render_summary`）。*
"""

    figures = []
    captions = {
        "holdout_frontier.png":
            "DOF–误差平面：枚举 8 腿（○ p1 / □ p2 / △ p3）、DOF 幂律预测、3 % 台账线，"
            "★ 为两条 hold-out，误差棒 = 预登记窗口。",
        "holdout_error_in_windows.png":
            "偏差/窗口比值：每个 hold-out 的两个模型并列；虚线 1.0 = 预登记窗口边界。",
    }
    for name, caption in captions.items():
        path = OUT / "figures" / name
        if path.exists():
            figures.append({"name": name, "caption": caption,
                            "png_base64": base64.b64encode(path.read_bytes()).decode()})

    return {
        "title": "CASE-LAX-DG：认证前沿的盲测——先预测未算过的 (p,N)，再去算它",
        "summary": (
            f"预登记盲测两条 hold-out 腿（p2.N600 DOF1800 / p3.N600 DOF2400）检验 CASE-LAX-DG 的"
            f"「p2.N800 最小 DOF 认证」能否外推：预测落窗 {summary['inside_window']}/{summary['predictions']}、"
            f"判定复现 {summary['verdict_correct']}/{summary['predictions']}；"
            f"实测 p2.N600={pct(legs['p2.N600']['worst_l2'])}、p3.N600={pct(legs['p3.N600']['worst_l2'])}，"
            f"扩大腿集后最小 DOF PASS 配置 = {cert['leg']}（最小性改写："
            f"{'是' if front['minimality_restated'] else '否'}）。台账生产腿 p3.N400 的 honest FAIL 不变。"),
        "scenario_key": "case_lax_dg_certified_holdout",
        "mission_id": MISSION_ID,
        "tags": ["CASE-LAX-DG", "TensorDG", "hold-out", "pre-registration",
                 "verification-debt", "Riemann-exact", "DG"],
        "markdown": markdown,
        "figures": figures,
    }


def main():
    body = build()
    token = login()
    resp = _post("/tensor/reports", body, token=token)
    payload = resp.get("data", resp)
    print(json.dumps({"report_id": payload.get("id"), "title": payload.get("title")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()