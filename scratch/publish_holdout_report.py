"""Publish the CASE-LAX-DG certified-frontier hold-out report to the platform."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PW = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path("/root/tp-agent/work/m21904")
OUT = ROOT / "docs/benchmarks/results/lax_certified_holdout"


def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


login = post("/auth/login", {"username": USER, "password": PW})
TOKEN = login.get("access_token") or login.get("token")

metrics = json.loads((OUT / "metrics.json").read_text())
pred = metrics["predictions"]
tol = metrics["tolerance"]

figures = []
for name, caption in (
    ("holdout_frontier.png",
     "图 1 DOF–误差平面：枚举腿（圆/方/三角按 p=1/2/3）、DOF 幂律拟合、3% 台账线，以及两条 hold-out 腿"
     "（★=实测，误差棒=预登记 LOO 窗口，×=预登记点预测）。"),
    ("holdout_error_in_windows.png",
     "图 2 两条 hold-out 腿的预测偏差 / 预登记窗口，两模型并列；1.0 虚线即预登记窗口，超过即模型外推不足。"),
):
    path = OUT / "figures" / name
    figures.append({"name": name, "caption": caption,
                    "png_base64": base64.b64encode(path.read_bytes()).decode()})

lines = []
add = lines.append
add("# CASE-LAX-DG · 认证前沿的 hold-out 检验：先预测一个没算过的配置，再去算它")
add("")
add("## ① 案例介绍")
add("")
add("Lax (1954) / Toro test 2 激波管：`[0,1]` 域、隔膜 `x0=0.5`、γ=1.4、快照 t=0.13；左态 "
    "`(ρ,u,p)=(0.445,0.698,3.528)`，右态 `(0.5,0,0.571)`；闭式星区 `p*=2.466098`、`u*=1.528723`。"
    "解含四道波：左稀疏波头/尾、接触间断、右激波，后两者是真跳变。")
add("")
add("台账锚点 `CASE-LAX-DG` 验收原文：`ref: 精确解｜容差 <3%`。上一张枚举研究"
    "（`lax_certified_configuration`，8 条腿）认证 `p2.N800`（DOF 2400，最差场量 2.570%）"
    "为最小 DOF 的 PASS 配置，但**明确把话收在枚举网格内**：「对未测量的 (p,N) 不作断言」。"
    "本单检验的正是这句没收的话：前沿能不能外推？")
add("")
add("## ② 怎么做的")
add("")
add(f"1. **预登记**（`predictions.json`，写入 {pred['registered_at_utc']}，早于任何求解）两条从未算过的 hold-out 腿："
    "`H1 = p2.N600`（DOF 1800，比认证腿小，打「最小 DOF」这句话）、"
    "`H2 = p3.N600`（DOF 2400，与认证腿完全同 DOF，问「固定 DOF 下误差是否只由 DOF 决定」）。")
add("2. **两个只用枚举腿拟合的预测器**：`dof`（一条 DOF 幂律）、`family`（每阶一条 N 幂律）。")
add("3. **窗口不是手挑的数**：窗口 = 该模型在枚举腿上的 leave-one-out 最大绝对误差，"
    "「在它看过的数据上校准、用在它没看过的数据上判」。")
add("4. 求解与度量**全部 import 自** `benchmarks.lax_case_pool.run_ladder`（同一求解器 P3 HLLC+正定性回退、"
    "SSP-RK3、CFL 0.35、10 点 Legendre–Gauss、全场相对 L2、取最差场量），容差从 `tensordg.cases.lax_1d` 导入。")
add("5. 两阶段 CLI 结构性地保证预登记：`--predict` 写预登记 → `--run` 才求解，且预登记缺失/时间倒挂时拒绝启动。")
add("")
add("## ③ 计算结果")
add("")
add("### 3.1 hold-out 实测（求解器直出，程序化注入）")
add("")
add("| leg | N | p | DOF | steps | 墙钟[s] | ρ L2 | u L2 | p L2 | 判定 |")
add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for row in metrics["legs"]:
    add(f"| `{row['leg']}` | {row['cells']} | {row['degree']} | {row['dof']} | "
        f"{row['steps']} | {row['wall_seconds']:.1f} | "
        f"{100*row['l2_errors']['density']:.3f}% | {100*row['l2_errors']['velocity']:.3f}% | "
        f"{100*row['l2_errors']['pressure']:.3f}% | {row['verdict']} |")
add("")
add("### 3.2 预登记预测 vs 实测")
add("")
add("| hold-out | DOF | 模型 | 预测最差 L2 | 窗口 | 实测 | 偏差 | 在窗口内 | 预测/实测判定 |")
add("|---|---:|---|---:|---:|---:|---:|---|---|")
for row in metrics["evaluation"]["legs"]:
    for model, item in row["per_model"].items():
        add(f"| `{row['leg']}` | {row['dof']} | `{model}` | "
            f"{100*item['predicted_worst_l2']:.3f}% | ±{100*item['window']:.3f}% | "
            f"{100*item['observed_worst_l2']:.3f}% | {100*item['abs_error']:.3f}% | "
            f"{'yes' if item['inside_window'] else 'NO'} | "
            f"{item['predicted_verdict']} / {item['observed_verdict']} |")
summary = metrics["evaluation"]["summary"]
add("")
add(f"预测落在预登记窗口内 **{summary['inside_window']}/{summary['predictions']}**；"
    f"判定复现 **{summary['verdict_correct']}/{summary['predictions']}**；"
    f"偏差中位数 **{summary['median_scaled_error']:.2f}×** 窗口。")
add("")
add("### 3.3 扩大腿集后的前沿重算")
add("")
add("| leg | DOF | 最差 L2 | 判定 | 来源 |")
add("|---|---:|---:|---|---|")
for row in metrics["frontier"]["rows"]:
    add(f"| `{row['leg']}` | {row['dof']} | {100*row['worst_l2']:.3f}% | "
        f"{'PASS' if row['worst_l2'] < tol else 'FAIL'} | {row['source']} |")
certified = metrics["frontier"]["minimal_dof_passing"]
add("")
add(f"最小 DOF 的 PASS 配置在扩大腿集后为 **`{certified['leg']}`**（DOF {certified['dof']}，"
    f"{100*certified['worst_l2']:.3f}%）；枚举认证原为 `p2.N800`；"
    f"最小性是否被改写：**{'是' if metrics['frontier']['minimality_restated'] else '否'}**。")
add("")
add("## ④ 结果图")
add("")
add("图 1 / 图 2 见附；数据源 `docs/benchmarks/results/lax_certified_holdout/figures/`。")
add("")
add("## ⑤ 注意点（局限与适用范围）")
add("")
add("1. 本单没有改任何求解器参数，改动只在「认证那句话能推到多远」。")
add("2. 两条 hold-out 不足以证明前沿处处成立，只证明在**刻意选在边界附近**的两个点上模型预测成立；"
    "边界（2.8–3.2%）正是最容易破的地方，故窗口取 LOO **最大**残差而非 RMS（保守侧）。")
add("3. 窗口是模型自校准尺度，不是统计置信区间；`inside_window` 失败的正确读法是「模型外推不足」，"
    "而非「求解器错了」。")
add("4. `(p+1)N` 折算忽略 CFL/时间步随 p 的变化（三阶 RK 稳定步长随 p 收紧），"
    "所以按阶数分别拟合的 `family` 在物理上更对；两模型并列正是为了让这个差异可见。")
add("")
add("## ⑥ 误差判定（是否 <3%）")
add("")
add("| 关键量 | 基准 | 数值 | <3%？ |")
add("|---|---|---|---|")
for row in metrics["legs"]:
    add(f"| `{row['leg']}` 最差场量 L2 | 闭式 Riemann 精确解 | "
        f"{100*row['worst_l2']:.3f}% | {'是' if row['verdict'] == 'PASS' else '否（honest）'} |")
add(f"| 台账生产腿 `p3.N400` 密度 L2（未改） | 闭式 Riemann 精确解 | 3.116% | 否（honest FAIL 仍在账） |")
add(f"| hold-out 预测偏差 | 模型自身 LOO 窗口 | 见 3.2 | "
    f"{'是' if summary['inside_window'] == summary['predictions'] else '部分否'} |")
add("")

body = {
    "title": "CASE-LAX-DG 认证前沿的 hold-out 检验：预登记预测两条未算过的 (p,N) 腿并实测",
    "summary": ("枚举研究的「最小 DOF 认证配置」只对枚举网格成立。本单预登记两条未测量的 (p,N) 腿"
                "（p2.N600 = DOF 1800；p3.N600 = DOF 2400 = 认证腿同 DOF），用只拟合枚举腿的两个误差模型"
                "（DOF 幂律 / 分阶幂律）预测其最差场量相对 L2，窗口取模型自身 LOO 最大残差，"
                "再用同一求解器与同一度量实测比对，并按扩大腿集重算前沿。"),
    "scenario_key": "case-lax-dg-certified-holdout",
    "mission_id": 21904,
    "tags": ["CASE-LAX-DG", "TensorDG", "hold-out", "pre-registration",
             "verification-debt", "certification-frontier"],
    "markdown": "\n".join(lines),
    "figures": figures,
}
result = post("/tensor/reports", body, TOKEN)
print(json.dumps(result, ensure_ascii=False)[:600])