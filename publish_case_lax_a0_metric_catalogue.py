"""Publish the CASE-LAX-DG A0 metric catalogue report to the platform (report discipline)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs/benchmarks/results/lax_a0_metric_catalogue"
metrics = json.loads((OUT / "metrics.json").read_text())
cat = metrics["catalogue"]
dec = metrics["decomposition"]
gates = metrics["gates"]
ctx = metrics["context"]
out = metrics["outcome"]
tol = metrics["config"]["tolerance"] * 100


def pct(key, field=None):
    entry = cat[key]
    field = field or entry["worst_field"]
    return 100 * entry["fields"][field]["dg"], field, entry["verdict"]


rows = []
for key, human in (("primitive/l2/pointwise_aligned_converged", "密度 · 点值（aligned 收敛求积）"),
                   ("primitive/l2/pointwise", "密度 · 点值（池子 10 阶）"),
                   ("conserved/l2/pointwise", "动量 · 守恒变量点值 L2"),
                   ("primitive/l2/cell_average", "速度 · 单元平均 L2（最差场）"),
                   ("primitive/l2/pointwise_uniform_order6", "密度 · 点值（未收敛 6 阶）")):
    value, field, verdict = pct(key)
    rows.append(f"| {human} | 精确 Riemann 解 | {value:.4f} % | {'是' if verdict == 'PASS' else '**否**'} |")

figs = []
for name, caption in (
    ("metric_catalogue.png", "同一钉扎腿 p3.N400 的多种度量：DG-P3 vs 同网格 P0 截断，红线=台账 3 % 门槛"),
    ("decomposition.png", "点值误差的四项精确分解占场量误差² 的份额（DG 次单元 / 单元均值 / 精确解次单元 / 2×对齐项）"),
    ("subcell_localisation.png", "逐单元次单元内容（密度），集中于四道解析波所在单元"),
):
    path = OUT / "figures" / name
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode(path.read_bytes()).decode()})

markdown = f"""# CASE-LAX-DG · A0 度量目录：判的是「点值」还是「单元平均」

**锚点** `CASE-LAX-DG`（算例池·Lax via DG，台账原文「ref: 精确解｜容差 < {tol:.2f} %」，status=open）
**仓库** TensorDG ｜ **分支** agent/m22337 ｜ **驱动** `benchmarks/lax_a0_metric_catalogue.py`（零求解器重跑，全部读已提交场缓存）

## ① 案例介绍

Lax (1954) 激波管 / Toro (2009) Test 2：`[0,1]`，隔膜 `x0=0.5`，γ=1.4，左态 `(ρ,u,p)=(0.445,0.698,3.528)`，右态 `(0.5,0,0.571)`，快照 `t=0.13`。精确解四道波，解析折点 `0.157637`、`0.287229`、`0.698734`、`0.822312`。钉扎腿 = `p3.N400`（DG-P3，400 单元，DOF 1600，HLLC，SSP-RK3）。

## ② 怎么做的

已审计过**求积轴**（`lax_a0_quadrature_audit`：场内折点使固定 Gauss 规则欠积分，必须按折点切段逼到收敛才可与容差比）。本单审计从未动过的**度量对象轴**——同一个「整场相对 L2」可以取自点值（Gauss 节点）或单元平均，可落在原始变量或守恒变量上、取 L1 或 L2，共 8 组。

三条先登记后跑数的问法：

1. **目录**：8 组度量逐个对 3 % 判 PASS/FAIL，另加 aligned 收敛求积与未收敛 6 阶读数做求积对照；
2. **判别力探针**：对已提交场做**同网格 P0 截断**（守恒变量上只保单元均值、丢高阶模态，再还原原始变量）。若某度量在此操作下逐位不变，它无法区分 P0 与 P3，不能用于裁决；
3. **误差四分**：`∫(q−q_ex)² = ∫(q−q̄)² + ∫(q̄−q̄_ex)² + ∫(q̄_ex−q_ex)² + 2∫(q−q̄)(q̄_ex−q_ex)`（DG 次单元 ⊕ 单元均值误差 ⊕ 精确解次单元 ⊕ 子单元对齐项），全部项用同一套折点对齐求积以保恒等式。

## ③ 计算结果

### 3.1 度量目录（同一钉扎腿）

| 度量对象 | 变量 | 范数 | 最差场 | DG-P3 | 同网格 P0 | 判定 |
|---|---|---|---|---|---|---|
| 点值（未收敛 6 阶） | primitive | L2 | density | {100*cat['primitive/l2/pointwise_uniform_order6']['worst_dg']:.4f} % | {100*cat['primitive/l2/pointwise_uniform_order6']['fields']['density']['p0_truncated']:.4f} % | **PASS（假象）** |
| 点值（池子 10 阶） | primitive | L2 | density | {100*cat['primitive/l2/pointwise']['worst_dg']:.4f} % | {100*cat['primitive/l2/pointwise']['fields']['density']['p0_truncated']:.4f} % | **FAIL** |
| 点值（aligned 收敛） | primitive | L2 | density | {100*cat['primitive/l2/pointwise_aligned_converged']['worst_dg']:.4f} % | {100*cat['primitive/l2/pointwise_aligned_converged']['fields']['density']['p0_truncated']:.4f} % | **FAIL** |
| 点值 | conserved | L2 | momentum | {100*cat['conserved/l2/pointwise']['worst_dg']:.4f} % | {100*cat['conserved/l2/pointwise']['fields']['momentum']['p0_truncated']:.4f} % | **FAIL** |
| 点值 | primitive | L1 | velocity | {100*cat['primitive/l1/pointwise']['worst_dg']:.4f} % | {100*cat['primitive/l1/pointwise']['fields']['velocity']['p0_truncated']:.4f} % | PASS |
| 单元平均 | primitive | L2 | velocity | {100*cat['primitive/l2/cell_average']['worst_dg']:.4f} % | {100*cat['primitive/l2/cell_average']['fields']['velocity']['p0_truncated']:.4f} % | PASS |
| 单元平均 | conserved | L2 | momentum | {100*cat['conserved/l2/cell_average']['worst_dg']:.4f} % | {100*cat['conserved/l2/cell_average']['fields']['momentum']['p0_truncated']:.4f} % | PASS |

### 3.2 排序翻转（DG/P0 比值，<1 表示 DG 更好）

| 度量 | density | velocity | pressure |
|---|---|---|---|
| 点值 L2（aligned 收敛） | {cat['primitive/l2/pointwise_aligned_converged']['fields']['density']['discrimination_ratio']:.3f} | {cat['primitive/l2/pointwise_aligned_converged']['fields']['velocity']['discrimination_ratio']:.3f} | {cat['primitive/l2/pointwise_aligned_converged']['fields']['pressure']['discrimination_ratio']:.3f} |
| 点值 L1 | {cat['primitive/l1/pointwise']['fields']['density']['discrimination_ratio']:.3f} | {cat['primitive/l1/pointwise']['fields']['velocity']['discrimination_ratio']:.3f} | {cat['primitive/l1/pointwise']['fields']['pressure']['discrimination_ratio']:.3f} |
| 守恒变量点值 L2 | {cat['conserved/l2/pointwise']['fields']['density']['discrimination_ratio']:.3f} | {cat['conserved/l2/pointwise']['fields']['momentum']['discrimination_ratio']:.3f} | {cat['conserved/l2/pointwise']['fields']['energy']['discrimination_ratio']:.3f} |
| 单元平均 L2 | {cat['primitive/l2/cell_average']['fields']['density']['discrimination_ratio']:.3f} | {cat['primitive/l2/cell_average']['fields']['velocity']['discrimination_ratio']:.3f} | {cat['primitive/l2/cell_average']['fields']['pressure']['discrimination_ratio']:.3f} |

单元平均度量在同网格 P0 截断下的最差漂移 = **{gates['cell_average_invariance_worst_relative_drift']['value']:.2e}**（→ 该度量对次单元内容完全免疫，比值恒 1.000，不能用于裁决）。点值 L1 上 P0 截断优于 DG-P3；点值 L2 上密度 DG 占优、速度/压力 P0 占优；守恒变量 L2 上 DG 全胜——**「高阶是否更准」在本配置下是度量族性质，不是格式性质**。

### 3.3 误差四分（钉扎腿，折点对齐求积）

| 场 | 总误差² | DG 次单元 | 单元均值项 | 精确解次单元 | 2×对齐项 | 次单元占比 | 恒等式残差 |
|---|---|---|---|---|---|---|---|
| density | {dec['density']['total']:.4e} | {dec['density']['dg_sub_cell']:.4e} | {dec['density']['cell_mean_error']:.4e} | {dec['density']['exact_sub_cell']:.4e} | {2*dec['density']['alignment']:.4e} | {100*dec['density']['dg_sub_cell_share']:.1f} % | {dec['density']['residual']:.1e} |
| velocity | {dec['velocity']['total']:.4e} | {dec['velocity']['dg_sub_cell']:.4e} | {dec['velocity']['cell_mean_error']:.4e} | {dec['velocity']['exact_sub_cell']:.4e} | {2*dec['velocity']['alignment']:.4e} | {100*dec['velocity']['dg_sub_cell_share']:.1f} % | {dec['velocity']['residual']:.1e} |
| pressure | {dec['pressure']['total']:.4e} | {dec['pressure']['dg_sub_cell']:.4e} | {dec['pressure']['cell_mean_error']:.4e} | {dec['pressure']['exact_sub_cell']:.4e} | {2*dec['pressure']['alignment']:.4e} | {100*dec['pressure']['dg_sub_cell_share']:.1f} % | {dec['pressure']['residual']:.1e} |

含解析折点的单元只有 **{dec['density']['kink_cells']} 个（网格的 {100*dec['density']['kink_cells']/400:.2f} %）**，却承载密度 DG 次单元能量的 **{100*dec['density']['dg_sub_cell_in_kink_cells']/dec['density']['dg_sub_cell']:.1f} %**；对齐项为负（密度占 -{abs(100*dec['density']['alignment_share']):.1f} % 的总误差²），说明 DG 的次单元形状在**抵消**精确解的次单元结构，而非纯噪声。

### 3.4 加密趋势（context only，p3.N800）

密度点值 L2 {100*ctx['norms']['pointwise']['density']:.4f} %（×{ctx['norms']['pointwise']['density']/cat['primitive/l2/pointwise_aligned_converged']['fields']['density']['dg']:.3f}），单元平均 {100*ctx['norms']['cell_average']['density']:.4f} %，DG 次单元能量 ×{ctx['decomposition']['density']['dg_sub_cell']/dec['density']['dg_sub_cell']:.3f}。

## ④ 结果图

（三张图见 figures 字段：度量目录条形图、四项分解份额图、次单元内容沿 x 的定位图。）

## ⑤ 注意点

1. 单元平均不是「更宽松的同一把尺子」，而是另一把尺子：它对次单元内容完全免疫（P0 截断漂移 {gates['cell_average_invariance_worst_relative_drift']['value']:.1e}）。用它把 A0 判成 PASS 等于让一阶重构过同一条容差——因此本单**不**主张换度量翻案。
2. P0 截断是探针不是方案：截断在守恒变量上做再还原原始变量，避免 `(ρ,u,p)` 上截断产生非守恒假象。
3. 求积轴不重复审计：点值裁决值取自已提交的 aligned 收敛求积（`lax_a0_quadrature_audit`），未收敛 6 阶读数单列作反面教材（它会制造 2.9696 % 的假 PASS）。
4. 本单不动算例池工件、不动生产腿、不改台账状态；口径变更属台账持有人权限。
5. 纯离线：全部数字来自 400×10 与 800×10 已提交场缓存，无求解器重跑，本机 load15 高企时可执行（< 20 s）。

## ⑥ 误差判定（是否 < {tol:.2f} %）

| 关键量（度量对象） | 基准 | 实测 | < {tol:.2f} % ? |
|---|---|---|---|
{chr(10).join(rows)}

**结论**：{out['statement']} {out['discrimination']} {out['recommendation']}

honest FAIL 如实保留：台账登记口径（整场点值相对 L2）下钉扎腿 `p3.N400` 密度 {100*cat['primitive/l2/pointwise_aligned_converged']['worst_dg']:.4f} % ≥ {tol:.2f} %，未达标；且**不能**通过更换度量对象在诚实前提下翻案。
"""

payload = {
    "title": "CASE-LAX-DG · A0 度量目录：点值 vs 单元平均（verdict 取决于度量对象，FAIL 仍站得住）",
    "summary": ("同一钉扎腿 p3.N400 上把整场相对 L2 的八种测量全部量化：点值族一致 FAIL"
                f"（收敛求积 3.0251 %），单元平均值族一律 PASS（≤0.98 %）但被证明对次单元内容完全免疫"
                f"（同网格 P0 截断漂移 3.2e-15）；并给出点值误差的四项精确分解"
                f"（密度次单元占 78.2 %，4 个折点单元承载 74.3 %，对齐项 -192.8 %）。"
                "同时把 tests/test_lax_1d.py 的假 PASS 闸改成断言 honest FAIL。"),
    "scenario_key": "case-lax-dg",
    "mission_id": 22337,
    "tags": ["TensorDG", "CASE-LAX-DG", "verification", "metric-convention", "honest-fail"],
    "markdown": markdown,
    "figures": figs,
}

base = os.environ["TENSOR_API_BASE"].rstrip("/")
user = os.environ["TENSOR_AGENT_USER"]
password = os.environ["TENSOR_AGENT_PASS"]
login = requests.post(f"{base}/auth/login", json={"username": user, "password": password}, timeout=60)
login.raise_for_status()
token = login.json().get("access_token") or login.json().get("token")
resp = requests.post(f"{base}/tensor/reports", headers={"Authorization": f"Bearer {token}"},
                     json=payload, timeout=180)
print(resp.status_code)
print(resp.text[:2000])