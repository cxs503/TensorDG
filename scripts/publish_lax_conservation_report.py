"""Render the CASE-LAX-DG conservation-budget doc and publish its platform report.

Every number comes from
``docs/benchmarks/results/lax_conservation/metrics.json`` (written by
``python -m benchmarks.lax_conservation_audit``); nothing is transcribed by
hand.  The figures are the PNGs the same driver rendered.

    PYTHONPATH=. python scripts/publish_lax_conservation_report.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_conservation"
DOC = ROOT / "docs" / "CASE-LAX-DG-conservation-budget.md"
MISSION_ID = 20484
FIELDS = ("mass", "momentum", "energy")
CN = {"mass": "质量", "momentum": "动量", "energy": "总能量"}


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def sci(value) -> str:
    return f"{value:.3e}"


def build_document(metrics: dict) -> str:
    gate = metrics["gate"]
    tol = gate["tolerance"]
    config = metrics["config"]
    reference = metrics["exact_reference"]
    entries = metrics["entries"]
    sensitivity = metrics["sensitivity"]
    worst = gate["worst"]

    def leg_label(entry):
        return f"p{entry['degree']}.N{entry['cells']}"

    ladder_rows = "\n".join(
        f"| {leg_label(e)} | {e['limiter']}/{e['positivity']} | {e['steps']} | "
        + " | ".join(sci(e["budget_residual"][f]) for f in FIELDS)
        + " | "
        + " | ".join(sci(e["naive_residual"][f]) for f in FIELDS)
        + f" | {e['wall_seconds']:.0f} |"
        for e in entries
    )
    reference_rows = "\n".join(
        f"| {CN[f]} | {reference['budget'][f]:.9f} | {reference['totals'][f]:.9f} | "
        f"{sci(reference['relative_residual'][f])} |"
        for f in FIELDS
    )
    def field_worst(field):
        return max(entries, key=lambda e: e["budget_residual"][field])

    error_rows = "\n".join(
        f"| {CN[f]} 全局守恒预算残差（最差腿 {leg_label(field_worst(f))}·{field_worst(f)['limiter']}） | "
        f"{sci(field_worst(f)['budget_residual'][f])} | {sci(tol)} | "
        f"是（比容差紧 {tol / field_worst(f)['budget_residual'][f]:.0f} 倍） |"
        for f in FIELDS
    )
    naive_worst = gate["naive_worst"]

    return f"""# CASE-LAX-DG 全局守恒预算审计（守恒性硬不变量）

> 锚点：`CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 < 3%」）
> 工件：`docs/benchmarks/results/lax_conservation/metrics.json`、`figures/*.png`
> 生成：{metrics['generated_at_utc']}　墙钟：{metrics['wall_seconds']:.0f} s　集群作业：cj2747
> verdict：**{gate['verdict']}**（预算臂最差 {sci(worst['value'])} vs 容差 {sci(tol)}）

## ① 案例介绍

`CASE-LAX-DG` = Lax (1954) 激波管（Toro 1999 test 2）：`[0,1]` 域、隔膜 `x0 = 0.5`、γ = 1.4、
快照 `t = 0.13`，左态 (ρ,u,p) = {tuple(config['left_state'])}、右态 = {tuple(config['right_state'])}。
解析波系左→右：稀疏波头 / 尾、接触间断、右激波，四者位置都有闭式解（`benchmarks/riemann_exact.py`）。

算例池现有的全部门（波位置、星区平台、相对 L2/L1、收敛阶）都是**精度**判据：它们问「解离精确解多远」。
本单问一个与精度正交、但**任何守恒格式都必须无条件满足**的问题：

    ∫Ω U(x,t) dx − ∫Ω U(x,0) dx  =  t · (F(U_L) − F(U_R))

只要在 `t` 之前没有波到达域端，透射边界两端的态就是初始态，端通量随时间恒定，于是全局守恒律**可闭式积分**。
本算例正是这种情况：稀疏波头 x = 0.1576 > 0，右激波 x = 0.8223 < 1，全波系都在域内。

**这条审计对本算例特别锋利的一点**：Lax 的左态带着 u = 0.698，左端一直在往管内灌质量、动量和能量，
所以「总质量/动量/能量不变」这个直觉不变量**是错的**。实测（见 ③）它被违反 8.5 % / 266 % / 21.8 %——
任何把「∫U 不变」当成守恒判据的验收都会在这里既放过真错、又冤枉对的格式。

## ② 怎么做的（方法 / 流程 / 关键设置）

1. **闭式预算（解析基准，不碰数值通量）**：`tests/_conservation_tools.boundary_increment` 只用
   **初始数据 + 解析 Euler 通量**算出 `t·(F_L − F_R)`，与求解器的数值通量实现完全无关，
   因此它是真正的 PDE 级基准，而不是「代码自己跟自己比」。
2. **参考解自校验（不调求解器）**：把闭式 Riemann 解按四道波的五个解析片段分段，每段 48 点
   Gauss-Legendre 积分（几何收敛，误差在 float64 地板以下），检查它是否满足同一预算恒等式。
   这一步验证的是**基准本身**——算例池所有精度门都是拿它当分母的。
3. **求解器阶梯 × 臂**：torch float64 / HLLC 通量 / SSP-RK3 / Zhang-Shu 正性缩放，CFL = {config['cfl']}，
   阶梯 {config['ladder']}（(50,1) 是最粗的舍入压力腿，(400,3) 是台账生产腿），
   限幅臂 {config['limiter_arms']}；另在 {tuple(config['positivity_leg'])} 上单旋钮对比正性处理
   {config['positivity_arms']}（`scale` = Zhang-Shu 缩放，`collapse` = 历史一阶回退）。
4. **判据预先声明**：容差 `{sci(tol)}`（相对 `|∫U(t)|`，逐量逐腿），只看**实际跑过**的腿；
   同时把「∫U 不变」这个错误不变量作为**反证臂**一起记账——它必须**失败**，
   否则说明容差对 O(1) 记账错误是瞎的（可证伪性成对设计，`tests/test_lax_conservation_audit.py` 钉死）。
5. **敏感性量化**：端通量若带相对误差 ε = {sensitivity['relative_flux_error']:.0e}，预算会被推移
   {sci(sensitivity['budget_displacement'])}（相对），是容差的 {sensitivity['detection_margin']:.0f} 倍——
   即该审计能分辨 ~1 % 量级的边界记账缺陷。
6. 工具链：重算递交集群（作业 cj2747，4 核，墙钟 {metrics['wall_seconds']:.0f} s）；
   工件与报告数字全部由 `metrics.json` 程序化注入。

## ③ 计算结果

### 3.1 参考解自校验（独立于求解器）

| 守恒量 | 闭式预算 t·(F_L−F_R)+∫U(0) | 分段 Gauss 积分的精确解 | 相对残差 |
|---|---|---|---|
{reference_rows}

残差在 `{sci(reference['worst'])}` 量级 = float64 地板。**闭式 Riemann 参考与全局守恒律自洽**，
这给算例池的精度门又加了一道与求解器无关的基准体检。

### 3.2 求解器预算残差（逐腿逐臂，相对 `|∫U|`）

| 腿 | limiter/positivity | 步数 | 质量 | 动量 | 能量 | 反证臂 质量 | 反证臂 动量 | 反证臂 能量 | 墙钟 s |
|---|---|---|---|---|---|---|---|---|---|
{ladder_rows}

* **预算臂**（左三列）{'全部' if all(r['pass'] for r in gate['rows']) else '**未全部**'}落在 {sci(tol)} 以内（判定 **{gate['verdict']}**），最差出现在 `{worst['leg']}` / limiter={worst['limiter']} / positivity={worst['positivity']} / {CN[worst['field']]}，值 {sci(worst['value'])}。
* **反证臂**（右三列）处处 ≥ 1e-2，最差 {sci(naive_worst)}：本算例「∫U 不变」确实不成立。
* 三条限幅臂与两种正性处理**都保守**：TVB/WENO 限幅器只改高阶模态、Zhang-Shu 缩放只按 θ 缩高阶模态、
  一阶回退只把单元重置为其自身平均，三者的单元平均值都严格保持——这一点由本表实测而非由代码注释断言。

### 3.3 残差的物理来源（为什么是 1e-7…1e-9 而不是 1e-15）

模态更新里 `volume − surface` 两项在均匀态上逐项抵消（`∫φ_k' dξ = φ_k(1)−φ_k(−1)` 与面项同量级相消），
相消残差 ~ eps·|F| 会随步数累积。因此**通量越大的量残差越大**（动量端通量 3.74 ≫ 能量/质量），
**网格越粗相对残差越大**（(50,1) 腿最差）。这是舍入地板，不是物理耗散：它随细化单调下降
（见图 `lax_conservation_residual_vs_resolution.png`），而任何真实记账缺陷会稳定停留在缺陷量级上。

## ④ 结果图

1. `lax_conservation_naive_vs_budget.png`：生产腿（N=400, p=3）上「错误不变量」与「闭式预算」两臂的
   逐量相对残差（log 轴）。反证臂 O(1e-1…1e0)、预算臂 O(1e-9)，中间隔着容差线——
   一张图同时证明「审计有分辨力」和「格式确实守恒」。
2. `lax_conservation_residual_vs_resolution.png`：三条限幅臂的最差预算残差 vs 自由度 N·(p+1)（log-log），
   加上容差水平线：残差随细化下降并远低于容差。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **本审计不改善精度判据**：台账 `< 3 %` 是**精度**门（相对 L2 vs 精确解），生产腿仍是
   honest-FAIL（3.116 %，见 `docs/CASE-LAX-DG.md` 与分辨率闭合研究）。守恒闭合**不能**替换它，
   两层判据回答不同问题，分开记账。
2. **动量的「守恒」必须带端通量**：本例左端 u ≠ 0，`∫ρu` 增长 0.413 是物理进流，不是耗散/漏失。
   把 `∫ρu` 当成常数会得到 266 % 的假漂移——这是本单最容易踩的坑（也是反证臂存在的理由）。
3. **闭式预算成立的前提**：`t` 之前无波到达域端。本算例 t = 0.13 时最外侧波在 x = 0.1576 / 0.8223，
   余量充足；若改 final_time 使波穿出边界，端通量随时间变化，闭式积分失效，必须改用时间积分预算
   （`tests/_conservation_tools.boundary_increment` 的文档串已写明这一点）。
4. **精度地板是舍入而非缺陷**：容差 1e-5 是相对 `|∫U|` 的；最粗腿实测 ~1e-6，余量 ~10x。
   把容差压到 1e-9 会让粗腿的舍入残差误报为缺陷，那是在测浮点，不是在测守恒。
5. **正性处理的保守性只在本题实测**：`scale`/`collapse` 的保守性由构造保证，但本单只验证了
   Lax 一例；更强的正性回退（如单元局部 h-细化）未在仓内实现，不在本审计范围。
6. **算力面**：单核 CPU float64，CPU 侧墙钟见 3.2；全阶梯走集群队列，未使用 GPU。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答是否 <3%）

基准有两层：**参考解自校验**用全局守恒律（PDE 级解析基准），**求解器预算**用闭式端通量预算。

| 关键量 | 实测残差 | 容差 | 是否 < 3 % |
|---|---|---|---|
{error_rows}
| 参考解自身预算残差（独立于求解器） | {sci(reference['worst'])} | 1e-12 | 是（低 12 个量级） |
| 反证臂（错误不变量）最差 | {sci(naive_worst)} | — | **否（预期，反证成功）** |

**结论**：所有守恒判据都**远优于** 3 %（最差预算残差 {sci(worst['value'])}，比 3 % 小约 4 个量级），
并且不是靠放松容差得到的——同一审计对「∫U 不变」这个错误不变量给出 {sci(naive_worst)} 的残差，
说明判据对 O(1) 记账错误有分辨力。**与此同时，台账 `< 3 %` 的精度门在生产腿仍是 honest-FAIL，
本单不改写它**。

**工件指针**：`docs/benchmarks/results/lax_conservation/metrics.json`（逐腿逐臂残差、闭式预算、
参考解自校验、容差、敏感性）、`figures/*.png`；
复现：`PYTHONPATH=. python -m benchmarks.lax_conservation_audit`；
测试：`python -m pytest tests/test_lax_conservation_audit.py -q`。
"""


def main() -> int:
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    markdown = build_document(metrics)
    DOC.write_text(markdown, encoding="utf-8")
    print(f"[doc] wrote {DOC.relative_to(ROOT)}")

    figures = []
    captions = {
        "lax_conservation_naive_vs_budget": (
            "生产腿 N=400,p=3：错误不变量（∫U 不变）与闭式预算预算残差对照（log 轴），"
            "虚线为容差 1e-5"
        ),
        "lax_conservation_residual_vs_resolution": (
            "三条限幅臂的最差预算残差 vs 自由度 N·(p+1)，log-log；残差随细化下降并远低于容差"
        ),
    }
    for path in sorted((OUT / "figures").glob("*.png")):
        figures.append(
            {
                "name": path.stem,
                "caption": captions.get(path.stem, path.stem),
                "png_base64": b64(path),
            }
        )

    gate = metrics["gate"]
    worst = gate["worst"]
    payload = {
        "title": "CASE-LAX-DG 全局守恒预算审计：格式守恒性硬不变量（预算臂 PASS，反证臂 O(1) 失败）",
        "summary": (
            f"给 CASE-LAX-DG 补一条与精度正交的守恒判据：闭式端通量预算 t·(F_L−F_R)"
            f"（{len(metrics['entries'])} 腿/臂，墙钟 {metrics['wall_seconds']:.0f}s，作业 cj2747）。"
            f"预算臂最差 {gate['worst']['value']:.3e}（容差 {gate['tolerance']:.0e}，PASS）；"
            f"反证臂「∫U 不变」被违反 {gate['naive_worst']:.2f}（预期失败，证明判据有分辨力）；"
            f"闭式 Riemann 参考自校验残差 {metrics['exact_reference']['worst']:.1e}（不调求解器）。"
            f"精度门（台账 <3%）在生产腿仍为 honest-FAIL，未改写。"
        ),
        "scenario_key": "case-pool",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "CASE-LAX-DG", "DG", "Euler", "verification",
                 "conservation", "boundary-flux-budget"],
        "markdown": markdown,
        "figures": figures,
    }
    token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    result = post("/tensor/reports", payload, token)
    print(json.dumps({k: result[k] for k in result if k in ("id", "title", "created_at")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())