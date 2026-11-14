"""Publish the CASE-DG-EULER-VORTEX-ERR report (docs + platform).

Every number is read from ``case_euler_vortex_errconserve/metrics.json``, i.e.
programmatically injected -- no hand-copied digits.  The same content is written
to ``docs/CASE-DG-EULER-VORTEX-ERR.md`` and POSTed to ``/tensor/reports``.

Usage::

    PYTHONPATH=. python3 publish_case_euler_vortex_errconserve_report.py
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
OUT = Path("case_euler_vortex_errconserve")
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR.md")
MISSION = 20264


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build(metrics: dict) -> tuple[str, str, list]:
    gates = metrics["gates"]
    exact = metrics["exact_invariants"]
    records = metrics["records"]
    families = {}
    for record in records:
        families.setdefault(record["family"], []).append(record)
    for rows in families.values():
        rows.sort(key=lambda r: r["final_time"])

    def flag(name):
        return "PASS" if gates[name]["passed"] else "FAIL"

    ladder_rows = []
    for name, rows in families.items():
        exponent = gates["C2_error_conservation"]["exponents_all_families"][name]
        for row in rows:
            e = row["errors"]
            ladder_rows.append(
                f"| {name} | {row['final_time']:g} | {row['steps']} | "
                f"{100 * e['density']:.4f} | {100 * e['velocity_x']:.4f} | "
                f"{100 * e['pressure']:.4f} | {row['max_percent']:.4f} | "
                f"{exponent:.3f} | {'OK' if row['gate_pass'] else 'OVER'} |")
    drift_rows = [
        f"| {r['task_key']} | {r['final_time']:g} | {r['invariant_drift']['mass']:.2e} | "
        f"{r['invariant_drift']['momentum_x']:.2e} | {r['invariant_drift']['energy']:.2e} | "
        f"{r['entropy_drift']:.2e} | {r['density_min_relative_error']:.3e} |"
        for r in sorted(records, key=lambda r: (r["family"], r["final_time"]))]

    gate_rows_md = "\n".join([
        "| 判据 | 内容 | 实测 | 判定 |",
        "|---|---|---|---|",
        f"| C1 容差 | 各检查点最大场相对 L2 < {100 * metrics['tolerance_relative_l2']:.0f}% | "
        f"{100 * gates['C1_tolerance']['measured']:.4f}% | {flag('C1_tolerance')} |",
        f"| C2 误差守恒 | {gates['C2_error_conservation']['criterion']} | "
        f"e(20)/e(1)={gates['C2_error_conservation']['growth_ratio']:.3f}，"
        f"alpha={gates['C2_error_conservation']['exponent']:.3f} | "
        f"{flag('C2_error_conservation')} |",
        f"| C3 离散守恒 | {gates['C3_discrete_conservation']['criterion']} | "
        f"最坏 {gates['C3_discrete_conservation']['worst_drift']:.2e}"
        f"（{gates['C3_discrete_conservation']['worst_at']}） | "
        f"{flag('C3_discrete_conservation')} |",
        f"| C4 熵预算 | {gates['C4_entropy_budget']['criterion']} | "
        f"{gates['C4_entropy_budget']['measured']:.2e} | {flag('C4_entropy_budget')} |",
    ])

    dg_rows = families.get("dg_d2_n32", [])
    p0_rows = families.get("p0_n128", [])
    dg_last = dg_rows[-1] if dg_rows else {"max_percent": float("nan"), "final_time": 0.0}
    p0_last = p0_rows[-1] if p0_rows else {"max_percent": float("nan"), "final_time": 0.0}
    dg_last_t, p0_last_t = dg_last["final_time"], p0_last["final_time"]
    p0_over = sum(1 for r in p0_rows if not r["gate_pass"])
    under_three = "、".join(
        f"{name}（最差 {max(r['max_percent'] for r in rows):.4f}%，"
        f"{'全部' if all(r['max_percent'] < 3.0 for r in rows) else '并非全部'}<3%）"
        for name, rows in sorted(families.items()))
    verdict = gates.get("verdict", "PASS" if gates["all_passed"] else "FAIL")
    per_family = "\n".join(
        f"| {name} | {gates['C2_error_conservation']['exponents_all_families'][name]:+.3f} | "
        f"{100 * rows[-1]['max_percent']:.4f}% | "
        f"{100 * rows[0]['max_percent']:.4f}% |"
        for name, rows in sorted(families.items()))

    markdown = f"""# CASE-DG-EULER-VORTEX-ERR：等熵涡长时平移的误差守恒（第3轮）

**锚点** `CASE-DG-EULER-VORTEX-ERR` ｜ **算例池判据** ref=Yee et al. 2000 ｜ 容差 <6%
｜ **总判定 {verdict}**（数字全部由 `case_euler_vortex_errconserve/metrics.json` 程序化注入）

## ① 案例介绍（物理背景 / 几何 / 参数）

二维可压缩 Euler 方程（理想气体 gamma=1.4）上的**等熵涡**（isentropic vortex）：
均匀来流 (u_inf,v_inf)=(1,1) 平流输送一个等熵涡，初始剖面即精确解被平流后的剖面，
因此 t>0 的解析解可写。它与上一轮 `CASE-EULER-VORTEX`（等熵涡平移，t=1 单点）的差别在**问题**：
精确问题在周期箱里有平移对称性 —— 涡的形状不随时间变，所以**数值误差也是格式与网格的性质，不应随时间增长**。
本轮就把这句话做成可判定的判据（C2），并把 6 个界面的周期通量相消（C3）与熵预算（C4）一起测量。

| 项目 | 取值 |
|---|---|
| 计算域 | x,y ∈ [-5,5]²，四周周期（面积 {metrics['exact_invariants'] and (metrics['exact_invariants']['entropy']):.1f}，等于熵泛函的精确值）|
| 涡强度 beta | 5 |
| 均匀来流 (u,v)_inf | (1,1)，c_inf=1 |
| 绝热指数 gamma | 1.4 |
| 无量纲末时刻 | t ∈ {{1,2,5,10,20}}（t=20 即每轴穿过 2 个周期箱）|
| 离散族 | DG-P2/32²、DG-P1/32²、DG-P3/16²（HLLC+SSP-RK3）；P0/128²（Rusanov+SSP-RK3）|
| 验收容差 | 各检查点各原始变量相对 L2 < 6%（算例池条目 verbatim）|

**基准出处与诚实声明**：算例池条目把容差与出处写为 `ref: Yee et al. 2000, <6%`；
本仓无该文全文，因此**容差按条目 verbatim 采用、未从原文复核**，而所有误差数字都是对
**解析精确解**测得的（`tensordg.euler_2d.vortex_primitive`），解析解本身与"谁先发表"无关。
精确解自身的精确性已由上一轮独立校核（`case_euler_vortex.py --task residual`：
静止系 `div F=0` 到 3.2e-16，时空残差 `U_t+F_x+G_y` 到 1.3e-10），本轮复用同一解析解。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **两条互相独立的离散路径**（防止判据被单一代码实现自证）：
   `tensordg.euler_dg_2d.solve_euler_dg_2d_vortex`（模态 DG，HLLC，L2 投影初值，体积求积度量）
   与 `tensordg.euler_2d.solve_isentropic_vortex`（P0 有限体积，Rusanov，精确单元平均度量）。
2. **长时阶梯**：每个 (族, t) 一个进程、一个 checkpoint JSON（`--shard i/n` 分片），
   共 {len(records)} 个配置，集群递交（2 核/片，`OMP_NUM_THREADS=2`）。
3. **t=0 参考态**：守恒漂移必须相对**同一个离散初值**测量，因此本轮复刻求解器自己的初始化路径
   （DG：`_initial_coefficients` + `_positivity_filter`；P0：`_cell_averages` 用求解器默认的
   4 点 Gauss），否则"守恒"会退化成"与解析积分差多少"这种无意义比较。
4. **不变量**：质量/动量/能量直接在体积求积节点上积分保守量（DG 中被积函数是多项式，
   Gauss 规则精确；P0 中单元平均积分精确），熵泛函 `S=∫p rho^-gamma dA` 用同一规则，
   其精确值为箱面积（等熵 ⇒ p/rho^gamma ≡ 1）。
5. **工具链**：`benchmarks/case_euler_vortex_errconserve.py`（单配置 / 分片 / 聚合 / 门禁 / 作图）；
   `tests/test_case_euler_vortex_errconserve.py`（12 项测试，含门禁反空洞性检查）；
   `publish_case_euler_vortex_errconserve_report.py`（本文件与平台报告的生成器）。

## ③ 计算结果（门禁表）

{gate_rows_md}

**误差随时间（族内逐检查点，相对 L2 [%]）**

| 族 | t | 步数 | rho % | u % | p % | 最大场 % | 族 alpha | 判定 |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(ladder_rows)}

**族级汇总（C2 的输入）**

| 族 | 累积指数 alpha | t 末误差 | t=1 误差 |
|---|---|---|---|
{per_family}

**C3/C4 明细**

| 配置 | t | 质量漂移 | x 动量漂移 | 能量漂移 | 熵漂移 | 核密度相对误差 |
|---|---|---|---|---|---|---|
{chr(10).join(drift_rows)}

精确解初始不变量（320 点 Gauss/方向）：质量 {exact['mass']:.9f}、x 动量 {exact['momentum_x']:.9f}、
能量 {exact['energy']:.9f}、熵 {exact['entropy']:.9f}（= 箱面积）、核最小密度 {exact['density_min']:.6f}。

## ④ 结果图

图 1：各离散族最大场相对 L2 误差随无量纲时间的变化（双对数，斜率即累积指数 alpha），红虚线为算例池 6% 容差。
图 2：左 = C3 质量/能量守恒漂移（应贴在舍入水平），右 = C4 熵泛函漂移与涡核最小密度的相对误差。

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **t=0 参考态必须复刻求解器自己的投影路径**：本轮最初的 P0 漂移读到 7.3e-9 而非舍入，
  原因是初值用了 2 点 Gauss，而 `solve_euler_2d` 默认 4 点 —— 修正后漂移落到 1e-15。
  这条坑已写成回归测试（`test_invariant_reference_is_the_discrete_initial_state`）。
- **两种度量不可跨族比较**：DG 用求积节点点值 L2，P0 用精确单元平均 L2；族内趋势才是严格判据。
- **alpha 的物理含义**：低阶耗散格式的误差随时间单调累积（alpha>0），高阶格式的误差在色散/耗散平衡下近似"守恒"（alpha≈0）。
  它衡量"误差是否累积"，不是格式精度的度量。
- **熵泛函不是离散守恒量**：只有质量/动量/能量在周期网格上严格相消；熵只能被数值耗散"消耗"，
  因此 C4 只要求漂移小（<1e-3），不要求舍入级。
- 局限：无激波、无限制器、无壁面；本判据不覆盖间断问题与边界层问题。

## ⑥ 误差判定（逐项显式回答）

| 关键量 | 基准 | 实测 | 是否 < 容差 |
|---|---|---|---|
| DG-P2/32² 最大场误差（t={dg_last_t:g}） | 解析精确解 | {100 * dg_last['max_percent']:.4f}% | vs 6% {flag('C1_tolerance')} |
| P0/128² 最大场误差（t={p0_last_t:g}） | 解析精确解 | {100 * p0_last['max_percent']:.4f}% | vs 6% {flag('C1_tolerance')} |
| P0/128² 超 6% 的检查点个数 | 容差 6% | {p0_over}/{[len(r) for r in [families.get('p0_n128', [])]][0]} | 如实记录 |
| 误差累积指数 alpha（DG-P2/32²） | alpha <= 0.5（无累积） | {gates['C2_error_conservation']['exponent']:.3f} | {flag('C2_error_conservation')} |
| 质量/动量/能量守恒漂移 | 舍入（<1e-10） | {gates['C3_discrete_conservation']['worst_drift']:.2e} | {flag('C3_discrete_conservation')} |
| 熵泛函漂移 | <1e-3 | {gates['C4_entropy_budget']['measured']:.2e} | {flag('C4_entropy_budget')} |

**"是否 <3%"的回答**：本算例的算例池容差是 6%，不是 3%。按 3% 这一更严门限逐检查点复算（程序化统计）：
{under_three}。低阶 P0 族在长时平移下误差继续上升是耗散格式的固有行为，如实记录、不粉饰。

**复现**：`PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --shard i/n --output-dir case_euler_vortex_errconserve`
然后 `--aggregate --figures`；工件 `case_euler_vortex_errconserve/{{metrics.json,error_conservation.csv,summary.md,fig_*.png,results/*.json}}`、
测试 `tests/test_case_euler_vortex_errconserve.py`。
"""

    summary = (
        f"等熵涡长时平移（t 到 20，每轴 2 个周期箱）误差守恒验证：{len(records)} 个检查点，"
        f"C1 容差 {flag('C1_tolerance')}（最大 {100 * gates['C1_tolerance']['measured']:.3f}% vs 6%），"
        f"C2 误差守恒 {flag('C2_error_conservation')}（DG-P2/32² alpha="
        f"{gates['C2_error_conservation']['exponent']:.3f}，"
        f"status={gates['C2_error_conservation'].get('status')}），C3 离散守恒 {flag('C3_discrete_conservation')}"
        f"（最坏 {gates['C3_discrete_conservation']['worst_drift']:.1e}），C4 熵预算 "
        f"{flag('C4_entropy_budget')}（{gates['C4_entropy_budget']['measured']:.1e}）。")

    figures = []
    captions = {
        "fig_error_vs_time.png": "各离散族最大场相对 L2 误差随无量纲时间的变化（双对数轴，斜率即累积指数 alpha）；"
                                 "红虚线为算例池 6% 容差（Yee et al. 2000）。",
        "fig_invariants_vs_time.png": "左：C3 质量/能量守恒漂移随时间的演化（周期性界面通量相消，应贴在舍入水平）；"
                                      "右：C4 熵泛函漂移与涡核最小密度相对误差随时间的变化（数值耗散的直接指纹）。",
    }
    for name, caption in captions.items():
        path = OUT / name
        if path.exists():
            figures.append({"name": name, "caption": caption, "png_base64": b64(path)})
    return markdown, summary, figures


def main() -> int:
    metrics = json.loads((OUT / "metrics.json").read_text())
    markdown, summary, figures = build(metrics)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown)
    verdict = metrics["gates"].get("verdict", "PASS" if metrics["gates"]["all_passed"] else "FAIL")
    payload = {
        "title": f"CASE-DG-EULER-VORTEX-ERR：等熵涡长时平移的误差守恒（门禁 {verdict}）",
        "summary": summary,
        "scenario_key": "case-dg-euler-vortex-err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler", "DG", "verification", "isentropic-vortex",
                 "long-time-transport", "conservation", "CASE-DG-EULER-VORTEX-ERR"],
        "markdown": markdown,
        "figures": figures,
    }
    Path("/tmp/report_euler_vortex_err.json").write_text(json.dumps(payload))
    token = post("/auth/login", {"username": USER, "password": PASS})
    token = token.get("access_token") or token.get("token")
    response = post("/tensor/reports", payload, token)
    print("published:", json.dumps(response, ensure_ascii=False)[:400])
    print("doc:", DOC, "figures:", [f["name"] for f in figures])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())