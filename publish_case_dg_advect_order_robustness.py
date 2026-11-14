"""Publish the CASE-DG-ADVECT-ORDER round-4 (robustness) progress report.

Every number in the markdown comes from
``validation/advection_order_robustness/metrics.json`` (this run) and
``validation/advection_order/ref_table.json`` (the mechanically parsed paper
table); nothing is retyped here.

    PYTHONPATH=. python publish_case_dg_advect_order_robustness.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 20906
OUT = pathlib.Path("validation/advection_order_robustness")
FIG = pathlib.Path("docs/figures")


def post(path, payload, token=None):
    req = urllib.request.Request(f"{BASE}/{path}", data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode())


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> dict:
    m = json.loads((OUT / "metrics.json").read_text())
    doc = (OUT / "summary.md").read_text()
    b1 = m["B1_estimator_independence"]["detail"]
    b2 = m["B2_per_mesh_vs_paper"]
    b3 = m["B3_cfl_window"]
    b4 = m["B4_prefactor_constancy"]["detail"]
    mk = lambda ok: "✅" if ok else "❌"  # noqa: E731
    dev = {d: 100 * abs(v["dev_fit_all"]) / abs(v["ref_fit_all"])
           for d, v in b1.items()}
    below3 = all(v < 3.0 for v in dev.values())

    head = f"""**判定：{m['verdict']}**（未过项：{', '.join(m['failed']) or '无'}）。
本轮不重复前三轮的"再拟合一次"：改问三个前几轮没问的问题——**阶的判定是否依赖
估计器（B1）**、**逐网格是否与论文自己印出的阶一致（B2）**、**阶是不是只在一个
CFL 上偶然成立（B3）**，外加**同阶不同前因子（B4）**的相对陈述。

## ⓪ 误差判定（是否 <3%）

算例判据是收敛率："观测阶 vs ref 同阶列的相对偏差"

| p | ref 拟合阶 | 本轮阶 | 相对偏差 | <3% |
|--:|--:|--:|--:|:--:|
""" + "\n".join(
        f"| P{d[1:]} | {b1[d]['ref_fit_all']:.4f} | {b1[d]['fit_all']:.4f} | "
        f"{dev[d]:.2f}% | {mk(dev[d] < 3.0)} |" for d in sorted(b1)) + f"""

**结论：{'是，P1–P4 观测阶相对偏差全部 <3%' if below3 else '否，见上表'}。**
逐网格（B2，共 {len(b2['rows'])} 行）最大偏差 {b2['max_gap']:.4f} 阶（容差
{m['thresholds']['RATE_TOL']}）。绝对误差：论文未公布时间步，不作 3% 承诺；只给前因子
恒定性（B4 离散度 ≤ {m['thresholds']['RATIO_SPREAD_TOL']:.0%}，见正文 §3.4）。

## 一句话结论

三个独立估计器互相之差 ≤ {max(v['estimator_spread'] for v in b1.values()):.4f} 阶 →
前三轮的 PASS **不依赖估计器**；阶在 `cfl ≤ {max(m['setup']['cfl_gated']):g}` 内与步长
无关（实测全 p 同时通过的最大 CFL = **{b3['validated_edge_cfl']}**）→ 观测到的确实是
**空间阶**，与第 2 轮 expm 误差拆分（时间误差占比 ≤1%）互证。
"""
    payload = {
        "title": ("CASE-DG-ADVECT-ORDER 第4轮：阶判定的估计器 / 渐近窗 / CFL 窗稳健性复测"
                  "（ref = Math. Comp. 2024, 表 tab:acc-sin-1d-uni 机械解析）"),
        "summary": (f"1-D 周期线性对流 P1–P4：三个估计器（全网格拟合 / 最细三网格 / 最细一对）"
                    f"互差 ≤{max(v['estimator_spread'] for v in b1.values()):.4f} 阶，"
                    f"逐网格对论文印刷阶最大偏差 {b2['max_gap']:.4f} 阶；"
                    f"CFL 窗审计给出阶成立窗口 cfl≤{max(m['setup']['cfl_gated']):g}"
                    f"（实测边缘 {b3['validated_edge_cfl']}）；前因子恒定性相对离散度 "
                    f"{max(v['relative_spread'] for v in b4.values()):.2%}。"
                    f"总判定 {m['verdict']}"),
        "scenario_key": "CASE-DG-ADVECT-ORDER",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "linear-advection", "convergence-order", "case-pool",
                 "verification", "estimator-robustness", "cfl-window"],
        "markdown": head + "\n\n---\n\n" + doc,
        "figures": [
            {"name": "advorder_robust_estimators",
             "caption": ("左：各阶 L2 误差（实线）vs 论文表同阶列（空心虚线叉），同阶不同前因子；"
                         "右：观测阶随 CFL 的变化（点线 = 论文同阶列拟合阶），绿色带 = 门控窗"),
             "png_base64": b64(FIG / "advorder_robust_estimators.png")},
            {"name": "advorder_robust_solution",
             "caption": ("N=40, cfl=0.2, t=1：P1–P4 数值解（点）vs 解析波（实线），"
                         "及高斯节点上的逐点误差"),
             "png_base64": b64(FIG / "advorder_robust_solution.png")},
        ],
    }
    pathlib.Path("/tmp/advorder_robustness_report.json").write_text(json.dumps(payload))
    token = post("auth/login", {"username": USER, "password": PASSWORD}).get("access_token")
    result = post("tensor/reports", payload, token)
    pathlib.Path("/tmp/advorder_robustness_report_result.json").write_text(
        json.dumps(result))
    print("report posted:", json.dumps(result)[:600])
    return result


if __name__ == "__main__":
    main()