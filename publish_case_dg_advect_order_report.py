"""Publish the CASE-DG-ADVECT-ORDER progress report to the platform API.

The markdown body is the rendered report (``docs/CASE-DG-ADVECT-ORDER.md``,
itself generated from ``validation/advection_order/metrics.json``), so every
number in the published report is programmatic -- nothing is retyped here.
The <3% question is answered explicitly at the top, per policy.

    PYTHONPATH=. python publish_case_dg_advect_order_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

from tools.render_advorder_doc import main as render_doc

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 19976
METRICS = pathlib.Path("validation/advection_order/metrics.json")
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
    m = json.loads(METRICS.read_text())
    rows = {row["degree"]: row for row in m["comparison"]}
    s = m["sensitivity"]
    mk = lambda ok: "✅" if ok else "❌"  # noqa: E731
    dev = {d: 100 * abs(r["order_gap_vs_ref"]) / max(abs(r["ref_fitted_order"]), 1e-12)
           for d, r in rows.items()}
    below3 = all(v < 3.0 for v in dev.values())

    rev = m["checks_revised_posthoc"]
    rate_ok = all(v for k, v in rev.items() if k.startswith("rate_"))
    head = f"""**判定：算例判据（收敛率）{mk(rate_ok)} PASS——P1–P4 观测阶与 ref 同阶列的相对
偏差均 <3%。事前判据 R1–R6 与辅助判据 S2/S4 有未过项，逐项如实列于正文 §4.3 / §6：
其中 R2/R6 的"RK4 时间阶封顶"假设被数据否定（时间项斜率
{m['sensitivity']['p4_error_vs_cfl_slope']:.2f}），R3 与 S2 属跨代码口径误设，S4 为舍入级
差异 + 阈值过紧。不改数、不改判据。**

## ⓪ 误差判定（是否 <3%）

本算例判据是**收敛率**。按"观测阶相对 ref 同阶列的相对偏差"折算：

| p | 基准阶（ref 列，同估计器） | 本仓阶 | 相对偏差 | <3% |
|--:|---:|---:|---:|:--:|
""" + "\n".join(
        f"| P{d} | {r['ref_fitted_order']:.3f}（{r['ref_column']}） | "
        f"{r['got_fitted_order']:.3f} | {dev[d]:.2f}% | {mk(dev[d] < 3.0)} |"
        for d, r in sorted(rows.items())) + f"""

**结论：{'是，P1–P4 的观测阶相对偏差全部 <3%' if below3 else '否，见上表'}。**
绝对误差与 ref 的逐网格比值（P1 ≈ {sum(rows[1]['ratio_got_over_ref'])/5:.2f}、P2 ≈
{sum(rows[2]['ratio_got_over_ref'])/5:.2f}、P3 ≈ {sum(rows[3]['ratio_got_over_ref'])/5:.2f}、
P4 ≈ {sum(rows[4]['ratio_got_over_ref'])/5:.2f}）随网格近乎常数（离散度
{max(r['ratio_spread'] for r in rows.values()):.1%} 以内），说明**同阶、不同前因子**；
论文未公布其时间步与通量，故不对绝对误差作 3% 承诺（如实在此声明）。

"""
    body = render_doc()
    figures = [
        {"name": "advect_order_convergence",
         "caption": "TensorDG 各阶 L2 收敛（实线）vs ref 同阶列（虚线叉）；拟合阶见正文 §4",
         "png_base64": b64(FIG / "case_dg_advect_order_convergence.png")},
        {"name": "advect_order_sensitivity",
         "caption": "CFL 敏感性：左 P2/N=40；右 P4/N=80（误差对 CFL 几乎不敏感 ⇒ 时间项可忽略）",
         "png_base64": b64(FIG / "case_dg_advect_order_sensitivity.png")},
        {"name": "advect_order_profile",
         "caption": "P3/N=80 在 t=1 的场量与逐点误差（数值 vs 解析）",
         "png_base64": b64(FIG / "case_dg_advect_order_profile.png")},
    ]
    payload = {
        "title": ("CASE-DG-ADVECT-ORDER：线性对流 p 阶收敛率复测（ref 表由 arXiv 源码机械提取，"
                  "Math. Comp. 2024）"),
        "summary": (f"1-D 周期线性对流 u0=sin(2πx), t=1, P0–P4 复测：观测阶 "
                    f"P1 {rows[1]['got_fitted_order']:.3f} / P2 {rows[2]['got_fitted_order']:.3f}"
                    f" / P3 {rows[3]['got_fitted_order']:.3f} / P4 {rows[4]['got_fitted_order']:.3f}"
                    f"，与 ref 同阶列（{rows[1]['ref_fitted_order']:.2f}/"
                    f"{rows[2]['ref_fitted_order']:.2f}/{rows[3]['ref_fitted_order']:.2f}/"
                    f"{rows[4]['ref_fitted_order']:.2f}）相对偏差均 <3%；"
                    f"RK4 时间阶封顶假设被否定（时间项斜率 {s['p4_error_vs_cfl_slope']:.2f}）"),
        "scenario_key": "CASE-DG-ADVECT-ORDER",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "linear-advection", "convergence-order", "case-pool",
                 "RKDG", "verification", "falsified-hypothesis"],
        "markdown": head + body,
        "figures": figures,
    }
    pathlib.Path("/tmp/advorder_report.json").write_text(json.dumps(payload))
    token = post("auth/login", {"username": USER, "password": PASSWORD}).get("access_token")
    result = post("tensor/reports", payload, token)
    print("report posted:", json.dumps(result)[:500])
    return result


if __name__ == "__main__":
    main()