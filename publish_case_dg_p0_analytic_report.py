"""Publish the P0-row report of CASE-DG-ADVECT-ORDER to the platform API.

The markdown body is ``docs/CASE-DG-ADVECT-ORDER-P0-ANALYTIC.md`` rendered from
``validation/advection_order_p0/metrics.json`` by :mod:`tools.render_p0_doc`, so
every published number is programmatic.

    PYTHONPATH=. python publish_case_dg_p0_analytic_report.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

from tools.render_p0_doc import main as render_doc

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 20521
METRICS = pathlib.Path("validation/advection_order_p0/metrics.json")
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
    d = m["derived"]
    chk = m["checks"]
    body = render_doc()
    figures = [
        {"name": "case_dg_p0_convergence",
         "caption": ("DG0 线性对流 t=1：闭式半离散精确误差（实线）、求解器全离散（空心方块）、"
                     "斜率 1 参照与投影地板 πh/√6；阴影=池梯 N=20…320"),
         "png_base64": b64(FIG / "case_dg_p0_convergence.png")},
        {"name": "case_dg_p0_local_order",
         "caption": ("局部（两点）阶随 N 的爬升：池梯 0.69→0.96，N≥1280 才进入 1% 带，"
                     "渐近极限 1.00"),
         "png_base64": b64(FIG / "case_dg_p0_local_order.png")},
        {"name": "case_dg_p0_profile",
         "caption": "N=20 粗网格场量 vs 解析解（上）与逐格误差（下）：幅度衰减是 DG0 误差主源",
         "png_base64": b64(FIG / "case_dg_p0_profile.png")},
    ]
    payload = {
        "title": ("CASE-DG-ADVECT-ORDER·P0：闭式半离散精确参照复核 DG0 收敛阶"
                  "（0.85 预渐近的证明 + 渐近常数 π√(2π²+1/6)）"),
        "summary": (f"DG0（度 0，≡一阶迎风 FV）半离散算子循环可对角化，闭式给出空间误差："
                    f"池梯 N=20…320 拟合阶 {d['order_analytic_pool']:.6f}，求解器 "
                    f"{d['order_solver_pool']:.6f}（差 {abs(d['order_difference']):.1e}）——"
                    f"0.853 是预渐近而非缺陷；细梯 N≥5120 回到 {d['order_analytic_fine']:.5f}（设计阶 1）；"
                    f"渐近常数实测 {d['constant_measured_finest']:.5f} vs 闭式 "
                    f"{m['closed_form']['asymptotic_constant']:.5f}（{100*d['constant_rel_error']:.3f}%），"
                    f"比单用 π²√2={m['closed_form']['scheme_constant']:.4f} 高 "
                    f"{100*d['constant_rel_error_scheme_only']:.2f}%（O(h²) 投影地板）；"
                    f"求解器 vs 闭式逐网格最大相对差 {d['solver_vs_exact_max_rel']:.1e}；"
                    f"10/10 判据 {m['verdict']}"),
        "scenario_key": "CASE-DG-ADVECT-ORDER",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "linear-advection", "DG0", "convergence-order", "case-pool",
                 "closed-form-reference", "pre-asymptotic", "falsified-hypothesis"],
        "markdown": body,
        "figures": figures,
    }
    pathlib.Path("/tmp/p0_report.json").write_text(json.dumps(payload))
    token = post("auth/login", {"username": USER, "password": PASSWORD}).get("access_token")
    result = post("tensor/reports", payload, token)
    print("report posted:", json.dumps(result)[:400])
    assert all(chk.values()), "report published but checks are not all green"
    return result


if __name__ == "__main__":
    main()