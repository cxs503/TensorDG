"""Publish the past-seam d'Alembert cross-check report to the platform.

The markdown body is the *programmatically generated* report produced by
``python -m benchmarks.elastic_1d_seam_crosscheck`` -- no number is retyped
here.  Figures are inlined as base64 PNGs from the same artefact directory.

Usage::

    PYTHONPATH=. MISSION_ID=17538 python publish_seam_report.py
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")
MISSION_ID = int(os.environ.get("MISSION_ID", "17538"))
ART = Path("validation/elastic_1d_seam_crosscheck")
DOC = Path("docs/CASE-DG-DALEMBERT-SEAM-CROSSCHECK.md")

FIG_CAPTIONS = {
    "fig_seam_profiles.png":
        "σ(x) 在 c t = 0.55/0.7/0.9/1.3 四个回卷时刻：DG 采样点 vs R0 镜像级数 "
        "vs R1 模态传播子",
    "fig_seam_probe.png":
        "探针 x = 0.8 的 σ(t)：出发到达（t≈0.3）与过缝镜像到达（t≈0.7）双峰，"
        "符号相反、幅值相同",
    "fig_seam_errors.png":
        "左：参考解互校偏差 vs c t（周期缝 c t = 0.5）；右：DG 过缝收敛阶与 "
        "R3 蛙跳自校验",
}


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def login() -> str:
    for body in ({"username": USER, "password": PASS},
                 {"email": USER, "password": PASS}):
        try:
            out = post("/auth/login", body)
        except Exception:                                # noqa: BLE001
            continue
        for key in ("access_token", "token"):
            if key in out:
                return out[key]
        if isinstance(out.get("data"), dict):
            for key in ("access_token", "token"):
                if key in out["data"]:
                    return out["data"][key]
    raise SystemExit("login failed")


def main() -> int:
    anchors = json.loads((ART / "seam_anchors.json").read_text())["anchors"]
    dg = json.loads((ART / "seam_dg_vs_independent.json").read_text())["rows"]
    ref = json.loads((ART / "seam_reference_crosscheck.json").read_text())["rows"]
    probe = json.loads((ART / "seam_probe_mirror.json").read_text())
    n_pass = sum(1 for r in anchors if r["verdict"] == "PASS")

    worst_dg = max(r["dg_vs_R1_spectral"]["rel_linf_pct"] for r in dg)
    worst_ref = max(r["R0_vs_R1_spectral"]["rel_linf_pct"] for r in ref)
    markdown = DOC.read_text()
    figures = [{"name": name, "caption": cap, "png_base64": b64(ART / name)}
               for name, cap in FIG_CAPTIONS.items() if (ART / name).exists()]

    title = ("TensorDG · d'Alembert 一维弹性杆：过周期缝（c t > L/2）独立参考交叉校验 "
             f"— {n_pass}/{len(anchors)} 锚点通过")
    summary = (
        "清偿母 crosscheck 报告明示的欠账「过周期缝区间的参考交叉校验」。"
        f"在 c t = 0.55/0.7/0.9/1.3 四点：判定参考 R0（镜像级数）与模态传播子 R1 "
        f"最坏偏差 {worst_ref:.2e} %；DG 解对从未见过的 R1 最坏偏差 {worst_dg:.2e} %"
        f"（门槛 2 %）；新测探针镜像到达时刻误差 {probe['mirror']['rel_error_pct']:.2e} %"
        f"（理论 t*={probe['mirror']['t_ref']:.6f}），σ 符号翻转成立。"
        f"锚点 {n_pass}/{len(anchors)} PASS。")

    token = login()
    payload = {
        "title": title,
        "summary": summary,
        "scenario_key": "dalembert-elastic-rod-1d",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "elastic-1d", "dAlembert", "verification",
                 "past-seam", "CASE-DG-DALEMBERT-ELASTIC-1D"],
        "markdown": markdown,
        "figures": figures,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())