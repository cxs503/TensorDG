#!/usr/bin/env python3
"""Publish the CASE-LAX-DG follow-up (R3 nested band + dt split) to the platform.

Markdown body is rendered from the committed artifact (``render_lax_case_pool_doc``),
so every number in the report is injected, never transcribed.  Figures are the
PNGs the same driver wrote.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_lax_case_pool_doc import METRICS, render  # noqa: E402

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = int(os.environ.get("MISSION_ID", "16993"))
FIGURES = ("lax_profiles.png", "lax_convergence.png", "time_refinement.png")


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    metrics = json.loads(METRICS.read_text())
    markdown = render(metrics)
    gates = metrics["gates"]
    r3 = gates["r3"]
    legacy = metrics["diagnostics"]["r3_legacy_band"]
    time_gate = gates.get("time", {})
    figures = []
    for name in FIGURES:
        path = METRICS.parent / "figures" / name
        if path.exists():
            figures.append({
                "name": name,
                "caption": f"CASE-LAX-DG / {name}（由 benchmarks/lax_case_pool.py 渲染，"
                           f"数字与图同源 metrics.json）",
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            })
    payload = {
        "title": ("CASE-LAX-DG：R3 区域规则换轨（4·dx → 固定 0.02 物理带）+ 时间误差分离，"
                  f"整体 honest-FAIL 保留（R3 {r3['status']}）"),
        "summary": ("闭合 m16606 审计 F1/F2/F5：R3 遮罩改为固定物理带宽（保留测度展布 2e-3 内，"
                    "旧规则仍 +35% 不嵌套），P1 子判决翻转而 P2/P3 仍不达设计阶；"
                    f"新增 dt 对折研究分离时间项，T 闸门 {time_gate.get('status')}；"
                    f"工件迁至 tracked docs/benchmarks/results/lax_case_pool/"),
        "scenario_key": "case_lax_dg_r3_nested_time_split",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "CASE-LAX-DG", "DG", "verification", "region-rule",
                 "temporal-error", "honest-FAIL"],
        "markdown": markdown,
        "figures": figures,
    }
    token = post("/auth/login", {"username": USER, "password": PASS})["access_token"]
    result = post("/tensor/reports", payload, token)
    print(json.dumps({"report_id": result.get("id"), "title": payload["title"],
                      "figures": len(figures)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())