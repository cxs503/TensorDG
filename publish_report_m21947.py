#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR report to platform."""
import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
OUT = Path("case_euler_vortex_errconserve")
MISSION = 21947

def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def main():
    # Login
    token_data = post("/auth/login", {"username": USER, "password": PASS})
    token = token_data.get("token", token_data.get("access_token", ""))
    print(f"Token: {token[:20]}...")

    # Load metrics
    metrics = json.loads((OUT / "metrics.json").read_text())
    gates = metrics.get("gates", {})
    records = metrics.get("records", [])

    # Build figures
    figures = []
    for png in OUT.glob("fig_*_m21947.png"):
        figures.append({
            "name": png.stem,
            "caption": png.stem.replace("fig_", "").replace("_m21947", "").replace("_", " "),
            "png_base64": b64(png)
        })

    # Build markdown from the doc
    doc_path = Path("docs/CASE-DG-EULER-VORTEX-ERR-m21947.md")
    markdown = doc_path.read_text()

    # Count passed gates
    passed = sum(1 for k, v in gates.items() if isinstance(v, dict) and v.get("passed"))
    total = sum(1 for k, v in gates.items() if isinstance(v, dict) and k != "verdict")

    report = {
        "title": "CASE-DG-EULER-VORTEX-ERR: 等熵涡移动误差守恒验证 (m21947)",
        "summary": f"DG格式等熵涡平流误差守恒验证：{passed}/{total}准则通过，DG族误差增长指数α<0.5（亚线性），快照容差远优于6%。C11证据溯源待集群重放完成。",
        "scenario_key": "CASE-DG-EULER-VORTEX-ERR",
        "mission_id": MISSION,
        "tags": ["DG", "euler", "vortex", "error-conservation", "verification", "m21947"],
        "markdown": markdown,
        "figures": figures,
    }

    result = post("/tensor/reports", report, token)
    print(f"Report published: {json.dumps(result, indent=2)}")
    return result

if __name__ == "__main__":
    main()
