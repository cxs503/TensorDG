#!/usr/bin/env python3
import requests
import json
import os
import base64
from pathlib import Path

api_base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
passwd = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")

# Login
resp = requests.post(f"{api_base}/auth/login", json={"username": user, "password": passwd})
token = resp.json()["access_token"]

# Read the markdown doc
doc_path = Path("docs/CERTIFIED-REFERENCE-DESIGN-ORDER-CEILING.md")
markdown = doc_path.read_text()

# Read the results
results_path = Path("artifacts/design_order_ceiling/results.json")
results = json.loads(results_path.read_text())

# Read the figure
fig_path = Path("artifacts/design_order_ceiling/design_order_ceiling.png")
if fig_path.exists():
    fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    figures = [{"name": "design_order_ceiling", "caption": "Declared-design-order ceiling: refusing P3 ladders was a policy artefact", "png_base64": fig_b64}]
else:
    figures = []

report = {
    "title": "TensorDG 声明设计阶数解除 WITHHELD_ORDER_TOO_HIGH 误拒：带宽覆盖率实测",
    "summary": "round-2 认证引擎用 P_MAX=6.0 作为观测阶的绝对上限，导致 P3 阶梯（p≈7）被误拒。本单引入 design_order 参数，将绝对上限替换为双侧窗口 2k+1±slack。实测 10 条物理阶梯，2 条 P3 阶梯新解除误拒，带宽覆盖率 100%，无回归。",
    "scenario_key": "tensorDG_certified_reference_design_order",
    "mission_id": "25494",
    "tags": ["TensorDG", "certified_reference", "design_order", "round2"],
    "markdown": markdown,
    "figures": figures
}

# Post the report
resp = requests.post(f"{api_base}/tensor/reports", json=report, headers={"Authorization": f"Bearer {token}"})
print(f"Status: {resp.status_code}")
print(f"Text: {resp.text}")
