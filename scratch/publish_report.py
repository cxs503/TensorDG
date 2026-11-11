"""Publish the m15258 progress report to the platform report API (scratch, untracked)."""
import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
DOC = Path("docs/reports/tensordg-ns-wall-flux-extrapolation.md")


def call(path, payload=None, token=None, method=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers,
                                 method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


status, body = call("/auth/login", {"username": USER, "password": PASS})
tokens = json.loads(body)
token = tokens.get("access_token") or tokens.get("token")
assert token, body[:400]

status, body = call("/tensor/missions/15258", token=token)
print("mission:", status, body[:500])

markdown = DOC.read_text()
payload = {
    "title": "TensorDG 等温壁热流后处理：内面通量外插把 P1 壁热流从 1 阶提到 2 阶（m15258）",
    "summary": "TensorDG 新增 tensordg.wall_flux_extrapolation：把等温壁前 points 个内面能量通量 "
               "Lagrange 外插到壁面，纯后处理不动算子；P1 壁热流观测阶 1.03→1.99，最细网格相对误差 "
               "1.05%→0.012%（face 0.26%→0.012%），curved 算例证明起作用的是外插模板而非单面超收敛。",
    "scenario_key": "tensordg-ns-wall-flux-extrapolation",
    "mission_id": 15258,
    "tags": ["TensorDG", "NS", "isothermal wall", "heat flux", "order of accuracy",
             "post-processing", "m15258"],
    "markdown": markdown,
}
fig = Path("docs/figures/ns_wall_flux_extrapolation.png").read_bytes()
payload["figures"] = [{
    "name": "ns_wall_flux_extrapolation.png",
    "caption": "(a) conduction state: solver vs extrapolated wall heat flux against the "
               "exact kappa(T_h-T_c)/L; (b) same state, error of both estimators (P1 "
               "direct order 1, extrapolated order 2); (c) curved T (flux varies): "
               "nearest interior face order 1, extrapolated points=2 order 2.",
    "png_base64": base64.b64encode(fig).decode(),
}]
status, body = call("/tensor/reports", payload, token)
print("report:", status, body[:800])