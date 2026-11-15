"""Publish the CKS1998 §5.4 cylinder report payload to the platform API.

Credentials are read from the deployment env file (never printed), the payload
comes from ``scripts/build_cylinder_stag_report.py`` -> ``/tmp/report_cyl.json``.

    PYTHONPATH=. python scripts/publish_cylinder_stag_report.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ENV = Path("/root/tp-agent/agent.env")
PAYLOAD = Path("/tmp/report_cyl.json")


def env() -> dict:
    out = {}
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def post(url: str, body: dict, token: str | None = None) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main() -> None:
    e = env()
    base = e["TENSOR_API_BASE"]
    login = post(f"{base}/auth/login",
                 {"username": e["TENSOR_AGENT_USER"], "password": e["TENSOR_AGENT_PASS"]})
    token = (login.get("access_token") or login.get("token")
             or login.get("data", {}).get("access_token"))
    if not token:
        raise SystemExit("login failed")
    payload = json.loads(PAYLOAD.read_text())
    res = post(f"{base}/tensor/reports", payload, token)
    rid = res.get("id") or res.get("report_id") or res.get("data", {}).get("id")
    print(json.dumps({"ok": True, "report_id": rid,
                      "title": payload["title"][:80], "figures": len(payload["figures"])}))


if __name__ == "__main__":
    main()