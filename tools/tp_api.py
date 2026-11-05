"""Tiny helper to talk to the TensorPlatform REST API from mission scripts.

Reads credentials from /root/tp-agent/agent.env (local, never committed) and
exposes `token()`, `get(path)`, `post(path, body)` for the :8010 API.
"""
from __future__ import annotations

import json
import os
import urllib.request


ENV_PATH = "/root/tp-agent/agent.env"


def _env():
    values = {}
    with open(ENV_PATH, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip()
    return values


def token() -> str:
    env = _env()
    base = env.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    body = json.dumps({
        "username": env["TENSOR_AGENT_USER"],
        "password": env["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(
        base + "/login", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    return payload["access_token"]


def _call(path: str, method: str = "GET", body=None, tok: str | None = None):
    env = _env()
    base = env.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    tok = tok or token()
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        base + "/" + path.lstrip("/"), data=data, method=method,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def get(path: str, tok: str | None = None):
    return _call(path, "GET", None, tok)


def post(path: str, body, tok: str | None = None):
    return _call(path, "POST", body, tok)


if __name__ == "__main__":
    print("token ok, len =", len(token()))