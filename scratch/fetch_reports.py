"""Fetch recent platform reports (token + list) without echoing secrets."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"]
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


tok = post("/auth/login", {"username": USER, "password": PWD})
token = tok.get("access_token") or tok.get("token")
out = get("/tensor/reports?limit=20", token)
items = out if isinstance(out, list) else out.get("items", out.get("reports", []))
for r in items:
    print(r.get("id"), "|", r.get("created_at", "")[:16], "|", (r.get("title") or "")[:90])