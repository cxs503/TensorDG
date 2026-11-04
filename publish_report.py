"""Publish the mission report to the platform API (no secrets on the command line)."""
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    request = urllib.request.Request(
        f"{BASE}/{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.status, response.read().decode()


status, body = post("auth/login", {"username": USER, "password": PASSWORD})
token = json.loads(body).get("access_token") or json.loads(body).get("token")
print("login", status, "token_len", len(token or ""))

report = json.load(open("/root/tp-agent/work/m3506/.report.json"))
status, body = post("tensor/reports", report, token)
print("report", status, body[:400])
status, body = post(f"tensor/reports?mission_id=3506", {}, token) if False else (None, "")
try:
    request = urllib.request.Request(
        f"{BASE}/tensor/reports?mission_id=3506&limit=3",
        headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read())
    entries = data if isinstance(data, list) else data.get("items", data)
    for entry in (entries if isinstance(entries, list) else [])[:3]:
        print("saved id", entry.get("id"), entry.get("title", "")[:60])
except Exception as error:  # noqa: BLE001 - read-back is best effort
    print("read-back failed:", error)