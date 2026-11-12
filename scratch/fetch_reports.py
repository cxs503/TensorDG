"""Read-only helper: list recent TensorPlatform reports (titles + todo sections)."""
import json
import os
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": "Bearer " + token} if token else {})},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


tok = post("/auth/login", {
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
})
token = tok.get("access_token") or tok.get("data", {}).get("access_token") or tok.get("token")
open("/tmp/tok.txt", "w").write(token)
data = get("/tensor/reports?limit=15", token)
items = data if isinstance(data, list) else (data.get("data") or data.get("items") or [])
print("n_reports", len(items))
for r in items[:15]:
    print(r.get("id"), "|", (r.get("title") or "")[:70], "|", r.get("scenario_key"), "|", str(r.get("created_at"))[:10])