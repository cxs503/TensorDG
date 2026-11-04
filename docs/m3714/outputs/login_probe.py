import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def get(path, token):
    req = urllib.request.Request(BASE + path)
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


login = post("/tensor/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
print("token ok:", bool(token))
reports = get("/tensor/reports?limit=5", token)
items = reports if isinstance(reports, list) else reports.get("items", reports.get("reports", []))
for r in items[:5]:
    print("scenario_key=", r.get("scenario_key"), "mission=", r.get("mission_id"))
with open("/tmp/token.txt", "w") as handle:
    handle.write(token)