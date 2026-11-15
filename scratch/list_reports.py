import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]


def post(path, body, token=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req) as f:
        return json.load(f)


def get(path, token):
    req = urllib.request.Request(base + path)
    req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req) as f:
        return json.load(f)


tok = post("/auth/login", {"username": user, "password": pw})
tok = tok.get("access_token") or tok.get("token")
open("/tmp/tok.txt", "w").write(tok)
d = get("/tensor/reports?limit=20", tok)
items = d if isinstance(d, list) else (d.get("items") or d.get("reports") or [])
for r in items:
    print(r.get("id"), "|", (r.get("created_at") or "")[:19], "|", (r.get("title") or "")[:95])