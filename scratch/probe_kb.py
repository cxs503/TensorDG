import json, os, urllib.request, urllib.parse

BASE = os.environ["TENSOR_API_BASE"]

def post(path, body, token=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def get(path, token):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

tok = None
for login_path in ("/auth/login", "/auth/token", "/login"):
    try:
        out = post(login_path, {"username": os.environ["TENSOR_AGENT_USER"],
                                "password": os.environ["TENSOR_AGENT_PASS"]})
        tok = out.get("access_token") or out.get("token")
        print("login ok via", login_path, "token len", len(tok or ""))
        break
    except Exception as exc:
        print("login failed", login_path, exc)
open("scratch/token.txt","w").write(tok or "")
for path in ("knowledge/capabilities?keyword=TensorDG",
             "knowledge/capabilities?q=TensorDG",
             "knowledge/negative_knowledge_ledger?keywords=TensorDG",
             "knowledge/negative?keywords=TensorDG",
             "knowledge/oss_solutions?keyword=Discontinuous+Galerkin+Navier-Stokes",
             "knowledge/search?q=Kovasznay",
             "knowledge/search?q=TensorDG"):
    try:
        out = get("/" + path, tok)
        text = json.dumps(out)[:600]
        print("OK", path, "->", text)
    except Exception as exc:
        print("ERR", path, exc)
