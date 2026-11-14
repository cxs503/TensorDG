import json, os, urllib.request, urllib.parse, sys

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER")
PWD = os.environ.get("TENSOR_AGENT_PASS")


def _call(req):
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return r.status, json.loads(r.read())
    except Exception as e:
        return "ERR", str(e)


def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return _call(req)


def get(path, token=None):
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return _call(req)


st, d = post("/auth/login", {"username": USER, "password": PWD})
tok = None
if isinstance(d, dict):
    tok = d.get("access_token") or d.get("token") or (d.get("data") or {}).get("access_token")
print("login", st, "token?", bool(tok))
if tok:
    open(os.path.join(os.path.dirname(__file__), ".tok"), "w").write(tok)

queries = {
    "knowledge_search_vortex": "/knowledge/search?query=" + urllib.parse.quote("isentropic vortex"),
    "knowledge_search_yee": "/knowledge/search?query=" + urllib.parse.quote("Yee entropy splitting"),
    "negkb_TensorDG": "/tensor/negative-knowledge/search?keywords=TensorDG&limit=50",
    "negkb_vortex": "/tensor/negative-knowledge/search?keywords=vortex&limit=50",
    "negkb_yee": "/tensor/negative-knowledge/search?keywords=Yee&limit=50",
}
out = {"login_status": st, "has_token": bool(tok)}
for k, p in queries.items():
    s, r = get(p, tok)
    out[k] = {"status": s, "data": r}
    print("==", k, s, str(r)[:1200])

json.dump(out, open(os.path.join(os.path.dirname(__file__), "kb_out.json"), "w"),
          ensure_ascii=False, indent=1)