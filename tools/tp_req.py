import json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
U = os.environ["TENSOR_AGENT_USER"]
P = os.environ["TENSOR_AGENT_PASS"]


def req(path, body=None, method=None, token=None):
    url = BASE + "/" + path.lstrip("/")
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "body": e.read().decode()[:800]}


def login():
    tok = req("auth/login", {"username": U, "password": P})
    return tok.get("access_token") or tok.get("data", {}).get("access_token", "")


if __name__ == "__main__":
    import sys
    t = login()
    open("runs/.tok", "w").write(t) if os.path.isdir("runs") else None
    path = sys.argv[1] if len(sys.argv) > 1 else "tensor/reports?limit=15"
    d = req(path, token=t)
    items = d if isinstance(d, list) else (d.get("data") or d.get("items") or [])
    if isinstance(items, list) and items and isinstance(items[0], dict):
        for r in items[:20]:
            print(r.get("id"), "|", str(r.get("title", ""))[:110], "|", str(r.get("created_at", ""))[:19])
    else:
        print(json.dumps(d, ensure_ascii=False)[:3000])