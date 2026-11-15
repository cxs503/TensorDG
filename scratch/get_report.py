import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok.txt").read().strip()


def get(path):
    req = urllib.request.Request(base + path)
    req.add_header("Authorization", "Bearer " + tok)
    with urllib.request.urlopen(req) as f:
        return json.load(f)


r = get("/tensor/reports/19743")
print("TITLE:", r.get("title"))
print("SCENARIO:", r.get("scenario_key"))
print("MISSION:", r.get("mission_id"))
md = r.get("markdown") or r.get("content") or ""
print("LEN", len(md))
print(md[:7000])