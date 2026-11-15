import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tp_req import req, login

t = login()
rid = sys.argv[1]
d = req("tensor/reports/%s" % rid, token=t)
r = d.get("data", d) if isinstance(d, dict) else d
for k in ("id", "title", "summary", "scenario_key", "mission_id", "tags"):
    v = r.get(k)
    if v is not None:
        print("%s: %s" % (k, v))
md = r.get("markdown") or ""
print("--- markdown len", len(md))
print(md[:6000])