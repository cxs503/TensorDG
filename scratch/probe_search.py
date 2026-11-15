import json, urllib.request, urllib.parse, socket
socket.setdefaulttimeout(20)
UA = {"User-Agent": "Mozilla/5.0 (research provenance agent)"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.read()

def gh(path):
    try:
        b = get("https://api.github.com" + path)
        return json.loads(b)
    except Exception as e:
        return {"_err": f"{type(e).__name__}: {e}"}

for q in ["Duisburg Test Case", "DTC hull geometry", "duisburg test case dtc"]:
    d = gh("/search/repositories?q=" + urllib.parse.quote(q) + "&per_page=5")
    print("==", q)
    if "_err" in d:
        print("  ERR", d["_err"]); continue
    for it in d.get("items", []):
        print("  ", it["full_name"], it.get("size"), it.get("description", "")[:90])

# Bing HTML search
for q in ["Duisburg Test Case DTC hull geometry download IGES",
          "\"DTC\" hull geometry zip container ship benchmark download"]:
    url = "https://www.bing.com/search?q=" + urllib.parse.quote(q) + "&count=20"
    try:
        h = get(url).decode("utf-8", "replace")
    except Exception as e:
        print("BING ERR", q, e); continue
    print("==", q, len(h))
    import re
    for m in re.finditer(r'<h2><a href="(http[^"]+)"[^>]*>(.*?)</a></h2>', h):
        print("   ", m.group(1)[:110], "|", re.sub("<[^>]+>", "", m.group(2))[:80])