import re, urllib.request, socket
socket.setdefaulttimeout(20)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "replace")

h = get("https://www.uni-due.de/inam/")
open("scratch/inam.html", "w").write(h)
print("len", len(h))
for kw in ["DTC", "Duisburg Test Case", "geometry", "Geometrie", "IGES", "benchmark"]:
    hits = [m.start() for m in re.finditer(kw, h, re.I)][:5]
    print(kw, len(hits))
    for p in hits[:3]:
        print("   ...", re.sub(r"\s+", " ", h[max(0, p-120):p+160]))
links = sorted(set(re.findall(r'href="([^"]+)"', h)))
print("links:", len(links))
for l in links[:60]:
    print("  ", l)