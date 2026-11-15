import re, urllib.request, urllib.parse, socket
socket.setdefaulttimeout(20)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.read(), r.status, r.geturl()

for u in ["https://www.uni-due.de/ist/", "https://www.uni-due.de/", "https://codeload.github.com/",
          "https://github.com/search?q=Duisburg+Test+Case&type=repositories"]:
    try:
        b, s, g = get(u)
        print(u, "->", s, len(b), g)
        if "github.com/search" in u:
            h = b.decode("utf-8", "replace")
            for m in re.finditer(r'href="(/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)"', h):
                print("   repo?", m.group(1))
    except Exception as e:
        print(u, "ERR", type(e).__name__, e)