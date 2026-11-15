import re, urllib.request, urllib.parse, socket
socket.setdefaulttimeout(25)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}

def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8", "replace")

qs = [
  "Duisburg Test Case DTC hull geometry download IGES",
  "DTC hull geometry zip duisburg test case download",
  "\"duisburg-test-case\" geometry",
]
for q in qs:
    h = get("https://www.bing.com/search?q=" + urllib.parse.quote(q) + "&count=30")
    open("/tmp/bing_%d.html" % abs(hash(q)) % 10000, "w").write(h) if False else None
    print("==", q, len(h))
    # cite blocks
    for m in re.finditer(r'<li class="b_algo".*?</li>', h, re.S):
        blk = m.group(0)
        a = re.search(r'<a[^>]+href="(http[^"]+)"[^>]*>(.*?)</a>', blk, re.S)
        if a:
            title = re.sub("<[^>]+>", "", a.group(2))
            print("   ", a.group(1)[:120])
            print("      ", title.strip()[:100])
    print()