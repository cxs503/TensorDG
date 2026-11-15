import urllib.request, socket
socket.setdefaulttimeout(12)
hosts = [
    "https://github.com/", "https://api.github.com/", "https://raw.githubusercontent.com/",
    "https://duckduckgo.com/", "https://www.bing.com/", "https://simman2014.dk/",
    "https://www.t2015.nmri.go.jp/", "https://www.hsva.de/", "https://www.simman2020.kr/",
    "https://web.archive.org/", "https://pypi.org/simple/", "https://zenodo.org/",
    "https://t2015.nmri.go.jp/file/Geometry_IGES_files/onrt/onrt_w_gap.zip",
    "https://t2015.nmri.go.jp/file/Geometry_IGES_files/onrt/propeller_IIHR2.zip",
    "http://www.duisburg-test-case.com/",
    "https://www.duisburg-test-case.com/",
]
for h in hosts:
    try:
        req = urllib.request.Request(h, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        r = urllib.request.urlopen(req)
        print(f"{h:75s} OK {r.status} {r.headers.get('content-length')}")
    except Exception as e:
        print(f"{h:75s} ERR {type(e).__name__}: {e}")