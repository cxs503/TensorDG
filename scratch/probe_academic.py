import urllib.request, socket
socket.setdefaulttimeout(15)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36"}
urls = [
 "https://api.openalex.org/works?search=Duisburg%20Test%20Case&per-page=3",
 "https://api.crossref.org/works?query.title=Duisburg+Test+Case+Post-Panamax+Container+Ship&rows=3",
 "https://www.mdpi.com/",
 "https://link.springer.com/",
 "https://arxiv.org/abs/2101.00001",
 "https://core.ac.uk/",
 "https://www.semanticscholar.org/",
 "https://api.semanticscholar.org/graph/v1/paper/search?query=Duisburg%20Test%20Case&limit=3",
 "https://www.frontiersin.org/",
 "https://ojs.cvut.cz/",
 "https://www.shipstructure.org/",
 "https://gitlab.com/",
]
for u in urls:
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers=UA))
        print(f"{u:80s} OK {r.status} {r.headers.get('content-type','')[:40]}")
    except Exception as e:
        print(f"{u:80s} ERR {type(e).__name__} {str(e)[:60]}")