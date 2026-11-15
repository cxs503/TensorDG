import hashlib, os, urllib.request, socket, zipfile, io
socket.setdefaulttimeout(120)
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120 Safari/537.36"}
url = "https://www.uni-due.de/imperia/md/content/inam/igs_ordner.zip"
os.makedirs("/root/.cache/hull_sources", exist_ok=True)
req = urllib.request.Request(url, headers=UA)
with urllib.request.urlopen(req) as r:
    data = r.read()
    print("status", r.status, "ctype", r.headers.get("content-type"), "clen", r.headers.get("content-length"))
print("bytes", len(data), "sha256", hashlib.sha256(data).hexdigest())
open("/root/.cache/hull_sources/dtc_igs_ordner.zip", "wb").write(data)
z = zipfile.ZipFile(io.BytesIO(data))
for i in z.infolist()[:40]:
    print(f"  {i.filename:60s} {i.file_size}")