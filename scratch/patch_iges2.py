"""Second patch: IGES parameter records in these archives are written free-format
('128,9,9,3,3,...;' — entity type followed directly by its parameter list) rather
than having the entity type right-justified in columns 1-8.  Parse records by
tokenising on commas, and fall back to the fixed 8-column form when present.
"""
import re
from pathlib import Path

p = Path("scripts/hull_source_hunt.py")
src = p.read_text()

NEW_LOOP = '''    blob = re.sub(r"\\s+", "", "".join(pd))
    xs, ys, zs = [], [], []
    for rec in blob.split(";"):
        rec = rec.strip()
        if len(rec) < 3:
            continue
        toks = [t for t in rec.split(",") if t]
        etype = None
        params = []
        if toks and re.fullmatch(r"\\d+", toks[0]):
            etype = int(toks[0])
            params = toks[1:]
        elif len(rec) >= 8 and re.fullmatch(r"\\d+", rec[0:8].strip() or "x"):
            etype = int(rec[0:8])
            params = [t for t in re.split(r"[,\\s]+", rec[8:]) if t]
        if etype != 116:                            # 116 = CARTESIAN POINT
            continue
        vals = []
        for tok in params[:3]:
            try:
                vals.append(float(tok))
            except ValueError:
                break
        if len(vals) >= 3:
            xs.append(vals[0]); ys.append(vals[1]); zs.append(vals[2])
'''

start = src.index('    blob = re.sub(r"\\s+", "", "".join(pd))')
end = src.index("    name_factor = {")
src = src[:start] + NEW_LOOP + src[end:]
p.write_text(src)
print("patched", len(src))