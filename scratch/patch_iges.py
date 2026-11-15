"""One-shot patch: IGES parameter-data column handling + units-name parsing.

Details fixed:
  * PD records are 80 columns; columns 65-72 hold the *directory-entry pointer*
    and 73-80 hold section letter + sequence number.  The old implementation kept
    8 extra columns of pointer digits, which corrupted entity-type parsing so no
    CARTESIAN POINT (type 116) was ever recognised.
  * The Global section carries the units as an integer flag (field 12) *and* as a
    Hollerith string (field 14, e.g. ``1HM``); the string is used when the flag is
    not a plain integer.
"""
import re
from pathlib import Path

p = Path("scripts/hull_source_hunt.py")
src = p.read_text()

NEW = '''def iges_bbox(text: str) -> dict:
    """Bounding box over all CARTESIAN POINT (type 116) entities of an IGES file.

    IGES records are 80 columns: 1-8 entity type, 9-64 parameter data, 65-72
    directory-entry pointer, 73 section letter, 74-80 sequence number.  Only the
    parameter data of the P section is scanned.
    """
    pd: list[str] = []
    units_flag = None
    units_name = None
    lines = text.splitlines()
    gtext = "".join(l[:72] for l in lines
                    if len(l) > 72 and l[72] == "G")
    if gtext:
        flds = [f.strip() for f in gtext.split(",")]
        for i, f in enumerate(flds):
            m = re.fullmatch(r"(\\d+)H([A-Za-z]+)", f)
            if m and i > 0:
                prev = flds[i - 1]
                if prev.isdigit() and 1 <= int(prev) <= 11:
                    units_flag = int(prev)
                    units_name = m.group(2).upper()
                    break
        if units_flag is None:                      # older writers: 12th field only
            for f in flds[:16]:
                if f.isdigit() and 1 <= int(f) <= 11:
                    units_flag = int(f)
                    break
    for line in lines:
        if len(line) > 72 and line[72] == "P":
            pd.append(line[:64])
    blob = re.sub(r"\\s+", "", "".join(pd))
    xs, ys, zs = [], [], []
    for rec in blob.split(";"):
        if len(rec) < 9:
            continue
        try:
            etype = int(rec[0:8])
        except ValueError:
            continue
        if etype != 116:                            # 116 = CARTESIAN POINT
            continue
        vals = []
        for tok in re.split(r"[,\\s]+", rec[8:]):
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                break
        if len(vals) >= 3:
            xs.append(vals[0]); ys.append(vals[1]); zs.append(vals[2])
    name_factor = {"M": 1.0, "MM": 0.001, "CM": 0.01, "IN": 0.0254, "FT": 0.3048,
                   "UM": 1e-6, "DM": 0.1, "KM": 1000.0}
    flag_factor = {1: 0.0254, 2: 0.001, 3: 0.01, 4: 1.0, 5: 0.3048, 6: 1.0,
                   9: 1e-6, 10: 0.1, 11: 1000.0}
    to_m = name_factor.get(units_name or "", flag_factor.get(units_flag, 1.0))
    if not xs:
        return {"points": 0, "units_flag": units_flag, "units_name": units_name,
                "units_to_m": to_m}
    return {
        "points": len(xs),
        "units_flag": units_flag,
        "units_name": units_name,
        "units_to_m": to_m,
        "bbox_raw": {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)], "z": [min(zs), max(zs)]},
        "span_m": {"x": round((max(xs) - min(xs)) * to_m, 4),
                   "y": round((max(ys) - min(ys)) * to_m, 4),
                   "z": round((max(zs) - min(zs)) * to_m, 4)},
    }


'''

start = src.index("def iges_bbox(text: str) -> dict:")
end = src.index("def iges_bbox_in_zip(")
src = src[:start] + NEW + src[end:]
p.write_text(src)
print("patched", p, len(src))