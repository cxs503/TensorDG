"""Fetch the external reference table for CASE-DG-ADVECT-ORDER (zero hand-copy).

The case is the 1-D periodic linear-advection convergence study with the smooth
manufactured solution ``u(x,0) = sin(2*pi*x)`` advected for one period.  The
reference numbers are **not** transcribed by hand: this script downloads the
author's LaTeX source of

    Zheng Sun, "Reducing polynomial degree by one for inner-stage operators
    affects neither stability type nor accuracy order of the Runge-Kutta
    discontinuous Galerkin method", Math. Comp. (2024), DOI 10.1090/mcom/4037,
    arXiv:2404.15453v2

and extracts the table labelled ``tab:acc-sin-1d-uni`` (1-D linear advection,
``u(x,0)=sin(2 pi x)``, ``t^n = 1``, uniform meshes ``N = 20 ... 320``, columns
``RK2DG1 / RK3DG2 / RK4DG3 / RK5DG4`` with L2 error and printed order).

Output: ``validation/advection_order/ref_table.json`` including the sha256 of
the downloaded e-print archive and of the extracted table block, so that the
numbers in later reports are traceable to one exact source revision.

Run: ``PYTHONPATH=. python tools/fetch_ref_advorder.py``
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import re
import tarfile
import urllib.request
from datetime import datetime, timezone

ARXIV_ID = "2404.15453v2"
EPRINT_URL = f"https://arxiv.org/e-print/{ARXIV_ID}"
TABLE_LABEL = "tab:acc-sin-1d-uni"
OUT = pathlib.Path("validation/advection_order/ref_table.json")

CITATION = {
    "author": "Zheng Sun",
    "title": ("Reducing polynomial degree by one for inner-stage operators "
              "affects neither stability type nor accuracy order of the "
              "Runge-Kutta discontinuous Galerkin method"),
    "journal": "Mathematics of Computation (AMS)",
    "doi": "10.1090/mcom/4037",
    "date": "2024-10-31",
    "preprint": f"arXiv:{ARXIV_ID}",
    "table_label": TABLE_LABEL,
    "experiment": ("1-D linear advection u_t + u_x = 0 on the periodic unit "
                   "interval, u(x,0)=sin(2 pi x), uniform meshes, t^n = 1"),
}

# (column header in the paper, DG degree of the final stage, RK stages used)
COLUMNS = [("RK2DG1", 1, 2), ("RK3DG2", 2, 3), ("RK4DG3", 3, 4), ("RK5DG4", 4, 5)]


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tensordg-agent/1.0"})
    with urllib.request.urlopen(req, timeout=120) as handle:
        return handle.read()


def _paper_tex(blob: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tex = [n for n in tar.getnames() if n.endswith(".tex")]
        if not tex:
            raise RuntimeError("no .tex file in the e-print archive")
        # The paper has a single source file; fail loudly if that ever changes.
        tex.sort(key=len, reverse=True)
        return tar.extractfile(tex[0]).read().decode("utf-8", "ignore")


def _table_block(tex: str, label: str) -> str:
    for match in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S):
        block = match.group(1)
        if f"\\label{{{label}}}" in block:
            return block
    raise RuntimeError(f"table label {label} not found in the source")


def _caption(block: str) -> str:
    match = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|$)", block, re.S)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def _parse_rows(block: str) -> list[dict]:
    """Parse the ``RKDG`` half of the table into ``{N: {column: {...}}}``."""
    body = block.split("sdA-RKDG")[0]  # only the standard RKDG block
    rows = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.split("&")]
        if len(cells) < 1 + 2 * len(COLUMNS):
            continue
        try:
            n = int(cells[0])
        except ValueError:
            continue  # header / rule lines
        record = {"N": n}
        for idx, (name, degree, rk) in enumerate(COLUMNS):
            err_raw = cells[1 + 2 * idx]
            order_raw = cells[2 + 2 * idx].replace("\\", "").strip()
            record[name] = {
                "degree": degree,
                "rk_stages": rk,
                "l2": float(err_raw.replace("\\", "").strip()),
                "order_printed": (None if order_raw in ("-", "")
                                  else float(order_raw)),
            }
        rows.append(record)
    if not rows:
        raise RuntimeError("no data rows parsed from the reference table")
    return rows


def main() -> dict:
    blob = _download(EPRINT_URL)
    tex = _paper_tex(blob)
    block = _table_block(tex, TABLE_LABEL)
    rows = _parse_rows(block)
    payload = {
        "case": "CASE-DG-ADVECT-ORDER",
        "citation": CITATION,
        "provenance": {
            "eprint_url": EPRINT_URL,
            "eprint_sha256": hashlib.sha256(blob).hexdigest(),
            "eprint_bytes": len(blob),
            "table_block_sha256": hashlib.sha256(block.encode()).hexdigest(),
            "table_caption": _caption(block),
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "extractor": "tools/fetch_ref_advorder.py",
            "note": ("numbers parsed mechanically from the author's LaTeX "
                     "source; no value is transcribed by hand"),
        },
        "columns": [{"name": n, "degree": d, "rk_stages": r}
                    for n, d, r in COLUMNS],
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(rows)} rows, N = {[r['N'] for r in rows]})")
    print("eprint sha256:", payload["provenance"]["eprint_sha256"])
    for row in rows:
        print(row["N"], {k: row[k]["l2"] for k in row if k != "N"})
    return payload


if __name__ == "__main__":
    main()