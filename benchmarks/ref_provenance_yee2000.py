"""Machine-checked provenance for the ``CASE-DG-EULER-VORTEX-ERR`` reference.

The case-pool brief names **Yee et al. 2000** as the reference and requires the
target numbers to be looked up rather than typed in by hand.  The honest way to do
that from inside a restricted network is:

1. resolve the bibliographic record mechanically (Crossref by title/author),
2. ask the OA indexes what full-text copy exists (Unpaywall, OpenAlex),
3. attempt the retrieval and log the *actual* HTTP outcome of every URL,
4. publish the resulting ledger next to the case so a reader can see exactly which
   numbers are first-hand and which are ``WITHHELD``.

Nothing here parses or transcribes the paper.  If the PDF cannot be fetched the
ledger says so and the case falls back to the analytic solution, which is
machine-verified by ``benchmarks/case_euler_vortex.py --task residual``.

    python benchmarks/ref_provenance_yee2000.py [--output artifacts/ref_provenance_yee2000.json]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TITLE_QUERY = "Entropy splitting and numerical dissipation Yee Vinokur Djomehri"
CROSSREF = "https://api.crossref.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"
OPENALEX = "https://api.openalex.org/works/doi:{doi}"
EMAIL = "tensor-agent@example.org"          # Unpaywall requires a contact address
USER_AGENT = "tensor-agent/1.0 (provenance ledger; case CASE-DG-EULER-VORTEX-ERR)"


def _get_json(url: str, timeout: float = 25.0):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def _probe(url: str, timeout: float = 40.0) -> dict:
    """Fetch ``url`` and report what actually came back (never raises)."""
    entry = {"url": url, "attempted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                  "Accept": "application/pdf,*/*"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(4_000_000)
            entry.update({"http_status": response.status, "bytes": len(payload),
                          "content_type": response.headers.get("Content-Type", "")})
            entry["pdf_magic"] = payload[:5] == b"%PDF-"
            entry["verdict"] = "RETRIEVED" if entry["pdf_magic"] else "NOT_A_PDF"
            if not entry["pdf_magic"] and "html" in entry["content_type"].lower():
                entry["pdf_links"] = _pdf_links(payload.decode("utf-8", "replace"), url)
    except urllib.error.HTTPError as exc:      # 403 / 404 / 429 ...
        entry.update({"http_status": exc.code, "verdict": "HTTP_ERROR"})
        body = exc.read()[:400].decode("utf-8", "replace")
        entry["body_excerpt"] = " ".join(body.split())[:240]
    except Exception as exc:                    # DNS failure, TLS refusal, timeout
        entry.update({"http_status": None, "verdict": "UNREACHABLE",
                      "error": f"{type(exc).__name__}: {exc}"})
    entry["elapsed_s"] = round(time.time() - started, 2)
    return entry


def _pdf_links(html: str, base: str, limit: int = 3) -> list[str]:
    """Absolute PDF-ish links found on a landing page (mechanical, no inference)."""
    import html as _html
    import re
    found = []
    for match in re.findall(r'href="([^"]+)"', html):
        target = _html.unescape(match)
        if ".pdf" in target.lower() or "viewcontent" in target.lower():
            absolute = urllib.parse.urljoin(base, target)
            if absolute not in found:
                found.append(absolute)
        if len(found) >= limit:
            break
    return found


def build_ledger(probe: bool = True) -> dict:
    ledger: dict = {
        "anchor": "CASE-DG-EULER-VORTEX-ERR",
        "mission": "m20056",
        "cited_as": "Yee et al. 2000",
        "protocol": ("bibliographic record via Crossref; OA location via Unpaywall and "
                     "OpenAlex; retrieval attempts probed and logged verbatim"),
        "queried_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # 1. bibliographic identity
    crossref_url = (CROSSREF + "?" + urllib.parse.urlencode({
        "query.bibliographic": TITLE_QUERY, "rows": 3,
        "select": "DOI,title,author,container-title,volume,page,issued",
    }))
    try:
        status, payload = _get_json(crossref_url)
        items = payload["message"]["items"]
        best = items[0]
        ledger["bibliographic"] = {
            "source": "Crossref",
            "http_status": status,
            "doi": best.get("DOI"),
            "title": (best.get("title") or [""])[0],
            "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip()
                        for a in best.get("author", [])],
            "journal": (best.get("container-title") or [""])[0],
            "volume": best.get("volume"),
            "page": best.get("page"),
            "issued": best.get("issued"),
            "alternatives": [{"doi": it.get("DOI"),
                              "title": (it.get("title") or [""])[0]}
                             for it in items[1:]],
        }
    except Exception as exc:
        ledger["bibliographic"] = {"error": f"{type(exc).__name__}: {exc}"}

    doi = (ledger.get("bibliographic") or {}).get("doi")
    if not doi:
        ledger["status"] = "UNRESOLVED"
        return ledger
    quoted = urllib.parse.quote(doi, safe="")

    # 2. open-access locations
    candidates: list[str] = []
    ledger["open_access"] = {}
    try:
        status, payload = _get_json(UNPAYWALL.format(doi=quoted) + f"?email={EMAIL}")
        locations = payload.get("oa_locations") or []
        ledger["open_access"]["unpaywall"] = {
            "http_status": status, "is_oa": payload.get("is_oa"),
            "oa_status": payload.get("oa_status"),
            "locations": [{"host_type": loc.get("host_type"),
                           "url": loc.get("url"),
                           "url_for_pdf": loc.get("url_for_pdf")} for loc in locations],
        }
        candidates += [loc.get("url_for_pdf") or loc.get("url") for loc in locations]
        candidates += [loc.get("url") for loc in locations if loc.get("url_for_pdf")]
    except Exception as exc:
        ledger["open_access"]["unpaywall"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        status, payload = _get_json(OPENALEX.format(doi=quoted))
        ledger["open_access"]["openalex"] = {
            "http_status": status,
            "oa_url": (payload.get("open_access") or {}).get("oa_url"),
            "locations": [{"is_oa": loc.get("is_oa"),
                           "landing_page_url": loc.get("landing_page_url"),
                           "pdf_url": loc.get("pdf_url")}
                          for loc in payload.get("locations") or []],
        }
        candidates.append((payload.get("open_access") or {}).get("oa_url"))
        candidates += [loc.get("pdf_url") for loc in payload.get("locations") or []]
        candidates += [loc.get("landing_page_url") for loc in payload.get("locations") or []]
    except Exception as exc:
        ledger["open_access"]["openalex"] = {"error": f"{type(exc).__name__}: {exc}"}

    ordered = []
    for url in candidates:
        if url and url not in ordered:
            ordered.append(url)
    ledger["candidate_urls"] = ordered

    # 3. retrieval attempts (landing pages then the PDF links they advertise)
    if probe:
        attempts = [_probe(url) for url in ordered]
        followed = []
        for attempt in attempts:
            for link in attempt.get("pdf_links") or []:
                if link not in ordered and link not in followed:
                    followed.append(link)
        attempts += [_probe(url) for url in followed]
        ledger["retrieval_attempts"] = attempts
    else:
        ledger["retrieval_attempts"] = []

    retrieved = [a for a in ledger["retrieval_attempts"] if a.get("verdict") == "RETRIEVED"]
    ledger["full_text_status"] = "RETRIEVED" if retrieved else "WITHHELD"
    ledger["numeric_target_policy"] = (
        "paper table used as numeric target" if retrieved else
        "paper table NOT retrieved -> published values graded WITHHELD; acceptance "
        "numeric target is the analytic isentropic vortex, machine-verified by "
        "benchmarks/case_euler_vortex.py --task residual (div F = 0 in the steady frame, "
        "U_t + F_x + G_y = 0 in the advected frame, isentropic law, zero divergence, "
        "radial momentum balance)")
    ledger["status"] = "RESOLVED" if retrieved else "RESOLVED_TABLE_WITHHELD"
    return ledger


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="artifacts/ref_provenance_yee2000.json")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip PDF retrieval attempts (bibliographic record only)")
    args = parser.parse_args(argv)
    ledger = build_ledger(probe=not args.no_probe)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=1, ensure_ascii=False))
    print(json.dumps({"status": ledger["status"],
                      "doi": (ledger.get("bibliographic") or {}).get("doi"),
                      "full_text_status": ledger.get("full_text_status"),
                      "attempts": [{k: a.get(k) for k in ("url", "http_status", "verdict")}
                                   for a in ledger.get("retrieval_attempts", [])]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())