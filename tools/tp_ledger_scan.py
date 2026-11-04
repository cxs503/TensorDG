"""Read the platform report ledger (read-only) to scan recent 局限/待办 sections.

Usage: python3 tools/tp_ledger_scan.py [limit]
Prints one line per report: id, created_at, title, and any '局限'/'待办'/'未完成' lines
truncated to a bounded length, so the output stays small enough for an agent context.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(path: str, token: str) -> dict:
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def login() -> str:
    d = _post(
        "/auth/login",
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        },
    )
    return d.get("access_token") or d.get("token")


KEY = re.compile(r"(局限|待办|未完成|honest[- ]?FAIL|建议)")


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tok = login()
    d = _get(f"/tensor/reports?limit={limit}", tok)
    items = d if isinstance(d, list) else d.get("items", d.get("reports", []))
    print(f"# {len(items)} reports")
    for r in items:
        print(f"\n--- id={r.get('id')} {str(r.get('created_at'))[:16]} :: {str(r.get('title'))[:100]}")
        md = r.get("markdown") or r.get("summary") or ""
        hits = [ln.strip() for ln in md.splitlines() if KEY.search(ln)]
        for ln in hits[:8]:
            print("   ", ln[:160])


if __name__ == "__main__":
    main()