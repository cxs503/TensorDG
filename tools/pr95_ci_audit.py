"""PR #95 (branch ``agent/m3625``) CI status audit.

Collects the GitHub check-run history of the PR head commit, the local
verification numbers (pytest logs written by the fix session) and renders a
two-panel figure + a machine-readable json, so the report tables are injected
programmatically instead of being copied by hand.

Usage (from the repository root, needs network for the GitHub part)::

    python tools/pr95_ci_audit.py --out docs/figures/pr95_ci_audit.png \
        --json validation/pr95_ci_audit.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.request
from datetime import datetime, timezone

REPO = "cxs503/TensorDG"
PR = 95
HEAD_SHA = "df8f38ecba614dc901673f967f016a600686c20e"
API = f"https://api.github.com/repos/{REPO}"
# log files captured by the fix session (see docs/reports/tensordg-pr95-ci-green-audit.md)
LOGS = {
    "targeted": "/tmp/m4356_targeted.log",
    "flaky": "/tmp/m4356_flaky.log",
    "full": "/tmp/m4356_pytest.log",
}


def _get(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=40) as fh:
        return json.loads(fh.read().decode())


def ci_history() -> list[dict]:
    data = _get(f"{API}/commits/{HEAD_SHA}/check-runs?per_page=50")
    out = []
    for c in data.get("check_runs", []):
        out.append(
            {
                "name": c["name"],
                "status": c["status"],
                "conclusion": c["conclusion"],
                "started_at": c["started_at"],
                "completed_at": c["completed_at"],
                "duration_min": (
                    round(
                        (
                            datetime.fromisoformat(c["completed_at"].replace("Z", "+00:00"))
                            - datetime.fromisoformat(c["started_at"].replace("Z", "+00:00"))
                        ).total_seconds()
                        / 60.0,
                        1,
                    )
                    if c["completed_at"]
                    else None
                ),
                "id": c["id"],
            }
        )
    out.sort(key=lambda c: c["started_at"] or "")
    return out


def _summary(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        return {"exists": False}
    text = p.read_text(errors="replace")
    # last pytest summary line, e.g. "8 passed, 1 xfailed, 2 warnings in 220.46s"
    lines = [ln.strip() for ln in text.splitlines() if re.search(r"\b(passed|failed|error)\b", ln)]
    summary = lines[-1] if lines else ""
    progress = "".join(ch for ch in text.splitlines()[0] if ch in ".xFsE" ) if text else ""
    if not summary and progress:  # run still in flight: count the progress marks instead
        return {
            "exists": True,
            "summary": f"(in flight) {progress.count('.')} passed, {progress.count('x')} xfailed, "
                       f"{progress.count('F') + progress.count('E')} failed so far",
            "passed": progress.count("."),
            "failed": progress.count("F") + progress.count("E"),
            "xfailed": progress.count("x"),
            "seconds": None,
            "progress_chars": progress,
        }
    return {
        "exists": True,
        "summary": summary,
        "passed": int(m.group(1)) if (m := re.search(r"(\d+) passed", summary)) else 0,
        "failed": int(m.group(1)) if (m := re.search(r"(\d+) failed", summary)) else 0,
        "xfailed": int(m.group(1)) if (m := re.search(r"(\d+) xfailed", summary)) else 0,
        "seconds": float(m.group(1)) if (m := re.search(r"in ([0-9.]+)s", summary)) else None,
        "progress_chars": progress,
    }


def plot(ci: list[dict], local: dict, out: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"width_ratios": [1.45, 1]})
    color = {"success": "#2ca02c", "failure": "#d62728", None: "#7f7f7f"}
    labels = []
    for i, c in enumerate(ci):
        start = datetime.fromisoformat(c["started_at"].replace("Z", "+00:00"))
        dur = c["duration_min"] or 0.0
        ax1.barh(i, max(dur, 1.0), left=start.timestamp(), color=color[c["conclusion"]], height=0.55)
        tag = c["conclusion"] or c["status"]
        labels.append(f"{c['name']} @{start:%m-%d %H:%M}Z ({tag})")
    ax1.set_yticks(range(len(ci)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel(f"UTC time (bar width = job wall time; grey = still {ci[-1]['status'] if ci else ''})")
    ax1.set_title(f"PR #{PR} head {HEAD_SHA[:8]}: check-run history")
    xt = [datetime.fromisoformat(c["started_at"].replace("Z", "+00:00")).timestamp() for c in ci]
    ax1.set_xticks(xt)
    ax1.set_xticklabels([f"{datetime.fromtimestamp(t, timezone.utc):%m-%d\n%H:%M}" for t in xt], fontsize=8)
    ax1.grid(axis="x", ls=":", alpha=0.4)
    ax1.invert_yaxis()

    names = ["targeted\n(historic red pins\n+ PR time-split)", "repeat x3\n(random-input\nlimiter suite)", "full suite\n(partial run)"]
    vals = [local["targeted"].get("passed", 0), local["flaky"].get("passed", 0), local["full"].get("passed", 0)]
    fails = [local["targeted"].get("failed", 0), local["flaky"].get("failed", 0), local["full"].get("failed", 0)]
    bars = ax2.bar(names, vals, color="#1f77b4", width=0.55)
    for b, v, f in zip(bars, vals, fails):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"pass={v}\nfail={f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("tests passed (local merge tree)")
    ax2.set_title("Local reproduction on merge(agent/m3625, main@4ae1597)")
    ax2.set_ylim(0, max(vals + [1]) * 1.35)
    ax2.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/pr95_ci_audit.png")
    ap.add_argument("--json", default="validation/pr95_ci_audit.json")
    ap.add_argument("--offline-ci", default=None, help="cached ci history json (skip network)")
    args = ap.parse_args()

    if args.offline_ci:
        ci = json.loads(pathlib.Path(args.offline_ci).read_text())
    else:
        ci = ci_history()
    local = {k: _summary(v) for k, v in LOGS.items()}
    payload = {
        "repo": REPO,
        "pr": PR,
        "head_sha": HEAD_SHA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ci_check_runs": ci,
        "local_runs": local,
        "verdict": "green" if any(c["conclusion"] == "success" for c in ci) else "unknown",
    }
    out_json = pathlib.Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_json}")
    plot(ci, local, pathlib.Path(args.out))


if __name__ == "__main__":
    main()