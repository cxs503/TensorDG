#!/usr/bin/env python3
"""Render the ``CASE-LAX-DG`` figures from a committed record (headless nodes).

The pool driver renders its figures inline, which makes the *record* hostage to
the plotting stack: the loong64 compute nodes carry ``numpy``/``torch`` but no
``matplotlib``, so ``python -m benchmarks.lax_case_pool`` died at
``render_figures`` *after* 22 min of solver time and wrote no ``metrics.json``
at all.  This script decouples the two: the cluster run writes the numbers
(``--skip-figures``), and the figures are rendered here from the record plus
``profiles.npz`` -- the same arrays the driver would have plotted.

Every input is read back from the artifact, so the figures cannot show a run
other than the one in ``metrics.json``::

    python scripts/render_lax_figures.py            # pool + limiter arms
    python scripts/render_lax_figures.py --pool     # only the pool record
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.lax_case_pool import (  # noqa: E402
    OUT as POOL_OUT, render_acceptance_figure, render_figures,
    render_time_figure,
)
from benchmarks.lax_case_pool_limiter_arms import OUT as ARMS_OUT  # noqa: E402
from benchmarks.lax_case_pool_limiter_arms import render_figures as render_arms  # noqa: E402

FIELDS = ("density", "velocity", "pressure")
PROVENANCE = ("scripts/render_lax_figures.py (record + profiles.npz; "
              "rendered off-node: the loong64 compute image has no matplotlib)")


def _as_list(value):
    """``render_*_figure`` helpers return either one ``Path`` or a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _names(files):
    return sorted({Path(p).name for p in files})


def render_pool(root=ROOT):
    rec_dir = root / POOL_OUT
    metrics = json.loads((rec_dir / "metrics.json").read_text())
    npz = np.load(rec_dir / "profiles.npz")
    profiles = {"points": npz["points"],
                "numerical": {f: npz[f"numerical_{f}"] for f in FIELDS},
                "exact": {f: npz[f"exact_{f}"] for f in FIELDS}}
    ladder = metrics["ladder"]
    # ``render_figures`` only needs the finest leg's identity for the title
    finest = {"cells": ladder[-1]["cells"], "degree": ladder[-1]["degree"]}
    fig_dir = rec_dir / "figures"
    # JSON stringifies the integer cell counts that double as dict keys
    entropy = [dict(e, relative_l1={int(k): v
                                    for k, v in e["relative_l1"].items()})
               for e in metrics["entropy_wave"]]
    files = _as_list(render_figures(metrics["waves"], ladder, finest, profiles,
                                    entropy, fig_dir))
    if metrics.get("time_refinement"):
        files.extend(_as_list(render_time_figure(metrics["time_refinement"], fig_dir)))
    files.extend(_as_list(render_acceptance_figure(ladder, metrics["acceptance"], fig_dir)))
    metrics["figures"] = _names(files)
    metrics["figures_provenance"] = PROVENANCE
    (rec_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics["figures"]


def render_arms_record(root=ROOT):
    rec_dir = root / ARMS_OUT
    metrics = json.loads((rec_dir / "metrics.json").read_text())
    npz = np.load(rec_dir / "profiles.npz")
    arm_results, profiles = {}, {}
    from benchmarks.lax_case_pool import exact_waves  # record drops "waves"
    for arm, record in metrics["arms"].items():
        arm_results[arm] = dict(record, waves=exact_waves())
        profiles[arm] = {
            "points": npz[f"{arm}_points"],
            "numerical": {f: npz[f"{arm}_numerical_{f}"] for f in FIELDS},
            "exact": {f: npz[f"{arm}_exact_{f}"] for f in FIELDS},
        }
    files = render_arms(arm_results, profiles, rec_dir / "figures")
    metrics["figures"] = _names(files)
    metrics["figures_provenance"] = PROVENANCE
    (rec_dir / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")
    return metrics["figures"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", action="store_true", help="pool record only")
    parser.add_argument("--arms", action="store_true", help="limiter arms only")
    args = parser.parse_args(argv)
    which = [("pool", render_pool), ("arms", render_arms_record)]
    if args.pool or args.arms:
        which = [(n, f) for n, f in which if (args.pool and n == "pool")
                 or (args.arms and n == "arms")]
    for name, fn in which:
        try:
            print(f"[figures] {name}: {fn()}")
        except Exception as exc:  # a missing figure must not hide the numbers
            print(f"[figures] {name}: FAILED {type(exc).__name__}: {exc}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
