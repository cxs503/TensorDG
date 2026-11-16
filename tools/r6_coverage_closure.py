#!/usr/bin/env python3
"""CASE-DG-EULER-VORTEX-ERR round 25: R6 reproducibility coverage closure.

Round 22 landed the R6 "reproduction" instrument (re-solve a checkpoint in a
fresh process, compare every published scalar with ``results/*.json``; a
relative deviation > 1e-9 means the committed bundle is stale).  It declared
four targets -- the C1/C2 decision point, the longest ladder entry, the highest
degree, and the second code path -- but only two of them were ever re-solved
(``dg_d2_n32_t1`` and ``p0_n128_t1``); the longest ladder entry
(``dg_d2_n32_t10``) and the highest degree (``dg_d3_n32_t1``) were left
``not_run``.  Rounds 23/24 carried that debt forward verbatim.

This driver pays it: it re-solves **all four declared R6 targets** through the
production entry point in fresh processes (threads 1 and 4, plus the repeated
re-solve and the inherited-environment harness control, exactly as round 22
defined), grades every published scalar against ``results/*.json``, and lands
the *complete* R6 section into the committed instrument artifact.  R1-R5 are
untouched; only the R6 section and the package verdict are recomputed.

Every number in the emitted document is read from the machine artefact -- no
digit is hand-copied.

Usage::

    PYTHONPATH=. python3 tools/r6_coverage_closure.py            # re-solve + land
    PYTHONPATH=. python3 tools/r6_coverage_closure.py --dry-run  # report only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.vortex_reference_instrument_certification import (  # noqa: E402
    REPRODUCTION_TARGETS,
    certify_reproduction,
    markdown,
    verdict,
)

R6_DIR = Path("case_euler_vortex_errconserve_r6")
JSON_PATH = R6_DIR / "reproduction.json"
MD_PATH = R6_DIR / "reproduction.md"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="grade re-solves already on disk; do not re-solve")
    args = parser.parse_args(argv)

    started = time.time()
    print(f"[r6-closure] re-solving {len(REPRODUCTION_TARGETS)} declared targets "
          f"({'re-solve' if not args.dry_run else 'reuse on-disk'}) ...", flush=True)
    r6 = certify_reproduction(
        targets=REPRODUCTION_TARGETS,
        solve=not args.dry_run,
        workdir=str(R6_DIR),
    )
    print(f"[r6-closure] R6 done in {time.time() - started:.0f}s: "
          f"reproduced={r6['coverage']['reproduced']}/"
          f"{r6['coverage']['declared_ladder_entries']}, "
          f"worst={r6['worst']:.3e} (limit {r6['limit']:.0e}), "
          f"passed={r6['passed']}, margin_kept={r6['margin_kept']}", flush=True)

    existing = json.loads(JSON_PATH.read_text())
    existing["R6_reproduction"] = r6
    existing["verdict"] = verdict(existing)
    existing["wall_seconds"] = time.time() - started

    R6_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True))
    MD_PATH.write_text(markdown(existing))
    print(f"[r6-closure] wrote {JSON_PATH} and {MD_PATH}")
    print(f"[r6-closure] verdict: {existing['verdict']['status']} "
          f"(failed={existing['verdict']['failed']})")
    return 0 if existing["verdict"]["status"] == "CERTIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
