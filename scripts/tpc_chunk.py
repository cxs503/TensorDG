"""Chunked local runner for the CKS1998 time-power closure ladder (m23825).

The pre-registered completion command (``--meshes 10 32``) is a ~12 min
single-process sweep; the local short-test cap is 120 s per command and the
cluster queue was saturated (job cj4804, 40+ min queued), so the ladder is
executed in per-CFL chunks here and reassembled with the module's own
``--assemble`` path -- the same code path the cluster job uses, so the
artifact is byte-identical in structure to the pre-registered one.

Scope declaration (honest): the auxiliary arms (drift-pair mesh 24, the
prescribed-dt positive control) are *not* re-run in this seat; they do not
feed the verdict (module docstring: the dt arm is "detection power, not a
verdict").  The verdict, the jackknife, the three-level order cross-check and
``mesh_independent`` all come from the ladder alone.

Usage (one command per chunk, each < 120 s)::

    python scripts/tpc_chunk.py --mesh 32 --cfls 0.025 0.05 0.1 0.2 0.4 0.5 \
        --out /tmp/tpc_m32a.json
    python scripts/tpc_chunk.py --mesh 32 --cfls 0.0125 --out /tmp/tpc_m32b.json
    python scripts/tpc_chunk.py --mesh 10 \
        --cfls 0.0125 0.025 0.05 0.1 0.2 0.4 0.5 --out /tmp/tpc_m10.json
    python -m benchmarks.cks1998_vortex_time_power_closure --assemble \
        /tmp/tpc_m32a.json /tmp/tpc_m32b.json /tmp/tpc_m10.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.cks1998_vortex_order_certification import TIME_PAIR  # noqa: E402
from benchmarks.cks1998_vortex_time_power import _record  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=int, required=True)
    ap.add_argument("--cfls", type=float, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    role = ("production" if args.mesh == TIME_PAIR[1][1] else "local")
    recs = [_record(1, args.mesh, "ladder", cfl=c, mesh_role=role)
            for c in args.cfls]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"degree": 1, "runs": recs}, indent=1))
    for r in recs:
        print(f"mesh={args.mesh} cfl={r['cfl']} steps={r.get('steps')} "
              f"err={r.get('error')}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
