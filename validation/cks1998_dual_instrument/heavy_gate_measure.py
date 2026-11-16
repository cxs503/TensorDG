"""Measure the three registered Sod gates against **both** reference instruments.

This is the cluster workload behind ``docs/CASE-REPRO-DG-CKS1998-DUAL-INSTRUMENT-GATES.md``.
It exists because two of the three gates in
``benchmarks/shock_reference_quadrature_audit.GATES`` need more than the local
single-command budget (a 3200-cell 2-D extruded P0 run and a 7281-step DG P2 run),
so their dual-instrument readings were left ``not_run`` in the committed artifact.

The workload deliberately does **not** reuse the committed profile cache: it runs the
three solver configurations from scratch, so the readings are an independent
reproduction of the documented gate numbers, not a re-scoring of an old profile.
Each profile is written to ``$HEAVY_GATE_CACHE`` and its sha256 is reported, so the
numbers can be re-scored later without re-running the solver.

stdout is one JSON document (nothing else); progress goes to stderr.

Usage
-----
    HEAVY_GATE_CACHE=/tmp/gate_cache python3 -m validation.cks1998_dual_instrument.heavy_gate_measure
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from pathlib import Path

GATE_KEYS = ("sod-1d-p0-n3200", "sod-2d-p0-extruded-n3200", "sod-dg-p2-tvb-n400")
DEFAULT_CACHE = "docs/benchmarks/results/shock_reference_quadrature/cache_cluster"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    from benchmarks import shock_reference_quadrature_audit as audit

    cache = Path(os.environ.get("HEAVY_GATE_CACHE", DEFAULT_CACHE)).resolve()
    keys = tuple(k.strip() for k in os.environ.get("HEAVY_GATES", ",".join(GATE_KEYS)).split(",") if k.strip())
    audit.CACHE = cache
    cache.mkdir(parents=True, exist_ok=True)

    import torch

    environment = {
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cache": str(cache),
    }
    print(json.dumps(environment), file=sys.stderr)

    gates = {}
    for gate in audit.GATES:
        if gate["key"] not in keys:
            continue
        print(f"measuring {gate['key']} ...", file=sys.stderr, flush=True)
        result = audit.measure_gate(gate)
        profile = cache / f"{_cache_name(gate)}.npz"
        if profile.exists():
            result["profile_sha256"] = _sha256(profile)
            result["profile_bytes"] = profile.stat().st_size
        gates[gate["key"]] = result
        print(f"  status={result.get('status')} flip={result.get('flip')}",
              file=sys.stderr, flush=True)

    document = {
        "case": "Sod / Toro test 1, t=0.2, gate tolerance 3 %",
        "tolerance": audit.TOLERANCE,
        "legacy_quadrature_order": audit.LEGACY_ORDER,
        "environment": environment,
        "gates": gates,
    }
    print(json.dumps(document, indent=2, ensure_ascii=False))
    return 0


def _cache_name(gate: dict) -> str:
    """The cache stem ``audit.cached`` uses for this gate (parsed off its runner)."""
    label = gate["key"]
    return {
        "sod-1d-p0-n3200": "sod-1d-p0-n3200",
        "sod-2d-p0-extruded-n3200": "sod-2d-p0-n3200",
        "sod-dg-p2-tvb-n400": "sod-dg-p2-n400",
    }[label]


if __name__ == "__main__":
    raise SystemExit(main())