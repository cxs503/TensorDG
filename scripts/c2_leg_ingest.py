"""Rebuild cluster-run convergence legs from a job's stdout in the working tree.

The heavy end of the C2 ladder runs on the job queue, which hands back stdout.
``scripts/c2_leg_job.py`` therefore emits each leg as
``LEG_JSON:<stem>:<sha256>`` / ``LEG_NPZ_B64:<stem>:<sha256>`` followed by the
base64 payload.  This ingester decodes it, **verifies the checksum** and writes
``<out>/legs/<stem>.{json,npz}`` -- so a truncated or edited capture fails loudly
instead of quietly becoming the record the C2 verdict is computed from.

Usage:
    python -m scripts.c2_leg_ingest <stdout-log> [<stdout-log> ...] \\
        --output-dir docs/benchmarks/results/riemann_2d_cks1998
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import re
import sys
from pathlib import Path

B64_LINE = re.compile(r"^[A-Za-z0-9+/=]+$")


def _payloads(lines):
    """Yield ``(marker, stem, sha256, bytes)`` for every emitter block."""
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("LEG_JSON:") or line.startswith("LEG_NPZ_B64:"):
            marker, stem, digest = line.split(":", 2)
            chunk = []
            index += 1
            while index < len(lines):
                candidate = lines[index].strip()
                if not candidate or not B64_LINE.match(candidate):
                    break
                chunk.append(candidate)
                index += 1
            if not chunk:
                raise ValueError(f"{marker} {stem}: no base64 payload follows")
            try:
                blob = base64.b64decode("".join(chunk), validate=True)
            except binascii.Error as error:
                raise ValueError(f"{marker} {stem}: bad base64 ({error})") from error
            yield marker, stem, digest, blob
            continue
        index += 1


def ingest(log_paths, output_dir, verbose=True):
    """Decode every emitter block from ``log_paths`` into ``output_dir/legs``."""
    legs = Path(output_dir) / "legs"
    legs.mkdir(parents=True, exist_ok=True)
    written = []
    for log_path in log_paths:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        for marker, stem, digest, blob in _payloads(lines):
            actual = hashlib.sha256(blob).hexdigest()
            if actual != digest:
                raise ValueError(
                    f"{log_path}: {marker} {stem} checksum mismatch "
                    f"(declared {digest[:16]}..., decoded {actual[:16]}...)"
                )
            suffix = ".json" if marker == "LEG_JSON" else ".npz"
            target = legs / f"{stem}{suffix}"
            target.write_bytes(blob)
            written.append(target)
            if verbose:
                print(f"  ingested {target} ({len(blob)} bytes, sha256 ok)")
    if not written:
        raise SystemExit("no LEG_JSON / LEG_NPZ_B64 blocks found in the given logs")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output-dir",
                        default="docs/benchmarks/results/riemann_2d_cks1998")
    args = parser.parse_args()
    files = ingest(args.logs, args.output_dir)
    print(f"\n{len(files)} file(s) restored under {Path(args.output_dir) / 'legs'}")
    sys.exit(0)


if __name__ == "__main__":
    main()