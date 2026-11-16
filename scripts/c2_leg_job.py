"""Run one CKS1998 2-D Riemann convergence leg and emit it as transportable text.

Why this wrapper exists
-----------------------
The ``full`` profile of ``benchmarks/riemann_2d_cks1998.py`` is four solves
(P1/P2 x 32x32/64x64 at ``t = 0.25``); the 64x64 P2 leg alone costs ~43 min of
one core, so the ladder has to run on the cluster job queue rather than on the
control machine.  A job hands back *stdout*, while gate C2 is recomputed from the
leg's raw cell-mean density array.  So the leg is emitted as a self-describing,
checksummed text payload that ``scripts/c2_leg_ingest.py`` can rebuild byte for
byte in the working tree -- no silent re-run, no hand-typed numbers.

Usage:
    OMP_NUM_THREADS=1 python -m scripts.c2_leg_job --degree 2 --cells 64 \\
        --time 0.25 --cfl 0.2 --output-dir <dir>
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.riemann_2d_cks1998 import leg_paths, leg_stem, run_leg  # noqa: E402
from tensordg.euler_dg_2d import LAX_LIU_CONFIG_3  # noqa: E402,F401

NPZ_MARK = "LEG_NPZ_B64"
JSON_MARK = "LEG_JSON"


def emit(degree, cells, final_time, cfl, output_dir):
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    record = run_leg(degree, cells, final_time=final_time, cfl=cfl,
                     output_dir=output_dir,
                     command=(f"python -m scripts.c2_leg_job --degree {degree} "
                              f"--cells {cells} --time {final_time:g} "
                              f"--cfl {cfl:g}"))
    json_path, npz_path = leg_paths(output_dir, degree, cells, final_time, cfl)
    stem = leg_stem(degree, cells, final_time, cfl)
    payload = json_path.read_bytes()
    blob = npz_path.read_bytes()
    print(f"{JSON_MARK}:{stem}:{hashlib.sha256(payload).hexdigest()}")
    print(base64.b64encode(payload).decode("ascii"))
    print(f"{NPZ_MARK}:{stem}:{hashlib.sha256(blob).hexdigest()}")
    print(base64.b64encode(blob).decode("ascii"))
    print("LEG_DONE", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--output-dir",
                        default=str(REPO / "docs" / "benchmarks" / "results"
                                    / "riemann_2d_cks1998"))
    args = parser.parse_args()
    emit(args.degree, args.cells, args.time, args.cfl, args.output_dir)


if __name__ == "__main__":
    main()