"""DG replay segfault probe, round 17: watchdog + heap-check variants.

Runs one vortex checkpoint the way ``tools/provenance_certify.py`` runs it
(``ec.run_task`` on the record's own config), with three diagnostics armed so a
native crash stops being opaque:

* ``faulthandler.enable()``      -- SIGSEGV/SIGABRT/SIGBUS prints the live Python frame
* a watchdog thread              -- every ``--watch`` seconds prints elapsed/RSS and
                                    dumps the live stack, so the *last* dump before a
                                    crash bounds the failure window
* ``resource`` high-water mark   -- distinguishes a memory leak (RSS climbing) from a
                                    fixed-footprint numerical run

The environment is left to the caller: ``MALLOC_CHECK_=3`` (glibc) turns silent heap
corruption into an abort with a message, and ``PYTHONMALLOC=malloc`` removes pymalloc
from the picture.  Both are outside the Python code under test, so they are shell-side
variants rather than flags here.

Usage:
    python3 scratch/repro_dg_segfault.py --cells 64 --degree 2 --final-time 2.0 --cfl 0.2
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import resource
import sys
import threading
import time

import torch

from benchmarks import case_euler_vortex_errconserve import ec


def rss_mb() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def start_watchdog(period: float, started: float) -> threading.Event:
    stop = threading.Event()

    def loop():
        while not stop.wait(period):
            print(f"[watch] t={time.time() - started:7.1f}s rss={rss_mb():8.1f}MB "
                  f"hwm={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:8.1f}MB",
                  flush=True)
            faulthandler.dump_traceback()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--final-time", type=float, default=2.0)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--watch", type=float, default=15.0,
                        help="watchdog period in seconds (0 disables)")
    parser.add_argument("--note", default="", help="label printed on the first line")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    faulthandler.enable()
    started = time.time()
    stop = start_watchdog(args.watch, started) if args.watch > 0 else None

    config = {"task": "dg", "degree": args.degree, "cells": args.cells,
              "final_time": args.final_time, "cfl": args.cfl, "strength": 5.0,
              "reference": ec.REFERENCE_MODEL}
    print(f"probe {args.note!r} config={config} threads={args.threads} "
          f"torch={torch.__version__} malloc_check={os.environ.get('MALLOC_CHECK_')!r} "
          f"pythonmalloc={os.environ.get('PYTHONMALLOC')!r}", flush=True)

    record = ec.run_task(config)
    print(f"run_task done in {time.time() - started:.1f} s "
          f"steps={record['steps']} max_percent={record['max_percent']:.6g} "
          f"hwm={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MB",
          flush=True)

    if stop is not None:
        stop.set()
    print(json.dumps({k: record[k] for k in ("task_key", "max_percent", "steps",
                                             "max_relative_l2")}))
    # Flush everything before interpreter teardown: if the crash lives in torch's
    # atexit path, the evidence must already be on stdout.
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())