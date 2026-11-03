"""Machine-checkable scale consistency of the shock limiters.

A modal DG solver on a Cartesian cell stores a physical slope ``s`` in the
mode-1 Legendre coefficient as ``c1 = s * dx / 2``.  A minmod/TVD limiter that
works from cell averages therefore has to write ``c1 = minmod(d_left, d_right)
/ 2``.  This script measures the ratio that the implementation actually
produces, so the value quoted in the verification report is read from an
artifact instead of being copied from a terminal.

Usage
-----
    PYTHONPATH=. python -m benchmarks.limiter_scale_check --tag post_fix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tensordg.limiters import _minmod, apply_tvb_limiter, apply_weno_limiter

AVERAGES = [1.0, 1.0, 1.0, 2.0, 2.6, 2.6]  # cell 3 has a non-zero minmod slope
DX = 0.1


def ratio(limited: torch.Tensor, target: torch.Tensor) -> float:
    mask = target.abs() > 1e-12
    if not bool(mask.any()):
        return float("nan")
    return float((limited[:, 1].abs() / target.abs())[mask].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="post_fix")
    args = parser.parse_args()

    ncells = len(AVERAGES)
    coeff = torch.zeros(ncells, 3, 1, dtype=torch.float64)
    coeff[:, 0, 0] = torch.tensor(AVERAGES, dtype=torch.float64)
    avg = coeff[:, 0]
    padded = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    target = _minmod(avg - padded[:-2], padded[2:] - avg) / 2.0

    out = {"tag": args.tag, "cell_averages": AVERAGES, "dx": DX, "cells": ncells,
           "target_mode1": [float(v) for v in target[:, 0]], "limiters": {}}
    for name, fn in (("tvb", apply_tvb_limiter), ("weno", apply_weno_limiter)):
        limited = fn(coeff, DX, tvb_constant=1e-9)
        out["limiters"][name] = {
            "mode1": [float(v) for v in limited[:, 1, 0]],
            "ratio_to_half_minmod": ratio(limited, target),
        }
        print(f"{name}: mode1={out['limiters'][name]['mode1']} "
              f"ratio={out['limiters'][name]['ratio_to_half_minmod']:.3f}")

    path = Path("validation/limiter_sod") / f"scale_consistency_{args.tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()