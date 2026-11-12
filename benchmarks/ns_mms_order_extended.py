"""Extended mesh ladder for the 2-D compressible Navier--Stokes DG MMS order study.

Why this driver exists (verification debt)
------------------------------------------
``benchmarks/verify_ns_mms_order.py`` published
``validation/ns_dg_2d/mms_order.json`` with two honest red flags that no
document in the repository disclosed (audit P5):

* ``summary[degree=2, density].within_quarter_order_of_design = false``
  (observed order 2.642 vs design 3, tolerance 2.75),
* ``summary[degree=3, density].within_quarter_order_of_design = false``
  (observed order 3.398 vs design 4, tolerance 3.75).

Both flags come from a ladder with **two mesh levels only** (degree 2: 8, 16;
degree 3: 8, 16), because explicit diffusion costs ``O(h^-3)`` and the original
driver capped the high-degree ladder at 16 cells.  A two-level ladder cannot
separate a genuine accuracy defect from pre-asymptotic behaviour: the first
interval (8 -> 16 cells, i.e. 4-8 points per wavelength of the manufactured
state) is the least reliable one.

This driver adds the missing third mesh level for both degrees (32 cells) and,
crucially, **decomposes the error** of every conserved variable into its
volume-weighted mean (a global balance / "mass offset" component) and its
zero-mean fluctuation.  Those two components are L2-orthogonal, so

    e_total^2 = e_mean^2 + e_fluct^2

decides *which* of them drags the density order down:

* if ``e_fluct`` converges at the design order ``p + 1`` while ``e_mean`` lags,
  the shortfall is a global balance (source/quadrature) effect, not a spatial
  derivative defect;
* if both lag, the discrete operator itself is below design order.

Nothing here re-derives the manufactured solution: the case construction,
solver call, quadrature and error norms are imported from
``benchmarks.verify_ns_mms_order`` so the extended ladder is measured with the
*identical* measuring stick as the published numbers.

Usage::

    PYTHONPATH=. python -m benchmarks.ns_mms_order_extended --shard 1 --shards 2
    PYTHONPATH=. python -m benchmarks.ns_mms_order_extended --degree 2 --ncells 32
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from benchmarks.verify_ns_mms_order import (
    FIELDS,
    l2_error,
    order_case,
    quadrature_coordinates,
)
from tensordg.euler_dg_2d import legendre_operators

#: Mesh levels added on top of the published ladders.  ``(degree, ncells)``.
EXTENDED_CASES = (
    (2, 32),
    (3, 32),
)

OUT_DIR = Path("validation/ns_dg_2d/extended")


def _git_sha() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return "unknown"

    return {
        "git_sha": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }


def error_decomposition(case: dict) -> dict:
    """Split each field's L2 error into mean (offset) and fluctuation parts.

    The mean is volume weighted, ``<e> = int e w / int w``, with ``e`` the
    numerical-minus-exact error on the quadrature nodes.  Because
    ``int (e - <e>) <e> w = 0``, the two contributions are orthogonal:
    ``e_total^2 = e_mean^2 + e_fluct^2`` exactly (verified in the test suite).
    """
    weights = case["weights"].double()
    dx, dy = case["dxdy"]
    cell_weight = (weights.view(-1, 1) * weights.view(1, -1)) * (0.25 * dx * dy)
    weight = cell_weight.view(1, 1, *cell_weight.shape)
    total_weight = float(weight.sum())
    at_quad, exact = case["at_quad"], case["exact"]
    out = {}
    for index, name in enumerate(FIELDS):
        diff = at_quad[..., index] - exact[..., index]
        ref = exact[..., index]
        mean = float((diff * weight).sum() / total_weight)
        e_mean = abs(mean)
        e_fluct = float(torch.sqrt(((diff - mean) ** 2 * weight).sum()))
        e_total, e_ref = l2_error(at_quad[..., index], ref, weights, dx, dy)
        out[name] = {
            "mean_offset": mean,
            "mean_l2": e_mean,
            "fluctuation_l2": e_fluct,
            "total_l2": e_total,
            "reference_l2": e_ref,
            "relative_l2": e_total / e_ref,
            "relative_fluctuation_l2": e_fluct / e_ref,
            "orthogonality_residual": abs(e_total ** 2 - (e_mean ** 2 + e_fluct ** 2)),
        }
    return out


def run_case(degree: int, ncells: int) -> dict:
    started = time.time()
    case = order_case(degree, ncells)
    wall = time.time() - started
    _x, _y, _dx, _dy = quadrature_coordinates(ncells, degree, case["quadrature"])
    _nodes, _w, _basis, _deriv, _signs = legendre_operators(degree, case["quadrature"])
    return {
        "degree": degree,
        "ncells": ncells,
        "quadrature": case["quadrature"],
        "steps": case["steps"],
        "diffusive_steps": case["diffusive_steps"],
        "dt": case["dt"],
        "wall_seconds": wall,
        "errors": case["errors"],
        "decomposition": error_decomposition(case),
        "provenance": _git_sha(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--ncells", type=int, default=None)
    parser.add_argument("--shard", type=int, default=1, help="1-based shard index")
    parser.add_argument("--shards", type=int, default=1, help="total shards")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    if args.degree is not None:
        cases = [(args.degree, args.ncells)]
    else:
        if not 1 <= args.shard <= args.shards:
            parser.error("shard index must be in [1, shards]")
        cases = list(EXTENDED_CASES)[args.shard - 1 :: args.shards]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for degree, ncells in cases:
        record = run_case(degree, ncells)
        path = args.out_dir / f"d{degree}_n{ncells}.json"
        path.write_text(json.dumps(record, indent=1, sort_keys=True))
        written.append(str(path))
        print(
            f"degree={degree} ncells={ncells:3d} steps={record['steps']:5d} "
            f"wall={record['wall_seconds']:.1f}s | "
            + " ".join(
                f"{f}: {record['errors'][f]['relative_l2']:.3e}"
                for f in FIELDS
            ),
            flush=True,
        )
    print(json.dumps({"written": written}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())