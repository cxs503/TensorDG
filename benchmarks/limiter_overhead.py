"""Limiter cost harness: kernel-count model + wall clock, three implementations.

Mission m3262.  The TVB/WENO limiter dev report (m274) recorded a **76-123 %
runtime overhead against a <30 % acceptance target** and the later verification
note (``docs/limiter_verification.md`` §5.5) says that number was never
re-measured after the implementation changed.  This harness re-measures it with
two independent metrics so the answer does not depend on the load state of one
machine.

1. **Kernel (dispatch) count.**  The limiter is element-wise tensor algebra on a
   few thousand doubles, so a call costs ATen dispatch overhead rather than
   floating point work; the number of kernels per call is a deterministic,
   machine-independent cost model.  The SSP-RK3 step of
   ``tensordg.euler_dg_1d`` issues

       K_step = 3 * (K_rhs + K_filter) + 3 * K_limiter ,

   hence ``overhead_model = 3 * K_limiter / (3 * (K_rhs + K_filter))``.

2. **Wall clock** of the same components in one process (median over repeats,
   plus min/max so the reader can see the noise of a shared host), and short
   end-to-end solves where the *per-step* time is compared (the limiter changes
   the stable step count, so the total time is not the overhead).

Three limiter implementations are compared:

* ``loop``    -- vectorised detector plus a Python loop over the flagged cells,
                 the classic shape of the m274 "pure Python per-cell loop"
                 implementation (the code itself is no longer in the repo, so
                 this is a reconstruction and is labelled as such);
* ``head``    -- the vectorised masked-assignment version of HEAD (ed8f38b),
                 kept verbatim in ``tests/_limiter_reference.py``;
* ``shipped`` -- the implementation of this mission.

Three states are measured, because the limiter has an early exit:

* ``worst``   -- random coefficients, detector fires almost everywhere;
* ``sod``     -- a Sod state with a smooth modulation, representative;
* ``smooth``  -- affine data, detector fires nowhere (early exit).

Usage
-----
    PYTHONPATH=. python benchmarks/limiter_overhead.py [--solves] [--scaling]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

import torch

from tensordg.euler_dg_1d import (
    _operators,
    _positivity_filter,
    _rhs,
    primitive_to_conservative,
)
from tensordg.limiters import (
    _troubled_cell_indicator,
    apply_tvb_limiter,
    apply_weno_limiter,
)

# The HEAD reference implementation lives with the tests (it is the oracle of
# the bit-equality test); importing it from there keeps a single copy.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
)
from _limiter_reference import ref_tvb_limiter, ref_weno_limiter  # noqa: E402


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

class _KernelCounter(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.ops: Dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: D102
        self.count += 1
        key = str(func)
        self.ops[key] = self.ops.get(key, 0) + 1
        return func(*args, **(kwargs or {}))


def count_kernels(fn) -> int:
    counter = _KernelCounter()
    with counter:
        fn()
    return counter.count


def timed_us(fn, repeats: int) -> Dict[str, float]:
    for _ in range(5):  # warm-up: fills the cached mask / padding index
        fn()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def measure(fn, repeats: int) -> Dict[str, float]:
    """Warm, count and time in that order (the count must be the warm count)."""
    for _ in range(5):
        fn()
    kernels = count_kernels(fn)
    return {"kernels": kernels, "warm_repeats": repeats, **timed_us(fn, repeats)}


# ---------------------------------------------------------------------------
# the m274-style per-cell loop (reconstruction, not the original code)
# ---------------------------------------------------------------------------

def loop_tvb_limiter(coeff: torch.Tensor, dx: float, tvb_constant: float = 1.0,
                     limit_boundaries: bool = False) -> torch.Tensor:
    """Vectorised detector + Python loop that rewrites each flagged cell.

    The m274 report describes the limiter as a "pure Python per-cell loop"; the
    code itself is gone.  This is the canonical shape of such an implementation:
    the troubled-cell test is vectorised, then every flagged cell is rewritten
    from Python.  It is measured only to show what the "every kernel is a
    dispatch" cost model does to a per-cell loop.
    """
    from tensordg.tvb import minmod as _mm

    ncells, nmodes, nvar = coeff.shape
    if nmodes <= 1:
        return coeff
    troubled = _troubled_cell_indicator(
        coeff, dx, tvb_constant, limit_boundaries=limit_boundaries
    )
    if not bool(troubled.any()):
        return coeff
    avg = coeff[:, 0]
    avg_pad = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = avg - avg_pad[:-2]
    d_right = avg_pad[2:] - avg
    slope = _mm(d_left, d_right)
    limited = coeff.clone()
    for i in torch.nonzero(troubled).flatten().tolist():
        limited[i, 1] = 0.5 * slope[i]
        if nmodes > 2:
            limited[i, 2:] = 0.0
    return limited


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

def sod_state(ncells: int, degree: int, dtype=torch.float64):
    """Sod data with a smooth modulation: a mix of smooth and shocked cells."""
    dx = 1.0 / ncells
    x = (torch.arange(ncells, dtype=dtype) + 0.5) * dx
    rho = torch.where(x < 0.5, torch.ones_like(x), torch.full_like(x, 0.125))
    u = torch.zeros_like(x)
    p = torch.where(x < 0.5, torch.ones_like(x), torch.full_like(x, 0.1))
    rho = rho * (1.0 + 0.05 * torch.sin(6.0 * x))
    coeff = torch.zeros(ncells, degree + 1, 3, dtype=dtype)
    coeff[:, 0] = primitive_to_conservative(rho, u, p)
    coeff[:, 1] = 1e-3 * torch.cos(4.0 * x)[:, None]
    return coeff


def worst_state(ncells: int, degree: int, dtype=torch.float64):
    torch.manual_seed(7)
    coeff = torch.randn(ncells, degree + 1, 3, dtype=dtype)
    coeff[:, 0].abs_().add_(1.0)
    return coeff


def smooth_state(ncells: int, degree: int, dtype=torch.float64):
    coeff = torch.zeros(ncells, degree + 1, 3, dtype=dtype)
    coeff[:, 0] = (1.0 + torch.linspace(0.0, 1.0, ncells, dtype=dtype))[:, None]
    coeff[:, 1] = 0.25
    return coeff


STATES = {"worst": worst_state, "sod": sod_state, "smooth": smooth_state}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncells", type=int, default=200)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--solves", action="store_true")
    ap.add_argument("--final-time", type=float, default=0.01)
    ap.add_argument("--solve-repeats", type=int, default=2)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--out", default="validation/limiter_overhead")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    ncells, degree = args.ncells, args.degree
    dx = 1.0 / ncells

    nodes, weights, phi, dphi, phi_left, phi_right = _operators(
        degree, torch.device("cpu"), torch.float64
    )
    check_points = torch.cat((phi, phi_left[None, :], phi_right[None, :]), dim=0)

    states = {name: builder(ncells, degree) for name, builder in STATES.items()}
    sod = states["sod"]

    rhs = lambda: _rhs(sod, dx, weights, phi, dphi, phi_left, phi_right)  # noqa: E731
    filt = lambda: _positivity_filter(sod, check_points)  # noqa: E731

    result = {
        "mission": "m3262",
        "ncells": ncells,
        "degree": degree,
        "dtype": str(sod.dtype),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "loadavg": open("/proc/loadavg").read().split()[:3],
        "components": {},
        "limiter": {},
        "flagged_fraction": {},
        "overhead_model": {},
        "criterion_30pct_model": {},
    }

    for name, fn in (("rhs", rhs), ("positivity_filter", filt)):
        result["components"][name] = measure(fn, args.repeats)
    step_us = (result["components"]["rhs"]["median_us"]
               + result["components"]["positivity_filter"]["median_us"])
    step_kernels = 3 * (result["components"]["rhs"]["kernels"]
                        + result["components"]["positivity_filter"]["kernels"])

    impls = {
        "tvb": {"loop": loop_tvb_limiter, "head": ref_tvb_limiter,
                "shipped": apply_tvb_limiter},
        "weno": {"head": ref_weno_limiter, "shipped": apply_weno_limiter},
    }
    for state_name, coeff in states.items():
        result["flagged_fraction"][state_name] = float(
            _troubled_cell_indicator(coeff, dx, 1.0).double().mean()
        )
        for kind, variants in impls.items():
            for impl, fn in variants.items():
                key = f"{kind}_{impl}_{state_name}"
                repeats = max(5, args.repeats // 40) if impl == "loop" else args.repeats
                entry = measure(lambda f=fn, c=coeff: f(c, dx, 1.0), repeats)
                entry["us_per_step_ratio"] = 3 * entry["median_us"] / step_us
                result["limiter"][key] = entry
                if state_name == "worst":
                    result["overhead_model"][key] = 3 * entry["kernels"] / step_kernels
                    result["criterion_30pct_model"][key] = (
                        3 * entry["kernels"] / step_kernels < 0.30
                    )

    if args.scaling:
        scaling = {"mission": "m3262", "degree": degree, "points": {}}
        for n in (50, 100, 200, 400, 800):
            c = worst_state(n, degree)
            d = 1.0 / n
            row = {}
            for label, fn in (
                ("tvb_shipped", lambda: apply_tvb_limiter(c, d, 1.0)),
                ("tvb_head", lambda: ref_tvb_limiter(c, d, 1.0)),
                ("weno_shipped", lambda: apply_weno_limiter(c, d, 1.0)),
                ("tvb_loop", lambda: loop_tvb_limiter(c, d, 1.0)),
            ):
                for _ in range(3):
                    fn()
                row[label] = count_kernels(fn)
            scaling["points"][str(n)] = row
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "scaling.json")
        with open(path, "w") as fh:
            json.dump(scaling, fh, indent=2, sort_keys=True)
        print(f"wrote {path}")

    if args.solves:
        from tensordg.euler_dg_1d import solve_euler_dg_1d

        left, right = (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)
        result["solves"] = {}
        for label, kind in (("none", None), ("tvb", "tvb"), ("weno", "weno")):
            walls = []
            for _ in range(args.solve_repeats):
                t0 = time.perf_counter()
                _, _, meta = solve_euler_dg_1d(
                    left, right, ncells=ncells, degree=degree,
                    final_time=args.final_time, limiter=kind,
                )
                walls.append(time.perf_counter() - t0)
            wall = min(walls)  # best of N on a shared host
            result["solves"][label] = {
                "wall_s": wall,
                "wall_s_all": walls,
                "steps": meta["steps"],
                "us_per_step": wall / meta["steps"] * 1e6,
            }
        base = result["solves"]["none"]["us_per_step"]
        for label in ("tvb", "weno"):
            result["solves"][label]["overhead_per_step"] = (
                result["solves"][label]["us_per_step"] / base - 1.0
            )

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"n{ncells}_p{degree}"
    path = os.path.join(args.out, f"{tag}.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in
                      ("components", "flagged_fraction", "overhead_model",
                       "criterion_30pct_model")}, indent=2))
    if args.solves:
        print(json.dumps(result["solves"], indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())