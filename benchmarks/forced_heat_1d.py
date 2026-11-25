"""Spatial SIPG convergence for two forced Fourier modes starting from zero."""
import json
import math
import subprocess
from pathlib import Path

import torch
from tensordg.heat_1d import heat_error, solve_heat_1d_exact_time


def reference(x, time):
    return sum(amplitude*(-math.expm1(-wave**2*time))*torch.sin(wave*x)
               for wave, amplitude in ((math.pi, 1.), (3*math.pi, -.2)))


def source(x):
    return math.pi**2*(torch.sin(math.pi*x)-1.8*torch.sin(3*math.pi*x))


if __name__ == "__main__":
    rows = []
    for degree in (1,2,3):
        previous = None
        for cells in (8,16,32):
            time = .04
            x, coeff, _ = solve_heat_1d_exact_time(cells, degree, time, source=source,
                                                   initial_temperature=torch.zeros_like)
            error = heat_error(x,coeff,time,reference=reference)["absolute_l2"]
            order = None if previous is None else math.log2(previous/error)
            passed = math.isfinite(error) and (order is None or order > degree+.7)
            rows.append(dict(degree=degree, cells=cells, absolute_l2=error, order=order, passed=passed))
            previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
