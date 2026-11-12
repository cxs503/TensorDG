"""Status measurement for the experimental 2-D P-SV elastic DG solver.

Writes ``validation/elastic_2d_psv/status.json`` -- every number quoted in
``docs/CASE-DG-ELASTIC-2D-PSV.md`` is produced here (no hand-copying).

Run: ``PYTHONPATH=. python benchmarks/elastic_2d_psv_status.py``
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from tensordg.elastic_1d import dalembert_pulse, initial_pulse_velocity, solve_elastic_1d
from tensordg.elastic_2d import (ElasticMedium, plane_wave, solve_elastic_2d,
                                 legendre_operators_2d, _rhs)

OUT = Path("validation/elastic_2d_psv")


def rel_l2(num, exact, weights):
    w2 = torch.outer(weights, weights).reshape(1, 1, weights.numel(),
                                               weights.numel(), 1)
    return ((((num - exact) ** 2 * w2).sum()
             / ((exact ** 2 * w2).sum())).sqrt().item())


def plane_wave_ladder(medium, t_end=0.02, cells=(8, 16, 32, 64)):
    rows = {}
    k = 2.0 * math.pi
    for mode in ("P", "S"):
        exact = plane_wave(mode, k, 0.0, 1.0, medium)
        for degree in (1, 2, 3):
            errs = []
            for n in cells:
                res = solve_elastic_2d(lambda x, y: exact(x, y, 0.0), nx=n, ny=1,
                                       degree=degree, final_time=t_end, cfl=0.1)
                x, y = res.quadrature_grid()
                errs.append(rel_l2(res.fields(), exact(x, y, t_end),
                                   res.ops["weights"]))
            eoc = [None] + [math.log(errs[i - 1] / errs[i]) / math.log(2.0)
                            for i in range(1, len(errs))]
            rows[f"{mode}_p{degree}"] = {"cells": list(cells), "rel_l2": errs,
                                         "eoc": eoc}
    return rows


def cfl_independence(medium, n=64, t_end=0.02):
    k = 2.0 * math.pi
    exact = plane_wave("P", k, 0.0, 1.0, medium)
    out = {}
    for cfl in (0.4, 0.2, 0.1, 0.05, 0.025):
        res = solve_elastic_2d(lambda x, y: exact(x, y, 0.0), nx=n, ny=1,
                               degree=2, final_time=t_end, cfl=cfl)
        x, y = res.quadrature_grid()
        out[str(cfl)] = {"rel_l2": rel_l2(res.fields(), exact(x, y, t_end),
                                          res.ops["weights"]),
                         "dt": res.times[0], "steps": res.times[1]}
    return out


def energy_drift(medium, n=32, t_end=0.08):
    k = 2.0 * math.pi
    exact = plane_wave("P", k, 0.0, 1.0, medium)
    res = solve_elastic_2d(lambda x, y: exact(x, y, 0.0), nx=n, ny=1, degree=2,
                           final_time=t_end, cfl=0.1,
                           snapshot_times=(0.02, 0.04, 0.06))
    snaps = sorted(res.snapshots.items())
    energies = [res.energy(c).item() for _, c in snaps]
    return {"times": [t for t, _ in snaps], "energy": energies,
            "ratio_to_first": [e / energies[0] for e in energies]}


def semidiscrete_spectrum(medium, nx=2, ny=1, degree=2):
    ops = legendre_operators_2d(degree)
    k = degree + 1
    hx, hy = 1.0 / nx, 1.0 / ny
    n = nx * ny * k * k * 5
    mat = np.zeros((n, n))
    for j in range(n):
        c = torch.zeros(nx, ny, k, k, 5, dtype=torch.float64).reshape(-1)
        c[j] = 1.0
        mat[:, j] = _rhs(c.reshape(nx, ny, k, k, 5), ops, medium, hx, hy,
                         "periodic", "periodic").reshape(-1).numpy()
    ev = np.linalg.eigvals(mat)
    top = np.sort(ev.real)[-5:]
    return {"nx": nx, "ny": ny, "degree": degree,
            "max_real_eigenvalue": float(ev.real.max()),
            "max_abs_imag": float(np.abs(ev.imag).max()),
            "top5_real": [float(v) for v in top]}


def pulse_cross_check(ns=(50, 100, 200), t_end=0.2):
    """Compare the 2-D solver in a y-invariant box against the 1-D library."""
    medium = ElasticMedium(rho=1.0, lam=0.5, mu=0.25)   # c_p = c_s + 1 = 1

    def init(x, y):
        out = torch.zeros(*x.shape, 5, dtype=torch.float64)
        out[..., 0] = torch.exp(-((x - 0.5) / 0.05) ** 2)
        return out

    rows = []
    for n in ns:
        res2 = solve_elastic_2d(init, nx=n, ny=1, degree=2, final_time=t_end,
                                cfl=0.3, medium=medium)
        x, y = res2.quadrature_grid()
        v2 = res2.fields()[:, 0, :, 0, 0].reshape(-1)
        x2 = x[:, 0, :, 0].reshape(-1)
        ve, _ = dalembert_pulse(x2, t_end, rho=1.0, young=1.0, center=0.5,
                                width=0.05)
        centres, coeff, _ = solve_elastic_1d(ncells=n, degree=2,
                                             final_time=t_end, cfl=0.3,
                                             bc="periodic", rho=1.0, young=1.0,
                                             initial_velocity=initial_pulse_velocity,
                                             initial_stress=lambda z: torch.zeros_like(z))
        v1 = coeff[:, 0, 0]
        v1_ref, _ = dalembert_pulse(centres, t_end, rho=1.0, young=1.0,
                                    center=0.5, width=0.05)
        rows.append({"cells": n,
                     "elastic_1d_vs_exact": float((v1 - v1_ref).norm() / v1_ref.norm()),
                     "elastic_2d_vs_exact": float((v2 - ve).norm() / ve.norm())})
    return {"t_end": t_end, "rows": rows}


def analytic_reference_bug(cells=(8, 16, 32)):
    """Error with the P-wave stress isotropic part omitted (the pre-fix bug)."""
    medium = ElasticMedium()
    k = 2.0 * math.pi
    exact = plane_wave("P", k, 0.0, 1.0, medium)
    lam, mu = medium.lam, medium.mu
    amp = -medium.zp

    def broken(x, y):
        s = torch.sin(k * x)
        out = torch.empty(*s.shape, 5, dtype=torch.float64)
        out[..., 0] = s
        out[..., 1] = 0.0
        out[..., 2] = amp * s      # only the n (x) n part
        out[..., 3] = 0.0
        out[..., 4] = 0.0
        return out

    rows = []
    for n in cells:
        res = solve_elastic_2d(broken, nx=n, ny=1, degree=2, final_time=0.02,
                               cfl=0.1, medium=medium)
        x, y = res.quadrature_grid()
        rows.append({"cells": n,
                     "rel_l2_vs_broken_reference": rel_l2(res.fields(),
                                                         broken(x, y),
                                                         res.ops["weights"]),
                     "rel_l2_vs_correct_reference": rel_l2(res.fields(),
                                                           exact(x, y, 0.02),
                                                           res.ops["weights"])})
    return {"cells": list(cells), "rows": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    medium = ElasticMedium()
    status = {
        "generated_by": "benchmarks/elastic_2d_psv_status.py",
        "solver_status": "EXPERIMENTAL / NOT VERIFIED (see docs/CASE-DG-ELASTIC-2D-PSV.md)",
        "medium": {"rho": medium.rho, "lam": medium.lam, "mu": medium.mu,
                   "cp": medium.cp, "cs": medium.cs},
        "plane_wave_ladder": plane_wave_ladder(medium),
        "cfl_independence": cfl_independence(medium),
        "energy_drift": energy_drift(medium),
        "semidiscrete_spectrum": semidiscrete_spectrum(medium),
        "pulse_cross_check": pulse_cross_check(),
        "analytic_reference_isotropic_term": analytic_reference_bug(),
    }
    try:
        status["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        status["commit"] = None
    path = OUT / "status.json"
    path.write_text(json.dumps(status, indent=2, sort_keys=True))
    print("wrote", path)
    print("max Re(eig) =", status["semidiscrete_spectrum"]["max_real_eigenvalue"])
    for key, row in status["plane_wave_ladder"].items():
        print(key, ["%.3e" % v for v in row["rel_l2"]])


if __name__ == "__main__":
    main()