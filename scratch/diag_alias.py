"""Diagnostic v2: is the DG moving-error growth smooth, or an aliasing mode?

Per checkpoint this reports

* relative L2 error of (rho, u, v, p) against the analytic vortex on the
  volume Gauss grid (same metric as the committed harness, so the numbers are
  directly comparable with ``case_euler_vortex_errconserve/metrics.json``),
* the Helmholtz split of the *velocity* error field, reconstructed on a
  **uniform** supersampled grid (8 nodes/cell/direction, exact for the modal
  polynomial): energy in the solenoidal (divergence-free) part vs the
  dilatational part.  A phase error of the vortex shape is mixed; a spurious
  grid-scale alias mode of a compressible solver is (almost) purely
  solenoidal and carries no density/pressure signature,
* the wavenumber of the spectral peak of the velocity error.

Usage:
    python scratch/diag_alias.py --t 2 --quad 4 --cfl 0.2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH, BG = 5.0, (1.0, 1.0)
UNIFORM_PER_CELL = 8


def run(t, quad, cfl, degree, cells):
    started = time.time()
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(quad, degree + 2))

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, STRENGTH, BG)

    xc, yc, coeff, meta = dg2.solve_euler_dg_2d(
        initial, ncells=cells, degree=degree, domain=DOMAIN, final_time=t,
        cfl=cfl, periodic=(True, True), initial_projection="l2",
        quadrature=max(quad, degree + 2))

    dx = (DOMAIN[0][1] - DOMAIN[0][0]) / cells
    dy = (DOMAIN[1][1] - DOMAIN[1][0]) / cells
    rec = {"t": t, "quad": quad, "cfl": cfl, "degree": degree, "cells": cells,
           "steps": int(meta["steps"])}

    # --- 1. committed metric: relative L2 on the volume Gauss grid ---------
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho, u, v, p = dg2._primitive(state)
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    ex = vortex_primitive(xq, yq, t, STRENGTH, BG)

    def rel(num, ref):
        return float(torch.sqrt(torch.mean((num - ref) ** 2)) /
                     torch.sqrt(torch.mean(ref ** 2)))

    rec["rel"] = {"rho": rel(rho, ex[0]), "u": rel(u, ex[1]),
                  "v": rel(v, ex[2]), "p": rel(p, ex[3])}
    rec["density_min"] = float(rho.min())

    # --- 2. uniform supersampled reconstruction + Helmholtz split ----------
    nu = UNIFORM_PER_CELL
    xi = np.linspace(-1.0 + 1.0 / nu, 1.0 - 1.0 / nu, nu)
    basis_u = torch.as_tensor(np.polynomial.legendre.legvander(xi, degree),
                              dtype=torch.float64)
    state_u = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis_u, basis_u)
    rho_u, u_u, v_u, p_u = dg2._primitive(state_u)
    xu = xc[:, None, None, None] + 0.5 * dx * torch.as_tensor(xi)[None, None, :, None]
    yu = yc[None, :, None, None] + 0.5 * dy * torch.as_tensor(xi)[None, None, None, :]
    exu = vortex_primitive(xu, yu, t, STRENGTH, BG)

    def flat(field):
        arr = np.asarray(field)
        return arr.transpose(0, 2, 1, 3).reshape(cells * nu, cells * nu)

    ue = flat(u_u - exu[1])
    ve = flat(v_u - exu[2])
    de = flat(rho_u - exu[0])
    pe = flat(p_u - exu[3])

    n = cells * nu
    L = DOMAIN[0][1] - DOMAIN[0][0]
    k1d = 2 * np.pi * np.fft.fftfreq(n, d=L / n)
    KX, KY = np.meshgrid(k1d, k1d, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = 1.0
    uh, vh = np.fft.fft2(ue), np.fft.fft2(ve)
    div_h = 1j * (KX * uh + KY * vh)
    phi = -div_h / k2
    us = np.fft.ifft2(uh - 1j * KX * phi).real
    vs = np.fft.ifft2(vh - 1j * KY * phi).real
    sol = float(np.sqrt(np.mean(us ** 2 + vs ** 2)))
    dil = float(np.sqrt(np.mean((ue - us) ** 2 + (ve - vs) ** 2)))
    rec.update({
        "vel_error_l2": float(np.sqrt(np.mean(ue ** 2 + ve ** 2))),
        "solenoidal_l2": sol,
        "dilatational_l2": dil,
        "solenoidal_fraction": sol ** 2 / max(sol ** 2 + dil ** 2, 1e-300),
        "rho_error_l2": float(np.sqrt(np.mean(de ** 2))),
        "p_error_l2": float(np.sqrt(np.mean(pe ** 2))),
    })
    mag = np.sqrt(np.abs(uh) ** 2 + np.abs(vh) ** 2)
    mag[0, 0] = 0.0
    idx = np.unravel_index(np.argmax(mag), mag.shape)
    i = idx[0] - (n if idx[0] > n // 2 else 0)
    j = idx[1] - (n if idx[1] > n // 2 else 0)
    rec["vel_err_peak_mode"] = [int(i), int(j)]
    rec["vel_err_peak_ppw"] = float(n / max(np.hypot(i, j), 1e-12))
    rec["vel_err_wavelength"] = float(L / max(np.hypot(i, j), 1e-12))
    rec["wall_seconds"] = time.time() - started
    return rec


#: (t, quad, cfl, degree, cells) presets, addressable by index for sharded runs.
PRESETS = [
    (2.0, 4, 0.2, 2, 32),   # 0 baseline (committed ladder config at t=2)
    (2.0, 6, 0.2, 2, 32),   # 1 moderate de-aliasing
    (2.0, 8, 0.2, 2, 32),   # 2 strong de-aliasing
    (5.0, 4, 0.2, 2, 32),   # 3 baseline, long time
    (5.0, 8, 0.2, 2, 32),   # 4 de-aliased, long time
    (2.0, 4, 0.1, 2, 32),   # 5 CFL sensitivity of the baseline
    (2.0, 8, 0.1, 2, 32),   # 6 CFL sensitivity, de-aliased
    (1.0, 8, 0.2, 2, 32),   # 7 de-aliased short time
    (2.0, 4, 0.2, 3, 16),   # 8 P3/16^2 baseline (same dofs as P2/32^2)
    (2.0, 8, 0.2, 3, 16),   # 9 P3/16^2 de-aliased
    (2.0, 4, 0.2, 1, 32),   # 10 P1/32^2 baseline
    (2.0, 8, 0.2, 1, 32),   # 11 P1/32^2 de-aliased
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=2.0)
    ap.add_argument("--quad", type=int, default=4)
    ap.add_argument("--cfl", type=float, default=0.2)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--preset", type=int, default=None)
    a = ap.parse_args()
    if a.preset is not None:
        a.t, a.quad, a.cfl, a.degree, a.cells = PRESETS[a.preset]
        a.tag = a.tag or (f"preset{a.preset}_t{a.t:g}_q{a.quad}_cfl{a.cfl:g}"
                          f"_d{a.degree}_n{a.cells}")
    rec = run(a.t, a.quad, a.cfl, a.degree, a.cells)
    outdir = Path("scratch/diag_alias")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"t{a.t:g}_q{a.quad}_cfl{a.cfl:g}_d{a.degree}_n{a.cells}"
    (outdir / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))