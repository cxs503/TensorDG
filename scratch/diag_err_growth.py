"""Diagnostic: DG-P2/N32 isentropic-vortex error trajectory, per field & structure."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive
from benchmarks.verify_euler_dg_2d import _reconstruct, _vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH, BG = 5.0, (1.0, 1.0)


def main(times):
    degree, cells = 2, 32
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    out = []
    for t in times:
        xc, yc, coeff, meta = dg2.solve_euler_dg_2d_vortex(
            ncells=cells, degree=degree, final_time=t, cfl=0.2,
            strength=STRENGTH, background_velocity=BG, domain=DOMAIN)
        err = _vortex_error(coeff, basis, nodes, DOMAIN, t, STRENGTH, BG)
        rec = {"t": t, "steps": meta["steps"],
               "errors": {k: float(err[k]) for k in err}}
        state = _reconstruct(coeff, basis)
        rho, u, v, p = dg2._primitive(state)
        dx = (DOMAIN[0][1] - DOMAIN[0][0]) / cells
        dy = (DOMAIN[1][1] - DOMAIN[1][0]) / cells
        xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
        yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
        ex_rho, ex_u, ex_v, ex_p = vortex_primitive(xq, yq, t, STRENGTH, BG)
        for name, num, ref in (("rho", rho, ex_rho), ("u", u, ex_u),
                               ("v", v, ex_v), ("p", p, ex_p)):
            d = (num - ref).numpy()
            rec[f"{name}_l2"] = float(np.sqrt(np.mean(d ** 2)))
            rec[f"{name}_absmax"] = float(np.abs(d).max())
            rec[f"{name}_l2_over_max"] = rec[f"{name}_l2"] / max(rec[f"{name}_absmax"], 1e-300)
        rec["density_min"] = float(rho.min())
        de = (rho - ex_rho).numpy()
        de_cell = de.mean(axis=(2, 3))
        spec = np.abs(np.fft.fft2(de_cell))
        n = spec.shape[0]
        rec["err_spec_frac_high"] = float(spec[n // 4:, :].sum() / max(spec.sum(), 1e-300))
        # velocity error vs density error ratio (both relative L2)
        rec["u_over_rho_rel"] = rec["errors"]["velocity_x"] / rec["errors"]["density"]
        out.append(rec)
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in rec.items()}), flush=True)
    outdir = Path("scratch/diag_err_growth")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"trajectory_t{times[0]:g}.json").write_text(json.dumps(out, indent=1))


TIMES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--index":
        i = int(sys.argv[2])
        main([TIMES[i]])
    else:
        main([float(a) for a in sys.argv[1:]] or TIMES)