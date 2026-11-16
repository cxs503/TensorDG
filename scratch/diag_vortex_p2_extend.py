"""Light diagnostic: extend the CKS1998 vortex p2 ladder with 24 and 32 cells
to see whether the observed order converges toward design order 3 as the mesh
refines (the committed certification only used 8/12/16).
"""
from __future__ import annotations
import math, time, json
from pathlib import Path
import torch
import tensordg.euler_dg_2d as dg2

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 1.0
CFL0 = 0.2
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)

def vortex_primitive(x, y, t):
    return dg2.vortex_primitive(x, y, t, STRENGTH, BACKGROUND)
def vortex_conservative(x, y, t):
    rho, u, v, p = vortex_primitive(x, y, t)
    return dg2.conservative_state(rho, u, v, p)
def initial(x, y):
    return vortex_primitive(x, y, 0.0)

def gauss_fields(coeff, degree, quadrature=None):
    order = quadrature if quadrature is not None else max(degree + 2, 3)
    nodes, weights, basis, _, _ = dg2.legendre_operators(degree, order)
    values = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    return values, nodes, weights

def error_norms(coeff, x, y, degree, final_time=FINAL_TIME, domain=DOMAIN):
    values, nodes, weights = gauss_fields(coeff, degree)
    (xmin, xmax), (ymin, ymax) = domain
    nx, ny = x.numel(), y.numel()
    hx, hy = (xmax - xmin) / nx, (ymax - ymin) / ny
    X = x[:, None, None, None] + nodes[None, None, :, None] * (hx / 2.0)
    Y = y[None, :, None, None] + nodes[None, None, None, :] * (hy / 2.0)
    exact = vortex_conservative(X, Y, final_time)
    jacobian = (weights[None, None, :, None] * weights[None, None, None, :]
                * (hx / 2.0) * (hy / 2.0))
    diff = (values - exact).abs()
    l1 = (diff * jacobian).sum()
    scale = (exact[..., 0].abs() * jacobian).sum()
    return {"L1_rel": float(l1 / scale)}

def run(degree, ncells):
    nx = ny = ncells
    t0 = time.time()
    x, y, coeff, meta = dg2.solve_euler_dg_2d(
        initial, ncells=(nx, ny), degree=degree, domain=DOMAIN,
        final_time=FINAL_TIME, cfl=CFL0, periodic=(True, True), device="cpu")
    wall = time.time() - t0
    err = error_norms(coeff, x, y, degree)["L1_rel"]
    h = (DOMAIN[0][1] - DOMAIN[0][0]) / nx
    return {"ncells": nx, "h": h, "err": err, "wall": round(wall, 2),
            "steps": meta["steps"], "dofs": meta["dofs"]}

def main():
    torch.set_num_threads(4)
    degree = 2
    # committed ladder: 8,12,16 (err from records_p2.json); extend with 24,32
    committed = {8: 0.001212487708760672, 12: 0.0003527612535566812,
                 16: 0.00015505979135996484}
    rows = []
    for n in [24, 32]:
        r = run(degree, n)
        rows.append(r)
        print(f"p{degree} n={n} h={r['h']:.4f} L1_rel={r['err']:.6e} "
              f"wall={r['wall']}s steps={r['steps']} dofs={r['dofs']}")
    # combine with committed
    all_rows = [{"ncells": n, "h": (10.0)/n, "err": e} for n, e in committed.items()]
    all_rows += rows
    all_rows.sort(key=lambda r: r["ncells"])
    print("\n--- full ladder (committed + new) ---")
    for r in all_rows:
        print(f"  n={r['ncells']} h={r['h']:.4f} L1_rel={r['err']:.6e}")
    print("\n--- observed order (adjacent pairs) ---")
    for a, b in zip(all_rows, all_rows[1:]):
        p = math.log(a["err"] / b["err"]) / math.log(a["h"] / b["h"])
        print(f"  n {a['ncells']}->{b['ncells']}: order {p:.4f}")
    import numpy as np
    xs = [math.log(r["h"]) for r in all_rows]
    ys = [math.log(r["err"]) for r in all_rows]
    p_lsq = np.polyfit(xs, ys, 1)[0]
    print(f"  least-squares order (all 5 levels): {p_lsq:.4f}")
    print(f"  design order: {degree+1}")
    out = Path(__file__).resolve().parent / "diag_vortex_p2_extend.json"
    out.write_text(json.dumps({"degree": degree, "rows": all_rows,
                               "p_lsq": float(p_lsq)}, indent=2))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
