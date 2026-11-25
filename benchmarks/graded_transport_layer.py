"""Outflow-layer resolution on equal and exponentially graded DG meshes."""
import json
from pathlib import Path
import subprocess

import numpy as np

from tensordg.advection_diffusion_1d import (advection_diffusion_error,exact_flux,
                                             numerical_face_fluxes,solve_advection_diffusion_1d)


def measure(n,degree,speed,graded):
    pe = 80.
    param = np.linspace(0.,1.,n+1)
    edges = np.expm1(4*param)/np.expm1(4.) if graded else param
    if speed>0:
        edges = 1.-edges[::-1].copy()
    centers,coeff,_ = solve_advection_diffusion_1d(degree=degree,peclet=pe,speed=speed,
                    left_value=.2,right_value=1.2,cell_edges=edges)
    error = advection_diffusion_error(centers,coeff,pe,speed,.2,1.2,cell_edges=edges)
    flux = numerical_face_fluxes(coeff,pe,speed,left_value=.2,right_value=1.2,cell_edges=edges)
    return dict(l2=error["absolute_l2"],face_flux_error=float(abs(flux-exact_flux(pe,speed,.2,1.2)).max()),
                max_cell_imbalance=float(flux.diff().abs().max()),min_width=float(np.diff(edges).min()))


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        for degree in (1,2,3):
            previous = None
            for n in (8,16,32,64):
                uniform = measure(n,degree,speed,False)
                graded = measure(n,degree,speed,True)
                order = float(np.log2(previous/graded["l2"])) if previous else None
                passed = (graded["l2"]<uniform["l2"] and graded["max_cell_imbalance"]<1e-8
                          and graded["face_flux_error"]<1e-3 and (n<64 or (order>degree+.7 and graded["face_flux_error"]<1e-8)))
                rows.append(dict(cells=n,degree=degree,speed=speed,uniform=uniform,graded=graded,
                                 graded_order=order,passed=passed))
                previous = graded["l2"]
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(peclet=80,commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
