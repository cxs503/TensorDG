"""Spatial SIPG convergence for two forced Fourier modes starting from zero."""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unequal-walls", action="store_true", help="superpose a stationary linear temperature profile")
    walls = (1.2,-.3) if parser.parse_args().unequal_walls else (0.,0.)
    def initial(x):
        return walls[0]+(walls[1]-walls[0])*x
    rows = []
    for degree in (1,2,3):
        previous = None
        previous_heat = None
        for cells in (8,16,32):
            time = .04
            x, coeff, meta = solve_heat_1d_exact_time(cells, degree, time, source=source,
                                                   initial_temperature=initial, wall_temperatures=walls, integrate_face_fluxes=True)
            error = heat_error(x,coeff,time,reference=lambda x,t: initial(x)+reference(x,t))["absolute_l2"]
            faces = torch.linspace(0.,1.,cells+1,dtype=coeff.dtype)
            heat = torch.tensor(meta["integrated_face_fluxes"],dtype=coeff.dtype)
            exact_heat = torch.full_like(faces, time*(walls[0]-walls[1]))
            deposited = torch.zeros(cells,dtype=coeff.dtype)
            for wave, amplitude in ((math.pi,1.), (3*math.pi,-.2)):
                cosine = torch.cos(wave*faces)
                exact_heat -= amplitude*wave*(time+math.expm1(-wave**2*time)/wave**2)*cosine
                deposited -= amplitude*wave*time*cosine.diff()
            flux_error = float((heat-exact_heat).abs().max())
            balance = float(((coeff[:,0]-initial(x))/cells + heat.diff()-deposited).abs().max())
            order = None if previous is None else math.log2(previous/error)
            passed = math.isfinite(error) and (order is None or order > degree+.7)
            heat_order = None if previous_heat is None else math.log2(previous_heat/flux_error)
            passed &= balance < 1e-10 and (heat_order is None or heat_order > 2*degree-.5)
            rows.append(dict(degree=degree, cells=cells, absolute_l2=error, order=order, max_face_heat_error=flux_error, face_heat_order=heat_order, max_cell_balance=balance, passed=passed))
            previous, previous_heat = error, flux_error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,wall_temperatures=walls,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
