"""SIPG startup from zero with two independently ramped walls and no volume source."""
import json
import math
from pathlib import Path
import subprocess

import torch
from tensordg.heat_1d import heat_error, solve_heat_1d_ramped_walls


def reference(x, t, heat=False):
    # w''=r(x)=2-3x, w(0)=w(1)=0. Expand the initial correction -w in sines.
    n = torch.arange(1,65,device=x.device)
    k = n.to(x.dtype)*math.pi
    amplitude = 2*(2+(-1.)**n)/k**3
    angle = x[...,None]*k
    if heat:
        # Sum the non-decaying cosine series analytically. Its derivative is
        # w and its mean on the slab is zero. Only the exponential tail remains.
        primitive = 1/40 + x**3/3 - x**4/8 - x**2/4
        modal_heat = amplitude*torch.exp(-t*k*k)/k
        return 1.5*t*t-t*(2*x-1.5*x**2-.5)-primitive+(torch.cos(angle)*modal_heat).sum(dim=-1)
    return t*(2-3*x)+x*x-.5*x**3-.5*x+(torch.sin(angle)*amplitude*torch.exp(-t*k*k)).sum(dim=-1)


if __name__ == "__main__":
    rows = []
    for degree in (1,2,3):
        previous = None
        for cells in (8,16,32):
            t = .06
            x,c,meta = solve_heat_1d_ramped_walls(cells,degree,t,wall_temperatures=(0.,0.),
                wall_rates=(2.,-1.),initial_temperature=torch.zeros_like,backend="exact-time",integrate_face_fluxes=True)
            error = heat_error(x,c,t,reference=reference)["absolute_l2"]
            faces = torch.linspace(0.,1.,cells+1,dtype=c.dtype)
            heat = c.new_tensor(meta["integrated_face_fluxes"])
            heat_error_max = float((heat-reference(faces,t,heat=True)).abs().max())
            balance = float((c[:,0]/cells+heat.diff()).abs().max())
            order = None if previous is None else math.log2(previous/error)
            passed = (cells != 32 or heat_error_max < {1:3e-5,2:6e-9,3:2e-12}[degree]) and (order is None or order>degree+.6) and balance<1e-10
            rows.append(dict(cells=cells,degree=degree,absolute_l2=error,order=order,
                             face_heat_error=heat_error_max,cell_balance=balance,passed=passed))
            previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
