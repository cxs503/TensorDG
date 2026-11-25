"""Complex forced thermal waves between independently phased Dirichlet walls."""
import cmath
import json
import math
from pathlib import Path
import subprocess

import torch
from tensordg import heat_error, solve_harmonic_heat_1d


def measure(n, degree, omega):
    k, c = 2.3, 1.7
    walls = (.8+.3j, -.4+.6j)
    source = .7-1.1j
    wave = cmath.sqrt(1j*omega*c/k)
    particular = source/(1j*omega*c)
    left, right = ((wall-particular)/cmath.sinh(wave) for wall in walls)
    def temperature(x, _):
        return particular+left*torch.sinh(wave*(1-x))+right*torch.sinh(wave*x)
    x,u,q = solve_harmonic_heat_1d(omega,ncells=n,degree=degree,conductivity=k,capacity=c,
                                   wall_values=walls,source=source)
    f = torch.linspace(0.,1.,n+1,dtype=x.dtype)
    target = k*wave*(left*torch.cosh(wave*(1-f))-right*torch.cosh(wave*f))
    error = heat_error(x,u,0.,reference=temperature)
    return dict(cells=n,degree=degree,omega=omega,temperature_l2=error["absolute_l2"],
                max_face_flux_error=float((q-target).abs().max()),
                max_wall_phase_error_rad=float(torch.angle(q[[0,-1]]/target[[0,-1]]).abs().max()),
                max_cell_balance=float((q.diff()+(1j*omega*c*u[:,0]-source)/n).abs().max()))


if __name__ == "__main__":
    rows = []
    for omega in (1.,20.,80.):
        for degree in (1,2,3):
            previous = None
            for n in (8,16,32):
                row = measure(n,degree,omega)
                error = row["temperature_l2"]
                order = None if previous is None else math.log2(previous/error)
                passed = row["max_cell_balance"]<1e-9 and (order is None or order>degree+.65)
                if n==32:
                    passed &= row["max_wall_phase_error_rad"]<(5e-3,3e-5,1e-7)[degree-1]
                rows.append(dict(**row,temperature_order=order,passed=passed))
                previous = error
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(r["passed"] for r in rows) else 1)
