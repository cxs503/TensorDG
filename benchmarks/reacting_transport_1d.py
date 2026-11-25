"""Smooth manufactured transport with a variable reaction and signed velocity."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import numerical_face_fluxes, sample_solution, solve_advection_diffusion_1d


def measure(n,degree,speed,boundary,mean_constrained,variable_velocity=False):
    pe = 6.
    d = abs(speed)/pe
    wave = 2*np.pi
    def field(x):
        return .3+.7*x+torch.sin(wave*x+.2)
    def velocity(x):
        return speed*(.2+torch.sin(wave*x+.3)) if variable_velocity else speed
    def velocity_gradient(x):
        return speed*wave*torch.cos(wave*x+.3) if variable_velocity else 0.
    def flux(x):
        return velocity(x)*field(x)-d*(.7+wave*torch.cos(wave*x+.2))
    def reaction(x):
        return 0. if mean_constrained else 1+x
    def source(x):
        return velocity_gradient(x)*field(x)+velocity(x)*(.7+wave*torch.cos(wave*x+.2))+d*wave**2*torch.sin(wave*x+.2)+reaction(x)*field(x)
    faces = torch.linspace(0.,1.,n+1,dtype=torch.float64)
    west,east = field(faces)[[0,-1]].tolist()
    exact_wall = flux(faces)[[0,-1]].tolist()
    imposed = tuple(q if kind=="F" else None for kind,q in zip(boundary,exact_wall))
    exchange = tuple((k,wall-normal*q/k) if kind=="R" else None for kind,k,wall,normal,q in
                     zip(boundary,(1.3,2.),(west,east),(-1.,1.),exact_wall))
    centers,coeff,_ = solve_advection_diffusion_1d(n,degree,pe,speed,left_value=west,right_value=east,
                                                   reaction=reaction,source=source,total_flux=imposed,boundary_exchange=exchange,velocity=velocity,
                                                   mean_value=.65 if mean_constrained else None)
    x,uh = sample_solution(centers,coeff,samples_per_cell=16)
    weights = torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])/(2*n)
    l2 = torch.sqrt(((uh-field(x)).square().reshape(n,-1) @ weights).sum())
    actual = numerical_face_fluxes(coeff,pe,speed,left_value=west,right_value=east,total_flux=imposed,boundary_exchange=exchange,velocity=velocity)
    net = (source(x)-reaction(x)*uh).reshape(n,-1) @ weights
    signs = coeff.new_tensor([(-1.)**j for j in range(degree+1)])
    wall_error = max(abs(float(coeff[0] @ signs)-west),abs(float(coeff[-1].sum())-east))
    return dict(wall_value_max=wall_error,global_balance=float(abs(actual[-1]-actual[0]-net.sum())),l2=float(l2),face_flux_max=float((actual-flux(faces)).abs().max()),
                cell_balance_max=float((actual.diff()-net).abs().max()))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("DD","DF","FD","FF","RD","DR","RF","FR","RR"), default="DD",
                        help="left/right boundary type: D for value, F for rightward total flux, R for outward reservoir exchange")
    parser.add_argument("--mean-constrained",action="store_true",
                        help="set reaction to zero and impose the exact mean under two flux boundaries")
    parser.add_argument("--variable-velocity",action="store_true",help="use sinusoidal conservative drift with two internal stagnation points")
    args = parser.parse_args()
    boundary = args.boundary
    if args.mean_constrained and boundary!="FF":
        parser.error("--mean-constrained requires --boundary FF")
    rows = []
    for speed in (-1.25,1.25):
        for degree in (1,2,3):
            previous = None
            for n in (8,16,32,64):
                errors = measure(n,degree,speed,boundary,args.mean_constrained,args.variable_velocity)
                order = float(np.log2(previous/errors["l2"])) if previous else None
                rows.append(dict(cells=n,degree=degree,speed=speed,errors=errors,l2_order=order,
                                 passed=errors["cell_balance_max"]<1e-10 and errors["global_balance"]<1e-9 and
                                 (order is None or order>degree+.75)))
                previous = errors["l2"]
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,boundary=boundary,variable_velocity=args.variable_velocity,mean_constrained=args.mean_constrained,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
