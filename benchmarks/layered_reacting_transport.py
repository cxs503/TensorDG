"""Graded, three-material DG transport in the cumulative diffusion-resistance coordinate."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import face_fluxes,numerical_face_fluxes,sample_solution,solve_advection_diffusion_1d


def measure(n,degree,speed,point_sources=False,surface_reaction=False,periodic=False):
    breaks = np.array([0.,.3,.75,1.])
    materials = np.array([.4,4.,.8])
    lengths = np.diff(breaks)
    param = np.arange(n+1)/n
    local = param+.35*param*(1-param)
    edges = torch.from_numpy(np.r_[np.concatenate([x+h*local[:-1] for x,h in zip(breaks,lengths)]),1.])
    ratio = torch.from_numpy(materials.repeat(n))
    wave = 2*np.pi/np.sum(lengths/materials)
    pe = 5.
    d = abs(speed)/pe
    extra_flux = np.array([0.,.8,-.5]) if point_sources else np.zeros(3)
    if periodic:
        # Make the concentration increment over the three linear segments zero.
        extra_flux -= np.average(extra_flux,weights=lengths/materials)
    slopes = -extra_flux/(d*materials)
    cell_flux = torch.from_numpy(extra_flux.repeat(n))
    interfaces = torch.tensor([n,2*n,3*n] if periodic else [n,2*n])
    next_cell = interfaces%(3*n)
    exact_jump = torch.from_numpy(np.diff(np.r_[extra_flux,extra_flux[0]] if periodic else extra_flux))
    sheet = edges.new_zeros(3*n if periodic else 3*n-1)
    sheet[interfaces-1] = exact_jump
    def phase(x):
        z = sum((x-a).clamp(0.,h)/m for a,h,m in zip(breaks,lengths,materials))
        return wave*z+.37
    def value(x):
        return .7+torch.sin(phase(x))+sum(b*(x-a).clamp(0.,h) for a,h,b in zip(breaks,lengths,slopes))
    def flux(x):
        return speed*value(x)-d*wave*torch.cos(phase(x))
    def reaction(x):
        return 0. if surface_reaction else 1+x
    def source(x):
        arg = phase(x)
        return speed*torch.from_numpy(slopes.repeat(n))[:,None]+(speed*wave*torch.cos(arg)+d*wave**2*torch.sin(arg))/ratio[:,None]+reaction(x)*value(x)
    opts = dict(periodic=periodic,interface_source=sheet,cell_edges=edges,diffusivity_ratio=ratio,left_value=value(edges)[0].item(),
                right_value=value(edges)[-1].item())
    if surface_reaction:
        uptake = torch.zeros_like(sheet)
        if periodic:
            uptake[-1] = 1.5
        else:
            uptake[interfaces-1] = edges.new_tensor([1.5,.6])
        sheet[interfaces-1] += uptake[interfaces-1]*value(edges[interfaces])
        opts.update(interface_reaction=uptake)
        if not periodic:
            opts.update(total_flux=(float(flux(edges[0])+extra_flux[0]),float(flux(edges[-1])+extra_flux[-1])))
    centers,coeff,_ = solve_advection_diffusion_1d(degree=degree,peclet=pe,speed=speed,
                    source=source,reaction=reaction,**opts)
    x,uh = sample_solution(centers,coeff,samples_per_cell=16,cell_edges=edges)
    x,uh = x.reshape(-1,16),uh.reshape(-1,16)
    weights = .5*edges.diff()[:,None]*torch.from_numpy(np.polynomial.legendre.leggauss(16)[1])
    west,east = numerical_face_fluxes(coeff,pe,speed,return_cell_fluxes=True,**opts)
    one_sided = face_fluxes(coeff,pe,speed,cell_edges=edges,diffusivity_ratio=ratio)
    net = (weights*(source(x)-reaction(x)*uh)).sum(dim=1)
    east_error = east-flux(edges[1:])-cell_flux
    west_error = west-flux(edges[:-1])-cell_flux
    return dict(l2=float(torch.sqrt((weights*(uh-value(x))**2).sum())),
                one_sided_flux=float((one_sided-flux(edges[1:])-cell_flux).abs().max()),
                interface_flux=float(max(east_error[interfaces-1].abs().max(),west_error[next_cell].abs().max())),
                interface_jump=float((west[next_cell]-east[interfaces-1]-exact_jump).abs().max()),
                cell_balance=float((east-west-net).abs().max()))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-sources",action="store_true",help="inject at the first material interface and remove at the second")
    parser.add_argument("--surface-reaction",action="store_true",help="prescribe both end fluxes and anchor concentration with two surface reactions")
    parser.add_argument("--periodic",action="store_true",help="join the outer materials and put surface reaction at the seam")
    args = parser.parse_args()
    rows = []
    for speed in (-1.25,1.25):
        for degree in (1,2,3):
            previous = {}
            for n in (4,8,16,32):
                errors = measure(n,degree,speed,args.point_sources,args.surface_reaction,args.periodic)
                orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("l2","one_sided_flux")} if previous else {}
                passed = errors["cell_balance"]<1e-9 and all(v>0 for v in orders.values())
                if n==32:
                    passed &= orders["l2"]>degree+.8 and orders["one_sided_flux"]>degree-.2 and max(errors["interface_flux"],errors["interface_jump"])<.01
                rows.append(dict(speed=speed,degree=degree,cells_per_material=n,errors=errors,orders=orders,passed=passed))
                previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,periodic=args.periodic,point_sources=args.point_sources,surface_reaction=args.surface_reaction,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
