from __future__ import annotations
import math, time, json
from pathlib import Path
import torch
import tensordg.euler_dg_2d as dg2
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0)); FINAL_TIME = 1.0; CFL0 = 0.2
STRENGTH = 5.0; BACKGROUND = (1.0, 1.0)
def vortex_primitive(x,y,t): return dg2.vortex_primitive(x,y,t,STRENGTH,BACKGROUND)
def vortex_conservative(x,y,t):
    rho,u,v,p = vortex_primitive(x,y,t); return dg2.conservative_state(rho,u,v,p)
def initial(x,y): return vortex_primitive(x,y,0.0)
def gauss_fields(coeff,degree,quadrature=None):
    order = quadrature if quadrature is not None else max(degree+2,3)
    nodes,weights,basis,_,_ = dg2.legendre_operators(degree,order)
    values = torch.einsum("xyklv,ak,bl->xyabv",coeff,basis,basis)
    return values,nodes,weights
def error_norms(coeff,x,y,degree,final_time=FINAL_TIME,domain=DOMAIN):
    values,nodes,weights = gauss_fields(coeff,degree)
    (xmin,xmax),(ymin,ymax) = domain
    nx,ny = x.numel(),y.numel(); hx,hy=(xmax-xmin)/nx,(ymax-ymin)/ny
    X = x[:,None,None,None]+nodes[None,None,:,None]*(hx/2.0)
    Y = y[None,:,None,None]+nodes[None,None,None,:]*(hy/2.0)
    exact = vortex_conservative(X,Y,final_time)
    jacobian = (weights[None,None,:,None]*weights[None,None,None,:]*(hx/2.0)*(hy/2.0))
    diff = (values-exact).abs()
    l1 = (diff*jacobian).sum(); scale=(exact[...,0].abs()*jacobian).sum()
    return {"L1_rel": float(l1/scale)}
def run(degree,ncells):
    nx=ny=ncells; t0=time.time()
    x,y,coeff,meta = dg2.solve_euler_dg_2d(initial,ncells=(nx,ny),degree=degree,
        domain=DOMAIN,final_time=FINAL_TIME,cfl=CFL0,periodic=(True,True),device="cpu")
    wall=time.time()-t0; err=error_norms(coeff,x,y,degree)["L1_rel"]
    h=(DOMAIN[0][1]-DOMAIN[0][0])/nx
    return {"ncells":nx,"h":h,"err":err,"wall":round(wall,2),"steps":meta["steps"],"dofs":meta["dofs"]}
torch.set_num_threads(4)
for n in [16, 24]:
    r = run(2, n)
    print(f"p2 n={n} h={r['h']:.4f} L1_rel={r['err']:.6e} wall={r['wall']}s steps={r['steps']} dofs={r['dofs']}")
