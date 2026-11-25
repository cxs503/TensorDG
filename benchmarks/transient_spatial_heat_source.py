"""Independent orthogonal heating modes in an insulated variable-capacity plate."""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from tensordg import l2_error_2d, poisson_face_powers_2d, solve_transient_diffusion_2d


def measure(steps, theta):
    nx,ny = 4,3
    times = np.linspace(0.,.5,steps+1)
    def mode(z):
        return z*z*(3.-2*z)
    def capacity(x,y):
        return 1.+x*y
    def exact(t,x,y):
        return math.sin(3*t)*mode(x)+math.cos(2*t)*mode(y)
    def source(t,x,y):
        rate = 3*math.cos(3*t)*mode(x)-2*math.sin(2*t)*mode(y)
        return capacity(x,y)*rate-math.sin(3*t)*(6-12*x)-math.cos(2*t)*(6-12*y)
    bc = dict.fromkeys(("left","right","bottom","top"),"neumann")
    x,y,states,means = solve_transient_diffusion_2d(
        lambda x,y:exact(0.,x,y),times,ncells=(nx,ny),degree=3,bc=bc,theta=theta,
        capacity=capacity,source_history=source,return_interval_means=True)
    z,w = np.polynomial.legendre.leggauss(9)
    xq = x.numpy()[:,None,None,None]+z[None,None,:,None]/(2*nx)
    yq = y.numpy()[None,:,None,None]+z/(2*ny)
    weights = np.outer(w,w)/(4*nx*ny)
    basis = np.polynomial.legendre.legvander(z,3)
    samples = np.einsum("txyij,qi,rj->txyqr",states.numpy(),basis,basis)
    stored = (samples*capacity(xq,yq)*weights).sum(axis=(-1,-2))
    field = l2_error_2d(x,y,states[-1],lambda x,y:exact(times[-1],x,y))["absolute_l2"]
    balance = transport = 0.
    xe,ye = np.linspace(0.,1.,nx+1),np.linspace(0.,1.,ny+1)
    for i,dt in enumerate(np.diff(times)):
        t0,t1 = times[i:i+2]
        qx,qy = poisson_face_powers_2d(means[i],bc=bc)
        heating = ((1-theta)*source(t0,xq,yq)+theta*source(t1,xq,yq))*weights
        residual = stored[i+1]-stored[i]+dt*((qx.diff(dim=0)+qy.diff(dim=1)).numpy()-heating.sum(axis=(-1,-2)))
        balance = max(balance,float(abs(residual).max()))
        avg_x = (math.cos(3*t0)-math.cos(3*t1))/(3*dt)
        avg_y = (math.sin(2*t1)-math.sin(2*t0))/(2*dt)
        transport = max(transport,float(abs(qx.numpy()+avg_x*(6*xe*(1-xe))[:,None]/ny).max()),
                        float(abs(qy.numpy()+avg_y*6*ye*(1-ye)/nx).max()))
    return dict(field=field,mean_face_power=transport,balance=balance)


if __name__=="__main__":
    rows = []
    for theta in (.5,1.):
        previous = {}
        for steps in (8,16,32,64):
            errors = measure(steps,theta)
            orders = {k:math.log2(previous[k]/errors[k]) for k in ("field","mean_face_power") if previous}
            passed = errors["balance"]<1e-11 and (steps<64 or min(orders.values())>(1.85 if theta==.5 else .9))
            rows.append(dict(theta=theta,steps=steps,errors=errors,orders=orders,passed=passed))
            previous = errors
    rev = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=rev,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
