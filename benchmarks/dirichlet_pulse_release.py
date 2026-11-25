"""Release after a Dirichlet pulse, checked against finite-interval drift-diffusion eigenmodes."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from tensordg.advection_diffusion_1d import sample_solution,solve_transient_advection_diffusion_1d


START,STOP = .017,.063
OBSERVE = np.array([.075,.10,.14,.2])


def reference(x,t,speed,modes=96):
    """Concentration at t and integral of rightward current from OBSERVE[0] to t."""
    d = abs(speed)/4
    b = speed/(2*d)
    k = np.pi*np.arange(1,modes+1)
    rate = d*(k*k+b*b)
    # In u=exp(b*x)*v, the driven left endpoint forces v's sine
    # coefficient with 2*D*k. Every mode decays freely after removal.
    at_stop = 2*d*k*(-np.expm1(-rate*(STOP-START)))/rate
    a = at_stop*np.exp(-rate*(t-STOP))
    integ = at_stop*np.exp(-rate*(OBSERVE[0]-STOP))*(-np.expm1(-rate*(t-OBSERVE[0])))/rate
    angle = x[:,None]*k
    field = np.exp(b*x)*(np.sin(angle) @ a)
    current = np.exp(b*x)*(((speed-d*b)*np.sin(angle)-d*k*np.cos(angle)) @ integ)
    return field,current


def measure(n,speed):
    # Refine the same partition at every level. This keeps the damping
    # interval after each switch proportional to the nominal step size.
    knots = np.r_[0.,START,STOP,OBSERVE]
    times = np.r_[0.,np.concatenate([np.linspace(a,b,int(np.ceil((b-a)/.2*32))*n//8+1)[1:]
                                     for a,b in zip(knots,knots[1:])])]
    centers,u,_,(west,east) = solve_transient_advection_diffusion_1d(times,0.,degree=3,ncells=n,
                peclet=4.,speed=speed,left_value=0.,right_value=0.,rannacher=True,
                boundary_value_history=lambda t:(float(START<=t<STOP),0.),source_jump_times=(START,STOP),
                return_face_transport=True)
    faces = np.linspace(0.,1.,n+1)
    ids = np.searchsorted(times,OBSERVE)
    weights = np.polynomial.legendre.leggauss(16)[1]/(2*n)
    field_error = transport_error = tail = 0.
    for t,i in zip(OBSERVE,ids):
        x,samples = sample_solution(centers,u[i],16)
        exact,_ = reference(x.numpy(),t,speed)
        _,transport = reference(faces,t,speed)
        field_error = max(field_error,float(np.sqrt(np.sum(weights*(samples.numpy()-exact).reshape(n,-1)**2))))
        measured = np.r_[west[ids[0]:i,0].sum().item(),east[ids[0]:i].sum(dim=0).numpy()]
        transport_error = max(transport_error,float(abs(measured-transport).max()))
        tail = max(tail,float(abs(reference(x.numpy(),t,speed,48)[0]-exact).max()),
                   float(abs(reference(faces,t,speed,48)[1]-transport).max()))
    return dict(field_l2=field_error,section_transport=transport_error,series_change=tail,
                mass_balance=float(abs(u[:,:,0].diff(dim=0)/n-west+east).max()))


if __name__=="__main__":
    rows = []
    for speed in (-1.,1.):
        previous = {}
        for n in (8,16,32,64):
            errors = measure(n,speed)
            orders = {k:float(np.log2(previous[k]/errors[k])) for k in ("field_l2","section_transport")} if previous else {}
            rows.append(dict(cells=n,speed=speed,errors=errors,orders=orders,
                             passed=errors["series_change"]<1e-12 and errors["mass_balance"]<1e-10
                             and all(v>0 for v in orders.values()) and (n<64 or min(orders.values())>1.8)))
            previous = errors
    revision = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=revision,post_pulse_times=OBSERVE.tolist(),runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
