import math, torch
import tensordg.ns_dg_2d as ns
from tensordg.wall_flux_extrapolation import wall_heat_flux_extrapolated_summary
MU=0.1; PR=0.71; K=MU*ns.CP/PR; TH,TC,P0=1.0,0.5,1.0; BETA=0.5
def s(y): return y + BETA*y*(1.0-y)
def state(x,y):
    ref=x*y+0.0*x+0.0*y
    T=TH+(TC-TH)*(s(y)+0.0*x)
    return (P0/T, torch.zeros_like(ref), torch.zeros_like(ref), P0*torch.ones_like(ref))
def case(ny,deg,wg):
    nx=2; dx,dy=1.0/nx,1.0/ny
    nodes,w,basis,deriv,signs=ns.legendre_operators(deg,deg+2,torch.device("cpu"),torch.float64)
    x=(torch.arange(nx,dtype=torch.float64)+0.5)*dx; y=(torch.arange(ny,dtype=torch.float64)+0.5)*dy
    coeff=ns._initial_coefficients(state,x,y,dx,dy,nodes,w,basis,torch.device("cpu"),torch.float64)
    pen=2.0*(deg+1.0)**2
    b={"xmin":None,"xmax":None,"ymin":{"velocity":(0.0,0.0),"temperature":TH},"ymax":{"velocity":(0.0,0.0),"temperature":TC}}
    direct=ns.wall_heat_flux_summary(coeff,basis,deriv,signs,w,dx,dy,b,MU,K,pen,wg)
    ex={p:wall_heat_flux_extrapolated_summary(coeff,basis,deriv,signs,w,dx,dy,b,MU,K,pen,wg,points=p) for p in (1,2,3)}
    return direct, ex
# analytic: T'(0)=(TC-TH)(1+BETA), T'(1)=(TC-TH)(1-BETA)
ref_bot=-K*(TC-TH)*(1+BETA); ref_top=-K*(TC-TH)*(1-BETA)
print("analytic bottom(top face at ymin, normal +y):", ref_bot, " top:", ref_top)
for deg in (1,2):
    for wg in ("face",):
        print(f"--- p{deg} {wg}")
        for ny in (8,16,32,64):
            d,e=case(ny,deg,wg)
            print(f" ny={ny:3d} direct={d['ymin']:.9f}/{d['ymax']:.9f} p1={e[1]['ymin']:.9f}/{e[1]['ymax']:.9f} p2={e[2]['ymin']:.9f}/{e[2]['ymax']:.9f} p3={e[3]['ymin']:.9f}/{e[3]['ymax']:.9f}")
        pass
