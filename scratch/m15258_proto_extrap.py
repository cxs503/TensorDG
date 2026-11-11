import math, torch, json
import tensordg.ns_dg_2d as ns

MU=0.1; PR=0.71; K=MU*ns.CP/PR; TH,TC,P0=1.0,0.5,1.0
def steady(x,y):
    ref=x*y+0.0*x+0.0*y
    T=TH+(TC-TH)*(y+0.0*x)
    return (P0/T, torch.zeros_like(ref), torch.zeros_like(ref), P0*torch.ones_like(ref))

def measure(ny, deg, wg):
    nx=2; dx,dy=1.0/nx,1.0/ny
    nodes,weights,basis,derivative,signs=ns.legendre_operators(deg,deg+2,torch.device("cpu"),torch.float64)
    x=(torch.arange(nx,dtype=torch.float64)+0.5)*dx
    y=(torch.arange(ny,dtype=torch.float64)+0.5)*dy
    coeff=ns._initial_coefficients(steady,x,y,dx,dy,nodes,weights,basis,torch.device("cpu"),torch.float64)
    pen=2.0*(deg+1.0)**2
    b={"xmin":None,"xmax":None,"ymin":{"velocity":(0.0,0.0),"temperature":TH},"ymax":{"velocity":(0.0,0.0),"temperature":TC}}
    right,left,_,_=ns._axis_fluxes_ns(coeff,basis,derivative,signs,weights,dx,dy,False,b["ymin"],b["ymax"],MU,K,pen,1,wg)
    # k=0 modal flux / 2 => face value; right[:,j] is face at top of cell j (y=(j+1)h)
    f_top=[float(right[0,j,0,3])/2.0 for j in range(ny)]   # face at y=(j+1)h
    f_bot=[float(left[0,j,0,3])/2.0 for j in range(ny)]    # face at y=j*h
    return {"f_interior":f_top, "wall_flux_face_est":float(left[0,0,0,3])/2.0,
            "wall_flux_top":float(right[0,ny-1,0,3])/2.0,"x":float(right[0,0,0,3])}

def extrap(vals, ys, y0, deg=2):
    # Lagrange extrapolation of vals at nodes ys to y0
    s=0.0
    for i,(yi,vi) in enumerate(zip(ys,vals)):
        w=1.0
        for j,yj in enumerate(ys):
            if i!=j: w*= (y0-yj)/(yi-yj)
        s+=vi*w
    return s

out={}
for deg in (1,2):
    for wg in ("taylor","face"):
        rows=[]
        for ny in (8,16,32,64):
            m=measure(ny,deg,wg)
            h=1.0/ny
            # interior faces between cells: y = h, 2h ... use first three interior faces (away from wall)
            ys=[ (j+1)*h for j in range(3)]
            lin=extrap(m["f_interior"][:2],ys[:2],0.0)
            quad=extrap(m["f_interior"][:3],ys[:3],0.0)
            rows.append(dict(ny=ny,h=h,wall=m["wall_flux_face_est"],lin=lin,quad=quad,
                             interior_first=m["f_interior"][0], top=m["wall_flux_top"]))
        def order(key):
            xs=[math.log(r["h"]) for r in rows]; ys=[math.log(abs(r[key]-K/2)) for r in rows]
            n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
            return sum((a-mx)*(b-my) for a,b in zip(xs,ys))/sum((a-mx)**2 for a in xs)
        out[f"p{deg}_{wg}"]={k:{"order":order(k),"err":{r["ny"]:r[k]-K/2 for r in rows}} for k in ("wall","lin","quad")}
        print(f"p{deg} {wg}: wall(order {order('wall'):.2f}, err64 {rows[-1]['wall']-K/2:+.3e}) "
              f"lin(order {order('lin'):.2f}, err64 {rows[-1]['lin']-K/2:+.3e})  "
              f"quad(order {order('quad'):.2f}, err64 {rows[-1]['quad']-K/2:+.3e})")
json.dump(out, open("scratch/m15258_proto_extrap.json","w"), indent=1)
