"""A reaction confined to half a slab fixes its absolute temperature under prescribed flux."""
import json
import math
from pathlib import Path
import subprocess

import torch

from tensordg.poisson_2d_bc import l2_error_2d, poisson_face_powers_2d, solve_poisson_2d_bc


def measure(n,degree):
    # Left of the interface u=1+x. Continuity of temperature and flux
    # fixes both integration constants for u''=4u on the right.
    def exact(x,y):
        z = 2*(x-.5)
        return torch.where(x<.5,1+x,1.5*torch.cosh(z)+.5*torch.sinh(z))

    east_derivative = 3*math.sinh(1)+math.cosh(1)
    bc = dict.fromkeys(("left","right","top","bottom"),"neumann")
    fluxes = dict(left=-1.,right=east_derivative)
    x,y,u,info = solve_poisson_2d_bc((n,3),degree,bc=bc,neumann_flux=fluxes,
                            forcing=lambda x,y:0*x+0*y,reaction=lambda x,y:4.*(x>.5),sparse=True)
    qx,qy = poisson_face_powers_2d(u,bc=bc,neumann_flux=fluxes)
    face_x = torch.arange(n+1,dtype=x.dtype)/n
    z = 2*(face_x-.5)
    axial = -torch.where(face_x<.5,1.,3*torch.sinh(z)+torch.cosh(z))
    loss = (4/n)*u[n//2:,:,0,0].sum()/len(y)
    return dict(n=n,degree=degree,anchored=info["temperature_anchored"],l2_error=l2_error_2d(x,y,u,exact)["absolute_l2"],
                section_power_error=float((qx.sum(dim=1)-axial).abs().max()),
                total_loss_error=float(abs(loss-(east_derivative-1))),
                transverse_power=float(qy.abs().max()))


if __name__ == "__main__":
    rows = []
    for degree in (1,2,3):
        previous = None
        for n in (4,8,16,32):
            row = measure(n,degree)
            order = None if previous is None else math.log2(previous/row["l2_error"])
            row.update(order=order,passed=row["anchored"] and (order is None or order>degree+.8)
                       and row["section_power_error"]<max(1e-9,.2*n**(-2*degree))
                       and row["total_loss_error"]<1e-9 and row["transverse_power"]<1e-9)
            rows.append(row)
            previous = row["l2_error"]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=Path(__file__).parent,text=True).strip()
    print(json.dumps(dict(commit=commit,runs=rows),indent=2,allow_nan=False))
    raise SystemExit(0 if all(row["passed"] for row in rows) else 1)
