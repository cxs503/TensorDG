import math, torch
from tensordg.elastic_2d import (ElasticMedium, legendre_operators_2d, _rhs,
                                 plane_wave, solve_elastic_2d, relative_l2_error)

med = ElasticMedium()
for deg in (1,2,3):
    ops = legendre_operators_2d(deg)
    K = deg+1
    c = torch.zeros(4,4,K,K,5, dtype=torch.float64)
    c[...,0,0,2] = 1.7; c[...,0,0,3] = -0.3; c[...,0,0,0] = 0.4
    r = _rhs(c, ops, med, 0.25, 0.25, "periodic", "periodic")
    # smooth state test: U = sin(2 pi x) * const  -> compare rhs to analytic
    print(f"deg={deg} constant-state |rhs|max = {r.abs().max().item():.3e}")

# operator test: exact derivative of a polynomial field through dmat
ops = legendre_operators_2d(3)
print("dmat(deg3)@coeff(P_3) expected P_3' = 1.5+7.5xi^2?")
c = torch.zeros(4,1); c[3]=1.0
print("got", (ops['dmat']@c).flatten().tolist())
