import math, torch
from tensordg.elastic_2d import (ElasticMedium, legendre_operators_2d, _rhs,
                                 modal_to_quadrature, quadrature_grid,
                                 flux_x, flux_y)

med = ElasticMedium()
for deg in (1,2,3):
    ops = legendre_operators_2d(deg)
    # (1) constant state -> zero RHS
    K = deg+1
    c = torch.zeros(4,4,K,K,5, dtype=torch.float64)
    c[...,2] = 1.7; c[...,3] = -0.3; c[...,0] = 0.4
    r = _rhs(c, ops, med, 0.25, 0.25, "periodic", "periodic")
    print(f"deg={deg} constant-state |rhs|max = {r.abs().max().item():.3e}")
