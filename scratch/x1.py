import torch, math
from tensordg.swe_1d import solve_swe_1d, _operators
from tensordg.swe_2d import solve_swe_2d, legendre_operators_2d


def h2d(x, y):
    return torch.where(x < 0.5, torch.ones_like(x), 0.2 * torch.ones_like(x))


def zero(x, y):
    return torch.zeros_like(x)


def flat(x, y):
    return torch.zeros_like(x)


for deg in (1, 2):
    for n in (32, 64):
        x2, y2, c2, zbf, meta2 = solve_swe_2d(h2d, zero, zero, flat, nx=n, ny=8,
                                               degree=deg, final_time=0.1, cfl=0.15,
                                               periodic=(False, False))
        nodes, w, phi, dphi, pl, pr = legendre_operators_2d(deg, c2.device, c2.dtype)
        vals = torch.einsum("xyklv,ak,bl->xyabv", c2, phi, phi)
        h2 = vals[..., 0]
        spread = float((h2.max(dim=3).values - h2.min(dim=3).values).abs().max())
        x1, c1, _zf1, meta1 = solve_swe_1d(lambda x: h2d(x, None), lambda x: zero(x, None),
                                     lambda x: flat(x, None), ncells=n, degree=deg,
                                     final_time=0.1, domain=(0.0, 1.0), cfl=0.15,
                                     periodic=False)
        nodes1, w1, phi1, dphi1, pl1, pr1 = _operators(deg, c1.device, c1.dtype)
        v1 = torch.einsum("cmv,qm->cqv", c1, phi1)
        h1q = v1[..., 0]
        h2q = h2[:, 0, :, 0]
        d = float((h2q - h1q).abs().max())
        drel = d / float(h1q.abs().max())
        print(f"deg={deg} n={n} steps2d={meta2['steps']} steps1d={meta1['steps']} "
              f"y-spread={spread:.3e} max|h2d-h1d|={d:.3e} rel={drel:.3e}")