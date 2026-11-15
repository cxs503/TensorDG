"""Probe: how large is the truncation error of the min-image periodic reference?

The published long-time errors of CASE-DG-EULER-VORTEX-ERR are measured against
``vortex_primitive(..., period=10)``, which keeps only the *nearest* image of the
free-space vortex.  The exact solution of the periodic problem is the full image
sum, so the min-image reference carries a truncation error.  Question: is that
error below the measured discretisation error (then the reference is certified),
or does it contaminate the finest checkpoints?
"""
import numpy as np
import torch

from tensordg.euler_2d import GAMMA, vortex_primitive

L = 10.0
STRENGTH = 5.0
BG = (1.0, 1.0)


def image_sum(x, y, time, images, strength=STRENGTH, background=BG):
    """Periodic solution as the image sum of the free-space vortex (perturbations)."""
    rho = torch.ones_like(x)
    u = torch.full_like(x, background[0])
    v = torch.full_like(x, background[1])
    p = torch.ones_like(x)
    for m in range(-images, images + 1):
        for n in range(-images, images + 1):
            r, uu, vv, pp = vortex_primitive(
                x - m * L, y - n * L, time, strength, background)
            rho = rho + (r - 1.0)
            u = u + (uu - background[0])
            v = v + (vv - background[1])
            p = p + (pp - 1.0)
    return rho, u, v, p


def rel_l2(a, b):
    return float(torch.sqrt(torch.mean((a - b) ** 2)) / torch.sqrt(torch.mean(b ** 2)))


def main():
    n = 400
    xs = torch.linspace(-5.0, 5.0, n, dtype=torch.float64)
    x, y = torch.meshgrid(xs, xs, indexing="xy")
    print(f"{'t':>5} {'K':>3} {'rho':>11} {'u':>11} {'v':>11} {'p':>11}")
    for t in (1.0, 2.0, 5.0, 10.0, 20.0):
        near = vortex_primitive(x, y, t, STRENGTH, BG, period=L)
        for images in (1, 2, 3, 4):
            full = image_sum(x, y, t, images)
            vals = [rel_l2(near[i], full[i]) for i in range(4)]
            print(f"{t:5.1f} {images:3d} " + " ".join(f"{v:11.3e}" for v in vals))
        ref = image_sum(x, y, t, 4)
        # periodic-invariance identity: displacement L is a lattice vector
        ref0 = image_sum(x, y, 0.0, 4)
        print(f"      full-image(t) vs full-image(0): "
              + " ".join(f"{rel_l2(ref[i], ref0[i]):11.3e}" for i in range(4)))
        near0 = vortex_primitive(x, y, 0.0, STRENGTH, BG, period=L)
        print(f"      min-image(t) vs min-image(0):  "
              + " ".join(f"{rel_l2(near[i], near0[i]):11.3e}" for i in range(4)))


if __name__ == "__main__":
    main()