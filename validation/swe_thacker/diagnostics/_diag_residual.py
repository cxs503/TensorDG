"""Scratch diagnostic 5: semi-discrete residual consistency (manufactured solution).

Projects the analytic solution onto the DG space, evaluates the semi-discrete
operator `_rhs` and compares it with the time derivative of the analytic solution
(the latter projected consistently).  A consistent operator must match to
truncation error.
"""
import math
import torch

from tensordg.swe_1d import _operators, _rhs, GRAVITY

g, h0, L, A = GRAVITY, 1.0, 1.0, 0.5
omega = math.sqrt(2 * g * h0) / L


def exact_fields(x, t):
    s = (A / L) * math.cos(omega * t)
    u = -(A * omega * L / (2 * h0)) * math.sin(omega * t)
    eta = h0 + (A * A / (4 * h0)) * math.sin(omega * t) ** 2 + s * x
    zb = -h0 * (1 - (x / L) ** 2)
    h = eta - zb
    return h, h * u, eta, zb


def ddt_fields(x, t):
    """Analytic ∂_t [h, hu]."""
    s = (A / L) * math.cos(omega * t)
    ds = -(A / L) * omega * math.sin(omega * t)
    dds = -(A / L) * omega ** 2 * math.cos(omega * t)
    u = -(s * 0 + ds * L ** 2 / (2 * h0))
    du = dds * L ** 2 / (2 * h0)
    eta_t = (A * A / (4 * h0)) * omega * math.sin(2 * omega * t) + ds * x
    h_t = eta_t  # z_b time independent
    hu_t = h_t * u + 0.0  # h*u_t + h_t*u
    h, _, _, _ = exact_fields(x, t)
    return h_t, h_t * u + h * du


def check(ncells, degree, t0=0.0, cfl_note=""):
    dev, dtype = torch.device("cpu"), torch.float64
    lower, upper = -L, L
    dx = (upper - lower) / ncells
    x = lower + (torch.arange(ncells, dtype=dtype) + 0.5) * dx
    nodes, weights, phi, dphi, phi_left, phi_right = _operators(degree, dev, dtype)
    node_x = x[:, None] + 0.5 * dx * nodes[None, :]

    h, hu, _, _ = exact_fields(node_x, t0)
    ht, hut = ddt_fields(node_x, t0)
    coeff = torch.zeros((ncells, degree + 1, 2), dtype=dtype)
    for k in range(degree + 1):
        f = (2 * k + 1) / 2.0
        coeff[:, k, 0] = f * torch.einsum("cq,q,q->c", h, weights, phi[:, k])
        coeff[:, k, 1] = f * torch.einsum("cq,q,q->c", hu, weights, phi[:, k])
    ref = torch.zeros_like(coeff)
    for k in range(degree + 1):
        f = (2 * k + 1) / 2.0
        ref[:, k, 0] = f * torch.einsum("cq,q,q->c", ht, weights, phi[:, k])
        ref[:, k, 1] = f * torch.einsum("cq,q,q->c", hut, weights, phi[:, k])

    zb_l = -h0 * (1 - ((lower + torch.arange(ncells, dtype=dtype) * dx) / L) ** 2)
    zb_r = -h0 * (1 - (((lower + torch.arange(ncells, dtype=dtype) * dx) + dx) / L) ** 2)
    zb_jump = zb_r - zb_l
    for periodic in (False, True):
        res = _rhs(coeff, dx, weights, phi, dphi, phi_left, phi_right, zb_jump,
                   "hll", periodic)
        # interior cells only (drop 2 at each end) to isolate interior consistency
        sl = slice(2, ncells - 2)
        err = (res[sl] - ref[sl])
        scale = ref[sl].abs().max()
        print(f"  P{degree} N={ncells:4d} t0={t0} periodic={periodic} "
              f"rel_residual={float(err.abs().max()/scale):.3e}", flush=True)


if __name__ == "__main__":
    print("semi-discrete residual check (analytic solution, hll, transmissive/periodic BC)")
    for n in (20, 40, 80):
        for deg in (1, 2):
            check(n, deg, 0.0)
    for n in (20, 40, 80):
        for deg in (1, 2):
            check(n, deg, 0.3)