"""Probe: does the wall-bounded *impulse-start* Couette reproduce, and is the
Dirichlet penalty sign the culprit?

m2589 reported that the impulse-start Couette of ``tensordg.ns_dg_2d`` blows up
at t ~ 0.0328 with a growth rate proportional to the SIPG penalty sigma, and
that only the temperature channel is responsible.  A growth rate proportional to
sigma is the fingerprint of an *anti*-dissipative boundary term, so this probe
sweeps the sign of the per-channel wall correction to see whether a sign
convention (rather than a deeper consistency error) explains it.

Nothing in the library is modified: the wall flux is monkey-patched per call.
"""
from __future__ import annotations

import torch

import tensordg.ns_dg_2d as ns
from tensordg.euler_dg_2d import GAMMA, _positivity_filter, _primitive, legendre_operators

MU, KAPPA, WALL_SPEED, WALL_T, PRESSURE = 0.1, 0.1 * 3.5 / 0.71, 1.0, 1.0, 1.0


def build_state(ncells=(2, 8), degree=1, rest=True, wall_temperature=WALL_T):
    nx, ny = ncells
    dev, dtype = torch.device("cpu"), torch.float64
    order = max(degree + 2, 3)
    nodes, weights, basis, derivative, signs = legendre_operators(
        degree, order, dev, dtype
    )
    dx, dy = 1.0 / nx, 1.0 / ny

    def initial(x, y):
        if rest:
            rho = torch.ones_like(y)
            u = torch.zeros_like(y)
            v = torch.zeros_like(y)
            return rho, u, v, torch.full_like(y, PRESSURE)
        u = WALL_SPEED * y
        temperature = WALL_T + MU * WALL_SPEED**2 / (2.0 * KAPPA) * y * (1.0 - y)
        return PRESSURE / temperature, u, torch.zeros_like(y), torch.full_like(y, PRESSURE)

    x = (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy
    coeff = ns._initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis, dev, dtype)
    boundary = {
        "xmin": None, "xmax": None,
        "ymin": {"velocity": (0.0, 0.0), "temperature": wall_temperature},
        "ymax": {"velocity": (WALL_SPEED, 0.0), "temperature": wall_temperature},
    }
    return dict(coeff=coeff, boundary=boundary, basis=basis, derivative=derivative,
                weights=weights, signs=signs, dx=dx, dy=dy, degree=degree)


def march(channel_weights=(1.0, 1.0, 1.0), penalty_scale=2.0, ncells=(2, 8),
          degree=1, cfl_viscous=1.0, max_steps=400, rest=True):
    d = build_state(ncells, degree, rest=rest)
    basis, derivative, weights, signs = d["basis"], d["derivative"], d["weights"], d["signs"]
    dx, dy, boundary = d["dx"], d["dy"], d["boundary"]
    h = min(dx, dy)
    penalty = penalty_scale * (degree + 1.0) ** 2
    wu, wv, wt = channel_weights
    original = ns._boundary_flux

    def channel_masked(side, wall, axis, h_, mu, kappa, pen, normal_scale, tangential_scale):
        inner_u, inner_v = side["u"], side["v"]
        wall_velocity = wall.get("velocity", (0.0, 0.0))
        face_u = torch.as_tensor(wall_velocity[0], device=inner_u.device,
                                dtype=inner_u.dtype).expand_as(inner_u)
        face_v = torch.as_tensor(wall_velocity[1], device=inner_v.device,
                                dtype=inner_v.dtype).expand_as(inner_v)
        wall_temperature = wall.get("temperature", None)
        face_T = (side["temperature"] if wall_temperature is None else
                  torch.as_tensor(wall_temperature, device=inner_u.device,
                                  dtype=inner_u.dtype).expand_as(inner_u))
        normal_u = side["u_n"] * normal_scale - wu * (pen / h_) * (inner_u - face_u)
        normal_v = side["v_n"] * normal_scale - wv * (pen / h_) * (inner_v - face_v)
        normal_T = side["T_n"] * normal_scale - wt * (pen / h_) * (
            side["temperature"] - face_T)
        zeros = torch.zeros_like(inner_u)
        pressure = side["pressure"]
        inviscid = (torch.stack([zeros, pressure, zeros, zeros], dim=-1) if axis == 0
                    else torch.stack([zeros, zeros, pressure, zeros], dim=-1))
        face = {"u": face_u, "v": face_v}
        tu, tv, tT = (side["u_t"] * tangential_scale, side["v_t"] * tangential_scale,
                      side["T_t"] * tangential_scale)
        if axis == 0:
            face.update(u_x=normal_u, v_x=normal_v, T_x=normal_T, u_y=tu, v_y=tv, T_y=tT)
        else:
            face.update(u_y=normal_u, v_y=normal_v, T_y=normal_T, u_x=tu, v_x=tv, T_x=tT)
        return inviscid - ns._viscous_flux(face, axis, mu, kappa)

    ns._boundary_flux = channel_masked
    try:
        coeff, time = d["coeff"].clone(), 0.0
        for step in range(max_steps):
            at = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
            _, ux, uy, pressure = _primitive(at)
            rho = at[..., 0].clamp_min(1e-12)
            sound = torch.sqrt(GAMMA * pressure.clamp_min(1e-12) / rho)
            rate = (float(torch.max(ux.abs() + sound))
                    + float(torch.max(uy.abs() + sound))) / h
            dt = 0.3 / (rate * (2 * degree + 1))
            diffusivity = max((4.0 / 3.0) * MU / float(rho.min()),
                              KAPPA / (float(rho.min()) * ns.CV))
            dt = min(dt, cfl_viscous * 0.1 * h * h / (diffusivity * (degree + 1.0) ** 4))
            k1 = ns._rhs_ns(coeff, basis, derivative, weights, signs, dx, dy,
                            (True, False), boundary, MU, KAPPA, penalty)
            u1 = _positivity_filter(coeff + dt * k1, basis, signs)
            k2 = ns._rhs_ns(u1, basis, derivative, weights, signs, dx, dy,
                            (True, False), boundary, MU, KAPPA, penalty)
            u2 = _positivity_filter(0.75 * coeff + 0.25 * (u1 + dt * k2), basis, signs)
            k3 = ns._rhs_ns(u2, basis, derivative, weights, signs, dx, dy,
                            (True, False), boundary, MU, KAPPA, penalty)
            coeff = _positivity_filter(coeff / 3.0 + (2.0 / 3.0) * (u2 + dt * k3),
                                       basis, signs)
            time += dt
            if (not bool(torch.isfinite(coeff).all())) or float(coeff.abs().max()) > 1e6:
                return dict(ok=False, time=time, steps=step + 1)
        return dict(ok=True, time=time, steps=max_steps)
    finally:
        ns._boundary_flux = original


if __name__ == "__main__":
    print("-- rest start, channel/sign sweep (2x8 P1, sigma=2) --")
    for label, w in (("+1 +1 +1 (shipped)", (1, 1, 1)), ("0 0 0 (no penalty)", (0, 0, 0)),
                     ("1 1 -1 (T flipped)", (1, 1, -1)), ("-1 -1 -1", (-1, -1, -1)),
                     ("1 1 0 (T off)", (1, 1, 0))):
        r = march(channel_weights=tuple(float(x) for x in w))
        print(f"{label:22s} ok={r['ok']} t={r['time']:.5f} steps={r['steps']}", flush=True)
    print("-- steady start (sanity: shipped sign, should be quiet) --")
    r = march(rest=False, max_steps=200)
    print("steady start", r, flush=True)