from __future__ import annotations
"""1-D Euler Sod shock tube, DG P0 (finite-volume limit), PyTorch implementation."""
import argparse
import math
import torch

GAMMA = 1.4


def primitive_to_conservative(rho, velocity, pressure):
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * velocity**2
    return torch.stack((rho, rho * velocity, energy), dim=-1)


def physical_flux(u):
    rho = u[..., 0].clamp_min(1e-12)
    velocity = u[..., 1] / rho
    pressure = (GAMMA - 1.0) * (u[..., 2] - 0.5 * u[..., 1]**2 / rho)
    return torch.stack((u[..., 1], u[..., 1] * velocity + pressure,
                        velocity * (u[..., 2] + pressure)), dim=-1)


def sound_speed(u):
    rho = u[..., 0].clamp_min(1e-12)
    velocity = u[..., 1] / rho
    pressure = (GAMMA - 1.0) * (u[..., 2] - 0.5 * u[..., 1]**2 / rho)
    return torch.sqrt((GAMMA * pressure.clamp_min(1e-12) / rho).clamp_min(0.0))


def rhs(u, dx):
    # Transmissive boundary conditions (zero-gradient ghost states).
    left = torch.cat((u[:1], u[:-1]), dim=0)
    right = torch.cat((u[1:], u[-1:]), dim=0)
    f_left, f_right = physical_flux(left), physical_flux(right)
    alpha = torch.maximum(left[..., 1] / left[..., 0].clamp_min(1e-12).abs() + sound_speed(left),
                          right[..., 1] / right[..., 0].clamp_min(1e-12).abs() + sound_speed(right))
    # Rusanov flux at each cell's left and right interfaces.
    flux_l = 0.5 * (f_left + physical_flux(u)) - 0.5 * alpha[:, None] * (u - left)
    flux_r = 0.5 * (physical_flux(u) + f_right) - 0.5 * alpha[:, None] * (right - u)
    # Boundary fluxes use the same physical state on both sides.
    flux_l[0] = physical_flux(u[:1])[0]
    flux_r[-1] = physical_flux(u[-1:])[0]
    return -(flux_r - flux_l) / dx


def solve_sod(ncells=400, final_time=0.2, cfl=0.4, device="cpu", dtype=torch.float64):
    if ncells < 20 or final_time < 0 or not 0 < cfl <= 0.5:
        raise ValueError("ncells >= 20, final_time >= 0, and 0 < cfl <= 0.5 are required")
    dev = torch.device(device)
    dx = 1.0 / ncells
    x = (torch.arange(ncells, device=dev, dtype=dtype) + 0.5) * dx
    rho = torch.where(x < 0.5, 1.0, 0.125)
    vel = torch.zeros_like(x)
    pressure = torch.where(x < 0.5, 1.0, 0.1)
    u = primitive_to_conservative(rho, vel, pressure)
    t, steps = 0.0, 0
    while t < final_time:
        speed = (u[:, 1] / u[:, 0].clamp_min(1e-12).abs() + sound_speed(u)).max().item()
        dt = min(cfl * dx / max(speed, 1e-12), final_time - t)
        # SSP-RK3; P0 DG is the conservative first-order finite-volume limit.
        u1 = u + dt * rhs(u, dx)
        u2 = 0.75 * u + 0.25 * (u1 + dt * rhs(u1, dx))
        u = (1.0 / 3.0) * u + (2.0 / 3.0) * (u2 + dt * rhs(u2, dx))
        # Fail fast rather than silently masking an inadmissible conservative state.
        rho_now = u[:, 0]
        p_now = (GAMMA - 1.0) * (u[:, 2] - 0.5 * u[:, 1]**2 / rho_now.clamp_min(1e-12))
        if not torch.isfinite(u).all() or (rho_now <= 0).any() or (p_now <= 0).any():
            raise FloatingPointError("Nonphysical density/pressure encountered; reduce CFL or refine mesh")
        t += dt
        steps += 1
    rho = u[:, 0]
    velocity = u[:, 1] / rho
    pressure = (GAMMA - 1.0) * (u[:, 2] - 0.5 * u[:, 1]**2 / rho)
    return x, {"density": rho, "velocity": velocity, "pressure": pressure,
               "conservative": u}, {"ncells": ncells, "final_time": final_time,
               "steps": steps, "device": str(dev), "cfl": cfl}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", type=int, default=400)
    p.add_argument("--time", type=float, default=0.2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="sod_results.csv")
    args = p.parse_args()
    x, fields, meta = solve_sod(args.cells, args.time, device=args.device)
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "density", "velocity", "pressure"])
        for i in range(x.numel()):
            writer.writerow([x[i].item(), fields["density"][i].item(),
                             fields["velocity"][i].item(), fields["pressure"][i].item()])
    print("1D Euler Sod shock tube | PyTorch DG P0 / finite-volume limit")
    print(f"cells={meta['ncells']} time={meta['final_time']} steps={meta['steps']} output={args.output}")
    print(f"density range: {fields['density'].min().item():.6g} .. {fields['density'].max().item():.6g}")
    print(f"pressure range: {fields['pressure'].min().item():.6g} .. {fields['pressure'].max().item():.6g}")

if __name__ == "__main__":
    main()
