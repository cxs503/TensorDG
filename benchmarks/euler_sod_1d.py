"""1-D Euler Sod shock tube, DG P0 (finite-volume limit), PyTorch implementation."""
from __future__ import annotations

import argparse

import torch

from tensordg.euler_1d import (
    GAMMA, physical_flux, primitive_to_conservative, rusanov_rhs,
    solve_riemann, sound_speed,
)

# Backward-compatible function name used by earlier benchmarks.
rhs = rusanov_rhs


def solve_sod(ncells=400, final_time=0.2, cfl=0.4, device="cpu", dtype=torch.float64):
    return solve_riemann(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=ncells,
        final_time=final_time, diaphragm=0.5, domain=(0.0, 1.0),
        cfl=cfl, flux="rusanov", device=device, dtype=dtype,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=400)
    parser.add_argument("--time", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="sod_results.csv")
    args = parser.parse_args()
    x, fields, meta = solve_sod(args.cells, args.time, device=args.device)
    import csv
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "density", "velocity", "pressure"])
        for i in range(x.numel()):
            writer.writerow([
                x[i].item(), fields["density"][i].item(),
                fields["velocity"][i].item(), fields["pressure"][i].item(),
            ])
    print("1D Euler Sod shock tube | PyTorch DG P0 / finite-volume limit")
    print(f"cells={meta['ncells']} time={meta['final_time']} steps={meta['steps']} output={args.output}")
    print(f"density range: {fields['density'].min().item():.6g} .. {fields['density'].max().item():.6g}")
    print(f"pressure range: {fields['pressure'].min().item():.6g} .. {fields['pressure'].max().item():.6g}")


if __name__ == "__main__":
    main()
