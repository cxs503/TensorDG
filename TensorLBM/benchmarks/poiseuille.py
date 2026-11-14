"""Poiseuille flow benchmark: analytical validation of mass-exchange schemes.

2D channel flow with constant body force, periodic in x, no-slip at walls.
Analytical solution: u(y) = (F / (2 * nu)) * y * (H - y)

This benchmark validates:
1. Mass conservation of each scheme
2. Velocity profile accuracy vs analytical solution
"""

import torch
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorlbm import D2Q9, MassExchangeSelector, Scheme
from tensorlbm.streaming import stream_periodic


def poiseuille_analytical(y, H, F, nu):
    """Analytical Poiseuille velocity profile."""
    return (F / (2 * nu)) * y * (H - y)


def run_poiseuille(
    scheme: Scheme,
    Nx: int = 64,
    Ny: int = 32,
    n_steps: int = 2000,
    F: float = 1e-5,
    tau: float = 0.8,
):
    """Run Poiseuille flow benchmark."""
    Q = D2Q9.Q
    omega = 1.0 / tau
    nu = D2Q9.cs2.item() * (tau - 0.5)
    H = Ny - 1

    # Initialize
    rho = torch.ones(Ny, Nx, dtype=torch.float64)
    ux = torch.zeros(Ny, Nx, dtype=torch.float64)
    uy = torch.zeros(Ny, Nx, dtype=torch.float64)
    f = D2Q9.equilibrium(rho, ux, uy)

    selector = MassExchangeSelector(scheme=scheme, omega=omega)
    fx = F

    # Run simulation
    for step in range(n_steps):
        rho, ux, uy = D2Q9.macroscopic(f)
        f_post = selector.collide(f, rho, ux, uy)

        # Apply body force
        for i in range(Q):
            f_post[i] += D2Q9.w[i] * 3.0 * D2Q9.e[i, 0].item() * fx

        # Streaming with bounce-back
        f = stream_with_bounceback(f_post, Ny, Nx)

    # Extract velocity profile
    mid_x = Nx // 2
    rho_final, ux_final, uy_final = D2Q9.macroscopic(f)
    ux_profile = ux_final[:, mid_x]

    # Analytical solution
    y_coords = torch.arange(Ny, dtype=torch.float64)
    u_analytical = poiseuille_analytical(y_coords, H, F, nu)

    # Compute error (exclude boundaries)
    interior = slice(1, -1)
    l2_error = torch.sqrt(torch.mean((ux_profile[interior] - u_analytical[interior]) ** 2)).item()
    u_max_analytical = F * H ** 2 / (8 * nu)
    rel_l2_error = l2_error / u_max_analytical if u_max_analytical > 0 else l2_error

    # Mass conservation check
    f_init = D2Q9.equilibrium(torch.ones(Ny, Nx), torch.zeros(Ny, Nx), torch.zeros(Ny, Nx))
    mass_conservation = selector.verify_mass_conservation(f_init, n_steps=50)

    return {
        "scheme": scheme.value,
        "Nx": Nx,
        "Ny": Ny,
        "n_steps": n_steps,
        "tau": tau,
        "nu": nu,
        "F": F,
        "H": H,
        "u_max_analytical": u_max_analytical,
        "l2_error": l2_error,
        "rel_l2_error": rel_l2_error,
        "mass_conservation": mass_conservation,
        "pass": rel_l2_error < 0.03,
    }


def stream_with_bounceback(f, Ny, Nx):
    """Stream with bounce-back at y=0 and y=Ny-1, periodic in x."""
    Q, _, _ = f.shape
    f_out = torch.empty_like(f)

    for i in range(Q):
        ex = int(D2Q9.e[i, 0])
        ey = int(D2Q9.e[i, 1])

        if ey == 0:
            f_out[i] = torch.roll(f[i], shifts=ex, dims=1)
        else:
            f_out[i] = torch.roll(f[i], shifts=(ey, ex), dims=(0, 1))

    # Bounce-back at walls
    for i in range(Q):
        if D2Q9.e[i, 1] < 0:
            f_out[i, 0, :] = f[D2Q9.opp[i], 0, :]
        if D2Q9.e[i, 1] > 0:
            f_out[i, Ny-1, :] = f[D2Q9.opp[i], Ny-1, :]

    return f_out


def main():
    """Run Poiseuille benchmark for all schemes."""
    results = []

    for scheme in [Scheme.BGK, Scheme.MRT, Scheme.TRT]:
        print(f"\nRunning Poiseuille benchmark with {scheme.value}...")
        result = run_poiseuille(scheme, Nx=64, Ny=32, n_steps=2000)
        results.append(result)
        print(f"  Scheme: {result['scheme']}")
        print(f"  Rel L2 error: {result['rel_l2_error']:.6f}")
        print(f"  Mass conservation pass: {result['mass_conservation']['pass']}")
        print(f"  Benchmark pass: {result['pass']}")

    # Save results
    output_dir = Path(__file__).parent.parent / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "poiseuille_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'poiseuille_results.json'}")

    # Summary
    print("\n" + "=" * 60)
    print("Poiseuille Flow Benchmark Summary")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  {r['scheme']:4s}: rel_L2={r['rel_l2_error']:.4%} [{status}]")


if __name__ == "__main__":
    main()
