"""MCMP cross-force operator validation benchmark.

Generates:
1. Symbolic verification results
2. Spectral analysis figures
3. Phase separation visualization
4. Laplace law measurement
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import sys
sys.path.insert(0, 'TensorLBM/src')

from tensorlbm import (
    symbolic_cross_force_verification,
    spectral_analysis_cross_force,
    setup_laplace_test,
    measure_laplace_pressure,
    MCMPSolver,
    pseudopotential_shan_chen,
)


def plot_spectral_analysis(spec_data, save_path='TensorLBM/outputs/spectral_analysis.png'):
    """Plot structure factor vs wave number."""
    k = np.array(spec_data['spectral_data']['k'])
    S_real = np.array(spec_data['spectral_data']['S_real'])
    S_imag = np.array(spec_data['spectral_data']['S_imag'])
    S_mag = np.array(spec_data['spectral_data']['S_mag'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Real part
    axes[0].plot(k, S_real, 'b-', linewidth=2)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Wave number k')
    axes[0].set_ylabel('Re[S(k)]')
    axes[0].set_title('Structure Factor (Real)')
    axes[0].grid(True, alpha=0.3)

    # Imaginary part
    axes[1].plot(k, S_imag, 'r-', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Wave number k')
    axes[1].set_ylabel('Im[S(k)]')
    axes[1].set_title('Structure Factor (Imaginary)')
    axes[1].grid(True, alpha=0.3)

    # Magnitude
    axes[2].plot(k, S_mag, 'g-', linewidth=2)
    axes[2].axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='S(0)=1')
    axes[2].set_xlabel('Wave number k')
    axes[2].set_ylabel('|S(k)|')
    axes[2].set_title('Structure Factor (Magnitude)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_phase_separation(solver, save_path='TensorLBM/outputs/phase_separation.png'):
    """Plot density fields showing phase separation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Component 0 density
    im0 = axes[0].imshow(solver.rho[0].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[0].set_title('Component 0 Density')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0])

    # Component 1 density
    im1 = axes[1].imshow(solver.rho[1].cpu().numpy(), cmap='viridis', aspect='auto')
    axes[1].set_title('Component 1 Density')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1])

    # Force magnitude
    force_mag = torch.sqrt(
        solver.cross_force.force[0, 0]**2 + solver.cross_force.force[0, 1]**2
    ).cpu().numpy()
    im2 = axes[2].imshow(force_mag, cmap='hot', aspect='auto')
    axes[2].set_title('Cross-Force Magnitude (Component 0)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_laplace_convergence(results, save_path='TensorLBM/outputs/laplace_convergence.png'):
    """Plot Laplace law error vs droplet radius."""
    radii = [r['R'] for r in results]
    errors = [r['relative_error'] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(radii, errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='15% threshold')
    ax.set_xlabel('Droplet Radius R')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Laplace Law Validation: Error vs Droplet Radius')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    import os
    os.makedirs('TensorLBM/outputs', exist_ok=True)

    # 1. Symbolic verification
    print("=" * 60)
    print("1. Symbolic Verification")
    print("=" * 60)
    sym_result = symbolic_cross_force_verification()
    print(f"Status: {sym_result['status']}")
    print(f"Checks: {sym_result['n_passed']}/{sym_result['n_checks']}")
    for name, check in sym_result['checks'].items():
        status = "PASS" if check.get('pass') else "FAIL"
        print(f"  [{status}] {name}: {check.get('value', 'N/A')}")

    # 2. Spectral analysis
    print("\n" + "=" * 60)
    print("2. Spectral Analysis")
    print("=" * 60)
    spec_result = spectral_analysis_cross_force(G=-1.5, rho0=1.0, grid_size=64)
    print(f"Stability: {spec_result['stability']}")
    print(f"High-freq damping: {spec_result['high_freq_damping_ratio']:.4f}")
    print(f"Isotropy error: {spec_result['isotropy_error_at_pi4']:.6f}")
    plot_spectral_analysis(spec_result)

    # 3. Phase separation
    print("\n" + "=" * 60)
    print("3. Phase Separation Test")
    print("=" * 60)
    torch.manual_seed(42)
    solver = setup_laplace_test(nx=64, ny=64, R=15.0)
    solver.run(200)
    solver._compute_macroscopic()
    solver.cross_force.compute_force(solver.rho)

    rho0_in = solver.rho[0][solver.rho[0] > 1.0].mean().item()
    rho0_out = solver.rho[0][solver.rho[0] < 0.8].mean().item()
    print(f"Component 0 density (high phase): {rho0_in:.4f}")
    print(f"Component 0 density (low phase): {rho0_out:.4f}")
    print(f"Phase separation ratio: {rho0_in/rho0_out:.2f}")
    plot_phase_separation(solver)

    # 4. Laplace law measurement
    print("\n" + "=" * 60)
    print("4. Laplace Law Measurement")
    print("=" * 60)
    laplace_results = []
    for R in [12, 15, 20, 25]:
        torch.manual_seed(42)
        s = setup_laplace_test(nx=64, ny=64, R=float(R))
        res = measure_laplace_pressure(s, R=float(R), n_equilibrate=200, n_sample=100)
        res['R'] = R
        laplace_results.append(res)
        print(f"R={R}: dp_measured={res['dp_measured']:.6f}, dp_theoretical={res['dp_theoretical']:.6f}, error={res['relative_error']*100:.1f}%")

    plot_laplace_convergence(laplace_results)

    # 5. Mass conservation
    print("\n" + "=" * 60)
    print("5. Mass Conservation")
    print("=" * 60)
    torch.manual_seed(42)
    solver2 = setup_laplace_test(nx=64, ny=64, R=15.0)
    mass_result = solver2.verify_mass_conservation(n_steps=100)
    print(f"Mass drift: {mass_result['mass_drift_relative']:.2e}")
    print(f"Max step drift: {mass_result['max_step_drift']:.2e}")
    print(f"Pass: {mass_result['pass']}")

    # Save summary
    summary = {
        "symbolic_verification": sym_result,
        "spectral_analysis": {
            "stability": spec_result['stability'],
            "high_freq_damping": spec_result['high_freq_damping_ratio'],
            "isotropy_error": spec_result['isotropy_error_at_pi4'],
        },
        "phase_separation": {
            "rho_high": rho0_in,
            "rho_low": rho0_out,
            "ratio": rho0_in / rho0_out,
        },
        "laplace_results": laplace_results,
        "mass_conservation": mass_result,
    }

    with open('TensorLBM/outputs/mcmp_xval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved: TensorLBM/outputs/mcmp_xval_summary.json")


if __name__ == '__main__':
    main()
