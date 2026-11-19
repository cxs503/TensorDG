"""Generate WENO limiter verification figures for the report.

Produces:
1. WENO-Z weight scale invariance comparison (JS vs Z)
2. WENO limiter odd symmetry verification
3. WENO limiter cell average conservation
4. WENO limiter slope boundedness
5. Forced limited representation EOC (second-order ceiling)
"""
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.set_default_dtype(torch.float64)

from tensordg.limiters import apply_weno_limiter, apply_tvb_limiter, _weno_weights
import tensordg.euler_dg_2d as dg2
from benchmarks.weno_limiter_smooth_vortex import _project_vortex, _vortex_error, DOMAIN

ARTIFACTS_DIR = "/root/tp-agent/work/m24758/artifacts/weno_verify"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def fig_to_base64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    import base64
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_weno_z_invariance():
    """Figure 1: WENO-Z weight scale invariance vs JS degeneracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    torch.manual_seed(1)
    beta = (torch.rand(3, 32) + 0.05) * 1e-3
    scales = [1e-8, 1e-4, 1e-1, 1e4, 1e8]
    
    js_drift = []
    z_drift = []
    
    for scale in scales:
        z_weights = _weno_weights(beta[0], beta[1], beta[2], variant="z")
        z_scaled = _weno_weights(*(scale * b for b in beta), variant="z")
        z_diff = float((z_weights[0] - z_scaled[0]).abs().max())
        z_drift.append(z_diff)
        
        js_weights = _weno_weights(beta[0], beta[1], beta[2], variant="js")
        js_scaled = _weno_weights(*(scale * b for b in beta), variant="js")
        js_diff = float((js_weights[0] - js_scaled[0]).abs().max())
        js_drift.append(js_diff)
    
    axes[0].semilogx(scales, js_drift, 'o-', label='JS variant', linewidth=2)
    axes[0].semilogx(scales, z_drift, 's-', label='WENO-Z variant', linewidth=2)
    axes[0].set_xlabel('Scale factor c')
    axes[0].set_ylabel('Max weight drift |w(cβ) - w(β)|')
    axes[0].set_title('Weight Drift vs Indicator Rescaling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1e-3, color='r', linestyle='--', alpha=0.5, label='Drift threshold')
    
    # Show ideal weights for uniform data
    beta_uniform = torch.full((32,), 0.37)
    w_js = _weno_weights(beta_uniform, beta_uniform, beta_uniform, variant="js")
    w_z = _weno_weights(beta_uniform, beta_uniform, beta_uniform, variant="z")
    
    x_pos = np.arange(3)
    width = 0.35
    axes[1].bar(x_pos - width/2, [float(w_js[k].mean().item()) for k in range(3)], width, label='JS', alpha=0.8)
    axes[1].bar(x_pos + width/2, [float(w_z[k].mean().item()) for k in range(3)], width, label='WENO-Z', alpha=0.8)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(['Stencil 0', 'Stencil 1', 'Stencil 2'])
    axes[1].set_ylabel('Weight value')
    axes[1].set_title('Ideal Weights (Uniform Data)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle('WENO-Z Scale Invariance vs JS Degeneracy', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{ARTIFACTS_DIR}/weno_z_invariance.png', dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS_DIR}/weno_z_invariance.png")


def plot_odd_symmetry():
    """Figure 2: WENO limiter odd symmetry verification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    torch.manual_seed(42)
    ncells = 20
    coeff = torch.randn(ncells, 3, 3)
    dx = 0.05
    
    limited_pos = apply_weno_limiter(coeff, dx, tvb_constant=1.0)
    limited_neg = apply_weno_limiter(-coeff, dx, tvb_constant=1.0)
    neg_limited = -limited_pos
    
    # Plot mode-1 for variable 0
    axes[0].plot(coeff[:, 0, 0].numpy(), 'o-', label='Original avg', markersize=4)
    axes[0].plot(limited_pos[:, 1, 0].numpy(), 's-', label='Limited mode-1 (+)', markersize=4)
    axes[0].plot(limited_neg[:, 1, 0].numpy(), '^-', label='Limited mode-1 (-)', markersize=4)
    axes[0].set_xlabel('Cell index')
    axes[0].set_ylabel('Mode-1 coefficient')
    axes[0].set_title('Odd Symmetry: Mode-1 Coefficients')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot the difference
    diff = (limited_neg - neg_limited).abs()
    axes[1].plot(diff[:, 0, 0].numpy(), 'o-', label='|Δ| mode-0', markersize=4)
    axes[1].plot(diff[:, 1, 0].numpy(), 's-', label='|Δ| mode-1', markersize=4)
    axes[1].set_xlabel('Cell index')
    axes[1].set_ylabel('Absolute difference')
    axes[1].set_title('Symmetry Error (should be ~0)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    # Use linear scale since values are very small
    axes[1].set_yscale('symlog', linthresh=1e-15)
    
    fig.suptitle('WENO Limiter Odd Symmetry Verification', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{ARTIFACTS_DIR}/weno_odd_symmetry.png', dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS_DIR}/weno_odd_symmetry.png")


def plot_conservation():
    """Figure 3: Cell average conservation verification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ncells = 16
    coeff = torch.zeros(ncells, 3, 2)
    coeff[:, 0, 0] = torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0, 2.8, 2.6, 2.4, 
                                    2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8])
    coeff[:, 0, 1] = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.8, 0.75,
                                    0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
    coeff[:, 1, :] = 0.3 * coeff[:, 0, :]
    coeff[:, 2, :] = -0.1 * coeff[:, 0, :]
    
    averages_original = coeff[:, 0, :].clone()
    limited = apply_weno_limiter(coeff, 1.0/ncells, tvb_constant=0.0)
    
    # Plot cell averages
    x = np.arange(ncells)
    axes[0].plot(x, averages_original[:, 0].numpy(), 'o-', label='Original avg', markersize=6)
    axes[0].plot(x, limited[:, 0, 0].numpy(), 's-', label='Limited avg', markersize=6, alpha=0.7)
    axes[0].set_xlabel('Cell index')
    axes[0].set_ylabel('Cell average (density)')
    axes[0].set_title('Cell Average Conservation (Var 0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot mode-1 changes
    mode1_change = (limited[:, 1, 0] - coeff[:, 1, 0]).abs()
    axes[1].bar(x, mode1_change.numpy(), alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Cell index')
    axes[1].set_ylabel('|Δ mode-1|')
    axes[1].set_title('Mode-1 Modification (Limited Cells)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('WENO Limiter Conservation Verification', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{ARTIFACTS_DIR}/weno_conservation.png', dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS_DIR}/weno_conservation.png")


def plot_slope_boundedness():
    """Figure 4: WENO limiter slope boundedness verification."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ncells = 25
    dx = 0.04
    coeff = torch.zeros(ncells, 3, 1)
    avg = torch.cumsum(torch.rand(ncells), dim=0)
    avg[10:] += 2.0 * torch.rand(ncells - 10)
    coeff[:, 0, 0] = avg
    
    padded = torch.cat([avg[:1], avg, avg[-1:]], dim=0)
    d_left = (avg - padded[:-2])[:, None]
    d_right = (padded[2:] - avg)[:, None]
    bound = 0.5 * torch.maximum(d_left.abs(), d_right.abs())
    
    limited = apply_weno_limiter(coeff, dx, tvb_constant=0.0)
    
    # Plot boundedness
    x = np.arange(ncells)
    ax.plot(x, limited[:, 1, 0].numpy(), 'o-', label='|c_1| (limited)', markersize=6)
    ax.plot(x, bound[:, 0].numpy(), 'r--', label='Bound (max|d_left|,|d_right|)/2', linewidth=2)
    ax.fill_between(x, 0, bound[:, 0].numpy(), alpha=0.2, color='red', label='Allowed region')
    ax.set_xlabel('Cell index')
    ax.set_ylabel('Slope coefficient')
    ax.set_title('WENO Limiter Slope Boundedness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('WENO Limiter Slope Boundedness (TVD Property)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{ARTIFACTS_DIR}/weno_slope_boundedness.png', dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS_DIR}/weno_slope_boundedness.png")


def plot_forced_eoc():
    """Figure 5: Forced limited representation EOC (second-order ceiling)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    degrees = [1, 2, 3]
    ncells_list = [16, 32, 64]
    
    for degree in degrees:
        errors = []
        hs = []
        for n in ncells_list:
            coeff, nodes, basis, dx, dy = _project_vortex(degree, n, n)
            limited = dg2.apply_weno_limiter_2d(coeff, dx, dy, 0.0, (True, True))
            error = _vortex_error(limited, basis, nodes, DOMAIN, 0.0)
            errors.append(error['density'])
            hs.append(10.0 / n)
        
        eoc = float(np.polyfit(np.log(np.asarray(hs, float)), np.log(np.asarray(errors, float)), 1)[0])
        ax.loglog(hs, errors, 'o-', label=f'p={degree}, EOC={eoc:.2f}', linewidth=2, markersize=8)
        
        # Reference second-order line
        ref_h = np.logspace(-1, -0.5, 50)
        ref_e = errors[0] * (hs[0] / ref_h) ** 2
        ax.loglog(ref_h, ref_e, 'k--', alpha=0.3, label='O(h²) reference')
    
    ax.set_xlabel('h (mesh spacing)')
    ax.set_ylabel('L2 error (density)')
    ax.set_title('Forced WENO Limited Representation: EOC')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle('Forced WENO Limited Representation: Second-Order Ceiling', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{ARTIFACTS_DIR}/weno_forced_eoc.png', dpi=150)
    plt.close()
    print(f"Saved {ARTIFACTS_DIR}/weno_forced_eoc.png")


if __name__ == "__main__":
    plot_weno_z_invariance()
    plot_odd_symmetry()
    plot_conservation()
    plot_slope_boundedness()
    plot_forced_eoc()
    print("\nAll figures generated successfully.")
