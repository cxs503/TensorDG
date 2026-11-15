"""Generate certification figure for CKS1998 vortex 2-D order certification."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CERT = REPO / "validation" / "cks1998_vortex_2d" / "certification.json"
FIG = REPO / "docs" / "figures" / "cks1998_vortex_certification.png"

def main():
    payload = json.loads(CERT.read_text())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: error decay with design order reference
    ax = axes[0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    
    for idx, name in enumerate(sorted(payload["degrees"].keys())):
        bundle = payload["degrees"][name]
        cert = bundle["certification"]
        levels = cert["levels"]
        hs = [l["h"] for l in levels]
        errs = [l["error"] for l in levels]
        
        ax.loglog(hs, errs, marker=markers[idx], color=colors[idx],
                  label=f"P{name[1:]} (measured)", linewidth=2, markersize=8)
        
        # Design order reference line through finest point
        degree = int(name[1:])
        h_ref = hs[-1]
        err_ref = errs[-1]
        p_plus_1 = degree + 1
        h_line = np.logspace(np.log10(hs[0]), np.log10(hs[-1]), 50)
        err_line = err_ref * (h_line / h_ref) ** p_plus_1
        ax.loglog(h_line, err_line, "--", color=colors[idx], alpha=0.4,
                  label=f"P{name[1:]} (O(h^{p_plus_1}))")
    
    ax.set_xlabel("h (cell size)", fontsize=12)
    ax.set_ylabel("L1 relative error (density)", fontsize=12)
    ax.set_title("CKS1998 §5.3 — 2D isentropic vortex: error decay", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    
    # Right: certification verdict summary
    ax = axes[1]
    names = sorted(payload["degrees"].keys())
    degrees = [int(n[1:]) for n in names]
    design_orders = [d + 1 for d in degrees]
    
    order_fine = [payload["degrees"][n]["certification"]["order_fine"] for n in names]
    delta_rel = [payload["degrees"][n]["certification"]["delta_rel"] for n in names]
    uncertainty = [payload["degrees"][n]["certification"]["uncertainty"] for n in names]
    delta_band = [payload["degrees"][n]["certification"]["delta_band_rel"] for n in names]
    verdicts = [payload["degrees"][n]["certification"]["verdict"] for n in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    bars1 = ax.bar(x - width, order_fine, width, label="Observed order", color="#1f77b4")
    bars2 = ax.bar(x, design_orders, width, label="Design order (p+1)", color="#2ca02c", alpha=0.7)
    
    # Error bars showing uncertainty band
    ax.errorbar(x, order_fine, yerr=uncertainty, fmt='none', color='red', 
                capsize=5, linewidth=2, label="Estimator uncertainty")
    
    # Tolerance line
    tol = payload["tolerance_rel"]
    ax.axhline(y=tol, color='red', linestyle='--', alpha=0.5, label=f"Tolerance ({tol*100:.0f}%)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Order / Relative deviation", fontsize=12)
    ax.set_title("CKS1998 §5.3 — Order certification verdicts", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add verdict labels
    for i, (xi, vf, vb, v) in enumerate(zip(x, order_fine, delta_band, verdicts)):
        color = "green" if v == "CERTIFIED" else ("orange" if v == "UNVERIFIED_BAND" else "red")
        ax.text(xi, max(order_fine) + 0.1, v, ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color=color)
    
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {FIG}")

if __name__ == "__main__":
    main()
