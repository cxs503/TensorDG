"""Generate the dispersion-analysis validation figure (p=0 vs closed form)."""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg import dispersion_analysis as da

REPO = Path(__file__).resolve().parents[1]

# p=0 branch vs closed form
thetas = np.linspace(1e-3, math.pi, 200)
branch = da.physical_branch(0, thetas)
closed = np.array([da.upwind_fv_symbol(t) for t in thetas])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# left: phase speed vs theta
c_num = -branch.imag / thetas
axes[0].plot(thetas, c_num, "o-", ms=3, label="assembled p=0")
axes[0].plot(thetas, np.sin(thetas) / thetas, "--", label="closed form sin(t)/t")
axes[0].set_xlabel(r"$\theta = k h$")
axes[0].set_ylabel(r"$c_{num}$")
axes[0].set_title("p=0 phase speed vs closed form")
axes[0].legend()
axes[0].grid(True)

# right: dissipation vs closed form (1-cos)/theta
decay = -branch.real / thetas
closed_decay = (1.0 - np.cos(thetas)) / thetas
axes[1].plot(thetas, decay, "o-", ms=3, label="assembled p=0")
axes[1].plot(thetas, closed_decay, "--", label=r"closed form $(1-\cos\theta)/\theta$")
axes[1].set_xlabel(r"$\theta = k h$")
axes[1].set_ylabel(r"$-\mathrm{Re}\,\lambda/\theta$")
axes[1].set_title("p=0 dissipation vs closed form")
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
out = REPO / "artifacts" / "dispersion_analysis_salvage_p0.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print("saved", out)

# also write a small numeric summary for the report
summary = {
    "p0_max_abs_error_vs_closed": float(np.abs(branch - closed).max()),
    "p0_phase_speed_at_1e-6": float(-da.physical_branch(0, np.array([1e-6]))[0].imag / 1e-6),
    "p1_ppw_phase": da.resolution_table(1)["ppw_phase"],
    "p1_ppw_amplitude": da.resolution_table(1)["ppw_amplitude"],
}
(REPO / "artifacts" / "dispersion_analysis_salvage_summary.json").write_text(
    json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
