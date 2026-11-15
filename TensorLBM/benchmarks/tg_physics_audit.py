"""Taylor-Green physics audit runner for XVAL host-staged halo close-out.

Runs the audit sweep of :mod:`tensorlbm.tg_physics_audit` (independent NumPy
D3Q19 BGK oracle), optionally cross-checks the subject implementation
``tensorlbm.host_staged_halo.HostStagedD3Q19`` (loaded from a TensorLBM source
tree via ``TENSORLBM_PORT_SRC``), and writes

    outputs/tg_physics_audit/table_tg_decay.csv
    outputs/tg_physics_audit/table_port_crosscheck.csv
    outputs/tg_physics_audit/summary.json
    outputs/tg_physics_audit/figures/fig1_decay_curves.png
    outputs/tg_physics_audit/figures/fig2_rate_error_vs_nu.png
    outputs/tg_physics_audit/figures/fig3_spectrum.png

Usage (inside the TensorLBM directory)::

    PYTHONPATH=src python benchmarks/tg_physics_audit.py
    TENSORLBM_PORT_SRC=/path/to/TensorLBM/src \
        PYTHONPATH=src python benchmarks/tg_physics_audit.py
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.tg_physics_audit import (  # noqa: E402
    AUDIT_PLAN,
    IndependentD3Q19BGK,
    analytic_decay_rate,
    audit_case,
    dominant_modes,
    fit_exponential_rate,
    mode_amplitude,
    taylor_green_velocity,
    taylor_green_velocity_double_scaled,
)

OUT = ROOT / "outputs" / "tg_physics_audit"
FIG = OUT / "figures"
PORT_DEFAULT = "/root/tensor-ecosystem/TensorLBM/src/tensorlbm/host_staged_halo.py"

# acceptance thresholds declared before the runs (owner standard: <15 %)
THRESH_ACCEPT = 0.15
THRESH_TIGHT = 0.03


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (f"{v:.6e}" if isinstance(v, float) else json.dumps(v) if isinstance(v, list) else v)
                    for k, v in row.items()
                }
            )


# ---------------------------------------------------------------------------
# Sweep (oracle)
# ---------------------------------------------------------------------------


PROFILES = {
    "full": {"grids": (16, 24, 32), "nus": (0.005, 0.01, 0.02, 0.05, 0.10),
             "steps": 600, "fit_window": (60, 480), "port_grids": (16, 24)},
    "quick": {"grids": (16, 24, 32), "nus": (0.01, 0.02, 0.05),
              "steps": 300, "fit_window": (40, 280), "port_grids": (16,)},
}


def sweep() -> list[dict]:
    plan = PROFILES[os.environ.get("TG_AUDIT_PROFILE", "full")]
    grids = plan["grids"]
    nus = plan["nus"]
    steps = plan["steps"]
    fit_window = plan["fit_window"]
    rows = []
    for n in grids:
        for nu in nus:
            for malformed in (False, True):
                t0 = time.time()
                row = audit_case(
                    n, nu, u0=float(AUDIT_PLAN["u0"]), steps=steps,
                    fit_window=fit_window, stride=20, malformed=malformed,
                )
                row["wall_s"] = time.time() - t0
                row["pass_accept_15pct"] = row["gamma_modal_rel_err"] <= THRESH_ACCEPT
                row["pass_tight_3pct"] = row["gamma_modal_rel_err"] <= THRESH_TIGHT
                row["pass_accept_15pct_ke"] = row["gamma_ke_rel_err"] <= THRESH_ACCEPT
                rows.append(row)
                print(
                    f"[sweep] n={n:2d} nu={nu:.3f} malformed={malformed!s:5s} "
                    f"modal_err={100 * row['gamma_modal_rel_err']:7.2f}% "
                    f"ke_err={100 * row['gamma_ke_rel_err']:7.2f}%",
                    flush=True,
                )
    return rows


# ---------------------------------------------------------------------------
# Port cross-check (subject implementation, torch)
# ---------------------------------------------------------------------------


def load_port_module():
    src = os.environ.get("TENSORLBM_PORT_SRC", PORT_DEFAULT)
    path = Path(src) / "tensorlbm" / "host_staged_halo.py" if Path(src).is_dir() else Path(src)
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_port_host_staged_halo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def port_crosscheck(port) -> list[dict]:
    """Decay + field agreement of the subject implementation vs the oracle."""
    import torch

    rows = []
    cases = [
        ("1x1x1", (1, 1, 1)),
        ("2x1x2", (2, 1, 2)),
        ("1x2x2", (1, 2, 2)),
        ("2x2x2", (2, 2, 2)),
    ]
    plan = PROFILES[os.environ.get("TG_AUDIT_PROFILE", "full")]
    for n in plan["port_grids"]:
        for nu in (0.02, 0.05):
            for label, builder in (("correct", taylor_green_velocity),
                                   ("malformed", taylor_green_velocity_double_scaled)):
                k = 2.0 * math.pi / n
                u_np = builder(n, n, n, float(AUDIT_PLAN["u0"]))
                u_t = torch.tensor(u_np, dtype=torch.float64)
                for name, (dx, dy, dz) in cases:
                    lay = port.HostStagedLayout(n, n, n, dx, dy, dz)
                    sim = port.HostStagedD3Q19(lay, nu=nu, dtype=torch.float64)
                    sim.set_velocity(u_t)
                    ke = []
                    ts = []
                    for step in range(1, 201):
                        sim.step(1)
                        if step % 20 == 0:
                            ts.append(float(step))
                            ke.append(sim.kinetic_energy())
                    g = fit_exponential_rate(np.array(ts), np.array(ke))
                    theo = analytic_decay_rate(nu, k, "kinetic_energy")
                    rows.append(
                        {
                            "grid": n,
                            "nu": nu,
                            "devices": name,
                            "ic": label,
                            "gamma_ke_port": g,
                            "gamma_ke_theory": theo,
                            "rel_err": abs(g - theo) / theo,
                            "pass_accept_15pct": abs(g - theo) / theo <= THRESH_ACCEPT,
                        }
                    )
                    print(
                        f"[port] n={n:2d} nu={nu:.3f} {label:9s} {name} "
                        f"ke_err={100 * abs(g - theo) / theo:7.2f}%",
                        flush=True,
                    )
    # field-level identity: port (decomposed, host-staged) vs oracle after 50 steps
    n, nu = 16, 0.02
    u = taylor_green_velocity(n, n, n, float(AUDIT_PLAN["u0"]))
    sim = port.HostStagedD3Q19(
        port.HostStagedLayout(n, n, n, 2, 1, 2), nu=nu, dtype=torch.float64
    )
    sim.set_velocity(torch.tensor(u, dtype=torch.float64))
    sim.step(50)
    port_u = sim.project_velocity().numpy()
    port_rho = sim.project_density().numpy()
    oracle = IndependentD3Q19BGK(n, nu=nu)
    oracle.set_velocity(u)
    oracle.step(50)
    rho_o, u_o = oracle.moments()
    rows.append(
        {
            "grid": n,
            "nu": nu,
            "devices": "2x1x2",
            "ic": "correct",
            "gamma_ke_port": "",
            "gamma_ke_theory": "",
            "rel_err": "",
            "pass_accept_15pct": "",
            "field_max_du": float(np.abs(port_u - u_o).max()),
            "field_max_drho": float(np.abs(port_rho - rho_o).max()),
        }
    )
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def decay_curves(n: int = 24, nu: float = 0.02):
    """Sample modal amplitude + kinetic energy traces for both ICs."""
    k = 2.0 * math.pi / n
    traces = {}
    for label, builder in (("correct", taylor_green_velocity),
                           ("malformed", taylor_green_velocity_double_scaled)):
        sim = IndependentD3Q19BGK(n, nu=nu)
        sim.set_velocity(builder(n, n, n, float(AUDIT_PLAN["u0"])))
        sim.step(20)
        t, amp, ke = [], [], []
        for step in range(20, 421):
            sim.step(1)
            if step % 10 == 0:
                rho, u = sim.moments()
                t.append(step)
                amp.append(mode_amplitude(u, (0, 1, 1)))
                ke.append(sim.kinetic_energy(rho, u))
        traces[label] = (np.array(t), np.array(amp), np.array(ke))
    return traces, k


def make_figures(rows: list[dict], port_rows: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    # fig1: decay traces, corrected vs malformed IC
    traces, k = decay_curves()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    nu = 0.02
    for label, marker in (("correct", "o"), ("malformed", "s")):
        t, amp, ke = traces[label]
        axes[0].semilogy(t[::3], amp[::3], marker, ms=3, label=f"{label} IC")
        axes[1].semilogy(t[::3], ke[::3], marker, ms=3, label=f"{label} IC")
    t, amp, ke = traces["correct"]
    axes[0].semilogy(t, amp[0] * np.exp(-2 * nu * k * k * (t - t[0])), "k-", lw=1,
                     label=r"analytic $e^{-2\nu k^2t}$")
    axes[1].semilogy(t, ke[0] * np.exp(-4 * nu * k * k * (t - t[0])), "k-", lw=1,
                     label=r"analytic $e^{-4\nu k^2t}$")
    axes[0].set_xlabel("lattice step"); axes[0].set_ylabel(r"$|\hat u(1,1)|$")
    axes[0].set_title("fundamental mode amplitude"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("lattice step"); axes[1].set_ylabel(r"$0.5\langle\rho u^2\rangle$")
    axes[1].set_title("kinetic energy"); axes[1].legend(fontsize=8)
    fig.suptitle(f"Taylor-Green decay, D3Q19 BGK, ${24}^3$, $\\nu$={nu}", fontsize=11)
    fig.tight_layout()
    p = FIG / "fig1_decay_curves.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(p)

    # fig2: rate error vs nu, corrected vs malformed, per grid
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for n in sorted({r["grid"] for r in rows}):
        for malformed, ls, mk in ((False, "-", "o"), (True, "--", "s")):
            sel = [r for r in rows if r["grid"] == n and r["malformed_ic"] == malformed]
            sel.sort(key=lambda r: r["nu"])
            xs = [r["nu"] for r in sel]
            ys = [100 * r["gamma_modal_rel_err"] for r in sel]
            ax.plot(xs, ys, ls, marker=mk, ms=4,
                    label=f"n={n}, {'malformed' if malformed else 'corrected'} IC")
    ax.axhline(100 * THRESH_ACCEPT, color="r", lw=1, ls=":",
               label="acceptance 15 %")
    ax.axhline(100 * THRESH_TIGHT, color="g", lw=1, ls=":",
               label="tight 3 %")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$"); ax.set_ylabel("fundamental-mode rate error [%]")
    ax.set_title("Taylor-Green decay-rate error (modal observable)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "fig2_rate_error_vs_nu.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(p)

    # fig3: late-time spectra (corrected vs malformed)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    n = 24
    for label, ax in (("correct", axes[0]), ("malformed", axes[1])):
        if label == "correct":
            _, _, _ = traces[label]
        sim = IndependentD3Q19BGK(n, nu=0.02)
        sim.set_velocity(
            (taylor_green_velocity if label == "correct" else taylor_green_velocity_double_scaled)(
                n, n, n, float(AUDIT_PLAN["u0"])
            )
        )
        sim.step(400)
        _, u = sim.moments()
        hat = np.fft.fftn(u, axes=(1, 2, 3)) / n ** 3
        power = (np.abs(hat) ** 2).sum(axis=0).sum(axis=0)  # sum over kz, vy -> kx axis
        ax.semilogy(np.arange(n), np.maximum(power, 1e-18), "o-", ms=3)
        ax.set_xlabel(r"$k_x$ (power summed over $k_y,k_z$)")
        ax.set_ylabel(r"modal energy $|\hat u|^2$")
        ax.set_title(f"{label} IC, 400 steps")
        modes = dominant_modes(u, top=3)
        ax.text(0.35, 0.75, "top modes: " + ", ".join(f"{m}:{f:.2f}" for m, f in modes),
                transform=ax.transAxes, fontsize=7)
    fig.suptitle("Late-time spectral content: malformed IC parks energy in the mean mode",
                 fontsize=10)
    fig.tight_layout()
    p = FIG / "fig3_spectrum.png"; fig.savefig(p, dpi=140); plt.close(fig); out.append(p)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = sweep()
    write_csv(OUT / "table_tg_decay.csv", rows)

    port_rows: list[dict] = []
    port = load_port_module()
    if port is not None:
        port_rows = port_crosscheck(port)
        write_csv(OUT / "table_port_crosscheck.csv", port_rows)
    else:
        print("[port] subject implementation not found; skipping cross-check", flush=True)

    figs = make_figures(rows, port_rows)

    def worst(sel):
        return max((r["gamma_modal_rel_err"] for r in sel), default=float("nan"))

    correct = [r for r in rows if not r["malformed_ic"]]
    malformed = [r for r in rows if r["malformed_ic"]]
    peak_err = [r for r in malformed if r["nu"] == 0.02]
    summary = {
        "case": "XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO / TG physics audit",
        "plan": {k: list(v) if isinstance(v, tuple) else v for k, v in AUDIT_PLAN.items()},
        "thresholds": {"acceptance": THRESH_ACCEPT, "tight": THRESH_TIGHT},
        "worst_modal_rel_err_corrected_ic": worst(correct),
        "worst_modal_rel_err_malformed_ic": worst(malformed),
        "all_corrected_within_acceptance": all(r["gamma_modal_rel_err"] <= THRESH_ACCEPT for r in correct),
        "all_corrected_within_tight": all(r["gamma_modal_rel_err"] <= THRESH_TIGHT for r in correct),
        "malformed_ic_errors_at_nu_0.02": [
            {"grid": r["grid"], "modal_rel_err": r["gamma_modal_rel_err"],
             "ke_rel_err": r["gamma_ke_rel_err"],
             "mean_momentum": r["mean_momentum0"]} for r in peak_err
        ],
        "port_crosscheck_rows": port_rows,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "port_module": os.environ.get("TENSORLBM_PORT_SRC", PORT_DEFAULT),
        },
        "artifacts": ["table_tg_decay.csv", "table_port_crosscheck.csv"]
        + [str(p.relative_to(OUT)) for p in figs],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in (
        "worst_modal_rel_err_corrected_ic",
        "worst_modal_rel_err_malformed_ic",
        "all_corrected_within_acceptance",
        "all_corrected_within_tight",
    )}, indent=2))
    return summary


if __name__ == "__main__":
    main()