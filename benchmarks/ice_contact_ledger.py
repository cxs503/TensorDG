"""Runner for the DEM-FEM interface ledger (anchor ``GAP-ICE-LEDGER-PHYSICS``).

Produces ``artifacts/ice_contact_ledger.json`` (all numbers) and
``docs/figures/ice_contact_ledger_*.png`` (figures).

The point of the exercise is the *grading rule* the ledger anchor demands:

* the ice-structure interface must be graded on a **physical residual**
  (momentum / energy / interface impulse / interface work) at ``<= 1e-3``,
* with a **real FEM participant** in the loop,
* and a backend-consistency number may only be reported as
  ``backend port parity`` -- it must not be used as conservation evidence.

The runner therefore also runs two *deliberate defects* to show that the
parity number stays comfortably below the old 5e-3 parity threshold while the
physical residuals blow through the 1e-3 gate.

Usage::

    python benchmarks/ice_contact_ledger.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordg.dem_fem_interface_ledger import (
    RESIDUAL_GATE,
    ContactSpec,
    CouplingConfig,
    FemRod,
    ParticleSpec,
    RodSpec,
    backend_port_parity,
    run_coupling,
    verify_contact_oscillator,
    verify_fem_eigenfrequencies,
    verify_fem_static,
    verify_stiff_rod_limit,
)

LEGACY_PARITY_GATE = 5.0e-3  # the parity threshold the anchor refuses


def _r(x: float) -> float:
    return float(f"{x:.6e}")


def run_ledger_case(name: str, **kwargs) -> dict:
    cfg = CouplingConfig(**kwargs)
    res = run_coupling(cfg, keep_history=True)
    ledger = res["ledger"]
    parity = backend_port_parity(res)
    hist = res["history"]
    return {
        "name": name,
        "config": {"force_split": cfg.force_split,
                   "frozen_coupling": cfg.frozen_coupling,
                   "dt": cfg.dt, "t_end": cfg.t_end, "n_elem": cfg.n_elem},
        "ledger": {k: (_r(v) if isinstance(v, float) else v)
                   for k, v in ledger.items()},
        "backend_port_parity": {k: (_r(v) if isinstance(v, float) else v)
                                for k, v in parity.items()},
        "history": {k: v for k, v in hist.items()},
    }


def dt_sweep(dts=(1.0e-6, 2.0e-6, 5.0e-6)) -> list[dict]:
    out = []
    for dt in dts:
        mono = run_coupling(CouplingConfig(dt=dt))["ledger"]
        stag = run_coupling(CouplingConfig(dt=dt, frozen_coupling=True))["ledger"]
        out.append({"dt": dt,
                    "monolithic_energy_residual": _r(mono["energy_residual"]),
                    "monolithic_impulse_residual": _r(mono["interface_impulse_residual"]),
                    "staggered_energy_residual": _r(stag["energy_residual"]),
                    "staggered_impulse_residual": _r(stag["interface_impulse_residual"])})
    return out


def make_figures(report: dict, figdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    figures = []

    # --- figure 1: contact force + energy partition ---------------------
    healthy = next(c for c in report["cases"] if c["name"] == "healthy")
    h = healthy["history"]
    t = h["time"].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(t * 1e3, h["force"].numpy(), lw=1.6, color="#1f4e79")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("interface contact force [N]")
    ax.set_title("DEM–FEM interface flux (healthy monolithic coupling)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    energy = h["energy"].numpy()
    tot = energy.sum(axis=1)
    ax.plot(t * 1e3, energy[:, 0], label="FEM kinetic")
    ax.plot(t * 1e3, energy[:, 1], label="FEM strain")
    ax.plot(t * 1e3, energy[:, 2], label="DEM kinetic")
    ax.plot(t * 1e3, energy[:, 3], label="contact spring")
    ax.plot(t * 1e3, tot, "k--", lw=1.2, label="total (conserved)")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("energy [J]")
    ax.set_title(f"energy partition, total drift {report['cases'][0]['ledger']['energy_residual']:.1e}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = figdir / "ice_contact_ledger_energy.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Left: interface contact force history of the healthy monolithic "
                               "DEM-FEM run. Right: energy partition between the DEM, FEM and "
                               "contact-spring reservoirs; the total (dashed) is conserved to "
                               f"{report['cases'][0]['ledger']['energy_residual']:.1e} relative.",
                    "path": str(p)})

    # --- figure 2: parity vs physics (the anchor's core claim) ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    names = [c["name"] for c in report["cases"]]
    phys = [max(c["ledger"]["momentum_residual"], c["ledger"]["energy_residual"],
                c["ledger"]["interface_impulse_residual"],
                c["ledger"]["interface_work_residual"]) for c in report["cases"]]
    parity = [c["backend_port_parity"]["value"] for c in report["cases"]]
    xs = range(len(names))
    ax.bar([x - 0.2 for x in xs], phys, width=0.4, label="max physical residual", color="#c0504d")
    ax.bar([x + 0.2 for x in xs], parity, width=0.4,
           label="backend port parity", color="#4f81bd")
    ax.axhline(RESIDUAL_GATE, color="k", ls="-", lw=1.2,
               label=f"physical gate {RESIDUAL_GATE:.0e}")
    ax.axhline(LEGACY_PARITY_GATE, color="k", ls=":", lw=1.2,
               label=f"legacy parity gate {LEGACY_PARITY_GATE:.0e}")
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("dimensionless")
    ax.set_title("parity passes where physics fails (defect cases)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = figdir / "ice_contact_ledger_parity_vs_physics.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Physical residuals (red) versus the backend port parity number "
                               "(blue) for the healthy run and the two injected interface "
                               "defects: the parity number stays under the legacy 5e-3 "
                               "threshold in every case, while the physical residual crosses "
                               "the 1e-3 gate for both defects.",
                    "path": str(p)})

    # --- figure 3: dt scaling ------------------------------------------
    sweep = report["dt_sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    dts = [s["dt"] for s in sweep]
    ax.loglog(dts, [s["monolithic_energy_residual"] for s in sweep], "o-",
              label="monolithic (conservative) RK4")
    ax.loglog(dts, [s["staggered_energy_residual"] for s in sweep], "s-",
              label="explicit staggered (frozen force)")
    ax.axhline(RESIDUAL_GATE, color="k", ls="-", lw=1.2, label=f"gate {RESIDUAL_GATE:.0e}")
    ax.set_xlabel("dt [s]")
    ax.set_ylabel("energy residual (relative)")
    ax.set_title("interface energy residual vs time step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = figdir / "ice_contact_ledger_dt_scaling.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figures.append({"name": p.name,
                    "caption": "Interface energy residual against dt: the monolithic conservative "
                               "coupling stays at integrator accuracy (O(dt^3-4)), while the "
                               "explicit staggered coupling leaks at O(dt) and crosses the 1e-3 "
                               "gate at the production time step.",
                    "path": str(p)})
    return figures


def main(quick: bool = False) -> dict:
    spec = RodSpec()
    rod = FemRod(spec, 32)

    static = verify_fem_static(rod, load=1.0e4)
    eigen = verify_fem_eigenfrequencies(spec, n_elems=(8, 16, 32, 64))
    oscillator = verify_contact_oscillator(dt=1.0e-7)
    stiff = verify_stiff_rod_limit()
    stiff.pop("ledger", None)

    cases = [run_ledger_case("healthy"),
             run_ledger_case("force_split_0.99", force_split=0.99),
             run_ledger_case("staggered", frozen_coupling=True)]
    if quick:
        sweep = dt_sweep(dts=(1.0e-6, 2.0e-6))
    else:
        sweep = dt_sweep()

    report = {
        "anchor": "GAP-ICE-LEDGER-PHYSICS",
        "title": "DEM-FEM interface energy/momentum ledger with physical residuals",
        "gate": RESIDUAL_GATE,
        "legacy_parity_gate": LEGACY_PARITY_GATE,
        "rod": {"length": spec.length, "density": spec.density,
                "young": spec.young, "area": spec.area,
                "wave_speed": spec.wave_speed, "total_mass": spec.total_mass},
        "participants": {
            "fem": {"type": "P1 consistent-mass axial rod (free-free)",
                    "n_elem": rod.n_elem, "dofs": rod.nfree,
                    "discrete_mass": _r(float(rod.momentum_functional.sum()))},
            "dem": {"type": "rigid particle train, linear penalty contact",
                    "n_particles": 1, "mass": 5.0, "radius": 0.05,
                    "v0": 1.0, "contact_stiffness": 1.0e7},
        },
        "reference_checks": {
            "fem_static_elongation": {k: (_r(v) if isinstance(v, float) else v)
                                      for k, v in static.items()},
            "fem_eigenfrequencies": eigen,
            "contact_oscillator_rigid_wall": {
                "analytic": oscillator["analytic"],
                "numerical": oscillator["numerical"],
                "relative_error_peak_force": _r(oscillator["relative_error_peak_force"]),
                "relative_error_duration": _r(oscillator["relative_error_duration"]),
            },
            "stiff_rod_two_body_limit": {
                "analytic": stiff["analytic"],
                "numerical": stiff["numerical"],
                "relative_error_peak_force": _r(stiff["relative_error_peak_force"]),
                "relative_error_duration": _r(stiff["relative_error_duration"]),
            },
        },
        "cases": cases,
        "dt_sweep": sweep,
    }

    outdir = Path(__file__).resolve().parent.parent / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = Path(__file__).resolve().parent.parent / "docs" / "figures"
    figures = make_figures(report, figdir)
    report["figures"] = [{"name": f["name"], "caption": f["caption"]} for f in figures]
    for case in report["cases"]:
        case.pop("history", None)
    (outdir / "ice_contact_ledger.json").write_text(
        json.dumps(report, indent=1,
                   default=lambda o: float(o) if hasattr(o, "item") else str(o)))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    rep = main(quick=args.quick)
    for case in rep["cases"]:
        led = case["ledger"]
        print(f"{case['name']:18s} mom={led['momentum_residual']:.3e} "
              f"eng={led['energy_residual']:.3e} "
              f"J={led['interface_impulse_residual']:.3e} "
              f"W={led['interface_work_residual']:.3e} "
              f"parity={case['backend_port_parity']['value']:.3e} "
              f"pass={led['passes_gate']}")