"""Cross-validation benchmark: fused (direction-selective) periodic ghost exchange.

Anchor ``XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED``.

Arms
----
A  selectivity conformance   — seam cell set vs upstream lbmpy ``comm_slices``
B  exactness                 — fused vs full-field-copy vs ghost-array pass
C  cost                      — element traffic and wall time per step
D  physics                   — analytic decay rates (2-D Taylor-Green, 3-D shear wave)

Writes ``outputs/fused_periodicity/xval_summary.json`` plus PNG figures.
Every number quoted in the report/docs is read back from that JSON.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.fused_periodicity import (
    D2Q9,
    D3Q19,
    D3Q27,
    FusedPeriodicBGK,
    FusedPeriodicStreamer,
    fit_exponential_rate,
    macroscopic,
    mode_amplitude,
    selection_matrix,
    shear_wave,
    taylor_green_2d,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "fused_periodicity"
OUT.mkdir(parents=True, exist_ok=True)


def arm_a_conformance() -> dict:
    try:
        from lbmpy.session import LBStencil, Stencil
        from pystencils.datahandling import create_data_handling
        from lbmpy.advanced_streaming.communication import LBMPeriodicityHandling
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": repr(exc)}

    cases = [("D2Q9", Stencil.D2Q9, D2Q9, (3, 4)), ("D2Q9", Stencil.D2Q9, D2Q9, (7, 5)),
             ("D3Q19", Stencil.D3Q19, D3Q19, (4, 5, 3)), ("D3Q27", Stencil.D3Q27, D3Q27, (3, 4, 2))]
    rows = []
    for name, enum, stencil, shape in cases:
        dh = create_data_handling(shape, periodicity=(True,) * len(shape), default_target="cpu")
        dh.add_array("pdfs", values_per_cell=len(stencil))
        dh.fill("pdfs", 0.0, ghost_layers=True)
        per = LBMPeriodicityHandling(stencil=LBStencil(enum), data_handling=dh, pdf_field_name="pdfs")
        lb = np.array(LBStencil(enum), dtype=np.int64)
        n_entries = sum(len(ts) for ts in per.comm_slices)
        cells = {}
        for ts in per.comm_slices:
            for _src, dst in ts:
                *spatial, pdf = dst
                axes = []
                for axis, entry in enumerate(spatial):
                    n = shape[axis]
                    ext = n + 2
                    if isinstance(entry, slice):
                        lo = 0 if entry.start is None else (entry.start + ext if entry.start < 0 else entry.start)
                        hi = ext if entry.stop is None else (entry.stop + ext if entry.stop < 0 else entry.stop)
                        idx = range(lo, hi)
                    else:
                        idx = [entry + ext if entry < 0 else entry]
                    axes.append([(int(i) - 1) % n for i in idx])
                key = tuple(int(v) for v in lb[pdf])
                for cell in __import__("itertools").product(*axes):
                    cells.setdefault(key, set()).add(cell)
        mine = selection_matrix(stencil, shape)
        mismatch = 0
        for q, velocity in enumerate(stencil):
            c = tuple(int(v) for v in velocity)
            shifted = {tuple((cell + e) % n for cell, e, n in zip(cell, c, shape))
                       for cell in cells.get(c, set())}
            ours = {tuple(int(v) for v in idx) for idx in np.argwhere(mine[q])}
            if shifted != ours:
                mismatch += 1
        rows.append({
            "lattice": name, "shape": list(shape),
            "upstream_copy_entries": int(n_entries),
            "upstream_seam_cells": int(sum(len(v) for v in cells.values())),
            "derived_seam_cells": int(mine.sum()),
            "populations_mismatched": mismatch,
        })
    return {"available": True, "rows": rows,
            "all_conform": all(r["populations_mismatched"] == 0 and
                               r["upstream_seam_cells"] == r["derived_seam_cells"] for r in rows)}


def arm_b_exactness() -> dict:
    rows = []
    for name, stencil, shape in [("D2Q9", D2Q9, (32, 40)), ("D3Q19", D3Q19, (12, 14, 10)),
                                 ("D3Q27", D3Q27, (10, 12, 8))]:
        rng = np.random.default_rng(2026)
        f = rng.standard_normal((len(stencil),) + shape)
        s = FusedPeriodicStreamer(stencil, shape)
        fused = s.stream_fused(f)
        fused_stats = s.stats.as_dict()
        fullcopy = s.stream_fullcopy(f)
        fullcopy_stats = s.stats.as_dict()
        ghost = s.stream_ghost(f)
        ghost_stats = s.stats.as_dict()
        rows.append({
            "lattice": name, "shape": list(shape),
            "max_abs_diff_fused_fullcopy": float(np.abs(fused - fullcopy).max()),
            "max_abs_diff_fused_ghost": float(np.abs(fused - ghost).max()),
            "bitwise_equal": bool(np.array_equal(fused, fullcopy) and np.array_equal(fused, ghost)),
            "fused": fused_stats, "fullcopy": fullcopy_stats, "ghost": ghost_stats,
            "element_ratio_fused_over_fullcopy":
                fused_stats["n_elem_total"] / fullcopy_stats["n_elem_total"],
        })
    return {"rows": rows, "all_bitwise_equal": all(r["bitwise_equal"] for r in rows)}


def _time_stream(fn, f, reps=20):
    fn(f)  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(f)
    return (time.perf_counter() - t0) / reps


def arm_c_cost() -> dict:
    rows = []
    for name, stencil in [("D2Q9", D2Q9), ("D3Q19", D3Q19), ("D3Q27", D3Q27)]:
        dim = stencil.shape[1]
        sizes = [32, 64, 128, 256] if dim == 2 else [16, 24, 32, 48]
        for n in sizes:
            shape = (n,) * dim
            rng = np.random.default_rng(1)
            f = rng.standard_normal((len(stencil),) + shape)
            s = FusedPeriodicStreamer(stencil, shape)
            t_fused = _time_stream(s.stream_fused, f)
            fused_stats = s.stats.as_dict()
            t_full = _time_stream(s.stream_fullcopy, f, reps=10)
            full_stats = s.stats.as_dict()
            t_ghost = _time_stream(s.stream_ghost, f, reps=10)
            ghost_stats = s.stats.as_dict()
            rows.append({
                "lattice": name, "n": n, "shape": list(shape),
                "fused_s": t_fused, "fullcopy_s": t_full, "ghost_s": t_ghost,
                "speedup_vs_fullcopy": t_full / t_fused,
                "speedup_vs_ghost": t_ghost / t_fused,
                "seam_fraction": fused_stats["seam_fraction"],
                "fused_elem": fused_stats["n_elem_total"],
                "fullcopy_elem": full_stats["n_elem_total"],
                "ghost_elem": ghost_stats["n_elem_total"],
                "seam_bytes": fused_stats["bytes_seam"],
                "fullcopy_bytes": full_stats["bytes_written"],
            })
    return {"rows": rows}


def arm_d_physics() -> dict:
    # --- 2-D Taylor-Green: velocity mode decays at nu * |k|^2 = 2 nu k^2 -----
    tg = []
    nu = 0.05
    for n in (48, 96):
        k = 2.0 * np.pi / n
        solver = FusedPeriodicBGK(D2Q9, (n, n), nu=nu, mode="fused")
        solver.init_from_velocity(taylor_green_2d((n, n), amp=0.02))
        _rho, u = solver.macroscopic()
        times, amps = [0], [mode_amplitude(u[0], (1, 1))]
        t_stop = int(1.5 / (2.0 * nu * k * k))
        chunk = max(1, t_stop // 12)
        while times[-1] < t_stop:
            solver.step(chunk)
            times.append(times[-1] + chunk)
            _rho, u = solver.macroscopic()
            amps.append(mode_amplitude(u[0], (1, 1)))
        rate = fit_exponential_rate(times, amps)
        analytic = -2.0 * nu * k * k
        tg.append({"n": n, "steps": int(times[-1]), "measured_rate": rate,
                   "analytic_rate": analytic,
                   "rel_error": abs(rate - analytic) / abs(analytic)})

    # --- 3-D D3Q19 Kolmogorov shear wave: exact decay at nu k^2 --------------
    sw = []
    for n in (24, 32):
        nu3 = 0.05
        k = 2.0 * np.pi / n
        solver = FusedPeriodicBGK(D3Q19, (n, n, n), nu=nu3, mode="fused")
        solver.init_from_velocity(shear_wave((n, n, n), axis=0, varying=1, amp=0.02))
        times, amps = [0], [mode_amplitude(solver.f[1] - solver.f[2], (0, 1, 0))]
        t_stop = int(1.2 / (nu3 * k * k))
        chunk = max(1, t_stop // 10)
        while times[-1] < t_stop:
            solver.step(chunk)
            times.append(times[-1] + chunk)
            amps.append(mode_amplitude(solver.f[1] - solver.f[2], (0, 1, 0)))
        rate = fit_exponential_rate(times, amps)
        analytic = -nu3 * k * k
        sw.append({"n": n, "steps": int(times[-1]), "measured_rate": rate,
                   "analytic_rate": analytic,
                   "rel_error": abs(rate - analytic) / abs(analytic),
                   "amp_trace": [float(a) for a in amps], "times": [int(t) for t in times]})

    # --- mass conservation over the same runs --------------------------------
    solver = FusedPeriodicBGK(D2Q9, (64, 64), nu=0.05, mode="fused")
    solver.init_from_velocity(taylor_green_2d((64, 64), amp=0.02))
    m0 = solver.total_mass()
    solver.step(500)
    mass_drift = abs(solver.total_mass() - m0) / m0
    return {"taylor_green_2d": tg, "shear_wave_3d": sw,
            "mass_drift_500_steps": mass_drift,
            "worst_physics_rel_error": max([r["rel_error"] for r in tg] + [r["rel_error"] for r in sw])}


def figures(summary: dict) -> list:
    made = []

    # Fig 1: cost / selectivity scaling
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, marker in [("D2Q9", "o"), ("D3Q19", "s"), ("D3Q27", "^")]:
        rows = [r for r in summary["arm_c_cost"]["rows"] if r["lattice"] == name]
        ns = [r["n"] for r in rows]
        axes[0].loglog(ns, [r["seam_fraction"] for r in rows], marker=marker, linestyle="-", label=name)
        axes[1].loglog(ns, [r["speedup_vs_fullcopy"] for r in rows], marker=marker, linestyle="-", label=name)
    axes[0].set_xlabel("grid size N per axis")
    axes[0].set_ylabel("seam elements / all streamed elements")
    axes[0].set_title("(a) direction-selective seam traffic fraction")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("grid size N per axis")
    axes[1].set_ylabel("t_fullcopy / t_fused")
    axes[1].set_title("(b) fused vs whole-field-copy streaming")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    for ax in axes:
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
    fig.tight_layout()
    p = OUT / "fig_cost_scaling.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append({"name": "fig_cost_scaling.png",
                 "caption": "Direction-selective seam traffic fraction and fused-vs-copy streaming "
                            "time ratio versus grid size, for D2Q9 / D3Q19 / D3Q27."})

    # Fig 2: analytic decay validation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for row in summary["arm_d_physics"]["shear_wave_3d"]:
        t = np.array(row["times"], dtype=float)
        a = np.array(row["amp_trace"], dtype=float)
        axes[0].semilogy(t, a / a[0], "o", label=f"fused N={row['n']}")
        axes[0].semilogy(t, np.exp(row["analytic_rate"] * t), "-",
                         label=f"analytic e^(nu k^2 t) N={row['n']}")
    axes[0].set_xlabel("time step")
    axes[0].set_ylabel("mode amplitude / initial")
    axes[0].set_title("(a) 3-D D3Q19 shear wave, nu k^2 decay")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    tg = summary["arm_d_physics"]["taylor_green_2d"]
    ns = [r["n"] for r in tg]
    err = [100 * r["rel_error"] for r in tg]
    axes[1].plot(ns, err, "o-", label="2-D Taylor-Green, |err| vs 2 nu k^2")
    axes[1].axhline(3.0, color="r", ls="--", label="3% line")
    axes[1].axhline(15.0, color="k", ls=":", label="15% acceptance")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("grid size N")
    axes[1].set_ylabel("relative rate error [%]")
    axes[1].set_title("(b) decay-rate error against analytic solution")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig_decay_validation.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append({"name": "fig_decay_validation.png",
                 "caption": "Measured mode decay of the fused-exchange solver against the analytic "
                            "rates (3-D shear wave nu k^2; 2-D Taylor-Green 2 nu k^2) and the "
                            "3% / 15% error bands."})

    # Fig 3: exchange field map (seam mask + one streamed population)
    shape = (24, 32)
    mask = selection_matrix(D2Q9, shape)
    s = FusedPeriodicStreamer(D2Q9, shape)
    u = taylor_green_2d(shape, amp=0.05)
    solver = FusedPeriodicBGK(D2Q9, shape, nu=0.05, mode="fused")
    solver.init_from_velocity(u)
    solver.step(120)
    _rho, uu = solver.macroscopic()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    im0 = axes[0].imshow(mask.sum(axis=0), origin="lower", cmap="viridis")
    axes[0].set_title("seam cells per node (summed over populations)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    im1 = axes[1].imshow(uu[0], origin="lower", cmap="RdBu_r")
    axes[1].set_title("ux after 120 fused steps, 24x32")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.suptitle("D2Q9 direction-selective seam map and streamed field", fontsize=10)
    fig.tight_layout()
    p = OUT / "fig_seam_map_field.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append({"name": "fig_seam_map_field.png",
                 "caption": "Per-node count of populations crossing the periodic seam (D2Q9, 24x32) "
                            "next to the velocity field produced by the fused-exchange solver."})
    return made


def main() -> dict:
    t0 = time.time()
    summary = {
        "anchor": "XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED",
        "upstream": "lssfau/lbmpy src/lbmpy/advanced_streaming/communication.py "
                    "(LBMPeriodicityHandling / get_communication_slices)",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine()},
        "arm_a_conformance": arm_a_conformance(),
        "arm_b_exactness": arm_b_exactness(),
        "arm_c_cost": arm_c_cost(),
        "arm_d_physics": arm_d_physics(),
    }
    summary["figures"] = figures(summary)
    summary["wall_seconds"] = time.time() - t0
    (OUT / "xval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "figures"}, indent=2)[:4000])
    print("figures:", [f["name"] for f in summary["figures"]])
    return summary


if __name__ == "__main__":
    main()