"""Cross-validation harness: sparse list-based AOS LBM (lbmpy pattern) in TensorLBM.

Runs four independent checks and writes every number to
``outputs/sparse_aos/xval_summary.json`` (no hand-copied digits) plus figures to
``outputs/sparse_aos/``:

1. ``equivalence``  — sparse AOS/SoA state vs the dense ``(Q, Ny, Nx)`` solver
   (``stream_periodic`` + ``bgk_collision``) on a fully periodic domain.
2. ``poiseuille``   — body-force channel flow vs the half-way bounce-back
   analytic profile at three resolutions (acceptance: rel-L2 < 15 %).
3. ``taylor_green`` — viscous decay rate vs the analytic
   ``exp(-nu k^2 t)`` law on the sparse (fully periodic) domain.
4. ``cost``         — memory and per-step wall clock of dense ``(Q, Ny, Nx)``,
   sparse AOS ``(n, Q)`` and sparse SoA ``(Q, n)`` on a partially solid domain.

Usage
-----
    cd TensorLBM && python3 benchmarks/sparse_aos_xval.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from tensorlbm import D2Q9, bgk_collision, stream_periodic  # noqa: E402
from tensorlbm.sparse_aos import (  # noqa: E402
    SparseAOSLBM,
    build_sparse_domain,
    channel_mask,
)

OUT = Path(__file__).resolve().parent.parent / "outputs" / "sparse_aos"
DTYPE = torch.float64
torch.set_num_threads(2)


# --------------------------------------------------------------------------- 1
def check_equivalence(resolutions=((16, 16, 80), (32, 32, 80))) -> list:
    """Sparse list-based solver vs dense solver on a fully periodic box."""
    rows = []
    for ny, nx, steps in resolutions:
        y = torch.arange(ny, dtype=DTYPE)
        x = torch.arange(nx, dtype=DTYPE)
        kx, ky = 2 * math.pi / nx, 2 * math.pi / ny
        amp = 0.04
        ux0 = amp * torch.sin(kx * x)[None, :] * torch.cos(ky * y)[:, None]
        uy0 = -amp * torch.cos(kx * x)[None, :] * torch.sin(ky * y)[:, None]
        rho0 = torch.ones(ny, nx, dtype=DTYPE)
        omega = 1.9

        f = D2Q9.equilibrium(rho0, ux0, uy0).to(DTYPE)
        for _ in range(steps):
            r, uxx, uyy = D2Q9.macroscopic(f)
            f = stream_periodic(bgk_collision(f, r, uxx, uyy, omega))
        dense = f

        row = {"ny": ny, "nx": nx, "steps": steps, "omega": omega}
        for layout in ("aos", "soa"):
            mask = torch.ones(ny, nx, dtype=torch.bool)
            dom = build_sparse_domain(mask, periodic=(True, True))
            solver = SparseAOSLBM(dom, omega=omega, layout=layout, dtype=DTYPE)
            solver.initialize(rho0, ux0, uy0)
            for _ in range(steps):
                solver.step()
            got = solver.to_dense()
            row[f"max_abs_diff_{layout}"] = float((got - dense).abs().max())
            row[f"rel_max_diff_{layout}"] = float(
                (got - dense).abs().max() / dense.abs().max()
            )
        rows.append(row)
        print(f"[equivalence] {ny}x{nx} steps={steps} -> {row['rel_max_diff_aos']:.3e}")
    return rows


# --------------------------------------------------------------------------- 2
def analytic_poiseuille(ny: int, tau: float, force: float) -> torch.Tensor:
    """Half-way bounce-back Poiseuille profile at the fluid-node rows."""
    nu = (tau - 0.5) / 3.0
    H = ny - 2.0
    y = torch.arange(1, ny - 1, dtype=DTYPE)
    d = y - 0.5
    return force / (2 * nu) * d * (H - d)


def check_poiseuille(
    resolutions=(18, 34, 66), nx: int = 8, tau: float = 0.8, force: float = 1e-5,
    max_steps: int = 60000, tol: float = 1e-10,
) -> dict:
    """Body-force channel flow, run to steady state; compare to analytic profile."""
    rows = []
    profiles = {}
    for ny in resolutions:
        dom = build_sparse_domain(channel_mask(ny, nx), periodic=(True, True))
        solver = SparseAOSLBM(dom, omega=1.0 / tau, dtype=DTYPE)
        solver.initialize(
            torch.ones(ny, nx, dtype=DTYPE),
            torch.zeros(ny, nx, dtype=DTYPE),
            torch.zeros(ny, nx, dtype=DTYPE),
        )
        m0 = solver.total_mass().item()
        prev = torch.zeros(ny - 2, dtype=DTYPE)
        steps = 0
        converged_at = None
        for it in range(1, max_steps + 1):
            solver.step(force=(force, 0.0))
            steps = it
            if it % 200 == 0:
                _, ux, _ = solver.macroscopic()
                prof = dom.scatter(ux)[1:-1, :].mean(dim=1)
                delta = float((prof - prev).abs().max())
                prev = prof
                if delta < tol:
                    converged_at = it
                    break

        _, ux, _ = solver.macroscopic()
        ux_dense = dom.scatter(ux).clone()
        ux_dense[~dom.mask] = 0.0
        prof = ux_dense[1:-1, :].mean(dim=1)
        ref = analytic_poiseuille(ny, tau, force)
        rel_l2 = float((prof - ref).pow(2).sum().sqrt() / ref.pow(2).sum().sqrt())
        # sensitivity: naive node-centred (y measured from node 0) reference
        nu = (tau - 0.5) / 3.0
        Hn = float(ny - 1)
        yn = torch.arange(1, ny - 1, dtype=DTYPE)
        ref_node = force / (2 * nu) * yn * (Hn - yn)
        rel_l2_node = float(
            (prof - ref_node).pow(2).sum().sqrt() / ref_node.pow(2).sum().sqrt()
        )
        rows.append(
            {
                "ny": ny,
                "nx": nx,
                "n_fluid": dom.n_fluid,
                "tau": tau,
                "nu": nu,
                "force": force,
                "steps_run": steps,
                "converged_at": converged_at,
                "tol": tol,
                "rel_l2_vs_halfway_wall": rel_l2,
                "rel_l2_vs_node_centred": rel_l2_node,
                "u_max_num": float(prof.max()),
                "u_max_ref": float(ref.max()),
                "u_max_ratio": float(prof.max() / ref.max()),
                "mass_rel_drift": float(abs(solver.total_mass().item() - m0) / m0),
            }
        )
        profiles[ny] = {
            "y": [float(v) for v in torch.arange(1, ny - 1, dtype=DTYPE)],
            "u_num": [float(v) for v in prof],
            "u_ref": [float(v) for v in ref],
        }
        print(
            f"[poiseuille] ny={ny} steps={steps} conv={converged_at} "
            f"rel-L2={100*rel_l2:.3f}% u_max/u_ref={float(prof.max()/ref.max()):.4f}"
        )
    worst = max(r["rel_l2_vs_halfway_wall"] for r in rows)
    return {
        "rows": rows,
        "profiles": profiles,
        "worst_rel_l2": worst,
        "acceptance_15pct": bool(worst < 0.15),
        "target_3pct": bool(worst < 0.03),
    }


# --------------------------------------------------------------------------- 3
def check_taylor_green(ny: int = 128, nx: int = 128, omega: float = 1.9,
                       steps: int = 20000, amp: float = 0.04) -> dict:
    """Sparse fully periodic box: viscous decay vs exp(-nu k^2 t)."""
    y = torch.arange(ny, dtype=DTYPE)
    x = torch.arange(nx, dtype=DTYPE)
    kx = 2 * math.pi / nx
    ky = 2 * math.pi / ny
    ux0 = amp * torch.sin(kx * x)[None, :] * torch.cos(ky * y)[:, None]
    uy0 = -amp * torch.cos(kx * x)[None, :] * torch.sin(ky * y)[:, None]
    rho0 = torch.ones(ny, nx, dtype=DTYPE)

    dom = build_sparse_domain(torch.ones(ny, nx, dtype=torch.bool), periodic=(True, True))
    solver = SparseAOSLBM(dom, omega=omega, dtype=DTYPE)
    solver.initialize(rho0, ux0, uy0)

    nu = (1.0 / omega - 0.5) / 3.0
    k2 = kx * kx + ky * ky
    # vorticity diffusion of this single mode: lambda = nu * k^2
    # (omega_z = 2 A k sin(kx)sin(ky) e^{-lambda t} -> d_t omega = nu lap omega)
    samples = []
    u_meas, u_analytic = [], []
    t0 = time.time()
    for it in range(steps + 1):
        if it % 1000 == 0:
            _, ux, uy = solver.macroscopic()
            # amplitude of the single Fourier mode: ux = A sin(kx x) cos(ky y)
            basis = torch.sin(kx * x)[None, :] * torch.cos(ky * y)[:, None]
            a_num = float(4.0 * (dom.scatter(ux) * basis).mean())  # <sin^2 cos^2> = 1/4
            a_ref = amp * math.exp(-nu * k2 * it)
            samples.append({"step": it, "amp_num": a_num, "amp_analytic": a_ref,
                            "ratio": a_num / a_ref})
            u_meas.append(a_num)
            u_analytic.append(a_ref)
        solver.step()
    wall = time.time() - t0

    # fit decay rate from the numerical samples
    steps_arr = [s["step"] for s in samples]
    ln = [math.log(abs(s["amp_num"])) for s in samples]
    n = len(steps_arr)
    mean_s = sum(steps_arr) / n
    mean_l = sum(ln) / n
    num = sum((s - mean_s) * (l - mean_l) for s, l in zip(steps_arr, ln))
    den = sum((s - mean_s) ** 2 for s in steps_arr)
    rate_num = -num / den
    rate_ref = nu * k2
    return {
        "ny": ny,
        "nx": nx,
        "steps": steps,
        "omega": omega,
        "nu": nu,
        "k2": k2,
        "decay_rate_num": rate_num,
        "decay_rate_analytic": rate_ref,
        "decay_rate_rel_err": abs(rate_num - rate_ref) / rate_ref,
        "amplitude_ratio_first": samples[0]["ratio"],
        "amplitude_ratio_last": samples[-1]["ratio"],
        "wall_seconds": wall,
        "samples": samples,
        "series": {"steps": steps_arr, "u_num": u_meas, "u_analytic": u_analytic},
    }


# --------------------------------------------------------------------------- 4
def check_cost(ny: int = 256, nx: int = 256, ny_solid: int = 256, n_steps: int = 60,
               seed: int = 7) -> dict:
    """Memory + per-step wall clock: dense vs sparse AOS vs sparse SoA."""
    g = torch.Generator().manual_seed(seed)
    # 55 % solid "porous" mask, deterministic
    mask = torch.rand(ny, nx, generator=g) < 0.45
    mask[0, :] = mask[-1, :] = False
    dom = build_sparse_domain(mask, periodic=(True, True))
    solver_aos = SparseAOSLBM(dom, omega=1.9, layout="aos", dtype=DTYPE)
    solver_soa = SparseAOSLBM(dom, omega=1.9, layout="soa", dtype=DTYPE)
    rho0 = torch.ones(ny, nx, dtype=DTYPE)
    zero = torch.zeros(ny, nx, dtype=DTYPE)
    solver_aos.initialize(rho0, zero, zero)
    solver_soa.initialize(rho0, zero, zero)
    # perturb so the collision is not a no-op
    solver_aos.f = solver_aos.f * (1.0 + 1e-3 * torch.randn(
        solver_aos.f.shape, generator=g, dtype=DTYPE))
    solver_soa.f = solver_aos.f.t().contiguous()

    def timeit(solver):
        solver.step()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            solver.step()
        return (time.perf_counter() - t0) / n_steps

    t_aos = timeit(solver_aos)
    t_soa = timeit(solver_soa)

    # dense full-grid baseline: same BGK + periodic streaming over ALL cells
    f_dense = solver_aos.to_dense(fill=1.0)
    f_dense = f_dense + 1e-6 * torch.randn(f_dense.shape, generator=g, dtype=DTYPE)

    def dense_step(f):
        r, uxx, uyy = D2Q9.macroscopic(f)
        return stream_periodic(bgk_collision(f, r, uxx, uyy, 1.9))

    f_dense = dense_step(f_dense)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        f_dense = dense_step(f_dense)
    t_dense = (time.perf_counter() - t0) / n_steps

    n_dense = ny * nx
    bytes_dense = D2Q9.Q * n_dense * 8
    return {
        "ny": ny,
        "nx": nx,
        "fluid_fraction": dom.fluid_fraction,
        "n_fluid": dom.n_fluid,
        "n_steps_timed": n_steps,
        "state_bytes_dense": bytes_dense,
        "state_bytes_sparse": solver_aos.state_bytes,
        "table_bytes_sparse_int64": solver_aos.table_bytes,
        "table_bytes_sparse_int32_equiv": solver_aos.table_bytes // 2,
        "bytes_ratio_state_only": solver_aos.state_bytes / bytes_dense,
        "seconds_per_step_aos": t_aos,
        "seconds_per_step_soa": t_soa,
        "seconds_per_step_dense_all_cells": t_dense,
        "speedup_aos_vs_dense": t_dense / t_aos,
        "speedup_soa_vs_dense": t_dense / t_soa,
        "aos_vs_soa_ratio": t_soa / t_aos,
        "aos_soa_max_abs_diff": float(
            (solver_aos.f - solver_soa.f.t()).abs().max()
        ),
    }


# ------------------------------------------------------------------- figures
def make_figures(summary: dict) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    figs = []

    # fig1: Poiseuille profiles
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for ny, pr in summary["poiseuille"]["profiles"].items():
        ax.plot(pr["u_num"], pr["y"], "o-", ms=3, lw=1.0, label=f"sparse AOS ny={ny}")
    pr0 = list(summary["poiseuille"]["profiles"].values())[-1]
    ax.plot(pr0["u_ref"], pr0["y"], "k--", lw=1.0, label="analytic (half-way wall)")
    ax.set_xlabel("$u_x$ [lattice units]")
    ax.set_ylabel("y [lattice]")
    ax.set_title("Poiseuille channel: sparse list-based AOS vs analytic")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p1 = OUT / "fig_poiseuille_profile.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=140)
    plt.close(fig)
    figs.append({"path": p1, "caption": "Poiseuille 通道：稀疏 AOS 数值剖面 vs 半程反弹壁面解析解（ny=18/34/66）"})

    # fig2: Taylor-Green decay
    ser = summary["taylor_green"]["series"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.semilogy(ser["steps"], ser["u_num"], "o", ms=3, label="sparse AOS")
    ax.semilogy(ser["steps"], ser["u_analytic"], "k--", lw=1.0,
                label=r"$\exp(-\nu k^2 t)$")
    ax.set_xlabel("step")
    ax.set_ylabel("first-mode amplitude of $u_x$")
    ax.set_title(
        f"Taylor-Green decay: rate err {100*summary['taylor_green']['decay_rate_rel_err']:.3f}%"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    p2 = OUT / "fig_taylor_green_decay.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    figs.append({"path": p2, "caption": "Taylor-Green 衰减：稀疏 AOS 基频幅值 vs 解析 exp(-νk²t)"})

    # fig3: cost
    c = summary["cost"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    labels = ["dense\n(Q,Ny,Nx)", "sparse AOS\n(n,Q)"]
    axes[0].bar(labels, [c["state_bytes_dense"] / 1e6, c["state_bytes_sparse"] / 1e6],
                color=["#888", "#1f77b4"])
    axes[0].set_ylabel("population state [MB]")
    axes[0].set_title(f"memory, fluid fraction {c['fluid_fraction']:.2f}")
    axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(["dense", "AOS", "SoA"],
                [1e3 * c["seconds_per_step_dense_all_cells"],
                 1e3 * c["seconds_per_step_aos"],
                 1e3 * c["seconds_per_step_soa"]],
                color=["#888", "#1f77b4", "#ff7f0e"])
    axes[1].set_ylabel("ms / step")
    axes[1].set_title("wall clock per step")
    axes[1].grid(alpha=0.3, axis="y")
    p3 = OUT / "fig_cost.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=140)
    plt.close(fig)
    figs.append({"path": p3, "caption": f"存储/耗时：dense vs sparse AOS vs SoA（{c['ny']}×{c['nx']}，流体占比 {c['fluid_fraction']:.2f}）"})

    # fig4: ux field cloud of the channel (last resolution)
    return figs


def make_field_figure(ny: int = 34, nx: int = 16, tau: float = 0.8,
                      force: float = 1e-5, steps: int = 20000) -> dict:
    """Steady channel field map (ux cloud with solid cells masked)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dom = build_sparse_domain(channel_mask(ny, nx), periodic=(True, True))
    solver = SparseAOSLBM(dom, omega=1.0 / tau, dtype=DTYPE)
    solver.initialize(torch.ones(ny, nx, dtype=DTYPE),
                      torch.zeros(ny, nx, dtype=DTYPE),
                      torch.zeros(ny, nx, dtype=DTYPE))
    for _ in range(steps):
        solver.step(force=(force, 0.0))
    _, ux, uy = solver.macroscopic()
    ux_dense = dom.scatter(ux).clone()
    ux_dense[~dom.mask] = float("nan")
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    im = ax.imshow(ux_dense, origin="lower", cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="$u_x$ [lattice]")
    ax.set_title(f"sparse AOS channel field, ny={ny}, nx={nx}, steps={steps}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    p = OUT / "fig_channel_ux_field.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return {"path": p, "caption": f"稀疏 AOS 通道定常 $u_x$ 云图（实心区为壁面，Ny={ny}, Nx={nx}）"}


def _reuse(name: str):
    """Load a previously computed check from outputs/sparse_aos/parts/. """
    p = OUT / "parts" / f"{name}.json"
    if p.exists():
        print(f"[reuse] {p}")
        return json.loads(p.read_text())
    return None


def main(reuse: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "anchor": "XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM",
        "upstream_pattern": "lssfau/lbmpy sparse-list-based-aos-lbm",
        "torch_version": torch.__version__,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
    }
    parts = {}
    if reuse:
        for name in ("equivalence", "poiseuille", "taylor_green", "cost"):
            parts[name] = _reuse(name)
    summary["equivalence"] = parts.get("equivalence") or check_equivalence()
    summary["poiseuille"] = parts.get("poiseuille") or check_poiseuille()
    summary["taylor_green"] = parts.get("taylor_green") or check_taylor_green()
    summary["cost"] = parts.get("cost") or check_cost()
    figs = make_figures(summary)
    field = make_field_figure()
    figs.append(field)
    summary["figures"] = [{"file": f["path"].name, "caption": f["caption"]} for f in figs]

    (OUT / "xval_summary.json").write_text(json.dumps(summary, indent=2))
    with open(OUT / "figure_captions.json", "w") as fh:
        json.dump(summary["figures"], fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("poiseuille", "taylor_green", "cost")}, indent=2)[:4000])
    print(f"wrote {OUT/'xval_summary.json'}")
    return summary


if __name__ == "__main__":
    main(reuse="--reuse" in sys.argv)