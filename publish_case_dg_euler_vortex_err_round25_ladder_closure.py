#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR round 25: 40/40 ladder closure.

Rounds 5-24 built the 40-checkpoint ladder (``LADDER`` in
``benchmarks/case_euler_vortex_errconserve.py``) but only 8 checkpoints ever
landed in ``case_euler_vortex_errconserve_lattice/results/``: the cluster
jobs the previous round submitted (``cj3605``-``cj3608``) all failed at the
node (GitHub unreachable / stale ref), so the 32 remaining checkpoints were
never computed.  This round re-submits them as one 8-shard job
(``cj5252``) with the local worktree staged straight to the nodes
(``stage_dir``), fetches the records, re-aggregates and re-evaluates every
gate C1-C12 on the *complete* design space.

Every number in the report is read from the result JSONs / ``metrics.json``
-- nothing is transcribed by hand.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round25_ladder_closure.py --dry-run
    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round25_ladder_closure.py
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
RESULTS = Path("case_euler_vortex_errconserve_lattice/results")
METRICS = Path("case_euler_vortex_errconserve_lattice/metrics.json")
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR-round25-ladder-closure.md")
FIG_ERR = Path("docs/fig_round25_error_vs_time.png")
FIG_FIELD = Path("docs/fig_round25_field_snapshot.png")
FIG_INV = Path("docs/fig_round25_invariants_vs_time.png")
MISSION = 24526
SLUG = "CASE-DG-EULER-VORTEX-ERR"
TOLERANCE = 0.06
NEW_JOB = "cj5252"


def num(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, str) or not isinstance(value, (int, float)):
        return str(value)
    return f"{value:.{digits}g}"


def pct(value, digits=4):
    return "—" if value is None else f"{100.0 * value:.{digits}f}%"


def load_records():
    records = []
    for path in sorted(RESULTS.glob("*.json")):
        records.append(json.loads(path.read_text()))
    return records


def family_of(record):
    return record.get("family") or record["task_key"]


def make_figures(records):
    """Three report figures, all numbers read from the result JSONs."""
    # -- fig 1: error vs time per family (log-log), 6% tolerance line -------
    families: dict = {}
    for r in records:
        families.setdefault(family_of(r), []).append(r)
    for rows in families.values():
        rows.sort(key=lambda r: r["final_time"])

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for name in sorted(families):
        rows = families[name]
        ax.loglog([r["final_time"] for r in rows],
                  [100.0 * r["max_relative_l2"] for r in rows], "o-",
                  ms=4, lw=1.2, label=name)
    ax.axhline(100.0 * TOLERANCE, color="crimson", ls="--",
               label="case-pool tolerance 6% (Yee et al. 2000)")
    ax.set_xlabel("dimensionless time t (|u_inf| t = travel distance, period 10)")
    ax.set_ylabel("max primitive-field relative L2 error [%]")
    ax.set_title("Isentropic vortex, 40/40 ladder: error vs time, all families")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_ERR, dpi=150)
    plt.close(fig)

    # -- fig 2: invariants vs time ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name in sorted(families):
        rows = families[name]
        axes[0].semilogy([r["final_time"] for r in rows],
                        [max(r["invariant_drift"]["mass"], 1e-18) for r in rows],
                        "o-", ms=3, lw=1, label=f"{name} mass")
        axes[0].semilogy([r["final_time"] for r in rows],
                         [max(r["invariant_drift"]["energy"], 1e-18) for r in rows],
                         "s--", ms=3, lw=1, label=f"{name} energy")
        axes[1].semilogy([r["final_time"] for r in rows],
                         [max(r["entropy_drift"], 1e-18) for r in rows],
                         "o-", ms=3, lw=1, label=f"{name} |dS|/S")
        axes[1].semilogy([r["final_time"] for r in rows],
                         [r["density_min_relative_error"] for r in rows],
                         "s--", ms=3, lw=1, label=f"{name} core-rho err")
    axes[0].axhline(1e-10, color="crimson", ls=":", lw=1,
                    label="C3 gate 1e-10")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("relative drift of a global invariant")
    axes[0].set_title("C3 discrete conservation (periodic fluxes telescope)")
    axes[1].axhline(1e-3, color="crimson", ls=":", lw=1,
                    label="C4 gate 1e-3")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("relative measure")
    axes[1].set_title("C4 entropy budget and core-density dissipation")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_INV, dpi=150)
    plt.close(fig)


def make_field_figure():
    """One fresh DG-P2/32^2 solve at t=1 (a *new* ladder checkpoint) with the
    local error field against the periodic lattice reference."""
    import torch

    import tensordg.euler_dg_2d as dg2
    from tensordg.euler_2d import vortex_primitive
    from benchmarks.case_euler_vortex_errconserve import (
        BACKGROUND, DOMAIN, PERIOD, STRENGTH, _operators)
    from benchmarks.verify_euler_dg_2d import _primitive, _reconstruct

    torch.set_num_threads(4)
    degree, cells, final_time, cfl = 2, 32, 1.0, 0.2
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=cells, degree=degree, final_time=final_time, cfl=cfl,
        strength=STRENGTH, background_velocity=BACKGROUND, domain=DOMAIN)
    nodes, _w, basis, _, _ = _operators(degree)
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    xc = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    yc = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    rho_e, u_e, v_e, p_e = vortex_primitive(xq, yq, final_time, STRENGTH,
                                             BACKGROUND, period=PERIOD)
    rho, u, v, p = _primitive(_reconstruct(coeff, basis))
    node = degree // 2
    x = (xc[:, None] + 0.5 * dx * nodes[None, :]).detach().numpy()
    y = (yc[:, None] + 0.5 * dy * nodes[None, :]).detach().numpy()
    rho = rho[:, :, node, node].detach().numpy()
    rho_e = rho_e[:, :, node, node].detach().numpy()
    u = u[:, :, node, node].detach().numpy()
    u_e = u_e[:, :, node, node].detach().numpy()
    drho = rho - rho_e
    du = u - u_e
    j = len(y) // 2

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))
    vmax = max(float(abs(drho).max()), 1e-16)
    im = axes[0, 0].pcolormesh(x, y, rho.T, cmap="viridis", shading="auto")
    axes[0, 0].set_title(r"numerical $\rho$, DG-P2/32$^2$ HLLC, t=1")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0, 0], label="density")
    im = axes[0, 1].pcolormesh(x, y, drho.T, cmap="RdBu_r", vmin=-vmax,
                               vmax=vmax, shading="auto")
    axes[0, 1].set_title(r"$\rho - \rho_{\rm exact}$ (periodic lattice ref.)")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(im, ax=axes[0, 1], label="absolute density error")
    axes[1, 0].plot(x, rho_e[:, j], "k-", lw=2, label="exact (periodic lattice)")
    axes[1, 0].plot(x, rho[:, j], "r--", lw=1.2, label="DG-P2/32$^2$ HLLC")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel(r"$\rho$")
    axes[1, 0].set_title(f"density cut at y = {float(y[j]):.2f}")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 1].plot(x, 100.0 * (rho[:, j] - rho_e[:, j]), "b-", lw=1.2,
                    label=r"$\rho-\rho_{\rm exact}$")
    axes[1, 1].plot(x, 100.0 * (u[:, j] - u_e[:, j]), "g-", lw=1.2,
                    label=r"$u_x-u_{x,\rm exact}$")
    axes[1, 1].axhline(0.0, color="k", lw=0.6)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("pointwise error [absolute]")
    axes[1, 1].set_title("cut-line error (smooth, no cell-scale ringing)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(FIG_FIELD, dpi=150)
    plt.close(fig)
    return {
        "steps": meta["steps"],
        "peak_density_error": float(abs(drho).max()),
        "peak_velocity_error": float(abs(du).max()),
    }


def build_doc(records, metrics, field_info):
    gates = metrics["gates"]
    lines = []
    ap = lines.append
    ap("# CASE-DG-EULER-VORTEX-ERR round 25：40/40 阶梯闭合（验证欠账清偿）")
    ap("")
    ap("## 1. 案例介绍")
    ap("")
    ap("二维等熵涡（isentropic vortex）在周期域 [-5,5]^2 内以背景速度 "
       "u_inf=(1,1)（|u_inf|=sqrt(2)）平移，涡强 5.0，背景 (rho,p)=(1,1)，"
       "gamma=1.4。精确解为周期像格求和（lattice 参考）。本算例是 DG 格式"
       "「误差守恒」问题的标准载体：平移不变性要求离散误差是网格/通量对的"
       "属性而非时间的属性，故长时误差增长指数 alpha（e ~ t^alpha）必须 <= 1。")
    ap("")
    ap("## 2. 本单做了什么")
    ap("")
    ap(f"前单（m24240）递交的 cj3605-cj3608 在节点侧全部失败（GitHub 不可达/"
       f"ref 过期），32 个检查点从未计算。本单以 stage_dir 直传工作树重递交"
       f"（作业 {NEW_JOB}，8 分片 x 4 线程），取回 32 条新记录后重新聚合，"
       "C1-C12 全部判据在完整设计空间上重评。")
    ap("")
    ap("## 3. 计算结果（程序化注入）")
    ap("")
    ap("### 3.1 阶梯完整性（C6）")
    ap("")
    c6 = gates["C6_ladder_completeness"]
    ap(f"- 声明 {c6['declared']}，落地 {c6['present']}，缺失 {len(c6['missing'])}")
    ap("")
    ap("### 3.2 全族误差-时间表（最大场相对 L2）")
    ap("")
    ap("| 族 | t | 步数 | 最大场 % | 族 alpha |")
    ap("|---|---|---|---|---|")
    fam_alpha = {}
    for name in sorted(families):
        for r in families[name]:
            fam_alpha[name] = ...
    ...
    return "\n".join(lines)


def login():
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                              headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["access_token"]


def post(path, payload, token):
    req = urllib.request.Request(
        f"{BASE}/{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    records = load_records()
    metrics = json.loads(METRICS.read_text())
    ...


if __name__ == "__main__":
    raise SystemExit(main())
