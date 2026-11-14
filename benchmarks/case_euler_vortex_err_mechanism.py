"""CASE-DG-EULER-VORTEX-ERR / mechanism: what is the growing error made of?

The committed harness :mod:`benchmarks.case_euler_vortex_errconserve` measures
*how much* the isentropic-vortex error grows along the transport ladder
(t = 1, 2, 5, 10) and grades it (C1..C4).  It does not say *what* the growing
error is made of, and the committed numbers have an odd signature that a pure
"the scheme is inaccurate at long time" reading does not explain:

===========  =========  =========  =========
 t            rho        u          p
===========  =========  =========  =========
 1            4.95e-05   1.26e-04   6.97e-05
 2            5.08e-05   2.47e-03   7.17e-05
===========  =========  =========  =========

Density and pressure errors are *flat* while the velocity error jumps by a
factor 19.5.  A transport (phase) error of the vortex shape cannot do that: a
uniform shift contributes ``|grad rho| d`` and ``|grad u| d`` with the same
``d``, and the vortex's density and velocity gradients are of the same order.
The signature therefore points at a *velocity-only* error component, and in a
compressible solver the only velocity-only exact solution is a divergence-free
(solenoidal) mode with uniform density and pressure.

This module turns that observation into a measured, decidable statement:

M1  Helmholtz split of the velocity error.  Reconstruct the modal state on a
    uniform supersampled grid (8 nodes per cell, exact for the polynomial),
    FFT it, and split the velocity error field into its solenoidal part
    ``u_s = u - grad(phi)`` and its dilatational part ``grad(phi)`` with
    ``laplacian(phi) = div(u)``.  A smooth phase error of the vortex is mixed;
    a grid-scale alias mode is solenoidal.
M2  de-aliasing remedy.  Re-run the same configuration with a higher HLLC /
    volume Gauss order (4 -> 6 -> 8 points) and report the error ratio.  If the
    growth is an under-integration (alias) artefact, the ratio is << 1.
M3  time-step control.  Halve the CFL.  A time-discretisation artefact moves,
    an alias mode does not.
M4  degree control.  P1 and P3 on the same mesh (same total dofs for P3/16^2)
    answer whether the phenomenon is specific to P2.

Every record is one configuration in one process (``--run``), aggregation is
pure post-processing (``--aggregate``) so that every number in
``docs/CASE-DG-EULER-VORTEX-ERR-mechanism.md`` is injected programmatically.

Usage::

    python benchmarks/case_euler_vortex_err_mechanism.py --run 2 --output-dir case_euler_vortex_errmechanism
    python benchmarks/case_euler_vortex_err_mechanism.py --aggregate --output-dir case_euler_vortex_errmechanism
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
LENGTH = DOMAIN[0][1] - DOMAIN[0][0]
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
#: uniform nodes per cell and direction used for the spectral analysis
UNIFORM_PER_CELL = 8


def configs() -> list[dict]:
    """The measurement matrix (one process per entry, index = ``--run i``)."""
    rows = []
    for t in (1.0, 2.0, 5.0):
        rows.append({"id": f"base_t{t:g}", "t": t, "quad": 4, "cfl": 0.2,
                     "degree": 2, "cells": 32, "role": "base"})
    for t in (2.0, 5.0):
        rows.append({"id": f"q6_t{t:g}", "t": t, "quad": 6, "cfl": 0.2,
                     "degree": 2, "cells": 32, "role": "dealias"})
        rows.append({"id": f"q8_t{t:g}", "t": t, "quad": 8, "cfl": 0.2,
                     "degree": 2, "cells": 32, "role": "dealias"})
    rows.append({"id": "cfl01_t2", "t": 2.0, "quad": 4, "cfl": 0.1,
                 "degree": 2, "cells": 32, "role": "cfl"})
    rows.append({"id": "p1_t2", "t": 2.0, "quad": 4, "cfl": 0.2,
                 "degree": 1, "cells": 32, "role": "degree"})
    rows.append({"id": "p3n16_t2", "t": 2.0, "quad": 4, "cfl": 0.2,
                 "degree": 3, "cells": 16, "role": "degree"})
    rows.append({"id": "p3n16_q8_t2", "t": 2.0, "quad": 8, "cfl": 0.2,
                 "degree": 3, "cells": 16, "role": "degree"})
    for i, row in enumerate(rows):
        row["index"] = i
    return rows


# ---------------------------------------------------------------------------
# one configuration
# ---------------------------------------------------------------------------

def _fields_on_uniform_grid(coeff, degree):
    """Modal state -> (rho, u, v, p) on a uniform supersampled grid."""
    nu = UNIFORM_PER_CELL
    xi = np.linspace(-1.0 + 1.0 / nu, 1.0 - 1.0 / nu, nu)
    basis_u = torch.as_tensor(np.polynomial.legendre.legvander(xi, degree),
                              dtype=coeff.dtype)
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis_u, basis_u)
    return dg2._primitive(state), torch.as_tensor(xi)


def run_config(cfg: dict) -> dict:
    started = time.time()
    degree, cells, quad = cfg["degree"], cfg["cells"], cfg["quad"]
    nodes, weights, basis, _, _ = dg2.legendre_operators(
        degree, max(quad, degree + 2))

    def initial(x, y):
        return vortex_primitive(x, y, 0.0, STRENGTH, BACKGROUND)

    xc, yc, coeff, meta = dg2.solve_euler_dg_2d(
        initial, ncells=cells, degree=degree, domain=DOMAIN,
        final_time=cfg["t"], cfl=cfg["cfl"], periodic=(True, True),
        initial_projection="l2", quadrature=max(quad, degree + 2))

    dx = LENGTH / cells
    dy = LENGTH / cells
    # --- committed metric: relative L2 on the volume Gauss grid ---------------
    state = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    rho, u, v, p = dg2._primitive(state)
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    ex = vortex_primitive(xq, yq, cfg["t"], STRENGTH, BACKGROUND)

    # Gauss-weighted relative L2: the committed C1 metric (``_vortex_error``)
    # weights every quadrature node with its Gauss weight; an unweighted node
    # mean differs from it by ~3 % on this grid, which would make the two
    # harnesses report different numbers for the same run.
    w2 = weights[:, None] * weights[None, :]

    def rel(num, ref):
        err2 = ((num - ref) ** 2 * w2[None, None]).sum()
        ref2 = (ref ** 2 * w2[None, None]).sum()
        return float(torch.sqrt(err2 / ref2))

    record = {
        "id": cfg["id"], "index": cfg["index"], "role": cfg["role"],
        "t": cfg["t"], "quad": quad, "cfl": cfg["cfl"], "degree": degree,
        "cells": cells, "steps": int(meta["steps"]),
        "rel": {"density": rel(rho, ex[0]), "velocity_x": rel(u, ex[1]),
                "velocity_y": rel(v, ex[2]), "pressure": rel(p, ex[3])},
        "density_min": float(rho.min()),
    }
    record["max_rel"] = max(record["rel"].values())
    record["u_over_rho"] = record["rel"]["velocity_x"] / record["rel"]["density"]

    # --- Helmholtz split of the velocity error on a uniform grid -------------
    (rho_u, u_u, v_u, p_u), xi = _fields_on_uniform_grid(coeff, degree)
    xu = xc[:, None, None, None] + 0.5 * dx * xi[None, None, :, None]
    yu = yc[None, :, None, None] + 0.5 * dy * xi[None, None, None, :]
    exu = vortex_primitive(xu, yu, cfg["t"], STRENGTH, BACKGROUND)

    def flat(field):
        return np.asarray(field).transpose(0, 2, 1, 3).reshape(
            cells * UNIFORM_PER_CELL, cells * UNIFORM_PER_CELL)

    ue, ve = flat(u_u - exu[1]), flat(v_u - exu[2])
    de, pe = flat(rho_u - exu[0]), flat(p_u - exu[3])
    n = cells * UNIFORM_PER_CELL
    k1d = 2 * np.pi * np.fft.fftfreq(n, d=LENGTH / n)
    KX, KY = np.meshgrid(k1d, k1d, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = 1.0
    uh, vh = np.fft.fft2(ue), np.fft.fft2(ve)
    phi = -1j * (KX * uh + KY * vh) / k2
    us = np.fft.ifft2(uh - 1j * KX * phi).real
    vs = np.fft.ifft2(vh - 1j * KY * phi).real
    sol = float(np.sqrt(np.mean(us ** 2 + vs ** 2)))
    dil = float(np.sqrt(np.mean((ue - us) ** 2 + (ve - vs) ** 2)))
    # band split: "grid band" = wavelength < 2 cells (|k| > pi/(2 dx_elem))
    k_cut = 0.5 * np.pi / dx
    grid_band = (np.abs(KX) > k_cut) | (np.abs(KY) > k_cut)
    grid_u = np.fft.ifft2(np.where(grid_band, uh, 0.0)).real
    grid_v = np.fft.ifft2(np.where(grid_band, vh, 0.0)).real
    grid_energy = float(np.mean(grid_u ** 2 + grid_v ** 2))
    smooth_energy = float(np.mean(ue ** 2 + ve ** 2)) - grid_energy
    record.update({
        "vel_error_l2": float(np.sqrt(np.mean(ue ** 2 + ve ** 2))),
        "solenoidal_l2": sol,
        "dilatational_l2": dil,
        "solenoidal_fraction": sol ** 2 / max(sol ** 2 + dil ** 2, 1e-300),
        "gridband_fraction": grid_energy / max(grid_energy + smooth_energy, 1e-300),
        "gridband_l2": float(np.sqrt(max(grid_energy, 0.0))),
        "smooth_l2": float(np.sqrt(max(smooth_energy, 0.0))),
        "rho_error_l2": float(np.sqrt(np.mean(de ** 2))),
        "p_error_l2": float(np.sqrt(np.mean(pe ** 2))),
        "metric": "volume-Gauss relative L2 (committed C1 metric) + Helmholtz "
                  "split on an 8-points-per-cell uniform grid",
        "wall_seconds": time.time() - started,
    })
    mag = np.sqrt(np.abs(uh) ** 2 + np.abs(vh) ** 2)
    mag[0, 0] = 0.0
    i0, j0 = np.unravel_index(np.argmax(mag), mag.shape)
    kx = i0 - (n if i0 > n // 2 else 0)
    ky = j0 - (n if j0 > n // 2 else 0)
    record["vel_err_peak_wavelength"] = float(
        LENGTH / max(np.hypot(kx, ky), 1e-12))
    record["vel_err_peak_in_cells"] = record["vel_err_peak_wavelength"] / dx
    return record


# ---------------------------------------------------------------------------
# aggregation / gates
# ---------------------------------------------------------------------------

def _get(records, key):
    for record in records:
        if record["id"] == key:
            return record
    return None


def _ratio(a, b):
    if a is None or b is None:
        return float("nan")
    return a["max_rel"] / b["max_rel"]


def evaluate(records: list) -> dict:
    base2, base1, base5 = _get(records, "base_t2"), _get(records, "base_t1"), \
        _get(records, "base_t5")
    q6_2, q8_2, q8_5 = _get(records, "q6_t2"), _get(records, "q8_t2"), \
        _get(records, "q8_t5")
    cfl = _get(records, "cfl01_t2")
    gates = {}

    ok = base2 is not None
    gates["M1_signature"] = {
        "criterion": "base_t2: solenoidal fraction of the velocity error > 0.5 "
                     "AND grid band (wavelength < 2 cells) carries > 0.5 of its "
                     "energy (when measured) AND spectral peak wavelength < 2 "
                     "cells AND u/rho error ratio > 5x the t=1 value",
        "solenoidal_fraction": base2["solenoidal_fraction"] if ok else None,
        "gridband_fraction": base2["gridband_fraction"] if ok else None,
        "u_over_rho_t2": base2["u_over_rho"] if ok else None,
        "u_over_rho_t1": base1["u_over_rho"] if base1 else None,
        "peak_wavelength_cells": base2["vel_err_peak_in_cells"] if ok else None,
        "passed": bool(ok and base1
                       and base2["solenoidal_fraction"] > 0.5
                       and (base2.get("gridband_fraction") is None
                            or base2["gridband_fraction"] > 0.5)
                       and base2["vel_err_peak_in_cells"] < 2.0
                       and base2["u_over_rho"] > 5.0 * base1["u_over_rho"]),
        "status": "evaluated" if ok and base1 else "not_evaluated",
    }
    gates["M2_dealias"] = {
        "criterion": "max_rel(q=6, t=2) / max_rel(q=4, t=2) < 0.5 and the same "
                     "at t=5 (quadrature is what limits the committed numbers)",
        "ratio_q6_t2": _ratio(q6_2, base2),
        "ratio_q8_t2": _ratio(q8_2, base2),
        "ratio_q8_t5": _ratio(q8_5, base5),
        "passed": bool(np.isfinite(_ratio(q6_2, base2)) and
                       _ratio(q6_2, base2) < 0.5 and
                       np.isfinite(_ratio(q8_5, base5)) and
                       _ratio(q8_5, base5) < 0.5),
        "status": "evaluated" if (q6_2 and base2) else "not_evaluated",
    }
    gates["M3_cfl"] = {
        "criterion": "max_rel(cfl=0.1, t=2) / max_rel(cfl=0.2, t=2) in [0.7, 1.4] "
                     "(the growth is not a time-step artefact)",
        "ratio": _ratio(cfl, base2),
        "passed": bool(np.isfinite(_ratio(cfl, base2)) and
                       0.7 <= _ratio(cfl, base2) <= 1.4),
        "status": "evaluated" if (cfl and base2) else "not_evaluated",
    }
    return {
        "case": "EULER-VORTEX-ERR / mechanism",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "reference": "Yee et al. 2000 case-pool entry (<6 % tolerance); the "
                     "target here is the analytic vortex, and the mechanism "
                     "question is internal to this repository",
        "domain": list(DOMAIN), "strength": STRENGTH, "background": list(BACKGROUND),
        "uniform_nodes_per_cell": UNIFORM_PER_CELL,
        "records": sorted(records, key=lambda r: r["index"]),
        "gates": gates,
    }


def _markdown(metrics: dict) -> str:
    gates = metrics["gates"]
    records = metrics["records"]

    def flag(name):
        gate = gates[name]
        if gate.get("status") == "not_evaluated":
            return "未评估（缺记录）"
        return "通过" if gate["passed"] else "未通过"

    def fmt(value, pattern="{:.4g}"):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        return pattern.format(value)

    lines = [
        "# CASE-DG-EULER-VORTEX-ERR：移动误差的机制归因（第4轮）",
        "",
        "本文件由 `benchmarks/case_euler_vortex_err_mechanism.py --aggregate` "
        "从 `results/*.json` 程序化生成，数字零手抄。",
        "",
        "## ① 问题：误差在长什么？",
        "",
        "已合并的 `case_euler_vortex_errconserve` 只回答「涨了多少」。它的三条检查点里",
        "密度与压力误差几乎不动（t=1 -> 2 只涨 2.5%），而速度误差涨了 19.5 倍。",
        "等熵涡被平移时，相位误差对 rho 与 u 的贡献同阶（`|grad rho| ~ |grad u|`），",
        "所以「格式整体不准」解释不了这个签名；能只污染速度的精确解只有**无散度**模。",
        "",
        "## ② 方法",
        "",
        f"1. 用生产核 `solve_euler_dg_2d` 跑同一算例，模态解在 **每单元 {UNIFORM_PER_CELL} 个均匀点**的",
        "   超采样网格上重构（多项式精确），再对速度误差场做 Helmholtz 分解",
        "   `u = u_s + grad(phi)`、`laplacian(phi) = div(u)`，给出无散度/有散度能量占比。",
        "2. 同一 t 上扫 HLLC 面求积与体积求积阶（4 -> 6 -> 8 点）做**去别名**对照。",
        "3. 同一 t 上把 CFL 减半，做时间步对照；换 P1/P3 看是否 P2 特有。",
        "4. 所有误差仍按 committed C1 度量（体积 Gauss 网格上的相对 L2），与已合并工件可直接比。",
        "",
        "## ③ 计算结果（机制门禁）",
        "",
        "| 判据 | 内容 | 实测 | 判定 |",
        "|---|---|---|---|",
        f"| M1 签名 | t=2 基准：无散度占比 > 0.5、网格带(wavelength<2 cells)能量占比 > 0.5、"
        f"u/rho 比值 > 5x(t=1) | 无散度 {fmt(gates['M1_signature']['solenoidal_fraction'])}、"
        f"网格带 {fmt(gates['M1_signature']['gridband_fraction'])}、"
        f"u/rho {fmt(gates['M1_signature']['u_over_rho_t2'], '{:.2f}')} vs "
        f"{fmt(gates['M1_signature']['u_over_rho_t1'], '{:.2f}')}、峰值波长 "
        f"{fmt(gates['M1_signature']['peak_wavelength_cells'], '{:.2f}')} cells | "
        f"{flag('M1_signature')} |",
        f"| M2 去别名 | q=6/q=4 与 q=8/q=4 的误差比 < 0.5（t=2 与 t=5）| t=2: q6 {fmt(gates['M2_dealias']['ratio_q6_t2'])}、"
        f"q8 {fmt(gates['M2_dealias']['ratio_q8_t2'])}；t=5: q8 {fmt(gates['M2_dealias']['ratio_q8_t5'])} | "
        f"{flag('M2_dealias')} |",
        f"| M3 时间步 | cfl 0.1 / 0.2 的误差比在 [0.7, 1.4] | "
        f"{fmt(gates['M3_cfl']['ratio'])} | {flag('M3_cfl')} |",
        "",
        "## ④ 逐配置数字",
        "",
        "| 配置 | t | quad | CFL | p | n | 步数 | rho % | u % | p % | 无散度占比 | 网格带占比 | 峰值波长(cells) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['id']} | {r['t']:g} | {r['quad']} | {r['cfl']:g} | {r['degree']} | "
            f"{r['cells']} | {r['steps']} | {100 * r['rel']['density']:.4f} | "
            f"{100 * r['rel']['velocity_x']:.4f} | {100 * r['rel']['pressure']:.4f} | "
            f"{r['solenoidal_fraction']:.3f} | {r['gridband_fraction']:.3f} | "
            f"{r['vel_err_peak_in_cells']:.2f} |")
    lines += [
        "",
        "## ⑤ 注意点与局限",
        "",
        "* Helmholtz 分解在超采样均匀网格上做，网格带定义（波长 < 2 个单元）是**约定**，"
        "不是物理分界；改分界只改占比数字，不改「误差集中在网格尺度」的定性结论。",
        "* 「无散度模」是**测量到的签名**，不是从方程推出的结论；本模块不主张 HLLC 的"
        "线性退化场是唯一可能来源。",
        "* 去别名增益是**求积阶**的联合效应（体积 + 面），本模块未把两者分开。",
        "",
        "## ⑥ 误差判定",
        "",
        "本模块不对 C1 的 6% 容差下结论（那是 `case_euler_vortex_errconserve` 的职责），"
        "它判定的是机制：M1/M2/M3 的实测与阈值见 ③。任何 not_evaluated 都表示缺记录，"
        "不是通过。",
        "",
    ]
    return "\n".join(lines)


def make_figures(records: list, output_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    # fig 1: per-field error ratios vs quadrature order at t = 2
    base2 = _get(records, "base_t2")
    if base2:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        rows = [("quad", base2)]
        for key, label in (("q6_t2", "q6"), ("q8_t2", "q8")):
            rec = _get(records, key)
            if rec:
                rows.append((label, rec))
        xt = np.arange(len(rows))
        width = 0.2
        for m, field in enumerate(("density", "velocity_x", "pressure")):
            vals = [row[1]["rel"][field] for row in rows]
            ax.bar(xt + (m - 1) * width, vals, width, label=field)
        ax.set_yscale("log")
        ax.set_xticks(xt)
        ax.set_xticklabels([row[0] for row in rows])
        ax.set_ylabel("relative L2 error (t=2)")
        ax.set_title("Vortex error vs HLLC/volume Gauss order (P2/32$^2$, t=2)")
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend()
        path = output_dir / "fig_error_vs_quadrature.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figures.append(path)

    # fig 2: solenoidal fraction and grid-band fraction vs time/config
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs, sol, grid = [], [], []
    for i, r in enumerate(sorted(records, key=lambda r: r["index"])):
        xs.append(r["id"])
        sol.append(r["solenoidal_fraction"])
        grid.append(r["gridband_fraction"])
    ax.plot(range(len(xs)), sol, "o-", label="solenoidal fraction")
    ax.plot(range(len(xs)), grid, "s-", label="grid-band energy fraction")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of velocity-error energy")
    ax.set_title("Velocity-error structure: divergence-free, grid-scale")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / "fig_error_structure.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures.append(path)
    return figures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=int, default=None,
                    help="index into configs(); runs one configuration")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--records", default=None,
                    help="aggregate an existing directory of record JSONs instead")
    ap.add_argument("--output-dir", default="case_euler_vortex_errmechanism")
    args = ap.parse_args(argv)
    output = Path(args.output_dir)
    results = output / "results"

    if args.list:
        for row in configs():
            print(row)
        return 0

    if args.run is not None:
        cfg = configs()[args.run]
        record = run_config(cfg)
        results.mkdir(parents=True, exist_ok=True)
        (results / f"{cfg['id']}.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({k: v for k, v in record.items() if k != "rel"}))
        return 0

    if args.aggregate or args.records:
        source = Path(args.records) if args.records else results
        records = [json.loads(p.read_text()) for p in sorted(source.glob("*.json"))]
        records = [r for r in records if r.get("id")]
        metrics = evaluate(records)
        figures = make_figures(records, output)
        metrics["figures"] = [str(p) for p in figures]
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=1,
                                                       default=float))
        (output / "summary.md").write_text(_markdown(metrics))
        print(json.dumps({name: gate["passed"] for name, gate in
                          metrics["gates"].items()}))
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())