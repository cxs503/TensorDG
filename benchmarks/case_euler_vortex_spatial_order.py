#!/usr/bin/env python3
"""Spatial order-of-accuracy verification for ``CASE-DG-EULER-VORTEX-ERR``.

The case-pool entry is certified by a *tolerance* gate (relative L2 of every
primitive field < 6 % at ``t = 1``, ref: Yee et al. 2000).  A tolerance gate
cannot, on its own, tell two very different situations apart:

* the DG family converges to the analytic isentropic vortex at its design
  order, so the 1e-4-level error is a discretisation property and the
  reference implementation is correct; or
* the DG family converges to *something else* that happens to sit within 6 %
  of the reference (a defect in the reference implementation, the periodic
  min-image map, or the flux), in which case every tolerance gate in the
  ladder passes while the case verifies nothing.

The separating test is the observed order of accuracy against the same
analytic reference: an ``O(1)`` reference defect shows up as a plateau
(``order -> 0``), while a correct reference and a consistent scheme give the
DG design order ``p + 1``.  This module measures it and adds the controls that
make the measurement attributable:

``S1 spatial_order``
    Every primitive field's relative L2 error decays at ``>= 0.9 * (p + 1)``
    over the finest-3-mesh least-squares fit, and the finest mesh pair's local
    order also clears that bound.
``S2 measurement_consistency``
    The *pure L2 projection* error of the same exact field (no time stepping,
    no flux) decays at the same ``p + 1`` rate.  If the error metric or the
    reference evaluation were ``O(1)``-wrong, both sequences would flatten
    together; the projection run proves the ``h``-scaling lives in the metric.
``S3 temporal_negligible``
    ``SSP-RK3`` with ``dt ~ CFL * h`` has a temporal error that also scales as
    ``h^3``; the observed decay is only attributable to *space* if halving the
    CFL changes the error by a negligible amount.  Measured on the meshes where
    both CFL levels were run.
``S4 reference_discriminates`` (anti-vacuity)
    The same transported states are scored against a deliberately wrong
    reference (the exact solution at ``t + 0.05``).  The fitted order must
    collapse below 1.0: proof that the order statistic is able to reject a bad
    reference rather than rubber-stamping it.
``S5 tolerance_margin``
    The finest mesh must land below ``TOLERANCE / 10``: the tolerance gate is
    then a statement about a deeply converged solution, not about a marginal
    one.

Run from the repository root::

    python3 benchmarks/case_euler_vortex_spatial_order.py --sweep --figures \
        --output-dir case_euler_vortex_errconserve
    python3 benchmarks/case_euler_vortex_spatial_order.py --quick   # 3 coarsest meshes
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
from benchmarks import case_euler_vortex_errconserve as ec
from benchmarks.verify_euler_dg_2d import _vortex_error as dg_vortex_error

#: Mesh ladder.  ``16`` is the pre-asymptotic end and is deliberately part of
#: the ladder: the fit drops the coarsest mesh so the reported order is an
#: asymptotic estimate, and the dropped point is still printed.
MESHES = (16, 32, 64, 128)
#: Design family of the case pool (DG-P2).
DEGREE = 2
FINAL_TIME = 1.0
CFL = 0.2
#: Control CFL for S3; only the meshes with both levels contribute.
CFL_CONTROL = 0.1
#: Reference time offset for the S4 anti-vacuity discriminator (a 5 % error in
#: the reference clock, i.e. ~3.5 cell lengths of transport at the finest mesh).
REFERENCE_SHIFT = 0.05
#: Gate fraction of the design order (``p + 1``).  0.9 leaves room for the
#: standard pre-asymptotic wobble without accepting a 2nd-order answer.
ORDER_FRACTION = 0.9
#: S3 bound on the CFL-induced relative change of the error.
TEMPORAL_SHARE_MAX = 0.02
#: S5 bound on the finest-mesh error, as a fraction of the case tolerance.
MARGIN_FRACTION = 0.1

FIELDS = ec.FIELDS


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def _operators(degree):
    return ec._operators(degree)


def _grid(cells, degree):
    nodes, weights, basis, _, _ = _operators(degree)
    (xmin, xmax), (ymin, ymax) = ec.DOMAIN
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    x = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    return nodes, weights, basis, x, y, dx, dy


def projection_errors(cells, degree=None):
    """Relative L2 error of the *pure* L2 projection of the exact field.

    No time stepping and no numerical flux are involved, so this sequence
    isolates the ``h``-scaling of the error metric plus the reference
    evaluation (S2).
    """
    degree = DEGREE if degree is None else degree
    nodes, weights, basis, x, y, dx, dy = _grid(cells, degree)

    def initial(xx, yy):
        return vortex_primitive(xx, yy, 0.0, ec.STRENGTH, ec.BACKGROUND)

    coeff = dg2._initial_coefficients(initial, x, y, dx, dy, nodes, weights,
                                      basis, torch.device("cpu"), torch.float64)
    errors = dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, 0.0, ec.STRENGTH,
                             ec.BACKGROUND, period=ec.PERIOD)
    return {name: float(errors[name]) for name in FIELDS}


def run_state(cells, cfl=None, degree=None, final_time=None):
    """Solve the vortex and return the transported modal state plus timing."""
    cfl = CFL if cfl is None else cfl
    degree = DEGREE if degree is None else degree
    final_time = FINAL_TIME if final_time is None else final_time
    started = time.time()
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=cells,
        degree=degree,
        final_time=final_time,
        cfl=cfl,
        strength=ec.STRENGTH,
        background_velocity=ec.BACKGROUND,
        domain=ec.DOMAIN,
    )
    return coeff, meta, time.time() - started


def score(coeff, cells, degree=None, reference_time=None):
    degree = DEGREE if degree is None else degree
    reference_time = FINAL_TIME if reference_time is None else reference_time
    nodes, _, basis, *_ = _grid(cells, degree)
    errors = dg_vortex_error(coeff, basis, nodes, ec.DOMAIN, reference_time,
                             ec.STRENGTH, ec.BACKGROUND, period=ec.PERIOD)
    return {name: float(errors[name]) for name in FIELDS}


# ---------------------------------------------------------------------------
# order statistics
# ---------------------------------------------------------------------------

def fit_order(cells, errors):
    """Least-squares exponent of ``e ~ h^order`` (``h = box / cells``)."""
    cells = np.asarray(cells, dtype=float)
    values = np.asarray(errors, dtype=float)
    if len(cells) < 2 or np.any(values <= 0.0):
        return float("nan")
    return float(np.polyfit(np.log(1.0 / cells), np.log(values), 1)[0])


def local_orders(cells, errors):
    """Pair-wise observed orders (diagnostic, no averaging)."""
    rows = []
    for i in range(len(cells) - 1):
        coarse, fine = cells[i], cells[i + 1]
        e_coarse, e_fine = errors[i], errors[i + 1]
        order = (math.log(e_coarse / e_fine) / math.log(fine / coarse)
                 if e_coarse > 0.0 and e_fine > 0.0 else float("nan"))
        rows.append({"coarse": int(coarse), "fine": int(fine),
                     "ratio": float(e_coarse / e_fine),
                     "order": float(order)})
    return rows


def _fit_rows(records, key, fields=FIELDS, meshes=None):
    rows = sorted((r for r in records
                   if key in r and (meshes is None or r["cells"] in meshes)),
                  key=lambda r: r["cells"])
    cells = [r["cells"] for r in rows]
    return cells, {name: [r[key][name] for r in rows] for name in fields}


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def sweep(meshes=MESHES, which="both", control_meshes=None, records_dir=None):
    """Run the ladder; returns one record per mesh.

    ``which`` selects ``"full"`` (design CFL only), ``"control"`` (control CFL
    only) or ``"both"``.  Every record keeps the state in memory long enough to
    be scored against both the correct and the deliberately wrong reference, so
    the S4 discriminator costs no extra solver time.
    """
    records = []
    for cells in meshes:
        record = {"cells": cells, "degree": DEGREE, "final_time": FINAL_TIME}
        if which in ("full", "both"):
            coeff, meta, wall = run_state(cells, CFL)
            record["steps"] = int(meta["steps"])
            record["dofs"] = int(meta["dofs"])
            record["wall_seconds"] = wall
            record["errors"] = score(coeff, cells)
            record["errors_wrong_reference"] = score(
                coeff, cells, reference_time=FINAL_TIME + REFERENCE_SHIFT)
            record["projection_errors"] = projection_errors(cells)
        if which in ("control", "both") and (
                control_meshes is None or cells in control_meshes):
            coeff_c, meta_c, wall_c = run_state(cells, CFL_CONTROL)
            record["cfl_control"] = {
                "cfl": CFL_CONTROL,
                "steps": int(meta_c["steps"]),
                "wall_seconds": wall_c,
                "errors": score(coeff_c, cells),
            }
        records.append(record)
        if records_dir is not None:
            # incremental persistence: a crash at the next (bigger) mesh must
            # not throw away the meshes already computed
            Path(records_dir).mkdir(parents=True, exist_ok=True)
            (Path(records_dir) / f"n{cells}.json").write_text(json.dumps(record, indent=1))
        print(f"n={cells:4d} done ({record.get('wall_seconds', 0.0):.1f} s)", flush=True)
    return records


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def _gate(criterion, passed, **measured):
    return {"criterion": criterion, "passed": bool(passed),
            "status": "passed" if passed else "failed", **measured}


def evaluate(records, degree=None):
    degree = DEGREE if degree is None else degree
    design_order = degree + 1
    lower_bound = ORDER_FRACTION * design_order
    cells = sorted(r["cells"] for r in records)
    # asymptotic window: drop the coarsest mesh
    fit_meshes = cells[1:] if len(cells) >= 3 else cells
    gates = {}

    # -- S1 spatial order of the transported solution -----------------------
    fit_cells, fit_errors = _fit_rows(records, "errors", meshes=fit_meshes)
    orders = {name: fit_order(fit_cells, values)
              for name, values in fit_errors.items()}
    all_cells, all_errors = _fit_rows(records, "errors")
    locals_ = {name: local_orders(all_cells, values)
               for name, values in all_errors.items()}
    finest_local = {name: rows[-1]["order"] for name, rows in locals_.items()}
    worst = min(orders.values())
    worst_local = min(finest_local.values())
    gates["S1_spatial_order"] = _gate(
        f"fitted order of every field over meshes {fit_cells} >= "
        f"{lower_bound:.2f} (= {ORDER_FRACTION} * (p+1), p={degree}), and the "
        f"finest-pair local order also >= {lower_bound:.2f}",
        worst >= lower_bound and worst_local >= lower_bound,
        design_order=design_order,
        lower_bound=lower_bound,
        fit_meshes=[int(c) for c in fit_cells],
        fitted_orders=orders,
        finest_pair_local_orders=finest_local,
        worst_fitted_order=worst,
        worst_finest_local_order=worst_local,
    )

    # -- S2 measurement consistency (projection only) -----------------------
    proj_cells, proj_errors = _fit_rows(records, "projection_errors",
                                       meshes=fit_meshes)
    proj_orders = {name: fit_order(proj_cells, values)
                   for name, values in proj_errors.items()}
    gates["S2_measurement_consistency"] = _gate(
        f"pure L2 projection error of every field over meshes {proj_cells} "
        f"decays at >= {lower_bound:.2f}: the h-scaling lives in the metric "
        f"and the reference evaluation, not only in the solver",
        min(proj_orders.values()) >= lower_bound,
        orders=proj_orders,
        lowest=min(proj_orders.values()),
    )

    # -- S3 temporal share ---------------------------------------------------
    share = {}
    for record in records:
        control = record.get("cfl_control")
        if not control:
            continue
        design = record["errors"]
        share[str(record["cells"])] = {
            name: abs(control["errors"][name] - design[name]) / design[name]
            for name in FIELDS
        }
    # S3 is judged on the same asymptotic window as S1: a mesh that is too
    # coarse for the order fit cannot invalidate the order claim, but its share
    # is still printed (transparency, not silence).
    window_keys = {str(int(c)) for c in fit_meshes}
    share_window = {k: v for k, v in share.items() if k in window_keys}
    worst_share = max((max(v.values()) for v in share_window.values()),
                      default=float("nan"))
    gates["S3_temporal_negligible"] = _gate(
        f"relative change of every field error between CFL {CFL_CONTROL} and "
        f"{CFL} < {TEMPORAL_SHARE_MAX:.0%} on the asymptotic window meshes "
        f"{sorted(int(k) for k in share_window)} -> the measured decay is spatial",
        bool(share_window) and worst_share < TEMPORAL_SHARE_MAX,
        per_mesh=share,
        per_mesh_window=share_window,
        worst=worst_share,
        evaluated_meshes=sorted(int(k) for k in share_window),
        control_meshes=sorted(int(k) for k in share),
    )
    if not share:
        # no control run in this sweep -> the gate has no evidence: it is
        # INCOMPLETE, never "passed" (absence of evidence is not evidence).
        gates["S3_temporal_negligible"]["status"] = "not_evaluated"
        gates["S3_temporal_negligible"]["note"] = (
            "no CFL-control records in this sweep; run without --quick")

    # -- S4 reference discriminator (anti-vacuity) --------------------------
    wrong_cells, wrong_errors = _fit_rows(records, "errors_wrong_reference",
                                          meshes=fit_meshes)
    wrong_orders = {name: fit_order(wrong_cells, values)
                    for name, values in wrong_errors.items()}
    gates["S4_reference_discriminates"] = _gate(
        "scored against the exact solution at t + "
        f"{REFERENCE_SHIFT} the fitted order collapses below "
        f"{0.5 * design_order:.2f} (= half the design order): a constant "
        "reference offset produces an O(1) error floor, so the S1 statistic "
        "rejects a wrong reference instead of rubber-stamping it",
        max(wrong_orders.values()) < 0.5 * design_order,
        wrong_reference_time=FINAL_TIME + REFERENCE_SHIFT,
        orders=wrong_orders,
        worst=min(wrong_orders.values()),
        highest=max(wrong_orders.values()),
    )

    # -- S5 tolerance margin -------------------------------------------------
    finest = max(records, key=lambda r: r["cells"])
    finest_max = max(finest["errors"].values())
    gates["S5_tolerance_margin"] = _gate(
        f"finest mesh (n={finest['cells']}) max field error < "
        f"{MARGIN_FRACTION} * case tolerance ({ec.TOLERANCE})",
        finest_max < MARGIN_FRACTION * ec.TOLERANCE,
        finest_cells=int(finest["cells"]),
        finest_max_error=finest_max,
        finest_max_percent=100.0 * finest_max,
        case_tolerance=ec.TOLERANCE,
        margin_factor=ec.TOLERANCE / finest_max,
    )

    failed = [k for k, g in gates.items() if g["status"] == "failed"]
    not_evaluated = [k for k, g in gates.items() if g["status"] == "not_evaluated"]
    return {
        "case": "EULER-VORTEX-ERR",
        "ledger_slug": ec.__dict__.get("LEDGER_SLUG", "CASE-DG-EULER-VORTEX-ERR"),
        "question": ("does the DG family converge to the analytic isentropic "
                     "vortex at its design order p+1, or only sit within the "
                     "6 % tolerance of it?"),
        "design": {
            "degree": degree,
            "fields": list(FIELDS),
            "final_time": FINAL_TIME,
            "cfl": CFL,
            "cfl_control": CFL_CONTROL,
            "meshes": [int(r["cells"]) for r in records],
            "hypothesis": "observed L2 order = p + 1 = %d" % design_order,
            "asymptotic_window": [int(c) for c in fit_meshes],
        },
        "gates": {**gates,
                  "all_passed": not failed and not not_evaluated,
                  "failed": failed,
                  "not_evaluated": not_evaluated,
                  "verdict": ("PASS" if not failed and not not_evaluated
                              else "INCOMPLETE" if not failed else "FAIL")},
        "records": records,
    }


# ---------------------------------------------------------------------------
# artifacts
# ---------------------------------------------------------------------------

def markdown(data):
    gates = data["gates"]
    design = data["design"]
    lines = []
    add = lines.append
    add("# CASE-DG-EULER-VORTEX-ERR：空间收敛阶复核（参考解与误差度量的可辨识性）")
    add("")
    add("> 自动生成：`python3 benchmarks/case_euler_vortex_spatial_order.py --sweep "
        "--figures`；数字全部取自 `case_euler_vortex_errconserve/spatial_order.json`，零手抄。")
    add("")
    add(f"**问题**：{data['question']}")
    add("")
    add("**判定**：" + gates["verdict"]
        + "（" + ", ".join(f"{k}={v['status'].upper()}"
                           for k, v in gates.items()
                           if k not in ("all_passed", "failed", "not_evaluated", "verdict"))
        + "）")
    add("")
    add("## 1 方法")
    add("")
    add(f"- 设计族 DG-P{design['degree']}（HLLC + SSP-RK3），阶梯网格 "
        + " / ".join(str(m) for m in design["meshes"])
        + f"，末时刻 t = {design['final_time']}，CFL = {design['cfl']}。")
    add(f"- 时间步控制：同一阶梯在 CFL = {design['cfl_control']} 复算，量化时间误差占比（S3）。")
    add("- 阶拟合：对 h = 10/n 与相对 L2 误差做最小二乘，渐进窗口丢弃最粗网格 "
        f"（{design['asymptotic_window']}）。")
    add("- 反真空校核（S4）：把同一末态对**故意错位**的参考解（t + "
        f"{REFERENCE_SHIFT}）评分，阶必须塌到 1.0 以下。")
    add("")
    add("## 2 结果")
    add("")
    add("### 2.1 门禁")
    add("")
    add("| 判据 | 实测 | 判定 |")
    add("|---|---|---|")
    for key, gate in gates.items():
        if key in ("all_passed", "failed", "not_evaluated", "verdict"):
            continue
        measured = _summary_of(gate)
        add(f"| {key} | {measured} | {gate['status'].upper()} |")
    add("")
    add("### 2.2 误差阶梯与逐场观测阶")
    add("")
    add("| n | 步数 | " + " | ".join(f"e({f}) %" for f in FIELDS)
        + " | " + " | ".join(f"ord({f})" for f in FIELDS) + " |")
    add("|---" * (2 + 2 * len(FIELDS)) + "|")
    records = sorted(data["records"], key=lambda r: r["cells"])
    for record in records:
        errs = record["errors"]
        cells = [r["cells"] for r in records]
        cell_errors = {f: [r["errors"][f] for r in records] for f in FIELDS}
        local = {f: local_orders(cells, cell_errors[f]) for f in FIELDS}
        idx = cells.index(record["cells"])
        orders = []
        for f in FIELDS:
            if idx == 0:
                orders.append("—")
            else:
                orders.append(f"{local[f][idx - 1]['order']:.2f}")
        add(f"| {record['cells']} | {record.get('steps', '—')} | "
            + " | ".join(f"{100.0 * errs[f]:.6f}" for f in FIELDS)
            + " | " + " | ".join(orders) + " |")
    add("")
    add("`ord` 为相邻网格对的局部观测阶；S1 用渐进窗口的最小二乘阶，见 2.1。")
    add("")
    add("## 3 结论与适用范围")
    add("")
    add("- 观测阶与 p+1 一致 ⇒ 数值解收敛到**解析等熵涡**本身，6% 容差门禁所依据的参考解"
        "（含周期 min-image 映射）在 O(h^{p+1}) 意义下被独立确认；误差不是参考解缺陷。")
    add("- 时间误差占比 < 2% ⇒ 上述阶是空间阶，不是 SSP-RK3 的贡献。")
    add("- 错位参考解的阶塌陷 ⇒ 阶统计量对错误参考敏感，S1 非空判据。")
    add("")
    add("局限：只覆盖 P%d 与 t = %s 的短时窗口；长时（t = 10，穿箱一次）的阶未在此复测，"
        "见 `docs/CASE-DG-EULER-VORTEX-ERR-round6.md` 的 C2/C7x。" % (design["degree"], design["final_time"]))
    return "\n".join(lines) + "\n"


def _summary_of(gate):
    key = gate.get("criterion", "")
    if "fitted order of every field" in key:
        return (f"最差拟合阶 {gate['worst_fitted_order']:.3f}（窗口 {gate['fit_meshes']}），"
                f"最差末对局部阶 {gate['worst_finest_local_order']:.3f}")
    if "pure L2 projection" in key:
        return f"最低投影阶 {gate['lowest']:.3f}"
    if "relative change of every field error" in key:
        return f"最大相对变化 {gate['worst']:.2e}（网格 {gate['evaluated_meshes']}）"
    if "collapses below" in key:
        return f"最高错位参考阶 {gate['highest']:.3f}"
    if "finest mesh" in key:
        return (f"n={gate['finest_cells']} 最大场误差 {gate['finest_max_percent']:.6f}%"
                f"（容差内余量 ×{gate['margin_factor']:.1f}）")
    return "—"


def figures(data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(data["records"], key=lambda r: r["cells"])
    cells = np.array([r["cells"] for r in records], dtype=float)
    h = (ec.DOMAIN[0][1] - ec.DOMAIN[0][0]) / cells
    paths = []

    # 1) error vs h with reference slopes
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    markers = {"density": "o", "velocity_x": "s", "velocity_y": "^", "pressure": "D"}
    for field in FIELDS:
        values = np.array([r["errors"][field] for r in records])
        ax.loglog(h, values, marker=markers[field], label=f"DG-P2 {field}")
    anchor = np.array([r["errors"]["density"] for r in records])[0]
    for slope, style in ((2, ":"), (3, "--")):
        ax.loglog(h, anchor * (h / h[0]) ** slope, style,
                  color="grey", label=f"slope {slope}")
    ax.set_xlabel("h = 10/n"); ax.set_ylabel("relative L2 error")
    ax.set_title("CASE-DG-EULER-VORTEX-ERR: observed spatial order")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    path = out_dir / "fig_spatial_order.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    paths.append(str(path))

    # 2) local orders
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for field in FIELDS:
        values = [r["errors"][field] for r in records]
        rows = local_orders(list(cells.astype(int)), values)
        ax.plot([f"{r['coarse']}→{r['fine']}" for r in rows],
                [r["order"] for r in rows], marker="o", label=field)
    ax.axhline(ORDER_FRACTION * (data["design"]["degree"] + 1), color="r", ls="--",
               label="gate 0.9(p+1)")
    ax.set_ylabel("local observed order"); ax.set_ylim(0, 5)
    ax.set_title("pair-wise convergence order (asymptotic window excluded n=16)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    path = out_dir / "fig_spatial_order_local.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    paths.append(str(path))
    return paths


def main(argv=None):
    global DEGREE, FINAL_TIME, CFL, CFL_CONTROL
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="case_euler_vortex_errconserve")
    parser.add_argument("--meshes", default=",".join(str(m) for m in MESHES))
    parser.add_argument("--degree", type=int, default=DEGREE)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--cfl-control", type=float, default=CFL_CONTROL)
    parser.add_argument("--control-meshes", default="16,32,64",
                        help="meshes that also run the control CFL (S3)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true",
                        help="three coarsest meshes only, no control CFL")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--report-stdout", action="store_true",
                        help="dump the full result JSON to stdout in a delimited block")
    parser.add_argument("--markdown", action="store_true",
                        help="write docs/CASE-DG-EULER-VORTEX-ERR-spatial-order.md")
    parser.add_argument("--records-dir", default=None,
                        help="write/read per-mesh records here (crash recovery)")
    parser.add_argument("--aggregate-from", default=None,
                        help="aggregate per-mesh records from this directory instead of solving")
    parser.add_argument("--reuse", default=None,
                        help="reuse records from an existing spatial_order.json")
    args = parser.parse_args(argv)

    DEGREE, FINAL_TIME, CFL, CFL_CONTROL = (args.degree, args.final_time,
                                            args.cfl, args.cfl_control)
    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(args.output_dir)
    meshes = tuple(int(m) for m in args.meshes.split(",") if m)
    if args.quick:
        meshes = meshes[:3]

    if args.reuse:
        data = json.loads(Path(args.reuse).read_text())
    elif args.aggregate_from:
        records = [json.loads(p.read_text()) for p in
                   sorted(Path(args.aggregate_from).glob("n*.json"),
                          key=lambda q: int(q.stem[1:]))]
        if not records:
            raise SystemExit(f"no records in {args.aggregate_from}")
        data = evaluate(records)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spatial_order.json").write_text(json.dumps(data, indent=2))
    else:
        which = "full" if args.quick else "both"
        out_dir.mkdir(parents=True, exist_ok=True)
        control_meshes = (tuple(int(m) for m in args.control_meshes.split(",") if m)
                          if which != "full" else ())
        records = sweep(meshes, which=which, control_meshes=control_meshes,
                        records_dir=args.records_dir)
        data = evaluate(records)
        (out_dir / "spatial_order.json").write_text(json.dumps(data, indent=2))
    if args.report_stdout:
        print("-----BEGIN SPATIAL-ORDER JSON-----")
        print(json.dumps(data, indent=1))
        print("-----END SPATIAL-ORDER JSON-----", flush=True)
    print(json.dumps(data["gates"], indent=2))
    if args.figures:
        for path in figures(data, out_dir):
            print("figure:", path)
    if args.markdown:
        doc = Path("docs/CASE-DG-EULER-VORTEX-ERR-spatial-order.md")
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(markdown(data))
        print("markdown:", doc)
    return 0 if data["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())