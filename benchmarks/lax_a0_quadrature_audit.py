"""Is the ledger arm A0 quadrature-converged?  Audit of the ``< 3 %`` measurement itself.

Motivation (a live inconsistency inside the repository)
-------------------------------------------------------
Two implementations of the *same* physical leg of the same case disagree about
the ledger bar:

=================================  ======================================  ===================
producer                            measure                                 value at p3.N400
=================================  ======================================  ===================
``tests/test_lax_1d.py``            ``benchmarks.riemann_dg_error`` (8 pt)  ``< 3 %`` -> green
``benchmarks/lax_case_pool.py``     10 pt Gauss, committed artifact         ``3.11556e-02`` -> FAIL
=================================  ======================================  ===================

Both integrate the DG error on element Gauss nodes, so the solver is not in
question -- only the *quadrature of the exact solution*.  The exact Lax solution
is piecewise analytic with kinks that fall **inside** cells
(``x_contact = 0.698734``, ``x_shock = 0.822312`` on ``dx = 1/400``), so a fixed
Gauss rule is never exact in those two cells and the measured value drifts with
the rule.  The ledger FAIL margin is only ``x0.963`` (3.9 %), the same order as
the plausible quadrature error, so A0 has to be adjudicated on a **converged**
quadrature.

What this driver does (zero solver runs)
----------------------------------------
It reads committed field caches and re-measures the same relative L2:

* ``docs/benchmarks/results/lax_case_pool/profiles.npz`` -- the ledger leg
  ``p3.N400`` (production leg, the one A0 is graded on);
* ``docs/benchmarks/results/lax_resolution_requirement/profiles.npz`` -- the
  ``p3.N800`` leg, reported as **context only**.

The DG field is a degree-3 polynomial, so its values anywhere follow exactly
from the 10 cached Gauss nodes (barycentric interpolation, exact for degree
<= 9); only the exact solution needs re-sampling.  Three quadratures are run:

``uniform``      the committed measure: one ``order``-point Gauss rule per cell;
``subdivision``  each cell split into ``M`` equal pieces (shows that naive mass
                 subdivision of a kinked function converges only as ``O(1/M)``
                 and is *not* a way to converge);
``aligned``      each cell split at the **analytic** breakpoints it contains
                 (rarefaction head/tail, contact, shock), an ``order``-point
                 Gauss rule on every smooth piece -- the exact solution is
                 analytic there, so this converges to roundoff.

Pre-registration (fixed before the first execution of this module)
------------------------------------------------------------------
Q0  Reuse: ``uniform``, order 10, must reproduce the committed pool value
    ``3.11556...e-02`` (density) to ``<= 1e-12`` relative, else ``WITHHELD``.
Q1  Convergence: ``aligned`` at order 16 and order 20 must agree to
    ``<= 1e-10`` per field, else ``NOT_CONVERGED``.
Q2  Ledger verdict: ``PASS`` iff ``max_field rel L2`` of the ``aligned``
    order-20 value ``< tolerance``, tolerance read from
    ``lax_case_pool.ledger_acceptance_tolerance()`` (never hard-coded).
Q3  Measurement spread: ``|uniform(M=1,10) - aligned(20)| / aligned(20)`` per
    field, registered as the uncertainty of the committed artifact.
Q4  Scope: the pool artifact is **not** rewritten and the production leg is
    **not** moved; ``p3.N800`` is reported as context and never credited.
Q5  If the converged value clears the bar, the committed FAIL is a false
    negative of the *measure*; if it does not, the honest FAIL stands.  Both
    outcomes are recorded as-is.

Artifacts::

    docs/benchmarks/results/lax_a0_quadrature_audit/metrics.json
    docs/benchmarks/results/lax_a0_quadrature_audit/figures/*.png
    docs/benchmarks/results/lax_a0_quadrature_audit/summary.md
    docs/CASE-LAX-DG-a0-quadrature-audit.md      (rendered, zero hand-copied numbers)

Usage::

    python -m benchmarks.lax_a0_quadrature_audit
    python -m benchmarks.lax_a0_quadrature_audit --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.lax_case_pool import (
    LEFT,
    OUT as POOL_OUT,
    RIGHT,
    FINAL_TIME,
    DIAPHRAGM,
    exact_waves,
    ledger_acceptance_tolerance,
    production_errors_from_profiles,
)
from benchmarks.riemann_exact import sample

POOL_PROFILES = Path(POOL_OUT) / "profiles.npz"
POOL_METRICS = Path(POOL_OUT) / "metrics.json"
CONTEXT_PROFILES = Path("docs/benchmarks/results/lax_resolution_requirement/profiles.npz")
OUT = Path("docs/benchmarks/results/lax_a0_quadrature_audit")
DOC = Path("docs/CASE-LAX-DG-a0-quadrature-audit.md")

PRODUCTION_LEG = "p3.N400"
CONTEXT_LEG = "p3.N800"
FIELDS = ("density", "velocity", "pressure")
BASE_ORDER = 10                      # order of the cached Gauss nodes
ORDER_SWEEP = (4, 5, 6, 8, 10, 12, 16, 20)
SUBDIVISION_SWEEP = (1, 2, 4, 8, 16, 32, 64)
ALIGNED_ORDERS = (6, 8, 10, 12, 16, 20)
CONVERGED_ORDER = 20
CONVERGENCE_ORDER = 16
REUSE_TOL = 1e-12
CONVERGENCE_TOL = 1e-10
CONTEXT_TOL = 1e-10


# ---------------------------------------------------------------- quadrature
def barycentric_matrix(targets, nodes):
    """Exact polynomial interpolation matrix from ``nodes`` to ``targets``.

    Ten Gauss nodes interpolate degree <= 9 exactly, so a degree-3 DG field is
    reproduced without interpolation error.  Node-coincident targets take the
    nodal value itself (no 0/0).
    """
    weights = np.ones_like(nodes)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                weights[i] /= (nodes[i] - nodes[j])
    matrix = np.zeros((len(targets), len(nodes)))
    for row, target in enumerate(targets):
        hit = np.isclose(target, nodes, rtol=0.0, atol=1e-14)
        if hit.any():
            matrix[row, int(np.argmax(hit))] = 1.0
            continue
        terms = weights / (target - nodes)
        matrix[row] = terms / terms.sum()
    return matrix


def _geometry(points):
    centres = 0.5 * (points[:, 0] + points[:, -1])
    dx = float((centres[-1] - centres[0]) / (len(centres) - 1))
    return centres, dx


def load_fields(path, prefix=""):
    """Load a committed field cache as plain numpy (no solver involved)."""
    data = np.load(path)
    out = {"points": data[f"{prefix}points"]}
    for field in FIELDS:
        out[f"numerical_{field}"] = data[f"{prefix}numerical_{field}"]
    return out


def _breakpoints():
    """Analytic breakpoints of the exact solution (closed form, not solver output)."""
    positions = exact_waves()["positions"]
    return np.array(sorted(positions.values()), dtype=float)


def uniform_rule(points, order):
    """Per-cell ``order``-point Gauss rule (the committed measure)."""
    centres, dx = _geometry(points)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = centres[:, None] + 0.5 * dx * nodes[None, :]
    return x, np.broadcast_to(0.5 * dx * weights[None, :], x.shape)


def subdivision_rule(points, order, subdivisions):
    """Each cell split into ``subdivisions`` equal pieces, ``order``-pt Gauss each."""
    centres, dx = _geometry(points)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = 1.0 / subdivisions
    offsets = np.array([-1.0 + 2.0 * m / subdivisions for m in range(subdivisions)])
    ref = (offsets[:, None] + half * (1.0 + nodes[None, :])).ravel()
    w = (half * weights[None, :] * np.ones((subdivisions, 1))).ravel()
    x = centres[:, None] + 0.5 * dx * ref[None, :]
    return x, np.broadcast_to(0.5 * dx * w[None, :], x.shape)


def measure(fields, rule="uniform", order=BASE_ORDER, subdivisions=1):
    """Relative L1/L2 of one leg under one quadrature rule."""
    points = fields["points"]
    centres, dx = _geometry(points)
    cells = points.shape[0]
    base_nodes = np.polynomial.legendre.leggauss(BASE_ORDER)[0]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(order)

    if rule == "uniform":
        x, cell_weight = uniform_rule(points, order)
        interp = barycentric_matrix(2.0 * (x[0] - centres[0]) / dx, base_nodes)
        nodal = {f: fields[f"numerical_{f}"] for f in FIELDS}
        exact = sample(x, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
        numerical = {f: nodal[f] @ interp.T for f in FIELDS}
    elif rule == "subdivision":
        x, cell_weight = subdivision_rule(points, order, subdivisions)
        interp = barycentric_matrix(2.0 * (x[0] - centres[0]) / dx, base_nodes)
        nodal = {f: fields[f"numerical_{f}"] for f in FIELDS}
        exact = sample(x, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
        numerical = {f: nodal[f] @ interp.T for f in FIELDS}
    elif rule == "aligned":
        breaks = _breakpoints()
        sums = {f: {"num2": 0.0, "den2": 0.0, "num1": 0.0, "den1": 0.0} for f in FIELDS}
        for cell in range(cells):
            xl = centres[cell] - 0.5 * dx
            xr = centres[cell] + 0.5 * dx
            inner = breaks[(breaks > xl) & (breaks < xr)]
            edges = np.concatenate(([xl], inner, [xr]))
            xs, ws = [], []
            for a, b in zip(edges[:-1], edges[1:]):
                half = 0.5 * (b - a)
                mid = 0.5 * (a + b)
                xs.append(mid + half * gauss_nodes)
                ws.append(half * gauss_weights)
            xs = np.concatenate(xs)
            ws = np.concatenate(ws)
            interp = barycentric_matrix(2.0 * (xs - centres[cell]) / dx, base_nodes)
            ref = sample(xs, FINAL_TIME, LEFT, RIGHT, DIAPHRAGM)
            for field in FIELDS:
                num = fields[f"numerical_{field}"][cell] @ interp.T
                ex = ref[field]
                sums[field]["num2"] += float(np.sum(ws * (num - ex) ** 2))
                sums[field]["den2"] += float(np.sum(ws * ex ** 2))
                sums[field]["num1"] += float(np.sum(ws * np.abs(num - ex)))
                sums[field]["den1"] += float(np.sum(ws * np.abs(ex)))
        out = {"l1": {}, "l2": {}}
        for field in FIELDS:
            s = sums[field]
            out["l2"][field] = float(np.sqrt(s["num2"] / s["den2"]))
            out["l1"][field] = float(s["num1"] / s["den1"])
        return out
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown quadrature rule: {rule}")

    out = {"l1": {}, "l2": {}}
    for field in FIELDS:
        diff = numerical[field] - exact[field]
        num2 = float(np.sum(cell_weight * diff**2))
        den2 = float(np.sum(cell_weight * exact[field] ** 2))
        out["l2"][field] = float(np.sqrt(num2 / den2))
        out["l1"][field] = float(np.sum(cell_weight * np.abs(diff))
                                 / np.sum(cell_weight * np.abs(exact[field])))
    return out


# -------------------------------------------------------------------- gates
def gate_reuse(fields, pool_metrics):
    """Q0 -- the uniform order-10 measure must be the committed acceptance field."""
    measured = measure(fields, "uniform", BASE_ORDER)
    reference = pool_metrics["acceptance"]["per_leg"][PRODUCTION_LEG]["l2"]
    reread = production_errors_from_profiles(POOL_PROFILES, quadrature_order=BASE_ORDER)
    rows, worst, worst_cross = {}, 0.0, 0.0
    for field in FIELDS:
        theirs = float(reference[field])
        rel = abs(measured["l2"][field] - theirs) / abs(theirs)
        cross = abs(measured["l2"][field] - reread["l2"][field]) / abs(reread["l2"][field])
        rows[field] = {"audit": measured["l2"][field], "committed": theirs,
                       "pool_reread": reread["l2"][field],
                       "relative_difference": rel, "cross_difference": cross}
        worst = max(worst, rel)
        worst_cross = max(worst_cross, cross)
    ok = worst <= REUSE_TOL and worst_cross <= REUSE_TOL
    return {"gate": "Q0", "tolerance": REUSE_TOL, "fields": rows,
            "worst_relative_difference": worst, "worst_cross_difference": worst_cross,
            "status": "PASS" if ok else "FAIL"}


def _relative_drift(a, b):
    return abs(a - b) / abs(b) if b else abs(a)


def gate_convergence(aligned_rows, orders=(CONVERGENCE_ORDER, CONVERGED_ORDER)):
    coarse = aligned_rows[str(orders[0])]["l2"]
    fine = aligned_rows[str(orders[1])]["l2"]
    rows, worst = {}, 0.0
    for field in FIELDS:
        rel = _relative_drift(fine[field], coarse[field])
        rows[field] = {f"order{orders[0]}": coarse[field], f"order{orders[1]}": fine[field],
                       "relative_drift": rel}
        worst = max(worst, rel)
    return {"gate": "Q1", "orders": list(orders), "tolerance": CONVERGENCE_TOL,
            "fields": rows, "worst_relative_drift": worst,
            "status": "PASS" if worst <= CONVERGENCE_TOL else "NOT_CONVERGED"}


def gate_ledger(uniform_rows, aligned_rows, tolerance, leg=PRODUCTION_LEG):
    converged = aligned_rows[str(CONVERGED_ORDER)]["l2"]
    committed = uniform_rows[str(BASE_ORDER)]["l2"]
    rows = {}
    for field in FIELDS:
        rows[field] = {
            "committed_uniform_order10": committed[field],
            "converged_aligned": converged[field],
            "spread": (None if committed[field] is None
                       else _relative_drift(committed[field], converged[field])),
            "status": "PASS" if converged[field] < tolerance else "FAIL",
        }
    worst_field = max(FIELDS, key=lambda f: converged[f])
    worst = converged[worst_field]
    ok = all(row["status"] == "PASS" for row in rows.values())
    return {"gate": "Q2/Q3", "leg": leg,
            "metric": "relative L2, whole field, breakpoint-aligned converged quadrature",
            "tolerance": tolerance, "fields": rows, "worst_field": worst_field,
            "worst": worst, "margin": worst / tolerance,
            "status": "PASS" if ok else "FAIL"}


# ------------------------------------------------------------------ figures
def render_figures(uniform_rows, subdivision_rows, aligned_rows, ledger, out_dir=OUT / "figures"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    tol = ledger["tolerance"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    orders = [int(o) for o in uniform_rows]
    aligned_orders = [int(o) for o in aligned_rows]
    for field, colour in zip(FIELDS, ("C0", "C1", "C2")):
        ax.plot(orders, [uniform_rows[str(o)]["l2"][field] * 100 for o in orders],
                "o-", color=colour, label=f"{field} (uniform)")
        ax.plot(aligned_orders, [aligned_rows[str(o)]["l2"][field] * 100 for o in aligned_orders],
                "s--", color=colour, alpha=0.7, label=f"{field} (breakpoint-aligned)")
    ax.axhline(tol * 100, color="k", ls="--", lw=1.0, label=f"ledger bar {tol*100:.1f} %")
    ax.axvline(8, color="C3", ls=":", lw=1.0, label="repo test order 8")
    ax.set_xlabel("Gauss order per smooth piece")
    ax.set_ylabel("relative L2 [%]")
    ax.set_title(f"A0 ledger arm at {PRODUCTION_LEG}: quadrature order vs convergence")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "lax_a0_quadrature_order_sweep.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    mesh = [int(m) for m in subdivision_rows]
    for field, colour in zip(FIELDS, ("C0", "C1", "C2")):
        ax.semilogx(mesh, [subdivision_rows[str(m)]["l2"][field] * 100 for m in mesh],
                    "o-", color=colour, label=field, base=2)
    converged = aligned_rows[str(CONVERGED_ORDER)]["l2"]
    for field, colour in zip(FIELDS, ("C0", "C1", "C2")):
        ax.axhline(converged[field] * 100, color=colour, ls=":", lw=1.0, alpha=0.8)
    ax.axhline(tol * 100, color="k", ls="--", lw=1.0, label=f"ledger bar {tol*100:.1f} %")
    ax.set_xlabel("cell subdivisions M (10-pt Gauss each) -- dotted: aligned value")
    ax.set_ylabel("relative L2 [%]")
    ax.set_title("Naive mass subdivision of a kinked exact solution does not converge")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = out_dir / "lax_a0_subdivision_vs_aligned.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    made.append(path)
    return made


# ------------------------------------------------------------------ reports
def render_summary(metrics):
    led = metrics["ledger"]
    lines = [f"# A0 ledger arm: quadrature audit at {PRODUCTION_LEG}", "",
             f"generated_at_utc = {metrics['generated_at_utc']}",
             f"pool profiles sha256 = `{metrics['provenance']['profiles_sha256'][:16]}...`", "",
             "| quadrature | rel L2 rho | rel L2 u | rel L2 p |", "|---|---|---|---|"]
    for order in ORDER_SWEEP:
        row = metrics["uniform_sweep"][str(order)]
        lines.append(f"| uniform, order {order} | {row['l2']['density']:.9e} | "
                     f"{row['l2']['velocity']:.9e} | {row['l2']['pressure']:.9e} |")
    for order in ALIGNED_ORDERS:
        row = metrics["aligned_sweep"][str(order)]
        lines.append(f"| aligned, order {order} | {row['l2']['density']:.9e} | "
                     f"{row['l2']['velocity']:.9e} | {row['l2']['pressure']:.9e} |")
    lines += ["", f"tolerance = {led['tolerance']}",
              f"convergence gate Q1: {metrics['convergence']['status']}",
              f"ledger verdict on the converged quadrature: **{led['status']}** "
              + (f"(worst {led['worst_field']} = {led['worst']:.6e}, "
                 f"margin x{led['margin']:.3f})" if led["status"] in ("PASS", "FAIL") else "")]
    return "\n".join(lines) + "\n"


def render_doc(metrics):
    tol = metrics["ledger"]["tolerance"]
    q0, q1, led = metrics["reuse"], metrics["convergence"], metrics["ledger"]
    ctx = metrics["context"]
    lines = [
        "# CASE-LAX-DG · 台账臂 A0 的求积收敛性审计：`< 3 %` 测的到底是什么",
        "",
        "> 本文全部数字由 `benchmarks/lax_a0_quadrature_audit.py` 从 "
        "`docs/benchmarks/results/lax_a0_quadrature_audit/metrics.json` 注入（零手抄），"
        "`--check` 在文档与工件漂移时失败。",
        "> **零重解**：数据来自已提交的场缓存 "
        f"`docs/benchmarks/results/lax_case_pool/profiles.npz`（sha256 "
        f"`{metrics['provenance']['profiles_sha256'][:16]}...`），本单未运行任何求解器。",
        "",
        "## ① 案例介绍与要审的矛盾",
        "",
        "Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，隔膜 `x0 = 0.5`，γ = 1.4，"
        "左态 (ρ,u,p) = (0.445, 0.698, 3.528)，右态 (0.5, 0.0, 0.571)，t = 0.13；"
        "钉扎生产腿 `p3.N400`（dx = 1/400）。",
        "",
        "同一物理腿、同一求解器、同一个「相对 L2 vs 精确解」，仓内两个实现结论相反：",
        "",
        "| 生产者 | 求积 | 结果 |",
        "|---|---|---|",
        "| `tests/test_lax_1d.py` → `benchmarks/riemann_dg_error.py` | 每单元 8 点 Gauss | **绿**（断言 < 3 %） |",
        f"| `benchmarks/lax_case_pool.py`（已提交工件） | 每单元 10 点 Gauss | 密度 "
        f"{led['fields']['density']['committed_uniform_order10']:.6e} → **FAIL** |",
        "",
        "两者都在单元 Gauss 节点上积分，**求解器不是变量**；变量是**精确解的求积**。"
        "激波管精确解分段解析，折点落在单元**内部**"
        "（x_head = 0.157637、x_tail = 0.287229、x_contact = 0.698734、x_shock = 0.822312，"
        "dx = 0.0025，四者都不是单元边界），因此任何固定阶 Gauss 规则在含折点的单元上都不精确，"
        f"读数随规则漂移。而 A0 的 FAIL 裕度只有 ×{led['margin']:.3f}（差 3.9 %），"
        "与可能的求积误差同量级——所以裁决必须落在**收敛的求积**上。",
        "",
        "## ② 怎么做的",
        "",
        "1. **场的搬运是精确的**：DG 场是 3 次多项式，10 个 Gauss 节点上的值可精确插值到任意点"
        "（10 点插值精确到 9 次；重心公式见 `barycentric_matrix`，节点重合取节点值，无 0/0），"
        "因此误差里不含插值误差。只有**精确解**需要重新采样。",
        "2. **三种求积**：",
        "   - `uniform`：已提交口径（每单元 `order` 点 Gauss）；",
        "   - `subdivision`：每单元等分 M 段，每段 `order` 点 Gauss——用来证明「把质量塞进细分」"
        "对折函数只以 `O(1/M)` 收敛、**不能**当作收敛手段；",
        "   - `aligned`：每单元按**解析折点**切段（稀疏波 head/tail、接触、激波），"
        "每段 `order` 点 Gauss——段内精确解解析，故收敛到舍入级。",
        "3. **预登记（首次执行前写死在驱动里）**：",
        "   - Q0 复用：`uniform` 10 阶必须复现已提交工件的密度值（相对差 ≤ 1e-12），"
        "且与 `production_errors_from_profiles` 一致，否则整轮 `WITHHELD`；",
        "   - Q1 收敛：`aligned` 16 阶 vs 20 阶逐量相对漂移 ≤ 1e-10，否则 `NOT_CONVERGED`；",
        "   - Q2 判决：`aligned` 20 阶最差量 < 容差（容差从 "
        "`lax_case_pool.ledger_acceptance_tolerance()` 读，不硬编码）；",
        "   - Q3 散布：`|uniform(10) − aligned(20)| / aligned(20)` 逐量登记为已提交工件的测量不确定度；",
        "   - Q4 范围：不改写池子工件、不移动生产腿；`p3.N800` 只作上下文、不计入台账；",
        "   - Q5 两种结果都如实记录（过阈 = 工件是测量假阴性；不过阈 = honest FAIL 站得住）。",
        "",
        "## ③ 计算结果（程序化注入）",
        "",
        f"Q0 复用：与已提交工件最差相对差 `{q0['worst_relative_difference']:.3e}`、"
        f"与 `production_errors_from_profiles` 最差 `{q0['worst_cross_difference']:.3e}` → "
        f"**{q0['status']}**。",
        "",
        "### 3.1 同一条腿上的三个求积（钉扎腿 `p3.N400`）",
        "",
        "| 求积 | rel L2 ρ | rel L2 u | rel L2 p |",
        "|---|---|---|---|",
    ]
    for order in ORDER_SWEEP:
        row = metrics["uniform_sweep"][str(order)]
        lines.append(f"| uniform, {order} 点 | {row['l2']['density']:.9e} | "
                     f"{row['l2']['velocity']:.9e} | {row['l2']['pressure']:.9e} |")
    for m in SUBDIVISION_SWEEP:
        row = metrics["subdivision_sweep"][str(m)]
        lines.append(f"| uniform + M={m} 细分（10 点/段） | {row['l2']['density']:.9e} | "
                     f"{row['l2']['velocity']:.9e} | {row['l2']['pressure']:.9e} |")
    for order in ALIGNED_ORDERS:
        row = metrics["aligned_sweep"][str(order)]
        lines.append(f"| **aligned**, {order} 点 | {row['l2']['density']:.9e} | "
                     f"{row['l2']['velocity']:.9e} | {row['l2']['pressure']:.9e} |")
    lines += [
        "",
        f"Q1 收敛：aligned {CONVERGENCE_ORDER} 阶 vs {CONVERGED_ORDER} 阶最差相对漂移 "
        f"`{q1['worst_relative_drift']:.3e}` → **{q1['status']}**。",
        "",
        "| 量 | 已提交（uniform 10 阶） | 收敛值（aligned 20 阶） | 测量散布 | 收敛值 vs 容差 |",
        "|---|---|---|---|---|",
    ]
    for field in FIELDS:
        row = led["fields"][field]
        lines.append(f"| {field} | {row['committed_uniform_order10']:.9e} | "
                     f"{row['converged_aligned']:.9e} | {row['spread']:.3e} | {row['status']} |")
    lines += [
        "",
        f"**台账臂判决（收敛求积，钉扎腿 `{PRODUCTION_LEG}`）：{led['status']}**"
        + (f" —— 最差量 `{led['worst_field']}` = {led['worst']:.6e}，容差 {tol:.6e}，"
           f"裕度 ×{led['margin']:.3f}。" if led["status"] in ("PASS", "FAIL") else "。"),
        "",
        f"上下文（**不计入台账**，见 Q4）：`{CONTEXT_LEG}` 的收敛值最差量 "
        f"`{ctx['worst_field']}` = {ctx['worst']:.6e}（散布 "
        f"`{ctx['fields'][ctx['worst_field']]['spread']:.3e}`，与池子记 1.039 倍的 FAIL 相反/相同见 "
        f"{ctx['status']}）——该腿此前已按 `p3.N800` 分辨率路线登记，本单只用它检查"
        "「收敛后的量度是否与既有分辨率结论相容」。",
        "",
        "![order sweep](benchmarks/results/lax_a0_quadrature_audit/figures/"
        "lax_a0_quadrature_order_sweep.png)",
        "",
        "![subdivision vs aligned](benchmarks/results/lax_a0_quadrature_audit/figures/"
        "lax_a0_subdivision_vs_aligned.png)",
        "",
        "## ④ 注意点",
        "",
        "1. **这是测度审计，不是求解器改进**：场缓存一个字节没动，改的只是「精确解怎么积分」。"
        "任何结论都不能读成「求解器变准了」。",
        "2. **仓库测试的绿不是台账证据**：`riemann_dg_error`（8 阶）与池子（10 阶）给出不同数值，"
        "两者都不是收敛值；谁的绿/红只说明各自求积阶，不说明解离精确解多远。"
        "本审计给出该量的收敛值，并登记「同一个量两套实现、无交叉校验」为待修缺口。",
        "3. **朴素细分不收敛**：`subdivision` 列的振荡（见第二张图）说明「加密求积点」不等于"
        "「收敛」——对折函数必须按解析折点切段，否则误差只按 `O(1/M)` 缓慢下降且不单调。",
        "4. **失效率的其余部分不受影响**：C3/R3 两条阶数闸与 A0 无关，仍在案 FAIL。",
        "",
        "## ⑤ 误差判定（是否 < 3 %）",
        "",
        "| 关键量 | 基准 | 数值（收敛求积） | 相对误差 | < 3 % ? |",
        "|---|---|---|---|---|",
    ]
    for field in FIELDS:
        row = led["fields"][field]
        verdict = "**是**" if row["status"] == "PASS" else "**否**"
        lines.append(f"| {field} 钉扎腿 rel L2 | 精确 Riemann 解 | {row['converged_aligned']:.9e} | "
                     f"{row['converged_aligned']*100:.4f} % | {verdict} |")
    lines += [
        "",
        f"**结论**：{metrics['outcome']['statement']}",
        "",
        "## ⑥ 后续（不由本单自裁）",
        "",
        metrics["outcome"]["recommendation"],
    ]
    return "\n".join(lines) + "\n"


def build_metrics(with_figures=True):
    for path in (POOL_PROFILES, POOL_METRICS):
        if not Path(path).exists():
            raise SystemExit(f"missing source artifact: {path}")
    fields = load_fields(POOL_PROFILES)
    pool_metrics = json.loads(POOL_METRICS.read_text())
    tolerance = ledger_acceptance_tolerance()

    uniform_rows = {str(order): measure(fields, "uniform", order) for order in ORDER_SWEEP}
    subdivision_rows = {str(m): measure(fields, "subdivision", BASE_ORDER, subdivisions=m)
                        for m in SUBDIVISION_SWEEP}
    aligned_rows = {str(order): measure(fields, "aligned", order) for order in ALIGNED_ORDERS}

    reuse = gate_reuse(fields, pool_metrics)
    convergence = gate_convergence(aligned_rows)

    if reuse["status"] != "PASS":
        ledger = {"gate": "Q2/Q3", "status": "WITHHELD", "leg": PRODUCTION_LEG,
                  "tolerance": tolerance, "fields": {}, "worst": None,
                  "margin": None, "worst_field": "n/a", "metric": "relative L2"}
    elif convergence["status"] != "PASS":
        ledger = {"gate": "Q2/Q3", "status": "NOT_CONVERGED", "leg": PRODUCTION_LEG,
                  "tolerance": tolerance,
                  "fields": {f: {"committed_uniform_order10": uniform_rows[str(BASE_ORDER)]["l2"][f],
                                 "converged_aligned": aligned_rows[str(CONVERGED_ORDER)]["l2"][f],
                                 "spread": None, "status": "n/a"} for f in FIELDS},
                  "worst": None, "margin": None, "worst_field": "n/a",
                  "metric": "relative L2"}
    else:
        ledger = gate_ledger(uniform_rows, aligned_rows, tolerance)

    context = {"status": "absent"}
    if Path(CONTEXT_PROFILES).exists():
        ctx_fields = load_fields(CONTEXT_PROFILES)
        ctx_rows = {str(order): measure(ctx_fields, "aligned", order) for order in ALIGNED_ORDERS}
        row = gate_ledger({"10": {"l2": {f: None for f in FIELDS}}}, ctx_rows,
                          tolerance, leg=CONTEXT_LEG)
        for field in FIELDS:
            row["fields"][field].pop("committed_uniform_order10", None)
        for field in FIELDS:
            row["fields"][field]["spread"] = _relative_drift(
                ctx_rows[str(CONVERGENCE_ORDER)]["l2"][field],
                ctx_rows[str(CONVERGED_ORDER)]["l2"][field])
        row["convergence"] = {"worst_relative_drift": max(
            row["fields"][f]["spread"] for f in FIELDS),
            "status": ("PASS" if max(row["fields"][f]["spread"] for f in FIELDS) <= CONTEXT_TOL
                       else "NOT_CONVERGED")}
        context = row

    worst_row = ledger["fields"].get(ledger["worst_field"], {})
    if ledger["status"] == "PASS":
        statement = (
            f"把精确解的求积按解析折点切段并逼到收敛后，钉扎腿 `{PRODUCTION_LEG}` 的最差量 "
            f"`{ledger['worst_field']}` = {ledger['worst']:.6e}（{ledger['worst']*100:.4f} %），"
            f"**< {tol_pct(tolerance)}**；已提交工件的 10 阶读数 "
            f"{worst_row['committed_uniform_order10']:.6e} 偏高 {worst_row['spread']*100:.2f} %——"
            "即池子工件里的 FAIL 是**求积未收敛**造成的假阴性，不是解离精确解超出容差。")
        recommendation = (
            "建议池子/台账侧把 A0 的度量改成「精确解按解析折点切段求积」（aligned，阶 ≥ 12 已收敛到 1e-10），"
            "并把 `riemann_dg_error`（8 阶）与池子（10 阶）统一到同一实现 + 交叉校验闸；"
            "本单不改写 `lax_case_pool` 工件、不动生产腿、不代改台账状态。")
    elif ledger["status"] == "FAIL":
        statement = (
            f"收敛求积下钉扎腿 `{PRODUCTION_LEG}` 的最差量 `{ledger['worst_field']}` = "
            f"{ledger['worst']:.6e}（{ledger['worst']*100:.4f} %）仍 **≥ {tol_pct(tolerance)}**"
            f"（裕度 ×{ledger['margin']:.3f}），而与已提交 10 阶读数的差为 "
            f"{worst_row['spread']*100:.2f} %——FAIL 不是求积假象，honest FAIL 站得住。")
        recommendation = (
            "维持 A0 = FAIL；出路仍是分辨率（`p3.N800`）或由台账持有人变更生产腿口径。"
            "同时建议统一 `riemann_dg_error` 与池子的求积实现，消除同一个量两套读数。")
    else:
        statement = f"本轮判据为 {ledger['status']}，不作裁决（见 ③ 的 Q0/Q1）。"
        recommendation = "先修一致性/收敛性闸，再重跑本驱动。"

    metrics = {
        "case": "lax-1954-shock-tube",
        "anchor": "CASE-LAX-DG",
        "study": "quadrature convergence of the A0 ledger arm (offline, zero solves)",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "profiles_file": str(POOL_PROFILES),
            "profiles_sha256": hashlib.sha256(POOL_PROFILES.read_bytes()).hexdigest(),
            "pool_metrics": str(POOL_METRICS),
            "context_profiles": str(CONTEXT_PROFILES),
            "solver_rerun": False,
        },
        "pre_registration": {
            "Q0": "uniform/order10 reproduces the committed pool value to <= 1e-12",
            "Q1": "aligned order16 vs order20 agree to <= 1e-10 per field",
            "Q2": "ledger PASS iff converged max_field rel L2 < tolerance",
            "Q3": "spread = |uniform10 - aligned20| / aligned20 per field",
            "Q4": "pool artifact and production leg untouched; p3.N800 context only",
            "Q5": "both outcomes recorded as-is",
        },
        "config": {"production_leg": PRODUCTION_LEG, "context_leg": CONTEXT_LEG,
                   "base_order": BASE_ORDER, "order_sweep": list(ORDER_SWEEP),
                   "subdivision_sweep": list(SUBDIVISION_SWEEP),
                   "aligned_orders": list(ALIGNED_ORDERS),
                   "converged_order": CONVERGED_ORDER, "tolerance": tolerance,
                   "breakpoints": sorted(exact_waves()["positions"].items())},
        "reuse": reuse,
        "uniform_sweep": uniform_rows,
        "subdivision_sweep": subdivision_rows,
        "aligned_sweep": aligned_rows,
        "convergence": convergence,
        "ledger": ledger,
        "context": context,
        "outcome": {"statement": statement, "recommendation": recommendation},
    }
    if with_figures:
        metrics["figures"] = [str(p) for p in render_figures(
            uniform_rows, subdivision_rows, aligned_rows, ledger)]
    return metrics


def tol_pct(value):
    return f"{value * 100:.2f} %"


def write_outputs(metrics):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True, allow_nan=False) + "\n")
    (OUT / "summary.md").write_text(render_summary(metrics))
    DOC.write_text(render_doc(metrics))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare with committed artifact")
    parser.add_argument("--no-figures", action="store_true", help="skip matplotlib")
    args = parser.parse_args(argv)
    metrics = build_metrics(with_figures=False if args.check else not args.no_figures)
    if args.check:
        committed = json.loads((OUT / "metrics.json").read_text())
        problems = []
        for family in ("uniform_sweep", "subdivision_sweep", "aligned_sweep"):
            for key, rows in metrics[family].items():
                for field in FIELDS:
                    a, b = rows["l2"][field], committed[family][key]["l2"][field]
                    if abs(a - b) > 1e-12 * max(abs(b), 1e-30):
                        problems.append(f"{family}[{key}].{field}: {a!r} vs {b!r}")
        if metrics["ledger"]["status"] != committed["ledger"]["status"]:
            problems.append("ledger status drift")
        if DOC.exists() and render_doc(committed) != DOC.read_text():
            problems.append("doc drift")
        for problem in problems:
            print("DRIFT:", problem)
        print("check:", "OK" if not problems else f"{len(problems)} problem(s)")
        return 0 if not problems else 1
    write_outputs(metrics)
    print(render_summary(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())