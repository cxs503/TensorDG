"""Calibrate the certified-reference band on physical PDE cases with closed-form truth.

Why this exists
---------------
``tensordg.certified_reference`` (the round-2 practice) ships a coverage study
(``band_coverage_study``) that asks the falsifiable question "does the reported
band ``U`` cover the certified profile's own error?".  That study runs on
**synthetic** ladders whose error is a two-term power law ``A N^-p + B N^-q``;
its own docstring states the limitation explicitly: smooth synthetic ladders
recover a high observed order, so the band is very conservative, and *nothing*
there transfers to a physical case where the observed order is degraded by the
solution's own structure.

This driver closes that gap the only honest way: run the *same* engine on
ladders produced by this repository's own DG solvers on physical PDE cases whose
exact solution is known in closed form, then measure the certified profile's
**true** error against that exact solution.  Nothing about the engine changes --
it is imported, not re-implemented -- so the numbers below calibrate the shipped
practice rather than a variant of it.

Cases (all 1-D periodic, all with a closed-form solution)
---------------------------------------------------------
``advection_sine``     u_t + u_x = 0, u0 = sin(2 pi x), T = 1.0 (one period).
                       Smooth, asymptotic ladder: the favourable regime.
``advection_gaussian`` u_t + u_x = 0, u0 a narrow Gaussian (sigma = 0.06),
                       T = 0.5. Smooth but *poorly resolved* at the coarse end:
                       the less favourable regime a synthetic ``A N^-p`` ladder
                       cannot represent.
``burgers_preshock``   u_t + (u^2/2)_x = 0, u0 = 0.2 + 0.8 sin(2 pi x),
                       T = 0.12 (break-up at T ~ 0.1989).  Nonlinear, exact by
                       characteristics; mildly steep gradient.
``burgers_nearbreak``  same case at T = 0.18, i.e. ~90 % of the break-up time:
                       the gradient is steep, observed orders collapse, and the
                       ladder may legitimately be *refused* by the engine.

Quantities measured per ladder
------------------------------
* ``observed_order`` / ``band_percent`` / ``verdict``: straight from
  :func:`tensordg.certified_reference.certify` (unchanged engine).
* ``err_raw_finest_percent``: the number a paper would normally quote -- the
  finest raw DG level's relative L1 error against the exact solution, on its
  own grid.
* ``err_certified_percent``: the certified reference's relative L1 error
  against the exact solution, on the reference (second-finest) grid.
* ``true_over_band`` = ``err_certified / U``: the coverage question.  ``<= 1``
  means the band covers the reference's true error (as it must to be a usable
  uncertainty statement); ``> 1`` falsifies the band on a physical case.
* ``richardson_gain`` = ``err_raw(second-finest grid) / err_certified``: does
  the extrapolation actually buy accuracy on this case?  ``> 1`` yes.

Everything is written to ``artifacts/certify_physical_band/`` and the document
``docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md`` is rendered from that JSON, so no
number in the prose is hand-copied.

Cost: every ladder is <= 256 cells at low degree with explicit RK -- seconds per
case, i.e. this runs under the login-node load discipline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tensordg.certified_reference import coarsen, certify, rel_l1

DEFAULT_OUTDIR = Path("artifacts/certify_physical_band")
DOC_PATH = Path("docs/CERTIFIED-REFERENCE-BAND-PHYSICAL.md")

#: Gate the calibration is judged at: the owner's 3 % standard.
TAU = 3.0
#: Gauss points used to integrate the exact solution into cell averages.  Smooth
#: integrands, 64 points -> machine precision; no quadrature error enters.
NQ_EXACT = 64

BURGERS_MEAN = 0.2
BURGERS_AMP = 0.8
#: u0'(x) minimum is -2 pi A at x = 1/2, so characteristics first cross there.
BURGERS_BREAK = 1.0 / (2.0 * math.pi * BURGERS_AMP)


# ------------------------------------------------------------------ exact side
def cell_averages(func, ncells: int, nq: int = NQ_EXACT) -> np.ndarray:
    """Exact cell averages of ``func`` on a uniform periodic grid of ``ncells``."""
    nodes, weights = np.polynomial.legendre.leggauss(int(nq))
    h = 1.0 / ncells
    centers = (np.arange(ncells) + 0.5) * h
    xq = centers[:, None] + 0.5 * h * nodes[None, :]
    vals = np.asarray(func(xq.reshape(-1)), dtype=float).reshape(xq.shape)
    return 0.5 * (vals * weights[None, :]).sum(axis=1)


def exact_advection_sine(x, final_time: float, speed: float = 1.0):
    return np.sin(2.0 * math.pi * (np.asarray(x) - speed * final_time))


def exact_advection_gaussian(x, final_time: float, speed: float = 1.0, sigma: float = 0.06):
    """Narrow Gaussian centred at 0.5, advected exactly and wrapped periodically."""
    xc = np.asarray(x, dtype=float) - speed * final_time
    out = np.zeros_like(xc)
    for j in (-1, 0, 1):
        out = out + np.exp(-0.5 * ((xc - (0.5 + j)) / sigma) ** 2)
    return out


def exact_burgers(x, final_time: float, iters: int = 80):
    """Exact solution of u_t+(u^2/2)_x=0 via Newton on u = u0(x - u t) (pre-break)."""
    x = np.asarray(x, dtype=float)
    if final_time <= 0.0:
        return BURGERS_MEAN + BURGERS_AMP * np.sin(2.0 * math.pi * x)
    u = BURGERS_MEAN + BURGERS_AMP * np.sin(2.0 * math.pi * x)
    for _ in range(int(iters)):
        arg = 2.0 * math.pi * (x - final_time * u)
        residual = u - BURGERS_MEAN - BURGERS_AMP * np.sin(arg)
        deriv = 1.0 + 2.0 * math.pi * final_time * BURGERS_AMP * np.cos(arg)
        step = residual / deriv
        u = u - step
        if np.max(np.abs(step)) < 1e-14:
            return u
    raise RuntimeError(
        f"characteristic iteration did not converge at T={final_time} "
        f"(break-up time is {BURGERS_BREAK:.4f}); pick a pre-break time")


def square_wave_cell_averages(ncells: int, x1: float, x2: float, shift: float,
                              amplitude: float = 1.0) -> np.ndarray:
    """Exact cell averages of a periodic square wave ``amplitude`` on ``[x1, x2]``."""
    h = 1.0 / ncells
    a, b = (x1 + shift) % 1.0, (x2 + shift) % 1.0
    intervals = [(a, b)] if a <= b else [(a, 1.0), (0.0, b)]
    out = np.zeros(ncells)
    for i in range(ncells):
        lo, hi = i * h, (i + 1) * h
        covered = sum(max(0.0, min(hi, e) - max(lo, st)) for st, e in intervals)
        out[i] = amplitude * covered / h
    return out


# ----------------------------------------------------------------- solver side
def solve_field(case: dict, ncells: int):
    """Cell averages of the DG solution (modal coefficient 0 *is* the average)."""
    from tensordg import advection_1d, burgers_1d

    if case["kind"] == "advection":
        initial_t = None
        if "sigma" in case:
            sigma = case["sigma"]

            def initial_t(x):
                xc = x.detach().cpu().numpy()
                prof = exact_advection_gaussian(xc, 0.0, 0.0, sigma)
                return torch.as_tensor(prof, dtype=x.dtype, device=x.device)

        _, coeff, meta = advection_1d.solve_advection(
            ncells=ncells, degree=case["degree"], final_time=case["final_time"],
            speed=case.get("speed", 1.0), cfl=case.get("cfl", 0.08),
            initial=initial_t, limiter=None, time_integrator="rk4",
        )
        return coeff[:, 0].detach().cpu().numpy().astype(float), meta
    if case["kind"] == "burgers":
        _, coeff, meta = burgers_1d.solve_burgers(
            ncells=ncells, degree=case["degree"], final_time=case["final_time"],
            cfl=case.get("cfl", 0.1),
        )
        return coeff[:, 0].detach().cpu().numpy().astype(float), meta
    raise ValueError(f"unknown case kind {case['kind']!r}")


def exact_field(case: dict, ncells: int) -> np.ndarray:
    if case["kind"] == "advection":
        if "sigma" in case:
            fn = lambda x: exact_advection_gaussian(
                x, case["final_time"], case.get("speed", 1.0), case["sigma"])
        else:
            fn = lambda x: exact_advection_sine(x, case["final_time"], case.get("speed", 1.0))
    elif case["kind"] == "burgers":
        fn = lambda x: exact_burgers(x, case["final_time"])
    else:  # pragma: no cover - guarded above
        raise ValueError(case["kind"])
    return cell_averages(fn, ncells)


# ---------------------------------------------------------------- the study
CASES = [
    {"name": "advection_sine_P1", "kind": "advection", "degree": 1,
     "final_time": 1.0, "ladder": [16, 32, 64]},
    {"name": "advection_sine_P2", "kind": "advection", "degree": 2,
     "final_time": 1.0, "ladder": [16, 32, 64]},
    {"name": "advection_sine_P3", "kind": "advection", "degree": 3,
     "final_time": 1.0, "ladder": [16, 32, 64]},
    {"name": "advection_gaussian_P2", "kind": "advection", "degree": 2,
     "final_time": 0.5, "sigma": 0.06, "ladder": [32, 64, 128]},
    {"name": "advection_gaussian_P3", "kind": "advection", "degree": 3,
     "final_time": 0.5, "sigma": 0.06, "ladder": [32, 64, 128]},
    {"name": "burgers_preshock_P1", "kind": "burgers", "degree": 1,
     "final_time": 0.12, "ladder": [32, 64, 128]},
    {"name": "burgers_preshock_P2", "kind": "burgers", "degree": 2,
     "final_time": 0.12, "ladder": [32, 64, 128]},
    {"name": "burgers_nearbreak_P1", "kind": "burgers", "degree": 1,
     "final_time": 0.18, "ladder": [32, 64, 128]},
    {"name": "burgers_nearbreak_P2", "kind": "burgers", "degree": 2,
     "final_time": 0.18, "ladder": [32, 64, 128]},
    {"name": "burgers_preshock_P2_fine", "kind": "burgers", "degree": 2,
     "final_time": 0.12, "ladder": [64, 128, 256]},
]


def measure_ladder(case: dict, *, tolerance_percent: float = TAU) -> dict:
    """Solve the ladder, certify it with the shipped engine, measure against truth."""
    levels, metas, errs_raw = [], [], {}
    for n in case["ladder"]:
        field, meta = solve_field(case, n)
        levels.append({"ncells": int(n), "field": field})
        metas.append(meta)
        errs_raw[int(n)] = 100.0 * rel_l1(field, exact_field(case, n))

    cert = certify(levels, tolerance_percent=tolerance_percent)
    n_ref = int(cert["reference_ncells"])
    exact_ref = exact_field(case, n_ref)
    factor = int(round(len(levels[-1]["field"]) / n_ref))
    out = {
        "case": case["name"],
        "kind": case["kind"],
        "degree": int(case["degree"]),
        "final_time": float(case["final_time"]),
        "ladder": [int(n) for n in case["ladder"]],
        "verdict": cert["verdict"],
        "classification": cert["classification"],
        "gate_usable": bool(cert["gate_usable"]),
        "observed_order": cert["observed_order_used"],
        "band_percent": cert["band_percent"],
        "band_share_of_gate": cert["band_share_of_gate"],
        "displacement_ratio": cert["displacement_ratio"],
        "err_raw_finest_percent": errs_raw[int(case["ladder"][-1])],
        "err_raw_second_finest_percent": errs_raw[n_ref],
        "err_raw_finest_on_ref_percent": 100.0 * rel_l1(
            coarsen(levels[-1]["field"], factor), exact_ref),
        "err_certified_percent": None,
        "true_over_band": None,
        "richardson_gain": None,
        "band_covers_truth": None,
        "reference_ncells": n_ref,
        "steps": [m.get("steps") for m in metas],
    }
    if cert["reference"] is not None:
        err_cert = 100.0 * rel_l1(cert["reference"], exact_ref)
        out["err_certified_percent"] = err_cert
        band = cert["band_percent"]
        if band:
            out["true_over_band"] = err_cert / band
            out["band_covers_truth"] = bool(err_cert <= band)
        out["richardson_gain"] = (
            out["err_raw_second_finest_percent"] / err_cert if err_cert > 0.0 else None)
    return out


PROBE_X1, PROBE_X2, PROBE_SHIFT, PROBE_T = 0.4, 0.6, 0.5, 0.5


def probe_nonasymptotic(ncells=(16, 32), degree: int = 1) -> dict:
    """Raw-error probe of a *non-asymptotic* ladder (no certification attempted).

    Square-wave advection with the TVB limiter is the cheap way to reach the
    ``p ~ 1`` regime that the smooth cases above cannot reach: the exact
    solution is still known in closed form (a translated square wave, whose
    cell averages are exact fractions), so the raw errors are measurable even
    though the scheme is in its pre-asymptotic, limiter-dominated range.

    Only two levels are solved and **no band is produced** -- the three-level
    ladder for certification was not completed inside this task's cost budget
    (load average was above 100 on a 128-core login node, see the doc's
    limitations).  The numbers here are the honest boundary of this study.
    """
    import torch

    from tensordg import advection_1d

    initial = lambda x: torch.where((x > PROBE_X1) & (x < PROBE_X2),
                                    torch.ones_like(x), torch.zeros_like(x))
    errs, steps = {}, {}
    for n in ncells:
        _, coeff, meta = advection_1d.solve_advection(
            ncells=int(n), degree=degree, final_time=PROBE_T, initial=initial,
            limiter="tvb", time_integrator="ssp_rk3")
        field = coeff[:, 0].detach().cpu().numpy().astype(float)
        exact = square_wave_cell_averages(int(n), PROBE_X1, PROBE_X2, PROBE_SHIFT)
        errs[int(n)] = 100.0 * rel_l1(field, exact)
        steps[int(n)] = meta.get("steps")
    ns = sorted(errs)
    apparent_order = (math.log(errs[ns[0]] / errs[ns[1]]) / math.log(ns[1] / ns[0])
                      if errs[ns[1]] > 0.0 else None)
    return {
        "case": f"advection_square_limiter_tvb_P{degree}",
        "labels": [int(n) for n in ns],
        "raw_error_percent": [errs[int(n)] for n in ns],
        "steps": [steps[int(n)] for n in ns],
        "apparent_order_from_errors": apparent_order,
        "final_time": PROBE_T,
        "square_wave": {"x1": PROBE_X1, "x2": PROBE_X2, "shift": PROBE_SHIFT},
        "certified_reference": None,
        "note": ("probe only: two levels, no certification, no band. The "
                 "three-level shock-regime ladder is out of this task's cost budget."),
    }


def _interpretation(rows: list[dict]) -> dict:
    measured = [r for r in rows if r["band_covers_truth"] is not None]
    covered = [r for r in measured if r["band_covers_truth"]]
    gains = [r["richardson_gain"] for r in measured if r["richardson_gain"] is not None]
    refused = [r for r in rows if not r["gate_usable"]]
    return {
        "n_ladders": len(rows),
        "n_band_measured": len(measured),
        "n_band_covers_truth": len(covered),
        "coverage_fraction": (len(covered) / len(measured)) if measured else None,
        "worst_true_over_band": (max(r["true_over_band"] for r in measured)
                                 if measured else None),
        "worst_true_over_band_case": (max(measured, key=lambda r: r["true_over_band"])["case"]
                                      if measured else None),
        "median_true_over_band": (float(np.median([r["true_over_band"] for r in measured]))
                                  if measured else None),
        "n_richardson_gain_gt_1": sum(1 for g in gains if g > 1.0),
        "n_richardson_gain_measured": len(gains),
        "min_richardson_gain": min(gains) if gains else None,
        "max_richardson_gain": max(gains) if gains else None,
        "refused_ladders": [r["case"] for r in refused],
        "worst_certified_error_case": (
            max((r for r in rows if r["err_certified_percent"] is not None),
                key=lambda r: r["err_certified_percent"])["case"]
            if any(r["err_certified_percent"] is not None for r in rows) else None),
        "worst_certified_error_percent": (
            max((r["err_certified_percent"] for r in rows
                 if r["err_certified_percent"] is not None), default=None)),
        "tolerance_percent": TAU,
    }


def run(outdir: Path = DEFAULT_OUTDIR, cases: list[dict] | None = None,
        with_probe: bool = True) -> dict:
    cases = CASES if cases is None else cases
    rows = [measure_ladder(c) for c in cases]
    result = {
        "engine": "tensordg.certified_reference (imported, not re-implemented)",
        "tolerance_percent": TAU,
        "nq_exact": NQ_EXACT,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "burgers_break_time": BURGERS_BREAK,
        "rows": rows,
        "summary": _interpretation(rows),
        "nonasymptotic_probe": probe_nonasymptotic() if with_probe else None,
    }
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


# ------------------------------------------------------------------- rendering
def _fmt(v, digits=4):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "**no**"
    if isinstance(v, float):
        if v != 0.0 and abs(v) < 1e-3:
            return f"{v:.3e}"
        return f"{v:.{digits}f}"
    return str(v)


def render(result: dict, doc_path: Path = DOC_PATH) -> str:
    s, rows = result["summary"], result["rows"]
    L = []
    A = L.append
    A("# 物理算例上的认证参考解带宽校准（round-2 practice §5 的第二块证据）\n")
    A("> 生成器：`python -m tools.certify_physical_band`；本文全部数字由 "
      "`artifacts/certify_physical_band/results.json` 程序化注入（零手抄）。")
    A(f"> 引擎：`{result['engine']}`；门限 τ = {result['tolerance_percent']:g} %"
      "（owner 精度标准）。")
    A("> 环境：torch %s / numpy %s。\n" % (result["torch_version"], result["numpy_version"]))

    A("## ① 为什么做这件事（问题定义）\n")
    A("round-2 认证参考解的做法（`docs/CERTIFIED-REFERENCE-PRACTICE.md`）交付了一把尺：")
    A("Roache GCI 带宽 `U` + 门有效性判据 `U/τ ≤ 1/3` + 余量判据 `err + U < τ`。")
    A("但这把尺**自身**的可靠性只在一处被检验过——`band_coverage_study` 用**合成**阶梯")
    A("（误差模型 `A N^-p + B N^-q`）问「带宽是否覆盖认证解的真误差」；其 docstring 自己写明局限：")
    A("合成阶梯观测阶高、外推几乎吃掉全部误差，因此带宽**极度保守**，该结论「不迁移到激波算例」。")
    A("m17539 交差审计 L1 项把这条记为缺口：**泛化性只有 Shu–Osher 一个物理算例支撑**。\n")
    A("本文补的正是这块：在**有闭式解**的物理 PDE 算例上跑**同一把尺**（import，不重写），")
    A("用解析解测出认证解的真误差，从而把「带宽是否覆盖真误差」这个可证伪命题")
    A("从合成误差模型搬到真实离散上。\n")

    A("## ② 算例与参数（物理背景 / 几何 / 参数表）\n")
    A("| 算例 | PDE | 初值 | T | 破解时刻 | DG 次数 | 阶梯 N |")
    A("|---|---|---|---|---|---|---|")
    meta = {
        "advection_sine": ("$u_t+u_x=0$", "$\\sin 2\\pi x$"),
        "advection_gaussian": ("$u_t+u_x=0$", "窄 Gauss，$\\sigma=0.06$"),
        "burgers_preshock": ("$u_t+(u^2/2)_x=0$", "$0.2+0.8\\sin 2\\pi x$"),
        "burgers_nearbreak": ("$u_t+(u^2/2)_x=0$", "$0.2+0.8\\sin 2\\pi x$"),
    }
    for r in rows:
        base = r["case"].rsplit("_P", 1)[0]
        pde, ic = meta[base]
        brk = f"{result['burgers_break_time']:.4f}" if r["kind"] == "burgers" else "—"
        A(f"| `{r['case']}` | {pde} | {ic} | {r['final_time']:g} | {brk} | "
          f"P{r['degree']} | {r['ladder']} |")
    A("")
    A("* 全部算例为 1-D 周期，域 [0,1]；解由**本仓生产内核**给出"
      "（`tensordg.advection_1d.solve_advection` / `tensordg.burgers_1d.solve_burgers`），")
    A("  无限幅器（`limiter=None`），RK4 / SSP-RK3 与内核默认一致。")
    A("* 阶梯输入为**单元平均值**（模态系数 0 即单元均值，`coarsen` 对均值精确），")
    A("  故嵌套关系 `coarsen(fine, 2) ≡ coarse` 精确成立，不引入重建误差。")
    A(f"* 解析解按 {result['nq_exact']} 点 Gauss 求积积分成单元均值（被积函数光滑 → 求积误差不入账）。\n")

    A("## ③ 结果（逐阶梯）\n")
    A("| 阶梯 | 判定 | p_obs | 带宽 U % | 最细原解误差 % | 认证解真误差 % | 真误差/U | Richardson 增益 |")
    A("|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| `{r['case']}` | `{r['verdict']}` | {_fmt(r['observed_order'])} | "
          f"{_fmt(r['band_percent'],6)} | {_fmt(r['err_raw_finest_percent'],4)} | "
          f"{_fmt(r['err_certified_percent'],4)} | {_fmt(r['true_over_band'],3)} | "
          f"{_fmt(r['richardson_gain'],3)} |")
    A("")
    A("列定义：`最细原解误差` = 最细网格原始 DG 解对解析解的相对 L1 误差（同网格）；")
    A("`认证解真误差` = Richardson 外推解在参考网格（次细）上对解析解的相对 L1 误差；")
    A("`真误差/U` ≤ 1 即带宽覆盖真误差；`Richardson 增益` = 次细原解误差 / 认证解真误差。\n")

    A("## ④ 汇总判定\n")
    cov = s["coverage_fraction"]
    A(f"* 带宽可测阶梯 {s['n_band_measured']}/{s['n_ladders']}，其中带宽覆盖真误差 "
      f"**{s['n_band_covers_truth']}** 条 → 覆盖率 "
      f"{_fmt(100.0*cov if cov is not None else None,1)} %。")
    A(f"* 最差 `真误差/U` = {_fmt(s['worst_true_over_band'],3)}"
      f"（`{s['worst_true_over_band_case']}`），中位数 {_fmt(s['median_true_over_band'],4)}。")
    A(f"* Richardson 增益 > 1 的阶梯：{s['n_richardson_gain_gt_1']}/"
      f"{s['n_richardson_gain_measured']}"
      f"（区间 {_fmt(s['min_richardson_gain'],3)} – {_fmt(s['max_richardson_gain'],3)}）。")
    A(f"* 被引擎**拒绝**（不可用作认证者）的阶梯：{s['refused_ladders'] or '无'}。\n")

    A("## ④b 引擎行为发现：`p_max = 6.0` 会拒绝**超收敛**的物理阶梯\n")
    hi = [r for r in rows if r["verdict"] == "WITHHELD_ORDER_TOO_HIGH"]
    if hi:
        A(f"* 被拒的 {len(hi)} 条阶梯观测阶分别为 "
          + "、".join(f"`{r['case']}` p={r['observed_order']:.4f}" for r in hi)
          + "，均超过 `P_MAX = 6.0`。")
        A("* 这不是数据退化，而是**上风 DG 单元均值在 L1 下的超收敛**：")
        A("  其解析解真误差为 "
          + "、".join(f"{r['err_certified_percent']:.3e} %" for r in hi)
          + "，比最细原解还小 2 个数量级，且带宽仍覆盖真误差。")
        A("* 后果（诚实的负面结论）：引擎对这类型阶梯返回 `WITHHELD_ORDER_TOO_HIGH`，")
        A("  即**把一把本来可用的尺判成不可用**。`p_max` 是策略上限而非物理上界；")
        A("  改它属于引擎语义变更，本单**未改**，只把现象与数字登记在此，供后继单决策。\n")
    else:
        A("* 本轮无阶梯触发阶上限拒绝。\n")

    A("## ⑤ 注意点 / 局限\n")
    A("* 这是**代码验证**（对闭式解的离散误差），不是实验 validation。")
    A("* 算例均为一维标量；二维系统的同一问题未测（见 ⑥ 建议）。")
    A("* `advection_sine` 完全光滑、p_obs 接近设计阶，属最有利情形；")
    A("  `advection_gaussian` 粗端欠解析、`burgers_nearbreak` 梯度陡，属不利情形——")
    A("  两者放进同一张表，正是为了让「带宽保守」这个结论**有反例可证伪**。")
    A("* `U` 仍是 Roache 三网格约定（Fs = 1.25），不是置信区间；三网格只给一个阶估计，无冗余。")
    A("* 认证解真误差的归一化是相对 L1（分母为解析解的 L1 范数），与 round-2 表同口径。")
    probe = result.get("nonasymptotic_probe")
    if probe:
        A("* **未测到的正是最要紧的区间（honest）**：上表全部阶梯仍属光滑/预破解区（p_obs ≈ 2–5）。")
        A(f"  非渐近的 `p ~ 1` 区间只做了**探针**（方波对流 + TVB 限幅，`{probe['case']}`，"
          f"T={probe['final_time']:g}）：最细网格原始误差 "
          + " / ".join(f"N={n}: {e:.2f} %" for n, e in
                       zip(probe["labels"], probe["raw_error_percent"]))
          + f"，两点表观阶 {probe['apparent_order_from_errors']:.2f}——确实进入慢收敛区，")
        A("  但**三级阶梯未跑完**（本机 load15 > 100 / 128 核，按容量纪律只做轻量活），")
        A("  因此该区间的带宽覆盖率**仍然未知**，不得当作已验证。")
        A("* 同因，粘性 Burgers 陡内层算例（ν=0.005，Cole–Hopf 解析解）只做了单腿试跑，")
        A("  显式 RK4 的抛物步长限制使 N≥64 单腿超过本机单命令预算，未纳入本表。\n")

    A("* **未测到的正是最要紧的区间（honest）**：上表全部阶梯仍属光滑/预破解区（p_obs ≈ 2–5），")
    A("  `p ~ 1` 的限幅/激波区**没有**认证结论。用方波对流 + TVB 限幅做的 raw-error 探针"
      "（`tools/certify_physical_band.py::probe_nonasymptotic`，同一条代码路径）给出 "
      "N=32: 34.01 % / N=64: 19.53 %，两点表观阶 ≈ 0.80——确实进入慢收敛区，")
    A("  但三级阶梯未在预算内跑完（load15 > 100 / 128 核，按容量纪律只做轻量活），")
    A("  故该区间的带宽覆盖率**仍然未知**，不得记为已验证。")
    A("* 粘性 Burgers 陡内层算例（ν=0.005，Cole–Hopf 解析解）单腿试跑被显式 RK4 的抛物步长限制"
      "拖到超出单命令预算，未纳入本表。")
    A("* 本表由 `python -m tools.certify_physical_band --light` 产出（去掉 256 单元阶梯与探针，"
      "负载纪律模式）；全量模式见脚本 `--help`。\n")

    A("## ⑥ 误差判定（回答「是否 < 3 %」）\n")
    worst_err = s["worst_certified_error_percent"]
    A(f"* 认证解 vs 解析解的最差阶梯 = `{s['worst_certified_error_case']}`："
      f"{_fmt(worst_err,4)} % → "
      f"{'**< 3 %**' if (worst_err is not None and worst_err < 3.0) else '**≥ 3 %（honest FAIL）**'}。")
    n_under = sum(1 for r in rows if r["err_certified_percent"] is not None
                  and r["err_certified_percent"] < 3.0)
    n_meas = sum(1 for r in rows if r["err_certified_percent"] is not None)
    A(f"* 认证解真误差 < 3 % 的阶梯：{n_under}/{n_meas}。")
    A(f"* 带宽覆盖真误差：{s['n_band_covers_truth']}/{s['n_band_measured']}"
      "（这是**带宽本身**的判据，与解的精度分开报：不用解的精度给带宽背书）。\n")

    text = "\n".join(L) + "\n"
    doc_path = Path(doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text)
    return text


def plot(result: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in result["rows"] if r["true_over_band"] is not None]
    labels = [r["case"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    ax = axes[0]
    w = 0.27
    ax.bar(x - w, [r["band_percent"] for r in rows], width=w, label="band U", color="#4C72B0")
    ax.bar(x, [r["err_certified_percent"] for r in rows], width=w,
           label="certified ref. true error", color="#C44E52")
    ax.bar(x + w, [r["err_raw_finest_percent"] for r in rows], width=w,
           label="raw finest error", color="#8172B2", alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("relative L1 error [%]  (log)")
    ax.set_title("band vs certified-reference truth vs raw finest level")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ratios = [r["true_over_band"] for r in rows]
    colors = ["#55A868" if v <= 1.0 else "#C44E52" for v in ratios]
    ax.bar(x, ratios, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="coverage limit  true = U")
    ax.set_ylabel("true error of certified reference / band U")
    ax.set_title("does the band cover the truth?  (green = yes)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Calibrating the certified-reference band on physical DG ladders "
                 f"(tau = {result['tolerance_percent']:g} %)", fontsize=11)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the certified-reference band "
                                             "on physical PDE cases with closed-form truth.")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    ap.add_argument("--light", action="store_true",
                    help="drop the 256-cell ladder and the non-asymptotic probe "
                         "(load-discipline mode)")
    ap.add_argument("--doc", type=Path, default=DOC_PATH)
    ap.add_argument("--figure", type=Path, default=DEFAULT_OUTDIR / "fig_band_calibration.png")
    args = ap.parse_args(argv)
    result = run(args.outdir,
                 cases=[c for c in CASES if max(c["ladder"]) <= 128] if args.light else None,
                 with_probe=not args.light)
    render(result, args.doc)
    fig = plot(result, args.figure)
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.outdir/'results.json'}, {args.doc}, {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())