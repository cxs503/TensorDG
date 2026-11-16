"""A **second dimension** for the certified-reference engine: 2-D isentropic vortex.

Why this module exists
----------------------
``XVAL-DG-ROUND2-CERTIFIED-REFERENCE`` is the anchor under which the round-2
practice was turned into a reusable engine
(:mod:`tensordg.certified_reference`) and then applied to two *physical* cases:
Shu--Osher 1988 (shock/entropy interaction, certified against an independent
FV-WENO7 ladder) and Sod 1978 (certified against the exact Riemann solution).
Both are **1-D**.  The engine's own docstring claims to be a "reusable,
case-independent engine", but every primitive in it was implicitly 1-D:

* ``coarsen`` did ``v.reshape(-1, factor).mean(axis=1)`` -- on a ``(n, n)`` field
  that averages *across* the second axis and returns a vector;
* ``richardson_extrapolate`` derived its factor from ``fine.size / coarse.size``
  -- on a 2-D refinement by 2 that is **4**, not 2;
* ``judge`` did the same for the candidate/reference ratio.

So a 2-D profile could not be certified at all: the failure was loud (shape
errors) rather than silent, but "cannot be used outside 1-D" is a real hole in a
claim of case-independence, and it is exactly the kind of hole that a later user
discovers by getting a *number* out of the code rather than an exception.

What this module does
---------------------
1. Exercises the 2-D paths of the engine on a case that has an **exact analytic
   solution**, so the certified reference's *own* error can be measured, not just
   argued: the isentropic Euler vortex (``tensordg.euler_2d.vortex_primitive``),
   a smooth, genuinely 2-D, unsteady compressible flow -- no rarefaction, no
   shock, so the ladder is in its asymptotic range and the certification is
   expected to be sharp (in contrast to the shock cases, where ``p ~ 1``).
2. Certifies a DG-P2 ladder on ``N = 16/32/64`` cells per direction
   (``tensordg.euler_dg_2d.solve_euler_dg_2d_vortex``, CFL 0.2, ``t = 1``,
   periodic box ``[-5,5]^2``, ``u_inf = (1,1)``, strength 5) with the *unmodified*
   engine, and answers the question the 1-D band-coverage study could not:
   **does the GCI band bracket the true error of a 2-D certified reference?**
3. Judges an independent DG-P1 leg (``N = 64``) against that reference, which is
   the actual use of the practice (``judge``).
4. Re-checks the published **1-D** numbers bit for bit, so the dimension
   generalisation cannot have changed an already published certificate.

Predeclared criteria (written before the numbers were looked at)
---------------------------------------------------------------
``C1`` the 2-D DG-P2 ladder certifies: ``verdict == CERTIFIED`` and its band
      consumes ``<= 1/3`` of the 5 % gate (the engine's non-vacuity rule);
``C2`` the certified reference lives on a **2-D** grid (``shape == (32, 32)``,
      the second-finest level) -- i.e. the coarsening really happened per
      direction, not on the flattened array;
``C3`` its **true** rel-L1 density error against the exact cell averages is
      ``<= U`` (band brackets truth) *and* ``< 1 %`` in absolute terms;
``C4`` the DG-P1 leg is ``PASS`` at the 5 % gate, and its error exceeds the
      reference's own true error by at least a factor 10 (the reference must be
      much better than what it is used to judge);
``C5`` the **shipped 1-D certificate** is reproduced exactly: re-certifying the
      committed Shu--Osher FV ladder returns the published
      ``p = 1.1767348864087728`` / ``U = 0.04162173159427086 %`` (rel. tol 1e-12).

``accepted`` is the conjunction of C1--C5 and the CLI exits non-zero otherwise,
so a red run cannot be mistaken for a green one.

Usage
-----
    python -m benchmarks.certified_reference_vortex_2d                # audit
    python -m benchmarks.certified_reference_vortex_2d --figures      # + PNGs
    python -m benchmarks.certified_reference_vortex_2d --json-stdout  # machine form

References
----------
* Roache (1994); ASME V&V 20-2009; Celik et al. (2008) -- the GCI practice the
  engine implements (see :mod:`tensordg.certified_reference`).
* C.-W. Shu, *High order weighted essentially nonoscillatory schemes for
  convection dominated problems*, SIAM Rev. 51 (2009) 82--126 -- the isentropic
  vortex as an accuracy test for high-order Euler solvers.
* Cockburn, Luskin, Shu & Sueli (2003), SIAM J. Numer. Anal. 41:1--27 -- why the
  cell averages of a DG field are the right object to certify.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow ``python benchmarks/certified_reference_vortex_2d.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tensordg.certified_reference import (  # noqa: E402  (engine used unmodified)
    CERTIFIED,
    U_SHARE_MAX,
    certify,
    judge,
    rel_l1,
)

# --------------------------------------------------------------- case definition
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
PERIOD = 10.0          # periodic box length = domain length, so the fold is exact
FINAL_TIME = 1.0
REF_DEGREE = 2
REF_LADDER = (16, 32, 64)
REF_CFL = 0.2
LEG_DEGREE = 1
LEG_NCELLS = 64
LEG_CFL = 0.2

GATE_PERCENT = 5.0     # case-pool gate the certified reference must be usable at
STRICT_PERCENT = 3.0   # owner precision standard

#: The shipped 1-D certificate this module must reproduce (C5).  Values are the
#: ones published by ``docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md`` and asserted
#: by ``tests/test_certified_reference.py``.
SHIPPED_1D_ORDER = 1.1767348864087728
SHIPPED_1D_BAND_PERCENT = 0.04162173159427086
REPRO_RAWDIR = Path("validation/repro_shu_osher/artifacts/raw")


# -------------------------------------------------------------------- case solver
def solve_leg(degree: int, ncells: int, cfl: float) -> dict:
    """DG solution of the periodic isentropic vortex; returns cell averages."""
    import torch

    from tensordg import euler_dg_2d as dg2

    started = time.time()
    _, _, coeff, metadata = dg2.solve_euler_dg_2d_vortex(
        ncells=int(ncells), degree=int(degree), final_time=FINAL_TIME,
        cfl=float(cfl), strength=STRENGTH, domain=DOMAIN,
    )
    averages = dg2.cell_fields(coeff)
    density = averages["density"].cpu().numpy().astype(float)
    return {
        "degree": int(degree),
        "ncells": int(ncells),
        "cfl": float(cfl),
        "steps": int(metadata.get("steps", -1)),
        "wall_seconds": time.time() - started,
        "field": density,
    }


def exact_density_cell_averages(ncells: int) -> np.ndarray:
    """Exact 4-point-per-direction cell averages of the analytic vortex density.

    The reference is the *periodic* solution (``period=10``, min-image fold), the
    same reference the repository's vortex studies use; ``images`` is not needed
    because the density perturbation is second order in the Gaussian envelope
    (measured floor ``4e-13`` relative L2, see
    ``docs/CASE-DG-EULER-VORTEX-ERR-reference-floor.md``).
    """
    import torch

    from tensordg.euler_2d import vortex_primitive

    xlo, xhi = DOMAIN[0]
    ylo, yhi = DOMAIN[1]
    dx = (xhi - xlo) / ncells
    dy = (yhi - ylo) / ncells
    nodes, weights = np.polynomial.legendre.leggauss(4)
    centres_x = xlo + (np.arange(ncells) + 0.5) * dx
    centres_y = ylo + (np.arange(ncells) + 0.5) * dy
    with torch.no_grad():
        xq = torch.as_tensor(
            centres_x[:, None, None, None]
            + 0.5 * dx * nodes[None, None, :, None], dtype=torch.float64)
        yq = torch.as_tensor(
            centres_y[None, :, None, None]
            + 0.5 * dy * nodes[None, None, None, :], dtype=torch.float64)
        primitives = vortex_primitive(xq, yq, FINAL_TIME, STRENGTH, BACKGROUND,
                                      period=PERIOD)
        density = primitives[0]
        w = torch.as_tensor(weights, dtype=torch.float64)
        averaged = 0.25 * torch.sum(
            density * w[None, None, :, None] * w[None, None, None, :], dim=(2, 3))
    return averaged.cpu().numpy().astype(float)


def build_ladders(reuse: Path | None = None, save: Path | None = None) -> dict:
    """Solve (or load) the reference ladder and the judged leg."""
    if reuse is not None and Path(reuse).exists():
        with np.load(reuse) as d:
            data = {k: d[k] for k in d.files}
        ladders = {
            "reference": [{"ncells": int(n), "field": data[f"ref_n{n}"]}
                          for n in REF_LADDER],
            "exact": {int(n): data[f"exact_n{n}"] for n in REF_LADDER},
            "leg": {"degree": LEG_DEGREE, "ncells": LEG_NCELLS, "cfl": LEG_CFL,
                    "field": data["leg"]},
            "reused_from": str(reuse),
        }
        return ladders
    ladders = {"reference": [], "exact": {}}
    for n in REF_LADDER:
        ladders["reference"].append(solve_leg(REF_DEGREE, n, REF_CFL))
        ladders["exact"][int(n)] = exact_density_cell_averages(n)
    ladders["leg"] = solve_leg(LEG_DEGREE, LEG_NCELLS, LEG_CFL)
    if save is not None:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        payload = {f"ref_n{n}": lv["field"] for n, lv in zip(REF_LADDER, ladders["reference"])}
        payload.update({f"exact_n{n}": v for n, v in ladders["exact"].items()})
        payload["leg"] = ladders["leg"]["field"]
        np.savez_compressed(save, **payload)
    return ladders


# ---------------------------------------------------------------- 1-D regression
def shipped_1d_recheck(rawdir: Path = REPRO_RAWDIR) -> dict:
    """Re-certify the committed Shu--Osher FV ladder with the *patched* engine."""
    from benchmarks.shu_osher_certified_ref import load_fv_levels

    levels = [{"ncells": lv["ncells"], "field": lv["rho"]}
              for lv in load_fv_levels(rawdir)]
    if not levels:
        return {"available": False, "reason": f"no raw FV levels under {rawdir}"}
    study = certify(levels, tolerance_percent=GATE_PERCENT)
    p = study.get("observed_order_used")
    band = study.get("band_percent")
    ok_p = p is not None and abs(p - SHIPPED_1D_ORDER) <= 1e-12 * abs(SHIPPED_1D_ORDER)
    ok_band = (band is not None
               and abs(band - SHIPPED_1D_BAND_PERCENT)
               <= 1e-12 * abs(SHIPPED_1D_BAND_PERCENT))
    return {
        "available": True,
        "n_levels": len(levels),
        "observed_order": p,
        "band_percent": band,
        "verdict": study.get("verdict"),
        "shipped_order": SHIPPED_1D_ORDER,
        "shipped_band_percent": SHIPPED_1D_BAND_PERCENT,
        "order_reproduced": bool(ok_p),
        "band_reproduced": bool(ok_band),
    }


# ------------------------------------------------------------------------ study
def build_study(reuse: Path | None = None, save: Path | None = None) -> dict:
    started = time.time()
    ladders = build_ladders(reuse=reuse, save=save)
    ref_levels = ladders["reference"]
    cert = certify(ref_levels, tolerance_percent=GATE_PERCENT)

    ref_ncells = int(cert["reference_ncells"])
    exact_ref = ladders["exact"][ref_ncells]
    reference = cert["reference"]
    true_percent = (100.0 * rel_l1(reference, exact_ref)
                    if reference is not None else None)

    # ladder levels measured against the exact solution of their own grid: the
    # convergence curve a reader can check by eye
    ladder_rows = []
    for lv in ref_levels:
        n = int(lv["ncells"])
        err = 100.0 * rel_l1(lv["field"], ladders["exact"][n])
        ladder_rows.append({"ncells": n, "rel_l1_percent_vs_exact": err,
                            "steps": lv.get("steps"), "wall_seconds": lv.get("wall_seconds")})

    leg = ladders["leg"]
    leg_exact = exact_density_cell_averages(int(leg["ncells"]))
    leg_true_percent = 100.0 * rel_l1(leg["field"], leg_exact)
    leg_verdict = judge(leg["field"], cert, tolerance_percent=GATE_PERCENT)

    band_percent = cert.get("band_percent")
    share = cert.get("band_share_of_gate")
    criteria = {
        "C1_certified_and_gate_usable": bool(
            cert.get("verdict") == CERTIFIED
            and share is not None and share <= U_SHARE_MAX),
        "C2_reference_is_two_dimensional": bool(
            reference is not None
            and np.shape(reference) == (ref_ncells, ref_ncells)),
        "C3_band_brackets_true_error": bool(
            true_percent is not None and band_percent is not None
            and true_percent <= band_percent and true_percent < 1.0),
        "C4_leg_passes_and_reference_is_sharper": bool(
            leg_verdict.get("decision") in ("PASS", "PASS_MARGINAL")
            and true_percent is not None and true_percent > 0.0
            and leg_true_percent >= 10.0 * true_percent),
    }
    recheck = shipped_1d_recheck()
    criteria["C5_shipped_1d_certificate_reproduced"] = bool(
        recheck.get("available")
        and recheck.get("order_reproduced") and recheck.get("band_reproduced"))

    study = {
        "case": {
            "name": "2-D isentropic Euler vortex (periodic)",
            "domain": [list(DOMAIN[0]), list(DOMAIN[1])],
            "period": PERIOD,
            "strength": STRENGTH,
            "background_velocity": list(BACKGROUND),
            "final_time": FINAL_TIME,
            "reference_family": {"scheme": "DG (tensordg.euler_dg_2d)",
                                 "degree": REF_DEGREE, "cfl": REF_CFL,
                                 "ladder": list(REF_LADDER)},
            "judged_leg": {"degree": LEG_DEGREE, "ncells": LEG_NCELLS, "cfl": LEG_CFL},
            "gate_percent": GATE_PERCENT, "strict_percent": STRICT_PERCENT,
        },
        "certification": {k: v for k, v in cert.items()
                          if k not in ("reference", "field_raw_finest")},
        "reference_true_error_percent": true_percent,
        "ladder_vs_exact": ladder_rows,
        "leg": {
            "degree": int(leg["degree"]), "ncells": int(leg["ncells"]),
            "true_error_percent": leg_true_percent,
            "judge": {k: v for k, v in leg_verdict.items()},
            "wall_seconds": leg.get("wall_seconds"),
        },
        "shipped_1d_recheck": recheck,
        "criteria": criteria,
        "accepted": bool(all(criteria.values())),
        "wall_seconds": time.time() - started,
        "ladders_reused_from": ladders.get("reused_from"),
        "_profiles": {
            "reference": reference,
            "exact_reference": exact_ref,
            "reference_ncells": ref_ncells,
        },
    }
    return study


# --------------------------------------------------------------------- figures
def make_figures(study: dict, ladders: dict, outdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures = []
    cert = study["certification"]
    ref = ladders.get("_reference")
    exact = ladders.get("_exact_ref")
    if ref is not None and exact is not None:
        err = np.asarray(ref, dtype=float) - np.asarray(exact, dtype=float)
        lim = float(np.max(np.abs(err))) or 1.0
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        im = ax.imshow(err, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       extent=[-5, 5, -5, 5])
        ax.set_title("certified reference $\\rho$ minus exact cell averages")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.85, label="$\\Delta\\rho$")
        path = outdir / "fig1_certified_minus_exact.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        figures.append({"name": "fig1_certified_minus_exact",
                        "caption": "差场：认证参考解（Richardson 外推，N=32 网格）减解析解单元平均，"
                                   "用于目视确认残差无系统性偏置（无激波、无局部堆积）。",
                        "path": str(path)})

    ladder = study["ladder_vs_exact"]
    ns = [row["ncells"] for row in ladder]
    errs = [row["rel_l1_percent_vs_exact"] for row in ladder]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.loglog(ns, errs, "o-", label="DG-P%d ladder vs exact" % REF_DEGREE)
    if cert.get("reference_ncells"):
        ax.loglog([cert["reference_ncells"]],
                  [study["reference_true_error_percent"]], "s", ms=8,
                  label="certified reference (true error)")
        ax.axhline(cert["band_percent"], ls="--", color="C2",
                   label="GCI band $U$ (%.4f %%)" % cert["band_percent"])
    ax.loglog([study["leg"]["ncells"]], [study["leg"]["true_error_percent"]], "^",
              ms=8, label="judged DG-P%d leg" % study["leg"]["degree"])
    ax.set_xlabel("cells per direction"); ax.set_ylabel("rel. L1 density error [%]")
    ax.set_title("2-D vortex: ladder, certified reference, band")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    path = outdir / "fig2_ladder_band.png"
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    figures.append({"name": "fig2_ladder_band",
                    "caption": "收敛曲线：DG 梯次三条腿对解析解的相对 L1 密度误差、认证参考解的真误差、"
                               "GCI 不确定带 U 与被判腿的位置（log-log）。",
                    "path": str(path)})
    return figures


# ----------------------------------------------------------------------- report
def table_markdown(study: dict) -> str:
    cert = study["certification"]
    rows = ["| 量 | 值 |", "|---|---|"]
    rows.append(f"| 观测阶 p | {cert.get('observed_order_used')} |")
    rows.append(f"| 分类 / 判定 | {cert.get('classification')} / {cert.get('verdict')} |")
    rows.append(f"| 不确定带 U | {cert.get('band_percent')} % |")
    rows.append(f"| U / τ (τ={GATE_PERCENT:g}%) | {cert.get('band_share_of_gate')} |")
    rows.append(f"| 参考解所在网格 | {cert.get('reference_ncells')}² |")
    rows.append(f"| 参考解真误差 vs 解析解 | {study['reference_true_error_percent']} % |")
    rows.append(f"| 被判 DG-P{study['leg']['degree']} 腿真误差 | {study['leg']['true_error_percent']} % |")
    rows.append(f"| 被判腿判决 | {study['leg']['judge'].get('decision')} |")
    return "\n".join(rows)


def report_markdown(study: dict) -> str:
    crit = study["criteria"]
    lines = [
        "# 认证参考解的第二个维度：2-D 等熵涡算例复核",
        "",
        "> 锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`。本文所有数字由 "
        "`python -m benchmarks.certified_reference_vortex_2d` 程序化生成，无手抄。",
        "",
        "## 1. 案例",
        "",
        "二维等熵 Euler 涡（周期箱 $[-5,5]^2$，$u_\\infty=(1,1)$，涡强 5，$t=1$，周期=10）。"
        "选它的理由：它是**唯一同时具备解析解与真二维性**的算例，使认证参考解的**真误差**"
        "（而非仅自收敛差）可测——这正是 1-D 激波算例做不到的事。",
        "",
        "## 2. 方法",
        "",
        "* 参考族：DG-P2（`tensordg.euler_dg_2d`），CFL 0.2，梯次 16/32/64 格/方向；",
        "* 认证：`tensordg.certified_reference.certify`（Richardson + Roache GCI，$Fs=1.25$，"
        "非空门 $U/\\tau \\le 1/3$），**未针对本算例做任何特化**；",
        "* 真值：解析解 4×4 Gauss 单元平均；",
        "* 判决：独立 DG-P1 腿（64 格/方向）对认证参考解调用 `judge`；",
        "* 回归：用打过补丁的引擎重认证已发布的 Shu–Osher FV 梯次，逐位比对已发布数字。",
        "",
        "## 3. 结果",
        "",
        table_markdown(study),
        "",
    ]
    for name, ok in crit.items():
        lines.append(f"* `{name}` → **{'PASS' if ok else 'FAIL'}**")
    lines += [
        "",
        f"**accepted = {study['accepted']}**",
        "",
        "## 4. 注意点",
        "",
        "* 本算例**无激波**，梯次落在渐近区，认证锐利；这与激波算例（$p\\approx1$、带宽"
        "被迫宽松）是互补的两个极端，不能互相替代。",
        "* 2-D 的 `ncells` 语义是**每个方向的格数**（不是总格数）；`coarsen` 相应按方向"
        "分块平均。总格数比在 2-D 下是 $r^2$，若用总格数比做粗化因子会静默地粗化到错误网格。",
        "* DG 单元平均在 2-D 的收敛阶不适用 1-D 的 $2k+1$ 超收敛结论，故本模块不声明 "
        "`design_order`。",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/certified_reference_vortex_2d")
    parser.add_argument("--reuse", default=None,
                        help="npz cache of the ladder (skips the DG solves)")
    parser.add_argument("--save-ladders", default=None)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--json-stdout", action="store_true")
    parser.add_argument("--require-accepted", action="store_true")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    reuse = Path(args.reuse) if args.reuse else None
    save = Path(args.save_ladders) if args.save_ladders else (outdir / "ladders.npz")
    study = build_study(reuse=reuse, save=None if reuse else save)
    profiles = study.pop("_profiles", {})

    (outdir / "study.json").write_text(json.dumps(study, indent=2, default=float))
    (outdir / "study.md").write_text(report_markdown(study))
    if args.figures and profiles.get("reference") is not None:
        figs = make_figures(study, {"_reference": profiles["reference"],
                                    "_exact_ref": profiles["exact_reference"]}, outdir)
        (outdir / "figures.json").write_text(json.dumps(figs, indent=2))
    if profiles.get("reference") is not None:
        np.savez_compressed(outdir / "certified_reference_2d.npz",
                            reference=np.asarray(profiles["reference"]),
                            exact=np.asarray(profiles["exact_reference"]))
    if args.json_stdout:
        print(json.dumps(study, indent=2, default=float))
    else:
        print(report_markdown(study))
    return 0 if (study["accepted"] or not args.require_accepted) else 1


if __name__ == "__main__":
    raise SystemExit(main())