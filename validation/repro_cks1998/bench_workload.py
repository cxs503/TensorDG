"""REPRO-DG-CKS1998 的 TensorBench workload：量出 got，打印成一份 JSON 文档。

设计约束（与平台 ``extract.format='json_stdout'`` 对齐）
--------------------------------------------------------
* **stdout 只放 JSON 文档**，其余日志一律走 stderr——平台从原始 stdout 字节里取文档，
  再按 spec 指针复算 ref/got；日志混进来只会增加解析风险。
* 量什么、ref 怎么来，全部由 ``validation/repro_cks1998/bench_spec.py`` 的度量表决定；
  本脚本只负责**测量**，不写死任何 ref 数字（ref 由理论阶 / 精确 Riemann 求解器给出）。
* 复用既有内核（``tensordg.advection_1d``、``tensordg.euler_dg_1d``）与既有参考解
  （``benchmarks.sod_exact``），不新起求解器。

档位
----
``smoke``（默认，约 1 s）: 极粗网格，只给测试用（指针/契约自检，不用于判定精度）
``bench``（集群作业用）: A1 三级 × 5 级网格（20/40/80/160/320）+ E2 Sod P2 N=400
                          + A2 限幅方波 L1 阶梯次（40/80/160/320，P1/P2/P3）
                          + E3 TVD 点值超调（100 单元，P1/P2，三种限幅策略）

不等式判据（A2 L1 阶 >= 1/2、E3 超调 <= 1 %）按 ``bench_spec.HINGE_RULES`` 做**单侧
hinge** 后进入 ref-vs-got 规格；原始测量值同时写进 ``validation.hinge``（raw/bound/kind）
与 ``rows``（逐网格 L1 误差、三种限幅器的峰值），使第三方可从 rows 重算并核对 hinge。

用法
----
    OMP_NUM_THREADS=1 python validation/repro_cks1998/bench_workload.py --profile smoke
    OMP_NUM_THREADS=1 python validation/repro_cks1998/bench_workload.py --profile bench \
        --out runs/cks1998_tensorbench/doc.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform as _platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tensordg.advection_1d import l2_error, solve_advection          # noqa: E402
from tensordg.convergence import observed_order                        # noqa: E402
from tensordg.euler_dg_1d import solve_euler_dg_1d                    # noqa: E402
from validation.repro_cks1998.bench_spec import BENCH_ID, BENCH_VER, ANCHOR  # noqa: E402
from validation.repro_cks1998.bench_spec import WORKLOAD_ID, WORKLOAD_VER    # noqa: E402
from validation.repro_cks1998.bench_spec import HINGE_RULES, hinge           # noqa: E402

SOD_LEFT = (1.0, 0.0, 1.0)      # (rho, u, p)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_T = 0.2
#: 台站选在解的光滑段：稀疏波内侧(0.30)、星区(0.55)、激波后平台(0.70)
SOD_STATIONS = {"sod_rho_x030": ("density", 0.30),
                "sod_rho_x070": ("density", 0.70),
                "sod_vel_x055": ("velocity", 0.55)}
#: 精确守恒积分（解析值，非求解器产出）：质量 / 总能量
EXACT_INTEGRALS = {"sod_mass_integral": 0.5625, "sod_energy_integral": 1.375}

#: 二维 Euler 光滑熵波的网格梯子（论文 §5.1 的 P^k 组合；相邻网格比不假设为 2）
SMOOTH_2D_LADDERS = {1: [16, 24, 32], 2: [12, 16, 24], 3: [8, 12, 16]}
SMOOTH_2D_SMOKE = {1: [4, 6], 2: [4, 6], 3: [3, 4]}
#: 二维守恒积分在哪个运行上测量（**预登记**：P2、p2 梯子中最细网格；smoke 档同比例缩小）
SMOOTH_2D_INTEGRAL_RUN = (2, SMOOTH_2D_LADDERS[2][-1])
SMOOTH_2D_INTEGRAL_RUN_SMOKE = (2, SMOOTH_2D_SMOKE[2][-1])

#: A2（限幅方波 L1 阶）与 E3（TVD 超调）的梯次。与仓内复现驱动
#: ``validation/repro_cks1998/repro_cks1998.py`` 的 quick/full 档同源，不另立口径。
INEQUALITY_PROFILES = {
    "smoke": {"discont_cells": [20, 40], "tvd_cells": 40, "tvd_degrees": [1, 2],
              # E3 并列门（论文 TVB M=0）的预登记分辨率 / 粗网格披露点。
              # 必须与 bench_spec.E3_PREREGISTRATION 的 *_by_profile 一致（有测试守护）。
              "tvd_paper_cells": 40, "tvd_paper_coarse_cells": 20},
    "bench": {"discont_cells": [40, 80, 160, 320], "tvd_cells": 100, "tvd_degrees": [1, 2],
              "tvd_paper_cells": 100, "tvd_paper_coarse_cells": 80},
}

PROFILES = {
    "smoke": {"smooth_cells": [8, 16], "degrees": [1, 2, 3], "sod": (1, 48),
              "cells2d": SMOOTH_2D_SMOKE,
              "integral_run": SMOOTH_2D_INTEGRAL_RUN_SMOKE,
              "inequality": INEQUALITY_PROFILES["smoke"]},
    "bench": {"smooth_cells": [20, 40, 80, 160, 320], "degrees": [1, 2, 3],
              "sod": (2, 400), "cells2d": SMOOTH_2D_LADDERS,
              "integral_run": SMOOTH_2D_INTEGRAL_RUN,
              "inequality": INEQUALITY_PROFILES["bench"]},
}


def log(msg: str) -> None:
    print(f"[workload] {msg}", file=sys.stderr, flush=True)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:                                    # pragma: no cover - 集群无 .git 时
        return os.environ.get("TENSORDG_REVISION", "unknown")


def _measure_smooth_2d(cells2d: dict, rows: list, metrics: dict, reference: dict,
                       integral_run: tuple = SMOOTH_2D_INTEGRAL_RUN) -> dict:
    """B1/B2：二维 Euler 光滑熵波（论文 §5.1）的观测阶与守恒积分。

    got 由同一条 2-D RKDG 求解器（``tensordg.euler_dg_2d``，无限制器、周期边界）
    实测；ref 是理论阶 ``k+1`` 与精确解的**闭式**守恒积分——两者都与本次运行无关。
    """
    from benchmarks.cks1998_smooth_2d import (DOMAIN, FINAL_TIME, conserved_integrals,
                                              error_norms, initial, run_case,
                                              EXACT_INTEGRALS)
    import tensordg.euler_dg_2d as dg2

    detail = {"ladders": {}, "integrals": None}
    last = {}
    for degree in sorted(cells2d):
        meshes, errs = list(cells2d[degree]), []
        for ncells in meshes:
            run = run_case(degree, ncells, None, 1.0, False)
            errs.append(run["norms"]["density"]["L1"])
            last[degree] = run
            rows.append({"section": "B1", "degree": degree, "ncells": ncells,
                         "l1_density": errs[-1],
                         "linf_density": run["norms"]["density"]["Linf"],
                         "wall_seconds": run["wall_seconds"]})
            log(f"B1 p{degree} n{ncells}: |rho|_L1={errs[-1]:.6e} "
                f"Linf={run['norms']['density']['Linf']:.6e}")
        order = observed_order(meshes, errs) if len(meshes) >= 2 else float("nan")
        key = f"sm2d_order_p{degree}"
        metrics[key] = {"value": float(order), "unit": "-"}
        reference[key] = float(degree + 1)
        detail["ladders"][f"p{degree}"] = {"cells": meshes, "l1_density": errs,
                                           "observed_order": float(order),
                                           "design_order": degree + 1}
        log(f"B1 p{degree}: observed order = {order:.4f} (theory {degree + 1})")

    # B2：在预登记的那次运行上测守恒积分（同一求解器、同一末时刻）
    degree, ncells = integral_run
    if degree in cells2d and ncells in cells2d[degree]:
        x, y, coeff, _meta = dg2.solve_euler_dg_2d(
            initial, ncells=ncells, degree=degree, domain=DOMAIN,
            final_time=FINAL_TIME, cfl=0.2, limiter=None, periodic=(True, True))
        totals = conserved_integrals(coeff, degree)
        got = {"sm2d_mass_integral": totals["mass"],
               "sm2d_energy_integral": totals["energy"]}
        ref = {"sm2d_mass_integral": EXACT_INTEGRALS["mass"],
               "sm2d_energy_integral": EXACT_INTEGRALS["energy"]}
        for key, value in got.items():
            metrics[key] = {"value": float(value), "unit": "-"}
            reference[key] = float(ref[key])
            rows.append({"section": "B2", "metric": key, "got": metrics[key]["value"],
                         "ref": reference[key], "degree": degree, "ncells": ncells})
            log(f"B2 {key}: got={metrics[key]['value']:.12f} ref={reference[key]:.1f}")
        detail["integrals"] = {"degree": degree, "ncells": ncells, "got": got, "ref": ref}
    else:
        log(f"B2 skipped: registered run (p{degree}, n{ncells}) not in this profile")
    return detail


def _rule(name: str) -> dict:
    for rule in HINGE_RULES:
        if rule["metric"] == name:
            return rule
    raise KeyError(f"HINGE_RULES 未登记度量 {name}")


def _measure_inequality(hp: dict, rows: list, metrics: dict, reference: dict) -> dict:
    """A2/E3：不等式判据的**原始测量** + 单侧 hinge 编码。

    测量函数直接复用仓内复现驱动（``repro_cks1998.run_discontinuous_advection`` /
    ``run_tvd``），保证「平台口径」与「仓内门口径」是同一份代码，不存在两套数。
    """
    from validation.repro_cks1998.repro_cks1998 import (run_discontinuous_advection,
                                                        run_tvd)

    hinge_payload: dict = {}

    def _emit(name: str, raw: float, extra: dict) -> None:
        rule = _rule(name)
        got = hinge(raw, rule["bound"], rule["kind"])
        metrics[name] = {"value": float(got), "unit": rule["unit"]}
        reference[name] = 1.0
        hinge_payload[name] = {"raw": float(raw), "bound": float(rule["bound"]),
                               "kind": rule["kind"], "hinge": float(got),
                               "criterion": rule["criterion"]}
        rows.append({"section": "hinge", "metric": name, "raw": float(raw),
                     "bound": float(rule["bound"]), "kind": rule["kind"],
                     "hinge": float(got), **extra})
        log(f"{name}: raw={raw:.6g} bound={rule['bound']:g} ({rule['kind']}) "
            f"-> hinge={got:.9f}")

    # ---- A2：限幅方波（TVB M=0 + SSP-RK3）的 L1 阶 ----
    a2 = run_discontinuous_advection({"discont_cells": list(hp["discont_cells"])})
    for key, block in a2["by_degree"].items():          # p1/p2/p3
        name = f"a2_l1_order_{key}"
        _emit(name, block["l1_order"],
              {"case": "A2_square_wave", "degree": int(key[1:]),
               "cells": list(hp["discont_cells"]), "l1_errors": block["l1_errors"]})

    # ---- E3：TVD 超调（**预登记在 strict 策略上**，见 GATES.tvd_overshoot）----
    e3 = run_tvd({"tvd_cells": int(hp["tvd_cells"]),
                  "tvd_degrees": list(hp["tvd_degrees"])})
    for degree in hp["tvd_degrees"]:
        strict = e3["cases"][f"p{degree}_strict_all"]
        # rows 里同时留点值极值：第三方可用 100*max(0, pt_max-1, -pt_min) 独立重算
        # 这条 raw（见 validation/repro_cks1998/hinge_audit.py），不必采信自述百分比。
        _emit(f"e3_overshoot_p{degree}", strict["overshoot_pointwise_pct"],
              {"case": "E3_tvd", "degree": int(degree),
               "limiter": "strict_all", "ncells": int(hp["tvd_cells"]),
               "overshoot_pointwise_pct": float(strict["overshoot_pointwise_pct"]),
               "avg_min": float(strict["avg_min"]), "avg_max": float(strict["avg_max"]),
               "pt_min": float(strict["pt_min"]), "pt_max": float(strict["pt_max"])})
    # ---- E3 并列门：**论文公开指示器**（TVB M=0）上的同一条 TVD 判据 ----
    # 旧门例 strict_all 是"构造即真"（审计 m22343 §3 F1），本门把它补成"可能为假"的判据；
    # 分辨率是判据本体的一部分（bench_spec.E3_PREREGISTRATION），文档行的 ncells /
    # limiter 与预登记不符 ⇒ validate_documents / ingest 拒收。
    paper_cells = int(hp["tvd_paper_cells"])
    coarse_cells = int(hp["tvd_paper_coarse_cells"])
    paper = e3 if paper_cells == int(hp["tvd_cells"]) else run_tvd(
        {"tvd_cells": paper_cells, "tvd_degrees": list(hp["tvd_degrees"])})
    for degree in hp["tvd_degrees"]:
        case = paper["cases"][f"p{degree}_tvb_M0"]
        _emit(f"e3_paper_overshoot_p{degree}", case["overshoot_pointwise_pct"],
              {"case": "E3_tvd_paper", "degree": int(degree), "limiter": "tvb_M0",
               "ncells": paper_cells,
               "overshoot_pointwise_pct": float(case["overshoot_pointwise_pct"]),
               "avg_min": float(case["avg_min"]), "avg_max": float(case["avg_max"]),
               "pt_min": float(case["pt_min"]), "pt_max": float(case["pt_max"]),
               "troubled_policy": "tvb", "tvb_constant": 0.0})
    # 分辨率披露（只披露不判定）：同一条判据在粗网格上的**实测**值，用以证明
    # "超调 ≤ 1 %" 对分辨率敏感，从而门必须钉住分辨率（审计 m22343 §3 F2）。
    coarse = e3 if coarse_cells == int(hp["tvd_cells"]) else run_tvd(
        {"tvd_cells": coarse_cells, "tvd_degrees": list(hp["tvd_degrees"])})
    for degree in hp["tvd_degrees"]:
        case = coarse["cases"][f"p{degree}_tvb_M0"]
        rows.append({"section": "E3_paper_resolution", "case": "E3_tvd_paper",
                     "degree": int(degree), "limiter": "tvb_M0",
                     "ncells": coarse_cells,
                     "overshoot_pointwise_pct": float(case["overshoot_pointwise_pct"]),
                     "exceeds_1pct": bool(case["overshoot_pointwise_pct"] > 1.0),
                     "gate_ncells": paper_cells, "preregistered": False,
                     "note": "coarse-grid disclosure only: shows the criterion is "
                             "resolution sensitive, hence the gate pins its resolution"})
        log(f"E3 paper-resolution disclosure p{degree} n{coarse_cells}: "
            f"{case['overshoot_pointwise_pct']:.3f} %")

    # 非门例（对照/公开指示器）也进 rows：它们的越界必须**看得见**，但不参与锚点判定
    for tag, case in e3["cases"].items():
        if tag.endswith("strict_all"):
            continue
        rows.append({"section": "E3_non_gate", "case": "E3_tvd", "tag": tag,
                     "degree": int(case["degree"]), "limiter": case["limiter"],
                     "ncells": int(case["ncells"]),
                     "overshoot_pointwise_pct": float(case["overshoot_pointwise_pct"]),
                     "exceeds_1pct": bool(case["overshoot_pointwise_pct"] > 1.0)})
        log(f"E3 non-gate {tag}: overshoot={case['overshoot_pointwise_pct']:.3f} %")

    return {"hinge": hinge_payload,
            "a2_pass": bool(a2["pass"]),
            "e3_pass": bool(e3["pass"]),
            "e3_control_violates": list(e3["control_violates"]),
            "profiles": {"discont_cells": list(hp["discont_cells"]),
                         "tvd_cells": int(hp["tvd_cells"]),
                         "tvd_degrees": list(hp["tvd_degrees"]),
                         "tvd_paper_cells": paper_cells,
                         "tvd_paper_coarse_cells": coarse_cells}}


def measure(profile: dict) -> dict:
    """返回 ``{"metrics": {...}, "reference": {...}, "rows": [...], "detail": {...}}``。"""
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    rows, metrics, reference = [], {}, {}

    # ---- A1：光滑正弦对流，观测阶 vs 理论阶 k+1 ----
    for degree in profile["degrees"]:
        cells, errs = list(profile["smooth_cells"]), []
        for ncells in cells:
            c, coeff, _meta = solve_advection(ncells=ncells, degree=degree,
                                              final_time=0.5, cfl=0.3)
            errs.append(l2_error(c, coeff, final_time=0.5))
            rows.append({"section": "A1", "degree": degree, "ncells": ncells,
                         "l2_error": errs[-1]})
        order = observed_order(cells, errs) if len(cells) >= 2 else float("nan")
        key = f"adv_order_p{degree}"
        metrics[key] = {"value": float(order), "unit": "-"}
        reference[key] = float(degree + 1)
        log(f"A1 p{degree}: order={order:.4f} (theory {degree + 1})")

    # ---- E2：Sod 激波管，DG vs 精确 Riemann 解 ----
    from validation.repro_cks1998.station_reference import (
        assert_station_cells_wave_free, exact_cell_averages, station_cell)
    from tensordg.euler_1d import GAMMA

    degree, ncells = profile["sod"]
    x, coeff, meta = solve_euler_dg_1d(SOD_LEFT, SOD_RIGHT, ncells=ncells, degree=degree,
                                       final_time=SOD_T, cfl=0.12, limiter="tvb",
                                       tvb_constant=1.0, limiter_stage="final")
    avg = coeff[:, 0]
    rho = avg[:, 0]
    vel = avg[:, 1] / rho
    pre = (GAMMA - 1.0) * (avg[:, 2] - 0.5 * avg[:, 1] ** 2 / rho)
    got = {"density": rho, "velocity": vel, "pressure": pre}
    # Reference-instrument policy (``station_reference``): the *exact* breakpoint-split
    # cell average.  ``benchmarks.riemann_exact.cell_averages`` -- the fixed Gauss rule
    # used before -- documents itself as unfit for tolerance comparisons (a cell that
    # straddles a wave is integrated inexactly, bias ``rho_q * jump``, no refinement
    # cure).  The guard makes the invariant it relied on explicit: every station cell
    # must be wave-free at this resolution.
    station_diag = assert_station_cells_wave_free(ncells, SOD_STATIONS)
    _, ref = exact_cell_averages(ncells, SOD_T)

    for key, (field, xs) in SOD_STATIONS.items():
        # Station -> cell is decided by one documented rule (``station_cell``), not by
        # ``int(x * ncells)``: that expression scores the left cell or the right one
        # depending on the last bit of the product for stations sitting on a face.
        idx = station_cell(ncells, xs)["index"]
        metrics[key] = {"value": float(got[field][idx]), "unit": "-"}
        reference[key] = float(ref[field][idx])
        rows.append({"section": "E2_station", "metric": key, "x": float((idx + 0.5) / ncells),
                     "got": metrics[key]["value"], "ref": reference[key]})

    # 守恒量：用同一套精确解析积分做 ref（与 DG 数值积分无关）
    energy = pre / (GAMMA - 1.0) + 0.5 * rho * vel ** 2
    dx = 1.0 / ncells
    integral = {"sod_mass_integral": float(rho.sum() * dx),
                "sod_energy_integral": float(energy.sum() * dx)}
    for key, ref_value in EXACT_INTEGRALS.items():
        metrics[key] = {"value": integral[key], "unit": "-"}
        reference[key] = float(ref_value)
        rows.append({"section": "E2_integral", "metric": key,
                     "got": integral[key], "ref": ref_value})

    rel_l2 = {f: float(np.sqrt(np.mean((got[f].numpy() - ref[f]) ** 2))
                       / np.sqrt(np.mean(ref[f] ** 2)))
              for f in ("density", "velocity", "pressure")}
    detail = {"sod": {"degree": degree, "ncells": ncells, "steps": meta["steps"],
                      "limiter": meta["limiter"], "rel_l2": rel_l2,
                      "worst_rel_l2": max(rel_l2.values()),
                      "stations": [
                          {"metric": row["metric"], "index": row["index"],
                           "x_left": row["x_left"], "x_right": row["x_right"],
                           "x_centre": row["x_centre"],
                           "gap_to_breakpoint_dx": row["gap_to_breakpoint_dx"],
                           "ref_exact": row["ref_exact"],
                           "ref_fixed_rule": row["ref_fixed_rule"],
                           "instrument_rel_delta": row["instrument_rel_delta"]}
                          for row in station_diag]}}
    log(f"E2 Sod p{degree} n{ncells}: worst rel L2 = {detail['sod']['worst_rel_l2']:.4%}; "
        f"tightest station margin = {min(r['gap_to_breakpoint_dx'] for r in station_diag):.3f} dx, "
        f"worst reference-instrument delta = "
        f"{max(r['instrument_rel_delta'] for r in station_diag):.2e}")

    # ---- A2/E3：不等式判据（hinge 编码进平台规格）------------------------
    if profile.get("inequality"):
        detail["inequality"] = _measure_inequality(profile["inequality"], rows,
                                                   metrics, reference)

    # ---- B1/B2：二维 Euler 光滑熵波（论文 §5.1）-------------------------
    if profile.get("cells2d"):
        detail["smooth_2d"] = _measure_smooth_2d(
            profile["cells2d"], rows, metrics, reference,
            profile.get("integral_run", SMOOTH_2D_INTEGRAL_RUN))

    return {"metrics": metrics, "reference": reference, "rows": rows, "detail": detail}


def build_doc(result: dict, profile_name: str, profile: dict, wall: float) -> dict:
    degree, ncells = profile["sod"]
    worst = result["detail"]["sod"]["worst_rel_l2"]
    return {
        "schema_version": "1.0.0",
        "run_id": f"{BENCH_ID}-{profile_name}-p{degree}_n{ncells}",
        "benchmark": {"benchmark_id": BENCH_ID, "version": BENCH_VER},
        "implementation": {"project": "TensorDG", "revision": git_revision()},
        "workload": {"workload_id": WORKLOAD_ID, "version": WORKLOAD_VER,
                     "profile": profile_name, "configuration": profile},
        "execution": {"status": "success", "wall_seconds": round(wall, 2),
                      "host": _platform.node(), "machine": _platform.machine(),
                      "python": _platform.python_version(),
                      "torch": torch.__version__, "numpy": np.__version__,
                      "torch_threads": torch.get_num_threads(),
                      "loadavg": list(os.getloadavg())},
        "metrics": result["metrics"],
        "reference": result["reference"],
        "validation": {"anchor": ANCHOR,
                       "sod_rel_l2": result["detail"]["sod"],
                       "smooth_2d": result["detail"].get("smooth_2d"),
                       "hinge": (result["detail"].get("inequality") or {}).get("hinge"),
                       "inequality": result["detail"].get("inequality"),
                       "note": ("A2/E3 的不等式判据经 bench_spec.hinge 做单侧编码后进入"
                                "ref-vs-got 规格（raw/bound/hinge 绑定在 validation.hinge，"
                                "逐网格原始量在 rows，可第三方重算）")},
        "rows": result["rows"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    ap.add_argument("--out", default=None, help="把同一份文档再写一份到文件")
    args = ap.parse_args()
    t0 = time.time()
    profile = PROFILES[args.profile]
    result = measure(profile)
    doc = build_doc(result, args.profile, profile, time.time() - t0)
    text = json.dumps(doc, ensure_ascii=False, indent=1)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)                       # stdout 只有文档
    return 0


if __name__ == "__main__":
    raise SystemExit(main())