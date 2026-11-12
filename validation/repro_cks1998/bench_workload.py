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

SOD_LEFT = (1.0, 0.0, 1.0)      # (rho, u, p)
SOD_RIGHT = (0.125, 0.0, 0.1)
SOD_T = 0.2
#: 台站选在解的光滑段：稀疏波内侧(0.30)、星区(0.55)、激波后平台(0.70)
SOD_STATIONS = {"sod_rho_x030": ("density", 0.30),
                "sod_rho_x070": ("density", 0.70),
                "sod_vel_x055": ("velocity", 0.55)}
#: 精确守恒积分（解析值，非求解器产出）：质量 / 总能量
EXACT_INTEGRALS = {"sod_mass_integral": 0.5625, "sod_energy_integral": 1.375}

PROFILES = {
    "smoke": {"smooth_cells": [8, 16], "degrees": [1, 2, 3], "sod": (1, 48)},
    "bench": {"smooth_cells": [20, 40, 80, 160, 320], "degrees": [1, 2, 3],
              "sod": (2, 400)},
}


def log(msg: str) -> None:
    print(f"[workload] {msg}", file=sys.stderr, flush=True)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       text=True).strip()
    except Exception:                                    # pragma: no cover - 集群无 .git 时
        return os.environ.get("TENSORDG_REVISION", "unknown")


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
    from benchmarks.sod_exact import cell_averages
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
    _, ref = cell_averages(ncells, SOD_T)

    for key, (field, xs) in SOD_STATIONS.items():
        idx = min(int(xs * ncells), ncells - 1)
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
                      "worst_rel_l2": max(rel_l2.values())}}
    log(f"E2 Sod p{degree} n{ncells}: worst rel L2 = {detail['sod']['worst_rel_l2']:.4%}")
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
                       "note": ("A2/E3 是不等式判据，不进入 ref-vs-got 规格；"
                                "见 tests/test_repro_cks1998.py 的仓内门")},
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