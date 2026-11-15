"""Hinge 编码的第三方交叉核验 + 门槛判据的判别力审计（m22343）。

背景（被审计对象）
------------------
m22274 把 CKS1998 的两条**不等式**判据（A2：限幅方波 L1 阶 ≥ 1/2；E3：点值超调 ≤ 1 %）
用**保序 hinge** 编进平台 ref-vs-got 规格：``got = clamp(raw)/bound``、``ref = 1``。
它在自家报告 §5.1 里诚实地登记了结构性局限：**hinge 由 workload 自己施加**，平台只看到
已饱和的 got，"raw → hinge" 这一步不在平台侧复算。于是同一份文档里"原始量"与"编码值"
是否自洽，此前**没有任何机器检查**——一个打印 ``got=1.0`` 的 workload（不测量）能原样过门。

本模块补上能补的那部分：在同一份 stdout 文档内做**三方一致性**核验

    rows（逐网格原始量）  --独立重算-->  raw  --clamp-->  hinge  ==  metrics.<name>.value

* A2 的 raw 由 ``rows`` 里的 (cells, l1_errors) 梯次**重算**（最小二乘斜率）；
* E3 的 raw 由 ``rows`` 里的点值极值 (pt_min, pt_max) 按 ``100*max(0, pt_max-1, -pt_min)``
  **重算**（过冲/欠冲定义与 ``repro_cks1998.overshoot_pct`` 同式，但不调用它）；
* 两者都与 ``validation.hinge.<name>.raw`` 自述值独立，再按 ``bench_spec.hinge`` 重新编码，
  与 ``metrics.<name>.value`` 逐位比较，最后核对"决定等价"（hinge 过门 ⟺ 原始不等式成立）。

任一处不一致 → ``DocError``，``bench_spec.validate_documents`` 因此拒收该 bundle，
``tensorbench_ingest.py --post`` 不会把伪造/串号的文档递交平台。

仍有局限（不粉饰）：这些重算仍**在同一份由被测代码产出的文档内**进行；它抓伪造与串号，
抓不到"workload 从一开始就不测量"。真正的独立复算必须在平台侧按 raw 重编码，属平台契约
改动范围（本仓无权）。

审计部分（``discriminating_power`` / ``a2_ladder_robustness``）
--------------------------------------------------------------
* ``a2_ladder_robustness``：把总梯次的最小二乘斜率与**各相邻网格对**的局部阶、以及
  **不相交子梯次**的斜率并列，检验"≥1/2"是否只靠某个幸运网格对；
* ``discriminating_power``：把 E3 的**门例配置**（strict_all）与**论文公开指示器**
  （TVB M=0）以及**不限幅对照**在同一批网格上并列，并在 N∈{40,80,100,200} 上扫描，
  回答三个问题：(i) 门例是不是"构造即真"（vacuity）；(ii) 门例配置换成论文配置是否仍过门；
  (iii) 判据对网格分辨率的敏感度（单分辨率预登记门的脆性）。
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

from validation.repro_cks1998.bench_spec import (E3_PREREGISTRATION, HINGE_RULES,
                                                 HINGE_TOL, DocError, get_pointer,
                                                 hinge)

#: raw 重算的比对容差：重算走的是同一批浮点数上的同式代数，差异只应来自 ulp。
RAW_REL_TOL = 1e-9


# ---------------------------------------------------------------------------
# 独立重算的工具函数（不 import tensordg.convergence / repro_cks1998 的实现）
# ---------------------------------------------------------------------------
def lsq_order(resolutions, errors) -> float:
    """log-log 最小二乘斜率（阶）。``h = 1/n``，故阶 = -slope。"""
    n = np.asarray(list(resolutions), dtype=float)
    e = np.asarray(list(errors), dtype=float)
    if n.size != e.size or n.size < 2:
        raise DocError("lsq_order: 需要等长的 >= 2 个网格/误差")
    if np.any(n <= 0) or np.any(e <= 0):
        raise DocError("lsq_order: 网格与误差都必须为正")
    return float(-np.polyfit(np.log(n), np.log(e), 1)[0])


def pairwise_orders(resolutions, errors) -> List[float]:
    n = np.asarray(list(resolutions), dtype=float)
    e = np.asarray(list(errors), dtype=float)
    return [float(math.log(e[i - 1] / e[i]) / math.log(n[i] / n[i - 1]))
            for i in range(1, n.size)]


def pointwise_overshoot_pct(pt_min: float, pt_max: float,
                            lo: float = 0.0, hi: float = 1.0) -> float:
    """点值超调（占初值幅值 %）；与 ``repro_cks1998.overshoot_pct`` 同式、独立实现。"""
    return 100.0 * max(0.0, pt_max - hi, lo - pt_min)


def _hinge_row(doc: Mapping[str, Any], metric: str) -> Dict[str, Any]:
    for row in doc.get("rows") or []:
        if row.get("metric") == metric:
            return row
    raise DocError(f"rows 里没有 {metric} 的原始量行：hinge 无法第三方重算")


def recompute_raw(doc: Mapping[str, Any], rule: Mapping[str, Any]) -> Tuple[float, str]:
    """从 ``rows`` 独立重算 hinge 的输入 raw。返回 (值, 重算路径说明)。"""
    metric, kind = rule["metric"], rule["kind"]
    row = _hinge_row(doc, metric)
    if metric.startswith("a2_l1_order"):
        cells = row.get("cells") or row.get("resolutions")
        errs = row.get("l1_errors")
        if not cells or not errs:
            raise DocError(f"{metric}: rows 行缺 cells/l1_errors，无法重算观测阶")
        return lsq_order(cells, errs), f"lsq slope over {len(list(errs))} grids from rows"
    if metric.startswith("e3"):        # e3_overshoot_* / e3_paper_overshoot_* 同一条路径
        if "limiter" in row and row.get("limiter") != "strict_all" and not metric.startswith("e3_paper"):
            raise DocError(f"{metric}: rows 行 limiter={row.get('limiter')!r} 与度量不符")
        if "pt_min" not in row or "pt_max" not in row:
            raise DocError(f"{metric}: rows 行缺 pt_min/pt_max，无法重算点值超调")
        return (pointwise_overshoot_pct(float(row["pt_min"]), float(row["pt_max"])),
                "100*max(0, pt_max-1, -pt_min) from rows")
    raise DocError(f"未知 hinge 度量 {metric!r}（本模块只认 a2_l1_order*/e3_overshoot*）")


# ---------------------------------------------------------------------------
# 三方一致性核验（被 validate_documents 调用 → ingest 门）
# ---------------------------------------------------------------------------
def verify_hinge_consistency(doc: Mapping[str, Any],
                             rel_tol: float = RAW_REL_TOL) -> Dict[str, Any]:
    """核对 rows ⟶ raw ⟶ hinge ⟶ metrics 的一致性；不一致抛 ``DocError``。

    返回报告（可写进 artifacts / 报告表）：
    ``{metric: {raw_doc, raw_rows, hinge_doc, hinge_recomputed, kind, bound,
                decision, holds, raw_rel_diff, consistent}}``
    """
    claimed = (doc.get("validation") or {}).get("hinge")
    if not claimed:
        raise DocError("文档缺 validation.hinge：启用 hinge 度量的 workload 必须发布"
                       "raw/bound/kind 绑定，否则 hinge 无法第三方核验")
    report: Dict[str, Any] = {}
    for rule in HINGE_RULES:
        name = rule["metric"]
        entry = claimed.get(name)
        if not isinstance(entry, Mapping):
            raise DocError(f"validation.hinge 缺度量 {name}")
        raw_doc = float(entry["raw"])
        bound = float(entry["bound"])
        kind = str(entry["kind"])
        if abs(bound - float(rule["bound"])) > 0.0 or kind != rule["kind"]:
            raise DocError(f"{name}: 文档里的界 ({bound}, {kind}) 与预登记 "
                           f"({rule['bound']}, {rule['kind']}) 不符")
        raw_rows, path = recompute_raw(doc, rule)
        scale = max(abs(raw_doc), abs(bound), 1e-300)
        raw_diff = abs(raw_rows - raw_doc) / scale
        if raw_diff > rel_tol:
            raise DocError(f"{name}: rows 重算的 raw={raw_rows!r} 与 validation.hinge."
                           f"raw={raw_doc!r} 不一致（相对差 {raw_diff:.3e} > {rel_tol:g}）")
        hinge_doc = float(entry["hinge"])
        hinge_rows = hinge(raw_rows, bound, kind)
        if abs(hinge_rows - hinge_doc) > HINGE_TOL:
            raise DocError(f"{name}: 由 rows 重编码的 hinge={hinge_rows!r} 与文档 "
                           f"hinge={hinge_doc!r} 不一致")
        got = float(get_pointer(doc, f"metrics.{name}.value"))
        if abs(got - hinge_doc) > HINGE_TOL:
            raise DocError(f"{name}: metrics 指针值 {got!r} != validation.hinge.hinge "
                           f"{hinge_doc!r}")
        holds = (raw_rows >= bound) if kind == "lower" else (raw_rows <= bound)
        decided_pass = abs(hinge_rows - 1.0) <= HINGE_TOL
        if holds != decided_pass:
            raise DocError(f"{name}: 决定不等价（原始不等式 {holds} vs hinge 过门 "
                           f"{decided_pass}）")
        report[name] = {"raw_doc": raw_doc, "raw_rows": raw_rows,
                        "raw_rel_diff": raw_diff, "raw_from": path,
                        "bound": bound, "kind": kind,
                        "hinge_doc": hinge_doc, "hinge_recomputed": hinge_rows,
                        "decision": "PASS" if decided_pass else "FAIL",
                        "raw_holds": bool(holds), "consistent": True}
    return report


# ---------------------------------------------------------------------------
# 判别力审计（不是门，是"门有没有信息量"的证据）
# ---------------------------------------------------------------------------
def a2_ladder_robustness(doc: Mapping[str, Any], bound: float = 0.5) -> Dict[str, Any]:
    """A2 的"≥1/2"是否只靠某个网格对：全梯次 / 相邻对 / 不相交子梯次并列。"""
    out: Dict[str, Any] = {}
    for rule in HINGE_RULES:
        if not rule["metric"].startswith("a2_l1_order"):
            continue
        row = _hinge_row(doc, rule["metric"])
        cells = [int(c) for c in row["cells"]]
        errs = [float(e) for e in row["l1_errors"]]
        pairs = pairwise_orders(cells, errs)
        subsets = {}
        if len(cells) >= 4:
            half = len(cells) // 2
            subsets["coarse_half"] = lsq_order(cells[:half + 1], errs[:half + 1])
            subsets["fine_half"] = lsq_order(cells[half:], errs[half:])
        out[rule["metric"]] = {
            "metric": rule["metric"], "cells": cells, "l1_errors": errs,
            "order_all": lsq_order(cells, errs), "pairs": pairs,
            "subsets": subsets, "bound": bound,
            "min_pair": min(pairs) if pairs else None,
            "all_pairs_above_bound": all(p >= bound for p in pairs),
            "all_subsets_above_bound": all(v >= bound for v in subsets.values()),
        }
    return out


def discriminating_power(cells: Tuple[int, ...] = (40, 80, 100, 200),
                         degrees: Tuple[int, ...] = (1, 2),
                         bound_pct: float = 1.0) -> Dict[str, Any]:
    """E3 门例 vs 论文公开指示器 vs 不限幅对照：分辨率扫描 + 判别力。

    直接调用仓内复现驱动 ``repro_cks1998.run_tvd``（与门例同源），只改网格数。
    """
    from validation.repro_cks1998.repro_cks1998 import run_tvd

    scan: List[Dict[str, Any]] = []
    for ncells in cells:
        res = run_tvd({"tvd_cells": int(ncells), "tvd_degrees": list(degrees)})
        for tag, case in res["cases"].items():
            scan.append({"ncells": int(ncells), "tag": tag,
                         "degree": int(case["degree"]),
                         "overshoot_pointwise_pct": float(case["overshoot_pointwise_pct"]),
                         "passes_1pct": bool(case["overshoot_pointwise_pct"] <= bound_pct)})
    def _by(kind: str, ncells: int) -> List[Dict[str, Any]]:
        return [r for r in scan if r["ncells"] == ncells and kind in r["tag"]]
    verdict = {}
    for ncells in cells:
        strict = _by("strict_all", ncells)
        tvb = _by("tvb_M0", ncells)
        unl = _by("unlimited", ncells)
        verdict[str(ncells)] = {
            "strict_all_max_pct": max(r["overshoot_pointwise_pct"] for r in strict),
            "tvb_M0_max_pct": max(r["overshoot_pointwise_pct"] for r in tvb),
            "unlimited_max_pct": max(r["overshoot_pointwise_pct"] for r in unl),
            "strict_all_passes": all(r["passes_1pct"] for r in strict),
            "tvb_M0_passes": all(r["passes_1pct"] for r in tvb),
            "unlimited_passes": all(r["passes_1pct"] for r in unl),
        }
    return {"bound_pct": bound_pct, "cells": list(cells), "degrees": list(degrees),
            "scan": scan, "by_resolution": verdict}


def main(argv: List[str] | None = None) -> int:
    import sys
    from pathlib import Path

    argv = list(sys.argv[1:] if argv is None else argv)
    doc_path = argv[0] if argv else None
    outdir = Path(argv[1]) if len(argv) > 1 else Path("artifacts/m22343_hinge_audit")
    outdir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {}
    if doc_path:
        doc = json.loads(Path(doc_path).read_text())
        payload["hinge_consistency"] = verify_hinge_consistency(doc)
        payload["a2_ladder_robustness"] = a2_ladder_robustness(doc)
    payload["discriminating_power"] = discriminating_power()
    path = outdir / "hinge_audit.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(f"→ {path}")
    for name, row in (payload.get("hinge_consistency") or {}).items():
        print(f"{name:20s} raw_doc={row['raw_doc']:.6g} raw_rows={row['raw_rows']:.6g} "
              f"hinge={row['hinge_doc']:.9f} {row['decision']}")
    for name, row in (payload.get("a2_ladder_robustness") or {}).items():
        print(f"{name:20s} order_all={row['order_all']:.4f} pairs="
              f"{[round(p, 4) for p in row['pairs']]} subsets="
              f"{ {k: round(v, 4) for k, v in row['subsets'].items()} }")
    for n, row in payload["discriminating_power"]["by_resolution"].items():
        print(f"E3 N={n:>4s} strict={row['strict_all_max_pct']:.3f}% "
              f"tvb_M0={row['tvb_M0_max_pct']:.3f}% "
              f"unlimited={row['unlimited_max_pct']:.3f}% "
              f"(pass: strict={row['strict_all_passes']} tvb={row['tvb_M0_passes']} "
              f"unlimited={row['unlimited_passes']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ---------------------------------------------------------------------------
# E3 并列门的**分辨率预登记**核验（审计 m22343 §3 F1/F2 的机器化闭合）
# ---------------------------------------------------------------------------
def verify_preregistration(doc: Mapping[str, Any]) -> Dict[str, Any]:
    """核对论文配置门的「分辨率/限幅策略/粗网格披露」是否与预登记一致。

    审计 m22343 §3 的两条发现：F1 旧门例 ``strict_all`` 是"构造即真"；F2 同一判据在
    论文配置下 N=80→100 **翻转**（1.389 % → 0.000 %），单分辨率门会把"网格选得好"
    当成"性质成立"。本函数把"门钉在哪一档"从散文变成可拒收的检查：

    * 该度量的 ``rows`` 行必须存在且**唯一**；
    * 行的 ``limiter`` 必须等于预登记的论文策略（``tvb_M0``）；
    * 行的 ``ncells`` 必须等于该档位（``workload.profile``）预登记的分辨率——换网格即抛；
    * ``E3_paper_resolution`` 披露行必须存在，且 ``ncells`` 等于预登记的粗网格点。

    返回 ``{metric: {limiter, ncells, coarse_ncells, gate_pct, coarse_pct,
    resolution_sensitive}}``；``resolution_sensitive=True`` 表示"门在预登记分辨率上过、
    而在粗网格上已越界"——即该判据**不是**分辨率无关的性质（报告必须披露）。
    """
    report: Dict[str, Any] = {}
    if not E3_PREREGISTRATION:
        return report
    profile = str(((doc.get("workload") or {}).get("profile")) or "")
    rows = list(doc.get("rows") or [])
    for metric, reg in E3_PREREGISTRATION.items():
        want = (reg.get("ncells_by_profile") or {}).get(profile)
        coarse_want = (reg.get("coarse_ncells_by_profile") or {}).get(profile)
        if want is None:
            raise DocError(f"{metric}: 档位 {profile!r} 未在 E3_PREREGISTRATION 登记分辨率")
        gate_rows = [r for r in rows if r.get("metric") == metric]
        if not gate_rows:
            raise DocError(f"{metric}: 预登记门例缺 rows 行（无法第三方重算）")
        if len(gate_rows) > 1:
            raise DocError(f"{metric}: rows 里出现 {len(gate_rows)} 行（必须唯一）")
        row = gate_rows[0]
        if row.get("limiter") != reg["limiter"]:
            raise DocError(f"{metric}: limiter={row.get('limiter')!r} 与预登记的论文策略 "
                           f"{reg['limiter']!r} 不一致")
        if int(row.get("ncells", -1)) != int(want):
            raise DocError(f"{metric}: ncells={row.get('ncells')!r} 与档位 {profile!r} 预登记分辨率 "
                           f"{want} 不一致（门不得悄悄换网格）")
        disc = [r for r in rows if r.get("section") == "E3_paper_resolution"
                and r.get("degree") == row.get("degree")]
        if not disc:
            raise DocError(f"{metric}: 缺粗网格分辨率披露行（E3_paper_resolution），"
                           "判据的分辨率敏感性不可见")
        if int(disc[0].get("ncells", -1)) != int(coarse_want):
            raise DocError(f"{metric}: 披露行 ncells={disc[0].get('ncells')!r} 与预登记的 "
                           f"粗网格 {coarse_want} 不一致")
        bound = float(_rule_bound(metric))
        gate_pct = float(row.get("overshoot_pointwise_pct"))
        coarse_pct = float(disc[0].get("overshoot_pointwise_pct"))
        recomputed_gate = pointwise_overshoot_pct(float(row["pt_min"]), float(row["pt_max"]))
        if abs(recomputed_gate - gate_pct) > 1e-9 * max(1.0, abs(gate_pct)):
            raise DocError(f"{metric}: 自述超调 {gate_pct:.6g} %% 与由 pt_min/pt_max 重算的 "
                           f"{recomputed_gate:.6g} %% 不一致")
        report[metric] = {"limiter": row["limiter"], "ncells": int(row["ncells"]),
                          "coarse_ncells": int(disc[0].get("ncells")),
                          "bound": bound, "gate_pct": gate_pct,
                          "coarse_pct": coarse_pct,
                          "gate_holds": bool(gate_pct <= bound),
                          "coarse_holds": bool(coarse_pct <= bound),
                          "resolution_sensitive": bool(coarse_pct > bound >= gate_pct)}
    return report


def _rule_bound(metric: str) -> float:
    for rule in HINGE_RULES:
        if rule["metric"] == metric:
            return float(rule["bound"])
    raise DocError(f"HINGE_RULES 未登记度量 {metric}")
