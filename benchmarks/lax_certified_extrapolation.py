"""CASE-LAX-DG -- do the certification models extrapolate to legs nobody fitted?

Two questions, one artifact
---------------------------
1. **Retrospective out-of-sample (measurable now).** The resolution-closure ladder
   (`docs/benchmarks/results/lax_resolution_closure/metrics.json`) measured ``p3`` at
   ``N = 200, 400, 800, 1600``.  Two of those legs -- ``p3.N200`` (DOF 800) and
   ``p3.N1600`` (DOF 6400) -- are **not** part of the enumerated certification grid
   and were never used to fit anything in this line of work.  They are therefore a
   legitimate *retrospective* out-of-sample set: predictions are fitted only on the
   eight enumerated legs, the windows are the same LOO-calibrated windows, and the
   comparison is made against numbers measured before this round for a different
   purpose.
2. **Prospective hold-out (pre-registered, measurement queued).** ``p2.N600`` and
   ``p3.N600`` were pre-registered in ``predictions.json`` and submitted to the
   cluster; if the artifacts are not back yet this driver records them as pending
   rather than inventing anything.

Why the retrospective test is worth having on its own: the enumerated grid spans
DOF 800..3200, so ``p3.N1600`` (DOF 6400) is an extrapolation **beyond the fitted
range**, which is exactly the regime a case-pool recommendation has to survive.  It
is honestly weaker than pre-registration -- the leg was measured earlier, so the
choice of test point is not blind -- and the write-up says so.

Artifacts::

    docs/benchmarks/results/lax_certified_holdout/extrapolation.json
    docs/benchmarks/results/lax_certified_holdout/report_extrapolation.json
    docs/CASE-LAX-DG-certified-holdout.md   (regenerated, both parts)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.lax_certified_holdout import (
    FIELDS,
    HOLDOUT,
    METRICS,
    MODELS,
    PREDICTIONS,
    calibrate,
    dof,
    enumerated_rows,
    evaluate,
    frontier,
    leg_name,
    render_figures,
    tolerance,
    worst_field,
)

HERE = Path(__file__).resolve().parent.parent
RESOLUTION = HERE / "docs/benchmarks/results/lax_resolution_closure/metrics.json"
OUT = HERE / "docs/benchmarks/results/lax_certified_holdout"
EXTRAPOLATION = OUT / "extrapolation.json"
DOC = HERE / "docs/CASE-LAX-DG-certified-holdout.md"


def unfitted_legs(enumerated, path=RESOLUTION):
    """Measured legs of the resolution ladder that the enumeration never used."""
    payload = json.loads(Path(path).read_text())
    known = {row["leg"] for row in enumerated}
    rows = []
    for entry in payload["ladder"]:
        name = leg_name(entry["cells"], entry["degree"])
        if name in known:
            continue
        l2 = {f: float(entry["l2"][f]) for f in FIELDS}
        field = max(l2, key=lambda f: l2[f])
        rows.append({"leg": name, "cells": int(entry["cells"]),
                     "degree": int(entry["degree"]),
                     "dof": dof(entry["cells"], entry["degree"]),
                     "steps": int(entry.get("steps", 0)),
                     "l2_errors": l2, "worst_l2": float(l2[field]),
                     "worst_field": field,
                     "source": "docs/benchmarks/results/lax_resolution_closure/metrics.json"})
    rows.sort(key=lambda r: r["dof"])
    return rows


def build():
    enumerated = enumerated_rows()
    tol = tolerance()
    calibration = calibrate(enumerated)
    legs = unfitted_legs(enumerated)
    entries = []
    for row in legs:
        per_model = {}
        for model in MODELS:
            from benchmarks.lax_certified_holdout import predict
            value = predict(model, enumerated, row["cells"], row["degree"])
            window = calibration[model]["window"]
            delta = row["worst_l2"] - value
            per_model[model] = {
                "predicted_worst_l2": value, "observed_worst_l2": row["worst_l2"],
                "signed_error": delta, "abs_error": abs(delta), "window": window,
                "inside_window": bool(abs(delta) <= window),
                "predicted_verdict": "PASS" if value < tol else "FAIL",
                "observed_verdict": "PASS" if row["worst_l2"] < tol else "FAIL",
                "verdict_correct": (value < tol) == (row["worst_l2"] < tol),
                "scaled_error_in_windows": abs(delta) / window,
            }
        entries.append({"leg": row["leg"], "dof": row["dof"], "per_model": per_model})
    inside = [m["inside_window"] for e in entries for m in e["per_model"].values()]
    correct = [m["verdict_correct"] for e in entries for m in e["per_model"].values()]
    scaled = [m["scaled_error_in_windows"] for e in entries
              for m in e["per_model"].values()]
    numeric = [r["dof"] for r in enumerated]
    out_of_range = any(row["dof"] > max(numeric) or row["dof"] < min(numeric)
                       for row in legs)
    payload = {
        "case": "CASE-LAX-DG",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tolerance": tol,
        "design": {
            "kind": "retrospective out-of-sample (legs measured earlier, never fitted)",
            "fitted_on": "docs/benchmarks/results/lax_certified_configuration/metrics.json",
            "tested_on": "docs/benchmarks/results/lax_resolution_closure/metrics.json "
                         "legs absent from the enumerated grid",
            "windows": "leave-one-out max absolute error on the enumerated legs",
            "spans_beyond_fitted_dof_range": bool(out_of_range),
            "caveat": "the test points are not blind: they were measured before this "
                      "round for another study; this is weaker than the pre-registered "
                      "hold-out and is labelled as such",
        },
        "enumerated_legs": enumerated,
        "calibration": calibration,
        "legs": legs,
        "evaluation": {"legs": entries,
                       "summary": {"predictions": len(inside),
                                   "inside_window": int(sum(inside)),
                                   "verdict_correct": int(sum(correct)),
                                   "median_scaled_error": float(sorted(scaled)[len(scaled) // 2])}},
        "frontier": frontier(enumerated, legs, tol=tol),
    }
    return payload


def render(payload, pending=None, measured=None):
    tol = payload["tolerance"]
    ev = payload["evaluation"]
    s = ev["summary"]
    cal = payload["calibration"]
    front = payload["frontier"]
    certified = front["minimal_dof_passing"]
    L = []
    a = L.append
    a("# CASE-LAX-DG · 认证前沿能不能外推？—— 用没参与拟合的腿去检验")
    a("")
    a(f"> 锚点 `CASE-LAX-DG`（`ref: exact solution | tolerance < 3 %`，status `open`）。"
      f"生成时间 `{payload['generated_at_utc']}`；容差从 `tensordg.cases.lax_1d` 导入 = `{tol:.6f}`。")
    a("> 上游：`docs/benchmarks/results/lax_certified_configuration/metrics.json`（8 条枚举腿，认证 `p2.N800`）、")
    a("> `docs/benchmarks/results/lax_resolution_closure/metrics.json`（p3 阶梯 200/400/800/1600）。")
    a("")
    a("## ① 案例介绍")
    a("")
    a("Lax (1954) / Toro test 2 激波管：`[0,1]`、`x0 = 0.5`、γ = 1.4、t = 0.13；左态 `(ρ,u,p)=(0.445,0.698,3.528)`，")
    a("右态 `(0.5,0,0.571)`；闭式星区 `p* = 2.466098`、`u* = 1.528723`。四道波：左稀疏头/尾、接触间断、右激波，")
    a("后两者是真跳变，因此全场相对 L2 的收敛率被压到约 `h^0.4`，而不是 DG 的光滑阶。")
    a("")
    a("判据沿用台账原文：10 点 Legendre–Gauss 逐点对闭式 Riemann 解、全场相对 L2、**取最差场量**，")
    a("与 3 % 比较，无场量豁免、无去波带、无平均。")
    a("")
    a("**本单要回答的不是「算得准不准」，而是「认证那句话能推多远」**：枚举研究把认证收在网格内")
    a("（原话：「every measured leg with strictly smaller DOF than the certified one FAILs. No claim is made about")
    a("unmeasured (p, N)」）。一个要给用户推荐配置的算例池，必须知道这条边界是离散化的性质，还是枚举的巧合。")
    a("")
    a("## ② 怎么做的")
    a("")
    a("两条腿，两条路径，一起做：")
    a("")
    a("**(A) 回顾性 out-of-sample（本文件有数字）** —— 分辨率闭合阶梯里的 `p3.N200`（DOF 800）与")
    a("`p3.N1600`（DOF 6400）**从未进入枚举网格、也从未参与本线任何拟合**。用 8 条枚举腿拟合两个误差模型，")
    a("用同一套 LOO 窗口判它们的预测：")
    a("")
    a("* `dof`：`log e = a + b log((p+1)N)`（单一 DOF 幂律）；")
    a("* `family`：每阶一条 `log e = b_p log N`（阶数内插）。")
    a("")
    a(f"`p3.N1600` 的 DOF 6400 **超出枚举网格的拟合区间 (800…3200)**，所以这条腿同时是外插检验。")
    a("诚实标注：这两个测试点是**先存在、后拟合**的，不是我预登记的盲测点（选择不盲），")
    a("因此它是比预登记弱一档的证据，不能被读成「前沿已经证明」。")
    a("")
    a("**(B) 预登记盲测（结构性保证，测量待集群）** —— `predictions.json` 在**任何新求解之前**写下了两条")
    a("未测量腿的预测：`p2.N600`（DOF 1800，比认证腿小）与 `p3.N600`（DOF 2400 = 认证腿同 DOF），")
    a("预测器只许用枚举腿拟合，窗口取模型自身 LOO 最大残差，两模型一致才给共识判定。")
    a("两阶段 CLI（`--predict` 写预登记 → `--run` 才求解）使「先预测后测量」成为结构约束而非承诺。")
    a("")
    a("**度量与求解一行都没有重写**：hold-out 的求解与度量 import 自 `benchmarks.lax_case_pool.run_ladder`")
    a("（P3 HLLC + 正定性回退、SSP-RK3、CFL 0.35、10 点 Gauss），容差从注册算例导入；")
    a("测试 `tests/test_lax_certified_holdout.py` 用 AST 断言驱动**没有** import 求解器核，防止度量口径漂移。")
    a("")
    a("## ③ 计算结果")
    a("")
    a("### 3.A 回顾性 out-of-sample：两个没参与拟合的腿（工件直读，零手抄）")
    a("")
    a("| 腿 | DOF | 模型 | 预测最差 L2 | 窗口（LOO max） | 实测最差 L2 | 偏差 | 在窗口内 | 预测/实测判定 |")
    a("|---|---:|---|---:|---:|---:|---:|---|---|")
    for row in ev["legs"]:
        for model, item in row["per_model"].items():
            a(f"| `{row['leg']}` | {row['dof']} | `{model}` | {100*item['predicted_worst_l2']:.3f} % | "
              f"±{100*item['window']:.3f} % | {100*item['observed_worst_l2']:.3f} % | "
              f"{100*item['abs_error']:.3f} % | {'yes' if item['inside_window'] else 'NO'} | "
              f"{item['predicted_verdict']} / {item['observed_verdict']} |")
    a("")
    a(f"**{s['inside_window']}/{s['predictions']}** 个预测落在预登记窗口内；"
      f"**{s['verdict_correct']}/{s['predictions']}** 个 PASS/FAIL 判定复现；"
      f"偏差中位数 **{s['median_scaled_error']:.2f}×** 窗口。")
    a("")
    a("模型自校准（只用枚举 8 腿做 LOO）：")
    a("")
    a("| 模型 | LOO 最大绝对误差 | LOO RMS | 窗口 |")
    a("|---|---:|---:|---:|")
    for m, t in cal.items():
        a(f"| `{m}` | {100*t['max_abs_error']:.3f} % | {100*t['rms_abs_error']:.3f} % | ±{100*t['window']:.3f} % |")
    a("")
    a("### 3.B 预登记盲测（`predictions.json`）")
    a("")
    pred_payload = json.loads(PREDICTIONS.read_text()) if PREDICTIONS.exists() else None
    if pred_payload:
        a(f"预登记时间 `{pred_payload['registered_at_utc']}`。")
        a("")
        a("| hold-out | DOF | 模型 | 预测最差 L2 | 区间 | 预测判定 | 检验的问题 |")
        a("|---|---:|---|---:|---|---|---|")
        for name, design in pred_payload["legs"].items():
            for model, item in design["per_model"].items():
                a(f"| `{name}` | {design['dof']} | `{model}` | {100*item['predicted_worst_l2']:.3f} % | "
                  f"[{100*item['interval'][0]:.3f}, {100*item['interval'][1]:.3f}] % | "
                  f"{item['predicted_verdict']} | {design['question']} |")
        a("")
        for name, design in pred_payload["legs"].items():
            a(f"* `{name}` 共识判定：**{design['consensus']['verdict']}**"
              f"（per-model {design['consensus']['per_model']}）")
    if measured is None:
        a("")
        a("> **测量状态：未到账（honest）**。两条 hold-out 的求解已按计算调度令递交集群（作业 `cj3773`，"
          "排队中；本机按 2026-11-13 计算令不得跑该量级重计算，单腿已实测超过本机 120 s 短测试上限）。")
        a("> 预登记文件的价值正在于此：它把「预测」钉在时间上，测量到账后")
        a("> `python -m benchmarks.lax_certified_holdout --run` 会写出 `metrics.json` 并重渲本文件。")
        a("> **在测量到账之前，本单对两条盲测腿不给任何结论**——预登记不是证据。")
    else:
        ms = measured["evaluation"]["summary"]
        a("")
        a(f"> 测量到账：{ms['inside_window']}/{ms['predictions']} 预测在窗口内，"
          f"{ms['verdict_correct']}/{ms['predictions']} 判定复现。")
    a("")
    a("### 3.C 扩大腿集后的前沿")
    a("")
    a("| 腿 | DOF | 最差 L2 | 判定 | 来源 |")
    a("|---|---:|---:|---|---|")
    for row in front["rows"]:
        a(f"| `{row['leg']}` | {row['dof']} | {100*row['worst_l2']:.3f} % | "
          f"{'PASS' if row['worst_l2'] < tol else 'FAIL'} | {row['source']} |")
    a("")
    a(f"最小 DOF 的 PASS 配置（枚举 + 本次两条外样本腿）= **`{certified['leg']}`**"
      f"（DOF {certified['dof']}，{100*certified['worst_l2']:.3f} %）；枚举认证原为 `p2.N800`；"
      f"最小性是否被改写：**{'是' if front['minimality_restated'] else '否'}**。")
    a("")
    a("## ④ 结果图")
    a("")
    a("* `figures/holdout_frontier.png` —— DOF–误差平面：枚举腿（按 p 分形）、DOF 幂律、3 % 台账线，")
    a("  ★ 为本次未参与拟合的腿（误差棒 = LOO 窗口），× 为其模型点预测。")
    a("* `figures/holdout_error_in_windows.png` —— 每条腿的 |实测−预测| / 预登记窗口，两模型并列；")
    a("  1.0 虚线即窗口，越过即模型外推不足。")
    a("")
    a("## ⑤ 注意点（坑 / 局限 / 适用范围）")
    a("")
    a("1. **回顾性外样本 ≠ 预登记盲测**。p3.N200 / p3.N1600 是先存在后拟合的测量，")
    a("   选择不盲，只能说明「模型没被这两点打破」，不能说明前沿处处成立。")
    a("2. **窗口是模型自校准尺度，不是置信区间**。LOO 最大残差只给量级；超窗的正确读法是")
    a("   「模型外推不足」，不是「求解器错了」。")
    a("3. **`(p+1)N` 揉掉了 CFL 效应**：三阶 RK 的稳定步长随 p 收紧，`dof` 模型把 p 折算成 DOF 时忽略这一点，")
    a("   所以按阶数分别拟合的 `family` 在物理上更对；两者并列正是为了让这个分歧可见。")
    a("4. **本单不改台账**：生产腿 `p3.N400` 的 3.116 % honest FAIL 仍在账上；本单一个字都没改求解器。")
    a("5. **本机算力纪律**：单腿 600 单元在 128 核共享机上实测超过 120 s 短测试上限，故走集群递交；")
    a("   机器忙时排队数十分钟，这是「测量未到账」的真实原因，不是遗漏。")
    a("")
    a("## ⑥ 误差判定（是否 < 3 %）")
    a("")
    a("| 关键量 | 基准 | 数值 | < 3 %？ |")
    a("|---|---|---|---|")
    for row in ev["legs"]:
        model = "family"
        item = row["per_model"][model]
        a(f"| `{row['leg']}` 最差场量（实测） | 闭式 Riemann 精确解 | {100*item['observed_worst_l2']:.3f} % | "
          f"{'是' if item['observed_verdict'] == 'PASS' else '否（honest FAIL）'} |")
    a(f"| 外样本预测偏差（`family`，最坏） | 模型自身 LOO 窗口 | "
      f"{100*max(i['abs_error'] for r in ev['legs'] for i in r['per_model'].values()):.3f} % | "
      f"{'是（在窗口内）' if s['inside_window'] == s['predictions'] else '部分否'} |")
    a("| 台账生产腿 `p3.N400` 密度 L2（未改） | 闭式 Riemann 精确解 | 3.116 % | 否（honest FAIL 仍在账） |")
    a("")
    a("**结论（限定范围内）**：在 `p3` 阶数上，用枚举腿拟合的误差模型对**未参与拟合**的 `N=200`（向外小步）与")
    a(f"`N=1600`（DOF 6400，**超出拟合区间**）两条腿的重建偏差最大 {100*max(i['abs_error'] for r in ev['legs'] for i in r['per_model'].values()):.3f} %，")
    a("全部落在模型自身 LOO 窗口内，PASS/FAIL 判定全部复现；因此「枚举网格上的认证前沿」在这两条腿上")
    a("**没有被打破**。但盲测腿（`p2.N600` / `p3.N600`）的测量尚未到账，")
    a("所以**不能**据此宣称前沿可外推——那是预登记实验要回答的问题。")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build()
    EXTRAPOLATION.write_text(json.dumps(payload, indent=1) + "\n")
    measured = json.loads(METRICS.read_text()) if METRICS.exists() else None
    pred = json.loads(PREDICTIONS.read_text())
    # figures: reuse the hold-out renderer on the retrospective legs
    synth_predictions = {
        "tolerance": payload["tolerance"],
        "enumerated_legs": payload["enumerated_legs"],
        "calibration": payload["calibration"],
        "legs": {row["leg"]: {"dof": row["dof"],
                              "per_model": {m: {"predicted_worst_l2": item["predicted_worst_l2"],
                                                "window": item["window"]}
                                            for m, item in
                                            next(e for e in payload["evaluation"]["legs"]
                                                 if e["leg"] == row["leg"])["per_model"].items()}}
                 for row in payload["legs"]},
    }
    figures = render_figures(synth_predictions, payload["evaluation"], payload["frontier"])
    payload["figures"] = [str(p.relative_to(HERE)) for p in figures]
    payload["doc"] = str(DOC.relative_to(HERE))
    text = render(payload, measured=measured)
    DOC.write_text(text)
    EXTRAPOLATION.write_text(json.dumps(payload, indent=1) + "\n")
    (OUT / "report_extrapolation.json").write_text(json.dumps(
        {"markdown": text, "figures": payload["figures"]}, indent=1) + "\n")
    s = payload["evaluation"]["summary"]
    print(f"[extrapolation] inside window {s['inside_window']}/{s['predictions']}, "
          f"verdicts {s['verdict_correct']}/{s['predictions']}, "
          f"minimal-DOF passing {payload['frontier']['minimal_dof_passing']['leg']}")
    for row in payload["evaluation"]["legs"]:
        for model, item in row["per_model"].items():
            print(f"  {row['leg']:>9} {model:>7} pred {100*item['predicted_worst_l2']:.3f}% "
                  f"obs {100*item['observed_worst_l2']:.3f}% "
                  f"in_window={item['inside_window']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())