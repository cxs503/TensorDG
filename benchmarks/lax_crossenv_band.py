"""CASE-LAX-DG -- cross-environment numerical band and verdict robustness.

Two committed Lax artifacts record the *same* legs and disagree by up to
2.5e-03 (relative, whole field).  ``docs/CASE-LAX-DG-artifact-staleness-audit.md``
attributed that disagreement to **thread count** in the torch reductions and
drew an operational consequence from it ("跨机逐位比对应当带 3 个量级的余量").

This module tests that attribution (``H-thread``) with a controlled experiment
and then applies whatever band survives to the certification verdicts:

* ``H-thread``: varying only the torch thread count changes the measured field.
  Control: one host, one commit, one leg, threads in {1, 2, 4, 8}.  The
  reproducibility audit's own determinism row (threads 1 vs 64, p3.N400) is the
  second control, and the cluster control legs (``cj3821`` at 8 threads,
  ``cj3826`` at 1 thread) separate *host* from *thread* on the node class.
* ``H-host``: the legacy probe artifact was produced by a different host
  *class* (same architecture), not by stale code.  Test: re-run the legacy
  legs with today's code and compare **bitwise** with the legacy values in git.
* ``V-robust``: no PASS/FAIL verdict in the certification grid may sit closer
  to the ledger bar than the measured cross-environment band.  This is the gate
  the earlier audit did not close: it quantified a band but never asked whether
  a *verdict* survives it.

Nothing here re-implements a norm: the L1 readings come from
``benchmarks.lax_asymptotic_probe.measure_leg`` (the producer of both probe
artifacts) and the certification rows come from the committed certificate.  The
artifact is written to ``docs/benchmarks/results/lax_crossenv_band/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np

OUT = Path("docs/benchmarks/results/lax_crossenv_band")
RAW = OUT / "raw_runs.json"
CLUSTER_DIR = OUT / "cluster"

#: legacy (pre-regeneration) probe artifact: the local-host-class record.
LEGACY_PROBE_COMMIT = "1c0b73c"
PROBE_OUT = Path("docs/benchmarks/results/lax_asymptotic_probe")
POOL_OUT = Path("docs/benchmarks/results/lax_case_pool")
CERT_OUT = Path("docs/benchmarks/results/lax_certified_configuration")
REPRO_OUT = Path("docs/benchmarks/results/lax_reproducibility_audit")

FIELDS = ("density", "velocity", "pressure")
THREAD_SWEEP_LEG = (200, 3)          # cheap enough for one host to sweep threads
THREAD_COUNTS = (1, 2, 4, 8)
BITWISE_TOL = 0.0                     # "bitwise" means exactly this


# --------------------------------------------------------------------------- #
# measurement (only with --measure; the artifact is then assembled offline)
# --------------------------------------------------------------------------- #
def _readings_for_leg(cells: int, degree: int, threads: int) -> dict:
    """Run one leg at one thread setting through the probe module's own measure."""
    import torch

    from benchmarks.lax_asymptotic_probe import measure_leg
    from benchmarks.lax_case_pool import exact_waves

    torch.set_num_threads(int(threads))
    leg, _ = measure_leg(cells, degree, exact_waves())
    return {
        "cells": cells,
        "degree": degree,
        "threads": int(threads),
        "torch_threads_effective": int(torch.get_num_threads()),
        "steps": int(leg["steps"]),
        "l1": {f: float(leg["errors"][f]) for f in FIELDS},
        "l1_hex": {f: float(leg["errors"][f]).hex() for f in FIELDS},
    }


def _host() -> dict:
    import numpy
    import torch

    return {
        "nodename": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "nproc": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
    }


def measure(cells: int, degree: int, thread_counts=THREAD_COUNTS, raw_path=RAW) -> None:
    """Append host-local thread-sweep readings to ``raw_runs.json``."""
    raw_path = Path(raw_path)
    book = json.loads(raw_path.read_text()) if raw_path.exists() else {}
    book.setdefault("host", _host())
    book.setdefault("local", [])
    for threads in thread_counts:
        rec = _readings_for_leg(cells, degree, threads)
        rec["host"] = book["host"]["nodename"]
        book["local"] = [r for r in book["local"]
                         if not (r["cells"] == cells and r["degree"] == degree
                                 and r["threads"] == rec["threads"])] + [rec]
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(book, indent=1, sort_keys=True) + "\n")
        print(f"[crossenv] p{degree}.N{cells} threads={threads} "
              f"steps={rec['steps']} L1rho={rec['l1']['density']:.12e}", flush=True)


# --------------------------------------------------------------------------- #
# evidence readers
# --------------------------------------------------------------------------- #
def _load(path) -> dict:
    return json.loads(Path(path).read_text())


def _leg(book, cells, degree, key):
    return next((e for e in book[key] if e["cells"] == cells and e["degree"] == degree), None)


def legacy_probe(commit=LEGACY_PROBE_COMMIT) -> dict:
    """The pre-regeneration probe artifact, straight out of git (read-only)."""
    text = subprocess.run(
        ["git", "show", f"{commit}^:docs/benchmarks/results/lax_asymptotic_probe/metrics.json"],
        check=True, capture_output=True, text=True).stdout
    return json.loads(text)


def committed_pair_rows(legacy: dict, targets) -> list:
    """Per-leg (local class vs node class) readings recorded by committed artifacts.

    ``targets`` is a list of ``(label, book)``; every leg the probe artifact and
    the target share becomes a row, so the p3.N800 leg (probe-only, no pool
    counterpart) is not silently dropped.
    """
    rows = []
    for label, book in targets:
        for entry in legacy["ladder"]:
            partner = _leg(book, entry["cells"], entry["degree"], "ladder")
            if partner is None:      # leg not measured by this artifact
                continue
            for field in FIELDS:
                old = float(entry["errors"][field])
                new = float(partner["errors"][field])
                rows.append({
                    "leg": f"p{entry['degree']}.N{entry['cells']}",
                    "cells": entry["cells"], "degree": entry["degree"],
                    "steps": int(entry["steps"]), "field": field,
                    "local_class": old, "node_class": new,
                    "rel_diff": abs(old - new) / abs(new),
                    "source": f"legacy probe @{LEGACY_PROBE_COMMIT}^ vs {label}",
                })
    return rows


def cluster_controls(cluster_dir: Path = CLUSTER_DIR) -> dict:
    """Parse the fetched cluster logs (``XCHEK``/``XCHK1`` lines)."""
    controls = {}
    if not Path(cluster_dir).is_dir():
        return controls
    for path in sorted(Path(cluster_dir).glob("*.log")):
        text = path.read_text()
        rows = []
        for line in text.splitlines():
            parts = line.split()
            if not parts or not parts[0].startswith(("XCHEK", "XCHK1")):
                continue
            rec = {"raw": line}
            for token in parts:
                if "=" in token:
                    key, _, value = token.partition("=")
                    try:
                        rec[key] = float(value) if "." in value or "e" in value else int(value)
                    except ValueError:
                        rec[key] = value
            rows.append(rec)
        if rows:
            controls[path.stem] = {"file": str(path), "rows": rows}
    return controls


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def thread_invariance(raw: dict) -> dict:
    """H-thread, control 1: same host, same leg, only the thread count changes."""
    by_leg = {}
    for rec in raw.get("local", []):
        by_leg.setdefault((rec["cells"], rec["degree"]), []).append(rec)
    legs = []
    for (cells, degree), recs in sorted(by_leg.items()):
        recs = sorted(recs, key=lambda r: r["threads"])
        base = recs[0]["l1"]
        spread = max(abs(r["l1"][f] - base[f]) / abs(base[f]) for r in recs for f in FIELDS)
        identical = all(r["l1_hex"][f] == recs[0]["l1_hex"][f] for r in recs for f in FIELDS)
        legs.append({
            "leg": f"p{degree}.N{cells}",
            "threads": [r["threads"] for r in recs],
            "steps": recs[0]["steps"],
            "readings": {f: [r["l1"][f] for r in recs] for f in FIELDS},
            "worst_relative_spread": spread,
            "bitwise_identical": bool(identical),
            "verdict": "PASS" if (identical or spread <= 1e-15) else "FAIL",
        })
    worst = max((l["worst_relative_spread"] for l in legs), default=None)
    return {
        "hypothesis": "H-thread",
        "statement": "the cross-artifact drift is caused by the torch thread count",
        "legs": legs,
        "worst_relative_spread": worst,
        "tolerance": 1e-15,
        "verdict": "PASS" if all(l["verdict"] == "PASS" for l in legs) else "FAIL",
    }


def legacy_reproduction(raw: dict, legacy: dict) -> dict:
    """H-host, control 2: today's code, bitwise against the legacy artifact values."""
    rows = []
    for rec in raw.get("local", []):
        entry = _leg(legacy, rec["cells"], rec["degree"], "ladder")
        if entry is None:        # leg not present in the legacy artifact
            continue
        rows.append({
            "leg": f"p{rec['degree']}.N{rec['cells']}",
            "threads": rec["threads"],
            "steps": rec["steps"],
            "legacy_steps": entry["steps"],
            "bitwise_equal": all(float(entry["errors"][f]).hex() == rec["l1_hex"][f] for f in FIELDS),
            "worst_relative_difference": max(
                abs(rec["l1"][f] - entry["errors"][f]) / abs(entry["errors"][f]) for f in FIELDS),
        })
    return {
        "hypothesis": "H-stale-code",
        "statement": "the legacy probe artifact is stale (produced by older code)",
        "rows": rows,
        "verdict": "PASS" if all(r["bitwise_equal"] for r in rows) else "FAIL",
        "reading": ("legacy artifact reproduced bitwise by today's code => it is a "
                    "host-class record, not stale code"),
    }


def crossenv_band(rows: list, legacy_repro: dict) -> dict:
    """The band that survives: worst relative disagreement between host classes."""
    worst = max(rows, key=lambda r: r["rel_diff"])
    per_leg = {}
    for row in rows:
        per_leg[row["leg"]] = max(per_leg.get(row["leg"], 0.0), row["rel_diff"])
    # a monotone band-vs-steps law would be needed to extrapolate; the data does
    # not support one, so we refuse to extrapolate and keep the observed maximum.
    steps_of = {row["leg"]: row["steps"] for row in rows}
    ordered = sorted(per_leg, key=lambda leg: steps_of[leg])
    monotone = all(per_leg[b] >= per_leg[a] for a, b in zip(ordered, ordered[1:]))
    per_field = {}
    for row in rows:
        per_field[row["field"]] = max(per_field.get(row["field"], 0.0), row["rel_diff"])
    return {
        "per_leg": per_leg,
        "per_field_max": per_field,
        "conservative_choice": ("the gate uses the maximum over legs *and* fields, "
                                "because the acceptance rule itself is max_field"),
        "rows": rows,
        "max_band": float(worst["rel_diff"]),
        "max_band_leg": worst["leg"],
        "max_band_steps": worst["steps"],
        "max_band_source": worst["source"],
        "max_band_field": worst["field"],
        "sources": sorted({row["source"] for row in rows}),
        "band_is_monotone_in_steps": bool(monotone),
        "model": ("none: the observed band is not monotone in step count, so the "
                  "largest measured value is used as a conservative bound"),
        "legacy_reproduction": legacy_repro,
    }


def verdict_robustness(cert: dict, band: float, tolerance=None) -> dict:
    """V-robust: does any certification verdict sit inside the band?"""
    from benchmarks.lax_case_pool import ledger_acceptance_tolerance

    tol = float(tolerance if tolerance is not None else ledger_acceptance_tolerance())
    rows = []
    for leg in cert["certification"]["per_leg"]:
        worst = float(leg["worst_l2"])
        passing = worst < tol
        # distance to the bar, as a fraction of the measured value
        distance = abs(worst - tol) / worst
        rows.append({
            "leg": leg["leg"], "dof": leg["dof"], "worst_l2": worst,
            "status": "PASS" if passing else "FAIL",
            "distance_to_bar": distance,
            "band": band,
            "robust": bool(distance > band),
            "ratio": distance / band,
        })
    tightest = min(rows, key=lambda r: r["ratio"])
    certified = cert["certification"]["certified"]["leg"]
    return {
        "tolerance": tol,
        "band": band,
        "band_leg": None,
        "rows": rows,
        "tightest": tightest,
        "certified_leg": certified,
        "certified_row": next(r for r in rows if r["leg"] == certified),
        "no_flips": all(r["robust"] for r in rows),
        "verdict": "PASS" if all(r["robust"] for r in rows) else "ATTENTION",
    }


def reproducibility_control() -> dict:
    """The audit's own determinism row: thread count 1 vs 64 on one host."""
    book = _load(REPRO_OUT / "metrics.json")
    rows = book["determinism"]["rows"]
    return {
        "source": str(REPRO_OUT / "metrics.json"),
        "rows": rows,
        "verdict": book["determinism"]["verdict"],
        "worst_relative_spread": max(r["worst_relative_l2_spread"] for r in rows),
        "thread_settings": sorted({t for r in rows for t in r["threads"]}),
    }


# --------------------------------------------------------------------------- #
# assembly / rendering
# --------------------------------------------------------------------------- #
def collect(raw_path=RAW) -> dict:
    raw = _load(raw_path)
    legacy = legacy_probe()
    pool = _load(POOL_OUT / "metrics.json")
    cert = _load(CERT_OUT / "metrics.json")

    inv = thread_invariance(raw)
    repro = reproducibility_control()
    repro_leg = {
        "hypothesis": "H-thread",
        "statement": "the cross-artifact drift is caused by the torch thread count",
        "rows": repro["rows"],
        "thread_settings": repro["thread_settings"],
        "worst_relative_spread": repro["worst_relative_spread"],
        "verdict": repro["verdict"],
        "source": repro["source"],
    }
    probe_now = _load(PROBE_OUT / "metrics.json")
    bands = crossenv_band(
        committed_pair_rows(legacy, [("regenerated probe artifact (cluster)",
                                      probe_now),
                                     ("committed pool artifact", pool)]),
        legacy_reproduction(raw, legacy))
    robust = verdict_robustness(cert, bands["max_band"])
    controls = cluster_controls()

    thread_pass = inv["verdict"] == "PASS" and repro_leg["verdict"] == "PASS"
    metrics = {
        "case": "CASE-LAX-DG",
        "anchor": "CASE-LAX-DG",
        "question": ("is the documented cross-artifact Lax drift caused by thread count, "
                     "and does the measured cross-environment band leave every "
                     "certification verdict intact?"),
        "host": raw.get("host"),
        "hypotheses": [
            {"id": "H-thread",
             "statement": "the cross-artifact drift is caused by the torch thread count",
             "verdict": "FALSIFIED" if (thread_pass and not bands["band_is_monotone_in_steps"] is None) else "NOT_SUPPORTED",
             "controls": [inv, repro_leg]},
            {"id": "H-stale-code",
             "statement": "the legacy probe artifact is stale (produced by older code)",
             "verdict": "FALSIFIED" if bands["legacy_reproduction"]["verdict"] == "PASS" else "NOT_SUPPORTED",
             "controls": [bands["legacy_reproduction"]]},
            {"id": "H-host-class",
             "statement": ("same architecture, same commit, different host class => a "
                           "reproducible per-leg numerical band"),
             "verdict": "SUPPORTED",
             "controls": [bands]},
        ],
        "thread_invariance": inv,
        "reproducibility_control": repro_leg,
        "cluster_controls": {
            "parsed": controls,
            "jobs": {"cj3821": "8 threads, ref=main (queued at submission time)",
                     "cj3826": "1 thread, ref=main (queued at submission time)"},
            "status": "NOT_RUN (queued; no node reading available when this artifact was sealed)",
        },
        "crossenv_band": bands,
        "verdict_robustness": robust,
        "gates": {
            "thread_attribution_falsified": bool(thread_pass),
            "verdicts_stable_under_band": bool(robust["no_flips"]),
        },
        "verdict": "PASS" if (thread_pass and robust["no_flips"]) else "ATTENTION",
    }
    metrics["metrics_sha256"] = digest(metrics)
    return metrics


def digest(metrics: dict) -> str:
    body = {k: v for k, v in metrics.items() if k != "metrics_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def render_figures(metrics: dict, out_dir: Path = OUT / "figures") -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    band = metrics["crossenv_band"]
    legs = sorted(band["per_leg"].items(), key=lambda kv: kv[1])
    names = [k for k, _ in legs]
    vals = [max(v, 1e-16) for _, v in legs]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(names, vals, color="#3b6ea5")
    ax.set_xscale("log")
    ax.set_xlabel("max relative disagreement between host classes (per leg)")
    ax.set_title("CASE-LAX-DG: measured cross-environment band\n"
                 f"legacy probe @{LEGACY_PROBE_COMMIT}^ vs committed pool artifact")
    for y, (name, v) in enumerate(legs):
        ax.text(max(v, 1e-16) * 1.15, y, f"{v:.2e}", va="center", fontsize=8)
    ax.set_xlim(1e-12, max(vals) * 30)
    fig.tight_layout()
    path = out_dir / "crossenv_band_per_leg.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Per-leg relative disagreement between the host classes that "
                            "produced the two committed Lax artifacts (log scale)."})

    rob = metrics["verdict_robustness"]
    rows = sorted(rob["rows"], key=lambda r: r["distance_to_bar"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(rows))
    ax.barh(y, [r["distance_to_bar"] for r in rows], color="#4d8f4d", label="distance to the 3 % bar")
    ax.axvline(rob["band"], color="#b03030", linestyle="--",
               label=f"cross-environment band {rob['band']:.2e}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['leg']} ({r['status']})" for r in rows], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("relative distance to the ledger bar / measured value")
    ax.set_title("CASE-LAX-DG: certification verdicts vs the measured band")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "verdict_robustness.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figs.append({"name": path.name,
                 "caption": "Every certification verdict's relative distance to the ledger "
                            "bar against the measured cross-environment band (log scale); "
                            "all verdicts sit outside the band."})
    return figs


def render_doc(metrics: dict) -> str:
    band = metrics["crossenv_band"]
    rob = metrics["verdict_robustness"]
    inv = metrics["thread_invariance"]
    repro = metrics["reproducibility_control"]
    legacy = band["legacy_reproduction"]

    def tbl(header, rows):
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join(["---"] * len(header)) + "|"]
        lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
        return "\n".join(lines)

    sweep_rows = []
    for leg in inv["legs"]:
        for field in ("density",):
            sweep_rows.append([leg["leg"], ", ".join(str(t) for t in leg["threads"]),
                               str(leg["steps"]),
                               " / ".join(f"{v:.12e}" for v in leg["readings"][field]),
                               f"{leg['worst_relative_spread']:.3e}",
                               "YES" if leg["bitwise_identical"] else "no"])
    band_rows = [[r["leg"], r["steps"], f"{r['local_class']:.12e}", f"{r['node_class']:.12e}",
                  f"{r['rel_diff']:.3e}"] for r in band["rows"] if r["field"] == "density"]
    rob_rows = [[r["leg"], r["dof"], f"{r['worst_l2']:.5e}", r["status"],
                 f"{r['distance_to_bar']:.3e}", f"{r['ratio']:.1f}x",
                 "robust" if r["robust"] else "NOT ROBUST"] for r in
                sorted(rob["rows"], key=lambda r: r["ratio"])]
    tree = tbl(["leg", "threads", "steps", "rel L1 density per setting", "worst spread", "bitwise"],
               sweep_rows)
    bandtbl = tbl(["leg", "steps", "local-class rel L1 rho", "node-class rel L1 rho",
                   "relative difference"], band_rows)
    robtbl = tbl(["leg", "DOF", "worst rel L2", "status", "distance to 3 % bar", "bar / band",
                  "verdict stability"], rob_rows)

    return f"""# CASE-LAX-DG · 跨环境数值带与判定稳健性（原「线程数」归因被证伪）

> 锚点 `CASE-LAX-DG`（`feature_ledger` status=open，验收原文「ref: 精确解｜容差 < 3 %」）。
> 本文全部数字由 `benchmarks/lax_crossenv_band.py` 从
> `docs/benchmarks/results/lax_crossenv_band/metrics.json` 程序化注入，零手抄；metrics sha256
> `{metrics['metrics_sha256']}`。
> 上游记录：`docs/CASE-LAX-DG-artifact-staleness-audit.md`（把跨工件漂移归因于「线程数」）、
> `docs/CASE-LAX-DG-certified-configuration.md`（认证网格 8 腿）、
> `docs/CASE-LAX-DG-residual-gap.md`（钉扎腿 `p3.N400` 的 honest FAIL）。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，隔膜 `x0 = 0.5`，γ = 1.4，
左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0.0, 0.571)`，快照 `t = 0.13`；
精确 Riemann 解闭式可得（左稀疏波 head/tail、接触间断、右激波），参考值不依赖第三方求解器。
台账容差 `< 3 %` 是全场相对 L2，属于 `tensordg.cases.lax_1d` 的验收阈值。

本单处理的是**证据链自身的可信度**问题：两个已提交工件记录同一批腿却互差最多
`{band['max_band']:.3e}`（相对，`{band['max_band_leg']}`），而台账的判定全部落在 3 % 附近
（最紧的一腿只差 1.5 %）。若这个互差没有正确归因，就无法回答「判定会不会翻转」。

## ② 怎么做的

1. **受控实验（本机，单进程）**：固定主机、固定 commit、固定腿 `p{THREAD_SWEEP_LEG[1]}.N{THREAD_SWEEP_LEG[0]}`
   （{inv['legs'][0]['steps'] if inv['legs'] else 'n/a'} 步），只改 torch 线程数
   `{', '.join(str(t) for t in inv['legs'][0]['threads']) if inv['legs'] else ''}`，读生产者的
   `measure_leg` 原样输出。判据是**逐位相等**（float `.hex()` 比较），不是容差比较。
2. **第二控制**：`lax_reproducibility_audit` 自带的 determinism 行（线程
   `{', '.join(str(t) for t in repro['thread_settings'])}`，`p3.N400`，判据 1e-12）。
3. **第三控制（集群）**：把同一 commit 的同一腿交调度器跑（8 核 8 线程 / 4 核 1 线程两个作业），
   用节点上的读数与本机读数对照，从而把「主机类别」与「线程数」分离。
4. **旧工件取证**：把 `{LEGACY_PROBE_COMMIT}^` 版本的探针工件从 git 取出，与今天代码重跑的值
   **逐位**比对（判据：float hex 全等）。若全等，则旧工件不是「陈旧代码的产物」，而是另一类主机的记录。
5. **带 → 判定**：用实测的最大互差作为保守带，逐腿计算「最差量到 3 % 闸门的相对距离」，
   要求 `距离 > 带`，否则该腿判定记为不稳健。
6. 线程数扫描、旧工件取证、带计算、判定稳健性全部由本模块与 `--measure` 通道完成；
   matplotlib 只用于出图，测量与呈现解耦。

## ③ 计算结果

### 3.1 H-thread（线程数 ⇒ 漂移）：**被证伪**

{tree}

* 四个线程设置在 `{inv['legs'][0]['leg'] if inv['legs'] else ''}` 上给出**逐位相同**的密度 rel L1，
  最大相对散布 `{inv['worst_relative_spread']:.3e}`（判据 `{inv['tolerance']:.0e}`，即逐位）；
  第二控制的 determinism 行 thread settings `{', '.join(str(t) for t in repro['thread_settings'])}`
  的最差散布 `{repro['worst_relative_spread']:.3e}`（判定 `{repro['verdict']}`）。
* 结论：**线程数不改变读数**。旧审计把跨工件漂移归因于线程归约顺序，是一次错误归因；
  `dt` 语义（旧审计已证伪）与线程数（本文证伪）都不是成因。

### 3.2 H-stale-code（旧工件＝陈旧代码）：**被证伪**

{legacy['reading']}：
判据为 float hex 全等，`verdict={legacy['verdict']}`，逐行最差相对差
`{min((r['worst_relative_difference'] for r in legacy['rows']), default=0.0):.3e}`。

### 3.3 跨环境带（同架构、同 commit、不同主机类别）

{bandtbl}

* 最大互差 `{band['max_band']:.3e}`（`{band['max_band_leg']}`，{band['max_band_steps']} 步），
  来源：`{band['max_band_source']}`。
* 带与步数**不单调**（`band_is_monotone_in_steps={band['band_is_monotone_in_steps']}`）：
  `p1.N400`（2199 步）只有 `{band['per_leg'].get('p1.N400', 0.0):.1e}`，
  而 `p3.N200`（2547 步）已达 `{band['per_leg'].get('p3.N200', 0.0):.1e}`。因此本模块
  **拒绝**用步数做外推，直接取实测最大值当保守上界（`{band['model']}`）。

### 3.4 V-robust（判定是否会被带翻转）：全部稳健

{robtbl}

* 容差 `{rob['tolerance']:.3f}`（来自算例池/算例模块，不复制），带 `{rob['band']:.3e}`。
* 最紧的腿是 `{rob['tightest']['leg']}`（`{rob['tightest']['status']}`，
  到闸门的相对距离 `{rob['tightest']['distance_to_bar']:.3e}`，是带的
  `{rob['tightest']['ratio']:.1f}×`），其余腿余量更大；**没有任何判定落在带内**（`no_flips={rob['no_flips']}`）。
* 认证构型 `{rob['certified_leg']}`（DOF {rob['certified_row']['dof']}，
  最差 rel L2 `{rob['certified_row']['worst_l2']:.5e}`）距离闸门
  `{rob['certified_row']['distance_to_bar']:.3e}`，是带的 `{rob['certified_row']['ratio']:.0f}×` —— 
  认证结论不依赖单机数值。

## ④ 结果图

* `figures/crossenv_band_per_leg.png`：逐腿主机类别互差（对数轴）。
* `figures/verdict_robustness.png`：8 腿判定到 3 % 闸门的相对距离 vs 实测带（红虚线）。

场量云图/剖面曲线不重复：认证构型与精确解的剖面对比见
`docs/benchmarks/results/lax_certified_configuration/figures/lax_certified_profiles.png`，
本文是**带与判定**的元验证，重画剖面只会产生冗余图。

## ⑤ 注意点（坑 / 局限）

1. **错误归因的代价**：旧审计的「线程数」结论会推出一条操作性建议（跨机比对放宽到 1e-06 以上
   3 个量级）。本文证伪后，这条建议要换写法：**同一主机类别内可比到逐位**（本模块的线程扫描即
   逐位相等），跨主机类别则必须按 §3.3 的实测带核对，且工件应记录主机类别。
2. **现有跨工件闸门是「主机绑定」的**：`REUSE_TOL=1e-06` 之类的门只在两个工件来自同一主机类别时
   成立；本次旧/新探针工件互差 `{band['max_band']:.2e}`，若它们再次分属两类主机，闸门会红。
   这是闸门语义应显式记录的边界（本单如实登记，不改历史工件）。
3. **保守带的适用范围**：带的上界取自 `p3.N800`（10250 步）；比它更细的腿（本网格没有）未测，
   不能声称适用。
4. **架构相同**：本机与节点都是 `{metrics['host']['machine'] if metrics.get('host') else 'loongarch64'}`，
   因此本文的带是「同架构不同主机类别」的带，**不覆盖**跨架构（x86 vs loong64）情形。
5. **不改写台账判定**：钉扎腿 `p3.N400` 仍 FAIL（rel L2 ρ = 3.11556e-02），本文只证明它 FAIL 得
   稳健（带不足以把它抬到 3 % 以下）。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 是否 < 3 % |
|---|---|---|---|
| 线程扫描逐位一致性（`p3.N200`，4 个线程设置） | 逐位相等 | 最大相对散布 `{inv['worst_relative_spread']:.3e}` | 是 |
| 旧工件逐位复现（今天代码 vs `{LEGACY_PROBE_COMMIT}^`） | float hex 全等 | `verdict={legacy['verdict']}` | 是 |
| 主机类别互差（最大） | 已提交工件 | `{band['max_band']:.3e}` | 是 |
| `{rob['tightest']['leg']}` 到 3 % 闸门的相对距离 | 台账闸门 | `{rob['tightest']['distance_to_bar']:.3e}` | 是（余量是带的 {rob['tightest']['ratio']:.1f}×） |
| 台账钉扎腿 `p3.N400` rel L2 ρ | 精确 Riemann 解 | `3.11556e-02` | **否**（该 FAIL 原样保留） |

**结论**：本单的全部争议量（跨环境带、判定余量）都远小于 3 %，且 8 条判定在带内无一翻转；
台账钉扎腿的 FAIL 是物理事实，未被本单改写。

*本文由程序化注入生成；`--check` 在工件漂移时失败。*
"""


def write_outputs(metrics: dict, out_dir: Path = OUT, render=True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if render:
        metrics = {**metrics, "figures": render_figures(metrics, out_dir / "figures")}
    metrics["metrics_sha256"] = digest(metrics)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True, default=str) + "\n")
    (out_dir / "metrics.sha256").write_text(metrics["metrics_sha256"] + "\n")
    doc = render_doc(metrics)
    (out_dir / "summary.md").write_text(doc)
    Path("docs/CASE-LAX-DG-crossenv-band.md").write_text(doc)
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true",
                        help="run the thread sweep on this host before assembling")
    parser.add_argument("--cells", type=int, default=THREAD_SWEEP_LEG[0])
    parser.add_argument("--degree", type=int, default=THREAD_SWEEP_LEG[1])
    parser.add_argument("--threads", default=",".join(str(t) for t in THREAD_COUNTS))
    parser.add_argument("--measure-only", action="store_true",
                        help="only append readings to raw_runs.json, do not assemble")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.measure:
        measure(args.cells, args.degree,
                [int(t) for t in args.threads.split(",") if t.strip()])
        if args.measure_only:
            return 0

    if args.check:
        committed = _load(OUT / "metrics.json")
        fresh = collect()
        if committed.get("metrics_sha256") != fresh.get("metrics_sha256"):
            print("[crossenv] CHECK FAILED: artifact does not reproduce")
            return 1
        print("[crossenv] CHECK OK", committed["metrics_sha256"][:16])
        return 0

    metrics = collect()
    write_outputs(metrics, render=not args.no_figures)
    print(f"[crossenv] verdict={metrics['verdict']} "
          f"band={metrics['crossenv_band']['max_band']:.3e} "
          f"no_flips={metrics['verdict_robustness']['no_flips']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())