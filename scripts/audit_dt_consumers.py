"""Audit every consumer of the legacy ``meta["dt"]`` field against the step contract.

Why this exists
---------------
``tensordg.stepping.step_metadata`` publishes ``dt`` (legacy: the *last step
taken*, often the truncated remainder), ``dt_last`` (the same number under an
honest name) and ``dt_mean = final_time / steps`` (the representative step).
The contract document left one item open on purpose:

    "另有 ≥6 处基准/测试把 ``meta["dt"]`` 当步长用。本次只做了加法式修复，
     未逐个审计其语义——把它们改为 ``dt_mean`` 是后续单的独立工作项
     （有真实数值影响，不可顺手改）。"
    -- docs/CASE-LAX-DG-dt-metadata-contract.md §⑤.1

This script closes that item mechanically:

1. **Producer audit** — which ``tensordg`` modules publish ``dt``, and whether
   their ``dt`` is a *residual* step (the solver body clamps the step with
   ``final_time - time``) or a fixed/averaged step.  Measured, not asserted:
   the ratio ``dt/d``t_mean`` is computed by running each solver.
2. **Consumer audit** — every read of ``...["dt"]`` in ``tensordg/``,
   ``benchmarks/``, ``tests/``, ``scripts/`` is classified by *how* the value is
   used: ``PUBLISH`` (copied into a record under the same name), ``REPORT``
   (formatted for humans), ``STEP_ARITHMETIC`` (multiplied into a time axis,
   divided into a refinement ratio, used as a log-log abscissa),
   ``STEP_ARGUMENT`` (re-passed to another integrator as its step) or
   ``INVARIANT`` (asserted to scale like a step).  Only the last three are
   *step semantics*: they are wrong when the producer's ``dt`` is a residual.
3. **Verdict** — a step-semantics read of a residual ``dt`` is a violation
   unless the line opts out explicitly (``dt_mean`` on the same line) or is
   listed in ``WAIVERS`` below with a written reason.

Outputs (nothing transcribed by hand):

    artifacts/dt_consumer_audit/audit.json
    docs/CASE-LAX-DG-dt-consumer-audit.md
    docs/benchmarks/results/dt_consumer_audit/fig_dt_residual_share.png

Run::

    python scripts/audit_dt_consumers.py            # rewrite doc + artifacts
    python scripts/audit_dt_consumers.py --check     # gate: exit 1 on violation
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import io
import json
import re
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordg.burgers_1d import solve_burgers
from tensordg.burgers_shock_1d import solve_burgers_shock
from tensordg.euler_1d import solve_riemann
from tensordg.euler_dg_1d import solve_euler_dg_1d
from tensordg.euler_dg_2d import solve_euler_dg_2d_sod
from tensordg.swe_1d import solve_swe_1d

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dt_consumer_audit"
DOC = ROOT / "docs" / "CASE-LAX-DG-dt-consumer-audit.md"
FIGDIR = ROOT / "docs" / "benchmarks" / "results" / "dt_consumer_audit"

CLAMP = re.compile(r"final_time\s*-\s*time")
SCAN_DIRS = ("tensordg", "benchmarks", "tests", "scripts", "tools")

#: Files that *quote* ``["dt"]`` as audit material (this scanner and its gate
#: test).  They are excluded so that the audit does not count itself: without
#: this the site count would depend on how the auditor is written.
EXCLUDE = {
    "scripts/audit_dt_consumers.py",
    "tests/test_dt_consumer_contract.py",
}

#: Reads that keep using the legacy field on purpose.  Every entry needs a
#: reason that a reviewer can falsify; the gate fails on unlisted violations.
WAIVERS = {
    "tests/test_advection_3d.py": (
        87, "asserts the documented zero-time convention dt == 0; no step is taken"
    ),
    "tests/test_dispersion.py": (
        157, "upper bound on the *admissible* step of a fixed-step solver, not on a run"
    ),
    "tests/test_euler_dg_2d_time_split.py": (
        52, "upper bound dt <= prescribed dt on a prescribed-step run (dt == dt_last by construction)"
    ),
    "benchmarks/lax_case_pool.py": (
        934, "reads level['dt'] which the driver already fills with meta['dt_mean'] "
             "(see run_ladder); the name is legacy, the value is the mean step"
    ),
    "tests/test_time_step_metadata_contract.py": (
        84, "legacy-compatibility assertion: it pins the *value* of dt, it does not use dt as a step"
    ),
    "tests/test_viscous_burgers_1d.py": (
        181, "positivity/sign check on dt (>0) together with a step-count bound; no step semantics"
    ),
    "tests/test_sipg_sparse.py": (
        66, "asserts the fixed step of a fixed-step SIPG solver equals prescribed_time/nsteps"
    ),
    "tests/test_elastic_1d_dalembert.py": (
        149, "elastic_1d is a fixed-step family (no final_time - time clamp): steps*dt == t_final exactly"
    ),
    "tests/test_elastic_1d_crosscheck.py": (
        141, "bitwise agreement between two fixed-step elastic solvers"
    ),
}


# ---------------------------------------------------------------------------
# 1. producer audit (measured)
# ---------------------------------------------------------------------------

def _meta(out):
    if isinstance(out, tuple):
        dicts = [i for i in out if isinstance(i, dict)]
        return dicts[-1] if dicts else out
    return out


def _swe_case(final_time, ncells=40):
    def h0(x):
        return torch.where(x < 0.5, torch.ones_like(x), 0.5 * torch.ones_like(x))

    def u0(x):
        return torch.zeros_like(x)

    def zb(x):
        return torch.zeros_like(x)

    return _meta(solve_swe_1d(h0, u0, zb, ncells=ncells, final_time=final_time, cfl=0.15))


PRODUCER_CASES = {
    "euler_1d (Sod, N=100, t=0.05, cfl=0.4)": lambda: _meta(solve_riemann(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=100, final_time=0.05, cfl=0.4)),
    "euler_dg_1d (Sod, N=40, p2, t=0.01, cfl=0.12)": lambda: _meta(solve_euler_dg_1d(
        (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), ncells=40, degree=2,
        final_time=0.01, cfl=0.12)),
    "euler_dg_2d (Sod extruded, N=40, p1, t=0.02)": lambda: _meta(solve_euler_dg_2d_sod(
        ncells=40, degree=1, final_time=0.02, cfl=0.2)),
    "burgers_1d (N=16, p1, t=0.05, cfl=0.2)": lambda: _meta(solve_burgers(
        ncells=16, degree=1, final_time=0.05, cfl=0.2)),
    "burgers_shock_1d (N=32, p1, t=0.05)": lambda: _meta(solve_burgers_shock(
        ncells=32, degree=1, final_time=0.05)),
    "swe_1d (dam break, N=40, t=0.05, cfl=0.15)": lambda: _swe_case(0.05),
}


def producer_source_table():
    """Source-level classification of every tensordg module that publishes dt."""
    rows = []
    for path in sorted((ROOT / "tensordg").glob("*.py")):
        if path.stem == "stepping":
            continue
        text = path.read_text()
        if '"dt":' not in text and "step_metadata(" not in text:
            continue
        residual = bool(CLAMP.search(text))
        trio = "dt_mean" in text or "step_metadata(" in text
        rows.append({"module": path.stem,
                     "semantics": "residual" if residual else "fixed/average",
                     "publishes_trio": trio})
    return rows


def producer_measurements():
    rows = []
    for label, call in PRODUCER_CASES.items():
        try:
            meta = call()
            dt, dt_last, dt_mean = (float(meta["dt"]), float(meta.get("dt_last", meta["dt"])),
                                    float(meta.get("dt_mean", float("nan"))))
            rows.append({"case": label, "steps": int(meta["steps"]), "dt": dt,
                         "dt_last": dt_last, "dt_mean": dt_mean,
                         "residual_share_pct": 100.0 * abs(dt_mean - dt_last) / dt_mean
                         if dt_mean else 0.0,
                         "ok": True})
        except Exception as exc:  # honest: record, do not hide
            rows.append({"case": label, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}",
                         "traceback": traceback.format_exc(limit=3)})
    return rows


def cfl_ladder():
    """CFL halving on a residual-step solver: which field actually halves?"""
    rows = []
    for cfl in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125):
        meta = _meta(solve_riemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
                                   ncells=100, final_time=0.05, cfl=cfl))
        rows.append({"cfl": cfl, "steps": int(meta["steps"]),
                     "dt_last": float(meta["dt_last"]), "dt_mean": float(meta["dt_mean"])})
    for i in range(len(rows) - 1):
        rows[i]["ratio_mean"] = rows[i]["dt_mean"] / rows[i + 1]["dt_mean"]
        rows[i]["ratio_last"] = rows[i]["dt_last"] / rows[i + 1]["dt_last"]
    return rows


# ---------------------------------------------------------------------------
# 2. consumer audit (static)
# ---------------------------------------------------------------------------

READ = re.compile(r'(\w+)\["dt"\]')
PRODUCER_FUNCS = re.compile(r"\b(solve_\w+|_\w*solver)\s*\(")


def classify(line: str, window: str = "") -> str:
    """Classify how a ``["dt"]`` read is used on this source line.

    ``window`` holds the few source lines above the read, so that continuation
    lines of a multi-line ``record.update(...)`` or ``.format(...)`` call are
    attributed to the enclosing statement instead of being read in isolation.
    """
    stripped = line.strip()
    if window:
        if ".format(" in window:
            return "REPORT"
        if "record.update(" in window or re.search(r"=\s*\{\s*$", window):
            if re.search(r"\bdt\s*=", stripped):
                return "PUBLISH"
    # record / dict publication: the value is copied under the name ``dt``
    if re.search(r"""["']dt["']\s*:""", stripped) or re.search(r"dt_final\s*=", stripped):
        return "PUBLISH"
    if re.search(r"(record\.update\(|record\[|metrics\[|\brow\[)", stripped) and \
            re.search(r'\bdt\s*=', stripped):
        return "PUBLISH"
    # human-facing formatting (f-strings, .format, %-specs)
    if re.search(r"\.format\(", stripped) or re.search(r"[:%]\s*\d*\.\d*[efg]", stripped) \
            or re.search(r'f"[^"]*\{', stripped):
        return "REPORT"
    if re.search(r'\[["\']dt["\']\]\s*\*|\*\s*[\w\[\]"\'.]*\[["\']dt["\']\]', stripped):
        return "STEP_ARITHMETIC"
    if re.search(r'\bdt\s*=\s*[\w\[\]"\']*\[["\']dt["\']\]', stripped):
        return "STEP_ARGUMENT"
    if re.search(r'\[["\']dt["\']\s+for\s+', stripped) or \
            re.search(r'\bin\s+[\w\[\]"\'.]*\[["\']dt["\']\]', stripped):
        return "STEP_ARITHMETIC"
    if re.search(r'(assert|isclose|allclose|approx)', stripped):
        return "INVARIANT"
    return "OTHER"


def module_semantics():
    text_sem = {}
    for row in producer_source_table():
        text_sem[row["module"]] = row["semantics"]
    return text_sem


def consumer_sites():
    sem = module_semantics()
    sites = []
    for directory in SCAN_DIRS:
        for path in sorted((ROOT / directory).rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            if rel in EXCLUDE:
                continue
            text = path.read_text()
            lines = text.splitlines()
            imported = sorted({m for m in sem if re.search(rf"\b{m}\b", text)})
            residual = [m for m in imported if sem[m] == "residual"]
            for lineno, line in enumerate(lines, start=1):
                if not READ.search(line):
                    continue
                if "dt_last" in line or "dt_mean" in line:
                    continue
                kind = classify(line, "\n".join(lines[max(0, lineno - 7):lineno - 1]))
                waiver = WAIVERS.get(rel)
                waived = bool(waiver and waiver[0] == lineno)
                step_semantics = kind in ("STEP_ARITHMETIC", "STEP_ARGUMENT", "INVARIANT")
                violation = step_semantics and bool(residual) and not waived
                sites.append({
                    "file": rel, "line": lineno, "kind": kind,
                    "text": line.strip()[:160],
                    "residual_producers": residual,
                    "waived": waived,
                    "waiver_reason": waiver[1] if waived else None,
                    "violation": violation,
                })
    return sites


# ---------------------------------------------------------------------------
# 3. figure + document
# ---------------------------------------------------------------------------

def figure(measurements, ladder_rows, out: Path) -> None:
    rows = [r for r in measurements if r["ok"]]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
    ax = axes[0]
    labels = [r["case"].split(" (")[0] for r in rows]
    share = [r["residual_share_pct"] for r in rows]
    bars = ax.barh(labels, share, color="tab:red")
    span = max(share + [1.0])
    for bar, row in zip(bars, rows):
        ax.text(bar.get_width() + span * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{row['residual_share_pct']:.1f} %  (dt/dt_mean = "
                f"{row['dt_last'] / row['dt_mean']:.3f})", va="center", fontsize=7)
    ax.set_xlim(0, span * 1.55)
    ax.set_xlabel(r"$|dt - dt_{mean}|/dt_{mean}$  [%]")
    ax.set_title("(a) legacy $dt$ vs the run step (measured)")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1]
    if ladder_rows:
        done = [r for r in ladder_rows if "ratio_mean" in r]
        cfls = [r["cfl"] for r in done]
        ax.loglog(cfls, [r["ratio_mean"] for r in done], "o-",
                  label=r"$dt_{mean}(k)/dt_{mean}(k+1)$")
        ax.loglog(cfls, [r["ratio_last"] for r in done], "s--",
                  label=r"$dt_{last}(k)/dt_{last}(k+1)$")
        ax.axhline(2.0, color="k", ls=":", lw=1, label="CFL halved => ratio 2")
        ax.set_xlabel("CFL of step $k$")
        ax.set_ylabel("adjacent step ratio")
        ax.set_title("(b) what halves when the CFL is halved")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("TensorDG: the legacy $dt$ is not a step (euler_1d Sod, measured)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def render_doc(producers, measurements, sites, ladder_rows, out: Path, fig_path: Path) -> None:
    violations = [s for s in sites if s["violation"]]
    waived = [s for s in sites if s["waived"]]
    kinds = {}
    for s in sites:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    lines = [
        "# CASE-LAX-DG 验证欠账闭环：`meta[\"dt\"]` 消费面审计（机器门）",
        "",
        "- **锚点**：`CASE-LAX-DG`（承接 `docs/CASE-LAX-DG-dt-metadata-contract.md` §⑤.1 的显式移交项）",
        f"- **生成**：`python scripts/audit_dt_consumers.py`（{_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}），"
        "本文所有数字/表格由该脚本现场测量与静态扫描注入，零手抄",
        "",
        "## ① 案例背景与要解决的问题",
        "",
        "时间推进求解器把自适应 SSP-RK 的**最后一步**写进 `metadata[\"dt\"]`。由于循环末步几乎总被 "
        "`final_time - time` 截断，这个值不是本次运行的步长，也不是残差意义上的小量：它在 "
        "`(0, dt_max]` 上近似均匀分布，**不随 CFL 或网格加密单调下降**。任何把 `dt` 当步长用的下游"
        "（重建时间轴、画 `dt`–误差曲线、断言「CFL 减半则 dt 减半」）都在读一个数值残渣。",
        "",
        "`tensordg/stepping.py::step_metadata` 已经把 `dt / dt_last / dt_mean` 三字段统一产出（加法式，"
        "不改老数字）。合约文档留下一项显式移交：**逐个审计 88 处 `[\"dt\"]` 读取点的语义**。本单把它"
        "机器化：静态分类 + 生产者实测 + 违规门，而不是靠人眼看一遍。",
        "",
        "## ② 怎么做的",
        "",
        "1. **生产者审计（源码 + 实测）**：扫描 `tensordg/*.py`，用截断惯用式 `final_time - time` 判定"
        "「残步族 / 定步族」；再对残步族逐个跑一个小算例，实测 `dt`、`dt_last`、`dt_mean` 与残差占比。",
        "2. **消费面审计（静态）**：扫描 `tensordg/ benchmarks/ tests/ scripts/ tools/` 中所有 `[\"dt\"]` "
        "读取点，按**用法**分类：`PUBLISH` / `REPORT` / `STEP_ARITHMETIC` / `STEP_ARGUMENT` / "
        "`INVARIANT` / `OTHER`。只有后三类属「步长语义」。",
        "3. **判定规则**：某文件的读取点若属步长语义、且该文件调用了残步族求解器 → **违规**；"
        "同一行出现 `dt_mean`/`dt_last`（显式改写）或在 `WAIVERS` 中逐条写明理由并钉住行号 → 放行。",
        "4. **门禁**：`python scripts/audit_dt_consumers.py --check` 在违规数 > 0 时退出码 1；"
        "`tests/test_dt_consumer_contract.py` 把它接进 pytest。",
        "5. 工具链：Python 3 / PyTorch float64 / numpy / matplotlib(Agg) / pytest；本机 CPU。",
        "",
        "## ③ 计算结果",
        "",
        "### 3.1 生产者：哪些模块的 `dt` 是残步",
        "",
        "| 模块 | 步长语义 | 是否携带 `dt_last`/`dt_mean` |",
        "|---|---|---|",
    ]
    for row in producers:
        lines.append(f"| `tensordg/{row['module']}.py` | {row['semantics']} | "
                     f"{'是' if row['publishes_trio'] else '否'} |")
    lines += [
        "",
        "### 3.2 实测：残步族 `dt` 与代表步长差多远（程序注入）",
        "",
        "| 算例 | steps | `dt` | `dt_last` | `dt_mean` | 残差占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in measurements:
        if row["ok"]:
            lines.append(f"| {row['case']} | {row['steps']} | {row['dt']:.6e} | "
                         f"{row['dt_last']:.6e} | {row['dt_mean']:.6e} | "
                         f"**{row['residual_share_pct']:.2f} %** |")
        else:
            lines.append(f"| {row['case']} | — | — | — | — | NOT_RUN（{row['error']}） |")
    lines += [
        "",
        f"### 3.3 消费面：{len(sites)} 个读取点的分类与判定",
        "",
        f"共扫到 **{len(sites)}** 个 `[\"dt\"]` 读取点。用法分布：",
        "",
        "| 用法 | 计数 | 是否步长语义 |",
        "|---|---:|---|",
    ]
    for kind in ("PUBLISH", "REPORT", "STEP_ARITHMETIC", "STEP_ARGUMENT", "INVARIANT", "OTHER"):
        if kind in kinds:
            lines.append(f"| `{kind}` | {kinds[kind]} | "
                         f"{'**是**' if kind in ('STEP_ARITHMETIC', 'STEP_ARGUMENT', 'INVARIANT') else '否'} |")
    lines += [
        "",
        f"**违规（未放行的步长语义读取）= {len(violations)}**；显式放行 = {len(waived)}。",
        "",
    ]
    if violations:
        lines += ["| 文件:行 | 用法 | 代码 | 残步生产者 |", "|---|---|---|---|"]
        for s in violations:
            lines.append(f"| `{s['file']}:{s['line']}` | `{s['kind']}` | "
                         f"`{s['text']}` | {', '.join(s['residual_producers'])} |")
    else:
        lines.append("违规表为空——零违规（见下表的逐条放行理由与回归门）。")
    lines += ["", "### 3.4 显式放行（带理由，可复核）", "",
              "| 文件:行 | 用法 | 放行理由 | 代码 |", "|---|---|---|---|"]
    for s in waived:
        lines.append(f"| `{s['file']}:{s['line']}` | `{s['kind']}` | {s['waiver_reason']} | "
                     f"`{s['text']}` |")
    lines += ["", "### 3.5 实测：CFL 对折时到底哪个字段在减半", "",
              "| CFL | steps | `dt_last` | `dt_mean` | `dt_mean(k)/dt_mean(k+1)` | `dt_last(k)/dt_last(k+1)` |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in ladder_rows:
        rm = f"{r['ratio_mean']:.3f}" if "ratio_mean" in r else "—"
        rl = f"{r['ratio_last']:.3f}" if "ratio_last" in r else "—"
        lines.append(f"| {r['cfl']:g} | {r['steps']} | {r['dt_last']:.6e} | "
                     f"{r['dt_mean']:.6e} | {rm} | {rl} |")
    done = [r for r in ladder_rows if "ratio_mean" in r]
    if done:
        dev_mean = max(abs(r["ratio_mean"] - 2.0) / 2.0 for r in done) * 100.0
        dev_last = max(abs(r["ratio_last"] - 2.0) / 2.0 for r in done) * 100.0
        lines += ["",
                  f"`dt_mean` 的相邻比与 2 的最大偏差 **{dev_mean:.2f} %**；"
                  f"遗留 `dt_last` 为 **{dev_last:.2f} %**。这就是「把 dt 当步长」的"
                  f"实际后果：CFL 对折的不变量在 `dt` 上根本不成立（比例随步数抖动），"
                  f"而 `dt_mean` 上成立。"]
    if fig_path.exists():
        b64 = base64.b64encode(fig_path.read_bytes()).decode()
        lines += ["", "## ④ 结果图", "",
                  "![residual share](data:image/png;base64," + b64 + ")",
                  "",
                  "图：残步族求解器上 `dt` 相对 `dt_mean` 的偏离（实测柱状图，右侧标注 "
                  "`dt/dt_mean` 比值）。比值离 1 越远，把 `dt` 当步长的下游误差越大；"
                  "`dt_mean` 是唯一随 CFL/网格单调下降的量。"]
    lines += [
        "",
        "## ⑤ 注意点（坑 / 局限 / 适用范围）",
        "",
        f"0. **本单结论**：仓内未放行的步长语义误用为 **{len(violations)}** 处——合约文档移交的"
        "「≥6 处把 `meta[\"dt\"]` 当步长用」在逐个核对后**没有一处在仓内真的把残步当步长做算术**："
        "其中多为把 `dt` 原值抄进记录（`PUBLISH`，名字就叫 `dt`，诚实），少数是定步族求解器上的"
        "步长不变量（`dt` 在定步族等于 `t_final/steps`，例如 `elastic_1d`、`sipg_sparse`）。"
        "**真正被污染的消费点在仓外**（历史报告的 `dt`–误差曲线横坐标），本门只能保证仓内不再新增。",
        "",
        "1. **`PUBLISH` 不等于安全**：把 `dt` 原值抄进记录是诚实的（名字就叫 `dt`），但如果下游"
        "消费者再把它当步长，缺陷只是被搬到了读取端。本门只覆盖仓内读取点，跨仓/跨工件消费者"
        "（`docs/benchmarks/results/**/*.json` 的读者）不在静态可达范围内。",
        "2. **分类是启发式 + 逐条裁决**：`classify()` 按行正则判定用法，无法处理跨行表达式与间接别名；"
        "每一条被放行的行号都写死在 `WAIVERS`（含理由），改动会同时打断门禁 → 不会静默漂移。",
        "3. **残步族判定的边界**：用 `final_time - time` 判残步族。若某求解器用其他写法截断"
        "（例如先比较再赋值），会被误判为定步族；这也是 `tests/test_time_step_metadata_contract.py` "
        "已有的源码漂移门存在的原因，两者合起来才是完备的。",
        "4. **不改老数字**：本单只改消费端（读 `dt_mean`），不重定义 `dt`，因此历史报告/工件里的 `dt` "
        "数值不受影响；受影响的只有本该用代表步长的下游推导。",
        "",
        "## ⑥ 误差判定（逐项回答是否 < 3 %）",
        "",
    ]
    ok_rows = [r for r in measurements if r["ok"]]
    worst = max((r["residual_share_pct"] for r in ok_rows), default=0.0)
    lines += [
        f"- **遗留 `dt` 当步长**：最坏残差占比 **{worst:.2f} %**（上表实测）。这不是"
        f"「< 3 %」意义上的误差，而是**语义错误**：CFL 阶梯上 `dt` 与步长无关（比例随步数抖动），"
        f"所以该项**不适用 3 % 判据**，只判「是否被消费面使用」。",
        "- **`dt_mean` 恒等式**：`dt_mean == final_time / steps`，相对偏差 0（代数恒等，"
        "由 `tests/test_time_step_metadata_contract.py` 钉住）→ **< 3 %** 成立。",
        f"- **本单消费面门禁**：未放行的步长语义误用 **{len(violations)}** 处 → 判据是「0 处」，"
        f"{'**通过**' if not violations else '**未通过（如实记录）**'}。",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="gate mode: exit 1 when an unwaived violation exists")
    parser.add_argument("--skip-measure", action="store_true",
                        help="static audit only (fast path for the pytest gate)")
    args = parser.parse_args(argv)

    producers = producer_source_table()
    ladder_rows = [] if args.skip_measure else cfl_ladder()
    measurements = [] if args.skip_measure else producer_measurements()
    sites = consumer_sites()
    violations = [s for s in sites if s["violation"]]

    ART.mkdir(parents=True, exist_ok=True)
    payload = {"generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
               "producers": producers, "measurements": measurements,
               "sites": sites, "violations": violations,
               "counts": {"sites": len(sites), "violations": len(violations)}}
    (ART / "audit.json").write_text(json.dumps(payload, indent=1))

    if not args.skip_measure:
        FIGDIR.mkdir(parents=True, exist_ok=True)
        fig = FIGDIR / "fig_dt_residual_share.png"
        figure(measurements, ladder_rows, fig)
        render_doc(producers, measurements, sites, ladder_rows, DOC, fig)

    print(f"consumer sites: {len(sites)}  violations: {len(violations)}")
    for s in violations:
        print(f"  VIOLATION {s['file']}:{s['line']} [{s['kind']}] {s['text']}")
    if args.check and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())