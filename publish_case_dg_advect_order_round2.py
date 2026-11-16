"""Render and publish the CASE-DG-ADVECT-ORDER round-2 report.

Every number *and every verdict claim* in the markdown is injected from
``validation/advection_order_matched_rk/metrics.json``; nothing is retyped here.
The <3% question is answered explicitly at the top (the case's declared
tolerance is the *convergence rate*, so the criterion is the order gap against
the reference column, plus the exact error split).

Rendering rules that matter for correctness of the report:

* the *design* time order of each rung is read from the metrics
  (``dt_order_probes[*].design_order``, i.e. the order of the tableau that was
  actually used) instead of being assumed to be ``degree + 1`` -- at ``p = 4``
  the stage-matched scheme is 5-stage *fourth* order (Butcher's barrier), so the
  ``degree + 1`` assumption silently marks a correct run as failed;
* the M2 (stage count) and M5 (measured time order) paragraphs are generated
  from ``metrics["checks"]``, so a PASS cannot be left documented as a FAIL and
  vice versa;
* the mesh ladder of each rung is printed from the metrics, because the
  fragments may have been run on different ladders.

    PYTHONPATH=. python publish_case_dg_advect_order_round2.py

``MISSION_ID`` (default 20415, the round-2 mission) can be overridden through
the environment so the same renderer can publish a re-open round.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = int(os.environ.get("MISSION_ID", "20415"))
METRICS = pathlib.Path("validation/advection_order_matched_rk/metrics.json")
DOC = pathlib.Path("docs/CASE-DG-ADVECT-ORDER-round2.md")
FIG = pathlib.Path("docs/figures")


def post(path, payload, token=None):
    req = urllib.request.Request(f"{BASE}/{path}", data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode())


def b64(path: pathlib.Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def design_order(probe: dict) -> int:
    """Order of the tableau that produced ``probe`` (not ``degree + 1``)."""
    order = probe.get("design_order")
    return probe["degree"] + 1 if order is None else int(order)


def render(m: dict) -> str:
    mk = lambda ok: {True: "✅", False: "❌", "not_run": "⬜NOT RUN"}[ok]  # noqa: E731
    rows = m["comparison"]
    checks = m["checks"]
    splits = m["error_split"]
    stage = m["time_share_vs_stages"]
    probes = m["dt_order_probes"]
    tol_rate = m["thresholds"]["RATE_TOL"]
    tol_dt = m["thresholds"]["DT_ORDER_TOL"]
    rate_ok = checks["M1_rate"]
    gap_max = max(abs(r["order_gap_vs_ref"]) for r in rows)
    gap_rel = max(abs(r["order_gap_vs_ref"]) / abs(r["ref_fitted_order"]) for r in rows)
    share_max = max(s["time_share"] for s in splits)
    ratio_means = "/".join(f"{r['ratio_mean']:.2f}" for r in rows)
    ladder_txt = "；".join(f"P{r['degree']}: {r['meshes']}" for r in rows)
    shared_ladder = len({tuple(r["meshes"]) for r in rows}) == 1
    scheme_txt = "，".join(
        f"P{r['degree']}={r['integrator']}({r['rk_stages']}级/{r['rk_order']}阶)"
        for r in rows)
    expm_nmax = max(s["ncells"] for s in splits)
    expm_dim = max(s["ncells"] * (s["degree"] + 1) for s in splits)
    measured = [p for p in probes if p["fitted_time_order"] is not None]
    probe_txt = "/".join(f"P{p['degree']}={p['fitted_time_order']:.3f}" for p in measured)
    design_txt = "/".join(f"P{p['degree']}={design_order(p)}" for p in probes)
    m5_dev = ("n/a" if len(measured) != len(probes) else
              f"{max(abs(p['fitted_time_order'] - design_order(p)) for p in probes):.3f}")

    # ---- M2 / M5 paragraphs, derived from the checks ----------------------
    if checks.get("M2_stages") is True:
        m2 = (f"- **M2 = PASS（级数匹配）**：逐阶所用表的级数与 ref 列一致"
              f"（{scheme_txt}，ref 级数 "
              f"{'/'.join(str(r['ref_rk_stages']) for r in rows)}），"
              f"且每阶都取该级数能达到的最高阶（`min(p+1, 可达阶)`）。"
              f"$p=4$ 的 ref 列 `RK5DG4` 是 **5 级 4 阶**：5 级 5 阶显式 RK 不存在"
              f"（Butcher 障碍），第 2 轮把它当成 `order = stages = p+1` 才留下假 FAIL。")
    else:
        m2 = (f"- **M2 = FAIL（未匹配）**：所用表级数与 ref 列不一致或阶不等于该级数可达阶"
              f"（{scheme_txt}）。")
    if checks.get("M5_dt_order") is True:
        m5 = (f"- **M5 = PASS（时间阶实测）**：固定网格、步数逐次加倍，对 `expm(L·T)u₀` 精确半离散解测斜率，"
              f"实测 {probe_txt}，设计阶 {design_txt}，最大偏差 {m5_dev}（容差 ±{tol_dt}）。"
              f"早期版本把 `dt`（随细分减小）当作分辨率变量传给假定分辨率递增的估计器，"
              f"得到负斜率并记为 NOT RUN；按步数拟合后四个阶全部实测。")
    else:
        m5 = (f"- **M5 = NOT RUN/FAIL（仪器缺陷）**：实测 {probe_txt or 'n/a'}，"
              f"设计阶 {design_txt}，偏差 {m5_dev} 超容差 ±{tol_dt} 或拟合点不足。")

    out = [f"""# CASE-DG-ADVECT-ORDER（第2轮）：同阶时间格式复测 + 精确误差拆分

**锚点** CASE-DG-ADVECT-ORDER ｜ **判据** 收敛率（观测阶 vs 外部 ref 同阶列，±{tol_rate}）｜
**算例** 1-D 周期线性对流，manufactured solution `u(x,0)=sin(2πx)`，`t=1` ｜
**仓库** TensorDG ｜ **总判定** **{m['verdict']}**（未过项：{', '.join(m['failed']) or '无'}；未测项：{', '.join(m.get('not_run') or []) or '无'}）

> 本文件由 `publish_case_dg_advect_order_round2.py` 从
> `validation/advection_order_matched_rk/metrics.json` 生成：数字与判据文字均程序化注入，零手抄。

## ⓪ 误差判定（是否 <3%）

1. **收敛率（本算例声明的判据）**：P1–P4 观测阶与 ref 同阶列的**绝对**偏差最大
   {gap_max:.3f}（相对偏差最大 {100*gap_rel:.2f}%）→ 绝对判据 ±{tol_rate} 阶：**{mk(rate_ok)}**；
   相对偏差 <3%：**{mk(gap_rel < 0.03)}**（{100*gap_rel:.2f}%）。
2. **时间误差占比**：cfl={m['setup']['cfl']} 下 `E_time/E_total` 最大 {100*share_max:.2f}%（阈 {100*m['thresholds']['TIME_SHARE_TOL']:.0f}%）
   → **{mk(checks['M4_time_share'])}**，即所测阶来自空间离散，不是时间格式污染。
3. **L2 幅值 vs ref**：**无基准可比** —— ref 表未给出 Δt/CFL，且同 N 比值为 {ratio_means}（逐网格近乎常数），
   说明两码同阶不同常数；本轮把"级数差"这一解释**证否**（见 §5），故幅值不作判据。

## 1. 案例介绍

一维周期线性对流 $u_t+cu_x=0$，$x\\in[0,1)$，$c=1$，$u(x,0)=\\sin(2\\pi x)$，$t^*=cT/L=1$。
精确解解析已知，故可测**观测收敛阶**（光滑解应 $p+1$）。

| 项 | 符号/定义 | 值 |
|---|---|---|
| 单元数 | $N$ | {', '.join(str(n) for n in m['setup']['cells'])} |
| 多项式阶 | $p$ | {', '.join(str(d) for d in m['setup']['degrees'])} |
| CFL | $c\\Delta t/h$ | {m['setup']['cfl']} |
| 时间格式 | 本轮 | 级数匹配表：`tableau_for_degree(p)` → {scheme_txt} |
| 无量纲末时 | $t^*$ | {m['setup']['final_time']} |

**ref 出处（数值，零手抄）**：{m['reference']['citation']['author']}，
*{m['reference']['citation']['title']}*，{m['reference']['citation']['journal']}，
DOI {m['reference']['citation']['doi']}，表 `{m['reference']['citation']['table_label']}`；
由 `tools/fetch_ref_advorder.py` 机械解析（仅取 `sdA-RKDG` 之前的**标准 RKDG** 半边，
列 `RK2DG1/RK3DG2/RK4DG3/RK5DG4`，级数 2/3/4/5）。

## 2. 怎么做的

1. **补能力**：新增 `tensordg/runge_kutta.py`（Euler / RK2-Heun / RK3-Kutta / RK4 / SSP-RK(5,4) / Butcher-RK5
   显式表 + `tableau_for_degree` 先匹配级数、再取该级数可达的最高阶），`solve_advection` 增加
   `euler|rk2|rk3|ssp_rk54|rk5` 入口，**`rk4`/`ssp_rk3` 走原手写分支，逐位不变**（有回归测试绑定存档 L2）。
2. **同阶复测（A）**：$p$ 用 $p+1$ 级格式（$p\\le3$ 时同时是 $p+1$ 阶）跑 $N$={m['setup']['cells']} 阶梯，
   与 ref 同阶列对比；阶估计沿用第 1 轮**同一最小二乘估计器**（`fit_with_floor`，$E<10^{{-12}}$ 舍入排除）。
3. **精确误差拆分（B、C）**：线性对流的 DG 半离散系统是线性的，故 `expm(L·T)u₀` 是其**精确解**
   （$L$ 由求解器自身 `_rhs` 逐单位向量生成，不另写一套空间算子）。
   与精确解比 → $E_{{time}}$（纯时间误差）；与连续解比 → $E_{{total}}$；$E_{{space}}$ 由精确半离散解给出。
4. **时间阶独立实测（D）**：固定网格、步数逐次加倍，对精确半离散解测斜率 = 设计阶（判据 ±{tol_dt}）。

## 3. 计算结果

### 3.1 同阶级数复测（A）

| p | 格式 | 级数 | 阶 | 可达阶 | ref 列 | ref 级数 | ref 拟合阶 | TensorDG 拟合阶 | 差 | 判定 |
|--:|---|---:|---:|---:|---|--:|---:|---:|---:|:--:|"""]
    for r in rows:
        out.append(f"| P{r['degree']} | {r['integrator']} | {r['rk_stages']} | {r['rk_order']} | "
                   f"{r['max_order_for_stages']} | {r['ref_column']} | "
                   f"{r['ref_rk_stages']} | {r['ref_fitted_order']:.3f} | {r['got_fitted_order']:.3f} | "
                   f"{r['order_gap_vs_ref']:+.3f} | {mk(r['rate_ok'])} |")

    ncols = len(m["setup"]["reference_meshes"])
    head = " | ".join(f"N={n}" for n in m["setup"]["reference_meshes"])
    out += ["", "### 3.2 L2 幅值比（ref Δt 未公布 → 仅作诊断）", "",
            f"| p | {head} | 均值 | 第1轮(RK4) 均值 | 落在 {m['thresholds']['RATIO_BAND']} |",
            "|--:|" + "---:|" * (ncols + 3) + ":--:|"]
    for r in rows:
        vals = [f"{v:.3f}" for v in r["ratio_got_over_ref"]]
        vals += ["n/a"] * (ncols - len(vals))
        q = " / ".join(vals)
        b = "n/a" if r["baseline_ratio_mean"] is None else f"{r['baseline_ratio_mean']:.3f}"
        out.append(f"| P{r['degree']} | {q} | {r['ratio_mean']:.3f} | {b} | {mk(r['ratio_in_band'])} |")

    out += ["", f"### 3.3 精确时间/空间误差拆分（cfl={m['setup']['cfl']}）", "",
            "| p | N | 格式 | $E_{total}$ | $E_{space}$ | $E_{time}$ | $E_{time}$ 占比 |",
            "|--:|--:|---|--:|--:|--:|--:|"]
    for s in splits:
        out.append(f"| P{s['degree']} | {s['ncells']} | {s['integrator']} | {s['E_total']:.3e} | "
                   f"{s['E_space']:.3e} | {s['E_time']:.3e} | {100*s['time_share']:.2f}% |")

    out += ["", "### 3.4 时间阶独立实测（D，固定网格 vs expm 精确解）", "",
            "| p | 格式 | 级数 | 设计阶 | 实测时间阶 | 拟合点数 | 判定 |", "|--:|---|--:|---:|---:|---:|:--:|"]
    for p in probes:
        design = design_order(p)
        ok = (p["fitted_time_order"] is not None and
              abs(p["fitted_time_order"] - design) <= tol_dt)
        shown = "n/a" if p["fitted_time_order"] is None else f"{p['fitted_time_order']:.3f}"
        out.append(f"| P{p['degree']} | {p['integrator']} | {p.get('design_stages', 'n/a')} | "
                   f"{design} | {shown} | {p.get('fit_points', 'n/a')} | {mk(ok)} |")

    out += ["", "### 3.5 时间误差占比：RK4 基准 vs 同阶格式", "",
            "| p | RK4 $E_{time}$ 占比 | 同阶格式 | 同阶 $E_{time}$ 占比 |", "|--:|---:|---|--:|"]
    for s in stage:
        out.append(f"| P{s['degree']} | {100*s['rk4_time_share']:.2f}% | "
                   f"{s['matched_integrator']} | {100*s['matched_time_share']:.2f}% |")

    out += ["", "## 4. 结果图", "",
            "- `docs/figures/advorder_matched_rk.png`：左=同阶级数阶梯 vs ref 同阶列（log-log），"
            "右=幅值比，RK4 基准（虚线）与同阶格式（实线）对比，绿带为比较窗口。",
            "- `docs/figures/advorder_error_split.png`：$E_{space}$–$E_{time}$ 逐点散点，虚线为 $E_{time}="
            f"{m['thresholds']['TIME_SHARE_TOL']:.2f}E_{{space}}$。", "",
            "## 5. 注意点（踩坑/局限）", "",
            "- **坑 1（表写错）**：Kutta-RK3 表初稿误写为 `a₃=(-1,1), b=(1/6,2/3,1/6)`，标量探针实测 1 阶；"
            "更正为 `a₃=(-1,2), b=(1/6,4/6,1/6)` 后标量/半离散双路均达 3 阶。若无独立阶检验，此错会静默污染整轮复测。",
            "- **坑 2（判据设计）**：全局误差斜率在细端受 float64 舍入污染（实测 12.5 阶这种假信号），"
            "故改用「一步局部误差」+ 拟合窗口下限 $10^{-14}$；这是第 1 轮 R6 型假信号同类问题的根治办法。",
            "- **坑 3（报告漂移）**：本报告的 M2/M5 结论曾以手写常量形式留在渲染器里，"
            "在实现修好之后仍把 PASS 印成 FAIL / NOT RUN，且渲染器本身在 Python 3.10/3.11 下是语法错误"
            "（f-string 内复用同引号）。现所有判据文字由 `metrics.json` 的 `checks` 生成，"
            "并由 `tests/test_advorder_round2_evidence.py` 断言「文档 ↔ 证据」一致。",
            m2,
            m5,
            f"- **局限 1（L2 幅值）**：ref 未公布 Δt/CFL 与通量形式，L2 幅值无法判定；"
            f"本轮只能排除「级数差」这一解释（比值 {ratio_means}，逐网格近乎常数）。",
            f"- **局限 2（稠密 expm 的规模）**：精确拆分用稠密 `expm`，仅适用小网格"
            f"（本轮最大 $N={expm_nmax}$，矩阵维 {expm_dim}）；大网格结论以"
            f"「$E_{{time}}$ 随 $\\Delta t^{{p+1}}$ 迅速衰减」外推，未逐网格做稠密矩阵。",
            f"- **局限 3（网格阶梯）**：各阶阶梯为 {ladder_txt}"
            f"（{'全阶同阶梯' if shared_ladder else '并非全阶同阶梯，逐阶记录在碎片里'}）。",
            "- **局限 4（双路径）**：`solve_advection` 的 `rk4`/`ssp_rk3` 分支为保持逐位兼容未并入通用驱动器，"
            "两条路径的等价性由阶测试与回归测试间接覆盖，未做逐位等价断言。", "",
            "## 6. 本轮修订（重开轮）", "",
            "1. **渲染器可用性**：本脚本曾含 f-string 内复用同引号的写法，在 "
            "`requires-python >= 3.10` 下是 **SyntaxError**（Python 3.11 实测无法 import），"
            "即第 2 轮之后它根本跑不起来，文档只能靠手改。",
            "2. **结论漂移**：M2/M5 的结论与 $p=4$ 的设计阶曾是手写常量，实现修好后仍把 PASS 印成 "
            "FAIL / NOT RUN。现由 `metrics.json` 的 `checks` 与 `dt_order_probes[*].design_order` 生成。",
            "3. **证据补齐**：P4 阶梯由 20–160 扩到 ref 全阶梯 20–320，`--assemble` 重新汇总后，"
            "`metrics.json` 是四阶齐全的规范包（此前存档只有单阶、判 FAIL）。",
            "4. **回归护栏**：`tests/test_advorder_round2_evidence.py` 断言"
            "「文档 == 渲染器对当前 metrics 的输出」，并把设计阶取自探针"
            "（`degree + 1` 的错误假设会把 $p=4$ 的正确答案判成 ❌）。", ""]
    return "\n".join(out)


def main() -> dict:
    metrics = json.loads(METRICS.read_text())
    md = render(metrics)
    DOC.write_text(md, encoding="utf-8")

    token = post("auth/login", {"username": USER, "password": PASSWORD})["access_token"]
    figs = []
    for name, caption in (
        ("advorder_matched_rk.png",
         "左：同阶级数（p+1 级）阶梯 vs 文献同阶列，斜率平行；右：L2 幅值比 RK4 基准 vs 同阶格式"),
        ("advorder_error_split.png",
         "cfl=0.2 下精确误差拆分：E_time 与 E_space 的逐点关系（虚线 E_time=0.1E_space）"),
    ):
        p = FIG / name
        if p.exists():
            figs.append({"name": name, "caption": caption, "png_base64": b64(p)})
    payload = {
        "title": "CASE-DG-ADVECT-ORDER 第2轮：同阶时间格式复测 + expm 精确时间/空间误差拆分",
        "summary": (f"TensorDG 以 p+1 级 RK 族（Euler/RK2/Kutta-RK3/RK4/SSP-RK(5,4)/Butcher-RK5）"
                    f"按同阶级数复测 P1–P4 收敛率：与文献列偏差最大 "
                    f"{max(abs(r['order_gap_vs_ref']) for r in metrics['comparison']):.3f} 阶；"
                    f"并用 expm 精确半离散解把误差拆成时间/空间两部分，"
                    f"cfl=0.2 下时间误差占比最大 "
                    f"{100*max(s['time_share'] for s in metrics['error_split']):.1f}%。"),
        "scenario_key": "CASE-DG-ADVECT-ORDER",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "收敛率复测", "Runge-Kutta", "误差拆分", "验证欠账"],
        "markdown": md,
        "figures": figs,
    }
    res = post("tensor/reports", payload, token)
    print("report id:", res.get("id"), "figures:", len(figs))
    return res


if __name__ == "__main__":
    main()