#!/usr/bin/env python3
"""Publish the time-discretisation pollution audit of the round-2 certified reference.

Recomputes the audit from committed raw profiles, regenerates the figures, runs
the test file that guards it, builds a paper-grade report JSON (every number
injected from the audit dict -- nothing hand-typed) and POSTs it to
``$TENSOR_API_BASE/tensor/reports``.  Also writes the same body to
``docs/CASE-XVAL-DG-ROUND2-TIME-POLLUTION.md``.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_time_pollution as tp  # noqa: E402

OUT = Path(tp.DEFAULT_OUT)
DOC = ROOT / "docs" / "CASE-XVAL-DG-ROUND2-TIME-POLLUTION.md"

FIG_CAPTIONS = {
    "fig_time_vs_space_leg.png":
        "各网格上的时间步误差 δ_t(CFL 0.3→0.15) 与决定带宽的空间差分 d_fine 对比（标注 R=δ_t/d_fine）",
    "fig_cfl_profile_difference.png":
        "N=3200 上已发布时间步（CFL=0.3）与 4 倍小步长（CFL=0.075）的密度剖面及其差值",
    "fig_leg_stability.png":
        "11 条 DG 腿分别对「CFL=0.3 参考」与「CFL→0 时间修正参考」的 L1 误差与 3%/5% 门槛",
}


def run_tests() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_xval_dg_round2_time_pollution.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    lines = (r.stdout + r.stderr).strip().splitlines()
    summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
    return "\n".join(summary[-2:]) if summary else f"rc={r.returncode}"


def _f(x, nd=6, dash="—"):
    if x is None:
        return dash
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _grid_table(res: dict) -> str:
    rows = ["| N | δ_t % = rel L1(0.3, 0.15) | rel L1(0.15, 0.075) % | 时间阶 p_t | "
            "d_fine（空间）% | R = δ_t/d_fine | 判定 |",
            "|---|---|---|---|---|---|---|"]
    for g in res["grids"]:
        d = g["differences"]
        second = _f(d[1]["rel_l1_percent"]) if len(d) > 1 else "—（无 0.075 级）"
        state = tp.classify_pollution(g.get("pollution_ratio"))
        rows.append(
            f"| {g['ncells']} | {_f(g['delta_t_percent'])} | {second} | "
            f"{_f(g['temporal_order'], 4)} | "
            f"{_f(100.0 * g['spatial_difference'] if g.get('spatial_difference') else None)} | "
            f"{_f(g.get('pollution_ratio'), 4)} | {state} |")
    return "\n".join(rows)


def _criteria_table(res: dict) -> str:
    c = res["criteria"]
    h = res["headline"]
    return "\n".join([
        "| 判据 | 陈述 | 实测 | 通过 |",
        "|---|---|---|---|",
        f"| T0 | 相邻时间差分严格递减（各网格） | "
        f"{json.dumps(c['T0']['monotone'], ensure_ascii=False)} | {'✅' if c['T0']['pass'] else '❌'} |",
        f"| T1 | δ_t/d_fine ≤ {tp.POLLUTION_MAX}（N={res['reference_grid']}） | "
        f"R = {_f(h['pollution_ratio'], 4)}（{h['state']}） | {'✅' if c['T1']['pass'] else '❌'} |",
        f"| T2 | 时间阶 p_t ≥ {tp.P_MIN_T} | {_f(c['T2']['temporal_order'], 4)} | "
        f"{'✅' if c['T2']['pass'] else '❌'} |",
        f"| T3 | 带宽变化 ≤ {tp.BAND_SHIFT_MAX} 且时间修正梯次仍 CERTIFIED | "
        f"{_f(c['T3']['band_shift_relative'], 4)} | {'✅' if c['T3']['pass'] else '❌'} |",
        f"| T4 | 3% 门槛无判定翻转、逐腿位移 ≤ {tp.LEG_STABILITY_MAX_PP} pp | "
        f"{c['T4']['n_flips']} 次翻转，最大 {_f(c['T4']['max_abs_move_pp'], 4)} pp | "
        f"{'✅' if c['T4']['pass'] else '❌'} |",
    ])


def _ladder_table(res: dict) -> str:
    c = res["criteria"]["T3"]
    rows = ["| 梯次 | 判定 | 观测阶 p | 带宽 U % | 参考网格 |", "|---|---|---|---|---|"]
    for label, key in (("原始三级（round 2 已发布）", "raw"),
                       ("原始四级（当前随仓发布）", "raw_four_level"),
                       ("时间修正三级（CFL→0）", "time_corrected")):
        b = c.get(key)
        if not b:
            rows.append(f"| {label} | 拒绝 | — | — | — |")
            continue
        rows.append(f"| {label} | {b['verdict']} | {_f(b['observed_order_used'], 6)} | "
                    f"{_f(b['band_percent'], 6)} | {b['reference_ncells']} |")
    return "\n".join(rows)


def _legs_table(res: dict) -> str:
    t4 = res["criteria"]["T4"]
    corr = t4["legs_time_corrected"] or [None] * len(t4["legs_raw"])
    rows = ["| DG 腿 | 参考=CFL0.3：L1 % | 参考=CFL→0：L1 % | 判定 3%（原始/修正） | 位移 pp |",
            "|---|---|---|---|---|"]
    for a, b in zip(t4["legs_raw"], corr):
        if b is None:
            rows.append(f"| {a['configuration']} | {_f(a['rel_l1_percent'])} | — | "
                        f"{a['decision_3pct']} / — | — |")
            continue
        rows.append(f"| {a['configuration']} | {_f(a['rel_l1_percent'])} | "
                    f"{_f(b['rel_l1_percent'])} | "
                    f"{a['decision_3pct']} / {b['decision_3pct']} | "
                    f"{_f(abs(b['rel_l1_percent'] - a['rel_l1_percent']), 4)} |")
    return "\n".join(rows)


def build_md(res: dict, tests: str) -> str:
    h = res["headline"]
    c = res["criteria"]
    ref = [g for g in res["grids"] if g["ncells"] == res["reference_grid"]][0]
    band_raw = c["T3"]["raw"]["band_percent"]
    band_corr = h["time_corrected_band_percent"]
    verdict_line = {
        "TIME_CLEAN": "**结论：时间步未污染认证参考**（R ≤ 0.10，时间修正后带宽与判定不变）",
        "TIME_MARGINAL": "**结论：时间步污染处于边界（0.10 < R ≤ 0.25），需在更小 CFL 重认证**",
        "TIME_POLLUTED": "**结论：认证参考被时间离散污染（R > 0.25 或时间修正改变判定），已发布带宽不可信**",
    }.get(res["verdict"], res["verdict"])

    return f"""# CASE-SHU-OSHER round-2 认证参考：时间离散污染审计

锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`
（TensorDG 的 round2-certified-reference 做法；验收：对标误差 <15%，本单按 owner 精度标准 3%）

{verdict_line}

## 1. 案例介绍（物理背景 / 几何 / 参数）

Shu–Osher 激波-熵波相互作用问题（Shu & Osher 1989）：1-D Euler 方程，
域 $x\\in[-5,5]$、终时 $t=1.8$、$\\gamma=1.4$。

| 量 | 左区 $x<-4$ | 右区 $x\\ge -4$ |
|---|---|---|
| $\\rho$ | 3.857143 | $1+0.2\\sin(5x)$ |
| $u$ | 2.629369 | 0 |
| $p$ | 10.3333 | 1 |

该问题无解析解，因此整条 round-2 认证链条（`fv_cfl0p3_n800/1600/3200/6400.npz`，
FV-WENO5/Jiang-Shu + HLLC + SSP-RK3）的全部梯次剖面都在**单一时间步 CFL=0.3** 下生成，
相邻差分 $d(N\\to2N)$ 被当作纯**空间**误差、Roache 带宽 $U=F_s d_{{fine}}/(r^p-1)$
被当作参考解自身不确定度发布。本单审计的正是这个隐含假设：时间离散是否污染了带宽。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **只细化时间**：固定网格 $N\\in\\{{800,1600,3200\\}}$，把 CFL 从已发布的 0.3
   细化到 0.15、0.075（比例 2），另在 $N=6400$ 补 0.15。任何两成员之差纯粹来自
   时间离散（同一网格、同一空间算子、同一初值）。
2. 计算**时间差分** $\\delta_t(N)=\\mathrm{{rel\\ L1}}(u_{{0.3}},u_{{0.15}})$（同一公共分母，
   取最细成员 L1 范数，避免归一化引入的阶偏差——见 §5 第 1 条坑）与
   **时间观测阶** $p_t=\\log_2[\\delta_t(0.3\\to0.15)/\\delta_t(0.15\\to0.075)]$。
3. 计算**空间差分** $d_{{fine}}$：按 `certify()` 的约定，在粗网格上比较
   $\\mathrm{{coarsen}}(u_N)$ 与 $u_{{N/2}}$。
4. **污染比** $R=\\delta_t(N)/d_{{fine}}(N)$；$F_s=1.25$、$r^p-1\\approx1.25$ 时带宽与
   $d_{{fine}}$ 同量级，故 $R$ 可直接读作「时间误差占带宽的比例」。
5. **时间修正重认证**：对每个网格用 $p_t$ 做时间 Richardson
   $u_T=u_{{0.075}}+(u_{{0.075}}-u_{{0.15}})/(2^{{p_t}}-1)$，把三个时间收敛后的剖面
   重新喂给同一个 `certify()`（同 $F_s$、同门槛），比较带宽/阶与原始发布值的位移。
6. **结论相关性检验**：把 11 条已发布 DG 腿分别对「原始参考」与「时间修正参考」
   在 3% 门槛下重新判定，统计翻转数（这是「污染是否 material」的唯一判据）。
7. 门槛（模块常量、先于数据固定）：T0 单调；T1 $R\\le0.10$；T2 $p_t\\ge0.5$；
   T3 带宽位移 $\\le25\\%$ 且修正梯次仍 CERTIFIED；T4 无翻转且逐腿位移 $\\le0.5$ pp。

工具链：`benchmarks/xval_dg_round2_time_pollution.py`（审计引擎，CLI 可复算）、
`validation/xval_dg_round2/gen_fv_level.py`（原样调用已发布数据生成路径，仅改 `--cfl`）、
集群 `run_compute` 递交（约 {sum(1 for _ in res['grids'])} 个网格 × 2 个 CFL 细化）。

## 3. 计算结果

### 3.1 逐网格时间腿 vs 空间腿

{_grid_table(res)}

### 3.2 判据

{_criteria_table(res)}

### 3.3 参考解：原始 vs 时间修正

{_ladder_table(res)}

### 3.4 已发布 DG 腿在两种参考下的判定稳定性

{_legs_table(res)}

## 4. 结果图

`fig_time_vs_space_leg.png`（时间腿 vs 空间腿，含 R 标注）、
`fig_cfl_profile_difference.png`（N=3200 剖面与 CFL 差）、
`fig_leg_stability.png`（逐腿误差双参考对比 + 3%/5% 门槛）。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **归一分母**：最初每个时间差分各自用「较细成员」的 L1 范数归一化，合成幂律
   上恢复的阶从 1.0 掉到 0.9978（误差自污染）。改为整条序列共用一个分母后，
   合成幂律恢复阶精确到 1e-14（`test_temporal_difference_is_pure_time_error`）。
   已发布数据上本单使用公共分母，因此 $\\delta_t$ 与 trail 里 `certify()`
   的差分不是同一归一化——两者比值 $R$ 只差一个 $O(1)$ 因子，判定门槛已按此设定。
2. **时间阶受激波限制**：SSP-RK3 在光滑区是三阶，但激波/接触间断处退化，
   故 $p_t$ 明显低于 3；$p_t$ 只用于时间修正，不当作精度结论。
3. **$N=6400$ 只有两级 CFL**：可算 $\\delta_t$ 与 $R$，但无第二对差分，因此不能给
   $p_t$，也不能做时间修正；四级梯次的时间修正（需 6400 的 0.075，约 2.5 CPU·h）
   未在本单范围内，如实列为待办。
4. **时间 Richardson 的合法性**：要求 $p_t$ 有限且 $\\ge$ 0.5（T2），否则修正被拒绝
   （`time_corrected_levels` 抛错、审计如实记 `refusal`），不把猜测当结果。
5. **适用范围**：本审计只回答「CFL=0.3 的时间步是否污染了 round-2 参考」，
   不涉及空间算子（WENO5/HLLC）本身的正确性，也不改变 DG 腿的误差定义。

## 6. 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 基准 | 测量值 | 相对误差 | < 3% ? |
|---|---|---|---|---|
| 污染比 $R$（N={res['reference_grid']}） | 0（理想：时间步无贡献） | {_f(h['pollution_ratio'], 4)} | — | {'**是**' if (h['pollution_ratio'] is not None and h['pollution_ratio'] < 0.03) else '**否**（R 非零本身即被量化，判据见 T1）'} |
| 带宽 U 原始（三级） | — | {_f(band_raw)} % | — | 见下行 |
| 带宽位移 \\|U_t−U\\|/U | 0（理想） | {_f(h['band_shift_relative'], 4)} | {_f(100.0 * h['band_shift_relative'] if h['band_shift_relative'] is not None else None, 3)} % | {'**是**' if (h['band_shift_relative'] is not None and h['band_shift_relative'] < 0.03) else ('**否**' if h['band_shift_relative'] is not None else '无基准')} |
| 时间修正后带宽 $U_t$ | — | {_f(band_corr)} % | — | 见下行 |
| DG 腿误差最大位移 | 0（理想） | {_f(c['T4']['max_abs_move_pp'], 4)} pp | — | {'**是**' if (c['T4']['max_abs_move_pp'] is not None and c['T4']['max_abs_move_pp'] < 0.03 * tp.STRICT_PERCENT) else '否/无基准'} |

DG 腿对认证参考的既有结论（< 3% 门槛逐腿判定）不受本单修改；本单只检验
「参考解的时间步是否让这些判定失去意义」。

## 7. 复现

```bash
# 生成 CFL 细化剖面（已在集群完成：cj4305-cj4308）
python validation/xval_dg_round2/gen_fv_level.py --ncells 3200 --cfl 0.15 \\
    --out validation/repro_shu_osher/artifacts/raw/fv_cfl0p15_n3200.npz
# 审计 + 图
python -m benchmarks.xval_dg_round2_time_pollution --figures --require-accepted
# 测试
python -m pytest -q tests/test_xval_dg_round2_time_pollution.py
```

测试结果：`{tests}`

## 8. 产物

| 文件 | 说明 |
|---|---|
| `benchmarks/xval_dg_round2_time_pollution.py` | T0–T4 审计引擎（预声明门槛、可 CLI 复算） |
| `tests/test_xval_dg_round2_time_pollution.py` | 22 项：合成幂律/拒绝路径/门槛钉扎/真实数据端到端 |
| `validation/xval_dg_round2/artifacts/time_pollution.json` | 完整审计（逐网格、逐判据、逐腿） |
| `validation/xval_dg_round2/artifacts/time_pollution.md` | 审计 Markdown（由审计字典渲染） |
| `validation/xval_dg_round2/artifacts/fig_*.png` | 三张结果图 |
| `validation/repro_shu_osher/artifacts/raw/fv_cfl0p1*/fv_cfl0p075*_n*.npz` | 新增 CFL 细化剖面（时间腿唯一证据） |
"""


def post(title: str, summary: str, md: str, fig_paths) -> str:
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{api}/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read())["access_token"]
    figures = []
    for p in fig_paths:
        p = Path(p)
        figures.append({"name": p.name,
                        "caption": FIG_CAPTIONS.get(p.name, p.name),
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    body = json.dumps({
        "title": title,
        "summary": summary,
        "scenario_key": "CASE-SHU-OSHER",
        "mission_id": 22686,
        "tags": ["TensorDG", "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
                 "time-discretisation", "certified-reference", "GCI", "verification"],
        "markdown": md,
        "figures": figures,
    }).encode()
    req = urllib.request.Request(f"{api}/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode()


def main() -> int:
    res = tp.audit()
    OUT.mkdir(parents=True, exist_ok=True)
    tp.write_artifacts(res, OUT)
    figs = tp.figures(res, tp.DEFAULT_RAW, OUT)
    tests = run_tests()
    md = build_md(res, tests)
    DOC.write_text(md)
    h = res["headline"]
    title = (f"CASE-SHU-OSHER 认证参考的时间离散污染审计：R = "
             f"{_f(h['pollution_ratio'], 4)}（{h['state']}），"
             f"时间修正带宽 {_f(h['raw_band_percent'])}% → "
             f"{_f(h['time_corrected_band_percent'])}%")
    ref = [g for g in res["grids"] if g["ncells"] == res["reference_grid"]][0]
    summary = (f"固定网格只细化 CFL（0.3/0.15/0.075）测量 round-2 认证参考的时间腿："
               f"N={res['reference_grid']} 上 δ_t={_f(ref['delta_t_percent'])}%，"
               f"R=δ_t/d_fine={_f(h['pollution_ratio'], 4)}（{h['state']}），"
               f"时间修正后带宽位移 {_f(100.0 * h['band_shift_relative'] if h['band_shift_relative'] is not None else None, 3)}%，"
               f"11 条 DG 腿 3% 判定翻转 {h['n_decision_flips']} 次 → "
               f"{res['verdict']}（accepted={res['accepted']}）。")
    resp = post(title, summary, md, figs)
    print(resp[:400])
    print(f"doc: {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())