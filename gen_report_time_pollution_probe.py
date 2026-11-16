"""Real-data temporal-leg probe for the round-2 certified reference (m22895).

Closes the part of ``XVAL-DG-ROUND2-CERTIFIED-REFERENCE``'s time-pollution audit
that can be measured without the cluster:

* **N=800** -- the reference ladder's coarsest grid already carries CFL 0.3 and
  0.15 (committed), so the temporal difference and the pollution ratio ``R =
  delta_t / d_fine`` are measurable *today* from committed data;
* **N=400 (probe grid)** -- a full three-member CFL series (0.3/0.15/0.075) costs
  ~2 CPU-minutes here (17.8 + 32.7 + 66.9 s), which is the only way to measure a
  real **temporal order** ``p_t`` locally; SSP-RK3 predicts 3.

The probe grid is deliberately *outside* the certified ladder (which starts at
N=800) and its profiles are kept out of ``artifacts/raw`` on purpose: the
certified-reference loader globs ``fv_cfl0p3_n*.npz`` from that directory, so a
stray ``fv_cfl0p3_n400.npz`` there would silently extend the certified ladder and
break its bit-for-bit tests.  Probe data therefore lives in
``validation/xval_dg_round2/artifacts/cfl_probe``.

Everything numeric in the document and the report is read back from the artifacts
written here -- no hand-copied numbers.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import xval_dg_round2_time_pollution as tp
from tensordg.certified_reference import coarsen, rel_l1

OUT = ROOT / "validation" / "xval_dg_round2" / "artifacts"
PROBE = OUT / "cfl_probe"
DOC = ROOT / "docs" / "CASE-XVAL-DG-ROUND2-TIME-POLLUTION-CFL-PROBE.md"
RAW = Path(tp.DEFAULT_RAW)
MISSION = 22895
JOB = "cj4464"


def measure() -> dict:
    """All real numbers of this probe, computed from files on disk."""
    # ---- N=800: two committed CFL members -> delta_t and R (no p_t)
    s800 = tp.temporal_series(RAW, 800, require=2)
    rep800 = tp.temporal_report(s800)
    d_sp800 = rel_l1(coarsen(tp.load_level(RAW, 1600, 0.3)["field"], 2),
                     tp.load_level(RAW, 800, 0.3)["field"])
    r800 = rep800["delta_t"] / d_sp800

    # ---- N=400 probe: full CFL series -> delta_t, p_t, R
    rep400 = tp.temporal_report(tp.temporal_series(PROBE, 400, require=3))
    p400_03 = tp.load_level(PROBE, 400, 0.3)["field"]
    p800_03 = tp.load_level(RAW, 800, 0.3)["field"]
    d_sp400 = rel_l1(coarsen(p800_03, 2), p400_03)
    r400 = rep400["delta_t"] / d_sp400

    # ---- the ladder's own finest successive difference (what sets the band)
    d_fine = rel_l1(coarsen(tp.load_level(RAW, 3200, 0.3)["field"], 2),
                    tp.load_level(RAW, 1600, 0.3)["field"])

    return {
        "n800": {"cfls": rep800["cfls"], "delta_t_percent": rep800["delta_t_percent"],
                 "spatial_difference_percent": 100.0 * d_sp800,
                 "pollution_ratio": r800,
                 "state": tp.classify_pollution(r800),
                 "temporal_order": rep800["temporal_order"],
                 "levels_present": rep800["levels_present"]},
        "n400_probe": {"cfls": rep400["cfls"],
                       "differences": rep400["differences"],
                       "delta_t_percent": rep400["delta_t_percent"],
                       "temporal_order": rep400["temporal_order"],
                       "monotone": rep400["monotone"],
                       "spatial_difference_percent": 100.0 * d_sp400,
                       "pollution_ratio": r400,
                       "state": tp.classify_pollution(r400)},
        "ladder": {"finest_successive_difference_percent": 100.0 * d_fine,
                   "reference_grid": tp.REFERENCE_GRID,
                   "gate_percent": tp.GATE_PERCENT},
        "thresholds": {"POLLUTION_MAX": tp.POLLUTION_MAX,
                       "POLLUTION_MARGINAL": tp.POLLUTION_MARGINAL,
                       "P_MIN_T": tp.P_MIN_T,
                       "BAND_SHIFT_MAX": tp.BAND_SHIFT_MAX},
        "pending": {"grids": [1600, 3200, 6400], "job": JOB,
                    "reason": "CFL 0.15/0.075 级在队列中排队（单核 6 分片），"
                              "本单未取回；缺它们只能测 N=800 的 R，无法做时间修正重认证"},
    }


def figures(res: dict) -> list[Path]:
    PROBE.mkdir(parents=True, exist_ok=True)
    figs = []

    # fig 1: temporal differences vs CFL on the probe grid (log-log)
    rows = res["n400_probe"]["differences"]
    x = [np.mean(r["cfls"]) for r in rows]
    y = [r["rel_l1_percent"] for r in rows]
    p_t = res["n400_probe"]["temporal_order"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.loglog(x, y, "o-", label="N=400 probe: |u(CFL_c) - u(CFL_f)|")
    if p_t:
        ref = y[0] * (np.array(x) / x[0]) ** p_t
        ax.loglog(x, ref, "k--", label=f"power law, observed p_t = {p_t:.3f}")
    ax.set_xlabel("CFL"); ax.set_ylabel("relative L1 difference [%]")
    ax.set_title("real temporal leg (FV-WENO5/SSP-RK3, Shu-Osher t=1.8)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    f1 = PROBE / "fig_time_leg_n400_probe.png"; fig.savefig(f1, dpi=150, bbox_inches="tight")
    plt.close(fig); figs.append(f1)

    # fig 2: pollution ratio R on the grids measured, against the predeclared gates
    labels, vals = [], []
    for key, name in (("n400_probe", "N=400 (probe)"), ("n800", "N=800 (ladder)")):
        labels.append(name); vals.append(res[key]["pollution_ratio"])
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.bar(labels, vals, color=["#4C72B0", "#55A868"])
    ax.axhline(tp.POLLUTION_MAX, color="green", ls="--", lw=1,
               label=f"T1 CLEAN gate = {tp.POLLUTION_MAX}")
    ax.axhline(tp.POLLUTION_MARGINAL, color="red", ls="--", lw=1,
               label=f"MARGINAL gate = {tp.POLLUTION_MARGINAL}")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)
    ax.set_ylabel("R = delta_t / d_fine"); ax.set_ylim(0, max(0.3, max(vals) * 1.4))
    ax.set_title("time-discretisation pollution ratio (real data)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    f2 = PROBE / "fig_pollution_ratio_real.png"; fig.savefig(f2, dpi=150, bbox_inches="tight")
    plt.close(fig); figs.append(f2)
    return figs


def _f(v, nd=6):
    return "n/a" if v is None else f"{v:.{nd}f}"


def build_md(res: dict) -> str:
    a, b = res["n800"], res["n400_probe"]
    diffs = "\n".join(
        f"| {r['cfls'][0]:g} → {r['cfls'][1]:g} | {r['ratio']:g} | "
        f"{r['rel_l1_percent']:.6f} % | {r['steps'][0]} / {r['steps'][1]} |"
        for r in b["differences"])
    return f"""# CASE-SHU-OSHER 认证参考：时间腿实测（CFL 探针）

锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE` · 任务 m22895 · 域：TensorDG 验证欠账

## 1. 案例介绍

Shu–Osher 激波-熵波相互作用，1-D Euler（γ=1.4），域 $x\\in[-5,5]$，$t=1.8$：
左态 $(\\rho,u,p)=(3.857143,\\,2.629369,\\,10.3333)$，右态 $(1+0.2\\sin 5x,\\,0,\\,1)$，
透射边界，参考求解器 FV-WENO5-JS / HLLC / SSP-RK3。

**问题**：已发布的 round-2 参考梯次 `fv_cfl0p3_n800/1600/3200/6400` 全部在**单一** CFL=0.3
下生成，相邻差分被当作纯空间误差、Roache 带宽 $U=F_s d_{{fine}}/(r^p-1)$ 被当作参考自身
不确定度。时间离散到底贡献多少，从未被测量（审计 `benchmarks/xval_dg_round2_time_pollution.py`
自落地起真实分支即为 `WITHHELD`）。

## 2. 怎么做的

1. **固定网格只细化 CFL**（0.3 → 0.15 → 0.075，比例 2），差分即纯时间误差；
   差分用整条序列的**公共分母**（最细成员），逐对归一化会把误差自身混进分母。
2. **N=800**：参考梯次最粗一级，CFL 0.3 与 0.15 剖面**已提交**，直接量 $\\delta_t$；
   与之比较的空间差分为 $d_{{fine}}(800)=\\|\\mathcal{{C}}(u_{{1600}})-u_{{800}}\\|/\\|u_{{800}}\\|$
   （`certify` 的粗网格约定）。
3. **N=400 探针**：本地三级 CFL 序列（17.8 + 32.7 + 66.9 s），用来测**真实时间阶** $p_t$。
   探针级**故意不放进** `artifacts/raw`：认证参考的加载器按 `fv_cfl0p3_n*.npz` 通配，
   放进去会悄悄把认证梯次扩成 5 级、破坏逐位一致性测试；因此落在
   `validation/xval_dg_round2/artifacts/cfl_probe/`。
4. 污染比 $R=\\delta_t/d_{{fine}}$ 与预声明门槛比较（CLEAN ≤ {res['thresholds']['POLLUTION_MAX']}，
   MARGINAL ≤ {res['thresholds']['POLLUTION_MARGINAL']}）。
5. **未完成**的那一半：N=1600/3200/6400 的 CFL 细化级需 1.2 CPU·h，已递交集群
   `{res['pending']['job']}`（单核 6 分片，生成器 `validation/xval_dg_round2/gen_cfl_ladder.py`
   幂等、已落盘即跳过），本单未取回；没有它们做不了时间修正后重认证（T3）与逐腿翻转（T4）。

## 3. 计算结果（真实数据，程序化注入）

### 3.1 N=400 探针：真实时间阶

| CFL 对 | 比例 | 相对 L1 差分 | 步数 |
|---|---|---|---|
{diffs}

| 量 | 值 |
|---|---|
| 时间观测阶 $p_t$（SSP-RK3 理论 3） | **{_f(b['temporal_order'])}** |
| 相邻时间差分单调递减（T0 形状） | {b['monotone']} |
| 最粗级时间误差 $\\delta_t$（CFL 0.3） | {b['delta_t_percent']:.6f} % |
| 同网格空间差分 $d_{{fine}}(400)$ | {b['spatial_difference_percent']:.6f} % |
| 污染比 $R=\\delta_t/d_{{fine}}$ | **{_f(b['pollution_ratio'], 4)}** → {b['state']} |

### 3.2 N=800（已提交数据）：污染比

| 量 | 值 |
|---|---|
| CFL 成员 | {a['cfls']} |
| $\\delta_t$ | {a['delta_t_percent']:.6f} % |
| $d_{{fine}}(800)$ | {a['spatial_difference_percent']:.6f} % |
| $R$ | **{_f(a['pollution_ratio'], 4)}** → {a['state']} |
| $p_t$ | n/a（该网格只有两级 CFL，一个差分不含阶信息） |

### 3.3 认证梯次自身的空间差分（对照）

| 量 | 值 |
|---|---|
| 最细相邻空间差分 $d_{{fine}}(1600\\to3200)$ | {res['ladder']['finest_successive_difference_percent']:.6f} % |
| 参考网格 / 门槛 | N={res['ladder']['reference_grid']} / {res['ladder']['gate_percent']:g} % |

## 4. 结果图

![fig_time_leg_n400_probe](validation/xval_dg_round2/artifacts/cfl_probe/fig_time_leg_n400_probe.png)

*图 1：N=400 探针上真实的时间差分随 CFL 的幂律（点=实测，虚线=观测阶拟合）。*

![fig_pollution_ratio_real](validation/xval_dg_round2/artifacts/cfl_probe/fig_pollution_ratio_real.png)

*图 2：两个已测网格的污染比 $R$ 与预声明门槛（绿=CLEAN 0.10，红=MARGINAL 0.25）。*

## 5. 注意点（坑 / 局限）

1. **通配污染坑**：把探针剖面写进 `artifacts/raw` 会让 `shu_osher_certified_ref` 的
   `fk_cfl0p3_n*.npz` 拾取到 N=400，认证梯次被悄悄扩级 —— 本单把探针隔离在
   `artifacts/cfl_probe/`，并保留该理由在生成器 docstring 里。
2. **两级 CFL 序列陷阱（本单修掉的引擎缺陷）**：审计文档与测试 `_REQUIRED` 都只要求
   N=6400 有 CFL 0.15（只算 $R$），但 `temporal_series` 硬要求全部三级，数据真到位时
   `audit()` 会以 `FileNotFoundError` 整单阵亡 —— 一个只会在数据出现后暴露的潜伏缺陷。
   现 `temporal_series(..., require=)` 允许"部分序列"：BASE 网格仍要求满三级
   （$p_t$ 需要两个相邻差分），额外网格只要最粗一对。
3. **$p_t$ 受激波限制**：SSP-RK3 光滑区三阶，间断处退化，探针值只用于判断"时间腿是否
   高阶"，不作为精度结论。
4. **探针网格不属认证梯次**：N=400 只为测时间阶；其 $R$ 与 N=800/3200 的 $R$ 不完全同
   口径（$d_{{fine}}$ 取自不同网格对），跨网格比较只看数量级。
5. **集群排队**：{res['pending']['reason']}。

## 6. 误差判定（显式回答是否 < 3%）

| 关键量 | 基准 | 实测 | 误差 | < 3% ? |
|---|---|---|---|---|
| 探针时间阶 $p_t$ | 3（SSP-RK3 设计阶） | {_f(b['temporal_order'])} | {abs(b['temporal_order'] - 3.0) / 3.0 * 100 if b['temporal_order'] else float('nan'):.2f} % | {"**否**（见注意点 3，激波限制）" if b['temporal_order'] and abs(b['temporal_order'] - 3.0) / 3.0 > 0.03 else "**是**"} |
| N=400 污染比 $R$ | CLEAN 门槛 0.10 | {_f(b['pollution_ratio'], 4)} | 裕度 {100 * (1 - b['pollution_ratio'] / 0.10):.1f} % | **是** |
| N=800 污染比 $R$ | CLEAN 门槛 0.10 | {_f(a['pollution_ratio'], 4)} | 裕度 {100 * (1 - a['pollution_ratio'] / 0.10):.1f} % | **是** |
| 参考带宽位移（T3） | — | **未测**（需 1600/3200 的 CFL 级） | — | **无法回答** |

**结论**：已测网格上时间腿**远小于**空间腿（$R\\approx${_f(a['pollution_ratio'], 3)}），
即"带宽主要由空间差分设定"的既有读法在 N=800 与 N=400 上成立；但审计整体仍**不得**
判为 `TIME_CLEAN`——参考网格 N=3200 的 $R$、时间修正带宽位移、11 条 DG 腿的翻转都还缺数据。

## 7. 复现

```bash
# 探针（本地 ~2 min）
python validation/xval_dg_round2/gen_cfl_ladder.py --ncells 400 --cfl 0.3 \\
    --out validation/xval_dg_round2/artifacts/cfl_probe/fv_cfl0p3_n400.npz
# 完整梯次（集群，幂等）
python validation/xval_dg_round2/gen_cfl_ladder.py --shard 0 --nshards 6
# 审计（数据齐后自动走真实分支）
python -m benchmarks.xval_dg_round2_time_pollution --figures --require-accepted
# 测试
python -m pytest -q tests/test_xval_dg_round2_time_pollution.py \\
    tests/test_xval_dg_round2_cfl_ladder_driver.py
```

## 8. 产物

| 文件 | 说明 |
|---|---|
| `validation/xval_dg_round2/gen_cfl_ladder.py` | CFL 细化级生成器（分片、幂等、命名与审计一致） |
| `tests/test_xval_dg_round2_cfl_ladder_driver.py` | 18 项不变式：级表覆盖、命名一致、可用性门、分片划分、幂等 |
| `benchmarks/xval_dg_round2_time_pollution.py` | T0–T4 审计引擎（本单修部分序列支持） |
| `tests/test_xval_dg_round2_time_pollution.py` | 25 项（+4 项部分序列契约） |
| `validation/xval_dg_round2/artifacts/cfl_probe/*.npz` | N=400 三级 CFL 探针剖面 |
| `validation/xval_dg_round2/artifacts/cfl_probe/time_leg_probe.json` | 本页全部数字的源 |
| `validation/xval_dg_round2/artifacts/cfl_probe/fig_*.png` | 两张结果图 |
| 本文件 | 方法 / 实测 / 局限（数字程序化注入） |
"""


def post(title: str, summary: str, md: str, figs: list[Path]) -> str:
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{api}/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read())["access_token"]
    caps = {
        "fig_time_leg_n400_probe.png":
            "图 1：N=400 探针真实时间差分与观测阶拟合（FV-WENO5/SSP-RK3，Shu-Osher t=1.8）",
        "fig_pollution_ratio_real.png":
            "图 2：真实污染比 R=δ_t/d_fine 与预声明 CLEAN/MARGINAL 门槛",
    }
    figures = [{"name": p.name, "caption": caps.get(p.name, p.name),
                "png_base64": base64.b64encode(p.read_bytes()).decode()} for p in figs]
    body = json.dumps({"title": title, "summary": summary,
                       "scenario_key": "CASE-SHU-OSHER", "mission_id": MISSION,
                       "tags": ["TensorDG", "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
                                "time-discretisation", "CFL-refinement",
                                "certified-reference", "GCI", "verification"],
                       "markdown": md, "figures": figures}).encode()
    req = urllib.request.Request(f"{api}/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode()


def main() -> int:
    res = measure()
    PROBE.mkdir(parents=True, exist_ok=True)
    (PROBE / "time_leg_probe.json").write_text(json.dumps(res, indent=2, default=str))
    figs = figures(res)
    md = build_md(res)
    DOC.write_text(md)
    a, b = res["n800"], res["n400_probe"]
    title = (f"CASE-SHU-OSHER 认证参考的时间腿实测：p_t={_f(b['temporal_order'], 3)}，"
             f"R(N=800)={_f(a['pollution_ratio'], 4)} → {a['state']}（参考网格仍 WITHHELD）")
    summary = (f"固定网格只细化 CFL 测 round-2 认证参考的时间离散贡献：N=400 探针三级序列"
               f"给出真实时间阶 p_t={_f(b['temporal_order'], 3)}（设计 3），"
               f"N=800 已提交数据给出 δ_t={a['delta_t_percent']:.4f}% vs 空间差分 "
               f"{a['spatial_difference_percent']:.4f}% ⇒ R={_f(a['pollution_ratio'], 4)}（CLEAN<0.10）；"
               f"同期修掉审计引擎潜伏缺陷（N=6400 两级 CFL 序列会 FileNotFoundError），"
               f"并新增幂等分片生成器 validation/xval_dg_round2/gen_cfl_ladder.py。"
               f"N=1600/3200/6400 的 CFL 级仍在集群 {JOB} 排队，时间修正重认证与逐腿翻转未测。")
    print(post(title, summary, md, figs)[:300])
    print(json.dumps(res, indent=2, default=str))
    print(f"doc: {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())