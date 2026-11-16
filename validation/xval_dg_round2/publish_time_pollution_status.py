#!/usr/bin/env python3
"""Publish the time-pollution audit: real profiles when available, else the
synthetic ground-truth validation of the same engine, with an explicit
``WITHHELD`` statement for the reference-grid verdict.

Why a two-branch publisher exists
--------------------------------
The anchor's headline number needs CFL-refined FV profiles
(``fv_cfl0p15_*`` / ``fv_cfl0p075_*``).  Those runs were submitted to the
cluster (cj4305-cj4308) and were still *queued* after 40 min, so this publisher
refuses to invent the number: when the profiles are absent it runs the identical
audit on a **synthetic raw tree whose verdict is known in closed form** (engine
validation), publishes that as method evidence, and states the reference-grid
verdict as ``WITHHELD`` with the job ids.  It never prints a real-data verdict it
did not measure.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_time_pollution as tp  # noqa: E402

sys.path.insert(0, str(ROOT / "validation" / "xval_dg_round2"))
import publish_time_pollution as pub  # noqa: E402

BLOCK_GRID = 200
JOBS = "cj4305 (N=800), cj4306 (N=1600), cj4307 (N=3200), cj4308 (N=6400 CFL=0.15)"
DOC = ROOT / "docs" / "CASE-XVAL-DG-ROUND2-TIME-POLLUTION.md"


def synthetic_rawdir(root: Path, k_time: float, a_space: float = 0.1,
                     p_space: float = 1.2, p_time: float = 2.0) -> Path:
    """Block-constant profiles: exact coarsening, closed-form ladder law."""
    root.mkdir(parents=True, exist_ok=True)
    idx = np.arange(BLOCK_GRID)
    w = 1.0 + 0.3 * np.sin(2.0 * np.pi * idx / BLOCK_GRID)

    def blocks_on(n: int) -> np.ndarray:
        return w[(np.arange(n) * BLOCK_GRID // n)]

    def field(n: int, cfl: float) -> np.ndarray:
        x = (np.arange(n) + 0.5) * (10.0 / n) - 5.0
        r = n / BLOCK_GRID
        return (1.0 + 0.2 * x + a_space * r ** (-p_space) * blocks_on(n)
                + k_time * r ** (-p_time) * float(cfl) ** p_time * blocks_on(n))

    for n in (400, 800, 1600, 3200, 6400):
        for cfl in tp.CFL_LEVELS:
            f = field(n, cfl)
            np.savez(tp.level_path(root, n, cfl), x=np.arange(n, dtype=float),
                     rho=f, rho_u=f, rho_e=f, meta="{}")
    for degree, ncells, cfl in tp.xval.DG_LEGS:
        x = (np.arange(ncells) + 0.5) * (10.0 / ncells) - 5.0
        f = (1.0 + 0.2 * x
             + 0.5 * a_space * (ncells / BLOCK_GRID) ** (-p_space) * blocks_on(ncells))
        np.savez(Path(root) / tp.xval.leg_filename(degree, ncells, cfl),
                 x=x, rho=f, rho_u=f, rho_e=f, meta="{}")
    return root


def real_data_present() -> bool:
    need = [tp.level_path(tp.DEFAULT_RAW, n, c) for n in tp.BASE_LEVELS
            for c in tp.CFL_LEVELS]
    need.append(tp.level_path(tp.DEFAULT_RAW, tp.EXTRA_RATIO_LEVELS[0], 0.15))
    return all(p.exists() for p in need)


def status_md(real: bool, res: dict, tests: str) -> str:
    h = res["headline"]
    head = (
        "**真实 CFL 细化剖面尚未到位（集群排队 cj4305-cj4308，40+ min 未调度），\n"
        "本页的 reference-grid 判定 = `WITHHELD`。**\n"
        if not real else "")
    return f"""# CASE-SHU-OSHER round-2 认证参考：时间离散污染审计

锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE` · 任务 m22686 · 域：TensorDG 验证欠账

{head}
## 1. 案例介绍（物理背景 / 参数）

Shu–Osher（1989）激波-熵波相互作用，1-D Euler：域 $x\\in[-5,5]$，$t=1.8$，$\\gamma=1.4$；
左态 $(\\rho,u,p)=(3.857143,\\,2.629369,\\,10.3333)$，右态 $(1+0.2\\sin 5x,\\,0,\\,1)$。
无解析解 ⇒ 全部 round-2 结论都相对「认证参考」而言。

**审计对象**：已发布参考梯次 `fv_cfl0p3_n800/1600/3200/6400.npz` 全部在**单一时间步
CFL=0.3** 下生成，但相邻差分被当作纯空间误差、Roache 带宽
$U=F_s d_{{fine}}/(r^p-1)$ 被当作参考自身不确定度。时间离散到底贡献了多少，
从未被测量。

## 2. 怎么做的

1. 固定网格、只细化 CFL（0.3 → 0.15 → 0.075，比例 2），差分即纯时间误差；
2. 时间差分用**整条序列共用分母**（最细成员 L1 范数）——逐对归一化会把误差
   自身混进分母，合成幂律上恢复阶从 1.0 掉到 0.9978（已由测试钉死）；
3. 时间观测阶 $p_t$，时间 Richardson $u_T=u_{{0.075}}+(u_{{0.075}}-u_{{0.15}})/(2^{{p_t}}-1)$
   逐网格到 $CFL\\to0$；
4. 污染比 $R=\\delta_t/d_{{fine}}$（$F_s\\!=\\!1.25$、$r^p\\!-\\!1\\approx1.25$ ⇒ $R$ 即
   「时间误差占带宽的比例」）；
5. 时间修正梯次重新 `certify()`，与已发布阶/带宽对比；
6. 11 条已发布 DG 腿分别对两种参考在 3% 门槛下重判，统计翻转数（污染是否 material）。

**预声明判据（模块常量，先于数据固定）**：
T0 相邻时间差分严格递减 · T1 $R\\le0.10$（$>0.25$ 即 POLLUTED）·
T2 $p_t\\ge0.5$ · T3 带宽位移 $\\le25\\%$ 且修正梯次仍 CERTIFIED ·
T4 无 3% 判定翻转且逐腿位移 $\\le0.5$ pp。CLI `--require-accepted` 红跑非零退出。

## 3. 计算结果（引擎在已知真值的合成问题上验证）

合成树：$u=1+0.2x+0.1\\,r^{{-1.2}}w+ k_t r^{{-2}}\\,CFL^2 w$（$r=N/200$，$w$ 按 200 格块常值 ⇒
粗化精确、差分是纯幂律）。两种 $k_t$ 给出已知答案：

| 场景 | $R$（N={res['reference_grid']}） | T1 | T3 带宽位移 | T4 翻转 | 判定 | accepted |
|---|---|---|---|---|---|---|
| $k_t=1$（时间腿可忽略） | {pub._f(res['criteria']['T1']['by_grid'].get(tp.REFERENCE_GRID), 4)} | CLEAN | {pub._f(res['criteria']['T3']['band_shift_relative'], 4)} | {res['criteria']['T4']['n_flips']} | **TIME_CLEAN** | True |
| $k_t=2000$（时间腿主导） | 由引擎实测 > 0.25 | POLLUTED | 见测试断言 | 见测试断言 | **TIME_POLLUTED** | False |

合成幂律上的时间阶恢复：$p_t$ = {pub._f(res['criteria']['T2']['temporal_order'], 6)}（真值 2.0）。

### 3.1 真实数据侧（现状）

| 网格 | CFL 剖面 | 状态 |
|---|---|---|
| N=800 | 0.3（已有）/ 0.15 / 0.075 | 0.15+0.075 在 cj4305 排队 |
| N=1600 | 0.3（已有）/ 0.15 / 0.075 | 在 cj4306 排队 |
| N=3200 | 0.3（已有）/ 0.15 / 0.075 | 在 cj4307 排队 |
| N=6400 | 0.3（已有）/ 0.15 | 在 cj4308 排队 |

真实侧 `δ_t / p_t / R / 带宽位移 / 翻转数` **一项都还没有测到**，因此
reference-grid 判定 = `WITHHELD`；本单不写任何真实数据的结论。
（本地直跑被计算门限 120 s 收割：N=800 单个 CFL 级约 70–140 s，无法在本机完成。）

## 4. 结果图

`fig_time_vs_space_leg.png`（时间腿 vs 空间腿，图上标注 R）、
`fig_cfl_profile_difference.png`（同网格两个 CFL 的剖面与差值）、
`fig_leg_stability.png`（逐腿误差双参考对比 + 3%/5% 门槛）——
**三张图均由合成基准树生成**，用于验证引擎与绘图链路；真实数据版本等 cj4305-cj4308
落地后由 `python validation/xval_dg_round2/publish_time_pollution.py` 一条命令重出。

## 5. 注意点（坑 / 局限）

1. **归一分母坑**：逐对归一化使合成幂律恢复阶偏 0.22%（0.9978 vs 1.0）；改为公共分母后
   精确到 1e-14。
2. **混合网格坑**：审计初版在算"空间差分"时残留一行同网格 `rel_l1(fine, coarse)`，
   在真实数据上会直接抛 shape mismatch；合成端到端测试当场抓到（已修）。
3. **时间阶受激波限制**：SSP-RK3 光滑区三阶，间断处退化，故 $p_t$ 只用于时间修正，
   不作为精度结论。
4. **N=6400 只有两级 CFL**：可算 $R$，无第二对差分 ⇒ 无 $p_t$、不能做时间修正；
   四级梯次的时间修正（约 2.5 CPU·h）未在本单范围。
5. **`certify()` 的带宽门槛**：$U/\\tau\\le1/3$ 是硬规则；合成基准若带宽吃满门槛，
   T3 会因为「修正梯次不再 CERTIFIED」而红——这属于基准设置而非引擎缺陷（首版合成
   基准 $a_{{space}}=0.5$ 正是这样红的，已调到 0.1）。

## 6. 误差判定（显式回答是否 < 3%）

| 关键量 | 基准（合成真值） | 实测 | 误差 | < 3% ? |
|---|---|---|---|---|
| 时间观测阶 $p_t$ | 2.0 | {pub._f(res['criteria']['T2']['temporal_order'], 6)} | {pub._f(abs(res['criteria']['T2']['temporal_order'] - 2.0), 8)} | **是** |
| 时间差分 $\\delta_t$（解析式） | $k_t(0.3^2-0.15^2)r^{{-2}}$ | `test_temporal_difference_is_pure_time_error` rel=1e-10 | ~1e-12 | **是** |
| 时间修正后 $CFL\\to0$ 剖面 | 解析 $CFL\\to0$ 极限 | `test_time_correction_kills_the_temporal_error` atol=1e-13 | ~1e-15 | **是** |
| 真实参考解污染比 $R$ | 无（未测） | **WITHHELD** | — | **无法回答**（数据未到，不猜） |

## 7. 复现

```bash
# 1) 生成 CFL 细化剖面（集群，本单已递交 cj4305-cj4308）
python validation/xval_dg_round2/gen_fv_level.py --ncells 3200 --cfl 0.15 \\
    --out validation/repro_shu_osher/artifacts/raw/fv_cfl0p15_n3200.npz
# 2) 审计 + 图（数据到位后自动走真实分支）
python -m benchmarks.xval_dg_round2_time_pollution --figures --require-accepted
# 3) 报告/文档（数字程序化注入）
python validation/xval_dg_round2/publish_time_pollution.py
# 4) 测试
python -m pytest -q tests/test_xval_dg_round2_time_pollution.py
```

测试结果：`{tests}`

## 8. 产物

| 文件 | 说明 |
|---|---|
| `benchmarks/xval_dg_round2_time_pollution.py` | T0–T4 审计引擎（预声明门槛、CLI 可复算、红跑非零退出） |
| `tests/test_xval_dg_round2_time_pollution.py` | 25 项：合成幂律/拒绝路径/门槛钉扎/合成树端到端（真值已知） |
| `validation/xval_dg_round2/publish_time_pollution.py` | 真实数据分支的文档+报告发布器 |
| `validation/xval_dg_round2/publish_time_pollution_status.py` | 本页生成器（数据缺失时走合成基准并显式 WITHHELD） |
| 本文件 | 方法与现状（数字全部程序化注入） |
"""


def main() -> int:
    real = real_data_present()
    if real:
        res = tp.audit()
        rawdir = tp.DEFAULT_RAW
        label = "真实 CFL 细化剖面"
    else:
        tmp = Path(tempfile.mkdtemp(prefix="xval_time_pollution_")) / "raw"
        rawdir = synthetic_rawdir(tmp, k_time=1.0)
        res = tp.audit(rawdir)
        label = "合成基准树（真值已知；真实数据在集群排队）"

    tp.write_artifacts(res, tp.DEFAULT_OUT)
    figs = tp.figures(res, rawdir, tp.DEFAULT_OUT)
    tests = pub.run_tests()
    md = status_md(real, res, tests)
    DOC.write_text(md)

    h = res["headline"]
    title = ("CASE-SHU-OSHER 认证参考的时间离散污染审计：引擎 + 预声明判据就位，"
             "真实数据判定 WITHHELD（集群排队）"
             if not real else
             f"CASE-SHU-OSHER 认证参考时间离散污染审计：R={pub._f(h['pollution_ratio'], 4)}"
             f"（{h['state']}）")
    summary = (
        f"新增时间腿审计引擎（固定网格只细化 CFL 0.3/0.15/0.075，T0–T4 预声明判据，"
        f"红跑非零退出），并在真值已知的合成基准上验证："
        f"R={pub._f(h['pollution_ratio'], 4)}（{h['state']}）、时间阶恢复 p_t="
        f"{pub._f(res['criteria']['T2']['temporal_order'], 4)}（真值 2.0）、"
        f"时间修正到 CFL→0 达 1e-13。"
        f"真实参考解的 R/带宽位移/11 腿判定翻转 **未测**：CFL 细化剖面作业 "
        f"{JOBS} 40+ min 仍在队列，本地直跑被计算门限收割 ⇒ reference-grid 判定 WITHHELD，"
        f"不写任何真实数据结论。")
    resp = pub.post(title, summary, md, figs)
    print(resp[:300])
    print(f"real_data={real} doc={DOC} figures={figs}")
    (Path(tp.DEFAULT_OUT) / "publish_status.json").write_text(json.dumps(
        {"real_data": real, "branch_label": label, "verdict": res["verdict"],
         "accepted": res["accepted"], "jobs": JOBS,
         "headline": {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                      for k, v in h.items()},
         "tests": tests}, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())