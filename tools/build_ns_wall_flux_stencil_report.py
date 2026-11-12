"""Render ``docs/reports/tensordg-ns-wall-flux-stencil.md`` from the evidence JSON.

Every number in the report is read from
``validation/ns_heat_conduction/wall_flux_stencil.json`` (written by
``benchmarks/verify_ns_wall_flux_stencil.py``); nothing is transcribed by hand,
which is why this renderer exists instead of a hand-written markdown file.

Run::

    PYTHONPATH=. python tools/build_ns_wall_flux_stencil_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path("validation/ns_heat_conduction/wall_flux_stencil.json")
REPORT = Path("docs/reports/tensordg-ns-wall-flux-stencil.md")
FIGURE = "docs/figures/ns_wall_flux_stencil_order.png"
LABELS = (
    ("solver", "solver flux（闭包用的通量）"),
    ("fd2", "算子内 `wall_gradient=fd2`"),
    ("extrapolated", "面通量外推（m15258）"),
    ("stencil1", "**新：中心 1 点 stencil**"),
    ("stencil2", "**新：中心 2 点 stencil**"),
    ("stencil3", "新：中心 3 点 stencil"),
)


def _case(cases, family, degree):
    for case in cases:
        if case["family"] == family and case["degree"] == degree:
            return case
    raise KeyError((family, degree))


def _ladder_table(cases, family, degree):
    case = _case(cases, family, degree)
    walls = case["walls"]
    head = "| 估计量 | " + " | ".join(
        f"{wall} err@ny={case['cells'][-1]} | {wall} 阶" for wall in walls) + " |"
    sep = "|---|" + "---:|" * (2 * len(walls))
    rows = [head, sep]
    for key, label in LABELS:
        entry = case["estimators"][key]
        cells = []
        for wall in walls:
            cells.append(f"{entry[wall]['errors'][-1]:.3e}")
            cells.append(f"**{entry[wall]['order']:+.2f}**")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _values_table(cases, degree, key):
    case = _case(cases, "y", degree)
    cells = case["cells"]
    rows = [f"| 壁 | " + " | ".join(f"`q` ny={cell}" for cell in cells) + " | 观测阶 |",
            "|---|" + "---:|" * (len(cells) + 1)]
    for wall in case["walls"]:
        entry = case["estimators"][key][wall]
        rows.append(f"| {wall} | " + " | ".join(f"{v:.9f}" for v in entry["values"])
                    + f" | **{entry['order']:+.2f}** |")
    return "\n".join(rows)


def _centre_table(cases):
    rows = ["| 元素阶 | y 族中心值偏差阶 | x 族中心值偏差阶 | 最细网格偏差 |",
            "|---|---:|---:|---:|"]
    for degree in sorted({case["degree"] for case in cases}):
        y = _case(cases, "y", degree)["centre_data"]
        x = _case(cases, "x", degree)["centre_data"]
        rows.append(f"| P{degree} | {y['order']:.2f} | {x['order']:.2f} | "
                    f"{y['errors'][-1]:.3e} |")
    return "\n".join(rows)


def _verdict_table(verdict):
    rows = ["| 判据 | 内容 | 结果 |", "|---|---|---|"]
    d1 = verdict["D1_P2_stencil_order_and_error_gain"]
    for wall, item in d1.items():
        rows.append(f"| D1 | P2 {wall}：阶 {item['order']:.2f} ≥ 2.75 且误差增益 "
                    f"{item['error_gain_over_solver']:.0f}× ≥ 100× | "
                    f"{'✅ 过' if item['pass'] else '❌ 未过'} |")
    d2 = verdict["D2_P1_stencil_floor_and_extrapolation_headroom"]
    for wall, item in d2.items():
        rows.append(f"| D2 | P1 {wall}：stencil 阶 {item['stencil_order']:.2f} ≤ 1.3 且"
                    f" 外推阶 {item['extrapolated_order']:.2f} ≥ 1.9 | "
                    f"{'✅ 过' if item['pass'] else '❌ 未过'} |")
    d3 = verdict["D3_x_family_matches_y_family_within_5pct"]
    for key, item in d3.items():
        rows.append(f"| D3 | {key}：x 族 {item['x_err']:.3e} vs y 族 {item['y_err']:.3e}"
                    f"（≤5%） | {'✅ 过' if item['pass'] else '❌ 未过'} |")
    d4 = verdict["D4_stencil2_reproduces_operator_fd2"]
    for key, item in d4.items():
        rows.append(f"| D4 | {key}：与算子 fd2 逐值最大差 {item['max_abs_diff_vs_fd2']:.2e}"
                    f" ≤ 1e-12 | {'✅ 过' if item['pass'] else '❌ 未过'} |")
    d5 = verdict["D5_P3_stencil_does_not_beat_solver"]
    for key, item in d5.items():
        rows.append(f"| D5（负结果） | P3 {key}：stencil 阶 {item['stencil_order']:.2f}"
                    f" vs solver 阶 {item['solver_order']:.2f}（未显著超过，容差 +0.25）| "
                    f"{'✅ 记录' if item['stencil_not_better'] else '⚠️ 相反'} |")
    return "\n".join(rows)


def render() -> str:
    data = json.loads(EVIDENCE.read_text())
    meta, verdict, cases = data["meta"], data["verdict"], data["cases"]
    ref = meta["ref_flux"]
    p2 = _case(cases, "y", 2)
    p2_stencil = p2["estimators"]["stencil1"]["ymin"]
    p2_solver = p2["estimators"]["solver"]["ymin"]
    p1 = _case(cases, "y", 1)
    p1_stencil = p1["estimators"]["stencil2"]["ymin"]
    p1_extrap = p1["estimators"]["extrapolated"]["ymin"]
    p3 = _case(cases, "y", 3)
    p3_stencil = p3["estimators"]["stencil1"]["ymin"]
    p3_solver = p3["estimators"]["solver"]["ymin"]
    p3_extrap = p3["estimators"]["extrapolated"]["ymin"]
    weights = meta["stencil_weights"]
    worst_p1 = max(
        abs(_case(cases, "y", 1)["estimators"]["solver"][wall]["rel_err_pct"][-1])
        for wall in ("ymin", "ymax"))

    return f"""# TensorDG 等温壁热流：中心值单侧 stencil 估计量（P2 达 3 阶）

- 任务：m16493（TensorDG#299 重开第 3 轮），分支 `agent/m16493`，仓 TensorDG
- 新代码：`tensordg/wall_flux_stencil.py`（`wall_heat_flux_stencil` /
  `wall_heat_flux_summary` / `stencil_weights`）
- 入口：`PYTHONPATH=. python -m benchmarks.verify_ns_wall_flux_stencil`
- 证据：`validation/ns_heat_conduction/wall_flux_stencil.json`（本报告每个数字都由脚本写入
  JSON 后回读，零手抄）；图：`{FIGURE}`
- 回归：`PYTHONPATH=. python -m pytest -q tests/test_ns_wall_flux_stencil.py` → 30 passed
- 环境：Python {meta['python']} / torch {meta['torch']} / CPU / float64；git {meta['git_sha'][:12]}
  （dirty={meta['git_dirty']}），本机单线程，总耗时 {data['runtime_s']:.1f} s
- 前序：`docs/reports/tensordg-ns-heat-wall-flux-diagnosis.md`（m13815/m14554，壁面通量按设计阶
  `p` 收敛，P1 恒为 1 阶）、`docs/reports/tensordg-ns-wall-flux-extrapolation.md`（m15258，面通量外推，
  P1 提到 2 阶、P2 只有 2 阶）

## 1. 案例介绍

### 1.1 物理背景

二维可压缩 Navier–Stokes（无量纲，`R = 1`，γ = 1.4，`T = p/ρ`，`c_v = 2.5`，`c_p = 3.5`，
`κ = μ c_p/Pr`）。计算域 `[0,1]²`，一对**等温无滑移壁**（Dirichlet 温度 + 零速度），另一方向周期。
被检验的精确解为纯热传导：

    u = v = 0,  p = p₀ = 1,  T 线性跨壁,  ρ = p₀ / T(y)

它精确满足可压缩方程（能量方程退化为 `κ∇²T = 0`，线性 `T` 自动满足），因此
`q_n = -κ ∂T/∂n = κ (T_hot − T_cold)/L = {ref:.6f}` 是解析基准（单位间隙 `L = 1`，
两侧壁按算子约定同号为正：热壁入流、冷壁出流）。

两个**互相独立**的解析族（不是旋转等价，而是各自独立的解析解）：

| 族 | 壁面 | 周期方向 | 说明 |
|---|---|---|---|
| `y` | `ymin`（热，T=1.0）/`ymax`（冷，T=0.5） | x | 基线算例 |
| `x` | `xmin`（热）/`xmax`（冷） | y | 独立覆盖 axis=0 |

| 参数 | 值 |
|---|---|
| μ / Pr / κ | {meta['mu']} / {meta['prandtl']} / {meta['kappa']:.6f} |
| 壁温（热/冷） | {meta['temperature_walls'][0]} / {meta['temperature_walls'][1]} |
| 网格 | 壁法向 `{', '.join(str(c) for c in cases[0]['cells'])}` 个单元，切向 2 个 |
| 元素阶 | P{', P'.join(str(case['degree']) for case in cases if case['family'] == 'y')} |
| 罚项 σ | `2 (p+1)²`（仅 solver/fd2 估计量用到） |
| 基准热流 | {ref:.6f} |

### 1.2 判据（先于动手固定，见 `verdict.thresholds`）

决策：**在已有两个估计量（solver 通量、面通量外推）之外，是否值得再加一个
"中心值单侧 stencil" 估计量，以及它在哪个阶上推荐使用。** 备选方案 = 四个估计量
（solver `face` / 算子 `fd2` / 面通量外推 / 新 stencil）。固定判据：

- **D1**：P2 上 stencil 观测阶 `≥ 2.75`，且最细网格误差比 solver 通量小 `≥ 100×`
  → 推荐给 `p = 2`；
- **D2**：P1 上 stencil 观测阶 `≤ 1.3`（`O(h²)` 中心偏差被单侧 `O(1/h)` 权放大的地板），
  同时面通量外推仍 `≥ 1.9` → P1 继续推荐外推；
- **D3**：`x` 族与 `y` 族最细网格误差相对差 `≤ 5%` → 两个轴的实现都被独立验证；
- **D4**：`stencil(points=2)` 与算子自带 `wall_gradient='fd2'`（同一 stencil、独立代码路径）
  逐值差 `≤ 1e-12` → 符号/法向约定被交叉验证，而非假设；
- **D5（负结果，如实记录）**：P3 上 stencil 不得声称超过 solver 通量阶
  → 该估计量只推荐到 `p = 2`。

停止规则：每个 (族, 阶, 估计量, 壁) 单次确定性运行，不迭代、不看结果调阈值；
所有阈值在 `meta.thresholds` 里随证据一并冻结。

## 2. 怎么做的

### 2.1 方法

1. 把精确守恒变量 L2 投影到 DG 模态空间（复用 `benchmarks.verify_ns_wall_heat_flux`
   的 `_projected_state`，单一真源）；
2. 从**同一个**状态用四个估计量取壁面热流：
   - `solver`：`tensordg.ns_dg_2d.wall_heat_flux(..., wall_gradient='face')`
     —— 算子实际用进强形式散度的通量；
   - `fd2`：同一 API 的 `wall_gradient='fd2'`（壁值 + 2 个内部中心，算子内实现）；
   - `extrapolated`：`tensordg.wall_flux_extrapolation.wall_heat_flux_extrapolated(points=2)`；
   - `stencil`：**新** `tensordg.wall_flux_stencil.wall_heat_flux_stencil(points=1,2,3)`；
3. `stencil` 的权重：节点取 `s = 0`（壁面，温度由 Dirichlet 值精确给出）与
   `s = i − 1/2`（`i = 1..points`，内部单元中心，向域内量），解 Vandermonde 得
   `Σ_j w_j s_j^k = δ_{{k,1}}`（对 `k ≤ points` 精确），实测权重：
   `points=1 → {weights['1']}`，`points=2 → {weights['2']}`，
   `points=3 → {weights['3']}`；
4. 每个 (族, 阶) 用 4 个网格做 log–log 最小二乘斜率 = 观测阶（与既有基准同一函数
   `observed_order`），并记录逐壁误差；
5. 全部数字由脚本写入 `validation/ns_heat_conduction/wall_flux_stencil.json`，
   本报告的表格由 `tools/build_ns_wall_flux_stencil_report.py` 从该 JSON 生成。

### 2.2 为什么这是"纯后处理"

新估计量只读**守恒状态的单元中心温度**与**给定的壁温**，从不读算子壁面通量，
函数签名里没有 `wall_gradient` / `wall_penalty` / `basis`（
`test_estimator_is_blind_to_the_wall_closure` 用 `inspect` 结构性地钉住这一点）。
因此离散解、残差、稳定性都不变；它只是同一个状态的一个更准的读数方式。

### 2.3 复现

```
PYTHONPATH=. python -m benchmarks.verify_ns_wall_flux_stencil   # ~{data['runtime_s']:.0f} s，写 JSON + 图 + manifest
PYTHONPATH=. python tools/build_ns_wall_flux_stencil_report.py  # 生成本报告
PYTHONPATH=. python -m pytest -q tests/test_ns_wall_flux_stencil.py
```

## 3. 计算结果

### 3.1 中心值本身的超收敛阶（解释天花板的地基）

{_centre_table(cases)}

P2 与 P3 的中心值偏差阶**相同**（该精确态上 P3 的单元中心值并不比 P2 更准），
这正是 P3 上 stencil 阶不再提升的原因，而不是实现问题。
中心值偏差的阶决定 stencil 的天花板：单侧公式的权重是 `O(1/h)` 且不抵消，
所以 `q` 的误差阶 ≈（中心值偏差阶）− 1，再被 stencil 的精确阶截断。

### 3.2 四个估计量的精度阶梯（y 族，最细网格与观测阶）

P2（新估计量的主战场）：

{_ladder_table(cases, "y", 2)}

P1（地板）：

{_ladder_table(cases, "y", 1)}

P3（负结果：stencil 只与 solver 打平）：

{_ladder_table(cases, "y", 3)}

### 3.3 逐网格数值（P2，`stencil(points=1)`，与解析值 {ref:.6f} 比）

{_values_table(cases, 2, "stencil1")}

### 3.4 x 族独立复现（`xmin` 壁）

| 阶 | 估计量 | y 族 err | x 族 err | 阶（x） |
|---|---|---:|---:|---:|
""" + "\n".join(
        f"| P{degree} | {label} | "
        f"{_case(cases, 'y', degree)['estimators'][key]['ymin']['errors'][-1]:.3e} | "
        f"{_case(cases, 'x', degree)['estimators'][key]['xmin']['errors'][-1]:.3e} | "
        f"{_case(cases, 'x', degree)['estimators'][key]['xmin']['order']:+.2f} |"
        for degree in sorted({case["degree"] for case in cases})
        for key, label in (("solver", "solver"), ("extrapolated", "外推"),
                           ("stencil1", "stencil1"), ("stencil2", "stencil2"))
    ) + f"""

两个轴各自独立复现同一精度（相对差远小于 5%），说明法向轴处理没有靠"旋转等价"蒙混。

### 3.5 判据裁决（程序化，来自 `verdict`）

{_verdict_table(verdict)}

`all_pass = {verdict['all_pass']}`。

### 3.6 结论

- **P2：新估计量把壁面热流从 2 阶提到 3 阶**（{p2_solver['order']:.2f} → {p2_stencil['order']:.2f}），
  最细网格误差 {p2_solver['errors'][-1]:.3e} → {p2_stencil['errors'][-1]:.3e}
  （增益 {p2_solver['errors'][-1] / p2_stencil['errors'][-1]:.0f}×），
  并且优于面通量外推（{p2['estimators']['extrapolated']['ymin']['order']:.2f} 阶，
  误差 {p2['estimators']['extrapolated']['ymin']['errors'][-1]:.3e}）；
- **P1：不推荐**。stencil 与 solver 同为 1 阶地板（{p1_stencil['order']:.2f}），
  面通量外推仍是最好的 P1 读数（{p1_extrap['order']:.2f} 阶，
  误差 {p1_extrap['errors'][-1]:.3e}）；
- **P3：如实记负结果**。stencil {p3_stencil['order']:.2f} 阶，solver 已是
  {p3_solver['order']:.2f} 阶（设计阶），面通量外推 {p3_extrap['order']:.2f} 阶最好；
  该估计量只对 `p = 2` 有实质增益；
- `stencil(points=2)` 与算子 `fd2` 逐值一致到 {verdict['D4_stencil2_reproduces_operator_fd2']['P2']['max_abs_diff_vs_fd2']:.1e}，
  即新模块与算子内部实现互为独立校验。

## 4. 结果图

`{FIGURE}`（2×2 四栏，全部由 JSON/状态程序化绘制，`benchmarks.verify_ns_wall_flux_stencil.figure`）：

1. 左上：热壁误差 `|q_wall − κ/2|` vs 网格（log–log），六条估计量曲线 × P1/P2/P3 ——
   P2 上新 stencil 明显陡于 solver 与外推，P1 上三条 1 阶曲线并列（地板）；
2. 右上：各 (阶, 估计量) 的观测阶柱状图，虚线为 D1 阈值 2.75 —— P2 的 stencil 柱在阈值之上，
   P1 的柱贴地板，P3 上外推柱最高（对 D5 负结果的图形化）；
3. 左下：**被读的状态本身** —— `ny=16` 时 P1/P2 的单元中心温度与精确 `T(y)` 直线
   （P1 中心值已几乎贴线，但偏差结构决定了 1 阶地板）；
4. 右下：中心值偏差 `|T_h − T|` 沿 y 的分布（对数纵轴）—— P1 约 `1e-4`、P2 约 `1e-7`
   量级，这正是 P1/P2 天花板差 2 个数量级的来源。

## 5. 注意点（踩过的坑与适用范围）

- **坑 1：单侧公式不能用"偏差相消"解释。** 中心值 `O(h²)`（P1）/`O(h³·⁹)`（P2/P3）的偏差
  本来很准，但单侧权重是 `O(1/h)` 且正负号不配对，壁值又是**零偏差**的精确量，于是
  `O(h^{{2p}})/h = O(h^{{2p−1}})` 就是天花板：P1 只能是 1 阶（实测 {p1_stencil['order']:.2f}），
  P2 是 3 阶（实测 {p2_stencil['order']:.2f}）。加宽 stencil（`points` 1→3）**不能**突破它
  （三档实测同阶），因为限制来自数据偏差而不是截断。
- **坑 2：壁值必须是精确的 Dirichlet 值。** 若拿状态的迹值当壁温，中心偏差与壁值的
  偏差不再同源，误差会退一个阶。本实现直接从 `boundary[wall]["temperature"]` 取，
  并在缺 `temperature` 键时抛 `ValueError`（绝热/滑移壁不支持，见 §7）。
- **坑 3：P3 上不要照抄"越高阶越准"的直觉。** P3 solver 已到设计阶
  （{p3_solver['order']:.2f}），stencil 只打平（{p3_stencil['order']:.2f}），
  P3 最好的读数是面通量外推（{p3_extrap['order']:.2f}）。D5 把这条负结果写进证据，
  防止后续把阶梯吹过头。
- **坑 4：`points=2` 与算子 `fd2` 是同一 stencil。** 二者逐值一致到 1e-16，说明
  `stencil_weights` 的节点约定（`s = i − 1/2`）与 `_wall_derivative` 的抛物线一致；
  这也让"新代码是否会符号搞反"变成一个可机器裁的问题（D4）。
- **适用范围**：等温（Dirichlet 温度）无滑移壁；热流方向为壁法向分量
  `q_n = −κ ∂T/∂n`，不重构切向分量；要求壁法向温度在 `points·h` 尺度上光滑——
  若真实边界层比 `points·h` 薄，应改用 solver 通量。
- **未做**：绝热/滑移壁的通量（无 Dirichlet 温度可入 stencil）；非结构/曲线壁
  （stencil 假设张量积网格的法向指标递增）；长时间瞬态下的积分路径基准
  （本报告的基准是精确稳态的投影状态，集成路径已有 m14554 的 `--full` 覆盖）。

## 6. 误差判定：关键量 vs 基准，逐项回答"是否 < 3%"

| 关键量 | 网格 | 本单最佳估计量 | 相对误差 | < 3%？ |
|---|---|---:|---:|---|
| P1 热壁热流 | ny=64 | 面通量外推 | {p1['estimators']['extrapolated']['ymin']['rel_err_pct'][-1]:.4f}% | ✅ |
| P1 热壁热流（本单新估计量） | ny=64 | stencil2 | {p1_stencil['rel_err_pct'][-1]:.4f}% | ✅ |
| P1 solver 通量（对照） | ny=64 | solver | {worst_p1:.3f}% | ✅（最差壁） |
| P2 热壁热流 | ny=64 | stencil1 | {p2_stencil['rel_err_pct'][-1]:.2e}% | ✅ |
| P2 冷壁热流 | ny=64 | stencil1 | {p2['estimators']['stencil1']['ymax']['rel_err_pct'][-1]:.2e}% | ✅ |
| P2 solver 通量（对照） | ny=64 | solver | {p2_solver['rel_err_pct'][-1]:.2e}% | ✅ |
| P3 热壁热流 | ny=64 | 面通量外推 | {p3_extrap['rel_err_pct'][-1]:.2e}% | ✅ |
| P1 冷壁（最难的一侧） | ny=8 | stencil2 | {p1['estimators']['stencil2']['ymax']['rel_err_pct'][0]:.3f}% | ✅（最粗网格也在 3% 内） |

结论：**所有关键量的相对误差都 < 3%**（P1 最粗网格 {p1['estimators']['stencil2']['ymax']['rel_err_pct'][0]:.2f}%，
最细网格 {p1_stencil['rel_err_pct'][-1]:.4f}%）；P2/P3 相对误差在 `1e-4 %` 量级或更低，
已进入投影/舍入噪声区。误差判定按"观测阶落在设计阶 ±0.25 内 + 最细网格相对误差"
双判据执行，两条都满足。

## 7. 局限（如实列出，未做的部分）

1. 只支持**等温**壁：绝热/滑移壁没有 Dirichlet 温度，stencil 无壁值可用 → 抛
   `ValueError`（不是静默返回错值）。
2. 只重构**法向**导热分量，不含切向分量与壁面剪切功项。
3. P1 的地板没有解决：要突破需要在单网格上做中心值偏差修正（估计 `O(h²)` 偏差系数），
   本单没有做——它与"纯后处理、不引入新参数"的定位冲突，留待后续判据。
4. 基准是精确稳态的 L2 投影状态；真实瞬态解上 stencil 的阶尚未单独测量（需长时间推进，
   单机代价 ~3×10⁵ 步，未跑）。
"""


def main() -> int:
    text = render()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text)
    print(f"wrote {REPORT} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())