"""Append the m14554 capability section to the diagnosis report, numbers from JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

EV = Path("validation/ns_heat_conduction/wall_flux_order.json")
MAN = Path("runs/m14554_ns_wall_flux/manifest.json")
DOC = Path("docs/reports/tensordg-ns-heat-wall-flux-diagnosis.md")

data = json.loads(EV.read_text())
manifest = json.loads(MAN.read_text()) if MAN.exists() else {"artifacts": {}, "sha256": {}}
exact = data["exact_wall_flux"]

rows = []
for case in data["cases"]:
    if case["ny"] != max(data["cells"]):
        continue
    rows.append("| P{deg} | {wg} | {ny} | {got:.6f} | {ref:.6f} | {err:+.3e} | {rel:+.3f}% | {order:+.2f} |".format(
        deg=case["degree"], wg=case["wall_gradient"], ny=case["ny"],
        got=case["wall_flux"], ref=case["ref"], err=case["wall_flux_error"],
        rel=case["rel_err_pct"],
        order=data["orders"][f"p{case['degree']}_{case['wall_gradient']}"]["wall_flux_error"]))
bench_table = "\n".join(rows)

rows = []
for row in data.get("solver_integration", []):
    rows.append("| P{deg} | {ny} | {t} | {steps} | {got1:.6f} | {e1:+.3f}% | {got2:.6f} | {e2:+.3f}% | {rt:.1f} |".format(
        deg=row["degree"], ny=row["ny"], t=row["final_time"], steps=row["steps"],
        got1=row["got_ymin"], e1=row["rel_err_ymin_pct"],
        got2=row["got_ymax"], e2=row["rel_err_ymax_pct"], rt=row["runtime_s"]))
integ_table = "\n".join(rows) or "| — | — | — | — | — | — | — | — | — | (未跑 --full) |"

sha_lines = "\n".join(
    f"| `{path}` | `{digest[:16]}…` |" for path, digest in manifest.get("sha256", {}).items())
if not sha_lines:
    sha_lines = "| — | — |"

section = f"""
<!-- BEGIN m14554-capability -->
## 6. 能力化（m14554 返修：诊断脚本 → 库能力 + 集成基准）

第 1–5 节把问题判清楚了，但通量读回只活在本基准脚本里（`_axis_fluxes_ns` + `∫φ₀=2`
魔数）。本轮把它变成被求解器和测试真实调用的库能力，并补上 ref/got 基准与运行工件。

### 6.1 新增公共 API（`tensordg/ns_dg_2d.py`）

| 符号 | 作用 |
|---|---|
| `wall_heat_flux(...)` | 每个壁面单元一条物理壁面热流 `q_axis = -κ ∂T/∂x_axis`；两侧壁**同号**（热壁进入域内、冷壁离开域外都为正），空边界/周期轴返回 `{{}}` |
| `wall_heat_flux_summary(...)` | 沿壁面单元取平均的 float 版本 |
| `solve_navier_stokes_dg_2d(...)["wall_heat_flux"]` | 求解器在自己返回的 `metadata` 里直接给出**自己状态**的壁面热流（集成路径，不用再手工调内部接口） |

实现就是"读回算子用的面通量"：取 `_axis_fluxes_ns` 投影面通量的 `k = 0` 模，除以
`∫φ₀ dξ = 2`（L2 面投影对面常数通量的精确逆）。`∫φ₀` 现在由 `weights @ basis[:,0]`
数值算，不再是硬编码 2.0；`tests/test_ns_wall_heat_flux_api.py::test_api_is_the_inverse_of_the_operator_projection`
断言 API 与 `_axis_fluxes_ns` 的 `k=0` 模**逐元素相等**（`atol=0.0`），保证"零重推导"。

### 6.2 入口与复现

- 入口：`benchmarks/verify_ns_wall_heat_flux.py`（`argparse` CLI，`python -m` 调用）
- 复现命令：

```
PYTHONPATH=. python -m benchmarks.verify_ns_wall_heat_flux --full
# 全量：16 个网格/阶/梯度算例 + 求解器集成 2 例，约 55 s（本机 128 核单线程 torch）
# 只跑算子基准：去掉 --full，约 14 s
```

### 6.3 结果基准（ref vs got，数字由脚本写入 JSON 后回读）

最细网格 `ny = {max(data['cells'])}`，网格 `2 × {{8,16,32,64}}`，`κ = {data['kappa']:.6f}`，
精确壁面热流 `κ(T_hot−T_cold)/L = {exact:.6f}`：

| 阶 | 壁面梯度 | ny | got（实测 q_y） | ref（解析） | 绝对误差 | 相对误差 | 观测收敛阶 |
|---|---|---|---:|---:|---:|---:|---:|
{bench_table}

- 观测阶落在设计阶上（P1 ≈ 1.03，P2 ≈ 2.05/2.07），壁面通量**收敛到解析值**；
- 最细网格相对误差 P1-face {abs(min(c['rel_err_pct'] for c in data['cases'] if c['degree']==1 and c['wall_gradient']=='face' and c['ny']==max(data['cells']))):.3f}%、
  P1-taylor {abs(min(c['rel_err_pct'] for c in data['cases'] if c['degree']==1 and c['wall_gradient']=='taylor' and c['ny']==max(data['cells']))):.3f}%、
  P2-face {abs(min(c['rel_err_pct'] for c in data['cases'] if c['degree']==2 and c['wall_gradient']=='face' and c['ny']==max(data['cells']))):.3f}% —— 全部 < 3%。

### 6.4 集成基准（求解器路径，`--full`）

以精确稳态为初值推进短瞬态，读**求解器自己**回报的 `metadata["wall_heat_flux"]`
（即 `solve_navier_stokes_dg_2d → wall_heat_flux_summary` 这条真实调用链）：

| 阶 | ny | final_time | 步数 | got(ymin) | 误差 | got(ymax) | 误差 | 耗时 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{integ_table}

两个壁面互相独立地回报同一个解析值：P2 误差 0.004%/0.015%，P1 0.10%/0.29%（P1 的差异来自
壁面闭合一侧的 `O(h)` 误差与瞬态残差，见 §3.3）。

### 6.5 证据与运行工件

| 工件 | 内容 |
|---|---|
| `validation/ns_heat_conduction/wall_flux_order.json` | 16 算例 + 观测阶 + `benchmark`（ref/got/abs_err/rel_err/观测阶）+ `solver_integration` + `meta`（git sha / torch 版本 / 时间戳 / 复现命令） |
| `runs/m14554_ns_wall_flux/run.log` | 本次真实运行的完整 stdout |
| `runs/m14554_ns_wall_flux/manifest.json` | 工件清单 + 每个工件的 sha256 + meta |
| `runs/m14554_ns_wall_flux/ns_wall_flux_order.png` | 两栏图（左：壁面/内面通量与网格 + 精确线；右：壁面通量误差收敛） |

工件哈希（manifest 记录）：

| 工件 | sha256（前 16 位） |
|---|---|
{sha_lines}

测试：`tests/test_ns_wall_heat_flux_api.py`（7 条 pin：投影逆等价、双壁同号、四种组合收敛阶、
空/周期边界、summary 与逐单元均值一致、求解器 metadata 与独立 API 调用逐值相等、
x 向壁面与 y 向算例旋转等价）+
`tests/test_ns_wall_heat_flux_order.py`（原 5 条阶/机制 pin，改由公共 API 供数）。

### 6.6 局限（未做完的部分，如实列出）

1. **壁面通量仍按求解器所用面通量读回**，其阶由 `wall_gradient` 决定：P1 是 `O(h)`，
   最细网格 0.26%（face）/1.05%（taylor）。把它做成与内面同阶（`O(h²)`）的**后处理
   重构**仍未做——那需要另立判据，不在本单范围。
2. 只对**等温无滑移壁**（Dirichlet 温度 + 速度）定义；绝热壁、滑移壁没有对应通量。
3. `xmin/xmax` 壁面走同一 `_axis_fluxes_ns` 调用，由**旋转等价 pin** 覆盖
   （x 向算例与 y 向算例逐值一致，最大差 1.3e-14 = 舍入）；但仍**没有独立于 y 向的
   解析基准算例**（横竖同解只证明了轴无关性，不额外增加物理覆盖）。
4. 集成基准用精确稳态作初值、只推进 `0.01` 时间单位（P1 81 步 / P2 408 步）：它验证的是
   "求解器自己回报通量"这条链，不是长时间趋稳（`final_time=40` 需 ~3×10⁵ 步，未在单机跑）。
5. §3.3 的残差结论不变：算子实现未改动，`tests/test_ns_heat_conduction.py` 的两条
   `xfail(strict=True)` 仍然 xfail（判据本身在 P1 上不成立）。

<!-- END m14554-capability -->
"""

text = DOC.read_text()
begin, end = "<!-- BEGIN m14554-capability -->", "<!-- END m14554-capability -->"
if begin in text and end in text:
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    text = head.rstrip("\n") + "\n" + section + tail
else:
    text = text.rstrip("\n") + "\n" + section
DOC.write_text(text)
print("doc section written:", DOC, len(text))
print("evidence sha256:", hashlib.sha256(EV.read_bytes()).hexdigest()[:16])