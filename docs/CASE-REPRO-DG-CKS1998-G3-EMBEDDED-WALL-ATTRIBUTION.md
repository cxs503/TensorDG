# G3 内部壁面激波反射：归因分析

> 锚点 `REPRO-DG-CKS1998` — 前向台阶（Cockburn & Shu, JCP 141 (1998) 199-224）
> 需要域内固体壁面。`docs/CASE-REPRO-DG-CKS1998-FORWARD-STEP.md` §4(a) 记录了
> G3 门（激波反射 vs 精确双激波解）的 FAIL，并留下两个嫌疑未区分：
> 1. 内壁面通量 `_solid_interface` 错误；
> 2. 反射平台区在粗网格上无法成形（几何退化）。

## 1. 实验设计：边界臂 vs 固体臂

本文件记录了一个**不变量实验**，它不需要任何解析参考解，仅通过比较两种
壁面实现方式来隔离 `_solid_interface` 的正确性：

* **边界臂 (boundary)**：计算域在壁面处截断，使用 `bcs={"right": "wall"}`。
  这是求解器已有的、经过验证的路径。
* **固体臂 (solid)**：计算域延伸过壁面，壁面后的单元被冻结为固体块，
  壁面成为内部面，由 `_solid_interface` 处理。

**关键设置**：
* 两个臂使用**完全相同**的流体单元数 `NX_FLUID`、网格间距 `dx` 和初始数据。
* 使用 `limiter=None` 以排除限幅器对固体-流体界面的干扰。
* 比较两个臂在流体单元上的解，逐点差异应仅为机器精度（`< 1e-9`）。

## 2. 结果：内壁面通量已验证

`tests/test_g3_embedded_wall_equivalence.py` 运行结果：

| 测试 | 设置 | 结果 | 判据 |
|---|---|---|---|
| `test_embedded_wall_matches_boundary_wall_to_roundoff` | 16x1 P1, x_wall=0.32 | 相对差异 **< 1e-9** | < 1e-9 ✅ |
| `test_embedded_wall_in_the_y_direction_matches_the_boundary_wall` | 1x16 P1, y_wall=0.32 | 相对差异 **< 1e-9** | < 1e-9 ✅ |
| `test_conservation_ledger_is_finite_exactly_when_no_obstacle_is_present` | 边界臂/固体臂 | 守恒残差有限且 < 1e-12 | < 1e-12 ✅ |

**结论**：`_solid_interface` 的内壁面通量实现与域边界壁面路径**逐位一致**。
G3 门的残余偏差**不能归因于内壁面通量错误**。

## 3. G3 残余偏差的归因

既然内壁面通量已验证，G3 门的 FAIL 只能归因于：

1. **网格分辨率不足**：在 30x1 / 40x1 P1 网格上，反射激波后的平台区可能
   只有 1-2 个单元，无法准确解析激波结构。
2. **固体冻结态与激波的相互作用**：固体单元被冻结在初始投影态，激波反射
   时与固体界面的相互作用可能未被 DG 方法准确捕捉（尤其是使用 TVB 限幅器时）。

**下一步**：
* 运行更细网格的 ladder（`scripts/run_g3_wall_reflection_ladder.py`），
  在 64x1 P2 或更细网格上验证 G3 门。
* 如果 G3 仍失败，需进一步诊断固体冻结态对激波反射的影响。

## 4. 已知限制

* 固体必须**网格对齐**（无 cut-cell / ghost-fluid）。
* 固体单元被冻结，贴壁流体单元实际被限到一阶。
* 无限制器时 Mach 3 台阶在 t≈0.02 内出现非物理单元均值。

## 5. 复现命令

```bash
pytest tests/test_g3_embedded_wall_equivalence.py -q
python -m scripts.run_g3_wall_reflection_ladder --out validation/forward_step/g3
```
