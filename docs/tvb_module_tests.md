# TensorDG TVB 模块直接单元测试验证报告

> 任务：m3666（自找活·tp-qa-2 verifier 席，原始成果）→ m4541（重开 + 补强）
> 日期：2026-11-04（m3666）/ 2026-11-05（m4541）
> 分支：`agent/m3666` → `agent/m4541`
> 环境：Python 3.11.6 / torch 2.7.1 / pytest 9.2.0

## 1. 案例介绍

**验证对象**：`tensordg/tvb.py` 模块中的 `troubled_from_differences` 函数。

**为什么需要这个测试**：该函数是 1D 和 2D 限制器共用的 TVB 问题单元检测核心实现
（m2549 从 `limiters.py` 中提取）。此前所有测试都通过 `limiters.py` 的
`_troubled_cell_indicator` 间接调用它，没有直接测试。作为限制器系统的基石，
直接测试可以：
1. 隔离验证尺度一致性准则（scale-consistent criterion）
2. 覆盖边界情况（仿射数据、光滑极值、间断）
3. 为未来开发者提供明确的行为文档

**物理背景**：TVB（Total Variation Bounded）准则由 Cockburn & Shu (1989) 提出，
用于检测 DG 解中需要限制的高阶模态。尺度一致性修复（m2500）将比较量从
`max(|d_left|, |d_right|)`（O(dx·u')）改为 `|d_left - minmod(d_left, d_right)|`
（O(dx²·u'') 光滑区 / O(dx·u') 间断处），使 TVB 常数 M 成为网格无关参数。

**m4541 复开时新增的验证目标**：判据中的**舍入底（round-off floor）**
`1e-12 * max(scale, 1)` 是一个与 dtype 无关的常数。它在 `float64`
（eps = 2.22e-16）下比机器精度高约 4 个数量级，但在 `float32`
（eps = 1.19e-7）下**低于一个 ulp**，等于没有保护作用。对本函数做 dtype 扫描后
确认这是真实缺陷：单精度下精确仿射剖面会被判为问题单元，进而被限制器削掉本已
精确的线性斜率（见 §3）。

## 2. 怎么做的

新增 `tests/test_tvb.py`：m3666 提供 18 条直接单元测试，m4541 追加 7 条
（dtype 舍入底 + 局限刻画），共 25 条，覆盖：

| 测试类别 | 测试数 | 验证内容 |
|---|---:|---|
| minmod 辅助函数 | 3 | 同号选小、异号归零、零值处理 |
| 仿射数据 | 2 | 线性剖面永不标记为问题单元 |
| 光滑极值 | 3 | M=0 时标记，大 M 时释放，阈值边界 |
| 间断 | 2 | 任何 M 下都标记 |
| 光滑减速 | 2 | M=0 时标记，适当 M 时释放 |
| 多变量 | 2 | 跨变量归约、形状保持 |
| 舍入底 | 1 | 近仿射数据不因舍入误判（float64） |
| 数据类型 | 2 | float32/float64 保持 |
| 尺度一致性 | 1 | 光滑波上 M 的网格无关性（float64） |
| **舍入底 dtype 自适应（m4541）** | **7** | float32 仿射数据不误判、两精度一致、float32 极值/间断/网格无关性、float64 逐位不变、offset/gradient 局限刻画 |

**m4541 的代码改动**（`tensordg/tvb.py`，最小改动）：

```python
floor_coeff = max(1e-12, 8.0 * torch.finfo(d_left.dtype).eps)
troubled = deviation > threshold + floor_coeff * scale.clamp_min(1.0)
```

- `float64`：`8·eps = 1.8e-15 < 1e-12` → 系数仍为 `1e-12`，**结果逐位不变**
  （由 `test_float64_result_unchanged_by_dtype_aware_floor` 与
  `tests/_limiter_reference.py` 的 `torch.equal` 等价性测试共同保证）。
- `float32`：系数升到 `9.5e-7`，比 eps 高约 8 倍、仍比真实可分辨特征
  （O(dx·u') 的间断、O(dx²·u'') 的极值）低 2~4 个数量级。

## 3. 计算结果

**测试结果（本地 python -m pytest）**：

```
tests/test_tvb.py ......................................... 25 passed
```

m3666 原 18 条全部保持通过；新增 7 条中 2 条为"带牙齿"的回归测试：
**把 `tensordg/tvb.py` 回退到修改前，`test_float32_affine_data_not_troubled` 与
`test_float32_and_float64_agree_on_affine_data` 立即 FAIL**（已实测），修复后恢复 GREEN。

**缺陷量化（`artifacts/m4541/tvb_roundoff_probe.py` 程序化生成）**：

精确仿射剖面 `u = 2x + 1`，201 单元，`dx = 1/200`，被标记为"问题单元"的比例：

| dtype | 舍入底 | M=0 | M=1e-3 | M≥1e-2 |
|---|---|---:|---:|---:|
| float32 | 历史 `1e-12` | **6.03 %** | **6.03 %** | 0 % |
| float32 | `max(1e-12, 8·eps)` | **0 %** | **0 %** | 0 % |
| float64 | 历史 `1e-12` | 0 % | 0 % | 0 % |
| float64 | `max(1e-12, 8·eps)` | 0 % | 0 % | 0 % |

物理后果（走**实际出厂限制器** `tensordg.limiters.apply_tvb_limiter`，M=0，
每单元内精确线性的场）：被削掉斜率模态的单元比例

| dtype | 历史舍入底 | dtype 自适应舍入底 |
|---|---:|---:|
| float32 | **5.97 %** | **0.00 %** |
| float64 | 0.00 % | 0.00 % |

尺度一致性（`u = sin 2πx`，M=100，问题单元比例应随加密不增长）：

| N | 40 | 80 | 160 | 320 |
|---|---:|---:|---:|---:|
| float32 | 0 % | 0 % | 0 % | 0 % |
| float64 | 0 % | 0 % | 0 % | 0 % |

严格 TVD（M=0）下光滑波的问题单元比例两精度均为 50.0 %（极值处应标记），
说明修复没有削弱判据的检出能力。

## 4. 误差判定

本测试为纯单元逻辑测试，断言均为精确匹配（`torch.equal`）或容差匹配
（`torch.allclose` / `pytest.approx`）。关键量逐项对照基准：

| 关键量 | 基准 | 修复前 | 修复后 | 相对基准误差 | 是否 <3% |
|---|---|---|---:|---:|---|
| float64 问题单元掩码 vs 历史表达式 | 逐位相同 | 逐位相同 | **逐位相同**（64×3 随机张量 × 3 组 (M,dx)） | 0 % | ✅ 是 |
| float32 仿射数据误判比例 | 0 % | 6.03 % | **0.00 %** | 修复前 6.03 pp；修复后 0 | 修复后 ✅ 是 |
| float32 线性场被削斜率比例 | 0 % | 5.97 % | **0.00 %** | 修复前 5.97 pp；修复后 0 | 修复后 ✅ 是 |
| M=100 光滑波问题单元比例（N=40→320） | 0 %，且不随 N 增长 | 0 % | **0 %** | 0 % | ✅ 是 |
| M=0 光滑极值检出比例 | 50 %（解析：极值两侧异号） | 50 % | **50 %** | 0 % | ✅ 是 |

结论：所有可量化关键量在修复后相对（解析/历史基准）误差均为 **0.00 %，全部 <3%**。
唯一"不达标"项是修复前的 float32 误判（6.03 %，超过 3 % 门槛）——这正是本单修掉的缺陷。

## 5. 注意点 / 局限

1. **仅测试核心函数**：`troubled_from_differences` 是纯张量运算，不涉及求解器循环。
   完整的限制器行为（含 SSP-RK 子步应用、正性过滤等）由现有测试套件覆盖。
2. **未测试 2D 方向分裂**：`tvb.py` 的函数是 1D 原语，2D 限制器在 `euler_dg_2d.py`
   中逐轴调用。2D 行为由 `test_limiter_2d.py` 覆盖（本单改动已跑 2D 限制器测试回归）。
3. **轻量级**：所有测试在 < 10 秒内完成，适合高负载环境。
4. **舍入底看不到 |u| 本身（先于本单即存在的局限，dtype 无关）**：
   floor 只用 `max(|d_left|, |d_right|)` 并 `clamp_min(1.0)`，因此它隐含假设
   单元平均值 `|u| = O(1)`。对"大 offset + 极小梯度"的数据（例如
   `u = 1e6 + 1e-3·x`，float64），差分量级 ~1e-4 而相减的舍入 ~eps·|u| ≈ 2e-10，
   仍会被低估的 floor 漏掉：实测 float64 下 M=0 有 32.7 % 单元被误判。
   该局限是**修复前就存在**的，本单的 dtype 自适应只解决精度依赖部分。
   由此界定的安全区为 `max|u| ≲ 1e4 · max|d|`，由
   `test_roundoff_floor_limitation_large_offset_small_gradient` 钉住：
   若将来彻底修好该 floor，这条刻画测试会失败并强制更新本节。
5. **未覆盖 bfloat16/float16**：这两种 dtype 的 eps（7.8e-3 / 9.8e-4）会使
   `8·eps` 系数大到可能吞掉真实特征；TensorDG 当前求解器路径均为 float32/float64，
   低精度路径如要启用需单独标定系数。

## 6. 交付物

| 文件 | 作用 |
|---|---|
| `tests/test_tvb.py` | 25 条直接单元测试（18 条 m3666 + 7 条 m4541） |
| `tensordg/tvb.py` | dtype 自适应舍入底（3 行 + 注释），float64 逐位不变 |
| `artifacts/m4541/tvb_roundoff_probe.py` | 复现 §3 全部数字与图表的探针脚本 |
| `artifacts/m4541/tvb_roundoff_probe.json` | 程序化生成的原始数值 |
| `artifacts/m4541/tvb_roundoff_floor.png` | 图：舍入底 dtype 对比 + 实际限制器误削比例 |

## 7. 复现命令

```bash
PYTHONPATH=. python -m pytest tests/test_tvb.py -v
PYTHONPATH=. python -m pytest tests/test_limiter_units.py tests/test_limiter_validation.py \
    tests/test_limiters.py tests/test_limiter_kernel_budget.py -v
PYTHONPATH=. python artifacts/m4541/tvb_roundoff_probe.py     # 生成 §3 数字与图
```
## 8. 本单验证执行记录（m4541）

| 套件 | 结果 | 覆盖 |
|---|---|---|
| `tests/test_tvb.py` | **25 passed** | 被改函数本身（含 7 条新增） |
| `tests/test_limiter_units.py` + `test_limiter_policy.py` + `test_limiter_validation.py` | **15 passed** | 1D 限制器语义/策略/验证 |
| `tests/test_limiters.py` + `test_limiter_kernel_budget.py` | **22 passed** | 1D 限制器 + 与 `tests/_limiter_reference.py` 的逐位等价 |
| `tests/test_euler_dg_2d_limiter.py` + `test_limiter_2d.py` + `test_limiter_2d_order.py` + `test_limiter_design_order_2d.py` | **41 passed**（851 s） | 2D 逐轴限制与设计精度阶 |
| 合计 | **103 passed, 0 failed** | tvb.py 的全部调用方（`limiters.py`、`euler_dg_2d.py`） |

**未完成项（如实记录）**：全仓 `pytest tests/ -q` 在共享 CPU 负载下跑到 76 % 后被外部因素截断
（后台日志文件被清空、无法取证），因此本单**只对上述 103 条定向测试做过验收**，不宣称全仓全绿。
由于 `troubled_from_differences` 的调用方只有 `limiters.py` 与 `euler_dg_2d.py`（已全覆盖），
剩余套件不受本单改动影响。

**回退实验（测试有效性）**：`git checkout tensordg/tvb.py` 还原修改前实现后，
`tests/test_tvb.py` 变为 `2 failed, 23 passed`，失败项正是
`test_float32_affine_data_not_troubled` 与 `test_float32_and_float64_agree_on_affine_data`。
