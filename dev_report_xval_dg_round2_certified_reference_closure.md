# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 失效测试补齐（4 级梯次等价性）

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）
> 分支：`agent/m21949`
> 日期：2026-11-15
> 类型：defect_fix（验证欠账 / 失效测试补齐）

## 1. 问题合同与背景

`tensordg.certified_reference`（round-2 认证参考引擎）的等价性测试
`tests/test_certified_reference.py` 声明了一个不变式：**引擎必须逐位复现
"已发布"的 CASE-SHU-OSHER round-2 认证**（`test_engine_reproduces_the_shipped_round2_certification_bit_for_bit`
与 `test_judge_reproduces_every_shipped_dg_leg_error`）。

该不变式的夹具 `_fv_levels()` 硬编码了 **3 级** FV 梯次（N=800/1600/3200）。

## 2. 缺陷定位（独立重建执行路径）

本单扫描时发现 **3 个测试 FAIL**（此前全绿）：

| 测试 | 失败原因 |
|---|---|
| `test_engine_reproduces_the_shipped_round2_certification_bit_for_bit` | 引擎 3 级 p=1.2825 vs legacy 4 级 p=1.1767 |
| `test_judge_reproduces_every_shipped_dg_leg_error` | 同上（参考网格 N=1600 vs N=3200） |
| `test_certified_reference_lax_demo.py::test_synthetic_converges` | 广播错误：`coarsen(r200,2)`=(100,) vs `coarsen(exact,2)`=(200,) |

**根因**：`benchmarks.shu_osher_certified_ref.load_fv_levels()` 用
`raw.glob("fv_cfl0p3_n*.npz")` 自动拾取**全部**已落盘的 FV 级——N=6400 级由
round-2 梯次扩展闭合（m21318）加入后，legacy 的 `convergence_study` 自动升级为
**4 级**（p=1.1767，带 U=0.0416%，参考在 N=3200）。但测试夹具 `_fv_levels()`
仍硬编码 3 级，于是等价性测试把 **3 级 vs 4 级** 两个不同对象拿来逐位比较，
必然 FAIL。

这是典型的「验证欠账」：**已合并功能（4 级梯次扩展）对既有不变式测试的
复核缺口**——功能落地了，但断言「已发布参考」的测试没跟着更新到当前发布版。

## 3. 修复（最小、可验证）

### 3.1 `tests/test_certified_reference.py`

`_fv_levels()` 从 3 级（800/1600/3200）改为 **4 级**（800/1600/3200/6400），
并新增 `FV_LEVELS` 常量 + 文档注释说明「当前已发布参考是 4 级」。
等价性断言改为对 **4 级** 数字（p=1.1767348864087728，band=0.04162173159427086）。

验证（修复后）：

```
legacy 4级: p=1.1767348864087728, ref_ncells=3200, band=0.04162173159427086
engine 4级: p=1.1767348864087728, ref_ncells=3200, band=0.04162173159427086  ← 逐位一致
```

### 3.2 `tests/test_certified_reference_lax_demo.py`

`test_synthetic_converges` 的广播错误：原代码把 `r200`（200 格）与 `exact`
（400 格）各按因子 2 粗化，落到 **100 vs 200** 两个不同网格，形状不匹配。
修复为在**公共网格**上比较：`exact` 粗化到 200 格与 `r200` 比（e200），
`r400` 与 `exact` 在 400 格上比（e400）。

验证（修复后）：`e400=0.0273 < e200=0.0569`，「更细网格误差更小」成立。

## 4. 测试状态

| 测试文件 | 修复前 | 修复后 |
|---|---|---|
| test_certified_reference.py | 2 FAIL | **19 PASS** |
| test_certified_reference_lax_demo.py | 1 FAIL | **10 PASS** |
| test_certified_reference_sod.py | 12 PASS | 12 PASS |
| test_ladder_extension_independent.py | PASS | PASS |
| test_xval_dg_round2_*.py（4 文件） | PASS | PASS |

**锚点全量：84 PASS, 0 FAIL**（含本单修复的 3 个）。

## 5. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3%? |
|---|---|---|---|---|
| 引擎 vs legacy 观测阶（4 级） | 1.1767348864087728 | 1.1767348864087728 | 0（逐位） | **yes** |
| 引擎 vs legacy 带 U（4 级） | 0.04162173159427086 | 0.04162173159427086 | 0（逐位） | **yes** |
| 最坏 DG 腿 rel L1（4 级参考） | 3% 门 | 2.4484% | — | **yes** |

**所有关键量误差均 < 3%（本单为逐位一致，误差=0），验证通过。**

## 6. 注意点

1. **本单是「验证欠账」类**：不是新增功能，而是把已合并的 4 级梯次扩展
   对既有不变式测试的复核缺口补齐——功能落地后测试没跟着更新到当前发布版。
2. **3 级历史数字保留**：`PUBLISHED_BAND_PERCENT`/`PUBLISHED_ORDER`（3 级）
   仍保留在文件头作历史记录，但等价性断言已改对 4 级当前发布版。
3. **系统负载**：本机 load15≈111，本单为纯测试/文档轻量活，未起重算力进程。

## 7. 产物

| 文件 | 说明 |
|---|---|
| `tests/test_certified_reference.py` | 夹具升级到 4 级梯次（+文档注释） |
| `tests/test_certified_reference_lax_demo.py` | 修复合成收敛测试的网格对齐 |
