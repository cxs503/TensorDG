# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 验证报告

> **Feature slug**: `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`
> **Domain**: TensorDG · 间断伽辽金（DG）求解器验证
> **算例**: Shu–Osher 激波-熵波相互作用（JCP 77 (1988) 439）
> **报告生成**: 2026-11-14 · 数字全部从工件直读，零手抄
> **验收判据**: 对标误差 < 15%（owner 精度标准：>15% 不可接受）

---

## ① 案例介绍（物理背景/几何/参数表）

### 物理背景
Shu–Osher 算例是 CFD 验证的经典基准：一道 Mach-3 激波与一道正弦熵波相互作用，产生复杂的波系结构（透射激波、接触间断、被压缩的熵波）。该算例**无闭式解析解**，因此验证必须依赖「认证参考解」（certified reference）——通过独立离散化 + 网格收敛指数（GCI）量化的参考剖面。

### 几何与参数
| 量 | 值 |
|---|---|
| 比热比 γ | 1.4 |
| 计算域 | x ∈ [−5, 5] |
| 终止时刻 | t = 1.8 |
| 左状态（x < −4） | ρ = 3.857143, u = 2.629369, p = 10.33333（Mach-3 激波后） |
| 右状态（x ≥ −4） | ρ = 1 + 0.2 sin(5x), u = 0, p = 1 |
| 边界条件 | 透射（transmissive） |

### 认证参考解的构建
- **求解器**: 独立 FV-WENO5/HLLC/SSP-RK3（与 DG 求解器完全不同的离散化）
- **网格序列**: N = 800, 1600, 3200（加密比 r = 2）
- **外推方法**: Richardson 外推 → 认证剖面（存储在第二细网格 N=1600 上）
- **不确定度**: Roache GCI（Fs = 1.25）→ U = 0.082797%
- **实测阶**: p = 1.2825（形式 5 阶因激波退化至 ~1 阶）

---

## ② 方法（怎么做的）

### 四重决策门（预注册，运行前固定）

| 决策 | 内容 | 阈值 | 判定 |
|---|---|---|---|
| **D1** | 参考解可用性 | CERTIFIED 且 U/τ ≤ 1/3 | PASS |
| **D2** | 每条 DG 腿误差 | rel L1 < 3%（owner 标准） | PASS |
| **D3** | 参考解 vs 闭式源 | 每项 < 1% | PASS |
| **D4** | 无自认证 | 全部 DG 族 WITHHELD | PASS |

### 工具链
- **核心引擎**: `tensordg/certified_reference.py`（618 行，case-independent）
- **基准驱动**: `benchmarks/xval_dg_round2_certified.py`（622 行）
- **平台集成**: `validation/xval_dg_round2/post_platform.py`
- **测试套件**: `tests/test_xval_dg_round2_integration.py`（11 测试）+ `tests/test_xval_oss_shu_osher.py`（5 测试 + 2 跳过）

### 闭式验证源（D3）
认证参考解与以下**闭式/解析源**交叉验证（不共享离散化）：
1. **Hugoniot 激波后平台密度**: ρ_L = 3.857143（Mach-3 跳跃条件）
2. **Hugoniot 激波位置**: x_L + W·t（W 由跳跃条件计算）
3. **全局质量守恒**: M(0) + ρ_L·u_L·t（精确质量预算）
4. **未扰动熵波**: 1 + 0.2 sin(5x)（x ≥ 2.6 区域，解析已知）

---

## ③ 计算结果（数字表，程序化注入零手抄）

### D1 — 参考解可用性

| 量 | 值 | 判定 |
|---|---|---|
| 分类 | monotonic | — |
| 实测阶 p | 1.2825 | — |
| GCI 不确定度 U | 0.082797% | — |
| U/τ（τ = 5%） | 0.0166 ≤ 0.3333 | **PASS** |
| 判决 | CERTIFIED | **PASS** |

### D2 — DG 腿对认证参考解的误差

| 配置 | rel L1 % | rel L2 % | err+U % | 3% 门 | 5% 门 |
|---|---|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.455701 | 5.262588 | 2.538498 | PASS | PASS |
| DG P1 N=400 CFL=0.4 | 1.547928 | 3.656721 | 1.630725 | PASS | PASS |
| DG P1 N=800 CFL=0.4 | 0.578643 | 1.517108 | 0.661440 | PASS | PASS |
| DG P2 N=200 CFL=0.4 | 2.420304 | 5.232715 | 2.503101 | PASS | PASS |
| DG P2 N=400 CFL=0.4 | 1.516026 | 3.609797 | 1.598823 | PASS | PASS |
| DG P2 N=800 CFL=0.4 | 0.566719 | 1.481060 | 0.649516 | PASS | PASS |
| DG P3 N=200 CFL=0.4 | 2.426617 | 5.252309 | 2.509415 | PASS | PASS |
| DG P3 N=400 CFL=0.4 | 1.522033 | 3.606569 | 1.604830 | PASS | PASS |
| DG P3 N=800 CFL=0.4 | 0.669867 | 1.490799 | 0.752664 | PASS | PASS |
| DG P2 N=400 CFL=0.1 | 1.519713 | 3.614589 | 1.602510 | PASS | PASS |
| DG P2 N=400 CFL=0.2 | 1.518092 | 3.611538 | 1.600889 | PASS | PASS |

**最差腿**: DG P1 N=200 CFL=0.4 → 2.455701%（< 3%，余量 18.1%）
**最佳腿**: DG P2 N=800 CFL=0.4 → 0.566719%
**含不确定度的最差值**: 2.538498%（< 3%，余量 15.4%）

### D3 — 参考解 vs 闭式源

| 量 | 类型 | 参考解值 | 闭式值 | 误差 % | < 1% ? |
|---|---|---|---|---|---|
| plateau_rho | relative | 3.857143 | 3.857143 | 1.225e-10 | **yes** |
| shock_position | relative | 2.403125 | 2.389367 | 0.5758 | **yes** |
| mass_integral | relative | 31.089152 | 31.089152 | 8.350e-07 | **yes** |
| preshock_sine_linf | bounded | 0.000864 | bound | 8.637e-04 | **yes** |

### D4 — 无自认证

| DG 族 | 判决 | 要求 |
|---|---|---|
| dg_p1 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |
| dg_p2 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |
| dg_p3 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |

**综合判定**: accepted = **True**（D1 ✓, D2 ✓, D3 ✓, D4 ✓）

---

## ④ 结果图

### 图 1: 认证参考解与 DG 腿剖面比较

![认证参考解与 DG 腿](docs/fig_xval_round2_profiles.png)

上图：认证参考解（黑色实线）及其 ±U 不确定度带（灰色阴影），叠加 DG P2 N=800 腿（蓝色虚线）。
下图：各腿与参考解的偏差，激波区域误差最大（限制器抹平效应）。

### 图 2: 闭式源偏差与腿误差

![闭式源偏差与腿误差](docs/fig_xval_round2_oracle.png)

左：参考解与四个闭式源的偏差（均 < 1%）。
右：11 条 DG 腿的 rel L1 误差，全部低于 3% 门线。

---

## ⑤ 注意点（踩过的坑/局限/适用范围）

### 局限
1. **无文献表格值**: Shu & Osher (1988) 仅给出图形，无数值表 → 无法做文献对标
2. **三网格 GCI**: 仅三级网格（800/1600/3200），无第四级冗余验证
3. **Fs = 1.25 是约定**: Roache 安全因子，非数据推导 → U 是约定 bound 而非置信区间
4. **实测阶非形式阶**: WENO5 形式 5 阶，实测 p = 1.2825（激波导致退化）
5. **DG 族拒绝仅针对此算例**: 在光滑解上同一引擎可在设计阶认证

### 适用范围
- 本验证是**代码间交叉验证**（two independent discretisations），≠ 实验 validation
- 适用于所有无闭式解的 CFD 验证算例：替换 FV 求解器 + 网格序列 + DG 腿配置即可
- 认证参考解方法论可迁移至其他激波/接触间断类算例

### 踩过的坑
- **外推位移陷阱**: 认证剖面存储在第二细网格上，其与原始层的距离 ≠ U（是 r^p/(r^p-1) 倍）
- **自认证风险**: DG 族自身不能认证自己（p < 0.5，U/τ > 2），必须用独立 FV  ladder
- **有界度量编码**: preshock_sine 的参考值为 0，不能编码为 |got-ref|/|ref|（会除零）

---

## ⑥ 误差判定（关键量 vs 基准，逐项列出并显式回答「是否 <3%」）

| 关键量 | 值 | 基准 | 误差 | < 3% ? |
|---|---|---|---|---|
| 认证参考解 GCI U | 0.082797% | 历史驱动 0.082797% | 0（逐位相同） | **yes** |
| 实测阶 p | 1.282509 | 历史驱动 1.282509 | 0（逐位相同） | **yes** |
| 认证剖面 vs 历史 ρ_certified | — | — | max abs diff = 0.0 | **yes** |
| 最差 DG 腿误差 | 2.455701% | 认证参考解 | 2.455701% | **yes**（余量 18.1%） |
| 最差腿含不确定度 | 2.538498% | 认证参考解 | 2.538498% | **yes**（余量 15.4%） |
| 平台密度误差 | 1.225e-10% | Hugoniot 闭式 | 1.225e-10% | **yes** |
| 激波位置误差 | 0.5758% | Hugoniot 闭式 | 0.5758% | **yes** |
| 质量积分误差 | 8.350e-07% | 精确质量预算 | 8.350e-07% | **yes** |
| 未扰动熵波偏差 | 0.000864% | 解析正弦波 | 0.000864% | **yes** |

**综合结论**: 所有关键量对标误差均 **< 3%**，远低于 15% 的 owner 精度标准。

---

## 测试验证

```
$ PYTHONPATH=. python -m pytest tests/test_xval_dg_round2_integration.py -v
======================== 11 passed in 5.42s =========================

$ PYTHONPATH=. python -m pytest tests/test_xval_oss_shu_osher.py -v
=================== 5 passed, 2 skipped in 0.18s ===================
```

- **T1**: CLI 写工件 + 四门全过 → PASS
- **T2**: 参考解可用 + 全部腿 < 3% → PASS
- **T3**: 参考解与闭式源一致 → PASS
- **T4**: 无 DG 族自认证 → PASS
- **T5**: 腿误差可独立复现（不依赖引擎） → PASS
- **T6**: 平台文档通过 TensorBench 契约验证 → PASS
- **T7**: 工件确定性（两次运行字节相同） → PASS

---

## 证据指针

- **Commit**: `c6ca0cc`（main，2026-11-14）
- **审计工件**: `validation/xval_dg_round2/artifacts/xval_round2_audit.json`
- **平台集成**: `validation/xval_dg_round2/post_platform.py`
- **核心引擎**: `tensordg/certified_reference.py`（618 行）
- **基准驱动**: `benchmarks/xval_dg_round2_certified.py`（622 行）
- **测试**: `tests/test_xval_dg_round2_integration.py`（11 测试）+ `tests/test_xval_oss_shu_osher.py`（5 测试）
- **OSS 交叉验证**: `benchmarks/xval_oss_shu_osher.py`
- **原始数据**: `validation/repro_shu_osher/artifacts/raw/`（14 个 npz 文件）
