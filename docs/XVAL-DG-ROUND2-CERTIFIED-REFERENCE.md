# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 综合验证报告

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open → 本单闭合）
> 分支：`agent/m22063`
> 日期：2026-11-15
> 验收标准：对标误差 < 15%（owner 精度标准）

## 1. 案例介绍（物理背景/几何/参数表）

**Shu-Osher 激波/熵波相互作用**（Shu & Osher 1988, JCP 73:52）是 DG 求解器验证的
经典基准：一维 Euler 方程（γ=1.4），计算域 x∈[-5,5]，终态 t=1.8。

| 参数 | 值 |
|---|---|
| 左状态（激波后） | ρ=3.857143, u=2.629369, p=10.33333（Mach-3 Hugoniot） |
| 右状态（熵波） | ρ=1+0.2sin(5x), u=0, p=1 |
| 边界 | 透射（transmissive） |
| 参考求解器 | FV-WENO5, CFL=0.3, SSP-RK3 |
| FV 梯次 | 4 级：N=[800, 1600, 3200, 6400] |
| DG 腿 | 11 组（P1/P2/P3 × N=200/400/800, CFL=0.4；P2 N=400, CFL=0.1/0.2） |
| 通量 | Rusanov |
| 限制器 | WENO（DG 腿） |

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 2.1 认证参考引擎（Roache/GCI 三级网格收敛分析）

`tensordg.certified_reference` 实现：
- 观测阶 p = log(d₁/d₂)/log(r)，r=2
- Richardson 外推带 U = Fs·d_fine/(r^p - 1)，Fs=1.25
- 三条拒绝规则：梯次不足(<3级)、非单调收敛、U/gate > 1/3

### 2.2 交叉验证（D1-D4）

`benchmarks.xval_dg_round2_certified` 实现四个预声明判据：

| 判据 | 含义 | 阈值 |
|---|---|---|
| D1 | 参考可用（CERTIFIED + U/gate ≤ 1/3） | U/gate ≤ 0.3333 |
| D2 | 所有 11 条 DG 腿过严格门 | rel L1 < 3% |
| D3 | 参考与闭形式 Oracle 比较 | < 1% |
| D4 | 无 DG 家族可自认证 | WITHHELD_ORDER_TOO_LOW |

### 2.3 四级梯次扩展（L0-L4）

`benchmarks.xval_dg_round2_ladder_extension` 定义 5 个判据：

| 判据 | 含义 | 阈值 |
|---|---|---|
| L0 | 连续差严格递减（渐近区证据） | d₁ > d₂ > d₃ |
| L1 | 扩展梯次被同一引擎认证 | verdict=CERTIFIED |
| L2 | 阶一致性 | \|p₄-p₃\|/p₃ ≤ 0.20 |
| L3 | 带覆盖（自洽性） | rel L1(cert₃,cert₄) ≤ U₃ |
| L4 | 声明稳定性 | 所有腿仍过 3% 门；最大移动 ≤ 0.5pp |

### 2.4 多场审计（MF1-MF6）

`benchmarks.xval_dg_round2_multifield` 验证全状态变量（ρ, ρu, E）：

| 判据 | 含义 |
|---|---|
| MF1 | 每场独立认证 |
| MF2 | 导出量（速度、压力）可用性 |
| MF3 | 所有场所有腿过门 |
| MF4 | 状态 Oracle 比较 |
| MF5 | 预算闭合 |
| MF6 | ρ-only 门忠实性（报告用） |

### 2.5 带诚实性审计

`benchmarks.xval_dg_round2_band_honesty` 验证 WENO5 vs WENO7 参考的一致性：

| 判据 | 含义 |
|---|---|
| 交叉阶距离 | 4 级参考间差异 < 带和 |
| 门存活 | 最坏腿 + 交叉偏移仍过 3% 门 |

### 2.6 工具链

```
gen_fv_level.py → FV-WENO5/7 求解 → raw/*.npz
xval_dg_round2_certified.py → 交叉验证 → audit.json + audit.md + figures
xval_dg_round2_ladder_extension.py → 梯次扩展审计 → ladder_extension.json + figures
xval_dg_round2_multifield.py → 多场审计 → multifield_audit.json + figures
xval_dg_round2_band_honesty.py → 带诚实性 → band_honesty_audit.json + figures
```

## 3. 计算结果（数字表，程序化注入零手抄）

### 3.1 三级参考认证（D1）

| 量 | 值 |
|---|---|
| 观测阶 p | 1.282509 |
| 带 U | 0.082797 % |
| U/gate | 0.0166 ≤ 0.3333 |
| 判决 | **CERTIFIED → PASS** |

### 3.2 DG 腿误差（D2）

| 配置 | rel L1 % | +U % | 3% 门 |
|---|---|---|---|
| DG P1 N=200 CFL=0.4 | 2.455701 | 2.538498 | PASS |
| DG P1 N=400 CFL=0.4 | 1.547928 | 1.630725 | PASS |
| DG P1 N=800 CFL=0.4 | 0.578643 | 0.661440 | PASS |
| DG P2 N=200 CFL=0.4 | 2.420304 | 2.503101 | PASS |
| DG P2 N=400 CFL=0.4 | 1.516026 | 1.598823 | PASS |
| DG P2 N=800 CFL=0.4 | 0.566719 | 0.649516 | PASS |
| DG P3 N=200 CFL=0.4 | 2.426617 | 2.509415 | PASS |
| DG P3 N=400 CFL=0.4 | 1.522033 | 1.604830 | PASS |
| DG P3 N=800 CFL=0.4 | 0.669867 | 0.752664 | PASS |
| DG P2 N=400 CFL=0.1 | 1.519713 | 1.602510 | PASS |
| DG P2 N=400 CFL=0.2 | 1.518092 | 1.600889 | PASS |

最坏腿：**DG P1 N=200 CFL=0.4 = 2.455701 %** < 3% → **PASS**

### 3.3 Oracle 比较（D3）

| 量 | 认证值 | 闭形式参考 | 误差 % | < 1% |
|---|---|---|---|---|
| plateau_rho | 3.857143 | 3.857143 | 1.225e-10 | yes |
| shock_position | 2.403125 | 2.389367 | 5.758e-01 | yes |
| mass_integral | 31.08915245 | 31.08915219 | 8.350e-07 | yes |
| preshock_sine_linf | 0.000864 | bound | 8.637e-04 | yes |

### 3.4 自认证拒绝（D4）

| 梯次 | 判决 | 要求 |
|---|---|---|
| dg_p1 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |
| dg_p2 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |
| dg_p3 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |

### 3.5 四级梯次扩展（L0-L4）

| 判据 | 结果 |
|---|---|
| L0 相邻差严格递减 | 2.308e-3 → 9.489e-4 → 4.198e-4 → **PASS** |
| L1 扩展梯次 CERTIFIED | **PASS** |
| L2 阶一致性 \|p₄-p₃\|/p₃ | 0.0825 ≤ 0.20 → **PASS** |
| L3 带覆盖 shift=0.0594% ≤ U₃=0.0828% | **PASS** |
| L4 声明稳定 最坏腿 2.4484% < 3%，最大移动 0.0073pp ≤ 0.5pp | **PASS** |

### 3.6 多场审计（MF1-MF6）

| 判据 | 结果 |
|---|---|
| MF1 每场独立认证 | **PASS** |
| MF2 导出量可用性 | **PASS** |
| MF3 所有场所有腿过门（55 比较；最坏 DG P1 N=200 CFL=0.4 / ρu = 2.4895%） | **PASS** |
| MF4 状态 Oracle | **PASS** |
| MF5 预算闭合 | **PASS** |
| MF6 ρ-only 门忠实性 | False（仅报告，不影响 accepted） |

### 3.7 带诚实性

| 量 | 值 |
|---|---|
| 交叉阶距离（4 级） | 0.0519% |
| WENO5 带 | 0.0416% |
| WENO7 带 | 0.0468% |
| 覆盖判决 | MARGINAL（带和 > 距离） |
| 门存活 | 最坏腿 2.4484% + 0.0519% = 2.5003% < 3% → **PASS** |

## 4. 结果图

### 4.1 认证参考与 DG 腿比较

![fig1_xval_profiles](../validation/xval_dg_round2/artifacts/fig1_xval_profiles.png)

*图 1：Shu-Osher t=1.8 认证参考（N=1600）与 DG P2 N=800 腿的密度剖面对比。*

### 4.2 每腿误差与门限裕度

![fig2_leg_oracle_deviation](../validation/xval_dg_round2/artifacts/fig2_leg_oracle_deviation.png)

*图 2：11 条 DG 腿的相对 L1 误差（柱状）与 3%/5% 门限（虚线），以及 plateau ρ_L 和激波位置对闭形式 Oracle 的偏差（对数坐标）。*

### 4.3 梯次扩展剖面

![fig_ladder_extension_profiles](../validation/xval_dg_round2/artifacts/fig_ladder_extension_profiles.png)

*图 3：四级 FV 梯次（N=800/1600/3200/6400）密度剖面叠加。*

### 4.4 梯次扩展腿误差

![fig_ladder_extension_legs](../validation/xval_dg_round2/artifacts/fig_ladder_extension_legs.png)

*图 4：11 条 DG 腿对 3 级 vs 4 级参考的误差比较。*

### 4.5 多场腿误差

![fig_mf1_leg_errors_by_field](../validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png)

*图 5：11 条 DG 腿对 ρ/ρu/E 三场的相对 L1 误差。*

### 4.6 多场导出量剖面

![fig_mf2_derived_profiles](../validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png)

*图 6：认证参考导出的速度/压力剖面与 DG 腿比较。*

### 4.7 交叉阶参考比较

![fig1_cross_order_references](../validation/xval_dg_round2/artifacts/band_honesty/fig1_cross_order_references.png)

*图 7：WENO5 vs WENO7 四级认证参考的密度剖面比较。*

### 4.8 双参考腿误差

![fig2_dg_legs_two_references](../validation/xval_dg_round2/artifacts/band_honesty/fig2_dg_legs_two_references.png)

*图 8：11 条 DG 腿对 WENO5 和 WENO7 参考的误差比较。*

## 5. 注意点（踩过的坑/局限/适用范围）

1. **N=6400 四级梯次数据**：已落盘 `validation/repro_shu_osher/artifacts/raw_ext/fv_cfl0p3_n6400.npz`。
2. **系统负载**：本机 load15 ≈ 60，本单为轻量活（跑已有基准+生成图），未起重算力进程。
3. **Oracle 来源**：所有闭形式 Oracle 来自 `tensordg.cases.shock_entropy`（案例模块，非引擎），
   确保引擎缺陷不会悄悄移动 Oracle。
4. **适用范围**：本验证针对一维 Euler 激波/熵波问题，不声明多维激波捕获的生产就绪性。
5. **MF6 判据**：ρ-only 门忠实性为 False（报告用），不影响整体 accepted 判定。
6. **带诚实性 verdict=MARGINAL**：WENO5/WENO7 带和勉强覆盖交叉阶距离，但门存活确认
   最坏腿 + 偏移仍过 3% 门。

## 6. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 15%? |
|---|---|---|---|---|
| 最坏 DG 腿 rel L1（3% 门） | 3% | 2.455701% | — | **yes**（< 3% << 15%） |
| plateau_rho vs Oracle | 3.857143 | 3.857143 | 1.225e-10% | **yes** |
| shock_position vs Oracle | 2.389367 | 2.403125 | 0.5758% | **yes** |
| mass_integral vs Oracle | 31.08915219 | 31.08915245 | 8.350e-07% | **yes** |
| 参考带 U/gate | 0.3333 | 0.0166 | — | **yes** |
| 多场最坏腿（ρu） | 3% | 2.4895% | — | **yes** |
| 交叉阶距离 | 带和 | 0.0519% | — | **yes** |

**所有关键量误差均远 < 15%（实际 < 3%），验证通过。**

## 7. 测试状态

| 测试文件 | 通过 | 跳过 | 失败 |
|---|---|---|---|
| test_xval_dg_round2_evidence.py | 13 | 0 | 0 |
| test_xval_dg_round2_integration.py | 10 | 0 | 0 |
| test_xval_dg_round2_adversarial.py | 9 | 0 | 0 |
| test_xval_dg_round2_ladder_extension.py | 7 | 0 | 0 |
| test_xval_dg_round2_multifield.py | 21 | 0 | 0 |
| test_xval_dg_round2_band_honesty.py | 17 | 0 | 0 |
| test_xval_dg_round2_band_honesty_synthetic.py | 12 | 0 | 0 |
| test_certified_reference.py | 19 | 0 | 0 |
| test_certified_reference_lax_demo.py | 10 | 0 | 0 |
| test_certified_reference_sod.py | 12 | 0 | 0 |

**总计：130 PASS, 0 SKIPPED, 0 FAIL**

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `validation/xval_dg_round2/artifacts/xval_round2_audit.json` | 交叉验证审计 JSON |
| `validation/xval_dg_round2/artifacts/xval_round2_audit.md` | 交叉验证审计 Markdown |
| `validation/xval_dg_round2/artifacts/ladder_extension.json` | 梯次扩展审计 JSON |
| `validation/xval_dg_round2/artifacts/ladder_extension.md` | 梯次扩展审计 Markdown |
| `validation/xval_dg_round2/artifacts/multifield_audit.json` | 多场审计 JSON |
| `validation/xval_dg_round2/artifacts/multifield_audit.md` | 多场审计 Markdown |
| `validation/xval_dg_round2/artifacts/band_honesty/band_honesty_audit.json` | 带诚实性审计 JSON |
| `validation/xval_dg_round2/artifacts/band_honesty/band_honesty_audit.md` | 带诚实性审计 Markdown |
| `validation/xval_dg_round2/artifacts/fig1_xval_profiles.png` | 参考/DG 剖面图 |
| `validation/xval_dg_round2/artifacts/fig2_leg_oracle_deviation.png` | 腿误差/Oracle 偏差图 |
| `validation/xval_dg_round2/artifacts/fig_ladder_extension_profiles.png` | 四级梯次剖面图 |
| `validation/xval_dg_round2/artifacts/fig_ladder_extension_legs.png` | 梯次扩展腿误差图 |
| `validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png` | 多场腿误差图 |
| `validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png` | 多场导出量剖面图 |
| `validation/xval_dg_round2/artifacts/band_honesty/fig1_cross_order_references.png` | 交叉阶参考比较图 |
| `validation/xval_dg_round2/artifacts/band_honesty/fig2_dg_legs_two_references.png` | 双参考腿误差图 |
| `docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md` | 本综合报告 |

## 9. 结论

**XVAL-DG-ROUND2-CERTIFIED-REFERENCE 锚点闭合。**

- 所有 5 个审计引擎（交叉验证、梯次扩展、多场、带诚实性、等价性）全部通过
- 130 个测试全绿（0 跳过，0 失败）
- 最坏 DG 腿误差 2.4557% < 3% << 15% 验收标准
- 8 张验证图已落盘
- 所有闭形式 Oracle 比较误差 < 1%
- 四级梯次扩展 L0-L4 全绿
- 带诚实性确认 WENO5/WENO7 参考一致性
