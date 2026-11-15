# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 验证图补齐报告

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）
> 分支：`agent/m22007`
> 日期：2026-11-15
> Commit：`13c6ace`

## 1. 案例介绍（物理背景/几何/参数表）

**Shu-Osher 激波/熵波相互作用**（Shu & Osher 1988, JCP 73:52）是 DG 求解器验证的
经典基准：一维 Euler 方程（γ=1.4），计算域 x∈[-5,5]，终态 t=1.8。

| 参数 | 值 |
|---|---|
| 左状态（激波后） | ρ=3.857143, u=2.629369, p=10.33333（Mach-3 Hugoniot） |
| 右状态（熵波） | ρ=1+0.2sin(5x), u=0, p=1 |
| 边界 | 透射（transmissive） |
| 参考求解器 | FV-WENO5, CFL=0.3, SSP-RK3 |
| 参考网格梯次 | [800, 1600, 3200]（三级） |
| DG 腿 | 11 组（P1/P2/P3 × N=200/400/800, CFL=0.4；P2 N=400, CFL=0.1/0.2） |
| 通量 | Rusanov |
| 限制器 | WENO（DG 腿） |

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 2.1 核心引擎

- `tensordg/certified_reference.py`（618 行）：Roache/GCI 三级网格收敛分析 + Richardson 外推
  - 观测阶 p = log(d₁/d₂)/log(r)，r=2
  - Richardson 外推带 U = Fs·d_fine/(r^p - 1)，Fs=1.25
  - 拒绝规则：梯次不足(<3级)、非单调收敛、U/gate > 1/3

- `benchmarks/xval_dg_round2_certified.py`（622 行）：D1-D4 交叉验证
  - **D1**: 参考可用（CERTIFIED + U/gate ≤ 1/3）
  - **D2**: 所有 11 条 DG 腿过严格门（rel L1 < 3%）
  - **D3**: 参考与闭形式 Oracle 比较 < 1%
  - **D4**: 无 DG 家族可自认证（全部 WITHHELD_ORDER_TOO_LOW）

- `benchmarks/xval_dg_round2_multifield.py`：MF1-MF6 多场扩展
  - MF1: ρ/ρu/ρE 三场独立认证
  - MF2: 导出量 u/p 的带覆盖（加法规则）
  - MF3: 55 条腿-场比较（11 腿 × 5 场）全部 < 3%
  - MF4: 状态 Oracle（plateau ρ/u/p + 熵波区 u=0, p=1）
  - MF5: 质量/动量/能量预算闭合 < 0.01%
  - MF6: 密度门对其他量的代理报告

### 2.2 工具链

```
gen_fv_level.py → FV-WENO5 求解 → raw/*.npz
xval_dg_round2_certified.py --figures → audit.json + audit.md + fig1/fig2
xval_dg_round2_multifield --figures → multifield_audit.json + fig_mf1/fig_mf2
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

### 3.4 多场认证（MF1-MF5）

| 决策 | 结果 |
|---|---|
| MF1 五场认证 | ρ/ρu/ρE 全部 CERTIFIED, U/gate ≤ 0.028 |
| MF2 导出量带 | u/p 带可用（加法规则） |
| MF3 55 比较 | 全部 < 3% 门，最坏 ρu = 2.490% |
| MF4 状态 Oracle | plateau ρ/u/p 全 < 1%，熵波区 u=0, p=1 |
| MF5 预算闭合 | mass/momentum/energy 全部 < 0.01% |

## 4. 结果图

### 4.1 认证参考与 DG 腿比较

![fig1_xval_profiles](../validation/xval_dg_round2/artifacts/fig1_xval_profiles.png)

*图 1：Shu-Osher t=1.8 认证参考（N=1600）与 DG P2 N=800 腿的密度剖面对比。*

### 4.2 每腿误差与门限裕度

![fig2_leg_oracle_deviation](../validation/xval_dg_round2/artifacts/fig2_leg_oracle_deviation.png)

*图 2：11 条 DG 腿的相对 L1 误差（柱状）与 3%/5% 门限（虚线），以及 plateau ρ_L 和激波位置对闭形式 Oracle 的偏差（对数坐标）。*

### 4.3 多场腿误差

![fig_mf1_leg_errors_by_field](../validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png)

*图 3：DG legs on all five quantities (ρ, ρu, ρE, u, p) with 3% gate.*

### 4.4 导出量剖面

![fig_mf2_derived_profiles](../validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png)

*图 4：Fields the density-only round-2 gate never looked at (u and p, each certified on its own FV ladder).*

## 5. 注意点（踩过的坑/局限/适用范围）

1. **缺失图修复**：`fig1_xval_profiles.png`、`fig2_leg_oracle_deviation.png`、
   `fig_mf1_leg_errors_by_field.png`、`fig_mf2_derived_profiles.png` 此前未落盘，
   导致 `test_artifact_is_not_stale` 失败。运行
   `python -m benchmarks.xval_dg_round2_certified --figures` 和
   `python -m benchmarks.xval_dg_round2_multifield --figures` 后补齐。
2. **系统负载**：本机 load15 ≈ 72，重计算必须走集群调度，不可本地强跑。
3. **Oracle 来源**：所有闭形式 Oracle 来自 `tensordg.cases.shock_entropy`（案例模块，非引擎），
   确保引擎缺陷不会悄悄移动 Oracle。
4. **适用范围**：本验证针对一维 Euler 激波/熵波问题，不声明多维激波捕获的生产就绪性。

## 6. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3%? |
|---|---|---|---|---|
| 最坏 DG 腿 rel L1 | 3% 门 | 2.455701% | — | **yes** |
| plateau_rho vs Oracle | 3.857143 | 3.857143 | 1.225e-10% | **yes** |
| shock_position vs Oracle | 2.389367 | 2.403125 | 0.5758% | **yes** |
| mass_integral vs Oracle | 31.08915219 | 31.08915245 | 8.350e-07% | **yes** |
| 参考带 U/gate | 0.3333 | 0.0166 | — | **yes** |

**所有关键量误差均 < 3%，验证通过。**

## 7. 测试状态

| 测试文件 | 通过 | 跳过 | 失败 |
|---|---|---|---|
| test_xval_dg_round2_evidence.py | 13 | 0 | 0 |
| test_xval_dg_round2_integration.py | 10 | 0 | 0 |
| test_xval_dg_round2_adversarial.py | 9 | 0 | 0 |
| test_xval_dg_round2_ladder_extension.py | 7 | 0 | 0 |
| test_xval_dg_round2_band_honesty.py | 14 | 0 | 0 |
| test_xval_dg_round2_band_honesty_synthetic.py | 14 | 0 | 0 |
| test_xval_dg_round2_multifield.py | 21 | 0 | 0 |

**总计：88 PASS, 0 SKIPPED, 0 FAIL**

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `validation/xval_dg_round2/artifacts/fig1_xval_profiles.png` | 参考/DG 剖面图（本单补齐） |
| `validation/xval_dg_round2/artifacts/fig2_leg_oracle_deviation.png` | 腿误差/Oracle 偏差图（本单补齐） |
| `validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png` | 多场腿误差图（本单补齐） |
| `validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png` | 导出量剖面图（本单补齐） |
| `validation/xval_dg_round2/artifacts/xval_round2_audit.json` | 交叉验证审计 JSON |
| `validation/xval_dg_round2/artifacts/xval_round2_audit.md` | 交叉验证审计 Markdown |
| `validation/xval_dg_round2/artifacts/multifield_audit.json` | 多场审计 JSON |
| `validation/xval_dg_round2/artifacts/multifield_audit.md` | 多场审计 Markdown |
