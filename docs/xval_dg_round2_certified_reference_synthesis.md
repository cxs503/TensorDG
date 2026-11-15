# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 综合验证报告

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）
> 分支：`agent/m21612`
> 日期：2026-11-15
> 状态：**主体完成，带宽诚实性审计待集群作业 cj3566 落盘**

---

## 1. 案例介绍（物理背景/几何/参数表）

**Shu-Osher 激波/熵波相互作用**（Shu & Osher 1988, JCP 77:439）是 DG 求解器验证的
经典基准：一维 Euler 方程（γ=1.4），计算域 x∈[-5,5]，终态 t=1.8。

| 参数 | 值 |
|---|---|
| 左状态（激波后） | ρ=3.857143, u=2.629369, p=10.33333（Mach-3 Hugoniot） |
| 右状态（熵波） | ρ=1+0.2sin(5x), u=0, p=1 |
| 边界 | 透射（transmissive） |
| 参考求解器 | FV-WENO5, CFL=0.3, SSP-RK3 |
| 参考网格梯次 | [800, 1600, 3200]（三级）→ 扩展至 [800, 1600, 3200, 6400]（四级） |
| DG 腿 | 11 组（P1/P2/P3 × N=200/400/800, CFL=0.4；P2 N=400, CFL=0.1/0.2） |
| 通量 | Rusanov（FV）/ HLLC（DG） |
| 限制器 | WENO（DG 腿） |

---

## 2. 方法体系（三层验证架构）

### 2.1 第一层：参考认证（certified_reference.py）

`tensordg/certified_reference.py` 实现 Roache/GCI 三级网格收敛分析：
- 观测阶 p = log(d₁/d₂)/log(r)，r=2
- Richardson 外推带 U = Fs·d_fine/(r^p - 1)，Fs=1.25
- 三条拒绝规则：梯次不足(<3级)、非单调收敛、U/gate > 1/3

### 2.2 第二层：梯次扩展（ladder_extension）

`benchmarks/xval_dg_round2_ladder_extension.py` 定义了 5 个预声明判据：

| 判据 | 含义 | 阈值 |
|---|---|---|
| L0 | 连续差严格递减（渐近区证据） | d₁ > d₂ > d₃ |
| L1 | 扩展梯次被同一引擎认证 | verdict=CERTIFIED |
| L2 | 阶一致性 | \|p₄-p₃\|/p₃ ≤ 0.20 |
| L3 | 带覆盖（自洽性） | rel L1(cert₃,cert₄) ≤ U₃ |
| L4 | 声明稳定性 | 所有腿仍过 3% 门；最大移动 ≤ 0.5pp |

### 2.3 第三层：交叉验证（xval_dg_round2_certified.py）

四个决策：
- **D1**: 参考可用（CERTIFIED + U/gate ≤ 1/3）
- **D2**: 所有 11 条 DG 腿过严格门（rel L1 < 3%）
- **D3**: 参考与闭形式 Oracle 比较 < 1%
- **D4**: 无 DG 家族可自认证（全部 WITHHELD_ORDER_TOO_LOW）

### 2.4 第四层：带宽诚实性审计（band_honesty）

`benchmarks/xval_dg_round2_band_honesty.py` 实现跨阶数审计：
- 用独立求解器 FV-WENO7（同物理、同通量、同时间积分、不同重建阶数）
- 分别认证 WENO5 和 WENO7 梯次
- 测量跨族距离 Δ = rel_l1(ref5, ref7)
- 三值判据：COVERED / MARGINAL / UNDER_COVERS
- 重判 11 条 DG 腿对双参考的误差，报告裕度移动

---

## 3. 计算结果（数字表，程序化注入零手抄）

### 3.1 三级参考认证（D1）

| 量 | 值 |
|---|---|
| 观测阶 p | 1.282509 |
| 带 U | 0.082797 % |
| U/gate | 0.0166 ≤ 0.3333 |
| 判决 | **CERTIFIED → PASS** |

### 3.2 四级梯次扩展（L0-L4）

| 判据 | 测量值 | 阈值 | 判定 |
|---|---|---|---|
| L0 相邻差 | 0.2308% → 0.0949% → 0.0420% | 严格递减 | PASS |
| L1 扩展认证 | CERTIFIED | CERTIFIED | PASS |
| L2 阶一致性 | \|p4-p3\|/p3 = 0.0825 | ≤ 0.20 | PASS |
| L3 带覆盖 | shift=0.0594% ≤ U3=0.0828% | 覆盖 | PASS |
| L4 声明稳定 | 最坏腿 2.4484% < 3%，移动 0.0073pp ≤ 0.5pp | 稳定 | PASS |

### 3.3 DG 腿误差（D2）

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

### 3.4 Oracle 比较（D3）

| 量 | 认证值 | 闭形式参考 | 误差 % | < 1% |
|---|---|---|---|---|
| plateau_rho | 3.857143 | 3.857143 | 1.225e-10 | yes |
| shock_position | 2.403125 | 2.389367 | 5.758e-01 | yes |
| mass_integral | 31.08915245 | 31.08915219 | 8.350e-07 | yes |
| preshock_sine_linf | 0.000864 | bound | 8.637e-04 | yes |

### 3.5 自认证拒绝（D4）

| 梯次 | 判决 | 要求 |
|---|---|---|
| dg_p1 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |
| dg_p2 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |
| dg_p3 | WITHHELD_ORDER_TOO_LOW | WITHHELD_ORDER_TOO_LOW |

**accepted = True**（D1✓ D2✓ D3✓ D4✓）

### 3.6 带宽诚实性审计（待 cj3566 落盘）

| 量 | 状态 |
|---|---|
| WENO7 梯次计算 | 集群作业 cj3566 排队中 |
| 跨阶数差 Δ | WITHHELD（待测量） |
| 覆盖判据 | WITHHELD（待测量） |
| DG 腿双参考重判 | WITHHELD（待测量） |

---

## 4. 测试状态

| 测试文件 | 通过 | 跳过 | 失败 |
|---|---|---|---|
| test_xval_dg_round2_evidence.py | 13 | 0 | 0 |
| test_xval_dg_round2_integration.py | 10 | 0 | 0 |
| test_xval_dg_round2_adversarial.py | 9 | 0 | 0 |
| test_xval_dg_round2_ladder_extension.py | 7 | 0 | 0 |
| test_xval_dg_round2_band_honesty.py | 16 | 1 | 0 |

**总计：55 PASS, 1 SKIPPED, 0 FAIL**

跳过的 1 条是 `test_audit_artifacts_consistent_if_present`：审计 JSON 尚不存在时
显式 SKIP（不伪装通过）。cj3566 落盘后该测试将自动转为 PASS。

---

## 5. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3%? |
|---|---|---|---|---|
| 最坏 DG 腿 rel L1 | 3% 门 | 2.455701% | — | **yes** |
| plateau_rho vs Oracle | 3.857143 | 3.857143 | 1.225e-10% | **yes** |
| shock_position vs Oracle | 2.389367 | 2.403125 | 0.5758% | **yes** |
| mass_integral vs Oracle | 31.08915219 | 31.08915245 | 8.350e-07% | **yes** |
| 参考带 U/gate | 0.3333 | 0.0166 | — | **yes** |

**所有已测量关键量误差均 < 3%，验证通过。**

---

## 6. 注意点（踩过的坑/局限/适用范围）

1. **N=6400 四级梯次数据**：已落盘 `validation/repro_shu_osher/artifacts/raw_ext/fv_cfl0p3_n6400.npz`。
2. **系统负载**：本机 load15 ≈ 70，重计算必须走集群调度，不可本地强跑。
3. **Oracle 来源**：所有闭形式 Oracle 来自 `tensordg.cases.shock_entropy`（案例模块，非引擎），
   确保引擎缺陷不会悄悄移动 Oracle。
4. **适用范围**：本验证针对一维 Euler 激波/熵波问题，不声明多维激波捕获的生产就绪性。
5. **带宽诚实性**：GCI 带宽回答的是「这一族再加密还会动多少」，不是「这一族离连续解有多远」。
   跨阶数审计（WENO5 vs WENO7）是回答后者的唯一途径，待 cj3566 落盘后闭合。

---

## 7. 产物清单

| 文件 | 说明 |
|---|---|
| `tensordg/certified_reference.py` | 认证引擎（可复用，case-independent） |
| `benchmarks/xval_dg_round2_certified.py` | 交叉验证引擎 |
| `benchmarks/xval_dg_round2_ladder_extension.py` | 梯次扩展审计引擎 |
| `benchmarks/xval_dg_round2_band_honesty.py` | 带宽诚实性审计引擎 |
| `benchmarks/fv_weno7_1d.py` | FV-WENO7 求解器（独立于 WENO5 路径） |
| `tools/derive_weno_tables.py` | WENO 系数表精确推导（sympy） |
| `validation/xval_dg_round2/artifacts/xval_round2_audit.json` | 交叉验证审计 JSON |
| `validation/xval_dg_round2/artifacts/xval_round2_audit.md` | 交叉验证审计 Markdown |
| `validation/xval_dg_round2/artifacts/ladder_extension.json` | 梯次扩展审计 JSON |
| `validation/xval_dg_round2/artifacts/fig1_xval_profiles.png` | 参考/DG 剖面图 |
| `validation/xval_dg_round2/artifacts/fig2_leg_oracle_deviation.png` | 腿误差/Oracle 偏差图 |
| `tests/test_xval_dg_round2_*.py` | 55 个测试（55 PASS, 1 SKIP） |
| `docs/xval_dg_round2_certified_reference_synthesis.md` | 本综合报告 |

---

## 8. 下一步

```bash
# 取回集群作业工件
run_compute action=fetch job_id=cj3566 --wait 3600

# 验证带宽诚实性审计
python -m pytest tests/test_xval_dg_round2_band_honesty.py -v

# 生成最终报告
python -m benchmarks.xval_dg_round2_band_honesty --figures
```

审计 JSON 的 `coverage.level` 与 `gate_survival.gate_survives_cross_family_shift`
两处即结论；无论 COVERED 还是 UNDER_COVERS，都应按 §5 的格式写进锚点回归报告。
