# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — Band-Honesty 闭合报告

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）
> 分支：`agent/m21780` 日期：2026-11-16
> 结论：**带宽诚实性审计已闭合——MARGINAL 覆盖 + 门限存活，11 条 DG 腿无翻转**

## 1. 案例介绍（物理背景/几何/参数表）

Shu-Osher 激波/熵波相互作用（Shu & Osher 1988, JCP 73:52）：一维 Euler 方程（γ=1.4），
计算域 x∈[-5,5]，终态 t=1.8。Mach-3 激波（左态 ρ=3.857143, u=2.629369, p=10.33333）
扫过 ρ=1+0.2sin(5x) 的静止正弦密度波。无闭式解。

| 参数 | 值 |
|---|---|
| 参考求解器 | FV-WENO5/HLLC/SSP-RK3（CFL=0.3） |
| 交叉验证求解器 | FV-WENO7/HLLC/SSP-RK3（CFL=0.3） |
| 梯次 | [800, 1600, 3200, 6400]（四级） |
| DG 腿 | 11 组（P1/P2/P3 × N=200/400/800, CFL=0.4；P2 N=400, CFL=0.1/0.2） |
| 门限 | 3%（rel L1 密度误差） |

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 2.1 带宽诚实性审计原理

已合并的锚点工件声称：FV-WENO5 参考解经四级 GCI 认证，带宽 U=0.0416%，11 条 DG 腿对它的
相对 L1 密度误差全部 < 3%（最坏 2.4484%）。但 U 来自单一族（WENO5）的网格收敛分析，回答的是
「这一族再加密还会动多少」，不是「这一族离连续解有多远」。

本审计引入**独立重建阶数**（WENO7）的梯次作为第二轴：
1. 用同一认证引擎分别认证 WENO5 和 WENO7 梯次；
2. 在两者认证剖面共处的网格（N=3200，四级梯次的参考层）上量 Δ = rel_l1(ref5, ref7)；
3. 三值判据：COVERED（Δ≤U5）/ MARGINAL（U5<Δ≤U5+U7）/ UNDER_COVERS（Δ>U5+U7）；
4. 将 11 条 DG 腿对两个认证参考各自重新判定，报告裕度移动和门限存活。

### 2.2 本单修复的缺陷

`benchmarks/xval_dg_round2_band_honesty.py` 的 `render_markdown` 函数在距离值为 `None` 时
（如 intra-family control 因 3 级/4 级参考层网格数不匹配而无法计算）会触发 `TypeError`。
本单新增 `_fmt(v, spec)` 辅助函数，将 `None` 格式化为 `"N/A"`，使 Markdown 渲染鲁棒。

### 2.3 数据来源

WENO7 梯次（n3200, n6400）来自集群作业 cj3525（2026-11-15 07:51 递交，n330 节点，succeeded）。
本单将 cj3525 的 WENO7 原始数据（n3200, n6400）复制到工作树，并重新运行审计生成 JSON/MD/PNG。

## 3. 计算结果（数字表，程序化注入零手抄）

### 3.1 认证参考

| 参考 | 判决 | 带宽 U % | 观测阶 | 参考层 |
|---|---|---|---|---|
| WENO5（4 级） | CERTIFIED | 0.041622 | — | N=3200 |
| WENO7（4 级） | CERTIFIED | 0.042237 | — | N=3200 |

### 3.2 跨阶数距离与覆盖判据

| 量 | 值 |
|---|---|
| Δ（4 级跨阶距离） | 0.051906 % |
| Δ（3 级跨阶距离） | 0.118764 % |
| Δ / U5 | 1.247 |
| **覆盖判据** | **MARGINAL** |
| 联合带覆盖（U5+U7） | True（0.0519 ≤ 0.0839） |

### 3.3 DG 腿重判

| 腿 | vs WENO5 ref % | vs WENO7 ref % | shift pp | pass both |
|---|---|---|---|---|
| dg_cfl0p4_limweno_n200_p1（最坏） | 2.4484 | 2.4504 | +0.0020 | True |
| dg_cfl0p4_limweno_n800_p3 | 0.6647 | 0.6686 | +0.0039 | True |
| …（其余 9 条） | 0.56–2.42 | 0.57–2.42 | +0.0018–0.0037 | True |

| 汇总量 | 值 |
|---|---|
| 最坏腿误差（vs WENO5） | 2.4484 % |
| 最坏腿误差（vs WENO7） | 2.4504 % |
| 最大 |shift| | 0.0039 pp |
| 判决翻转（3% 门） | **无** |
| 门限存活（最坏 + Δ < 3%） | **True**（2.5003 % < 3.0 %） |

## 4. 结果图

### 4.1 跨阶数参考剖面对比

![fig1](validation/xval_dg_round2/artifacts/band_honesty/fig1_cross_order_references.png)

*图 1：Shu-Osher t=1.8 的 WENO5 与 WENO7 认证参考剖面（N=3200）及差值。Δ = 0.0519 %。*

### 4.2 DG 腿对双参考的误差比较

![fig2](validation/xval_dg_round2/artifacts/band_honesty/fig2_dg_legs_two_references.png)

*图 2：11 条 DG 腿对 WENO5 和 WENO7 认证参考的相对 L1 误差。3% 门限（虚线）无翻转。*

## 5. 注意点（踩过的坑/局限/适用范围）

1. **None 值渲染缺陷（本单已修复）**：`render_markdown` 原未处理 `intra_family_weno5_3v4_percent`
   和 `weno5_shipped_vs_generic_percent` 为 `None` 的情况（3 级/4 级参考层网格数不匹配时发生），
   导致 `TypeError: unsupported format string passed to NoneType.__format__`。
   修复：新增 `_fmt(v, spec)` 辅助函数，`None` → `"N/A"`。
2. **intra-family control 为 N/A 的原因**：WENO5 的 3 级梯次（800/1600/3200）参考层在 N=1600，
   4 级梯次（800/1600/3200/6400）参考层在 N=3200，网格数不同无法直接计算 rel_l1。
   这是认证引擎的设计行为（参考层取次细层），不是缺陷。
3. **MARGINAL 判据的物理含义**：Δ = 0.0519 % 略大于 U5 = 0.0416 %（比值 1.25），说明 WENO5
   认证带**不完全覆盖**跨族差异，但 WENO5+WENO7 联合带（0.0839 %）完全覆盖。这意味着参考解
   的「真实不确定度」比单一族 GCI 带宽大约 25%，但仍远小于 3% 门限。
4. **适用范围**：本审计针对一维 Euler 激波/熵波问题；不声明多维激波捕获的带宽诚实性。

## 6. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3%? |
|---|---|---|---|---|
| 最坏 DG 腿 vs WENO5 ref | 3% 门 | 2.4484 % | — | **yes** |
| 最坏 DG 腿 vs WENO7 ref | 3% 门 | 2.4504 % | — | **yes** |
| 最坏腿 + Δ（门限存活） | 3% 门 | 2.5003 % | — | **yes** |
| Δ / U5（覆盖比） | — | 1.247 | — | MARGINAL |
| 判决翻转 | 0 | 0 | — | **yes** |

**所有 DG 腿对两个独立认证参考均通过 3% 门限，无翻转，门限存活。**

## 7. 测试状态

| 测试文件 | 通过 | 跳过 | 失败 |
|---|---|---|---|
| test_xval_dg_round2_band_honesty.py | **17** | **0** | **0** |
| test_xval_dg_round2_band_honesty_synthetic.py | 11 | 0 | 0 |
| test_xval_dg_round2_evidence.py | 13 | 0 | 0 |
| test_xval_dg_round2_integration.py | 10 | 0 | 0 |
| test_xval_dg_round2_adversarial.py | 9 | 0 | 0 |
| test_xval_dg_round2_ladder_extension.py | 7 | 0 | 0 |

**总计：67 PASS, 0 SKIPPED, 0 FAIL**

> 注：`test_audit_artifacts_consistent_if_present` 从 SKIPPED 转为 PASSED（审计 JSON 已存在）。

## 8. 产物清单

| 文件 | 说明 |
|---|---|
| `benchmarks/xval_dg_round2_band_honesty.py` | 修复 None 渲染缺陷（+10/-5 行） |
| `validation/xval_dg_round2/artifacts/band_honesty/band_honesty_audit.json` | 完整审计 JSON |
| `validation/xval_dg_round2/artifacts/band_honesty/band_honesty_audit.md` | 审计 Markdown |
| `validation/xval_dg_round2/artifacts/band_honesty/fig1_cross_order_references.png` | 跨阶参考对比图 |
| `validation/xval_dg_round2/artifacts/band_honesty/fig2_dg_legs_two_references.png` | DG 腿双参考误差图 |
| `validation/xval_dg_round2/artifacts/band_honesty/raw_weno7/fv_weno7_cfl0p3_n3200.npz` | WENO7 N=3200 原始数据 |
| `validation/xval_dg_round2/artifacts/band_honesty/raw_weno7/fv_weno7_cfl0p3_n6400.npz` | WENO7 N=6400 原始数据 |

## 9. 集群作业

| 作业号 | 命令 | 状态 |
|---|---|---|
| cj3525 | WENO7 四级梯次求解 + 审计 | succeeded（n330, rc=0） |
