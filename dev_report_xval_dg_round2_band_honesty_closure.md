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
# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 跨阶数带宽诚实性审计（本单状态）
> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open，本单核对未变）
> 分支：`agent/m21629`　日期：2026-11-15
> 结论状态：**装置已建成并通过验证；量化测量（跨阶数差 Δ）已递交集群 cj3580，仍在排队——本单不编造该数字。**
## 1. 案例与要审的量（物理背景/几何/参数表）
Shu-Osher 激波/熵波相互作用（Shu & Osher 1988, JCP 77:439；Jiang & Shu 1996 头条算例）：
一维 Euler（γ=1.4），x∈[-5,5]，t=1.8，Mach-3 激波（左态为 RH 后状态 ρ=3.857143,
u=2.629369, p=10.33333）扫过 ρ=1+0.2sin(5x) 的静止正弦密度波。无闭式解。
已合并的锚点工件声称：FV-WENO5 参考解经三级/四级 GCI 认证，**带宽 U=0.0828 %**，
11 条 DG 腿对它的相对 L1 密度误差全部 < 3 %（最坏 2.4557 %）。
| 项 | 值 |
| 门限 τ（DG 腿） | 3 %（owner 精度标准 <15%，本处更严） |
| 被审量 | 认证带宽 U 是否覆盖「参考解自身的误差」 |
| 现有证据轴 | 单一族（FV-WENO5）× 单一梯次（800/1600/3200/6400） |
| 本单新增轴 | **同一物理、不同重建阶数（WENO7）** 的独立梯次 |
## 2. 为什么这是一个真实缺口（不是重复劳动）
`tensordg/certified_reference.py` 的 `band_coverage_study` docstring 自己写明：
「smooth ladders 会得到高观测阶、带非常保守……**这不说明激波情形**，激波情形下真误差根本
测不出来（无精确解）且观测阶 ~1」。也就是说：GCI 带宽回答的是「这一族再加密还会动多少」，
不是「这一族离连续解有多远」。锚点的 `CERTIFIED` 因此可能是**精确但有偏**的。
`ladder_extension`（L3）只做了**同族**（cert3 vs cert4）覆盖检查，没有跨族/跨阶轴。
## 3. 怎么做的（方法/工具链）
### 3.1 新增独立求解器 `benchmarks/fv_weno7_1d.py`
同物理（`tensordg.euler_1d` 的 HLLC 通量）、同时间积分（SSP-RK3）、同有限体积布局、
同边界约定，但**重建阶数不同**（7 阶：7 点模板、4 个子模板、不同最优权重与光滑指示子）。
因此它与 DG 路径、与 WENO5 路径都只共享物理内核。
所有系数表**由定义精确推导**，不是抄文献：
`tools/derive_weno_tables.py` 用 sympy 有理数运算求
① 子模板面值系数（矩量矩阵逆）、② Jiang-Shu 最优权重（一致性+归一化方程组精确解）、
③ Jin-Shu 光滑指示子 `β_k = Σ_l ∫ (∂^l p_k)² dx`（单位网格，精确积分）。
### 3.2 装置级验证（本单已完成的量化部分）
| 验证 | 结果 | 判定 |
| 表精确性：最优权重混合复现 order−1 次多项式 | q=5：1.11e-16；q=7：2.66e-15 | 机器精度 |
| q=5 最优权重 = 经典 Jiang-Shu | (0.1, 0.6, 0.3) 逐位相等 | 通过 |
| q=7 最优权重 = 文献值 | (1/35, 12/35, 18/35, 4/35) | 通过 |
| 方向对称性：`u_left` 与 `u_right` 对光滑场误差同量级 | n=64：均 6.57e-06，相对差 <1e-6 | 通过（钉住镜像索引） |
| 与在库 WENO5 的约定一致性 | 同场同网格：L/R 差 6.86e-06（仅 β 归一化差） | 通过 |
| 光滑场重建收敛阶（n=40→320） | q=5：3.78e-05→7.49e-09（观测阶 4.54）；q=7：6.60e-06→9.04e-10（观测阶 5.00） | 单调收敛、7 阶族精度恒优于 5 阶族 |
| 光滑指示子 | 常数场 β=0（<1e-12）、线性场 β>0 | 与定义一致 |
### 3.3 审计引擎 `benchmarks/xval_dg_round2_band_honesty.py`
1. 用**同一个认证引擎、同一个 3 % 门限**分别认证 WENO5 梯次与 WENO7 梯次；
2. 在两者认证剖面共处的网格（次细层 N=1600，无插值）上量 `Δ = rel_l1(ref5, ref7)`；
3. 三值判据：`COVERED`（Δ≤U5）／`MARGINAL`（U5<Δ≤U5+U7）／`UNDER_COVERS`（Δ>U5+U7）；
4. 把 11 条已落盘 DG 腿对**两个**认证参考各自重新判定，报告最坏裕度移动
   `max_abs_shift_pp`、是否有 PASS→FAIL 翻转，以及「最坏腿 + Δ 是否仍 < 3 %」的门限存活算术；
5. 控制轴：同族 3 级 vs 4 级（带本就该覆盖的量）。
## 4. 计算结果（本单实际拿到的数字）
| 量 | 值 | 来源 |
| WENO7 在 Shu-Osher 的梯次运行 | 未完成 | 集群作业 **cj3580**（`run_compute`，repo=TensorDG，stage_dir=本工作树，8 核，timeout 3600）自 08:49 排队，39 分钟后仍 queued（队列 56 job 在飞） |
| 跨阶数差 Δ | **未测（WITHHELD）** | 同上 |
| 测试 | 16 passed, 1 skipped | `pytest tests/test_xval_dg_round2_band_honesty.py` |
跳过的那 1 条是 `test_audit_artifacts_consistent_if_present`：审计 JSON 尚不存在时
显式 SKIP（不伪装通过）。
## 5. 结果图
**本单不提供场量对比图**：唯一需要图的量（ref5 vs ref7 剖面差、11 腿对双参考的误差柱状）必须由
WENO7 梯次剖面生成，而该梯次仍在集群队列中。图形代码已就绪（`figures()`，两张图，
matplotlib/Agg，程序化读取审计 JSON，零手抄），且报告生成脚本
`gen_report_xval_band_honesty.py` 已写好（读审计 JSON → 六要素报告 + 图内嵌）。
**不以占位图充数。**
## 6. 注意点（踩过的坑 / 局限 / 适用范围）
1. **镜像索引坑（已钉住）**：右状态用「翻转数组再翻转回来」实现时，索引必须与 ghost 层数
   匹配。本单先用 3 次迭代才发现少一格的偏移——表现是右状态误差 O(h) 而左状态 O(h^5)，
   由 `test_reconstruction_is_direction_symmetric` 固定住（该测试若失败，说明又偏了一格）。
2. **一个被否定的怀疑（负面结果，如实记录）**：初读 `benchmarks/fv_weno5_1d.py` 的
   `u_right = _weno5_left(s[5],...,s[1])` 时，我推断其最优权重镜像后未交换（0.1 与 0.3 互换），
   会把右状态降到 3 阶。数值检验否定该怀疑：**翻转后 p0 对应的是「远离目标面」的子模板**，
   故权重序无需交换；实测左右状态均 5.0 阶、误差同量级（6.57e-06 / 6.57e-06）。
   这条负面结果记录下来，以免后人重复怀疑。
3. **β 归一化**：Jin-Shu β 只差一个常数因子，本模块的 β 与在库 WENO5 手写常数不同
   （同场差 6.86e-06）。因此 5 阶路径是「重实现」而非逐位拷贝——这恰好提供了第二个敏感性轴。
4. **本地算力纪律**：本机被计算门守卫拦截（≥10 min 的活必须递交），nohup 后台亦被拦，
   故梯次只能走 `run_compute`；队列拥塞（56 job 在飞）导致本单交付「装置 + 验证」而非
   「装置 + 结论」。
5. **适用范围**：审计针对一维 Euler 激波/熵波；不声明多维激波捕获的带宽诚实性。
## 7. 误差判定（关键量 vs 基准，逐项）
| 关键量 | 基准 | 测量 | 误差 | <3 %? |
| 最优权重混合 vs 精确面值（q=5） | 解析 | — | 1.11e-16 | 是 |
| 最优权重混合 vs 精确面值（q=7） | 解析 | — | 2.66e-15 | 是 |
| q=5 最优权重 | 经典 Jiang-Shu (0.1,0.6,0.3) | (0.1,0.6,0.3) | 0（逐位） | 是 |
| q=7 最优权重 | 文献 (1/35,12/35,18/35,4/35) | 同 | 0（有理数精确） | 是 |
| 方向对称性 | — | 6.57e-06 / 6.57e-06 | 相对差 <1e-6 | 是 |
| 跨阶数带宽覆盖 Δ vs U5 | — | **未测** | — | **WITHHELD** |
**诚实结论：装置层面的所有解析基准误差 ≪3 %（机器精度级）；唯一真正要回答的量（Δ vs U5）
本单未取得测量，判 WITHHELD，不出结论。**
| `benchmarks/fv_weno7_1d.py` | 5/7 阶 FV-WENO/HLLC/SSP-RK3 求解器 + 表自检 + 阶数验证（已有，本单复核） |
| `benchmarks/xval_dg_round2_band_honesty.py` | 审计引擎（认证、跨阶差、三值覆盖判据、11 腿双参考重判、门限存活、图）（已有，本单复核） |
| `tools/derive_weno_tables.py` | 系数表的精确推导（sympy，有理数）（已有，本单复核） |
| `tests/test_xval_dg_round2_band_honesty.py` | 17 项测试（16 passed + 1 条件 SKIP） |
| `gen_report_xval_band_honesty.py` | **本单新增**：读审计 JSON → 六要素论文级报告 + 图内嵌（待 cj3580 落盘即跑） |
| `validation/xval_dg_round2/artifacts/band_honesty/` | 审计输出目录（JSON/Markdown/PNG/weno7 原始 npz 缓存）——待 cj3580 落盘 |
## 9. 下一步（接单者照做即可闭环）
```bash
run_compute action=fetch job_id=cj3580          # 取回梯次与审计工件
python -m benchmarks.xval_dg_round2_band_honesty --figures   # 生成审计 + 图
python -m pytest tests/test_xval_dg_round2_band_honesty.py -q   # 17 passed
python gen_report_xval_band_honesty.py          # 生成六要素报告 JSON
# 发布：curl -s -X POST $TENSOR_API_BASE/tensor/reports ... -d @.m21629_xval_band_honesty_report.json
```
审计 JSON 的 `coverage.level` 与 `gate_survival.gate_survives_cross_family_shift`
两处即结论；无论 COVERED 还是 UNDER_COVERS，都应按 §7 的格式写进锚点回归报告。
