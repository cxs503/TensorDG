# CASE-LAX-DG · 认证腿上的时间误差预算（P3 N = 400 / P3 N = 800）

> 锚点：`feature_ledger` `CASE-LAX-DG`（验收原文「ref: 精确解｜容差 < 3 %」，status 仍 `open`）。
> 本文全部数字由 `python -m benchmarks.lax_time_budget_certified` 从工件
> `docs/benchmarks/results/lax_time_budget_certified/metrics.json` 注入（零手抄；
> `--check` 在漂移时失败）。测量运行 2026-11-15T05:27:02Z，墙钟 1664.2 s，
> 工件 sha256 前缀 `60123ae7139612bf`。
> 总体判定：**PASS**。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，γ = 1.4，隔膜 `x0 = 0.5`，
左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0.0, 0.571)`，
快照 `t = 0.13`。参考值 = 精确 Riemann 解（闭式，不依赖任何第三方求解器）。

**待办缺口（本单要回答的）**：算例池记录里两条相反方向的关键读数——钉扎腿 `p3.N400`
密度 rel L2 = `3.11556e-02`（未达标）、
分辨率声明腿 `p3.N800` = `2.31490e-02`（达标）
——**都是在单一 CFL = 0.35 下取的**，而池子的时间门 `gate_time`
（注册容差 5%）从未在这两条腿上、也从未在台账所用的 rel L2 范数上评过。
若不测，「N=800 达标」有被 `dt` 误差注水的风险，而「N=400 不达标」有被误判为分辨率问题的风险。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. 网格固定（P3, N = 400 与 800），只动 `dt`：CFL ∈ { 0.35000, 0.17500, 0.08750 }，
   即三次精确 2× 细化（与池子 `TIME_CFL_DIVISORS = (1,2,4)` 同口径）。
2. 误差口径沿用池子本身：单元 Gauss 节点（求积阶 10）、Legendre 求积加权、
   全场、不挖波带、不给任何量豁免；参考值由 `benchmarks.lax_case_pool.exact_reference` 供给。
3. 时间诊断直接复用池子的注册实现：`temporal_order`（三次精确对分给出闭式阶，不做拟合）、
   `temporal_share`（可辨识时的模型份额）、`temporal_bound`（序列平/非单调时的 4× dt 实测敏感度，作为诚实上界）。
4. 预登记规则（跑之前写死在驱动里 `TIME_BUDGET_RULE`）：T1 temporal share of the baseline-CFL whole-field relative L2 error < 5 % for every leg and every field (order-identifiable -> temporal_share, else the measured 4x-dt bound); T2 p3.N800 max-field rel L2 < ledger tolerance at every CFL level; T3 p3.N400 max-field rel L2 >= ledger tolerance at every CFL level (the FAIL is not a dt artifact); T4 baseline-CFL L2 reproduces the committed resolution artifact; T5 steps and dt_mean halve with the CFL divisor (ratio in [1.9, 2.1]). Overall PASS iff T1-T5 hold.
5. **不改任何已提交数字、不移动 A0 生产腿**：本文只在独立工件里追加时间维度的陈述。

## ③ 计算结果（程序化注入）

### 3.1 全部测量点（rel L2，台账度量）

| leg | CFL | steps | dt_mean | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 对 3% 判据 |
|---|---|---|---|---|---|---|---|---|
| p3.N400 | 0.35000 | 5109 | 2.5445e-05 | 3.11556e-02 | 2.24805e-02 | 1.30501e-02 | 3.11556e-02 (`density`) | FAIL |
| p3.N400 | 0.17500 | 10217 | 1.2724e-05 | 3.12877e-02 | 2.28046e-02 | 1.32258e-02 | 3.12877e-02 (`density`) | FAIL |
| p3.N400 | 0.08750 | 20429 | 6.3635e-06 | 3.13361e-02 | 2.29407e-02 | 1.32887e-02 | 3.13361e-02 (`density`) | FAIL |
| p3.N800 | 0.35000 | 10250 | 1.2683e-05 | 2.31490e-02 | 1.38257e-02 | 8.99402e-03 | 2.31490e-02 (`density`) | PASS |
| p3.N800 | 0.17500 | 20497 | 6.3424e-06 | 2.32623e-02 | 1.39665e-02 | 9.12024e-03 | 2.32623e-02 (`density`) | PASS |
| p3.N800 | 0.08750 | 40986 | 3.1718e-06 | 2.32990e-02 | 1.40098e-02 | 9.16361e-03 | 2.32990e-02 (`density`) | PASS |

### 3.2 时间误差预算（T1）

| leg | 量 | rel L2 @ 最粗 dt | rel L2 @ 最细 dt | 观测时间阶 | 时间份额/上界 | T1 |
|---|---|---|---|---|---|---|
| p3.N400 | density | 3.115562e-02 | 3.133607e-02 | null（不可辨识） | bound 5.758e-03 | PASS |
| p3.N400 | velocity | 2.248053e-02 | 2.294072e-02 | null（不可辨识） | bound 2.006e-02 | PASS |
| p3.N400 | pressure | 1.305012e-02 | 1.328873e-02 | null（不可辨识） | bound 1.796e-02 | PASS |
| p3.N800 | density | 2.314899e-02 | 2.329896e-02 | null（不可辨识） | bound 6.437e-03 | PASS |
| p3.N800 | velocity | 1.382570e-02 | 1.400977e-02 | null（不可辨识） | bound 1.314e-02 | PASS |
| p3.N800 | pressure | 8.994023e-03 | 9.163610e-03 | null（不可辨识） | bound 1.851e-02 | PASS |

最坏读数（两腿取最坏）：`velocity`，按 **bound** 读 =
`2.006e-02`，注册门 `5%` → 整体 T1 = **PASS**。

### 3.3 台账判定的 dt 稳健性（T2 / T3）

| leg | 层级 | 最差 rel L2 | 最差量 | 容差裕度（×） | verdict |
|---|---|---|---|---|---|
| p3.N400 | cfl=0.35000 | 3.11556e-02 | density | 1.039 | FAIL |
| p3.N400 | cfl=0.17500 | 3.12877e-02 | density | 1.043 | FAIL |
| p3.N400 | cfl=0.08750 | 3.13361e-02 | density | 1.045 | FAIL |
| p3.N800 | cfl=0.35000 | 2.31490e-02 | density | 0.772 | PASS |
| p3.N800 | cfl=0.17500 | 2.32623e-02 | density | 0.775 | PASS |
| p3.N800 | cfl=0.08750 | 2.32990e-02 | density | 0.777 | PASS |

* T2「N = 800 达标不是 dt 假象」：**PASS**
  （三档 CFL 的 verdict 一致性：`PASS`）；
* T3「N = 400 不达标不是时间步问题」：**PASS**
  （`FAIL`）。

### 3.4 复用检查（T4）与细化有效性（T5）

基线 CFL 的读数 vs 已提交 `lax_resolution_requirement` 工件（相对容差 `1e-09`）：
最差相对差 `0.000e+00` → **PASS**。

| leg | field | 已提交 | 本次 | 相对差 |
|---|---|---|---|---|
| p3.N400 | density | 3.115562e-02 | 3.115562e-02 | 0.00e+00 |
| p3.N400 | velocity | 2.248053e-02 | 2.248053e-02 | 0.00e+00 |
| p3.N400 | pressure | 1.305012e-02 | 1.305012e-02 | 0.00e+00 |
| p3.N800 | density | 2.314899e-02 | 2.314899e-02 | 0.00e+00 |
| p3.N800 | velocity | 1.382570e-02 | 1.382570e-02 | 0.00e+00 |
| p3.N800 | pressure | 8.994023e-03 | 8.994023e-03 | 0.00e+00 |

| leg | steps | 步数比（相邻 CFL 对分） | dt_mean 比 | T5 |
|---|---|---|---|---|
| p3.N400 | [5109, 10217, 20429] | 1.9998, 1.9995 | 1.9998, 1.9995 | PASS |
| p3.N800 | [10250, 20497, 40986] | 1.9997, 1.9996 | 1.9997, 1.9996 | PASS |

## ④ 结果图

* `docs/benchmarks/results/lax_time_budget_certified/figures/lax_time_budget_l2_vs_cfl.png`
* `docs/benchmarks/results/lax_time_budget_certified/figures/lax_time_budget_share.png`
* `docs/benchmarks/results/lax_time_budget_certified/figures/lax_time_budget_profiles.png`

* `lax_time_budget_l2_vs_cfl.png`：两腿的 rel L2（ρ/u/p）随 dt 细化，虚线为台账 3 % 容差；
* `lax_time_budget_share.png`：各腿各量的时间误差份额 vs 5 % 注册门；
* `lax_time_budget_profiles.png`：基线 CFL 与 4× 细化 dt 的密度剖面叠加（vs 精确解），看细化 dt 是否真的移动了场。

渲染来源：rendered from the committed profiles cache。

## ⑤ 注意点

1. **只回答时间维度**：本文不重算空间阶、不重判 A0 生产腿；`p3.N400` 的 FAIL 原样保留。
   若要把锚点翻成 done，仍需台账持有人显式变更「生产腿」口径。
2. **份额 vs 上界**：激波主导的 norm 在 dt 细化下常常是平的（序列不可辨识 ⇒ `temporal_order = null`），
   此时 `temporal_share` 无定义，只能报 4× dt 实测敏感度作为上界；本文按池子注册的读法取上界，
   不把「测不出」写成「等于零」。
3. **口径敏感性**：误差在 Gauss 节点上按 Legendre 求积加权；换单元平均 / L∞ 会给出不同数字，
   口径写在工件 `config` 里供重裁。
4. **算力**：N = 800 / CFL = 0.08750 需 40986 步，
   属重计算，本机不起，已递交集群调度执行。

## ⑥ 误差判定（是否 < 3%）

| 关键量 | 基准 | 数值 | 相对误差 | < 3% ? |
|---|---|---|---|---|
| rel L2 density @ p3.N800, 最细 dt | 精确 Riemann 解 | 2.32990e-02 | 2.330 % | 是 |
| rel L2 velocity @ p3.N800, 最细 dt | 精确 Riemann 解 | 1.40098e-02 | 1.401 % | 是 |
| rel L2 pressure @ p3.N800, 最细 dt | 精确 Riemann 解 | 9.16361e-03 | 0.916 % | 是 |
| rel L2 density @ p3.N400（钉扎腿，未改） | 精确 Riemann 解 | 3.11556e-02 | 3.116 % | 否 |
| 时间误差份额（最坏字段） | 池子注册门 | 2.006e-02 | 2.006 % | 是 |

**结论**：在 P3 / N = 800 上、且在最细 dt 下，三量 rel L2 全部
达标（< 3%）；
钉扎腿 `p3.N400` 的 FAIL 在三档 dt 下都不被时间细化救回。
时间误差份额 低于注册门 5%，
因此上述空间分辨率结论不是 `dt` 误差造成的。
