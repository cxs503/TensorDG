# CASE-DG-PREFINEMENT-EXPONENTIAL-2D — 二维 p 型加密（光滑解）指数收敛

**锚点**：`CASE-DG-PREFINEMENT-EXPONENTIAL`（算例池·p 型加密指数收敛，光滑解）
**文献**：Babuska & Suri (1994) *The p- and h-p versions of the finite element method,
basic principles and properties*, SIAM Review 36(4):578–632, DOI 10.21236/ada260237
（解析解 ⇒ 误差对多项式阶 p 指数衰减：`e(p) ≤ C exp(-σ p)`）
**工件**：`docs/benchmarks/results/p-refinement-exponential-2d/`（JSON + CSV + 4 图 + provenance + transfer 诊断）
**驱动**：`benchmarks/p_refinement_exponential_2d.py`、`benchmarks/p_refinement_2d_transfer.py`
**算例定义**：`tensordg/cases/prefinement_exponential_2d.py`（已登记入 `tensordg.cases.CASES`）
**门禁**：`tests/test_p_refinement_exponential_2d.py`（12 道活闸门，实算求解器，非只读工件）

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

一维算例（m16180/m16499/m16918）已把 1D 的 p 型指数收敛钉死在 σ=3.9271。其 verifier 审计
（`docs/AUDIT-m16499-CASE-DG-PREFINEMENT-EXPONENTIAL.md`）把「**二维 / 变系数未覆盖**」列为未闭合局限；
同时 `tensordg/advection_2d.py` 还带着一维曾经有过的那种无依据硬上限 `degree in (0,1,2,3)`，
使任何 p>3 的二维阶梯根本跑不起来。本单同时关闭这两项。

| 项 | 取值 |
|---|---|
| PDE | `u_t + a_x u_x + a_y u_y = 0`（线性对流，周期） |
| 区域 | `[0,1]×[0,1]` 周期 |
| 速度 | `a = (1.0, 0.5)` |
| 初值 | `u0 = 1 + sin(2πx)·sin(2πy)`（整函数，解析） |
| 精确解 | `ue = 1 + sin(2π(x - a_x t))·sin(2π(y - a_y t))` |
| 网格 | `8×8` 单元固定（隔离 p 加密与 h 加密） |
| 阶 | `p = 1..6` |
| 时间 | `T = 1.0`；RK4，CFL 逐级折半直到相邻两级 L2 相对变化 < 1 % |
| DOF | `8·8·(p+1)²` = 256 … 3136 |

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **解除硬上限**：`advection_2d.solve_advection_2d` 的 `degree not in (0,1,2,3)` 换成显式
   `0 ≤ degree ≤ MAX_DEGREE(=8)`；张量积算子本就是 degree-generic（Legendre 值与导数在 Gauss 点上构造），
   上限只约束 Python 侧基装配。`benchmarks` 中的 `guard_check()` 把「p=4/5/6/8 接受、p=9 拒绝」记进工件。
2. **新算例模块** `tensordg/cases/prefinement_exponential_2d.py`：按包内约定提供
   `prefinement_parameters()` / `solve_level()` / `p_ladder()` / `anchor_table()`，benchmark、tests、下游只读这一处常量。
3. **每级三重测量**（不信任单一误差例程）：
   - 生产路径 `advection_2d.error_2d`（torch，Gauss 阶 `2p+4`）；
   - 独立路径 `independent_relative_l2`（纯 numpy、阶 `2p+6`，代码独立）→ 记录 `path_dispersion`；
   - 舍入地板信噪比 `roundoff_floor(||u||)` / `signal_to_noise`。
4. **判据**：沿用一维算例在 m16918 收紧后的复合预登记判据（同一实现
   `convergence.exponential_decay_criterion`）——`σ>1` ∧ 逐段衰减率不减 ∧ `e(p_min)/e(p_max)>1e3` ∧ 误差单调下降；
   另跑 10 条已知衰减律的**证伪对照**（合成序列，与实测无关）。
5. **可移植性诊断** `benchmarks/p_refinement_2d_transfer.py`：不覆盖上面预登记判决，单独回答
   「二维数据是求解器错，还是那条判据不可移植」。三项检查：旋转对称性（实算）、
   逐段衰减率的代数闭式比较（`p^-k` ⇒ 率按 `ln2/ln(6/5)=3.8` 倍下降；指数律 ⇒ 率恒定）、
   指数拟合 vs 最优代数拟合的 log-SSE 比较。

## 3. 计算结果（程序化注入，零手抄）

主表来自 `docs/benchmarks/results/p-refinement-exponential-2d/p_refinement_2d_table.csv`：

| p | DOF | 相对 L2 | 独立求积 L2 | 路径相对离散 | 信噪比 | CFL | 步数 |
|---|---|---|---|---|---|---|---|
| 1 | 256 | 3.350858e-02 | 3.350858e-02 | 3.6e-14 | 1.4e+14 | 0.05 | 720 |
| 2 | 576 | 1.500029e-03 | 1.500029e-03 | 2.8e-14 | 6.0e+12 | 0.05 | 1200 |
| 3 | 1024 | 7.127515e-05 | 7.127515e-05 | 6.9e-14 | 2.9e+11 | 0.05 | 1680 |
| 4 | 1600 | 2.760981e-06 | 2.760981e-06 | 4.6e-12 | 1.1e+10 | 0.05 | 2160 |
| 5 | 2304 | 9.083434e-08 | 9.083434e-08 | 1.5e-10 | 3.7e+08 | 0.05 | 2640 |
| 6 | 3136 | 2.539602e-09 | 2.539602e-09 | 8.6e-09 | 1.0e+07 | 0.05 | 3120 |

- 最小二乘指数衰减率 **σ = 3.2675**（对数域斜率 −3.2675）
- 逐段衰减率 **3.1063 / 3.0467 / 3.2510 / 3.4143 / 3.5770**
- 总降幅 `e(1)/e(6) = 1.32e+07`（>1e3 门），逐级缩减 22.3× / 21.0× / 25.8× / 30.4× / 35.8×
- 全部 6 级 `temporal_converged = true`；`max path dispersion = 8.6e-09`；`min SNR = 1.0e+07`
- 时限：整套（6 级 + 4 图 + guard）199.4 s 单核

### 图

1. `p_refinement_2d_semilog.png` — 半对数误差–阶曲线 + 指数拟合（σ=3.267）+ Babuska-Suri 参考斜率 0.5
2. `p_refinement_2d_log10.png` — log10(误差) 对 p 的线性拟合（斜率 −3.268）
3. `p_refinement_2d_fields.png` — p=1 与 p=6 的数值解云图 + 逐点误差场（p=6 最大误差 ~3e-9）
4. `p_refinement_2d_vs_1d.png` — 误差对 DOF：与已发布一维算例（σ=3.93，16 单元）同图比较

## 4. 判据判定（诚实申报，不粉饰）

| 判据 | 来源 | 实测 | 判决 |
|---|---|---|---|
| 半对数斜率 < −0.5/阶，即 **σ > 0.5** | **锚点预登记**（Babuska-Suri 原文容差） | σ = 3.2675（余量 6.5×） | **PASS** |
| σ > 1 | m16918 复合判据 | 3.2675 | PASS |
| `e(p1)/e(p6) > 1e3` | m16918 复合判据 | 1.32e7 | PASS |
| 误差单调下降 | m16918 复合判据 | 是 | PASS |
| **逐段衰减率不减** | m16918 复合判据（**一维标定**） | 3.1063→3.0467 处 **−1.92 %** 单点回落 | **FAIL** |
| 10 条已知衰减律证伪对照 | 合成表 | 全部落预期侧 | PASS |
| 极限解除（p=4..8 可算、p=9 拒绝） | 本单 | 是 | PASS |

**因此工件里的复合判据 verdict 记为 `FAIL`（预登记判据不因看到数据而修改），而锚点自身的 σ>0.5 判据
以 6.5 倍余量通过。** 这个 FAIL 的归因见 §5，`tests/test_p_refinement_exponential_2d.py` 把这条 FAIL
连同「哪一条子条件失败」一起锁成测试，任何人偷偷改判据都会红。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **错误归属已排除求解器**：`transfer_diagnostics.json` 记录
   - 旋转对称性：把速度换成分量互换 `(0.5,1.0)` 后逐级 L2 **逐位一致**（最大相对差 3.8e-15）⇒ y 向通量/索引无缺陷；
   - 速度 `(1.0,1.0)` 复核：3.282→3.050，同样出现该回落 ⇒ 与具体速度无关；
   - 指数拟合 vs 最优代数律 `p^-k` 拟合：log-SSE 比值 **103×** 偏向指数律 ⇒ 数据确实是谱收敛，不是代数律。
2. **判据不可移植的根因（代数闭式，非数据调参）**：`p^-k` 的逐段衰减率是 `k·ln((p+1)/p)`，
   p=1..6 内按 3.8 倍系统下降；纯指数律的逐段率**恒定**。所以「率不减」不是指数律的普适性质，
   而是一维低阶端恰好单调的一维标定结果。可移植条款应为 **`末段率 ≥ 首段率`**；
   该条款先在**合成律表**上单独验证（分离 10 条已知律全部正确，`portable_falsification.gate_pass=true`），
   再应用到实测阶梯（通过）。**明确标注：这是 post-hoc 分析，不用于改写已发布 verdict**，
   仅作为后续二维/三维预登记的建议。
3. **适用范围**：一维/二维周期常系数线性对流、整函数初值。p=6 时 SNR 已降至 1e7（信噪比随 p 指数下降），
   `MAX_DEGREE=8` 是代码上限而非可验证上限；三维与变系数未覆盖。
4. **一维对照**：一维阶梯逐段率 3.576→4.269 自始单调，二维在 p=1..3 段有一个 1.92 % 的预渐近回落，
   说明低阶端率序列对维度敏感——这正是不宜把「单调不减」当作普适判据的实证。

## 6. 误差判定（逐项，显式回答）

本算例的「基准」是**解析解**（不是实验值），故不适用「实验误差 <3%」的表述；可判定的是：
每一级的离散误差**是否被独立路径复现**，以及**收敛律是否符合预登记判据**。

| 关键量 | 基准 | 实测偏差 | 判决 |
|---|---|---|---|
| 各级 L2 误差（p=1..6） | 独立 numpy 高 2 阶求积 | 相对离散 ≤ 8.6e-09 | PASS（≪1 %） |
| σ（误差–阶指数率） | 锚点判据 σ>0.5 | 3.2675（余量 553 %） | PASS |
| σ（复合判据 σ>1） | m16918 | 3.2675 | PASS |
| 总降幅 | >1e3 | 1.32e7（余量 1.3e4 倍） | PASS |
| 逐段率单调性 | m16918（一维标定） | −1.92 % 单点回落 | **FAIL（诚实申报）** |
| 一维已发布 σ 复核 | 1D 工件 σ=3.9271 | 未在本单重跑（沿用 m16499 复核结论） | WITHHELD（本单未重测） |

结论一句话：**二维 p 型加密确认为指数收敛（σ=3.27，锚点判据以 6.5× 余量通过），
但一维标定的复合判据「逐段率不减」不可移植，实测出现 −1.92 % 单点回落，故工件 verdict 记 FAIL，
归因分析证明这是判据而非求解器的问题，并给出可移植替代条款（合成表验证通过，标注 post-hoc）。**

## 复现

```bash
PYTHONPATH=. python -m benchmarks.p_refinement_exponential_2d        # 全量 p=1..6，~200 s
PYTHONPATH=. python -m benchmarks.p_refinement_exponential_2d --quick # p=1..3 smoke
PYTHONPATH=. python -m benchmarks.p_refinement_2d_transfer            # 可移植性/对称性诊断
PYTHONPATH=. python -m pytest -q tests/test_p_refinement_exponential_2d.py
```