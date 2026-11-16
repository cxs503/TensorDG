# CKS1998 2-D Riemann Problem Verification (Lax & Liu 1998, Config 3)

> **锚点**: `REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) / SIAM Review 43(1) (2001)）
> **算例**: 四象限 Riemann 问题（Lax & Liu 1998, Configuration 3，两道激波相互作用）
> **求解器**: `tensordg.euler_dg_2d.solve_euler_dg_2d`（P1/P2 DG, HLLC 通量, SSP-RK3, TVB 限幅）
> **机器记录**: `docs/benchmarks/results/riemann_2d_cks1998/metrics.json`（由本文档 §复现命令 生成）
> **状态**: 门判定**只**由记录渲染（见 ③ 的数字表与 ⑥ 的最低判据）；本文正文不手写任何运行数字。

---

## ① 案例介绍

四象限 Riemann 问题是二维 Euler 方程的经典基准（Lax & Liu 1998, 19 种构型）。
Configuration 3 是**两道激波相互作用**的情形，对激波捕捉格式的二维稳定性和
正性保持能力构成严格考验。CKS1998 论文（§V）将此作为 RKDG 方法在二维系统上的
代表性算例。

### 初始条件（输入，不是运行结果）

| 象限 | 区域 | $\rho$ | $u$ | $v$ | $p$ |
|------|------|-------:|----:|----:|----:|
| 左下 | $x<0.5, y<0.5$ | 0.138 | 1.206 | 1.206 | 0.3 |
| 右下 | $x>0.5, y<0.5$ | 0.5323 | 0.0 | 1.206 | 0.3 |
| 左上 | $x<0.5, y>0.5$ | 0.5323 | 1.206 | 0.0 | 0.3 |
| 右上 | $x>0.5, y>0.5$ | 1.5 | 0.0 | 0.0 | 1.5 |

$\gamma = 1.4$，透射边界（transmissive），$t = 0.25$（完整算例时刻）。
该表与求解器内的 `tensordg.euler_dg_2d.LAX_LIU_CONFIG_3` 逐值绑定
（`tests/test_riemann_2d_cks1998_doc_binding.py::test_documented_initial_states_match_solver`），
文档抄错初值也会失败。

### 记录里出现的构型

`metrics.json` 的两个 run 行是：

1. **物理构型**（`periodic=(False, False)`，透射边界）——本题的真实边界条件，质量
   会通过边界流动，因此"总质量变化"是物理量而不是守恒误差；
2. **周期控制**（`periodic=(True, True)`）——同一初值、无边界，任何守恒量漂移都只能
   来自格式本身，用来把"物理通量"和"格式误差"分开。

`--quick` 是**烟测构型**，参数固定在 `benchmarks/riemann_2d_cks1998.py::QUICK_CONFIG`
（32×32 P1、$t=0.1$、CFL=0.2），记录里的 `runs[*].final_time` 会如实写出来。选 $t=0.1$
而不是完整算例时刻 $t=0.25$ 的理由是工程性的、也是可复现性要求：整条复现命令（两次求解
＋出图＋渲染本文数字表）必须留在**一条有界的单核命令**里，任何人 clone 后都能在分钟级
重建记录。$t=0.25$ 由 `full` 构型与 32×32/64×64 自收敛一起跑，属分钟级以上的计算，
递交计算队列执行。

因此本文档的结论边界是：**在 $t=0.1$、32×32 P1 上**各门成立；这不是对 $t=0.25$ 的
声明，$t=0.25$ 的证据要等 `full` 记录。

---

## ② 怎么做的

- **空间离散**: 模态 Legendre DG（P1/P2），张量积基，L2 投影初值
- **数值通量**: HLLC（Toro 三波求解器）
- **时间积分**: SSP-RK3
- **限幅器**: TVB（Cockburn-Shu 1989），$M = 1.0$
- **正性**: 初值经 `_positivity_filter()`（Zhang-Shu 缩放）保证容许；运行时
  `_admissible_cells()` 逐单元检查
- **测量**: 正性用**独立**的求积重构（`_check_positivity`，非求解器自带的容许性判据），
  避免"用被测对象自己的检查证明被测对象"
- **通量平衡**: 追踪四类守恒量（质量、两向动量、能量）的边界通量积分，验证离散恒等式
  $Q(t) - Q(0) = \int_0^t F_{\text{boundary}}$
- **门判定**: `judge(rows)` 是纯函数——只吃测量行，不吃任何预设结论；`NOT_RUN` 不计 PASS
- **图**: 密度云图由**已经测量过的行**直接绘制（旧版为了画图又重解了一遍，既浪费一次完整
  求解，也让"图"和"记录行"成了两条不同轨迹；现在 `provenance.figure_source_row` 指向存图的行）

---

## ③ 计算结果

下表**由 `metrics.json` 程序化渲染**（`--sync-doc`），每行标出数据在记录里的路径。
测试 `test_document_number_block_is_rendered_from_record` 逐字节比对，手改任何一格都会失败。

<!-- DOC-NUMBERS:BEGIN (由 `python -m benchmarks.riemann_2d_cks1998 --quick --sync-doc` 从 metrics.json 渲染，禁止手改) -->
| 量 | metrics.json 路径 | 值 |
|---|---:|:---:|
| profile | `profile` | quick |
| 物理构型 ncells | `runs[0].ncells` | 32 |
| 物理构型 degree | `runs[0].degree` | 1 |
| 物理构型 final_time | `runs[0].final_time` | 0.1 |
| 物理构型 cfl | `runs[0].cfl` | 0.2 |
| 物理构型 时间步数 | `runs[0].steps` | 301 |
| C1 rho_min（格点最低值，含周期控制行） | `verdicts.C1_positivity.worst_rho_min` | 6.808e-02 |
| C1 p_min（格点最低值，含周期控制行） | `verdicts.C1_positivity.worst_p_min` | 7.300e-02 |
| C1 判定 | `verdicts.C1_positivity` | PASS |
| C3 判定 | `verdicts.C3_finite` | PASS |
| 物理构型 初始质量 | `runs[0].initial_mass` | 0.67565 |
| 物理构型 末态质量 | `runs[0].total_mass` | 0.758495 |
| 物理构型 质量变化（物理边界通量，不是守恒误差） | `runs[0].conservation.mass.change` | 0.0828453 |
| C4 通量平衡最坏相对残差 | `verdicts.C4_flux_balance.worst_relative_residual` | 1.203e-14 |
| C4 判定 | `verdicts.C4_flux_balance` | PASS |
| 周期控制 时间步数 | `runs[1].steps` | 299 |
| C5 周期控制最坏相对漂移 | `verdicts.C5_periodic_drift.worst_relative_change` | 2.892e-14 |
| C5 判定 | `verdicts.C5_periodic_drift` | PASS |
| C6 最坏质量界比（|dM| / 先验通量界） | `verdicts.C6_physical_magnitude.worst_mass_bound_ratio` | 8.004e-03 |
| C6 最坏极值因子（max(rho,p) / 初值极值） | `verdicts.C6_physical_magnitude.worst_extremum_factor` | 1.19361 |
| C6 判定 | `verdicts.C6_physical_magnitude` | PASS |
| C2 判定 | `verdicts.C2_self_convergence` | NOT_RUN |
| 结果图文件 | `provenance.figure` | riemann_2d_density_p1_32.png |
| 结果图取自的记录行 | `provenance.figure_source_row` | 0 |
<!-- DOC-NUMBERS:END -->

读表要点：

- **质量变化**是透射边界下的物理边界通量，不是守恒误差。判定守恒的是 **C4 通量平衡
  恒等式**（离散层面 $Q(t)-Q(0)$ 与边界通量积分之差），它必须在机器精度量级闭合；
- **C6 质量界比**把"质量变化"除以一个**先验**通量界
  `MASS_FLUX_SAFETY · ρ_max0 · s_max0 · P · t`（只由四个初始状态与几何算出，
  见 `a_priori_bounds()`），所以它不会随着跑出什么结果而自动通过；
- **C6 极值因子**是 `max(ρ_max/ρ_max0, p_max/p_max0)`，检验解没有跑到初值极值的
  1.5 倍之外（这是**合理性**因子，不是极值原理：二维 Euler 的熵解没有严格最大模原理，
  激波聚焦可以把密度压到初值之上）。

---

## ④ 结果图

密度云图（全域 + 波相互作用区放大）由驱动在跑完后自动生成，随记录一起存放在
`docs/benchmarks/results/riemann_2d_cks1998/` 下。文件名与它取自哪一行见 ③ 表末两行
（`provenance.figure` / `provenance.figure_source_row`，同样由记录渲染，正文不手抄）。

图与记录行的绑定：`provenance.figure_source_row` 指向 `runs` 中绘制所用的行，测试校验该行
存在、是物理构型、且网格/阶数与文件名一致（`test_figure_comes_from_a_recorded_row`）。

---

## ⑤ 注意点

### TVB 限幅器与正性

TVB（minmod 族）限幅器控制振荡幅度，但**不保证** Gauss 点上的 $\rho>0$、$p>0$。
本实现通过初值 `_positivity_filter()` 保证初始容许，并用独立求积重构逐点检查正性；
正性是否成立看 ③ 表中 C1 行与判定（记录同时给出**格点最低值**和**单元平均值最低值**，
周期控制行也在统计范围内，所以最低值可以低于初值最小值 0.138——这是格式的欠冲，
C1 要求的是 > 0 而不是"不欠冲"）。

### 通量平衡 vs 质量守恒

旧版文档把透射构型的质量变化（物理量）当成"守恒误差"并给出一个从未被代码算过的百分比，
这一版把两者彻底分开：**C4 只判定离散通量平衡恒等式**，质量变化本身照实报告在 ③ 表里，
不做"通过/不通过"判定（`verdicts.C4_flux_balance.detail` 写明了这一点）。

### 周期控制

周期域上无边界通量，四类守恒量的漂移应到机器精度量级。C5 的尺度取
`max(|边界通量|, |总质量|)`，因此它是相对漂移而不是绝对残差。

### 记录与文档的一致性

本文档 `metrics.json` 里的数字全部由
`python -m benchmarks.riemann_2d_cks1998 --quick --sync-doc` 写入。改动求解器或驱动之后
**必须**重跑该命令并提交新的记录；否则 `tests/test_riemann_2d_cks1998_doc_binding.py`
会失败——这是有意的：文档不允许比记录新，记录不允许比代码新（`judge(runs)` 必须复现
记录里的 `verdicts`）。

---

## ⑥ 误差判定

### 与解析解/文献的比较：**本算例无法做**

Lax & Liu 1998 Config 3 在 $t>0$ 没有闭式解，CKS1998 论文给出的也是图形与定性描述，
不存在可作为 % 误差基准的参考场。因此本文档**不声明**"关键量 vs 基准 <3%"——
没有基准就无法计算该百分比，任何这样的声明都会是编造的。取而代之的是下列可机器判定的
最低判据（全部由 ③ 表渲染结果给出 PASS/FAIL/NOT_RUN，正文不重复数字）：

| 门 | 描述 | 判据 |
|---:|------|-----:|
| C1 | 正性：格点与单元平均 $\rho>0$、$p>0$（含周期控制行） | 严格 > 0 |
| C2 | 网格自收敛：64×64 与块平均后 32×32 的密度相对 $L1$ | < 0.05 |
| C3 | 有限性：无 NaN/Inf | — |
| C4 | 离散通量平衡：$|Q(t)-Q(0)-\int F|$ / 尺度 | < 10⁻¹⁰ |
| C5 | 周期控制：同数据、无边界时的相对漂移 | < 10⁻¹² |
| C6 | 物理量级：质量界比 ≤ 1.0，极值因子 ≤ 1.5（界由初值先验给出） | 见左 |

判据在**任何运行之前**写定于 `benchmarks/riemann_2d_cks1998.py::GATES`，记录里原样存留
（`metrics.json.gates`），所以"跑完再调容差"在结构上做不到。

**C2 的现状**：`quick` 构型只有一套网格，C2 记录为 `NOT_RUN`（不是 PASS）。
真自收敛需要 32×32 与 64×64 两套网格，下面的完整命令实现了它，但那是分钟级计算，
必须递交计算队列：

```bash
OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998        # full profile，含 C2
```

C2 未跑之前，本算例的结论边界是：**格式在完整性（C3）、正性（C1）、离散守恒恒等式
（C4）、无边界的守恒性（C5）与物理量级（C6）上是自洽的；网格收敛性（C2）尚未在
本记录中成立**。这是本锚点当前的诚实边界。

---

## 复现命令

```bash
# 冒烟 + 生成记录 + 同步本文档的数字表
OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998 --quick --sync-doc

# 完整收敛研究（P1/P2, 32x32 和 64x64，含 C2）——分钟级，递交计算队列
OMP_NUM_THREADS=1 python -m benchmarks.riemann_2d_cks1998

# 回归测试（驱动 / 守恒 / C6 门 / 文档绑定）
OMP_NUM_THREADS=1 python -m pytest tests/test_riemann_2d_cks1998.py \
    tests/test_riemann_2d_conservation.py tests/test_riemann2d_c6_divergence_gate.py \
    tests/test_riemann_2d_cks1998_doc_binding.py -q
```

## 产物清单

| 文件 | 内容 |
|------|------|
| `benchmarks/riemann_2d_cks1998.py` | 基准驱动（quick/full 模式，C1–C6 门，`judge()` 纯函数，文档数字渲染 `render_doc_numbers_block()` / `--sync-doc`） |
| `tests/test_riemann_2d_cks1998.py` | 求解器回归测试（有限性、初值、有界振荡、时间推进） |
| `tests/test_riemann_2d_cks1998_doc_binding.py` | 文档 ↔ 记录 ↔ 判定 ↔ 图 的三角绑定 |
| `docs/benchmarks/results/riemann_2d_cks1998/metrics.json` | 数值结果 + 门判定 + 出处（`provenance`） |
| `docs/CASE-DG-CKS1998-2D-RIEMANN.md` | 本报告（数字由记录渲染） |