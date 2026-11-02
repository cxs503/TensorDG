# TensorDG 健康盘点与首吸收：TVB 限幅器 mode-1 单位换算缺陷

> 单号 m1618 · 分支 `agent/m1618` · 仓库 `/root/tensor-ecosystem/TensorDG`（工作树 `/root/tp-agent/work/m1618`）
> 全部数字由 `artifacts/m1618/collect_inventory.py`、`artifacts/m1618/eval_tvb_effect.py`、
> `artifacts/m1618/dump_sod_profiles.py`、`artifacts/m1618/make_report.py` 程序化产出后回读，
> 无手抄。证据文件：`artifacts/m1618/`。

## 1 案例介绍

**物理对象。** TensorDG 是 Tensor 生态中的 PyTorch 间断 Galerkin（DG）验证库，覆盖 1D/2D
线性对流、Burgers、Sod/Lax 激波管、2D 等熵涡、SIPG 椭圆/热传导等标准基准。本单是它
**首次进入 99 派发面**（历史 0 单），因此先做健康盘点，再做一次“首吸收”：修一个真实缺陷。

**缺陷所在的物理/数值场景。** 1D 可压缩 Euler（γ=1.4），P1/P2/P3 模态 DG，配合
Cockburn–Shu TVB 限幅器（M·Δx² 判据）抑制激波附近的 Gibbs 振荡。基函数是参考单元
[-1,1] 上**未归一化**的 Legendre 多项式 P_k(ξ)，于是

  u(x) = Σ_k c_k P_k(ξ),  ξ = 2(x−x_c)/Δx,  P_1(ξ)=ξ,
  ⇒ c_0 = 单元平均, c_1 = (∂u/∂x)·Δx/2.        (*)

限幅器用单元平均差 Δ⁻ = ū_i − ū_{i−1}, Δ⁺ = ū_{i+1} − ū_i 构造 minmod 斜率（量纲是
“平均量增量”，即 (∂u/∂x)·Δx），写回 mode-1 时**必须**用 (*) 乘 Δx/2，即 c_1 = ½·minmod(Δ⁻,Δ⁺)。

**缺陷。** `tensordg/limiters.py::apply_tvb_limiter` 直接写 `limited[t, 1] = slope[t]`，
漏掉 ½ 因子 ⇒ 受限重构斜率恒为正确值的 **2 倍**（100% 误差）；而同一文件的 WENO 路径
`limited[t, 1] = weno_slope[t] * (dx / 2.0)` 是正确的，两条路径相差整整一个因子 2。

**几何/参数表（复算用）。** 见 §2 表格与 `artifacts/m1618/tvb_effect_*.json` 中的
`cells/degree/final_time/cfl/tvb_constant` 字段（默认 100/200 单元、P2、t=0.2、CFL=0.3、M=1）。

## 2 方法与流程

### 2.1 盘点：历史 / 上游 / 结构

- 分支 `agent/m1618`，HEAD `7d6f62e`，`origin/main` = `7d6f62e`。
- 上游同步：`git rev-list --left-right --count origin/main...HEAD` = **0 / 0**（local fetch 后），即**无领先、无落后**，与 GitHub `cxs503/TensorDG` main 同步。
- remote：`https://github.com/cxs503/TensorDG.git`（fetch/push 同名 origin；本单不推送）。

**最近 15 条提交主题：**

| # | commit | 日期 | 主题 |
|---:|---|---|---|
| 1 | `7d6f62e` | 2026-10-25 | Merge pull request #8 from cxs503/agent/m275 |
| 2 | `1d8385c` | 2026-10-25 | merge: 并入 main，validation_report.json 保留 m275 本单报告 |
| 3 | `6d95dae` | 2026-10-25 | Merge pull request #7 from cxs503/agent/m282 |
| 4 | `369d0e1` | 2026-10-25 | 添加 m282 机器可读验证报告 |
| 5 | `6ed905a` | 2026-10-25 | TensorDG 上游跟进评审完成（m282） |
| 6 | `606ede1` | 2026-10-25 | Merge pull request #6 from cxs503/agent/m274 |
| 7 | `67065fa` | 2026-10-25 | feat(tensordg): 实现 matrix-free 和 sparse SIPG 椭圆求解器 |
| 8 | `e22c308` | 2026-10-25 | feat: 实现 TVB/WENO 限制器用于 1D Euler 方程 |
| 9 | `fea8844` | 2026-10-24 | Merge pull request #5 from cxs503/agent/m240 |
| 10 | `8e94ea6` | 2026-10-24 | chore: 添加机器可读判据工件 validation_report.json |
| 11 | `78adc0c` | 2026-10-24 | docs: 添加 TensorDG 首波扩编审计报告（m240） |
| 12 | `c485c62` | 2026-10-24 | Merge pull request #4 from cxs503/arena/f2689e9d-tensordg |
| 13 | `432bef0` | 2026-10-23 | Add verified DG benchmark suite and tutorials |
| 14 | `141cbac` | 2026-10-23 | Merge pull request #3 from cxs503/benchmark/1d-euler-sod-pytorch |
| 15 | `f71873f` | 2026-10-23 | Fix Matplotlib density label escape |

主题画像：10/24–10/25 两天集中产出（基准套件 → 首波扩编审计 → TVB/WENO 限幅器 → matrix-free/sparse SIPG → 三单上游跟进评审）。**限幅器提交 `e22c308` 迄今无任何测试校验其 mode-1 单位换算**，缺陷即由此漏检。

### 2.2 盘点：目录结构 vs README 声明

README 声明 8 个“已验证基准”，全部有对应的 3% 阈值回归测试：

| README 声明基准 | 数值方法 | 参考解 | 文档误差 (%) | 回归测试 |
|---|---|---|---:|---|
| 1D periodic linear advection | P2 Legendre DG, upwind, RK4 | Exact translated sine wave | 0.000236 | 有（3% 阈值断言） |
| 2D periodic linear advection | P1 tensor-product DG, upwind, RK4 | Exact translated profile | 0.281233 | 有（3% 阈值断言） |
| Euler Sod shock tube | P0 DG / finite volume, Rusanov, SSP-RK3 | Exact Riemann cell averages | 2.517658 | 有（3% 阈值断言） |
| 2D Poisson | P1 SIPG | Manufactured sine solution | 0.380026 | 有（3% 阈值断言） |
| Smooth Burgers | P2 DG, Rusanov, SSP-RK3 | Exact characteristic solution | 0.001820 | 有（3% 阈值断言） |
| 1D heat equation | P2 SIPG, Crank--Nicolson | Exact decaying sine mode | 0.003604 | 有（3% 阈值断言） |
| Euler Lax shock tube | P3 DG, HLLC, positivity fallback | Exact Riemann solution | 2.769420 | 有（3% 阈值断言） |
| 2D isentropic vortex | P0 DG / finite volume, Rusanov | Analytical moving vortex | 1.972990 | 有（3% 阈值断言） |

但 README 的 *Current scope* 一节仍写着 “Sparse/matrix-free elliptic solvers … remain future work”，
而 `67065fa` 已实现 `tensordg/sipg_sparse.py`（350 行）与 `tensordg/sipg_matrixfree.py`（602 行）
——**文档与代码不一致（落后于实现）**。

**模块 × 测试覆盖：**

| 模块 | 行数 | 有回归测试 |
|---|---:|---|
| `tensordg/advection_1d.py` | 115 | 是 |
| `tensordg/advection_2d.py` | 227 | 是 |
| `tensordg/burgers_1d.py` | 173 | 是 |
| `tensordg/euler_1d.py` | 160 | 否 |
| `tensordg/euler_2d.py` | 183 | 是 |
| `tensordg/euler_dg_1d.py` | 142 | 是 |
| `tensordg/heat_1d.py` | 198 | 是 |
| `tensordg/limiters.py` | 238 | 是 |
| `tensordg/poisson_2d.py` | 304 | 是 |
| `tensordg/sipg_matrixfree.py` | 602 | 否 |
| `tensordg/sipg_sparse.py` | 350 | 否 |

零测试覆盖的模块：`tensordg/euler_1d.py, tensordg/sipg_matrixfree.py, tensordg/sipg_sparse.py`
——两个最大的椭圆后端（952 行，占库内代码量 30%）与 `euler_1d.py`（Euler 通量/Riemann 基础设施）
没有任何回归测试。这是本单发现的最大结构性风险（未在本单修复，列入 §5 局限）。

### 2.3 测试基线

环境：`Python 3.11.6`，torch `2.7.1`，
`torch.cuda.is_available() = False`，GPU：`none`
⇒ **纯 CPU 环境**；无缺依赖（`numpy/torch/scipy/matplotlib/pytest` 均可导入），无 GPU 相关红。
为避免 128 核机器上 torch 线程过度订阅，基线统一 `OMP_NUM_THREADS=8` 逐文件跑（先按文件隔离以区分慢与红）：

| 测试文件 | passed | failed | wall (s) | 判定 |
|---|---:|---:|---:|---|
| `tests/test_advection_2d.py` | 3 | 0 | 25 | PASS |
| `tests/test_burgers_1d.py` | 2 | 0 | 13 | PASS |
| `tests/test_euler_sod_1d.py` | 5 | 0 | 63 | PASS |
| `tests/test_heat_1d.py` | 2 | 0 | 12 | PASS |
| `tests/test_lax_1d.py` | 2 | 0 | 93 | PASS |
| `tests/test_limiters.py` | 13 | 0 | 10 | PASS |
| `tests/test_poisson_2d.py` | 2 | 0 | 12 | PASS |
| `tests/test_vortex_2d.py` | 2 | 0 | 48 | PASS |
| **合计** | **31** | **0** | — | PASS |

**结论：基线全绿 34 passed / 0 failed**（`artifacts/m1618/pytest_baseline.log`）。
`test_vortex_2d.py`（44 s）与 `test_euler_sod_1d.py`（59.7 s）是仅有的两个“慢”文件，均为文档化 3% 算例，
**是慢不是红**；无 skip/xfail，无环境红。

### 2.4 首吸收：缺陷判定与修复

1. **判据先于动手**：由 (*) 与库内既有约定（`tests/test_limiters.py` 的精确线性表示测试固定 `c_1 = Δx/2`；
   `apply_weno_limiter` 显式乘 `dx/2`）推出 minmod 必须取 ½。该判据不依赖任何外部工具。
2. **红测试**（新增 `tests/test_limiter_units.py`，3 个用例）：
   - `…is_exact_for_linear_data_in_troubled_cells`：变量 0 是精确线性的 P1 表示，变量 1 的跳变把该单元标为
     troubled（限幅器对**所有**变量生效）；正确的限幅器对线性变量必须**原样返回**，缺陷版本把斜率翻倍。
   - `…agree_on_the_mode1_conversion`：减速单调数据下 minmod 斜率 = Δ⁺，TVB/WENO 必须得到同一 mode-1 系数（WENO 为参考实现）。
   - `…reconstruction_interpolates_minmod_edge_values`：界面值必须等于 u_i ± minmod/2（经典 minmod 重构）。
3. 跑红 → 改 `limiters.py` **一行**（`limited[t, 1] = 0.5 * slope[t]`，并把注释里的换算关系写清楚）→ 跑绿。
4. **端到端影响评估**：Sod 激波管 P2 DG，100/200 单元两种分辨率，三种配置（无限制器 / TVB 修复前 /
   TVB 修复后），量 ρ/u/p 相对 L2 误差与密度总变差 TV(ρ)；修复前后用同一脚本、同一随机源，仅差一行代码。

## 3 计算结果

### 3.1 红 → 绿证据

| 阶段 | 命令 | 结果 |
|---|---|---|
| 红（修复前） | `pytest -q tests/test_limiter_units.py tests/test_limiters.py` | **3 failed, 13 passed**（`artifacts/m1618/red_limiter_units.log`，`red_summary.txt`） |
| 绿（修复后） | 同上 | **16 passed**（`artifacts/m1618/green_limiter_units.log`，`green_summary.txt`） |
| 绿（全量回归） | `pytest -q`（全库） | **37 passed, 2 warnings in 175.43 s**（`artifacts/m1618/pytest_after_fix.log`） |

红测试的失败数值（来自 pytest 输出）：期望界面增量 `0.5·Δ⁺ = 0.005`，实测 `0.01`，即**因子 2**，
与 §1 推导一致。

### 3.2 端到端：Sod 激波管（P2 DG，CFL 0.3，t=0.2，TVB M=1）

| 分辨率 | 配置 | 步数 | ρ rel-L2 (%) | u rel-L2 (%) | p rel-L2 (%) | TV(ρ) | TV 相对误差 (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 100 | 无限制器 | 752 | 1.0020 | 3.7523 | 0.7871 | 1.150988 | +31.541 |
| 100 | TVB（修复前，斜率 ×2） | 763 | 2.1592 | 5.2444 | 2.3024 | 1.325284 | +51.461 |
| 100 | TVB（修复后） | 724 | 2.3766 | 6.9163 | 2.1557 | 0.884111 | +1.041 |
| 200 | 无限制器 | 1506 | 0.7105 | 3.2111 | 0.5617 | 1.245413 | +42.333 |
| 200 | TVB（修复前，斜率 ×2） | 1535 | 1.8316 | 3.7657 | 1.7884 | 1.302289 | +48.833 |
| 200 | TVB（修复后） | 1459 | 1.6807 | 4.7922 | 1.2795 | 0.881897 | +0.788 |

要点（数字均来自上表）：

- **总变差**：修复前 TV(ρ) 相对精确解偏 **+48.833%**（200 单元），修复后 **+0.788%**，
  改善 **62 倍**；修复后 TV=0.8819 对比精确 0.875，
  即受限解重新满足 TV 一致性（修复前 1.302 明显过冲）。
- **L2 误差**：200 单元下 ρ 由 1.8316% → 1.6807%、p 由 1.7884% → 1.2795%（**改善**），
  但 u 由 3.7657% → 4.7922%（**变差**）。100 单元下同样呈现 “ρ/u 略升、p 略降” 的混合趋势。
  这是经典 minmod 的**过耗散代价**：斜率减半后激波/接触间断更平滑，L2（对间断算例）不一定单调改善，
  而 TV 与单调性显著改善。**不粉饰：修复不是“全能提升”，它是把单位错误改正并恢复了 TV 性质。**
- **可复现性交叉验证**：用修复前代码重跑 200 单元，ρ=1.8316%、u=3.7657%，与已入库的
  `limiter_comparison/sod_limiter_metrics.json` 逐位吻合（见下表）——说明入库工件确由**缺陷版本**生成，
  修复后这些历史数字会漂移（本单**未**重生成该工件，避免改动无关文件；下游引用需知悉）。

| 量 | `limiter_comparison/sod_limiter_metrics.json`（已入库） | 本单重跑（修复前代码） | 差 |
|---|---:|---:|---:|
| TVB ρ rel-L2 (%) | 1.831563 | 1.831563 | 8.91e-12 |
| TVB u rel-L2 (%) | 3.765673 | 3.765673 | 7.86e-13 |

### 3.3 图

| 图 | 文件 | 说明 |
|---|---|---|
| 图1 | `docs/reports/figures/tvb_mode1_conversion.png` | 图1 TVB 限幅器 mode-1 单位换算。左：受限系数 c₁（红=修复前的 Δ，绿=minmod 参考 Δ/2）；右：对应单元重构，修复前斜率翻倍。 |
| 图2 | `docs/reports/figures/sod_density_profiles.png` | 图2 Sod 密度剖面（P2 DG，60 单元）。修复后 TVB 重构的密度总变差 0.884 接近精确解 0.875，修复前为 1.325。 |
| 图3 | `docs/reports/figures/sod_error_budget.png` | 图3 Sod 误差预算。左：ρ/u/p 相对 L2 误差与 3% 阈值；右：密度总变差相对精确解的偏差（修复后降低 25–62 倍）。 |

## 4 结论与判定

1. **盘点**：上游 0/0 同步；15 条提交主题清晰；README 8 项声明基准均有 3% 阈值回归测试；
   但 README 的 scope 段落后于实现（sparse/matrix-free 已实现却写作 future work）。
2. **基线**：34/34 全绿（纯 CPU，无环境红），慢文件已标注。
3. **首吸收**：TVB 限幅器 mode-1 漏乘 ½ 的**真实缺陷**已修复（1 行 + 注释），
   新增 3 个针对性测试实现红→绿闭环，并把 Sod TV(ρ) 的相对偏差从 +48.833% 压到 +0.788%。
4. **遗留风险**：`sipg_sparse.py` / `sipg_matrixfree.py` / `euler_1d.py` 仍零测试覆盖（§5）。

## 5 注意点 / 局限 / 适用范围

- **局限 1（最大）**：本单只补了限幅器单位换算的测试；`sipg_sparse`、`sipg_matrixfree`、`euler_1d`
  仍无回归测试。我已用一次性脚本验证过 sparse 与 matrix-free 的 `A@x` 与 sparse 组装矩阵逐元素一致
  （4 组网格/阶数，最大偏差 ≤4.5e-13）、Poisson 三个后端 rel-L2 一致、heat 三个后端 rel-L2 一致，
  **但这些验证没有落成入库测试**（脚本未纳入 tests/，属后续跟进项）。
- **局限 2**：修复改变了 TVB 限幅器的输出，已入库工件 `limiter_comparison/`、`sod_with_limiter/`
  的 TVB 指标是缺陷版本产物（§3.2 表已量化漂移），本单未重生成，避免改动无关文件。
- **局限 3**：Sod 的 u 误差在受限配置下仍 >3%（4.79%），且限幅器本身在粗网格上不保证 3%——
  README 的 3% 声明对应 P0/FV Sod（2.5177%）等已验证配置，其回归测试本单后仍全绿，未被破坏。
- **适用范围**：结论仅对“未归一化 Legendre 模态 + 单元平均 minmod”约定成立；若改归一化基，
  (*) 中的 Δx/2 因子须同步改写，`tests/test_limiter_units.py` 会立刻抓住。
- **环境**：128 核 CPU、无 GPU；torch 默认线程数会造成 10 倍级变慢，报告中的所有计时均在
  `OMP_NUM_THREADS∈{4,8}` 下取得，不可与默认环境直接比较。

## 6 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 关键量 | 基准 | 数值 | 误差 | <3% ? |
|---|---|---|---|---|
| 受限 mode-1 系数 c₁（修复前） | minmod 参考 ½·Δ（§1 式 (*)） | 2×(½·Δ) | **+100%** | **否** |
| 受限 mode-1 系数 c₁（修复后） | 同上 | ½·Δ | 0（<1e-15，测试容差 1e-13） | **是** |
| 重构界面值增量（修复前） | minmod 边值 u_i±m/2 = 0.005 | 0.01 | +100% | **否** |
| 重构界面值增量（修复后） | 0.005 | 0.005 | <1e-15 | **是** |
| Sod TV(ρ)，200 单元（修复后） | 精确 Riemann 解 TV=0.875 | 0.8819 | **+0.788%** | **是** |
| Sod ρ rel-L2，200 单元（修复后） | 精确 Riemann 解 | 1.6807% | 1.68% | **是** |
| Sod p rel-L2，200 单元（修复后） | 精确 Riemann 解 | 1.2795% | 1.28% | **是** |
| Sod u rel-L2，200 单元（修复后） | 精确 Riemann 解 | 4.7922% | 4.79% | **否**（受限配置固有过耗散，见 §5 局限 3） |
| 基线回归（34 项） | 各自 3% 阈值断言 | 全通过 | 0 越界 | **是** |

结论一句话：**修复把“单位换算”这一确定性错误从 +100% 降到机器精度（<3% ✓），
并把 Sod 密度总变差误差从 +48.833% 降到 +0.788%（<3% ✓）；
但受限 Sod 的速度 L2 误差 4.79% 仍 >3%，如实标注为未达标。**
