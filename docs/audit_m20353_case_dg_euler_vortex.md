# 交差审计 · CASE-DG-EULER-VORTEX-ERR（锚点任务 m20332，审计任务 m20353）

> 本报告的数字全部由 `tools/audit_case_dg_euler_vortex_err.py` 从锚点工件（`case_euler_vortex/*.json`、`convergence.csv`、求解器返回的模态系数）程序化重算后由 `tools/publish_audit_m20353.py` 渲染，无手抄。判据来自代码常量与实际工件，不采信报告自述。

## ① 案例介绍（物理背景 / 参数表）

二维可压缩 Euler 方程的等熵涡（isentropic vortex）是**解析精确解**：均匀来流平移输运一个等熵涡，t>0 的精确解即初始剖面的平移。锚点做的是**移动误差是否随网格收敛**与**全域守恒量是否精确守恒**两件事，二者都是格式正确性的基本判据。

| 项目 | 取值 |
|---|---|
| 方程 | 2D Euler, ideal gas $\gamma=1.4$ |
| 计算域 | $[-5,5]^2$，四周周期 |
| 涡强度 $\beta$ | 5（峰值切向 Mach 0.7958）|
| 来流 $(u,v)_\infty$ | (1,1) |
| 末时刻 | t=1（敏感性另测 t=0.5, 2）|
| 通量 | DG: HLLC；P0 基线: Rusanov |
| 时间推进 | SSP-RK3 |
| 验收容差 | 各原始变量相对 $L_2$ < 5%（`benchmarks/case_euler_vortex.py:TOLERANCE`）|
| 守恒容差 | < 1e-12（机器精度量级）|
| 外部锚点 | Shu, ICASE Report 97-65 / NASA CR-97-206253 (1998) §4.4.2 |

## ② 怎么做的（审计方法与工具链）

1. **判据取自代码，不取自报告**：容差从 `benchmarks/case_euler_vortex.py:TOLERANCE` 解析得到；每个配置的 `gate_pass` 由 `max_relative_l2` 重新比较得出，再与存储的 flag 对账。
2. **误差阶独立重算**：$p=\log(e_{coarse}/e_{fine})/\log(\Delta x_{coarse}/\Delta x_{fine})$ 逐原始变量重算，与工件里存的 `eoc` 对照（差值 > 5e-3 判为不自洽）。
3. **守恒漂移独立重导**：不用求解器自报的 `metadata['conservation_drift']`，而是自己调用 `solve_euler_dg_2d_vortex`（t=0 得初始 L2 投影，t=1 得推进态），从模态零阶系数按单元面积加权求和重算全域积分与相对漂移。
4. **报告六段与图片存在性检查**：`docs/case_dg_euler_vortex_conservation.md` 的 ①–⑥ 是否齐全、正文引用的图片是否真在树上。
5. **工件字节绑定**：对 17 个关键工件取 sha256，保证判级绑定到确切字节。
6. **独立复现**：把两个门禁配置（DG P2 32²、P0 128²）在新节点上重跑，逐配置比对新旧误差（相对差 > 1e-6 判为不可复现）。
7. **基准台账核对**：查 `tensor_benchmark_cases` 里该锚点是否有登记行，有则逐项比容差。

## ③ 计算结果（核对数字表）

### 3.1 精度：逐配置 vs 5% 容差（重算，非自述）

| 配置 | 网格 | t | CFL | $\varepsilon_{max}$ | 占容差 | 重算判定 | 工件自述 |
|---|---|---|---|---|---|---|---|
| p0_d0 | 32² | 1 | 0.4 | 7.799898% | 156.00% | FAIL | FAIL |
| p0_d0 | 64² | 1 | 0.4 | 4.926163% | 98.52% | PASS | PASS |
| p0_d0 | 96² | 1 | 0.4 | 3.584806% | 71.70% | PASS | PASS |
| p0_d0 | 128² | 1 | 0.4 | 2.816997% | 56.34% | PASS | PASS |
| dg_d1 | 16² | 1 | 0.2 | 0.837279% | 16.75% | PASS | PASS |
| dg_d1 | 32² | 1 | 0.2 | 0.201151% | 4.02% | PASS | PASS |
| dg_d1 | 64² | 1 | 0.2 | 0.049499% | 0.99% | PASS | PASS |
| dg_d2 | 16² | 1 | 0.2 | 0.081949% | 1.64% | PASS | PASS |
| dg_d2 | 32² | 1 | 0.2 | 0.012652% | 0.25% | PASS | PASS |
| dg_d2 | 48² | 1 | 0.2 | 0.008864% | 0.18% | PASS | PASS |
| dg_d3 | 12² | 1 | 0.2 | 0.034576% | 0.69% | PASS | PASS |
| dg_d3 | 16² | 1 | 0.2 | 0.011623% | 0.23% | PASS | PASS |
| dg_d3 | 24² | 1 | 0.2 | 0.008629% | 0.17% | PASS | PASS |

门禁配置（算例池指定）dg_d2_n32_t1_cfl0.2、p0_n128_t1_cfl0.4 全部通过：`True`；全梯中唯一未过 5% 的配置是 P0 32²（7.799898%，即门禁有牙的反例）。

### 3.2 收敛阶独立重算（工件 eoc vs 重算 eoc）

| 族 | 网格对 | 重算阶（density / vx / vy / p）| 自洽 |
|---|---|---|---|
| p0_d0 | 32→64 | 0.610 / 0.665 / 0.662 / 0.572 | 是 |
| p0_d0 | 64→96 | 0.757 / 0.789 / 0.784 / 0.733 | 是 |
| p0_d0 | 96→128 | 0.822 / 0.844 / 0.838 / 0.806 | 是 |
| dg_d1 | 16→32 | 2.038 / 2.057 / 2.032 / 2.003 | 是 |
| dg_d1 | 32→64 | 2.034 / 2.023 / 2.018 / 2.013 | 是 |
| dg_d2 | 16→32 | 3.005 / 2.696 / 2.694 / 3.013 | 是 |
| dg_d2 | 32→48 | 3.082 / 0.876 / 0.877 / 3.069 | 是 |
| dg_d3 | 12→16 | 3.879 / 3.790 / 3.774 / 4.100 | 是 |
| dg_d3 | 16→24 | 3.956 / 0.734 / 0.735 / 3.950 | 是 |

重算 9 对，不自洽 0 个 → 工件里的 `eoc` 确实由存储误差导出。

### 3.3 守恒漂移独立重导（不用求解器自述）

| 配置 | 审计独立重算漂移 | 求解器自述 | 工件存储 | < 1e-12 |
|---|---|---|---|---|
| DG P1 8x8 | 2.89e-15 | 2.89e-15 | 2.89e-15 | 是 |
| DG P3 6x6 | 5.44e-15 | 5.44e-15 | 5.44e-15 | 是 |

重算与存储逐位一致（相对差 < 1e-13 判为一致），因此「守恒 ~1e-15」不是求解器自证。

### 3.4 独立复现（新节点重跑 vs 工件）

| 配置 | 工件 $\varepsilon_{max}$ | 重跑 $\varepsilon_{max}$ | 相对差 |
|---|---|---|---|
| dg_d2_n32_t1_cfl0.2 | 1.265198591e-04 | 1.265198591e-04 | 7.19e-13 |
| p0_n128_t1_cfl0.4 | 2.816997317e-02 | 2.816997317e-02 | 2.09e-15 |

共享 2 个配置，最大相对差 7.19e-13（阈值 1e-06），不一致 0 个；作业号 cj2661。

### 3.5 三件套核对

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件 | 17/17 存在于锚点树 | `case_euler_vortex/results/*.json`（18 条，含 `exact_solution_residual.json`）、`metrics.json`、`conservation_results.json`、`convergence.csv`、2 张守恒图 |
| benchmark ref vs got | 容差 5% 取自代码；门禁配置 got=0.012652%（DG P2 32²）/ 2.816997%（P0 128²） | `case_euler_vortex/metrics.json` 与逐配置结果文件 |
| repro_cmd 可执行 | **原样不可执行**（见 §⑤ F1），已修 | `tools/run_case_euler_vortex.sh` |
| 报告六段齐全 | 是（docs/case_dg_euler_vortex_conservation.md）| sha256 `dd5102696a122396…` |
| 报告引用图片存在 | 是 | ['fig_conservation_drift.png', 'fig_error_vs_conservation.png'] |
| metrics 声明图片存在 | **否**（4/4 缺失，已被前序门禁滤除，本次重新生成入库）| `case_euler_vortex/fig_*.png` |
| `tensor_benchmark_cases` 登记 | **审计前 0 行**（唯一裁判无条目），本次补登记 | 见 `artifacts/m20353/benchmark_case_row.json` |

### 3.6 结果图

**图 1｜误差 vs 网格：log-log 收敛曲线（从工件重算）与 5% 容差线**

![fig_audit_convergence.png](fig_audit_convergence.png)

**图 2｜逐配置容差余量（bar<100 即通过；重算而非自述）**

![fig_audit_tolerance_margin.png](fig_audit_tolerance_margin.png)

**图 3｜守恒漂移 vs 解误差：粗网格大误差仍机器精度守恒（解耦证据）**

![fig_audit_conservation_vs_error.png](fig_audit_conservation_vs_error.png)

**图 4｜场量云图：t=1 密度场（DG P2 32²，等熵涡已平移）**

![fig_density_field.png](fig_density_field.png)

## ④ 判级（L0–L5）

**结论：L4（基准验证）**。理由逐条绑定证据：

| 级别门槛 | 是否满足 | 证据指针 |
|---|---|---|
| L1 数学/单元验证 | 是 | 精确解残差：稳态系 3.17e-16、时空残差 1.32e-10（`case_euler_vortex/results/exact_solution_residual.json`）|
| L2 独立功能实现 | 是 | `tensordg/euler_dg_2d.py:1025 solve_euler_dg_2d_vortex`、`tensordg/euler_2d.py:241 solve_isentropic_vortex`（输入→计算→输出，无 stub）|
| L3 系统集成 | 是 | 主 DG 推进器 `solve_euler_dg_2d` 内新增守恒 instrumentation（`tensordg/euler_dg_2d.py:915-1000`）；被 `benchmarks/convergence_euler.py`、`design_order_euler_dg_2d.py`、`limiter_2d_order_study.py` 等多处调用 |
| L4 基准验证 | 是 | 完整算例梯 13 条记录 + 解析解误差分析 + EOC 自洽 + 独立重跑复现（§3.4）+ 容差门禁 + 守恒独立重导（§3.3）|
| L5 科研工程成果 | 否 | 守恒漂移为周期域数值通量逐项相消的**必然**性质（机器误差级），属格式正确性不变量，非新方法/新物理；无外部实验或文献新结论 |

数字层面：门禁配置 $\varepsilon_{max}$ 为 0.012652%，占 5% 容差的 0.25%；守恒漂移 1e-14 量级 vs 1e-12 判据。**没有虚标迹象**：自述 gate flag、eoc、漂移三项全部与独立重算一致。

## ⑤ 注意点（发现的缺陷、踩过的坑、局限）

| 级别 | 缺陷 | 证据 | 处置 |
|---|---|---|---|
| **blocking（交差链）** | `tensor_benchmark_cases` 中锚点 0 行 → 唯一裁判无条目，自动门禁无法绑定容差 | `SELECT count(*) … WHERE anchor_slug='CASE-DG-EULER-VORTEX-ERR'` = 0（审计前）| 本次按工件实测值补登记一行（got 值程序化取自动化产物）|
| **major（可复现性）** | 报告 §复现命令与 `tools/run_case_euler_vortex.sh` 的 `python3 benchmarks/case_euler_vortex.py` 在干净检出上 `ModuleNotFoundError: tensordg`（本机与集群节点 n330 各复现一次，作业 cj2650 rc=1）| 集群 cj2650 日志 | 改 `-m benchmarks.case_euler_vortex`，新增回归测试 `test_documented_repro_command_is_executable` |
| **major（证据链）** | `metrics.json` 声明的 4 张精度图在树上不存在（前序门禁 `m15349` 滤除），违反「declared ⇒ present」 | `metrics.figures` vs `git ls-files` | 用文档化 `--aggregate` 重新生成 4 图入库（重生成 `metrics.json` 与存储值逐键一致，故图表号可信）|
| **major（测试可重复性）** | 锚点守恒测试套件在集群节点 n330 上 **segfault**（`tensordg/euler_dg_2d.py:676 _admissible_cells` 内 torch einsum），且此前已有断言失败标记 | 作业 cj2649 rc=255 | 见 §未跑检查；本审计以独立重导（§3.3）替代 |
| **major（HEAD 测试红）** | 审计 HEAD 上 `tests/test_case_euler_vortex.py` 1 failed / 6 passed（`test_aggregate_builds_tables_and_gate`，作业 cj2661）| 集群 cj2661 日志 | 根因未定位；该测试用 `tmp_path` 与 `docs_dir=None`，不触碰本审计新增资产，故判为**既有缺陷，非本单引入**，如实记 unknown |
| minor | 锚点报告/守恒工件在 `agent/m20332` 分支（commit `2b48f2e`），审计时尚未并入 main：审计 HEAD `f527c8d` 上只有精度半（`case_euler_vortex/`、`docs/` 缺守恒报告）| `git merge-base --is-ancestor` = 假 | 本审计按锚点树核验，并在 HEAD 上补齐缺失图片 |
| minor | 守恒容差 1e-12 对该性质而言过于宽松（实测 1e-14），门禁对真实回归的灵敏度有限 | §3.3 | 记录不修改（不改他人验收线）|

**未跑的检查（unknown / not_run，不算通过）**：
- 守恒测试套件在集群上的全绿仍未建立（cj2649 segfault；断言失败明细未取到）。
- 非周期边界（壁面/远场）的守恒审计未跑（锚点未主张，属适用范围声明）。
- 时间误差的网格无关性未单独研究（锚点自述，本审计未复核）。
- GPU/loong64 原生长时（t=2）梯未复跑。
- HEAD 上 `test_aggregate_builds_tables_and_gate` 的失败根因未定位（未修）。

## ⑥ 误差判定（逐项显式回答「是否 <3%」）

| 关键量 | 基准 | 实测 | <3%? | <5%（算例池判据）? |
|---|---|---|---|---|
| DG P2 32² 全场最大相对 $L_2$ | 精确等熵涡 | 0.012652% | 是 | 是 |
| P0 128² 全场最大相对 $L_2$ | 精确等熵涡 | 2.816997% | 是 | 是 |
| P0 32²（反例）| 精确等熵涡 | 7.799898% | 否 | 否（门禁确实会拒）|
| 守恒漂移（独立重算）| 精确守恒 0 | 5.44e-15 | 是（判据 1e-12）| — |

**结论**：精度判据在算例池门禁配置上以 **0.0127%（DG）/ 2.82%（P0）** 远优于 3% 与 5%；守恒判据以 1e-14 优于 1e-12 四个量级。**唯一不达标项是刻意保留的反例 P0 32²（7.80%）**，它证明门禁有牙而非失败。
