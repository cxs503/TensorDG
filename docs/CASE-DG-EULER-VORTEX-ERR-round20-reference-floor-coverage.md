# CASE-DG-EULER-VORTEX-ERR round 20：参考解地板的覆盖面（与真实最不利点）

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜算例池·等熵涡移动误差守恒｜ref: Yee, Sandham & Djomehri (JCP 162, 2000)｜容差 6.0%（t = 1 快照）

## ① 案例与本节问题

等熵涡在周期域上平移，误差读数**全部**是 `||u_h - R||` 的比值。`R` 用最小镜像折返（`min_image`）求值时丢掉了邻居镜像的尾巴，因此每个读数都带一个参考解缺陷。round 7 量到了这个缺陷（`O(tail)` 进速度、`O(tail^2)` 进密度/压力），并把它写成**最坏情况上界**：

```
|e_lattice - e_published| / e_published  <=  floor / e_published
```

本节问的不是「地板多大」（round 7 已答），而是**这个上界覆盖了谁**：已发布的地板工件是否对每一条已发布读数都给过界？

## ② 怎么做的

1. **把检查点来源从硬编码阶梯改为记录自身**。原实现遍历常量 `LADDER`，而 `LADDER` 与实际落盘的读数在随后几轮里出现分叉：工件里 11 条记录（`dg_d1_n32` 长时、`dg_d1_n64` 全部、`dg_d3_n32` 全部、`dg_d2_n32_t20`）写着 `published_errors = null` —— 地板算出来了，**却没有和任何实测比**。新实现 `configs_from_results()` 逐条读 `results/*.json` 里记录的 `config`（它才是该次计算的权威，含方向），因此不可能再漏族。
2. **修一个方向 bug**。原 `floor_at()` 用模块常量 `BACKGROUND = (1, 1)` 求参考，而 C10 方向轴的记录沿 `axis0` / `obl30` 平移；折返折叠依赖位移方向，所以对方向族算出的地板是**另一个算例的**。现按记录自身的 `background_velocity` 与 `strength` 求值。
3. **把覆盖面变成断言**：`coverage.uncovered` 必须为空，并新增回归测试 `tests/test_vortex_reference_upgrade.py`。
4. **量化实测位移的工具**（`benchmarks/vortex_reference_upgrade.py`）：上界是最坏情况（假设参考缺陷与解误差同向），实测才是裁决。工具在同一检查点上比较 `min_image`（已发布）与 `lattice`（精确镜像和）的重跑读数，并带一个**同条件重跑控制**（`min_image` 再跑一次）把求解器噪声与参考效应分开。

## ③ 计算结果（程序化注入，零手抄）

覆盖面：`{'records_with_config': 40, 'compared': 40, 'uncovered': []}`。全部 40 条检查点都有「地板 / 实测」比值，`uncovered = []`。

### 3.1 最不利检查点（上界最大处）

| 排名 | 检查点 | 方向 | t | 像半宽 K | 速度地板 | 实测最大误差 | 地板/实测 | 主导场 |
|---|---|---|---|---|---|---|---|---|
| 1 | dg_d3_n32_t1_cfl0.2 | diag45 | 1 | 2 | 1.476e-04% | 0.0006% | 0.2531 | velocity_x |
| 2 | dg_d3_n32_t2_cfl0.2 | diag45 | 2 | 2 | 1.449e-04% | 0.0006% | 0.2477 | velocity_y |
| 3 | dg_d3_n32_t5_cfl0.2 | diag45 | 5 | 2 | 1.464e-04% | 0.0006% | 0.2462 | velocity_x |
| 4 | dg_d3_n32_t10_cfl0.2 | diag45 | 10 | 3 | 1.464e-04% | 0.0006% | 0.2383 | velocity_x |
| 5 | dg_d2_n64_t1_cfl0.2 | diag45 | 1 | 2 | 1.464e-04% | 0.0011% | 0.1274 | velocity_y |
| 6 | dg_d2_n64_t2_cfl0.2 | diag45 | 2 | 2 | 1.460e-04% | 0.0012% | 0.1267 | velocity_y |
| 7 | dg_d2_n64_t10_cfl0.2 | diag45 | 10 | 3 | 1.464e-04% | 0.0012% | 0.1196 | velocity_y |
| 8 | dg_d2_n32_t1_cfl0.2_dirobl30 | obl30 | 1 | 2 | 2.056e-04% | 0.0134% | 0.0158 | velocity_x |

**结论 1（实质修正）**：最不利点是 `dg_d3_n32_t1_cfl0.2`，地板/实测 = **25.31%**；round 7 工件写下的最不利点是 `dg_d2_n64_t1_cfl0.2`（12.74%），它只是**被覆盖子集**的最不利点：`dg_d1_n64`、`dg_d3_n32`、`dg_d1_n32` 长时与 `dg_d2_n32_t20` 共 11 条记录当时在地板工件里没有比对（`published_errors = null`）。这不是数值变化，是覆盖面变化：p = 3 在 32² 上解误差更小，所以同一地板在它身上占的比例更大。

### 3.2 地板的方向依赖（C10 轴）

同一族（DG-P2 / 32² / CFL 0.2）、同一速度模长 |u∞| = √2，只改平移方向：

| 方向 | t | 速度地板（相对 L2） | 实测最大误差 | 地板/实测 |
|---|---|---|---|---|
| axis0 | 1 | 8.826e-04% | 0.0597% | 0.0156 |
| axis0 | 2 | 9.077e-04% | 0.0611% | 0.0150 |
| axis0 | 5 | 9.134e-04% | 0.0686% | 0.0133 |
| axis0 | 10 | 9.030e-04% | 0.0887% | 0.0102 |
| diag45 | 1 | 1.446e-04% | 0.0094% | 0.0153 |
| diag45 | 2 | 1.462e-04% | 0.0097% | 0.0152 |
| diag45 | 5 | 1.464e-04% | 0.0111% | 0.0132 |
| diag45 | 10 | 1.464e-04% | 0.0152% | 0.0097 |
| diag45 | 20 | 1.464e-04% | 0.0252% | 0.0058 |
| obl30 | 1 | 2.056e-04% | 0.0134% | 0.0158 |
| obl30 | 2 | 2.016e-04% | 0.0138% | 0.0148 |
| obl30 | 5 | 2.039e-04% | 0.0158% | 0.0136 |
| obl30 | 10 | 2.018e-04% | 0.0211% | 0.0099 |

**结论 2**：地板与方向强相关——`axis0` 最大 9.134e-06，`diag45` 1.464e-06，`obl30` 2.056e-06。轴向（`axis0`）比斜向（`obl30`）大约 4 倍：轴向折返把涡心折到单坐标的 |shift| = 5 处，丢掉的镜像离得最近；`obl30` 的两个分量都落在 2.9 与 7.1 附近，最近被丢镜像远得多。因此 **C10 的三条方向读数并不共享同一个参考解不确定度**，「方向不变性」的判据必须逐方向带自己的界（round 7 的工件只对 `diag45` 给过界）。

### 3.3 界是否仍然是界

全部 40 条检查点的 `地板/实测` 最大值 = 0.2531 < 1，即没有任何一条读数的判据被参考解接管（地板没有超过实测误差本身）。

## ④ 结果图

`fig_reference_floor_coverage.png`（随报告内嵌）：左 = 逐检查点「地板/实测」（对数轴，按族着色），红色虚线 = 1（判据被参考解接管的临界）；右 = 三条方向的地板随时间的双对数曲线，显示 `axis0` 比 `obl30` 高两个量级。

## ⑤ 注意点 / 局限

1. **上界不是实测**。3.1 与 3.2 的比值是**最坏情况**：它假设参考缺陷与解误差完全同向。真实位移由 `benchmarks/vortex_reference_upgrade.py` 量；本单已递交lattice 参考重跑（`cj4355` = dg_d2_n64 t=1/2/5/10，`cj4361` = dg_d3_n32 t=1/2/5/10，`cj4357` = 同条件 min_image 重跑控制），但队列深、窗口内未落盘，**故本报告不给出任何实测位移数字**，只给出上界与其覆盖面。round 7 在 32² 设计点上量到：换参考只动速度、位移 `7.6e-6`（相对），是当时地板比值的 5e-4 —— 说明上界确实非常保守。
2. **`lattice` 不是 Euler 精确解**：镜像和精确的是线性叠加，非线性交叉项为 `O(tail^2) ~ 1e-11`，与地板本身同量级，故只用于速度场。
3. **容差出处**：6% 是算例池条目原文照搬，未从 Yee 等（2000）正文重新推导。
4. **本节的修正方向是「补覆盖」，不是「改判据」**：C1–C11 的判据文本、容差与已落盘读数一律未动，重算后的 `max_floor_over_error` 仍 < 1。

## ⑥ 误差判定（逐项对基准，显式回答「是否 < 3%」）

| 关键量 | 基准 | 实测/读数 | 相对基准 | 是否 < 3% |
|---|---|---|---|---|
| 设计点 DG-P2/32² t=1 最大场相对 L2 | 等熵涡闭式解 | 0.009431% | 6% 锚点容差 | 是 |
| C1 最差族 p0_n64_t1 | 等熵涡闭式解 | 4.9260% | 6% 锚点容差（余量 1.22×） | **否**（4.93% > 3%，但通过 6% 口径） |
| 参考地板上界（最不利 dg_d3_n32_t1_cfl0.2） | 该记录实测误差 | 25.31% | 1（上界不得 ≥ 实测） | 是（< 1） |
| 参考地板上界（design point dg_d2_n32_t1） | 该记录实测误差 | 1.53% | 1 | 是 |
| 方向间地板差异（axis0 vs obl30） | 各自实测误差 | 4× | — | 不适用（比值，非误差） |

## ⑦ 复现

```bash
PYTHONPATH=. python3 benchmarks/vortex_reference_floor.py \
    --results-dir case_euler_vortex_errconserve \
    --output case_euler_vortex_errconserve_lattice/reference_floor.json \
    --markdown case_euler_vortex_errconserve_lattice/reference_floor.md \
    --figures
PYTHONPATH=. python3 -m pytest tests/test_vortex_reference_upgrade.py -q
PYTHONPATH=. python3 gen_report_m22770.py
```

实测位移（队列落盘后）：

```bash
run_compute fetch cj4355   # lattice：dg_d2_n64 t=1/2/5/10
run_compute fetch cj4361   # lattice：dg_d3_n32 t=1/2/5/10
run_compute fetch cj4357   # min_image 重跑控制：dg_d2_n64 t=1,10
PYTHONPATH=. python3 benchmarks/vortex_reference_upgrade.py \
    --rerun-dir case_euler_vortex_errconserve_rerun --figures
```

## ⑧ 顺带闭合：已提交的判决与已提交的证书不同步

`tests/test_metrics_provenance_freshness.py` 在本单开局时是**红**的（3 条断言失败），
原因是 round 19 落地的那份重放证书从未触发 `--aggregate` 重算：仓库里已提交的
`metrics.json` 仍写着 `replayed: 18 / unverified: 1`，而同一提交里的证书已支持 19 份重放，
对应 `C11 = not_evaluated`、整包 `INCOMPLETE` —— 与 round 19 报告宣称的 `PASS` 不一致。
本单重跑 `--aggregate --figures` 后：`verdict = PASS`、`all_passed = True`、`failed = []`、
`C11 = passed`（最差重放偏差 1.133e-12，门槛 1e-9），`tests/test_metrics_provenance_freshness.py`
与 `tests/test_case_euler_vortex_errconserve.py` 共 49 条全部通过。
**教训**：证书是独立落盘的数据，判决只是投影；任何证书落地后不重算聚合器，
仓库里的判决就会落后于自己的证据，而这条现已由回归测试把守。
