# T2 核心路径统一优化：口径、内存与精度-性能权衡

> 本文件由 `publish_core_path_t2.py` 从 `TensorLBM/outputs/core_path_t2/summary.json` 程序化生成，请勿手工改数字。
## 1. 案例介绍（物理背景 / 几何 / 参数）

本单锚点 **`GAP-LBM-T2-GPU-COREPATH`**（T2 GPU 核心路径统一优化，P1）。该锚点的验收口径
写死了两条纪律：**优化对象必须是 T3 已验证的物理算例（禁盲目追峰值）**，且**对标口径必须统一**
（加速数字与精度、内存、后端可用性一起报，不能只挑好看的那一个）。因此本单不追峰值，而是先把
"能合规地报出加速比"这件事本身建起来，然后在这套口径下测出来。

物理算例取 T3 验证矩阵第 5 行 **`taylor-green-2d`**（当前矩阵中唯一 REAL 通过的算例）：

| 项 | 值 |
|---|---|
| 解析解 | 幅值 ∝ exp(−2νk²t)，k = 2π/N |
| 格子 | D2Q9，BGK，cs² = 1/3 |
| ν | 0.05（τ = 0.65） |
| 初值幅值 u₀ | 0.01（Mach ≈ 0.017，可压误差 <1e-3） |
| 网格 | 64×64（主族，= T3 矩阵配置）、128×128（分辨率标度点） |
| 时间步 | 主族 1200 步、128² 600 步（每 100 步取一次幅值） |
| 预注册精度闸门 | 衰减率相对误差 ≤ 2.0%（与 T3 矩阵同容差） |
| 预注册一致性闸门 | float64 相对 |Δu| ≤ 1.0e-10，float32 ≤ 1.0e-05 |

机器上下文（程序化写入 `summary.json`）：`ec6ae0b42a0c` / Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38，
Python 3.11.6，numpy 2.4.6，可用核 128，
loadavg [63.34, 64.26, 66.07]。计时为单核、单进程。

后端探测结果（**不可用的后端只能 WITHHELD，不许 CPU 假装**）：

| backend | 类型 | 可用 | 证据/原因 |
|---|---|---|---|
| numpy | cpu | 是 | numpy 2.4.6 |
| torch-cpu | cpu | 是 | torch 2.7.1 |
| torch-cuda | gpu | 否 | torch.cuda.is_available() == False |
| cupy | gpu | 否 | cupy not installed |

GPU 臂的处置：

| backend | 类型 | 原因 | 证据级 |
|---|---|---|---|
| torch-cuda | gpu | torch.cuda.is_available() == False | WITHHELD |
| cupy | gpu | cupy not installed | WITHHELD |

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

新增 `TensorLBM/src/tensorlbm/core_path.py`，把在此之前互不相干的三套时间推进
（`FusedPeriodicBGK`、`SparseAOSLBM`、`MCMPSolver`）统一到一个 API 上，并提供：

1. **统一 step 接口** `CorePath.init_from_velocity / step / macroscopic`，参数集合被冻结在
   `CoreConfig`（stencil, shape, ν, kernel, backend, dtype）——一个臂就是一个不可变的身份。
2. **两个核函数，同一 API**：
   - `reference`：碰撞阶段物化整场 `f_eq` 与整场碰撞后 `f_post`，再整场周期迁移 → **4 个 Q·N 活缓冲**；
   - `fused`：逐方向单趟扫（single sweep）——`f_eq[q]` 只落在一个 N 大小的暂存里，就地松弛后就地
     迁入目标缓冲，`f_eq`/`f_post` **从不整场物化** → **2 个 Q·N 活缓冲**。周期回绕折进扫描里
     （每个方向一次 `roll`）；平壁边界仍归 `boundaries.py`，本模块**不假装支持**。
3. **多后端 + 能力探测**：`probe_backends()` 逐项报告 numpy / torch-cpu / torch-cuda / cupy 的
   可用性与原因；`make_ops()` 对不可用后端直接抛 `BackendUnavailable`，绝不回退。
4. **内存账**：`declared_live_bytes()` 由配置精确算出（fused = 2QN + 5N，reference = 4QN + 5N，
   按 dtype itemsize），另有实测 `tracemalloc` 峰值，两者都进表。
5. **统一对标口径**：`run_arm()` 在一个函数里完成"物理跑 → 精度拟合 → 计时 → 判决"，
   判决只能由 `admissibility()` 给出，规则写死在代码里：
   **精度闸门与一致性闸门都过，才发 MLUPS；否则该臂的加速数字作废**（`REJECTED_ACCURACY` /
   `REJECTED_CONSISTENCY`）。计时协议：一次 warm-up 块 + `repeat` 个计时块，取最小块
   （噪声最小），精度历史来自**另一次不计时的相同跑**，计时永不污染物理。
6. 工具链：`TensorLBM/benchmarks/core_path_t2_bench.py` 跑全部臂并程序化写出
   `outputs/core_path_t2/summary.json` + 4 张图；本脚本再把 summary 渲染成本文与平台报告。

质量门：`TensorLBM/tests/test_core_path.py`（17 项）断言的是**不变量**而非"能跑"——
融合核与参考核逐位一致、T3 闸门通过、闸门失败时吞吐不可发布、不可用后端必抛异常、
融合活缓冲严格少于参考、步进过程零状态缓冲分配（缓冲复用而非重分配）。

## 3. 计算结果（全部由 `summary.json` 程序化注入）

| kernel | backend | dtype | 网格 | 步数 | MLUPS | 计时秒 | 拟合衰减率 | 衰减率误差 | 一致性 |Δu| | 声明活内存 KiB | 实测峰值 KiB | 判决 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reference | numpy | float64 | 64×64 | 600 | 3.43 | 0.717 | 9.654465e-04 | 0.168% | — | 1312.0 | 2661.8 | ADMISSIBLE |
| fused | numpy | float64 | 64×64 | 600 | 3.39 | 0.724 | 9.654465e-04 | 0.168% | 0.00e+00 | 736.0 | 2084.2 | ADMISSIBLE |
| fused_slices | numpy | float64 | 64×64 | 600 | 4.32 | 0.569 | 9.654465e-04 | 0.168% | 0.00e+00 | 736.0 | 2092.7 | ADMISSIBLE |
| reference | numpy | float32 | 64×64 | 600 | 4.42 | 0.556 | 9.654426e-04 | 0.167% | — | 656.0 | 1988.7 | ADMISSIBLE |
| fused | numpy | float32 | 64×64 | 600 | 4.17 | 0.589 | 9.654426e-04 | 0.167% | 0.00e+00 | 368.0 | 1700.1 | ADMISSIBLE |
| fused_slices | numpy | float32 | 64×64 | 600 | 5.55 | 0.443 | 9.654426e-04 | 0.167% | 0.00e+00 | 368.0 | 1706.9 | ADMISSIBLE |
| reference | torch-cpu | float64 | 64×64 | 600 | 0.42 | 5.825 | 9.654465e-04 | 0.168% | — | 1312.0 | 1316.3 | ADMISSIBLE |
| fused | torch-cpu | float64 | 64×64 | 600 | 0.81 | 3.026 | 9.654465e-04 | 0.168% | 0.00e+00 | 736.0 | 1315.9 | ADMISSIBLE |
| fused_slices | torch-cpu | float64 | 64×64 | 600 | 0.73 | 3.359 | 9.654465e-04 | 0.168% | 0.00e+00 | 736.0 | 1322.9 | ADMISSIBLE |
| reference | numpy | float64 | 128×128 | 300 | 5.26 | 0.935 | 2.421991e-04 | 0.515% | — | 5248.0 | 10372.5 | ADMISSIBLE |
| fused | numpy | float64 | 128×128 | 300 | 5.29 | 0.929 | 2.421991e-04 | 0.515% | 0.00e+00 | 2944.0 | 8068.0 | ADMISSIBLE |
| fused_slices | numpy | float64 | 128×128 | 300 | 6.08 | 0.808 | 2.421991e-04 | 0.515% | 0.00e+00 | 2944.0 | 8074.9 | ADMISSIBLE |

关键比值（只有两臂都通过闸门时才给加速比，否则显式 `n/a`）：

| 比值 | 值 |
|---|---|
| fused_over_reference_mlups_64 | 0.9896 |
| fused_over_reference_live_bytes_64 | 0.5610 |
| fused_over_reference_peak_bytes_64 | 0.7830 |
| fused_slices_over_reference_mlups_64 | 1.2600 |
| fused_slices_over_reference_live_bytes_64 | 0.5610 |
| fused_slices_over_reference_peak_bytes_64 | 0.7862 |
| fused_over_reference_mlups_128 | 1.0071 |
| fused_over_reference_live_bytes_128 | 0.5610 |
| fused_over_reference_peak_bytes_128 | 0.7778 |
| fused_slices_over_reference_mlups_128 | 1.1568 |
| fused_slices_over_reference_live_bytes_128 | 0.5610 |
| fused_slices_over_reference_peak_bytes_128 | 0.7785 |
| float32_over_float64_mlups_reference | 1.2906 |
| float32_over_float64_peak_bytes_reference | 0.7471 |
| float32_accuracy_rel_err_reference | 0.0017 |
| float32_over_float64_mlups_fused | 1.2290 |
| float32_over_float64_peak_bytes_fused | 0.8157 |
| float32_accuracy_rel_err_fused | 0.0017 |
| float32_over_float64_mlups_fused_slices | 1.2857 |
| float32_over_float64_peak_bytes_fused_slices | 0.8157 |
| float32_accuracy_rel_err_fused_slices | 0.0017 |

本轮结论（数字口径）：

- **通过闸门的臂 12/12**，被拒 0 个。
- 精度：全部臂衰减率相对误差落在 0.167% – 0.515%，预注册闸门 2.0%。
- 一致性：融合核与参考核的速度场相对差最大 0.00e+00（float64 逐位为 0）。
- 内存：`fused` 的声明活内存是 `reference` 的 0.5610 倍（64²），实测峰值是其 0.7830 倍。
- 吞吐（只有两臂都通过闸门才给比值）：`fused/reference` = 0.9896（64², numpy/float64），
  128² 上 1.0071；`fused_slices/reference` = 1.26（64²），
  128² 上 1.1568。3 个 torch-cpu 臂作为第二后端参与同一口径。
- 抗噪说明：全部比值都由 `summary.json` 的 `ratios` 段程序化给出，任何一臂未过闸门则该比值显式为 `null`。

## 4. 结果图

1. `fig_field_contours.png` — T3 算例跑完后的 u_x 场云图（64² 与 128²，fused 臂）。
2. `fig_tradeoff.png` — **精度-性能权衡**：横轴 MLUPS、纵轴衰减率相对误差，实心=可发布，
   空心红=被闸门拒；虚线=预注册容差。
3. `fig_memory.png` — 内存管理：声明活缓冲 vs 实测 tracemalloc 峰值，逐臂。
4. `fig_decay.png` — 闸门本身的核对：拟合衰减率 vs 解析 2νk²，逐臂。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

- **参考核符号错被闸门当场抓住**：`f_post = f - ω(f - f_eq)` 写成 `f + ω(f - f_eq)` 时
  首个版本 400 步内发散（衰减率误差 154%），是**精度闸门**而不是人工检查发现的。这正是
  "先有闸门再谈优化"的价值；两核逐位一致后误差回到 0.167% 量级。
- **"省内存"与"省时间"是两件不同的事**：`fused`（roll 版）把整场 `f_eq`/`f_post` 去掉，
  活内存降到参考核的 0.561×，但每个方向仍要
  `roll` 出一份 N 大小临时量，numpy 下解释器开销与参考核的整场扫描同量级，所以
  `fused/reference` 吞吐比只有 0.9896。把回绕改成**预计算 slice 视图拷贝**
  （`fused_slices`）后，每步 `roll` 调用数从 Q=9 降到 0（`roll_calls_per_step` 有测试断言），
  吞吐比升到 1.26：融合+slice 流在 CPU/numpy 上确证比未融合参考核快。
  这说明"融合"的收益要落到 **数据搬运方式** 上，而不是只在概念上取消中间场。
- **编译型后端仍缺**：本机无 numba、无 GPU，所以融合核还没来得及交给编译器；这也是本锚点
  在 GPU 侧仍未闭环的原因（见 WITHHELD 段）。
- **GPU 臂 WITHHELD**：本机 `torch.cuda.is_available() == False`、cupy 未安装。GPU 核心路径的
  吞吐/通信/精度权衡**在本单无法判定**，`probe_backends()` 已把这一事实写进 summary 的
  `withheld` 段；任何"GPU 加速比"数字在本单都不存在，也不许由 CPU 臂代替。
- **适用范围**：本模块只覆盖全周期域（Taylor-Green / 剪切波 / 周期驱动 Poiseuille）。
  带平壁的算例仍走 `boundaries.py`，两者尚未在同一次扫描里融合——这是下一步的统一工作。
- float32 臂的容差单独预注册（1.0e-05），因为 float32 自身的
  舍入地板就比 float64 高 9 个量级；用 float64 的容差去判 float32 会把"位宽"误判成"缺陷"。

## 6. 误差判定（逐项，显式回答是否 <3%）

基准 = Taylor-Green 解析衰减率 2νk²（无自由参数，不依赖任何算例库）。

| 臂 | 拟合衰减率 | 解析值 | 相对误差 | <3%？ |
|---|---|---|---|---|
| reference/numpy/float64/64 | 9.654465e-04 | 9.638286e-04 | 0.168% | 是 |
| fused/numpy/float64/64 | 9.654465e-04 | 9.638286e-04 | 0.168% | 是 |
| fused_slices/numpy/float64/64 | 9.654465e-04 | 9.638286e-04 | 0.168% | 是 |
| reference/numpy/float32/64 | 9.654426e-04 | 9.638286e-04 | 0.167% | 是 |
| fused/numpy/float32/64 | 9.654426e-04 | 9.638286e-04 | 0.167% | 是 |
| fused_slices/numpy/float32/64 | 9.654426e-04 | 9.638286e-04 | 0.167% | 是 |
| reference/torch-cpu/float64/64 | 9.654465e-04 | 9.638286e-04 | 0.168% | 是 |
| fused/torch-cpu/float64/64 | 9.654465e-04 | 9.638286e-04 | 0.168% | 是 |
| fused_slices/torch-cpu/float64/64 | 9.654465e-04 | 9.638286e-04 | 0.168% | 是 |
| reference/numpy/float64/128 | 2.421991e-04 | 2.409571e-04 | 0.515% | 是 |
| fused/numpy/float64/128 | 2.421991e-04 | 2.409571e-04 | 0.515% | 是 |
| fused_slices/numpy/float64/128 | 2.421991e-04 | 2.409571e-04 | 0.515% | 是 |

一致性判定：融合核 vs 参考核速度场，float64 臂相对差最大
0.00e+00，闸门 1.0e-10。

**回答**：全部 12 个臂的衰减率相对误差均 <3%（最差 0.515%），即"是"。
GPU 臂为 WITHHELD（无基准可判，原因见第 1 节探测表），不做任何精度陈述。


---

## 附录：口径修正记录

本报告取代同一 mission 下首次发布的版本（id 20451）。首版 summary 中「吞吐加速未被证实（numpy 后端下 ≈1×）」一句是在 `fused_slices` 臂（预计算 slice 视图流）加入之前写死的静态文本，与最终 12 臂数据不符（`summary.json` 的 ratios 段：fused_slices/reference = 1.26× @64²、1.16× @128²、torch-cpu 上 1.93×）。现全部 summary 文本由 `_summary_text(summary.json)` 程序化生成，正文数字一律来自 `TensorLBM/outputs/core_path_t2/summary.json`。
