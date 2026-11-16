# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 认证参考引擎的第二个维度（2-D 泛化）

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open，开工前核对未 done）
> 分支：`agent/m22834`　日期：2026-11-16
> 类型：capability_extension（验证欠账：把已合并的 round-2 认证做法从 1-D 推广到 2-D 并加不变式测试）
> 诚实声明：**2-D 物理算例（等熵涡）的数值尚未产出**——集群作业 `cj4418` 递交后排队 47 min 未起跑，
> 本单按实登记为 not_run/WITHHELD，不写任何未跑出的数字。

## 1. 缺口：引擎自称「case-independent」，实际只能吃 1-D

`tensordg/certified_reference.py` 的文档字符串主张自己是「reusable, case-independent engine」，
但三个原语都隐式假设一维：

| 原语 | 1-D 写法 | 2-D 的后果 |
|---|---|---|
| `coarsen` | `v.reshape(-1, factor).mean(axis=1)` | 在 `(n,n)` 场上沿第二轴平均，返回**向量** |
| `richardson_extrapolate` | `factor = round(fine.size/coarse.size)` | 每方向加密 2 倍时因子是 **4**，粗化到错误网格 |
| `judge` | 同上（候选/参考元素数比） | 同上；候选会被粗化到不存在的网格上 |

前序两轮工作（Shu–Osher 交叉族审计、Sod 解析真值）都是 1-D 算例，**没有任何一条测试
覆盖 2-D**，所以这个洞一直没有暴露。失败方式是抛形状异常（不是静默出错），但
「除 1-D 外不可用」对一个「case-independent」的主张是实质缺口。

## 2. 改动（最小、向后兼容）

`tensordg/certified_reference.py`：

1. `coarsen(values, factor)`：`ndim<=1` 走**原样**代码路径（1-D 逐位不变）；`ndim>=2` 时
   按方向分块平均 `reshape(n0//f, f, n1//f, f).mean(axis=(1,3))`，即 `h -> h*factor` 的正确语义
   （`ncells` 语义钉死为**每方向格数**，与 `certify` 里的 `ncells/base` 因子一致）。
2. 新增 `nesting_factor(fine, coarse)`：按**形状**而非元素数求每方向加密因子；
   2-D 的 `2n x 2n vs n x n` 元素数比是 4，加密因子才是 2——用元素数比会静默粗化两次。
3. `richardson_extrapolate` / `judge` 改用 `nesting_factor`；1-D 下两者与原实现等价
   （1-D 形状比 == 元素数比）。
4. `nesting_factor` 进 `__all__`。

新增 `benchmarks/certified_reference_vortex_2d.py`：二维等熵 Euler 涡（周期箱 `[-5,5]^2`，
`u_inf=(1,1)`，涡强 5，`t=1`）的认证算例驱动——DG-P2 梯次 16/32/64 格/方向 + 独立 DG-P1 腿判决，
并**用解析解**测认证参考解的**真误差**（C3 带宽是否覆盖真误差），外加 C5：用打过补丁的引擎
重认证已发布的 Shu–Osher FV 梯次，必须**逐位**复现已发布数字。

## 3. 验证（已跑，命令与结果）

```
python -m pytest tests/test_certified_reference_2d.py -q          # 11 passed
python -m pytest tests/test_certified_reference.py \
                 tests/test_certified_reference_sod.py -q          # 31 passed
```

新增测试断言的是**不变式**，不是「跑一遍」：

| 测试 | 断言的不变式 |
|---|---|
| `test_one_dimensional_coarsening_is_unchanged` | 1-D 粗化数值与原行为一致；不整除时抛错 |
| `test_two_dimensional_coarsening_is_a_per_direction_block_average` | 2-D 粗化 == 逐方向块平均（手算 `(4,4)->(2,2)` 为 `[[2.5,4.5],[10.5,12.5]]`），且与「先展平再平均」的总均值一致 |
| `test_nesting_factor_uses_shapes_not_element_counts` | `(8,8)/(4,4) -> 2`（不是 4）；非嵌套 `(8,6)/(4,4) -> None` |
| `test_richardson_in_two_dimensions_removes_a_power_law_error` | 2-D Richardson 把 `a*h^3` 误差消到 `rel_L1 < 1e-8`；非嵌套对仍报错 |
| `test_two_dimensional_ladder_certifies_on_a_two_dimensional_grid` | 2-D 梯次 `CERTIFIED`，参考解形状 `(32,32)`，`refinement_fine==2`，观测阶复原注入的 `p=3`（±0.05） |
| `test_two_dimensional_certification_band_covers_the_true_error` | 2-D 带宽 `U` 覆盖参考解真误差 |
| `test_one_dimensional_certificate_is_bit_identical_to_the_shipped_behaviour` | 1-D 认证路径逐位不变（逐级与手写 `reshape(-1,2).mean` 比对） |
| `test_judge_coarsens_a_two_dimensional_candidate_by_the_per_direction_factor` | 候选在 2× 细网格上时按**每方向因子 2** 粗化到参考网格、判决为 PASS |
| `test_judge_refuses_a_two_dimensional_profile_that_does_not_nest` | 非嵌套 2-D 候选被拒（`ValueError`） |

**1-D 已发布证书未被移动**：打过补丁的引擎重认证 `validation/repro_shu_osher/artifacts/raw`
的 FV 梯次（4 级），得到 `p = 1.1767348864087728`、`U = 0.04162173159427086 %`，
与 `docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md` 发布值**逐位相同**（rel 1e-12 判据）；
原有等价性/失效测试 31 项全绿。

## 4. 未完成项（如实登记，not_run）

1. **2-D 物理算例数值未产出**：`python -m benchmarks.certified_reference_vortex_2d --figures`
   已递交集群（`cj4418`，8 核，`stage_dir`=本工作树），排队 47 min 未起跑（队列 in-flight 83）。
   因此 C1–C4 的数值判据（2-D 梯次是否 CERTIFIED、带宽是否覆盖 DG-P2 梯次的真误差、
   被判 P1 腿的判决）**全部为 not_run/WITHHELD**，本文不写任何估计值。
   复现命令：`run_compute submit repo=TensorDG stage_dir=<worktree> cores=8
   cmd='cd <worktree> && OMP_NUM_THREADS=8 python -m benchmarks.certified_reference_vortex_2d
   --outdir artifacts/certified_reference_vortex_2d --figures --json-stdout'`。
2. 因此本单**无结果图**：模块的绘图路径（差场云图 + 收敛/带宽曲线）已实现但依赖上条数值产出。
3. 2-D 的 DG 单元平均不适用 1-D 的 `2k+1` 超收敛结论，故模块刻意不声明 `design_order`；
   这一点由后续数值结果检验（若观测阶接近 `k+1` 属预期）。
4. `tests/test_public_api_contract.py` 在本工作树为 **5 failed / 22 passed**，但
   `git stash` 后（未打本单补丁）同样 **5 failed** —— 与本单改动无关的既有 RED，
   本单未触碰也未宣称修复。

## 5. 误差判定（回答「是否 < 3%」）

| 关键量 | 基准 | 本单测量 | 误差 | < 3% ? |
|---|---|---|---|---|
| 1-D 已发布观测阶 p | 1.1767348864087728 | 1.1767348864087728 | 0（逐位） | **yes** |
| 1-D 已发布带宽 U | 0.04162173159427086 % | 0.04162173159427086 % | 0（逐位） | **yes** |
| 2-D 参考解真误差 | 解析解（4×4 Gauss 单元平均） | not_run（cj4418 排队） | — | **WITHHELD** |
| 2-D 带宽覆盖性 | 真误差 ≤ U | not_run | — | **WITHHELD** |

2-D 物理量的误差判定**尚未做出**，须由 `cj4418` 的产出补齐（或由后续单继承）。

## 6. 注意点

1. **`ncells` 语义**：2-D 下是每方向格数，不是总格数。谁若按总格数传，`certify` 的粗化因子会
   变成 `r^2`，静默认证错网格；`nesting_factor` 的形状判据与测试就是防这一条。
2. **合成阶梯的坑（本单实测）**：若用「解析场在单元中心取值 + `a*n^-p`」构造 2-D 阶梯，
   会额外引入 `O(h^2)` 的单元平均/中心差，观测阶实测 **2.04** 而不是注入的 3.00；
   这是夹具缺陷不是引擎缺陷——改用双线性场（单元平均 == 中心值）后观测阶精确复原。
   同一个坑在真实算例里对应「参考解必须是单元平均，而不是节点值」。
3. **本单不改任何已发布数字**：改动只影响 `ndim>=2`，而 1-D 路径由逐位比对与 31 项既有测试双重锁定。

## 7. 产物

| 文件 | 说明 |
|---|---|
| `tensordg/certified_reference.py` | 2-D/N-D 粗化 + `nesting_factor`；1-D 路径逐位不变 |
| `tests/test_certified_reference_2d.py` | 11 项 2-D/1-D 不变式测试（全绿） |
| `benchmarks/certified_reference_vortex_2d.py` | 2-D 等熵涡认证驱动（C1–C5 判据、图、可复现 CLI） |
| `docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE-2D.md` | 本报告 |