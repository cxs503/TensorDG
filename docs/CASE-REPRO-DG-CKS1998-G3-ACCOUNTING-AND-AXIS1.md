# CKS1998 · G3 内壁反射：守恒账本缺陷修复 + axis=1 内壁等价性

> 锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, *JCP* **141** (1998) 199–224，§5.3 前向台阶）。
> 上一单 `docs/CASE-REPRO-DG-CKS1998-G3-EMBEDDED-WALL-ATTRIBUTION.md`（m21992）把 G3 的
> 残余偏差**归因排除**了 `_solid_interface`（两臂逐位相等），并留下两条明确的欠账：
> ⑤.3「无固体臂的守恒残差返回 `nan`，让守恒门静默失效」、⑤.5「y 向内壁面（`axis=1`）从未独立对照」。
> 本单清偿这两条；**反射平台的定量梯次仍未取回（见 §4，honest）**。

## 1. 缺陷：无障碍运行时 `conservation_residual_total` 全是 `nan`

`tensordg/euler_dg_2d.py` 里最强的那条恒等式是

```
域积分变化  ==  边界通量账本  +  障碍物冲量账本
```

它此前被 `if solid_balance is not None and totals_initial is not None:` 守卫。而
`solid_balance` 只在**网格里存在固体单元**时才分配，于是在**无障碍**的算例（G3 的
`wall_mode="boundary"` 臂、以及任何只用 `bcs="wall"` 的历史调用）上整块返回 `nan`。
`abs(nan) < tol` 恒为 `False`，所以这类守恒门**不是失败，也不是通过，而是静默不可判**。

**物理上 `nan` 是错的**：没有固体单元，就没有任何内面能把动量推给流体，障碍冲量**恰好为 0**。
修复只改守卫与冲量取值（最小集）：

| 位置 | 改前 | 改后 |
|---|---|---|
| `conservation_residual_total` 守卫 | `solid_balance is not None and totals_initial is not None` | `totals_initial is not None`（= 边界账本被跟踪） |
| 冲量项 | `float(solid_balance[i])` | `0.0 if solid_balance is None else float(...)` |
| 条目 | 4 个字段 | 增 `has_obstacle: bool`，把「无障碍」与「未跟踪」分开 |
| 未跟踪（`track_boundary_flux=False` 且非周期） | `nan` | 仍 `nan`（有语义，不是缺陷） |

行为差异（同一算例，`gate_wall_reflection`，16×1 P1，t_end=wall_time+0.03）：

| 臂 | 改前 质量相对残差 | 改后 质量相对残差 | 改后 能量相对残差 | 步数 |
|---|---:|---:|---:|---:|
| `boundary`（无障碍） | **nan** | **9.496e-15** | 7.880e-15 | 255 |
| `solid`（有障碍） | 7.377e-15 | 7.377e-15（不变） | 4.491e-15 | 234 |

`solid` 臂逐位不变，说明这修复**只解锁了原本不可判的那条路径**，没有改动已通过的门。
证据文件：`validation/forward_step/g3/accounting_fix_evidence.json`。

回归门（新增）：
`tests/test_g3_embedded_wall_equivalence.py::test_conservation_ledger_is_finite_exactly_when_no_obstacle_is_present`
断言 4 个守恒量 × 5 个字段**全部有限**、相对残差 < 1e-12、`has_obstacle` 与冲量取值为 0.0。
即"`nan` 残差本身就是失败"，不再能冒充非答案。

## 2. 新覆盖：`axis=1`（y 向）内壁面的等价性

`_solid_interface` 有 `axis=0` / `axis=1` 两个分支，但**所有**历史内壁实验（G3、台阶）都把
障碍放在 x 向面后，只走到 `axis=0`。台阶的**竖直面** `x = 0.6, y∈[0,0.2]` 属于 `axis=1`，
归因文档 ⑤.5 因此把它记为未证。本单把 G3 实验整体旋转 90°：

* 两臂：`bcs top="wall"`（域边界，`ny_fluid=16`）vs 顶部后方 3 个冻结固体单元（`axis=1` 内面）；
* 相同 `dx`、相同流体单元数、相同初值（同一 `WallReflection` 双激波 Riemann 态，沿 +y 传播）；
* 判据不引入第二个解析参考：两臂流体单元必须**逐位相等**。

结果：`max|Δ| ≤ 1e-9·max|coeff|` 通过，两臂步数相同，贴壁密度高于入射波后态（反射平台确实出现），
s 向质量/能量残差 < 1e-12。测试：
`tests/test_g3_embedded_wall_equivalence.py::test_embedded_wall_in_the_y_direction_matches_the_boundary_wall`
（**1 passed in 96.62 s**，本机 8 线程实跑）。

## 3. 梯次驱动器：扩到可判网格 + 表格/作图程序化

`scripts/run_g3_wall_reflection_ladder.py` 的梯次只到 116×1 P1。按该脚本自己的注释，反射平台在
29/58 上缓冲后只剩 2/8 个可采样单元，而门要求 `min_plateau_cells=3`。本单追加
`232×1 P1`、`116×1 P2` 两级。新增 `scripts/report_g3_ladder.py`：读 `g3_ladder.json` 直接生成
markdown 表与收敛图（`docs/figures/g3_ladder.png`，平台误差/L1 vs dx，两臂 + 3 % 门线），
**零手抄**，作业回件即可出数。

## 4. 未完成（honest）

**反射平台的定量梯次未取回。** 作业 **cj3879**（8 核，n315，`rc=0`，46 min 后 `succeeded`）
跑完了梯次与等价性测试，但 `fetch` 取到的包是**递交时的 stage 快照**（2858 条目，
无 `validation/forward_step/g3/` 目录），因此本单**没有**任何梯次数字可报。
如实记为 **WITHHELD / not_run**，不写成通过；G3「内壁反射在可判网格上误差 < 3 %」这一条
**仍无证据**。复现只需一条命令（作业本身已验证可跑到终态）：

```bash
run_compute submit repo=TensorDG stage_dir=<worktree> cores=8 timeout=3600 cmd='... run_g3_wall_reflection_ladder --out <dir>'
```

**平台侧缺陷（供下一单）**：`run_compute fetch` 对 `stage_dir` 作业返回的是输入快照而非运行后
工作树，任何"作业把结果写进检出目录"的用法都会静默丢产物——本单踩到。建议 fetch 改打运行后
目录，或在回执里显式区分 `stage` / `result` 包。

其余未做：G3 的 **y 向**定量（本单只做了 y 向**等价性**，没有 y 向解析参考的定量门）；
台阶算例（Mach 3）的阻力系数与激波位置仍未取回；论文表格逐格对照仍 WITHHELD（仓内无 PDF）。

## 5. 误差判定（逐项，显式回答「是否 < 3 %」）

| 项 | 基准 | 实测 | 误差 | < 3 % ? |
|---|---|---|---:|---|
| 无障碍臂守恒恒等式（质量） | 0 | 9.496e-15 | 9.5e-15 | **是**（改前不可判 `nan`） |
| 无障碍臂守恒恒等式（能量） | 0 | 7.880e-15 | 7.9e-15 | **是** |
| 有障碍臂守恒恒等式（质量/能量） | 0 | 7.377e-15 / 4.491e-15 | 同左 | **是**（未变） |
| x 向两臂解（全模态） | `boundary` 臂 | `solid` 臂 | 0（≤1e-9 相对门内） | **是**（精确相等，沿用 m21992） |
| **y 向两臂解（全模态）** | `boundary` 臂 | `solid` 臂 | ≤1e-9 相对 | **是**（本单新增） |
| 反射平台误差 vs 精确双激波解 | rho5/p5 | **无数据** | — | **未判（WITHHELD）** |

## 6. 复现

```bash
pytest tests/test_g3_embedded_wall_equivalence.py -q -k "finite_exactly"   # 账本有限性门
pytest tests/test_g3_embedded_wall_equivalence.py -q -k "y_direction"      # axis=1 等价门（~97 s）
python -m scripts.run_g3_wall_reflection_ladder --out validation/forward_step/g3   # 梯次（集群）
python -m scripts.report_g3_ladder --json validation/forward_step/g3/g3_ladder.json # 表+图
```
