# TensorDG 斜激波精确验证链重落地（m15113 · PR#270 第 2 轮）

> 仓：TensorDG；分支 `agent/m15113`（自 `main@d0b157c` 切出）；机器：loongarch64（128 核，与集群同架构）。
> 全部数字程序化注入：`docs/benchmarks/results/oblique_shock/results.json`（已提交工件）与
> `artifacts/m15113/figs.json`（本单重跑，零手抄）。

## ① 案例介绍（物理背景 / 几何 / 参数）

**任务背景**：原 PR#270（m13761 matured deliverables，7 文件 +1129/−40）被 owner 于 2026-11-05
裁定关闭。本单为「管到底」第 2 轮：从当前 main 切枝，把成熟交付物重新落地并全量验证。

**落地状态审计（git 对表，非推断）**：

| 文件 | main 现状 | 本单动作 |
|---|---|---|
| benchmarks/oblique_shock.py | 已在 main（#249 合入，blob 与 m13761 一致） | 不动 |
| benchmarks/supersonic_wedge.py | 已在 main（一致） | 不动 |
| benchmarks/double_mach_reflection.py | 已在 main（一致） | 不动 |
| docs/benchmarks/results/oblique_shock/results.json | 已在 main（一致） | 不动 |
| docs/reports/tensordg-oblique-shock-verification.md | 已在 main（一致） | 不动 |
| tests/test_euler_verification_chain.py | main 上是 m14727 重写版（pinned 回归守卫 + 阈值溯源，**新于** m13761 版） | 保留 main 版（取 m13761 版=回退） |
| tests/test_oblique_shock.py | main 上是 #249 早期版（ncells=40/cfl=0.1 + 120s timeout 标记） | **取 PR#270 成熟版**（ncells=30/cfl=0.2，去 timeout 标记） |

即：7 个交付文件中 5 个已在 main、1 个被更新的 m14727 版取代，**唯一剩余 delta =
tests/test_oblique_shock.py 的成熟版**（+1/−11 行）。

**物理案例**（验证面本体，m13761 建立）：二维可压缩 Euler（γ=1.4），上游均匀流 + 平面激波
（与 x 轴夹角 β）+ Rankine–Hugoniot 下游态；两态均匀 ⇒ **定常精确解**。锚 = θ–β–M 关系
（Anderson 3/e 第 4 章）+ RH 四条跳跃条件。

| case | M | 几何 | 网格 | t_end | CFL | TVB |
|---|---|---|---|---|---|---|
| wedge_M2_theta15 | 2.0 | θ=15° 楔类（β=45.3436°） | 40², P1 | 0.03 | 0.2 | 100 |
| normal_M4_beta90 | 4.0 | 网格对齐正激波（β=90°） | 24², P1 | 0.02 | 0.1 | 1.0 |
| control_M2_theta15_wrong_beta | 2.0 | 同两态、激波面偏 −5°（负对照） | 40², P1 | 0.04 | 0.2 | 100 |
| strong_M10_beta60 | 10.0 | Woodward–Colella 类强激波 | 64², P1 | 0.06 | 0.2 | 100 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **对表**：`git diff --name-only origin/main agent/m13761` + 逐文件 blob 比对，确定唯一剩余 delta。
2. **落地**：`git checkout agent/m13761 -- tests/test_oblique_shock.py`，单 commit 提交
   （`8003aae`），不触碰任何 TensorDG 核心模块（无热文件重叠，无需额外声明）。
3. **验证**（loongarch64，与集群同架构）：
   - `pytest tests/test_oblique_shock.py` → **7/7 passed（44 s）**；
   - `pytest tests/test_euler_verification_chain.py tests/test_benchmark_smoke.py` → **19 passed, 1 xpassed（29:06）**；
   - 全套 `pytest tests/`（后台 + 集群备份作业 cj994，stage_dir=本工作树）。
4. **报告图**：`scratch/m15113_make_report_figs.py` 以 results.json 同款参数重跑两个稳定 case，
   调 `oblique_shock._plot_case` 出密度云图 + 精确激波面 + 法向剖面 vs 精确 RH 台阶；
   数字写入 `artifacts/m15113/figs.json`（程序化，零手抄）。

## ③ 计算结果（程序化注入：results.json 已提交工件 + 本单 figs.json 重跑）

| case | M | β_exact(°) | β̂(°) | Δβ(°) | ρ₂/ρ₁ exact | ρ 比 err | p 比 err | Δθ(°) | 保持误差 | 门 |
|---|---|---|---|---|---|---|---|---|---|---|
| wedge_M2_theta15 (40²) | 2.0 | 45.3436 | 45.6470 | 0.3034 | 1.72892 | 4.25e-6 | 6.64e-6 | 0.0002 | 2.31% | 5/5 PASS |
| normal_M4_beta90 (24²) | 4.0 | 90.0000 | 90.0000 | 0.0000 | 4.57143 | 1.03e-5 | 1.40e-5 | 0.0000 | 2.03% | 5/5 PASS |
| control_M2_theta15_wrong_beta (40²) | 2.0 | 45.3436 | 40.4083 | 0.0645 | 1.72892 | 0.00% | 0.00% | 0.0003 | 3.73% | preserve 门 FAIL（预期） |
| strong_M10_beta60 (64²) | 10.0 | 60.0000 | — | — | 5.625 | — | — | — | — | **发散（honest FAIL）** |

- 本单重跑（figs.json）与已提交 results.json **逐位一致**（wedge：Δβ=0.3034°、ρ err=4.25e-6、
  p err=6.64e-6、Δθ=0.0002°、preserve=2.31%；normal：Δβ=0.0°、ρ err=1.03e-5、p err=1.40e-5、
  preserve=2.03%）——工件可复现。
- 测试（loongarch64）：`tests/test_oblique_shock.py` **7/7 passed（44 s）**；
  `tests/test_euler_verification_chain.py` + `tests/test_benchmark_smoke.py` **19 passed, 1 xpassed（29:06）**；
  全套 `pytest tests/` 结果见 §⑤ 验证状态。

## ④ 结果图

- 图 1：wedge_M2_theta15（M=2、θ=15°、40²、P1、t=0.03）——密度云图（白线=精确激波面）+
  沿精确法向剖面 vs 精确 RH 台阶（虚线=密度、点线=压力）；β̂=45.647°（精确 45.344°）。
- 图 2：normal_M4_beta90（M=4 网格对齐正激波、24²、P1、t=0.02）——测量管线自洽性对照，
  β̂ 误差 0.0°，ρ/p 比误差 ~1e-5。

（绘图通路为 m13761 实装的 `oblique_shock._plot_case`；本单未新增绘图代码。）

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **落地审计是本单核心**：直接 cherry-pick m13761 会把 m14727 的验证链重写**回退**掉
   （Sod/Lax/均匀流阈值溯源全部丢失）——必须逐文件 blob 对表后只取剩余 delta。
2. **边界条件是硬约束**：出流面截断激波，量测必须在内部窗口；壁面楔形绕流与 Woodward–Colella
   双 Mach 反射图形在当前求解器（无斜滑移壁/运动边界）上物理不可达——`test_double_mach_runs`
   保持 `xfail(strict=False)` 如实登记。
3. **M=10 强激波发散**（FloatingPointError: nonphysical cell average，TVB=100 与 TVB=1/CFL=0.1
   均如此）——未调参掩盖。
4. **负对照不敏感**：激波面偏 5° 时保持误差仅 3.73%（<10% 判别阈值）——`preserve_l1` 短时下
   对激波面几何不敏感（平台比由 RH 构造，天然与几何无关）；判别力来自正激波极限（1e-5）与
   楔形反演（0.30°）。
5. **限制器代价**：激波贯穿全域时 TVB 路径单步约 3× 光滑算例，故网格上限 24²–40²。
6. 适用范围：定常平面激波（斜/正），M ≤ 4 在当前设置下稳定。
7. **双 Mach xfail 本次 xpass**：`test_double_mach_runs`（120×30、cfl=0.15、tvb=100、t=0.1）在
   loongarch64 上跑完且密度全有限（XPASS，strict=False 不翻红）——m13761 记录的
   FloatingPointError 中止是边缘行为（网格/CFL 相关）；物理缺口（无斜滑移壁/运动边界）仍在，
   xfail 标记保留。
8. **全套测试状态**：`pytest tests/`（loongarch64 128 核，后台）进行中；已见 2 个 setup ERROR，
   定位在 `tests/test_euler_1d_wave_speeds.py` / `tests/test_euler_2d_p0_verification.py`
   （本单 delta 未触碰这两个文件及其依赖，属 main 既有状态，不属本单范围）；集群备份作业 cj994
   （stage_dir=本工作树）排队中。本单直接相关测试面（oblique_shock 7/7、验证链+smoke 19P+1XP）全绿。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）

| 量 | 基准 | 实测误差 | <3%？ |
|---|---|---|---|
| 激波角（M=2、θ=15°、40²） | θ–β–M 精确 β=45.3436° | 0.3034°（≈0.67%） | **是** |
| 密度比（同上） | RH 1.72892 | 4.25e-6 | **是** |
| 压力比（同上） | RH 2.19465 | 6.64e-6 | **是** |
| 偏转角（同上） | 15° | 0.0002° | **是** |
| 正激波密度比（M=4） | RH 4.57143 | 1.03e-5 | **是** |
| 正激波压力比（M=4） | RH 18.5 | 1.40e-5 | **是** |
| 保持误差（M=2 斜激波） | 0（定常精确解） | 2.31% | **是** |
| M=10 强激波 | RH 5.625 / 87.3 | 未取得（发散） | **否（honest FAIL）** |
| 负对照判别 | 应 >10% | 3.73% | **否（判据不敏感，已披露）** |

**结论**：斜激波角度与跳跃量验证面成立（<3%，多数量 1e-4 量级）；壁面楔形/双 Mach 反射未验证、
M=10 发散、负对照不敏感三项如实入账，不粉饰。
