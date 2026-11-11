# TensorDG 质量审计报告（A+B，owner 2026-11-11 全面质量审计令）

- **仓库**：`cxs503/TensorDG`，ref=`main` @ `81ddba9`（worktree `agent/m14490`）
- **审计员**：`audit2-ab`（mission 14490）
- **唯一核心标准**：可验证的功能交付（不是任务数 / PR 数 / 行数）
- **证据分级**：REAL / MOCK / STUB / WITHHELD / NOT_SUPPORTED；成熟度 L0–L5（owner 六级）
- **报告纪律**：每个结论附指针（文件:行号 / 工件路径 / 测试名 / PR#）。无证据即低判；诚实 FAIL 不粉饰。

---

## 0. 审计口径与方法

| 步骤 | 做法 | 机器可复现脚本 |
|---|---|---|
| 资产盘点 | `git diff --numstat <parent> <merge>` 逐 mission 统计，按路径分类 core/tests/docs/bench/validation/tools | §8-1 |
| 占位扫描 | AST 遍历 `tensordg/`，识别 `pass` / `raise NotImplementedError` / `...` / 常量返回函数 | §8-2 |
| 接入分析 | 每模块统计 `__init__` 导出 + tests/benchmarks/validation/tools 外部引用 | §8-3 |
| 功能真实性 | 逐 scope 六问（入口/真算/接入/IO/可运行算例/独立基准），与 PR 报告声明逐条对表 | §2 |
| 运行验证 | 本地 `pytest` 逐文件执行（单文件 < 8 min）；全量套件递交集群（窗口内未派发，见 §6） | §7 |
| 工件核验 | 直读 `validation/**/*.json`、`docs/benchmarks/results/**/*.json` 的判据/阈值/实测值，检查 provenance 与结论一致性 | §8-4 |

**级别判据（本报告采用）**

| L | 判据 | 本报告的证据门槛 |
|---|---|---|
| L0 | 文档成果（调研/报告/知识整理） | 只有 md，无算法代码、无工件 |
| L1 | 数学与单元验证（公式实现 + 解析解对照，证局部逻辑） | 代码存在 + 单元测试/解析对照，但无完整算例 |
| L2 | 独立功能实现（真算法：输入→计算→输出） | 代码被独立调用执行过（测试或脚本），输出可读 |
| L3 | 系统集成（接入主求解器/平台/工作流协同运行） | 被主路径/CI/其他求解器实际调用（非孤岛） |
| L4 | 基准验证（完整算例 + 收敛/守恒 + 误差分析 + 基准对照可复现） | 有提交工件（JSON）+ 判据 + 实测误差 + 可重跑脚本 |
| L5 | 科研与工程成果（解决此前未解问题/新方法/新规律） | L4 之上且有此前不存在的结论（本轮未给任何模块判 L5） |

> **附 100 页报告不改变级别**：本报告所有 L≥2 判定都指向「执行证据」（测试名 / 工件路径 / 运行日志），不指向文档篇幅。

---

## 1. ① 资产盘点

### 1.1 存量（main HEAD 全量，按 worktree 实测行数）

| 类别 | 文件数 | 行数 | 性质 |
|---|---:|---:|---|
| 核心算法模块 `tensordg/*.py` | 30 | 13,374 | 真算法库；324 个公开 def/class；`__init__.py` 导出 69 个符号 |
| 测试 `tests/` | 102 | 20,837 | pytest，`--collect-only` = **1091 个用例** |
| 基准脚本 `benchmarks/` | 108 | 32,448 | 算例驱动 + 图/JSON 生成 |
| 验证脚本 `validation/` | 32 | 4,614 | 独立验证运行器（另含大 JSON 工件，见末行） |
| 工具 `tools/` | 36 | 8,565 | 报告/图/推导工具 |
| `scratch/` | 12 | 1,693 | 一次性诊断脚本（已入 main） |
| 文档 `docs/**.md` + 根 md | 127 | 19,589 | 报告/教程/验证说明 |
| JSON 工件（benchmarks+validation+docs） | 117 | 85,224 | 数值证据本体 |

### 1.2 agent PR 增量（78 个 mission，`Merge #N (mXXXX)` 提交，numstat vs 父提交）

**合计新增 123,651 行**，按路径分类：

| 类别 | 新增行 | 占比 |
|---|---:|---:|
| 核心算法 `tensordg/` | 8,064 | **6.5 %** |
| 测试 `tests/` | 14,525 | 11.7 % |
| 文档 `docs/` + md | 22,966 | 18.6 % |
| 基准 `benchmarks/` | 18,996 | 15.4 % |
| 验证 `validation/`（含大 JSON 工件） | 53,013 | **42.9 %** |
| `tools/` + `scratch/` | 5,633 | 4.6 % |
| 其他（`.report.json` 等） | 454 | 0.4 % |

**解读**：88.6 % 的合入行不是求解器代码。这是「验证驱动型仓库」的正常形态（先有工件再有结论），
但也意味着**行数绝不能当功能交付的代理指标**——本报告因此对每个 scope 逐条落到执行证据。

最近 12 个 mission 的分类明细（新增行）：

| mission | 日期 | 总增 | core | tests | docs | bench | validation | tools/scratch | 内容 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| m13815 | 11-11 | 894 | 0 | 139 | 136 | 148 | 235 | 236 | NS 壁面热流收敛阶诊断 |
| m13810 | 11-11 | 1,125 | 0 | 99 | 115 | 0 | 911 | 0 | Thacker 长时 P2 验证 |
| m13761 | 11-11 | 1,129 | 0 | 134 | 262 | 733 | 0 | 0 | 斜激波精确验证 + 审计 #247 |
| m12101 | 11-11 | 2,137 | 370 | 248 | 239 | 0 | 564 | 716 | 2D SWE 静水平衡 |
| m11149 | 11-11 | 1,300 | 0 | 350 | 134 | 816 | 0 | 0 | 「Euler 验证链」（后经 m13761 部分证伪） |
| m10754 | 11-10 | 2,827 | 79 | 279 | 126 | 0 | 2,343 | 0 | SWE 岸线静止态 |
| m10591 | 11-10 | 4,905 | 1,153 | 337 | 2,659 | 756 | 0 | 0 | 统一验证（对流扩散 + 色散波） |
| m9711 | 11-10 | 798 | 71 | 109 | 55 | 0 | 282 | 281 | SWE 静水平衡 helper 契约 |
| m9198 | 11-10 | 1,061 | 0 | 207 | 292 | 562 | 0 | 0 | 曲边等参几何验证重写 |
| m8805 | 11-10 | 1,282 | 165 | 402 | 0 | 0 | 715 | 0 | SWE 静水重构 |
| m8649 | 11-10 | 749 | 0 | 122 | 99 | 0 | 206 | 322 | SWE WB 结果工件 |
| m8317 | 11-09 | 1,297 | 319 | 321 | 212 | 0 | 445 | 0 | （多提交 mission） |

> 78 个 mission **无 0 行空合并**（同 mission 多提交取最大差异统计；
> m8805/m8749/m8740/m7557/m7536/m7528/m7201 携带 ≥2 个提交，属 rebase/重推，非重复计数）。

### 1.3 占位实现（stub / NotImplemented / TODO / 空函数）

- AST 全量扫描 `tensordg/` 30 个文件、324 个公开 def/class：**STUB 函数 = 0**，`raise NotImplementedError` = 0，`TODO/FIXME` = 0。
- 唯一「常量返回」函数：`tensordg/burgers_shock_1d.py:92 shock_ctime()` 恒返回 `0.0`，docstring 明写「always 0」= 契约常量，非占位。
- **本仓库的「空壳」不是空函数，而是「有算法但无任何算例/工件」的模块**：
  `sipg_matrixfree.py`(621) / `sipg_sparse.py`(350) / `ice_edge_bench.py`(317) 合计 **1,288 行**，
  只被测试引用，`benchmarks/` 与 `validation/` 中**零调用方**（见 §1.5）。

### 1.4 重复实现（功能重叠清单）

| 重叠组 | 文件（行数） | 重叠内容 | 判定 |
|---|---|---|---|
| SIPG 椭圆/抛物装配 | `poisson_2d.py`(363) + `sipg_sparse.py`(350) + `sipg_matrixfree.py`(621) | 同一 Poisson-2D / Heat-1D 离散的三套装配（稠密/稀疏迭代/无矩阵 PCG） | **实质性重复**（1,334 行覆盖 2 个 PDE）；matrix-free 属性能路径，但缺对比工件说明收益 |
| 变系数对流速度场 | `advection_2d_var.py`(389) + `advection_diffusion.py`(635) | 旋转场/变形场（solid-body rotation / deformational flow）两套独立实现 | **实质性重复**；m10591 未复用 m4933 的 `rotated_profile` |
| 浅水静水重构（Audusse） | `swe_1d._hydrostatic_interface_flux`(:168) + `swe_2d._hydrostatic_flux_2d`(:112) | 同方案 1D/2D 两份实现 | 设计性重复（维度不同），但 m12101 未复用 swe_1d 已校验的判据 |
| 精确解参照 | `benchmarks/riemann_exact.py` / `sod_exact.py` / `swe_well_balanced_exact.py` vs `tensordg/swe_exact.py` | 精确 Riemann / 浅水精确解 | **刻意重复（非缺陷）**：参照必须独立于被验对象，否则循环验证（`tests/test_verification_ladder.py:30-35` 明写该设计意图） |
| TVB 判据 | `tvb.py`(68) vs `limiters.apply_tvb_limiter` | m2549 已把判定抽成单一实现 | 非重复（已收敛为单源） |

### 1.5 未接入模块（无任何 benchmark/validation 调用方）

| 模块 | 行数 | 唯一调用方 | 后果 |
|---|---:|---|---|
| `sipg_matrixfree.py` | 621 | `tests/test_matrixfree_verification.py` | 无算例工件；「matrix-free 更快」**无基准数据** |
| `sipg_sparse.py` | 350 | `tests/test_sipg_sparse.py` | 同上（稀疏 vs 稠密无对比工件） |
| `ice_edge_bench.py` | 317 | `tests/test_dg_advection_bench.py`、`test_advection_2d_var.py` | 冰区应用（域语义）躺在求解器仓里，无端到端算例 |

另：`advection_diffusion_1d.py`(286)、`burgers_shock_1d.py`(758)、`mms_ns_2d.py`(135)、
`ns_poiseuille.py`(199) 不在 `__init__.py` 导出面（69 符号）内，但均有 benchmark 或 validation 调用方
→ 属「有接入、未进公共 API」，不算孤岛。

---

## 2. ② 功能真实性审查（六问表）

六问：**Q1** 代码入口在哪 · **Q2** 是否真的执行计算 · **Q3** 是否接入主程序/工作流 · **Q4** 是否有实际输入输出 · **Q5** 是否有完整运行案例 · **Q6** 是否有独立基准验证。
「运行证据」列 = 本次审计**实际执行**过的 pytest / 工件（不是文档转述；未跑记 not_run）。

| # | scope | Q1 入口（文件:行） | Q2 真算 | Q3 接入 | Q4 I/O | Q5 运行案例 | Q6 独立基准 | 运行证据（本次实测） |
|---|---|---|---|---|---|---|---|---|
| 1 | 线性对流 1D/2D | `advection_1d.py:76 solve_advection`、`advection_2d.py` | ✅ DG 模态+RK4 | ✅ `__init__` 导出 + CI `benchmarks.linear_advection_1d/2d` | ✅ 系数/误差 | ✅ `results/linear-advection-{1,2}d` | ✅ 解析平移解；EOC 2.041/2.999/3.990、1.972/2.999/4.023（`verification_ladder/gates_verified.json` C1/C1b PASS） | `test_verification_ladder.py` **16 passed, 3 skipped (28.8 s)** |
| 2 | 变系数对流 + 冰区 bench | `advection_2d_var.py:144 rotated_profile`、`ice_edge_bench.py` | ✅ 变速度 RK | ⚠️ 仅测试引用 | ✅ | ⚠️ `validation/verify_advection_2d_var.py` | ✅ 解析解 | not_run |
| 3 | 对流–扩散 2D（m10591） | `advection_diffusion.py:551 run_solid_body_rotation` | ✅ 真求解（旋转/变形/衰减） | ✅ `__init__` 导出 + `benchmarks/dg/unified_verification.py` | ✅ 场量/质量漂移 | ✅ `results/dg-unified-verification/dg_unified_verification.json` | ✅ 解析解 + 算子一致性（散度 5.0e-11） | `test_advection_diffusion_2d.py` **15 passed (71 s)** |
| 4 | 一维对流扩散（指数边界层） | `advection_diffusion_1d.py` | ✅ | ✅ benchmark + tool | ✅ | ⚠️ Pe=100 P1 阶 1.627 vs 理论 2 → `results/advection_diffusion/metrics.json` `overall_verdict=FAIL` | ✅ 解析解 | not_run |
| 5 | 色散波（m10591） | `dispersive.py:162/340/406` | ✅ 色散算子 + KdV 传播 | ✅ `__init__` 导出 + 统一验证脚本 | ✅ kh 表/波速 | ✅ `dg_unified_verification.json`（E3 5.0e-5、E4 7.4e-7、E5 5.2e-4） | ✅ Airy 1.9e-16、KdV 速度误差 7.1e-5 | `test_dispersive_waves.py` **10 passed (18 s)** |
| 6 | Burgers（光滑+激波） | `burgers_1d.py:101`、`burgers_shock_1d.py` | ✅ | ✅ benchmark + ladder | ✅ | ✅ `results/burgers-1d/metrics.json` passed | ✅ 特征解 / Rankine–Hugoniot | not_run |
| 7 | 收敛/GCI 工具 | `convergence.py` | ✅ 纯工具函数 | ✅ 全仓 34 处引用（最多） | ✅ | ✅ `results/convergence/metrics.json` passed | —（判据工具本身） | not_run |
| 8 | 曲边等参几何（m9198） | `curved_geometry.py`（Piola/Jacobian/自由流） | ✅ | ✅ `benchmarks/verified/dg_curved_geometry/run_verification.py` | ✅ | ✅ `result.json` + 同目录 `gates_verified.json` | ✅ 自由流保持 drift ≤8.7e-16、度规恒等 ≤2.2e-16 | `test_curved_geometry.py` **46 passed (459 s)** |
| 9 | 可微 / 反问题 | `differentiable.py:101..401 advance_*` | ✅ autograd 前向 | ✅ `tests/inverse_study.py` + 教程 | ✅ 梯度/目标 | ✅ `validation/diffusive_differentiability/evidence.json`（目标误差 1.3e-3） | ✅ 解析波速/扩散率 + 故障注入矩阵 | not_run |
| 10 | Euler 1D（Riemann/HLLC/Sod/Lax） | `euler_1d.py:274 solve_riemann`、`euler_dg_1d.py:66` | ✅ | ✅ 8 测试文件 + CI Sod/Lax | ✅ | ✅ `results/euler-sod-1d`(2.5177%)、`lax-1d`(2.7694%) | ✅ 精确 Riemann（`benchmarks/riemann_exact.py` 独立实现） | not_run |
| 11 | Euler 2D（DG+限制器+涡） | `euler_dg_2d.py:574 solve_euler_dg_2d` | ✅ | ✅ 被 `ns_dg_2d`/`swe_2d` 复用 + CI 涡脚本 | ✅ | ✅ `results/isentropic-vortex-2d`（1.973%） | ✅ 解析移动涡/设计阶 | `test_euler_verification_chain.py` 部分 **FAIL**（§2.2） |
| 12 | 斜激波精确验证（m13761） | `benchmarks/oblique_shock.py`（RH 两状态精确解） | ✅ | ✅ `tests/test_oblique_shock.py` | ✅ | ✅ `results/oblique_shock/results.json` | ✅ θ–β–M：β 误差 0.303°(0.67%)、ρ 比 4.2e-6、p 比 6.6e-6、保持 L1 2.31% | `test_oblique_shock.py` **7 passed (480 s)** |
| 13 | 楔形绕流（#247→m13761 重写） | `benchmarks/supersonic_wedge.py:85` | ✅ 现建真 RH 态 | ⚠️ 无 CI 步骤 | ✅ metrics.json（运行后） | ❌ **无提交结果工件** | ✅ θ–β–M 解析角 | `test_euler_verification_chain.py::TestSupersonicWedge` **FAIL**（§2.2） |
| 14 | 双 Mach 反射（#247 声明） | `benchmarks/double_mach_reflection.py` | ⚠️ 只有周期/透射面 → **无反射壁、无 Mach 茎**（`:19-30` 自承） | ❌ 无 CI | ✅ 场量 | ⚠️ 仅初始态 RH 检查 | ❌ 无（Woodward–Colella 只作「定性」断言） | `...::TestDoubleMachReflection` **FAIL**（§2.2） |
| 15 | 热传导 1D | `heat_1d.py:146 solve_heat_1d` | ✅ SIPG+Crank–Nicolson | ✅ `__init__` + CI | ✅ | ✅ `results/heat-1d/metrics.json`（3.6e-6） | ✅ 解析衰减模态 | not_run |
| 16 | Poisson 2D（BC/稀疏/无矩阵） | `poisson_2d.py:166`、`poisson_2d_bc.py:215`、`sipg_sparse.py:246`、`sipg_matrixfree.py:263` | ✅ | ⚠️ 仅稠密路径进 CI | ✅ | ✅ `results/poisson-2d`（0.380%）；稀疏/无矩阵仅单测 | ✅ 制造解 | not_run（稠密）；稀疏/无矩阵见 §1.5 |
| 17 | 限制器（TVB/WENO） | `limiters.py`、`tvb.py` | ✅ | ✅ 被 Euler 1D/2D、SWE 复用（46 处） | ✅ | ✅ `docs/limiter_policy_verification.md`、`limiter_2d_verification.md` | ✅ Gibbs 界 / 阶次不退化 | not_run（本次窗口） |
| 18 | SWE 1D（Ritter/Thacker/静水平衡） | `swe_1d.py:422 solve_swe_1d`、`:168 _hydrostatic_interface_flux` | ✅ Audusse 静水重构 + HLL | ✅ `__init__` 导出 + 31 处 bench/val 引用 | ✅ 水深/流速/eta | ✅ `validation/swe_thacker/*.json` | ✅ Ritter 自相似 / Thacker 解析周期解 | `test_swe_1d.py` **30 passed (345 s)**；`test_thacker_longtime_p2.py` **6 passed (73 s)** |
| 19 | SWE 2D（m12101 静水平衡） | `swe_2d.py:339 solve_swe_2d`、`:112 _hydrostatic_flux_2d` | ✅ | ✅ `__init__` 导出 + validation 运行器 | ✅ | ✅ `validation/swe_2d_wb/wellbalanced_2d_results.json` | ✅ 静止湖（P2 判别：默认 u 1.1e-2 vs WB 2.8e-12） | `test_swe_2d_well_balanced.py` **17 passed (50 s)** |
| 20 | SWE 精确解库 | `swe_exact.py`（Stoker/Ritter/Thacker） | ✅ 解析式实现 | ✅ 8 处测试引用 | ✅ | ✅ 由各 SWE 测试消费 | 本体即基准 | 间接覆盖（`test_swe_1d.py`） |
| 21 | NS DG 2D（+MMS/壁面） | `ns_dg_2d.py`(1377)、`mms_ns_2d.py`、`ns_poiseuille.py` | ✅ | ✅ 11 测试文件 | ✅ | ⚠️ `validation/ns_dg_2d/mms_order.json` `all_within_quarter_order=false`（诚实 FAIL） | ✅ MMS 制造解 / Couette–Poiseuille 解析 | `test_ns_dg_2d_wall.py` 通过 + 2 `xfail(strict)` |
| 22 | NS 壁面热流阶（m13815） | `benchmarks/verify_ns_wall_heat_flux.py`；`validation/ns_heat_conduction/wall_flux_order.json` | ✅ | ✅ `tests/test_ns_wall_heat_flux_order.py` | ✅ 壁面通量 vs κ/2=0.24648 | ✅ 最细 P2 误差 8.5e-6（0.003%） | ✅ 解析等温壁导热 | `test_ns_wall_heat_flux_order.py` **8 passed (8 s)** |
| 23 | Thacker 长时 P2（m13810） | `validation/swe_thacker/run_thacker_longtime.py` | ✅ | ✅ `tests/test_thacker_longtime_p2.py` | ✅ | ✅ `thacker_longtime_results.json` `verdict=PASS`（体积误差 ≤1e-12） | ✅ Thacker 解析周期解 + EOC 带 | **6 passed (73 s)** |
| 24 | 岸线静止态（m10754） | `validation/swe_thacker/run_shoreline_rest.py` | ✅ | ✅ 工件契约测试 | ✅ | ✅ `shoreline_rest_state_results.json` | ✅ 静止解 | `test_swe_shoreline_rest_artifact.py` **7 passed (0.06 s)**（只校验工件自洽，不重解方程） |

### 2.1 与 PR/报告声明的差异（逐条对表）

| 声明（出处） | 代码实况 | 差异判定 |
|---|---|---|
| 「双 Mach 反射密度 1.4~22+，与 Woodward–Colella 一致」`docs/dg_euler_verification_report.md:67`（#247/m11149） | `euler_dg_2d` 只有周期/零梯度面 → 无反射壁、无精确后激波边界；`benchmarks/double_mach_reflection.py:19-30` 自承「triple point / Mach stem 无法形成」 | **声明不成立**（数字无工件、物理不可发生） |
| 「超声速楔形绕流激波角误差 <5°」同上 `:17`、`:73` | m11149 版 `_initial_wedge` 返回**均匀来流**（无楔、无激波）；m13761 已重写为 RH 精确态，但**仍无提交工件** | **声明不成立**（当时）；重写后代码 REAL，验证面缺工件 |
| 「Sod 误差 <2%、Lax 星区压力/速度」同上 `:49-56` | 数值量级与 `results/euler-sod-1d`、`lax-1d` 一致，但**该报告未附任何工件/运行 ID**；那些工件由其它 mission 产生 | **声明无独立证据**（借用别处结论且未溯源） |
| 「验证链建立」同上 `:9-18`（含双 Mach、楔形两格） | m13761 自审明确「应予撤回」`docs/reports/tensordg-oblique-shock-verification.md:111` | **已在别处撤回，原报告未加勘误、仍在 main** |
| 「zero-to-one ladder 机器门 C1–C6」（`README.md:49`） | `docs/benchmarks/results/verification_ladder/gates_verified.json` `verdict=FAIL`，C3 的 5 项 Sod 激波位置检查未过 | **口径夸大**：`docs/verification_ladder.md:96` 有 Known limitations 披露，但 README 宣传句未标 C3 FAIL |
| 「等温壁能量闭合不一致」`validation/ns_heat_conduction/evidence.json`（m3210, 11-04） | m13815（11-11）实测壁面通量按 h^p 收敛、最细 P2 误差 0.003%，判定「m3210 判据选错」`docs/reports/tensordg-ns-heat-wall-flux-diagnosis.md:136` | **旧工件结论已被推翻但未回写**（两工件并存、无 supersession 标记） |
| m10591「统一验证 5 gates 全 PASS」`docs/benchmarks/dg-unified-verification.md:23-27` | 工件逐 gate 值 + 阈值 + pass，provenance 齐全 | **声明与工件一致**（本轮核验通过） |
| m12101「2D 静水平衡 PASS」`docs/tensordg-swe-2d-wellbalanced.md:176` | 工件含 P2 对照：默认路径 u_max=1.1e-2、WB 路径 2.8e-12 | **一致**，且带负对照（诚实） |
| m13810「Thacker 长时 P2 PASS」`thacker_longtime_results.json:verdict` | verdict=PASS，体积误差 ≤1e-12，EOC 在带内 | **一致** |
| m13815「壁面通量误差 0.003%，闭合一致」`...wall-flux-diagnosis.md:135` | 工件 `wall_flux_order.json` 阶次 1.05/2.05 与实测误差一致 | **一致**（且推翻自己前序结论，诚实） |

### 2.2 本次实测的测试失败（真发现：main 全量 pytest 会红）

`tests/test_euler_verification_chain.py`（#247/m11149 建立，m13761 部分修补）在**当前 main** 上实跑：
**15 个用例中 5 个 FAIL**（本地逐用例复现，非并发/资源噪声）：

| 用例 | 实测 | 断言 | 机制判定 |
|---|---|---|---|
| `TestSodShockTube::test_sod_p2_50cells_full` | 密度单元均值误差 **2.993 %** | `< 2 %`（`:70`） | **阈值/声明过紧**：#247 报告写「<2%」，求解器实际 2.99%（README 口径的 Sod 是 P0/FV 2.5177%） |
| `TestLaxShockTube::test_lax_p2_full` | 未取到 traceback（单跑 >115 s 被工具收割） | `< 3 %`（`:129`） | 未能定性（honest：not_run 完整输出） |
| `TestVortexConvergence::test_vortex_p3_eoc` | 未取到 traceback | `EOC ≥ 3.8`（`:167` 起） | 未能定性（honest） |
| `TestUniformConvergence::test_uniform_p1_eoc` | **EOC = −0.004** | `≥ 1.8`（`:216`） | **测试设计错误**：把 t=0.2 的数值密度与 **t=0 的初值** 比（`:207` `rho_exact = 1 + 0.1 sin(2πx)`，非演化精确解），误差 O(1) 且不随 h 收敛 → 该用例在任何求解器下都不可能通过 |
| `TestUniformConvergence::test_uniform_p2_eoc` | 同上机制 | `≥ 2.8`（`:245`） | 同上 |

判定：**这是 #247 遗留的红测试**，与 m13761 无关——m13761 已把真正的「双 Mach / 楔形」两组
正确降级为 `xfail(strict=False)` + 精确初始态断言（见 `git show 2d87170 -- tests/test_euler_verification_chain.py`）。
仓库现状因此是：**乐观报告（未撤回）+ 自相矛盾的超标测试 + 一个数学上不可能通过的自证用例**。

### 2.3 工件契约测试 vs 求解器测试（口径提醒）

`tests/test_swe_shoreline_rest_artifact.py`（7 用例，0.06 s）等只校验**已提交 JSON 的自洽性**
（schema、阈值、值与判据一致性），**不重解方程**。它们证明「工件没被手改坏」，不证明
「今天重跑还能得到同样结果」。本报告在 L 判级时区分对待（契约测试 ≠ L4）。

---

## 3. ③ 成熟度判级（逐 scope 落表）

| scope | L | 证据指针（文件:行 / 工件 / 测试名 / PR#） | 说明 |
|---|---:|---|---|
| `advection_1d` + `advection_2d` | **4** | `tensordg/advection_1d.py:76`；`verification_ladder/gates_verified.json` C1/C1b PASS（EOC 2.041/2.999/3.990、1.972/2.999/4.023）；`tests/test_verification_ladder.py` 16 passed | 完整算例 + 收敛阶 + 解析基准 |
| `advection_2d_var` / `ice_edge_bench` | **2** | `advection_2d_var.py:144`、`ice_edge_bench.py`；`validation/verify_advection_2d_var.py`；仅测试接入 | 真算法，但无 CI、无提交工件、无端到端算例 |
| `advection_diffusion`（m10591） | **4** | `advection_diffusion.py:551`；`dg_unified_verification.json` E1 0.0156 / E2 0.0166 / E5 5.2e-4 PASS，散度 5.0e-11；`test_advection_diffusion_2d.py` 15 passed | 解析解 + 质量守恒 + 算子一致性 |
| `advection_diffusion_1d` | **3** | `results/advection_diffusion/metrics.json`：Pe=1/10 各阶通过，Pe=100 P1 阶 1.627（FAIL 入账） | 有工件，但高 Pe 未达标 → 不升 4 |
| `dispersive`（m10591） | **4** | `dispersive.py:162/340/406`；Airy 1.9e-16、KdV 7.1e-5、E3/E4/E5 PASS；`test_dispersive_waves.py` 10 passed | 模型表 + 传播算例 + 门 |
| `burgers_1d` / `burgers_shock_1d` | **4** | `results/burgers-1d/metrics.json` passed；`docs/burgers_shock_verification.md`（RH 位置到舍入、Gibbs 界） | 平滑 OOA + 激波位置 |
| `convergence`（GCI/Richardson） | **2** | `tensordg/convergence.py`；`results/convergence/metrics.json` passed | 判据工具，非物理能力 |
| `curved_geometry`（m9198） | **4** | `benchmarks/verified/dg_curved_geometry/result.json`（J1 ≤8.7e-16、J2 ≤2.2e-16、all_gates_pass）；`test_curved_geometry.py` **46 passed (459 s)** | 自由流保持 + 度规恒等 + 阶次 |
| `differentiable`（反问题） | **4** | `validation/diffusive_differentiability/evidence.json`（1.3e-3）；`docs/tutorials/differentiable-inverse-problems.md`（故障注入 37.5% 假收敛） | 独立基准 + 负对照 |
| `euler_1d` / `euler_dg_1d` | **4** | `results/euler-sod-1d`（2.5177%）、`results/lax-1d`（2.7694%）；精确 Riemann 独立参照 `benchmarks/riemann_exact.py` | 一维激波面成熟（注：`test_euler_verification_chain.py` 中 #247 写的 tighter 阈值会红，见 §2.2） |
| `euler_2d` / `euler_dg_2d`（光滑部分） | **4** | `results/isentropic-vortex-2d`（1.973%）；`test_euler_verification_chain.py` 的 vortex/uniform/conservation 组 | 光滑涡 + 设计阶 |
| `euler_2d` 强激波（双 Mach / 楔形绕流） | **1** | `double_mach_reflection.py:19-30`（自承无反射壁）；`supersonic_wedge.py`（无提交工件）；两组测试已被 m13761 转 xfail | 只有初始态/RH 公式级正确性；#247 声明不成立 |
| 斜激波精确验证（m13761） | **4** | `benchmarks/oblique_shock.py`；`results/oblique_shock/results.json`（β 0.303°、ρ 4.2e-6、p 6.6e-6、保持 2.31%）；`test_oblique_shock.py` **7 passed (480 s)** | 精确解 + 多工况 + 负对照（不敏感已披露） |
| `heat_1d` | **4** | `results/heat-1d/metrics.json`（3.6e-6） | |
| `limiters` / `tvb` | **4** | `docs/limiter_policy_verification.md`、`limiter_2d_verification.md`；`tests/test_tvb.py` 等 | 判据单源 + Gibbs 界 + 阶次不退化 |
| `poisson_2d`（稠密 SIPG） | **4** | `results/poisson-2d/metrics.json`（0.380%）；CI `python -m benchmarks.poisson_2d` | |
| `sipg_sparse` / `sipg_matrixfree` | **2** | `sipg_sparse.py:246`、`sipg_matrixfree.py:263`；仅单测 | 真算法但无基准/性能工件、无 CI 步骤 → 不得高于 2 |
| `swe_1d`（含静水平衡/岸线） | **4** | `swe_1d.py:422/168`；`thacker_longtime_results.json` PASS；`shoreline_rest_state_results.json`；`test_swe_1d.py` **30 passed (345 s)**、`test_thacker_longtime_p2.py` **6 passed** | 解析解 + 体积守恒 1e-12 + 收敛带 |
| `swe_2d`（m12101 静水平衡） | **4** | `swe_2d.py:339/112`；`wellbalanced_2d_results.json`（P2：默认 u 1.1e-2 vs WB 2.8e-12；溃坝质量漂移 2.2e-15）；`test_swe_2d_well_balanced.py` **17 passed** | 带负对照（P1 两路径同为舍入级，P2 才辨别）——工件如实保留 |
| `swe_exact` | **2** | `tensordg/swe_exact.py`（Stoker/Ritter/Thacker）；被 8 处测试消费 | 参照库本体 |
| `ns_dg_2d` + MMS | **3** | `validation/ns_dg_2d/mms_order.json`（`all_within_quarter_order=false`）；`test_ns_dg_2d_wall.py`（2 strict xfail 记录开放缺陷） | 有 MMS 研究但未全部达标 → 不升 4 |
| NS 壁面热流量化（m13815） | **4** | `validation/ns_heat_conduction/wall_flux_order.json`（P2 8.5e-6=0.003%，阶 1.05/2.05）；`test_ns_wall_heat_flux_order.py` **8 passed** | 解析基准 + 阶次 + 判据证伪（自推翻前序结论） |
| `ns_poiseuille` / `mms_ns_2d` | **2** | `tests/test_ns_couette_poiseuille.py`、`test_mms_ns_2d.py` | 单测级 |
| docs-only 声明（#247 报告、README 宣传语） | **0** | `docs/dg_euler_verification_report.md` | 文档成果，不计功能 |

### 3.1 L 分布统计（26 条 verdict，以 `tensor_maturity_verdicts` 回读为准）

| 级别 | 条数 | 占比 | scope |
|---|---:|---:|---|
| L5 | 0 | 0 % | —（本轮无「此前未解问题 + 新方法」级证据） |
| L4 | 15 | 58 % | advection_1d+2d、advection_diffusion、burgers、curved_geometry、differentiable、dispersive、euler_1d、euler_2d 光滑、heat_1d、limiters+tvb、ns wall heat flux、oblique_shock、poisson_2d、swe_1d、swe_2d |
| L3 | 3 | 12 % | advection_diffusion_1d、ns_dg_2d+MMS、verification_ladder C1–C6（C3 FAIL） |
| L2 | 5 | 19 % | advection_2d_var+ice_edge_bench、convergence、ns_poiseuille+mms_ns_2d、sipg_sparse+sipg_matrixfree、swe_exact |
| L1 | 2 | 8 % | euler_2d 强激波（双 Mach/楔形）、euler_verification_chain（PR#247 / m11149） |
| L0 | 1 | 4 % | docs-only 声明（#247 报告、README 宣传语） |

> 判级说明：L4 的 15 个 scope 全部指向**已提交工件 + 本次执行过的测试**；
> 无工件的（sipg_*、advection_2d_var、ns_poiseuille）即便代码是真算法也只判 2。
> 有工件但未达判据的（高 Pe 对流扩散、NS MMS）判 3，不因报告篇幅升级。

---

## 4. Top 问题（按严重度）

| # | 严重度 | 问题 | 证据 | 建议修复 |
|---|---|---|---|---|
| P1 | **阻断** | main 上 `tests/test_euler_verification_chain.py` **5 个 FAIL**（/15）：阈值过紧 1 个 + 测试设计错误 2 个 + 未定性 2 个；CI `pytest -q` 必红 | §2.2 表；`test_sod_p2_50cells_full` → `density cell-average error 2.993% >= 2%`；`test_uniform_p1_eoc` → `P1 uniform EOC = -0.004 < 1.8` | Sod/Lax 阈值按仓库口径放宽并说明；`TestUniformConvergence` 两个用例改为与**演化精确解**比（或按 m13761 方式 `xfail(strict=True)` 登记设计缺陷） |
| P2 | **高** | `docs/dg_euler_verification_report.md`（#247）仍原样留在 main，其「双 Mach 密度 1.4~22+」「楔形角误差 <5°」无工件且已被后续 mission 证伪，无勘误标记 | `docs/dg_euler_verification_report.md:17,67,73`；撤回写在 `docs/reports/tensordg-oblique-shock-verification.md:111`；`git log` 该文件自 729240e 后未修改 | 文件头加「SUPERSEDED / 勘误」块并指向 m13761 报告；或归档到 `docs/archive/` |
| P3 | 中 | README 宣传「ladder 机器门 C1–C6」而工件 `verdict=FAIL`（C3 Sod 位置 5 项不过） | `README.md:49` vs `verification_ladder/gates_verified.json` `gates.C3.status=FAIL` | README 句补「C3 目前 FAIL（见 docs/verification_ladder.md Known limitations）」 |
| P4 | 中 | `validation/ns_heat_conduction/evidence.json` 结论「闭合不一致」已被 m13815 推翻，旧工件未回写、两工件无 supersession 关系 | 旧件 `git log` = bcd8af3(11-04)；新件 = 81ddba9(11-11)；`wall-flux-diagnosis.md:136` 声明改判 | 旧件加 `superseded_by` 字段或移入 `validation/ns_heat_conduction/archive/` |
| P5 | 中 | `validation/ns_dg_2d/mms_order.json` `all_within_quarter_order=false`，仓内**无任何文档披露**（grep 0 命中） | `validation/ns_dg_2d/mms_order.json`；`grep -rn "within_quarter" docs/ tests/` = 空 | 在 NS 文档/README 已知局限中登记 |
| P6 | 中 | `sipg_matrixfree`/`sipg_sparse`（971 行）+ `ice_edge_bench`（317 行）= **1,288 行核心代码只在单测里活**：无 benchmark、无 CI 步骤、无性能/残差工件 | §1.5；`grep -l sipg_ benchmarks/ validation/` = 0 | 补 CI smoke + 稠密/稀疏/无矩阵耗时与残差工件，或明确标注为实验 API |
| P7 | 中 | 变系数对流两套速度场实现（`advection_2d_var` vs `advection_diffusion`） | §1.4 | 择一为单源，另一处改 import |
| P8 | 低 | 工件 provenance 出现 `git_dirty=true`（`wellbalanced_2d_results.json`、`gates_verified.json`） | 两工件 `provenance.git_dirty` | 干净 worktree 重跑并记录 commit |
| P9 | 低 | 两个诚实 FAIL（`advection_diffusion` Pe=100、`riemann_vacuum` C2b 控制）未在 README 汇总面登记 | `results/advection_diffusion/metrics.json`、`validation/riemann_vacuum/vacuum_sweep.json` | 建 README「已知 FAIL 台账」小节 |

---

## 5. 「曾声称为功能、实为文档/测试/预设」膨胀清单

| 声明 | 实际是什么 | 膨胀性质 |
|---|---|---|
| 「双 Mach 反射验证通过（密度 1.4~22+）」#247 | 一段初始条件 + 定性断言；求解器无反射壁能力；无 JSON 工件 | **预设/不可发生** |
| 「超声速楔形绕流激波角误差 <5°」#247 | m11149 版构造的是均匀来流（`_initial_wedge` 均匀场），无楔无激波 | **假算例**（m13761 已重写并披露） |
| 「Euler 验证链建立」#247 | 1D Sod/Lax/涡三格确有工件（借用），2D 两格是上述两条 | **拼接式夸大**：把别处证据算进本 PR 的「链」 |
| README「ladder C1–C6」 | C3 FAIL 仍在，工件自己写着 FAIL | **口径夸大**（细节有披露，宣传句未标） |
| 「等温壁闭合不一致」m3210 | 判据本身有阶（P1 残差含 1/h 放大），m13815 用通量重测证伪 | **判据误用已自纠**，但旧结论未回写 |
| `test_swe_shoreline_rest_artifact` 7 用例 | 只校验 JSON 自洽，不重解方程（0.06 s 完成即证据） | **测试面 ≠ 求解面**（非缺陷，但不能当 L4 证据） |
| #247 的正向项 | Sod/Lax/涡三格确有工件，但其阈值比仓库口径更紧（2%），**今天跑都不过** | **测试写死超标断言**：验证未成立却已合入 |

---

## 6. 未完成 / 未能验证（honest）

1. **全量测试套件未跑完**：已递交集群作业 `cj837`（`python3 -m pytest tests -q` @ `TensorDG@main`），审计窗口内排队 20+ 分钟未派发（not_run）。
   本地按文件执行 **12 个文件（11 个跑完 + 1 个跑完 14/15）**，覆盖 **203 passed / 5 failed / 7 skipped+xfail**（合计 215/1091 ≈ 19.7 %）。
2. `tests/` 102 个文件里本次实跑 12 个（§7），其余靠工件核验 + 静态阅读 → 未跑即记 `not_run`，不记 pass。
3. 基准算例重跑：仅重跑 `oblique_shock` 的 pytest 路径（7 passed）；`benchmarks/oblique_shock.py` 单 case 在本机 >110 s 被工具超时收割，**工件数值未逐位重生成比对**。
4. 未审计 TensorDG 之外的跨仓契约（TensorPlatform 平台侧集成 / TensorPhysics 编排），本轮 scope 限单仓。
5. 审计机与集群同为 loong64；若与 GitHub CI（x86）不一致，判读需注架构。
6. 我**没有**修任何代码（本单为审计单）：P1 的红测试只定位、不改。

---

## 7. 本次实跑记录（命令 + 结果）

| 命令（cwd=worktree） | 结果 | 耗时 |
|---|---|---|
| `python3 -m pytest tests --collect-only -q -p no:cacheprovider` | **1091 tests collected** | 13.8 s |
| `python3 -m pytest tests/test_swe_2d_well_balanced.py -q` | **17 passed** | 49.7 s |
| `python3 -m pytest tests/test_dispersive_waves.py -q` | **10 passed** | 18.3 s |
| `python3 -m pytest tests/test_advection_diffusion_2d.py -q` | **15 passed** | 71.1 s |
| `python3 -m pytest tests/test_ns_wall_heat_flux_order.py -q` | **8 passed** | 8.2 s |
| `python3 -m pytest tests/test_swe_shoreline_rest_artifact.py -q` | **7 passed** | 0.06 s |
| `python3 -m pytest tests/test_swe_hydrostatic_reconstruction.py -q` / `test_swe_wellbalanced_hydrostatic.py -q` | 启动后截稿前无最终摘要 → **not_run（不记 pass）** | — |
| `python3 -m pytest tests/test_verification_ladder.py -q` | **16 passed, 3 skipped** | 28.8 s |
| `python3 -m pytest tests/test_swe_1d.py -q` | **30 passed** | 345 s |
| `python3 -m pytest tests/test_thacker_longtime_p2.py -q` | **6 passed** | 73.1 s |
| `python3 -m pytest tests/test_curved_geometry.py -q` | **46 passed** | 459 s |
| `python3 -m pytest tests/test_oblique_shock.py -q` | **7 passed** | 480 s |
| `python3 -m pytest tests/test_ns_dg_2d_wall.py -q` | **32 passed, 4 xfailed** | 335 s |
| `python3 -m pytest tests/test_euler_verification_chain.py -q` | **9 passed / 5 failed**（14/15 跑完；第 15 个 supersonic wedge P2 120×60 在截稿时仍在跑）（§2.2） | >40 min |
| `python3 -m pytest "tests/test_euler_verification_chain.py::TestSodShockTube" -q --tb=short` | **1 failed**：`density cell-average error 2.993% >= 2%` | 100.7 s |
| 集群 `cj837`：`pytest tests -q` @ TensorDG main | **queued，未派发**（not_run） | — |

---

## 8. 复现（审计脚本）

```bash
# 1) 逐 mission 分类盘点（core/tests/docs/bench/validation/tools）
python3 - <<'PY'
import subprocess, collections, re
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
def cat(p):
    if p.startswith('tensordg/'): return 'core'
    if p.startswith('tests/'): return 'tests'
    if p.startswith('docs/') or p.endswith('.md'): return 'docs'
    if p.startswith('benchmarks/'): return 'bench'
    if p.startswith('validation/'): return 'val'
    return 'tools' if p.startswith(('tools/','scratch/')) else 'other'
by = collections.defaultdict(list)
for L in sh("git log --format='%h|%p|%ct|%s'").strip().split('\n'):
    h, p, ct, s = L.split('|', 3); m = re.search(r'\((m\d+)\)', s)
    if m: by[m.group(1)].append((int(ct), h, p.split()[0]))
tot = collections.Counter()
for mid, cs in by.items():
    cs.sort(reverse=True)
    for ct, h, par in cs:                      # 同 mission 取最大差异提交
        st = sh(f"git diff --numstat {par} {h}"); loc = collections.Counter()
        for line in st.strip().split('\n'):
            if not line.strip(): continue
            a, d, p = line.split('\t'); loc[cat(p)] += int(a) if a.isdigit() else 0
        if sum(loc.values()): tot += loc; break
print(len(by), 'missions', dict(tot))
PY

# 2) 占位实现 AST 扫描（stub / 常量返回）
python3 - <<'PY'
import ast, glob
stub = []; const = []
for p in sorted(glob.glob('tensordg/*.py')):
    for n in ast.walk(ast.parse(open(p).read())):
        if isinstance(n, ast.FunctionDef):
            body = [b for b in n.body if not (isinstance(b, ast.Expr)
                    and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str))]
            if len(body) == 1:
                b = body[0]
                if isinstance(b, (ast.Pass, ast.Raise)): stub.append((p, n.lineno, n.name))
                if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) \
                        and b.value.value is Ellipsis: stub.append((p, n.lineno, n.name))
                if isinstance(b, ast.Return) and isinstance(b.value, ast.Constant):
                    const.append((p, n.lineno, n.name, b.value.value))
print('stub:', stub); print('const-return:', const)
PY

# 3) 接入分析（模块 → 外部引用方）
python3 - <<'PY'
import glob, re
mods = [p.split('/')[-1][:-3] for p in glob.glob('tensordg/*.py') if not p.endswith('__init__.py')]
corpus = {p: open(p, errors='ignore').read() for d in
          ('tensordg','tests','benchmarks','validation','tools','scratch')
          for p in glob.glob(d + '/**/*.py', recursive=True)}
for m in sorted(mods):
    files = [p for p, t in corpus.items() if p != f'tensordg/{m}.py' and re.search(rf'\b{m}\b', t)]
    print(f"{m:22s} refs={len(files):3d} test_only={all(p.startswith('tests/') for p in files)}")
PY

# 4) 工件判据核验（示例：全部 JSON 里的 verdict/gates/pass）
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob('docs/**/*.json', recursive=True) + glob.glob('validation/**/*.json', recursive=True)):
    d = json.load(open(f))
    hit = {k: v for k, v in d.items() if isinstance(v, (int, float, str)) and
           any(s in k.lower() for s in ('verdict', 'passed', 'all_gates_pass', 'all_within'))}
    if hit: print(f, hit)
PY

# 5) 红测试定位（P1）
python3 -m pytest tests/test_euler_verification_chain.py -q -p no:cacheprovider --tb=line
```

---

## 9. verdicts 落表（tensor_maturity_verdicts）

见 `docs/audit_verdicts.sql`（本单执行的 INSERT 语句原文）与 §9.1 回读结果。
`verdict_by='audit2-ab'`，`mission_id=14490`，每条 evidence 含文件:行号 / 工件路径 / 测试名。

### 9.1 落表回读（`psql -U foundation -d tensorplatform`）

```
SELECT level, count(*) FROM tensor_maturity_verdicts
 WHERE repo='TensorDG' AND verdict_by='audit2-ab' GROUP BY level ORDER BY level DESC;
 level | count
-------+-------
     4 |    15
     3 |     3
     2 |     5
     1 |     2
     0 |     1
```

- 共 **26 条**，`repo='TensorDG'`、`mission_id=14490`、`verdict_by='audit2-ab'`；
- INSERT 语句原文：`docs/audit_verdicts.sql`（只 INSERT，未 ALTER 表）；
- 每条 `evidence` 为 JSONB，含 `entry`（文件:行）/ `artifact`（工件路径）/ `test`（测试名与本次运行结果）/ `gap`（缺口）。

---

## 10. 平台报告与图

- 论文级进展报告已发布：`POST /api/v1/tensor/reports` → **report_id = 13605**（mission 14490）。
- 图（程序化生成，数据源 = git numstat / DB 回读 / pytest 日志）：
  - `docs/figures/audit_asset_inventory.png`（78 mission 增量分类）
  - `docs/figures/audit_maturity_levels.png`（L 分布）
  - `docs/figures/audit_test_status.png`（逐文件 pytest 结果）

---

## 11. 技能产物对表（codebase-investigation / adversarial-review）

### 11.1 codebase-investigation 必交产物

| 项 | 内容 |
|---|---|
| 仓库 / ref / commit | `cxs503/TensorDG`，`main` @ `81ddba9`，worktree `agent/m14490`（干净，见 `git status`） |
| 已检视路径 | `tensordg/*.py`(30)、`tests/*.py`(102 中 12 实跑、其余静态)、`benchmarks/dg/*`、`benchmarks/verified/dg_curved_geometry/*`、`validation/{swe_thacker,swe_2d_wb,ns_dg_2d,ns_heat_conduction,diffusive_differentiability,gate_power}/*`、`docs/benchmarks/results/*`、`docs/{dg_euler_verification_report.md, verification_ladder.md, DG_CURVED_GEOMETRY.md, tensordg-*.md}`、`.github/workflows/tests.yml`、`pyproject.toml`、`README.md`、`AGENTS.md` |
| 架构与执行地图 | 入口 = `tensordg/__init__.py`（69 符号）+ 各 `solve_*`；算例层 = `benchmarks/*`（CI 直接跑 12 条命令）；验证层 = `validation/*/run_*.py → *.json`；门禁 = `tests/*` + `gates_verified.json` 契约测试 |
| 测试与覆盖局限 | 1091 用例；慢块由环境变量门控（`TENSORDG_LADDER_SLOW`）→ 默认 skip 3 个；契约式测试（0.06 s）不重解方程 |
| 并发变更注记 | 本审计期间 main 未变；最后一次合入为 `81ddba9`（m13815）。审计未与其它写者冲突（本单只新增 `audit_report.md`、`docs/audit_verdicts.sql`、3 张图） |
| 运行时检查 | §7 表（14 条命令）；集群 `cj837` pending |
| 确认事实 / 推断 / 未知 | **确认（runtime）**：5 个红测试中的 3 个机制、12 文件 203 passed；**确认（source）**：双 Mach 无反射壁、楔形无工件、C3 FAIL、MMS 未达标、1,288 行孤岛模块；**未知（not_run）**：`test_lax_p2_full` / `test_vortex_p3_eoc` 的具体断言值（>115 s，未取到 traceback）、全量 1091 用例结果 |
| 边界风险 | (a) 若 CI 在 x86 上跑，浮点差异可能改变 2.993 % vs 2 % 的边缘判定——但 EOC=−0.004 的用例与架构无关，红是确定的；(b) 修改测试须避免「把 FAIL 改绿」式掩盖 |
| 最小下一步 | `pytest tests/test_euler_verification_chain.py -q` 复现（>40 min，建议集群）→ 把 `TestUniformConvergence` 两用例改为与细网格自收敛参照比（或 `xfail(strict=True)` 登记设计缺陷）→ 报告加勘误块 |

### 11.2 adversarial-review 必交产物

- **被审对象**：`81ddba9` + 78 个 mission 的合入内容（重点 #247/m11149、#251/m13815、#249/m13761、#248/m12101、#245/m10591、#241/m9198、#243/m9711、#240/m8805、#246/m10754、#244/m8805）。
- **独立性限制**：审计员与被审 mission 无共同作者关系；但受限于（i）全量套件未跑完、（ii）部分重算被工具超时收割、（iii）无法访问 PR 讨论与合并门禁日志。
- **已做检查**：静态（AST/接入/重复/工件判据）+ 运行时（12 文件 pytest）。
- **未做检查（not_run，不记 pass）**：全量 1091 用例；`oblique_shock` 工件逐位重生成；x86 CI 环境复现。
- **严重度排序发现**：P1 红测试（阻断，含复现命令）→ P2 未撤回报告（高）→ P3/P4/P5 口径与工件状态（中）→ P6/P7 孤岛与重复（中）→ P8/P9（低）。
- **处置（disposition）**：**request changes** —— P1 必须先修（CI 绿），P2 必须加勘误；L4 的 15 个 scope 判定**approve with notes**（notes = provenance `git_dirty`、负对照不敏感等已披露项）。
