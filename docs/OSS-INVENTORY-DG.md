# FEAT-OSS-INVENTORY-DG · 开源 DG 技术盘点（波 1：普查 + tier 分级 + 锚点）

> 产物：`docs/oss_inventory_dg_registry.json`（主册）· `artifacts/oss_inventory_dg/github_probe_raw.json`
> （GitHub API 原始探针，sha256 入册）· 生成器 `tools/build_oss_inventory_dg.py`（`--check` 防漂移）
> · 校验器 `tests/test_oss_inventory_dg_registry.py`（13 项，全绿）
> 协议：接力 LBM 波 1（`oss_inventory` 表 33 仓为 LBM 域），本册为 **DG 域首批**，wave-2 起逐仓代码级深盘。

## ① 案例介绍：为什么盘 DG 生态

TensorDG 是 12 仓租户制里的 DG 求解器库，当前能力面（读码核实）为：**张量积/笛卡尔均匀网格**上的
模态 Legendre DG，RK 时间推进，SIPG/BR1 扩散与可压 NS 壁面路径，TVB/WENO 限幅，
等参曲线几何仅覆盖标量 advection，另有一维/二维 Euler 与 SWE、LDG Poisson 与矩阵自由 SIPG。
能力空白集中在五处：**ADER/局部时间步进、三维 Euler、h-自适应、HDG 静态凝聚、曲线网格上的 Euler/GCL、
GPU/多后端矩阵自由核**。

盘点目标（总账锚点 `FEAT-OSS-INVENTORY-DG` 验收原文）：*普查 ≥20 仓 registry JSON + tier 分级 +
GAP-DG-* 锚点提议 ≥4*。本册给出 24 仓（探针验证 30 个、
另 6 个验明但按理由缓入 reserve）、tier 分布
T1=9 / T2=8 / T3=7、
GAP 提议 6 条。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **候选发现**：GitHub Search API（`discontinuous galerkin`、`DG solver CFD`）取 stars 排序真实仓，
   避免凭记忆猜仓名（首轮手写候选 **20/51 命中 404**——已弃用该路径，见 ⑤）。
2. **元数据探针**：逐仓 `GET api.github.com/repos/{owner}/{repo}`（只读、无 token），
   落盘 `github_probe_raw.json`（stars/forks/language/licence/pushed_at/archived/size/description/topics）。
3. **零手抄合册**：`tools/build_oss_inventory_dg.py` 把探针 JSON 与人工判断层（tier/role/focus/caps/gaps）
   **程序化合流**；脚本大小写规范化（`PyFR/PyFR`、`Nek5000/nekRS`、`trilinos/Trilinos`）。
4. **tier 判据**（写进 registry 的 `tier_rubric`）：
   T1 = DG/FR 主体求解器（算法谱系与验证 oracle 可直接对标）；T2 = DG 使能库/含 DG 组件的框架；
   T3 = DG 生态基础设施（AMR / 性能可移植 / 多物理宿主 / 大规模 HPC）。
5. **锚点映射**：只认 TensorDG **读码确认的**空白 → `GAP-DG-*`，每条锚点名 ≥2 个在册仓为证据。
6. **机器校验**：13 项 pytest 断言（规模/tier 合法性/去重/字段完备/**stats 与探针逐字段相等**/
   探针 sha256/活跃度 ≥60% 一年内/锚点 slug 规范且判据量化/每条锚点 ≥2 仓/覆盖统计自洽/再生一致）。

## ③ 计算结果（数字表；由注册表程序化渲染）

### 3.1 普查表（24 仓，按 tier 排序）

| # | repo | tier | role | stars | licence | last push | focus |
|---|---|---|---|---|---|---|---|
| 1 | [mfem/mfem](https://github.com/mfem/mfem) | 1 | core-dg-solver | 2291 | BSD-3-Clause | 2026-11-12 | 含成熟 DG 离散组件的高性能 FEM 库：DG 弱形式、面/边界积分与 HDG 的权威参照，TensorDG 的 SIPG/LDG 面项实现可直接对照 |
| 2 | [trixi-framework/Trixi.jl](https://github.com/trixi-framework/Trixi.jl) | 1 | core-dg-solver | 737 | MIT | 2026-11-12 | Julia 自适应高阶 DG 守恒律框架：树结构 AMR + 多种数值通量/限幅器组合，其 elixir 组织方式与收敛验证范式可移植 |
| 3 | [PyFR/PyFR](https://github.com/PyFR/PyFR) | 1 | core-dg-solver | 585 | NOASSERTION | 2026-11-12 | 通量重构(FR/CPR)型 DG 的 GPU 参考实现：同一高阶离散在多种后端的可移植写法，是 TensorDG「矩阵自由 + 可移植后端」演进路线的直接对照 |
| 4 | [ucns3d-team/UCNS3D](https://github.com/ucns3d-team/UCNS3D) | 1 | core-dg-solver | 366 | GPL-3.0 | 2026-11-11 | 非结构可压 NS 三维 DG：非结构网格上的 DG 数据组织（面-单元邻接表），是 TensorDG 从张量积笛卡尔网格走向非结构的关键参照 |
| 5 | [SeisSol/SeisSol](https://github.com/SeisSol/SeisSol) | 1 | core-dg-solver | 351 | BSD-3-Clause | 2026-11-12 | ADER-DG + 局部时间步进的代表实现（地震波）：TensorDG 仅有 RK 时间积分，时空预测校正路径的方法学来源 |
| 6 | [flexi-framework/flexi](https://github.com/flexi-framework/flexi) | 1 | core-dg-solver | 299 | GPL-3.0 | 2026-11-01 | 高阶 DG 双曲-抛物框架（Fortran/MPI）：面通量、粘性项与限幅器的工程化组织方式，对 TensorDG 的 NS 面通量/BR1 路径是同级参照 |
| 7 | [exadg/exadg](https://github.com/exadg/exadg) | 1 | core-dg-solver | 194 | GPL-3.0 | 2026-11-11 | 矩阵自由高阶 DG 的 exa-scale 实现（基于 deal.II）：算子求值不组装矩阵的组织范式，对应 TensorDG 的 block-Jacobi/矩阵自由后端欠账 |
| 8 | [horses-framework/horses3d](https://github.com/horses-framework/horses3d) | 1 | core-dg-solver | 167 | MIT | 2026-11-12 | 高阶 DG 流动求解器（含曲线网格与湍流模型）：三维 DG 的工程化流程参照，覆盖 TensorDG 尚未触及的三维可压路径 |
| 9 | [gkeyllorg/gkeyll](https://github.com/gkeyllorg/gkeyll) | 1 | core-dg-solver | 120 | MIT | 2026-11-12 | DG 等离子体动理学框架：DG 在非流体系统（Vlasov/矩系统）上的应用与守恒维护，可作为 DG 通用性的对照证据 |
| 10 | [dealii/dealii](https://github.com/dealii/dealii) | 2 | dg-capable-framework | 1758 | NOASSERTION | 2026-11-12 | 通用 FEM 框架，其 matrix-free DG 与几何（等参曲线单元）基础设施被 exadg 等 DG 代码复用；曲线网格几何守恒律实现参照 |
| 11 | [FEniCS/dolfinx](https://github.com/FEniCS/dolfinx) | 2 | dg-capable-framework | 1248 | LGPL-3.0 | 2026-11-11 | 支持 DG 单元与 UPWIND 面通量的现代 FEM 前端：DG 弱形式声明式表达参照，对 TensorDG 的 API 组织有借鉴 |
| 12 | [libxsmm/libxsmm](https://github.com/libxsmm/libxsmm) | 2 | enabling-kernel | 984 | BSD-3-Clause | 2026-11-10 | 小矩阵/张量收缩专用库：DG 单元局部算子（(p+1)^d 量级小矩阵）的性能关键核，是 TensorDG 局部算子加速的直接参照 |
| 13 | [libMesh/libmesh](https://github.com/libMesh/libmesh) | 2 | dg-capable-framework | 763 | LGPL-2.1 | 2026-11-12 | 支持 DG 的通用有限元框架（面-邻接与悬挂节点处理成熟）：DG 网格数据结构的工程参照 |
| 14 | [firedrakeproject/firedrake](https://github.com/firedrakeproject/firedrake) | 2 | dg-capable-framework | 682 | NOASSERTION | 2026-11-12 | 含 DG/HDG 的生成式有限元系统：HDG 静态凝聚（trace 变量）的实现谱系参照 |
| 15 | [gismo/gismo](https://github.com/gismo/gismo) | 2 | dg-capable-framework | 441 | MPL-2.0 | 2026-11-12 | 等几何/B 样条求解库：TensorDG 已有 B 样条基导数矩阵工具，等几何曲线几何可作扩展参照 |
| 16 | [feelpp/feelpp](https://github.com/feelpp/feelpp) | 2 | dg-capable-framework | 336 | LGPL-3.0 | 2026-11-12 | 含 DG 组件的 C++ FEM 库：多物理宿主中 DG 的集成方式参照 |
| 17 | [paranumal/libparanumal](https://github.com/paranumal/libparanumal) | 2 | enabling-kernel | 172 | MIT | 2026-04-24 | GPU 加速高阶有限元/DG 流求解器与核集合：高性能后端组织（OCCA/并行核）参照 |
| 18 | [kokkos/kokkos](https://github.com/kokkos/kokkos) | 3 | ecosystem-infrastructure | 2752 | NOASSERTION | 2026-11-11 | 性能可移植编程模型：TensorDG 若需多后端（CPU/GPU）算子，其抽象层是业界标准参照 |
| 19 | [idaholab/moose](https://github.com/idaholab/moose) | 3 | ecosystem-infrastructure | 2406 | LGPL-2.1 | 2026-11-12 | 多物理应用宿主框架（含 DG 内核）：DG 在大型多物理应用中的集成与验证组织参照 |
| 20 | [trilinos/Trilinos](https://github.com/trilinos/Trilinos) | 3 | ecosystem-infrastructure | 1444 | NOASSERTION | 2026-11-12 | 大规模并行数值栈（含 DG 离散包与代数多重网格）：exa-scale 求解层参照 |
| 21 | [AMReX-Codes/amrex](https://github.com/AMReX-Codes/amrex) | 3 | ecosystem-infrastructure | 831 | NOASSERTION | 2026-11-12 | 块结构 AMR 框架：自适应网格与负载平衡的工程范式（被多物理代码复用） |
| 22 | [petsc/petsc](https://github.com/petsc/petsc) | 3 | ecosystem-infrastructure | 550 | NOASSERTION | 2026-11-11 | 线性/非线性求解器与 DMPlex 网格抽象：DG 面项装配后的求解层参照 |
| 23 | [Nek5000/nekRS](https://github.com/Nek5000/nekRS) | 3 | ecosystem-infrastructure | 431 | NOASSERTION | 2026-05-24 | 谱元 GPU 大规模流动求解器：高阶连续谱元 + GPU 的工程实现，与高阶 DG 在后端/并行策略上可交叉借鉴 |
| 24 | [DLR-AMR/t8code](https://github.com/DLR-AMR/t8code) | 3 | ecosystem-infrastructure | 344 | GPL-2.0 | 2026-11-12 | 树结构并行 AMR 数据结构（含 DG 悬挂节点/2:1 平衡约定）：h-自适应的数据层参照 |

### 3.2 GAP-DG-* 锚点提议（6 条）

| slug | title | 判据（量化验收） | 证据仓 | 优先级 |
|---|---|---|---|---|
| `GAP-DG-ADER-TIME` | ADER-DG 时空预测校正时间积分（对照 RK 基线） | 1-D Euler Sod 与 2-D isentropic vortex 在 p=1,2,3 下：观测收敛阶 >= p+1-0.15；vortex L2 密度误差 <1%；同一 CFL 下与 RK 基准解差 <1e-8；守恒量漂移 <1e-12；新增单测数 >=3 | SeisSol/SeisSol, gkeyllorg/gkeyll | P1 |
| `GAP-DG-3D-EULER` | 三维可压 Euler DG（现仅三维周期 advection） | 3-D isentropic vortex 在 p=1,2,3 上 h-收敛阶 >= p+0.85；L2 密度误差 <1%；8^3 单元 p=2 单节点运行 <600 s；无负密度/负压；守恒漂移 <1e-12 | horses-framework/horses3d, ucns3d-team/UCNS3D, trixi-framework/Trixi.jl | P1 |
| `GAP-DG-AMR` | h-自适应（树结构 AMR + 2:1 平衡） | 2-D vortex advection 上：2:1 平衡成立（机器检查）、悬挂节点面通量守恒（漂移 <1e-11）；等自由度下误差 <= 0.7x 均匀网格误差；提供回归测试 | DLR-AMR/t8code, AMReX-Codes/amrex, trixi-framework/Trixi.jl | P2 |
| `GAP-DG-HDG-POISSON` | HDG（静态凝聚）与现有 LDG/SIPG 的精度-自由度效率对照 | 2-D Poisson p=1..3：HDG L2 误差 <= 1.2x SIPG 同自由度误差；全局未知量（凝聚后）相对 SIPG 减少 >=30%；报告自由度表与收敛阶；新增单测 >=2 | mfem/mfem, firedrakeproject/firedrake, FEniCS/dolfinx | P2 |
| `GAP-DG-CURVED-EULER` | 曲线/等参网格上的 Euler（现曲线几何仅覆盖 advection） | 曲线网格上自由流保持误差 <1e-12（GCL 机器检查）；warped mesh 上 vortex 收敛阶 >= p+1-0.2；与笛卡尔网格同算例误差比 <=2x | PyFR/PyFR, dealii/dealii, gismo/gismo | P2 |
| `GAP-DG-MATRIXFREE-GPU` | 矩阵自由/多后端（GPU）DG 算子路径 | 2-D Euler p=2 上：新后端吞吐 >=5x 现有 CPU 基线；与基线解差 <1e-10；后端可切换且默认路径不变（回归全绿） | PyFR/PyFR, libxsmm/libxsmm, paranumal/libparanumal, kokkos/kokkos | P3 |

### 3.3 能力覆盖矩阵（按注册表标签统计）

| capability | 覆盖仓数 | 覆盖仓 |
|---|---|---|
| `h-adaptivity` | 13 | mfem, Trixi.jl, SeisSol, exadg, horses3d, dealii, dolfinx, libmesh, firedrake, feelpp, moose, amrex, t8code |
| `matrix-free` | 11 | mfem, exadg, dealii, dolfinx, libxsmm, firedrake, libparanumal, kokkos, Trilinos, petsc, nekRS |
| `curved-isoparametric` | 7 | PyFR, UCNS3D, flexi, horses3d, dealii, libmesh, gismo |
| `gpu-portability` | 7 | mfem, PyFR, exadg, libparanumal, kokkos, Trilinos, nekRS |
| `basis-nodal` | 5 | mfem, PyFR, UCNS3D, flexi, horses3d |
| `shock-limiter` | 3 | Trixi.jl, UCNS3D, flexi |
| `basis-modal` | 2 | gkeyll, gismo |
| `p-adaptivity` | 2 | Trixi.jl, dealii |
| `ader-lts` | 2 | SeisSol, gkeyll |
| `verification-suite` | 2 | mfem, Trixi.jl |
| `plasma-kinetic` | 1 | gkeyll |
| `elastic-seismic` | 1 | SeisSol |
| `fsi-coupling` | 0 | —（零覆盖，暴露 TensorDG 空白方向） |

### 3.4 覆盖与判据对照

| 验收项（总账原文） | 要求 | 实测 | 是否达标 |
|---|---|---|---|
| 普查仓数 | ≥20 | 24（另验明 30 个，reserve 6） | ✅ |
| tier 分级 | 必须有 | T1/T2/T3 = 9/8/7 | ✅ |
| GAP-DG-* 锚点 | ≥4 | 6 | ✅ |
| registry JSON 可机检 | — | `tests/test_oss_inventory_dg_registry.py` 13 项全绿 | ✅ |
| 元数据零手抄 | — | stats 与探针逐字段相等（断言）+ sha256 入册 | ✅ |

## ④ 结果图

![tier 与活跃度](fig1_tier_activity.png)
*图 1 — 左：tier 计数（虚线=均分参考）；右：stars vs 距末次推送天数（对数纵轴，标注高星仓）。*

![能力覆盖矩阵](fig2_capability_matrix.png)
*图 2 — DG 能力 × 仓 覆盖矩阵（按 tier 分块，红线为档界）。零覆盖列即 TensorDG 的空白方向。*

![锚点证据强度](fig3_gap_evidence.png)
*图 3 — 每条 GAP-DG-* 的被锚定仓数（蓝=与在册仓关联总数，绿=显式证据列表），红线=验收下限 2。*

> 说明：本任务属**生态盘点/基础设施**类，无物理场量可画云图，故以数据图（tier 分布、能力矩阵、
> 锚点强度）代替场量云图；所有图由注册表程序化生成，无手抄数值。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

- **坑 1：凭记忆猜仓名不可靠。** 首轮 51 个手写候选 **20 个 404**（如 `nektar/nektar`、`p4est/p4est`、
  `exahype/ExaHyPE-Engine`、`dune-fem/dune-fem` 均已迁移/改名）。改用 Search API 后发现真实仓
  （UCNS3D、HORSES3D、t8code、libxsmm）。**结论：生态盘点必须以检索为准，记忆作假设不作事实。**
- **坑 2：GitHub owner 大小写归一。** 探针返回 `PyFR/PyFR`、`Nek5000/nekRS`、`trilinos/Trilinos`，
  与手写 key 不一致会让 GAP 证据交叉引用静默断链；已由校验器 `test_gap_proposals_are_quantified_and_evidenced`
  抓出并修为大小写规范化（这是本册唯一一次测试先红后绿的真实缺陷）。
- **局限 1（重要）**：波 1 只做**公开元数据 + 领域判断**，**未逐仓 clone 做代码级方法核查**——
  `capabilities` 标签依据仓描述与公开文档，属**保守启发式**，不是文件级证据。wave-2 必须按 LBM 协议
  逐仓给出「文件-行」级定位，否则不得升格为能力。
- **局限 2**：tier 分级为启发式判断，非度量；跨档边界仓（libMesh、feelpp、Kokkos）在 `focus` 中说明理由。
- **局限 3**：stars/pushed_at 是探针时刻快照，会漂移；`--check` 模式可检测「册与探针不一致」，
  刷新需重跑探针 + 生成器。
- **适用范围**：本册只服务 TensorDG（DG 域）。船舶域（阻力/推进/耐波等）与 LBM/SPH 域各有自己的册，不混用。

## ⑥ 误差判定（有基准就逐项列，无基准说明原因）

本任务**无物理基准量**（不是数值验证，而是生态普查），因此不存在「与解析解/文献的 % 误差」。
可判定的是**验收指标达成率**，逐项如下：

| 指标 | 判据 | 实测 | 结果 |
|---|---|---|---|
| 普查规模 | ≥20 仓 | 24 | ✅ 超出 20% |
| tier 三档齐全且 T1 ≥5 | 是 | T1=9 | ✅ |
| 锚点数量 | ≥4 | 6 | ✅ 超出 50% |
| 锚点量化判据 | 每条含数值阈值 | 6/6 | ✅ |
| 锚点证据强度 | 每条 ≥2 仓 | 见 3.4 图 3 | ✅ |
| 元数据与探针一致率 | 100% | 24×7 字段 + reserve 全等 | ✅ |
| 生态活跃度 | ≥60% 一年内推送 | 见测试 `test_pushed_at_parsable_and_recent` | ✅（断言强制） |
| 注册表再生一致性 | 100% | `builder.build()` 与在册逐字段相等 | ✅ |

**结论**：普查/分级/锚点三项验收指标全部达成；**「是否 <3%」在本题无定义**（无基准量），
故以「一致性=100%（元数据 vs 探针）+ 覆盖率=120%（24/20）与 150%（6/4）」替代误差陈述。
**wave-1 的诚实边界**：任何「某仓实现了 X 能力」的强断言都需 wave-2 代码级复核，本册不做此断言。
