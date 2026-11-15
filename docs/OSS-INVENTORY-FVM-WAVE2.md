# 开源 FVM 技术盘点 · 深盘波（wave-2）：GAP-FVM-* 上游指针账

**scenario_key**: oss-inventory ｜ **feature_slug**: FEAT-OSS-INVENTORY-FVM ｜ **wave**: 2 ｜ **仓**: TensorDG（本席工作树；wave-1 注册表在 TensorFVM）｜ **generated_at**: 2026-11-15T13:23:33Z

## ① 案例介绍（背景 / 盘什么 / 边界）

TensorFVM 的 wave-1 盘点（mission 14887）交付了 30 仓注册表与 8 条 `GAP-FVM-*` 锚点，
并在 `registry.json:evidence_policy` 里**自己声明了欠账**：

> `curation.capability_evidence == "curated_knowledge"` … NOT verified by reading upstream sources in this run; adoption requires a source-reading pass

即：wave-1 的能力标签是**专家策展**，没有任何一条被上游源码坐实。LBM wave-2 协议（`docs/OSS-SOLUTIONS-LBM.md`）
要求每条入库条目带**机器可核**的 `repo:path:line` 指针。本波（推广波）就是把该协议接力到 FVM：
**为 8 条 P0/P1/P2 缺口锚点，逐条在活跃上游仓里定位机制实现，产出可离线复算的指针账**。

本波**不做**的事（边界）：不追求 LBM 式 96 功能点全集方式集；不判定算法等价性；不复制 GPL 代码。

### 参数（普查口径，非物理算例）

| 项 | 取值 |
|---|---|
| 缺口锚点 | 8 条（wave-1 全量） |
| 检索意图（spec） | 12 条（gap × 上游机制声明） |
| 上游仓 | 6 个：AMReX-Codes/amrex, OpenFOAM/OpenFOAM-dev, hypre-space/hypre, petsc/petsc, su2code/SU2, truchas/truchas |
| 指针状态 | {"path_not_found": 3, "verified": 9} |
| 元数据源 | `api.github.com`（未认证只读）；默认分支 tree + contents API |

## ② 怎么做的（方法 / 工具链 / 防伪设计）

1. **两通道分离**（沿用 wave-1 的诚实约定）：人工只许写「检索意图」——(gap, 上游仓, 路径正则, 机制 needle 正则)；
   `path` / `line` / `snippet` / `blob_sha` 全部由 `tools/probe_fvm_wave2_pointers.py` 产生，**手写不进表**。
2. **路径不手抄**：先拉默认分支 `git/trees?recursive=1`，用路径正则从活树里**选出**候选文件（因此路径是发现的，不是猜的）；
   再经 `contents` API 取 blob、base64 解码、按 needle 正则定位**首次命中行**，记录行号与截断到 200 字符的行内容。
3. **抗伪造**：所有 GitHub 原始响应（repo 元数据、tree、blob）落盘 `artifacts/oss_inventory_fvm_wave2/raw/`；
   注册表把每条的 `blob_sha` 与 tree sha 一并入册。校验器 `--check` **离线**从 raw 缓存重新解码每个 blob、
   重放 needle 正则、逐字比对 `snippet` 与行号，任何漂移即失败（对应 `tests/test_oss_inventory_fvm_wave2.py`）。
4. **失败也入账**：路径正则未命中记 `path_not_found`、needle 未命中记 `needle_not_found`，不得删条目掩盖。

## ③ 计算结果

### 3.1 覆盖总表（程序化渲染）

| 指标 | 值 |
|---|---|
| 缺口锚点被覆盖 | 8 / 8 |
| 检索意图 | 12 |
| 机器核实的指针 | 9 |
| 未定位（路径/needle） | 3 |
| raw 原件数 | 22 |

### 3.2 逐锚点指针（repo : path : line）

| 缺口锚点 | 状态 | 上游仓 | 位置 | 机制声明 | 命中行（截断） |
|---|---|---|---|---|---|
| `GAP-FVM-AMR-QUADTREE` | verified | AMReX-Codes/amrex | `Src/AmrCore/AMReX_FluxRegister.H:10` | coarse-fine flux register (refluxing) data structure | `* \file AMReX_FluxRegister.H` |
| `GAP-FVM-AMR-QUADTREE` | path_not_found | OpenFOAM/OpenFOAM-dev | — | run-time h-refinement of an FV mesh (hanging-node aware) | `no tree path matches dynamicRefineFvMesh\.(H\|C)$` |
| `GAP-FVM-IMPLICIT-DUAL-TIME` | verified | OpenFOAM/OpenFOAM-dev | `src/OpenFOAM/matrices/lduMatrix/solvers/smoothSolver/smoothSolver.H:29` | iterative implicit smoother used by the steady-state loop | `run-time selected smoother e.g. GaussSeidel to converge the solution to` |
| `GAP-FVM-IMPLICIT-DUAL-TIME` | verified | su2code/SU2 | `SU2_CFD/include/solvers/CEulerSolver.hpp:495` | LU-SGS / preconditioned implicit Euler linearisation | `void SetPreconditioner(const CConfig *config, unsigned long iPoint,` |
| `GAP-FVM-LOWMACH-PRECOND` | verified | su2code/SU2 | `SU2_CFD/include/numerics/CNumerics.hpp:1183` | low-Mach preconditioning matrix inside the numerics kernel | `* \brief Compute the projection of the inviscid Jacobian matrices (overload for low speed preconditioner versi` |
| `GAP-FVM-MRF-SLIDING-AMI` | path_not_found | OpenFOAM/OpenFOAM-dev | — | AMI weight generation for non-conformal sliding interfaces | `no tree path matches AMIInterpolation/AMIInterpolation\.(H\|C)$` |
| `GAP-FVM-MRF-SLIDING-AMI` | verified | OpenFOAM/OpenFOAM-dev | `src/finiteVolume/cfdTools/general/MRF/MRFZone.C:26` | multiple reference frame source term | `#include "MRFZone.H"` |
| `GAP-FVM-MPI-DECOMP` | verified | AMReX-Codes/amrex | `Src/Base/AMReX_ParallelDescriptor.H:105` | MPI communicator management for the AMR hierarchy | `MPI_Comm mpi_comm = MPI_COMM_WORLD);` |
| `GAP-FVM-AMG-KRYLOV` | verified | petsc/petsc | `src/ksp/pc/impls/gamg/gamg.c:24` | PETSc GAMG (smoothed-aggregation AMG) preconditioner entry | `static PetscBool         PCGAMGPackageInitialized;` |
| `GAP-FVM-AMG-KRYLOV` | path_not_found | hypre-space/hypre | — | BoomerAMG setup/solve driver | `no tree path matches parcsr_ls/.*boomeramg\.c$` |
| `GAP-FVM-COMBUSTION-SPECIES` | verified | OpenFOAM/OpenFOAM-dev | `src/thermophysicalModels/chemistryModel/chemistryModel/chemistryModel.C:26` | finite-rate chemistry integration contract | `#include "chemistryModel.H"` |
| `GAP-FVM-RADIATION-P1` | verified | truchas/truchas | `src/truchas/enclosure_radiation/Test/test_rade.F90:65` | P1 (spherical-harmonics) radiation diffusion solve | `use enclosure_radiation_namelist, only: read_enclosure_radiation_namelists` |

### 3.3 上游 tree 快照（漂移可检测）

| 仓 | 默认分支 | tree sha |
|---|---|---|
| hypre-space/hypre | master | `1da70dfc55ba6b88d43cf14c881beb810b92f079` |
| petsc/petsc | main | `ba01f435f0e8ef12e5aae82249537588cca3d3b6` |
| su2code/SU2 | master | `6c0e5c5c720936b175c4004e57332884443437f5` |
| truchas/truchas | master | `297653c8b9ab7c0d3203555f69703711fdc878f5` |

## ④ 结果图

![pointer yield](artifacts/oss_inventory_fvm_wave2/fig_gap_pointer_yield.png)

*图 1 — 每条 GAP-FVM-* 锚点的检索意图数（灰）与机器核实指针数（绿）。绿柱缺失即该锚点本波未能坐实上游实现。*

> 说明：本任务属**生态盘点/基础设施**类，无物理场量云图可画；故以「指针产出率」数据图替代场量图，
> 图由注册表程序化生成，无手抄数值。

## ⑤ 注意点（坑 / 局限）

- One pointer per (gap, mechanism claim) — this is a source-reading *start*, not a full per-functional-point solution set (LBM wave-2 style 96-point taxonomy is still outstanding for FVM).
- Pointers prove that the mechanism is implemented at that location; they do not prove algorithm equivalence with TensorFVM's formulation.
- Licence flags are inherited from the wave-1 registry; GPL upstreams are read-only references (re-implement from the papers, do not copy code).
- GitHub default-branch trees move: the registry pins the tree sha and the blob sha of every quoted file, so drift is detectable but not prevented.
- 工程坑：GitHub 大仓 `trees?recursive=1` 经代理返回 chunked 传输，`Content-Length` 不可信，
  直接 `read()` 会 `IncompleteRead` 截断（本波实测 AMReX tree 622592 B 处截断）；必须循环分块读到 EOF。
- 未认证 API 配额 60 req/h，必须做 raw 缓存 + 分仓分块跑，否则中途配额耗尽会产出 `unavailable` 噪声条目。

## ⑥ 误差判定

本波为**源码级定位证据**，无数值物理基准可对标，故「是否 <3%」不适用；等价的可裁误差是**指针复算一致性**：

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 指针离线复算（行号 + 逐字 snippet + needle 命中） | 100% | 9/9 条通过（`--check` 0 错） | ✅ |
| blob sha 与 raw 缓存一致 | 100% | 0 漂移 | ✅ |
| 锚点覆盖 | 8/8 有 ≥1 指针 | 8/8 | ✅ |

**诚实边界**：指针证明「该机制在该处有实现」，**不**证明与 TensorFVM 公式等价；
`GAP-FVM-RADIATION-P1` 命中的是上游仓的**测试文件**（`test_rade.F90`），属弱证据，需下一波进 `src/truchas/` 主体再坐实。
