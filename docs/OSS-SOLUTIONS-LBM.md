# FEAT-OSS-INVENTORY-LBM · 开源实现方式集（波 2：按功能点归集）

> 本文件由 `tools/build_oss_solutions_lbm.py` 程序化生成，禁止手改；`--check` 防漂移。条目文本（method/files/transfer）逐字取自平台表 `oss_solutions`，指针与迁移级别由脚本提取，零手抄。

## 1. 案例介绍：盘什么、为什么

LBM 是本生态中唯一已有「24+ 开源项目代码级盘点」的求解器域（波 1 落 `oss_inventory`）；但条目按各任务当时的锚点散落，同一功能点（如插值 bounce-back）的实现分散在 pylbm / lettuce / LUMA 三个仓的不同 slug 下，开发单开工前无法一眼看到「这个功能点别人怎么做的、指针在哪、能不能搬」。本波把这批条目按 **LBM 功能点**重新归集，形成可直接检索的实现方式集，并经 PANORAMA 每单注入全员可用。

- 归集范围：`oss_solutions` 全量 **294** 条快照（sha256 `ab99ec48adf5ee87…`）→ 选出 **162** 条作为方式集条目（solution 155 + 反例 7）
- 覆盖仓库：inventory 33 仓中 **24** 仓被引用，方式集共涉及 **24** 个上游仓
- 功能点：**96** 个（验收门槛 ≥40）

## 2. 怎么做的（方法 / 流程 / 工具链）

1. **快照**：`--dump` 用 psql 把 `oss_solutions` / `oss_inventory` 全量导出为 `artifacts/oss_solutions_lbm/source_rows.json` / `artifacts/oss_solutions_lbm/source_inventory.json`，sha256 入册。
2. **归集判据（人工层，只此一处）**：`tools/oss_solutions_lbm_taxonomy.py` 定义 96 个规范功能点（10 类）+ 每条目对应的源行选择器 `(repo, tag 子串)`；选择器必须唯一命中 1 行，否则构建直接失败。
3. **零手抄提取**：`upstream_ref` 由正则从源行 `files` 字段提取 `path:line`；`migration_level` 由 `transfer` 字段前缀程序化分类（direct/adapt/reference/none）；`method`/`files`/`transfer` 原样保留。
4. **指针探测（可选）**：`--probe-list` 输出待验指针，`tools/probe_oss_pointers.py` 在计算节点按仓浅克隆并回读 `path:line` 的真实行内容，结果写入 `artifacts/oss_solutions_lbm/pointer_probe.json` 后被本脚本并入条目。
5. **入库注入**：`artifacts/oss_solutions_lbm/insert.sql` 以 feature_slug=FEAT-OSS-INVENTORY-LBM 写入 `oss_solutions`；PANORAMA 的「开源实现方式集」表按 feature_slug 聚合，因此本方式集自动对全员可见（开发单开工前可 `SELECT * FROM oss_solutions WHERE feature_slug='FEAT-OSS-INVENTORY-LBM'`）。
6. **校验**：`tests/test_oss_solutions_lbm_registry.py` 断言规模、指针格式、选区唯一性、迁移级别枚举、统计自洽、再生一致。

## 3. 计算结果（由注册表程序化渲染）

### 3.1 规模与来源

| 指标 | 值 |
|---|---|
| 功能点 | 96 |
| 方式集条目 | 162（solution 155 / 反例 7）|
| 带 `文件:行` 指针的条目 | 159 |
| 涉及上游仓 | 24 |
| 指针实测通过 | 0 / 0（探测覆盖率 0.0%）|

### 3.2 迁移评估分布（程序化分类）

| 迁移级别 | 条目数 |
|---|---|
| 可直接迁移 (`direct`) | 45 |
| 需适配后迁移 (`adapt`) | 67 |
| 仅作参照 (`reference`) | 47 |
| 不可迁移/不适用 (`none`) | 1 |
| 原条目未标注 (`unlabelled`) | 2 |

### 3.3 分类 × 功能点 × 条目

| 类别 | 功能点数 | 条目数 | 涉及仓 |
|---|---|---|---|
| 碰撞算子与平衡态 | 10 | 17 | 2b-t/LB-t, PorousMediaSimulation/openLBMPM, cfdemons/LUMA, espressomd/espresso, joconnor22/LIFE, lbm-principles-practice/code, lettucecfd/lettuce, loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow, lssfau/lbmpy, pylbm/pylbm, sorush-khajepor/listLBM |
| 边界条件 | 10 | 17 | 2b-t/LB-t, cfdemons/LUMA, espressomd/espresso, haimingz/vivsim, hietwll/LBM_Taichi, je-santos/MPLBM-UT, joconnor22/LIFE, lanl/MF-LBM, lbm-principles-practice/code, lettucecfd/lettuce, loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow, pylbm/pylbm, sailfish-team/sailfish, sorush-khajepor/listLBM |
| 多相 / 相场 / 润湿 | 12 | 22 | AminZar69/CudaPhaseField, CFD-GO/TCLB, PorousMediaSimulation/openLBMPM, ebinan92/Fingering_dynamics, je-santos/MPLBM-UT, lanl/MF-LBM, lbm-principles-practice/code, lssfau/lbmpy, lssfau/walberla, ludwig-cf/ludwig, omalaspinas/palabos, sailfish-team/sailfish |
| 自由面与多孔介质 | 11 | 20 | Maarten-vd-Sande/lbm, ProjectPhysX/FluidX3D, je-santos/MPLBM-UT, lanl/MF-LBM, lettucecfd/lettuce, lssfau/lbmpy, lssfau/walberla, omalaspinas/palabos, sailfish-team/sailfish, sorush-khajepor/listLBM |
| 数据布局与性能可移植 | 10 | 16 | 2b-t/LB-t, AminZar69/CudaPhaseField, PorousMediaSimulation/openLBMPM, ProjectPhysX/FluidX3D, haimingz/vivsim, hietwll/LBM_Taichi, lanl/MF-LBM, lettucecfd/lettuce, loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow, lssfau/lbmpy, lssfau/walberla, sailfish-team/sailfish |
| 并行与多卡 | 6 | 12 | CFD-GO/TCLB, ProjectPhysX/FluidX3D, cfdemons/LUMA, haimingz/vivsim, lanl/MF-LBM, lbm-principles-practice/code, lssfau/lbmpy, lssfau/walberla, ludwig-cf/ludwig, omalaspinas/palabos, pylbm/pylbm, sailfish-team/sailfish |
| 流固耦合与受力 | 6 | 9 | ProjectPhysX/FluidX3D, cfdemons/LUMA, haimingz/vivsim, joconnor22/LIFE, ludwig-cf/ludwig |
| 网格加密 | 1 | 3 | cfdemons/LUMA, haimingz/vivsim, lssfau/walberla |
| 案例组织 / CI / 验证面 | 23 | 39 | 2b-t/LB-t, CFD-GO/TCLB, ProjectPhysX/FluidX3D, cfdemons/LUMA, ebinan92/Fingering_dynamics, haimingz/vivsim, hietwll/LBM_Taichi, je-santos/MPLBM-UT, joconnor22/LIFE, lanl/MF-LBM, lbm-principles-practice/code, lettucecfd/lettuce, loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow, lssfau/lbmpy, lssfau/walberla, ludwig-cf/ludwig, omalaspinas/palabos, pylbm/pylbm, sailfish-team/sailfish |
| 缺口与反例（不可迁移证据） | 7 | 7 | 2b-t/LB-t, AminZar69/CudaPhaseField, ebinan92/Fingering_dynamics, lssfau/walberla, ludwig-cf/ludwig, sorush-khajepor/listLBM |

### 3.4 功能点清单（方式数 / 代表性指针）

| 功能点 | 名称 | 方式数 | 首个上游指针 | 迁移级别 |
|---|---|---|---|---|
| LBM-FP-01 | BGK 单松弛基线 + 单相 Poiseuille 驱动 | 2 | `lettucecfd/lettuce` lettuce/ext/_flows/poiseuille.py:30 | 仅作参照 |
| LBM-FP-02 | TRT 两松弛算子 + magic 参数 lambda | 3 | `2b-t/LB-t` src/population/collision/collision_trt.hpp:61 | 可直接迁移 |
| LBM-FP-03 | MRT 多松弛碰撞 + 热涨落 | 1 | `espressomd/espresso` src/walberla_bridge/src/lattice_boltzmann/LBCollisionSetup.impl.hpp:30 | 仅作参照 |
| LBM-FP-04 | KBC 熵稳定碰撞（KBC-N4 / KBC-D） | 1 | `cfdemons/LUMA` LUMA/src/GridObj_ops_lbm_optimised.cpp:1122 | 仅作参照 |
| LBM-FP-05 | 中心矩（central-moment）碰撞 | 2 | `joconnor22/LIFE` src/Grid.cpp:106 | 需适配后迁移 |
| LBM-FP-06 | Chapman-Enskog 自动等效方程推导 | 1 | `pylbm/pylbm` pylbm/analysis/equivalent_equation.py:1 | 可直接迁移 |
| LBM-FP-07 | von Neumann 线性稳定性分析 | 1 | `pylbm/pylbm` pylbm/analysis/stability.py:1 | 可直接迁移 |
| LBM-FP-08 | 符号化代码生成（sympy / 模板 DSL） | 2 | `pylbm/pylbm` pylbm/scheme.py:1 | 可直接迁移 |
| LBM-FP-09 | 外力项注入方案族（Guo / EDM / 速度平移 / 四阶） | 3 | `PorousMediaSimulation/openLBMPM` ShanChen2D/ExplicitD2Q9GPU.py:255 | 需适配后迁移 |
| LBM-FP-10 | 格子定义表 + 自检（D3Q19 stride 陷阱） | 1 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` LatFlow/D3Q19.py:44 | 需适配后迁移 |
| LBM-FP-11 | half-way bounce-back 静壁 + 奇偶写回 | 2 | `2b-t/LB-t` src/population/boundary/boundary_bounceback.hpp:35 | 可直接迁移 |
| LBM-FP-12 | 插值 bounce-back（Bouzidi / BFL / IBB1） | 3 | `pylbm/pylbm` pylbm/boundary.py:810 | 可直接迁移 |
| LBM-FP-13 | Zou-He 进出口 / 通道重建 | 2 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` car.py:59 | 需适配后迁移 |
| LBM-FP-14 | Guo 插值速度/压力边界（非平衡外插） | 1 | `2b-t/LB-t` src/population/boundary/boundary_guo.hpp:43 | 可直接迁移 |
| LBM-FP-15 | 正则化速度/压力边界（Latt & Chopard 2008） | 2 | `cfdemons/LUMA` LUMA/src/GridObj_ops_lbm_optimised.cpp:313 | 仅作参照 |
| LBM-FP-16 | 平衡态外插边界 | 1 | `hietwll/LBM_Taichi` lbm_solver.py:130 | 仅作参照 |
| LBM-FP-17 | Lees-Edwards 剪切边界 | 1 | `espressomd/espresso` src/walberla_bridge/src/lattice_boltzmann/LBCollisionSetup.impl.hpp:85 | 仅作参照 |
| LBM-FP-18 | 标量 Dirichlet 边界（反反弹 / Inamuro 湿节点） | 1 | `lbm-principles-practice/code` chapter8/film_antibb.cpp:152 | 可直接迁移 |
| LBM-FP-19 | 润湿/接触角边界（几何法 / 壁面序参量 / 反演标定） | 3 | `lanl/MF-LBM` Phase_gradient.F90:209 | 需适配后迁移 |
| LBM-FP-20 | 壁面法向预处理（iso8 平滑） | 1 | `lanl/MF-LBM` Geometry_preprocessing.F90:236 | 仅作参照 |
| LBM-FP-21 | Shan-Chen 伪势单组分 + Young-Laplace 标定 | 1 | `lbm-principles-practice/code` chapter9/shanchen.cpp:126 | 需适配后迁移 |
| LBM-FP-22 | 两组分 Shan-Chen（MCMP）分层 Poiseuille | 4 | `je-santos/MPLBM-UT` src/2-phase_LBM/ShanChen.cpp:598 | 需适配后迁移 |
| LBM-FP-23 | 高密度比两相（Lee 两相 / Maxwell / Peng-Robinson EOS） | 3 | `omalaspinas/palabos` src/multiPhysics/twoPhaseModel3D.h:1279 | 仅作参照 |
| LBM-FP-24 | 自由能（SOY）压力张量平衡态 | 2 | `sailfish-team/sailfish` sailfish/sym_equilibrium.py:15 | 需适配后迁移 |
| LBM-FP-25 | 相场 Allen-Cahn（守恒型 / 含 CSF 力链） | 3 | `CFD-GO/TCLB` models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:649 | 需适配后迁移 |
| LBM-FP-26 | Cahn-Hilliard 自由能核心 | 2 | `ebinan92/Fingering_dynamics` lattice_boltzmann/fingering.py:149 | 可直接迁移 |
| LBM-FP-27 | 三组分双阱化学势 / sigma→lambda 映射 | 1 | `AminZar69/CudaPhaseField` src/global_subroutines.f90:329 | 需适配后迁移 |
| LBM-FP-28 | 接触角双实现交叉验证（壁面通量 vs 几何 ghost） | 2 | `lssfau/walberla` src/lbm/free_surface/surface_geometry/ContactAngle.h:34 | 需适配后迁移 |
| LBM-FP-29 | 组分通量边界（化学势 Zou-He） | 1 | `ebinan92/Fingering_dynamics` lattice_boltzmann/fingering.py:303 | 需适配后迁移 |
| LBM-FP-30 | 涨落 LBM（ghost-mode 噪声源） | 1 | `ludwig-cf/ludwig` src/collision.c:487 | 仅作参照 |
| LBM-FP-31 | 双 PDF 部分密度 / 色梯度模型 | 1 | `lanl/MF-LBM` Module.F90:203 | 仅作参照 |
| LBM-FP-32 | 相场 EOS 驱动压力演化（两相耦合链） | 1 | `CFD-GO/TCLB` models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:580 | 需适配后迁移 |
| LBM-FP-33 | 自由面 fill-level 家族（重构 / refill / 可配置族） | 2 | `lssfau/walberla` src/lbm/free_surface/dynamics/PdfReconstructionModel.h:62 | 需适配后迁移 |
| LBM-FP-34 | 多余质量分配模型族（9 种可切换） | 1 | `lssfau/walberla` src/lbm/free_surface/dynamics/ExcessMassDistributionModel.h:85 | 需适配后迁移 |
| LBM-FP-35 | VOF 质量交换 / 表面法向 / 界面初始化（Thuerey 系） | 4 | `Maarten-vd-Sande/lbm` lbm/streamcollide.py:77 | 需适配后迁移 |
| LBM-FP-36 | 自由面 PLIC 曲率 + 单元状态机 | 1 | `ProjectPhysX/FluidX3D` src/kernel.cpp:1677 | 需适配后迁移 |
| LBM-FP-37 | 单元类型位标志 + 邻接缓存 | 1 | `Maarten-vd-Sande/lbm` lbm/util.py:31 | 仅作参照 |
| LBM-FP-38 | 稀疏/多孔 list-based 组织 | 3 | `lssfau/lbmpy` src/lbmpy/sparse/mapping.py:14 | 需适配后迁移 |
| LBM-FP-39 | 部分饱和多孔（PSB / 部分 bounce-back） | 2 | `lettucecfd/lettuce` lettuce/ext/_boundary/partially_saturated_boundary.py:6 | 仅作参照 |
| LBM-FP-40 | 渗透率 / Darcy 评价流程 | 1 | `lssfau/walberla` src/lbm/evaluations/Permeability.h:42 | 可直接迁移 |
| LBM-FP-41 | 体素图像几何摄入 + 孔隙率报表 | 2 | `sorush-khajepor/listLBM` imgproc/readsolid2d_mod.f90:1 | 需适配后迁移 |
| LBM-FP-42 | 间接寻址边界节点（墙表） | 2 | `lanl/MF-LBM` Module.F90:90 | 需适配后迁移 |
| LBM-FP-43 | 三角网格 GPU 体素化（STL 光线投射） | 1 | `ProjectPhysX/FluidX3D` src/kernel.cpp:2273 | 需适配后迁移 |
| LBM-FP-44 | A-A pattern（单数组奇偶步索引） | 3 | `2b-t/LB-t` src/population/indexing/aa_pattern.hpp:56 | 需适配后迁移 |
| LBM-FP-45 | 融合 stream+collide（in-place / 生成式 sweep） | 3 | `ProjectPhysX/FluidX3D` src/kernel.cpp:1491 | 需适配后迁移 |
| LBM-FP-46 | SoA per-direction / 寄存器内矩→平衡态流水 | 2 | `sailfish-team/sailfish` sailfish/subdomain_runner.py:564 | 仅作参照 |
| LBM-FP-47 | 半精度 DDF 存储压缩（FP16S/FP16C + DDF-shift） | 1 | `ProjectPhysX/FluidX3D` src/kernel.cpp:1036 | 需适配后迁移 |
| LBM-FP-48 | 共享内存 warp/tile 交换 | 2 | `PorousMediaSimulation/openLBMPM` ShanChen2D/AccelerateGPU2D.py:1261 | 需适配后迁移 |
| LBM-FP-49 | GPU 寄存器压力（maxregcount）调优 | 1 | `AminZar69/CudaPhaseField` src/precision_m.f90:5 | 仅作参照 |
| LBM-FP-50 | Taichi JIT 单相 LBM | 1 | `hietwll/LBM_Taichi` lbm_solver.py:12 | 仅作参照 |
| LBM-FP-51 | JAX 纯函数 + static slice A-A | 1 | `haimingz/vivsim` vivsim/lbm/basic.py:60 | 可直接迁移 |
| LBM-FP-52 | 卷积表达 streaming（固定核） | 1 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` utils.py:4 | 仅作参照 |
| LBM-FP-53 | 算子索引掩码 substep（uint8 mask） | 1 | `lettucecfd/lettuce` lettuce/_simulation.py:83 | 可直接迁移 |
| LBM-FP-54 | MPI halo 交换（envelope / slab / cart 分解） | 3 | `omalaspinas/palabos` CMakeLists.txt:79 | 需适配后迁移 |
| LBM-FP-55 | 通信/计算异步重叠（waitany / post-wait / OpenACC） | 3 | `CFD-GO/TCLB` src/Lattice.cu.Rt:327 | 仅作参照 |
| LBM-FP-56 | 多 GPU / 多设备（ppermute / ZMQ / 多流） | 3 | `haimingz/vivsim` vivsim/multidevice.py:13 | 仅作参照 |
| LBM-FP-57 | MPI 拓扑自动分解 | 1 | `pylbm/pylbm` pylbm/mpi_topology.py:1 | 可直接迁移 |
| LBM-FP-58 | Ghost 交换融合进生成核（方向选择性） | 1 | `lssfau/lbmpy` src/lbmpy/advanced_streaming/communication.py:20 | 仅作参照 |
| LBM-FP-59 | 单节点多 GPU host-staged halo | 1 | `ProjectPhysX/FluidX3D` src/lbm.cpp:96 | 仅作参照 |
| LBM-FP-60 | IBM 直接力 + MDF 模板核 | 3 | `cfdemons/LUMA` LUMA/src/ObjectManager_ops_ibm.cpp:255 | 需适配后迁移 |
| LBM-FP-61 | 强耦合 FSI（Aitken 松弛 / 不动点） | 2 | `joconnor22/LIFE` src/Objects.cpp:26 | 仅作参照 |
| LBM-FP-62 | 动量交换法求固体受力/扭矩 | 1 | `ProjectPhysX/FluidX3D` src/kernel.cpp:1911 | 可直接迁移 |
| LBM-FP-63 | 粒子近场润滑修正 | 1 | `ludwig-cf/ludwig` src/lubrication.h:25 | 需适配后迁移 |
| LBM-FP-64 | 结构侧共旋梁有限元（Newmark/Newton） | 1 | `joconnor22/LIFE` src/FEMElement.cpp:72 | 仅作参照 |
| LBM-FP-65 | 静态浸没障碍集合 | 1 | `joconnor22/LIFE` src/IBMBody.cpp:55 | 仅作参照 |
| LBM-FP-66 | 静态块加密 2:1（Schornbaum-Ruede 式等值面修正） | 3 | `lssfau/walberla` src/lbm/lattice_model/CollisionModel.h:47 | 仅作参照 |
| LBM-FP-67 | 编译期参数 + 运行期几何配置 | 2 | `joconnor22/LIFE` inc/params.h:43 | 需适配后迁移 |
| LBM-FP-68 | 单点配置对象 + 单位换算 schema | 2 | `lssfau/lbmpy` src/lbmpy/creationfunctions.py:105 | 可直接迁移 |
| LBM-FP-69 | prm/YAML 驱动配置 + 模板生成流水线 | 2 | `lssfau/walberla` apps/benchmarks/PhaseFieldAllenCahn/CMakeLists.txt:1 | 可直接迁移 |
| LBM-FP-70 | 算例目录分类法（每物理一例 + 回归钩子） | 3 | `omalaspinas/palabos` CMakeLists.txt:159 | 可直接迁移 |
| LBM-FP-71 | CI 分级门禁（staged gates / 单 workflow / 格式门） | 3 | `lssfau/lbmpy` gitlab-ci.yml:1 | 仅作参照 |
| LBM-FP-72 | 目录即构建/CI 单元（model matrix） | 2 | `CFD-GO/TCLB` models/flow/d2q9/Dynamics.R:1 | 需适配后迁移 |
| LBM-FP-73 | 逐点参考回归 / 黄金场比对 | 3 | `lssfau/lbmpy` tests/shan_chen/test_shan_chen_two_phase.py:85 | 仅作参照 |
| LBM-FP-74 | 解析解加权 Lp 误差门（1% 级） | 3 | `lssfau/walberla` src/field/AccuracyEvaluation.h:72 | 可直接迁移 |
| LBM-FP-75 | sweep 等价回归（配置族同结果） | 1 | `lssfau/walberla` tests/lbm/SweepEquivalenceTest.cpp:30 | 仅作参照 |
| LBM-FP-76 | 浮点域容差 / 跨编译差异回归 | 1 | `ludwig-cf/ludwig` tests/awk-fp-diff.sh:5 | 可直接迁移 |
| LBM-FP-77 | 文献基准数据（Ghia 1982 顶盖方腔） | 1 | `hietwll/LBM_Taichi` data/ghia1982.dat | 可直接迁移 |
| LBM-FP-78 | Young-Laplace 解析 + 液滴标定验收链 | 1 | `je-santos/MPLBM-UT` examples/young_laplace_validation/2_phase_sim.py:121 | 需适配后迁移 |
| LBM-FP-79 | 长跑 checkpoint 状态机 / 压力阶梯续算 | 2 | `lanl/MF-LBM` IO_multiphase.F90:579 | 需适配后迁移 |
| LBM-FP-80 | 原地误差打印（in-situ 解析锚 L2） | 2 | `ProjectPhysX/FluidX3D` src/setup.cpp:84 | 可直接迁移 |
| LBM-FP-81 | 单元测试基架（trait 矩阵 / pytest 基线） | 2 | `2b-t/LB-t` test/run_tests.cpp:10 | 需适配后迁移 |
| LBM-FP-82 | 编译期回归 diff 基线工具 | 2 | `cfdemons/LUMA` LUMA/cases/testsuite/scripts/runTestSuite.sh:29 | 可直接迁移 |
| LBM-FP-83 | 多物理 app 分层 + scenario 类组织 | 1 | `lssfau/walberla` apps/showcases/PhaseFieldAllenCahn/CPU/multiphase_rising_bubble.py:34 | 可直接迁移 |
| LBM-FP-84 | 单文件配置切换 + SI 单位同时表达 | 1 | `ProjectPhysX/FluidX3D` src/setup.cpp:7 | 仅作参照 |
| LBM-FP-85 | 每步回调钩子 + 掩码合成边界 | 1 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` LatFlow/Domain.py:225 | 需适配后迁移 |
| LBM-FP-86 | 可微 LBM / BPTT 穿过时间步学参数 | 1 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` LatFlow/Domain.py:134 | 需适配后迁移 |
| LBM-FP-87 | LES 结构函数代理湍流闭合 | 1 | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` LatFlow/Domain.py:90 | 需适配后迁移 |
| LBM-FP-88 | 谱方法参考臂（Hele-Shaw 指进） | 1 | `ebinan92/Fingering_dynamics` spectrum_method/main.py:49 | 需适配后迁移 |
| LBM-FP-89 | 指令级 DSL 测试门 | 1 | `CFD-GO/TCLB` tools/tests.sh:154 | 仅作参照 |
| LBM-GAP-01 | 反例：仓内无两组分 MCMP（仅单相/两相） | 1 | `lssfau/walberla` apps/showcases/PhaseFieldAllenCahn/CPU/multiphase_RTI_3D.py:43 | 仅作参照 |
| LBM-GAP-02 | 反例：仓内无高密度比能力 | 1 | `ludwig-cf/ludwig` src/fe_ternary.c:12 | 需适配后迁移 |
| LBM-GAP-03 | 反例：多相只有布局模式、无物理实现 | 1 | `2b-t/LB-t` src/population/population.hpp:39 | 仅作参照 |
| LBM-GAP-04 | 反例：相场仓不支持伪势（不可混用） | 1 | `AminZar69/CudaPhaseField` src/device_var.f90:25 | 不可迁移/不适用 |
| LBM-GAP-05 | 反例：无测试无 CI（不可信实现） | 1 | `ebinan92/Fingering_dynamics` lattice_boltzmann/validation.py:22 | 原条目未标注 |
| LBM-GAP-06 | 反例：cupy drop-in 未融合 kernel | 1 | `ebinan92/Fingering_dynamics` lattice_boltzmann/fingering_periodic_gpu.py:2 | 原条目未标注 |
| LBM-GAP-07 | 反例：单文件算例 + 改 CMake 路径的组织方式 | 1 | `sorush-khajepor/listLBM` CMakeLists.txt:5 | 仅作参照 |

## 4. 结果图

图见平台报告（`scenario_key=oss-solutions`）：①分类×迁移级别堆叠柱；②指针探测通过率与覆盖仓数。仓内不重复存放二进制图，数据源为本注册表 `category_stats` / `stats.migration_level_counts`。

## 5. 注意点 / 局限

1. **文本是搬运而非重写**：条目 `method/files/transfer` 逐字取自源行，源行本身可能是某任务的一次性摘录；指针只保证「原文这么写」，语义正确性由指针探测 + 人工抽检兜底。
2. **指针为源行当时的行号**，探测在仓库当前默认分支上回读（`pointer_probe.head`），上游如有提交行号可能漂移——探测记录 `status` 与真实行内容，漂移会显式暴露为 `missing_file` / `line_oob`，不会被当作通过。
3. **inventory 中 9 仓 summary 为空**（JuliusWeinmiller/ThermalMultiphaseOpenLB, geoelements/mcmp-lbm, openLB/openLB, Autodesk/XLB, Zitzeronion/Swalbe.jl, UvaCsl/HemoCell, Xiangyu-Hu/SPHinXsys, apes-suite/musubi, hemelb-codes/hemelb），且 9 仓无任何 oss_solutions 条目（JuliusWeinmiller/ThermalMultiphaseOpenLB, geoelements/mcmp-lbm, openLB/openLB, Autodesk/XLB, Zitzeronion/Swalbe.jl, UvaCsl/HemoCell, Xiangyu-Hu/SPHinXsys, apes-suite/musubi, hemelb-codes/hemelb）——这些仓的代码级盘点仍欠账，不在本波覆盖内。
4. **反例行（LBM-GAP-*）不是实现**：它们记录「该仓没有这个东西」，用于防止下游误引，不计入新增能力。
5. **SPHinXsys 等跨域仓的 SPH 条目未纳入**（其锚点属 SPH 域），避免污染 LBM 方式集。

## 6. 误差判定

本任务为代码级盘点，没有物理量基准，因此按「机械可验指标」判定，逐项给出是否达标（不声称 <3% 之类无意义的物理误差）：

| 指标 | 目标 | 实测 | 判定 |
|---|---|---|---|
| 功能点数 | ≥40 | 96 | 达标 |
| 方式集条目 | ≥40 | 162 | 达标 |
| 每条目带文件指针 | 100% | 100.0% | 达标 |
| 其中带行号（`文件:行`） | 越高越好 | 159 条 (98.1%) | 另 3 条源行只给文件级指针（未伪造行号）|
| 每条目带迁移评估 | 100% | 100.0%（脚本强制：transfer 逐字必填）| 达标 |
| 行号指针实测命中 | 高覆盖 | **未运行（not_run）** | 本机无外网（`github.com` 连接被重置），探测工具与输入已就绪：`tools/probe_oss_pointers.py` + `artifacts/oss_solutions_lbm/pointer_probe_input.json`，须在有网计算节点执行；**未运行不等于通过** |

## 7. 附录：条目全文（按功能点）

### LBM-FP-01 BGK 单松弛基线 + 单相 Poiseuille 驱动（碰撞算子与平衡态）

- 迁移判断：TensorLBM 单相主干的最小参考：常压梯度/周期驱动通道基线，迁移成本最低
- **lettucecfd/lettuce** `lettuce/ext/_flows/poiseuille.py:30` ｜ 仅作参照
  - 方法：Flow 对象自带 analytic_solution(t) 与壁面 bounce-back 掩码，配 ErrorReporter 按间隔记录相对范数误差；MCMP 两组分分层 Poiseuille 可复用该骨架（单场 f 扩两组分 + 界面力项）。
  - 迁移评估（原文）：reference xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN
- **joconnor22/LIFE** `examples/PELskin/params.h:47` ｜ 仅作参照
  - 方法：本仓无多相，但提供了该 slug 所需驱动子机制的单相实现：左右壁设 eFluid 得到 x 周期，压差以常体积力 dpdx 注入（force_xy=(ρ·Drho·gravityX+dpdx)·ΔxΔt²/Δm，每格一次、不随 ρ 更新），配合正则化壁面对上下边界闭合，形成压力驱动的通道层流；重启/输出正常。
  - 迁移评估（原文）：reference xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-02 TRT 两松弛算子 + magic 参数 lambda（碰撞算子与平衡态）

- 迁移判断：TRT 是壁面位置无关性的标准做法；三仓给出查表法与生成式两种写法
- **2b-t/LB-t** `src/population/collision/collision_trt.hpp:61` ｜ 可直接迁移
  - 方法：[综合推荐 F14 碰撞算子族] MIT：TRT 双松弛 + magic λ=0.25 的显式写法（壁面位置无关的经典取值）；路线：碰撞算子按『SRT→TRT(magic)→MRT/central-moment』三级推进，先固定 magic 取值（λ=0.25 与 3/16 各跑一遍，记录壁面滑移差），再上 central-moment（LIFE 公式）与 KBC。
  - 迁移评估（原文）：fits xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER
- **lbm-principles-practice/code** `chapter8/gaussian_2d_trt.cpp:47` ｜ 可直接迁移
  - 方法：Split populations into symmetric and antisymmetric parts f_pm=0.5*(f_i +- f_complement) and the corresponding equilibrium parts, relax each with its own rate omega_plus/omega_minus, and derive one rate from the other through the magic parameter Lambda: omega_plus=1/(0.5+Lambda*omega_minus/(1-0.5*omega_minus)). The discrete TRT equilibria are generated from two constant tables weights_trt[] and com
  - 迁移评估（原文）：fits xval:XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS
- **lettucecfd/lettuce** `lettuce/ext/_collision/trt_collision.py:16` ｜ 可直接迁移
  - 方法：TRT 用对称/反对称两支分布差分别以 tau_plus/tau_minus 松弛；魔法参数 Λ=(τ₊−½)(τ₋−½)=3/16 提高稳定性（probe 中 τ=0.5088 → τ₋=21.76）。
  - 迁移评估（原文）：fits xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN

### LBM-FP-03 MRT 多松弛碰撞 + 热涨落（碰撞算子与平衡态）

- 迁移判断：矩空间松弛矩阵的实现范式，TensorLBM 目前只有 BGK/TRT
- **espressomd/espresso** `src/walberla_bridge/src/lattice_boltzmann/LBCollisionSetup.impl.hpp:30` ｜ 仅作参照
  - 方法：MRT 松弛矩阵从粘度自动推导，支持热涨落 (kT 参数)，奇数模式松弛消除数值不稳定
  - 迁移评估（原文）：reference

### LBM-FP-04 KBC 熵稳定碰撞（KBC-N4 / KBC-D）（碰撞算子与平衡态）

- 迁移判断：高 Re/低黏参数下的稳定性替代碰撞，属 TensorLBM 碰撞族空白
- **cfdemons/LUMA** `LUMA/src/GridObj_ops_lbm_optimised.cpp:1122` ｜ 仅作参照
  - 方法：构造 13 个（3D）或 3 个（2D）矩及其平衡值，把 fneq 分解为熵部 ds 与镜像部 dh；用熵线闭合求 γ = 2/ω − (2−2/ω)(ds·dh)/(dh·dh)；碰撞写作 f* = f − (ω/2)(2ds + γdh) + F_i。D3Q27 需 27 全套速度表，2D 用 D2Q9 的 KBC-D。
  - 迁移评估（原文）：reference

### LBM-FP-05 中心矩（central-moment）碰撞（碰撞算子与平衡态）

- 迁移判断：伽利略不变性更好的碰撞族；两套独立实现（部分松弛 / 完整中心矩）
- **joconnor22/LIFE** `src/Grid.cpp:106` ｜ 需适配后迁移
  - 方法：以局部速度 u_n 为原点把 9 个分布变换到中心矩 k0..k8：k0=ρ、k1/k2=F/2、k3=2ρc_s²、k4=(1−ω)·Σf(cx²−cy²)、k5=(1−ω)·Σf cx cy、k6/k7=½F c_s²、k8=ρc_s⁴，再逆变换回 fStar（显式展开 9 个方向）；平衡态用四阶 Hermite 乘积形式 ¼ρw(9cx²ux²+6cx ux−3ux²+2)(…)。等价两松弛率（剪切矩松弛、其余瞬时）。
  - 迁移评估（原文）：adapt xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION
- **joconnor22/LIFE** `src/Grid.cpp:106` ｜ 仅作参照
  - 方法：D2Q9 central moments (De Rosis 2019): f→k in frame shifted by u, relax only k4/k5 by omega, inverse transform via 9 explicit polynomials to f*. Force injected as moments k1/k2=half-step F and k6/k7=F*cs^2.
  - 迁移评估（原文）：reference xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-06 Chapman-Enskog 自动等效方程推导（碰撞算子与平衡态）

- 迁移判断：把「离散模型→宏观方程」推导自动化，可作新格子/新模型的验收前置
- **pylbm/pylbm** `pylbm/analysis/equivalent_equation.py:1` ｜ 可直接迁移
  - 方法：自动推导 LBM 格式的等价方程（修正方程），计算一阶/二阶修正项，用于精度分析
  - 迁移评估（原文）：fits xval:XVAL-LBM-CARTESIAN-MPI

### LBM-FP-07 von Neumann 线性稳定性分析（碰撞算子与平衡态）

- 迁移判断：参数选取（tau、forcing、格子）的稳定性判据可在计算前算出
- **pylbm/pylbm** `pylbm/analysis/stability.py:1` ｜ 可直接迁移
  - 方法：自动计算 LBM 格式的放大矩阵，对给定波矢计算特征值，判断 L2 稳定性
  - 迁移评估（原文）：fits xval:XVAL-LBM-CARTESIAN-MPI

### LBM-FP-08 符号化代码生成（sympy / 模板 DSL）（碰撞算子与平衡态）

- 迁移判断：TensorLBM 内核仍手写；生成式路线把新格子/新边界的实现成本降到声明级
- **pylbm/pylbm** `pylbm/scheme.py:1` ｜ 可直接迁移
  - 方法：用 SymPy 定义 LBM 格式（速度/多项式/平衡态/松弛参数），自动推导矩矩阵 M/M⁻¹，生成 Cython/NumPy/OpenCL kernel
  - 迁移评估（原文）：fits xval:XVAL-LBM-CARTESIAN-MPI
- **lettucecfd/lettuce** `lettuce/base.py:30` ｜ 需适配后迁移
  - 方法：[综合推荐 F07 GPU 融合 kernel 与存储压缩] MIT：算子按组件模板生成、JIT 可选，Python 侧落地成本最低；路线：近期（可运行）：lettuce/vivsim 的框架内融合（MIT，fits）＋ A-A pattern 省一半内存；中期（性能）：照 walberla/lbmpy 的生成式 sweep 设计自研 codegen（禁止搬码）；存储压缩按 FluidX3D FP16 的公开算法自行实现。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN

### LBM-FP-09 外力项注入方案族（Guo / EDM / 速度平移 / 四阶）（碰撞算子与平衡态）

- 迁移判断：同一物理力的多种 forcing 离散；TensorLBM 已可插拔，可对标权重与四阶写法
- **PorousMediaSimulation/openLBMPM** `ShanChen2D/ExplicitD2Q9GPU.py:255` ｜ 需适配后迁移
  - 方法：Two-component layered/anti-parallel-wall Poiseuille: force split as f<-f-0.5F before collision and +F after, u=[sum(f c)+0.5(F0+F1)]/rho_tot, viscosity-weighted u_eq=sum[(sum(f c)+0.5F)/tau]/sum(rho/tau); isotropy 4/8/10 stencils (Sigma w c_x^2 = 1) suppress spurious currents and allow high viscosity ratio.
  - 迁移评估（原文）：adapt xval:XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE
- **sorush-khajepor/listLBM** `collision/pseudopotential/potentialfunc_mod.f90:26` ｜ 需适配后迁移
  - 方法：把 psi(rho) 作为过程指针策略对象注入碰撞；SC 相互作用力只覆写 getforce，统一由父类 Guo 力项（速度平移 u+F/2rho + 源项 (1-0.5omega)w[...]）注入，换 EOS/换多相模型不动碰撞主体。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN
- **lssfau/lbmpy** `tests/shan_chen/test_shan_chen_two_component.py:55` ｜ 可直接迁移
  - 方法：每组分的力 F_sigma = -psi_sigma(x) * sum_{sigma'} g_{sigma sigma'} * sum_d w_d psi_sigma'(x+c_d) c_d，psi 用指数势 rho0*(1-exp(-rho/rho0))；两套 LBMConfig 各自 density_input 但共享 velocity_input = u_bary = sum(rho_sigma u_sigma)/sum(rho_sigma)，各配 Guo 力项、collide_only 内核 + 独立 pull streaming，每步前同步 rho 的 halo。
  - 迁移评估（原文）：fits xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN

### LBM-FP-10 格子定义表 + 自检（D3Q19 stride 陷阱）（碰撞算子与平衡态）

- 迁移判断：表格驱动格子定义缺自检导致步长错误，是可复用的负面教训
- **loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow** `LatFlow/D3Q19.py:44` ｜ 需适配后迁移
  - 方法：每套格子由 WEIGHTS/LVELOC/BOUNCE/STREAM 四张常量表定义，Domain 用 reshape 统一装配；但上游未对 STREAM 与 LVELOC 做一致性自检，导致 D3Q19 的 q=3 槽位抄成 q=15 的槽位、+y 分布沿 (0,1,1) 迁移。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

### LBM-FP-11 half-way bounce-back 静壁 + 奇偶写回（边界条件）

- 迁移判断：最基础壁面实现；两仓给出「同位写回」与「掩码索引」两种组织
- **2b-t/LB-t** `src/population/boundary/boundary_bounceback.hpp:35` ｜ 可直接迁移
  - 方法：壁面格点在碰撞后把 n 方向的 incoming 分布写到对偶方向 !n 与翻转时间步标签的槽位，等价标准 half-way BB；无插值/无曲率修正。
  - 迁移评估（原文）：fits xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER
- **haimingz/vivsim** `vivsim/lbm/boundary/bb.py:98` ｜ 可直接迁移
  - 方法：用布尔 mask 标记固体格点，一行 f.at[:,mask].set(f[:,mask][OPP_DIRS]) 完成局部 bounce-back，几何任意（其算例把文字光栅化成 mask）。JAX 下无动态索引开销，天然适配多孔介质/复杂几何。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

### LBM-FP-12 插值 bounce-back（Bouzidi / BFL / IBB1）（边界条件）

- 迁移判断：曲面与斜壁的二阶壁面距离修正；TensorLBM 已有 alpha 格式，可对照三仓的 q 法处理
- **pylbm/pylbm** `pylbm/boundary.py:810` ｜ 可直接迁移
  - 方法：统一框架 LinearBounceBack 实现 α 参数化插值 bounce-back，含 CLI(4/(1+2δ))/YLI(2/(1+δ))/BFL 变体，支持后碰撞修正
  - 迁移评估（原文）：fits xval:XVAL-LBM-CARTESIAN-MPI
- **lettucecfd/lettuce** `examples/advanced_projects/efficient_bounce_back_obstacle/boundary/linear_interpolated_bounce_back_boundary.py:9` ｜ 需适配后迁移
  - 方法：由墙体距离参数 d 预计算 (f_index_lt, d_lt)/(f_index_gt, d_gt)，按 d≤0.5 与 d>0.5 两分支做线性插值反弹；节点力矩交换可直接输出壁面力。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN
- **cfdemons/LUMA** `LUMA/src/GridObj_ops_lbm_optimised.cpp:1012` ｜ 需适配后迁移
  - 方法：对与壁相交的链接按墙距比 q∈[0,1] 做二阶插值反弹：墙更近当前点用 f*=(1−2q)(f̄₊−f̄)+f̄（f̄ 为预迁移反向值，f̄₊ 为再远一格的预迁移值）；墙更近源点用 f*=(1−2q)(f−f̄)/(2−2q)+f̄（迁移后值）。q 由点云几何逐标记点逐方向预计算并取最小（最近墙优先），同一链接两条 BFL 同时命中时取最近；壁面力用该链接的动量交换累加到标记点。
  - 迁移评估（原文）：adapt

### LBM-FP-13 Zou-He 进出口 / 通道重建（边界条件）

- 迁移判断：速度入口-压力出口的标准非平衡态重构，TensorLBM 通道算例的直接对照
- **loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow** `car.py:59` ｜ 需适配后迁移
  - 方法：2D D2Q9 通道进出口的分布重建：进口先用已知分量反算密度 rho=(f0+f2+f4+2(f3+f6+f7))/(1-u)，再解 f1,f5,f8；出口用 vx=-1+(...) 反算 f3,f6,f7；每步在 setup_step 里覆写第一/最后一行格点。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL
- **sorush-khajepor/listLBM** `stream/stream_mod.f90:45` ｜ 仅作参照
  - 方法：streaming 就地对 halfqs 一半方向与邻居交换 pdf（单趟无临时数组）；固体子链表跳过第二趟交换+空白碰撞=半步 bounce-back，且固体节点用固定 rho_wall/零速的「湿节点」参与伪势作用；进/出口用 Zou-He 型带切向与镜像对角修正的速度/密度边界，角点由自动生成的镜像对角表处理，出流用一阶外推，角上可复合。
  - 迁移评估（原文）：reference xval:XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN

### LBM-FP-14 Guo 插值速度/压力边界（非平衡外插）（边界条件）

- 迁移判断：非格点对齐进出口的二阶处理，与 Zou-He 互补
- **2b-t/LB-t** `src/population/boundary/boundary_guo.hpp:43` ｜ 可直接迁移
  - 方法：在边界邻点算局部矩与非平衡态 fneq=f-feq，按边界类型/方向用目标宏观量替换（速度边界强制 u，压力边界强制 rho 与切向速度），回写 feq+fneq。
  - 迁移评估（原文）：fits xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER

### LBM-FP-15 正则化速度/压力边界（Latt & Chopard 2008）（边界条件）

- 迁移判断：TensorLBM 已落地该族；两套独立 OSS 实现可作端口对账的第二参考臂
- **cfdemons/LUMA** `LUMA/src/GridObj_ops_lbm_optimised.cpp:313` ｜ 仅作参照
  - 方法：按法向把各速度方向分为流入侧 f⁺（含对角）与 f⁰，由已知 f 求非平衡应力张量分量 S_xx,S_yy,S_xy(,S_zz,S_xz,S_yz)，再用 f = feq + (w_i/2c_s⁴)[(c_x²−c_s²)S_xx+…+2c_xc_yS_xy] 重构未知方向；压力边界以 ρ=L_RHOIN+Δp 给定目标密度；角点/棱边方向数不足时沿法外推密度（一阶），压力边界在角点显式拒绝。
  - 迁移评估（原文）：reference
- **joconnor22/LIFE** `src/Grid.cpp:387` ｜ 仅作参照
  - 方法：Zou-He-type closure rho=(2*fplus+fzero)/(1-n.u) for velocity BCs, u_n=n*(1-(2*fplus+fzero)/rho) for pressure BCs; unknowns via anti-symmetric reconstruction then f=feq+(wi/2cs^4)[(cxi^2-cs^2)Sxx+(cyi^2-cs^2)Syy+2cx*cy*Sxy] from off-equilibrium stress. Corner: extrapolate rho + buried link=feq.
  - 迁移评估（原文）：reference xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-16 平衡态外插边界（边界条件）

- 迁移判断：由内点平衡态外推壁面分布的轻量替代，适合复杂几何入口
- **hietwll/LBM_Taichi** `lbm_solver.py:130` ｜ 仅作参照
  - 方法：边界 f = f_eq(boundary) - f_eq(inner) + f_old(inner)；一阶外推，非标准 Zou-He
  - 迁移评估（原文）：reference xval:XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION

### LBM-FP-17 Lees-Edwards 剪切边界（边界条件）

- 迁移判断：剪切/流变算例的周期剪切实现，TensorLBM 尚无
- **espressomd/espresso** `src/walberla_bridge/src/lattice_boltzmann/LBCollisionSetup.impl.hpp:85` ｜ 仅作参照
  - 方法：周期性剪切流边界，修改碰撞步施加剪切速度，PDF/速度/力在边界处插值
  - 迁移评估（原文）：reference

### LBM-FP-18 标量 Dirichlet 边界（反反弹 / Inamuro 湿节点）（边界条件）

- 迁移判断：温度/浓度/组分标量壁面条件的统一写法
- **lbm-principles-practice/code** `chapter8/film_antibb.cpp:152` ｜ 可直接迁移
  - 方法：[综合推荐 F15 边界算子族] MIT：反反弹 + Inamuro 湿节点，标量 Dirichlet（速度/压力）最干净的实现；路线：边界库按『壁面（半程 BB / 插值 BB / 正则化）→ 出入口（Zou-He 反算 / 反反弹 Dirichlet / EE）→ 特殊（Lees-Edwards 剪切、湿节点固定密度、平衡态外插 EE）』三组收编；每组附壁面偏移与角点处理约定。EE 的实现只见于无许可证仓（hietwll/LBM_Taichi，all-rights-reserved），只可按公式自写。
  - 迁移评估（原文）：fits xval:XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS

### LBM-FP-19 润湿/接触角边界（几何法 / 壁面序参量 / 反演标定）（边界条件）

- 迁移判断：接触角边界三套实现路线，可直接交叉验证
- **lanl/MF-LBM** `Phase_gradient.F90:209` ｜ 需适配后迁移
  - 方法：[综合推荐 F06 润湿 / 接触角] BSD-3 可搬：几何法把色梯度绕 n_w 旋转到 θ，无需 G_ads 吸附系数标定；路线：两条腿：几何法（MF-LBM，法向旋转，适配任意多相路线）作默认；吸附系数反演（MPLBM-UT 公式）作标定与交叉验证。壁面法向用 iso8 平滑预处理，否则阶梯壁面会把接触角误差放大。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE
- **sailfish-team/sailfish** `sailfish/lb_binary.py:161` ｜ 需适配后迁移
  - 方法：润湿边界通过壁面 φ 梯度参数 bc_wall_grad_phase 实现：半程反弹时用一阶 stencil、全程反弹时用一/二阶 stencil 把壁外虚拟 φ 设为 φ_wall−order*grad_phase，从而控制接触角；注释指 Kusumaatmaja 学位论文 p.18 option 2。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE
- **je-santos/MPLBM-UT** `src/2-phase_LBM/ShanChen.cpp:573` ｜ 需适配后迁移
  - 方法：由壁面黏附系数反算接触角 cosθ=4G_ads/(Gc(ρ_w-ρ_nw))，并同时给非湿相侧 -G_ads；用液滴三点标定（G_ads=-0.4/0/+0.4 ↔ 156°/90°/24°）校验。移植时须去掉原实现的 abs() 以保留湿相/非湿相方向。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION

### LBM-FP-20 壁面法向预处理（iso8 平滑）（边界条件）

- 迁移判断：体素阶梯壁法向噪声的处理链，直接影响润湿与受力精度
- **lanl/MF-LBM** `Geometry_preprocessing.F90:236` ｜ 仅作参照
  - 方法：预处理阶段先把 int1 壁面数组做多级平滑，再用 ISO8（8 层、权重 4/45…1/5040）各向同性差分求固体表面法向，写入界面格的 n_w 表供主循环润湿模型使用；同一张表还用于 φ 在壁面的加权外推（权重 w_equ 累加/归一）。
  - 迁移评估（原文）：reference xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE

### LBM-FP-21 Shan-Chen 伪势单组分 + Young-Laplace 标定（多相 / 相场 / 润湿）

- 迁移判断：伪势法最常用起点；G→sigma 标定流程可直接复用到 TensorLBM SCMP
- **lbm-principles-practice/code** `chapter9/shanchen.cpp:126` ｜ 需适配后迁移
  - 方法：[综合推荐 F02 伪势 SCMP/MCMP 力项与势函数] MIT 且自带『标定器 + 护栏』：把 ψ/EOS/G 与 Laplace 压差一起校准，正是 GAP-LBM-MCMP 验收要的工装形态；路线：伪势层拆成三个可插拔点：ψ(ρ) 策略、互作用力 stencil（含 EDM 展开）、力项注入（Guo/EDM）。标定护栏直接吃 lbm-principles 的 Young-Laplace 标定器（fits），先钉 σ 再谈密度比。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS

### LBM-FP-22 两组分 Shan-Chen（MCMP）分层 Poiseuille（多相 / 相场 / 润湿）

- 迁移判断：MCMP 的标准验收算例；多仓实现（psi 场 / 双格子 / 分层解析入口）
- **je-santos/MPLBM-UT** `src/2-phase_LBM/ShanChen.cpp:598` ｜ 需适配后迁移
  - 方法：两套 MultiBlockLattice3D 各持一组分密度/黏度，用 Palabos ShanChenMultiComponentProcessor3D 以单一 G 耦合产生相分离力；润湿性由壁面 bounce-back 黏附系数按几何 tag 分 5 类给出。分层 Poiseuille 只需把图像几何换成解析通道几何即可复用此骨架。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION
- **omalaspinas/palabos** `src/multiPhysics/shanChenProcessor3D.h:54` ｜ 需适配后迁移
  - 方法：多组分 Shan-Chen 显式力 F=-G*psi(x)*sum_i t_i*psi(x+c_i)*c_i；先把 psi 缓存成标量场再访问邻居，力写回格子 external force 字段，由 ShanChenExternalForceBGKdynamics/Regularized 在 collide 内并入；D3Q19 走完全展开特化。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
- **lbm-principles-practice/code** `chapter9/shanchen.cpp:29` ｜ 需适配后迁移
  - 方法：[综合推荐 F01 两组分分层 Poiseuille 对标] 唯一 MIT 的可搬两组分 SC 骨架：单格子多组分分布 + 平面界面初值，最小可验证（原表即 adapt：通道几何/边界需改写，故不升 fits）；路线：以 lbm-principles-practice/code（MIT）的双组分 SC + Zou-He 反算为母版起最小可证单元：单通道 D2Q9、两组分各自演化、界面位置与分相流量由积分量输出；解析锚用 TCLB 的分层剖面 U=6·Uavg·Y(H−Y)/H²（reference，仅取公式）。力项注入取 lbmpy 的重心 Guo 形式，速度混合取 listLBM 的策略化写法并显式记录「谁混合、按 ρσ 还是按 τ」这一自由度。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS
- **CFD-GO/TCLB** `models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:266` ｜ 仅作参照
  - 方法：分层 Poiseuille 解析剖面直接作为初始/入口条件（U=6*Uavg*Y*(H-Y)/H^2，圆管 2*Uavg*(1-(r/R)^2)），并由 updateMyGlobals 把两相流量、动能、界面位置积成 globals 做误差度量；可复用于两组分分层 Poiseuille 解析对标。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP

### LBM-FP-23 高密度比两相（Lee 两相 / Maxwell / Peng-Robinson EOS）（多相 / 相场 / 润湿）

- 迁移判断：伪势法密度比上限的核心瓶颈；三条不同 EOS/分割路线
- **omalaspinas/palabos** `src/multiPhysics/twoPhaseModel3D.h:1279` ｜ 仅作参照
  - 方法：Lee/He-Lee 两相扩散界面以显式 densityRatio 参数化，界面用相场 C 与化学势 mu 的三步分裂（C 与梯度 -> gradC/rho/mu -> gradMu/laplaceMu），表面张力与接触角由参数注入；SC 路线无任何高密度比机制。
  - 迁移评估（原文）：reference xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
- **lssfau/lbmpy** `src/lbmpy/phasefield/high_density_ratio_model.py:8` ｜ 需适配后迁移
  - 方法：高密度比不走 psi(rho) 势，而是由 EOS 积分得游离能 psi_eos=int(p/rho^2 drho)，再用 Maxwell 等压条件 psi_eos(rho_g)=psi_eos(rho_l) 解积分常数，得 f_bulk=lambda/2*(psi_eos-psi_0)^2 + kappa 界面项；力由压力张量 pi_ab=(f-sum phi mu)delta_ab - kappa d_a phi d_b phi 取散度施加。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN
- **PorousMediaSimulation/openLBMPM** `ShanChen2D/OptimizedD2Q9GPU.py:112` ｜ 仅作参照
  - 方法：Peng-Robinson EOS pseudo-potential psi=sqrt(2(p-rho/3)/(c0 G)) with p from R,T,a,b,alpha; written but never invoked (calEffectiveMassPR is an empty stub) - reference only for wiring an EOS-based high-density-ratio SC branch.
  - 迁移评估（原文）：reference xval:XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE

### LBM-FP-24 自由能（SOY）压力张量平衡态（多相 / 相场 / 润湿）

- 迁移判断：自由能法的解析表面张力载体；两仓给出平衡态压力张量写法
- **sailfish-team/sailfish** `sailfish/sym_equilibrium.py:15` ｜ 需适配后迁移
  - 方法：自由能二元模型的平衡态由符号推导：流体分布含体自由能压强 pb=ρ/3+A(−φ²/2+3φ⁴/4) 与 κ 各向异性修正（wxx/wyy/wzz/wxy/wyz/wxz），序参量分布含化学势 μ=A(−φ+φ³)−κ∇²φ；二阶矩闭包为 P_ab=[pb−κφ∇²φ]δ+ρu_a u_b+κ(∂_aφ∂_bφ−½|∇φ|²δ)，我方独立复核残差 1e-16。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE
- **ludwig-cf/ludwig** `src/symmetric.c:327` ｜ 仅作参照
  - 方法：以 F=1/2A phi^2+1/4B phi^4+1/2 kappa(grad phi)^2 的自由能路线替代 Shan-Chen：压强张量 P_ab 直接进碰撞平衡应力，界面力以 -phi grad mu 加到 hydro->force，再用 Guo 速度移位入碰撞；两组分分层 Poiseuille 可照此搭（非伪势路线）。
  - 迁移评估（原文）：reference xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES

### LBM-FP-25 相场 Allen-Cahn（守恒型 / 含 CSF 力链）（多相 / 相场 / 润湿）

- 迁移判断：相场路线三套工程实现，含保守 AC 与表面张力力链
- **CFD-GO/TCLB** `models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:649` ｜ 需适配后迁移
  - 方法：相场用第三套分布 h 承载：heq=C*feq(u)，界面力 Fphi=w_i*((1-4(C-0.5)^2)/W)*(e_i.n)，更新 h - omega_phi*(h-heq+0.5*Fphi) + Fphi；化学势 mu=4*(12*sigma/W)*(C-phi_l)*(C-phi_h)*(C-phi_avg)-1.5*sigma*W*laplace(C)，laplace 用各向同性 27 点模板。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP
- **lssfau/walberla** `apps/benchmarks/PhaseFieldAllenCahn/multiphase_codegen.py:29` ｜ 仅作参照
  - 方法：相场 LB(D3Q15,h) 推进 C，流体 LB(D3Q19,g) 推进 u；界面追踪力与流体力分别由 lbmpy 的 interface_tracking_force / hydrodynamic_force 注入；ρ 由 C 线性混合 ρ=ρ_L+C(ρ_H-ρ_L)。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **lssfau/lbmpy** `src/lbmpy/phasefield_allen_cahn/kernel_equations.py:19` ｜ 仅作参照
  - 方法：化学势 mu = d f/d phi - kappa*laplacian(phi) 符号生成，界面演化用 interface_tracking_force 并入碰撞规则；流体侧力分解为 pressure_force(由 rho(phi) 差) + viscous_force(按 tau 与密度比) + surface_tension_force(CSF) + 可选 thermocapillary_surface_tension_force(sigma(T) 线性)；相场方程用 RK 推进。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN

### LBM-FP-26 Cahn-Hilliard 自由能核心（多相 / 相场 / 润湿）

- 迁移判断：CH 方程扩散界面演化的独立实现，可作 AC 路线的对照
- **ebinan92/Fingering_dynamics** `lattice_boltzmann/fingering.py:149` ｜ 可直接迁移
  - 方法：守恒型 Cahn-Hilliard 相场内核：序参量 psi 由第二个分布 g 承载（psi=sum g），mu=a*psi(1-psi^2)-kappa*lap(psi)，kink 解 psi=tanh(x/xi) 与 sigma=4*kappa/(3*xi)（等价 kappa=0.75*xi*sigma）经本报告解析+数值复核一致，可作为内核验收门。
  - 迁移评估（原文）：fits xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION
- **ludwig-cf/ludwig** `src/free_energy.h:45` ｜ 需适配后迁移
  - 方法：[综合推荐 F05 相场 Allen-Cahn / Cahn-Hilliard] BSD-3 可搬：守恒 CH 的面通量形式 + 守恒漂移补偿 PHI_CONSERVE_* 是可直接落地的方程骨架；路线：方程骨架取 ludwig（BSD，可搬），力项分层取 walberla 的双力项分离（重写），三元/顺序参数取 CudaPhaseField 的保守 AC + 调和 λ 构造（公式）；解析锚用自由能 kink 解。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES

### LBM-FP-27 三组分双阱化学势 / sigma→lambda 映射（多相 / 相场 / 润湿）

- 迁移判断：三相（水-油-气）化学势参数化，TensorLBM 三元路线的方法来源
- **AminZar69/CudaPhaseField** `src/global_subroutines.f90:329` ｜ 需适配后迁移
  - 方法：Ternary chemical potential mu_k = (12/w)[lambda_k phi_k(1-phi_k)(1-2phi_k) - 2 lambda_t phi1 phi2 (1-phi1-phi2)] - (3/4) w lambda_k lap(phi_k), with lambda_k = sigma_jk + sigma_ki - sigma_ij recovered from the 3x3 surface-tension matrix and lambda_t = 3/(1/lambda_1+1/lambda_2+1/lambda_3) (harmonic mean). Verified: bulk term vanishes at all three pure phases; shipped sigma is degenerate (all 0.1) s
  - 迁移评估（原文）：adapt xval:XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING

### LBM-FP-28 接触角双实现交叉验证（壁面通量 vs 几何 ghost）（多相 / 相场 / 润湿）

- 迁移判断：本生态已用「两套互不共享代码的润湿实现」做 XVAL；OSS 侧同构流程可直接抄
- **lssfau/walberla** `src/lbm/free_surface/surface_geometry/ContactAngle.h:34` ｜ 需适配后迁移
  - 方法：自由面侧只缓存 sin/cos 的 ContactAngle + DetectWettingSweep/ObstacleFillLevelSweep 做几何修正；相场侧由 codegen 生成 ContactAngle boundary kernel（公式在 lbmpy，按界面厚度与角度反解壁面 C 的 Neumann 条件）。两套不共享代码。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **ebinan92/Fingering_dynamics** `lattice_boltzmann/fingering.py:102` ｜ 可直接迁移
  - 方法：润湿不用 Shan-Chen 吸附系数 G_ads, 而是把壁面格点 psi 固定为 psi_wall 并作为差分虚拟节点参与 nabla/lap。优点：无 G_ads 标定；缺点：接触角与 psi_wall 的映射需自建标定表（本仓没做）。
  - 迁移评估（原文）：fits xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION

### LBM-FP-29 组分通量边界（化学势 Zou-He）（多相 / 相场 / 润湿）

- 迁移判断：两组分守恒 + 化学势通量入口，指进/排驱算例的入口条件
- **ebinan92/Fingering_dynamics** `lattice_boltzmann/fingering.py:303` ｜ 需适配后迁移
  - 方法：多相 Zou-He：入口同时补质量式 rho_in=1/(1-u_x)(sum f_k + ... - psi_x*mu/2) 与序参量式 psi_in=1-sum(g), 未知方向按 w1+w5+w8 归一分配；出口同式但 u_x=+u0。移植时必须补全出口角点并把出口改成压力/回压型。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION

### LBM-FP-30 涨落 LBM（ghost-mode 噪声源）（多相 / 相场 / 润湿）

- 迁移判断：热涨落/布朗运动需要离散涨落源，TensorLBM 无
- **ludwig-cf/ludwig** `src/collision.c:487` ｜ 仅作参照
  - 方法：水动力模态确定性演化、ghost 模态按 1/tau 与 kT 给方差受热噪声；可向 CH 注入随机通量。提供「热涨落驱动的界面毛细波/Brownian」验证面。
  - 迁移评估（原文）：reference xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES

### LBM-FP-31 双 PDF 部分密度 / 色梯度模型（多相 / 相场 / 润湿）

- 迁移判断：两组分的另一种写法（部分密度 + 色梯度力），可作 MCMP 对照
- **lanl/MF-LBM** `Module.F90:203` ｜ 仅作参照
  - 方法：用两套完整 PDF（f 与 g）分别承载两个"组元"，宏观量取和：den=ρ1+ρ2、φ=(ρ1-ρ2)/den、u 由总 PDF 的一阶矩给出；再着色把总 PDF 拆回两套（主方向权重 w_equ_1、对角 1/(36√2)），β 控界面锐度；粘度按 φ 线性插值倒数后转松弛率。
  - 迁移评估（原文）：reference xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE

### LBM-FP-32 相场 EOS 驱动压力演化（两相耦合链）（多相 / 相场 / 润湿）

- 迁移判断：AC 与 EOS 压力演化的完整耦合链，缺失时相场只剩界面输运
- **CFD-GO/TCLB** `models/multiphase/d3q27_pf_velocity/Dynamics.c.Rt:580` ｜ 需适配后迁移
  - 方法：水动力分布零阶矩存压力、密度由相场 phi 代数映射，界面用保守 Allen-Cahn（含反扩散系数 1/W 的 Fphi 项），表面力取 mu*grad(phi) 并加 (0.5-tau)(rho_h-rho_l)(tau_ij*grad(phi)) 黏性力，配合 force_fixed_iterator 定点迭代自洽；是绕开伪势高密度比瓶颈的正解配方。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP

### LBM-FP-33 自由面 fill-level 家族（重构 / refill / 可配置族）（自由面与多孔介质）

- 迁移判断：自由面法的核心状态机；walberla 把 fill-level 模型做成可切换族
- **lssfau/walberla** `src/lbm/free_surface/dynamics/PdfReconstructionModel.h:62` ｜ 需适配后迁移
  - 方法：fill-level(0..1) 标记 liquid/interface/gas；新液格按法向选 6 种 PDF 重建模型之一，新界面格按 5 种再填充模型之一给 f；多余/欠缺质量用 ExcessMassDistribution 在界面邻域再分配保守恒；曲率由法向场 Parker-Youngs 中心差分得出，界面密度偏差 Δρ=6σκ 限幅 0.1。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **lssfau/walberla** `src/lbm/free_surface/dynamics/SurfaceDynamicsHandler.h:188` ｜ 需适配后迁移
  - 方法：flag 三态(liquid/interface/gas)+连续 fill level；时间步为固定 sweep 序列：边界→力加权→StreamReconstructAdvect 融合→碰撞/流式→胞元转换→回填→多余质量分配→标志复位→块状态更新；每个环节是可配置的模型族（重建 5 × 回填 5 × 多余质量 9 × 曲率 3）。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-FP-34 多余质量分配模型族（9 种可切换）（自由面与多孔介质）

- 迁移判断：自由面质量守恒的关键补丁；9 种可切换实现是难得的方法学清单
- **lssfau/walberla** `src/lbm/free_surface/dynamics/ExcessMassDistributionModel.h:85` ｜ 需适配后迁移
  - 方法：[综合推荐 F04 自由面（fill-level / VOF / mass tracking）] 把『多余质量怎么分配』显式做成 9 种可切换模型族——这正是自由面实现最容易悄悄错的地方；路线：自由面按三件套拆解实现：状态机（flag 转移）、质量账本（F→I→G 转移与超额质量分配）、界面分布重建（法向由 fill level 差分 + 非平衡反弹）。质量分配必须做成可选模型族并带守恒断言（Σmass 漂移）。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-FP-35 VOF 质量交换 / 表面法向 / 界面初始化（Thuerey 系）（自由面与多孔介质）

- 迁移判断：Thuerey 2006 VOF 自由面三件套，TensorLBM 自由面欠账的直接来源
- **Maarten-vd-Sande/lbm** `lbm/streamcollide.py:77` ｜ 需适配后迁移
  - 方法：每格独立 mass 场做体积质量账本，F→I 直接取上游分布并记账，I→I 交换按两侧填充率 ε 平均加权；转相时把超额质量按 max(0,e·n) 加权投给 interface 邻格，下一步收集回 mass。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AVERAGE-SURROUND-INIT
- **Maarten-vd-Sande/lbm** `lbm/streamcollide.py:181` ｜ 需适配后迁移
  - 方法：界面法向由填充率 ε 的相邻差分 n=0.5*(ε_{i-1}-ε_{i+1}, ε_{j-1}-ε_{j+1}) 给出，再用 n·e_opp>0 判据把"指向液体外"的方向用 feq+feq_opp-f_opp 重构覆盖。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AVERAGE-SURROUND-INIT
- **Maarten-vd-Sande/lbm** `lbm/update_types.py:32` ｜ 需适配后迁移
  - 方法：空腔格升级为界面格时，用周围 FLUID/INTERFACE 邻格的 rho 与 u 的算术平均初始化该格宏观量并重建 feq，避免新格从零注入伪压力。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AVERAGE-SURROUND-INIT
- **omalaspinas/palabos** `src/multiPhysics/freeSurfaceUtil3D.h:55` ｜ 可直接迁移
  - 方法：8 态 flag（empty/interface/fluid/wall/protect/temporarilyProtect/protectEmpty/slipWall）+ 界面格 mass 做体积记账，质量变化用入射分布累加（Thuerey eq.6），超额质量强制再分配、丢失质量单独统计成可观测量，表面张力由曲率法向注入。
  - 迁移评估（原文）：fits xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD

### LBM-FP-36 自由面 PLIC 曲率 + 单元状态机（自由面与多孔介质）

- 迁移判断：工程级自由面（溃坝/船舶）实现，含 PLIC 曲率与状态机
- **ProjectPhysX/FluidX3D** `src/kernel.cpp:1677` ｜ 需适配后迁移
  - 方法：用 TYPE_F/I/G 加转移标记 IF/IG/GI 的状态机逐格判定界面演化；界面格存质量 mass 与充填率 phi=clamp(mass/rho,0,1)；气体侧 DDF 由 feq(1−ρ_lap, u_shifted) 与质量通量重建；ρ_lap=6σK，K 由 D3Q27 邻域的 PLIC 立方体-平面反解 + 5 参数 Monge 面最小二乘求得；多余质量均分给 F/I 邻居，分不掉则留本地以保守恒。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR

### LBM-FP-37 单元类型位标志 + 邻接缓存（自由面与多孔介质）

- 迁移判断：掩码/固体/界面统一成 bitflag，减少分支与内存
- **Maarten-vd-Sande/lbm** `lbm/util.py:31` ｜ 仅作参照
  - 方法：用一个整数位域同时打包格型（FLUID/INTERFACE/GAS/OBSTACLE/INLET/OUTLET）、邻域存在性（NO_FLUID_NEIGH/NO_EMPTY_NEIGH/NO_IFACE_NEIGH）与本步转换意图（TO_FLUID/TO_GAS），内层循环用位与 O(1) 跳过空单元。
  - 迁移评估（原文）：reference xval:XVAL-LBM-AVERAGE-SURROUND-INIT

### LBM-FP-38 稀疏/多孔 list-based 组织（自由面与多孔介质）

- 迁移判断：只迭代流体节点的三种组织（list-based AoS / 子链表 / 邻接表）
- **lssfau/lbmpy** `src/lbmpy/sparse/mapping.py:14` ｜ 需适配后迁移
  - 方法：由 flag 场映射出「流体单元列表 + 每方向邻居索引」index array，pdf 存成 pdf_arr[n_fluid_cells, Q]（AoS），碰撞流式在 1D 索引空间一次完成（streaming 变成索引查表），边界另用 SparseLbBoundaryMapper 出 UBB 赋值 kernel。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN
- **lssfau/walberla** `python/lbmpy_walberla/sparse.py:11` ｜ 需适配后迁移
  - 方法：稀疏 LBM 用活跃 cell 列表 + src/dst/idx 索引数组 + SoA 布局，由 create_lb_update_rule_sparse 生成 gather 式 kernel；多孔介质用 Darcy 反算渗透率评估器（给粘性与流轴）；
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **sorush-khajepor/listLBM** `lists/parts/part_mod.f90:14` ｜ 需适配后迁移
  - 方法：[综合推荐 F09 稀疏 / 多孔介质几何] 子链表分类 + 显式邻接表 + 固体节点剔除，是唯一带『体素图像→孔隙率报表』全链路的实现；路线：几何摄入走『体素/图像 → EDT 分层 → tag 压缩』（MPLBM-UT fits）；存储走 list 型子链表 + 显式邻接表（listLBM 算法级重写）；亚网格层预留 PSB 钩子（lettuce 公式）。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN

### LBM-FP-39 部分饱和多孔（PSB / 部分 bounce-back）（自由面与多孔介质）

- 迁移判断：多孔介质灰区节点的阻力模型，渗透率算例的必要件
- **lettucecfd/lettuce** `lettuce/ext/_boundary/partially_saturated_boundary.py:6` ｜ 仅作参照
  - 方法：固体节点按饱和度 ε 在 BGK 碰撞与全反射 BB 之间加权混合，B=ε(τ−0.5)/((1−ε)+(τ−0.5))，可表示部分填充/多孔介质体，无需解析曲面。
  - 迁移评估（原文）：reference xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN
- **omalaspinas/palabos** `src/boundaryCondition/partialBBdynamics.h:12` ｜ 可直接迁移
  - 方法：部分 bounce-back 动力学对象以「部分反射」表示多孔骨架/半渗透壁，配合稀疏块结构与孔隙几何示例；动量交换 bounce-back 另用于 IBM 受力统计。
  - 迁移评估（原文）：fits xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD

### LBM-FP-40 渗透率 / Darcy 评价流程（自由面与多孔介质）

- 迁移判断：稳态流量反算渗透率的验收链，可作多孔算例判据
- **lssfau/walberla** `src/lbm/evaluations/Permeability.h:42` ｜ 可直接迁移
  - 方法：在指定 sampleVolume（局部 cell interval）内沿 flowAxis 统计平均流速与压力梯度，按 Darcy 律反算渗透率；可设 calcFrequency 周期性输出。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-FP-41 体素图像几何摄入 + 孔隙率报表（自由面与多孔介质）

- 迁移判断：CT/图像几何到格子掩码的工程链（含 EDT 压缩）
- **sorush-khajepor/listLBM** `imgproc/readsolid2d_mod.f90:1` ｜ 需适配后迁移
  - 方法：由体素图(.dat)或解析盒子标记 cell kind（wall/blackghost/blackwall），自动为周期方向预置 ghost 节点、自动识别角/边边界情形，最后输出 total/computational/black/Porosity 与各子链表单元数报表。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN
- **je-santos/MPLBM-UT** `src/python/mplbm_utils/pore_utils.py:19` ｜ 可直接迁移
  - 方法：对二值图像做欧氏距离变换，距孔隙 1 层的固体设为 bounce-back 壁（携带润湿 tag），更深的固体设为 NoDynamics（零计算），其余为流体；几何按 tag 值（2608-2611）压缩为 int16 落盘，模拟端按 tag 分派 dynamics。
  - 迁移评估（原文）：fits xval:XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION

### LBM-FP-42 间接寻址边界节点（墙表）（自由面与多孔介质）

- 迁移判断：稀疏边界节点用索引表替代全网格掩码，显存/带宽收益明确
- **lanl/MF-LBM** `Module.F90:90` ｜ 需适配后迁移
  - 方法：用 int1 三维壁面数组表示几何；预处理阶段把固体格与"流-固界面格"各自展开成**一维列表**：固体格存 18 邻居方向号 + 各方向的流体邻居数 + 局部权重（la_weight），界面格存壁面单位法向 (nwx,nwy,nwz) 与局部接触角 θ；主循环只遍历列表即可完成 φ 在壁面的外推、色梯度法向的壁面修正与润湿旋转，代价 O(边界节点数) 而非 O(N) 全场扫描。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE
- **sailfish-team/sailfish** `sailfish/templates/boundary.mako:34` ｜ 需适配后迁移
  - 方法：间接寻址：维护 nodes[] 稠密→稀疏映射，内核以 nodes[dense_gi+offset] 访问活跃流体节点，未激活节点不占存储也不参与计算；配套 subdomain 端只需给出 encoded_map 的稀疏节点表。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE

### LBM-FP-43 三角网格 GPU 体素化（STL 光线投射）（自由面与多孔介质）

- 迁移判断：船体/复杂几何入流的必备前置，TensorLBM 几何摄入欠账
- **ProjectPhysX/FluidX3D** `src/kernel.cpp:2273` ｜ 需适配后迁移
  - 方法：在 GPU 上对每个格点沿 3 个方向做三角网格射线求交（按 mesh bounding box 裁剪，无需 BVH），命中则置 TYPE_S 并按给定线速度/角速度写该格 u，实现 STL 一次性上格子且自动带移动壁速度；voxelize/unvoxelize 可在运行中反复调用做运动物体。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR

### LBM-FP-44 A-A pattern（单数组奇偶步索引）（数据布局与性能可移植）

- 迁移判断：省一半内存的经典布局；三仓给出索引与融合两种写法
- **2b-t/LB-t** `src/population/indexing/aa_pattern.hpp:56` ｜ 需适配后迁移
  - 方法：单一扁平数组存全部分布，偶时间步正常写/对偶槽读、奇时间步镜像；靠 PAD 使 ND 为偶数、正负速度半区对称，MASK 清零静止分量第二槽避免重复写。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER
- **lssfau/walberla** `python/lbmpy_walberla/alternating_sweeps.py:66` ｜ 可直接迁移
  - 方法：对同一碰撞规则生成 even/odd 两份 sweep；用 TimestepTracker 计时间步奇偶，运行时整数条件选路（EvenIntegerCondition）；in-place 布局下免去全场 swap。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **lanl/MF-LBM** `Kernel_multiphase.F90:45` ｜ 需适配后迁移
  - 方法：奇数步在**单个 kernel** 内完成「从 18 个邻居 pull 分布 → 求 ρ1/ρ2/φ 与宏观量 → CSF 力 → MRT 碰撞 → R-K 再着色 → push 到邻居的反方向槽位」；偶数步不做流动，只在原地把反方向槽位互换再碰撞。每步每方向只发生 1 读 1 写，两套 PDF 交错寻址，避免全场 copy 与两套完整分布数组并存。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE

### LBM-FP-45 融合 stream+collide（in-place / 生成式 sweep）（数据布局与性能可移植）

- 迁移判断：带宽极限优化的主路线，TensorLBM 融合核的直接参照
- **ProjectPhysX/FluidX3D** `src/kernel.cpp:1491` ｜ 需适配后迁移
  - 方法：单个 OpenCL kernel 内一次完成 拉取(stream)→求 ρ,u→Guo 力项→碰撞→写回邻居，用 A-A 双半格 + Esoteric-Pull（按时间步奇偶翻转方向索引导址）实现单 cell 原地更新、只维护一块 DDF 缓冲；固壁格直接 return，由邻居写入的镜像值形成隐式 bounce-back。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR
- **lssfau/walberla** `apps/showcases/PhaseFieldAllenCahn/GPU/multiphase_codegen.py:154` ｜ 仅作参照
  - 方法：kernel 由 Python 生成，目标 Target.GPU，靠 kernel_type 选 stream_pull_collide / collide_stream_push 融合方向；GPUField 同时支持 AoS(zyxf)/SoA(fzyx) 且可开 pitched memory；交替 pull/push 用编译期奇偶分支实现 in-place 双场流式；GPU 通信用就地打包(CombinedInPlaceGpuPackInfo)与 GPU 直通缓冲。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **lssfau/lbmpy** `src/lbmpy/creationfunctions.py:298` ｜ 需适配后迁移
  - 方法：用单一 kernel_type 选择器在「碰撞+宏观量输出+流式」之间融合或分离（default_stream_collide / collide_only / esotwist），pdf 数组按 fzyx（逐方向连续 SoA）布局，AA-pattern 单数组原地省一半内存免 swap，esotwist 只访问必要方向。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN

### LBM-FP-46 SoA per-direction / 寄存器内矩→平衡态流水（数据布局与性能可移植）

- 迁移判断：两套 CPU 侧融合写法，用来判定是否必须上 GPU
- **sailfish-team/sailfish** `sailfish/subdomain_runner.py:564` ｜ 仅作参照
  - 方法：每个 lattice 一个平坦 buffer 长 Q*N_active，按方向分片连续（idx=gi+q*N+offset），使传播成为整片平移；碰撞与传播融合在同一内核（*CollideAndPropagate*），跨方向部分用块内共享内存传播或 warp shuffle，另提供 AB 双缓冲与 AA 原地读时传播两种访问模式。
  - 迁移评估（原文）：reference xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE
- **2b-t/LB-t** `src/population/collision/collision_bgk_avx2.hpp:41` ｜ 仅作参照
  - 方法：无 GPU 实现；AVX2 路径演示了把矩求和、平衡态构造、碰撞回写全部保持在寄存器流水内、不落中间数组的单遍结构（可作为 GPU 融合 kernel 的算法骨架参考）。
  - 迁移评估（原文）：reference xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER

### LBM-FP-47 半精度 DDF 存储压缩（FP16S/FP16C + DDF-shift）（数据布局与性能可移植）

- 迁移判断：显存/带宽减半的具体做法与误差代价，TensorLBM 存储优化候选
- **ProjectPhysX/FluidX3D** `src/kernel.cpp:1036` ｜ 需适配后迁移
  - 方法：平衡态函数直接返回 f^eq−w（「扰动法/DDF-shifting」），宏观量用 Σf+1 复原，于是 DDF 数值集中在 ±1 附近、动态范围变小，可安全压到 IEEE half（硬件 vload_half*2^-15）或自定义 1-4-11 格式；算术全程仍 FP32。D3Q19 每格显存 93→55 B。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR

### LBM-FP-48 共享内存 warp/tile 交换（数据布局与性能可移植）

- 迁移判断：GPU 上不做全局流式的共享内存流式写法
- **PorousMediaSimulation/openLBMPM** `ShanChen2D/AccelerateGPU2D.py:1261` ｜ 需适配后迁移
  - 方法：Legacy branch performs streaming inside shared memory: 32-thread warp per lattice row, nine cuda.shared.array((32,)) buffers, boundary threads pull from adjacent rows => 8 global neighbour fetches removed; still one kernel per stage (not fused).
  - 迁移评估（原文）：adapt xval:XVAL-LBM-COMPRESSED-FLUID-NODE-NEIGHBOUR-TABLE
- **AminZar69/CudaPhaseField** `src/global_subroutines.f90:104` ｜ 需适配后迁移
  - 方法：(a) REAL pattern - gradient_cal and chemical_potential stage phi1/phi2/phi3 into shared tiles s_phi(0:TILE+1,0:TILE+1) with a one-cell halo ring loaded by the tx==1/TILE and ty==1/TILE threads plus four dedicated corner threads, then syncthreads and run the 8-neighbour stencil entirely from shared memory; TILE is a compile-time constant that must equal the launch blockDim. (b) REAL but partial fus
  - 迁移评估（原文）：adapt xval:XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING

### LBM-FP-49 GPU 寄存器压力（maxregcount）调优（数据布局与性能可移植）

- 迁移判断：寄存器溢出是融合核头号性能坑，属可复用的调优顺序经验
- **AminZar69/CudaPhaseField** `src/precision_m.f90:5` ｜ 仅作参照
  - 方法：Register pressure, not bandwidth, is what breaks the biggest kernel: collision_h needs 18 populations + 18 equilibria + 18 source arrays live at once and overflowed the register budget, silently failing the launch. The fix is a compile-time cap applied to every build mode (maxregcount:64) documented as a permanent constraint rather than a diagnostic. A second, toolchain-level lesson is recorded in
  - 迁移评估（原文）：reference xval:XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING

### LBM-FP-50 Taichi JIT 单相 LBM（数据布局与性能可移植）

- 迁移判断：免手写 CUDA 的可移植 GPU 路线，用来衡量另一条技术栈成本
- **hietwll/LBM_Taichi** `lbm_solver.py:12` ｜ 仅作参照
  - 方法：Taichi @ti.kernel 自动 JIT 编译 D2Q9-BGK collide-stream 到 GPU；双缓冲 f_old/f_new；AoS Vector.field(9) 布局
  - 迁移评估（原文）：reference xval:XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION

### LBM-FP-51 JAX 纯函数 + static slice A-A（数据布局与性能可移植）

- 迁移判断：函数式/可微路线下的布局与融合写法
- **haimingz/vivsim** `vivsim/lbm/basic.py:60` ｜ 可直接迁移
  - 方法：用「纯函数 + 静态 slice 拼接」代替 jnp.roll 实现 streaming（每方向一条 concatenate），配 jit(donate_argnums=0) 与 lax.scan 把时间循环整体融合；分布函数按 AA-pattern (q,NX,NY) 布局使空间维连续、便于 XLA 合并访存；benchmark 用 warmup+repeat+block_until_ready 逐算子计时。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

### LBM-FP-52 卷积表达 streaming（固定核）（数据布局与性能可移植）

- 迁移判断：把 stream/bounce 表达为卷积，可借深度学习框架做融合
- **loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow** `utils.py:4` ｜ 仅作参照
  - 方法：把离散分布按 [1,*Ndim,q] 存放（q 当通道），stream = 一次 3x3(2D)/3x3x3(3D) 常量卷积核卷积、bounce = 一次 1x1 卷积（核为 BOUNCE 反相置换矩阵），迁移与方向重排/壁面反射都交给框架的卷积实现融合。
  - 迁移评估（原文）：reference xval:XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

### LBM-FP-53 算子索引掩码 substep（uint8 mask）（数据布局与性能可移植）

- 迁移判断：用 mask 替代分支的算子组织，向量化友好
- **lettucecfd/lettuce** `lettuce/_simulation.py:83` ｜ 可直接迁移
  - 方法：no_collision_mask 存 uint8 算子索引（pre_boundary i / collision_index / post_boundary i）而非布尔；collide 阶段逐算子 torch.where(mask==i, op(f), f, out=f) 就地施加；no_streaming_mask 同法抑制指定方向对流。
  - 迁移评估（原文）：fits xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN

### LBM-FP-54 MPI halo 交换（envelope / slab / cart 分解）（并行与多卡）

- 迁移判断：三种分区与交换写法，TensorLBM MPI 路径的对账面
- **omalaspinas/palabos** `CMakeLists.txt:79` ｜ 需适配后迁移
  - 方法：MPI 规则块分解 + envelope ghost 层；通信器可切：ParallelBlockCommunicator3D 内部用 sendRecvPool 做非阻塞收发（通信与计算重叠），BlockingCommunicator3D 作阻塞对照；多块管理支持稀疏块结构。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
- **lbm-principles-practice/code** `chapter13/mpi/nonblocking/main.cpp:53` ｜ 可直接迁移
  - 方法：Split NY into contiguous rank slabs with one halo row per side; because the layout is direction-major with contiguous y-rows, each halo block is (ndir-1)*NX contiguous doubles so MPI_Isend/MPI_Irecv needs no packing; call MPI_Testall after every interior y-row so communication overlaps the remaining collision work; combine global norms in one MPI_Allreduce over seven scalars; a blocking Send/Recv 
  - 迁移评估（原文）：fits xval:XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS
- **cfdemons/LUMA** `LUMA/src/MpiManager.cpp:107` ｜ 需适配后迁移
  - 方法：[综合推荐 F08 分布式多卡 / MPI] Apache-2.0 可搬：MPI_Cart 拓扑 + 按活跃单元数的 smart decomposition 择优，正是体素多孔域要的负载均衡；路线：分区层用 LUMA 的 Cart + 活跃单元负载评估（Apache，可搬）；通信层先把 halo 做成显式 non-blocking post/wait 两段（ludwig 的 post-wait-split）以便插计算遮盖，再图 ghost 交换融合。
  - 迁移评估（原文）：adapt

### LBM-FP-55 通信/计算异步重叠（waitany / post-wait / OpenACC）（并行与多卡）

- 迁移判断：长跑算例的扩展效率关键；三种重叠范式
- **CFD-GO/TCLB** `src/Lattice.cu.Rt:327` ｜ 仅作参照
  - 方法：27 方向独立 MPI 缓冲；每步先用 outStream 异步 D2H，再 Irecv/Isend 全量发起并 MPI_Waitany 收一个拷一个（H2D 走 inStream），中间插入 interior kernel 覆盖通信；单 rank 单 GPU，多卡靠 MPI 域分解。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP
- **ludwig-cf/ludwig** `src/lb_data.c:1431` ｜ 需适配后迁移
  - 方法：MPI 三维域分解，halo 交换拆成 post(非阻塞 isend/irecv + 预建通信图) 与 wait 两段，中间可插本地计算隐藏延迟；胶体跨进程字节级 MPI_Irecv/Isend + 分段规约。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES
- **lanl/MF-LBM** `Main_multiphase.F90:354` ｜ 需适配后迁移
  - 方法：把每步 kernel 拆成「边界条带 kernel（z/y 各自一条 async 队列）」+「内部大块 kernel」，中间插入 halo 打包与 MPI 异步收发；GPU 版（#ifdef _openacc）把内部 kernel 排在 mpi_send_req **之前** 提交，CPU 版相反，以此在两种平台上都让通信藏在大块计算后面；最后统一 sync 把 halo 灌回。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE

### LBM-FP-56 多 GPU / 多设备（ppermute / ZMQ / 多流）（并行与多卡）

- 迁移判断：多卡三种通信栈；TensorLBM 多卡欠账的直接参照
- **haimingz/vivsim** `vivsim/multidevice.py:13` ｜ 仅作参照
  - 方法：域沿 x/y 均分到设备 mesh，每步用 jax.lax.ppermute 在设备轴上交换相邻子域边界分布函数（周期环形拓扑）实现跨设备迁移。最小可用但阻塞、无计算通信重叠。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT
- **sailfish-team/sailfish** `sailfish/connector.py:22` ｜ 仅作参照
  - 方法：子域切分 + 连接器抽象（multiprocessing 共享数组 / ZMQ PAIR / 跨主机 ZMQ / blosc 压缩版）交换边界分布层；主从进程用 socket 交换端口与统计；集群通过 execnet(SSH) 或 PBS 启动每节点每 GPU 一个 runner。
  - 迁移评估（原文）：reference xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE
- **lssfau/walberla** `src/blockforest/GlobalLoadBalancing.h:56` ｜ 需适配后迁移
  - 方法：块结构分区（Hilbert 曲线建块序）→ Metis/ParMetis 按块负载迁移；每层每块绑定 CUDA stream，块级 halo 通信与计算在不同流上重叠；pack/unpack 有 GPU 侧实现。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-FP-57 MPI 拓扑自动分解（并行与多卡）

- 迁移判断：按格点尺寸自动选分解形状的轻量工具
- **pylbm/pylbm** `pylbm/mpi_topology.py:1` ｜ 可直接迁移
  - 方法：基于 mpi4py 的笛卡尔拓扑管理，自动分区（Compute_dims）或用户指定，非阻塞 Irecv/Isend 通信
  - 迁移评估（原文）：fits xval:XVAL-LBM-CARTESIAN-MPI

### LBM-FP-58 Ghost 交换融合进生成核（方向选择性）（并行与多卡）

- 迁移判断：把 halo 打包融进生成核的选择性交换（5/19 面），通信量可量化省下
- **lssfau/lbmpy** `src/lbmpy/advanced_streaming/communication.py:20` ｜ 仅作参照
  - 方法：本仓不做域分解，只把周期边界 ghost 交换编译进生成 kernel（拿 src/dst slice 直接生成 CUDA copy kernel），注释称比 cupy 数组拷贝快近 2 倍；域分解交给 pystencils DataHandling/waLBerla。无异步通信隐藏。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN

### LBM-FP-59 单节点多 GPU host-staged halo（并行与多卡）

- 迁移判断：单机多卡 + 主机中转的带宽实现，与 TensorLBM 现役路径同构
- **ProjectPhysX/FluidX3D** `src/lbm.cpp:96` ｜ 仅作参照
  - 方法：按 (Dx,Dy,Dz,Ox,Oy,Oz) 把域切成每 GPU 一块 LBM_Domain（各自 cl::Context/queue 与独立编译单元）；halo 用 transfer_extract_*/transfer__insert_* kernel 做 VRAM 内选择性拷贝，再经 host staging buffer 以 CPU 指针 swap 交换 +/− 方向缓冲；逐方向 x→y→z 串行推进，每方向前后各一次 finish_queue 全局同步。
  - 迁移评估（原文）：reference xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR

### LBM-FP-60 IBM 直接力 + MDF 模板核（流固耦合与受力）

- 迁移判断：浸没边界三套写法（RKPM 校正 / Peskin 张量积模板 / 静态标记点）
- **cfdemons/LUMA** `LUMA/src/ObjectManager_ops_ibm.cpp:255` ｜ 需适配后迁移
  - 方法：[综合推荐 F13 IBM / 流固耦合与受力] Apache-2.0：Peskin 3 点核 + RKPM ε 守恒校正 + 强耦合子迭代，谱系最完整且可搬；路线：实现层先用 vivsim 的预计算 stencil（MIT 可直接搬），物理层照 LUMA 的三件套（力→ε 校正→子迭代）补上『力守恒』这一常被省略的环节；强耦合收敛判据取 LIFE 的 Aitken 形式。
  - 迁移评估（原文）：adapt
- **haimingz/vivsim** `vivsim/ib/stencil.py:7` ｜ 需适配后迁移
  - 方法：把 IBM stencil 一次性预计算成 (weights, flat_indices) 常数，运行期只做一次 einsum 插值与一次散播；插值/散布严格互为伴随（本次验证残差 2e-16），MDF 用 lax.fori_loop 迭代且可整体 jit。可直接搬到任何张量后端。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT
- **joconnor22/LIFE** `src/Objects.cpp:234` ｜ 需适配后迁移
  - 方法：IBM bodies (rigid circle / rigid+flexible filament) represented by Lagrange markers; force spreading via Dirac delta creates effective porous resistance. Epsilon smoothing from interpolation matrix.
  - 迁移评估（原文）：adapt xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-61 强耦合 FSI（Aitken 松弛 / 不动点）（流固耦合与受力）

- 迁移判断：弱-强耦合的分界与松弛策略；流固/拍动问题的必需件
- **joconnor22/LIFE** `src/Objects.cpp:26` ｜ 仅作参照
  - 方法：每时间步内 do{预测(2.5U_n−2U_nm1+0.5U_nm2)→插值→FEM 全量重解(Newton-Raphson, TOL=1e-10, MAXIT=20)→残差}while(res>1e-8 且 it<20)；子迭代 k≥2 时用 Aitken：relax=−relax·(r_{k−1}·(r_k−r_{k−1}))/|r_k−r_{k−1}|²，位移 U=U_km1+relax·(U−U_km1)，残差范数按 ref_L·√(DOF) 归一。收敛后一次性铺展 IBM 力。
  - 迁移评估（原文）：reference xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION
- **haimingz/vivsim** `examples/2d/vortex_induced_vibration.py:70` ｜ 需适配后迁移
  - 方法：全库纯函数 + 状态作 scan carry 传递，FSI 用固定点子迭代（FSI_ITER/IB_ITER）耦合，使整个时间循环结构上可被 jax.grad 穿透；但仓内无一处求导用例，属"结构就绪、能力未兑现"。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

### LBM-FP-62 动量交换法求固体受力/扭矩（流固耦合与受力）

- 迁移判断：阻力量化的格子侧算法（船舶阻力 LBM 路线的核心量）
- **ProjectPhysX/FluidX3D** `src/kernel.cpp:1911` ｜ 可直接迁移
  - 方法：由 DDF 直接算流体对固壁的力：对每个与固体相邻的流体格把 (f_i+f_ī) 的动量通量累加成 F（代码注释指出因 bounce-back 反射需 ×2），再用原子加按 TYPE_S 标记累出物体合力/力矩/形心，可直接对标船体阻力/扭矩验收。
  - 迁移评估（原文）：fits xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR

### LBM-FP-63 粒子近场润滑修正（流固耦合与受力）

- 迁移判断：颗粒-壁间隙小于格子时的解析补丁，精度门槛所在
- **ludwig-cf/ludwig** `src/lubrication.h:25` ｜ 需适配后迁移
  - 方法：BBL 之外叠解析近场润滑力（球-球径向/切向、粒子-壁面亏空），截断距离 rch 控制启用范围，力/力矩并入刚体总量。低分辨率下把阻力算对的关键。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES

### LBM-FP-64 结构侧共旋梁有限元（Newmark/Newton）（流固耦合与受力）

- 迁移判断：FSI 结构求解器的最小可靠臂
- **joconnor22/LIFE** `src/FEMElement.cpp:72` ｜ 仅作参照
  - 方法：2 节点 3 自由度（u,v,θ）平面梁：局部内力只用轴向伸长 u=(L²−L0²)/(L+L0) 与端弯矩 M1=2EI/L0(2θ1+θ2)、M2=2EI/L0(θ1+2θ2)，几何非线性由 K_NL 的 V0/L0、F0/L0 项承担；M 用一致质量（ρAL/420 的 140/156/22L0/54/13L0…），K_L 为标准梁刚度；Newmark-β(α=0.25,δ=0.5) 做时间积分，牛顿-拉夫逊线性化解 LAPACK。IBM 标记点按 ζ 映射到单元并用形函数插值位移/速度。
  - 迁移评估（原文）：reference xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-65 静态浸没障碍集合（流固耦合与受力）

- 迁移判断：基准算例里静止障碍物的集合写法
- **joconnor22/LIFE** `src/IBMBody.cpp:55` ｜ 仅作参照
  - 方法：「稀疏/多孔/复杂几何」在本仓用静态刚性 IBM 标记环表达：geometry.config 的 CIRCLE 行按周长生成等距标记点（num=floor(2πr/Δx)），可一行写多体、按间距阵列（space_x/space_y），标记点速度恒 0 即为无滑移障碍；配合 3 点 δ 核 + ε 校正把障碍力铺到格子，等价于任意曲线/栅板/植被阵列的多孔障碍，但**不是** mask bounce-back、也无稀疏索引压缩。
  - 迁移评估（原文）：reference xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-66 静态块加密 2:1（Schornbaum-Ruede 式等值面修正）（网格加密）

- 迁移判断：网格加密三套实现（静态块 / explode-coalesce / 缓冲层），TensorLBM 无
- **lssfau/walberla** `src/lbm/lattice_model/CollisionModel.h:47` ｜ 仅作参照
  - 方法：块森林 2:1 加密 + 等值面边界流式修正（EqualLevelBorderStreamCorrection）+ 按旋度/涡量选加密区 + 加密层 ω 按 levelDependentRelaxationParameter 重标定 + 时间步 PDF 打包同步。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **cfdemons/LUMA** `LUMA/src/GridObj_ops_lbm_optimised.cpp:36` ｜ 需适配后迁移
  - 方法：[综合推荐 F16 网格加密 / 可微与生成式框架] Apache-2.0：重叠网格 explode/coalesce + 2:1 子循环，块加密的工程化写法；路线：加密路线取 LUMA 的重叠网格（Apache 可搬）+ walberla 的等值面流式修正公式；可微/涨落路线只做设计预留（LOL 的 BPTT 与 ludwig 的 ghost-mode 涨落），两者许可证/框架均不可搬。
  - 迁移评估（原文）：adapt
- **haimingz/vivsim** `vivsim/multigrid.py:21` ｜ 仅作参照
  - 方法：块结构 2:1 局部加密——细/粗块之间加一层 buffer，迁移后按方向做 fine→coarse 四点平均、coarse→fine 复制，并按 level 重算松弛参数以保持黏度一致。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

### LBM-FP-67 编译期参数 + 运行期几何配置（案例组织 / CI / 验证面）

- 迁移判断：「参数编译期、几何运行期」的分层配置约定
- **joconnor22/LIFE** `inc/params.h:43` ｜ 需适配后迁移
  - 方法：一个算例=一个目录，两份输入：params.h（网格/物性/边界类型/ω/步数，编译期常量，改则须重编译）+ input/geometry.config（运行期文本：CIRCLE/FILAMENT 关键字行，含编号、起始坐标、阵列间距、半径或 长度/厚度/角度/FLEX|RIGID/单元数/BC/密度/E）；geometryReadIn 先扫一遍算 body/node 数预留内存，再逐行构造；重启时可就地增删体。
  - 迁移评估（原文）：adapt xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION
- **cfdemons/LUMA** `LUMA/input/geometry.config:11` ｜ 可直接迁移
  - 方法：每个算例一个定义头 definitions_NNN.h（宏给维度/分辨率/雷诺数/壁面 BC 类型/加密层数/输出频率/体列表），运行时配套 input/ 目录下的 geometry.config（体定义表：FROM_FILE/FILAMENT_ARRAY/CIRCLE_SPHERE/SQUARE_CUBE/PLATE，含 LEV/REG/FLEX_RIGID/BC/DENSITY/YOUNG_MOD 等后缀参数，无关项填 NULL）与 inlet_profile.in / point-cloud .in；runTestSuite.sh 按算例号自动分发对应输入、选 1 或 8 rank，并把定义头拷入结果目录留痕。
  - 迁移评估（原文）：fits

### LBM-FP-68 单点配置对象 + 单位换算 schema（案例组织 / CI / 验证面）

- 迁移判断：算例配置的单一真源（含 LBM↔SI 基准量），TensorLBM 算例注册表的对照
- **lssfau/lbmpy** `src/lbmpy/creationfunctions.py:105` ｜ 可直接迁移
  - 方法：用一个 dataclass LBMConfig 集中全部方法/数值/输出配置（stencil, method, relaxation_rates, compressible, zero_centered, equilibrium_order, c_s_sq, force_model, force, kernel_type, streaming_pattern, output, velocity_input, density_input ...），场景工厂只负责几何与初边值，step 对象负责时间推进组合。
  - 迁移评估（原文）：fits xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN
- **2b-t/LB-t** `include/lbt/simulation/simulation.hpp:682` ｜ 可直接迁移
  - 方法：[综合推荐 F10 案例组织与运行管理] MIT：七段式 JSON schema + LBM↔SI 三基准量换算，是唯一把单位换算内建进案例格式的；路线：案例格式取 LB-t 的 JSON schema（含单位换算与三基准量），配置对象取 lbmpy 的单一事实源写法，运行管理取 MF-LBM 的 checkpoint 状态机；把 listLBM 的单文件算例明确列为反例（不迁移）。
  - 迁移评估（原文）：fits xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER

### LBM-FP-69 prm/YAML 驱动配置 + 模板生成流水线（案例组织 / CI / 验证面）

- 迁移判断：参数文件→算例目录的生成链，可复现性做法
- **lssfau/walberla** `apps/benchmarks/PhaseFieldAllenCahn/CMakeLists.txt:1` ｜ 可直接迁移
  - 方法：算例=可执行文件+参数文件(.prm)+python 扫描驱动；CMake 把 *.prm/*.py 链进 build 目录，算例目录可在 build 里直接跑；benchmark.py 负责参数扫描与结果收集。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **je-santos/MPLBM-UT** `src/python/mplbm_utils/create_palabos_input_file.py:93` ｜ 可直接迁移
  - 方法：每个算例目录一个 input.yml + 一个 runner 脚本；create_palabos_input_file 用 f-string 模板把 yml 展开成求解器 XML，再写 mpirun bash 脚本并 subprocess 提交；后处理脚本从固定文本标记（'Pressure difference ='）解析输出。
  - 迁移评估（原文）：fits xval:XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION

### LBM-FP-70 算例目录分类法（每物理一例 + 回归钩子）（案例组织 / CI / 验证面）

- 迁移判断：算例组织三套范式（examples 分类 / 每例一回归 / 每例一文件）
- **omalaspinas/palabos** `CMakeLists.txt:159` ｜ 可直接迁移
  - 方法：examples 按用途分五条主线（showCases 演示 / codesByTopic 单主题最小复现 / benchmarks 性能 / tutorial 教学 / gpuExamples 加速版），每例独立 CMakeLists 并在顶层 CMake 全量 add_subdirectory 注册。
  - 迁移评估（原文）：fits xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD
- **sailfish-team/sailfish** `examples/binary_fluid/sc_poiseuille_2d.py:19` ｜ 需适配后迁移
  - 方法：每个物理案例一个可执行脚本（binary_fluid/ 下 SC 与 free-energy 各成系列），脚本只声明几何/初值/边界/参数，物理模型由 sim 类装配；对应的 regtest 用同名参数覆盖（minimal PBT），使"例—验证"一一对应。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE
- **haimingz/vivsim** `examples/2d/poiseuille_channel.py:33` ｜ 可直接迁移
  - 方法：案例组织范式——每个算例文件顶部集中参数块（几何/物理/LBM 三节），中部一个纯函数 step(f)（碰撞→力项→迁移→边界，顺序与所用 forcing 严格配套），底部 update_chunk+scan 跑；可视化/后处理只读 f。可作为我们案例目录模板。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

### LBM-FP-71 CI 分级门禁（staged gates / 单 workflow / 格式门）（案例组织 / CI / 验证面）

- 迁移判断：三仓 CI 组织：快速门→全量门的分级与格式/缓存门
- **lssfau/lbmpy** `gitlab-ci.yml:1` ｜ 仅作参照
  - 方法：GitLab CI 分 8 个 stage（Code Quality / Unit Tests / Reporting / Prerelease-Tests / integration / nightly / docs / deploy），lint 用 nox -s lint，单测带 coverage 产物，nightly 由变量开关控制，docs 单独构建。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN
- **lssfau/walberla** `github/workflows/basicTests.yml:1` ｜ 仅作参照
  - 方法：单一 GitHub Actions job：CMake 配置(MPI off, tests on, OpenMP on) → 并行构建 → ctest 全量跑。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **omalaspinas/palabos** `gitlab-ci.yml:57` ｜ 可直接迁移
  - 方法：GitLab CI 分阶段 prepare_docker -> check_format -> regression_tests -> parallel_regression_tests -> build_cmake，用 12 套 ccache key（gcc/clang × ser/par × rel/deb）矩阵缓存编译产物；AppVeyor 另覆盖 Windows/macOS。
  - 迁移评估（原文）：fits xval:XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD

### LBM-FP-72 目录即构建/CI 单元（model matrix）（案例组织 / CI / 验证面）

- 迁移判断：以算例目录为构建与 CI 单元的工程范式，TensorLBM 算例注册表同构
- **CFD-GO/TCLB** `models/flow/d2q9/Dynamics.R:1` ｜ 需适配后迁移
  - 方法：每个模型一个目录 models/<物理类>/<格子>_<方法>/，三件套 Dynamics.R（声明密度/设置/节点类型/输出量）+ Dynamics.c.Rt（模板内核）+ conf.mk（编译选项），算例独立放 example/<类>/<name>.xml；make <model> 即编该模型，模型名同时是 CI 矩阵键。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP
- **CFD-GO/TCLB** `github/workflows/cpu_test.yml:7` ｜ 仅作参照
  - 方法：CI 以模型为矩阵项：CPU 流水线对 16 个模型做 configure->compile->tests.sh->gcov 覆盖；GPU/HIP 流水线只做编译守卫；另一条流水线跑 tests/solid 独立单测。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP

### LBM-FP-73 逐点参考回归 / 黄金场比对（案例组织 / CI / 验证面）

- 迁移判断：不做全局容差而做逐点/黄金场比对，噪声下的判据更稳
- **lssfau/lbmpy** `tests/shan_chen/test_shan_chen_two_phase.py:85` ｜ 仅作参照
  - 方法：[综合推荐 F11 测试与验证面] 逐点参考回归（沿中心线 64 点 np.allclose 对比外部参考实现）——比单一积分量误差强得多；路线：验证面按三层建：符号层（等价方程/稳定性，pylbm 公式）、解析剖面层（lettuce 式 1% 门）、逐点参考场层（lbmpy 式中枢线/剖面 np.allclose）。基准数据集一律取论文原表（如 Ghia 1982），不取无许可证仓库的数据文件。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ALLEN-CAHN-CSF-FORCE-CHAIN
- **sailfish-team/sailfish** `regtest/poiseuille.py:64` ｜ 可直接迁移
  - 方法：验证面三件套：(a) Poiseuille 解析剖面对比，对每档黏度算 max(v_sim)/max(v_th)−1 并画随黏度曲线；(b) Ghia 等 Re=100/400/1000 的 LDC golden 场按分量存 npz，逐例比对；(c) 扫描 G∈[3,5.5] 用密度场极值/序参量定位 spinodal 分解阈值（G≈4.0）。
  - 迁移评估（原文）：fits xval:XVAL-LBM-FREE-ENERGY-BINARY-EQUILIBRIUM-PRESSURE
- **joconnor22/LIFE** `testing/store-ref-data.sh:27` ｜ 可直接迁移
  - 方法：基线生成脚本把 nSteps/tinfo/tVTK/tRestart 统一改写为 500 后跑出 Results/ 并连同 params.h+geometry.config 存为 RefData；回归脚本逐案例 make clean && make、跑 2 次（TurekHron 二次测 restart）、再用 diff -r Results RefData 做逐字节比对，全绿才算过；配合 #ifdef ORDERED 让 OpenMP 归约可复现，使位精确比对可行。物理验证面=7 个算例（含 Turek-Hron FSI2 公开基准）+ TotalForces.out 的 Cd/Cl（0.5ρU²L 归一）。
  - 迁移评估（原文）：fits xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-74 解析解加权 Lp 误差门（1% 级）（案例组织 / CI / 验证面）

- 迁移判断：解析解加权 Lp 误差 + 门限的验收面，可直接复用到本生态精度标准
- **lssfau/walberla** `src/field/AccuracyEvaluation.h:72` ｜ 可直接迁移
  - 方法：把解析解作为 std::function 传入，按网格体元加权求离散 Lp 误差；线图评价器沿指定线段比较仿真与解析；Poiseuille 算例现场构造解析剖面并用三种分母（参考/最大/平均速度）报相对误差。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION
- **lettucecfd/lettuce** `lettuce/ext/_reporter/error_reporter.py:11` ｜ 可直接迁移
  - 方法：算例对象自带 analytic_solution(t) 返回 (p,u)；ErrorReporter 每 interval 步算 ‖u−u_ref‖/N^(d/2) 与压力同量并追加列表；pytest 用 rel=1% 峰值 + abs=1%·u_max 双门断言。
  - 迁移评估（原文）：fits xval:XVAL-LBM-COMPONENT-PROTOCOL-TEMPLATE-CODEGEN
- **lbm-principles-practice/code** `chapter6/force_poiseuille_BB.m:95` ｜ 可直接迁移
  - 方法：Every benchmark compares the simulated field with a closed-form solution and reports a relative L2 norm computed only over fluid nodes (wall/virtual layers excluded); steady problems iterate until |mean(u)/mean(u_old)-1| < 1e-12; Taylor-Green runs report the sum-square error, the L2 norms of rho and u, and the analytic decay time td=1/(nu*(kx^2+ky^2)) at fixed sampling intervals.
  - 迁移评估（原文）：fits xval:XVAL-LBM-ANALYTIC-SOLUTION-RELATIVE-L2-HARNESS

### LBM-FP-75 sweep 等价回归（配置族同结果）（案例组织 / CI / 验证面）

- 迁移判断：同一算例不同实现路径（sweep 变体）必须给出相同结果
- **lssfau/walberla** `tests/lbm/SweepEquivalenceTest.cpp:30` ｜ 仅作参照
  - 方法：同一算例上对 CellwiseSweep / SplitSweep / SplitPureSweep / 生成 sweep 做逐 cell 数值比对，保证"换流式实现不改结果"；另有 codegen 测试目录与自由面子模块(动力学/表面几何/气泡)独立测试；多相唯一解析解对标是微通道热毛细 analytic 解（但只落 CSV，无误差统计脚本）。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-FP-76 浮点域容差 / 跨编译差异回归（案例组织 / CI / 验证面）

- 迁移判断：跨编译器/跨域的数值差异必须被容差化而非掩盖
- **ludwig-cf/ludwig** `tests/awk-fp-diff.sh:5` ｜ 可直接迁移
  - 方法：浮点按域容差比对（tests/awk-fp-diff.sh 由 test-diff.sh 驱动），单元测试按 feature 一个 test_*.c；容差为回归容差而非物理基准。
  - 迁移评估（原文）：fits xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES

### LBM-FP-77 文献基准数据（Ghia 1982 顶盖方腔）（案例组织 / CI / 验证面）

- 迁移判断：经典 CFD 基准数据表，可直接作为外锚
- **hietwll/LBM_Taichi** `data/ghia1982.dat` ｜ 可直接迁移
  - 方法：纯文本 18×12 矩阵：y/u(Re=[100..10000])/x/v(Re=[100..10000])；顶盖腔中心线速度基准
  - 迁移评估（原文）：fits xval:XVAL-LBM-EQUILIBRIUM-DIFFERENCE-EXTRAPOLATION

### LBM-FP-78 Young-Laplace 解析 + 液滴标定验收链（案例组织 / CI / 验证面）

- 迁移判断：多相验收的解析锚 + 标定流程，本生态同构判据的 OSS 对照
- **je-santos/MPLBM-UT** `examples/young_laplace_validation/2_phase_sim.py:121` ｜ 需适配后迁移
  - 方法：用 Pc=2σcosθ/r 造解析 Pc(Sw) 与 LBM 结果同图比较（Young-Laplace 验证算例），另以平面基底液滴算例扫 G_ads 三点标定接触角；但仅绘图、无断言，需我们补成参数化 pytest 与门禁。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION

### LBM-FP-79 长跑 checkpoint 状态机 / 压力阶梯续算（案例组织 / CI / 验证面）

- 迁移判断：长跑算例断点续算与状态机，可复用性强（HPC 必需）
- **lanl/MF-LBM** `IO_multiphase.F90:579` ｜ 需适配后迁移
  - 方法：用 job_status.txt 承载算例状态机（new/continue/done/failed/reached_max_step），checkpoint 按**墙钟小时**定时（两级备份：主 + 第二备份），保存完整 PDF+φ（必要时含对流出流历史场），重启时读状态文件走 continue 路径避免重算；同时用 d_vol_* 支持按"注入孔隙体积"触发输出/监控，使长算例可无人值守分段。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE
- **je-santos/MPLBM-UT** `src/2-phase_LBM/ShanChen.cpp:526` ｜ 可直接迁移
  - 方法：以 num_pc_steps 级递增出口密度实现准静态排驱，级差由 Δρ=6σcosθ/r_min 定（等价 Laplace 2σ|cosθ|/r）；每级跑到平均速度相对变化率<conv 判为定常；级间以 run_num.dat + saveBinaryBlock(两个 lattice) 断点续算。
  - 迁移评估（原文）：fits xval:XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION

### LBM-FP-80 原地误差打印（in-situ 解析锚 L2）（案例组织 / CI / 验证面）

- 迁移判断：求解器自带解析锚收敛打印，属低成本常驻验证面
- **ProjectPhysX/FluidX3D** `src/setup.cpp:84` ｜ 可直接迁移
  - 方法：每个算例自带「运行中算误差、误差不再下降即判定收敛」的 L2 判据：Poiseuille 圆柱管每 1000 步算 ‖u_num−u_max(1−r²/R²)‖₂/‖u_ref‖₂ 并要求下降；体积力由解析式反推 f=4u_maxρν/R² 使入口条件与解析解一致；另有 Stokes 阻力、Taylor-Green、lid-driven cavity、Karman、注水质量守恒等 40+ 算例。
  - 迁移评估（原文）：fits xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR
- **haimingz/vivsim** `examples/2d/flow_pass_cylinder.py:80` ｜ 需适配后迁移
  - 方法：每个算例末尾把误差算成数字打印：圆柱绕流与 3 篇 DOI 文献的 Cd=1.389/Cl=0.718 比百分比误差，Couette/Taylor-Green/ABC 打印 rel-L2，Poiseuille 叠解析剖面。优点是参考值连出处写进源码，缺点是全仓无 assert、无 pytest、无 CI 门。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

### LBM-FP-81 单元测试基架（trait 矩阵 / pytest 基线）（案例组织 / CI / 验证面）

- 迁移判断：格子×边界×碰撞的 trait 矩阵测试基架，覆盖组合爆炸的工程解法
- **2b-t/LB-t** `test/run_tests.cpp:10` ｜ 需适配后迁移
  - 方法：为 population/indexing/lattice 各写"参数化+模板"测试基类，同一套断言对多个格子与访问模式实例化，覆盖 >1000 用例。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER
- **pylbm/pylbm** `tests/test_*.py` ｜ 可直接迁移
  - 方法：tests/ 目录含 pytest 套件，baseline/ 和 reference/ 存放验证数据，支持参数化测试和收敛性测试
  - 迁移评估（原文）：fits xval:XVAL-LBM-CARTESIAN-MPI

### LBM-FP-82 编译期回归 diff 基线工具（案例组织 / CI / 验证面）

- 迁移判断：把旧实现输出当基线做 diff 的回归工具化做法
- **cfdemons/LUMA** `LUMA/cases/testsuite/scripts/runTestSuite.sh:29` ｜ 可直接迁移
  - 方法：壳脚本逐算例编译+运行，把输出与 base/caseNNN 快照逐文件 diff——但实际只对单个 ASCII 转储 io_lite.Lev0.Reg0.Rnk0.100.dat 做 diff -q 判通过，失败时用 diff -y 生成并排差异日志；同时用 hdf5/restart/LiftDrag CSV 作为人工核查工件；同一物理用例分别以 1 rank 与 8 rank 运行以覆盖 MPI 路径。
  - 迁移评估（原文）：fits
- **joconnor22/LIFE** `testing/run-tests.sh` ｜ 可直接迁移
  - 方法：Shell-script regression testing: run all example cases, diff Results/ against stored RefData/ (binary-exact, no tolerance). TurekHron case also exercises restart capability.
  - 迁移评估（原文）：fits xval:XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION

### LBM-FP-83 多物理 app 分层 + scenario 类组织（案例组织 / CI / 验证面）

- 迁移判断：4 层 app 结构（scenario/算例/驱动/参数），大型求解器组织范式
- **lssfau/walberla** `apps/showcases/PhaseFieldAllenCahn/CPU/multiphase_rising_bubble.py:34` ｜ 可直接迁移
  - 方法：算例按 examples/showcases/benchmarks/tutorials 四层组织，每算例一个 .prm 参数文件 + 一个 Python Scenario 类（网格、物理参数、输出频率、每步回调一次配齐），C++ 侧只读 prm 装配；多相场景把模型族（重建/回填/多余质量/曲率）全部暴露到 prm。
  - 迁移评估（原文）：fits xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-FP-84 单文件配置切换 + SI 单位同时表达（案例组织 / CI / 验证面）

- 迁移判断：单文件开关 + 物理单位直读的算例配置风格
- **ProjectPhysX/FluidX3D** `src/setup.cpp:7` ｜ 仅作参照
  - 方法：所有算例写在 setup.cpp 一个文件里、用注释开关切换；每个算例统一四段「①解析反推体积力/粘度 ②units.set_m_kg_s 建立 SI↔lattice 三标度 ③几何按坐标谓词写 flag ④run + 出图/写 VTK」，无量纲数（Re/We/Ca/Fr）由 units.si_* 直接打印。
  - 迁移评估（原文）：reference xval:XVAL-LBM-SINGLE-COMPONENT-POISEUILLE-FORCE-ANCHOR

### LBM-FP-85 每步回调钩子 + 掩码合成边界（案例组织 / CI / 验证面）

- 迁移判断：把边界/探针挂到步进钩子，免改主循环即可扩展
- **loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow** `LatFlow/Domain.py:225` ｜ 需适配后迁移
  - 方法：Solve(sess,Tf,initialize_step,setup_step,save_step,save_interval) 把每步的前处理、输出节奏以回调注入；边界条件用单个张量表达式 f=f*(1-boundary)+f_bc*boundary 合成任意几何。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

### LBM-FP-86 可微 LBM / BPTT 穿过时间步学参数（案例组织 / CI / 验证面）

- 迁移判断：可微求解器路线（LES 参数/反问题），TensorLBM 未触及
- **loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow** `LatFlow/Domain.py:134` ｜ 需适配后迁移
  - 方法：提供 graph_unroll 分支让 collide/stream 返回图内张量而非写回 Variable，Unroll() 在 Python 侧串起 N 步并返回状态链；tau 由场量与标量 Sc 在图内算出，故梯度可从 N 步后的场回传到模型常数。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

### LBM-FP-87 LES 结构函数代理湍流闭合（案例组织 / CI / 验证面）

- 迁移判断：低成本 LES 闭合（非平衡应力绝对值卷积），工程精度/成本折中
- **loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow** `LatFlow/Domain.py:90` ｜ 需适配后迁移
  - 方法：用 Q=sqrt(2*Σ_i (f_i-f_i^eq)^2 * EEk_i)（EEk=Σ_{n,m}|c_n c_m|）作为 Π_ab 模的标量代理，再 tau=0.5*(tau0+sqrt(tau0^2+6*Q*Sc/rho)) 做 Smagorinsky 型涡粘。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

### LBM-FP-88 谱方法参考臂（Hele-Shaw 指进）（案例组织 / CI / 验证面）

- 迁移判断：为界面失稳类算例提供独立谱方法参考臂（非 LBM 第二实现）
- **ebinan92/Fingering_dynamics** `spectrum_method/main.py:49` ｜ 需适配后迁移
  - 方法：谱方法指进：涡量-流函数 phi_hat=-i*R*k_y*c_hat/(k^2+i*R(k_x*c_x+k_y*c_y)), 扩散用积分因子 exp(-k^2*dt) 精确、非线性项用 AB2（启动 RK4）。可作指进线性稳定性/分辨率基准，LBM 只负责非线性阶段与润湿壁面。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION

### LBM-FP-89 指令级 DSL 测试门（案例组织 / CI / 验证面）

- 迁移判断：把模型 DSL 的指令流本身纳入测试，模型生成器的验收面
- **CFD-GO/TCLB** `tools/tests.sh:154` ｜ 仅作参照
  - 方法：测试用指令式 DSL 描述：run/fail/exists/sha1/csvconcatenate 等行内命令，测试脚本放 tests/external/<model>/*.test（git submodule），逐条打印 OK/FAIL 并计时，配合 tools/csvdiff 做阈值比较。
  - 迁移评估（原文）：reference xval:XVAL-LBM-ASYNC-COPY-WAITANY-OVERLAP

### LBM-GAP-01 反例：仓内无两组分 MCMP（仅单相/两相）（缺口与反例（不可迁移证据））

- 迁移判断：避免误判某仓可为 MCMP 提供实现；取用前必查
- **lssfau/walberla** `apps/showcases/PhaseFieldAllenCahn/CPU/multiphase_RTI_3D.py:43` ｜ 仅作参照
  - 方法：本仓无 MCMP（无组分概念、无组分输运方程）；两组分分层 Poiseuille 无法直接映射。可借用的只是相场的分层初始化几何写法与变密度平衡态装配，把"两组分"降级为"两相分层"来近似。
  - 迁移评估（原文）：reference xval:XVAL-LBM-BLOCK-AMR-EQUAL-LEVEL-STREAM-CORRECTION

### LBM-GAP-02 反例：仓内无高密度比能力（缺口与反例（不可迁移证据））

- 迁移判断：密度比上限不能靠该仓补齐
- **ludwig-cf/ludwig** `src/fe_ternary.c:12` ｜ 需适配后迁移
  - 方法：本仓无密度比机制（rho 为旁观量，三元模型注释明示），只可借用其自由能->mu->力链路与sigma/xi 标度关系来反解界面参数；高密度比需自建变密度平衡态。
  - 迁移评估（原文）：adapt xval:XVAL-LBM-FLUCTUATING-LBM-GHOST-MODES

### LBM-GAP-03 反例：多相只有布局模式、无物理实现（缺口与反例（不可迁移证据））

- 迁移判断：仅有布局模式（layout pattern），不可当作多相实现
- **2b-t/LB-t** `src/population/population.hpp:39` ｜ 仅作参照
  - 方法：该仓无多相实现；唯一可借的是 NPOP 并排存多套分布的布局机制与"组分索引 p 进算子签名"的接口范式（从未实例化 NPOP>1，故只是范式不是实现）。
  - 迁移评估（原文）：reference xval:XVAL-LBM-CASE-SCHEMA-JSON-UNITS-CONVERTER

### LBM-GAP-04 反例：相场仓不支持伪势（不可混用）（缺口与反例（不可迁移证据））

- 迁移判断：相场与伪势不可混用，选型时必须先分类
- **AminZar69/CudaPhaseField** `src/device_var.f90:25` ｜ 不可迁移/不适用
  - 方法：No contribution. The code has no Shan-Chen pseudo-potential, no interaction strength G, no colour-gradient, no component-resolved pressure boundary, no Poiseuille/layered-flow setup; its multiphase physics is a ternary Allen-Cahn phase field with algebraic third phase. Mapping it onto the MCMP Poiseuille gap would be a category error.
  - 迁移评估（原文）：none xval:XVAL-LBM-MAXREGCOUNT-PINNED-BEFORE-TUNING

### LBM-GAP-05 反例：无测试无 CI（不可信实现）（缺口与反例（不可迁移证据））

- 迁移判断：无断言无 CI 的实现只能取「想法」不可取「结论」
- **ebinan92/Fingering_dynamics** `lattice_boltzmann/validation.py:22` ｜ 原条目未标注
  - 方法：无 tests/、无 CI、无断言、无 requirements.txt；唯一含 Theta 的行从未被使用。验证面反面教材：任何从本仓搬来的物理都必须由我们自带验收门。
  - 迁移评估（原文）：reject xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION

### LBM-GAP-06 反例：cupy drop-in 未融合 kernel（缺口与反例（不可迁移证据））

- 迁移判断：换框架不等于提速；无融合的 GPU 移植会掉带宽
- **ebinan92/Fingering_dynamics** `lattice_boltzmann/fingering_periodic_gpu.py:2` ｜ 原条目未标注
  - 方法：GPU 版只是 numpy->cupy 直替换（78 处 cp.）, 无 RawKernel/ElementwiseKernel、无融合、无多卡；每步仍有 ~20 个算子级 kernel 与全网格临时量。只能参考算子求值顺序，性能路径不可继承（负面结论，用于关闭该能力的捷径假设）。
  - 迁移评估（原文）：reject-as-is xval:XVAL-LBM-CUPY-DROPIN-NO-KERNEL-FUSION

### LBM-GAP-07 反例：单文件算例 + 改 CMake 路径的组织方式（缺口与反例（不可迁移证据））

- 迁移判断：反面工程范式，TensorLBM 算例注册表须避免
- **sorush-khajepor/listLBM** `CMakeLists.txt:5` ｜ 仅作参照
  - 方法：每个算例是一个独立 main.f90，换算例必须手改顶层 CMakeLists.txt 的 example_path/project_name 再重新 cmake 构建；无任何自动化测试与 CI，验证结果只以 PNG/PDF 存在。
  - 迁移评估（原文）：reference xval:XVAL-LBM-EXAMPLE-PER-MAIN-ANTIPATTERN

