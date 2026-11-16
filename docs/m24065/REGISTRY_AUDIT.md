# TensorDG 能力注册表注记补全 · REGISTRY_AUDIT (m24065)

> 锚点=CAP-REG-DG · 对象=PR#786 交付的空注册表基础设施（0 条目）· 本单=为真实能力逐个加 `@capability` 注记并如实评定 maturity。
> 基准裁判：注册表类交付无文献/实验锚（`tensor_benchmark_cases` anchor_slug='CAP-REG-DG' → 0 rows，同 CAP-REG-DEM 先例）→ 无 ref vs got 物理误差可列；判据以本地纪律 + 双模对拍为准。

## ① 案例介绍（对象/参数表）
对象=TensorDG 能力注册表（元工具）。PR#786 只交付了 `tensordg/registry.py`（@capability 装饰器 + --static AST + import + --parity + 本地纪律校验），但全仓 0 处注记 → 双模均 0 条目（m24035 判 L1，C003 FAIL）。
| 项 | 值 |
|---|---|
| 注记模块数 | 29 个 `tensordg/*.py`（solver 17 / limiter 3 / analysis 4 / geometry 2 / time_integration 2 / infrastructure 1） |
| 条目总数 | 29 |
| maturity 分布 | {"planned": 27, "mature": 2} |
| category 分布 | {"solver": 17, "limiter": 3, "analysis": 4, "geometry": 2, "infrastructure": 1, "time_integration": 2} |
| source_commit | 375183574098ecdf0ee71198b3c55b419c706cfe |

## ② 怎么做的（方法/流程/工具链）
1) 扫 `tensordg/` 全包，识别真实求解器/限制器/时间积分/几何/分析入口函数（`solve_*`/`apply_*`/`main` 等公开入口）。
2) 逐个加 `@capability(name, category, maturity, evidence/verified/endpoint, description, tags)` 注记；`from .registry import capability` 局部导入（自注册，不改公共 API）。
3) maturity 如实评定：
   - **mature**（2）：`dispersion-analysis`（CLI 实跑，p0 闭式色散锚 max|λ-闭式|=0，ANALYTIC 级）+ `certified-reference`（Shu-Osher 认证参考，XVALUE 级）——均满足 mature 需 verified(ANALYTIC/XVALUE)+endpoint(wired+cli:)。
   - **planned**（27）：其余已实现能力，均带 `evidence.feature_ledger` 锚（feature_ledger 表 domain=TensorDG 真实 slug）。
4) 修 3 处注册表基础设施缺陷（否则静态/导入双模无法如实校验）：
   - 静态 AST 抽取器原先丢弃 evidence/endpoint/verified → 静态模式对 planned/mature 纪律全盲；
   - import 模式原先按 `dir(pkg)` 前缀匹配（恒 0 命中）且脚本直跑时读错 registry 实例 → 改为导入全部子模块后读规范 `tensordg.registry` 单例；
   - 规则5 指针解析原先取 `func.split('.')[0]`（import 模式 3 段路径误判为包名）→ 改为取函数名前一节。
工具链：python3.11 + torch 2.7.1 + psql(feature_ledger 取证) + 平台 API(8010)。

## ③ 计算结果（数字表，程序化注入零手抄）
| 量 | 期望 | 复算 | 判定 |
|---|---|---|---|
| --static exit / json.loads | C001 exit0+可解析 | exit 0，json OK | PASS |
| 条目数 | C003 >0 | **29** | PASS |
| 本地纪律 problems | C003 全过 | **0**（static）/ **0**（parity 双模） | PASS |
| --parity 双模 | C002 一致或差异如实 | static=29 import=29 parity=True | PASS |
| REGISTRY_AUDIT.md | C003 在案 | 本文件 | PASS |

### 条目清单（29 条，按 category/name 排序）
| name | category | maturity | function | feature_ledger 锚 / verified |
|---|---|---|---|---|
| certified-reference | analysis | mature | `certified_reference.main` | verified=XVALUE (no ledger anchor needed for mature) |
| convergence-study | analysis | planned | `convergence.observed_order` | CASE-DG-ADVECT-ORDER (nearest-case) |
| dispersion-analysis | analysis | mature | `dispersion_analysis.main` | verified=ANALYTIC (no ledger anchor needed for mature) |
| xfamily-order | analysis | planned | `xfamily_order.cross_family_difference` | XVAL-DG-ROUND2-CERTIFIED-REFERENCE (exact) |
| curved-geometry | geometry | planned | `curved_geometry.gll_nodes_weights` | CASE-DG-ADVECT-ORDER (nearest-case) |
| wall-flux-extrapolation | geometry | planned | `wall_flux_extrapolation.wall_heat_flux_extrapolated` | CASE-DG-ADVECT-ORDER (nearest-case) |
| capability-registry | infrastructure | planned | `registry.main` | CAP-REG-DG (exact) |
| artificial-viscosity | limiter | planned | `artificial_viscosity.av_sensor` | CASE-DG-BLASTWAVE-PERSSON (exact) |
| char-limiter-2d | limiter | planned | `char_limiter.apply_characteristic_tvb_limiter_2d` | CASE-SHU-OSHER-DG (nearest-case) |
| tvb-weno-limiters | limiter | planned | `limiters.apply_limiter` | CASE-SHU-OSHER-DG (nearest-case) |
| acoustics-2d | solver | planned | `acoustics.solve_acoustic_2d` | CASE-ACOUSTIC-PULSE (exact) |
| advection-1d | solver | planned | `advection_1d.solve_advection` | CASE-DG-ADVECT-ORDER (exact) |
| advection-2d | solver | planned | `advection_2d.solve_advection_2d` | CASE-DG-SOLID-BODY-ROTATION-2D (exact) |
| advection-3d | solver | planned | `advection_3d.solve_advection_3d` | CASE-DG-ADVECT-ORDER (nearest-case) |
| advection-diffusion-1d | solver | planned | `advection_diffusion_1d.solve_advection_diffusion_1d` | CASE-DG-ADVECT-ORDER (nearest-case) |
| advection-diffusion-2d | solver | planned | `advection_diffusion.solve` | CASE-DG-SOLID-BODY-ROTATION-2D (nearest-case) |
| burgers-1d | solver | planned | `burgers_1d.solve_burgers` | CASE-BURGERS-SIN (exact) |
| burgers-shock-1d | solver | planned | `burgers_shock_1d.solve_burgers_shock` | CASE-DG-SOD-SHOCK (exact) |
| elastic-1d | solver | planned | `elastic_1d.solve_elastic_1d` | CASE-DG-DALEMBERT-ELASTIC-1D (exact) |
| euler-dg-1d | solver | planned | `euler_dg_1d.solve_euler_dg_1d` | CASE-SOD-DG (exact) |
| euler-dg-2d | solver | planned | `euler_dg_2d.solve_euler_dg_2d` | CASE-EULER-VORTEX (exact) |
| heat-1d | solver | planned | `heat_1d.solve_heat_1d` | CASE-DG-ADVECT-ORDER (nearest-case) |
| kovasznay-ns | solver | planned | `kovasznay.solve_kovasznay_dg_2d` | CASE-DG-KOVASZNAY-NS (exact) |
| ldg-dirichlet-1d | solver | planned | `ldg_dirichlet_1d.solve_ldg_dirichlet_1d` | CASE-DG-LDG-POISSON (exact) |
| poisson-2d-ldg | solver | planned | `poisson_2d_ldg.solve_poisson_2d_ldg` | CASE-DG-LDG-POISSON (exact) |
| swe-1d | solver | planned | `swe_1d.solve_swe_1d` | CASE-DG-RITTER-DAMBREAK-SWE (exact) |
| swe-2d | solver | planned | `swe_2d.solve_swe_2d` | CASE-DG-RITTER-DAMBREAK-SWE (nearest-case) |
| rk-stability | time_integration | planned | `rk_stability.stability_function` | CASE-DG-ADVECT-ORDER (nearest-case) |
| runge-kutta | time_integration | planned | `runge_kutta.rk_step` | CASE-DG-ADVECT-ORDER (nearest-case) |

### maturity 评定依据
- **mature（2）**：`dispersion-analysis`、`certified-reference`。判据=有真实 CLI endpoint（`python -m tensordg.<mod>`，wired）+ verified 达 ANALYTIC/XVALUE 级。`dispersion-analysis` 已实跑（p0 闭式锚 0.0、p=0..4 耗散/相位指数 2/4/6/8/10 与理论 2p+2/2p+2 吻合）；`certified-reference` 为 m17539 Shu-Osher 认证参考（XVALUE）。
- **planned（27）**：已实现且多数有测试，但按本地纪律「mature 需 verified+wired」，库函数无 endpoint 不得称 mature；故如实记 planned 并挂 feature_ledger 需求锚。锚分两类：**exact**（该能力有专属 ledger 行）与 **nearest-case**（无专属行，挂最近算例锚，已在表中标注，非虚构——ledger 按算例组织，求解器由算例验证）。

## ④ 结果图
本单=后端/元工具注记，无物理场量、无算例云图；可图量仅为「条目数 29 / problems 0 / parity True」标量，绘图无信息量，故不附 png（纯文档/后端任务无图，理由已写明）。

## ⑤ 注意点（坑/局限/适用范围）
- **maturity 语义**：本注册表 maturity 是「交付/认证成熟度」而非「实现状态」；planned≠未实现，而是「已实现但未达 mature 认证（verified+wired）」。27 条 planned 中多数已有测试与解析/交叉验证，升级 mature 只差 endpoint 接线 + verified 字段补全（后续单）。
- **nearest-case 锚**：12 条无专属 ledger 行，挂最近算例锚（表内标注 nearest-case）；非虚构，但非 1:1，升级时需补专属 ledger 行。
- **import 模式依赖 torch**：本主机 torch 2.7.1 可用，双模对拍成立；torch-free 主机仅 --static 可用（设计如此）。
- 注册表装饰器签名（name/category）仍偏离双试点规范（id/kind + module_path/entry/params/wired/tested/test_path + REGISTRY 单例）——本单不扩签名（判据冻结），仅补注记 + 修 3 处抽取/校验缺陷。

## ⑥ 误差判定
本单=注册表/元工具交付，无物理场量、无 ref vs got 数值对照（无基准裁判行，原因=注册表类交付无文献/实验锚）。关键量=条目数/problems/parity，均为程序化注入（registry JSON），无手抄；「是否<3%」不适用（无物理基准），已声明原因。
