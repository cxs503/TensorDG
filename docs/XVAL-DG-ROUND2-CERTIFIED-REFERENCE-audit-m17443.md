# 交差审计：m17365 XVAL-DG-ROUND2-CERTIFIED-REFERENCE（TensorDG）

审计单：m17443（board-audit）｜被审单：m17365（status=succeeded，finish 记录 `fail_stage=budget_exhausted+forced_finish`、`head_commit=null`）
审计日期：2026-11-13｜仓库：TensorDG｜性质：**交差核验，非重做**

## 结论（一句话）

交付的**数值是真的、可复现的、且经本单独立复算逐位吻合**（11 条 DG 腿 + 认证参考解 + 两轴敏感性全部重复出来，OSS 腿重跑与缓存逐位相同）；但 **L3/L4 均不成立**：交付物在 m17365 工作树内**全部未提交**（`git status` 全为 untracked、`head_commit=null`、main `70bd4c1` 无该文件）、**无任何调用方**、**锚点在 `tensor_benchmark_cases` 零登记（0 行）**、平台 L4 裁判 `CLAIMED_NOT_PROVEN rows=0`，且**随交付发出的一条测试是坏的**（`1 failed, 6 passed`，失败原因被交付自述错标）。**定级 L2**（真代码输入→计算→输出，已独立复算），返修条件见 §7。

## ① 案例与判据（ref 出处）

- 算例：CASE-SHU-OSHER（Shu & Osher, JCP 77 (1988) 439-471；γ=1.4，x∈[−5,5]，x<−4 为 Mach-3 激波后状态 (3.857143, 2.629369, 10.33333)，x≥−4 为 ρ=1+0.2 sin(5x)、u=0、p=1，t=1.8）。原文**只给图不给数表**→ 文献数值 **WITHHELD**，判据只能是「两套独立离散互相认证」+ 既有闭式锚（离散 id=19 已登记）。
- 预注册判据（开工前定，`xval_oss_shu_osher.py:80-83`）：**D1** 两认证参考解 rel L1 < 1.00 %；**D2** 差量 ≤ 3.0×√(U_ours²+U_oss²)；**D3** 全部 DG 腿对 OSS 认证参考解 < 5.0 %；owner 通用精度线 <15 %（本单实际用 <3 % 回答）。
- **基准裁判（唯一裁判）实况**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DG-ROUND2-CERTIFIED-REFERENCE'` → **0 行**（见 `artifacts/audit_m17365/logs/registry_and_judge.log`）。同类 XVAL 锚点（`XVAL-FEM-TPB1976`, id=14）已登记，说明本锚点属**漏登记**。故本单只能以被它重判的 `CASE-SHU-OSHER-DG`（id=19，tol `<5% rel L1`，DG ladder N=200/400/800×P1/P2/P3，CFL=0.4，状态 passed）+ 交付自身预注册判据作裁判，并明确记录该登记缺口。

## ② 方法与本单核验手段

交付分两条腿：
1. **本仓腿**：`benchmarks/fv_weno5_1d.py`（FV-WENO5 分量重构 / HLLC-Davis / SSP-RK3 / 3 层 ghost 透射），N=800/1600/3200，CFL=0.3，raw npz 已在 main（`validation/repro_shu_osher/artifacts/raw/fv_cfl0p3_n*.npz`，本单 md5 与 m17365 一致）。
2. **OSS 腿**：Clawpack **5.14.0 / pyclaw SharpClaw**（纯 Python 核，`euler_hllc_1D`，WENO5 `recon.weno`，SSP104，`BC.extrap`），同 N/CFL；原始剖面缓存 `artifacts/xval_m17365/oss_raw/*.npz`。
3. 认证机制（coarsen / rel L1 / 实测阶 / Roache GCI Fs=1.25 / Richardson）**刻意共用同一把尺**（`benchmarks/shu_osher_certified_ref.py`），这是 XVAL 的正当做法。

本单核验（全部为**审计员自己重写**的实现，不 import 交付模块的度量函数）：
- `artifacts/audit_m17365/independent_recompute.py`：自写 coarsen / rel L1 / rel L2 / 实测阶 / GCI / Richardson / 窗口（x≤−3.0 平台、−3.0<x<2.6 熵波区、x≥2.6 未扰正弦），从**原始 npz** 重算认证参考解、D1/D2、同网格差、11 条 DG 腿、三轴敏感性。
- OSS 腿真伪：从 `/tmp/oss/clawpack-5.14.0` **重跑 N=800 全时程**（55.2 s），与交付缓存逐位比对；并把 OSS 树的 6 个关键文件与官方 sdist tarball 逐文件 sha256 对照（`logs/oss_tree_integrity.log`）。
- repro 链：在有缓存条件下重跑 `python -m benchmarks.xval_oss_shu_osher` 全流程并与交付 `xval.json` 逐字段 diff（`logs/reproduction.log`）。
- 测试：在 `git archive` 干净副本上跑 `pytest tests/test_xval_oss_shu_osher.py`（交付原样 + 三行修复后各一次）。

**独立性限制（如实声明）**：审计与交付共用同一套**原始 npz**（FV/DG 腿）作为输入；审计未重跑 DG 求解器本身（N=800 P3 单腿分钟级，本单预算内未做），故「DG raw 是否确由 TensorDG 主求解器产出」本单 **not_run**；OSS 腿、聚合、认证、判定、报告数字注入则全部已独立复核。

## ③ 参数表（本单复核值）

| 量 | 值 | 出处 |
|---|---|---|
| γ / 域 / 终止时刻 | 1.4 / [−5,5] / 1.8 | 交付 doc ①，与本单读码一致 |
| 加密序列 / 加密比 r | 800,1600,3200 / 2 | `xval.json:ladder` |
| CFL（基准腿）/ GCI Fs | 0.3 / 1.25 | `xval.json:predeclared` |
| 实测阶 p：ours / OSS | 1.282509 / 1.294025 | 本单复算（逐位等于交付） |
| GCI U：ours / OSS | 0.082797 % / 0.082076 % | 本单复算（逐位等于交付） |
| OSS 版本 / 改写 | Clawpack 5.14.0（`clawpack/__init__.py:27 __version__='5.14.0'`，PKG-INFO 一致）/ 仅新增 1 个 82 B 空 shim | 本单 tarball 对照 |
| 参考解网格 | N=1600（第二细层） | `certified_refs.npz` x 长度 1600 |

## ④ 对照表（ref vs got，打勾/叉）

### 4.1 认证参考解跨实现差（D1/D2）——本单复算 vs 交付

| 量 | 本单复算 | 交付 | 一致 | <3 % |
|---|---|---|---|---|
| rel L1(ours_cert, oss_cert) | **0.011501 %** | 0.011500 % | ✅ 逐位 | ✅ 余量 ×261 |
| 合成带 3.0×√(U²+U²) | 0.349752 % | 0.349752 % | ✅ 逐位 | D2 PASS |
| ours p / GCI | 1.282509 / 0.082797 % | 同 | ✅ | — |
| oss p / GCI | 1.294025 / 0.082076 % | 同 | ✅ | — |
| `certified_refs.npz` 与复算剖面 max\|dev\| | 0.0 / 0.0 | — | ✅ 逐位 | — |

### 4.2 同网格原始解差（无外推，最硬口径）

| N | 本单复算 rel L1 % | 交付 rel L1 % | 一致 |
|---|---|---|---|
| 800 | 0.009555 | 0.009555 | ✅ |
| 1600 | 0.006647 | 0.006647 | ✅ |
| 3200 | 0.007803 | 0.007803 | ✅ |

### 4.3 11 条 DG 腿 vs OSS 认证参考解（D3，<5 % 门）

| 配置 | 复算 rel L1 % | 交付 rel L1 % | 复算 rel L2 % | 5 % 门 |
|---|---|---|---|---|
| P1 N=200 CFL=0.4 | 2.4559 | 2.4559 | 5.2632 | PASS |
| P1 N=400 CFL=0.4 | 1.5480 | 1.5480 | 3.6568 | PASS |
| P1 N=800 CFL=0.4 | 0.5768 | 0.5768 | 1.5166 | PASS |
| P2 N=200 CFL=0.4 | 2.4205 | 2.4205 | 5.2333 | PASS |
| P2 N=400 CFL=0.4 | 1.5161 | 1.5161 | 3.6099 | PASS |
| P2 N=800 CFL=0.4 | 0.5648 | 0.5648 | 1.4806 | PASS |
| P3 N=200 CFL=0.4 | 2.4268 | 2.4268 | 5.2529 | PASS |
| P3 N=400 CFL=0.4 | 1.5221 | 1.5221 | 3.6067 | PASS |
| P3 N=800 CFL=0.4 | 0.6678 | 0.6678 | 1.4905 | PASS |
| P2 N=400 CFL=0.1 | 1.5198 | 1.5198 | 3.6147 | PASS |
| P2 N=400 CFL=0.2 | 1.5182 | 1.5182 | 3.6117 | PASS |

D3 汇总：**11/11 PASS，全部 <3 %**（最差腿 2.4559 %，对 5 % 门余量 50.9 %）。

### 4.4 敏感性（本单复算，来自缓存 ladder 重算）

| 轴 | 变体 | p_obs | GCI % | 认证解差 vs base % | 交付 |
|---|---|---|---|---|---|
| CFL | 0.3 / 0.6 | 1.294025 / 1.297905 | 0.082076 / 0.081539 | 0.005126 | ✅ 逐位一致 |
| 时间积分 | SSP104 / SSP33 | 1.294025 / 1.291541 | 0.082076 / 0.081706 | 0.006705 | ✅ 逐位一致 |
| 重构 | 分量 / char_decomp=1 | 1.294025 / 1.173265 | 0.082076 / 0.061890 | 0.034311 | ⚠ 交付写「未完成」，实际缓存三层已齐（见 F4） |

### 4.5 真伪与不可篡改

- **OSS 腿重跑**：N=800、CFL=0.3、t=1.8，本单 55.2 s 全新跑出，与交付 `oss_cfl0p3_n800.npz` **max\|Δ\|=0.0（逐位相同）** → 缓存确由 Clawpack 实跑产出，非伪造、非本仓 FV 拷贝。
- **OSS 树未篡改**：`clawpack/__init__.py`、`riemann/riemann/euler_1D_py.py`、`pyclaw/src/pyclaw/{limiters/recon.py, sharpclaw/solver.py, solver.py, state.py}` 与官方 `clawpack-5.14.0.tar.gz` **6/6 sha256 相同**；全树差异只有 1 个新增空 shim（`riemann/riemann/euler_1D_py.py` 之外唯一一处，82 B，内容 `# env shim: ...`）+ `pyclaw.log` 运行残留。
- **聚合层可复现**：`build_report` 用缓存 ladder 重跑后与交付 `xval.json` 逐字段 diff，**仅** `sensitivity.char_decomp` 这一块不同（交付时第三层缓存尚未落盘），其余字段（D1、11 腿、同网格、CFL/SSP33 轴、provenance、oss_meta）**全部逐位相同**。

## ⑤ 收敛性/敏感性判读

- 两实现都由 successive differences 测得 p≈1.28/1.29（形式 5 阶，激波把 L1 阶压到 ~1），**两套独立代码给出同一结论**，这本身是跨实现自洽证据；认证参考解自身不确定度 U≈0.082 %，远小于被判定量的误差（2.46 % 最差腿），故「判定不被参考解噪声支配」成立。
- 认证参考解自身 §③ 参数表数值可信：本单用独立代码逐位复现。
- 3 % 与 15 % 两条线均满足（D1 0.0115 %；最差腿 2.4559 %）。

## ⑥ 注意点、失败模式与工件

### 严重度排序（findings）

- **F1（阻断 L3，高）交付物零提交**：m17365 `git status` 全 untracked、`head_commit=null`；本单在 main `70bd4c1` 上 `git ls-files | grep -i xval` **空**。即「认证参考解跨实现可与 OSS 交叉验证」这一结论目前在**版本库层面不存在**，工作树被判删即丢失。修复：提交并走 merge-master。
- **F2（阻断 L4，高）锚点零登记 + 机器裁判 rows=0**：`tensor_benchmark_cases` 该 anchor 0 行；`/tensor/bench/l4-verdict/17365` → `CLAIMED_NOT_PROVEN`，`failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]`，`rule: L4 requires >=1 platform-recomputed benchmark row`。**L4 不能自报**，必须落 `tensor_benchmark_cases`(anchor=本锚点, metric=认证解 rel L1 与 DG 腿 rel L1, tolerance=<1%/`<5%`) + `tensorbench_results` 行（含 ref/got/error 与集群作业号）。
- **F3（缺陷，中）随交付发出 1 条坏测试**：`tests/test_xval_oss_shu_osher.py:73` 以三元解包接收 dict（`x, rho_small, meta = xval.run_oss_level(...)`）→ `ValueError: too many values to unpack (expected 3)`。实测 `1 failed, 6 passed`（`logs/pytest_as_delivered.log`）。三行修复后 `7 passed in 2.42 s`（`logs/pytest_after_fix.log` + `proposed_fix/test_oss_leg_signature.patch`）。**交付自述把失败原因写成「积分测试阈值断言，未修完」与实际 TypeError 不符**——根因错标，属证据陈述瑕疵。
- **F4（陈述与实物不一致，中低）char_decomp 敏感性被写「未完成」，实际缓存三层已在盘**（`oss_cfl0p3_char1_n3200.npz` 落盘 05:22，晚于 `xval.json` 05:16）。已复算：p=1.173265、GCI=0.061890 %、与 base 认证解差 0.034311 %。方向是**欠报**（不是虚标），但报告应与实物对齐（重跑一次 aggregator 即可）。
- **F5（溯源缺口，中低）HLLC 求解器文件无哈希**：`OSS_PROVENANCE` 里写的路径 `clawpack/riemann/euler_1D_py.py` 在 5.14 sdist 中不存在（实际在 `riemann/riemann/euler_1D_py.py`），于是交付 provenance 表把**最该被哈希的那一个文件**标成 `missing`。本单补 hash：`4c4b6f5a8323b7128203d5722fc93c8553d07a6af35d6c61c13723b86149365b`，并与 tarball 相同。
- **F6（元数据小 bug，低）**：`verify_oss_tree` 用旧式 `MICRO = 0` 判版本 → 交付报 `clawpack unknown`（真实 5.14.0，`clawpack/__init__.py:27`）；`oss_meta.steps` 恒为 `-1`（`solver.num_step` 不存在）。两项不影响数值。
- **F7（内部措辞矛盾，低）**：doc ②「OSS 树**未修改**，仅以 sys.path 前置导入」 vs ⑥ 失败模式 2「以空 shim 文件占位」。实测确为 1 个新增 82 B shim；② 措辞应为「未修改任何既有文件，新增 1 个未被 1D 路径引用的空 shim」。

### 未跑项（unknown / not_run，不记为 passed）

- DG 求解器原始产出未重跑（只复算其落盘剖面）→ 见 §2 独立性限制。
- 全量 OSS ladder 从零重跑（800/1600/3200 共约 12 min）：本单只重跑 N=800（55.2 s，逐位相同）+ 其余用缓存复算。
- `char_decomp=2`（特征型）在 OSS 纯 Python 核 `NotImplementedError`：未绕道验证，交付已如实标注。
- 与文献数值的直接对比：**WITHHELD**（原文无数表，未转录）。

### 工件

`artifacts/audit_m17365/`：`independent_recompute.py`（审计员自写度量）、`independent_recompute.json`、`delivered/`（交付件冻结副本 + `logs/delivered_sha256.txt`）、`logs/{pytest_as_delivered,pytest_after_fix,reproduction,oss_tree_integrity,registry_and_judge}.log`、`proposed_fix/test_oss_leg_signature.patch`、`fig_audit_certified_refs.png`、`fig_audit_dg_ladder.png`、`make_figures.py`。

## ⑦ 判级与处置

**级别 = L2**（独立功能实现：真代码、输入→计算→输出、本单独立复算逐位复现）。理由与逐级判据：

| 级别 | 判 | 依据 |
|---|---|---|
| L0 文档 | 达成 | 六段齐全（doc 95 行） |
| L1 数学/单测 | 达成 | 6 条单测过；D1/D2/D3 判定复现（修复后 7 passed） |
| **L2 独立功能实现** | **达成（本单定级）** | `benchmarks/xval_oss_shu_osher.py` 432 行真代码；入口 `python -m benchmarks.xval_oss_shu_osher` 可跑；`artifacts/xval_m17365/{xval.json,certified_refs.npz,oss_raw/*.npz}` 实物在盘；独立复算逐位吻合 |
| L3 系统集成 | **不达** | 交付零提交（F1）、`tensordg/__init__.py` 无导出、`git grep` 无调用方、平台侧仅一条 report 无能力登记 |
| L4 基准验证 | **不达（机器裁判否决）** | `tensorbench_results` rows=0、锚点零登记（F2）；数值质量本身达 L4 强度，但基准门不认自报 |
| L5 科研工程 | 不达 | 无外部可用产物/无集成 |

**处置：approve with notes → request changes（返修，不消账为 L3/L4）**。返修清单（按性价比）：
1. F3 三行修测试（patch 已给，`7 passed` 已验证）；
2. F1 提交三件套 + docs + figs 到分支并 merge-master；
3. F2 登记 `tensor_benchmark_cases`（anchor=XVAL-DG-ROUND2-CERTIFIED-REFERENCE，两条 case：认证解 rel L1 tol `<1%`、DG 腿 rel L1 tol `<5%`）+ 落 `tensorbench_results`（ref/got/error + 集群作业号）；
4. F4 重跑 aggregator 补齐 char_decomp 敏感性；F5 修正 provenance 路径并补 HLLC 文件 sha256；F6/F7 措辞与小 bug。

**误差判定（显式回答是否 <3 %）**：认证参考解跨实现差 **0.011501 % < 3 %**（且 < 预注册 1 %）；最差 DG 腿 **2.4559 % < 3 %**（对 5 % 门余量 50.9 %）；参考解自身不确定度 0.082 % ≪ 判定量。**结论：数值层面 3 % 线全部满足**；不满足的是交付纪律（提交/登记/测试）而非数值。