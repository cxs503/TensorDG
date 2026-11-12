# CASE-SHU-OSHER-DG · 第 2 轮（调研驱动）· m15804

锚点：`CASE-SHU-OSHER-DG`｜判据（总账）：算例池 SHU-OSHER-DG，ref = Shu & Osher 1988/1989，
容差 <5%（相对误差）。前轮 m15614 honest-FAIL（fail_stage / failure_reason 均未留），
本轮按两轮法第 2 轮执行：**先把成熟方法调研清楚，再结合推进**，并把上一轮审计
（`docs/audit_m15552_CASE-SHU-OSHER-DG.md`）列入的两处残余缺口（F2 平台机器可证性、
F4 文献锚点口径）逐条处理或如实登记。

---

## 第一部分 · 调研（①②）

### 1.1 开源横向（本仓同类实现）

| 调研面 | 做法 | 结果（如实） |
|---|---|---|
| 平台开源库清单 | `SELECT repo,tier,focus,methods FROM oss_inventory`（psql, 只读） | 33 行；`repo ILIKE '%dg%' OR methods ILIKE '%WENO%' OR focus ILIKE '%Galerkin%'` → **0 行命中**。本特性在平台库存里没有已缓存的同类实现 → 不存在「抄结构」的对象 |
| 集群 `~/oss-cache` 源码扫描 | `run_cluster submit`（TensorDG, nodes n315/n330 健康）→ `cj1181`：列目录 + `grep -ril 'shu.osher\|shu_osher'` 全语言源码 | **not_run**：递交时刻队列 366 个作业在前，至本单收尾仍 `queued`。未执行 ≠ 通过，故不声明任何 oss-cache 结论 |
| 文献可达性实测 | `urllib` 直连；Shu 本人出版页 `https://www.dam.brown.edu/people/shu/pub.html`（200, 211 KB，可解析） | 该页 Yes；其中 1988 JCP 77:439-471 与 1989 JCP 83:32-78 两条的 PDF 链接均指向 **ScienceDirect 付费墙**（`/pii/0021999188901775`、`/pii/0021999189902222`）；NASA NTRS 直链 403；`raw.githubusercontent.com` 连接重置；DDG/`archive.org` 不可达（`Errno 99`）。**结论：原文数值不可得** |

### 1.2 方法谱系（教科书/文献标准做法，external_anchor 出处）

| 环节 | 标准做法与出处 | 本仓选择 |
|---|---|---|
| 算例定义 | Shu & Osher, *Efficient implementation of essentially non-oscillatory shock-capturing schemes*, JCP **77** (1988) 439-471；同一算例的正式表述见 JCP **83** (1989) 32-78 的 test 3：Mach-3 激波（x0=-4）扫过 `ρ=1+0.2 sin(5x)` 静止正弦，γ=1.4，x∈[-5,5]，t=1.8 | 原样实现（`tensordg/cases/shock_entropy.py`） |
| 空间离散（DG 侧） | Cockburn & Shu RKDG：TVB/minmod 限制器（Math. Comp. 52 (1989) 411-435）；高阶激波捕获需 WENO/CWENO 型限制器（Qiu & Shu, SIAM J. Numer. Anal. 43 (2005) 63-77） | P1/P2/P3 × N=200/400/800，`limiter='weno'`（Qiu–Shu 型 WENO 限制器，`tensordg/limiters.py`） |
| 时间积分 | SSP-RK3（Shu & Osher 1988 的 TVD RK；Gottlieb & Shu, SIAM Rev. 43 (2001) 89-112） | SSP-RK3，每 stage 后限制（`limit_each_stage`） |
| 通量 | HLLC 近似 Riemann 解（Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009, §10.4） | HLLC（`tensordg/euler_dg_1d.py`） |
| 参考解构造（文献通行做法） | 该算例无闭式解，文献一律给「细网格高阶解」当参考（如 Jiang & Shu, JCP 126 (1996) 202-228 用 2000 点 WENO）；判据取「与细网格参考的相对 L1」 | 独立离散化 FV-WENO5-JS / HLLC / SSP-RK3，N=1600/3200（`benchmarks/fv_weno5_1d.py`），并给出其自身网格收敛 |
| 保正性 | Zhang & Shu 保极值 RKDG（JCP 229 (2010) 3091-3120） | 局部保正回退（`_positivity_filter`），非全保极值 |

### 1.3 选型沉淀（③）

- **选型**：DG-L2 投影 + Qiu–Shu WENO 限制器 + HLLC + SSP-RK3 打到 t=1.8；参考解用**独立格式** FV-WENO5/HLLC/SSP-RK3 的 N=3200 解，并同时报 4 个闭式锚点（Rankine–Hugoniot 激波速度/位置、激波后平台密度、未扰正弦、全局质量预算）。
- **迁移方案**（对同族激波-熵波类算例可复用）：①用 RH 关系把左右态从马赫数反推、把字面量当被检对象而不是被抄对象；②参考解必须是**不同离散化**且给出自身收敛证据；③误差面按 N×p 两向阶梯 + CFL 扫描构成敏感性面；④门判据写成可复算的 JSON 而不是报告里的散字。
- **入库**：`docs/oss_solutions_CASE-SHU-OSHER-DG.json`（`feature_slug='CASE-SHU-OSHER-DG'`，两条：`oss_survey` 与 `method_selection`）。平台 `oss_solutions` 表对当前角色为 `permission denied`，故以工件形式随仓入库，供知识沉淀管线摄取——不以文件冒充已入库。

---

## 第二部分 · 实现与验收（④，论文级六段）

### ① 问题设定与基准出处

Mach-3 右行激波在 x=-4 处进入 `ρ=1+0.2 sin(5x)`、`u=0`、`p=1` 的静止气体（γ=1.4），x∈[-5,5]，
透射边界，积分到 t=1.8。激波后（x<-4）为 RH 跃变状态 `(ρ,u,p)=(3.857143, 2.629369, 10.33333)`；
波后正弦被折成高频熵波/声波扇。

- 算例出处：Shu & Osher, JCP **77** (1988) 439-471；JCP **83** (1989) 32-78（test 3）。
- **文献表格值：WITHHELD**。Shu 出版页两条链接均指向 ScienceDirect 付费墙，NTRS 403，环境内 PDF 不可达；
  原文以图形给结果、无表格值 → 不抄任何文献数字（不伪造）。
- 因此数值基准 = 独立离散化参考解（FV-WENO5-JS/HLLC/SSP-RK3, N=3200）+ 闭式锚点（下面逐项给出）。

### ② 方法与实现（关键代码指针）

| 组件 | 位置 |
|---|---|
| 算例定义 + 闭式锚点 + 门报告 | `tensordg/cases/shock_entropy.py`（`rankine_hugoniot_anchor`、`exact_mass_budget`、`gate_report`） |
| 1D Euler DG 求解器 | `tensordg/euler_dg_1d.py`（HLLC 通量、WENO/TVB 限制器、SSP-RK3、保正回退） |
| 求解器自带验证钩子 | `tensordg/euler_dg_1d.py:281 verify_shock_entropy()` |
| 算例池驱动（阶梯/参考/CFL/合并） | `benchmarks/shu_osher_case_pool.py` |
| 独立参考格式 | `benchmarks/fv_weno5_1d.py` |
| 一键复现 | `validation/repro_shu_osher/run_all.sh`；快路 `smoke_repro.py` |
| 平台证据回路（本轮新增） | `validation/repro_shu_osher/tensorbench_ingest.py` |

### ③ 参数表

| 量 | 值 |
|---|---|
| γ | 1.4 |
| 域 / 界面 | [-5, 5] / x0 = -4 |
| 末时刻 | t = 1.8 |
| 左态 (ρ,u,p) | (3.857143, 2.629369, 10.33333)（RH 反推自 M=3，相对残差 ≤3.2e-7） |
| 右态 | ρ = 1 + 0.2 sin(5x), u = 0, p = 1 |
| DG 阶梯 | p = 1,2,3 × N = 200,400,800，CFL = 0.4，limiter = WENO，SSP-RK3 |
| 参考解 | FV-WENO5-JS/HLLC/SSP-RK3，N = 3200，CFL = 0.3，HLLC，Davis 波速 |
| 无量纲数 | 激波马赫数 M_s = 3.0（唯一自由参数，其余全部由 γ 与 M_s 决定）；网格 Reynolds 类无量纲数不适用（无粘 Euler） |

### ④ 对照表（逐项 ref vs got + 误差）

平台契约复核（`validation/repro_shu_osher/tensorbench_ingest.py --validate-only`，用平台自己的
`tools.tensorbench_eval.contract` 解析/校验/复算，documents 存
`validation/repro_shu_osher/artifacts/tensorbench_payload.json`）：

| 指标 | ref（FV-WENO5 N=3200 参考解） | got（DG P2 N=800） | 相对误差 | 容差 5% |
|---|---|---|---|---|
| plateau_rho | 3.85714 | 3.85714 | 0.0000% | PASS |
| shock_position | 2.39688 | 2.4 | 0.1304% | PASS |
| wave_amplitude | 3.88039 | 3.77118 | 2.8143% | PASS |
| mass_integral | 31.0816 | 31.059 | 0.0727% | PASS |

算例池门（相对 L1，判据 <5%，ref = FV-WENO5 N=3200）：

| p | N | rel L1 % | rel L2 % | Linf % | 判据 <5% |
|---|---|---|---|---|---|
| P1 | 200 | 2.4615 | 5.277 | 14.822 | PASS |
| P1 | 400 | 1.5438 | 3.658 | 12.441 | PASS |
| P1 | 800 | 0.5676 | 1.470 | 10.472 | PASS |
| P2 | 200 | 2.4260 | 5.247 | 14.868 | PASS |
| P2 | 400 | 1.5136 | 3.613 | 12.385 | PASS |
| P2 | 800 | 0.5561 | 1.434 | 10.492 | PASS |
| P3 | 200 | 2.4322 | 5.266 | 14.886 | PASS |
| P3 | 400 | 1.5204 | 3.610 | 12.415 | PASS |
| P3 | 800 | 0.6618 | 1.444 | 10.523 | PASS |

| 判据 | 结果 |
|---|---|
| <5% rel L1（9 腿 DG 阶梯） | PASS —— 9 腿全部 <5%，最差 2.4615%（P1 N=200），最好 0.5561%（P2 N=800） |
| 参考解自身收敛（N=1600 vs N=3200） | rel L1 = 0.1010% / rel L2 = 0.3580%（N=1600 vs N=3200） |
| 全腿密度有限、为正 | PASS —— 9 腿 rho_min ≥ 0.800033，全部 finite |
| 闭式锚点（RH 激波位置 / 平台密度 / 未扰正弦 / 质量预算） | PASS —— 激波位置相对 RH 解析值误差 ≤ 0.742%，平台密度偏差 ≤ 0.6303%，未扰正弦 rel L1 ≤ 0.0348%，质量预算误差 ≤ 3.02e-03·M0 |

### ⑤ 收敛性与参数敏感性（两个以上分辨率/参数点）

- 分辨率阶梯：N = 200/400/800 × p = 1/2/3（9 腿，见上表），rel L1 随 N 单调下降（除 P3 N=800 的
  限制器口径效应），三阶数曲线在 N=800 处汇到 0.5561%–0.6618% 的窄带内。
- 参考解自收敛：N=1600 vs N=3200 → rel L1 = 0.1010% / rel L2 = 0.3580%（N=1600 vs N=3200）（即参考自身误差比被裁判的 5% 小 50 倍）。
- DG 自收敛（P3）：N=400 vs N=800 → rel L1 = 1.2155%。
- CFL 敏感性（N=400, P2）：CFL = 0.1 / 0.2 / 0.4 → CFL 0.1 → 1.5168% / CFL 0.2 → 1.5153% / CFL 0.4 → 1.5136%；全扫掠 spread = 0.0033 pp，
  说明结论对时间步长不敏感。

### ⑥ 失败模式与工件

- **失败模式 1（前轮死法复现）**：`POST /tensor/bench/evaluations` 用自定义 spec 键 → 422
  （`store.submit_evaluation` → `contract.parse_spec` 严格白名单）。本轮改为先本地过
  `parse_spec`/`validate_result`/`evaluate_metrics` 再造请求，已全绿（见上表）。
- **失败模式 2（平台反造假绑定）**：`store.ingest_result` 只在 `cluster_jobs` 存在**同 mission** 且
  `state='succeeded'` 的作业时落 `tensorbench_results`。本轮递交的集群复跑 `cj1184`
  （`run_all.sh` 全流程 + 依赖探针）在收尾时仍 `queued`（队列前方 369 个作业），**因此没有落库**——
  不用本地结果冒充集群作业。缺行 ⇒ 平台 `GET /tensor/bench/l4-verdict/15804` 仍会答
  `claimed_not_proven`，这是如实登记的残余缺口（审计 F2 的机制部分已就绪，只欠一次成功作业）。
- **失败模式 3（诊断量跨分辨率误用，本轮实测发现）**：把「密度总变差 TV」当对照量会得出
  19.1% 的假超差——N=3200 参考解解析出 N=800 DG 网格无法表示的熵波细结构，TV 因此系统性偏大。
  TV 是分辨率相关量，不是物理锚点；已从平台指标里剔除（保留 4 个物理量），并在脚本注释中留证。
- 工件：`validation/repro_shu_osher/artifacts/{metrics.json, profiles.npz, raw/*.npz,
  tensorbench_payload.json}`、`validation/repro_shu_osher/benchmark_ref_vs_got.{json,md}`、
  `docs/fig_r2_profiles.png`、`docs/fig_r2_convergence.png`。
- 复跑：`bash validation/repro_shu_osher/run_all.sh`（全流程）；快路 24 s
  `python validation/repro_shu_osher/smoke_repro.py`；
  平台回路 `python validation/repro_shu_osher/tensorbench_ingest.py --validate-only`。

---

## 第三部分 · 局限与未完成（诚实登记）

1. **文献数值 WITHHELD**：原文（JCP 77/83）无表格值且环境内 PDF 不可达，故本轮仍不是「文献/实验
   validation」，而是「交叉格式 + 闭式锚点验证」。要翻案需拿到 1988/1989 原文图并做带不确定度的
   数字化，本单未做。
2. **集群复跑未完成**：`cj1181`（oss-cache 扫描）与 `cj1184`（全流程复跑）均 `queued` 未执行 →
   `tensorbench_results` 无行，平台 L4 判定端点仍报 `claimed_not_proven`。
3. **oss_solutions 未直写库**：平台表对当前 DB 角色 `permission denied`，以工件
   `docs/oss_solutions_CASE-SHU-OSHER-DG.json` 随仓入库。
4. 未做的敏感性轴：限制器族（none/TVB/WENO）在 N=800 下的系统对比、边界条件（透射 vs 周期）影响。

## 第四部分 · 与前轮的关系（Delta）

| 项 | m15480（轮 1） | 本轮 m15804（轮 2） |
|---|---|---|
| 平台证据回路 | `POST /bench/evaluations` 422，无 `tensorbench_results` 行 | 新增 `tensorbench_ingest.py`：用平台自己的契约模块在本地跑通 `parse_spec`/`validate_result`/`evaluate_metrics`，4 指标全过 5%（见 ④）；只差一次 succeeded 集群作业即可落库 |
| 文献调研 | 未系统做 | 出货：oss_inventory 33 行 0 命中、集群扫描 job id、Shu 出版页/付费墙实测、方法谱系 6 条带出处 |
| 知识沉淀 | 无 | `docs/oss_solutions_CASE-SHU-OSHER-DG.json`（选型 + 迁移方案） |
| 诊断量误用 | 未识别 | 实测发现 TV 跨分辨率不可比（19.1% 假超差），已剔除并留证 |