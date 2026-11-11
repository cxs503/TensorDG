# CASE-EULER-VORTEX 第2轮：调研 → 选型 → 结合实现

**锚点**：CASE-EULER-VORTEX（算例池：EULER-VORTEX｜ref=精确解｜容差 <5%）
**前轮**：m15260 honest-FAIL，`fail_stage`/`failure_reason` 均未留（失败无记录 = 本轮无法继承任何线索）
**本轮结论**：验收门 **PASS**（门禁配置 P2/32² 全场最大相对 L2 = 0.0127%，P0/128² = 2.817%，判据 <5%）

---

## ① 开源横向调研（OSS inventory）

| 渠道 | 做了什么 | 结果 |
|---|---|---|
| 本机 `~/oss-cache` | `ls` | 仅 3 个 LBM 仓（Maarten-vd-Sande_lbm / hschwane / pylbm），**无 DG/Euler 求解器**，对本算例无迁移价值 |
| 平台 `oss_inventory` 表 | `SELECT ... WHERE feature_slug ILIKE '%VORTEX%'` | **未能执行**：以本机 root 角色直连 tensorplatform 库对该表无 SELECT 权限（`ERROR: permission denied`），information_schema 亦不可见；API 侧无 oss 只读端点。此项列为「未跑」，不冒充已查 |
| 集群节点 `~/oss-cache` | `run_cluster submit`（作业 `cj1079`：`ls -1 ~/oss-cache`） | 递交成功但排队 288 位，本轮窗口内未回件；未取回前不引用其内容 |
| 已知同类实现（结构参照，非源码迁移） | 领域通行做法 | Trixi.jl / deal.II / FEniCS-DG 一类实现对本算例的共同结构：**解析解算子与数值算子分离**、误差在求积节点上取 L2 范数、分辨率阶梯逐级给观测阶、限制器默认关闭（光滑算例）。本仓 `tensordg/euler_dg_2d.py` 已具备同构结构，迁移点因此不是「换算法」而是「换验证编排」 |

**横向调研的净结论**：没有可直接搬运的源码级资产；本算例的缺口在**验证编排与门禁**（契约化算例、可机检门禁、可复现工件），不在数值内核。

## ② 方法谱系（文献/教科书标准做法 + 标准算例）

| 环节 | 标准做法 | 出处（external_anchor） |
|---|---|---|
| 算例本身 | 二维等熵涡：域 [-5,5]²，β=5，来流 (1,1)，γ=1.4，周期边界，t>0 精确解为初始剖面向下游平移 | C.-W. Shu, *ENO and WENO schemes for hyperbolic conservation laws*, ICASE Report 97-65 / NASA CR-97-206253 (1998), §4.4.2 |
| 精确解形式 | ρ=T^{1/(γ-1)}, T=1-(γ-1)β²/(8γπ²)·exp(1-r²), u=u∞-β/(2π)·e^{(1-r²)/2}·y', v=v∞+β/(2π)·e^{(1-r²)/2}·x', p=ρ^γ | 同上；亦见 Yee, Sandham & Djomehri, *Low-dissipative high-order shock-capturing methods using characteristic-based filters*, JCP 150 (1999) 中同一算例 |
| 空间离散 | 模态 DG（Legendre 张量积），界面 Riemann 通量 Rusanov 或 HLLC | Cockburn & Shu, *The Runge-Kutta discontinuous Galerkin method for conservation laws V*, JCP 141 (1998) |
| HLLC 通量 | Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer (2009), Ch. 10 |
| 时间推进 | SSP-RK3（凸组合保证 TVD/正性） | Gottlieb & Shu, *Total variation diminishing Runge-Kutta schemes*, Math. Comp. 67 (1998) |
| 误差口径 | 各原始变量相对 L2：‖q_h-q_ex‖₂/‖q_ex‖₂，求积节点上取值 | 高阶方法社区通行；本仓 `benchmarks/verify_euler_dg_2d.py::_vortex_error` |
| 收敛判据 | 观测阶 p=log(e_c/e_f)/log(h_c/h_f) 对齐设计阶 p+1 | 同上 |

**方法谱系净结论**：本算例的正确判据由**精确解 + 相对 L2 + 设计阶**三者构成；任何一项缺失都会让「误差<5%」变成一个不可判定的口号（这正是前轮可能踩的坑：无记录 = 无法区分是跑不动还是判据错配）。

## ③ 选型沉淀（`oss_solutions`，feature_slug=`CASE-EULER-VORTEX`）

> DB 写入信道不可用（同 ①，无 oss_inventory/oss_solutions 的写权限），选型记录以仓内工件形式沉淀，字段与 `oss_solutions` 对齐，可被后续摄入器直接入库。

机器可读记录：`docs/oss_solutions_CASE-EULER-VORTEX.json`

选型要点（每个仓/方法 → adapt / reference / reject）：

| 候选 | transfer | 理由 |
|---|---|---|
| 本仓 P0 路径（`euler_2d.py`） | **adapt** | 作为低阶基线纳入门禁：它证明<5% 判据**有判别力**（32² 单场 7.8% 判叉、128² 2.82% 判勾） |
| 本仓高阶 DG（`euler_dg_2d.py`） | **adapt** | 主路径：P2/32² 达 0.0127%，设计阶 3.0/3.1 实测 |
| Trixi.jl / deal.II 类实现的结构 | **reference** | 只借验证编排结构（解析解算子分离、求积节点 L2、阶梯观测阶），不搬代码 |
| 前轮 m15260 的做法 | **reject-as-is** | 无 fail_stage/failure_reason，不可复现、不可继承 → 本轮以「每配置一进程一检查点 + 门禁脚本」取代一次性长跑 |
| 限制器开启的 DG | **reject（本算例）** | 光滑算例，限制器只降阶（本仓 `docs/limiter_verification.md` 已验证），破坏设计阶 |

**迁移方案（已落地）**：
1. `benchmarks/case_euler_vortex.py`：契约化算例（域/β/来流/t/CFL/网格/容差写在模块常量里，一处定义），一条配置一次进程，记录落 `results/<key>.json`（被杀进程只损失自己那一格）。
2. 参考解**自校核**（本轮新增，防「拿错答案当基准」）：静止系 `div F=0`、平动系 `U_t+F_x+G_y=0`、等熵律、无散旋流、径向动量平衡五条独立判据。
3. `benchmarks/_case_euler_vortex_report.py`：把记录聚合成表/图/门禁，**报告数字程序化注入（零手抄）**，直接生成 `docs/CASE-EULER-VORTEX.md`。
4. `tools/run_case_euler_vortex.sh`：并行编排（每配置 `setsid --fork`，单线程），18 个配置全量 ~1 分钟墙钟（单配置 0.7 s ~ 60 s）。

## 与前轮失败的对照（本轮不再重蹈）

| 前轮死法 | 本轮对策 |
|---|---|
| 无 fail_stage / failure_reason | 每配置独立检查点 JSON；跑到哪一步就是哪一步的证据；门禁脚本显式输出 PASS/MISSING/FAIL 三态 |
| 一次性长跑被超时整组收割（本机单条 shell 120 s 上限） | 每个配置 `setsid --fork` 脱离进程组，逐配置落盘；实测单配置最长 60 s（P2/32²） |
| 判据口径可能错配（单元平均 vs 点值） | 两条路径各自声明度量口径，跨族只比量级，族内比收敛阶 |

## 未跑 / 局限

- `oss_inventory` / `negative_knowledge_ledger` 的 SQL 查询与 `oss_solutions` 入库**未跑**（权限）；集群作业 `cj1079` 未回件。以上均未在报告中当作已知事实引用。
- 文献出处来自领域知识，本环境无全文本访问；引用给出报告号/期刊卷期以便复核。