# AUDIT m16499 — CASE-DG-PREFINEMENT-EXPONENTIAL（独立复核 + 门禁加固）

- **锚点**：`CASE-DG-PREFINEMENT-EXPONENTIAL`（feature_ledger status=open）
- **被审对象**：`benchmarks/p_refinement_exponential.py`、`tests/test_p_refinement_exponential.py`、
  `docs/benchmarks/results/p-refinement-exponential/*`、`docs/CASE-DG-PREFINEMENT-EXPONENTIAL.md`
- **复核方**：m16499（verifier 席，本单）；被审代码来自 #318 (m16180) 合入版本，commit `5998d1d`
- **结论（disposition）**：**approve with changes** — 已发布数值**可复现**（下表，全部由本单重算取得），
  但原门禁自证、工件溯源越界、判据非鉴别性三项缺陷已在本单修复或登记；修复后门禁通过。
- **独立性限制**：复核与被审代码同仓同架构（loong64, torch 2.7.1），未与第三方 DG 实现交叉；
  重算使用同一 `solve_advection`，独立性仅体现在测量路径（四套 Gauss 阶数 + 独立 NumPy 基函数装配 + 另一时间积分器）。

## 1. 复核做了什么

| # | 检查 | 手段 | 结果 |
|---|---|---|---|
| C1 | 工件可复现性 | 用当前代码整跑 `python -m benchmarks.p_refinement_exponential`（79.5 s） | 六个 p 的 L2 误差与已发布 JSON **逐位一致**（7.464013e-03 / 2.089339e-04 / 5.041733e-06 / 9.731719e-08 / 1.592692e-09 / 2.228252e-11），σ=3.9271 |
| C2 | 测量路径独立性 | 新写 `benchmarks/p_refinement_independent_check.py`：自建 NumPy Legendre 基，nq=2p+4/16/32/64 四套 Gauss 求积重算 L2 | 四套求积相对离散度：p1 1.3e-15、p2 3.4e-13、p3 4.1e-12、p4 8.7e-10、p5 7.5e-9、p6 1.3e-6（见 F5） |
| C3 | 范数独立性 | 同解另算 L1、L∞ | 三者同阶单调下降，σ 结论不随范数改变 |
| C4 | 时间离散独立性 | RK4 vs SSP-RK3（Cockburn–Shu 1998）同空间算子 | p≤3（同 cfl）相对差 ≤1.4e-5；p≥4 降至 cfl/8 后差 4.4e-8（p4）/1.5e-5（p5）/3.4e-2（p6，绝对 7.6e-13，仍为 RK3 时间误差，见 F6） |
| C4b | 固定 cfl 的时间无关性 | 同 p 把 cfl 减半两次（0.08→0.04→0.02） | 末次相对变化 ≤5.8e-6（p=6）——**cfl=0.08 本身已时间收敛**，驱动里的 0.08→0.04 半步属冗余但不影响结论 |
| C5 | 工件交叉核对 | 重算值 vs 存储 JSON | p1–p3 相对差 ≤3.6e-9；p4 1.6e-8；p5 1.2e-6；p6 1.6e-3（= 该点时间误差份额） |
| C6 | 判据鉴别力（证伪对照） | 把**已知规律**序列喂给 `p_decay_diagnostics` | 常数 σ=-0.000；p⁻¹ σ=0.549；p⁻² σ=1.099；exp(σ=0.5) σ=0.500；exp(σ=3.93) σ=3.927 → 见 F4 |
| C7 | 门禁有效性 | 跑旧门禁 `tests/test_p_refinement_exponential.py` | **12 passed in 0.14 s，零次求解器调用** → 见 F1 |
| C8 | 溯源卫生 | 读工件 `figures` 路径 + 哈希 | 原工件指向 `/root/tp-agent/work/m16180/...`（他人工作树绝对路径），且无 sha256 → 见 F2 |
| C9 | 工件证据完整性 | 工件 schema 是否含逐 p 时间收敛记录 | 原 JSON 无 `levels`（cfl/steps/temporal_change），“时间已收敛”无法从工件自证 → 见 F3 |

## 2. 发现（按严重度）

### F1（HIGH，已修复）门禁自证：12 道闸门只读 JSON，不碰求解器
旧 `tests/test_p_refinement_exponential.py` 全部断言建立在
`p_refinement_results.json` 上（`test_verdict_passes` 直接断言 `art["verdict"]=="PASS"`）。
实测 **12 passed / 0.14 s / 零 solver 调用**：工件被手改或求解器回归，门禁都不会变色。
- 修复：新增 `tests/test_p_refinement_exponential_live.py`（10 道活闸门，~11.8 s）：用 shipped
  `solve_advection` 在 N=16、cfl=0.08 重算 p=1..3，经**独立 NumPy 求积**再测误差，
  断言单调性、σ>0.5（预登记判据）、σ>1 且相邻衰减率不减（鉴别判据）、求积独立性（<1e-9）、
  与存储工件一致（<1e-6）。
- 证据：`pytest -q tests/test_p_refinement_exponential_live.py tests/test_p_refinement_exponential.py`
  → **22 passed, 1 skipped**（skip = 集群独立复核工件未落地时跳过溯源检查）。

### F2（MEDIUM，已修复）工件溯源越界：figure 路径写进他人工作树、无哈希
原 JSON：`"p_refinement_semilog.png": "/root/tp-agent/work/m16180/docs/benchmarks/..."` —— 换机/换
worktree 即失效，且无法判断工件是否由当前代码产生。
- 修复：`generate_figures` 改为写**仓内相对路径**；`main()` 新增 `provenance.json`
  （`artifact_sha256` + python/numpy/torch 版本 + git commit + 配置 + 运行时长）。
- 回归守卫：`test_artifact_figure_paths_are_relocatable`、`test_independent_check_artifact_provenance`。

### F3（MEDIUM，已修复）工件缺逐 p 时间收敛证据
原 `build_report` 丢弃了 `levels`，于是“时间误差 <1%”这一关键声明在工件里无据可查（只在正文表里）。
- 修复：工件 `results.levels` 现含每 p 的 `cfl/dt/steps/temporal_change/temporal_converged/refinements`。
- 复跑结果：全部 p 首个 refinement 即收敛（cfl 0.08→0.04，`temporal_change` ≤ 1.6e-3 = 0.16%，阈值 1%）。

### F4（HIGH·方法论，未改判据，已登记 + 建议收紧）预登记判据 σ>0.5 **不具备鉴别力**
证伪对照（C6）：

| 已知误差律 | 测得 σ | 旧判据 σ>0.5 |
|---|---|---|
| 常数（不收敛） | 0.000 | 拒 |
| 代数 p⁻¹ | 0.343（3 点拟合时 0.549） | 拒（6 点）/ **过（3 点，假阳性）** |
| 代数 p⁻² | 0.685 | **过（假阳性）** |
| 代数 p⁻³ | 1.028 | **过（假阳性）** |
| 真指数 σ=0.5 | 0.500 | 拒（边界） |
| 真指数 σ=3.93 | 3.927 | 过 |

即：该判据会**放过代数收敛、卡掉恰好等于阈值的指数收敛**。本案实测 σ=3.93、相邻衰减率
3.576→3.724→3.948→4.113→4.269 单调递增（代数律的相邻衰减率必递减），故“指数收敛”结论成立；
但**判据本身需收紧**。建议后续算例预登记为复合判据：`σ>1` **且**相邻衰减率不减 **且** 误差比
`e(p_min)/e(p_max)>10³`。已用 `test_algebraic_law_fails_the_discriminating_criterion` 钉住该弱点，
防止后人把 `σ>0.5` 当充分条件。

### F5（LOW）p=6 已接近 float64 测量地板
p=6 误差 2.228e-11，而 `(u_h - u_exact)` 是两个 O(1) 量之差，点值相对不确定度 ~1e-16/2e-11 ≈ 5e-6；
实测四套求积相对离散度 p=6 达 **1.3e-6**（对应绝对 3e-17 ≈ roundoff 地板），p≥4 逐级升高的正是这一项。
结论：σ 与收敛律不受影响；但若把 p 推到 `MAX_DEGREE=8`，误差测量本身将进入 roundoff 主导，
届时应改用补偿求和/更高精度或改用模态系数解析残差度量。原文档“p=6 接近 machine precision 限制”的判断成立且已量化。

### F6（LOW，已修正协议）跨积分器对比必须随阶数缩放 dt
初次把 SSP-RK3 放在 RK4 的 cfl=0.08 下比较，p=6 出现 116 倍差异——这是 RK3 的时间误差
（~ (k·dt)⁴）压过空间误差 2.2e-11，**不是空间解不一致**。修复：`independent_check` 对 p≥4 用
cfl/8 跑 RK3 并记录 `ssp_rk3_cfl`；修复后 p4/p5 差 4.4e-8/1.5e-5，p6 仍余 3.4%（绝对 7.6e-13），
说明要到 0.1% 以下需再降 dt 约 4 倍——这**是 RK3 的时间误差**，不能当作空间解分歧的证据。

## 3. 复核数值表（本单重算，非抄录）

| p | DOFs | L2（重算） | 与旧工件相对差 | 求积离散度 | 时间误差份额 |
|---|---|---|---|---|---|
| 1 | 32 | 7.464013e-03 | ≤3.6e-9 | 1.3e-15 | ~0 |
| 2 | 48 | 2.089339e-04 | ≤3.6e-9 | 3.4e-13 | ~0 |
| 3 | 64 | 5.041733e-06 | ≤3.6e-9 | 4.1e-12 | ~0 |
| 4 | 80 | 9.731719e-08 | 1.6e-8 | 8.7e-10 | ~0 |
| 5 | 96 | 1.592692e-09 | 1.2e-6 | 7.5e-9 | ~1e-6 |
| 6 | 112 | 2.228252e-11 | 1.6e-3 | 1.3e-6 | 1.6e-3（RK4 自收敛记录） |

σ（最小二乘指数律衰减率）= **3.927**，判据阈值 0.5；鉴别判据（σ>1 且衰减率不减）= **PASS**。
Babuska & Suri 1994 的 `‖e‖≤C exp(-σp)` 只给形式，不含 σ 数值，故本表**无可比的文献常数**：
对照量是“指数律形状 + 单调递增的相邻衰减率”，不是某个外部数字。误差判定逐项：
- σ=3.927 vs 阈值 0.5（>7.8 倍裕度）：**通过**；σ>1 鉴别判据：**通过**；
- 工件—重算差 ≤3.6e-9（p≤3）≪1%：**通过**（p=6 的 1.6e-3 由时间误差份额解释，非空间解不一致）；
- 与文献相对误差：**无法给出百分比**——Babuska–Suri 未提供该算例的 σ 参考值，故本单以
  “判据 + 求积/范数/积分器三重独立性”替代外部数值对照，并在 §2 明确此为适用范围限制。

## 4. 未做的检查（unknown / not_run，不得视为通过）

- 与**第三方 DG 实现**（如 deal.II / Firedrake / Nektar）同算例逐点对比：not_run（无本地依赖，超出本单范围）。
- p=7、8（`MAX_DEGREE=8` 允许）：not_run——F5 表明该区间测量已被 roundoff 主导，需换度量方式。
- 二维/曲线单元上的 p 收敛：not_run（本算例为一维周期线性对流，结论不外推到 2D 变系数问题）。
- 集群侧独立复核作业 `cj1326`（loong64 双节点，`stage_dir` 送本工作树）已递交但排队未派发：
  本单证据全部来自控制面本机重算，未使用集群结果。

## 5. 交付工件

| 文件 | 说明 |
|---|---|
| `benchmarks/p_refinement_independent_check.py` | 独立复核脚本（多求积阶/多范数/跨积分器/证伪对照/工件哈希） |
| `tests/test_p_refinement_exponential_live.py` | 10 道活闸门（求解器联动，~11.8 s） |
| `docs/benchmarks/results/p-refinement-exponential/provenance.json` | 当前工件 sha256 + 环境 + commit |
| `docs/benchmarks/results/p-refinement-exponential/p_refinement_results.json` | 重跑工件（含 `levels`、相对路径 figures） |
| `docs/CASE-DG-PREFINEMENT-EXPONENTIAL.md` | §7 增补独立复核结论与判据收紧建议 |