# 交差核验：m22309 `XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`（board-audit m22345）

**被审计交付**：`agent/m22309 @ ebfeee4`（PR #621，平台台账 state=merged @2026-11-15T20:04:15Z） “TensorLBM: static immersed-marker obstacles (LIFE convention) + Re=100 cylinder xval”
**唯一裁判**：`tensor_benchmark_cases` 中该 anchor 的行 **不存在（0 行）**；因此落到仓内钉住的裁判
`TensorLBM/src/tensorlbm/t3_matrix.py:113-131`（`CASES['cylinder-2d-re100']`，Cd=1.35 带 1.32–1.38、St=0.165 带 0.164–0.166，
判据 `|Δ|/ref ≤ 0.10`，生态上限 0.15）。
**裁定**：**L3（confirmed-grade，基准未过）**；L4 拒绝。落表 `tensor_maturity_verdicts` id=900。

---

## 1. 案例介绍（物理背景 / 几何 / 参数）

锚点要求复现 `joconnor22/LIFE` 的 **静态拉格朗日标记障碍**做法：障碍物不是固体 mask，而是按周长等距展开的
标记点云（`floor(2πr/dx)` 个，间距 ≈ 1 dx，相邻 < 0.5 dx 剪掉），标记速度恒 0；标记力经紧支撑 delta 核铺到格子，
带 ε 松弛的多直接强迫。物理判据是 **Re=100 圆柱绕流** 的 Cd 与 St。

| 项 | 值（来自交付实现，非手抄） |
|---|---|
| 格子 / 碰撞 | D2Q9、BGK、float64 |
| 通道 | 320 × 162（20 D × 10 D，堵塞比 10%） |
| 来流 / Re | u0 = 0.05，Re = 100（nu = 0.008，tau = 0.524） |
| 标记 | n = 50（`floor(2πr/dx)`），实距 1.00531 dx |
| 步数 | 1200（时间平均窗口 = 后 45%，即 540 步 = 15.12 D ≈ 2.5 个脱落周期） |
| 基准 | Cd = 1.35（1.32–1.38）、St = 0.165（0.164–0.166），判据 ≤10% |

## 2. 怎么做的（核验方法）

1. **取证**：读 `report_m22309.json`（六段 + 3 图，base64 内嵌）、`docs/CASE-XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES.md`、
   `TensorLBM/outputs/cylinder_ibm/*`、`TensorLBM/src/tensorlbm/{immersed.py,cylinder.py,t3_matrix.py}`、`tests/test_immersed_ibm.py`。
2. **裁判比对**：`psql -c "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES'"`
   → **0 行**（`ILIKE '%IMMERSED%'` 亦 0），故用仓内 T3 钉值逐项比对 got。
3. **测试**：`python3 -m pytest tests/test_immersed_ibm.py -q` → **13 passed in 18.64 s**（断言的是不变量：核单位分解 <1e-12、
   spread/interpolate 精确伴随 1e-13、节点力 = −标记牵引力 1e-13、静止场零虚假力、Guo 源项零质量、壁面动量交换符号、开边界）。
4. **复现**：以完全相同参数独立重跑 D=16/1200 步 → `cd_mean = 1.6806693327048954`，与工件**逐位一致**（71.4 s）；
   另把交付复制到 `/tmp/m22309_repro` 跑 `benchmarks/cylinder_ibm_xval.py --quick`（D=12，1440 步）→ rc=0、写出 summary。
5. **集成核验**：`PROBES['cylinder_ibm_cd_st'](d_cells=16, n_steps=300, threads=4)` 端到端 → `solver_steps=300, measured=0.287, pass=False`
   （求解器确实推进，T3 harness 不会把它降级为 MOCK）。
6. **独立诊断**：用同一次运行的 Cd/Cl 时程做子窗口分解、FFT 主峰、零点率交叉校验（图 1、图 2 由
   `artifacts/audit_m22345/make_audit_figures.py` 从 `.npz` 程序化生成，零手抄）。

## 3. 计算结果（got vs 基准，逐项）

| 关键量 | got（交付） | 基准 | 相对误差 | 判据 | 结论 |
|---|---|---|---|---|---|
| Cd（D=16） | 1.68067 | 1.35（带 1.32–1.38） | **24.49%**（按带宽 27.32%） | ≤10% / 生态 ≤15% | **FAIL** |
| St（频谱，D=16） | 0.08856 | 0.165（带 0.164–0.166） | **46.33%** | ≤10% | **FAIL / WITHHELD** |
| St（零点交叉，同一次运行） | 0.23704 | 0.165 | 43.66% | — | 三个估计互相矛盾（见 §5） |
| St（审计方 FFT 主峰） | 0.1311 | 0.165 | 20.6% | — | 同上 |
| 标记无滑移残差 | 0.925% of U | 0 | 0.93% | — | 通过（L1 级不变量） |
| 质量漂移 | +3.0e-03 | 0 | — | — | 通过 |
| 控制体动量收支失配 | 100.18% | 0 | — | — | **WITHHELD**（交付已声明不计入验收，审计同意） |
| D=12 复现（--quick） | Cd 1.67926 / St 0.09231 | 同上 | 24.39% / 44.06% | — | 无收敛趋势 |

**细化阶梯只有 1 个点（D=16）**，标记密度研究表为空（`spacing_study=[]`）；summary 自注
“job cj4045 跑全阶梯”，审计时 `cj4045 state=queued, age 18.5 min` —— 全阶梯从未产出。

## 4. 结果图

- `artifacts/audit_m22345/fig1_audit_cd_window_drift.png`：Cd(t)/Cl(t) + 平均窗口起点 + 四个子窗口均值
  （1.8669 / 1.7127 / 1.3307，后半段仍在下降；cd_amp/cd_mean = 0.829）。
- `artifacts/audit_m22345/fig2_audit_error_vs_tolerance.png`：Cd/St 实测 vs 基准，±10%（T3）与 ±15%（生态）带，
  两个分辨率（D=12、D=16）与三个 St 估计。
- 交付侧图（沿用，不改动）：`TensorLBM/outputs/cylinder_ibm/fig_cd_cl_history.png`、`fig_ladder.png`、`fig_vorticity.png`。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **基准裁判在库里不存在**：该 anchor 在 `tensor_benchmark_cases` 无行，机器裁定只能依赖仓内 T3 钉值；
   审计不自行补登裁判（不自造尺子）——登记作为返修项提给独立工单。
2. **“阶梯/密度研究”是叙述而非证据**：报告 §3.1 标题“细化阶梯”、§5 第 4 条“收敛阶梯显示随 D 下降”、
   “标记加密到 0.5 dx 改变有限，说明主导因素是核宽”，对应数据均不存在（ladder 1 点、spacing_study 空、cj4045 未出队）。
   **“核宽是主导误差源”是假设，不是本单结果。**
3. **24.49% 不是收敛后的系统偏差**：平均窗口只含 ~2.5 个脱落周期，子窗口均值 1.87→1.71→1.33，摆幅 83% 的均值，
   因此该数字里包含未收敛成分；据此把误差归因于“扩散界面固有偏差”证据不足。
4. **St 三估计互相矛盾**（0.0886 / 0.1311 / 0.2370）：`n_periods=2 < 5`，`wake_periodic=false`，
   St 应记 **WITHHELD**，不宜作为“46.33% 偏差”引用。
5. **主干状态（审计中途变化）**：审计取证时三个本地主干指针（工作树 m22345 @93a5f96、`/opt/mm-mirror/TensorDG.git` main @93a5f96、
   `/root/tensor-ecosystem/TensorDG` @8b8e3ec）都还没有 `immersed.py`/`cylinder.py`，`t3_matrix_run.py --only cylinder-2d-re100 --quick`
   报 **NOT_SUPPORTED**；但平台 PR 台账 `tensor_dev_missions.result.pr_gate` 记录 **PR #621 state=merged @ 2026-11-15T20:04:15Z**，
   即合并落在审计窗口内、本地克隆滞后。审计方**未**在合并后主干上重跑复现（本单禁 pull/push），故“主干可复现”一项记 **not_run**，
   而不是“不可复现”；若 gate 的 grep 打在过期克隆上，机器裁定会给 L2 而非 L3 —— 这是滞后，不是虚标。
6. **适用范围**：静态、刚性、零速标记；不支持柔性/运动体（与 LIFE 一致）。
7. **诚实项（加分不粉饰）**：交付 summary 与报告 §6 明确写 `within_15pct=False`、`within_t3_10pct=False`、
   `<3%: 否`，控制体收支标 WITHHELD —— 未发现 PASS 虚标，虚标仅出现在 §2 的阶梯/密度叙述。

## 6. 误差判定（显式回答）

| 关键量 | 相对误差 | 是否 < 3% | 是否 < 15% |
|---|---|---|---|
| Cd（D=16，交付最细） | 24.49% | **否** | **否** |
| St（频谱口径） | 46.33% | **否** | **否**（且应记 WITHHELD） |
| 标记无滑移残差 | 0.93% | 是 | 是 |
| 核单位分解偏差 | 3.3e-16 | 是 | 是 |

**是否 < 3%：否。** 关键物理量（Cd、St）双双出带；基准未验证 ⇒ L4 不成立。

## 7. 判级与处置

| 级别 | 判定 | 依据 |
|---|---|---|
| L0 | 否 | 有真代码 |
| L1 | 通过 | 13 个不变量测试全绿（核/伴随/力守恒/开边界） |
| L2 | 通过 | `immersed.py` + `cylinder.py` 输入→计算→输出，审计独立重跑结果逐位一致 |
| **L3** | **确认**（合并后主干可复核；审计时本地克隆滞后） | 包公开 API 导出（`__init__.py:28-36`）+ T3 注册探针（`t3_matrix.py:113-131, :447-493`，`solver_steps>0` 故为 REAL/FAIL 而非 MOCK）+ 基准驱动可执行；审计方直接驱动 `PROBES['cylinder_ibm_cd_st']` 得 `solver_steps=300` |
| L4 | **拒绝** | 基准 Cd 24.49%、St 46.33%（>10%/>15%）；无收敛点（阶梯 1 点）；无 DB 裁判行；St 三估计自相矛盾 |
| L5 | 否 | 无第二独立代码/实验对标，且算例是 10% 堵塞通道 vs 无限域基准，未做堵塞修正 |

**处置**：L3 确认 + 返修清单（reopened for L4）：
① 合并后在平稳尾迹上重跑阶梯（D=16/24/32，平均 ≥5 个周期，Cd 收敛）；
② 跑标记密度研究或撤回“核宽主导”表述；
③ 为该 anchor 在 `tensor_benchmark_cases` 登记裁判行；
④ 在合并后主干（PR #621 已 merged）重跑文档里的复现命令并记录 `not_run` 清单。
## 8. 证据包（审计方产出，可机器复核）

| 文件 | 用途 |
|---|---|
| `artifacts/audit_m22345/audited_src/**` | `agent/m22309@ebfeee4` 六个被审计文件的**只读证据快照**（sha256 见 `audited_files_sha256.txt`，非重新交付） |
| `artifacts/audit_m22345/entry_repro.sh` | 审计方实际执行过的四条入口命令 |
| `artifacts/audit_m22345/cd_cl_history_D16.npz` | 独立重跑 D=16/1200 步时程（cd_mean 与工件逐位一致） |
| `artifacts/audit_m22345/quick_repro_D12_{summary,}.json` | `--quick` 复现（D=12，rc=0） |
| `artifacts/audit_m22345/fig1_audit_cd_window_drift.png` / `fig2_audit_error_vs_tolerance.png` | §4 两图 |
| `artifacts/audit_m22345/verdict_evidence.json` | 落表 evidence 原文 |
| 平台报告 | id 20114（本审计首版）+ 修订版（合并状态更正） |
