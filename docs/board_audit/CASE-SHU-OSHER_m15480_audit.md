# 交差核验单：m15480 · CASE-SHU-OSHER-DG（TensorDG）· 审计任务 m15550

**对象** = mission **15480**，worktree `/root/tp-agent/work/m15480`，交付 commit `537c9ad`
"library-owned case entry, committed run artifacts, live repro + evidence manifest"。
**性质** = 交差核验（非重做）。基准裁判口径先定后测：`<5%` 相对 L1（密度，DG 单元平均 vs 独立参考）。
**审计位置** = `/root/tp-agent/work/m15550`（本审计自带数据副本 `docs/board_audit/data/`，404 KB，
byte-copy 自被审 worktree 的 `runs/shu_osher_case_pool/`，使复核可脱离被审树重跑）。
**审计代码** = `docs/board_audit/audit_shu_osher_m15550.py`（**不 import** 交付件的度量助手
`benchmarks.shu_osher_case_pool.coarsen/relative_errors` 与 `tensordg.cases.shock_entropy.relative_l1_error`，
自写体积平均 + 相对 L1，避免"用自己的尺子量自己"）。
**审计产物** = `audit_shu_osher_m15550.json`、`fig_profiles_and_gate.png`。

---

## 1. 三件套取证（逐项）

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 文件真存在 | **PASS** | `runs/shu_osher_case_pool/raw/` 14 个 npz + `metrics.json` + `profiles.npz`；`artifacts.sha256.json` **16/16 数据工件 sha256 全对**（0 mismatch 0 missing）；3 张 png 在 `docs/figures/` 与 `validation/repro_shu_osher/artifacts/` **md5 完全一致**（098c5af0 / bcc95a4c / 7a3247e4） |
| benchmark ref vs got 数值 | **PASS（审计独立重算）** | 9 条 DG 阶梯由审计代码从 raw npz 重算，与 `metrics.json` 表值 **drift 0.0e+00 pp（9/9）**，区间 **0.556 %–2.462 %**，全部 `<5 %`；最差 P1 N=200 2.4615 %，最佳 P2 N=800 **0.5561 %**（亦 `<3 %`）。CFL 支线（P2 N=400：cfl 0.1→1.5168 %，0.2→1.5153 %，0.4→1.5136 %）同样 `<5 %`。所有配置 `rho_min > 0`（0.8000–0.8009，无负密度） |
| repro_cmd 可执行 | **PASS（脚本层）/ 本地未跑完（如实）** | `validation/repro_shu_osher/{run_all.sh,smoke_repro.py}` 存在且入口清晰；`smoke_repro.py` 与 `docker` 无关、无网络依赖。**审计本地在 120 s harness 上限内未跑完**（宿主 128 核被并行 pytest 占满，同刻多条 pytest 各占 800 %+ CPU）——是环境上限，不是脚本失败；已递交集群作业 **cj1158**（stage_dir=被审树，smoke + 三支 pytest）作为可执行性证明，交差时该作业仍在排队（前面 353 个作业） |
| 报告六段齐全 | **PASS** | `tensor_reports` id=14555（mission 15480）含 ①案例介绍 ②怎么做的 ③计算结果 ④结果图 ⑤注意点 ⑥误差判定，`figures` 3 张内嵌 |

## 2. 基准裁判（唯一裁判）核对结果

任务指定判据 `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-SHU-OSHER'`
→ **0 行**。同表对 `%SHU%/%OSHER%/%entropy%` 全模糊匹配 → **仍 0 行**（该表现有 17 行，TensorDG 仅 1 行 = `CASE-EULER-VORTEX`）。
平台侧最接近的登记在 `tensor_case_pool`：

| id | case_code | repo | tol_policy | status | mission_id | linked_anchor |
|---|---|---|---|---|---|---|
| 25 | SHU-OSHER | TensorFVM | `<5%` | dispatched | 15262 | 空 |
| 83 | SHU-OSHER-DG | **TensorDG** | `<5%` | **dispatched** | **15297** | **空** |

交付件自带的平台自检 `validation/repro_shu_osher/artifacts/platform_bench.json` 也如实记录了这一点：
`l4_verdict.status = CLAIMED_NOT_PROVEN`，`rows = 0`，失败项 `execution_success / source_cluster_job /
cluster_job_bound / ref_got_error_present / within_tolerance`，理由"no tensorbench_results row …
the L4 benchmark claim is self-declared, not measured"。

**判定口径**：平台登记裁判行**不存在**，因此 `<5%` 这条线在库内/案池里是**先声明**的（案池 83 的 `tol_policy`，
以及仓库 `benchmark_ref_vs_got.json`），但**没有平台侧可机器复核的裁判行**。本审计以"审计独立重算"顶替裁判：
got 数值真实、在容差内、无虚标。**这改变的是"确认路径"，不改变数值结论**。

## 3. 判级

**L4 — 基准验证（confirmed on substance）**：完整算例（几何/初值/边界/终时全给定）+ 误差分析（9 配置 ×
L1/L2/L∞ + 参考解自收敛 + RH 闭式锚 + 质量预算）+ 可复现（raw 工件 sha256 可查、repro 脚本存在、审计独立重算 0 漂移）。
不判 **L5**：无文献表值/实验数据对标（Shu & Osher 1988/1989 该题只给曲线图，交付件**诚实声明 WITHHELD**，
未抄数），且方法为既有 DG+WENO 限制器，不是新方法。

## 4. 发现（按严重度）

**F1｜阻断平台级确认（不阻断物理结论）**：`tensor_benchmark_cases` 无本锚点行 → 平台 L4 端点是
`CLAIMED_NOT_PROVEN`；`tensorbench_results` 无行、无集群作业绑定（run 在控制面 99 上跑）。修复=登记裁判行
（anchor_slug 需统一：交付件用 `CASE-SHU-OSHER-DG`，任务口径用 `CASE-SHU-OSHER`）+ 一次 `run_cluster` 绑定。
**F2｜记账缺口**：案池 83（SHU-OSHER-DG/TensorDG）仍 `status='dispatched'`、`mission_id=15297`、`linked_anchor` 空——
m15480 交付后未回写自己的案池行，结案状态与交付事实脱节。
**F3｜图注错位（文档级）**：平台报告 id=14555 的 `figures` 数组把图名与图注错配——
`shu_osher_case_pool_cfl.png` 挂的是"t=1.8 密度剖面"注（内容属 profiles），`..._profiles.png` 挂的是
"CFL 敏感性"注。同名文件在仓库内 md5 一致，故是**发布侧图注旋转**，读者会误标图。
**F4｜图内标题与数据矛盾（文档级）**：`plot_cfl` 的标题写 "CFL sensitivity: DG P2, N=800"，而该图实际扫描的是
`metrics.json → cfl_sweep` = **N=400, P2**（三条腿 400/2/0.1|0.2|0.4）；仓库 doc 的图注写 N=400 才是对的。
**F5｜文档瑕疵（文档级）**：报告 ③ 的表格只有 `|---|…|` 分隔行、**无表头**（12 列无列名，读者无法映射字段）；
⑤ 引用"见正文 ②b"，但全文无 ②b 小节（悬空引用）。
**F6｜口径提示（非缺陷）**：`<5 %` 属**跨格式代码验证**（DG vs 独立 FV-WENO5），不是实验/文献验证；
参考解自身 1600 vs 3200 收敛 0.1010 %（审计重算，与表值一致），误差地板比判据小两个量级，判据可信。
**F7｜已报未门控（信息项）**：P3 各腿质量误差（0.0012 %/0.0189 %/**0.3019 %** of M0）如实入表但未设门；
`conservation_drift` 在本算例按设计为 NaN（透射边界），以质量预算锚替代——无造假期望值迹象。
**F8｜平台记账（非本交付缺陷）**：同一交付已有 m15552 的 verdict（id=337, level 4, gate pending），
本单 m15550 是**第二次**交差核验 → 存在重复派单/重复审计，建议平台去重（本 verdict 用不同 scope 串落表，不覆盖旧行）。

## 5. 独立复核的具体数字（审计自算，非转抄）

* RH 反算（由左态反推，不看字面值）：激波速度 **3.5496481**、激波 Mach **3.0000002**、
  `rho_L` 相对残差 **1.0e-08**、`u_L` **0.0**、`p_L` **1.7e-07** → 左态确为 Mach-3 打入 ρ=p=1 的 RH 后状态；
  解析激波位置 `x_s = -4 + s·1.8 = 2.389367`。
* 质量预算：`M0 = 12.833811`、面通量 `ρu = 10.141852`、`M(1.8) = 31.089145`。
* 参考解贴锚：安静区板值偏差 **0.0 %**、未扰动正弦区 `1+0.2sin(5x)` 偏差 **0.0 %**。
* 阶梯：见 `audit_shu_osher_m15550.json → legs[]`（含每腿 `drift_pp`、`rho_min`、质量误差）。

## 6. 处置（disposition）

**approve with notes**：数值与可复现性成立，L4 在实体上确认；**F1/F2 需返修平台侧登记**，
F3–F5 为文档级小修。平台上 `maturity_gate` 若以 `tensor_benchmark_cases` / `tensorbench_results` 为机器判据，
则本条 L4 预期被判 **reopened（缺裁判行/缺集群绑定）**——这是 F1 的正确表现形态，**不是**数值造假。

**未做/未验证项（state unknown, not passed）**：`smoke_repro.py` 全流程未在审计本地跑完（120 s 上限），
`run_all.sh` 全量 ~30 min 未跑（作业 cj1158 排队中）；文献表值对标**不存在**（WITHHELD）。