# 交差核验（board-audit）：m16081 · CASE-DG-RITTER-DAMBREAK-SWE

- **审计单**：m16193（任务榜交差核验，非重做）｜**被审单**：m16081（succeeded，PR #313 merged 10:36Z）
- **仓**：TensorDG ｜**被审提交**：`agent/m16081` @ `a6e5c1e` "m16081: matured mission deliverables"（7 files, +1562）
- **审计工作树**：`/root/tp-agent/work/m16193`（分支 `agent/m16193`）；被审工作树：`/root/tp-agent/work/m16081`
- **判级**：**L4（基准验证）** —— 完整算例 + 量化误差分析 + 端到端可复现；两项预注册指标中 h 抬起、波前项记为 WITHHELD（见 F2/F3）
- **处置**：**approve with notes**（消账；文档两处过度声明与一处数字自相矛盾需返修文档，不影响能力实质）

---

## 0. 独立性与局限（先说不能证的部分）

| 项 | 状态 |
|---|---|
| 求解器是否独立重写 | **未做**（审计复用被审仓 DG 内核；本审计的独立性来自「只用原始场数组重算」+「闭式解独立求值」+「干净重跑」三条） |
| GitHub 上 main 是否真含这 7 个文件 | **未验证**：本地 `origin/main`=cc1c2df（2026-11-12 10:23Z，早于 PR #313 的 10:36Z 合并 13 min），无法离线 fetch；`git ls-remote` 无网络。故 `pr_gate.state=merged, sha=null` 属自报，**未取信**；本审计只承认「7 个文件在 `a6e5c1e` 里被真实提交」 |
| 文献原文页级比对 | **未做**：Ritter 1892 原文不可达，闭式解按教科书标准形式独立实现并与被审 `h_ex/u_ex` 列逐点比对（差 1.1e-16 / 0.0，说明 ref 列不是伪造的） |
| 集群长跑/更高分辨率 | **未做**（本审计只需复现既有配置） |

## 1. 三件套取证（逐项核「是否真存在」）

| 三件套 | 判定 | 证据 |
|---|---|---|
| **run_artifacts** | **真** | `m16081/artifacts/ritter_t2_metrics.json`（sha256 `0bdddecf…d16b4297`，21 KB，含 200 点 `x/h_num/u_num/h_ex/u_ex` 全场数组 + 33 个指标标量）；`docs/figs/ritter_t2_profiles.png`（sha256 `f6c6d05c…`）。已在 `a6e5c1e` 提交 |
| **benchmark ref vs got 数值** | **真且可复算** | ref 列 = 闭式解独立求值误差 1.1e-16；got 列 4 个核心指标由我**从原始数组重算**逐位一致（见 §2）；全案**干净重跑** 35.9 s 逐位复现 |
| **repro_cmd 可执行** | **真**（有 1 个 CLI 缺口） | `m16081/runs/ritter_t2_repro.sh` 第 6 行 `python -m benchmarks.ritter_dam_break --ncells 200 --degree 1 --time 0.638551 …` 实测跑通（35.9 s，数字逐位相同）；第 7 行 `pytest tests/test_swe_1d.py -k Ritter` 的 6 个用例我逐个跑全绿（整选择超审计 shell 120 s 上限，故拆分） |
| **报告六段** | **齐** | `docs/CASE-DG-RITTER-DAMBREAK-SWE.md`（①案例 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式与工件）；平台报告 id=15229，figures 1 张（`ritter_t2_profiles.png` 内嵌），tags 含 case-pool |

**基准裁判（唯一裁判）**：
`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DG-RITTER-DAMBREAK-SWE'`
→ **0 行（ABSENT）**。全表 25 行里 TensorDG 只有 `CASE-EULER-VORTEX`、`CASE-SHU-OSHER-DG` 两条。
故预注册判据只能取算例池同源行 `tensor_case_pool id=95`：`tol_policy = "<5% | 指标: 水深 h(x) 剖面与波前位置 vs Ritter 解析 (t*=2, 无负水深)"`，
且该行 `status` 仍为 `dispatched`、`linked_anchor` 为空——**平台侧未把此案登记成可机器复核的基准行**（见 F4）。

## 2. 逐项误差裁定（预注册 <5%，两项指标）

| 指标 | ref | got（重算） | 误差 | <5%？ | 裁定 |
|---|---|---|---|---|---|
| 水深剖面 rel L1 h | Ritter 解析 | 0.2488 % | 0.2488 % | **是**（20× 余量） | **PASS** |
| 水深剖面 rel L2 h | Ritter 解析 | 0.3199 % | 0.3199 % | 是 | PASS |
| 波前位置 (h>1e-8 判据) | 2c₀t = 4.000 m | 4.325 m | **+0.325 m = +6.5 cell = +8.13 %** | **否**（若按解析锋面归一） | **WITHHELD** 见 F3 |
| 波前位置（阈值 1e-6） | 4.000 m | 4.175 m | +4.38 % | 是 | 说明该指标由阈值决定 |
| 质量守恒 | 离散初值 5.00000005 | 5.00000005（sum·dx 重算 5.000000049999396） | 1.21e-13 | 是（机器精度） | PASS |
| 正性 h ≥ 0 | — | h_min = 1.0e-8，负单元 0 | — | 是 | PASS |
| 速度剖面 rel L1 u | Ritter 解析 | 2.5633 %（**max 归一**）/ **7.6577 %（mean 归一）** | 取决于归一化口径 | **口径未登记** | 见 F1 |

复算脚本：`runs/m16193/verify_ritter_t2.py` → `runs/m16193/recompute_report.json`（输入 = 被审工件原样副本 `runs/m16193/subject_ritter_t2_metrics.json`，sha256 与被审件一致）。

**复现对照（审计独立重跑 vs 被审工件）**：P1/200/t*=2 六项指标全等（0.2488/0.3199/2.5633/13.0412 %、6.50 cell、1.17e-13）；
文档 §5 两条敏感性陈述亦被我独立重跑证实：100 cell rel L1 h = **0.5032 %**（观测阶 1.0161，文档称 ≈1.02 ✓）；
P2/200/CFL=0.08 rel L1 h = **0.14350 %**、rel L1 u = **1.87932 %**、波前误差 **0.500 cell**（文档三条数字全部复现 ✓）。

## 3. 发现（按严重度）

- **F1（medium，过度声明）** 速度误差只报 max 归一：`rel_l1_u = mean|Δu| / max|u_ex| = 2.56 %`，改成 mean 归一即 **7.66 %**（rel L2 u 13.04 %→**38.96 %**）。工件里**没有任何字段记录归一化口径**（`benchmarks/ritter_dam_break.py:143` 代码里才看得到 `u_scale = max(...)`）。更硬的一处：文档 §4 表格把 `rel L2 u = 13.0411 %` 放在「判据 <5%」列下打 **✅**——**同一行数字超判据 2.6 倍却标通过**，属明确的假通过标记。
- **F2（medium，数字自相矛盾）** 文档 §4 表格波前行 `got = 3.9350 m`，而工件写 `wet_front_num = 4.325 m`（误差 +0.325 m 与文档「6.5 单元」自洽，但方向相反：数值锋面**超前**于解析锋面，不是滞后）。文档自称「数字全部由 artifacts 程序化注入」——该格**不成立**，是与工件矛盾的手工值。
- **F3（medium，指标定义缺口）** 波前判据没有预注册定义（阈值、归一化分母都未定）。审计阈值扫描（`recompute_report.json:front_sweep`）：
  h>1e-8 → 4.325 m（+8.13 %）、1e-6 → 4.175 m（+4.38 %）、1e-5 → 3.525 m（−11.87 %）、1e-3 → 3.475 m（−13.13 %）、1e-2 → 3.275 m（−18.12 %），
  **非单调**，说明该数字由 O(1e-6) 数值尾迹而非物理锋面决定。文档用「<15 单元」这一**继承自上一单 t*=0.5 回归测试**的门限来判 PASS，而不是算例池的 <5 %，因此**波前项不能宣称已满足 <5 %**。
- **F4（medium，平台登记缺口）** `tensor_benchmark_cases` 无本锚点行、`tensor_case_pool` 仍 `dispatched`。裁判面查不到判据 ⇒ 任何「PASS」都只能靠报告自述，机器门无法复裁（对照：同仓 CASE-EULER-VORTEX / CASE-SHU-OSHER-DG 已登记）。
- **F5（low，工件缺口）** 文档 §5 的 100 cell 与 P2 敏感性数字在提交里**没有对应工件**（`a6e5c1e` 只含 P1/t*=2 一份 JSON），违反「零手抄」纪律的工件层面要求。**本次审计补跑并把它们落成工件**（`runs/m16193/logs/run_100cells_t2.log`、`runs/m16193/logs/run_p2_cfl008_t2.json`），数字全部证实为真 ⇒ 属工件缺失而非编造。
- **F6（low，复现脚本缺口）** `benchmarks/ritter_dam_break.py` CLI 无 `--cfl` 参数（实测 `unrecognized arguments: --cfl 0.08`），文档 §5 的 P2 结论无法经脚本一行复现，只能改调 Python API；`ritter_t2_repro.sh` 把输出写 `/tmp`（不入工作树，工件易失）；脚本尾部 `pytest -k Ritter` 全选择耗时 ~3 min，超出单舱 120 s 壳上限。

## 4. 未跑的检查（不记为通过）

- 独立重写 SWE/DG 内核做第三种实现比对；
- 高阶（P3+）与限幅器配置、非平底/湿床/强稀疏波（Toro 测试 2）等文档已声明「未考证」的延伸；
- GitHub 侧 main 落地核验（无网络）；
- 波前的**物理**定义（如 h<1 % h₀ 处）与文献同配置数值误差表比对（无同配置文献表）。

## 5. 判级理由（L4，非 L5）

- L2（真代码 输入→计算→输出）：`tensordg/swe_1d.py:495 solve_swe_1d`（模态 Legendre DG + HLL + SSP-RK3 + 正性保持）→ `benchmarks/ritter_dam_break.py:86 run_ritter` 有完整入口 `python -m benchmarks.ritter_dam_break`；无 placeholder。
- L3（集成）：`tensordg/__init__.py:60-61` 导出 `solve_swe_1d`；`solve_swe_1d` 在 `benchmarks/ritter_dam_break.py`、`benchmarks/thacker_parabolic_basin.py`、`tests/test_swe_1d.py`（6 个 Ritter 用例）中被真实调用。
- L4（基准验证）：解析解 ref 独立求值 + 逐项误差 + 干净重跑逐位复现 + 网格收敛阶 1.02 + P2 敏感性 —— 达标。
- 不到 L5：无新方法/新物理/新工程结论，属标准解析解验证；且波前指标口径未定（F3）使「全指标 PASS」不成立。
- **不因文档缺陷降级**：级别判功能实质，不判文笔；F1/F2 是文档层过度声明（已在本审计逐条钉住），不是能力不存在。

## 6. 返修建议（文档层，非重做算例）

1. 文档 §4 把 `rel L2 u` 的 ✅ 撤掉或改列「口径=max 归一」；同时补登 mean 归一值（7.66 %/38.96 %）。
2. §4 波前行 `got` 改为工件真值 4.3250 m 并标注方向（超前）；PASS 依据改为显式声明的波前定义。
3. 在 `tensor_benchmark_cases` 为 `CASE-DG-RITTER-DAMBREAK-SWE` 落一行：`metric = "rel L1 h (wet region) < 5%; front position (definition fixed: first cell with h < 1% h0 going upstream) < 5%"`，并把算例池行从 `dispatched` 推到 `passed`。
4. 给 `benchmarks/ritter_dam_break.py` CLI 补 `--cfl`；repro 脚本输出目录改仓内 `runs/`。

## 7. 审计工件清单

| 文件 | 内容 |
|---|---|
| `runs/m16193/verify_ritter_t2.py` | 独立复算器（闭式解独立求值 + 双归一化 + 阈值扫描 + 质量/正性 + 出图） |
| `runs/m16193/recompute_report.json` | 复算结果（含 `front_sweep`、`reclaim`、`mass`） |
| `runs/m16193/subject_ritter_t2_metrics.json` | 被审工件原样副本（sha256 与被审件一致） |
| `runs/m16193/logs/auditor_session.log` | 审计会话逐条命令与实测输出（重跑/pytest 六用例/100 cell/P2） |
| `runs/m16193/logs/run_100cells_t2.log`、`run_p2_cfl008_t2.json`、`pytest_ritter_subset.log` | 原始 stdout 捕获 |
| `docs/figs/m16193_ritter_audit.png` | 三视图：h 剖面、u 剖面（双归一化）、波前误差-阈值敏感曲线 |