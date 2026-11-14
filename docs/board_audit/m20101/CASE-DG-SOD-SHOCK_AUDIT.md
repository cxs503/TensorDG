# 交差审计 · CASE-DG-SOD-SHOCK / m20005（TensorDG）— m20101

**审计对象** mission `m20005`（repo=TensorDG, status=succeeded, headline「Sod 1978 密度峰位/平台特征级验证，最大误差 0.753%（判据 <4%）PASS」）
**审计类型** 交差核验（非重做）｜**审计员** board-audit / m20101｜**日期** 2026-11-14
**裁定** **L4（基准验证）· approve with notes**（confirmed）

---

## 0. 身份绑定（本次审计实际评测的字节）

subject worktree `/root/tp-agent/work/m20005`，git HEAD `e18b8acc0323de55a166f0075c87e8a608cba09f`。

| 文件 | 字节 | sha256(前 16) |
|---|---|---|
| `benchmarks/case_sod_peak_plateau.py` | 25122 | `cfa9953b91384ecc` |
| `benchmarks/riemann_exact.py` | 5726 | `51c3bd8c97cc6090` |
| `tests/test_case_sod_peak_plateau.py` | 4016 | `d53719b94030697d` |
| `docs/CASE-DG-SOD-SHOCK.md` | 15182 | `32d7525d45dc4e24` |
| `tensordg/euler_dg_1d.py`（被复用的求解器） | 19683 | `4378b85c682c06cd` |
| `tensordg/euler_1d.py`（P0 有限体积极限） | 15115 | `3b02bc8b308cf2ec` |
| `artifacts/case_sod_peak_plateau/metrics.json` | 168221 | `64de5e05143659fb` |
| `artifacts/case_sod_peak_plateau/features.csv` | 29802 | `07ef832955824c8a` |
| `artifacts/case_sod_peak_plateau/profiles.csv` | 5736 | `6bd59d5b24f335af` |
| `artifacts/.../density_peak_plateau.png` | 70473 | `129d7aa3cc1ad983` |
| `artifacts/.../feature_convergence.png` | 48701 | `a05c960e5c4f0a43` |
| `artifacts/.../sensitivity.png` | 46323 | `ce718c10e675184d` |

12/12 工件存在且非空；完整清单（含 sha256 全串）见 `runs/m20101/audit_metrics.json → artifact_identity`。

**审计产物**：`runs/m20101/verify_sod_shock_audit.py`（独立复核器，1 次运行 ~35 s）、
`runs/m20101/audit_metrics.json`（全部原始数字）、`docs/figs/m20101_sod_audit.png`（本审计自绘图，
不是被审方的图）。

---

## 1. 取证：三件套是否真的存在

| 项 | 判据 | 结果 |
|---|---|---|
| **run_artifacts** 文件 | 工件目录存在且非空 | ✅ `artifacts/case_sod_peak_plateau/` 6 个文件（metrics.json / features.csv / profiles.csv / 3×png），12/12 核验文件存在 |
| **benchmark ref vs got 数值** | 工件内逐项 ref/got/rel_err 表 | ✅ `metrics.json.ladder[*].table` 13 个配置 × 13 行（7 门禁 + 6 旁证），每行有 ref/got/abs_err/rel_err_pct/pass/normalisation |
| **repro_cmd 可执行** | 命令能跑出同一批数字 | ✅（门禁配置 + pytest 端到端复跑，见 §3）；⚠️ `--profile full` 全量 26 配置本机超 10 min 上限，未由审计员端到端重跑（见 §6） |
| **报告六段齐全** | ①案例 ②方法 ③参数表 ④对照表 ⑤收敛/敏感 ⑥注意点/失败模式（+⑦误差判定） | ✅ `docs/CASE-DG-SOD-SHOCK.md` 七节全；平台 report id **18265**（markdown 11004 字符、3 图、⑦ 逐项显式回答「是否 <3%」） |

**基准裁判（唯一裁判）核对** ——

```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-DG-SOD-SHOCK';
-- -> 0 rows（全表 121 行，TensorDG 侧仅有 REPRO-DG-CKS1998 / CASE-EULER-VORTEX /
--    CASE-DG-SOLID-BODY-ROTATION-2D / CASE-SHU-OSHER-DG / XVAL-DG-ROUND2-CERTIFIED-REFERENCE）
SELECT id,case_code,repo,tol_policy,tier,linked_anchor,status,mission_id
  FROM tensor_case_pool WHERE case_code ILIKE '%SOD%';
-- -> 198 | DG-SOD-SHOCK | TensorDG | <4% | core | (NULL) | dispatched | 20005
```

结论：**判据表里没有这一锚点的行**，4% 容差登记在 `tensor_case_pool` 的 `tol_policy`（`<4%`，
与代码 `benchmarks/case_sod_peak_plateau.py:65 TOLERANCE = 0.04` 一致），但 `linked_anchor` 为 NULL
→ `tensor_benchmark_cases` 无法机器复核；且 case_pool 状态仍停在 `dispatched`（任务已 succeeded）。
**这不是虚标，是登记缺口**（见 F1）。因此本次审计不依赖该行，改用更强的外部锚点自行复核（§2）。

---

## 2. ref 是不是真 ref：从零独立重推

被审方的 ref 由仓库内 `benchmarks/riemann_exact.py` 程序化计算。审计员在
`runs/m20101/verify_sod_shock_audit.py` 里**另写一套**精确 Riemann 求解器（Toro ψ 函数 + 400 步二分，
不 import 仓库任何模块），重推 Sod 波结构：

| 量 | 审计员独立值 | 被审方 ref | 相对差 |
|---|---|---|---|
| p\* | 0.3031301781 | 0.3031301781 | 1.8e-16 |
| u\* | 0.9274526200 | 0.9274526200 | 0 |
| ρ_L\*（密度峰） | 0.4263194282 | 0.4263194282 | 1.3e-16 |
| ρ_R\*（平台） | 0.2655737117 | 0.2655737117 | 2.1e-16 |
| c_L\* | 0.9977254326 | 0.9977254326 | 0 |
| S（激波速度） | 1.7521557320 | 1.7521557320 | 1.3e-16 |
| x_head / x_tail / x_contact / x_shock | 0.2633568087 / 0.4859454375 / 0.6854905240 / 0.8504311464 | 同 | ≤1e-16 |
| 峰台宽度 / 平台宽度 | 0.1995450865 / 0.1649406224 | 同 | ≤6.7e-16 |

15 个量最大相对差 **6.7e-16** → ref 是解析解本身（Toro 2009 §4.3；Sod 1978 初始数据），
不是拟合/手填/自洽循环。外锚点成立。

---

## 3. 复现：门禁配置端到端重跑

| 检查 | 命令 | 结果 |
|---|---|---|
| 单测 | `python -m pytest tests/test_case_sod_peak_plateau.py -q` | **4 passed in 38.29 s**（2026-11-14，审计员本机 cwd=m20005） |
| 门禁配置重跑 | `run_single(degree=2, ncells=100, cfl=0.15)` | 1517 步（= 工件 steps 1517），31.5 s，`pass=True`，7/7 门禁通过，最大误差 **0.7534 %** |
| 与已提交工件逐位比对 | 13 行 got 值 vs `metrics.json` | 最大绝对差 **3.1e-13** → 工件是真跑出来的（bit-level 复现） |
| 文档 ↔ 工件零手抄 | 解析 `docs/CASE-DG-SOD-SHOCK.md` ④ 表 13×7=**182 个单元格**（got + rel_err%）与 metrics.json 逐项比对 | **0 处不一致** |

门禁配置 7 项（P2/100/CFL 0.15，审计员重跑值）：

| 指标 | ref | got | 相对误差 | 判据 4% | <3% |
|---|---|---|---|---|---|
| 密度峰 ρ_L\* | 0.42631943 | 0.42765168 | 0.3125 % | ✅ | 是 |
| 峰位 x_tail | 0.48594544 | 0.48386739 | 0.2078 % | ✅ | 是 |
| 平台 ρ_R\* | 0.26557371 | 0.26571372 | 0.0527 % | ✅ | 是 |
| 接触面 x_contact | 0.68549052 | 0.68491583 | 0.0575 % | ✅ | 是 |
| 激波 x_shock | 0.85043115 | 0.85010253 | 0.0329 % | ✅ | 是 |
| 平台宽度 | 0.16494062 | 0.16518670 | 0.1492 % | ✅ | 是 |
| 峰台宽度 | 0.19954509 | 0.20104844 | **0.7534 %** | ✅ | 是 |

→ 任务榜 headline 的「7 项特征最大相对误差 0.753%（<4%，且全部 <3%）」**逐字属实**。

---

## 4. 判级（L0–L5）

| 级别 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 解析关系、单测 | ✅ `exact_features` 与文献常数交叉校核（rel<1e-5）、提取器喂精确剖面必须无偏 |
| L2 独立功能实现 | 真代码 输入→计算→输出 | ✅ `benchmarks/case_sod_peak_plateau.py:1-527`（exact_features / pointwise_primitive / measure_density_features / compare_features / run_single / main），无 MOCK/STUB/random（grep 空） |
| L3 系统集成 | 接主求解器/平台 | ✅ 直接调用生产求解器 `tensordg/euler_dg_1d.py:131 solve_euler_dg_1d`（HLLC + SSP-RK3 + Zhang–Shu 正性缩放）与 `tensordg/euler_1d.py::solve_riemann`（P0），未另起沙盒；测试进 CI（`.github/workflows/tests.yml:46` → `pytest -q`） |
| **L4 基准验证** | 完整算例 + 误差分析 + 可复现 | ✅ **达成**：13 配置阶梯（P0–P3 × N=25…1600）× 7 CFL × 6 限制器敏感性；逐项 ref/got 误差表；4% 判据逐项判定 + 3% 显式回答；工件齐全；审计员 bit-level 复现；失败配置（P0/200 平台宽 6.59 %、P3/25 ρ峰 4.70 %）在 ⑥ 逐条自曝未粉饰 |
| L5 科研工程成果 | 方法/科研增量 | ✗ 未达：复用既有 DG 求解器做标准验证，无新数值方法、无外部对标实验数据、无同行评议产出 |

**裁定：L4。** 级别=功能实质，与报告篇幅无关；本单报告 11 k 字不改变它。

---

## 5. 缺陷与发现（按严重度）

### F1 · MEDIUM（不阻断 L4，阻断机器认证）· 判据未进唯一裁判表
`tensor_benchmark_cases` 对本锚点 **0 行**；`tensor_case_pool` id=198 `linked_anchor=NULL`、
`status` 仍为 `dispatched`（mission 已 succeeded）。于是 (a) 4 % 容差的**机器复核通道不存在**，
(b) **7 项门禁指标清单未登记**——容差登记为 `tol_policy='<4%'`，未列指标名，指标集由实现方自定
（本单自定的 7 项比锚点描述的「峰位/平台」更严，方向是诚实的，但机制上可被择优）。
**复现**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DG-SOD-SHOCK'` → 0 rows；
`SELECT * FROM tensor_case_pool WHERE id=198` → `linked_anchor NULL, status dispatched`。
**建议**：回填 benchmark 行（7 指标 + 4 %）+ 置 case_pool `passed` + 填 `linked_anchor`。

### F2 · MEDIUM · 度量可调参数敏感性（`peak_width` 是脆弱量）
提取器有两个声明过的可调量（`TRIM=0.10`、`PEAK_EPS=0.01`）。审计员在**同一** P2/100 剖面上做
4×3 扫描：11/12 变体全部 7 项 <4 %；**唯一越界**为 `trim=0.05, peak_eps=0.02` → `peak_width` **4.53 %**。
`peak_width` 在扫描区间内从 0.14 %（trim .10/eps .005）漂到 4.53 %，跨度 ~30×（它由两次检测相减得
到，误差继承放大）；`trim≥0.10` 的所有变体 ≤2.3 %。→ 结论不脆（声明配置 0.753 %，余量 5.3×），
但 `peak_width` 的读数强依赖度量定义，报告 ④ 表把它与其余 6 项同列等权，读者会低估这一点。
**复现**：`runs/m20101/audit_metrics.json → extractor_robustness`（12 变体逐项误差）。

### F3 · LOW · 测试里有一条无效断言
`tests/test_case_sod_peak_plateau.py:83`：`assert run["max_rel_err_pct"] < 100 * TOLERANCE`，即 **<400 %**。
真正起作用的是上一行 `assert run["pass"]`（4 %）。该行不提供任何额外保护，属冗余而非有害；
若哪天 `pass` 判定逻辑被改坏，这行会给出虚假安全感。

### F4 · LOW · `metrics.json:all_pass=false` 与「PASS」口径相反
`all_pass` 是**梯 13 个配置的 AND**（`case_sod_peak_plateau.py:430`），P0/200 与 P3/25 本就出 ✗，
于是顶层标志为 `false`；而报告/榜单的 PASS 指**预登记门禁配置**。文档 ⑥ 有披露，但顶层字段语义
对下游脚本可读性是误导性的（自动化消费者按 `all_pass` 会判 FAIL）。建议加
`gate_pass` 字段与 `all_pass` 并列。

### F5 · LOW · 报告 ⑦ 表漏列误差最大的那一项
⑦「是否 <3%」表只列 6 项，缺 `峰台宽度`（正是 0.753 % 这一项，也是报告正文引用的「最大误差」）。
结论方向仍正确（0.753 %<3 %），但表与正文口径不一致，易被读成「最大误差指标未列入表」。

### F6 · INFO · 无法从工件证伪的过程性声明
「预登记判据配置 P2/100/CFL 0.15」的**先验性**无法由工件证明。正面证据：阶梯上 P1/N=400 更好
（0.450 %），报告明确拒绝事后反选；负面可能：P2/100 恰是 DG 文献常用配置，也可能是事后择优。
审计员对其**不背书也不指控**。

### 正向确认（非缺陷）
- 文档数字与工件 **182/182 单元格零手抄**（§3）。
- 未发现 MOCK/STUB：`grep -i "mock|stub|TODO|np.random"` 在算例/ref/求解器三个文件均空。
- 失败配置自曝：⑥ 列 P0 粗网格（平台宽 6.59 %、峰台宽 11.59 %）与 P3/25（ρ 峰 4.70 %）的 ✗，
  与 `metrics.json` 一致（审计员复核 ladder 13 行 pass 计数 5–7/13，与文档 ✗ 位置一致）。

---

## 6. 未做的检查（unknown ≠ passed）

1. `--profile full`（26 配置全量）**未由审计员端到端重跑**：本机重计算有 <10 min 硬拦，
   任务声明的集群作业 `cj2436` 记录未能从 `compute_ledger_mission_v` 取回（查询超时 120 s 被收割）。
   → 已核的是门禁配置（bit-level）+ pytest；阶梯其余 12 配置与 CFL/限制器敏感性**未逐条重跑**，
   但其代码路径与门禁配置相同（同一 `run_single`），工件内部数字经 182/182 零手抄校验。
2. **未**独立重写 DG 求解器内核（`euler_dg_1d.py` 的 HLLC/SSP-RK3/正性缩放只做了静态阅读 +
   在本算例上的行为验证；其自身正确性由其它锚点 REPRO-DG-CKS1998 等背书，不在本单范围）。
3. **未**验证 `profiles.csv` 全量剖面与 png 的逐像素一致（只核文件非空 + 大小）。
4. **未**与该算例的第三方文献数值表（如 Sod 1978 原文表格）逐格比对——本单外锚点是解析解，
   比表格更硬，故未做。
5. 无网络：GitHub 侧分支/PR 状态未核。

---

## 7. 处置

**Disposition: approve with notes（L4 / confirmed）**。
主张全部属实（0.753 % 最大误差、7/7 通过、<3 % 逐项达成、182 格零手抄、bit-level 可复现），
无虚标、无粉饰；不足在于**平台侧登记**（F1 判据未进唯一裁判表、case_pool 状态滞留）与
**度量定义的可调敏感性披露不足**（F2/F5），以及两处低危实现瑕疵（F3/F4）。
建议返修不需要重跑科学计算，只需：回填 benchmark 行 + 翻转 case_pool 状态（F1）；
在 ④ 表为 `peak_width` 加脚注并补 ⑦ 第 7 行（F2/F5）；`all_pass` 旁增 `gate_pass`（F4）。
---

## 8. 产物与裁定落账

| 项 | 值 |
|---|---|
| verdict 行 | `tensor_maturity_verdicts` **id=832**（`TensorDG` / `m20005 CASE-DG-SOD-SHOCK` / `mission_id=20101` / **level=4** / `verdict_by='board-audit'` / `gate_state=pending`） |
| 机器预检 | 直接 import 平台守护 `maturity_gate.judge(4, evidence, 'TensorDG', 20101)` → **`(4, [])`**（零缺项）；`callers_verified(...)` → `True`。按 `maturity_gate.py:317` 分支，`gl>=2` → 下一轮（周期 30 min）应置 `gate_state='confirmed'` 并 `ingest_verdict` 写 knowledge_base |
| 平台报告 | `tensor_reports` **id=18318**（六要素 + 2 图，数字全部由 `runs/m20101/audit_metrics.json` 程序化注入） |
| 审计产物 | `docs/board_audit/m20101/CASE-DG-SOD-SHOCK_AUDIT.md`、`runs/m20101/verify_sod_shock_audit.py`、`runs/m20101/audit_metrics.json`、`runs/m20101/publish_audit_report.py`、`docs/figs/m20101_sod_audit.png`、`docs/figs/m20101_sod_robustness.png` |

**自主性限制（independence limitations）**：审计员与被审单同处一台宿主、同一 Python/numpy 环境；
「独立」体现在**不 import 被审方的 ref 模块、不采信其报告文字、重新推导解析解并重跑门禁配置**，
而非环境隔离。被审方的 `--profile full` 全量作业（自报集群 `cj2436`）未由审计员重跑，
故阶梯中除门禁配置外的 12 个配置只经「工件内部一致性 + 代码路径同一」两级旁证，未经独立重算。
