# 交差核验审计：m16200 · CASE-DG-SOLID-BODY-ROTATION-2D（TensorDG）

- **审计单**：m16228（board-audit），工作树 `/root/tp-agent/work/m16228`
- **被审单**：m16200（TensorDG，状态 `succeeded`，`finished_at=2026-11-12T11:10:04Z`，model=qwen3.6-plus，73 tool calls）
- **锚点**：`CASE-DG-SOLID-BODY-ROTATION-2D`｜ref LeVeque 2002 *Finite Volume Methods for Hyperbolic Problems* Ch.19, DOI 10.1017/cbo9780511791253
- **声明判据**（m16200 goal 原文）：`容差: <5% | 指标: 旋转一周后 L1 误差与极值衰减 vs 初值 (CFL=0.5, 网格 100x100)`
- **本单性质**：交差核验，非重做。审计侧只增不改（未修改被审任何文件），新增文件全部落在 m16228 工作树。

---

## 0. 一句话结论

被审单的**代码是真的、可跑、可复现、非硬编码**（审计侧独立复现 7/7 指标逐位相同；新配置产生工件里没有的新数值；既有 14 条解算器测试在改动后全绿），
但**被声明的验收算例（100×100, CFL=0.5, L1<5%）从未执行过**：没有 n=100 工件、3 条专门断言该判据的测试全部 `pytest.skip`、集群作业 cj1265 至今 `queued` 未派发；
**平台基准表里也没有本锚点的登记行**（`tensor_benchmark_cases` 0 行），且**三个交付文件在被审工作树里全部未入 git**（`git ls-files` 无命中）。
⇒ 判定 **L3**（独立功能实现 + 接入本库主解算器类并驱动端到端；L1/L2/L3 机器可核证据齐），**L4 不成立**（基准验证缺最后一环）。

---

## 1. 取证：三件套逐项核验

被审工作树 `/root/tp-agent/work/m16200`（branch `agent/m16200`）。

| 项 | 声明 | 核验结果 |
|---|---|---|
| impl_files | `tensordg/advection_2d_var.py`（+77 行：`l1_error` + 3 个 profile）| **真存在，非占位**。`l1_error` @ 行 403（Gauss 积分 L1/L2 范数，含可选 disk 掩膜）；`cone_profile` @158、`slotted_cylinder` @172、`smooth_bump` @187。sha256 `bdc6724e55bed1ea8c11529ee2a663c086dbf85dfc9c11c7b8bf6439bfde2905` |
| impl_files | `benchmarks/case_dg_solid_body_rotation_2d.py`（199 行）| **真存在**。`run_case` @54、`main` @134、`TOLERANCE=0.05` @51、`DISK_RADIUS=0.28` @48。sha256 `038e989c…ccd5f7` |
| tests | `tests/test_case_dg_solid_body_rotation_2d.py`（97 行，7 条）| **真存在但 3/7 恒 skip**（见 F4）。sha256 `376d5842…97e7c37` |
| run_artifacts | 6 个 JSON 工件 | **存在**（P1 n=16/25/32，P2 n=16/25，+ 主工件 `case_dg_solid_body_rotation_2d.json`= P2 n=25 副本）。**缺 `…_p2_n100.json`** |
| benchmark ref vs got | 声明 L1<5% @100×100 | **ref 侧无数值基准、got 侧无 100×100 值**；只有 n≤32 的 got 值（见 §3 表）。登记表 `tensor_benchmark_cases WHERE anchor_slug='CASE-DG-SOLID-BODY-ROTATION-2D'` → **0 行**（全表 29 行；TensorDG 只有 CASE-EULER-VORTEX / CASE-SHU-OSHER-DG 两行） |
| repro_cmd | 报告 §5.2 | **可执行且可复现**（审计侧照抄执行，7/7 指标逐位相同，见 §2）；前提是命令里那段 `PYTHONPATH=<worktree>` |

**git 落地核验（关键）**：
```
$ git -C /root/tp-agent/work/m16200 ls-files | grep solid_body   # → 无命中（仅命中一个历史 png）
$ git -C /root/tp-agent/work/m16228 log --all --oneline -1 -- benchmarks/case_dg_solid_body_rotation_2d.py  # → 空
$ git -C /root/tp-agent/work/m16200 status --short
 M tensordg/advection_2d_var.py
?? artifacts/case_dg_solid_body_rotation_2d*.json (×6)
?? benchmarks/case_dg_solid_body_rotation_2d.py
?? docs/case_dg_solid_body_rotation_2d_report.md
?? tests/test_case_dg_solid_body_rotation_2d.py
```
m16200 的 `result.evidence.head_commit = null`，`git_status` 里就是上面这串 `??`。**即：succeeded 是「dev_acceptance 自动批准」的结果，不等于交付入仓。**

## 2. 审计侧独立复现（不是读报告）

1. **复现被审算例 P2 n=25**（报告 §5.2 原命令，只把输出改到 /tmp）：
   `cd /root/tp-agent/work/m16200 && PYTHONPATH=$PWD python3 benchmarks/case_dg_solid_body_rotation_2d.py --degree 2 --ncells 25 --cfl 0.5 --output /tmp/audit_repro_p2n25.json`（25.4 s，1506 步）
   → `steps / dt / l1_relative / l2_relative / peak_retention_avg / peak_retention_nodal / mass_drift_rel` **7/7 与被审工件逐位相同**（1.6467948585547286e-03 等）。工件见 `runs/m16228/auditor_repro_p2n25.json`。
2. **反硬编码探针（新配置，工件里不存在）**：`--degree 1 --ncells 20`（8.2 s，723 步）→ `l1_relative = 9.162989994671315e-02`，`peak_retention_nodal = 0.9087`，判据 FAIL。该数值不在任何被审工件中 ⇒ 数值确由解算器现算，**未发现硬编码/回填**。工件 `runs/m16228/auditor_novel_p1n20.json`。
3. **非回归**：以被审的改动后模块跑既有解算器测试
   `PYTHONPATH=/root/tp-agent/work/m16200 python3 -c "import tensordg.advection_2d_var as m; print('MODULE',m.__file__,hasattr(m.DGAdvection2D,'l1_error')); ... pytest.main(['tests/test_advection_2d_var.py','-q'])"`
   → 进程内打印 `MODULE /root/tp-agent/work/m16200/tensordg/advection_2d_var.py has_l1_error True`，随后 **14 passed in 141.55 s**。即新增能力未破坏既有行为（日志 `runs/m16228/logs/pytest_advection_regression.log`）。
4. **被审测试自身**：`python3 -m pytest tests/test_case_dg_solid_body_rotation_2d.py -q` → **4 passed, 3 skipped**（skip 的正是 3 条 100×100 验收断言）。
5. **集群债**：`run_cluster ps cj1265` → `queued`（10:55 递交，审计时刻 11:20 仍 queued，无节点派发）；`cj1271`（收敛研究）→ `queued`。**报告 §5.3 承诺的「结果出来即可」至今为空头。**

## 3. 实际存在的数字（审计侧从工件重算，零手抄）

| 算例 | DOFs | 步数 | L1 rel | L2 rel | peak 保留(nodal) | mass drift | L1<5% | peak>95% |
|---|---|---|---|---|---|---|---|---|
| P1 n=16 | 1,024 | 579 | 15.06% | 14.38% | 87.29% | 1.96e-16 | ✗ | ✗ |
| P1 n=20 *(审计新跑)* | 1,600 | 723 | 9.16% | 9.04% | 90.87% | — | ✗ | ✗ |
| P1 n=25 | 2,500 | 904 | 5.29% | 5.39% | 95.93% | 0.0 | ✗ | ✓ |
| P1 n=32 | 4,096 | 1,157 | 2.82% | 2.93% | 97.48% | 1.96e-16 | ✓ | ✓ |
| P2 n=16 | 2,304 | 964 | 0.863% | 0.870% | 99.40% | 3.93e-16 | ✓ | ✓ |
| P2 n=25 | 5,625 | 1,506 | 0.165% | 0.173% | 99.83% | 1.96e-16 | ✓ | ✓ |
| **P2 n=100（声明验收算例）** | 90,000 | ~6,000 | **—** | **—** | **—** | **—** | **未执行** | **未执行** |

审计侧重算的观测阶（工件间）：
`P1 16→25: EOC(L1)=2.345`；`P1 25→32: EOC=2.546`；`P2 16→25: EOC=3.711`（报告写 3.76，取整差异不影响结论）。
mass drift ≤ 3.93e-16 全部算例 —— 守恒性达标。

**disk 范数覆盖面（审计侧算的）**：`l1_reference / mass_initial = 0.03321102100432308 / 0.035342863255146094 = 0.93968`，
即被审的 L1/L2 误差**只覆盖初值的 93.97%**，**6.03% 的质量落在 disk r=0.28 之外不计入误差**。

## 4. 发现（按严重度）

### F1【阻断·声明 vs 证据】声明的验收算例从未执行
- 证据：无 `artifacts/case_dg_solid_body_rotation_2d_p2_n100.json`；`tests/test_case_dg_solid_body_rotation_2d.py` 的 3 条断言读该工件 → `pytest.skip`（实跑 4 passed / 3 skipped）；`cj1265` 状态 `queued`；报告 §4.2 自己写 `P2, n=100 ⏳ pending`。
- 影响：m16200 的 `succeeded` 与「验收判据满足」之间没有证据链。锚点判据 `L1<5% @100×100` 在本次审计中**只能是 NOT_RUN**，不能记为 PASS。
- 复现：`cd /root/tp-agent/work/m16200 && python3 -m pytest tests/test_case_dg_solid_body_rotation_2d.py -q` → `4 passed, 3 skipped`。

### F2【阻断·登记缺失】平台基准表无本锚点行
- 证据：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DG-SOLID-BODY-ROTATION-2D'` → **0 行**。全表 29 行，TensorDG 仅 2 行。
- 影响：容差 `<5%` 目前只存在于(1) mission goal 文本、(2) 被审自己脚本/测试里的 `TOLERANCE = 0.05` 常量 —— **自己给自己定判据**，正是 gate 最易被走过场的形态。建议把该行登记进 `tensor_benchmark_cases`（ref_source=LeVeque 2002 Ch.19，metric=L1 rel + peak retention，tolerance=<5%，grid 100×100 写进 case_name）后再谈 L4。

### F3【阻断·落地】三个交付文件全部未入 git
- 证据：见 §1 `git ls-files` / `git log --all -- …` / `git status`。`result.evidence.head_commit = null`。
- 影响：worktree 一旦回收（`worktree_recyclable` 机制）交付即蒸发；「自动推送分支」没有可推内容。这是本轮 `succeeded` 里最大的实质缺口，与代码质量无关，是流程缺口。

### F4【中·测试形同虚设】验收断言遇缺工件即 skip
- 证据：`tests/test_case_dg_solid_body_rotation_2d.py:20-27` 的 `_load_artifact()` 在文件缺失时 `pytest.skip`，而 3 条验收测试（`test_p2_n100_cfl05_l1_error` / `…_peak_retention` / `…_mass_conservation`）正是唯一对声明判据负责的断言。
- 影响：**绿测试不代表声明判据通过**——这是我这一单最想拦下的反例：`4 passed` 里没有一条覆盖 100×100。验收测试对"缺工件"应当 FAIL 而非 SKIP。
- 另外可跑通的 4 条断言（`e32<e16`、`e25<e16`、n=32/n=25 过 5%）都只验证「粗网格还行」，属执行而非不变量。

### F5【中·指标口径】disk 范数排除了 6.03% 的初值质量
- 证据：见 §3 末行。理由（报告 §2.3：周期接缝处速度场不连续，接缝误差不属于 PDE）是合理的，但结论必须带口径：所谓 0.16% 是**在 94% 场域上**的 L1 相对误差，尾部的数值耗散/色散误差被掩膜遮掉。
- 建议：报告与台账引用该数字时固定写「disk r=0.28 (93.97% mass) 口径」，另报一个全 Ω 的 L1 作为对照。

### F6【中·声明未落地】三个"新增 profile"，实际 0 个被跑过
- 证据：`grep -rn "cone_profile\|slotted_cylinder" --include=*.py .` → 仅定义行（158/172），**零调用方**；`smooth_bump` 只接进 `--profile` 选项（脚本行 68-69、139），**零工件**。报告 §2.2 把三者列为本次新增。
- 影响：LeVeque/Zalesak 类**间断**算例（slotted cylinder）恰恰是 DG 限幅器真正的考题；本次只跑了 C^∞ 高斯，因此对本仓间断输运能力**没有任何新证据**。这是 "声明了能力但没验证" 的典型，判级时不能计入。

### F7【低·外推当结论】"100×100 预计轻松通过"是外推
- 证据：EOC 实测 3.71（P2 16→25）高于设计阶 p+1=3；P1 也 2.35/2.55 > 2。粗网格处于预渐近区（色散主导），EOC 不能外推到 n=100；且 P2 的 EOC 只有两个点。
- 影响：报告 §8「The definitive 100×100 P2 case … expected to easily pass」属推测，不能作为通过依据（报告已在 §5.3 标注 pending，属诚实标注，但倡议性结论仍应删）。

### F8【低·可复现性隐患】repro_cmd 依赖 PYTHONPATH
- 证据：不带 `PYTHONPATH` 直接 `python3 benchmarks/case_dg_solid_body_rotation_2d.py` → `ImportError: cannot import name 'smooth_bump' from 'tensordg.advection_2d_var' (/root/tp-agent/work/m14673/tensordg/advection_2d_var.py)`（脚本目录入 sys.path，命中了别的 worktree 的已装包）。
- 报告 §5.2 里确实写了 `PYTHONPATH=<worktree>`，按原文执行可跑通（§2 已复现）⇒ **不是缺陷，是脆弱点**：建议脚本内自行 `sys.path.insert(0, repo_root)` 或用 `python -m benchmarks.…`。

### 诚实面（该给的分）
- 被审报告**没有编数**：n=100 pending 在 §4.2/§5.3/§8 三处如实标注；六段结构齐（①问题+ref DOI ②方法+代码指针 ③参数表 ω=2π/CFL/σ=0.075/dt 公式 ④got 表+逐条对勾/叉 ⑤两分辨率 EOC ⑥失败模式+repro_cmd+工件表）；审计反硬编码探针未发现回填。

## 5. 判级（L0–L5）

| 级 | 判据 | 结论 | 依据 |
|---|---|---|---|
| L1 数学/单元验证 | tests 非空 | **✓** | 7 条测试（4 条实跑绿，3 条跳过——跳过项记 NOT_RUN，不算通过）|
| L2 独立功能实现 | 真代码 输入→计算→输出 + 工件 | **✓** | `l1_error` 真实现（行 403）；harness 现算；审计新配置 n=20 产生新数值 9.16%；P2 n=25 逐位复现 |
| L3 系统集成 | 接入主求解器/平台 + 调用方 | **✓** | 新能力长在库主解算器类 `DGAdvection2D` 上（同文件行 403），harness 经 `project/solve/peak/l1_error` 端到端驱动；既有 14 条 `tests/test_advection_2d_var.py` 在改动后全绿（模块路径已在进程内打印确认） |
| L4 基准验证 | 完整算例+误差分析+可复现 | **✗** | 声明算例 100×100 **未执行**（F1）、无登记基准行（F2）、交付未入仓（F3）|
| L5 科研工程 | L4 + 外部锚 | **✗** | L4 未达；外部锚（LeVeque 2002）仅引用未对照数值 |

**Verdict：L3。** 差异写明：L4 的缺口**不是**"报告写得不漂亮"，而是"被声明的那个算例本身没跑"——任何百页报告都不能替代 n=100 的那一行数字。

## 6. 处置（disposition）

**Approve with notes（核定 L3，拒绝 L4 主张）。** 具体动作建议：

1. **算例池保持 open**：`CASE-DG-SOLID-BODY-ROTATION-2D` 不得记为带判据的 passed；通过与否的唯一裁判是 n=100 的 got 值。
2. **落地修补（先做，成本最低）**：把 4 个文件（solver 改动 + harness + tests + 报告 + 6 工件）真正 commit 到 `agent/m16200` 并推 PR——现在它们是纯 untracked。
3. **补跑**：`cj1265` 已 30 min 无派发，需重新递交或改本地长跑（预计 1500–2000 s）；结果写入 `artifacts/case_dg_solid_body_rotation_2d_p2_n100.json` 后 3 条验收测试才会真正执行。
4. **测试纪律**：把 `_load_artifact` 的 `pytest.skip` 改成 `pytest.fail`（或 `assert path.exists()`），让"缺工件"变红而不是静默变绿。
5. **登记基准行**：`tensor_benchmark_cases` 插入锚点行（case_name 写死 100×100/CFL0.5/disk r=0.28，metric=L1 rel + peak retention，tolerance=<5%，ref_source=LeVeque 2002 Ch.19），避免自定判据。
6. **补间断算例**：把已写好的 `slotted_cylinder` 接进 `--profile` 并跑出一个工件——否则不要把它写进"本次新增能力"。

## 7. 独立性与未跑项（诚实披露）

- **独立性限制**：解算器内核未重写；独立性来自(a)对既有工件的重算、(b)新配置现算、(c)照抄 repro_cmd 逐位复现、(d)改动后既有测试重跑、(e)直接查库/查 git/查集群状态。
- **未跑（unknown，非 passed）**：100×100 CFL=0.5 验收算例（受 1800 s 墙钟与集群排队限制）；P3 档；`smooth_bump`/`cone_profile`/`slotted_cylinder` 三条 profile 的算例；`cj1265`/`cj1271` 结果（无结果可取）；GitHub 侧是否已推送（沙箱无网）。
- **审计侧未修改被审文件**：`git -C /root/tp-agent/work/m16200 status` 与审计前一致（仍是那 1 M + 9 ??）。

## 8. 工件（本单产出）

| 路径 | 说明 |
|---|---|
| `runs/m16228/auditor_repro_p2n25.json` | 照抄 repro_cmd 的复现工件（与 m16200 工件 7/7 逐位相同）|
| `runs/m16228/auditor_novel_p1n20.json` | 反硬编码探针（工件中不存在的新配置）|
| `runs/m16228/case_dg_solid_body_rotation_2d_p*.json` | 被审 5 个档位工件的审计侧副本（sha256 见 §1）|
| `runs/m16228/logs/pytest_advection_regression.log` | 改动后既有 14 条解算器测试 14 passed（含 MODULE 路径自证行）|
| `docs/figs/m16228_sbr_audit.png` | 误差-网格图（含 5% 门、100×100 NOT RUN 阴影带）+ peak 保留柱图 |
| `tools/plot_m16228_sbr_audit.py` | 上图生成脚本（只读工件，可重跑）|
| `docs/audit_m16228_case_dg_solid_body_rotation_2d.md` | 本文件 |