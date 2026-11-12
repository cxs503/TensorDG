# m16227 交差核验：CASE-DG-KOVASZNAY-NS（TensorDG / m16218）

**性质**：交差核验（非重做）。**判定**：**L4**（基准验证），处置 **confirmed（含 4 条非阻塞整改项）**。
**verdict**：`tensor_maturity_verdicts.id=448`（repo=TensorDG, scope=`m16218 CASE-DG-KOVASZNAY-NS`, mission_id=16227, verdict_by=board-audit）。
**工件**：`artifacts/audit_m16227/audit.json`（机读）、本文件、`scripts/audit/m16227_board_audit.py`（可重跑）。

## 0. 独立性边界

被审树 = `/root/tp-agent/work/m16218`（HEAD `cc1c2df`，branch `agent/m16075` 经 push-merge 并入）。
本审计在 `/root/tp-agent/work/m16227` 独立重算，**未调用**被审仓的误差函数来"复核"其误差：
关键 got 值用 numpy 独立实现 5 点 Gauss–Legendre 张量求积（Jacobian `dx*dy/4`）从
`validation/kovasznay/fields_primary.npz` 重算；ref 用闭式解独立重算后与存档 `u_exact/v_exact` 对照。
未做：`primary` 全量重跑（本地 597 s 单线程，已递交集群 cj1277/cj1280，排队 449 未出队 → 记 not_run）。

## 1. 裁判表（唯一裁判：`tensor_benchmark_cases`）

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-DG-KOVASZNAY-NS';
-- 0 rows
```

**该锚点在基准表里没有登记行**（这是发现 F1，不是我们漏查）。判据只能回溯到算例池：
`tensor_case_pool` id=94，`case_code=DG-KOVASZNAY-NS`，`tol_policy="<1% | 指标: 速度 L2 相对误差 vs 精确解 (Re=40, k=3, 网格 24x24)"`，
`tier=core`，`status=dispatched`，`mission_id=16075`，`linked_anchor=NULL`。

| 项 | 判据（算例池#94） | got（存档工件） | 独立重算 | 判定 |
|---|---|---|---|---|
| 速度相对 L2，P3/24²，Re=40 | < 1 % | 8.818188869671156e-05（0.0088 %） | **8.818188869671156e-05**（逐位一致） | PASS，余量 113.4× |
| u 分量相对 L2 | — | 8.452181340142625e-05 | 8.452181340142625e-05 | 一致 |
| v 分量相对 L2 | — | 3.1207594975243293e-04 | 3.1207594975243293e-04 | 一致 |
| ref 场 vs 闭式解 | — | 存档 `u_exact/v_exact` | 最大偏差 **0.0** | ref 未被篡改 |
| 闭式解自残差（连续性/x/y 动量） | 真解要求 ~机器精度 | 4.44e-16 / 3.77e-15 / 5.27e-16 | — | PASS |

参考解是**闭式解的程序求值**（不是抄表），故无"抄错表"风险面；相应地也没有文献给出的误差数字可逐位比对 → L5 的外部锚点缺席（见 §4）。

## 2. 三件套取证（真存在，非占位）

| 键 | 指针 | 核验 |
|---|---|---|
| impl_files | `tensordg/kovasznay.py`（527 行）、`tensordg/ns_dg_2d.py`（1670 行，本单新增 `_boundary_flux` 的 `"state"` 分支 +25 行） | 非 placeholder：HLLC 鬼状态通量、模态求值器、求积误差度量、分块重启驱动 |
| entry | `benchmarks/verify_kovasznay_dg_2d.py`（338 行，`--job`） | 可执行（见 §3） |
| run_artifacts | `validation/kovasznay/{primary,summary,p1_12,p1_24,p2_12,p2_24,p3_12,mach030,bc_extrap}.json`、`fields_primary.npz`、`docs/figures/kovasznay_{fields,steady_state,convergence}.png` | 全部存在且非空 |
| tests | `tests/test_kovasznay_dg_2d.py`（8 项，判据测试非 smoke） | 本审计重跑：**8 passed in 79.95 s** |
| integration_callers | `tensordg/kovasznay.py` → `from .ns_dg_2d import _rhs_ns, solve_navier_stokes_dg_2d`；`tensordg/__init__.py` 导出；`ns_dg_2d._boundary_flux` 的 `"state"` 分支由本算例的进出口消费 | grep 命中；算例不是自立求解器，而是驱动主 NS DG 求解器 |
| repro_cmd | `cd /root/tp-agent/work/m16218 && PYTHONPATH=. python -m benchmarks.verify_kovasznay_dg_2d --job primary` | 见 §3 |

## 3. 可复现性（本审计亲自跑）

- `--job p1_12` 在被审树的**副本**（`/tmp/repro16218`）重跑：`validation/kovasznay/p1_12.json` 逐叶比对，**除 `wall_seconds`（97.65 s → 94.80 s，−3.0 %）外所有叶完全一致**（速度误差 2.4296e-02、步数 1040、残差 3.12e-04）。
- `pytest tests/test_kovasznay_dg_2d.py`：8 passed（含 `test_steady_error_shrinks_with_degree` 75.5 s）。
- `primary` 全量重跑：not_run（集群 cj1277/cj1280 排队未出队；99 控制面单跑 597 s 超时）。

## 4. 分级判定（L0–L5，按功能实质不按报告篇幅）

| 级 | 判据 | 证据 | 判 |
|---|---|---|---|
| L1 数学/单元验证 | 闭式解残差 + 判据测试 | 残差 3.8e-15；8 项测试全绿 | ✓ |
| L2 独立功能实现 | 输入→计算→输出 | `solve_kovasznay_dg_2d` 真实驱动 `solve_navier_stokes_dg_2d`（SSP-RK3 推进、HLLC+SIPG、分块重启），输出 JSON+场文件 | ✓ |
| L3 系统集成 | 接入主求解器 | 能力长在主求解器内部（`ns_dg_2d._boundary_flux` 的 `"state"` 分支，缺省时逐位向后兼容），算例经主求解器调用；`tensordg/__init__.py` 导出 | ✓ |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | 判据 < 1 %，got 0.0088 %（余量 113×）；网格阶数收敛观测阶 P1 2.08 / P2 2.92（理论 2/3）；负对照 `p1_12` 2.43 % ✗、`bc_extrap` 337.1 % ✗ 说明度量有判别力（非恒真）；复现见 §3 | ✓ |
| L5 科研工程成果 | 独立第三方数值锚点 | 仅闭式解自校核 + 文献 DOI（Kovasznay 1948；Bassi & Rebay 1997），**无第三方误差数字比对** | ✗（未声明） |

**L4 成立，L5 不声明。** 平台侧报告 15318 六要素齐全（①–⑦ 含"是否<3 %"）并附 3 张图。

## 5. 发现（按严重度；均不改变级别）

| ID | 严重度 | 发现 | 证据 | 整改建议 |
|---|---|---|---|---|
| F1 | non-blocking（平台侧记账缺口） | `tensor_benchmark_cases` 对本锚点 **0 行**，判据只存在于 `tensor_case_pool#94`，且其 `linked_anchor=NULL` | §1 SQL | 把算例池 core 项回填为基准表行（metric=`rel L2 velocity`，tolerance=`<1%`，status=passed）并回填 `linked_anchor` |
| F2 | minor（出处） | m16218 工作树有未提交改动：`summary.json` 多出 mach030/bc_extrap 两行、`kovasznay_convergence.png` 重生成（`tensordg/__init__.py` 亦未提交） | `git status`；提交对应 `git log -1 -- validation/kovasznay/mach030.json` = `64bfa60` | 逐运行 JSON 已入库，故引用数字仍可溯源；建议补一次聚合提交，避免推送分支丢聚合与导出 |
| F3 | minor（措辞超标） | `docs/CASE-DG-KOVASZNAY-NS.md` 第 31 行称 k=3 波数变体"已跑且列入表 1"，而表 1 / §5.3 记 `wavenumber3`「未跑」 | 文档自相矛盾 | 改 31 行为"未跑（§5.3）"；平台报告 15318 未复述该错误声明 |
| F4 | minor（实验设计） | 门禁配置推进到 t=0.08，其余扫描全部 t=0.2；误差在窗口内仍漂移 +44.8 %（报告 §5.4 已披露） | `validation/kovasznay/primary.json` history vs `RUNS` 表 | 用同一窗口（t=0.2）复跑门禁配置，或把 0.08 写进算例卡作为口径 |
| F5 | note（声明强度） | 报告 15318 的"M² 拟合指数 1.98"仅由 2 个马赫点得出，是斜率不是拟合 | 报告 §⑤ 表只有 primary/mach030 两行 | 补 c=120 点后再写指数 |

## 6. 未跑项（如实记 not_run，不算通过）

- `primary` 重跑（集群排队）。
- t=0.2 的门禁配置探针（cj1280，排队中）。
- `mach120` / `bc_adjoint` / `wavenumber3` / `ic_uniform` / `ic_exact_ref` 五个已定义未运行配置（被审报告同样标注「未跑」）。
- 与文献误差数字的逐位比对：**无此数据**（ref 为闭式解，文献只给格式与设定）。

## 7. 处置

**confirmed（L4）**：判据达标（0.0088 % < 1 %，余量 113×）、结果经本审计独立重算逐位复现、测试与集成调用方为真。
F1 为平台侧基准登记缺口，F2–F5 为非阻塞整改项，均不改变功能级别；
若平台愿提升到 L5，需补独立第三方（文献/实验）数值锚点，本审计不作该声明。