# dev report · TensorDG：CASE-LAX-DG 分辨率闭合验证测试（重开 #676 第 1 轮）

**锚点**：`CASE-LAX-DG`（算例池 · Lax via DG，台账验收原文「ref: 精确解｜容差 < 3%」）
**分支**：`agent/m23026`（自 `main` c199bbdf 切出）
**范围**：`tests/test_lax_resolution_closure.py`（唯一代码文件）+ 本报告
**重叠声明**：本单**不触碰** `tensordg/` 下任何核心模块，也不改 `benchmarks/lax_resolution_closure.py`
与 `scripts/render_lax_closure_doc.py`；只读取它们的已登记常量与已提交工件。因此不与其他在开 PR
的 TensorDG 核心模块热文件路径重叠。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域、隔膜 `x0 = 0.5`、γ = 1.4、快照 `t = 0.13`。
精确解含四道波：左稀疏波（head → tail）、接触间断、右激波，四者位置与平台值均有闭式解。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |
| 解析波系 | p\* = 2.466098、u\* = 1.528723、ρ\*\_L = 0.344568、ρ\*\_R = 1.304085 | |

登记配置（工件 `config` 块）：p = 3、CFL = 0.35、t = 0.13、
域 = [0.0, 1.0]、R3 带半宽 = 0.02、阶梯 N = [200, 400, 800, 1600]、
限幅器臂 = ['none', 'tvb', 'weno']（生产网格 N = 400）。

## ② 怎么做的（方法 / 流程 / 工具链）

**背景**：台账把验收读成生产腿 `p3.N400` 的全场相对 L2，该腿实测 3.116 %，高于 3 % 门，是**诚实的 FAIL**。分辨率闭合研究（`benchmarks/lax_resolution_closure.py`；本次工件墙钟 1385 s ≈ 23 min）
把同一登记配置细化到 N = [200, 400, 800, 1600]，证明 3 % 门在 N = 800 上可达
（三量最差 2.315 %）。工件与文档已提交，但**结论此前无法被 `pytest` 复核**：
任何读者只能信任文档里的手写数字。

**本单工作**：把该结论做成可执行、不可漂移、不可空过的护栏。**22 条测试**分七组，全部只读已提交工件
（`docs/benchmarks/results/lax_resolution_closure/metrics.json`），不重跑求解器：

| 组 | 目的 | 代表测试 |
|---|---|---|
| A 判决再推导 | verdict 由阶梯重新算出，而非在 JSON 里断言 | `test_closure_mesh_is_rederived_from_the_ladder` |
| B 台账门 | N=800 三量 < 3 %；N=400 仍 ≥ 3 %（honest FAIL 不得被改口径） | `test_n800_passes_ledger_bar` / `test_n400_still_fails_ledger_bar` |
| C 限幅器归因 | 单旋钮：无限幅臂最准，TVB 使 L2 密度恶化 ≥ 2× | `test_tvb_degrades_accuracy_vs_unlimited` |
| D 视图一致性 | `summary.md` 与主文档都是工件的**渲染视图**，逐字节比对 | `test_case_doc_is_a_rendered_view_of_the_artifact` / `test_summary_md_is_exactly_the_rendered_view` |
| E 代码↔工件 | 驱动常量（阶梯/阶数/生产网格/限幅器臂/CFL）与工件 `config` 一致，防止「改了驱动没重跑」 | `test_driver_constants_and_artifact_config_agree` |
| F 报告↔模块 | 本报告点名的测试必须真实存在，条数必须相符 | `test_dev_report_test_inventory_matches_the_module` |
| G 门空过修复 | 工件/摘要缺失由 `skip` 改为**硬失败**（`docs/` 下工件是被跟踪的，缺失=证据被删） | `test_committed_views_and_figures_are_on_disk` |

工具链：`pytest` + 纯 `json`/`pathlib`/`re`；D 组直接调用 `benchmarks.lax_resolution_closure.render_summary`
与 `scripts.render_lax_closure_doc.render` 重新渲染后与提交文件逐字节比较；B 组数字由工件算出，
报告表格由生成脚本从工件注入（零手抄），并由 F/G 组反向校验。

### 护栏清单（按文件顺序；前 16 条为既有护栏，后 6 条为本单新增：D/E/F/G 组 + 视图存在性）

- `test_ladder_is_the_registered_production_arm`
- `test_production_leg_number_is_the_ladder_entry`
- `test_closure_mesh_is_rederived_from_the_ladder`
- `test_extrapolation_never_counts_as_a_pass`
- `test_every_field_converges_along_the_ladder`
- `test_limiter_arms_change_only_the_limiter`
- `test_figures_are_declared_on_disk`
- `test_n800_passes_ledger_bar`
- `test_n800_worst_field_is_density`
- `test_n400_still_fails_ledger_bar`
- `test_unlimited_arm_is_most_accurate_at_n400`
- `test_tvb_degrades_accuracy_vs_unlimited`
- `test_closure_n800_l2_matches_documentation`
- `test_summary_md_verdict_matches_the_gate`
- `test_summary_md_tables_match_metrics`
- `test_ingest_refreshes_a_stale_summary`
- `test_committed_views_and_figures_are_on_disk`
- `test_driver_constants_and_artifact_config_agree`
- `test_case_doc_is_a_rendered_view_of_the_artifact`
- `test_summary_md_is_exactly_the_rendered_view`
- `test_dev_report_test_inventory_matches_the_module`
- `test_dev_report_ladder_table_matches_the_artifact`

## ③ 计算结果（程序化注入）

### 3.1 测试执行（本机，2026-11-16）

```
python -m pytest tests/test_lax_resolution_closure.py -q
→ 22 passed
```

既有 16 条在本次改动前后均通过；新增 6 条在修复本报告之前**故意红**（F 组两条），
说明护栏确实能抓住「报告与模块不一致」这一缺陷类，而非空断言。

### 3.2 p3 无限幅阶梯（登记臂，数字表由工件注入并被 `test_dev_report_ladder_table_matches_the_artifact` 反向校验）

<!-- LADDER-TABLE-START -->
| N | ρ L2 | u L2 | p L2 |
|---|---|---|---|
| 200 | 4.200 % | 3.390 % | 1.655 % |
| 400 | 3.116 % | 2.248 % | 1.305 % |
| 800 | 2.315 % | 1.383 % | 0.899 % |
| 1600 | 1.875 % | 1.475 % | 0.917 % |
<!-- LADDER-TABLE-END -->

### 3.3 生产网格 N = 400 单旋钮限幅器对比

| limiter | ρ L2 | u L2 | p L2 |
|---|---|---|---|
| none | 3.116 % | 2.248 % | 1.305 % |
| tvb | 7.039 % | 2.981 % | 1.423 % |
| weno | 6.443 % | 2.940 % | 1.414 % |

TVB 使 ρ L2 由 3.116 % 恶化到 7.039 %，比值 2.26×。

### 3.4 闭合判决（工件 `gate`）

- verdict：**PASS**，闭合网格 = **800**
- 最差量：`density`
- 最细对阶 [800, 1600] = 0.304052
- 外推闭网格 = 1600（仅模型预测，**不计为通过**）
- 工件生成：2026-11-14T11:26:01Z，墙钟 1385 s

## ④ 结果图

本单是**验证护栏**类交付，不产生新的物理场图；结论全部来自对已提交工件的机器化复核。
被护栏锁定的两张已提交图（由原驱动生成，`test_committed_views_and_figures_are_on_disk` 校验存在且非空）：

- `docs/benchmarks/results/lax_resolution_closure/figures/closure_ladder.png` — 登记 p3 无限幅臂上
  三量全场相对 L2 随网格数变化；虚线 = 3 % 台账门，绿线 = 达标网格。
- `docs/benchmarks/results/lax_resolution_closure/figures/closure_arms.png` — 生产网格 N = 400
  上单旋钮（limiter）对比；两种限幅器都比无限幅更差（问题单元退化为线性重构）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **测试不重跑求解器**：22 条测试全部读已提交工件，跑得很快，但工件本身的正确性由原集群运行（墙钟 1385 s）
   保证；本单只保证「工件、视图、驱动常量、报告」四者不互相漂移。
2. **N = 800 通过 ≠ 台账原文已闭合**：台账登记的生产腿是 `p3.N400`，那条腿的诚实结论仍是 FAIL
   （3.116 %）。本单证明的是「同一配置在 N = 800 上可达 3 % 门」，
   是接受准则的分辨率限定，不是改口径。`test_n400_still_fails_ledger_bar` 专门把这条钉死。
3. **门空过是真实缺陷类**：原实现的 `pytest.skip` 在工件缺失时会让 16 条断言全部静默通过；
   工件被 git 跟踪，缺失只可能是证据被删，因此改为硬失败。
4. **限幅器不是补救手段**：TVB/WENO 在 N = 400 上都让 ρ L2 恶化（2.26× / 2.07×），
   因为本仓限幅器把问题单元退化为线性重构，在 p3 上属过度限制。
5. **适用范围**：护栏绑定的是「已登记的 p3 无限幅配置 + 2026-11-14T11:26:01Z 的工件」。重跑驱动后
   必须同时重渲染 `docs/CASE-LAX-DG-resolution-closure.md` 与 `summary.md`，否则 D 组会红。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | 是否 < 3 % |
|---|---|---|---|
| N=800 密度 L2 | 精确 Riemann 解 | 2.315 % | 是 |
| N=800 速度 L2 | 精确 Riemann 解 | 1.383 % | 是 |
| N=800 压力 L2 | 精确 Riemann 解 | 0.899 % | 是 |
| N=400 密度 L2（台账生产腿） | 精确 Riemann 解 | 3.116 % | 否（honest FAIL，保留） |

**结论**：登记配置在 N = 800 上三量全部 < 3 %，闭合 verdict = PASS；
生产腿 N = 400 仍为 FAIL。22 条测试把这两句话、连同它们所依赖的工件/视图/驱动常量/
本报告一起锁死，任一环节漂移都会让 CI 变红。
