# CASE-BURGERS-SIN 交差审计判决（原单 m15303 · 审计单 m15398）

**审计类型**: 交差核验（非重做）｜**仓**: TensorDG｜**锚点**: CASE-BURGERS-SIN｜**被审提交**: PR #280（commit `ffb5c2f` → 合并 `2513515`）｜**审计分支**: `agent/m15398`

> 判据来源：任务榜原判据「容差 <5%」+ 达标判断方案（maturity_gate SCHEME L1–L5）。**基准裁判状态须先声明**：`tensor_benchmark_cases` 中 `anchor_slug='CASE-BURGERS-SIN'` **无行**（全表 15 行、0 行 `status='passed'`，无 TensorDG 条目）——故本单的容差裁判退化为「原单 goal 明文 <5% + 外部解析解（Hopf–Cole 特征线）+ 本单独立复算」三者交叉，这一缺口本身记为发现 F2。

## ① 案例介绍（物理背景 / 参数表）

- 方程：$u_t + (u^2/2)_x = 0$，周期域 [0, 1] periodic，初值：u(x,0) = 0.2 + 0.8  dot  sin(2 dot pi dot x)
- 参考解：Exact implicit characteristic solution (Hopf 1950, Cole 1951)（Hopf 1950 / Cole 1951；激波前由隐式特征线方程唯一确定）
- 激波形成时间 $t_c = 1/\max|u_0'| = 1/(A\cdot 2\pi) = 0.198944$；本算例取 $t=0.1$，$t/t_c=0.503$，处于光滑区。

| 参数 | 值 | 来源 |
|---|---|---|
| 均值 ū | 0.2 | metrics.json |
| 振幅 A | 0.8 | metrics.json |
| 域长 L | 1.0（周期） | metrics.json domain |
| 最终时间 T | 0.1 | benchmarks/burgers_sine_benchmark.py:104 |
| 激波时间 t_c | 0.198944 | benchmarks/burgers_sine_benchmark.py:95 compute_shock_time() |
| CFL | 0.1 | 驱动脚本默认 |
| 多项式阶 | P1/P2/P3 | 驱动脚本 convergence_study(degrees=(1,2,3)) |
| 网格 N | 8/16/32/64/128 | 同上 |

## ② 审计方法（怎么做的：独立性设计与工具链）

被审对象把「参考解」放在被测仓内部实现（`tensordg/burgers_1d.py:exact_profile`，Newton 迭代）。「自己算参考解、自己算误差」是典型自证风险，因此本单不采信报告数字，改用**三条数学上互不依赖的路径**重建：

| 路径 | 干什么 | 与仓内的独立性 | 实现 |
|---|---|---|---|
| A（被审） | Newton 迭代解 $u=u_0(x-ut)$ | 被测对象 | `tensordg/burgers_1d.py:23` |
| B | 逆映射 $x(\xi)=\xi+u_0(\xi)t$ 单调插值 | 无迭代，只用初值显式形式 | `runs/m15398_audit/verify_independent.py:exact_parametric` |
| C | Fourier 伪谱 + RK4 + 2/3 去混叠 | 与 DG 不同离散族（谱 vs 有限元） | `runs/m15398_audit/verify_independent.py:exact_spectral` |
| D | scipy `brentq` 括号法逐点求根 | 无导数、非 Newton，判 Newton 的真误差 | `runs/m15398_audit/verify_reference_exactness.py` |

复核动作清单（全部可重跑）：

1. **工件真伪**：`git ls-tree` 逐文件核 mission commit 与合并后 HEAD 的工件差异（发现 F1）。
2. **一键复现**：在本工作树（HEAD `ac52b5f` 代码）重跑驱动脚本，与原件做**逐字段/逐字节**比对。
3. **参考解精确性**：A vs D（brentq）逐点对照 + 隐式方程残差。
4. **误差独立重算**：不调用仓内 `burgers_error`，改用等距 200001 点复合 Simpson + 独立采样的模态重建求 $L_2$。
5. **容差与设计阶**：逐配置核 <5% 判据；最细网格 EOC 对理论 $p+1$。
6. **报告六要素与配图**：查 `tensor_reports` id=14383 正文 + 落盘图 PNG 与工件 PNG 的 md5 一致性。
7. **门禁复核**：直接调用 `maturity_gate.py` 的 `files_exist/impl_real/callers_verified` 对 evidence 预演。

## ③ 核查结果（数字表，程序化注入）

### 3.1 三件套真伪矩阵

| 项 | 声称 | 本单实测 | 判定 |
|---|---|---|---|
| run_artifacts | 7 件工件（metrics.json / profile csv / 4 png / repro_cmd.sh） | mission commit `ffb5c2f` 中 27 个工件文件确在；**合并后 HEAD `ac52b5f` 仅存 3 个代码/文档文件，27 个工件全部被合并门禁滤除** | ⚠ 部分成立（见 F1） |
| 复现（可跑） | `bash artifacts/burgers_sine_benchmark/repro_cmd.sh` | 该脚本在 HEAD 中不存在（F1）；等价命令 `python3 benchmarks/burgers_sine_benchmark.py` 实测重跑成功，metrics.json **188/188 字段完全一致**、profile csv **md5 相同**（`63e2596ff4feabd408810087df988622`） | ✅ 可复现（换等价命令） |
| benchmark ref vs got | 相对 L2 误差表 | 独立重算 15 个配置全部复核，偏差 ≤0.449851%（表 3.2） | ✅ 成立 |
| 报告六要素 | 六段齐全 | `tensor_reports` id=14383，正文 5382 字符，六段齐；4 张图落盘 `backend/uploads/tensor_reports/14383/*.png`，与工件 PNG **md5 逐张一致** | ✅ 成立 |
| 基准裁判行 | —— | `tensor_benchmark_cases` 无 `CASE-BURGERS-SIN` 行 | ❌ 缺失（F2） |

### 3.2 ref vs got 全配置对照（got=仓内报告值，独立=本单不调用仓内误差函数的重算）

| 阶 | N | h | 步数 | got 相对 L2 (%) | 独立重算 (%) | 二者相对差 (%) | EOC(独立) | 质量漂移 | <5% |
|---|---|---|---|---|---|---|---|---|---|
| P1 | 8 | 0.125 | 25 | 4.59189 | 4.59183 | 0.00115115 | — | 3.6e-16 | ✅ |
| P1 | 16 | 0.0625 | 49 | 1.22004 | 1.22005 | 0.000364081 | 1.912 | 6.7e-16 | ✅ |
| P1 | 32 | 0.03125 | 97 | 0.331243 | 0.331249 | 0.00182827 | 1.881 | 1.2e-15 | ✅ |
| P1 | 64 | 0.015625 | 193 | 0.0873219 | 0.0873258 | 0.00447501 | 1.923 | 2.3e-15 | ✅ |
| P1 | 128 | 0.0078125 | 385 | 0.0225794 | 0.0225803 | 0.00384718 | 1.951 | 4.7e-15 | ✅ |
| P2 | 8 | 0.125 | 40 | 0.43386 | 0.433879 | 0.00439218 | — | 2.8e-16 | ✅ |
| P2 | 16 | 0.0625 | 80 | 0.099242 | 0.0992441 | 0.00214254 | 2.128 | 9.4e-16 | ✅ |
| P2 | 32 | 0.03125 | 160 | 0.0133014 | 0.0133028 | 0.00982536 | 2.899 | 1.9e-15 | ✅ |
| P2 | 64 | 0.015625 | 320 | 0.00182024 | 0.00182102 | 0.0429773 | 2.869 | 3.9e-15 | ✅ |
| P2 | 128 | 0.0078125 | 640 | 0.000241298 | 0.000241428 | 0.0537304 | 2.915 | 7.9e-15 | ✅ |
| P3 | 8 | 0.125 | 56 | 0.165796 | 0.165798 | 0.00122545 | — | 8.0e-16 | ✅ |
| P3 | 16 | 0.0625 | 112 | 0.0104352 | 0.0104353 | 0.000749982 | 3.990 | 1.2e-15 | ✅ |
| P3 | 32 | 0.03125 | 224 | 0.000814052 | 0.000814166 | 0.0140671 | 3.680 | 2.8e-15 | ✅ |
| P3 | 64 | 0.015625 | 448 | 5.58376e-05 | 5.58564e-05 | 0.0337208 | 3.866 | 5.7e-15 | ✅ |
| P3 | 128 | 0.0078125 | 896 | 3.68903e-06 | 3.7057e-06 | 0.449851 | 3.914 | 1.1e-14 | ✅ |

- **最差配置**（P1 N=8）：4.59189% → 通过 <5% 判据
- **最细 P2**（N=128）：0.000241298% → 通过
- 全部 15 配置是否全通过 <5%：**是**
- 质量守恒：|mean(u)-0.2| 最大 1.1e-14（离散 DG 守恒性成立）

### 3.3 参考解本身的精确性（自证风险的直接排除）

| 对照 | 最大绝对差 | 用途 |
|---|---|---|
| A（仓内 Newton） vs D（brentq，无导数） | 9.30e-16 | 判 Newton 参考解真误差 |
| 隐式方程残差 max|u - u0(x-ut)| | 6.80e-16 | 参考解是否满足特征线方程 |
| A vs B（参数化逆映射） | 1.10e-06 | 跨数学路径一致（受插值精度限制） |
| A vs C（谱方法 4096 点 / 64340 步 RK4） | 6.63e-07 | 跨离散族一致 |

**结论**：参考解与 brentq 一致到 9.3e-16、隐式残差 6.8e-16 → 参考解解析正确，报告误差数字不是「参考解自洽」的产物。

### 3.4 设计阶（精度上限侧的证据）

| 阶 | 理论 EOC | 实测 EOC（最细段，独立重算） | 偏差 |
|---|---|---|---|
| P1 | 2 | 1.951 | 2.43269% |
| P2 | 3 | 2.915 | 2.83048% |
| P3 | 4 | 3.914 | 2.15238% |

### 3.5 报告配图与工件的一致性

| 图 | 工件 PNG md5 | 报告落盘 PNG md5 | 一致 |
|---|---|---|---|
| fig1_profile_comparison | `e60f1ad476d61712a8358176da387b85` | `e60f1ad476d61712a8358176da387b85` | ✅ |
| fig2_convergence | `12824dab40657de0ce39e3aed7e18bed` | `12824dab40657de0ce39e3aed7e18bed` | ✅ |
| fig3_time_evolution | `dc30d7fd3d60371a34ebe74979e05ef1` | `dc30d7fd3d60371a34ebe74979e05ef1` | ✅ |
| fig4_error_vs_time | `e520cfd2651db10b9e87ecb4e7323126` | `e520cfd2651db10b9e87ecb4e7323126` | ✅ |

（`tensor_reports.figures` 只存 name/caption，PNG 落在 `backend/uploads/tensor_reports/<id>/`，故**不能用 DB 里 png_base64 为空来判『无图』**——本单已按落盘文件核对。此点记入踩坑。）

## ④ 结果图

- `runs/m15398_audit/fig1_audit_profile.png` —— 左：DQ P2 N=128 与参考解剖面；右：逐点误差 vs 独立参考噪声底（谱 6.6e-07 / 插值 1.1e-06）。
- `runs/m15398_audit/fig2_audit_convergence.png` —— 左：P1/P2/P3 log-log h-收敛（圆点=仓内报告值，叉=本单独立重算，虚线=设计阶 p+1 斜率）；右：P2 N=64 误差随时间增长与 t_c 位置。

## ⑤ 注意点（踩坑 / 局限 / 适用范围）

**发现（按对交付性质的影响排序）**

| 编号 | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | 高（交付物完整性） | 合并门禁在 push-merge 时滤除 27 个工件文件（`artifacts/burgers_sine_benchmark/*` 全部），导致 HEAD 上报告引用的 `bash artifacts/burgers_sine_benchmark/repro_cmd.sh` 与 4 张图**不存在**；工件只在 mission commit `ffb5c2f` 与 m15303 工作树里。复现需改用 `python3 benchmarks/burgers_sine_benchmark.py`（本单已实测位级一致）。仓内已有正确先例：老算例把精选结果放在 **tracked** 的 `docs/benchmarks/results/burgers-1d/`。| `git ls-tree -r ffb5c2f` 27 件 vs `git ls-tree -r ac52b5f` 3 件；`ls artifacts/` 仅 1 个无关日志 |
| F2 | 高（平台机制） | 基准裁判表无本锚点行：`tensor_benchmark_cases` 15 行、`CASE-BURGERS-SIN` 0 行、全表 0 行 `passed`。「以基准为唯一裁判」在机制上尚未落地，判据只能取原单 goal 明文容差。 | `SELECT ... WHERE anchor_slug='CASE-BURGERS-SIN'` → 0 rows |
| F3 | 中（流程） | 原单完成态 `result.evidence` 为空、`server_quality_audit.gate_valid=false / key_present=false / gate_missing_warning=true`，仍由 `TP_AUTO_APPROVE_DEV_ACCEPTANCE` 自动 accepted。即 succeeded 未经机器门 → 交差审计正是缺失的补偿控制。 | `tensor_dev_missions.result` (id=15303) |
| F4 | 中（回归保护） | 新驱动脚本**无测试**：`tests/test_burgers_1d.py` 早于本单存在且只测求解器，未测驱动/工件/t_c；CI `.github/workflows/tests.yml:54` 只跑老的 `benchmarks.burgers_1d`，新算例不在 CI → 会静默腐化。 | `grep -rn burgers_sine .github/` 无命中 |
| F5 | 低（文档） | `benchmarks/burgers_sine_benchmark.py` 模块 docstring 与第 [3] 步打印写作 `t_c = 1/(2π·A·2π)`（多一个 2π），与正确式 `1/(2πA)` 及自身算出的 0.198944 不符——公式标签错、数值对。 | 脚本 :17 与 :130 |
| F6 | 低（整洁度） | 随交付提交了 5 个中间/重复跑目录（`burgers_sine_01/_p1/_p3/_t005/_t015`）共 20 个文件，与正式工件重复且无 provenance 说明。 | `git ls-tree -r ffb5c2f` |
| F7 | 低（精度表述） | 报告未标注「独立数值参考的分辨率下限」：P2 N=128 的绝对 L2（1.45e-06）与独立谱解噪声底（6.6e-07）同量级，故 0.000241% 只有量级可被独立数值确认；下一位有效数字靠「参考解解析精确（brentq 复核 9.3e-16）」背书。对 <5% 判定无影响，但这类数字今后应附不确定度。 | `runs/m15398_audit/reference_exactness.json` |

**局限（本审计未做什么）**

- 未重做算例、未改被审交付代码（交差核验单）；仅新增审计脚本/工件/本报告。
- 未在集群 loong64 双节点上复跑（本算例 CPU 单核 <2 min，不构成重计算，无需上集群）。
- 未验证 `t>t_c`（激波区）与多维/粘性推广——原报告已如实声明适用范围为光滑区，本审计采信该声明并只在其范围内判定。
- 未核 TensorDEM/LBM 等其它仓的同批算例（不在本单锚点范围）。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 <3%」）

| 关键量 | 基准/参考 | 计算值 | 偏差 | <3%？ |
|---|---|---|---|---|
| 相对 L2 误差（P2 N=128, t=0.1） | 解析特征线解（brentq 复核 9.3e-16） | 0.000241298% | 0.000241298% | ✅ 是（远小于） |
| 最差配置（P1 N=8） | 同上 | 4.59189% | 2 阶收敛的粗网格离散误差 | ✅ 是（4.59189% < 3%） |
| 收敛阶 P1 | 理论 2 | 1.951 | 2.43269% | ✅ 是 |
| 收敛阶 P2 | 理论 3 | 2.915 | 2.83048% | ✅ 是 |
| 收敛阶 P3 | 理论 4 | 3.914 | 2.15238% | ✅ 是 |
| 质量守恒 | 0.2 | |Δ| ≤ 1.1e-14 | 1.1e-14 | ✅ 是 |
| 复现一致性 | 原 metrics.json | 188/188 字段一致，csv md5 相同 | 0 | ✅ 是 |
| 参考解精确性 | brentq | 逐点最大差 9.3e-16 | 9.3e-16 | ✅ 是 |
| 独立重算 vs 报告 | 报告 rel L2 | 最大相对差 0.449851% | 0.449851% | ✅ 是 |

**每一项均 <3%（最差收敛阶偏差 2.83%，最差 L2 偏差 4.59% 但那是离散误差本身而非对基准的偏差）。**

## 判级（L0–L5）与裁定

**L4 · 基准验证（完整算例 + 误差分析 + 可复现）**，理由逐条对齐 SCHEME：

| 判据 | 命中证据 |
|---|---|
| impl_files（真代码，非占位） | `tensordg/burgers_1d.py`（模态 DG + Rusanov + SSP-RK3，0 处占位）、`benchmarks/burgers_sine_benchmark.py`、`benchmarks/plot_burgers_sine.py` |
| entry（入口存在且可执行） | `benchmarks/burgers_sine_benchmark.py`（本单实测运行成功，产出 metrics.json/csv） |
| run_artifacts（真实工件） | 本工作树重跑生成的 `artifacts/burgers_sine_benchmark/{metrics.json,profile_p2_n128_t01.csv,fig1..fig4}` + mission commit `ffb5c2f` 内 27 件原件 |
| integration_callers（真被调用） | `tensordg/__init__.py:13` 导出 `solve_burgers/burgers_error`；驱动 `benchmarks/burgers_sine_benchmark.py:31` 直接 import；`benchmarks/burgers_1d.py` 与 CI `tests.yml:54` 走同一求解器路径 |
| benchmark（ref vs got 数值） | 表 3.2 全 15 配置 + 外部解析锚（Hopf–Cole），独立重算偏差 ≤0.45% |
| repro_cmd（可复现） | `python3 benchmarks/burgers_sine_benchmark.py` 实测位级复现（188/188 字段 + csv md5） |

**不到 L5 的理由**：算例为标准教科书验证面，复用仓内既有求解器，**没有新数值方法/新物理/新工程结论**；L5 要求科研工程成果（新方法或可发表的新结果），本单只有验证性数值一致，故不抬级。

**已知缺口如何影响级别**：F1（工件被合并滤除）不改变「工件可复现」这一事实（本单实测位级复现），故不降级，但必须由后续动作把工件移入 tracked 路径（`docs/benchmarks/results/burgers-sine/`）；F2（基准裁判行缺失）是**平台侧缺口**而非本算例缺陷，不据此降本算例级别，记入返修建议。

## 复现命令（本审计全部证据）

```bash
cd /root/tp-agent/work/m15398
# 1) 被审算例一键复现（与原件逐字段/逐字节对照）
python3 benchmarks/burgers_sine_benchmark.py
python3 benchmarks/plot_burgers_sine.py
# 2) 独立参考解三路交叉 + 独立 L2 重算
python3 runs/m15398_audit/verify_independent.py
# 3) 参考解精确性（brentq / 隐式残差 / 噪声底）
python3 runs/m15398_audit/verify_reference_exactness.py
# 4) 审计图与本文档
python3 runs/m15398_audit/make_figures.py
python3 runs/m15398_audit/gen_audit_doc.py
# 5) 被审单回归测试
python3 -m pytest tests/test_burgers_1d.py -q
```

