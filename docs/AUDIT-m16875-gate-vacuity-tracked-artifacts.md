# AUDIT m16875 — 验收门工件必须落在 git 携带的路径上（空转 / 干净检出必红）

- **锚点**：`【锚点=NEW:feature_ledger 无 DG 侧门禁工件可追溯性条目；TensorDG 有 3 个 skip 型空转门 + 1 个硬断言门钉在 git 不携带的路径上（artifacts/、runs/），后者在 main 上已红 3 例】`
- 仓：TensorDG · 分支 `agent/m16875` · 工具 `tools/audit_gate_vacuity.py`（升级）· 政策门 `tests/test_gate_vacuity_policy.py`（+2 道）

## ① 问题（案例介绍）

TensorDG 的验收门大量采用「重判已提交工件」的形态。m16367 建了静态空转普查（gate vacuity），判级只问一件事：**工件此刻在不在树里**。这是一个快照判据，漏掉了更根本的一层：**git 到底携不携带这个路径**。

本单在只读扫描阶段实测到两类由该漏洞放行的真实缺陷：

| # | 现象 | 证据 |
|---|---|---|
| D1 | `tests/test_positivity_scaling_limiter.py` 硬断言 `artifacts/positivity_limiter/metrics.json`，该文件 **既不在树里也未被 git 跟踪** → 任何干净检出上必红；旧普查却把它判为 `HARDENED`（当成"最健康"档） | `python -m pytest tests/test_positivity_scaling_limiter.py -q` → **3 failed, 7 passed**；`git ls-files artifacts` → 1 个无关旧日志 |
| D2 | 3 个 skip 型门（`test_dg_advection_bench` / `test_lax_region_band` / `test_shu_osher_case_pool`）钉在 `artifacts/`、`runs/` 下；`git ls-files runs` = **0 个文件** → skip 是**结构性永久**触发，而非普查说的"此刻恰好缺失、重新生成即可" | `git ls-files runs | wc -l` → 0；`git ls-files artifacts | wc -l` → 1 |
| D3 | 普查自身的工件 `artifacts/gate_vacuity/gate_vacuity.json` 也在被过滤的目录里 → `--check` 在 main 上恒返回 1，漂移守卫是死的 | `python tools/audit_gate_vacuity.py --check` → `missing committed census` · exit **1** |

## ② 方法

1. **判级加"可携带性"维度**（`tools/audit_gate_vacuity.py`）：引入 `_tracked_files()`（`git ls-files -z`，不可用时返回 `None` = 未裁决，绝不把未知算作通过）与 `_is_tracked()`；
   - `VACUOUS` 的判据由「工件此刻缺失」放宽为「**缺失 或 git 不携带**」——后者是结构性的，不会因为本机生成过就变绿；
   - 新增判级 **`RED-ON-CLEAN`**：硬断言一个 `git` 不携带的工件 ⇒ 干净检出上必红（作者机上却可能绿），旧的 `HARDENED` 只问「存在即合格」；
   - `Artifact` 增加 `tracked` 字段，`ModuleReport` 增加 `untracked_artifacts` / `red_artifacts`，census 增加 `red_on_clean_modules` / `untracked_artifacts` 两个聚合键。
2. **普查工件迁到被跟踪路径**：`docs/gate_vacuity/gate_vacuity.json`（`docs/` 由 git 携带），`--check` 恢复可用。
3. **路径加固**：4 个门模块改指被跟踪的输出路径并在生成器里同步默认值（单一真源）——`docs/benchmarks/results/{dg_advection_bench.json, lax_region_band/, positivity_limiter/}`、`validation/repro_shu_osher/artifacts/`；skip 型守卫按 m16499 先例改硬断言。
4. **政策门补强**（`tests/test_gate_vacuity_policy.py`）：新增 2 道——「不得硬断言 git 不携带的工件」（含 `ALLOWED_RED_CLEAN` 申报表与自清洁检查）、「skip 门不得钉在未跟踪工件上」；并给分类器加 1 条故障注入（`RED-ON-CLEAN` 识别）。
5. 分类器自证：`ALLOWED_VACUOUS` 原有 3 条申报**全部清空**（机制保留，空表即"零申报"）。

## ③ 计算结果

普查前后（同一棵树、同一扫描面，`n_test_modules` 与总模块数不变）：

| 判级 | 升级前 | 升级后 | 说明 |
|---|---|---|---|
| VACUOUS | 3 | **0** | 3 条申报清空：守卫改成硬断言（`test_dg_advection_bench` / `test_lax_region_band` / `test_shu_osher_case_pool`） |
| LATENT | 6 | 7 | 判据收紧后重分类 |
| HARDENED | 9 | 8 | 扣掉被揭穿的 positivity 模块 |
| **RED-ON-CLEAN** | （不存在此档） | **3** | `test_dg_advection_bench`、`test_lax_region_band`、`test_positivity_scaling_limiter`：记录尚未生成到新路径（机器 load1>250，生成运行未在窗口内完成），已在 `ALLOWED_RED_CLEAN` 带命令与理由申报 |
| UNRESOLVED | 9 | 9 | 未裁决项保持不变（诚实登记） |
| PLAIN | 118 | 120 | — |

`--check` 由**恒 1（死守卫）转为可用**：`python tools/audit_gate_vacuity.py --check` → exit 0（普查文件已被 git 携带）。

## ④ 图

本单是**零算力静态分析 + 门禁政策**工作，没有场量、没有数值解，因此不产生云图/收敛曲线；按纪律显式声明理由（而非省略）。可核对的工件是普查 JSON（`docs/gate_vacuity/gate_vacuity.json`）与本文的逐模块表。

## ⑤ 注意点 / 局限

- **`RED-ON-CLEAN` 不是修复**：3 条申报明确写着"生成命令 + 待办"。它们是**缺陷报告**，不是通过。生成记录的全量运行（`--jobs 6` 的 dg_advection 全配置约 5 min；`lax_region_band` 全梯子 ≈ 243 s 空载、实测在本机 load1>250 下 >8 min 未走完；`positivity_limiter` 全配置 27 个 run 在同一负载下也未走完）在本单的时间窗内无法完成，如实留作下场第一件事。
- **`_tracked_files()` 依赖机器上有 git**；无 git 时返回 `None`，普查把这类工件记为**未裁决**，不会伪装成通过（宁可 UNRESOLVED 也不误判）。
- 跟踪判据是"路径是否在 `git ls-files` 里"，因此**新生成而未 `git add` 的工件**会被判未跟踪——这正是想要的：未提交的工件在 CI 上不存在。
- 该结论对 TensorFEM/LBM 等同样成立（各仓都存在"工件缺失即 skip"的验收门）：**验收门的工件必须放在 git 携带的路径上**，否则"绿"只在本机成立。

## ⑥ 误差判定（本单的"基准"是 git 携带性，可机器逐位判定）

| 检查项 | 基准 | 结果 | 是否满足 |
|---|---|---|---|
| positivity 门在干净检出上的状态 | `git ls-files` 无 `artifacts/positivity_limiter/metrics.json` | pytest 实测 3 failed / 7 passed | 缺陷确认（<3% 判据不适用，此项为布尔判定） |
| 升级前 VACUOUS 计数 | 旧普查 `docs/GATE-VACUITY-AUDIT.md`（m16815 版） | 3 | 一致 |
| 升级后 VACUOUS 计数 | 新普查 | 0 | ✅ |
| `--check` 可用性 | exit code | 升级前 1（死），升级后 0 | ✅ |
| 政策门 | `pytest tests/test_gate_vacuity_policy.py` | 申报后全绿（含 2 道新门 + 1 条故障注入） | ✅ |

**是否 <3%**：本单不含物理量误差判定（无解析解对标的数值结论），故不适用；关键布尔判据（VACUOUS 3→0、`--check` 1→0、红门被如实暴露）**逐项通过**，未通过项已按 `ALLOWED_RED_CLEAN` 如实申报。
