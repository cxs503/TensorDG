# TensorDG 上游差集巡更（m17042）

**结论先行**

1. **差集为空**：本仓 `HEAD` 与上游 `origin/main` 指向同一提交 `5b513c9`，
   `git rev-list --left-right --count` 得 ahead=0、behind=0，verdict=`EMPTY`。
   仓内不存在"未推上游的本地提交"，因此本单执行任务的第二分支：盘点最近 15 个上游首父提交的
   主题演化，并对每条落地做风险标注。
2. **最近 15 次落地里 3 次是"零内容落地"**（提交树 == 父提交树），
   提交信息却写着 `Merge #NNN ... via merge-master`——在台账里算"已合入"，实际交付物被门禁全部滤除。
3. **本地分支动物园与上游脱节**：扫描 247 个本地 `agent/*` 引用，
   128 个持有一共 910 个"上游根本不存在"的文件；
   去重后 803 个缺失路径中 **151 个仍被上游的 docs/tests 消费**
   （其中 `runs/lax_case_pool/metrics.json` 直接让 2 个验收门在干净检出上静默 skip，见 ⑤）。
   上游在线 `agent/*` 分支 153 个（`git ls-remote` 实测），本地 247 个 → 其中 94 个上游已删；
   **这 94 个里有 38 个持有 295 个 main 上不存在的文件**——
   即这些文件的唯一副本就在本工作树的本地 ref 里。
4. **读法本身是误差源**：60 个首父提交里有 20 个真合并，
   用 `git diff-tree`（不带 `-m`）统计会**整体丢 93 个文件**
   （占该窗口 367 个文件的 25.3%）。

---

## ① 案例介绍（巡更对象与参数表）

| 项 | 值 |
|---|---|
| 仓库 | TensorDG（PyTorch DG 求解器，12 仓租户制中的领域求解器仓） |
| 本地工作树 | `/root/tp-agent/work/m17042` |
| 分支 / HEAD | `agent/m17042` / `5b513c9`（= 上游 `origin/main`） |
| 上游解析方式 | `branch_upstream` → `origin/main` |
| 差集判据 | `git rev-list --left-right --count HEAD...upstream` |
| 落地窗口 | 上游首父链最近 15 个提交（2026-11-13 … 2026-11-13） |
| 分支审计范围 | 本地 `refs/heads/agent/*`（247 个） |
| 基线窗口 | 首父链最近 60 个提交（用于读法对照） |

本仓的可验证对象不是物理场，而是**版本历史本身**：落地提交 = "一次任务（mission）向 main 交付了什么"。
因此"参数表"是判据与口径，而不是几何/雷诺数。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

全部可由两个脚本重放：

```
python tools/patrol_upstream_diff.py --write --out artifacts/patrol_upstream_diff          # 窗口 15 + 分支审计
python tools/patrol_upstream_diff.py --limit 60 --no-branches \
       --write --out artifacts/patrol_upstream_diff/window60                              # 基线窗口
python tools/render_patrol_figures.py                                                     # 图
python tools/render_patrol_report.py                                                      # 本文件
```

四个口径，各自解决一个已知误读：

1. **首父差集**：每条落地一律用 `git diff <sha>^1 <sha> --numstat`，而不是 `git show` 的默认输出。
   实测 `20` 个真合并会被 `git diff-tree`（无 `-m`）读成 0 文件，
   这正是"任务明明交付了几千行，台账却显示空"的机制之一。
2. **零内容落地判据**：`first_parent_numstat` 返回空 **且** 该提交有父提交（根提交不算），
   才判 `R3-CONTENT-EMPTY-LANDING`；含父提交数的 `has_parent` 与 `files_plumbing` 一起入 JSON，可复核。
3. **分支对账按 blob**：对每个本地分支取 `merge-base(upstream, branch)`，把该分支改过的路径的 blob OID
   与上游同路径 blob 逐一比对，分 `LANDED / DIFFERS / ABSENT_ON_UPSTREAM`。
   不用 `git cherry`（patch-id）：squash/门禁改写会改变 patch-id，会对"确已合入"的工作误报未落地——
   实测 `agent/m16925` 在 patch-id 口径下是 UNLANDED，按 blob 口径是 CONTENT_LANDED。
4. **引用分类（消费者 vs 生产者）**：缺失路径若被上游 `*.md`/`*.py` 提到，用一次
   `git grep -F -f <paths>` 判定引用者属于 `docs/tests`（消费者，缺失即缺陷）还是
   `benchmarks/tools/scripts`（生产者，缺失是运行时产物，正常）。

## ③ 计算结果（最近 15 个落地的主题演化）

| # | sha | 日期 | 提交主题（mission / 通道） | 文件数 | 行数 | 主题分布 | 风险码 |
|---:|---|---|---|---:|---|---|---|
| 1 | `5b513c9` | 2026-11-13 | Merge #375 (m16938) via merge-master | 10 | +3543/−242 | validation-artifacts×4, benchmarks×2, docs×2, core-solver×1, tests×1 | R6, R7 |
| 2 | `04a7fd4` | 2026-11-13 | Merge #374 (m16925) via merge-master | 22 | +1987/−67 | scratch×16, docs×2, benchmarks×1, core-solver×1, tests×1, validation-artifacts×1 | R4, R6, R8, R9 |
| 3 | `c3b346e` | 2026-11-13 | Merge #373 (m16918) via merge-master | 12 | +1418/−159 | docs×6, benchmarks×2, scratch×1, core-solver×1, tests×1, tooling×1 | R4, R6 |
| 4 | `25c48db` | 2026-11-13 | Merge #372 (m15447) via merge-master | 0 | +0/−0 | —（0 文件） | R3 |
| 5 | `95dc7f8` | 2026-11-13 | Merge #371 (m16928) via merge-master | 4 | +330/−1 | core-solver×2, docs×1, tests×1 | — |
| 6 | `771aafb` | 2026-11-13 | Merge #370 (m15389) via merge-master | 0 | +0/−0 | —（0 文件） | R3 |
| 7 | `65c3fc8` | 2026-11-13 | Merge #369 (m16875) via merge-master | 16 | +4302/−73 | docs×7, tests×5, benchmarks×2, tooling×2 | R5, R6, R8 |
| 8 | `402dcbb` | 2026-11-13 | Merge #367 (m16859) via merge-master | 6 | +1331/−1 | validation-artifacts×3, docs×1, core-solver×1, tests×1 | R6, R9 |
| 9 | `941fd5c` | 2026-11-13 | Merge #366 (m15398) via merge-master | 0 | +0/−0 | —（0 文件） | R3 |
| 10 | `de4c1e5` | 2026-11-13 | Merge #361 (m16803) via merge-master | 6 | +1424/−17 | benchmarks×2, docs×1, core-solver×1, tests×1, tooling×1 | R6 |
| 11 | `ac49329` | 2026-11-13 | Merge #368 (m16874) via push-merge | 11 | +1668/−6 | core-solver×4, validation-artifacts×4, benchmarks×1, docs×1, tests×1 | R2, R6 |
| 12 | `75f7351` | 2026-11-13 | Merge #364 (m16867) via push-merge | 1 | +104/−0 | docs×1 | R2, R5 |
| 13 | `5ee54fd` | 2026-11-13 | Merge #363 (m16814) via merge-master | 19 | +3158/−3 | validation-artifacts×14, docs×2, scratch×1, tests×1, tooling×1 | R4, R5, R6, R9 |
| 14 | `8d43758` | 2026-11-13 | Merge #362 (m16805) via push-merge | 1 | +72/−0 | docs×1 | R2, R5 |
| 15 | `40ba2e0` | 2026-11-13 | Merge #360 (m16815) via push-merge | 5 | +188/−10 | docs×3, tests×1, tooling×1 | R2, R5, R8 |

聚合（`patrol.json → summary`）：落地 15 次、文件 113 个、
新增 19525 行 / 删除 579 行；
主题按改动文件计 {"docs": 28, "validation-artifacts": 26, "scratch": 18, "tests": 14, "core-solver": 11, "benchmarks": 10, "tooling": 6}；
落地通道 {"merge-master": 11, "push-merge": 4}。

**主题演化读数**：窗口内交付重量集中在校验资产与文档（docs 28 文件、
validation-artifacts 26 文件），
核心求解器 `tensordg/` 仅 11 文件；
5 次落地完全没有触碰核心代码——
这与"以可复现基准驱动增长"的章程（docs/PROJECT_CHARTER.md）一致，但也意味着
**近期增量主要在证伪/复核面，而非新离散格式能力**。

## ④ 风险标注

| 风险码 | 命中次数（窗口 15） | 含义 |
|---|---:|---|
| R6-SCOPE-CREEP | 8 | 单次落地横跨 ≥4 个顶层主题 |
| R5-NO-CORE-CHANGE | 5 | 落地不触碰 `tensordg/` 核心求解器 |
| R2-MERGE-CHANNEL-PUSH_MERGE | 4 | 落地未经 `merge-master` 通道（旁路通道，审计链更短） |
| R3-CONTENT-EMPTY-LANDING | 3 | 落地提交的树 == 父提交树：记为已合入但零内容 |
| R4-SCRATCH-IN-HISTORY | 3 | 把 `scratch/` 或根目录散脚本带进历史 |
| R8-BINARY-FIGURE-ASSET | 3 | 落地新增 `*/figures/*.png` 等二进制资产（门禁常滤除） |
| R9-ARTIFACT-NO-SHA256 | 3 | 验证产物落地却未同步 `*.sha256` 指纹 |
| R7-BIG-ARTIFACT-CHURN | 1 | 单文件 ≥2000 行的验证/运行产物写入 |
| R1-GATE-FILTER-DECLARED | 0 | 落地提交对自己的描述里就写了被门禁滤除（自证丢失） |

**门禁滤除是常态而非意外**：`patrol.json → gate_filter_declared` 显示最近 200 个上游提交里
有 **30** 个在自己的提交信息里写明"文件被门禁滤除"
（最近一例 `f9489be`：
`m16874: matured mission deliverables（3 个文件被门禁滤除）`）。
R1 在窗口内计数为 0，只是因为首父链上看到的是**落地提交**，自证语句留在任务自己的工作提交（第二父）上——
因此 R3（零内容落地）不是孤例，而是同一个门禁行为的极端表现。

**R3 零内容落地明细**（最高优先级：台账与事实背离）

| sha | 提交主题 | 日期 |
|---|---|---|
| `25c48db` | Merge #372 (m15447) via merge-master | 2026-11-13 |
| `771aafb` | Merge #370 (m15389) via merge-master | 2026-11-13 |
| `941fd5c` | Merge #366 (m15398) via merge-master | 2026-11-13 |

**R7 大数据写入**：`5b513c9` 单次落地把
`validation/artificial_viscosity/profiles.csv` 等物证一次性增加 ~2.5k 行原始剖面（可追溯，但仓库体积增长快）。

## ⑤ 分支动物园审计（上游已删分支、本地仍持有的内容）

| 状态 | 分支数 |
|---|---:|
| LOST_FILES | 128 |
| MERGED_REF | 42 |
| CONTENT_LANDED | 37 |
| DRIFTED | 28 |
| NO_MERGE_BASE | 12 |

`LOST_FILES` 样例（缺失文件仅上游无、分支有）：

| 分支 | tip | 上游缺失 | 内容漂移 | 缺失文件（前 2） |
|---|---|---:|---:|---|
| `agent/m14501` | `dabd92e` | 45 | 0 | `artifacts/adv2d_var_tests.log`, `artifacts/adv_diff_tests.log` … |
| `agent/m14549` | `21be271` | 34 | 7 | `artifacts/curved_geometry/evidence.json`, `artifacts/curved_geometry/figures/annulus_field.png` … |
| `agent/m16133` | `97f56fe` | 32 | 0 | `runs/m16133_audit/_source_m16072/author_doc.md`, `runs/m16133_audit/_source_m16072/raw/dg-av-p2n100-eps0.2.json` … |
| `agent/m16319` | `2b8ae9e` | 30 | 0 | `runs/audit_m16248/api.py`, `runs/audit_m16248/branch/benchmarks/p_refinement_exponential.py` … |
| `agent/m15303` | `ffb5c2f` | 27 | 0 | `artifacts/burgers_sine_01/burgers_1d_convergence.png`, `artifacts/burgers_sine_01/burgers_1d_profile.png` … |
| `agent/m15398` | `7325e96` | 27 | 0 | `artifacts/burgers_sine_benchmark/fig1_profile_comparison.png`, `artifacts/burgers_sine_benchmark/fig2_convergence.png` … |

**消费者引用的缺失路径**（缺失即缺陷，按持有分支数排序）：

| 路径 | 持有分支数 | 上游引用者 |
|---|---:|---|
| `runs/lax_case_pool/profiles.npz` | 5 | `benchmarks/lax_case_pool.py`, `docs/CASE-LAX-DG-artifact-gap-analysis.md`, `docs/CASE-LAX-DG-artifact-generation.md` … |
| `runs/lax_case_pool/metrics.json` | 4 | `benchmarks/lax_case_pool.py`, `docs/AUDIT-m16606-CASE-LAX-DG-R3-region-rule.md`, `docs/CASE-LAX-DG-artifact-gap-analysis.md` … |
| `runs/shu_osher_case_pool/metrics.json` | 3 | `audit_report_m15297.md`, `benchmarks/report_shu_osher_case_pool.py`, `benchmarks/shu_osher_case_pool.py` … |
| `runs/shu_osher_case_pool/profiles.npz` | 3 | `audit_report_m15297.md`, `benchmarks/shu_osher_case_pool.py`, `docs/board_audit/make_verdict_m15550.py` |
| `runs/lax_case_pool/summary.md` | 3 | `benchmarks/lax_case_pool.py`, `docs/CASE-LAX-DG.md` |
| `artifacts/gate_vacuity/gate_vacuity.json` | 3 | `docs/AUDIT-m16815-gate-vacuity-red-main.md`, `docs/AUDIT-m16875-gate-vacuity-tracked-artifacts.md` |
| `runs/lax_case_pool/figures/lax_profiles.png` | 3 | `docs/CASE-LAX-DG-artifact-gap-analysis.md`, `docs/CASE-LAX-DG-artifact-generation.md`, `docs/CASE-LAX-DG.md` |
| `runs/lax_case_pool/figures/lax_convergence.png` | 3 | `docs/CASE-LAX-DG-artifact-gap-analysis.md`, `docs/CASE-LAX-DG-artifact-generation.md`, `docs/CASE-LAX-DG.md` |
| `runs/lax_case_pool/figures/entropy_wave_order.png` | 3 | `docs/CASE-LAX-DG-artifact-gap-analysis.md`, `docs/CASE-LAX-DG-artifact-generation.md`, `docs/CASE-LAX-DG.md` |
| `artifacts/evidence_integrity_audit.json` | 3 | `docs/evidence_integrity_audit.md`, `tests/test_evidence_integrity.py`, `tools/audit_evidence_integrity.py` |

运行时复核（本机干净检出，`runs/lax_case_pool/metrics.json` 确实不存在）：

```
$ python -m pytest tests/test_lax_case_pool.py -q
9 passed, 2 skipped, 1 warning in 26.00s
```

`tests/test_lax_case_pool.py:256,287` 的 `@pytest.mark.skipif(not METRICS.exists(), ...)` 因此在干净检出上恒 skip：
断言的数字（Lax 案例池剖面误差）**没有在该检出上被判过**。这与 `tools/audit_gate_vacuity.py`
的模块级结论不矛盾——该工具按模块定性，而此处是模块内**单测级**空转，属其粒度外的残余盲区
（同现象另一例：`tests/test_evidence_integrity.py:79`）。

## ⑥ 注意点（局限 / 适用范围）

- **差集空 ≠ 无风险**：上游同步只说明"本地没落在上游之后"，不说明"上游的内容完整"。R3 与 ⑤ 的两类问题
  都发生在**上游这一侧**。
- **本地 247 个 `agent/*` 引用是工作树态，不是上游态**。本仓 `remote.origin.fetch` 只映射 `refs/heads/main`，
  因此 `origin/agent/*` 从不落地为本地跟踪引用（"`git fetch --prune` 后为 0"是 refspec 现象，**不是**上游删了分支）；
  真实上游态要用 `git ls-remote --heads origin 'refs/heads/agent/*'` 测：本次实测 153 个在线。
  本地 247 个引用里 94 个上游已删，
  其中 38 个共持有 295 个 main 上不存在的文件。
  本次审计把它们当作"证据是否真的到了 main"的探针，而不是把它们当作待合并工作。
- **`ABSENT_ON_UPSTREAM` 不等于丢失**：产物类路径（`artifacts/*.png`、`runs/*/raw/*.npz`）本就由
  benchmark 在运行时生成，门禁滤除是设计行为。真正可判缺陷的是**消费者引用**的那一类（本单 151 条）。
- **图不是场量云图**：本单对象是版本历史，无 PDE 场量；①-④ 图给的是落地主题构成与读法偏差、
  分支状态与缺失路径排名（见下图）。此处显式声明替代理由。
- **12 个分支 `NO_MERGE_BASE`**（`git merge-base` 为空，孤立历史）未做内容对账，属未判读项，不按"已检查"计。

## ⑦ 误差判定（有无基准、是否 <3%）

本单**没有解析解/实验基准**——被测量是版本历史计数，不是物理量，故"<3% 误差"这一判据不适用；
按纪律显式声明原因，并改用**可机器复核的对照差值**：

| 对照 | 数值 | 相对差 | 判定 |
|---|---|---:|---|
| 读法对照：plumbing（`diff-tree` 无 `-m`）vs 首父 diff，60 提交窗口丢失文件 | 93 / 367 | 25.3% | 远超 3%：读法不可互换，必须以首父 diff 为准 |
| 零内容落地占比（窗口 15） | 3 / 15 | 20.0% | 台账口径失真，需在台账侧修正 |
| 缺失路径被消费者引用占比 | 151 / 803 | 18.8% | 缺陷类缺失不可忽略 |
| 差集核对（HEAD vs 上游） | ahead=0 / behind=0 | 0 | **精确一致**（唯一"误差 0"的项，因为它是 git 的确定性输出） |

## ⑧ 建议的下一步（最小增量）

1. 把 `tools/patrol_upstream_diff.py --check`（零内容落地或 `LOST_FILES` 存在即非零退出）挂到巡更节点，
   让"零内容落地"在合并当天就变红，而不是靠人事后读提交信息。
2. 给 `tests/test_lax_case_pool.py` 的物证缺失补**硬断言**（与 `tests/test_dg_advection_bench.py:149`
   的既有写法一致），消灭单测级空转。
3. 把 `consumer_referenced_count` 纳入 `tools/audit_evidence_integrity.py` 的孤儿判定：
   被 docs/tests 引用却不在上游 = 不合格物证，而不是"待补"。

图（由 `tools/render_patrol_figures.py` 生成）：

- `docs/figures/patrol_m17042/fig_landing_themes.png`：左=每次落地的主题构成（红色标注零内容落地）；
  右=plumbing 读法与首父读法的对照总量。
- `docs/figures/patrol_m17042/fig_branch_zoo.png`：左=分支状态分布；右=消费者引用的缺失路径排名。

机器可读证据：`artifacts/patrol_upstream_diff/patrol.json`、`artifacts/patrol_upstream_diff/window60/patrol.json`、
`artifacts/patrol_upstream_diff/patrol.md`。
