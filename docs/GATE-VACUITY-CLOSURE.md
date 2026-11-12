# 空转门禁闭环：把「工件缺失即 skip」改成「缺失即红」并搬进 tracked 证据树

- **锚点**：`NEW: 空转门禁闭环（2 个 VACUOUS 门禁归零 + 审计器 cwd 缺陷修复）`
  （`feature_ledger` 无对应条目；建议新增 `GATE-DG-VACUITY-CLOSURE`，验收见 §6）
- **本单**：m16795（verifier 席自找活）· 仓 `TensorDG` · 分支 `agent/m16795`
- **被审/被修对象**：`tools/audit_gate_vacuity.py`（审计器本体）、
  `tests/test_shu_osher_case_pool.py`、`tests/test_dg_advection_bench.py`、
  `tests/test_gate_vacuity_policy.py`、`benchmarks/shu_osher_case_pool.py`、
  `tools/run_dg_advection_bench.py`
- **独立性限制**：审计器与被审门禁同仓同架构（loong64 / torch 2.7.1 / numpy 2.4.6）；
  独立性体现在**测量路径**（静态 AST 普查 + 门禁逐条实跑）而非实现路径。物理数值侧
  本单不新增算例，只用已提交的 `validation/repro_shu_osher/artifacts/raw/*` 重建汇总。

## 1. 症状（动手前的基线，可复现）

`python -m pytest tests/test_gate_vacuity_policy.py` 在**本单开始时的树**上：

```
5 failed, 5 passed
FAILED test_committed_census_matches_a_fresh_scan
FAILED test_report_is_rendered_from_the_census
FAILED test_no_unregistered_vacuous_gate
FAILED test_allowance_list_is_self_cleaning
FAILED test_census_covers_every_test_module
```

即：**空转门禁的机器门自己是红的**——它的普查工件
`artifacts/gate_vacuity/gate_vacuity.json` 从未落进 `main`，而同一条任务写的
`docs/GATE-VACUITY-AUDIT.md` 落了。这是「合并门过滤工件」的现场证据（F1）。

## 2. 根因（三类，全部有工件证据）

| # | 根因 | 证据 |
|---|---|---|
| F1 | **`artifacts/` 与 `runs/` 事实上不被 git 跟踪**，证据写进去就等于没写 | `git ls-files runs \| wc -l` = **0**（整个仓库历史）；`git ls-files artifacts` = **1**（一个 .log）；`git ls-files validation` = **240** |
| F2 | 门禁用「工件缺失 → `pytest.skip`」守护验收断言 | `tests/test_shu_osher_case_pool.py` L65/L113、`tests/test_dg_advection_bench.py` L147 |
| F3 | 门禁的工件路径是 **cwd 相对**（`Path("runs/...")`），审计器也用 `Path.cwd()` 解析 | 审计器 `_resolve()` / artifact 循环；基线 `cwd_dependent_artifacts` 共 **8** 个常量、**6** 个模块 |

F3 的后果不是理论的：审计在别的工作树里跑时，`Path("benchmarks/data/…")` 被解析成
**那个工作树的 cwd**，于是 `tests/test_shu_osher_case_pool.py` 被误判成 VACUOUS
（`missing_artifacts` 里出现 `/root/tp-agent/work/m16367/...`），而该文件其实是已提交的
tracked 输入。**审计器自己也有 cwd 依赖**。

## 3. 修法（最小改动、可验证）

1. **证据搬进 tracked 树**（F1）：汇总工件改落 `validation/`：
   - `validation/shu_osher_case_pool/{metrics.json,profiles.npz,raw/*.npz}`
   - `validation/dg_advection_bench/dg_advection_bench.json`
   - `validation/gate_vacuity/gate_vacuity.json`（普查 JSON 本体）
2. **生成器路径锚定仓库根**（F3）：`benchmarks/shu_osher_case_pool.py` 新增
   `ROOT = Path(__file__).resolve().parents[1]`，`OUT = ROOT/"validation"/"shu_osher_case_pool"`；
   `tools/run_dg_advection_bench.py` / `tools/render_bench_tables.py` 同步。**从任意 cwd 跑结果一致**。
3. **skip → 硬断言**（F2）：两个门禁改成 `assert <工件>.exists(), "<一键复现命令>"`，
   缺失即红并在报错里给出重生成命令。`ALLOWED_VACUOUS` 里两条登记随之退役（自清理门强制）。
4. **审计器本身去 cwd 依赖**（F3）：`_resolve()` 与工件存在性判定改用 `ROOT` 而非 `Path.cwd()`；
   `JSON_OUT` 从 `artifacts/gate_vacuity/` 搬到 `validation/gate_vacuity/`。
5. **代价可控的重建路径**：新增 `python -m benchmarks.shu_osher_case_pool seed`
   —— 把已提交的阶梯原始腿（`validation/repro_shu_osher/artifacts/raw`，同 key 同 meta）
   拷进证据目录，并用 `_verify_seed()` 逐腿复核 `meta`（ncells/degree/cfl/limiter），
   再 `merge` 重建 `metrics.json`。**不重算 9 腿阶梯（数十分钟）、不抄任何数字。**

## 4. 结果（程序化注入，零手抄）

`validation/shu_osher_case_pool/metrics.json`（本单重建，9 腿）：

| ncells | p | rel L1 vs FV-WENO5 参考 (%) | <5% 判据 |
|---|---|---|---|
| 200 | 1 | 2.462 | ✅ |
| 400 | 1 | 1.544 | ✅ |
| 800 | 1 | 0.568 | ✅ |
| 200 | 2 | 2.426 | ✅ |
| 400 | 2 | 1.514 | ✅ |
| 800 | 2 | 0.556 | ✅ |
| 200 | 3 | 2.432 | ✅ |
| 400 | 3 | 1.520 | ✅ |
| 800 | 3 | 0.662 | ✅ |

`seed --verify-only`：**11/11** 腿 meta 与阶梯定义一致（9 DG + 2 FV 参考）。

空转普查（本单前后对照，`tools/audit_gate_vacuity.py`）：

| 判级 | 修前 | 修后 |
|---|---|---|
| VACUOUS | **2** | **1**（仅 `test_lax_region_band.py`，已带理由登记） |
| LATENT | 7 | 6 |
| HARDENED | 9 | 11 |
| UNRESOLVED | 9 | 9 |
| PLAIN | 117 | 117 |
| 缺失工件 | 2 | 1（`runs/lax_region_band/metrics.json`） |
| cwd 依赖常量 | 8 | 7（`test_shu_osher_case_pool.py::COMMITTED_REFERENCE` 已消除） |

`HARDENED +2` = 两个原 VACUOUS 模块；`VACUOUS 2→1` = 闭环两个中的两个（一个等价替换为
`lax_region_band` 的显式登记，那是 m16606 的工件，重算 244 s，不在本单预算内）。

## 5. 注意点（坑与局限）

- **坑**：`benchmarks/shu_osher_case_pool.py all-ladder --reverse` 并非「便宜优先」——
  `reversed(LADDER)` 的第一腿是 `(800, 3)`，本机单腿 wall > 3 min（OMP 超订 + 机器被
  另一条 1 天的 pytest 占 247% CPU）。要重算整阶梯请直接上 `run_cluster`。
- **坑**：`runs/` 在合并门被滤除这条事实，只能从 `git ls-files` 的统计反推（没有显式的
  过滤清单）；因此本单把「证据必须落 `validation/`」写成了生成器默认行为，而不是靠人记住。
- **局限 1**：本单**没有**重跑 9 腿 DG 阶梯与 FV 参考（预算/算力），
  `metrics.json` 由**已提交的**原始腿重建；这些腿的物理正确性由
  `tests/test_shu_osher_case_pool.py` 的解析锚（Rankine–Hugoniot、精确质量收支、
  unshocked sine）与 `validation/repro_shu_osher/evidence.json` 承担，不是本单新增证据。
- **局限 2**：剩余 7 个 cwd 依赖常量（`test_ns_*`、`test_oblique_shock_theta_beta`、
  `test_shu_osher_1d`）未改，仍被普查逐条列出——诚实登记，未算通过。
- **局限 3**：`test_lax_region_band.py` 仍是 VACUOUS（带理由登记），其修复路径与本单相同
  （重算 + 落 `validation/`）。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 本单值 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 空转普查 VACUOUS 计数 | 目标 0（未登记） | 已登记 1 / 未登记 0 | —（计数，非连续量） | 不适用；**未登记 VACUOUS = 0 达标** |
| shu_osher 最粗 P3 rel L1 | 5% 验收线 | 2.432% | 相对验收线余量 2.57 pp | ✅ 是 |
| shu_osher 最细 P3 rel L1 | 5% 验收线 | 0.662% | 余量 4.34 pp | ✅ 是 |
| seed 复核通过率 | 11/11 | 11/11 | 0 | ✅ 是 |
| 门禁策略测试 | 全绿 | 17 passed | — | ✅ 是 |

无物理基准的新增量的部分（门禁/审计器）不适用「<3%」判据，已按计数/布尔量判定并说明。

建议新增总账条目：`GATE-DG-VACUITY-CLOSURE`（域：TensorDG/质量门禁；
验收：`ALLOWED_VACUOUS` 中未登记 VACUOUS=0 且 `tests/test_gate_vacuity_policy.py` 全绿，
cwd 依赖常量逐条清零）。
