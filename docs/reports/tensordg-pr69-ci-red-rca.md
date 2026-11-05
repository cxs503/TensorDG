# PR #69 CI 红根因分析与修复（m4078 / 第 2 轮）

> 任务：m4078（修复·TensorDG#69·第 2 轮）
> PR：#69，分支 `agent/m3666`，head `bed2559a2ad4178f24c97f25ce3fb18c7c87c1e4`
> 日期：2026-11-05
> 结论：**PR #69 自身内容无缺陷**；红测来自「分支基线过期」，修复 = 把 PR head 树对齐到当前 `origin/main`（PR 内容仍只有 2 个文件）。

## 1. 现象

PR #69（`m3666: matured mission deliverables`）只新增两个文件：

| 文件 | 内容 |
|---|---|
| `tests/test_tvb.py` | 18 条针对 `tensordg/tvb.py`（`minmod` / `troubled_from_differences`）的直接单元测试 |
| `docs/tvb_module_tests.md` | 对应验证报告 |

GitHub 上该 head 的 `test` job（push 事件，run `41857114140` / job `126668605448`）在
`pytest -q` 步骤失败，后续 11 个 benchmark 步骤全部被 skip：

```
4 failed, 402 passed, 4 skipped, 7 xfailed, 3 warnings in 8319.75s (2:18:39)
```

失败项与 PR #69 毫无交集：

```
FAILED tests/test_ns_heat_conduction.py::test_shipped_wrapper_runs_at_its_documented_defaults
       [XPASS(strict)] known defect: solve_heat_conduction_dg_2d raises FloatingPointError
FAILED tests/test_ns_heat_conduction.py::test_exact_steady_state_is_approximately_preserved
       [XPASS(strict)] known defect: even starting on the exact steady state the run collapses
FAILED tests/test_ns_heat_wall_closure.py::test_conduction_entry_point_still_crashes_on_the_documented_default
       Failed: DID NOT RAISE FloatingPointError
FAILED tests/test_ns_heat_wall_closure.py::test_interior_energy_residual_decays_but_wall_rows_do_not
       assert 1.1254006470996836 < 1.1
```

原始证据（自动抽取，无手抄）：`artifacts/pr69_ci_evidence/ci_job_summary.txt`。

## 2. 根因

`agent/m3666` 的基线是 `619167f`（#53 的 merge 提交），而 PR 的 base 分支 `main`
已经走到 `efd020d`（两者之间 main 侧累计改动 66 个文件）。四条红测全部属于
**attainted ratchet（棘轮测试）**：它们用 `xfail(strict=True)` / 「必须抛异常」的写法
把当时的**已知缺陷**钉成红灯，缺陷一旦被修好，这些钉子就会反向变红。

修复该缺陷的是 main 上的 `605e8c1`（m3759，NS 等温壁 m3139 壁面导数修复 +
正性保护），它同时做了两件事：

1. 把两条已经 XPASS 的严格 xfail 翻成正向断言
   （`test_shipped_wrapper_runs_at_its_documented_defaults`、
   `test_exact_steady_state_is_approximately_preserved`）；
2. 删除「入口必然崩溃」的负向钉子
   （`test_conduction_entry_point_still_crashes_on_the_documented_default`），
   并把壁面能量残差比的上界由拍脑袋的 `< 1.1` 改为按实测重标定的
   `< 1.1 * 1.06`（实测 16/32 比值 1.1254，旧界比实测低 2.3%，任何机器上必红）。

也就是说：**这些测试钉住的是一个在旧基线上真实存在、在 main 上已被修复的缺陷**。
PR #69 的 push 事件 CI 跑的是「旧基线 + 2 个新文件」的树，因此复现了旧钉子，
属于基线过期（stale base）而不是 PR 引入的回归。GitHub 的 `pull_request` 事件
（merge 提交 = main + 2 文件）则不出现这 4 条失败——这正是「本地/合并树绿、
head 树红」的分裂来源。

## 3. 修复

不做任何源码/断言「就地修补」（那等于把已修好的缺陷重新钉回红灯，是错的），
而是把 PR head 树对齐到当前 main：

* 工作树从 `origin/main`（`efd020d`）切出；
* 仅取入 PR #69 的两个文件（`tests/test_tvb.py`、`docs/tvb_module_tests.md`）；
* 结果树 = `main` + PR 两个文件，即 `agent/m3666` rebase 到 main 后的内容。

因此 PR #69 的 diff 仍然**只有 2 个文件**，未改动任何无关文件（那 66 个 main 侧
改动全部是 main 已经有的内容，不是本 PR 引入的）。

## 4. 验证（本地）

| 验证项 | 命令 | 结果 |
|---|---|---|
| PR 新增测试独立跑 | `pytest -q tests/test_tvb.py` | **18 passed**（此前同样 18 passed，说明 PR 内容本身绿） |
| 四条历史红测（同一棵树） | `pytest -q tests/test_ns_heat_conduction.py tests/test_ns_heat_wall_closure.py` | 见 `artifacts/pr69_ci_evidence/local_verification.txt`（全绿 / 只剩预期的 strict-xfail） |
| 树一致性 | `git diff --name-status origin/main HEAD` | 只有 `A docs/tvb_module_tests.md`、`A tests/test_tvb.py` |

判定：修复的充分条件是「head 树 ⊇ main 的测试钉翻转」，已满足；PR 内容零改动。

## 5. 注意点 / 局限

* CI job 单跑 `pytest -q` 耗时 2 小时 18 分（GitHub 免费 runner 2 核 + 全仓重计算
  测试），本地共用机也存在同样量级，因此本轮只对**受影响子集**（4 条历史红测 +
  PR 新测）做了确定性复现，未在本地重跑 400+ 条全量套件；全量绿的依据是
  main 在 `40455d90` 上 `test` job 已 `success`，而本树 = main + 2 个纯新增测试文件。
* 这类红测不值得单独立项修：**棘轮测试（strict xfail / 「必须崩」钉子）一旦缺陷被
  修复，过期分支上的 CI 必然红**。根治手段是分支合并前 rebase 到 main（或让
  pr-gate 在过期分支上先跑 merge-tree 检查），而不是改断言。

## 6. 误差判定

本任务为 CI/工程修复，无物理量基准，因此不适用「关键量 vs 基准误差 < 3%」判据；
唯一可量化判据是测试结果，逐项如下：

| 判据 | 基准 | 实测 | 是否达标 |
|---|---|---|---|
| PR 新增测试通过数 | 18/18 | 18/18 | 是（100%，0 失败） |
| 历史红测失败数（对齐后树） | 0 | 0 | 是（>3% 判据不适用，按 0 失败） |
| PR diff 文件数 | 2 | 2 | 是（0% 偏离） |