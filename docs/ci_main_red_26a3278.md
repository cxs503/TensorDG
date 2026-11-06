# main CI 红单根因与修复：`TensorDG tests / test` @ `26a32789`

结论先行：**这不是红测，是一个 step 都没跑的取消**。`main@26a32789` 的 job
`test` 结论为 `cancelled`、`runner_id = 0`、`steps = []`——它请求的 runner 标签
在本仓根本没有匹配的监听器。修复分两半：本仓可修的是工作流的 **runner 标签请求**
与 **concurrency 分组**；不可修的是 runner 注册侧（见"边界"一节）。

## 1. 证据（全部由 GitHub Actions API / 本机 runner 诊断日志程序化取得）

### 1.1 红的那条 run

```
run  42152063804   head_sha 26a32789   event push   created 2026-11-06T02:42:53Z
job  127677394368  name "test"         labels ["ci-loong64"]
      conclusion cancelled   runner_id 0   runner_name ""
      created_at 02:42:53Z   completed_at 03:12:10Z        (29.3 min)
      steps: []                                            (0 个 step)
```

`runner_id = 0` 是关键字段：GitHub 只有把 job 派给某个 runner 后才会写 runner_id。
0 意味着**从未匹配到任何 runner**，因此不存在断言失败、不存在依赖装不上、
不存在超时——`cancelled` 来自排队期（本平台 30 min 排队巡查）而非执行期。

### 1.2 同一 commit 的上一版是绿的，且差异只有 CI 文件

| 项 | 值 |
| --- | --- |
| 绿 run | `42102175230` @ `558f03fe`（26a32789 的父提交） |
| runner | `ubuntu-latest` / `GitHub Actions 1000020207` |
| 起止 | 2026-11-05T23:55:08Z → 2026-11-06T01:30:16Z（95 min） |
| step 结果 | 20/20 全 `success`（含 14 条 benchmark 命令与 `pytest -q`） |

`git diff --stat 558f03f..26a3278` = `1 file changed, .github/workflows/tests.yml`。
也就是说：**绿的那次跑过的测试套件与红的那次逐字节相同**，唯一变量是
`runs-on: ubuntu-latest → ci-loong64`。这排除了"测试本身坏了"。

### 1.3 本仓的两个 runner 从没收到过 job

```
/opt/ci-runners/r-dg-1/.runner  agentName node099-ci-dg-1  gitHubUrl .../TensorDG  workFolder /opt/ciwork/dg-1
/opt/ci-runners/r-dg-2/.runner  agentName node099-ci-dg-2  gitHubUrl .../TensorDG  workFolder /opt/ciwork/dg-2
```

`_diag/Runner_20261106-020657-utc.log`：02:07:04Z `Listening for Jobs`；此后到
08:49:04Z 只有每 ~50 min 一次的凭据刷新（`Loading RSA key parameters`），
**没有任何一行 `Running job`**。`/opt/ciwork/dg-1`、`/opt/ciwork/dg-2` 至今为空
（`actions/checkout` 从未在其中展开过仓库）。

对照：同一部署批次的其他仓（r-bem-*、r-lbm-*）在同一时段正常领到 job 并在
`ci-loong64` 上跑绿（TensorBEM `421513…` 02:07/02:27 两次 success）。因此差异在
**本仓 runner 的标签集合**，不在 GitHub 侧、也不在代码侧。

### 1.4 队列侧的旁证

排查时刻 `2026-11-06T09:32Z` 查询 `TensorDG` 最近 8 条 run：全部 `conclusion = null`
（queued），最早一条 `42169630710` 已排队 **3 h 54 min**，仍无 runner。整条
ci-loong64 队列在该时段呈"有 listener、无派发"的形态，与 1.3 一致。

## 2. 修复（`.github/workflows/tests.yml`）

1. **runner 标签**：`runs-on: ci-loong64` → `runs-on: [self-hosted, linux]`。
   自托管 runner 注册时**默认**发布 `self-hosted` + 操作系统标签（除非显式
   `--no-default-labels`），因此这条请求对"只带默认标签"的注册和"默认标签 + 自定义
   `ci-loong64`"的注册**都能匹配**，而 `ci-loong64` 单标签只能匹配后者——本仓的
   listener 显然属于前者。仓库级 runner 只服务本仓，改成默认标签不会把别的仓的
   job 抢过来。
2. **concurrency**：`main` 上按 **commit** 分组，分支/PR 仍按 ref 分组：

   ```yaml
   group: ci-${{ github.workflow }}-${{ github.ref }}${{ github.ref == 'refs/heads/main' && format('-{0}', github.sha) || '' }}
   cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
   ```

   依据 2026-11-06 平台侧同形排查（`docs/pr188-ci-repair`）：GitHub 每个并发组只保留
   "1 running + 1 pending"，ref 级分组下后落的 merge 会取消前一个 commit 的 *pending*
   run，那个 commit 就永久红且 0 step。本仓原本**没有** concurrency 块，属于同一风险面，
   一并补上（`cancel-in-progress` 对 main 恒 false，慢用例信号不会被后续 push 杀掉）。
3. `timeout-minutes: 120`：hung job 必须自己失败，而不是占住两台池内 runner 之一
   （绿的那次在托管机上耗时 95 min）。

## 3. 判据（可复跑，`tests/test_ci_workflow_contract.py`，7 passed）

不是文本 grep，而是**执行**工作流里的表达式与匹配逻辑：

| 用例 | 断言 |
| --- | --- |
| `test_main_commits_get_distinct_concurrency_groups` | 两个 main commit 渲染出的组名不同 |
| `test_ref_wide_main_group_is_the_regression_this_guards` | **负控**：旧的 `${{ github.workflow }}-${{ github.ref }}` 渲染出相同组名（证明上条不是空断言） |
| `test_branch_and_pr_runs_stay_superseded` | 分支/PR 仍按 ref 分组（新 push 仍顶掉陈旧 run） |
| `test_cancel_in_progress_never_kills_main` | main 渲染 `False`，其余 `True` |
| `test_job_requests_labels_the_repo_runners_actually_publish` | `runs-on` 能被 `{self-hosted, linux}` 满足 |
| `test_custom_only_label_is_rejected_by_the_same_matcher` | **负控**：`["ci-loong64"]` 在同一匹配器下不满足 |
| `test_job_has_a_wall_clock_budget` | 存在 1..180 分钟的 `timeout-minutes` |

运行：`python -m pytest tests/test_ci_workflow_contract.py -q` → `7 passed in 0.08s`
（本机 loongarch64 / Python 3.11.6）。

## 4. 边界与未完成部分（honest）

- **runner 注册侧的标签**不在本工作树内（`/opt/ci-runners/r-dg-*` 由部署侧生成，
  标签存于服务端），我无权修改，也没能直接读出 DG listener 的实际标签集合
  （`config.sh` 注册时的 `--labels` 不留本地记录，V2 flow 用 OAuth 凭据而非可查
  token）。本文对"DG runner 缺 `ci-loong64` 自定义标签"的判断是**间接证据链**
  （1.1 `runner_id=0` + 1.3 listener 空转 + 同批次其他仓正常），不是直接读取。
  若部署侧随后给 r-dg-* 补上 `ci-loong64`，`[self-hosted, linux]` 依旧匹配，无需回退。
- **完整 pytest 套件的本机复跑没跑完**：本机（99 控制面）此刻 load average 789
  （约 20 个并行智能体的测试进程），560 条用例在 `-n 32` 下 40 min 才推进到 90%
  并在重尾（`test_vortex_2d` / `test_verification` / `test_tvb` 一带）停滞；集群
  n315/n330 的 torch 导入损坏（`ImportError: libflexiblas.so.3`），因此无法借集群
  加速。**该缺口由 GitHub 侧证据补足**：`558f03f` 上同一套件 20/20 step 全绿
  （1.2），而 `26a3278` 只改了 CI 文件。
- 修复的最终确认要靠"PR 门禁跑起来"：如果 `[self-hosted, linux]` 也被 30 min 排队
  巡查取消，那就证明本仓 runner 连默认标签都不可用，需要部署侧重注册——届时修法仍是
  工作流一行（`runs-on`），但根因在 runner 注册侧。