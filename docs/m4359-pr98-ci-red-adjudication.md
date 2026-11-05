# PR #98 (agent/m3640) CI 红裁定 — 陈旧棘轮副本，非算法缺陷

mission m4359 · 2026-11-05 · 仓库 TensorDG · PR #98 head `1097b52`

## 1. 现象

CI 报 `test=failure` 的那次运行是 **push 事件**、跑在分支树本身（branch-only tree）上：

| run id | event | tree | 结论 |
|---|---|---|---|
| 41864712278 | push | 分支树 `1097b52` 单树（未与 main 合并） | **failure**（job 126695846024，step `Run pytest -q`） |
| 42001266988 | pull_request | GitHub 自动生成的 merge tree（`1097b52` × main `4ae1597`） | success |
| 42018092166 | pull_request | 同上 | success |

即：**同一 head 的合并树已经绿了两次**，红的只是分支树自己。合并树才是合并后的真实状态，
也是 pr-gate 判据（`merge_governor.py` L134-152 取 head sha 上最新 workflow run 的 conclusion）。

## 2. 根因（机器可判，不靠叙述）

PR #98 的分支基点 `619167f`（Merge #53）早于 main 上 m3139 壁面修复的**重裁**（`c051cef`，m3694/m4083）。
分支树把**已修好的求解器**与**修复前的棘轮测试副本**配在一起：

```
base blob  tensordg/ns_dg_2d.py
  origin/main : 39267a40a895f7842fc6bd244799ecbb8dccaf46
  推动基点 619167f: 39267a40a895f7842fc6bd244799ecbb8dccaf46   ← 逐字节相同（m3139 修复已在）
分支相对 main 的 ns_dg_2d.py 差异只有 m3640 自己的 callable-boundary 一处：
  + def stage_boundary(time_value) / boundary.values() 上的 callable 分支 / meta 序列化
```

而两份测试文件在分支上是**陈旧副本**（main 已重裁）：

| 文件 | 分支 vs main |
|---|---|
| `tests/test_ns_heat_conduction.py` | 47 行差异（main 删除 `strict xfail` 段，改正性/完成性断言） |
| `tests/test_ns_heat_wall_closure.py` | 74 行差异（main 把壁面残差比 pin `1.1` → `1.2/1.1` + 单调性检查） |

陈旧副本遇到已修好的求解器 → `XPASS(strict)`、`assert 1.1254 < 1.1` 一类必红。
这与 m4083 对 PR #74 的裁定同型（同一对文件、同一机理，见 `docs/m4083-pr74-ci-fix.md`）。

## 3. 本地复现（分支树，非合并树）

```
git worktree add --detach /tmp/bo1097 1097b52
cd /tmp/bo1097 && PYTHONPATH=. python3 -m pytest -q -n 4 -p no:randomly \
    tests/test_ns_heat_conduction.py tests/test_ns_heat_wall_closure.py
→ ...x.x..F..F      # 两个用例失败（xx=非 strict xfail 通过项，F=棘轮断言失败）
```

原始日志摘录：`artifacts/m4359/pr98_branch_tree_red.txt`。

对照（合并树，本工作树 = `origin/main` + PR #98 内容）：同一组测试文件取 main 版本，
m3640 新增的 `tests/test_ns_transient_wall_heat.py` 在本机（loongarch64，torch float64）
`8 passed in 121.04s`，其中三个关键判据的余量：

| 判据 | 实测 | 阈值 | 余量 |
|---|---|---|---|
| 壁面热流相对误差（P1, 16 cells, dT=0.005, t=0.7） | 0.405 % | < 3 % | 7.4× |
| 剖面相对 L2 | 0.00021 | < 0.01 | 47× |
| 网格 8→16 观测阶 | 0.806 | > 0.6 | 1.34× |
| `dT/T0→0` 外推误差 | 0.265 % | < 3 % | 11× |

（数据来自分支内 `validation/ns_dg_2d_transient_wall/summary.json`，程序化提取，非手抄。）

## 4. 修复方式与它为什么"零代码改动"

merge tree 里 main 版本的两份测试文件天然取胜（分支没有改它们），所以 PR #98 **不需要分支提交**：
红只存在于"分支单树"这一从未被合并的中间态上。修复 = 以当前 main 为基点重新取树，即 GitHub
已经算出的 merge tree；该 merge tree 的两次 pull_request 运行均 success。

本工作树（`agent/m4359`）因此只携带本裁定文档与证据，不含 `agent/m3640` 的 12 个文件
（与其内容的重复投影会制造 add/add 噪声）。

## 5. 注意点 / 局限（如实）

* **未能自行完成合并**：把 PR 合进 main 属特权动作（平台纪律：agent 分支推送由 tp_pusher，
  main 由 pr-gate/merge_governor 记账）。本沙箱的策略门直接拦截了向 main 的写操作，
  故本单交付到"CI 绿 + 裁定结论 + 证据"，合并记账留给 pr-gate。
* **合并此刻被 CI 队列卡住**：`1097b52` 上另有一个 pull_request 运行（run 42059360789）自
  2026-11-05T15:52Z 起处于 `queued`（GitHub Actions 排队积压，同时 main 的 push run 42023293691
  也自 11:58Z 起排队），因此 `mergeable_state=unstable`（`mergeable=true`）。队列消化前
  pr-gate 只会看到"最新 run 未完成"。另：`/root/tp-agent/merge_governor.log` 最后一条为 08:50Z，
  治理守护进程在观测时刻无新活动，若队列长期不消化需人工触发合议。
* **分支树红不是本单引入**：`1097b52` 的 push 运行自 2026-11-04T16:10Z 起就红，与 m4359 无关。
* 差分只覆盖上述两个文件；若未来 main 再次重裁这两份棘轮，陈旧分支会以完全相同的方式复红，
  根治手段是让 agent 分支在推送前 rebase 到 main tip（tp_pusher 已按 main-tip 嫁接树基，
  但**分支自身树**仍保留旧测试副本，这是本事故的结构性来源）。

## 6. 误差判定

本单是 CI/语义一致性裁定，无可对标实验或解析解的可比量，故不适用"误差 < 3%"判据；
唯一可量化的数值是上表 §3 中 m3640 基准自身的 4 项判据，实测余量分别为 7.4×/47×/1.34×/11×，
**全部满足 < 3 %（或对应阶数/阈值）要求，且均非临界通过**（最小余量 1.34× 在观测阶一项）。