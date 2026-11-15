# dev report · CASE-LAX-DG：AO 臂闭环（闭合臂集最后一条未跑臂）

锚点 `CASE-LAX-DG`（台账原文「ref: 精确解｜容差 < 3 %」，status open）。
平台报告 id **20226**；工件 `docs/benchmarks/results/lax_scheme_variants/metrics.json`（节点 cj4120 生成）。

## 1 欠账是什么

预登记臂集 6 条臂（`base / cfl020 / cfl050 / pos_none / av / ao`）里，`ao` 从来没有数字：
上一单（m22357）递交的 `cj4072` 被读成「已递交集群、待完成」。查作业真身：
`cluster/jobs/cj4072.log` → `rc=1`、`output_tail = ModuleNotFoundError: No module named
'benchmarks.lax_scheme_variants'`，节点检出 head `caaa480`（远早于该模块入库），而该次递交
`stage_dir=None` → 跑的是节点上陈旧的 main 克隆。**代码不在节点上，「succeeded」的状态字只是包装成功。**
后果：`verdict.arms_not_run = ["ao"]`，「配置空间内不可达」这一结论缺一块。

## 2 做了什么

1. 复跑改带 `stage_dir=<worktree>`（tar 直达节点），命令
   `python3 -m benchmarks.lax_scheme_variants --variant ao --aggregate --figures --report`，**cj4120**（4 核、
   n315、AO 腿墙钟 189.1 s、4936 steps）。
2. 节点无 matplotlib（`--figures` 阶段 rc=1），图与文档改在本机从同一份 `metrics.json` 重生成；
   数值工件（`legs/ao.json`/`ao.npz`/`metrics.json`）直接取自节点 tarball，未改一个数。
3. 测试加固：`tests/test_lax_ao_arm_closure.py` 新增 `TestClosedArmSetComplete`（5 条）——把「臂集闭合」
   绑定到磁盘上的腿文件：每条预登记臂都要有腿记录、`metrics.rows` 与 `VARIANT_ORDER` 逐项对齐、
   `not_run`/`error` 只允许 `pos_none`（预登记崩溃属结果不属于缺口）、AO 行与 AO 腿文件逐位一致。
   这些测试在 AO 腿文件落盘前**是红的**（4 failed），落盘后全绿——证明断言的是不变量而非空跑。

## 3 结果（程序化注入）

| 臂 | 旋钮 | steps | L2 ρ | 可用 |
|---|---|---|---|---|
| `base`（控制） | — | 5109 | `3.115562e-02` | False |
| `cfl020` | cfl=0.20 | 8746 | `3.264379e-02` | False |
| `cfl050` | cfl=0.50 | 3578 | `3.099249e-02` | False |
| `pos_none` | positivity=none | — | 崩溃（`FloatingPointError`） | False |
| `av` | limiter=av | 4970 | `3.958778e-02` | False |
| **`ao`（本单补齐）** | **limiter=ao** | **4936** | **`5.627794e-02` = 5.6278 %** | **False** |

* AO 最差场量 = `density` `5.627794e-02`，裕度/容差 `1.8759`；
* 相对 `base`（3.1156 %）= **1.806×**；相对 `av`（3.9588 %）= 1.422×。
  即：**AO 限幅在钉扎腿上比无限幅和 AV 都更差**，上一单「AO 应落在 base 与 av 之间」的预判被数据否掉。
* 可用臂集合 `[]`、sweep 判定 **FAIL**、`arms_not_run = []`、
  `arms_errored = ['pos_none']`、`scope_complete = False`
  （False 只因为 `pos_none` 是预登记崩溃，不是缺口）、控制臂复现 `True`
  （最差相对差 `1.305e-06`）。
* 可行性界不变：floor `1.8270e-02`（占预算 60.90 %）、
  所需 solver 腿 `2.3795e-02`、实测 `2.4110e-02`、实测/所需 `1.0133`。

## 4 误差判定（vs 精确 Riemann 解，逐项）

| 关键量 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|
| L2 ρ @ `ao` (p3.N400) | `5.627794e-02` | 5.6278 % | **否** |
| L2 ρ @ `base` (p3.N400) | `3.115562e-02` | 3.1156 % | **否** |
| L2 ρ @ `av` (p3.N400) | `3.958778e-02` | 3.9588 % | **否** |

**结论 honest-FAIL**：补齐 AO 臂后，钉扎腿 `p3.N400` 在已提交格式的配置空间内**仍无可用臂**，
「配置空间内不可达」现在是有完整臂集支撑的结论，而不是「还缺一臂」的悬案。分辨率出路（`p3.N800` = 2.3149 %，
见 `docs/CASE-LAX-DG-resolution-requirement.md`）未动口径、未在本单重开。

## 5 坑 / 可复用知识

1. **集群作业状态字 ≠ 结果**：`cj4072` 平台侧 `succeeded`、节点日志 `rc=1`。判作业必须看
   `cluster/jobs/<jid>.log` 的 `rc` 与 `output_tail`（上一单把 rc=1 读成「待完成」，白等一轮）。
2. **依赖未入 main 的改动，递交必须带 `stage_dir`**：`ref=main` 会命中节点上（可能陈旧的）克隆，
   失败形态是「模块不存在」，与「代码写错」不可区分。
3. **节点与本地镜像不同构**：n315 没有 matplotlib。数值与绘图应拆成两步——数值在节点、出图在本地，
   否则 `--figures` 的一步 rc=1 会让人误以为整条命令失败（数值其实已经写盘）。
4. 墙钟不作排序判据（AO 189 s vs base 99 s 受排队/节点噪声影响）。

## 6 未完成 / 边界

* `tests -k lax` 全量（含 N=800 分辨率腿等重测）本机被 120 s 硬拦，已递交 **cj4169**（8 核，3600 s）
  到集群；本单落盘时其状态见 finish 记录。
* 本单不判断「更好的格式（子单元限幅、激波贴合、r-自适应）能否达标」；不动容差、不动范数、不动生产腿。
