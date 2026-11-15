# 开发报告 · m22527 · CASE-DG-EULER-VORTEX-ERR 守恒审计：记录契约修复 + 分辨率负对照

> 任务：重开 TensorDG#632（原 PR 于 2026-11-05 由 owner 裁定关闭）· 管到底第 1 轮。
> 分支：`agent/m22527`，从当前 `main`（d6c4a66）切出。
> 本单**不修改** `tensordg/` 任何核心模块，因此与 TensorDG 核心模块上的在开 PR 无重叠。

## 1. 起点核查：PR#632 的内容已经在 main 上了

| 原 PR#632 文件 | 当前 main 状态 | 结论 |
|---|---|---|
| `benchmarks/case_euler_vortex.py` | 存在，含 `CONSERVATION_NAMES` / `run_p0` / `run_dg` / observer 守恒采样 | 已在 |
| `case_euler_vortex/conservation_results.json` | 存在（8 配置，DG 阶梯 + P0 阶梯） | 已在 |
| `docs/case_dg_euler_vortex_conservation.md` | 存在（109 行，六段式） | 已在 |
| `publish_case_dg_euler_vortex_conservation.py` | 存在（303 行） | 已在 |
| `tensordg/euler_2d.py`（+7/−1） | `conservative_state` 已导出 | 已在 |
| `tensordg/euler_dg_2d.py`（+22） | `conservation_drift` / 边界通量账本已在 | 已在 |
| `tests/test_euler_vortex_conservation.py` | 存在（118 行） | 已在，**但跑不过** |

同 mission（m20332）的更早合并 `#457` 已经把同一批文件带进 main。所以本轮的“重开”如果只是
把 #632 的 diff 再贴一遍，就是**空完成**——`git diff` 会是空的。因此先做起点审计，再决定增量。

## 2. 起点审计发现（这就是本轮真正要修的东西）

```
$ python -m pytest tests/test_euler_vortex_conservation.py -x
FAILED TestConservationVsSolutionError::test_coarse_mesh_conservation_still_exact
E   KeyError: 'max_relative_l2'
```

根因：`max_relative_l2` / `max_percent` / `gate_pass` 只在 `run_task()` 里附加，
而验收测试与发布脚本都直接调用 `run_p0()` / `run_dg()`。于是 **#632 的验收测试在 main 上是红的**，
而报告里的表格是发布脚本自己补字段后算出来的——两者用了两个不同的记录契约。

即：mission 的交付物（测试）与交付物（报告）互相不兼容，这属于**缺陷**，不是风格问题。

## 3. 决定与备选方案（experimental-design）

| 备选 | 做法 | 取舍 |
|---|---|---|
| A. 重贴 #632 diff | 原样重放 | **否**：diff 为空，空完成 |
| B. 只修 KeyError | `run_task` 外再补字段 | 可行但只修一行，审计本身仍无“分辨率”证据 |
| C. **B + 审计分辨率负对照 + 逐步漂移上限**（采用） | 统一记录契约；注入已知泄漏验证审计能读回；P0 路径给逐步上限 | 增量可测、单一功能点（守恒审计的可信度）、不动核心模块 |

**假设**：既然 ~1e-15 的读数与“度量写错/写瞎”在输出上不可区分，那么一个只会输出 1e-15 的守恒
研究，其证据强度取决于审计本身的**分辨率**是否被独立验证过。给出负对照即可把
“格式守恒”与“审计无效”分开。

**预登记阈值（先写后测，不因结果调整）**：

- 守恒容差 `CONSERVATION_TOL = 1e-12`（沿用原报告口径；比机器精度高 ~3 个量级，比任何物理级泄漏低 ~3 个量级）。
- 负对照：注入 δ 后审计读回值必须落在 `[0.5δ, 2δ]`（因子 2 的允许带，覆盖格式化与求和舍入）。
- 负对照判定：`δ > 容差 ⇒ FAIL`，`δ < 容差 ⇒ PASS`（容差为硬线，不留余量）。
- 逐步上限：`max_step_drift < 1e-12`，且必然满足 `max_step_drift ≥ 末态漂移`（上限是末态的父集最大值）。
- 解误差：案例池 5% 相对 L2；报告纪律追问的 3% 线逐项列出，不达标如实写。

**分析方式**：全部数字由脚本注入 markdown（`publish_*.py` 生成报告），并用测试反向校验
“报告里的每个漂移字符串都来自 artifact”，杜绝手抄。

## 4. 改动清单（6 个文件 + 3 个新工件）

| 文件 | 改动 |
|---|---|
| `benchmarks/case_euler_vortex.py` | 新增 `CONSERVATION_TOL`、`conservation_totals()`、`conservation_drift_from_totals()`、`conservation_summary()`、`_finalize_record()`；`run_p0` 增加**逐步漂移上限**（复用已有 observer，不改求解器）；`run_dg` 显式声明该路径无逐步钩子（内存 `NaN` / 落盘 `null`，不伪造）；两个入口统一附加 `max_relative_l2/max_percent/gate_pass/conservation_pass` |
| `tests/test_euler_vortex_conservation.py` | 修好原验收测试（记录契约）；新增 `TestRecordSchema`（两入口字段完备）、`TestPerStepCeiling`（逐步上限 + DG 诚实声明缺失）、`TestAuditResolution`（负对照：δ=1e-3/1e-6/1e-10 注入 + 容差线判别力 + NaN 判负）、`TestPublishedArtifacts`（工件是严格 JSON、报告数字与工件一致、负对照数据自洽）；模块级 `torch.set_num_threads(1)` 固定归约顺序并提速 |
| `publish_case_dg_euler_vortex_conservation.py` | 增加负对照 `run_audit_resolution()`（注入已知泄漏的阶梯）；新增图 3；表 1 增加“逐步漂移上限”列；新增表 2/3/4/5（逐项判定、负对照、逐配置守恒判定、解误差 vs 3% 线）；`_jsonable()` 让 artifact 成为严格 JSON（非有限值 → `null`）；固定单线程 |
| `case_euler_vortex/conservation_results.json` | 重新生成：8 配置，新增 `conservation_pass` / `conservation_tolerance` / `max_step_drift(_name/_step)` 字段 |
| `case_euler_vortex/audit_resolution.json` | 新增：5 个注入档位的审计响应 |
| `case_euler_vortex/fig_*.png` | 重新生成 2 张 + 新增 `fig_audit_resolution.png`（原两张图此前**从未入库**，本单补入） |
| `docs/case_dg_euler_vortex_conservation.md` | 由脚本重生成：新增表 3（负对照）、表 4/5（显式误差判定）、逐步上限列、注意点（DG 无逐步钩子、记录契约、线程数） |

未改动：`tensordg/euler_2d.py`、`tensordg/euler_dg_2d.py` 及任何其它核心模块（**热文件声明：零改动**）。

## 5. 验证

```
$ python -m pytest tests/test_euler_vortex_conservation.py -q
21 passed in 66.85s

$ python -m pytest tests/test_case_euler_vortex.py tests/test_case_euler_vortex_yee.py \
      tests/test_case_euler_vortex_err_mechanism.py -q
11 passed

$ python publish_case_dg_euler_vortex_conservation.py     # 6 个工件一次性重建
max_conservation_drift_dg = 1.0387e-14
max_conservation_drift_p0 = 2.8029e-15
max_step_drift_p0          = 2.8029e-15
all_pass = true, audit_resolution_pass = true, n_configs = 8
```

结果摘要（详见 `docs/case_dg_euler_vortex_conservation.md`）：

- 8 个配置的末态守恒漂移 ≤ `1.04e-14`，全部 < `1e-12`；
- P0 路径逐步漂移上限 `2.80e-15`（= 末态漂移，说明中途没有更大的瞬时涨落）；
- 负对照：δ ∈ {1e-3, 1e-6, 1e-9, 1e-12, 1e-13} 读回比 0.9992–1.0001，判定在容差线处翻转；
- 解误差：DG P2 16² = 0.0819%，DG P2 32² = 0.0127%，P0 验收档 128² = 2.8170%（均 < 3%）；
  故意取粗的 P0 16²/32² = 10.59%/7.80%（> 3%，如实标注：**精度**问题，不是守恒问题）。

## 6. 诚实披露与局限

1. **`tests/test_vortex_err_projection_baseline.py::test_measured_amplification_separates_conserved_from_accumulating`
   在干净 main 上就是红的**（`git stash` 后复跑仍然 FAIL），与本单改动无关，本单不修（避免跨功能点扩散）。
2. **DG 路径没有逐步钩子**：`solve_euler_dg_2d()` 无 observer 参数，逐步漂移上限只能给 P0。
   补齐需要改核心求解循环 `tensordg/euler_dg_2d.py`（热文件），本单刻意不做并在此声明。
3. **负对照是度量层对照，不是求解器对照**：它证明“审计能读回泄漏”，不能证明“求解器在任意
   条件下都守恒”；后者靠 8 个配置 + limiter 用例的经验证据。
4. **P0 的漂移仍是端点 + 逐步上限，而不是逐步明细**：observer 只保留极值，不存全时程序列
   （全序列会让 artifact 体积随步数线性增长）。
5. 全量 3432 条测试未在本机整体跑完（单机 120 s 上限），已递交集群作业做子集回归与全量回归，
   结果见报告与作业回执。

## 7. 复现

```bash
python publish_case_dg_euler_vortex_conservation.py
python -m pytest tests/test_euler_vortex_conservation.py -v
```