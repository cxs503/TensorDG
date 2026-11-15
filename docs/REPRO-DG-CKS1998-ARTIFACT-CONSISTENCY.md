# REPRO-DG-CKS1998 · 证据可核验性修复（artifact-consistency bundle）

- 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, RKDG benchmark set，SIAM Review 43(1)/JCP 141）
- 仓/分支：TensorDG · `agent/m21336` · 任务 m21336（自找活 tp-impl-7 平台席）
- 性质：**验证欠账闭合 / 缺陷修复**（不是新算法），针对审计 m15345 已记录但未修完的三处缺陷。

## 1. 修了什么（每项都有可失败→通过的守卫测试）

| # | 缺陷（出处） | 现象 | 修复 | 守卫测试 |
|---|---|---|---|---|
| D1 | 审计 m15345 **M1** 的复发 | `validation/repro_cks1998/evidence.md` 的 sha256 清单里 3 行 `artifacts/full/*.png` 在 git 中不存在 → 新克隆无法核验任何一行（清单本身不可证） | `tools/render_cks1998_evidence.py` 只把 **git 已跟踪** 文件写进 sha256 清单；本地生成但未入库的文件另设「generated locally, not committed」节，明说不入清单的原因 | `test_every_manifest_row_matches_the_bytes_on_disk`、`test_manifest_only_lists_files_that_travel_with_the_repository`、`test_generated_but_untracked_artifacts_are_declared_as_such` |
| D2 | 审计 m15345 **M2 同族 / L2 口径混用** | 已提交的 `benchmark_ref_vs_got.json` 是 **quick 软档**（54 行，profile=quick）生成的，而锚点 E2 门声明在 **full 档**；full 工件早在库里，表却从未重生成 | 用 full 工件重生成判据表：`profile=full`，78 行，门例 **15/15 PASS**，未进门例 9 行如实 FAIL、6 行 UNSCORED | `test_committed_ref_vs_got_table_is_rebuilt_from_the_gate_profile`（逐行 dict 相等 + profile 必须为 full） |
| D3 | 图不可由工件重绘 | `figure_smooth` 用模块全局 `PROFILES[CURRENT_PROFILE]`（缺省 `quick`）而不是 payload 的网格序列 → 在「从已提交 full metrics.json 重绘图」这一唯一可复现路径上直接抛 `ValueError: shapes (3,) and (5,)`；`figure_sod`/`figure_tvd` 同样依赖该全局 | 三个绘图函数全部改为 **payload 驱动**：网格数取自 `rows[*]["ncells"]`，Sod 取 gate 例 `(degree, ncells)`，TVD 取 `cases` 的 degree 与 `ncells` | `test_smooth_figure_rebuilds_from_committed_full_metrics`（修复前必崩）、`test_sod_figure_reads_its_case_from_the_payload`、`test_tvd_figure_reads_degrees_and_cells_from_the_payload`（monkeypatch 记录实参） |
| D4 | 判据表 verdict 可能被「洗白」 | 审计 M2 关掉了 builder 侧的 `informative`，但没有独立复核：测试只检查 builder 自己的输出格式 | 新增**不信任 builder** 的重算：从 `(ref, got, tolerance)` 文本独立复算 PASS/FAIL，覆盖 20+ 行（E2 站点行 + E3 行） | `test_scored_rows_are_exactly_their_tolerance_rule` |

另修一处 `tools/render_cks1998_evidence.py` 的潜在崩溃：判据表抬头引用了已不存在的
`counts['informative']`（M2 修复后该键被 `unscored/gated_*` 取代），重渲染必 `KeyError`。

## 2. 怎么验证的

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_cks1998_artifact_consistency.py -q   # 8 passed
OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark                            # profile=full, gated 15/15
OMP_NUM_THREADS=1 python -m tools.render_cks1998_evidence                           # 清单 12 行全部可核验
```

独立复核（本单实跑）：清单 12 行逐行 `sha256sum` 与 bytes 一致且全部 `git ls-files` 命中；
修复前同一检查报 3 行 `missing`（`artifacts/full/a1_l2_order.png|e2_sod.png|e3_limiter.png`）。

## 3. 误差判定

本单是证据/口径修复，不产生新的物理量，**不涉及 >3% 的精度判定**。受影响的既有判定不变：

| 关键量 | 基准 | 实测 | 判定 |
|---|---|---|---|
| E2 Sod 相对 L2（门例 P2/N=400，最差分量 velocity） | < 3 % | 2.368 % | PASS（余量 0.632 pp） |
| E2 Sod 粗网格 P2/N=200 | < 3 % | 4.173 % | **FAIL（如实登记，非门例）** |
| E3 TVD 严格策略超调 | ≤ 1 % | 0.000 % | PASS |
| E3 TVB M=0 指示器（P1,N=80） | ≤ 1 % | 1.389 % | **FAIL（如实登记，非门例）** |

## 4. 未完成 / 局限（honest WITHHELD）

- **E2 门余量的旋钮敏感性研究未出数**：`validation/repro_cks1998/sod_gate_margin.py`
  已写好（ladder P1/P2/P3 × N、P2/N=400 的 cfl 与 TVB M 扫描、log-log 交点 N\*(3%)），
  并已递交集群（cj3357 ladder / cj3358 cfl / cj3359 tvb），**排队中未完成**。
  因此「3 % 门是否只是单点运气」这一问本单 **不作任何数字声明**（WITHHELD）。
  待作业返回后：`--study assemble` 会从记录的原始行重新推导观测阶、N\* 与逐项 verdict，
  并写 `artifacts/sod_gate_margin/{metrics.json,summary.md,ladder.csv,knob_sensitivity.csv}` 与两张图。
- `figure_sod` / `figure_tvd` 仍需调用求解器（payload 只存站点/峰值，不存整条场）：
  已做到「参数由 payload 决定」，但尚未做到「完全不重算」。要彻底消除重算，需把
  场剖面落成 CSV 工件（约 1–2 MB），本单未做，避免为图改动已核验的 `metrics.json`。
- full 档 3 张 PNG 仍未入库（maturation 门曾滤除二进制）；本单的选择是**不假装有**，
  在 evidence.md 里显式声明为「本地生成、未提交」。

## 5. 给下次自找活的线索

1. 集群作业返回后跑 `sod_gate_margin.py --study assemble`，把 N\*(3%) 与 cfl/M 敏感性写进锚点报告；
   若某旋钮值跌破 3 %，那是锚点门的真实边界，应写成门例适用范围而不是 FAIL。
2. 同类「清单不可核验」缺陷可用同一守卫模式扫其他锚点（`grep -l "sha256" validation/*/evidence.md`）。
3. `figure_sod/figure_tvd` 的场剖面落盘（CSV）可让全部图零重算，适合作为下一次小增量。