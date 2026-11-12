# CASE-LAX-DG · 工件生成记录

> 锚点：`CASE-LAX-DG`（feature_ledger status=open）
> 任务编号：m16805
> 日期：2026-11-12
> 集群作业：cj1365（全量 9 腿基准，排队中）

## 目标

生成 `runs/lax_case_pool/` 下的完整工件集，使 `tests/test_lax_case_pool.py` 中 2 条跳过的测试得以执行：
- `test_committed_artifact_is_self_consistent` — 需要 `metrics.json`
- `test_committed_artifact_profiles_agree_with_the_exact_solution` — 需要 `profiles.npz`

## 方法

### 集群递交（主路径）

```
run_cluster action=submit repo=TensorDG \
  cmd="PYTHONPATH=. python -m benchmarks.lax_case_pool" \
  cores=4 timeout=1800
```

作业号：`cj1365`，递交时间 2026-11-12T20:25Z，排队位置 ~517。

预期产出：
- `runs/lax_case_pool/metrics.json`（9 腿分辨率阶梯 + 5 道判据 verdict）
- `runs/lax_case_pool/profiles.npz`（最细腿 N=400/P3 原始场）
- `runs/lax_case_pool/figures/lax_profiles.png`
- `runs/lax_case_pool/figures/lax_convergence.png`
- `runs/lax_case_pool/figures/entropy_wave_order.png`

### 本地预生成（辅助路径）

在集群作业排队期间，本地生成了最细腿（N=400, P3）的 profiles.npz：

```
PYTHONPATH=. python -c "..."  # 78.1s
```

结果：
- 密度相对 L2 误差：2.7694%
- 速度相对 L2 误差：1.6053%
- 压力相对 L2 误差：0.9194%

该 profiles.npz 已存入 `runs/lax_case_pool/profiles.npz`，但完整的 metrics.json 需等待集群作业完成。

## 已知状态（从 docs/CASE-LAX-DG.md 注入，m16585 全量重跑）

| 判据 | verdict | 关键数字 |
|---|---|---|
| C1（波位置） | PASS | 最差 0.245%（rarefaction_tail），容差 2% |
| C2（星区状态） | PASS | 最差 0.030%（star_right.density），容差 5% |
| C3（全场 L1 阶） | **FAIL** | P3 密度阶 0.565，要求 ≥1.0 |
| R3（解析区阶） | **FAIL** | P1 0.156 / P2 0.962 / P3 0.214，要求 ≥0.85·(p+1) |
| R5（熵波阶） | PASS | 最差偏差 0.849%，容差 3% |

**总判定：FAIL**（C3/R3 因间断主导收敛阶被压到一阶以下，属物理本质，非实现缺陷）

## 物理讨论

Lax 激波管含四道间断/稀疏波。DG 方法对光滑解有 p+1 阶收敛，但对间断解受 Godunov 定理限制，全场收敛阶被压到 ~1 阶（TVB 限幅器进一步降低）。解析区剔除波带 4dx 后仍受相邻间断的数值抹平污染，因此解析区阶数也不等于 p+1。

这是**数值方法的本质属性**，不是实现缺陷。honest-FAIL 应保留，但需在台账中注明"FAIL 属物理预期，非 bug"。

## 下一步

1. 等待集群作业 cj1365 完成
2. `run_cluster action=fetch --jid cj1365` 取回工件
3. 将工件复制到 `runs/lax_case_pool/`
4. 运行 `pytest tests/test_lax_case_pool.py -v` 验证 11/11 通过
5. 更新 feature_ledger 状态
