# CASE · REPRO-DG-CKS1998：TVB 指示器的「发货核 vs 论文原文」A/B（验证欠账）

> 锚点 `REPRO-DG-CKS1998`（feature_ledger，open）。本单补的是一处**从未被任何门覆盖的
> 配置档**，不是重复既有算例：仓里所有限幅门都是 `M = 0`（严格 TVD），而 A1 观测阶门
> 跑的是 `limiter=None`（无限幅）。论文真正设计的档位——**有限 `M` + 光滑解带极值**——
> 在本单之前只有散文描述，没有断言。

## 0. 被审对象

`tensordg.tvb.troubled_from_differences`。论文（Cockburn & Shu, Math. Comp. 52 (1989)
eq. (2.10)；JCP 141 (1998) §2.2）释放单元的条件是**某一侧单元均值差已退化为 O(h²)**：

```
released  ⟺  min(|a₁|, |a₂|) ≤ M h²      a₁ = ūᵢ − ūᵢ₋₁,  a₂ = ūᵢ₊₁ − ūᵢ
```

发货核改为比较「单侧差偏离 minmod 斜率的量」与 `M h²`；对光滑数据该偏离 = `|u''| h²`，
于是光滑段的读法是 `troubled ⟺ |u''| > M`。两者在极值释放档等价、在**单调光滑段**不等价。
该偏离在 `tensordg/tvb.py` 的 docstring 里已自陈（为修「每个光滑极值都被置旗」），
但**没有任何测试或工件量化过它的后果**。

## 1. 判据的标尺来自初值本身，不来自拟合

测试解 `u₀ = sin(2πx)`（标量线性对流，周期域，T = 0.5，`ssp_rk3`，CFL 0.3）的二阶导数有
闭式 `max|u''| = 4π² = 39.478417…`。**全部 `M` 都写成它的倍数**，唯一允许被当作"正确值"的
只有 1.0 倍这一档——这是论文「M 应由解的曲率定」的直接落点，避免审计反复点名的"构造即真"。

## 2. 四条门（`GATES`，均可拒收）

| 门 | 断言 | 非 vacuous 的保证 |
|---|---|---|
| `G1_inert_above_curvature` | `M = max|u''|` ⇒ 置旗率 **0**、解与无限幅**逐位相同**（`max|Δ| = 0`）、阶 ≥ p+½ | — |
| `G2_indicator_fires_below_half_curvature` | `M = ¼max|u''|` ⇒ 置旗率 > 0 | 证明 G1 不是"限幅器根本没开" |
| `C1_paper_form_is_stricter` | 论文版置旗率 ≥ 发货版（同一 M、同一网格） | 证明 A/B 的两条分支真的不同 |
| `D1_public_default_constant` | 公开默认 `tvb_constant=1.0` 在本算例上损失 ≥ ¼ 阶 | 记录用户"顺手开限幅器"的静默代价 |

## 3. 结果（数字全部来自工件，零手抄）

- 逐行表：`docs/benchmarks/results/tvb_criterion_ab/table_smoke.md`（由
  `gen_report_cks1998_tvb_criterion_ab.py` 从 JSON 直接生成）
- 工件：`validation/repro_cks1998/artifacts/tvb_criterion_ab/tvb_criterion_ab_smoke.json`
- 图：`docs/figures/cks1998_tvb_criterion_ab.png`（阶 / 置旗率 vs 理论标尺 M）、
  `docs/figures/cks1998_tvb_criterion_ab_ladder.png`（P2 误差阶梯）

要点（数值见上表）：

1. `M ≥ max|u''|` 时发货核**完全不介入**，解与无限幅逐位相同，p1/p2 观测阶分别为
   2.009 / 3.000 —— 与理论 p+1 吻合到 0.45 % 与 0.00 %，**远优于 3 %**。
2. `M < ½max|u''|` 时置旗率 70–100 %，观测阶塌到 ~1.5（p=2 掉 1.48 阶）——极值削平
   的经典代价，如实入账。
3. 论文原文判据在同标尺下**即使在 `M = 2·max|u''|` 仍置旗 85–90 %**（单调段单侧差是
   O(h)，永远大于 `M h²`），1D 正弦上观测阶掉到 ~1.1。→ 发货核是**更宽松**的那一支，
   对它做正偏差判据时必须知道这一点。
4. 公开默认 `M = 1.0` 等价于 `M/max|u''| = 0.03`，即"几乎全单元置旗"，p2 静默损失 1.48 阶。
   默认值按仓规不改（向后兼容），但这条坑现在有工件与门盯着。

## 4. 复现

```bash
OMP_NUM_THREADS=1 python -m validation.repro_cks1998.tvb_criterion_ab --profile smoke
OMP_NUM_THREADS=1 python -m pytest tests/test_cks1998_tvb_criterion_ab.py -q     # 12 passed
OMP_NUM_THREADS=1 python gen_report_cks1998_tvb_criterion_ab.py --profile smoke
```

`--profile bench`（p=3、N=160）本机超时不跑，递交 `run_compute`（作业 cj4751）。

## 5. 局限 / 未做

- **只有单一波形**（`sin(2πx)`）量化了 `M` 的含义；一般解的 `|u''|` 估计（自适应 `M`）不在本单。
- **不**声称发货判据在激波算例上更优；激波档由既有 `tvb_M0` 门覆盖
  （`docs/CASE-REPRO-DG-CKS1998-E3-PAPER-PREREG.md`）。
- `--profile bench` 的 p=3 / N=160 数据未进本报告表格（作业 cj4751 排队中），结论的
  适用性限定在 p ≤ 2、N ≤ 80 的实测范围；p=3 的定性结论（`M ≥ max|u''|` ⇒ 不介入）由
  同一判据的构造保证，但**未被本单实测**，不得当作已证。