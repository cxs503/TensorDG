# REPRO-DG-CKS1998 · A2/E3 不等式判据的 hinge 编码（闭合最后两条 NOT_ENCODED）

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199-224 / SIAM Review 43(1) 2001）
> 单号：m22274　分支：`agent/m22274`　集群作业：`cj4015`
> 平台裁定：**PENDING（集群作业 cj4015 排队中，POST /tensor/bench/results 未递交）**（rows=0，failed_checks=['platform_rows_pending']）

## ① 案例介绍

CKS1998 的 §3.2/§4 给出两条**不等式**判据（不是 ref-vs-got）：

* **A2**：限幅（TVB M=0 + SSP-RK3）输运方波的 **L1 收敛阶 ≥ 1/2**（半阶地板）；
* **E3**：限幅器的 **TVD 性质**——不产生新极值，点值超调 **≤ 1 %**（初值界 [0,1]）。

两者在此前都登记在 `bench_spec.NOT_ENCODED`：平台的相对误差契约是**双侧**的
（`|got-ref|/|ref| ≤ tol`），把界直接当 ref 会伪造度量——超调 0 %（完美 TVD）对 1 % 的界
算成 100 % 误差，比真正越界的 1.389 % 更"差"。于是这两条论文判据长期只活在 Markdown 与
仓内门里，平台机器裁判（L4）覆盖不到。

## ② 怎么做的

1. **保序 hinge 编码**（`bench_spec.hinge`，唯一真源 `bench_spec.HINGE_RULES`）：
   下界 `x ≥ b ⇒ got = min(x,b)/b`；上界 `x ≤ b ⇒ got = max(x,b)/b`；一律 `ref = 1`。
   `min/max` 与 `/b` 都是 IEEE 精确运算，故"满足即逐位饱和到 1.0、误差 0"，
   "越界则误差 = 相对亏空/超出量" ⇒ **PASS ⟺ 原始不等式成立**（逐点等价见
   `tests/test_cks1998_inequality_hinge.py::test_hinge_decision_is_equivalent_to_the_raw_inequality`）。
2. **容差 = 1e-9 是浮点护栏，不是放宽**：hinge 在界上逐位等于 1、误差恰好 0；1e-9 只吸收
   JSON 往返的 1 ulp（比观测阶有效数字低 6 个量级）。
3. **测量复用仓内驱动**：workload 直接调用 `repro_cks1998.run_discontinuous_advection`
   与 `run_tvd`，平台口径与仓内门口径是同一份代码（不存在两套数）。
4. **可审计**：原始值同时写进 `validation.hinge.<metric> = {raw,bound,kind,hinge}` 与
   `rows`（A2 逐网格 L1 误差、E3 三种限幅器峰值）。第三方可从 `rows` 重算观测阶并核对
   `hinge == clamp(raw)`（`test_hinge_matches_raw_and_rows_recompute_the_order`）。
5. **取证回路**：集群作业 cj4015（stage_dir，节点不可达 GitHub）→ 原始 stdout →
   `POST /tensor/bench/results`（平台按指针复算）→ `GET /tensor/bench/l4-verdict/22274`。
   **本报告的数据源：local bench-configuration partial (A2 cells 40/80/160, E3 cells 100, degrees 1-2)**；平台 ingest 是否已完成见 ③ 与 ⑥ 的裁定行。

## ③ 计算结果

A2（限幅方波，TVB M=0 + SSP-RK3；观测阶 = L1 误差的最小二乘斜率）：

| 阶 | 网格 | L1 误差 | 观测阶 | 地板 | 判定 |
|---|---|---|---:|---:|---|
| P1 | [40, 80, 160] | 1.1435e-01, 7.3007e-02, 4.6466e-02 | 0.6496 | 0.5 | PASS |
| P2 | [40, 80, 160] | 1.1435e-01, 7.3008e-02, 4.6466e-02 | 0.6496 | 0.5 | PASS |
| P3 | [40, 80, 160] | 1.1435e-01, 7.3008e-02, 4.6466e-02 | 0.6496 | 0.5 | PASS |

E3（超调，% of 初值界）：**门例 = strict 策略**（预登记，`GATES.tvd_overshoot = 0.01`），
非门例（`unlimited` 对照 / 公开的 `tvb_M0` 指示器）同样进 `rows`，越界必须看得见：

| 策略 | 点值超调 | 对 1 % 门 |
|---|---:|---|
| p1_unlimited | 6.326 % | **>1 %（可见越界）** |
| p1_tvb_M0 | 0.000 % | ≤1 % |
| p2_unlimited | 8.573 % | **>1 %（可见越界）** |
| p2_tvb_M0 | 0.000 % | ≤1 % |

平台复算（ref/got/误差全部由平台从作业 stdout 按 spec 指针重算）：

| 度量 | 原始测量 raw | 界（kind） | hinge got | ref | 本地复算 rel_error | 判定 |
|---|---:|---|---:|---:|---:|---|
| `a2_l1_order_p1` | 0.64962 | 0.5（lower） | 1.000000000 | 1.0 | 0 | PASS |
| `a2_l1_order_p2` | 0.64962 | 0.5（lower） | 1.000000000 | 1.0 | 0 | PASS |
| `a2_l1_order_p3` | 0.649621 | 0.5（lower） | 1.000000000 | 1.0 | 0 | PASS |
| `e3_overshoot_p1` | 0 | 1（upper） | 1.000000000 | 1.0 | 0 | PASS |
| `e3_overshoot_p2` | 0 | 1（upper） | 1.000000000 | 1.0 | 0 | PASS |

**L4 裁定：`PENDING（集群作业 cj4015 排队中，POST /tensor/bench/results 未递交）`，rows=0，failed_checks=['platform_rows_pending']**
（hinge 之前该锚点覆盖 13 条度量；本单 +5 ⇒ 18 条，`NOT_ENCODED` 清空）。

## ④ 结果图

`artifacts/cks1998_hinge/a2_l1_ladder.png`（L1 阶梯 + 观测阶/地板）、
`artifacts/cks1998_hinge/e3_overshoot.png`（三策略超调 vs 1 % 门，门例与非门例分色）。

## ⑤ 注意点 / 局限（honest）

1. **hinge 由 workload 施加**——裁判链的自证成分：平台拿到的是已饱和的 got。缓解是
   原始值同文档可见且可重算（rows + validation.hinge），回归门钉死 `hinge == clamp(raw)`；
   但严格说，"raw → hinge"这一步不在平台侧复算，这是本编码的**结构性局限**。
2. **A2 的 1/2 地板在 P2/P3 上没有渐近意义**：限幅后退化为 O(h^1/2)，实测三阶都在 0.65
   左右（同一数值），编码判的是"不低于地板"，不是"设计阶"。
3. **E3 的非门例越界**（`unlimited` 6–8 %、`tvb_M0` 在粗网格 N=80 上 1.389 %）**不进锚点判定**
   ——预登记的 E3 门是 strict 策略；这些越界留在 rows/docs 里，不是被吞掉。
4. **hinge 只覆盖一维 A2/E3**：二维系统算例的不等式判据（无）与 §5.3/§5.4 的非光滑判据
   仍是另一条线，本单不涉及。
5. 网格/容差**未改动任何既有项**：13 条旧度量容差逐字不变（`FROZEN_TOLERANCES` 只增不改）。

## ⑥ 误差判定（是否 < 3 %）

hinge 度量的 ref 是界（1.0），误差是**相对界的亏空/超出量**，不是物理量的 3 % 口径：

* A2 三条：原始观测阶 0.65 对地板 0.5 ⇒ 优于地板 **30 %**（hinge 误差 0），**< 3 %** 成立；
* E3 两条（strict）：超调 0.000 % 对 1 % 的界 ⇒ hinge 误差 0，**< 3 %** 成立；
* 全部 5 条 hinge 度量最差平台相对误差 = **0**（1e-9 级以下），逐项 **< 3 %**；
* 既有 13 条度量本单未重测口径，平台历史行（m20922/m22274）保持不变。
