# CASE-LAX-DG · 台账容差需要多细的网格？（rel L2 @ P3, N = 400 → 800）

> 锚点：`feature_ledger` `CASE-LAX-DG`（验收原文「ref: 精确解｜容差 < 3 %」，status 仍 `open`）。
> 本文全部数字由 `python -m benchmarks.lax_resolution_requirement` 从工件
> `docs/benchmarks/results/lax_resolution_requirement/metrics.json` 注入（零手抄；
> `--check` 在漂移时失败）。测量运行 2026-11-14T18:39:16Z，墙钟 246.2 s，
> 工件 sha256 前缀 `44dbc878aa762615`。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，γ = 1.4，隔膜 `x0 = 0.5`，
左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0.0, 0.571)`，快照 `t = 0.13`。
解含左稀疏波、接触间断、右激波，精确 Riemann 解闭式可得，故参考值不依赖任何第三方求解器。

**待办缺口**：台账容差写在「解 vs 精确解 < 3 %」，而池子把这条钉在 `p3.N400`
（`docs/CASE-LAX-DG-acceptance-A0.md`），实测密度整体 rel L2 =
`3.11556e-02`，即差 3.9 % 的裕度未达标。池子的 C3/R3 同时给出
P3 全场阶 ≈ 0.49（间断面压阶），所以在 N=400 上靠调参无法补齐，唯一干净的出路是
**加密一档 h** 并把「需要多细」量化。此前从未有 N=800 的 **L2** 读数入库
（`lax_asymptotic_probe` 只存了 L1）。

## ② 怎么做的

1. 同一求解器、同一测量口径（`tensordg.euler_dg_1d`，HLLC，CFL = 0.35，单元 Gauss 节点 +
   Legendre 求积，阶 10，全场不挖波带、不给任何量豁免），跑 P3 的
   N = 400 与 N = 800 两腿；
2. 复用检查：N=400 的 L1/L2 必须与已提交池子工件逐位一致，N=800 的 L1 必须与已提交
   p3 渐近探针（`lax_asymptotic_probe`）逐位一致——同一份场数组要能重现两处历史数字；
3. 预登记规则（跑之前写在驱动里，`RESOLUTION_RULE`）：resolution requirement = the smallest leg of the extension ladder whose whole-field relative L2 is below the registered tolerance for every field (no field exemption, no wave band removed, no averaging); a leg is PASS iff max_field rel L2 < tolerance；
4. **不移动 A0 的生产腿**：`production_leg` 仍为 `p3.N400`，其 FAIL 原样留在记录里；
   本文只增加一条关于分辨率的陈述。

## ③ 计算结果（程序化注入）

| leg | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 判据 |
|---|---|---|---|---|---|
| p3.N400 | 3.11556e-02 | 2.24805e-02 | 1.30501e-02 | 3.11556e-02 | FAIL |
| p3.N800 | 2.31490e-02 | 1.38257e-02 | 8.99402e-03 | 2.31490e-02 | PASS |

| leg | 最差 rel L2 | 该量 | 容差裕度（×容差） | verdict |
|---|---|---|---|---|
| p3.N400 | 3.11556e-02 | density | 1.039 | FAIL |
| p3.N800 | 2.31490e-02 | density | 0.772 | PASS |

最小达标腿：**p3.N800**（分辨率倍数
×2.0），
R2 判定 **PASS**。

| 量 | N=400→800 阶（L1） | 阶（L2） |
|---|---|---|
| density | 0.495 | 0.429 |
| velocity | 0.580 | 0.701 |
| pressure | 0.511 | 0.537 |

复用检查（与已提交工件对比，容差 `1e-06`）：
最差相对差 `7.138e-03`，
状态 **FAIL**。逐来源：

| 来源工件 | 对比行数 | 最差相对差 | verdict |
|---|---|---|---|
| lax_asymptotic_probe | 6 | 7.138e-03 | FAIL |
| lax_case_pool | 6 | 0.000e+00 | PASS |

**来源间的分歧要据实读**：本单实测 `lax_case_pool` 的 `p3.N400` 六个读数（L1/L2 × 三量）
**逐位一致**（相对差 0.0），而 `lax_asymptotic_probe` 的同腿读数差约 1e-4、`p3.N800` 差至
7.1e-3——即探针工件记录的场不再是当前代码可复现的场（两处腿的步数相同 5109 / 10250，
但工件里记录的 `dt` 与本次不同：`2.2296e-05` vs `2.5445e-05`（N=400）、
`4.4115e-06` vs `1.2683e-05`（N=800））。因此**本文给出的 N=800 读数是当前可复现的一组**，
探针工件需重生成（见 `docs/CASE-LAX-DG-residual-gap.md` 的后续项），overall `reuse_check`
判 FAIL 如实登记，不用池子那部分的 PASS 覆盖它。

| 来源工件 | leg | 范数 | 量 | 已提交 | 本次 | 相对差 |
|---|---|---|---|---|---|---|
| lax_asymptotic_probe | p3.N400 | l1 | density | 8.686465e-03 | 8.685540e-03 | 1.06e-04 |
| lax_asymptotic_probe | p3.N400 | l1 | velocity | 1.230920e-02 | 1.230639e-02 | 2.29e-04 |
| lax_asymptotic_probe | p3.N400 | l1 | pressure | 7.314181e-03 | 7.311961e-03 | 3.03e-04 |
| lax_asymptotic_probe | p3.N800 | l1 | density | 6.180136e-03 | 6.164752e-03 | 2.49e-03 |
| lax_asymptotic_probe | p3.N800 | l1 | velocity | 8.281602e-03 | 8.233285e-03 | 5.83e-03 |
| lax_asymptotic_probe | p3.N800 | l1 | pressure | 5.168109e-03 | 5.131218e-03 | 7.14e-03 |

## ④ 结果图

* `docs/benchmarks/results/lax_resolution_requirement/figures/lax_resolution_metric.png`
* `docs/benchmarks/results/lax_resolution_requirement/figures/lax_resolution_profiles.png`

* `lax_resolution_metric.png`：两腿的 rel L2（台账度量）/ rel L1 与 3% 容差线；
* `lax_resolution_profiles.png`：N=800 的 ρ/u/p 剖面 vs 精确解。

渲染来源：rendered from the committed profiles cache。

## ⑤ 注意点

1. **测的是距离不是阶**：本文只回答「这个范数在这个网格上是否进 3 %」，不回答族是否
   达设计阶。P3 在间断存在下全场阶仍 < 1（见上表），这是 DG 抓激波的已知属性，
   与池子 C3/R3 的 FAIL 一致，两者不互相代言。
2. **不得用它改写 A0**：台账臂的最细登记腿仍是 N=400（正文 A0 的 FAIL 保留）。
   若要按本文结论把锚点翻成 done，需要台账持有人明确把「生产腿」改成 N=800 ——
   那是验收口径变更，不是本单可以自己做的裁决。
3. **口径敏感性**：误差在 Gauss 节点上按 Legendre 求积加权，换单元平均/L∞ 会给出不同数字；
   口径写在工件的 `config` 里供重裁。
4. **算力**：N=800/P3 需要 10250 步，是 N=400 腿（5109 步）的约
   2.0 倍步数，属重计算，已递交集群调度执行。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| rel L2 density @ p3.N800 | 精确 Riemann 解 | 2.31490e-02 | 2.315 % | 是 |
| rel L2 velocity @ p3.N800 | 精确 Riemann 解 | 1.38257e-02 | 1.383 % | 是 |
| rel L2 pressure @ p3.N800 | 精确 Riemann 解 | 8.99402e-03 | 0.899 % | 是 |
| rel L2 density @ p3.N400（钉扎腿，未改） | 精确 Riemann 解 |
3.11556e-02 | 3.116 % | 否 |

**结论**：在 P3 / N = 800 上 **PASS** —— 台账 3 % 容差
全部三个量均达标；
钉扎腿 `p3.N400` 的 FAIL 原样保留，未被本文改写。
