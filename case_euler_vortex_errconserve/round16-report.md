# CASE-DG-EULER-VORTEX-ERR round 16：重放崩溃隔离与证据入库

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜等熵涡移动误差守恒｜ref: Yee et al. (JCP 162, 2000)｜容差 6.0%（t=1 快照）｜mission 22520
> 整包裁决：**INCOMPLETE**

## ① 案例介绍

二维等熵涡在周期方箱 [-5,5]² 内匀速平移，真解在平移下不变形，因此离散误差的累积律是网格属性而非时间属性。本轮不改物理，只处理「证据」：把已经算出来的重放证书真正入库，并让重放崩溃不再抹掉同批其余证据。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5,5]²，周期，min-image 折返参考解 |
| 涡强 β / 背景态 | 5.0 / ρ=1, p=1, γ=1.4 |
| 输运速度 | |u_inf|=√2 |
| 离散 | 模态 DG p=1,2,3（HLLC+SSP-RK3）+ P0 有限体积（Rusanov）交叉校验 |
| 检查点 | 40 条记录（声明阶梯） |
| 判据 | C1–C11（容差/误差守恒/离散守恒/熵预算/交叉阶/阶梯完整/设计空间/阶数/设计阶/方向/源流完整） |

## ② 怎么做的

1. **严格入库已在飞的证据**：`cj4124`（分片 4/4）的工件里确实有两条新证书（`dg_d2_n32_t10_cfl0.4`、`dg_d2_n32_t5_cfl0.2`，`replay_delta` 均为 0），经 `tools/land_provenance_certificates.py` 逐条核对后入库（拒孤儿/拒配置不符/只能被更好者替换）。
2. **定位「递交≠证据」的机制**：`cj4121`–`cj4123` 三个分片作业状态 `succeeded`、`rc=0`，但日志尾部是 `Segmentation fault (core dumped)`——分片内顺序重放，一次崩溃抹掉该分片剩余全部未写盘证书。
3. **修复**：`tools/provenance_certify.py` 新增 `--isolate`，每条检查点在独立子进程重放，父进程按 `returncode` 记账（负值=被信号杀死，记为 `crashed (signal N)`），继续尝试后续 key，结束后从磁盘重读 bundle 报告进度（不用陈旧副本）。
4. **在飞验证**：递交三条单 key 作业 `cj4184`(n64_t2,1 线程)、`cj4185`(n32_t10_cfl0.2,1 线程)、`cj4186`(n64_t10,4 线程)，各自成败互不牵连。

## ③ 计算结果（程序化读取 `metrics.json`）

| 判据 | 状态 | 读数 |
|---|---|---|
| C1_tolerance | passed | 0.04926 |
| C1x_crosscheck_raw | None | 0.143 |
| C2_error_conservation | passed | — |
| C3_discrete_conservation | passed | — |
| C4_entropy_budget | passed | 6.336e-07 |
| C5_crosscheck_order | passed | — |
| C6_ladder_completeness | passed | — |
| C7_design_space_invariance | passed | 0.4401 |
| C8_degree_invariance | passed | 0.4401 |
| C10_direction_invariance | passed | 0.3129 |
| C9_design_order | passed | 2.042 |
| C7x_design_space_spread | informational | — |
| C7x_amplification_prediction | informational | — |
| C11_provenance_integrity | not_evaluated | 1.133e-12 |

C11 明细（证书逐条 `replay_delta`）：

- `recorded` 21 条（记录自带参考模型）；`replayed` 16 条；`unverified` 3 条；`certificate_rejected` 0 条。
- 重放最大偏差 1.133e-12（最差点 `dg_d2_n32_t5_cfl0.4`，容差 —）。
- 磁盘证书 16 份；仍未验证：['dg_d2_n32_t10_cfl0.2', 'dg_d2_n64_t10_cfl0.2', 'dg_d2_n64_t2_cfl0.2']。

## ④ 结果图

![图](fig_round16_error_and_provenance.png)

左：各族最大场相对 L2 误差随输运时刻的变化（对数纵轴，红线=6% 快照容差）；右：C11 溯源分类计数（本轮把 `unverified` 从 5 压到 3）。

## ⑤ 注意点 / 局限

1. **段错误根因未定**：发生在较大 DG 检查点的 `replay()` 内部（torch float64 CPU 路径），P0/numpy 路径与较小 DG 检查点（2342 steps）从未崩过；线程数是被怀疑但未证实的变量。本轮只修「崩溃不该抹掉同批其余证据」。
2. **C11 尚未闭合**：3 条 `unverified` 仍在，整包裁决 INCOMPLETE，不写成通过。
3. 证书是数据而非叙述：入库要过物理字段一致性检查，弱证据不能替换强证据。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| t=1 最大场相对 L2（设计点 DG-P2/32²、P0/64²） | Yee et al. 2000 算例定义（解析等熵涡） | 0.0493% | 是（门槛 6%，上列为最差点 p0_n64_t1_cfl0.4） |
| 离散守恒漂移 | 周期界面通量望远镜求和（应为舍入级） | — | 是 |
| 交叉校验阶（P0） | 理论 1 | — | 否（0.849，属长时 P0 的观测阶，判据只要求 ≥0.8，已如实登记） |
| 设计阶（DG-P1/P2） | 理论 p+1 | 2.042 | 否（2.042 vs 2/3，判据要求 ≥0.8(p+1)，未达 3% 相对精度，如实登记） |
| 源流完整性 | 每条记录都能复现 | 1.133e-12（相对） | 已入库者 是；3 条未验证 |

结论：**数值判据 C1–C10 通过，C11 未闭合 ⇒ 整包 INCOMPLETE**（honest FAIL/未判定不粉饰）。注意 C5/C9 的「<3%」问句回答是否定的：它们是收敛阶读数，不是与基准值的相对误差。