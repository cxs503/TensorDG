# CASE-DG-EULER-VORTEX-ERR round 15：证据账重算（未验证 7→5）、陈旧判决复核、重放分片重递交

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜算例池·等熵涡移动误差守恒｜ref: Yee, Sandham & Djomehri (JCP 162, 2000)
> 容差 6.0%（t=1 快照，最大场相对 L2）｜本单 mission 22435
> 整包裁决（写作时）：**INCOMPLETE**（C11：40 条记录中 5 条仍为 `unverified`）

## ① 案例介绍

二维等熵涡在周期方箱 [-5,5]² 内匀速平移，真解在平移下不变形，故本算例判定的是「误差是网格属性，
不是时间属性」——即误差随时间的累积律，而不是单点精度。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5, 5]²，周期，min-image 折返参考解 |
| 涡强 β | 5.0 |
| 背景态 | ρ=1, p=1, γ=1.4 |
| 输运速度 | \|u_inf\|=√2（方向轴见 C10） |
| 容差（快照，t=1） | 6%（最大场相对 L2） |
| 离散 | 模态 DG p=1,2,3（HLLC + SSP-RK3，L2 投影）+ P0 有限体积（Rusanov）为交叉校验 |
| 阶梯 | 40 个声明检查点（网格×CFL×阶数×方向×时刻） |
| 判据 | C1–C11（容差/误差守恒/离散守恒/熵预算/交叉阶/阶梯完整性/设计空间不变/阶数不变/设计阶/方向不变/**源流完整**） |

## ② 怎么做的（本轮做了什么）

1. **先核锚点**：`feature_ledger` 中 `CASE-DG-EULER-VORTEX-ERR` 状态 `open`（非 done/verified），
   允许实质增量。
2. **重算证据账**：把 `results/` 的 40 条记录 + 磁盘上的 `provenance_certificates.json`
   重新过一遍聚合器（`python3 benchmarks/case_euler_vortex_errconserve.py --aggregate`，只读记录、无重算）。
   结果：**`unverified` 由 7 降为 5、`replayed` 由 12 升为 14**。
   原因是 `metrics.json` 是**陈旧判决**：它自报 `certificates_on_disk: 12`，
   而磁盘上的证书文件早已有 14 条（`dg_d2_n32_t2_cfl0.2` 认证于 2026-11-15T21:00:54、
   `dg_d2_n64_t1_cfl0.2` 认证于 2026-11-15T14:55:08，两条 `replay_delta` 分别为 0.0 与 9.75e-13，
   均远小于 1e-9 容差）。**证据已落地、判决没跟上**，整包因此被少算了一条通过证据。
   这就是本轮新增回归测试 `tests/test_metrics_provenance_freshness.py` 要钉住的东西：
   `metrics.json` 的 provenance 计数必须与「记录 + 当前证书文件」重算的结果一致，否则即为陈旧判决。
3. **复核上一轮声称的在途证据（重要，属事故级发现）**：round 14 文档记载已递交 5 个重放分片
   `cj3888`–`cj3892`。本轮逐个查队列与取回工件：
   * `cj3888`/`cj3889`/`cj3890`/`cj3891` 队列状态 `succeeded`、rc=0，但取回的工件
     （`/srv/tensor/agent/cluster/fetch/cj3888_n315.tar.gz` 等）里**没有** `provenance_certificates.json`；
     其内容是别的仓的交付物（`DELIVERABLE_WAVE_FLUME*.md`、`FLUCTUATING_HYDRO_SUMMARY.md`、
     `assets/geometry_registry.json` 等 LBM/SPH/FEM 内容）；
   * `cj3892` 的作业日志显示启动我们的分片时 `ssh ... mkdir -p ~/jobs/cj3892 && ... provenance_certify.py`
     在 30 s 内 `subprocess.TimeoutExpired`；`cj3888` 日志尾部是另一件事的流体计算并以
     `free(): corrupted unsorted chunks` 收尾。
   ⇒ 这 5 个「已递交」作业**没有产出任何证书**。结论：递交号 ≠ 证据，取回工件必须**逐条核对内容是否属于本仓本算例**，
     否则会把别的任务的成功回执当作自己的证据（这正是 `tools/land_provenance_certificates.py`
     设计成严格过滤的原因：拒绝孤儿证书、拒绝配置不符的证书）。
4. **重递交**：本单以 `stage_dir=/srv/tensor/agent/work/m22435`（自己的工作树，不走 GitHub main）
   递交 4 个按成本均衡分片的重放作业：`cj4121`(1/4)、`cj4122`(2/4)、`cj4123`(3/4)、`cj4124`(4/4)，
   每个 8 核、`--threads 8`、`--max-seconds 7000`，命令尾部把证书文件复制成 `certs_shardN.json`
   并打印 `CERTS_BEGIN…CERTS_END` 包裹的原文，使**日志本身**成为可取回的工件（不再依赖取回 tar 的内容归属）。
   分片划分（`tools/provenance_certify.py` 的 cost proxy，cells²·(p+1)²·t/cfl 均衡）：

   | 分片 | 检查点 | 成本代理 |
   |---|---|---|
   | 1/4 | `dg_d2_n64_t10_cfl0.2` | 1 843 200 |
   | 2/4 | `dg_d2_n32_t10_cfl0.2` | 460 800 |
   | 3/4 | `dg_d2_n64_t2_cfl0.2` | 368 640 |
   | 4/4 | `dg_d2_n32_t10_cfl0.4`、`dg_d2_n32_t5_cfl0.2` | 460 800 |

   交错网格，每点只被一个分片试着认证一次，因此结果可合并、不会重复付费。

## ③ 计算结果（重算后的判据表，程序化导出）

`metrics.json` 由聚合器写出，下表为 `gates` 的原样摘要（`--aggregate` 输出）：

| 判据 | 状态 | 读数 | 说明 |
|---|---|---|---|
| C1_tolerance | passed | 4.9260% | 最差点 `p0_n64_t1_cfl0.4`，< 6% |
| C1x_crosscheck_raw | informational | 14.3047% | 长时 P0 原始误差（超快照容差，如实登记） |
| C2_error_conservation | passed | α=0.199 | 设计族 e~t^α ≤ 1 |
| C3_discrete_conservation | passed | 3.9e-13 | 质量/动量/能量漂移 |
| C4_entropy_budget | passed | 6.34e-07 | \|ΔS\|/S(0) < 1e-3 |
| C5_crosscheck_order | passed | 0.849 | P0 观测阶 ≥ 0.8 |
| C6_ladder_completeness | passed | 40/40 | 声明检查点全部有记录 |
| C7_design_space_invariance | passed | α=0.4401 | 每个设计点 α ≤ 1 |
| C8_degree_invariance | passed | α=0.4401 | p=1,2,3 全部可判 |
| C9_design_order | passed | 2.0424 | ≥ 0.8(p+1) |
| C10_direction_invariance | passed | α=0.3129 | 两个方向均满足 |
| **C11_provenance_integrity** | **not_evaluated** | 见下 | 40 条中 5 条 `unverified` |

C11 计数的本轮变化（同一聚合器、同一记录集，只是证书账更新）：

| 计数 | 重算前（陈旧 `metrics.json`） | 重算后 |
|---|---|---|
| recorded（记录自带参考模型） | 21 | 21 |
| replayed（证书复现 1e-9 内） | 12 | **14** |
| unverified（无证书） | 7 | **5** |
| certificates_on_disk | 12（陈旧） | **14** |
| C11 状态 | not_evaluated | not_evaluated（待 5 条） |

仍未验证的 5 条：`dg_d2_n32_t5_cfl0.2`、`dg_d2_n32_t10_cfl0.2`、`dg_d2_n32_t10_cfl0.4`、
`dg_d2_n64_t2_cfl0.2`、`dg_d2_n64_t10_cfl0.2`（正是本轮 4 个分片在算的点）。

## ④ 结果图

本轮为**证据账/判决刷新与递交复核**，没有产生新的场量计算，故不新增云图；已发表的场图与
误差-时间曲线仍在 `docs/CASE-DG-EULER-VORTEX-ERR-round12-evidence-closure.md`
（`results/fig_error_vs_time.png`、`fig_design_space.png`、`fig_invariants_vs_time.png`）。
按报告纪律，本轮图缺的**理由**是：唯一变化的量是「证书计数」这一离散表，把它画成图不增加信息；
表中的数字由 `metrics.json` 程序化导出，无手抄。

## ⑤ 注意点（踩过的坑与适用范围）

1. **判决会陈旧**：证书、记录都是独立落盘的数据，`metrics.json` 是它们的**投影**。
   任何一处落地后必须重跑 `--aggregate`，否则整包的裁决落后于证据——本轮就是靠这条把 7 修回 5。
   新增测试 `tests/test_metrics_provenance_freshness.py` 把「投影必须与源一致」变成断言，而不是纪律。
2. **递交号不是证据**：`succeeded` + rc=0 的作业号可以指向完全另一件事的产出。取回工件必须验内容归属
   （本轮的 `cj3888`–`cj3892` 就是反例：tar 里是别的仓的交付物，证书一条也没有）。
   为此本单让作业把待取回的证书原文写进**日志**（`CERTS_BEGIN/END`），日志与作业号一一绑定。
3. **成本→时间不是线性的**：cost proxy 只用于分片均衡，不预测墙钟（`dg_d2_n64_t10_cfl0.2`
   历史墙钟 2666 s，proxy 1 843 200）。分片按 proxy 均衡后，最重分片仍是它，因此整包闭合的
   关键路径就是这一个点。
4. **适用范围**：C11 只回答「这条记录的数字是不是它声称的那次计算产生的」，不回答「这个格式准不准」——
   后者是 C1–C10 的事。两者不可互相替代。

## ⑥ 误差判定

| 关键量 | 实测 | 基准 | 是否 < 3% |
|---|---|---|---|
| C1 t=1 最大场相对 L2（最差族 `p0_n64`） | 4.9260% | 等熵涡闭式周期解 | 否（4.93% > 3%，但设计容差为 6%，通过口径是 6%）|
| C1 最差 DG 族 t=1（`dg_d2_n64`） | 0.00115% | 闭式解 | 是 |
| C2 累积指数 α | 0.199 | 判据 α ≤ 1（误差不随时间放大） | 是 |
| C5 P0 观测阶 | 0.849 | 一阶 ≥ 0.8 | 是（偏差 6.1% 于 1.0，但判据是 0.8 下限）|
| C9 DG 观测阶（p=1） | 2.0424 | 0.8·(p+1)=1.6 | 是 |
| C11 证书复现偏差（14 条已落地证书） | ≤ 1.13e-12 | 1e-9 容差 | 是 |
| C11 未验证记录占比 | 5/40 = 12.5% | 0（全部可判） | 否（C11 因此不可判） |

**诚实结论**：本轮把证据账从「陈旧」修正为「与磁盘一致」，把 C11 的未验证数从 7 降到 5，
并把剩余 5 条的重放递交为 `cj4121`–`cj4124`；**证书尚未取回落盘，故 C11 仍 `not_evaluated`，
整包仍 INCOMPLETE**。不宣称「C11 已闭合」「整包 PASS」。

## ⑦ 复现

```
# 证据账重算（只读记录，不重算物理）
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --threads 2
python3 -c "import json;m=json.load(open('case_euler_vortex_errconserve/metrics.json'));print(m['provenance']['counts'])"
# -> {'recorded': 21, 'unverified': 5, 'replayed': 14}

# 判决新鲜度回归（新增）
PYTHONPATH=. python3 -m pytest -q tests/test_metrics_provenance_freshness.py
```