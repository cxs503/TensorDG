# CASE-DG-EULER-VORTEX-ERR round 19：C11 溯源闭合，整包裁决转为 PASS

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜算例池·等熵涡移动误差守恒｜ref: Yee, Sandham & Djomehri (JCP 162, 2000)｜容差 6.0%（t=1 快照，最大场相对 L2）
> 作业 `cj4263`（节点 `n315`，单线程）｜mission 22683

**结论**：`verdict = PASS`，`all_passed = True`，`failed = []`，`not_evaluated = []`。C1–C11 全部通过；此前五轮（round 14–18）唯一挡板——缺失的最后一份重放证书——已落地。

## ① 案例介绍（物理背景 / 几何 / 参数）

等熵涡在均匀流中以常速平移，是二维 Euler 方程唯一有闭式解的“输运 + 全非线性”算例：无粘无热传导，涡核密度/压力扰动自相似，**解析解随时间只是平移**。因此它同时检验三件事：格式的空间精度、时间推进的相位（移动）误差、以及误差本身是否随时间放大（“误差守恒”）。

| 项 | 取值 | 来源 |
|---|---:|---|
| 计算域 | 周期 [−5, 5]²（min-image 折返参考） | `PERIOD` / `REFERENCE_MODEL` |
| 均匀流 | (u∞, v∞) = (1, 1)，|u∞| = √2 | `BACKGROUND` |
| 涡强度 | β = 5 | 记录 `config.strength` |
| 设计点 | DG-P2 / 32²，CFL 0.2，t = 1 | `LADDER` 设计点 |
| 本记录 | DG-P2 / 64²，CFL 0.2，t = 10，9369 步 | `results/dg_d2_n64_t10_cfl0.2.json` |
| 自由度 | 147456 | 同上 |
| 容差锚点 | 6.0%（t=1 最大场相对 L2） | 锚点验收口径 |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **取证而不是取状态**：`run_compute fetch cj4263` 取回工件 tar（47 MB），证书逐字从工件内的 `provenance_certificates.json` 读出。作业状态 `succeeded`+`rc=0` 不作数（round 15 三个分片就是“成功”却零证据）。
2. **严格入库**：`tools/land_provenance_certificates.py <tar> --report` 逐条过滤——检查点必须在本地存在；`task/degree/cells/cfl/final_time/strength` 必须与本地记录一致；必须命名当前参考模型 `min_image`；`replay_delta` 有限且 ≤ 1e-9；重放百分比与存储百分比同容差一致；已有证书只能被更好的替换。结果：`landed 1`、`rejected 0`、`kept 18`，磁盘证书 19 份。
3. **重放策略（本轮的实质发现）**：该检查点在 4 线程与 8 线程下重放均在 `tensordg/euler_dg_2d.py::_traces` 的 `torch.einsum` 内 SIGSEGV（round 17/18 记录），而**单线程**重放一次通过。round 19 的重放固定 1 线程：`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1 python3 tools/provenance_certify.py --keys dg_d2_n64_t10_cfl0.2 --threads 1 --isolate`。
4. **判决由未改动的聚合器重算**：`python3 -m benchmarks.case_euler_vortex_errconserve --aggregate --figures`。本轮没有改动任何判据文本、容差、或已落盘的读数。

工具链：PyTorch(float64, CPU) DG 求解器 `tensordg/euler_dg_2d.py` + SSP-RK3、算例驱动器 `benchmarks/case_euler_vortex_errconserve.py`、重放器 `tools/provenance_certify.py`、入库器 `tools/land_provenance_certificates.py`。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 本轮落地的证书

| 字段 | 值 |
|---|---:|
| `task_key` | dg_d2_n64_t10_cfl0.2 |
| `reference_model` | min_image |
| `replay_delta` | 0 |
| `stored_max_percent` | 0.00122451 |
| `replayed_max_percent` | 0.00122451 |
| `steps` | 9369 |
| `wall_seconds` | 2084.37 |
| `certified_at` | 2026-11-16T10:00:49 |
| 记录自身 `wall_seconds` | 2665.53 |

`replay_delta = 0` 表示该检查点的误差读数被**逐位复现**（四个场的绝对偏差全为 0），这比 1e-9 的准入门槛严格 9 个数量级。

### 3.2 十一项判据

| 判据 | 状态 | 读数 |
|---|---|---|
| C1 容差（t=1 最大场相对 L2） | ✅ 通过 | 0.04926 |
| C2 误差守恒（累积指数 α） | ✅ 通过 | — |
| C3 离散守恒 | ✅ 通过 | — |
| C4 熵预算 | ✅ 通过 | 6.3361e-07 |
| C5 P0 交叉阶 | ✅ 通过 | — |
| C6 阶梯完整性 | ✅ 通过 | 40/40 |
| C7 设计空间不变性 | ✅ 通过 | 0.44007 |
| C8 阶数不变性 | ✅ 通过 | 0.44007 |
| C10 方向不变性 | ✅ 通过 | 0.31288 |
| C9 设计阶 | ✅ 通过 | 2.0424 |
| C11 证据溯源完整性 | ✅ 通过 | recorded 21 / replayed 19 / unverified 0 |

重放偏差最大值：1.133e-12（最差 `dg_d2_n32_t5_cfl0.4`，门槛 1e-9）。

## ④ 结果图

`fig_round19_c11_closure.png`（随报告内嵌）：

- 左：全部 19 份重放证书的复现偏差，对数轴，红色虚线为 1e-9 准入线——**没有一份接近门槛**（最差 1.13e-12，余量 882×）；
- 右：C11 的记录分布 `recorded 21 / replayed 19 / unverified 0`，整包 `verdict = PASS`。

误差–时间曲线、设计空间并列图、离散不变量漂移图由 `--aggregate --figures` 重算并保留在 `case_euler_vortex_errconserve/`（`fig_error_vs_time.png` 等），本轮物理读数未变，故不重复内嵌。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **作业回执不是证据，工件才是**。三次“成功”的作业曾带来零证据；本轮判据是工件里的证书文件与其中的 `replay_delta`。
2. **崩溃是线程相关的，但只用单样本证明过**。4/8 线程各崩一次（`_traces` 内原生 SIGSEGV），1 线程在本算例 DG 阶梯上 3/3 成功（`dg_d2_n32_t10`、`dg_d2_n64_t2`、`dg_d2_n64_t10`）。这是**策略依据**，不是因果定论——证书里目前**不记录线程数**，所以“这份证书是几线程跑出来的”只能从递交命令行读到，属于已知缺口（未合并分支上的 round 18 提交正是补这个字段；本单不去重复实现）。
3. **C11 只回答溯源，不回答精度**。它证明“这条记录的数字是它声称的那次计算产生的”，格式准不准是 C1–C10 的事，两者不可互相替代。
4. **适用范围**：锚点容差是 **t = 1 快照**口径。同一阶梯里 `p0_n64_t10` 的读数（14.30%）超出 6%，那是长时一次格式的**已知**表现，不在锚点声明范围，也不能用 C1 的通过去覆盖——本报告如实并列两者。
5. **容差出处的诚实边界**：6% 这个数是算例池条目**原文照搬**，并未从 Yee 等（2000）正文重新推导（仓库里没有该文原文，`metrics.json` 的 `reference` 字段已写明这一点）。被检验的数字测的是**解析等熵涡解**，与“哪篇论文先提出该算例”无关。
6. **判决会陈旧**：证书与记录是独立落盘的数据，`metrics.json` 只是投影；任何一份证书落地后必须重跑 `--aggregate`（`tests/test_metrics_provenance_freshness.py` 把这条变成断言）。

## ⑥ 误差判定（逐项对基准，显式回答“是否 < 3%”）

| 关键量 | 基准 | 实测 | 相对基准 | 是否 < 3% |
|---|---|---:|---:|---|
| 设计点 DG-P2/32² t=1 最大场相对 L2 | 等熵涡闭式解 | 0.009431% | 6% 锚点容差余量 636× | 是（<0.02%） |
| C1 最差族（t=1，`p0_n64_t1_cfl0.4`）最大场相对 L2 | 等熵涡闭式解 | 4.926% | 锚点容差 6% | **否**（>3%，但通过 6% 的验收口径） |
| 本轮重放复现偏差 `dg_d2_n64_t10_cfl0.2` | 该记录自身存储误差 | 0 | 准入门槛 1e-9 | 是（余量 ≥ 9 个数量级） |
| 全部 19 份证书的最差复现偏差 | 各自存储误差 | 1.13e-12 | 准入门槛 1e-9 | 是 |
| C9 设计阶（DG-P1/P2） | 理论 p+1 | 2.042 | 判据 ≥ 0.8(p+1) | 收敛阶读数，非相对误差（不适用 3% 口径） |
| C5 P0 交叉校验阶 | 理论 1 | 0.8488 | 判据 ≥ 0.8 | 收敛阶读数，非相对误差（不适用 3% 口径） |
| 溯源覆盖 | 40/40 条记录可判 | 40/40 | 0 | 是 |

**诚实结论**：本锚点的验收容差是 6%（t=1 快照），最差族 4.926% 通过、余量 1.22×；若按 3% 的严格口径，该族**不达标**，本报告不粉饰。设计点（DG-P2/32²）0.0094% 远优于 3%。本轮新增的实质事实只有一条：C11 由 `not_evaluated` 变为 `passed`，整包 `INCOMPLETE → PASS`。

## ⑦ 复现

```bash
run_compute fetch cj4263                      # 取回工件 tar
PYTHONPATH=. python3 tools/land_provenance_certificates.py \
    /srv/tensor/agent/cluster/fetch/cj4263_n315.tar.gz --report /tmp/landing_r19.json
PYTHONPATH=. python3 -m benchmarks.case_euler_vortex_errconserve --aggregate --figures
PYTHONPATH=. python3 -m pytest tests/test_case_euler_vortex_errconserve.py -q   # 44 passed
PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round19_c11_closure.py
```
