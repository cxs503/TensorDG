# CASE-DG-EULER-VORTEX-ERR round 17：把「已在集群算完但从未入库」的溯源证书收账

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜等熵涡移动误差守恒｜ref: Yee et al. (JCP 162, 2000)｜容差 6.0%（t=1 快照）｜mission 22612
> 整包裁决：**INCOMPLETE**（C1–C10 通过；C11 未闭合，`unverified` 由 3 条压到 1 条）

## ① 起因：上一轮把「递交≠证据」写成了结论，但没有回头检查在飞作业里到底有没有证据

round 16 的结论是「段错误抹掉了同批未写盘的证书，C11 仍有 3 条 `unverified`」，
并把三条单 key 作业 `cj4184`/`cj4185`/`cj4186` 留在了集群队列里当作后续。
本轮先做的不是重算，而是**去取回那三条作业的工件**——因为「作业成功但证据没入库」正是
round 15 已经犯过一次的错。

结果与 round 16 的叙述不一致：

| 作业 | 配置 | 线程 | 取回的证书 | `replay_delta` |
|---|---|---|---|---|
| `cj4184` | `dg_d2_n64_t2_cfl0.2`（1874 steps） | 1 | **有** | `0.000e+00`（位相同复现） |
| `cj4185` | `dg_d2_n32_t10_cfl0.2`（4684 steps） | 1 | **有** | `1.128e-12` |
| `cj4186` | `dg_d2_n64_t10_cfl0.2`（9369 steps） | 4 | 无 | — |

也就是说：三条里**两条的证据其实已经算出来了**，只是没人搬运；只有最贵的那条（4 线程）
既没有证书也没有存活。round 16 报告的「3 条 `unverified`」里，有 2 条是**账务问题，不是算力问题**。

## ② 怎么做的

1. `run_compute fetch` 取回 `cj4184_n330.tar.gz`（47 MB）、`cj4185_n330.tar.gz`、`cj4186_n330.tar.gz`；
   直接读工件内 `case_euler_vortex_errconserve/provenance_certificates.json`，得到上表。
2. 用**严格搬运器**入库，不做手工拷贝：
   `PYTHONPATH=. python3 tools/land_provenance_certificates.py <tar> ... --report /tmp/landing_r17.json`
   逐条过「检查点必须本地存在 / 物理字段必须与本地记录一致 / 必须命名当前参考模型 /
   `replay_delta` 有限且 ≤ `PROVENANCE_REPLAY_TOLERANCE` / 重放百分比与存储百分比同容差一致 /
   已有证书只能被更好者替换」。
   输出：`cj4184` landed 1（`dg_d2_n64_t2_cfl0.2`）、`cj4185` landed 1（`dg_d2_n32_t10_cfl0.2`），
   rejected 0，kept 16/16；磁盘证书 16 → **18**。
3. 重聚合：`python3 -m benchmarks.case_euler_vortex_errconserve --aggregate --threads 2`
   → C11 `unverified` 由 `['dg_d2_n32_t10_cfl0.2','dg_d2_n64_t10_cfl0.2','dg_d2_n64_t2_cfl0.2']`
   收敛为 `['dg_d2_n64_t10_cfl0.2']`，`certificates_on_disk = 18`，`failed = []`。
4. 剩下的 1 条按「1 线程」重放递交 `cj4263`（本轮时间窗内未完成，故不写成通过）：
   `PYTHONPATH=. python3 tools/provenance_certify.py --keys dg_d2_n64_t10_cfl0.2 --threads 1 --max-seconds 5000`
5. 新增 `scratch/repro_dg_segfault.py`：崩溃不再只靠「进程没了」来判定——`faulthandler` 捕获
   SIGSEGV/SIGABRT/SIGBUS 打印存活 Python 栈，watchdog 线程每 15 s 打印耗时/RSS/high-water mark
   （用来区分「内存漏增」与「固定足迹的纯数值长跑」），并在解释器退出前强制 flush，
   以便把「torch 退出期崩溃」与「跑数中途崩溃」分开。

## ③ 计算结果（由 `--aggregate` 程序化写出）

| 判据 | 状态 | 读数 |
|---|---|---|
| C1_tolerance | passed | 0.04926（% 级，门槛 6%） |
| C2_error_conservation | passed | — |
| C3_discrete_conservation | passed | — |
| C4_entropy_budget | passed | 6.336e-07 |
| C5_crosscheck_order | passed | — |
| C6_ladder_completeness | passed | — |
| C7_design_space_invariance | passed | 0.4401 |
| C8_degree_invariance | passed | 0.4401 |
| C9_design_order | passed | 2.042 |
| C10_direction_invariance | passed | 0.3129 |
| C7x_* | informational | — |
| **C11_provenance_integrity** | **not_evaluated** | `unverified = ['dg_d2_n64_t10_cfl0.2']` |

`all_passed = false`，`verdict = INCOMPLETE`，`failed = []`：没有判据被判失败，
只有一条证据缺席。

## ④ 结果图

`fig_round17_provenance.png`（随报告发布）：左图 C11 分类计数 round16 → round17
（`recorded 21 / replayed 16→18 / unverified 3→1 / rejected 0`）；右图三条在飞作业的取证结局。

## ⑤ 注意点 / 局限

1. **段错误根因仍未定，但嫌疑面被收窄**：`cj4184`（1 线程，1874 steps，405 s 级）与
   `cj4185`（1 线程，4684 steps，474 s 级）都完整跑完并写出证书；唯一失败的是
   `cj4186`（**4 线程**，9369 steps，2665 s 级）。「线程数」从 round 16 的「怀疑变量」
   升级为「与失败唯一共变的工作点」，仍非因果证明（n=1 失败样本），所以下一轮的重放统一用 1 线程，
   并且在 `scratch/repro_dg_segfault.py` 里带上了能出栈的诊断。
2. **本文不声称 C11 已通过**：18/19 条记录有证书，`dg_d2_n64_t10_cfl0.2` 仍缺，
   `verdict = INCOMPLETE` 原样保留在 `metrics.json` 里。
3. **取证纪律**：证书一律经 `tools/land_provenance_certificates.py` 入库，禁止手工搬
   `provenance_certificates.json`——否则「哪台机器算的、对不对得上本地记录」就无人核对了。
4. **作业 rc 不可信**（本轮踩到的坑）：`cmd | tail` 的返回码是 `tail` 的，`rc=0` 完全不能证明
   子进程没被信号杀死（round 15/16 的 `cj4121`–`cj4123` 就是这样「成功」的）。本轮的取回判据改成
   **读工件里的证书文件**，而不是读作业状态。

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 < 3% |
|---|---|---|---|---|
| t=1 最大场相对 L2（设计点 DG-P2/32²、P0/64²） | 解析等熵涡（Yee et al. 2000 算例定义） | 0.0493% | 相对容差 6% 余量 ×122 | 是（相对容差口径） |
| `dg_d2_n64_t2_cfl0.2` 重放复现 | 自身存储误差 | `replay_delta = 0.0` | 0 | 是（位相同） |
| `dg_d2_n32_t10_cfl0.2` 重放复现 | 自身存储误差 | `1.128e-12` | 1.1e-10 % | 是 |
| 设计阶 DG-P1/P2 | 理论 p+1 | 2.042 | — | 否（收敛阶读数，非相对误差；判据要求 ≥0.8(p+1)，如实登记） |
| 交叉校验阶 P0 | 理论 1 | 0.849 | — | 否（长时 P0 观测阶，判据只要求 ≥0.8） |
| 溯源完整性（C11） | 19 条记录均有证书 | 18/19 | 缺 1 | **否（1 条未验证 ⇒ 整包 INCOMPLETE）** |

结论：数值判据 C1–C10 全过、`failed` 为空；C11 因 1 条 `unverified` 未闭合，本报告如实记为
INCOMPLETE，不写成通过。