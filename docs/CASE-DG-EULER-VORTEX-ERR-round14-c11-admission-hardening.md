# CASE-DG-EULER-VORTEX-ERR round 14：C11 准入门槛下沉到判据本身

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜算例池·等熵涡移动误差守恒｜ref: Yee, Sandham & Djomehri (JCP 162, 2000)
> 容差 6.0%（t=1 快照，最大场相对 L2）｜本单 mission 22090
> 整包裁决（本轮结束时）：**INCOMPLETE**（C11 有 5 条旧读数仍无重放证书）

## ① 本轮的缺口（怎么发现的）

round 12/13 把「重放证书」作为 C11 的证据形态引进来，并写了一个准入过滤器
`tools/land_provenance_certificates.py`：落地前校验检查点存在、物理配置逐字段一致、
参考模型一致、重放偏差 ≤ 1e-9、重放百分比与存档百分比吻合。问题在于**这层校验只在
落地工具里**：

* `provenance_certificates.json` 是一个普通 JSON 文件，手改、未来合并、
  键错位都能把一条「属于别的检查点」的证书放进去；
* 而 C11（`benchmarks/case_euler_vortex_errconserve.py: provenance_audit`）只看两件事：
  证书声明的参考模型是否等于聚合口径，以及它自报的 `replay_delta` 是否 ≤ 1e-9。

也就是说：一条声称 `reference_model=min_image`、`replay_delta=0`，但配置写着
**另一套网格/时刻**的证书，会被判为 `replayed`，从而让 C11 变 `passed`。
这正是本算例历史上踩过的坑（round 1–3 的「误差放大」是参考解缺陷，不是格式问题），
所以「证据必须绑定到它声称的检查点」不能只是工具约定，必须是判据自己算出来的。

## ② 怎么做的（最小改动）

1. 新增 `certificate_mismatch(record, certificate)`（`benchmarks/case_euler_vortex_errconserve.py`）：
   从记录与证书**自身**重新推导三处绑定——
   * 物理/离散字段 `task/degree/cells/cfl/final_time/strength`（另一套网格就是另一个问题）；
   * 证书自称重放的检查点名 `task_key`；
   * 它自称复现的存档百分比 `stored_max_percent`（`replay_delta` 很小但绝对数不同，
     是「配错对」而不是「复现」）。
2. `provenance_audit` 把这三点并入既有的 `problems` 列表：命中即 `certificate_rejected`，
   并在行内保留 `certificate_mismatch` 作为可审计的拒绝理由。原有两条检查
   （参考模型、重放偏差）一条未动。
3. `evaluate_gates` 的既有规则「`mismatched or rejected` ⇒ C11 = failed」不变，
   因此被拒证书**不可能**让整包变 PASS。
4. 新增 `tests/test_provenance_gate_admission.py`（13 例）钉住判据（不是钉住工具）。

## ③ 计算结果（判据加固的可验证后果）

| 场景 | 加固前 | 加固后 |
|---|---|---|
| 证书配置另一套网格（cells 32 → 64） | `replayed`（被采纳） | `certificate_rejected`，理由含 `cells` |
| 证书配置另一时刻（t=2 → 10） | `replayed` | `certificate_rejected`，理由含 `final_time` |
| 证书 CFL / 阶数不同 | `replayed` | `certificate_rejected` |
| 证书 `task_key` 指向别的检查点 | `replayed` | `certificate_rejected`，理由含 `task_key` |
| `stored_max_percent` 只有记录的一半 | `replayed` | `certificate_rejected`，理由含 `stored_max_percent` |
| 参考模型不同 / delta > 1e-9 / delta = nan | rejected | rejected（原样保留） |
| **已落地的 14 条真实证书** | `replayed` | **`replayed`（全部仍然通过）** |

表格最后一行的意义：这次加固**关的是洞，不是提高验收门槛**。用磁盘上真实的
`provenance_certificates.json` + `results/*.json` 复算，14 条证书全部仍判 `replayed`，
被拒 0 条（`tests/test_provenance_gate_admission.py::test_landed_bundle_is_still_admitted`）。

测试：`PYTHONPATH=. python3 -m pytest -q tests/test_provenance_gate_admission.py tests/test_case_euler_vortex_errconserve.py tests/test_provenance_certify_shard.py tests/test_land_provenance_certificates.py`
→ **77 passed**（本机短测试，<8 核 <10 min）。

## ④ 未完成的部分（honest）

欠账本体**没有**在本轮付清：C11 仍有 5 条旧读数无证书
（`dg_d2_n32_t10_cfl0.2`、`dg_d2_n32_t10_cfl0.4`、`dg_d2_n32_t5_cfl0.2`、
`dg_d2_n64_t10_cfl0.2`、`dg_d2_n64_t2_cfl0.2`，后两条是最贵的两条）。

已按 round 13 的协议把 5 个重放分片递交集群（`cj3888`–`cj3892`，
`tools/provenance_certify.py --remaining --shard i/5 --threads 8`），但本单结束时
它们仍在排队（集群在飞 57–60 个作业、200 个在跑，调度器派发）；因此
**证书尚未取回落盘，`metrics.json` 未重算，包裁决仍为 INCOMPLETE**。
不宣称任何「C11 已闭合」「整包 PASS」的结论。

## ⑤ 注意点

1. **判据不能依赖工具的礼貌**：准入过滤是便利，不是证据链的一环。凡是判据要用的性质，
   必须在判据里重算。
2. **加固不得反向收紧**：新增检查只可能把 `replayed` 降级，不会把 `unverified` 升级；
   已落地证书零变化是这一点的证据。
3. **本轮的 `metrics.json` 未重算**：`--aggregate --figures` 属重计算，本机被计算门禁拦截；
   且加固对现有工件是恒等变换（14/14 仍 `replayed`，未证实仍 5），故没有重算的必要性证据
   需求——这一点是**推断**，等 5 条证书落地后由聚合器一次性重判。
4. **图的缺失**：本轮是判据/测试加固，无新场量计算，因此不生成新图；既有场图与
   误差-时间曲线见 `docs/CASE-DG-EULER-VORTEX-ERR-round12-evidence-closure.md`
   的 `fig_error_vs_time.png` / `fig_design_space.png` / `fig_invariants_vs_time.png`。

## ⑥ 误差判定

| 关键量 | 实测 | 基准 | < 3%？ |
|---|---|---|---|
| DG p=2 32² t=1 设计闸门（既有读数，未变） | 0.009431% | 闭式周期解 | 是 |
| 证书准入：14 条真实证书被误拒 | 0 / 14 | 应全部采纳 | 是（无误拒） |
| 证书准入：6 类伪造/错位证书被误收 | 0 / 6 | 应全部拒绝 | 是（无误收） |

结论：本轮不改动任何既有验收读数；加固后误拒 0、误收 0，均有测试断言。
**整包裁决仍为 INCOMPLETE**，C11 待 5 条证书落地后由聚合器重判。
