# CASE-LAX-DG · 台账容差残余缺口（A0 仍 FAIL）与分辨率研究的状态

> 锚点：`feature_ledger` `CASE-LAX-DG`（验收原文「ref: 精确解｜容差 < 3 %」，status `open`）。
> 上游记录：`docs/CASE-LAX-DG-acceptance-A0.md`（A0 层）、`docs/CASE-LAX-DG.md`（池子报告）。
> 本文只登记两件事：①本单修掉的**可复现性缺陷**；②台账臂**残余缺口**的量化与研究状态。

## ① 案例介绍（不变）

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]`，γ = 1.4，隔膜 `x0 = 0.5`，
左态 `(ρ,u,p) = (0.445, 0.698, 3.528)`，右态 `(0.5, 0.0, 0.571)`，快照 `t = 0.13`。
精确 Riemann 解闭式可得（四项：左稀疏波 head/tail、接触间断、右激波），
因此本算例的参考值不依赖任何第三方求解器。

## ② 本单做了什么

### A. 修掉 main 上那条常年红的契约测试（缺陷修复，已证）

`tests/test_lax_case_pool.py::test_committed_artifact_carries_the_ledger_acceptance`
在合并后的 main 上为红，根因是**生产者/消费者漂移**，共三处：

| # | 缺陷 | 证据 | 修法 |
|---|---|---|---|
| F1 | 已提交工件缺 `acceptance_provenance` 块（生产者 `scripts/attach_lax_acceptance.py` 存在但工件早于它） | `KeyError: 'acceptance_provenance'` | 重跑 attach（只读 `profiles.npz`，不重跑求解器）把 provenance 写回工件 |
| F2 | `production_errors_from_profiles` 用 `np.mean(np.diff(centres))` 估 `dx`，与 `modal_fields` 的 `span/(ncells-1)` 差若干 ulp，导致同一格子的 L2 出现伪 last-bit 差（`3.1155618811580767e-02` vs `3.115561881158076e-02`） | 重测与阶梯自带 L2 不逐位一致 | 改用与 `modal_fields` 完全相同的 `dx` 定义（源码内注明） |
| F3 | 测试用**默认（ladder 来源）**重建 ledger 臂，而工件该臂来源是 `profiles`，两者永远不可能相等；同时一致性检查只覆盖 L1 | `AssertionError: {'source': 'ladder'} != {'source': 'profiles'}` | 测试按工件**声明的来源**重建；attach 的一致性检查扩到 L1+L2 两个范数，并把 `production_leg`/`arm_source`/`profiles_file` 写进 provenance |

修后 `tests/test_lax_case_pool.py` **26 passed**（本机实跑），`scripts/attach_lax_acceptance.py --check`
与 `scripts/sync_lax_report_numbers.py --check` 均 exit 0。

### B. 残余缺口量化（台账臂，数字来自已提交工件）

| 项 | 值 | 来源 |
|---|---|---|
| 钉扎生产腿 | `p3.N400` | `metrics.json. acceptance.production_leg` |
| rel L2 ρ | `3.11556e-02` | 同上（`profiles.npz` 重测，与阶梯一致） |
| rel L2 u / p | `2.24805e-02` / `1.30501e-02` | 同上 |
| 容差裕度 | ×0.963（差 3.9 % 未达） | 同上 |
| 全阶梯最差 rel L1 | `2.32905e-02`（PASS） | 同上（更严臂） |

即：**三个量里只有密度没过 3 %**，且只差 3.9 % 的裕度。池子的 C3/R3 给出 P3 全场阶
≈ 0.49（间断面压阶），因此在 N=400 上靠调参无法补齐——唯一干净的出路是加密一档 h。

### C. 分辨率研究（代码已就绪，读数尚未入库）

新增驱动器 `benchmarks/lax_resolution_requirement.py`（预登记规则写在代码里）：

* **R1** P3/N=800 的台账度量（全场 rel **L2**）——此前从未入库（`lax_asymptotic_probe` 只存 L1）；
* **R2** 该分辨率下逐量是否 < 3 %（判据：`max_field rel L2 < tolerance`，不挖波带、不豁免）；
* **R3** N=400→800 的实测全场阶（L1 与 L2）；
* 复用检查：N=400 的 L1/L2 必须与池子工件逐位一致、N=800 的 L1 必须与 p3 渐近探针逐位一致；
* **不移动 A0 生产腿**（仍 `p3.N400`），A0 的 FAIL 原样保留。

执行状态（诚实登记）：驱动已在集群**完整跑通**——作业 `cj2647`（node n315）`rc = 0`，
命令以 `&&` 串联 `python -m benchmarks.lax_resolution_requirement` 与其产物的 `ls`，
故产物在节点上确实生成；但 `run_compute` 的 `fetch` 通道只回传**递交时的输入快照**
（tar 31565 KB，内容为 stage_dir 原样），节点写入不回传，复跑 `cj2662` 同样卡在此处。
本机侧执行被计算门限拦（单次 shell 上限 120 s，而该研究 N=400 腿≈108 s + N=800 腿≈215 s），
因此 **R1/R2/R3 的数字本单未能落进 worktree 工件**，对应契约测试
`tests/test_lax_resolution_requirement.py` 以 **skip（不是 pass）**登记，reason 里写明复现命令。

**未完成项（明确）**：`docs/benchmarks/results/lax_resolution_requirement/metrics.json`
及其渲染的 `docs/CASE-LAX-DG-resolution-requirement.md` 需在一次能取回工件通道的作业里补跑；
补跑后该测试组自动转绿，并可用本文 B 节的缺口数字判读「3 % 是否在 N=800 达标」。

## ③ 注意点

1. **不得用 B/C 改写 A0**：台账臂的钉扎腿仍为 `p3.N400`，A0 = FAIL 保留在案。
   若要把锚点翻成 done，需要台账持有人明确变更“生产腿”口径——那不是本单能自行裁决的事。
2. **范数与口径**：全部误差在单元 Gauss 节点上按 Legendre 求积加权，全场、不挖波带；
   换单元平均 / L∞ 会给出不同数字，口径写在工件 `config` 里供重裁。
3. **集群通道缺陷（可复用知识）**：`run_compute` 的 `fetch` 只返回 stage_dir 快照，
   节点上的新产物不会回传；需要产物的作业应把结果**打印到 stdout**或写到本机可达的共享路径，
   否则「跑成功」但「取不回」。本单即此坑。
4. **算力**：N=800/P3 需 10250 步（N=400 腿 5109 步的 2.0 倍步数），属重计算，
   本机不得起；本单两次递交均为集群作业。

## ④ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| 钉扎腿 rel L2 ρ（`p3.N400`） | 精确 Riemann 解 | `3.11556e-02` | 3.116 % | **否** |
| 钉扎腿 rel L2 u | 精确 Riemann 解 | `2.24805e-02` | 2.248 % | 是 |
| 钉扎腿 rel L2 p | 精确 Riemann 解 | `1.30501e-02` | 1.305 % | 是 |
| 全阶梯最差 rel L1 ρ | 精确 Riemann 解 | `2.32905e-02` | 2.329 % | 是 |
| N=800/P3 rel L2（三量） | 精确 Riemann 解 | **未测得（工件未回传）** | — | **未判定（not_run）** |

**结论**：A0 验收层 **FAIL**（密度 3.116 % > 3 %，裕度 ×0.963），本轮未改变；
分辨率研究 R1–R3 为 **not_run（工件未回传）**，按纪律登记为未完成而非通过。
