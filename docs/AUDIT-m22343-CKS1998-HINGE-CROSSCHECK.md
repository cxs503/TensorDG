# AUDIT · REPRO-DG-CKS1998 hinge 编码的独立核验（m22343 · verifier 席）

> 被审计对象：m22274 / 报告 20081「CKS1998 A2/E3 不等式判据的 hinge 编码」
> （commit `779f822`，`bench_spec.HINGE_RULES` + `bench_workload._measure_inequality`）
> 本单：m22343，锚点 `REPRO-DG-CKS1998`，分支 `agent/m22343`
> 集群作业：`cj4054`（`--profile bench` 全度量 stdout，2026-11-15 19:54 递交，排队中）

## 0. 审计独立性声明（先说局限）

* 我是同一工作树、同一作者体系的 verifier 席：**不是**外部审计。独立性仅体现在
  "重算路径与被审对象不共享实现"上（本文 §2 的重算公式全部在本单新写的
  `validation/repro_cks1998/hinge_audit.py` 里独立实现，不 import 被核验的实现）。
* 被审对象**没有**原始实验日志外的第三方参考；A2/E3 的比对基准是 CKS1998 论文的
  **不等式**（半阶地板、TVD 无新极值），不是某张具体表格里的数字，故"逐项数值对标"
  在论文层面不可得（论文给的是收敛**率**与**性质**，非可复制的标量参考值）。
* 未做：平台侧（TensorPlatform）契约改动 —— 越界，禁碰。

## 1. 被审主张 vs 证据（逐条绑定）

| 被审主张（报告 20081） | 证据（原始指针） | 审计核验 | 判定 |
|---|---|---|---|
| hinge 编码 **PASS ⟺ 原始不等式** | `bench_spec.hinge/min/max` + `tests/test_cks1998_inequality_hinge.py::test_hinge_decision_is_equivalent_to_the_raw_inequality` | 审计独立复算：`min/max` 与 `/b` 为 IEEE 精确运算，界上逐位饱和；等价性对任意 x 成立（代数事实，非经验） | 成立 |
| A2 实测观测阶 0.6496 ≥ 0.5 | 报告 ③ 表（cells 40/80/160，P1–P3） | 本审计在 smoke 档独立重跑：0.6501 / 0.6501 / 0.6501（三阶同值）；bench 档参见 `cj4054` | 成立（余量 30 %） |
| E3 门例 strict 超调 0.000 % ≤ 1 % | 报告 ③ 表 | 独立重跑一致（0.000 %） | 成立，但**门例选择偏松**（见 §3 F1） |
| "超调 0 % 对 1 % 的界"此前不可编码 | 报告 ①② + `NOT_ENCODED` 历史 | 复述正确：双侧相对误差会把完美 TVD 判成 100 % 误差 | 成立 |
| 平台裁定 = PENDING（rows=0） | `GET /tensor/bench/l4-verdict/22274` → `CLAIMED_NOT_PROVEN, rows=0` | **审计实测仍是 rows=0**：hinge 的五条度量至今未进 `tensorbench_results` | 主张如实，未夸大 |
| 覆盖 18 条度量（13 + 5） | `bench_spec.METRICS` 长度 | `13 + len(HINGE_RULES) = 13 + 5 = 18` 成立；平台实际入库仍只有旧 13 条（mission 20922/19918） | 部分成立（规格 18 / 入库 13+0） |

## 2. 本单补的缺口：hinge 不再"自证"（代码增量）

m22274 自己披露的结构性局限（报告 §5.1）：**hinge 由 workload 施加**，平台只看到已饱和的
got，"raw → hinge" 不在平台侧复算。⇒ 在此之前，一个**不测量、直接打印 `got=1.0`** 的
workload 可以通过本仓 ingest 并把平台判绿。这是材料级缺陷（gate 可被绕过），不是文字问题。

本单新增 `validation/repro_cks1998/hinge_audit.py::verify_hinge_consistency`，在同一份
stdout 文档内做**四方一致性**核验，并接进 `bench_spec.validate_documents`（⇒
`tensorbench_ingest.py --post` 拒发不一致的 bundle）：

```
rows（逐网格原始量） --独立重算--> raw --clamp--> hinge == metrics.<name>.value
```

* A2：从 `rows` 的 `(cells, l1_errors)` 用 log-log 最小二乘**重算**观测阶；
* E3：从 `rows` 的 `(pt_min, pt_max)` 用 `100*max(0, pt_max-1, -pt_min)` **重算**点值超调
  （为此 `bench_workload._measure_inequality` 的 E3 门例行新增 `pt_min/pt_max/avg_*`，
  属**纯增量**字段，不改变任何既有数值）；
* 再核对：`validation.hinge.bound/kind` 与预登记 `HINGE_RULES` 一致、重编码的 hinge 与
  文档 hinge 逐位一致（`HINGE_TOL=1e-9`）、`metrics` 指针值 = hinge、以及
  **决定等价**（hinge 过门 ⟺ 原始不等式成立）。

回归门 `tests/test_cks1998_hinge_audit.py`（**11 条，实跑 8 passed + 3 依赖真实 smoke 载荷**）：

| 门 | 断言 |
|---|---|
| 诚实文档 | 四方一致、`raw_rel_diff < 1e-12`、决定 PASS |
| 伪造饱和 got | 越界梯次（阶 0.2）被伪造成 `hinge=1.0` → 抛 `DocError("rows 重算的 raw …")` |
| raw 与 hinge 不服 | 只改 `validation.hinge.raw` → 抛 |
| 篡改点值极值 | `pt_max=1.05` 而自述 raw=0 → 抛 |
| 界漂移 | `bound` 改成 2.0 → 抛（与预登记不符） |
| 缺 hinge 块 | 抛（启用 hinge 度量却不发布 raw 绑定） |
| 判别力 | E3 超调 5 % ⇒ hinge=5.0 ⇒ 相对误差 4.0 ≫ 1e-9 ⇒ **FAIL** |
| 真实载荷 | `--profile smoke` 全量文档 5 条 hinge 全部一致且 PASS；篡改 `pt_max` 后 `validate_documents` 抛 |
| 局部阶披露 | A2 每条相邻网格对的最小阶也 ≥ 0.5（不只总斜率） |
| 判别力扫描 | N=40 P1：`unlimited` 判红、`strict_all` 判绿，且 `unlimited > strict` |

**仍未闭合的那一层（不粉饰）**：这些重算仍在**同一份由被测代码产出的文档内**，抓得到伪造与
串号，抓不到"workload 从一开始就不测量"。真正的独立复算必须由**平台侧按 raw 重新编码**
（契约改动，本仓无权，属 TensorPlatform 范围）。

## 3. 严重度排序的发现

**F1（中 · 门例选择偏松 = vacuity 风险，非阻塞）**
E3 的门例配置是 `troubled_policy="strict_all"`（逐单元 minmod，**按构造**即无新极值），
而 CKS1998 §4 讨论的是**TVB 指示器**（`tvb_constant=0`）。把门钉在"构造即真"的配置上，
0.000 % 是代码性质而非数值证据。仓内同时保留了非门例（公开指示器 `tvb_M0`）的越界记录
（N=80 时 P1 1.389 %），说明该量**确实会越界**、判据有判别力；但门例本身不承担判别任务。
建议：把 E3 的门例改为**论文配置**（tvb_M0）并预登记分辨率，`strict_all` 降为对照；
或双门并列（论文配置 + 严格策略），任一越界即在 rows 可见。本单**未改门**（改门属
预登记变更，需与 owner/账本同步，且 m22274 已按 strict 预登记），仅登记风险。

**F2（中 · 单分辨率门的脆性，可复现）**
同一判据在 N=80 与 N=100 之间翻转（`tvb_M0` P1：1.389 % → 0.000 %）。⇒ "超调 ≤ 1 %"
在论文配置下**不是分辨率无关的性质**（点值超调随网格振荡）。单点预登记门会把
"网格选得好"当成"性质成立"。建议门改为两分辨率对（粗网格必须也给数），并把
分辨率敏感性写进验收（本单已把该扫描做成本仓工具
`python -m validation.repro_cks1998.hinge_audit <doc> <outdir>`）。

**F3（低 · 判据的"物理含义"弱）**
A2 的 ≥1/2 地板在 P1/P2/P3 上实测都是 0.650（三阶同值）：限幅后退化为同一 O(h^{0.65})，
"阶"不再区分设计精度。门是对的，但**不能**用它的 PASS 推断实现的高阶性；报告 20081 §5.2
已如实登记，本审计确认该披露充分。

**F4（低 · 口径残差）**
`HINGE_TOL=1e-9` 作为平台容差在语义上是"浮点护栏"而非判据容差；平台库里
`tolerance_rel=1e-9` 看起来像"极严的物理容差"。属可读性风险，已在 §2 的核验里显式区分
（`bound` 进 raw 比较、`HINGE_TOL` 只进 ulp 比较）。

**F5（结构 · 平台侧未覆盖）**
hinge 的五条度量**至今 rows=0**（`l4-verdict/22274` 实测）。本单递交 `cj4054` 以产生
`--profile bench` 的全度量 stdout，打通 `POST /tensor/bench/results` 的最后一段；
本审计完成时作业仍在排队，故本单**不宣称** L4 已闭合（honest：`not_run`，附命令）。

## 4. 处置（disposition）

* 对 m22274 的 hinge 编码：**approve with notes**（编码本身正确、等价性成立、披露充分；
  F1/F2 是门设计问题，不推翻其结论）。
* 对本单增量：hinge 的"自证"缺陷已用**文档内四方核验**降级为"可交叉重算"，ingest 门生效。
* 建议后续（可拆单）：① 平台侧按 raw 重编码 hinge（需 TensorPlatform 战役单）；
  ② E3 双门（论文配置 + 严格策略）；③ 把 `cj4054` 的 stdout 走 ingest 收尾 L4。

## 5. 复现命令

```bash
# 四方一致性 + 判别力审计（本单新增）
OMP_NUM_THREADS=1 TORCH_THREADS=1 \
  python validation/repro_cks1998/bench_workload.py --profile smoke \
    --out artifacts/m22343_hinge_audit/smoke_doc.json > /dev/null
python -m validation.repro_cks1998.hinge_audit \
  artifacts/m22343_hinge_audit/smoke_doc.json artifacts/m22343_hinge_audit
# 回归门
python -m pytest tests/test_cks1998_hinge_audit.py -q
# L4 收尾（cj4054 成功后）
python validation/repro_cks1998/tensorbench_ingest.py \
  --stdout <cj4054 stdout> --post --job-id 4054 --mission 22343
```

**未运行 / 无法判定（标 unknown，不标 passed）**：`discriminating_power` 的四分辨率扫描
（N=40/80/100/200 × P1/P2 × 三种限幅器）在本机被 120 s shell 上限截断（`exit=-9`），
本轮计入 `not_run`；命令同上，应在集群跑（或拆分网格）。