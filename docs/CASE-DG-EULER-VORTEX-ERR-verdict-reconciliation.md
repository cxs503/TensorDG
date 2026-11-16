# CASE-DG-EULER-VORTEX-ERR：判读台账对账（单一事实源）

> 本报告的每个数字都由 `tools/verdict_reconcile.py` 从已入库工件程序化注入，无手抄；判据由脚本计算。原始对账件：`artifacts/verdict_reconciliation_CASE-DG-EULER-VORTEX-ERR.json`。

## ① 案例介绍

二维可压缩 Euler（γ = 1.4）等熵涡在双周期域 [-5, 5]² 上被均匀来流 (u, v)∞ = (1, 1) 刚性输运；β = 5，域长 L = 10、|u∞| = √2 的分量速度各为 1，故沿每一轴的平移周期为 T = L/u∞ = 10。t = T 时解析精确解与初始条件逐点相同——这正是本锚点「移动误差守恒」要判的东西：不是短时精度，而是一个整周期后误差是否仍被 6 % 容差覆盖。

本单不重算算例，而是审**判读**：同一个台账行在同一 commit 上被 7 个工作树、10 个断言判读的工件覆盖，其中 2 个互相矛盾。锚点能否关闭，取决于读哪一份。

| 项 | 值 |
|---|---|
| 台账 slug | `CASE-DG-EULER-VORTEX-ERR` |
| 容差（算例池原文） | 6 % |
| 权威工件 | `case_euler_vortex_yee/metrics.json` |
| 权威判读 | **PASS** （0.015163 %） |
| 对账后判定 | **PASS** |
| 阻塞问题 | 0 |

## ② 怎么做的

**方法**：不新增物理算例，而是把「判读」本身当作被测对象——为台账行建立单一事实源，并用两个可证伪的判据把失效模式钉住。

| 环节 | 实现 | 说明 |
|---|---|---|
| 枚举 | `tools/verdict_reconcile.py::collect` | 只认 `git ls-files` 跟踪的工件：读者 checkout 得到的那份证据才作数 |
| 提取 | `extract` | 四种工件形态（aggregation / gates / certification / config）归一到同一结构 |
| 三值归约 | `gate_state` + `tri_state` | 未求值的门禁只能是 `INCOMPLETE`，不许塌成 `FAIL` |
| 作用域声明 | `SCOPE_REGISTRY` | 断言判读的工件必须显式声明 `anchor` / `study:*` / `superseded`，未声明即阻塞 |
| 溯源扫描 | `producer_scan` | 没有任何脚本写它的路径 = 孤儿工件，判读已被冻结 |
| 参考模型证书 | `certificate` | 在门禁配置的同一块求积格上，直接算自由空间参考与周期参考的解析解之差 |
| 落盘 | `--mark` / `--report` | 被取代件旁写 `SUPERSEDED.json`，同时产出对账 JSON |

**工具链**：Python 3.11 + torch（解析解与求积）、pytest、matplotlib、git。

### 两个被测缺陷

**D1｜同一 slug、同一容差、同一门禁配置，两份工件给出相反判读。**

| 工件 | 门禁读数 | 发布判读 |
|---|---|---|
| `case_euler_vortex/metrics_yee.json` | 16.443285 % | `overall_passed: false` |
| `case_euler_vortex_yee/metrics.json` | 0.015163 % | `overall_passed: true` |

两份工件的 `reference` 字符串**逐字相同**、`config_id` 也相同（`dg_d2_n32_t10_cfl0.2`），因此从文件内容本身无法分辨谁是新账——谁先被读到，锚点就朝谁倒。

**D2｜「没跑」被发布成「失败」。** 多个研究工件暴露 `all(gate.passed)`，而它们各自输出目录里的大部分门禁根本没求值（`measured: nan`、`passed: None`，检查点在另一个树里），布尔值因此是 `False`——对任何消费者而言与「物理失败」不可区分。覆盖度事实被当成科学结论发布。

## ③ 计算结果

### 表 1：对账后的事实源（逐工件）

| 工件 | 作用域 | 分类 | 三值判读 | 门禁 % | 未跑门禁 | 生产者 |
|---|---|---|---|---|---|---|
| `case_euler_vortex/metrics_yee.json` | superseded | SUPERSEDED | FAIL | 16.4433 | 0 | 0 |
| `case_euler_vortex_errconserve/instrument_certification.json` | study:instrument-certification | STUDY | PASS | — | 0 | 4 |
| `case_euler_vortex_errconserve/metrics.json` | study:error-conservation-ladder | STUDY | PASS | — | 0 | 203 |
| `case_euler_vortex_errconserve/spatial_order.json` | study:spatial-order | STUDY | PASS | — | 0 | 2 |
| `case_euler_vortex_errconserve_lattice/lattice_adjudication.json` | study:lattice-adjudication | STUDY | INCOMPLETE | — | 0 | 1 |
| `case_euler_vortex_errconserve_lattice/metrics.json` | study:lattice-reference | STUDY | FAIL | — | 7 | 203 |
| `case_euler_vortex_errconserve_longtime/horizon.json` | study:longtime-horizon | STUDY | INCOMPLETE | — | 0 | 1 |
| `case_euler_vortex_errconserve_r6/reproduction.json` | study:reproduction-r6 | STUDY | PASS | — | 0 | 1 |
| `case_euler_vortex_errmechanism/metrics.json` | study:error-mechanism | STUDY | INCOMPLETE | — | 3 | 203 |
| `case_euler_vortex_yee/metrics.json` | anchor | AUTHORITATIVE | PASS | 0.0151626 | 0 | 203 |

### 表 2：D1 的根因证书（参考模型，不是格式）

| 量 | 值 |
|---|---|
| 被取代件的门禁读数 | 16.443285 % |
| 证书预测（自由空间 vs 周期解析解，t = 10） | 16.443616 % |
| 相对偏差 | 2.013e-05 |
| 与实测误差之比 | 1084.5 × |
| 证书匹配带（准入） | 1% |

证书在 t = 10 给出 16.4436 %，被取代件发布 16.4433 %，相对偏差 2.01e-05——这不是格式误差可以解释的量级，而是同一个量被两种参考模型各算了一遍。t = 1 时两者只差 0.0087 %（涡核仍在盒内，两个模型一致），差异随核心离开盒子的程度单调增长，到 t = 10（恰好一个平移周期）达到极值。**结论：该工件量的是参考模型错配，不是格式误差。**

### 表 3：D2 的改正（未跑 ≠ 失败）

| 工件 | 原发布布尔 | 未跑门禁数 | 三值判读 |
|---|---|---|---|
| `case_euler_vortex_errconserve_lattice/lattice_adjudication.json` | True | 0 | **INCOMPLETE** |
| `case_euler_vortex_errconserve_longtime/horizon.json` | — | 0 | **INCOMPLETE** |
| `case_euler_vortex_errmechanism/metrics.json` | False | 3 | **INCOMPLETE** |


反向的坑（自我复核查出）：`C6_ladder_completeness` 的判据是**计数**（已入库检查点 vs 已声明检查点），它实测 `status: failed` 却没有数字读数。只看 `measured` 的规则会把它塌成「未判」——把一个真实的失败藏起来。因此三值归约以工件自报的 `status` 为最强证据、`measured` 只在无 `status` 时兜底：

- `case_euler_vortex_errconserve_lattice/metrics.json`：`failed_gates=['C6_ladder_completeness']`、`not_run_gates=7 个`、判读 **FAIL**（真实失败与未跑门禁同时如实登记，互不掩盖）

## ④ 结果图

![verdict lollipop](docs/fig_verdict_lollipop.png)

*图 1：三个数字落在误差轴上的位置（对数轴）。权威工件 0.0151626 % 远在 6 % 容差线左侧；被取代件的读数与证书预测几乎重合——它量的是参考模型，不是格式。*

![certificate vs time](docs/fig_verdict_certificate_vs_time.png)

*图 2：证书随时间的取值（三个原始场 + 逐时刻最大值）。t = 1 时两个参考模型一致，涡核离开盒子的过程中差异单调增长，t = 10 达到被取代件发布的那个读数。*

![tri-state](docs/fig_verdict_tristate.png)

*图 3：「没跑」被发布成「失败」的现状与改正。柱长为该输出目录里从未求值的门禁数，标注为原布尔值 → 三值判读。*

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **本单没有重算任何物理算例**。它审计的是判读与证据链，能量守恒/收敛阶等物理结论仍以既有研究报告为准；本单不构成对它们的独立复算。
2. **作用域声明是人工维护的表**（`SCOPE_REGISTRY`）。这是有意的：把作用域写成「猜出来的」就等于把 D1 换个形式再犯一次。代价是新增断言工件必须显式登记，未登记会被 `--check` 阻塞——这是特性不是缺陷，但确实需要下一单维护。
3. **证书带 1 %**。同一差异两种算法之间实测差 ~2e-5，而真实格式误差比 16 % 小三个量级，1 % 分离得很开；但如果将来出现量级介于两者之间的可疑数字，该带需要重新论证。
4. **被取代件没有删除**。它被保留为历史证据并在旁边写了 `SUPERSEDED.json`——删证据会让「为什么曾经是 16 %」变成不可复查的口述。
5. **权威判读是 PASS，但台账 slug 仍是 open**。本单只负责让判读可被机械认定，关账动作属于台账流程，不在本单范围内。
6. **适用范围**：判读对账思路与域无关（任何「多份工件断言同一验收行」的场景都适用）；证书部分绑定本算例的自由空间/周期解析解，换算例需换证书。

## ⑥ 误差判定

| 关键量 | 基准 | 实测 | 误差 | 是否 <3 %？ |
|---|---|---|---|---|
| 权威工件的门禁读数 vs 6 % 容差 | ≤ 6 % | 0.0151626 % | 0.25 % of tolerance | ✓ 是 |
| 证书预测 vs 被取代件读数 | 被取代件自身 | 16.443616 % | 相对偏差 2.01e-05 | ✓ 是（远优） |
| D2 改正后的未跑门禁数 vs 命题「全部未求值」 | 原布尔 False | 3 个门禁 | 一致 | ✓ 是 |
| 阻塞问题数 vs 0 | 0 | 0 | 0 | ✓ 是 |

**逐项回答「是否 <3 %」**：

- 权威判读：0.0151626 % 对 6 % 容差，占容差 0.25 %，**是**。
- 根因证书：与被取代件读数相对偏差 2.01e-05，远小于 3 %，**是**。
- D2 改正的判据是「未跑门禁数 = 未求值门禁数」，属计数一致性（离散、无容差），已由 `tests/test_euler_vortex_verdict_reconciliation.py::test_incomplete_study_is_not_published_as_fail` 逐件断言，**是**。
- 本单未重算格式误差，因此不对「DG 格式精度」下新的误差判定；那一量沿用既有研究报告，本单不重复声称。

## ⑦ 复现

```bash
# 对账（阻塞则退出码 1）
PYTHONPATH=. python3 tools/verdict_reconcile.py --check
# 落盘：被取代件标记 + 对账 JSON
PYTHONPATH=. python3 tools/verdict_reconcile.py --mark \
    --report artifacts/verdict_reconciliation_CASE-DG-EULER-VORTEX-ERR.json
# 门禁测试
PYTHONPATH=. python3 -m pytest tests/test_euler_vortex_verdict_reconciliation.py -q
```
