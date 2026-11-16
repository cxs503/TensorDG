# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — DG 腿时间步轴（D0–D3）

> 锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（status=open，核对于开工前为 open，未被 done）
> 分支 `agent/m23262`　类型：验证欠账 / 缺陷修复　报告 id 20877

## 1. 缺口（读代码 + 账本得出，非猜测）

锚点已有四轮工作（认证引擎、四级梯次扩展、WENO7 带宽诚实性、多场/时间污染）。但
`benchmarks/xval_dg_round2_time_pollution.py` 的真实数据分支自建立起就是
`WITHHELD_MISSING_INPUTS`：6 个 CFL 细化 FV-WENO5 剖面从未落盘，T0–T5 **没有任何数字**。

同时发现一个**从未被问过的问题**：11 条已发布 DG 腿里 9 条在 CFL=0.4 报价，腿表从不说明
该误差是空间误差还是空间+时间误差。其中 P2/N=400 的 CFL 0.4/0.2/0.1 三条腿**已提交**且
只差时间步 —— 这条轴零新增算力即可分离。

## 2. 做了什么

1. `benchmarks/xval_dg_round2_time_pollution.py`：新增 `dg_temporal_axis()`（D0 可分辨性 /
   D1 污染比 / D2 决策稳定性 / D3 时间阶可用性）与 `markdown_dg_axis()`；接入 `audit()`
   为 `dg_temporal_axis` 字段；`audit_or_withheld()` 在 FV 侧 withheld 时**仍**报出 DG 轴
   （它只读已提交剖面）；修 2 个缺陷：`_jsonable` 把 bool 误转成 int（int 检查在 bool 之前）、
   `p_t` 除以零（两条 CFL 腿完全相同时崩溃）。
2. `validation/xval_dg_round2/gen_dg_time_axis.py`：产物驱动（json/md/png）。
3. `publish_dg_time_axis.py`：报告体与图注全部从 artifact 程序化注入。
4. `tests/test_xval_dg_round2_dg_time_axis.py`：10 项，含**负控制**（合成树里时间项主导时
   必须判 POLLUTED、必须报出决策翻转；缺腿必须响亮拒绝）。

## 3. 结果（程序化注入，零手抄）

| 腿 | CFL | err % | 判决 3% |
|---|---|---|---|
| DG P2 N=400 | 0.4 | 1.513771 | PASS |
| DG P2 N=400 | 0.2 | 1.515633 | PASS |
| DG P2 N=400 | 0.1 | 1.517166 | PASS |

* 时间差 rel L1(0.4→0.2)=5.065e-3 %，rel L1(0.2→0.1)=**2.502e-3 %**；认证带 U=0.041622 %
* **D0** resolvable=False → 时间误差**低于参考自带带**，只能界定（≤U），不能测量
* **D1** R_leg = 2.502e-3 / 1.5172 = **0.001649** ≪ 0.10 → CLEAN
* **D2** 三条腿同判决 PASS，最大移动 0.0034 pp ≪ 0.5
* **D3** 时间阶 1.0178 **不可用**（误差随 CFL 减小而微升，序列非单调 → 舍入/限制器交互）
* **状态 `DG_AXIS_CLEAN`**：已发布 CFL=0.4 腿误差是空间误差

## 4. 误差判定

| 关键量 | 基准 | 测量 | 误差 | < 3 %? |
|---|---|---|---|---|
| 时间差 | U=0.041622 % | 2.502e-3 % | 低 93.99 % | 是（不可分辨） |
| 腿误差随 CFL 漂移 | 0 | 0.003395 pp | — | 是 |
| 认证参考复现 | p=1.176735 / U=0.041622 % | 逐位一致 | 0 | 是 |

## 5. 注意点 / 未完成（honest not_run）

* FV 侧 T0–T5 仍 `unknown`：6 个 CFL 剖面已递交 cj4715（6 分片）/cj4716（补 N=800 CFL=0.075），
  本单结束时**仍在排队（前方 120+ 作业）**，工件未取回。`time_pollution.json` 保持
  `WITHHELD_MISSING_INPUTS`，未被覆盖成任何数字。
* D3 是诚实 FAIL（时间阶不可用），不是通过项。
* 适用范围：一维 Shu–Osher、DG P2 N=400、WENO 限制器；不外推到多维/其它 CFL。

## 6. 验证

```
python -m pytest tests/test_xval_dg_round2_dg_time_axis.py tests/test_xval_dg_round2_time_pollution.py -q
→ 40 passed, 4 skipped
```
