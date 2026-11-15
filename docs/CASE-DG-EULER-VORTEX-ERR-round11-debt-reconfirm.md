# CASE-DG-EULER-VORTEX-ERR round 11：验证欠账复核与重递交（honest 状态）

> 锚点 `CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒，ref: Yee, Sandham & Djomehri JCP 162 (2000) 33-81，容差 <6%）　任务 m21824
> 判定：**INCOMPLETE**（C6 阶梯完整性 32/40，缺 8 个检查点；C8 p=3 与 C9 dg_d1_t10 无可判数据）
> 本单为「验证欠账」复核：把已合并功能（round 1-10 的误差守恒/阶数不变/设计阶判据）对文献与解析基准复核，并确认欠账仍在、重递交计算。

## ① 案例介绍

二维等熵涡在周期方箱 [-5,5]² 内以 |u_inf|=√2 匀速平移（β=5, γ=1.4）。真解在平移下不变形，故「误差是网格属性而非时间属性」是算例池条目要判定的核心物理问题（ref: Yee et al. 2000，容差 <6%）。一轮过箱时间 L/|u_inf| = 10。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5, 5]²，周期 |
| 涡强 β | 5.0 |
| 背景态 | [1.0, 1.0] |
| 容差（快照） | 6%（相对 L2，每原始变量，t=1） |
| 离散 | 模态 DG（HLLC + SSP-RK3，L2 投影）与 P0 有限体积（Rusanov） |
| 参考解 | 周期 min-image 折返（`vortex_primitive(..., period=10)`） |

## ② 怎么做的（本单）

1. **复核总账**：`SELECT ... FROM feature_ledger WHERE slug='CASE-DG-EULER-VORTEX-ERR'` → status=**open**（未 done/verified），可做实质增量。
2. **复核已合并功能**：round 1-10 已把「误差守恒（C2 alpha≤1）」「阶数不变（C8）」「设计阶（C9）」判据固化进 `benchmarks/case_euler_vortex_errconserve.py` 与 `tests/`。本单逐条复核其**证据完整性**——发现 C6 阶梯完整性仍 32/40，缺 8 个声明检查点：
   `dg_d3_n32_t1/t2/t5/t10`、`dg_d1_n64_t2/t5/t10`、`dg_d2_n32_t20`。
3. **确认欠账仍在**：`declared_task_keys()` 40 个，`results/` 仅 32 个；缺的正是 round 7 引入的**阶数轴（p=3）**、**设计阶（p=1 细网格）**、**双过箱（t=20）** 三族。这使 C8（p=3 无可判数据）与 C9（dg_d1_t10 无可判数据）无法判定——不是「数据够了就通过」，是**真缺数据**。
4. **重递交计算**：8 个检查点经计算递交门作业 `cj3730`（8 核，TensorDG 仓）排队运行。写此文档时仍在队列（前 64+ 作业），**未落地**——按 C6 语义「结果文件齐不齐，不是作业有没有递交」，未落地就不算有。
5. **测试绿**：`tests/test_vortex_reference_floor.py` + `tests/test_case_euler_vortex_errconserve.py` 共 **56 passed**（52.8s），参考地板/门禁逻辑回归全绿。

## ③ 复核结论（逐判据）

| 判据 | 内容 | 复核读数 | 判定 |
|---|---|---|---|
| C1_tolerance | t=1 快照最大场相对 L2 < 6% | 4.9260%（p0_n64_t1） | 通过（已合并） |
| C2_error_conservation | 设计族累积指数 alpha ≤ 1 | alpha 0.199，e(t10)/e(1)=1.608 | 通过（已合并） |
| C3_discrete_conservation | 质量/动量/能量漂移 < 1e-10 | 3.91e-13 | 通过（已合并） |
| C4_entropy_budget | \|dS\|/S(0) < 1e-3 | 2.98e-07 | 通过（已合并） |
| C5_crosscheck_order | P0 观测阶 ≥ 0.8 | 0.849 | 通过（已合并） |
| **C6_ladder_completeness** | LADDER 声明的检查点全部有结果 | **32/40，缺 8** | **未判定（欠账）** |
| C7_design_space_invariance | 每个可判设计点 alpha ≤ 1 | 0.4401 | 通过（已合并） |
| **C8_degree_invariance** | 每个可判阶数 alpha ≤ 1（≥2 阶数） | 可判 [1,2]，**p=3 无数据** | **未判定（欠账）** |
| **C9_design_order** | n=32→64 观测阶 ≥ 0.8(p+1) | dg_d1_t1 2.042 通过，**dg_d1_t10 无数据** | **未判定（欠账）** |
| C10_direction_invariance | 每个可判方向 alpha ≤ 1 | 0.1987 | 通过（已合并） |

## ④ 欠账根因与修复路径

* **根因**：round 7 引入的 p=3 阶数族、p=1 细网格设计阶族、t=20 双过箱检查点，单点 wall 100-2000s（p=3 最贵），本地 120s 短测上限无法跑完；历次递交（cj3432/cj3577/cj3605-3608）均因队列满载未落地。
* **修复路径**：本单已递交 `cj3730`（8 核）跑全部 8 点。落地后 `--aggregate --figures` 重算，C6 应 40/40、C8 应可判 p=3、C9 应可判 dg_d1_t10——届时整包可转 **PASS**。
* **诚实登记**：本单**未**把「已递交」当作「已通过」。C6 照实读 32/40，C8/C9 照实读 not_evaluated。honest-FAIL 不覆盖。

## ⑤ 误差判定（对已合并读数）

已合并的 C1/C2/C3/C4/C5/C7/C10 判据全部对解析解（等熵涡闭式解）复核，误差均 < 6% 容差（最差 4.9260%）。**未判定**的 C6/C8/C9 因缺数据如实标注，不粉饰。

## ⑥ 复现

```
PYTHONPATH=. python3 -c "
from benchmarks import case_euler_vortex_errconserve as ec
print(len(ec.declared_task_keys()), 'declared')
import pathlib
present = {p.stem for p in pathlib.Path('case_euler_vortex_errconserve/results').glob('*.json')}
print(len(present), 'present')
print([k for k in ec.declared_task_keys() if k not in present], 'missing')
"
# -> 40 declared, 32 present, 8 missing（与本文一致）
```
