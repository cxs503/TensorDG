# CLOSEOUT m17687 — p 加密地板研究：时间步认证协议 + 外样本 σ 外推（p=7/8 实测仍未落地）

> 锚点 `CASE-DG-PREFINEMENT-EXPONENTIAL`（feature_ledger=open）· 上游
> `docs/CLOSEOUT-m16918-prefinement-criterion.md`（地板量化）、
> `docs/CLOSEOUT-m17067-exact-time-floor.md`（精确时间积分实测 p=7/8）
> 本单 m17687 · 数字由 `publish_floor_protocol_report.py` 从工件直读（零手抄）

## 0. 这一单做了什么（以及**没**做什么）

**做了**：`benchmarks/p_refinement_floor.py` 的两项协议加固——

1. **时间步自适应认证** `_temporal_converged_chain`：每个 p 从 `cfl` 起 `cfl→cfl/2` 反复细化，
   停止条件 `change <= max(1%, 10×roundoff 相对不确定度)`；并带 `step_budget` 上限。
   未认证档位如实返回 `temporal_converged=False` + `stop_reason`
   （`step_budget` / `refinement_budget_exhausted` / `no_refinement_allowed`），
   绝不静默当成空间误差引用。
2. **外样本外推检验** `_extrapolation_check`：用**已发布** p=6 点 + σ 外推
   `error(p)=e(6)·exp(-σ(p-6))`，对本次实测的每个 p>6 点给
   `measured/predicted`，即对 Babuska & Suri (1994) 解析解指数律做**预测检验**而非再拟合。
3. 配套：`tools/plot_floor_high_degree.py`（双面板图：误差-p 阶梯 + 局部衰减率 vs σ 带）、
   `tests/test_p_refinement_floor_high_degree.py`（工件闸 + 两条活闸，活闸已在本地通过）。

**没做（honest）**：p=7/p=8 的**实测**在本单窗口内未取得。计划中的高 p 运行
（`python -m benchmarks.p_refinement_floor --degrees 5 6 7 8 --temporal-refinements 3 --step-budget 50000`，
约 7.6 万 RK 步）递交集群作业 **cj1635**（TensorDG，stage_dir=本工作树）后仍在队列中未派发；
本地试跑因控制面单次命令 120 s 上限被收割。故 `measurement_floor.json` 仍是 p=[5, 6] 的旧工件，
新增的 p>6 内样本外推检验**尚无数据**，其工件闸按设计 `skip`（不是 pass）。
p=7/8 的现状仍以上游 m17067 的精确时间结论为准：p=7 可解析（2.716e-13），
p=8 只可作上界（算子表示地板 ~4.75e-13 大于信号）。

## ① 案例介绍
一维周期线性对流 `u_t+u_x=0`，初值 `sin(2πx)`（整函数），`T=1`（一整周期，精确解=初值平移），
固定网格 `ncells=16` 只做 p 加密。理论基准 Babuska & Suri (1994) SIAM Review 36:578–632：
解析解下 DG/FEM 误差随 p 指数衰减 `‖u-u_h‖ <= C exp(-σ p)`。
已发布阶梯 p=1..6 的 σ = **3.9271**，float64 测量地板 = **1.570e-16**（`eps64·‖u_h‖`），
可测裕度 = 10×地板 = 1.570e-15。

## ② 怎么做的（本单新增协议）
1. 对每个 p：`cfl` 起点取已发布工件的 CFL（无发布值时用 0.08）；
2. 阶梯细化：每档 `cfl/2` 重解（出货 RK4 + 出货 `solve_advection`，不改格式），
   用 longdouble 求积路径测 L2，比较相邻两档误差；
3. 停止：`|e_prev - e_cur|/e_cur <= max(0.01, 10·eps64·‖u_h‖/e_cur)` ⇒ 认证；
   预算命中 ⇒ 未认证（带 `stop_reason` 与 `total_steps` 入工件）；
4. 外样本：`measured/predicted` 逐点入工件，带宽 ±2×；
5. 门禁：`tests/test_p_refinement_floor_high_degree.py`（工件一致性 + 活闸；
   活闸在 p=2 与 p=4/CFL=2 上直接驱动新函数，验证「认证」与「预算耗尽如实报告」两条路径）。

## ③ 计算结果（当前工件，p=[5, 6]）
| p | L2 (longdouble) | 相对离散 | SNR(地板) | 可测 |
|---|---|---|---|---|
| 5 | 1.5927e-09 | 5.5e-09 | 1.01e+07 | 是 |
| 6 | 2.2283e-11 | 8.4e-07 | 1.42e+05 | 是 |

已发布阶梯（主工件）误差：p=1: 7.464e-03, p=2: 2.089e-04, p=3: 5.042e-06, p=4: 9.732e-08, p=5: 1.593e-09, p=6: 2.228e-11。
按 σ=3.927 外推：p=7 → 4.390e-13（SNR 2796）、
p=8 → 8.648e-15（SNR 55）——**外推值，不是实测**。

## ④ 结果图
`measurement_floor_protocol.png`：已发布 p=1..6 阶梯、地板研究重测点（p=5,6）、σ 外推（虚线段，
明确标注 unmeasured）与 float64 地板/可测裕度两条水平线。图中外推段**只作预测**，不冒充实测。

## ⑤ 注意点 / 局限
- **p>6 的实测为本单欠账**（集群 cj1635 未派发、本地受 120 s 命令上限）：本节结论不覆盖 p=7/8 的实测事实。
- 相对时间收敛判据在高 p 会退化：误差趋近 `eps64·‖u_h‖` 时「1% 相对变化」等价于要求噪声自洽，
  故认证阈值改为 `max(1%, 10×roundoff 不确定度)`；这**不改**出厂 `solve_advection` 的时间判据，
  只作用于本验证脚本。
- 步数预算 / 细化上限是显式协议参数：命中预算的档位标未认证，不得引用为空间误差。
- 适用面：一维周期线性对流光滑解；二维/变系数/激波不在范围。

## ⑥ 误差判定（逐项，回答「是否<3%」）
| 项 | 基准 | 结果 | 是否<3% |
|---|---|---|---|
| 新协议活闸（认证路径） | p=2 已知收敛档 | 通过（见 pytest） | ✅ |
| 新协议活闸（预算耗尽如实报告） | max_refinements=0 | `temporal_converged=False` | ✅（行为正确） |
| 已发布阶梯复现 | 主工件 | 本单未重跑主工件 | ⚪ 未测（无新基准） |
| p=7/8 实测 vs σ 外推 | ±2× 带宽 | **无数据**（作业未派发） | ⚪ 未测 |

**结论**：本单交付的是**测量协议的认证化**（可验证、有活闸、未认证即拒引），
不是 p=7/8 的新实测；锚点的「p>6 实测」仍为 WITHHELD，需在集群空出后跑 cj1635 或等价命令。


## 7. 补记（同一工作窗口内，p=7 实测已落地）

递交的集群作业仍在排队，但同一协议在控制面本地以小预算（p=6,7，细化上限 1 次，
`runtime 169 s`）跑完后工件已入库 `measurement_floor.json`（levels = [6, 7]）：

| p | cfl 起点→认证点 | 解算次数 | 时间相对变化 | 认证 | 停止原因 | L2 (longdouble) | SNR | 可测 |
|---|---|---|---|---|---|---|---|---|
| 6 | 0.04 → 0.02 | 2 | 5.92e-06 | 是 | converged | 2.2282e-11 | 1.42e+05 | 是 |
| 7 | 0.08 → 0.04 | 2 | 1.77e+00 | **否** | refinement_budget_exhausted | 2.7505e-13 | 1.75e+03 | 是 |

**外样本检验（p=6 已发布点 + σ 外推）**：实测 2.7505e-13 vs 外推 4.3898e-13
⇒ 比值 **0.627**（±2× 带宽内，True）。

**与上游 m17067 精确时间积分的独立交叉**：p=7 精确时间值 2.7158e-13，本单出厂 RK4（cfl 0.04，
未认证档位）2.7505e-13 ⇒ 相对差 **1.28%**，
两条独立路径（矩阵指数 vs RK4+认证协议）在 3% 内一致；p=7 档位仍未通过 1% 时间认证（`refinement_budget_exhausted`，
需 cfl≈0.01 才收敛，见 m17067 的 CFL 扫描），故引用时应写「RK4/cfl 0.04 实测 + 时间误差未认证」。
p=8 仍未实测（集群作业未派发），维持 m17067 的上界结论。
