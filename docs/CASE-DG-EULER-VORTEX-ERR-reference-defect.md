# CASE-DG-EULER-VORTEX-ERR：长时"误差放大"实为参考解缺陷（m20695）

- **锚点**：`CASE-DG-EULER-VORTEX-ERR`（总账 status=open，本单做实质增量，不自行改 status）
- **承接线索**：`docs/AUDIT-m20486-CASE-DG-EULER-VORTEX-ERR.md` §5 自留线索
  「t=2→t=5 的 58 倍跳变……强烈提示 DG 长时积分的放大机制，已列入下次自找活线索，本单不下结论」
- **结论（本单）**：放大机制**不是** DG 格式的，而是**度量**的 —— 精确解按自由空间
  （无周期折返）平移评估，而两个求解器都以 `periodic=(True,True)` 封闭面。

## 1. 缺陷（defect）与机制

`tensordg/euler_2d.py::vortex_primitive` 只做 `x_shift = x - u_inf·t`。周期箱
`[-5,5]^2`、`u_inf=(1,1)` 下，t=5 时涡心已到箱角 (5,5)：自由空间参考在对角箱角
(-5,-5) 处几乎未受扰动（半径 14 → ~e^-98），而**周期解在那里必须有一个 1/4 涡心**。
于是"参考"不再是同一个数学对象，测出的"误差"是参考之间的距离。

**量化（纯解析扫掠，不跑求解器，`ec.reference_defect_sweep`）**：

| t | rho % | u % | p % |
|---|---|---|---|
| 1 | 0.000 | 0.009 | 0.000 |
| 2 | 0.000 | 0.263 | 0.001 |
| 3 | 0.068 | 2.688 | 0.096 |
| 4 | 1.684 | 9.701 | 2.305 |
| 5 | 6.175 | 14.175 | 8.022 |
| 7.5 | 6.866 | 16.150 | 8.930 |
| 10 | 6.866 | 16.147 | 8.930 |

旧发布读数（DG-P2/32²，自由空间参考）：t=5 **14.4302%**、t=10 **16.4433%**；
上表对应参考缺陷 14.175%/16.147% —— 两者在 1.5% 相对差内重合，即**旧读数的绝大部分
可被参考缺陷完全解释**。t=1 的参考缺陷仅 0.009%（u），故 t=1 的 0.0127% 是真实离散误差。

## 2. 修复

| 文件 | 改动 |
|---|---|
| `tensordg/euler_2d.py` | `vortex_primitive(..., period=None)`：折返位移到 `[-L/2,L/2)`（min-image 周期解）；`vortex_error(..., period=None)` 透传。默认 `None` = 旧语义，全部既有调用点不受影响 |
| `benchmarks/verify_euler_dg_2d.py` | `_vortex_error(..., period=None)` 透传 |
| `benchmarks/case_euler_vortex_errconserve.py` | 新增 `PERIOD` 常量，两个度量路径改用周期参考；设计族阶梯扩到 t=1/2/5/10；`metrics.json` 新增 `reference_model` 与 `reference_defect_percent` 程序化块 |
| `tests/test_case_euler_vortex_errconserve.py` | +5 项不变量测试（见 §3） |

截断：折返只保留最近镜像；其余镜像在半径 ≥5 处贡献 ~6e-6 振幅（`reference_model.truncated_images` 已写明）。

## 3. 验证（本单实跑）

`PYTHONPATH=. python3 -m pytest tests/test_case_euler_vortex_errconserve.py -k "periodic or reference or inside or dg_metric or p0_metric"` → **6 passed**

新增 5 项（全部为不变量/反空洞断言，不是"跑通即算过"）：

1. `test_periodic_reference_is_invariant_under_a_box_translation`：φ(x+10)≡φ(x)（1e-12）。
2. `test_free_space_reference_still_matches_while_the_centre_is_inside`：t=0 **逐位相同**、
   t=1 <1e-3、t=2 <5e-3 相对 L2 —— 修复不是重新调参。
3. `test_free_space_reference_is_the_wrong_solution_once_the_vortex_wraps`：t=5 缺陷 >6% 容差、
   t=10 >15%，且速度分量 > 密度分量（符号位移 vs 纯径向）。
4. `test_p0_metric_reports_zero_for_the_exact_periodic_state`：把**精确周期解**喂进度量，
   密度误差 0.0，速度/压力 3.6e-4（`avg(rho u)/avg(rho) != avg(u)` 的 O(h²) 残差）；
   同一状态对自由空间参考报 13.9% —— 判别力 400 倍。
5. `test_dg_metric_reports_projection_error_only_for_the_exact_periodic_state`：DG 模态路径同理
   （投影误差 <1e-2 vs 自由空间 13.9%）。

**实测前后对比**（同一状态，`p0 n=64 t=5`，本单本地跑）：

| 分量 | 自由空间参考（旧） | 周期参考（新） |
|---|---|---|
| rho | 3.334% | 5.726% |
| u | 7.763% | 12.024% |
| p | 4.426% | 8.476% |

注意方向：P0 一阶格式在 t=5 已把涡核抹平（ρ_min 0.93 vs 精确 0.494），数值解在**四个**
箱角都有 1/4 核；旧参考只在**一个**箱角有核，因而**系统性低估**了真实误差（"半个对的
参考"比"全对的参考"更容易得分）。即旧度量在 P0 族上低报、在 DG 族上高报 —— 两种方向的
偏差同源，都来自非周期参考。

## 4. 未完成 / 待办（honest，不粉饰）

- **阶梯重跑未完成**：全部 12 个检查点需用周期参考重新求解（结果 json 内含度量值，
  不能只重算度量）。已递交 `run_compute` 作业 **cj2908**（6 分片、6 核、stage_dir=本工作树），
  两次递交（cj2872 排队 26 min 未出结果后取消、cj2908）均因集群拥塞**尚未产出**。
  本机重计算被 `LOCAL-COMPUTE BLOCKED` 硬拦，未绕过。
- 因此 `case_euler_vortex_errconserve/metrics.json` **仍是修复前的旧证据包**（自由空间参考），
  不得引用为修复后结论 —— 见同目录 `PENDING_REFERENCE_RERUN.md`。
- **修复后 C2（alpha<=1）判定未定**：旧 alpha=3.3~4.4 已确认被参考缺陷污染，
  但真实 DG 长时误差尚未测得，故本单**不声称** C2 通过。
- Yee et al. (2000) 原文不在库内，6% 容差仍按算例池条目 verbatim 采用 → **WITHHELD**。
- 未做独立第三方环境复现（同一 checkout）。

## 5. 给下次自找活的线索

1. 取回 cj2908（或重跑）→ `--aggregate --figures` → 用 `publish_case_dg_vortex_reference_defect.py`
   重发报告并重判 C1/C2/C3/C4/C5；若 DG 族 alpha 仍 >1，则**这**才是真的格式级问题
   （下一步：固定网格 CFL 细化 0.2→0.05 分离时间/空间误差）。
2. 同一缺陷在**其他**长时周期算例里同样存在：`benchmarks/isentropic_vortex_2d.py`
   （`--time` 可设 >2）、`convergence_euler.py`、`limiter_2d_order_study.py`、
   `euler_dg_2d_time_split.py` 都调用不带 `period` 的度量 —— 只要 t 使涡心出箱即失真。
   建议逐个审计（`grep -n "vortex_error(" benchmarks/`）。
3. `_vortex_error` 的 `period` 参数目前只有本算例使用；建议给 `case_euler_vortex.py`
   增加 t>4 的长时检查点以覆盖该路径。
