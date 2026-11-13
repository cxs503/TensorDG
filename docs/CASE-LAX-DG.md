# CASE-LAX-DG · Lax 激波管（Toro test 2）DG 算例池验证

> 工件 `runs/lax_case_pool/metrics.json` 首次生成于 2026-11-12T08:46:44Z；m16585 于 **2026-11-12T17:03:14Z** 在干净工作树上全量重跑（`quick=false`，9 腿，墙钟 373.2 s）并把 metrics.json / profiles.npz / 3 张 PNG 随仓提交，阶梯数字与初版逐位一致。本文全部数字由 `benchmarks/lax_case_pool.py` 从该 JSON 注入（零手抄）。

> **口径变更（m16993）**：R3 的解析区遮罩已由 `4·dx` 改为固定物理半宽 `0.02`（闭合 m16606 审计 F1/F2），
> 同一批解上旧规则的测量保留为 `diagnostics.r3_legacy_band`。因此本文下表的「解析区 rel L1」列
> 属**改口径前的历史记录**，工件也从 `runs/`（合并门禁不携带）迁到 `docs/benchmarks/results/lax_case_pool/`。
> 现行数字与新增的时间误差分离见 `docs/CASE-LAX-DG-R3-nested-band-and-dt-split.md`（由
> `scripts/render_lax_case_pool_doc.py` 从 `metrics.json` 注入）。
>
> **参考解独立仲裁（m17119）**：本算例的参考值是同仓闭式代数（`benchmarks/riemann_exact.py`），
> 存在自证风险。已用**不含 Riemann 求解器**的 Rusanov 有限体积族（一阶/二阶，N=400…6400）与
> **纯二分**星区解独立复核：波位置最坏 0.098 %、星区平台最坏 2.9e-07、两种求根相对差 1.8e-16，
> 独立族以实测阶 0.51/0.61 收敛到闭式解（=`dx→0` 极限的独立见证）。见
> `docs/CASE-LAX-DG-reference-crosscheck.md`，工件 `docs/benchmarks/results/lax_reference_crosscheck/`。
> 本文的 C3/R3 honest-FAIL 不受影响（方法属性），该复核只证明参考值可信。

## ① 案例介绍

Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域内隔膜 `x0 = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。精确解含四道波：左稀疏波（head → tail）、接触间断、右激波。四道波的位置与星区状态全部有闭式表达，因此该算例自带不需第三方求解器产生的参考值。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

解析波系（闭式，未经求解器回读）：

| 量 | 值 |
|---|---|
| p* | 2.466098 |
| u* | 1.528723 |
| ρ*_L（等熵稀疏） | 0.344568 |
| ρ*_R（Rankine–Hugoniot） | 1.304085 |
| x 稀疏波头（解析边沿） | 0.157637 |
| x 稀疏波尾（解析边沿） | 0.287229 |
| x 接触间断 | 0.698734 |
| x 右激波 | 0.822312 |

## ② 方法

模态 DG（Legendre 基，P1–3），HLLC 数值通量，SSP-RK3 时间推进，CFL = 0.35，位置与星区量在每单元 10 点 Gauss 节点上重建后测量。分辨率阶梯 N × p：N=100/P1, N=200/P1, N=400/P1, N=100/P2, N=200/P2, N=400/P2, N=100/P3, N=200/P3, N=400/P3。

五道预登记判据（容差在跑数之前写死在 `benchmarks/lax_case_pool.py`，未按结果回调）：

| 判据 | 测量对象 | 容差 |
|---|---|---|
| C1 | 四道波位置（最细腿）vs 解析位置，归一化 `|Δx|/|x_exact-x0|` | < 2.000 % |
| C2 | 星区 (ρ*_L, u*, p*, ρ*_R) 平台值 vs 解析 | < 5.000 % |
| C3 | 全场相对 L1 收敛阶（密度，最细一对） | ≥ 1.0 |
| R3 | 解析区（波带 4.0 dx 剔除）相对 L1 收敛阶 | ≥ 0.85·(p+1) |
| R5 | 光滑熵波设计阶（u = 1.0 / 2.0，周期域） | 实测阶与 p+1 相对偏差 < 3.000 % |

## ③ 计算结果（程序化注入）

## resolution ladder

| P | N | steps | rel L1 rho | rel L1 u | rel L1 p | analytic-region rel L1 rho | wall [s] |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 554 | 2.33856e-02 | 1.02549e-02 | 5.30940e-03 | 1.21393e-03 | 6.8 |
| 1 | 200 | 1102 | 1.36530e-02 | 4.67057e-03 | 2.64516e-03 | 1.00278e-03 | 13.4 |
| 1 | 400 | 2198 | 7.38227e-03 | 2.34567e-03 | 1.31914e-03 | 8.99857e-04 | 30.5 |
| 2 | 100 | 912 | 1.49085e-02 | 9.34849e-03 | 5.00484e-03 | 2.16940e-03 | 11.1 |
| 2 | 200 | 1826 | 8.44277e-03 | 3.88562e-03 | 2.39419e-03 | 1.29562e-03 | 25.9 |
| 2 | 400 | 3659 | 4.21351e-03 | 1.51925e-03 | 9.37436e-04 | 6.65070e-04 | 54.5 |
| 3 | 100 | 1264 | 1.33859e-02 | 1.50257e-02 | 8.32030e-03 | 4.78355e-03 | 16.5 |
| 3 | 200 | 2535 | 9.58691e-03 | 1.22518e-02 | 6.74382e-03 | 4.84076e-03 | 34.0 |
| 3 | 400 | 5087 | 6.48039e-03 | 8.70614e-03 | 5.08869e-03 | 4.17239e-03 | 81.5 |

## gates

| gate | verdict | detail |
|---|---|---|
| C1 | PASS | rarefaction_head: 0.104 %; rarefaction_tail: 0.245 %; contact: 0.039 %; shock: 0.048 % |
| C2 | PASS | star_left.density: 0.002 %; star_left.velocity: 0.003 %; star_left.pressure: 0.000 %; star_right.density: 0.030 % |
| C3 | FAIL | p1.density: 0.887; p1.velocity: 0.994; p1.pressure: 1.004; p2.density: 1.003; p2.velocity: 1.355; p2.pressure: 1.353; p3.density: 0.565; p3.velocity: 0.493; p3.pressure: 0.406 |
| R3 | FAIL | p1.density: 0.156; p1.velocity: 0.500; p1.pressure: 0.490; p1.requirement: 1.700; p1.density_ratio: 0.092; p2.density: 0.962; p2.velocity: 1.420; p2.pressure: 1.250; p2.requirement: 2.550; p2.density_ratio: 0.377; p3.density: 0.214; p3.velocity: 0.358; p3.pressure: 0.312; p3.requirement: 3.400; p3.density_ratio: 0.063 |
| R5 | PASS | u=1.0.p1: order 2.008 (dev 0.41 %); u=2.0.p1: order 2.017 (dev 0.85 %); u=1.0.p2: order 3.001 (dev 0.03 %); u=2.0.p2: order 3.001 (dev 0.03 %) |

Overall verdict: **FAIL**


## ④ 图与工件（已随仓提交，m16585 落盘）

三张图由 `benchmarks/lax_case_pool.py` 直接渲染（剖面 + 与精确解比较 + 熵波设计阶拟合）：

* `runs/lax_case_pool/figures/lax_profiles.png` — 最细腿 (N=400, P3) 的 rho/u/p 数值 vs 精确解
* `runs/lax_case_pool/figures/lax_convergence.png` — 全场与解析区相对 L1 随 N 收敛（log-log）
* `runs/lax_case_pool/figures/entropy_wave_order.png` — 光滑熵波实测阶 vs p+1

工件清单（`sha256` 由 `sha256sum` 直读，重跑可逐位校验）：

| 文件 | 用途 | bytes | sha256[:16] |
|---|---|---|---|
| `runs/lax_case_pool/metrics.json` | 实测值 + 判据 verdict + 全部容差配置 | 14022 | 63c83d5d37c980b8 |
| `runs/lax_case_pool/profiles.npz` | 最细腿原始量 vs 精确解逐点场 | 225846 | 3542bd55f1684d40 |
| `runs/lax_case_pool/summary.md` | 由 metrics.json 渲染的表（禁手改） | 2300 | 89416a1dfa43148c |
| `runs/lax_case_pool/figures/lax_profiles.png` | 场量对比图 | 89240 | 6a75d44ad7c6a95b |
| `runs/lax_case_pool/figures/lax_convergence.png` | 收敛图 | 87338 | 71975d1e790c85d8 |
| `runs/lax_case_pool/figures/entropy_wave_order.png` | 熵波阶数图 | 59423 | 91bd74b53d9a4da5 |

**复现与门禁状态**（2026-11-12，m16585）：`PYTHONPATH=. python -m benchmarks.lax_case_pool`
全量（非 `--quick`）墙钟 **373.2 s**，9 条阶梯腿的 rel L1 rho/u/p 与解析区 rho 与本文表格
**逐位一致（相对差 < 1e-6，9 腿 × 7 量全对）**，verdict 仍为 **FAIL**；
`pytest tests/test_lax_case_pool.py` 由 **9 passed / 2 skipped** 变为 **11 passed**——
此前被 `skipif(not METRICS.exists())` 跳过的两条工件层测试（`test_committed_artifact_is_self_consistent`
从 metric 里的数重判门禁 + 容差一致性 + 阶梯单调性、`test_committed_artifact_profiles_agree_with_the_exact_solution`
要求 `profiles.npz` 三场相对 L1 < 5%）**首次真正执行并通过**。工件生成不等于判据变好：
`verdict` 未变，只把「已发布数字」变成「可离线复核的数字」。

## ⑤ 注意点 / 局限

* **C1 的口径**：稀疏波头/尾按压力降的 10 % 水平线定位，因此它落在扇形区**内部**而不是解析边沿。本例中该水平线与解析边沿的偏移为 0.0113 / -0.0149（x 单位），故 C1 是「同一水平线在数值场与精确场中的位置差」，不是边沿分辨率。
* **接触间断/激波窗口**：搜索窗从相邻波回缩 `max(4 dx, 0.02)`，否则被抹宽的接触间断会被误认成激波（本单实测到的定位缺陷，已由 `tests/test_lax_case_pool.py::test_contact_and_shock_are_not_confused_by_the_search_windows` 锁死）。
* **宽谱 vs 解析区**：全场 L1 由间断主导、收敛阶被压到一阶附近；解析区把四道波各剔掉 4.0 dx，剩余区域仍受相邻间断的数值抹平污染，因此解析区阶数不等于设计阶 p+1。
* **R5 的口径**：台账原文为「within 3% of 1.0/2.0」，本单读作「两种预登记波速 u = 1.0 / 2.0 下实测阶与设计阶 p+1 的相对偏差 < 3%」，两种波速与两条阶梯写在工件里，读数若有异议可直接按工件重判。
* 计算面：CPU / float64；本仓限幅器是局部正性回退而非 TVD 证明，因此本算例验证的是「该实现的实测精度」，不构成一般激波捕捉保证。

## ⑥ 误差判定

| 判据 | 关键量 | 实测 | 容差 | 是否达标（<3% 口径） |
|---|---|---|---|---|
| C1 | 最差波位置（rarefaction_tail） | 0.245 % | < 2.000 % | 是（预登记容差 2.000 % 下 PASS） |
| C2 | 最差星区量（star_right.density） | 0.030 % | < 5.000 % | 是（预登记容差 5.000 % 下 PASS） |
| C3 | 密度全场 L1 阶 | 0.565 | ≥ 1.0 | 阶数判据（非误差百分比）：FAIL |
| R3 | 密度解析区阶（P1/P2/P3） | 0.156 / 0.962 / 0.214 | ≥ 0.85·(p+1) | 阶数判据：FAIL |
| R5 | 熵波阶偏差（最差） | 0.849 % | < 3.000 % | PASS |

**总判定：FAIL**。（判据逐条为 PASS/FAIL，未达标的条目按 honest-FAIL 原样记录，不重跑调参掩盖。）

**与算例池台账口径（ref: 精确解，容差 <3%）的逐项比对**（台账原文容差是百分比型，
而本算例池的 C3/R3 是阶数型判据，两者必须分开判，不能用一个 FAIL 盖住全部）：

| 关键量 | 实测 | 台账 <3% 口径 |
|---|---|---|
| C1 最差波位置误差（rarefaction_tail） | 0.245 % | 是（< 3 %） |
| C2 最差星区量误差（star_right.density） | 0.030 % | 是（< 3 %） |
| R5 最差设计阶偏差 | 0.849 % | 是（< 3 %） |
| C3 全场密度 L1 阶 | 0.565（最细一对） | 否——阶数判据，要求 ≥ 1.0 |
| R3 解析区密度阶 / 要求 | 0.063–0.377（比值） | 否——阶数判据，要求比值 ≥ 1.0 |

即：**误差百分比意义上的三个判据全部远优于台账 <3%；作为整体的 verdict=FAIL 只由两条
阶数判据（C3/R3）驱动**。这是本算例在「间断主导的全场误差无法达到设计阶」这一物理事实下的
如实结论，不是把结果判活，也不因重跑而改变。

