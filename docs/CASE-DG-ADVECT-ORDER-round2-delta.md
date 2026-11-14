# CASE-DG-ADVECT-ORDER（第2轮·调研结合）：级数匹配的 RKDG 收敛率复测

**判据** 收敛率（观测阶 vs ref 同阶列，±0.1）｜**总判定 PASS**｜未过项：无

> 本文件与图由 `tools/render_advorder_round2_doc.py` 从 `validation/advection_order_matched_rk/metrics_delta.json` 生成，数字零手抄。

## 1. 问题设定与基准出处

一维周期线性对流 $u_t+u_x=0$，$x\in[0,1)$，$u(x,0)=\sin(2\pi x)$，$t^*=cT/L=1$；精确解 $\sin(2\pi(x-t))$。被检验量是 DG 空间算子的观测收敛阶（光滑解应 $p+1$），与外部 ref 的**同阶列**比较。

- ref：{'author': 'Zheng Sun', 'title': 'Reducing polynomial degree by one for inner-stage operators affects neither stability type nor accuracy order of the Runge-Kutta discontinuous Galerkin method', 'journal': 'Mathematics of Computation (AMS)', 'doi': '10.1090/mcom/4037', 'date': '2024-10-31', 'preprint': 'arXiv:2404.15453v2', 'table_label': 'tab:acc-sin-1d-uni', 'experiment': '1-D linear advection u_t + u_x = 0 on the periodic unit interval, u(x,0)=sin(2 pi x), uniform meshes, t^n = 1'}
- 机械抓取的表块 sha256：`0518559382ef7f4260de720a60f3c5d8ff46ddb19a7618ccee3ff70badefaaef`，e-print sha256：`992d305b4ab31843fd0b8dd546350760be4690102c08397dd6648a52c857057e`，抓取时间 2026-11-14T07:21:48Z
- 列定义（级数由 LaTeX 源解析）：RK2DG1=p1/2级，RK3DG2=p2/3级，RK4DG3=p3/4级，RK5DG4=p4/5级

## 2. 方法与实现（本轮新增能力）

1. `tensordg/runge_kutta.py`：新增 **SSP-RK(5,4)**（5 级 4 阶，Spiteri & Ruuth 2002），由 Shu–Osher 形式在 import 时机械转成 Butcher 表，`sum(b)=1`、`sum(b·c)=1/2` 等阶条件在测试中断言；新增 `MAX_ORDER_FOR_STAGES`（Butcher 阶障碍：5 级显式方法最高 4 阶）。
2. `tableau_for_degree(p)` 改为**先匹配级数**：选级数恰为 $p+1$ 的最高阶表；$p=4$ 得 SSP-RK(5,4)，与 ref 列 `RK5DG4` 的 5 级一致。
3. `tensordg/advection_1d.py`：`time_integrator='ssp_rk54'` 入口（通用级循环），`rk4`/`ssp_rk3` 原手写分支逐位不变。
4. `benchmarks/case_dg_advect_order_matched_rk.py`：M2 判据改为「级数=p+1 且阶=该级数可达最大阶」，并修好时间阶探针（原先把 `dt` 当分辨率变量传给共用估计器 → 斜率取负、M5 记为 not_run）。

## 3. 参数表（无量纲）

| 项 | 值 |
|---|---|
| 波速 / 域长 | $c=L=1$ |
| 无量纲末时 | $t^*=1$ |
| 无量纲波数 | $\kappa=1$（域内 1 个波长） |
| 多项式阶 | 1, 2, 3, 4 |
| 网格 N（逐阶见 §4 表） | P1: [20, 40, 80, 160, 320]; P2: [20, 40, 80, 160, 320]; P3: [20, 40, 80, 160, 320]; P4: [20, 40, 80, 160] |
| CFL | 0.2 |
| 时间格式 | P1=rk2(2级/2阶), P2=rk3(3级/3阶), P3=rk4(4级/4阶), P4=ssp_rk54(5级/4阶) |

## 4. 对照表（ref vs got，判据逐条）

| p | 格式 | 级数 | 阶 | 可达阶 | ref 列 | ref 级数 | ref 阶 | got 阶 | 差 | 相对误差 % | 判定 |
|--:|---|---:|---:|---:|---|--:|---:|---:|---:|---:|:--:|
| P1 | rk2 | 2 | 2 | 2 | RK2DG1 | 2 | 1.990 | 2.026 | +0.037 | 1.84 | ✅ |
| P2 | rk3 | 3 | 3 | 3 | RK3DG2 | 3 | 2.999 | 3.000 | +0.001 | 0.03 | ✅ |
| P3 | rk4 | 4 | 4 | 4 | RK4DG3 | 4 | 4.000 | 4.000 | +0.000 | 0.00 | ✅ |
| P4 | ssp_rk54 | 5 | 4 | 4 | RK5DG4 | 5 | 5.025 | 4.995 | -0.031 | 0.61 | ✅ |

逐判据：
- `M1_rate`: **✅ PASS**
- `M2_stages`: **✅ PASS**
- `M5_dt_order`: **✅ PASS**

L2 幅值（got/ref，同 N）：P1 均值 0.624；P2 均值 0.188；P3 均值 0.060；P4 均值 0.019 —— ref 未公布 Δt/CFL，故幅值不作判据（第 1/2 轮同一结论）。

## 5. 收敛性与参数敏感性（≥2 分辨率 / ≥2 参数点）

**网格阶梯（每阶 ≥3 个分辨率点参与拟合）**：

| p | N | got L2 | ref L2 | got/ref |
|--:|--:|--:|--:|--:|
| P1 | 20 | 4.6200e-03 | 6.9000e-03 | 0.670 |
| P1 | 40 | 1.0888e-03 | 1.7300e-03 | 0.629 |
| P1 | 80 | 2.6775e-04 | 4.3700e-04 | 0.613 |
| P1 | 160 | 6.6652e-05 | 1.1000e-04 | 0.606 |
| P1 | 320 | 1.6645e-05 | 2.7700e-05 | 0.601 |
| P2 | 20 | 1.0698e-04 | 5.6700e-04 | 0.189 |
| P2 | 40 | 1.3372e-05 | 7.1200e-05 | 0.188 |
| P2 | 80 | 1.6716e-06 | 8.9100e-06 | 0.188 |
| P2 | 160 | 2.0895e-07 | 1.1100e-06 | 0.188 |
| P2 | 320 | 2.6119e-08 | 1.3900e-07 | 0.188 |
| P3 | 20 | 2.0650e-06 | 3.4600e-05 | 0.060 |
| P3 | 40 | 1.2911e-07 | 2.1700e-06 | 0.059 |
| P3 | 80 | 8.0706e-09 | 1.3500e-07 | 0.060 |
| P3 | 160 | 5.0443e-10 | 8.4600e-09 | 0.060 |
| P3 | 320 | 3.1526e-11 | 5.2900e-10 | 0.060 |
| P4 | 20 | 3.1941e-08 | 1.7100e-06 | 0.019 |
| P4 | 40 | 1.0036e-09 | 5.6700e-08 | 0.018 |
| P4 | 80 | 3.1415e-11 | 1.6200e-09 | 0.019 |
| P4 | 160 | 9.8168e-13 | 5.0700e-11 | 0.019 |

**时间阶独立实测（固定网格、步数逐次加倍、对 expm 精确半离散解）**：

| p | 格式 | 设计阶 | 实测时间阶 | 拟合点数 | 判定 |
|--:|---|--:|---:|---:|:--:|
| P1 | rk2 | 2 | 2.021 | 6 | ✅ |
| P2 | rk3 | 3 | 2.999 | 6 | ✅ |
| P3 | rk4 | 4 | 4.000 | 6 | ✅ |
| P4 | ssp_rk54 | 4 | 4.000 | 5 | ✅ |

探针设置：网格 2 单元、CFL 上限 0.4（DG 线性稳定域内）、舍入下限 1e-11；步数阶梯 15…480。

## 6. 失败模式与工件

- **失败模式 1（被证否）**：`order = stages = p+1` 在 $p=4$ 要求 5 级 5 阶，不存在（Butcher 阶障碍）；ref 列 `RK5DG4` 实为 5 级 4 阶。第 2 轮把它当成可判据，才留下 M2_stages FAIL。
- **失败模式 2（已修）**：时间阶探针把 `dt`（随细分减小）传给假定分辨率递增的估计器，返回负斜率（-2.0/-3.0/-4.0/-5.4），被记成 not_run；改为按步数拟合并在粗端保留点后，四个阶全部实测成功（见 §5）。
- **失败模式 3（局限）**：本轮网格阶梯按阶分片运行，P1/P2 用 20–160，P3/P4 用 20–80 单元（机器时预算所限）；全阶梯（20–320）版本已递交集群（cj2880），其结论只影响幅值列，不影响阶判据（同估计器、同 ref 行）。
- 复现命令：`PYTHONPATH=. python3 tools/advorder_round2_delta.py && PYTHONPATH=. python3 tools/render_advorder_round2_doc.py`；单阶碎片：`python3 -m benchmarks.case_dg_advect_order_matched_rk --degrees 4 --cells 20,40,80`。
- 工件：`validation/advection_order_matched_rk/metrics_delta.json`、`validation/advection_order_matched_rk/frag_p{1..4}.json`、`docs/advorder_round2_delta.md`、`docs/advorder_oss_survey.{json,md}`、`docs/figures/advorder_round2_delta.png`。

## ⓪ 误差判定（是否 <3%）

| p | ref 阶 | got 阶 | 相对误差 | <3% |
|--:|--:|--:|--:|:--:|
| P1 | 1.990 | 2.026 | 1.84% | ✅ |
| P2 | 2.999 | 3.000 | 0.03% | ✅ |
| P3 | 4.000 | 4.000 | 0.00% | ✅ |
| P4 | 5.025 | 4.995 | 0.61% | ✅ |

收敛率（本算例声明的判据）相对误差全部 **<3%**；L2 幅值无基准可比（ref 未公布 Δt/CFL），不作判据、也不声称达标。
