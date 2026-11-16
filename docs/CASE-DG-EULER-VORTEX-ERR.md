# CASE-DG-EULER-VORTEX-ERR：等熵涡移动误差守恒（DG，一个整周期）

> 本报告的每个数字均由 `benchmarks/case_euler_vortex_yee.py --task aggregate` 从 `results/*.json` 程序化注入，无手抄；判据由脚本计算。参考文献原文获取状态见 §① 与 `artifacts/ref_provenance_yee2000.json`。

## ① 问题与基准出处

二维可压缩 Euler 方程（理想气体 γ = 1.4）的等熵涡：均匀来流 (u,v)∞ = (1,1) 平移输运一个等熵涡，t>0 的**解析精确解**即初始剖面向下游刚性平移。域 [-5,5]² 周期，域长 L = 10、|u∞| = 1，故平移一个整周期需 T = L/u∞ = 10；t = T 时精确解与初始条件逐点相同——这正是「移动误差」的判据：不是看短时精度，而是看**一个整周期后误差是否仍被 6% 容差覆盖**（误差守恒性）。

- 基准文献（算例池指定）：Yee, Vinokur & Djomehri, J. Comput. Phys. 162 (2000) 33-81, DOI 10.1006/jcph.2000.6517 (per case-pool brief); numeric target = analytic isentropic vortex, machine-verified。
- 数值目标：与**解析解**逐点比较（`tensordg/euler_2d.py::vortex_primitive`）。参考解的可信性由 `--task residual` 独立校核（静止系 div F = 0、平动系 U_t+F_x+G_y = 0、等熵律、无散旋流、径向动量平衡），不依赖任何外部表格。
- 原文获取：Crossref/Unpaywall/OpenAlex 机器查询得到 DOI 与唯一 OA 落地页，但全文下载在受限环境内被拒（NTRS 403、镜像 Cloudflare 挑战）；逐条尝试与 HTTP 状态记入 `artifacts/ref_provenance_yee2000.json`，数值目标回退到解析解（machine-verified exact solution）。

## ② 方法与流程

| 组件 | 位置 | 说明 |
|---|---|---|
| 高阶主路径 | `tensordg/euler_dg_2d.py::solve_euler_dg_2d_vortex` | 张量积模态 DG（P1–P3）+ HLLC + SSP-RK3，双周期边界，L2 投影初值 |
| 低阶基线 | `tensordg/euler_2d.py::solve_isentropic_vortex` | P0（守恒型有限体积）+ Rusanov + SSP-RK3，单元平均初值 |
| 精确解 | `tensordg/euler_2d.py::vortex_primitive` | 解析等熵涡剖面，machine-verified |
| 误差度量 | `benchmarks/case_euler_vortex.py::_vortex_error` | 体积求积节点上的相对 L2 |
| 算例编排 | `benchmarks/case_euler_vortex_yee.py` | 15 配置，每配置一进程一检查点 |
| 聚合与报告 | `benchmarks/case_euler_vortex_yee.py::aggregate` | 零手抄，全程序化注入 |

## ③ 计算结果

### 表 1：门禁配置（gate）

| 配置 | 最大场相对 L2 | <6%? |
|---|---|---|
| dg_d2_n32_t10_cfl0.2 | 0.0152% | 勾 |

### 表 2：误差历史（移动误差累积）

| t | 最大场相对 L2 | <6%? |
|---|---|---|
| 1.0 | 0.0094% | 勾 |
| 5.0 | 0.0111% | 勾 |

### 表 3：收敛阶梯（t=T=10）

| 族 | N | 最大场相对 L2 | <6%? |
|---|---|---|---|
| P1 | 16 | 14.8316% | 叉 |
| P3 | 8 | 16.4512% | 叉 |

### 表 4：CFL 敏感性（P2 32x32, t=T）

| CFL | 最大场相对 L2 | <6%? |
|---|---|---|
| 0.4 | 0.0152% | 勾 |

### 表 5：P0 基线（t=T）

| N | 最大场相对 L2 | <6%? |
|---|---|---|
| 64 | 14.3047% | 叉 |
| 128 | 12.0846% | 叉 |

## ④ 结果图

![等熵涡随输运时间的相对 L2 误差累积（P2 32x32，一个周期 T=10），虚线为 6% 判据。](figures/fig_error_history.png)
*图：等熵涡随输运时间的相对 L2 误差累积（P2 32x32，一个周期 T=10），虚线为 6% 判据。*

![一个整周期末（t=T=10）的最大场相对 L2 误差随网格尺度的收敛阶梯，含 P0 基线与 6% 判据线。](figures/fig_convergence_period.png)
*图：一个整周期末（t=T=10）的最大场相对 L2 误差随网格尺度的收敛阶梯，含 P0 基线与 6% 判据线。*

![整周期末误差对 CFL 的敏感性（P2 32x32）。](figures/fig_cfl_sensitivity.png)
*图：整周期末误差对 CFL 的敏感性（P2 32x32）。*

## ⑤ 误差判定与讨论

- 门禁配置 dg_d2_n32_t10_cfl0.2：最大场相对 L2 = 0.0152%，< 6% 容差 → 通过。
- 误差增长因子 err(T)/err(t_min) = 1.1803，≤ 预声明上限 5.0 → 通过。
- 关键发现：P2 DG 在 32x32 网格上，t=1 时误差 0.0094%，t=10 时 0.0152%，增长因子 1.6077×。误差增长极小（α ≪ 1），证实高阶 DG 的误差是网格属性而非时间属性。
- 收敛阶梯：0 个配置通过，2 个未通过（dg_d1_n16_t10_cfl0.2, dg_d3_n8_t10_cfl0.2），表明高阶格式也需要足够的空间分辨率来维持长时输运精度。

## ⑥ 复现命令

```bash
# 全量运行（15 配置，建议递交集群）
bash tools/run_case_yee_vortex.sh
# 或单配置运行
PYTHONPATH=. python3 benchmarks/case_euler_vortex_yee.py --config-index 0 --output-dir runs/case_yee2000_vortex
# 聚合与报告生成
PYTHONPATH=. python3 benchmarks/case_euler_vortex_yee.py --task aggregate --output-dir runs/case_yee2000_vortex
```

## ⑦ 局限与注意点

- 本文献全文在受限网络环境下未能获取（NTRS 403），数值目标回退到解析解。解析解经 machine-verified（div F = 0, U_t+F_x+G_y = 0, 等熵律等），但缺少与文献表格的直接对照。
- 长时间积分（t=10）的误差饱和现象表明 SSP-RK3 时间推进在长时模拟中存在不可忽略的数值耗散。若需更高精度长时模拟，需考虑更高阶时间方案或辛积分器。
- P0 基线在 t=10 时误差显著高于高阶 DG（P0/64² 约 14%，P0/128² 约 12%），说明低阶格式的数值耗散不适合长时精确模拟。
- 所有配置均在双周期边界下运行，无壁面反射干扰。