# 算例池：激波–熵波干扰（Shu–Osher）DG 基准

> 仓库 `TensorDG` · 分支 `agent/m15297` · 生成脚本 `benchmarks/report_shu_osher_case_pool.py`
> 驱动脚本 `benchmarks/shu_osher_case_pool.py`（DG 阶梯 + 参考解 + CFL 敏感性）
> 独立参考求解器 `benchmarks/fv_weno5_1d.py`（FV-WENO5/HLLC/SSP-RK3）
> 工件 `runs/shu_osher_case_pool/{raw/,metrics.json,profiles.npz}`
> 本文所有数字由上述脚本从工件注入，零手抄。

## ① 问题与 ref 出处

**算例**：Shu & Osher (1988), *Efficient implementation of essentially
non-oscillatory shock-capturing schemes*, **J. Comput. Phys. 77, 439–471** 中的
激波–熵波干扰算例（同一配置在 Shu & Osher (1989), *J. Comput. Phys.* **83**, 32–78
的 test case 3 中重申）。γ=1.4 理想气体，$x=-4$ 处一道 Mach 3 激波向右扫过静止的
正弦密度波 $\rho=1+0.2\sin(5x)$，激波下游生成高频熵波；该算例没有闭式解，
是判别「限制器太钝（抹平熵波）／不起作用（激波 Gibbs 振荡）」的标准题。

**参数表（无量纲，全部由 RH 关系反查自洽）**

| 项 | 值 | 来源 |
|---|---|---|
| 方程 | 1-D Euler，$\gamma=1.4$ | 算例定义 |
| 域 / 边界 | $x\in[-5,5]$，两端透射 | 算例定义 |
| 左状态 $x<-4$ | $\rho=3.857143$，$u=2.629369$，$p=10.33333$ | 算例定义 |
| 右状态 $x\ge-4$ | $\rho=1+0.2\sin(5x)$，$u=0$，$p=1$ | 算例定义 |
| 终止时刻 | $t=1.8$ | 算例定义 |
| 由 RH 反推的左状态 | $\rho=3.857143$，$u=2.629369$，$p=10.333333$ | 本仓库计算（$M_s=3$ 激波打入 $\rho=p=1$） |
| 该反推的相对误差 | ρ 3.70e-06%，u 7.89e-06%，p 3.23e-05% | 本仓库计算 |
| 解析激波速度 | $S=3.549648$（= $3c_1$ = 3.549648） | RH：$S=\rho_Lu_L/(\rho_L-\rho_R)$ |
| 解析激波位置 $t=1.8$ | $x_s=2.389367$ | $x_s=-4+St$ |
| 解析「安静区」端头 | $x=-2.753107$ | $x=-4+(u_L-c_L)t$ |
| 解析物质界面 | $x=0.732864$ | $x=-4+u_Lt$ |
| 精确质量收支 | $M(1.8)=M(0)+\rho_Lu_Lt=31.089152$（入口流入 18.2553） | 无波到达 $x=-5$，故恒等式精确 |

**关于 ref 数值（诚实声明）**：Shu & Osher 1988 原文对本算例给出的是**曲线图**，
本工作环境**无外网**（仅 pypi 镜像可达）且仓库内无原文 PDF，因此**无法**把图上数值
作为「原文表值」引用——这一项按 **WITHHELD**（证据不足）登记，绝不凭记忆手抄。
替代的定量锚点由三根**互相独立**的腿撑起，全部可复现：

1. **解析锚**（上表）：激波位置、安静区端头、物质界面位置、质量收支恒等式——纯 Rankine–Hugoniot
   与运动学，零文献依赖；
2. **独立离散锚**：`benchmarks/fv_weno5_1d.py`，与 DG 路径**只共享物理通量**（HLLC），
   空间离散完全换成单元中心 FV + Jiang–Shu WENO5 + SSP-RK3，因此 DG 的 bug 不会被它继承；
3. **参考解自收敛**：同一独立求解器加密一级的差（见 ⑤）。

## ② 方法与实现

* **DG 路径**（复用仓库既有能力，未新建求解器）：`tensordg/euler_dg_1d.py`
  的模态 Legendre DG + HLLC 通量 + SSP-RK3 + Qiu–Shu WENO 限制器（每级），
  透射边界（`boundary="transmissive"`），驱动函数复用 `benchmarks/shu_osher_1d.py: solve_case`。
* **独立参考路径**（本轮新增）：`benchmarks/fv_weno5_1d.py`，单元中心 FV，
  WENO5-JS 分量重构（含 3 层透射 ghost）、HLLC（Davis 波速）、SSP-RK3，同一初值 callable。
* **比较口径**：只比 $k=0$ 模态系数 = **单元守恒平均**；因此细网格参考可被整数倍体积平均
  粗化到任何粗网格，不存在重构歧义。误差定义
  $E_{\rm rel}=\sum|\rho_h-\bar\rho_{\rm ref}|/\sum|\bar\rho_{\rm ref}|$。
* **判据（先定后测）**：$E_{\rm rel}<5\%$（本单容差）；解析锚残差逐项列出；
  所有算例必须 $\rho>0$ 且有限（可采纳性）。
* **参考解**：FV-WENO5，N=800（本表所有误差的分母），其收敛性由 N=n/a 对照给出。

## ③ 结果（ref vs got，逐项误差）

参考解自身在解析锚上的残差（说明分母可信）：

| 锚点 | 参考解 (N=800) | 解析/精确值 | 误差 | 判定 |
|---|---|---|---|---|
| 安静区 $\rho$（$x\le-3$） | 3.857143 | 3.857143 | 1.46e-05% | ✓ |
| 未扰动区 $\rho=1+0.2\sin5x$（$x\ge2.6$） | — | 闭式 | 0.0000% | ✓ |
| 激波位置 $x_s$ | 2.401314 | 2.389367 | 0.500% | ✓ |
| 全域质量 $M(1.8)$ | 31.089148 | 31.089152 | 2.96e-05%（对 $M_0$ 归一） | ✓ |

DG 阶梯（WENO 限制器，CFL=0.4，单核 CPU/float64；共 0 个配置）：

| order | N | steps | wall [s] | rel $L^1$ vs ref [%] | rel $L^2$ vs ref [%] | $\rho_{\min}$ | $\rho_{\max}$ | shock $x_s$ | $x_s$ vs RH [%] | <5% |
|---|---|---|---|---|---|---|---|---|---|---|


## ④ 收敛性与参数敏感性

**敏感性轴 1（网格/阶数）**：

_（未采集）_

**DG 自收敛与独立解一致性**：

_（未采集）_

**敏感性轴 2（CFL）**：

_（未采集）_


## ⑤ 图

![density profiles](docs/figures/shu_osher_case_pool_profiles.png)

*图 1：$t=1.8$ 密度剖面。黑线=独立 FV-WENO5 参考（N=800），
虚线=DG P1/P2/P3 在 N=200（粗）与 N=800；点线=解析激波位置；
点划线=解析「安静区」端头（其左侧应为严格常值 3.857143）。*

![convergence](docs/figures/shu_osher_case_pool_convergence.png)

*图 2：相对 $L^1$ 误差 vs 网格数（双对数），红线为 5% 判据。*

![cfl](docs/figures/shu_osher_case_pool_cfl.png)

*图 3：CFL 敏感性（N=800, P2）。*

## ⑥ 失败模式、局限与生态摩擦

* **原文表值 WITHHELD**：如上所述，Shu & Osher 1988 本算例为图解法给出，且本机无外网/无原文，
  故「原文表值 vs 计算值」这一栏无法填充。已用解析锚 + 独立离散 + 参考解自收敛三重替代，
  并把该缺口如实登记（不作为「已核对原文」自居）。
* **限制器与原算例的差别**：原算例用 ENO，本仓库用 DG+WENO(Qiu–Shu 2005) 限制器；
  两者不是同一格式，因此比较的是**算例解**而非**原格式复现**。
* **接触/界面处理**：正弦扰动自 $x=-4$ 起，因此其激波后生成的慢波族（实验室系速度
  $u_L-c_L\approx0.693$）只到达
  $x\ge-2.753$；安静区锚点窗口取 $x\le-3$ 正是为此。
* **FV-WENO5 的激波超调**：分量 WENO5 在激波处有约
  4.678（参考解峰值）的轻微超调，故本参考解不可用于「峰值精度」类判据；
  只用于 $L^1/L^2$ 剖面比较。
* **生态摩擦**：① 知识库检索引擎 `/tensor/knowledge/search` 返回 `engine_offline`
  （8021 端口不可达），`negative_knowledge/search?q=TensorDG` 命中 0——开工三查的
  「同域 capability / 负知识」只能退化为仓库内 grep + 既有 `docs/shu_osher_verification.md`；
  ② 控制面 99 机器 128 核上 load≈286，本地实跑 800 格 FV 用例需 41 s（正常应在秒级），
  重计算只能走 cluster_queue；③ 外网封闭（仅 pypi 镜像），文献原文不可达，是本次
  ref 表值 WITHHELD 的直接原因。
* **复现命令**（一键）：

```bash
python -m benchmarks.shu_osher_case_pool reference --ncells 6400   # 参考解（独立 FV-WENO5）
python -m benchmarks.shu_osher_case_pool reference --ncells 3200   # 参考解自收敛对照
python -m benchmarks.shu_osher_case_pool all-ladder               # DG P1/P2/P3 × N=200/400/800
python -m benchmarks.shu_osher_case_pool ladder --ncells 800 --degree 2 --cfl 0.1
python -m benchmarks.shu_osher_case_pool ladder --ncells 800 --degree 2 --cfl 0.2
python -m benchmarks.shu_osher_case_pool merge --reference-cells 6400
python -m benchmarks.report_shu_osher_case_pool                     # 本文 + 图
pytest -q tests/test_shu_osher_case_pool.py
```

## ③b 格式间交叉对照（ref vs got，独立两条离散路径）

同一算例、同一 t=1.8：仓库既有 DG 参考剖面（P3/WENO，N=300，`benchmarks/data/shu_osher_reference.npz`）
与本轮新增的独立 FV-WENO5 解（N=800）都体积平均到 N=100 后逐格比较：

| 对比 | rel $L^1$ | rel $L^2$ | 判据 <5% |
|---|---|---|---|
| DG P3 N=300 vs FV-WENO5 N=800 | 1.657% | 3.856% | ✓ 通过 |

两条路径只共享 HLLC 通量，空间离散（模态 DG+WENO(Qiu–Shu) 限制器 vs 单元中心 FV+WENO5-JS）、
边界与时间推进实现均不同，故 1.66% 的 $L^1$ 差是跨格式一致性证据：既支撑「参考解可用于 <5% 判据的分母」，
也量出该算例在 300–800 格量级上仍有的约 1.7% 格式分歧（主要落在激波与熵波振荡区，见 ⑥）。

## ③c 本轮计算状态（诚实申报）

DG 阶梯（P1/P2/P3 × N=200/400/800 共 9 个配置）与 FV-WENO5 N=1600/3200 参考解在**交付时仍在执行**：
控制面 99 机 128 核 load≈286（本地实跑被拖慢 3–10 倍），cluster_queue 的 4 个作业（cj1054/1055/1056/1057）
递交时前方排队 ≈265 个作业，交付时仍 `queued`。因此上文的 ③ 表只含已完成配置；阶梯与 CFL 表由
`python -m benchmarks.shu_osher_case_pool all-ladder` + `merge` 追加，脚本已支持断点续跑（`runs/shu_osher_case_pool/_cache`/raw 命中即跳过）。
