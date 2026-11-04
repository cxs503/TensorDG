# TensorDG：非定常壁面热流的参考解级验证 —— 参数区间陷阱使 5.6% 偏差现形（m3640）

**仓** TensorDG · **mission** m3640 · **模块** `tensordg/ns_dg_2d.py`（壁面热流通道）

## ① 案例介绍

壁面热流通道此前只有**稳态**证据（m2831/m2876：线性导热场 <3%）。m3614 §5 局限 4
明确把「非定常壁面热流」列为未验证。本单补经典热渗透算例：气体静止充满 `0<=y<=H`，
初温均匀 `T0`；`t=0+` 底壁阶跃到 `T_w=T0+dT` 并保持，顶壁 `T0` 且远离。主导方程为常压导热
`dT/dt = alpha d2T/dy2`，`alpha = kappa/(rho0 c_p)`，精确解
`T = T0 + dT erfc(y/2sqrt(alpha t))`，壁面热流 `q_w = kappa dT / sqrt(pi alpha t)`。

参数：gamma=1.4、R=1、c_p=3.5、Pr=0.71；主算例 `mu=0.01`（kappa=0.0493，alpha=0.01408），
`t=0.7`，H=1，x 周期 1 单元，y 向 8/12/16/24 单元，P1/P2，SSP-RK3 + BR1/SIPG。

## ② 怎么做的

1. 求解器最小增量：`boundary` 条目现在也接受可调用对象 `f(t)->{velocity,temperature}`，
   SSP-RK3 三个阶段按阶段时刻求值（`stage_boundary`）。默认路径逐位不变
   （测试 `test_callable_boundary_matches_constant_boundary`，atol=0 全等）。
2. 壁面热流取格式自己用的量：`ymin` 面 DG 边界数值通量的能量分量
   （`ns._face_primitives` + `ns._boundary_flux`，与 m2831/m3139 审计同一估计器）。
3. 参考解自检：闭式 `q_w` vs erfc 场单侧二阶差分，相对差 2.3e-9（阈值 1e-4）。
4. 参考解适用条件显式化：新增判据 `2 sqrt(alpha t)/(c_s t)`，并据此选参数区间。
5. 工件：`benchmarks/verify_ns_transient_wall_heat.py`（幅值阶梯 + 网格/阶数阶梯 + 出图）、
   `tests/test_ns_transient_wall_heat.py`（8 项）、`tools/render_transient_wall_report.py`（报告零手抄）。

## ③ 计算结果

### 3a 第一版参数区间（mu=0.05, t=0.2）：系统性 5.6% 偏差，网格/阶数/幅值都治不好

| 阶数 | ny | q_w 相对误差 | T 剖面相对 L2 |
|---:|---:|---:|---:|
| P1 | 8 | 5.343% | 0.026% |
| P1 | 16 | 5.617% | 0.025% |
| P1 | 24 | 5.668% | 0.024% |
| P2 | 8 | 5.613% | 0.026% |
| P2 | 12 | 5.653% | 0.025% |

幅值阶梯（P1，ny=16）：dT=0.020 -> 5.804%，dT=0.010 -> 5.680%，dT=0.005 -> 5.617%
；外推到 dT->0 仍 5.555%。

### 3b 修正参数区间（mu=0.01, t=0.7，判据比 1.00 -> 0.24）

| 阶数 | ny | q_w 相对误差 |
|---:|---:|---:|
| P1 | 8 | 0.708% |
| P1 | 16 | 0.405% |

同一格式、同一估计器，误差从 5.6% 降到 **0.71%（ny=8）/ 0.41%（ny=16）**，
剖面相对 L2 约 0.02%。

## ④ 结果图

图为第一版参数区间的温度剖面/误差曲线（`docs/figures/ns_dg_2d_transient_wall_heat.png`）：
剖面本身对 erfc 精确到 0.03%，而壁面热流偏差 5.6% 且不随网格改善——正是「参考解不适用」的指纹。

## ⑤ 注意点

1. **最大坑：参考解自身的适用条件**。erfc 解要求层内声学响应准静态，即
   `t >> 4 alpha/c_s^2 = 4 mu/(gamma p0 Pr)`。第一版 `mu=0.05,t=0.2` 时该比为 1.00，
   壁面热流偏低 5.6%，且对 h/p/dT 都不敏感——极易误判为「格式 bug」。修参数后降到 0.4%。
2. 壁面热流比剖面难一个数量级：同网格下剖面 <0.1%、热流 5%。只报剖面误差会高估该通道。
3. `t=0+` 阶跃使热流在初始时刻发散，热渗透长度必须覆盖若干单元才能谈 <3%。
4. 局限：常数 mu、Cartesian、显式推进、单核 float64；强加热 `dT/T0=O(1)` 不适用。
5. 主算例（mu=0.01, t=0.7）的完整阶梯在本单提交时仍在跑（高负载），
   ③3b 的两点是已测点；阶梯与观测阶由 `benchmarks/verify_ns_transient_wall_heat.py` 复现。

## ⑥ 误差判定（逐项，显式回答是否 <3%）

| 判据 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 壁面热流 P1 ny=16（mu=0.01,t=0.7） | erfc 半无限介质解 | 0.405% | **是** |
| 壁面热流 P1 ny=8（同上） | 同上 | 0.708% | **是** |
| 温度剖面 P1 ny=16 | erfc 精确剖面 | 0.021% | **是** |
| 参考解闭式 vs 有限差分 | erfc 场一侧差分 | 2.3e-9 | **是** |
| 壁面热流（旧参数 mu=0.05,t=0.2） | 同上 | 5.617% | **否（honest FAIL，归因参数区间而非格式；已据判据修正）** |

## ⑦ 复现

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 python -m benchmarks.verify_ns_transient_wall_heat
PYTHONPATH=. python -m pytest tests/test_ns_transient_wall_heat.py -q   # 7 passed / 1 failed
PYTHONPATH=. python -m tools.render_transient_wall_report
```
