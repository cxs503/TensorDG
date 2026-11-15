# XVAL-LBM 静态浸没标记障碍（LIFE static-immersed-marker-obstacles）在 TensorLBM 的复现与对标

**锚点** `XVAL-LBM-STATIC-IMMERSED-MARKER-OBSTACLES`（`joconnor22/LIFE` 做法，功能点 LBM-FP-65）。
**判据** Re=100 圆柱绕流 `Cd = 1.35`（带 1.32–1.38）、`St = 0.165`
（带 0.164–0.166），生态验收 ≤ 15%。

## 1. LIFE 做法的三个要素（实现即照此）

1. 障碍物 = **静态拉格朗日标记点云**，`CIRCLE` 行按周长展开 `floor(2*pi*r/dx)` 个等距标记
   （`src/IBMBody.cpp:55`）；相邻 < 0.5 dx 的标记被剪掉（`IBMBody.cpp:64`）。**没有 mask bounce-back，
   没有稀疏索引压缩**——障碍物就是这圈标记。
2. 标记速度恒 0 = 无滑移；标记力经紧支撑 delta 核铺到格子（`IBMNode.cpp:218`、`Objects.cpp:410`），
   带 ε 松弛校正。
3. 标记间距 ≈ dx。

## 2. 本仓实现（新增代码）

| 文件 | 内容 |
|---|---|
| `TensorLBM/src/tensorlbm/immersed.py` | 3 点/4 点 delta 核（单位分解 + 一阶矩为零）、`make_circle_markers`（LIFE 展开/剪枝）、`StaticMarkerObstacles`（预计算紧支撑 stencil，spread/interpolate **精确伴随**）、`multi_direct_forcing`（多直接强迫 + ε 松弛 + gain 校正）、`guo_source`（Guo 2002 源项） |
| `TensorLBM/src/tensorlbm/cylinder.py` | 标记圆柱通道算例：D2Q9 BGK、入口定速度 / 出口定压（非平衡外推，Guo 2000）、上下半程反弹、Cd/Cl 时间序列、St 频谱 + 零点交叉校验、标记滑移、质量漂移、控制体动量收支诊断 |
| `TensorLBM/benchmarks/cylinder_ibm_xval.py` | 细化阶梯 + 标记密度研究，程序化写 `outputs/cylinder_ibm/xval_summary.json` |
| `TensorLBM/tests/test_immersed_ibm.py` | 13 个不变量测试（核单位分解、伴随性、力守恒、静止场零虚假力、均匀流阻力方向、Guo 源项零质量、壁面动量交换符号、开边界条件、物理短跑） |
| `TensorLBM/src/tensorlbm/t3_matrix.py` | `cylinder-2d-re100` 从 `NOT_SUPPORTED`（无弯曲壁面 + 无力积分）改为注册探针 `cylinder_ibm_cd_st` |

## 3. 关键数值结论（本机短阶梯，D=16）

| 项 | 值 |
|---|---|
| 标记数 | 50.0（间距 1.0053 dx） |
| Cd | 1.6807（文献 1.35，误差 24.5%） |
| St（频谱/零点交叉） | 0.0886 / 0.2370（文献 0.165，误差 46.3%） |
| Cl,rms | 0.0572（脱落已建立，窗口内 2 个周期） |
| 标记无滑移残差 | 0.93% of U |
| 质量漂移 | +3.00e-03 |
| 控制体动量收支失配 | 100.2%（诊断，WITHHELD，不计入验收） |

## 4. 坑与局限（详见平台报告第 5 节）

1. 完美对称初始场永不脱落（`Cl ≡ 0`）→ 必须加横向扰动，本单取 10% u0 高斯。
2. 两端都外推密度 = 没有压力参考，实测 `dp` 无界漂移、收支失配 400 倍 → 出口必须钉 `rho = 1`。
3. 低 tau 下多直接强迫会过松弛（速度响应 `gain = 3/2 - omega/2`），不校正则牵引力振荡、
   Cd 在 0.3–2.7 乱跳；校正后单步收敛。
4. 扩散界面把实体抹宽约 1 dx → Cd 系统性偏高；该偏差随 D 收敛，标记加密作用有限。
5. 通道高 10 D（堵塞比 10.0%）对文献无限域值抬高 Cd，未做堵塞修正。
6. 仅支持静态刚性零速标记（LIFE 同样不支持运动/柔性体）。

## 5. 复现

```
python3 -m pytest TensorLBM/tests/test_immersed_ibm.py -q
cd TensorLBM && python3 benchmarks/cylinder_ibm_xval.py --ladder 16,24,32 --steps-per-diameter 120 --spacing 0.5
python3 publish_xval_lbm_immersed_markers.py
```
