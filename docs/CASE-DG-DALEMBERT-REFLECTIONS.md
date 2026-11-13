# CASE-DG-DALEMBERT-REFLECTIONS — d'Alembert 解越过第一次缝合：多次反射镜像级数验证

> 算例键 `dalembert-elastic-rod-1d`（锚 `CASE-DG-DALEMBERT-ELASTIC-1D`）· 仓库 TensorDG · 工件目录
> `validation/elastic_1d_reflections/` · 生成于 2026-11-12 · 本机运行 194.9 s · 起点 commit `95dc7f8`
> 入口 `python -m benchmarks.elastic_1d_reflections --output-dir validation/elastic_1d_reflections`

本报告是对已合并算例（m16246，`docs/CASE-DG-DALEMBERT-ELASTIC-1D.md`，L4）的**验证欠账复核**，
不是重做：那条算例已 9/9 锚点通过，但它自己记录的局限 #3 是「闭式解仅在 `c t ≤ 0.45` 且脉冲未穿缝时
有效」——也就是说**物理上最重要的多次反射/长时累积色散区段，一直没有被验证过**。本节点把该区段补上，
并顺带发现并修掉了参考解的一处**静默失效**缺陷。

## ① 案例介绍（物理背景 / 几何 / 参数）

一维均质线弹性细杆纵波，速度—应力一阶双曲系统（与主算例同一方程组）：

```
v_t - sigma_x/rho = 0 ,   sigma_t - E v_x = 0 ,   c = sqrt(E/rho) ,   lambda± = ±c
w+ = sigma - rho c v （随 +c），w- = sigma + rho c v （随 -c）
```

初始为静止杆内的 Gauss 速度脉冲 `v0 = exp(-((x-L/2)/w)^2)`、`sigma(x,0)=0`。d'Alembert 解把它劈成
两列半幅反向行波。**关键点**：闭式解 `v=(V(x-ct)+V(x+ct))/2` 只是**单幅像**（n=0），一旦半波行至端部
（周期缝 / 固支端 / 自由端）就漏掉了反射回来的那一幅——真实解必须写成**镜像级数**。

长度 1 的杆，参数全无量纲（与主算例同一套）：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| length | 1.0 | rho / young | 1.0 / 1.0 |
| c | 1.0 | 脉冲中心 center | 0.5 |
| 脉冲宽 width | 0.05 | 初始应力 | 0（应力自由释放） |
| 每次遍历时间 `L/c` | 1.0 | 名义缝合限 `min(center, L-center)` | 0.5 |

两种端部条件，对应两种物理反射符号：

| 场景 | 端点 | 镜像构造 | 反射符号 |
|---|---|---|---|
| `periodic` | 周期缝 $x=0\equiv L$ | $\tilde V(y)=\sum_n V(y-nL)$ | 同号（波形穿过缝） |
| `clamped-free` | $x=0$ 固支（$v=0$）、$x=L$ 自由（$\sigma=0$） | 关于 0 **奇**延拓 + 关于 L **偶**延拓 ⇒ $4L$ 周期，$V(y+2L)=-V(y)$ | 固支端反号、自由端同号 |

参考出处：K. F. Graff, *Wave Motion in Elastic Solids*（Ohio State Univ. Press 1975；Dover 1991）第 1 章
细杆纵波——杆速 $c=\sqrt{E/\rho}$、固支端 $\sigma$ 加倍/自由端 $v$ 加倍的反射规则、行波叠加。
**零手抄声明**：本环境读不到原书，报告与代码中**没有**任何数字抄自原书表格；全部 ref 值由镜像级数
或模态闭式在程序内算出，且镜像级数在缝前退化为原单幅闭式（实测偏差 ≤1e-16，见 ④#9）。

## ② 方法与实现

| 环节 | 实现 | 位置 |
|---|---|---|
| 参考解（任意 t） | 周期镜像级数、固支-自由 $4L$ 周期镜像 | `tensordg/elastic_1d.py::_dalembert_periodic_images` / `_dalembert_fixed_free_images` |
| 缺陷修复 | `dalembert_pulse` 越界时**抛 ValueError**（不再静默返回绕缝后近零场），`strict=False` 保留旧行为 | `tensordg/elastic_1d.py::dalembert_pulse` |
| 判据度量 | **能量范数**相对误差 $\|(dv, d\sigma/\rho c)\|/\|(v_{ex},\sigma_{ex}/\rho c)\|$；plain $\sigma$ 相对 L2 在整数次遍历处**退化**（$\sigma_{ex}\equiv0$），故不作判据 | `benchmarks/elastic_1d_reflections.py::energy_norm_error` |
| 驱动与判据 | 遍历数扫描 + 长时收敛 + 故障注入 + 参考解自校 | `benchmarks/elastic_1d_reflections.py` |
| 测试 | 20 条（参考解自校/边界条件/缺陷门禁/端到端/度量退化/故障注入/驱动产物） | `tests/test_elastic_1d_reflections.py` |

求解器本体（模态 Legendre DG + 精确迎风 + SSP-RK3 + 特征边界）**未改动**——本节点只新增参考解与判据，
符合「验证欠账」类工作不改被测对象的纪律。

## ③ 计算结果

### (a) 多次反射遍历扫描（N=200, P2, CFL=0.3）

| 场景 | t | 遍历数 `ct/L` | 能量范数误差 | 峰值速度 got / 精确 | 步数 |
|---|---|---|---|---|---|
| periodic | 0.20 | 0.2 | 0.00227% | 0.500001 / 0.5（双半波） | 667 |
| periodic | 1.00 | 1.0 | 0.00231% | 0.999993 / 1.0 | 3334 |
| periodic | 2.00 | 2.0 | 0.00243% | 0.999987 / 1.0 | 6667 |
| periodic | 4.00 | 4.0 | 0.00288% | 0.999976 / 1.0 | 13334 |
| clamped-free | 0.50 | 0.5 | 0.00228% | 0.999995 / 1.0 | 1667 |
| clamped-free | 1.50 | 1.5 | 0.00236% | 0.999990 / 1.0 | 5000 |
| clamped-free | 3.50 | 3.5 | 0.00274% | 0.999979 / 1.0 | 11667 |

误差随遍历数**单调微增**（0.00227% → 0.00288%，4 次遍历涨 27%），峰值速度同期从 0.500001 掉到
0.999976（相对 1.0 少 2.4e-5）——这就是**累积数值耗散/色散的真实量级**，此前从未被测到。

### (b) 长时收敛（t=2.5，非整数遍历，避开 $\sigma_{ex}\equiv0$ 的退化时刻）

| N | 能量范数误差 | 步数 |
|---|---|---|
| 50 | 0.6405% | 2084 |
| 100 | 0.03062% | 4167 |
| 200 | 0.00252% | 8334 |

测得 EOC = **3.996**（理论 p+1 = 3）。如实说明：该值**高于**理论阶，属本范数/本区间的超收敛表现，
只能作为「不劣于理论阶」的证据，不能反过来当作四阶格式的声明。

### (c) 参考解有效性扫描（解析量，非求解器）

单幅闭式 vs 镜像级数的偏差（以单位脉冲应力幅 $\rho c/2$ 归一）：

| `c t / L` | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 |
|---|---|---|---|---|---|---|---|
| 偏差 | 0 | ~0 | 0（4.8e-8 绝对） | 0.0109% | **1.6901%** | 35.33% | 99.96% |

**这是本节点最实质的发现之一**：名义有效限是「半波中心到达缝」的 `c t = 0.5`，但**有限脉宽**让
Gauss 尾部提前越缝——`c t = 0.35`（center − 3·width）时偏差已达 1e-4 量级，`c t = 0.40` 时偏差
**1.69%，已经顶到 2% 验收容差**。主算例用 t=0.2，不受影响；但原文档写的有效窗 `c t ≤ 0.45` 偏宽。

## ④ ref vs got 逐项判据（13/13 PASS，最差 0.2876% << 2%）

| # | 量 | ref | got | 误差 | 判据 | 结论 |
|---|---|---|---|---|---|---|
| 1 | periodic 2 遍历 能量范数误差 | 0 | 0.24263% | 0.24263% | <2% | ✅ |
| 2 | periodic 4 遍历 能量范数误差 | 0 | 0.28763% | 0.28763% | <2% | ✅ |
| 3 | clamped-free 3.5 遍历 能量范数误差 | 0 | 0.27425% | 0.27425% | <2% | ✅ |
| 4 | periodic 重组峰值速度（t=L/c，精确 1.0） | 1.0 | 0.9999927 | 0.00073% | <2% | ✅ |
| 5 | clamped-free 3.5 遍历峰值速度 | 0.99999 | 0.99998 | 0.00145% | <2% | ✅ |
| 6 | periodic 4 遍历弹性能漂移 | 0 | 0.07637% | 0.07637% | <2% | ✅ |
| 7 | 固支端速度泄漏 $\max\|v(0)\|/\max\|v\|$ | 0 | 0.05165% | 0.05165% | <2% | ✅ |
| 8 | 自由端应力泄漏 $\max\|\sigma(L)\|/\max\|\sigma\|$ | 0 | 0.05165% | 0.05165% | <2% | ✅ |
| 9 | 镜像解 vs 单幅闭式（t=0.2，自校） | 0 | 0（≤1e-16） | 0 | <2% | ✅ |
| 10 | 镜像解边界条件违例（t 至 7.5） | 0 | ~0 | 0 | <2% | ✅ |
| 11 | `dalembert_pulse(ct=0.6)` 必须抛错而非静默给错 | ValueError | ValueError | 0 | 门禁 | ✅ |
| 12 | 故障注入：同一数值解 vs **错号**反射级数必须被拒 | 检出 | 检出 | 0 | 门禁 | ✅ |
| 13 | 故障注入：错号参考解的度量值须超容差 | >2% | **100.0%** | 0 | 门禁 | ✅ |

**误差判定（显式回答 <3%？）**：全部关键量相对误差 **<0.29%**，**是** <3%，也 <2% 锚点容差；
最差项为 4 遍历周期算例的 0.28763%（误差随遍历数增长是物理上应有的累积效应，非判据缺陷）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **静默错参考是本节点真正修掉的缺陷**：`dalembert_pulse` 在 `c t > 0.5` 时返回的不是「近似解」而是
   **幅值近零的错解**（t=0.6 时 max|v| = 0.00395，真实应为 O(1)），任何下游用它当 ref 都会得到
   无意义的误差而不报错。现在抛 `ValueError` 并指向镜像解版本。
2. **$\sigma$ 的相对 L2 在整数遍历处退化**：`c t = nL` 时 $V(x-ct)=V(x+ct)$，$\sigma_{ex}\equiv0$，
   分母为零（早期版本的表里因此出现过 2.3e10 这种无意义数字）。判据改用能量范数后恒有定义。
3. **有限脉宽压缩有效窗**：见 ③(c)，名义限 0.5 ≠ 实用限 0.35。任何用单幅闭式做 ref 的下游算例，
   应把 `c t ≤ center − 3·width` 写进判据，或直接用镜像解。
4. **镜像级数的截断**：`n_images=6` 对 width/L=0.05 足够（`exp(-(nL/w)^2)` 在 |n|=1 即 1e-174）；
   更宽脉冲需增大 n_images，代码未自动判定。
5. **判据度量是采样和而非真正的单元求积**：`stress_profile` 给等距 Gauss 点，能量范数用等权和；
   对同为 P2 的场这是一致求积，但换 degree 时权重不严格等价，跨阶比较只作参考。
6. **EOC 超收敛**：测得 3.996 > p+1=3，如实报告且不作强声明。
7. 本节点**未**覆盖：非均质/变截面杆（阻抗失配反射 $R=(Z_2-Z_1)/(Z_1+Z_2)$）、2D/3D 弹性波、
   材料色散——这些仍是该算例的 L5 欠账。

## ⑥ 工件与证据指针

| 文件 | 用途 | sha256[:12] |
|---|---|---|
| `validation/elastic_1d_reflections/elastic_1d_reflections.json` | 遍历扫描 + 收敛 + 有效性扫描 + 故障注入 | 41f1cbd84477 |
| `validation/elastic_1d_reflections/benchmark_ref_vs_got.json` | 13 行 ref/got/误差/判定 | fdaebaff815c |
| `validation/elastic_1d_reflections/reflection_profiles.csv` | 5 条剖面 num vs exact 全量 | d8bd78d15c17 |
| `validation/elastic_1d_reflections/elastic_1d_reflections.png` | 四面板图（剖面/反射/误差累积/收敛） | 29c20dc437ff |
| `validation/elastic_1d_reflections/run_manifest.json` | 复现命令 + 各工件 sha256 | — |
| `docs/figures/elastic_1d_reflections.png` | 论文图副本 | — |

- `impl_files`: `tensordg/elastic_1d.py`（镜像参考解 + 越界门禁）
- `entry`: `benchmarks/elastic_1d_reflections.py`；`tests`: `tests/test_elastic_1d_reflections.py`（20 条）
- `repro_cmd`: `python -m benchmarks.elastic_1d_reflections --output-dir validation/elastic_1d_reflections`
  （194.9 s，单核；集群可递交）
- 复现测试：`python -m pytest tests/test_elastic_1d_reflections.py -q`

**与锚 `CASE-DG-DALEMBERT-ELASTIC-1D` 的关系**：锚本体的 9 锚点验收（L4）已由 m16246 完成并通过审计，
本节点补的是它自述的局限 #3（长时/多次反射无验证）与一处会误导下游的静默错参考缺陷；
锚的验收数字未变，也未重跑替换。