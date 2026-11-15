# T3 国际标准验证矩阵（GAP-LBM-T3-BENCH）

**状态：矩阵骨架 + 墙边界算符实测通过；九例中 1 例 REAL 通过、2 例探针已落地但
待满分辨率跑完（WITHHELD）、6 例诚实标注 NOT_SUPPORTED（附缺口与预注册判据）。**

本文件是 `src/tensorlbm/t3_matrix.py` 的规格与证据说明。矩阵不是一份"已验证"清单，
而是一份**可执行、可审计、不可事后调参**的规格：判据与容差在跑之前写死在代码里，
跑不动的例子永远显示 NOT_SUPPORTED，不会因为"看起来合理"而变成 PASS。

---

## 1. 为什么做成这个形状

T3 线的失效模式是：某个算例被"验证过一次"，数字被抄进报告，之后再没人重跑；或者
跑出来 7% 误差、把容差从 5% 改成 10% 就算通过。因此矩阵的三条硬约束：

1. **容差预注册**：`CASES[*].tolerance` 与 `tolerance_basis` 在运行前固定；
   `validate_spec()` 拒绝任何 > 15%（生态验收上限）的容差、拒绝没有基线的容差。
2. **无探针不得 REAL**：`probe is None` 的行只能是 `NOT_SUPPORTED`，并必须写清
   `gap`（缺什么能力）。`validate_spec()` 拒绝"有 probe 又有 gap"或"没 probe 也没 gap"
   的行。
3. **探针必须真跑求解器**：runner 读探针返回的 `solver_steps`，为 0 则把该行降级
   为 `MOCK` 且判 FAIL（`tests/test_t3_matrix.py::test_probe_without_solver_steps_is_downgraded_to_mock`）。
   探针抛异常 → `WITHHELD`，`pass=None`，绝不当作通过。

证据分级：`REAL` / `WITHHELD` / `NOT_SUPPORTED` / `MOCK` / `STUB`。

## 2. 九算例矩阵（`python benchmarks/t3_matrix_run.py --quick|--full`）

| # | case | 案例 | 参照 | 判据（预注册容差） | 当前证据 |
|---|---|---|---|---|---|
| 1 | `cavity-2d-ghia` | 方腔 Re=100/400 | Ghia, Ghia & Shin 1982, JCP 48, 387, Table I/II | 两向中心线 L∞ 误差 ≤ 5% | 探针已落地；满分辨率跑排队中（cj3655）→ **WITHHELD** |
| 2 | `cylinder-2d-re100` | 圆柱 Re=100 | Cd=1.32–1.38, St=0.164–0.166（Park 1998 / Schäfer–Turek 1996） | Cd、St 相对误差 ≤ 10% | **NOT_SUPPORTED**：无浸入曲线壁面 + 无力积分（阻力）诊断 |
| 3 | `sphere-3d` | 球体 Re=100/300 | Clift–Grace–Weber Cd 曲线 | ≤ 10% | **NOT_SUPPORTED**：本包只有 D2Q9，无任何 3D 格点 |
| 4 | `turbulent-channel-re180` | 湍流槽道 Re_τ=180 | Kim–Moin–Moser 1987（κ=0.41, B=5.2） | u+ 对对数律 ≤ 5% | **NOT_SUPPORTED**：无亚格子/壁面函数、无时均诊断、2D 无法承载 |
| 5 | `taylor-green-2d` | Taylor-Green 涡衰减 | 解析 e^{-2νk²t} | 衰减率相对误差 ≤ 2% | **REAL 通过**：实测 4.78e-5（n=64, ν=0.05, 1500 步） |
| 6 | `static-droplet-laplace` | 静态液滴 Laplace 定律 | Δp·R = σ 常数（Laplace 定律） | Δp·R 相对离散 ≤ 5% | 探针已落地；粗配置（64², R=8/10/12, 1500 步）离散 38% → **WITHHELD**（未平衡/域偏小，满配置排队） |
| 7 | `rayleigh-taylor-2d` | Rayleigh-Taylor | n=√(A g k)（Chandrasekhar 1961） | 增长直线性拟合 ≤ 5% | **NOT_SUPPORTED**：无重力/浮力体力、无 Atwood 比初值、无界面振幅诊断 |
| 8 | `dam-break-ritter` | 溃坝 Ritter 解 | Ritter 1892 浅水精确解 | h(x) 相对 L2 ≤ 5% | **NOT_SUPPORTED**：无自由表面（干床）模型；Ritter 参照本身已在 DG 算例池 |
| 9 | `ship-resistance-waves` | 船体阻力/兴波 | ITTC 2014 KCS/KVLCC2 + 本域 TensorBEM | Ct ≤ 10% | **NOT_SUPPORTED**：需 3D+自由面+运动船体+消波，属 TensorBEM/FVM 域 |

覆盖率（REAL 通过）：**1/9**。这是如实数字，不是可交付的完成度——矩阵的价值在于
把剩下 8 项的**缺口和判据**钉死，任何后续单子要么把某行推到 REAL 通过，要么保持
NOT_SUPPORTED，不能含糊。

## 3. 本轮真正做实的东西：墙边界算符（`src/tensorlbm/boundaries.py`）

矩阵要能跑方腔，先得有壁面。此前 `tensorlbm` 只有周期边界，**没有任何壁面算符**，
方腔/圆柱/球体/槽道这四类案例在离散层面根本无法开始——这也是它们此前"未验证"
的物理原因，而不是"没时间跑"。

新增 `HalfwayBounceBack`：半程反弹（link-wise bounce-back），带移动壁面动量项
`f_i = f_ī^post + 6 w_i ρ_w (e_i·u_w)`，固体掩膜任意、周期 roll 拉取、逐方向预计算
掩膜与源项。参照 Krüger et al. 2017 ch.8 / Ladd 1994。

### 3.1 实测（`outputs/t3_wall_bc/summary.json`，程序化写出）

| 算例 | 解析解 | 实测误差 | 结论 |
|---|---|---|---|
| 平面 Couette（周期 x，上下壁，上壁 U，Ny=17, τ=1.0, 4000 步收敛） | u = U·Y 精确线性 | L∞ = **2.78e-8**（机器精度） | 移动壁面动量项的符号与量级**精确正确** |
| 压力驱动 Poiseuille（Ny=13, τ=0.8, F=1e-5, 3500 步收敛） | u = F/(2ν)·Y(H−Y) | L∞ = **0.76%**（Ny=33 时 0.096%） | 静止壁面 + 抛物剖面正确，随分辨率下降 |
| Taylor-Green（n=64, ν=0.05, 1500 步） | 幅值 e^{-2νk²t} | 衰减率相对误差 **4.8e-5** | 体相 BGK 与解析率一致 |

**关键约定（踩过的坑，务必沿用）**：半程反弹把壁面放在**节点外半格**，所以
`N` 个节点、两侧各一层固体时，通道宽度是 **H = N − 2**（不是 N−1），流体节点坐标
`Y = (y−0.5)/H`。最初的实现里 H=N−1 让 Poiseuille 的峰值速度差了 6.25%
（0.01200 vs 0.0128，比例恰为 31²/32²），这也是 Ghia 对比里最容易埋进去的系统偏差。

第二个坑：移动壁面的 `u_w` 存在**固体节点**上，必须随掩膜一起 shift 到流体节点侧再
投到 `e_i`，否则 `e_i·u_w≡0`，流场永远不动（本轮的第一次 Couette 就是这么"收敛"到
全零的，而且残差判据还判它收敛了——这就是为什么探针必须有 `amplitude_last < 0.5 *
amplitude_first`、`total x-momentum > 0` 这类**反冻结**断言）。

第三：收敛判据不能只看"残差小"。Couette 初始全零场在 200 步内残差也是零，于是
"收敛"到零流。现 `_run()` 增加了 `min_steps`，测试另有反冻结断言。

## 4. 质量门（CI）

```
python -m pytest tests/test_t3_matrix.py -q
python benchmarks/t3_matrix_run.py --quick     # 退出码 0 = 规格有效且无执行失败
```

测试覆盖：规格合法性（9 例、容差 ≤15%、repro_cmd 指向自身）、越界容差被拒、
未注册探针被拒、缺 gap 被拒、`solver_steps=0` 降级 MOCK、探针异常 → WITHHELD、
NOT_SUPPORTED 不计入通过率、Couette 机器精度、Poiseuille 抛物剖面、移动壁面动量
符号、`H=N−2` 约定、Taylor-Green 衰减率（含粗网格不允许藏错）。

`tests/conftest.py` 把 `src/` 放到 `sys.path` 最前：此前测试实际导入的是**安装副本**，
仓库源码的改动不会被测到（本轮新增模块一导入就暴露了这一点）。

## 5. 局限与未完成（如实）

- **方腔满分辨率未完成**：N=129、Re=100/400、判据 5% 的运行已递交集群（作业 cj3655），
  本单窗口内一直在排队（队列 50+）。本机被 120 s 墙钟限制，N=49 以上跑不完，
  因此该行保持 **WITHHELD**，不给数字。已知的中间观察：Re=100 在 N=65 上 3000 步内
  流场已建立（u 中心线负值区出现），但是否进入 5% 带未测。
- **静态液滴满配置未完成**：粗配置失败（38% 离散）说明 Shan-Chen 在 64²、1500 步、
  R=8–12 下既未平衡、界面也贴边界；满配置（96², R=12/16/20, 6000 步）同样排队中。
  同时指出：`mcmp.measure_laplace_pressure` 拿"σ 的经验估计式"当参照，这**不是**
  独立基准，故本矩阵改用 Laplace 定律的正确内容（Δp·R 常数）作为判据。
- **6 例 NOT_SUPPORTED 是能力缺口，不是工作量缺口**：分别缺 3D 格点、自由表面、
  浸入边界/力积分、亚格子与壁面函数、重力体力。这些都需要各自立单。
- `tolerance_basis` 里的"文献散布 ~4%"等属经验值，未逐条复核原文；判据结构（而非
  这些附注数字）才是机器校验对象。

## 6. 给下一张单子的线索

1. `cylinder-2d-re100`：在 `HalfwayBounceBack` 之上加阶梯/浸入固体 + 动量交换法算
   Cd，再补 St（升力谱峰）。这是 9 例里距现成能力最近的一项。
2. `cavity-2d-ghia`：把 cj3655 的满分辨率结果回填进矩阵（`--full`），确认 5% 带。
3. `static-droplet-laplace`：先做平衡性研究（Δp 随步数是否平），再谈线性度。
4. 别再把"σ 经验公式"当参照；矩阵已把它替换为 Laplace 定律本身。