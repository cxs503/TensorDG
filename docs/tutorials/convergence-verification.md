# TensorDG 收敛阶与 GCI 网格无关性验收

本文给出 TensorDG 的**系统化收敛阶验收**方法：如何用实测精度阶（observed order）对标理论阶，如何用 Richardson 外推与 GCI（Grid Convergence Index）判断网格无关性，以及哪些结论已被机器判据证实、哪些仍有局限。

配套工具：

- 计算库 [`tensordg/convergence.py`](../../tensordg/convergence.py)
- 验收驱动 [`benchmarks/convergence_study.py`](../../benchmarks/convergence_study.py)
- 回归测试 [`tests/test_verification.py`](../../tests/test_verification.py)
- 程序化生成的实测结果 [`docs/benchmarks/results/convergence/summary.md`](../benchmarks/results/convergence/summary.md)

> 数字纪律：本文**不手抄任何实测数字**。全部结果表、CSV、图均由
> `python -m benchmarks.convergence_study` 从当次运行直接写入
> `docs/benchmarks/results/convergence/`，正文只引用该目录。

## 1. 为什么需要独立的一层“验证面”

单个算例的误差小不等于实现正确：误差小可能来自参考值本身由被测实现生成、来自网格恰好适配误差相位、或来自时间/空间误差互相抵消。收敛阶验收把“实现是否与离散理论一致”变成可机器判决的问题：

1. 用**至少 3 个**（推荐 4 个）网格序列测 L2 误差；
2. 用最小二乘拟合 log(e) 对 log(h) 的斜率，得到实测阶 `p_obs`；
3. 判 `|p_obs − p_theory| ≤ 0.2`（本仓库阈值）；
4. 用三网格 Richardson 外推 + GCI 判断是否进入渐近区，并把 GCI 与解析解的真实误差直接比较。

关键纪律：**参考值只能来自解析解或 Richardson/GCI 外推本身**，绝不能由被测实现生成。

## 2. 理论阶对照表

| 问题 | 方法 | 多项式次数 | L2 空间阶（理论） | 时间方向 |
|---|---|---|---:|---|
| 2D Poisson 制造解 | SIPG（penalty `20·(p+1)²/h`） | P1 / P2 / P3 | 2 / 3 / 4 | 稳态 |
| 1D 周期线性对流 | 模态 Legendre DG + upwind + RK4 | P1 / P2 | 2 / 3 | RK4（非定常，CFL 固定） |
| 1D 热方程 | SIPG + Crank–Nicolson | P1 / P2 | 2 / 3 | CN 二阶（需固定 dt 才能测空间阶） |

时间方向另有独立验收：固定网格、只加密 dt，用域平均量测 Crank–Nicolson 二阶。

## 3. 工具链与判据

`tensordg/convergence.py`（纯 NumPy，不依赖求解器内部）：

| 函数 | 作用 |
|---|---|
| `pairwise_orders(n, e)` | 相邻两级网格的实测阶 |
| `observed_order(n, e)` | 全序列最小二乘实测阶 |
| `richardson_extrapolate(f1, f2, r, p)` | Richardson 外推 |
| `gci_study(fine, medium, coarse, r, Fs)` | Roache 阶估计 + GCI + 渐近比 |
| `convergence_table(...)` | 生成 JSON 化收敛表与判据 |
| `p_decay_diagnostics(p, e)` | p 加密衰减率、单调性、超线性判据 |
| `modal_point_value_1d/2d`、`modal_mean_1d/2d`、`modal_l2_norm_1d` | 从模态系数取标量 QoI（点值/均值/L2 范数） |

判据（`benchmarks/convergence_study.py` 内置，逐条输出 PASS / FAIL / NOT_APPLICABLE）：

1. `|p_obs − p_theory| ≤ 0.2`，并额外检查每个相邻对的阶偏差；
2. `GCI(fine) ≥ 真误差`，其中真误差由**解析解**给出；
3. 渐近比 `GCI_medium / (r^p · GCI_fine) ∈ [0.8, 1.25]`；
4. p 加密：误差逐级下降且下降因子 > `p_fine + 1`（超线性，接近谱收敛）。

GCI 采用 Roache (1994) / Celik et al. (2008) 定义，安全因子 `Fs = 1.25`（观测阶本身有不确定性时）：

```
p = |ln|ε32/ε21| + ln((r21^p − s)/(r32^p − s))| / ln(r21),   s = sign(ε32/ε21)
f* = f1 − ε21 / (r21^p − 1)
GCI_fine = Fs · |ε21 / f1| / (r21^p − 1) × 100%
```

## 4. 运行

```bash
python -m pip install -e ".[dev]"

# 完整 h/p/时间收敛研究 + 图 + 程序化 summary（约 40 s，CPU/float64）
python -m benchmarks.convergence_study

# 回归测试（并入套件）
pytest -q tests/test_verification.py
```

输出：

| 文件 | 内容 |
|---|---|
| `results/convergence/metrics.json` | 全部研究、判据、PASS/FAIL 状态的机器可读记录 |
| `results/convergence/summary.md` | 由 metrics 渲染的 Markdown 表（零手抄） |
| `results/convergence/h_studies.csv` | 逐网格误差与相邻阶 |
| `results/convergence/h_convergence.png` | log–log 误差 vs h，附理论斜率参考线 |
| `results/convergence/order_vs_theory.png` | 实测阶 vs 理论阶柱状图 |
| `results/convergence/gci_check.png` | GCI 估计 vs 解析真误差（对数纵轴） |
| `results/convergence/p_convergence.png` | 固定网格下 p 加密的半对数误差衰减 |

## 5. 结论（三态）

**已证实（机器判据 + 解析参考）**

- SIPG Poisson 的实测 L2 阶在 P1/P2/P3 上与理论 `p+1` 一致（偏差远小于 0.2），三网格序列处于渐近区（渐近比≈1）；
- 1D 周期对流 P1/P2 的实测 L2 阶与理论 2/3 一致；
- 1D 热方程在**固定 dt**（`time_steps=1000`）下 P1/P2 空间阶与理论 2/3 一致，且单独加密 dt 时 Crank–Nicolson 呈现二阶时间收敛；
- matrix-free PCG（Jacobi 预分解）后端的误差与 dense 直接解在 1e-9 相对量级内一致，实测阶不变，CG 残差低于 1e-9；
- p 加密在光滑解上呈超线性衰减（每步 p 误差下降数十倍），与 DG 谱收敛一致；
- GCI（Fs=1.25）对 P1/P2 算例给出的估计与解析真误差之比约为 1.2–1.3，即**保守但不过度保守地包住真误差**。

**推测**

- Poisson 点值 QoI 的实测阶明显高于 `p+1`（P1≈2、P2≈4、P3≈6，即近似 `2p`），提示 DG 在光滑解上的点值超收敛；本仓库仅把它用于 GCI 流程演示，未将其作为理论阶判据。
- P2 SIPG 在最粗网格对（2→4）上阶略低于 3（LSQ 拟合后仍在阈值内），推测是粗网格下惩罚项与边界面的渐近不足所致。

**未验证 / 明确局限**

- 只覆盖 CPU + `float64`、均匀 Cartesian 网格；GPU、`float32`、非结构网格未验证；
- 只覆盖 p ≤ 3 与二维以内；Euler 激波算例（Sod/Lax/涡）不做收敛阶断言——间断解无经典收敛阶，其验收仍走精确 Riemann 参考解的误差门（见其他 benchmark 文档）；
- GCI 只估计“被加密参数”对应的离散误差分量。热方程固定 dt 时，crank–Nicolson 时间误差构成地板，h 序列的 GCI 会**低估总误差**（实测：P2 均值 QoI 的 GCI 约为真误差的 3%）。该条在驱动里被明确标为 `NOT_APPLICABLE` 并附原因，而不是判 PASS；
- 测试通过 ≠ 物理正确：收敛阶只证明实现与离散理论一致，不理解物理的算例仍需与实验/文献比对。

## 6. 常见坑

1. **两点估阶不可靠**：单对网格的阶容易被非渐近效应污染（如 P2 SIPG 的 2.90），至少三网格 + 最小二乘。
2. **测空间阶必须压住时间误差**：`solve_heat_1d` 默认 `dt ≈ 0.02 h`，P2 会出现 2 阶（时间主导）而非 3 阶；实测阶前应先显式固定 `time_steps`。
3. **QoI 选择影响结论**：域平均量往往超收敛，点值超收敛更强；判断“是否达到理论阶”时必须用 L2 误差范数，QoI 只用于 GCI 流程。
4. **旋回/振荡序列不要硬套 GCI**：`gci_study` 在无法解出正阶时返回 `None` 并置 `in_asymptotic_range=False`，不要把它当成 0 误差。

## 7. 参考

- Roache, P.J. (1994) *Perspective: a method for uniform reporting of grid refinement studies*, J. Fluids Eng. 116(3), 405–413.
- Celik, I.B. et al. (2008) *Procedure for estimation and reporting of uncertainty due to discretization in CFD applications*, J. Fluids Eng. 130(7), 078001.
- Cockburn, B., Shu, C.-W. (1989) *TVB Runge–Kutta local projection DG methods*, Math. Comp. 52, 411–435.