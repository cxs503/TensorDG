# 脉冲启动 Couette 壁面瞬态：对 Stokes 级数解析解的定量验证（m3661）

> 上游：TensorDG@c1c2420（main）· 分支 `agent/m3661` · 2026-11-04
> 复核对象：`tensordg/ns_dg_2d.py` 的 Dirichlet（壁面）路径在**脉冲启动**瞬态下的精度
> 前序账：`docs/reports/tensordg-ns-wall-verification.md`（m2589，缺陷定位）、
> `docs/reports/tensordg-ns-wall-scope-audit.md`（m2786，把瞬态精度列为唯一 honest FAIL）、
> `docs/reports/tensordg-ns-wall-orientation-fix.md`（m2876，罚项朝向修复）
> 本文全部数字由 `benchmarks/verify_ns_impulse_couette.py` 程序化写入
> `validation/ns_dg_2d_impulse/impulse.json`，再由该 JSON 生成下表；零手抄。

---

## 1 案例介绍（物理背景 / 几何 / 参数）

上下两块等温无滑移平板之间的流体，**下板静止、上板在 t = 0 瞬间以 U 起动**（脉冲启动，
impulsively started Couette）。这是壁面边界层/起动瞬态的最小解析算例，也是 m2496–m2876
几轮报告的同一个「P0 最小复现」：它曾以 `FloatingPointError` 在 t ≈ 0.033 崩掉，修好后
只被验证到「能推进到 t = 0.05–0.1」，**从未与任何参考解比较过精度**——这就是本文补的欠账。

| 参数 | 值 |
|---|---|
| 计算域 | [0,1] × [0,1]，x 周期（1 个单元即够，解与 x 无关）、y 两道 Dirichlet 壁 |
| 网格 | 1 × ny 个 Cartesian 单元，ny ∈ {8, 16}；P1 模态 Legendre DG |
| 壁面条件 | ymin: u = 0；ymax: u = U，v = 0，两壁 T = T_w（无穿透、等温） |
| 粘性/热 | BR1 强形式 + SIPG，σ = 2(p+1)²，μ = 0.05 常数，Pr = 0.71，κ = μc_p/Pr = 0.24648 |
| 气体 | 量纲一，R = 1（T = p/ρ），γ = 1.4，c_v = 2.5，c_p = 3.5 |
| 流动参数 | U = 0.3，p = 1，T_w = 1 ⇒ ρ_w = 1，ν = μ/ρ = 0.05 |
| 时间推进 | SSP-RK3，声学 CFL 0.3 + 粘性 CFL 1.0，推进到 t = 0.2 |
| 参考解 | 沟道内 Stokes 第一问题的级数解（见 §2.1） |

**为什么 U = 0.3 而不是 1**：级数解是**不可压**平行流的精确解；压缩求解器与它的差别只有
粘性耗散使 T（进而 ρ = p/T、ν）缓慢漂移，量级 μU²/(8κ) = 2.28e-3。U = 1 时该漂移升到 2.5 %
（且 Mach ≈ 0.85，声学瞬态不可忽略），会把参考解的适用性搅进来；U = 0.3（Mach ≈ 0.25）
把这项干扰压到 0.1 % 量级，测到的误差就可以归给 DG 壁面算子本身。

### 1.1 参考解（本单新增 `ns.impulsive_couette_exact`）

平行流下 u = u(y,t)、v = 0、p = 常数，对流项 u∂ₓu + v∂_yu 恒为零，动量方程退化为热方程，
故 Stokes 级数在沟道内是精确解：

    u(y,t) = U s + Σ_{n≥1} 2U(-1)^n/(nπ) · sin(nπ s) · exp(−n²π²νt/H²),   s = y/H

满足 u(0,t) = 0、u(H,t) = U（t > 0）、u(y,0) = 0（y < H）。级数在 exp(−n²π²νt/H²) < 1e-14
处截断（本例 t = 0.2 时约 40 项）。

## 2 怎么做的（方法 / 工具链 / 判据）

1. **参考解不是自证的**：先用一台与被测代码完全无关的隐式有限差分解
   （u_t = νu_yy，401 节点、dt = 2e-4、1000 步）解同一个初边值问题，与级数比较；
   相对 L2 差 **5.40e-4**（地板来自差分格式自身的 O(dt) 误差）。该比较已固化为回归测试。
2. **DG 侧误差**：把最终时刻的模态系数在每单元的 Gauss–Legendre 节点上重构出 u，
   与级数在**同一批节点**上比较，按 GL 权重求相对 L2 范数（y 方向 Jacobian 在分子分母抵消）。
3. **判据（先于实验、可机器裁）**：
   - *profile*：每个算例的相对 L2(u) < 3 %；
   - *order*：加密后误差单调下降且实测斜率 > 0.5（初始数据在壁面有跳跃，经典阶会被削，
     故判据不写死为 p+1）；
   - *bounds*：u ∈ [−0.02U, 1.001U] 且沿 y 单调（不依赖参考解的容许性判据）；
   - *heating*：max|T − T_w| ≤ 3 × μU²/(8κ)（瞬态比定常 Brinkman 抛物线更冷）。
4. **工具链**：`benchmarks/verify_ns_impulse_couette.py`（`--only degree:ny` 单算例模式
   → `validation/ns_dg_2d_impulse/cases/case_p*_ny*.json`；`--merge` 汇总
   `impulse.json` + 图），判据固化在 `tests/test_ns_impulse_couette.py`（3 passed / 18.7 s）。

## 3 计算结果（直读 `impulse.json`）

| P | ny | h | 步数 | dt | 相对 L2(u) | max|u_num−u_exact| | u_min | u_max | max|T−T_w| |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.125 | 203 | 4.212e-05 | **2.9690 %** | 5.391e-03 | −1.181e-03 | 0.27585 | 2.792e-03 |
| 1 | 16 | 0.0625 | 809 | 7.524e-05 | **0.7544 %** | 1.727e-03 | −4.389e-06 | 0.28820 | 2.744e-03 |

- 加密斜率（8 → 16 格）：**1.976**（P1 的设计阶 2）。壁面 u_min 从 −1.18e-3 收敛到
  −4.4e-6，过冲随 h 消失，不是持续振荡。
- 质量（Σ 模态 0 模态和，x 周期均匀）：8.0000726（ny=8）/ 16.000405（ny=16），相对漂移
  **9.1e-6 / 2.5e-5**——脉冲启动下质量守恒到积分精度。
- 粘性加热：两网格上 max|T−T_w| = 2.79e-3 / 2.74e-3，与解析耗散标度 2.28e-3 同量级
  （比值 1.22 / 1.20 ≤ 3），确认「T 漂移 ≤ 0.3 % ⇒ ν ≈ 常数」的前提成立。
- 图：`docs/figures/ns_dg_2d_impulse_couette.png`
  （(a) t = 0.2 的 u(y) 剖面 vs 级数；(b) 误差–h 双对数与 3 % 门（虚线为斜率 1 参考）；
  (c) 各算例距 3 % 门的柱状图）。

## 4 误差判定（逐项，显式回答是否 <3 %）

| 关键量 | 基准 | 实测 | 是否 <3 % |
|---|---|---:|---|
| 相对 L2(u)，P1 / 8 格，t = 0.2 | Stokes 级数 | 2.9690 % | ✅（余量仅 1 %，见 §5.1） |
| 相对 L2(u)，P1 / 16 格，t = 0.2 | Stokes 级数 | 0.7544 % | ✅ |
| max 点误差，P1 / 16 格 | Stokes 级数 | 1.727e-03 / U = 0.58 % | ✅ |
| 参考解自身：级数 vs 独立隐式差分 | 隐式 FD（401 节点） | 5.40e-4 | ✅ |
| 加密斜率 8→16 | P1 设计阶 2 | 1.976 | ✅（\|Δ\| = 0.024 < 0.2） |
| 粘性加热 max\|T−T_w\| | μU²/(8κ) 解析标度 | 1.22 × 标度 | ✅（≤ 3 × 标度） |
| P2/P3 的瞬态精度 | Stokes 级数 | **未测**（算力纪律，见 §5.2） | — 未声明达标 |

**显式回答**：「脉冲启动瞬态精度 <3 %」——**是**，在两个已测网格上（2.97 % / 0.75 %），
且加密斜率等于 P1 设计阶。**但** P2/P3 与更细网格未测，不能把结论外推到「所有阶都达标」。

## 5 注意点（坑 / 局限 / 适用范围）

1. **坑（判据边缘）**：P1 / 8 格的 2.969 % 距 3 % 门只剩 1 % 余量。它不是「刚好过」而是
   「网格太粗」：t = 0.2 时边界层厚 δ = √(νt) = 0.10，8 格时 h/δ = 1.25，壁面层只有不到
   一个单元。报告里必须把 8 格与 16 格并列，只引 16 格的 0.75 % 会掩盖这个分辨率依赖。
2. **局限（未测面）**：P2/P3 与 ny = 32 未跑——本机 load15 在 100–350 之间（nproc = 128），
   按容量纪律只起单线程小算例；P2 / 8 格估算需 ~5 min，ny = 32 需 ~4 min/阶。判据与工具
   已就位（`--only 2:8`、`--only 1:32` 各一条命令），后续在低负载窗口或集群补跑即可并入
   同一 `impulse.json`。
3. **局限（参考解的适用边界）**：级数解是不可压平行流精确解。本例用 U = 0.3 把压缩性
   修正压到 0.1 %（max|T−T_w|/T_w = 0.28 %，其中一部分还是真实耗散而非误差）。U → 1
   （Mach ≈ 0.85）时不可用本参考解下 3 % 判决。
4. **坑（级数在 t → 0 的抵消）**：级数系数在 t → 0 时按 (−1)^n 交替抵消到 13 位有效数字，
   即使双精度也会有 ~1.4e-3 的舍入噪声（测试里已把 t = 1e-8 的阈值定在 5e-3 并注明原因）。
   不要用级数在 t ≲ 1e-6 做小时间验证。
5. **前序 honest FAIL 的状态更新**：m2786 §6 保留的唯一 honest FAIL（「瞬态精度无法给出且
   不达标」）在**当前 main 上已不成立**——本单实测脉冲启动 P1 / 8 格 203 步到 t = 0.2、
   P2 / 8 格 1025 步到 t = 0.1 均正常推进且与级数在 3 % 内。致失稳通道由 m2782（罚项朝向）
   与 m2876（`outward_sign`）闭合；本文把「能跑」升级为「跑得对」。

## 6 复现

```bash
PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette --only 1:8     # 14 s
PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette --only 1:16    # 57 s
PYTHONPATH=. python -m benchmarks.verify_ns_impulse_couette --merge        # 汇总 + 图
PYTHONPATH=. python -m pytest -q tests/test_ns_impulse_couette.py          # 3 passed
```

判据汇总（`impulse.json.verdict`）：`profile_below_three_percent` /
`bounded_and_physical` / `monotone_in_y` / `heating_within_scale` /
`refinement_slope_above_half` 全部 true → `overall: PASS`。