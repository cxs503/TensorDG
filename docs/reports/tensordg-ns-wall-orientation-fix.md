# TensorDG 压缩 Navier–Stokes 壁面通道：Dirichlet 罚项定向缺陷的定位与修复

> 上游重锚说明（链上仲裁）：本单的根因发现与修法已由链上 m2782（`outward_sign`）
> 先行合入——机制相同（min 面罚项沿外法向取号），关键字名不同（本单原名
> `orientation`）。本重锚版把调用点改到 `outward_sign`，保留本单全部独立证据
> （FD Jacobian 谱守卫、双侧 3% 门、verify 脚本与再生成的 JSON/PNG）。

> 上游：TensorDG@0ec9e1e（main）
> 前序报告：`docs/reports/tensordg-ns-dg-2d.md`（m2496，粘性通量）、
> `docs/reports/tensordg-ns-wall-verification.md`（m2589，壁面路径 honest-FAIL + 三条判据）
> 本单：把 m2589 §6 item 1「P0 修壁面瞬态稳定性」做完
> 证据文件：`validation/ns_dg_2d_wall/orientation_fix.json`（本报告所有数字由该 JSON 直读）
> 图：`docs/figures/ns_wall_orientation_fix.png`
> 复现：`PYTHONPATH=. python -m tools.verify_ns_wall_orientation`

---

## 1. 案例介绍

**物理背景**：二维平面槽道内可压缩黏性流（γ=1.4、单位气体常数、常黏度 μ）。
下壁静止、上壁以 U 突然起动（impulsive-start Couette），这是壁面 Dirichlet
（无滑移 + 等温）路径的最小可判别算例：解析稳态是

```
u(y) = U y/H,        T(y) = T_w + A y(1-y)/H^2,   A = mu U^2 / (2 kappa),   p = const
```

（Brinkman 抛物线，黏性耗散与热传导平衡），壁面通量解析可写：

| 量 | 解析值（μ=0.1, U=1, κ=0.492958, H=1） |
|---|---:|
| τ_xy（壁面剪应力） | μU/H = 0.100000 |
| 能量通量（两侧壁） | −κA = −μU²/2 = −0.050000 |

**几何与参数表**

| 项 | 值 |
|---|---|
| 计算域 | x∈[0,1] 周期；y∈[0,1] 壁面 |
| 单元数 | (nx, ny) = (2, 8) / (2,16) / (2,32) / (2,64) |
| 多项式阶 | P1（另测 P2） |
| μ / Pr / κ | 0.1 / 0.71 / 0.492957746 |
| 初值 | ρ=1, u=v=0, p=1（L2 投影） |
| 壁面 | y=0: (u,v)=(0,0), T=1；y=1: (u,v)=(1,0), T=1 |
| SIPG 罚系数 σ | 2(p+1)² = 8 |
| 时间推进 | SSP-RK3，显式，黏性步长上限 VISCOUS_STABILITY=0.1 |

**缺口**：m2589 记录了 honest-FAIL —— 脉冲起动 Couette 在 t≈0.03–0.08 触发
`nonphysical cell average`，且与时间步无关（减小 dt 只增加步数、不改变失稳物理时刻），
说明是**空间（半离散）算子不稳定**。该缺陷此前无修法。

---

## 2. 怎么做的

### 2.1 判据先于实验（四条，全部机器裁）

1. **半离散谱判据**：壁面通道算子在脉冲起动状态处的 Jacobian 的所有特征值必须
   满足 Re λ ≤ 0（正实部 = 指数增长 = 失稳）。这是"空间不稳定"的直接判据，
   比时间推进快 3 个数量级（64 DOF，2 s）。
2. **壁面通量判据**：两侧壁的能量通量与解析 −κA 的相对误差 <3%，且随 h 单调收敛。
3. **对称性判据**：两条壁面（max 面 / min 面）的通量误差必须同量级 ——
   只有一个面错，是"定向"缺陷的指纹。
4. **瞬态判据**（沿用 m2589 的 strict-xfail）：脉冲起动 Couette 必须推进到 t=0.1 且保持物理。

### 2.2 定位方法

1. 先用**时间推进**确认症状（t≈0.08 失稳，与 dt 无关）。
2. 再对 `tensordg.ns_dg_2d._rhs_ns` 做**有限差分 Jacobian**（eps=1e-6，64 DOF 全量），
   求特征值：负实部占多数，但出现 13 个正实部、max Re = +86.7。
3. 取正特征值对应的**特征向量**（按 DOF 展成 (nx,ny,k,l,4)）→ **全部集中在 y=0（下壁）
   单元行，且以 E（总能）与 ρv（法向动量）为主**。上壁行幅值只有下壁的 1/100。
4. 于是检查 `_boundary_flux` 在 min 面 / max 面是否用了同一套符号 → 见 §2.3。

### 2.3 缺陷与修复

`_boundary_flux` 的 Dirichlet 罚项（BR1/SIPG 的梯度修正）写成

```python
normal_q = ∂q/∂axis * normal_scale - (penalty/h) * (q_inner - q_wall)
```

**下面两项在 min 面与 max 面上共用一个符号**：

- `∂q/∂axis`：正确 —— 内点梯度在 `+axis` 约定下对两个面都成立（弱形式的 `signs`
  因子负责把 min 面翻向）。
- 罚项：**错误**。罚项模拟的是"穿过壁面流出单元"的通量，必须沿**外法向**；
  max 面外法向 = +axis，min 面外法向 = −axis。

后果（min 面）：比壁面更热的单元反而**被加热**（热通量符号反了），比壁面更快的流体
反而**被加速** —— 即罚项在 min 面是**反耗散**的。这正是"底壁能量通量误差 3.17%
（上壁仅 0.56%）"和"失稳模式只在下壁行"的来源。

修复（最小改动）：给 `_boundary_flux` 增加 `orientation` 参数，max 面取 +1、min 面取 −1，
只作用于罚项；调用点 `_axis_fluxes_ns` 的 `left_boundary` 传 `orientation=-1.0`。

```diff
-        normal_u = side["u_n"] * normal_scale - (penalty / h) * (inner_u - face_u)
+        sign = orientation
+        normal_u = side["u_n"] * normal_scale - sign * (penalty / h) * (inner_u - face_u)
...
-        left_boundary = _boundary_flux(left, wall_left, axis, h, mu, kappa, penalty,
-                                       normal_scale, tangential_scale)
+        left_boundary = _boundary_flux(left, wall_left, axis, h, mu, kappa, penalty,
+                                       normal_scale, tangential_scale,
+                                       orientation=-1.0)
```

物理自检（min 面，+axis 约定）：单元比壁面热（T⁻>T_D）时修正后的 ∂T/∂y 应为
`∂T/∂y + (σ/h)(T⁻−T_D)`，使 −y 方向的热通量（进入壁面）**增大**；
比壁面快（u⁻>u_D）时修正后的 τ 应给出 −x 方向的阻力。两者都是 `orientation=−1` 的结果。

---

## 3. 计算结果（`orientation_fix.json` 直读）

### 3.1 谱（判据 1）：正实部特征值清零

| 变体 | DOF | max Re λ | 正实部个数 |
|---|---:|---:|---:|
| 修复前（两面同号） | 64 | **+86.3305** | **6** |
| 修复后（外法向定向） | 64 | **+1.24e−14** | **0** |

谱半径 \|λ\|max 基本不变（378.0 → 381.7），说明修复没有改变算子的刚性量级，
只把反耗散模式消掉；显式步长的稳定域因此不受影响。

在 (2,4) P1 上把网格加密/升阶后重复（见 §5），修复后正实部始终为 0。

### 3.2 壁面通量（判据 2、3）

| ny | 底壁能量通量 | 相对误差 | 顶壁能量通量 | 相对误差 | 剪应力相对误差 |
|---:|---:|---:|---:|---:|---:|
| 8 | −0.048133 | 3.734 % | −0.052201 | 4.402 % | 0.334 % |
| 16 | −0.049080 | 1.841 % | −0.051109 | 2.219 % | 0.189 % |
| 32 | −0.049543 | 0.914 % | −0.050557 | 1.114 % | 0.100 % |
| 64 | −0.049772 | 0.455 % | −0.050279 | 0.558 % | 0.051 % |
| 观测阶 | | **1.52** | | **1.49** | — |

- 解析基准：−κA = −0.05（两侧壁相同：顶壁的壁面功 u·τ = μU²/H 与 Fourier 项 −μU²/(2H)
  相消，净值同为 −μU²/2）。
- **两侧壁误差已同量级**（4.40 % vs 3.73 %），不再有 m2589 的 0.56 % vs 3.17 % 不对称。
- 64 格底壁从 **3.17 % → 0.455 %**（同一量、同一网格，前序报告的缺陷值）。

### 3.3 解析稳态残差（m2589 判据 1，仍未过）

| ny | max\|R(U_exact)\| |
|---:|---:|
| 8 | 0.04518 |
| 16 | 0.04864 |
| 32 | 0.05111 |
| 64 | 0.05237 |
| 观测阶 | −0.11 |

比 m2589 的 0.3045（32 格）小了 6 倍，但**仍不随 h 衰减**：max 出现在壁面行，
机制见 §5 局限 (1)。该项 **honest-FAIL 保留**（对应 strict-xfail 测试未退休）。

### 3.4 瞬态（判据 4）

| 项 | 值 |
|---|---|
| 算例 | `solve_couette_dg_2d(ncells=(2,8), degree=1, μ=0.1, T=0.1)` |
| 结果 | **physical** |
| 步数 / 末步 dt | 203 / 1.69e−4 |
| 状态检查 | ρ>0、E>0、\|u\|max=0.919（趋向上壁速度 1）、T∈[1.000, 1.032]（Brinkman 加热，物理） |

### 3.5 图

`docs/figures/ns_wall_orientation_fix.png`
左：修复前/后半离散 Jacobian 的 max Re λ（+86.33 → 机器零）；
右：两侧壁热通量相对误差 vs ny（log-log，含 O(h) 参考线与 3% 门）。

---

## 4. 误差判定（逐项，显式回答「是否 <3%」）

| 关键量 | 实测 | 基准 | 相对误差 | 是否 <3% |
|---|---:|---|---:|---|
| max Re λ（修复前，2×4 P1） | +86.3305 | 0（须 ≤0） | — | ❌（失稳） |
| max Re λ（修复后，2×4 P1） | +1.24e−14 | 0 | 1.2e−14（机器零） | ✅ |
| 底壁能量通量（64 格） | −0.049772 | −0.05 | 0.455 % | ✅ |
| 顶壁能量通量（64 格） | −0.050279 | −0.05 | 0.558 % | ✅ |
| 底壁能量通量（32 格） | −0.049543 | −0.05 | 0.914 % | ✅ |
| 顶壁能量通量（32 格） | −0.050557 | −0.05 | 1.114 % | ✅ |
| 底壁能量通量（8 格） | −0.048133 | −0.05 | 3.734 % | ❌（需 ≥16 格） |
| 壁面剪应力（32 格） | 0.100100 | 0.100 | 0.100 % | ✅ |
| 脉冲 Couette 推进到 t=0.1 | physical（203 步） | 应保持物理 | 不可比（布尔判据） | ✅ |
| 解析稳态残差（32 格） | 0.0511 | 0（一致离散应为 O(h)） | — | ❌ **honest-FAIL 保留** |
| 壁面通量观测阶 | 1.52 / 1.49 | 设计阶 2 | — | ❌（仍非二阶） |

**显式回答**：
- 「瞬态可用」→ **是**（m2589 的 P0 缺陷闭合，strict-xfail 已退休为普通测试）；
- 「壁面通量 <3%」→ **是，但只在 ny≥16 时**（32 格起两侧壁均 <1.2 %；8 格不达标）；
- 「壁面通量二阶收敛」→ **否**（1.5 阶），对应 xfail 测试保留并改写理由；
- 「解析稳态残差 O(h)」→ **否**，honest-FAIL 保留，原因见 §5(1)。

---

## 5. 注意点（坑 / 局限 / 适用边界）

1. **局限（最重要，未完）**：墙法向梯度仍是**单侧单元梯度**，通量误差 O(h) →
   强形式残差里被 1/h 放大成 **O(1)**，所以"解析稳态残差 vs h"不收敛、壁面通量只有
   1.5 阶。要过 m2589 的判据 1/2 必须换壁面梯度（BR1 lifting 的正规形式、跨单元
   单侧二阶重构、或守恒律反算壁面通量）。本轮没做，不在本报告声明范围内。
2. **坑：时间推进不是定位工具**。失稳在 t≈0.08 才可见，且 dt 缩 4× 只改步数；
   换成 Jacobian 谱后 2 s 就给出 +86.7 与"下壁行 + E/ρv"的定位，且可当回归门用
   （`test_wall_penalty_has_no_unstable_mode`）。
3. **坑：不对称是定向缺陷的指纹**。前序报告把"底壁 3.17 % / 顶壁 0.56 %"当成"壁面只有
   一阶"的整体结论，其实两个面误差差 5.6 倍本身就是符号缺陷的证据 —— 判据里应显式加
   "两面同量级"这一条。
4. **坑：能量通量的解析符号容易搞错**。能量分量是 `-(u·τ + κ ∂T/∂y)`（+axis 约定），
   顶壁 u·τ = μU²/H 与 κ∂T/∂y = −μU²/(2H) 相消，净值 = −μU²/2 = 底壁同值。本单第一版
   把顶壁基准写成 +0.05，得到了 200 % 的假误差（已修正；测试与工具同步）。
5. **坑：`_boundary_flux` 是"求值"接口，也是"求解"接口**。测试/工具直接调用它取壁面通量时
   必须传同一个 `orientation`，否则会拿修复前的约定去比修复后的解（`tools/diag_ns_wall.py`
   与 `tests/test_ns_dg_2d_wall.py` 的底壁调用已改为 `orientation=-1.0`）。
6. **适用边界**：结论只覆盖 `ns_dg_2d` 的强形式 BR1/SIPG 壁面路径（Cartesian、常 μ、
   P1/P2）。周期/无壁结论（m2496 的 2.02–2.07 阶）不受影响，已复跑通过。
7. **未做**：瞬态 vs 解析瞬态解（Fourier 级数
   `u = U[y/H − (2/π)Σ(−1)^{n+1} sin(nπy/H)e^{−νn²π²t}/n]`）的场量误差未给 ——
   在 load15≈126 的机器上把边界层解析到能判 3% 需要的步数约 10⁴–10⁵，本单只做轻量核算。
   这是下一次自找活最值得做的一项（判据已可一键写）。

---

## 6. 测试与产物

| 文件 | 作用 |
|---|---|
| `tensordg/ns_dg_2d.py` | 修复：`_boundary_flux(..., orientation)` + min 面传 −1 |
| `tests/test_ns_dg_2d_wall.py` | 退休瞬态 xfail → `test_impulsively_started_couette_stays_physical`；新增 `test_wall_penalty_has_no_unstable_mode`（谱门）与 `test_wall_heat_flux_is_below_three_percent_on_both_walls`（3 % 双侧门）；两条残余精度 xfail 改写理由 |
| `tools/verify_ns_wall_orientation.py` | 谱 + 通量 + 残差 + 瞬态，写 JSON 与图 |
| `validation/ns_dg_2d_wall/orientation_fix.json` | 全部数字（本报告零手抄） |
| `docs/figures/ns_wall_orientation_fix.png` | 结果图 |

```bash
PYTHONPATH=. python -m pytest -q tests/test_ns_dg_2d_wall.py tests/test_ns_dg_2d.py
PYTHONPATH=. python -m tools.verify_ns_wall_orientation
```
\ No newline at end of file
