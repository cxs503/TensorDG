# CASE · REPRO-DG-CKS1998 · 曲边 DG 反射壁线性不稳定：定位、修复与圆柱算例的入口

> 锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199–224, Part V）
> 单号 m21767（自找活·platform 席）　分支 `agent/m21767`
> 对象 `tensordg/euler_curved.py` + `tests/test_euler_curved_boundaries.py`
> 前序 `docs/AUDIT-m21473-...`（不收敛，根因未定位）→ `docs/CASE-DG-CKS1998-CURVED-OPERATOR-SIGN-FIX.md`
> （内部面定向修复）→ **本单**（壁面定向缺陷 = 模块自认「反射壁线性不稳定，圆柱算例不可复现」的根因）

## ① 案例介绍（物理背景 / 几何 / 参数）

CKS1998 的曲边腿要能算 §「绕圆柱」算例，前提是反射壁在**稳态附近线性稳定**。
`euler_curved.py` 的 docstring 在 HEAD 上诚实记录：均匀滑移流（精确稳态，残差 1e-14）
的显式 SSP-RK3 步进会放大舍入误差，约 700 步后炸掉，并据此**拒绝**声称任何圆柱结果。
本单处理的就是这条「自认欠账」：定位根因 → 最小修复 → 用线性谱 + 长步进双证据闭合，
再把圆柱算例的入口（`benchmarks/cylinder_euler_steady.py`）建起来。

几何：`channel(6×4, p=1)` 直通道（周期 x + 上下壁面），基态 `(ρ,u,v,p)=(1,0.3,0,1)`；
另有周期/出流/远场边界组合做对照。剖面：`cylinder_annulus_mesh`（极坐标 O 型网格，
内圆 = 壁面，外圆 = 远场）。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **线性谱定位**：对 `rhs` 在均匀滑移基态处做有限差分 Jacobian（6×4, p=1, 384 dof），
   求 `max Re eig`（脚本 `scripts/curved_wall_stability_scan.py`）。
   upwind DG 的谱应是「虚轴 − 耗散」，正实部即为**反耗散**边界项的铁证。
2. **边界族隔离**：同一网格/状态只换边界种类（周期 / 壁面 / 出流 / 远场）。
3. **通量变体扫描**（`scripts/curved_wall_flux_variants.py`，monkeypatch 只换壁面通量）：
   镜像 HLLC（HEAD）/ LLF / 纯 `(0,pn,0)` / 全反射 / 远场泄漏。
4. **定向协变性探针**：直接对 `wall_normal_flux` 检查 `F(n) + F(−n) = 0` 是否成立。
5. **修复与验证**：最小改动 `_boundary_normal_flux`；再用线性谱 + 3000 步长步进
   （+ 反向对照）双证据，回归门替换原「诚实负面」测试。

## ③ 计算结果（全部脚本写出，零手抄）

### 3.1 边界族隔离（修复前，6×4, p=1，`max Re eig`）

| 边界组合 | max Re（修复前） | max Re（修复后） | 稳态残差 |
|---|---:|---:|---:|
| 全周期（无壁面） | +1.14e−07 | +1.14e−07 | 5.5e−15 |
| 周期 x + 壁面 y | **+9.194** | **+9.84e−08** | 5.5e−15 |
| 出流 x + 壁面 y | **+9.079** | +2.59e−04 | 5.5e−15 |
| 远场 x + 壁面 y | **+9.113** | −5.68e−02 | 5.5e−15 |
| 四边全壁面（非稳态，仅诊断） | +1.37e+01 | +3.74e−04 | 4.79e+00 |

结论：正实部**只**由壁面带来（周期边界干净），且修复后全部消失/转负。
注意「稳态残差」两栏相同（5.5e−15）——即**算子一致**，缺陷纯粹在稳定性。

### 3.2 壁面通量变体（6×4, p=1, 周期 x + 壁面 y）

| 变体 | max Re |
|---|---:|
| V0 镜像 HLLC（HEAD） | +9.194 |
| V1 LLF/镜像 | +9.194 |
| V2 纯 `(0,pn,0)`（无耗散） | **+9.93e−08** |
| V3 全反射 ghost + HLLC | +9.194 |
| V4 远场泄漏 ghost | +4.477 |

只有**无耗散**的 V2 稳定 ⇒ 出问题的不是「耗散大小」而是**耗散的符号/定向**。

### 3.3 根因（定向协变性探针）

`HLLC` 只在「法向取反 **且** 左右状态互换」下协变。`_boundary_normal_flux` 对低面
（`axis-lo`）传的是**参考方向法向**（指向单元内部）、且始终把内部态放在 `left`：

```
u·n = +0.01 (n 为外法向)   F(n) + F(-n) = [+1.3e-16, +1.83e-02, +1.54e-02, +5.3e-16]
u·n = -0.01                F(n) + F(-n) = [+1.3e-16, -1.83e-02, -1.54e-02, -5.3e-16]
u·n =  0                    F(n) + F(-n) = [0, 0, 0, 0]
```

即镜像 HLLC 壁面通量**不是奇函数**（非协变部分 ∝ ρc·u_n，量级 2.4 % of |F| at u_n=1 % a）。
强形式面项要求 `F̂·n_out`；低面用 `F̂(n_ref = −n_out)` 再整体取负，非协变部分因此**反向
进入**——耗散变成反耗散。这正好解释 §3.2：去掉耗散项（V2）反而稳定。

### 3.4 修复（最小改动）

```python
if kind == "wall":
    if side < 0:      # 参考法向指向单元内部 → 用外法向重述 Riemann 问题
        return -wall_normal_flux(inner, -unit_x, -unit_y, self.gamma)
    return wall_normal_flux(inner, unit_x, unit_y, self.gamma)
```

依据协变性 `F̂(n_out; U_in, U_ghost) = −F̂(n_ref; U_ghost, U_in)`，取负后与调用方的
「参考定向 + 低面取负」约定一致。收敛态仍是 `(0,pn,0)`，故稳态残差不变（§3.1）。

### 3.5 长步进（时间域）证据

`tests/test_euler_curved_boundaries.py::test_wall_boundary_march_is_stable_and_uncovariant_flux_is_not`：
均匀滑移精确稳态 + 3000 步 SSP-RK3（`dt = stable_dt`），要求 `max|U−U0| < 1e−8`
（修复前 ~700 步即 >1e3 炸掉）；**反向对照**把低面壁面通量翻回修复前的定向，必须炸掉，
否则门空转。`tests/test_euler_curved_boundaries.py` 12 项全绿，`tests/test_euler_curved.py` 11 项全绿。

## ④ 结果图

本单是**边界算子缺陷修复 + 线性谱证据**，核心「图」是 §3.1/3.2 的谱表（`max Re eig`），
数值表本身即判定；无新场量云图。圆柱算例的 Cp(θ) 图由
`benchmarks/cylinder_euler_steady.py` 输出（`validation/cylinder_steady/metrics_*.json`，
含 wall-quadrature 的 θ/Cp 数组），但本单收尾时该作业仍在队列（cj3709），故**不贴图**。

## ⑤ 注意点 / 局限（honest）

1. **圆柱算例仍未给出定量结论**：入口已建（`benchmarks/cylinder_euler_steady.py`），
   亚音速 M=0.38 稳态跑的作业 `cj3709` 在本单收尾时仍排队（前置 65 位），
   因此「Cp_stag vs 等熵总压 1.0427、前后对称性、d'Alembert 阻力→0」三项**均为未跑**，
   不写任何数字，判 **WITHHELD**。
2. **马赫 3 超音速不可算**：模块无曲边限制器，P1 在弓形激波处振荡出负压
   （实测 15×24 P1 M=3 在 100 步内 `blew_up=true`）。这与模块 docstring 的
   「shock limiters on curved elements 未实现」一致，**不是**本单修复能解决的。
3. **出流边界仍是「无为」边界**（零梯度拷贝，无特征辐射条件）：修复后其 `max Re = +2.6e−04`
   仍为正（量级极小，属既有已知局限），远场边界修复后为负（−5.7e−02，耗散）。
4. **V4 远场泄漏对照 +4.477**：说明「壁面漏流」也不会让它稳定，进一步支持定向而非 ghost 构造是根因。
5. 本单未改动任何容差/判据；原「诚实负面」测试被**升级**为「稳定门 + 反向对照」，未删除任何证据。

## ⑥ 误差判定（显式回答「是否 < 3 %」）

| 关键量 | 基准（解析） | 实测 | 相对偏差 | 是否 < 3 % |
|---|---|---|---:|---|
| 均匀滑移稳态残差 | 0 | 5.54e−15 | — | 是（round-off） |
| 壁面边界 max Re | ≤ 0（upwind DG 应无正实部） | +9.84e−08（修复后） | — | 是（修复前 +9.194 为 fail） |
| 3000 步稳态漂移 | 0 | < 1e−8（门限） | — | 是 |
| 圆柱 Cp_stag | 1.0427（M=0.38 等熵总压） | 未跑（cj3709 排队） | — | **无法判定（WITHHELD）** |
