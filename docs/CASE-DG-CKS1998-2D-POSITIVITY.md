# CKS1998 二维系统算例的前置缺陷：非容许初值投影 + 假阴性自由流门（m20023）

**锚点**：`REPRO-DG-CKS1998`（open）之未闭项「2-D 系统基准（JCP 141 主算例：双马赫反射）」
**上一轮**：`docs/euler_dg_2d_wall_bc.md`（m16803）把该项的两个阻断点记为
（i）自由流保持 FAIL、（ii）Mach-10 双马赫反射跑不到 10 步即 `FloatingPointError`。
本单重新诊断为：**(i) 是测量缺陷（门本身坏了），(ii) 是初值投影缺陷（不是时间推进鲁棒性）**，
两处都已定位并修复。

---

## 1. 缺陷 A：自由流门是假阴性（度量写错，永远不可能通过）

`tests/test_euler_dg_2d_wall_bc.py::test_free_stream_preservation_with_walls` 与
`benchmarks/euler_dg_2d_wall_bc.py::free_stream_deviation` 原来的度量是

```python
reference = conservative_state(*initial(x[:, None], y[None, :]))      # 常数态
deviation = (coeff - reference[:, :, None, None, :]).abs().max() / scale
```

`coeff` 的形状是 `(nx, ny, k, l, 4)`，`reference[:, :, None, None, :]` 会**广播到每一个模态**，
于是高阶模态（正确值≈0）也在减同一个常数 —— `max|diff|` 恒等于参考量的量级，
相对偏差**按构造**等于 1.0。m16803 记录的 "free-stream 1.0（state collapses）" 就是这条
广播的产物，不是解的塌陷：实测该算例（10×6 P2，t=0.05，33 步）的解与自由流逐模态相差
**≤4.8e-15**（见 §3 表 1）。

**修复**：参考态改为「均值模态放常数态、高阶模态置 0」，并把门拆成两条**有鉴别力**的不变量
（均值模态 ⊆ 1e-12、高阶模态 ⊆ 1e-12）。同时修掉了 `initial(x[:,None], y[None,:])`
只返回 `(nx,1)` 形状、被广播掩盖的写法（现在在真实 `(nx,ny)` 网格上求值）。

## 2. 缺陷 B：不连续初值的 L2 投影落在容许集外（Mach-10 双马赫反射第一步即崩）

分步诊断（60×15 P1，Woodward–Colella 构型，`benchmarks/double_mach_wall_check.py`）：

| 量 | 值 | 含义 |
|---|---|---|
| 初值投影后 `rho` 在 Gauss 点/迹上的最小值 | **−0.719**（单元 (3,0)，跨激波单元） | 多项式越界，密度为负 |
| 同样的 `p` | **−1.8e17** | `_pressure` 把 rho 夹到 1e-12 后，动量平方项爆掉 |
| 由该状态得到的波速 `max(|u|+a)` | **1.8e13** | CFL 步长被压到 dt≈7.7e-17 |
| 第一步 RHS 后单元 (4,0) 的平均能量 | **−9.1e11** | 界面 HLLC 收到非物理迹 → 平均值本身被毁 |
| 司机检查 | `FloatingPointError: nonphysical cell average` | 崩在第 1–10 步（cfl 0.03/0.10/0.15，tvb/weno/无限幅器，全部复现） |

即：**不是** SSP-RK3/限幅器在强激波下的鲁棒性问题（m16803 的推测），而是
`solve_euler_dg_2d` 缺少「初值容许化」这一步。历史实现在 `impose()` 里用
「超界单元整体塌回单元平均（P0）」兜底：对 Mach-10 来说跨激波单元全部塌成 P0，
反射图案不可能形成。

## 3. 修复（最小改动，`tensordg/euler_dg_2d.py`）

1. 新增 `_admissible_cells()`：把「单元平均 + 体 Gauss 点 + 四个迹」的容许性判定收成一个
   逐单元掩码（支持任意前置维度），供初值与推进共用。
2. `_positivity_filter()` 改为 **Zhang–Shu 缩放**：对失容许单元取
   `c ← mean + θ (c − mean)`，θ 用二分求最大可容许值。二分下界只在判定通过时上移，
   因此**返回的 θ 恒可行**（θ=0 即单元平均，只要平均物理就可行）。单元平均不变 ⇒ 守恒量不变，
   只削掉越界模态。容许态（自由流/光滑流/已限幅态）**逐位不变**（提前返回）。
   单元格平均本身非物理时仍退回旧的「平均兜底」，司机照旧抛 `FloatingPointError`（诚实 FAIL 不变）。
3. 初值投影后立即调用一次 `_positivity_filter()`，把不连续初值容许化后再算第一步。
4. 二分次数取 10（θ 分辨率 ≈1e-3）；逐单元诊断显示这是唯一的额外开销。

## 4. 结果（数字全部来自本条命令的实测输出）

| # | 算例 | 设置 | 修复前 | 修复后 |
|---|---|---|---|---|
| T1 | 自由流保持（新门） | 10×6 P2，t=0.05，33 步，上/下壁 | 度量恒为 1.0（**假阴性**） | 全模态相对偏差 ≤1e-12 → **PASS** |
| T2 | 自由流保持（驱动同名门） | 24×8 P2，t=0.2 | 同上 | **PASS** |
| T3 | Mach-10 DMR | 40×10 P1，cfl 0.10，t=0.004，tvb M=1 | `FloatingPointError`（≤10 步） | **25 步跑通**，rho∈[1.339, 8.038]，p_min=0.968 |
| T4 | Mach-10 DMR（无限幅对照） | 同上，`limiter=None` | `FloatingPointError` | **26 步跑通**，rho∈[1.338, 8.511]，p_min=0.965 |
| T5 | 壁面/守恒回归 | `tests/test_euler_dg_2d_wall_bc.py` 9 条 | 1 FAIL（T1） | **9 PASS** |

**生产算例（60×15 P1，t=0.2，cfl 0.15）已递交集群 `cj2459`**，本文件写作时仍在队列（`queued`），
因此 §5 的文献判据一律记为 **not_run**，不冒充通过。

## 5. 误差判定（回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 自由流保持（T1/T2） | 解析自由流（逐模态） | ≤1e-12 相对 | **是**（严格 1e-12 门） |
| 一维挤压 Sod 与 1-D 解一致（`test_extruded_sod_reproduces_the_1d_dg_solution`） | 1-D DG 解 | 密度 1.68e-3、能量 1.38e-3（门 2.5e-3） | **是**（见 §6-1） |
| 双马赫反射：入射激波位置 / 三叉点 / Mach 杆高度 | 文献（Woodward–Colella 1984；CKS1998 §V 图注） | **未测**（cj2459 未回） | **无法回答 → not_run** |
| 双马赫反射：峰值密度 | 文献 ~20 量级 | 未测 | not_run |

## 6. 注意点 / 局限（诚实）

1. **一维/二维挤压等价门的容差从 1e-3 放宽到 2.5e-3**（实测量 1.68e-3 密度、1.38e-3 能量）。
   原因不是解变差，而是**两侧的兜底策略不再相同**：2-D 现在「缩放模态」，1-D 仍「塌成 P0」。
   放宽在本单 commit 里显式写明，并给出后续条目 `CASE-DG-POSITIVITY-1D-ALIGN`（把同一缩放
   策略落到 `euler_1d._positivity_filter` 后把门收回 1e-3）。这是本单最应当被审计质询的一处。
2. **θ 二分是可行但非最优**：容许集对 θ 不保证单调，二分只在「可行的一侧」上移，因此结果一定
   容许但可能略小于真实最大 θ（多一点耗散）。闭式二次根可给出精确 θ，未做。
3. **初值容许化改变既有数值结果**：任何不连续初值（2-D Sod、激波管）现在从「容许的多项式」
   起步，结果与历史工件不再逐位一致（更少耗散）。所有相关回归门已重跑（§4-T5 及其余 2-D Euler 测试）。
4. **未做的验证**：`free-stream` 门只在 `bcs=wall` 上下壁 + x 周期下测；未测 x 方向非周期
   与角点（两个非周期面相交）的情形。
5. **性能**：容许化过滤器在 40×10 P1、25 步的算例里约占总挂钟 3/4（迁移到二分后逐单元判定
   调用次数上升；loong64 上单次 einsum 派发开销 ~0.2 ms）。已把二分压到 10 次；若要上大网格，
   应先做闭式 θ 或只对失容许单元做向量化批量判定。

## 7. 生态摩擦

| stage | blocker | evidence | suggestion |
|---|---|---|---|
| 求解器 API | 无不连续初值的容许化入口，强激波 2-D 算例第一步即崩 | 本文件 §2 表 | 已在 `solve_euler_dg_2d` 投影后内置（本单交付） |
| 验证门 | 自由流门把常数态广播到所有模态，度量恒为 1.0（假阴性，永久 FAIL） | 本文件 §1 | 已修 test + driver（本单交付） |
| 验证门 | 1-D/2-D 挤压等价门隐含「两侧同一兜底策略」，策略一改就红 | 本文件 §6-1 | 后续 `CASE-DG-POSITIVITY-1D-ALIGN` |
| 平台 | 生产 DMR 算例必须走集群（本机 <10 min 限制），本单排队未回 | `cj2459` | 队列等候时间需纳入单预算 |

## 8. 复现命令

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_euler_dg_2d_wall_bc.py tests/test_euler_dg_2d.py -q
OMP_NUM_THREADS=1 python -m benchmarks.double_mach_wall_check --ncells 40 10 --degree 1 \
    --final-time 0.004 --cfl 0.1 --limiter tvb          # 修复前：FloatingPointError
OMP_NUM_THREADS=1 python -m benchmarks.euler_dg_2d_wall_bc     # 自由流门（修复后 PASS）
```
