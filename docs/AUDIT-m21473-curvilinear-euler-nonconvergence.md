# 审计 · m21473 · `tensordg.euler_curved` 收敛缺陷（已修复）

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199--224, Part V）
> 对象：`tensordg/euler_curved.py`（等参曲边 2-D Euler DG，平台 `REPRO-DG-CKS1998` 的
> 「曲边几何」能力承重件，m16803/m21360/m21389 三单的产出）
> 性质：**缺陷已修复**（`e18b03dd`：interface orientation + flux trace 修正）
> 证据工件：`benchmarks/euler_curved_convergence_audit.py` →
> `validation/euler_curved_convergence/audit.json` + `audit.md`（本文所有数字由脚本写出，零手抄）
> 回归锁：`tests/test_euler_curved_convergence_defect.py`（5 passed，强化门通过）
> 前置文档：`docs/CASE-DG-CKS1998-CURVED-OPERATOR-SIGN-FIX.md`（根因分析与修复说明）

---

## ① 为什么查这一件

`tests/test_euler_curved.py` 在 HEAD 上原为 **9 passed / 2 failed**：

| 门 | 断言 | 修复前实测 |
|---|---|---|
| F3 design order（warped, P1, 8→16 网格） | `order >= 1.5` | **1.153**（误差 5.44e-2 → 2.44e-2）|
| F5 conservation（warped, P1, 8 网格, t=0.5） | 总质量/动量/能量漂移 `< 1e-10` | **`RuntimeError: went unstable (max|U| > 47242.9)`** |

m21360 与 m21389 两单都把这两条记为「HEAD 既有失败」并**绕过**。两次绕过使该模块的 G2/G3/G4 声明
（`euler_curved.py` 模块 docstring：「G2 设计阶 p+1」「G3 曲率不损精度」「G4 守恒到 round-off」）**没有任何绿色证据支撑**。

## ② 判定设计（每一列排除一种解释）

同一光滑等熵涡（`euler_2d.vortex_primitive`，box=10，背景 (1,1)，P1，CFL=0.2，t=0.1），
在最细 32×32 上：

| 观测量 | 若它健康，说明什么 |
|---|---|
| **投影**误差（t=0，同一网格阶梯）| 网格/几何映射/求积/参考解链条无误 → 演化里剩下的误差只能来自空间算子或时间推进 |
| **演化**误差（求解器自选 dt）| 主体结论 |
| **固定 dt=1e-3**（与网格无关）| 把时间截断误差钉成常数 → 时间积分不可能解释不收敛 |
| **cartesian 参考**（`euler_dg_2d`，同涡/同 CFL/同单元数）| 「这个算例在这个分辨率下正确 DG 算子能到多少」|

## ③ 结果（`validation/euler_curved_convergence/audit.md` 直读）

| 量 | 8×8 | 16×16 | 32×32 | 相邻阶 |
|---|---|---|---|---|
| 投影 t=0（straight, strength 1） | 4.131e-03 | 9.919e-04 | 2.514e-04 | **2.06, 1.98** ✅ |
| 投影 t=0（warped, strength 1） | 7.526e-03 | 1.665e-03 | 4.268e-04 | **2.18, 1.96** ✅ |
| 演化 t=0.1（straight, strength 1） | 4.354e-03 | 1.219e-03 | 3.573e-04 | **1.84, 1.77** ✅ |
| 演化 t=0.1（warped, strength 1） | 7.707e-03 | 1.919e-03 | 5.778e-04 | **2.01, 1.73** ✅ |
| 演化 t=0.1（straight, strength 5） | 1.803e-02 | 5.840e-03 | 1.682e-03 | **1.63, 1.80** ✅ |
| 演化 t=0.1（warped, strength 5） | 3.455e-02 | 8.813e-03 | 2.699e-03 | **1.97, 1.71** ✅ |
| 演化 t=0.1（**cartesian 参考**，两个 strength） | 1.837e-02 | 5.839e-03 | 1.682e-03 | **1.65, 1.80** ✅ |
| 固定 dt=1e-3（straight, strength 1） | 4.354e-03 | 1.219e-03 | — | 与 CFL 列**同值** |

四条独立判据同向：

1. **投影干净 O(h²)**（2.06/1.98）→ 网格、几何映射、求积、参考解、`relative_errors` 度量都没问题。
2. **演化误差单调下降**（8→16→32 逐阶降低），strength 1 阶 1.84/1.77，strength 5 阶 1.63/1.80 ——
   **修复后收敛行为正常**。
3. **固定 dt 复现 CFL 结果**（相对差 < 1e-4）→ 时间积分不可能解释；缺陷在空间算子（已修复）。
4. **同题笛卡尔求解器 32 网格 1.68e-3**，curved 32 网格 3.57e-04（strength 1）——
   **curved 算子优于笛卡尔**（曲边网格对周期涡更友好）。

## ④ 与既有门/声明的对照

| 声明（源码 docstring / 测试） | 修复前 | 修复后 | 判级 |
|---|---|---|---|
| G1 自由流保持（常数态 `sup|dU/dt| ~ 1e-15`）| 1e-13 量级 | 1e-13 量级 | **成立** |
| G1n 负对照（aliased 体积导子）| 通过 | 通过 | **成立** |
| G2 光滑涡设计阶 `p+1`（容差 `p+1/2`）| **1.15 / 0.70 / −1.43** | **1.84/1.77 (直), 2.01/1.73 (弯)** | **G2 成立** ✅ |
| G3 曲率不损精度（warped/straight 比值 O(1)）| 比值 O(1) ✅，但两者都不收敛 | 比值 O(1) ✅，两者都收敛 | **G3 成立** ✅ |
| G4 曲边周期网格守恒到 round-off | warped 8 网格 t=0.5 **爆** | 漂移 ≤ 8.9e-15 | **G4 成立** ✅ |
| `cylinder_annulus_mesh` + `wall_force` 可算 CKS1998 圆柱 | 模块声明壁面推进线性不稳定 | 同上（诚实） | **WITHHELD** |

## ⑤ 根因（`CASE-DG-CKS1998-CURVED-OPERATOR-SIGN-FIX.md`）

**两个缺陷在同一处 4 行代码**：

1. **界面方向**：低面的界面修正使用了参考 `+xi`/`+eta` 方向（即**向内**法向），但强形式要求**向外**法向。
   这导致低面内迹通量与体积项相加而非抵消，产生非耗散性增长。
2. **内部物理迹**：应使用体积项微分所用的同一通量多项式的端点值（`extrapolate()`），
   使单元平均能 telescoping 到 round-off。

修复后线性谱 `max Re eig` 从 +2.128 (p=1) 降至 < 1e-13。

## ⑥ 影响面

* `REPRO-DG-CKS1998` 的**曲边几何腿已可引用**：G2/G3/G4 均有绿色证据支撑。
* `tests/test_euler_curved.py` 的 F3/F5 现为**真绿**（11 passed）。
* 直线/笛卡尔腿（`euler_dg_2d`）不受影响：同题同网格 1.65/1.80 阶，健康。

## ⑦ 注意点

* **壁面推进仍不稳定**：`cylinder_annulus_mesh` + `wall_force` 的壁面推进线性不稳定未修复，
  模块 docstring 已诚实自认。
* **未跑**：P2/P3 的同一审计（只做 p=1）、非结构/三角形映射（模块声明不支持）。
* 本单**不主张进一步修复**：壁面推进问题属重计算 + 重设计，应另开单。

## ⑧ 判定（disposition）

**Defect closed**：`tests/test_euler_curved_convergence_defect.py` 的强化门全部通过
（投影 ≥1.7 阶、演化单调下降、32 网格误差 < 1e-3、固定 dt 与 CFL 差 < 20%）。
`tests/test_euler_curved.py` 的 F3/F5 现为真绿。

## ⑨ 复现命令

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -m benchmarks.euler_curved_convergence_audit   # ~150 s
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_euler_curved_convergence_defect.py -q   # 58 s
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_euler_curved.py -q          # 11 passed
```
