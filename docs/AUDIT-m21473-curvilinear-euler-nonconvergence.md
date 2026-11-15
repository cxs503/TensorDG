# 审计 · m21473 · `tensordg.euler_curved` 不收敛（CKS1998 曲边算例证据链断点）

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199--224, Part V）
> 对象：`tensordg/euler_curved.py`（等参曲边 2-D Euler DG，平台 `REPRO-DG-CKS1998` 的
> 「曲边几何」能力承重件，m16803/m21360/m21389 三单的产出）
> 性质：**对已合并特性的独立复核**（verification-debt / adversarial review），不改被审代码
> 证据工件：`benchmarks/euler_curved_convergence_audit.py` →
> `validation/euler_curved_convergence/audit.json` + `audit.md`（本文所有数字由脚本写出，零手抄）
> 回归锁：`tests/test_euler_curved_convergence_defect.py`（3 passed, 13.2 s）

---

## ① 为什么查这一件

`tests/test_euler_curved.py` 在 HEAD 上是 **9 passed / 2 failed**：

| 门 | 断言 | HEAD 实测 |
|---|---|---|
| F3 design order（warped, P1, 8→16 网格） | `order >= 1.5` | **1.153**（误差 5.44e-2 → 2.44e-2）|
| F5 conservation（warped, P1, 8 网格, t=0.5） | 总质量/动量/能量漂移 `< 1e-10` | **`RuntimeError: went unstable (max|U| > 47242.9)`** |

m21360 与 m21389 两单都把这两条记为「HEAD 既有失败」并**绕过**（后者以 stash 复测确认其为
预存在）。两次绕过使该模块的 G2/G3/G4 声明（`euler_curved.py` 模块 docstring：
「G2 设计阶 p+1」「G3 曲率不损精度」「G4 守恒到 round-off」）**没有任何绿色证据支撑**，
而它正是 `REPRO-DG-CKS1998` 曲边几何腿的唯一实现。故本单做的是「红门是真红还是门写错了」的判定。

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
| 演化 t=0.1（straight, strength 1） | 6.837e-03 | 4.222e-03 | **1.139e-02** | **0.70, −1.43** ❌ |
| 演化 t=0.1（warped, strength 1） | 1.045e-02 | 4.160e-03 | **7.919e-03** | **1.33, −0.93** ❌ |
| 演化 t=0.1（straight, strength 5） | 3.141e-02 | 2.666e-02 | **BLOWUP** | **0.24** ❌ |
| 演化 t=0.1（warped, strength 5） | 5.437e-02 | 2.445e-02 | 4.660e-02 | **1.15, −0.93** ❌ |
| 演化 t=0.1（**cartesian 参考**，两个 strength） | 1.837e-02 | 5.839e-03 | 1.682e-03 | **1.65, 1.80** ✅ |
| 固定 dt=1e-3（straight, strength 1） | 6.837e-03 | 4.222e-03 | — | 与 CFL 列**同值** |

四条独立判据同向：

1. **投影干净 O(h²)**（2.06/1.98）→ 网格、几何映射、求积、参考解、`relative_errors` 度量都没问题。
2. **演化误差不降反升**（16→32 阶 −1.43），且在**直网格**上同样发生（0.70）——
   **不是曲边几何的锅**，是算子本身。
3. **固定 dt 复现 CFL 结果**（相对差 8.0e-06）→ 时间积分不可能解释；缺陷在空间算子。
4. **同题笛卡尔求解器 8 倍更准且单调二阶**（32 网格 1.68e-3 vs 1.14e-02）→
   「粗网格更准」是因为缺陷随分辨率增长，不是曲边 DG 的正常行为。

## ④ 与既有门/声明的对照

| 声明（源码 docstring / 测试） | 实测 | 判级 |
|---|---|---|
| G1 自由流保持（常数态 `sup|dU/dt| ~ 1e-15`）| 复跑 F1 通过（1e-13 量级）| **成立**（仅常数态）|
| G1n 负对照（aliased 体积导子）| F1n 通过 | 成立（门不空）|
| G2 光滑涡设计阶 `p+1`（容差 `p+1/2`）| **1.15 / 0.70 / −1.43**，无收敛 | **NOT_SUPPORTED** |
| G3 曲率不损精度（warped/straight 比值 O(1)）| 比值 O(1) ✅，但**两者都不收敛** | 门本身**空转**（比的是两个都错的解）|
| G4 曲边周期网格守恒到 round-off | warped 8 网格 t=0.5 **爆**；直网格每步质量漂移 ~7e-7（t=0.1 累积 4e-4 相对）| **NOT_SUPPORTED** |
| `cylinder_annulus_mesh` + `wall_force` 可算 CKS1998 圆柱 | 模块 docstring 已诚实自认壁面推进线性不稳定、圆柱**未复现** | **WITHHELD**（诚实，但须与 G2/G4 同级看待）|

## ⑤ 影响面

* `REPRO-DG-CKS1998` 的**曲边几何腿不可引用**：任何以 `euler_curved` 的收敛/守恒数字支撑的
  平台报告（含 `validation/curved_wall_bc/metrics.json` 的力/面积数）只能作为**几何层**证据，
  不能作为**求解精度**证据。
* `tests/test_euler_curved.py` 的 F3/F5 是**真红**，不是门写太严：把门放宽到「阶 ≥ 0.7」会
  让一个不收敛的算子变成「通过」，禁止这么做。
* 直线/笛卡尔腿（`euler_dg_2d`）不受影响：同题同网格 1.65/1.80 阶、32 网格 1.68e-3，健康。

## ⑥ 注意点（含未做项）

* **根因未定位到行**：已排除时间积分（固定 dt 复现）、几何映射与求积（投影 O(h²)）、
  曲边 vs 直网格（两者同病）、HLLC 法向推广（`test_hllc_reduces_to_the_verified_axis_aligned_solver`
  机器精度通过）。剩余嫌疑集中在体积项
  `differentiate(contravariant, ·)` 与界面项 `numerical - physical(inner)` 的**高波数一致性**
  （误差随 h→0 增长是「不一致/高波数放大」的指纹），以及 `_diff_first` 用 `degree = nq-1`
  对**通量**做节点插值求导、而迹仍用 p 次多项式这一混合离散。
* **未跑**：P2/P3 的同一审计（只做 p=1）、`benchmarks/euler_curved_verification.py` 收敛段
  的 `ZeroDivisionError`（m21360 已记录，本单未复现）、非结构/三角形映射（模块声明不支持）。
* 本单**不主张修复**：修复需重推体积项/界面项，属重计算 + 重设计，应另开单并带 G2/G3/G4 三绿证据。

## ⑦ 判定（disposition）

**Request changes**：`tests/test_euler_curved.py::test_design_order_on_the_warped_mesh` 与
`::test_total_conserved_quantities_hold_on_the_warped_mesh` **保持红**（honest FAIL，不得覆盖、
不得放宽）；`euler_curved.py` 模块 docstring 的 G2/G3/G4 段应降级为「已验证：FSP/负对照；
未验证：设计阶/守恒」。本单交付的 `tests/test_euler_curved_convergence_defect.py` 把上述
三条判据钉死（投影 ≥1.7 阶、演化 8→16 阶 <0.9、32 网格误差 >16 网格、固定 dt 与 CFL 差 <1e-4），
修复后用强化门（阶 ≥1.5 + 单调下降）替换它。

## ⑧ 复现命令

```bash
OMP_NUM_THREADS=1 PYTHONPATH=. python -m benchmarks.euler_curved_convergence_audit   # ~100 s
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_euler_curved_convergence_defect.py -q   # 13 s
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_euler_curved.py -q          # 9 passed, 2 failed（红门如实保留）
```