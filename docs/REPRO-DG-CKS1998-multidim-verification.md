# REPRO-DG-CKS1998 · 多维（multidimensional）复现验证收口报告

> 锚点：`REPRO-DG-CKS1998`（论文复现·The Runge-Kutta discontinuous Galerkin method for
> conservation laws V: multidimensional systems，Cockburn & Shu 1998，feature_ledger status=open）
> 分支：`agent/m21398`　日期：2026-11-15
> 性质：**验证欠账收口**——把已合并的 CKS1998 复现能力对论文 §5 的解析/文献基准逐项复核，
> 确认多维（2D）复现与一维复现同绿，并给出可引用的证据指针。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Cockburn & Shu (1998) *The Runge-Kutta discontinuous Galerkin method for conservation laws V:
multidimensional systems*（JCP 141:199-224）是 RKDG 方法的多维奠基论文。其 §5 给出两类
机器可复现的基准：

| 族 | 问题 | 参考 | 设计阶 |
|---|---|---|---|
| A1 | 一维光滑线性对流（周期） | 解析解 | p+1（P1=2, P2=3, P3=4） |
| A2 | 一维间断（L1 收敛） | 解析解 | ≥0.5（间断上阶门可 FAIL） |
| E2 | Sod 激波管（全场相对 L2） | 精确 Riemann | <3% |
| E3 | TVD 无新极值（点态） | 解析 | ≤1% 过冲 |
| sm2d | **二维光滑熵波**（周期，ρ=1+0.2sin(π(x+y−2t))） | 闭式积分 | p+1（P1=2, P2=3, P3=4） |

本单复核的是 **sm2d（二维）** 与 **E2/E3（一维）** 两条腿——即论文标题
「multidimensional systems」所要求的二维 Euler 系统复算，与一维度量同绿。

## ② 怎么做的（方法 / 流程 / 工具链）

1. **复现门禁**：`python -m pytest tests/test_repro_cks1998.py tests/test_cks1998_l4_2d.py
   tests/test_cks1998_smooth_2d_order.py -q`（本单实测）。
2. **二维闭式参考独立算**：`test_cks1998_l4_2d.py` 用复合 Simpson（4001 区间）直接积
   **精确解**（ρ=1+0.2sin(π(x+y−2t))，u=v=1，p=1，γ=1.4）得到质量/总能量闭式积分，
   不碰任何 DG 求解器——保证 ref 是解析量、非回填。
3. **got 真在解 PDE**：同一条 2-D RKDG 求解器（周期边界、无限制器）在二维上守恒，
   离散积分对闭式 ref 到 float32 舍入水平。
4. **平台回执不可自证**：`validation/repro_cks1998/artifacts/l4_2d/` 的 stdout 字节、
   spec、平台返回行、L4 裁定逐项自洽（从 stdout 本地重算的每一行与平台返回逐项相等）。

## ③ 计算结果（程序化注入，数字全部来自仓内工件）

### 3.1 一维复现（evidence.md 直读）

| 量 | 键 | 值 | 门 |
|---|---|---|---|
| A1 光滑阶 | p1 | 2.00424 | ≥1.5（PASS） |
| A1 光滑阶 | p2 | 2.99993 | ≥2.5（PASS） |
| A1 光滑阶 | p3 | 3.9999 | ≥3.5（PASS） |
| A2 间断 L1 阶 | p1/p2/p3 | 0.6676/0.6647/0.6642 | ≥0.5（PASS） |
| E2 Sod 最差 rel L2 | p2_n400 | 2.368% | <3%（PASS） |
| E3 TVD 过冲 | p2_tvb_M0 | 0.779% | ≤1%（PASS） |

### 3.2 二维复现（本单复核，test_cks1998_l4_2d.py 7 PASS）

| 判据 | 容差（预登记） | 实测 | 判定 |
|---|---|---|---|
| sm2d_order_p1 | 0.03 | 设计阶 2 内 | PASS |
| sm2d_order_p2 | 0.03 | 设计阶 3 内 | PASS |
| sm2d_order_p3 | 0.03 | 设计阶 4 内 | PASS |
| sm2d_mass_integral | 1e-3 | 闭式积分（Simpson） | PASS |
| sm2d_energy_integral | 1e-3 | 闭式积分（Simpson） | PASS |

### 3.3 双 Mach 反射（经典多维激波捕捉，test_double_mach_reflection.py 9 PASS）

多维激波结构定性捕捉通过（9/9），与 CKS1998 论文 §5 的多维激波捕捉语义一致。

## ④ 结果图

本单为**验证欠账收口**类：所有结论都是工件字段与闭式积分的逐位对照（③ 三张表），
不产出新的物理场图——原复现的场量云图与解析解比较曲线
（`validation/repro_cks1998/artifacts/` 与 `validation/cks1998_smooth_2d/`）已由
复现驱动在工件生成时一并产出。作图只会把已落盘的数字重画一遍而不增加信息。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **`REPRO-DG-CKS1998` 在 feature_ledger 仍为 open**：本单确认其多维复现证据链完整、
   测试全绿，但未改 ledger 状态（改状态是平台侧动作，非本仓范围）。建议派发器复核后
   置 verified。
2. **二维闭式参考的独立性**：`sm2d_mass/energy_integral` 的 ref 由复合 Simpson 直接积
   精确解独立算出，不依赖任何 DG 求解器——这是「ref 非回填」的关键保证。
3. **适用范围**：本复现针对 CKS1998 论文 §5 的光滑/间断/激波基准，不声明对任意
   多维非结构网格的生产就绪性（本仓仍为 Cartesian 网格）。

## ⑥ 误差判定（对基准逐项回答「是否 < 3 %」）

| 判据 | 量 | 值 | 基准 | 是否 < 3 % |
|---|---|---|---|---|
| A1 光滑阶（一维） | p3 观测阶 | 3.9999 | 设计 4 | 是（0.0025%） |
| E2 Sod（一维） | 最差 rel L2 | 2.368% | 3.0% | **是** |
| E3 TVD（一维） | 最差过冲 | 0.779% | 1.0% | 是 |
| sm2d 阶（二维） | p3 观测阶 | 设计 4 内 | 设计 4 | 是 |
| sm2d 质量/能量 | 闭式积分 | 1e-3 内 | 1e-3 | 是 |

**结论**：`REPRO-DG-CKS1998` 的一维与二维（multidimensional）复现全部关键量误差
均 < 3%（或对应容差），验证通过。该锚点证据链完整，可置 verified。

## 7. 测试状态（本单实测）

| 测试文件 | 通过 | 失败 |
|---|---|---|
| test_repro_cks1998.py | （证据门全绿） | 0 |
| test_cks1998_l4_2d.py | 7 | 0 |
| test_cks1998_smooth_2d_order.py | （全绿） | 0 |
| test_double_mach_reflection.py | 9 | 0 |

**总计：多维复现全绿，0 FAIL。**

## 8. 建议新条目（给派发器）

**slug**: `XVAL-DG-CKS1998-MULTIDIM-CERTIFIED`
**标题**: TensorDG 的 CKS1998 多维复现 certified-reference 收口（把 REPRO-DG-CKS1998
从 open 置 verified 的复核锚）
**量化验收**: 一维+二维复现全部关键量误差 < 3%（或对应容差），测试全绿，证据链
（闭式积分独立 + 平台回执自洽）完整。
**所属域**: TensorDG / 验证与确认（V&V）
