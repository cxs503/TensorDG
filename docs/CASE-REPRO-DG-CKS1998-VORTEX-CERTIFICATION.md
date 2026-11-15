# REPRO-DG-CKS1998 · CKS1998 §5.3 涡旋二维阶次认证报告

> 锚点：`REPRO-DG-CKS1998`（论文复现·The Runge-Kutta discontinuous Galerkin method for
> conservation laws V: multidimensional systems，Cockburn & Shu 1998，JCP 141:199-224）
> 分支：`agent/m22140`  日期：2026-11-15
> 性质：**验证欠账补全**——为 CKS1998 §5.3 等熵涡旋二维算例新增阶次认证模块，
> 与 §5.1 光滑二维算例的认证体系对齐（估计量不确定度 + 渐近区检验 + 时空分离）。

---

## ① 案例介绍（物理背景 / 几何 / 参数表）

Cockburn & Shu (1998) §5.3 给出二维 Euler 方程组的等熵涡旋光滑解作为精度验证基准：

| 参数 | 值 |
|---|---|
| 域 | `[-5, 5]²`，周期边界 |
| 涡旋强度 β | 5.0 |
| 远场状态 | ρ=1, u=1, v=1, p=1, γ=1.4 |
| 精确解 | ρ = [1 − (γ−1)β²/(8γπ²) exp(1−r²)]^{1/(γ−1)} |
| 终时 | T = 1.0（涡旋回到初始位置） |
| CFL | 0.2（SSP-RK3 时间积分） |
| 网格序列 | P1: 16/24/32, P2: 12/16/24, P3: 8/12/16 |

设计阶：P^k 的 L2 误差阶应为 k+1（P1=2, P2=3, P3=4）。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **新建认证模块** `benchmarks/cks1998_vortex_order_certification.py`：
   - 复用 §5.1 光滑二维算例的认证框架（`certify()` 函数）
   - 预登记容差：`TOLERANCE_REL = 0.03`（3%），`SIGMA_BAND = (0.9, 1.1)`（渐近区）
   - 三值判定：`CERTIFIED` / `UNVERIFIED_BAND` / `FAIL`

2. **认证要素**：
   - **相邻网格阶** `p_i = ln(e_i/e_{i+1}) / ln(r_i)`
   - **最小二乘阶** `p_lsq`（全网格 / 最细三层）
   - **Richardson 外推残差** `ρ_R = |e_R| / e_fine`（检验网格序列是否渐近）
   - **估计量不确定度** `U = max|X - p_fine|`（跨估计量最大偏差）
   - **渐近比** `σ = p_{coarse} / p_fine`（应在 0.9–1.1 内）

3. **测试** `tests/test_cks1998_vortex_order_certification.py`（13 条）：
   - 估计器单元测试（合成数据，毫秒级）
   - 非空性检验（退化阶必须报 FAIL）
   - 工件可复现检验（从 committed levels 重算 verdict）

4. **运行**：`OMP_NUM_THREADS=1 python -m benchmarks.cks1998_vortex_order_certification --quick`

## ③ 计算结果（程序化注入，零手抄）

### 3.1 误差衰减表

| 配置 | P^k | 网格 | h | 步数 | ρ L1_rel | EOC L1 | 设计 p+1 |
|---|---|---|---|---|---|---|---|
| p1 | 1 | 16 | 6.250e-01 | 142 | 1.595e-03 | -- | 2 |
| p1 | 1 | 24 | 4.167e-01 | 212 | 6.908e-04 | 2.06 | 2 |
| p1 | 1 | 32 | 3.125e-01 | 282 | 3.807e-04 | 2.07 | 2 |
| p2 | 2 | 12 | 8.333e-01 | 176 | 3.528e-04 | -- | 3 |
| p2 | 2 | 16 | 6.250e-01 | 234 | 1.551e-04 | 2.86 | 3 |
| p2 | 2 | 24 | 4.167e-01 | 352 | 4.748e-05 | 2.92 | 3 |
| p3 | 3 | 8 | 1.250e+00 | 164 | 2.187e-04 | -- | 4 |
| p3 | 3 | 12 | 8.333e-01 | 246 | 4.339e-05 | 3.99 | 4 |
| p3 | 3 | 16 | 6.250e-01 | 328 | 1.421e-05 | 3.88 | 4 |

### 3.2 认证判定表

| degree | design | order_fine | delta_rel | uncertainty | delta_band | asymptotic_ok | extrapolation_consistent | verdict |
|---|---|---|---|---|---|---|---|---|
| p1 | 2 | 2.0708 | 0.0354 | 0.0071 | 0.0390 | True | True | **FAIL** |
| p2 | 3 | 2.9189 | 0.0270 | 0.0617 | 0.0476 | True | True | **UNVERIFIED_BAND** |
| p3 | 4 | 3.8799 | 0.0300 | 0.1092 | 0.0573 | True | True | **FAIL** |

### 3.3 关键发现

- **P1**：观测阶 2.07，相对偏差 3.54% > 3% 容差 → **FAIL**
- **P2**：观测阶 2.92，相对偏差 2.70% < 3%，但估计量不确定度 6.17% 使带宽超容差 → **UNVERIFIED_BAND**
- **P3**：观测阶 3.88，相对偏差 3.00% ≈ 3%，不确定度 10.92% 使带宽超容差 → **FAIL**

**与 §5.1 光滑二维算例对比**：§5.1 的 P1/P2/P3 均为 `CERTIFIED`（delta_band < 3%），
而 §5.3 涡旋算例无一认证通过。这说明涡旋算例的误差衰减在有限网格上尚未进入严格的渐近区，
或估计量分散度较大（高阶多项式对网格分辨率更敏感）。

## ④ 结果图

![CKS1998 Vortex Certification](docs/figures/cks1998_vortex_certification.png)

左图：误差衰减曲线与设计阶参考线；右图：阶次认证判定（绿=CERTIFIED，橙=UNVERIFIED_BAND，红=FAIL）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **honest FAIL**：P1 和 P3 的认证判定为 FAIL，P2 为 UNVERIFIED_BAND。
   这**不是求解器缺陷**——点估计 EOC（2.07/2.92/3.88）均满足论文的 p+0.5 容差，
   但认证框架要求更严格的 3% 带宽容差（delta + U/(p+1) ≤ 3%）。
   这是"点估计通过但认证不通过"的典型案例，正是本模块要揭示的。

2. **网格分辨率限制**：涡旋算例的 P3 在 16×16 网格上 delta_rel = 3.00%，恰在容差边界。
   若增加更细网格（如 24×24），可能进入渐近区使认证通过。但计算成本显著增加。

3. **与 §5.1 的差异**：§5.1 光滑算例（ρ = 1 + 0.2sin(π(x+y−2t))）的误差衰减更规则，
   估计量不确定度更小（P1: 0.7%, P2: 1.5%, P3: 0.9%），而涡旋算例的 P3 不确定度达 10.9%。
   这反映了涡旋解的高频成分对高阶 DG 的影响。

4. **时空耦合未测**：本认证未做 CFL 减半的时空分离检验（§5.1 认证模块有此功能）。
   后续可增加 `--time-separation` 选项。

## ⑥ 误差判定（显式回答「是否 < 3%」）

| 量 | 实测 | 容差 | 是否 < 3% |
|---|---|---|---|
| P1 delta_rel | 3.54% | 3% | **否**（FAIL） |
| P2 delta_rel | 2.70% | 3% | **是**（但 delta_band = 4.76% > 3% → UNVERIFIED_BAND） |
| P3 delta_rel | 3.00% | 3% | **边界**（delta_band = 5.73% > 3% → FAIL） |

**结论**：涡旋算例的点估计阶次满足论文的 p+0.5 容差，但在更严格的 3% 认证容差下，
P1/P3 为 FAIL，P2 为 UNVERIFIED_BAND。这是诚实的负面结果，反映了有限网格上
高阶 DG 对涡旋解的收敛行为尚未完全进入渐近区。

---

## 产物清单

| 文件 | 说明 |
|---|---|
| `benchmarks/cks1998_vortex_order_certification.py` | 阶次认证模块（~450 行） |
| `tests/test_cks1998_vortex_order_certification.py` | 测试 13 条（全绿） |
| `validation/cks1998_vortex_2d/certification.json` | 认证工件 |
| `docs/figures/cks1998_vortex_certification.png` | 认证结果图 |
| `tools/gen_vortex_certification_fig.py` | 绘图脚本 |
| `docs/CASE-REPRO-DG-CKS1998-VORTEX-CERTIFICATION.md` | 本报告 |
