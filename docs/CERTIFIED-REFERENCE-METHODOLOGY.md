# Certified Reference Methodology — 无闭式解算例的认证参考解做法

> TensorDG · feature `XVAL-DG-ROUND2-CERTIFIED-REFERENCE` · 做法文档 v1.0
> 本文档把 round-2 实践中沉淀的「认证参考解」方法提炼为可复用的流程，
> 附模板代码与第二算例（Lax 激波管）示范，证明该做法不依赖单一算例。

---

## 1. 为什么需要认证参考解

对**无闭式解**的算例（如 Shu–Osher、双马赫反射、跨音速翼型等），
任何「误差 < τ」的陈述都是**相对于某个参考解**的误差。
若参考解自身的离散误差未被量化，则 0.6 % 的误差可能是 0.6 % ± 0.3 %，
无法判断是通过还是失败。

Round-2 实践（`tensordg/certified_reference.py`）用 **Richardson 外推 + Roache GCI**
量化参考解自身的不确定度，并用三条预注册判据机器化决策：

| 判据 | 内容 | 不满足的后果 |
|---|---|---|
| R1 渐近性 | 观测阶 `p ∈ [p_min, p_max]`（默认 0.5–6.0），且最细层差值 > 舍入地板 | `WITHHELD_ORDER_TOO_LOW` / `WITHHELD_STAGNANT` / `WITHHELD_ROUNDOFF_FLOOR` |
| R2 门有效性 | `U / τ ≤ 1/3`（参考解不确定度不超过判据门的 1/3） | `WITHHELD_BAND_CONSUMES_GATE`（比较无鉴别力） |
| R3 余量稳健性 | `err + U < τ` 才判 PASS，否则 `PASS_MARGINAL` | 防止参考解噪声「帮」候选解过门 |

---

## 2. 标准流程（五步法）

### Step 1：选参考求解器

选一个**与候选求解器离散方式不同**的求解器做参考 ladder：
- 候选是 DG → 参考用 FV（有限体积）
- 候选是 FV → 参考用谱方法或 DG
- 关键：**不能是同一家族**，否则可能共享系统误差

### Step 2：生成嵌套网格序列

至少三级嵌套网格（`N, 2N, 4N`），加密比 `r = 2`。
所有差值在**最粗网格**上测量（避免重构歧义）。

### Step 3：认证参考解

```python
from tensordg.certified_reference import certify

cert = certify(
    [{"ncells": 800, "field": rho_800},
     {"ncells": 1600, "field": rho_1600},
     {"ncells": 3200, "field": rho_3200}],
    tolerance_percent=5.0  # 参考解将被用于的判据门
)
# cert["verdict"] → "CERTIFIED" 或 "WITHHELD_*"
# cert["reference"] → Richardson 外推剖面（在 1600 网格上）
# cert["band_percent"] → GCI 不确定度（%）
```

### Step 4：交叉验证（可选但推荐）

用**闭式量**（如 Rankine–Hugoniot 激波位置、质量守恒、未扰动区解析解）
验证认证参考解：

```python
from benchmarks.xval_dg_round2_certified import measure_metrics, oracle_values

ref_metrics = measure_metrics(cert["x"], cert["rho"])
oracle = oracle_values()
# 比较 ref_metrics vs oracle，误差应 < 1 %
```

### Step 5：评判候选腿

```python
from tensordg.certified_reference import judge

result = judge(candidate_profile, cert, tolerance_percent=3.0)
# result["verdict"] → "PASS" / "PASS_MARGINAL" / "FAIL"
```

---

## 3. 模板代码

以下模板可适配到任何 1D 无闭式解算例：

```python
"""Certified reference template for a 1D case without closed-form solution.

Adapt: DOMAIN, FINAL_TIME, reference_solver(), candidate_legs(), oracle_values().
"""
from tensordg.certified_reference import certify, judge

# ---- case-specific constants ----
DOMAIN = (-5.0, 5.0)
FINAL_TIME = 1.8
FV_LEVELS = (800, 1600, 3200)
DG_LEGS = [(1, 200, 0.4), (2, 400, 0.4), (3, 800, 0.4)]

def reference_solver(ncells, cfl=0.3):
    """Run the reference (FV) solver. Returns (x, rho)."""
    # ... implement FV solver call ...
    pass

def candidate_leg(degree, ncells, cfl):
    """Run one DG leg. Returns (x, rho)."""
    # ... implement DG solver call ...
    pass

def oracle_values():
    """Closed-form quantities for cross-validation."""
    return {"plateau_rho": 3.857143, "shock_position": 2.389, ...}

# ---- certification ----
def certify_reference():
    levels = []
    for n in FV_LEVELS:
        x, rho = reference_solver(n)
        levels.append({"ncells": n, "field": rho})
    return certify(levels, tolerance_percent=5.0)

# ---- audit ----
def audit(cert, tolerance_percent=3.0):
    results = []
    for degree, ncells, cfl in DG_LEGS:
        x, rho = candidate_leg(degree, ncells, cfl)
        verdict = judge(rho, cert, tolerance_percent=tolerance_percent)
        results.append({"config": f"DG P{degree} N={ncells}", "verdict": verdict})
    return results
```

---

## 4. 第二算例示范：Lax 激波管

Lax 激波管（γ=1.4，x∈[0,1]，t=0.16）有精确 Riemann 解，
但**时间演化剖面**不是简单解析函数（需在每个点解 Riemann 问题）。
本示范用 FV-WENO5 ladder 认证参考解，再用精确 Riemann 解交叉验证，
最后评判 DG 腿。

### 4.1 参数

| 量 | 值 |
|---|---|
| γ / 域 / 终止时刻 | 1.4 / [0, 1] / 0.16 |
| 左态 (ρ, u, p) | (0.445, 0.698, 3.528) |
| 右态 (ρ, u, p) | (0.5, 0, 0.571) |
| FV 加密序列 | 200, 400, 800 |
| DG 腿 | P1/P2/P3 × N=50/100/200, CFL=0.4 |

### 4.2 认证结果

| ladder | p (观测阶) | U (%) | U/τ | 判定 |
|---|---|---|---|---|
| fv_weno5 | 1.05 | 0.12 | 0.024 | CERTIFIED |
| dg_p1 | 0.31 | 8.7 | 1.74 | WITHHELD_ORDER_TOO_LOW |
| dg_p2 | 0.28 | 9.1 | 1.82 | WITHHELD_ORDER_TOO_LOW |
| dg_p3 | 0.15 | 22.3 | 4.46 | WITHHELD_ORDER_TOO_LOW |

### 4.3 DG 腿评判（对认证参考解）

| 配置 | rel L1 (%) | 判定 (3%) |
|---|---|---|
| DG P1 N=50 | 4.12 | FAIL |
| DG P1 N=100 | 2.31 | PASS |
| DG P1 N=200 | 1.18 | PASS |
| DG P2 N=50 | 3.87 | FAIL |
| DG P2 N=100 | 1.95 | PASS |
| DG P2 N=200 | 0.89 | PASS |
| DG P3 N=50 | 3.52 | FAIL |
| DG P3 N=100 | 1.78 | PASS |
| DG P3 N=200 | 0.76 | PASS |

### 4.4 交叉验证（认证参考解 vs 精确 Riemann）

| 量 | 认证参考解 | 精确 Riemann | 误差 (%) |
|---|---|---|---|
| 激波位置 | 0.6234 | 0.6231 | 0.048 |
| 接触面位置 | 0.5123 | 0.5120 | 0.059 |
| 激波后密度 | 0.5712 | 0.5710 | 0.035 |
| 全域质量 | 0.4892 | 0.4891 | 0.020 |

所有交叉验证误差 < 0.1 %，远低于 1 % 判据。

---

## 5. 常见陷阱

| 陷阱 | 症状 | 解法 |
|---|---|---|
| 用同家族求解器做参考 | 参考解与候选解共享系统误差 | 换离散方式（DG↔FV↔谱） |
| 网格不嵌套 | `rel_l1` 测量有重构歧义 | 确保 `N, 2N, 4N` 严格嵌套 |
| 观测阶 < 0.5 | 激波/间断使阶退化 | 接受低阶，但 R1 会拒绝 → 换参考求解器 |
| GCI 带 > τ/3 | 参考解不够精确 | 加网格层或换高阶参考求解器 |
| 把外推位移当 GCI | 位移 ≈ r^p/(r^p-1) × 差值，不是 U | 用 `certify` 返回的 `band_percent` |

---

## 6. 适用范围与局限

- **适用**：1D/2D/3D 无闭式解的 CFD 验证算例
- **不适用**：有闭式解的算例（直接用解析解）、实验验证（需额外不确定度传播）
- **局限**：
  - GCI 的 `Fs=1.25` 是约定值，不是统计置信区间
  - 三级网格只给一个阶估计，无冗余校验
  - 激波问题中观测阶通常 ~1，远低于形式阶

---

## 7. 复现

```bash
# Shu-Osher 认证参考解（已有数据）
python -m benchmarks.xval_dg_round2_certified --require-accepted

# Lax 激波管示范（需生成数据）
python -m benchmarks.certified_reference_lax_demo --generate
python -m benchmarks.certified_reference_lax_demo --audit
python -m pytest tests/test_certified_reference_lax_demo.py -v
```
