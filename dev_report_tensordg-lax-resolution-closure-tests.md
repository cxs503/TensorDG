# dev report · TensorDG：CASE-LAX-DG 分辨率闭合验证测试

**锚点**：`CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 <3%」，`feature_ledger` status=open）
**分支**：`agent/m20699` **范围**：`tests/test_lax_resolution_closure.py`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域内隔膜 `x0 = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。精确解含四道波：左稀疏波（head → tail）、接触间断、右激波。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

解析波系（闭式）：p\* = 2.466098、u\* = 1.528723、ρ\*\_L = 0.344568、ρ\*\_R = 1.304085。

## ② 怎么做的（方法 / 流程 / 工具链）

**问题**：CASE-LAX-DG 的台账验收原文是「ref: 精确解｜容差 <3%」。标准算例池阶梯（P1/P2/P3 × N=100/200/400）在 p3.N400 上的全场相对 L2 密度误差为 3.116 %，**略高于 3 % 门槛**（honest FAIL）。但分辨率闭合研究（`docs/CASE-LAX-DG-resolution-closure.md`）已将 P3 阶梯扩展到 N = [200, 400, 800, 1600]，证明 **N = 800 时三量最差 L2 = 2.315 % < 3 %**。

**缺口**：分辨率闭合研究的工件存在，但**没有测试**来机器化验证这一结论。任何读者只能信任文档中的数字，无法通过 `pytest` 自动复核。

**本单工作**：新增 `tests/test_lax_resolution_closure.py`（10 条测试），读取已提交的分辨率闭合工件（`docs/benchmarks/results/lax_resolution_closure/metrics.json`），机器化验证：

1. N = 800 腿三量 L2 全部 < 3 %；
2. 密度始终是最差场量；
3. 收敛阶梯单调递减；
4. 闭合 verdict = PASS，closure_mesh = 800；
5. N = 400 仍 FAIL（与算例池工件逐位一致）；
6. 无限幅臂在 N = 400 上最准确，TVB/WENO 均退化；
7. TVB 使 L2 密度恶化 ≥ 2×；
8. N = 800 的 L2 密度与文档公布值（2.315 %）一致；
9. 闭合研究配置与声明一致（P3, CFL 0.35, N=[200,400,800,1600]）。

## ③ 计算结果（程序化注入）

### 3.1 测试执行

```
python -m pytest tests/test_lax_resolution_closure.py -v
→ 10 passed, 0.13 s
```

### 3.2 关键验证数字（从工件读取，零手抄）

| 测试 | 验证内容 | 实测值 | 门槛 | 结果 |
|---|---|---|---|---|
| `test_n800_passes_ledger_bar` | N=800 三量 L2 < 3 % | ρ: 2.315 %, u: 1.383 %, p: 0.899 % | < 3 % | PASS |
| `test_closure_ladder_is_monotone` | L2 密度单调递减 | 4.200 % → 3.116 % → 2.315 % → 1.875 % | 单调 | PASS |
| `test_closure_verdict_is_pass` | 闭合 verdict | PASS, closure_mesh = 800 | PASS | PASS |
| `test_n400_still_fails_ledger_bar` | N=400 仍 FAIL | 3.116 % | ≥ 3 % | PASS |
| `test_tvb_degrades_accuracy_vs_unlimited` | TVB 退化比 | 7.039 % / 3.116 % = 2.26× | ≥ 2× | PASS |

### 3.3 原有测试回归

```
python -m pytest tests/test_lax_case_pool.py -v
→ 26 passed, 21.6 s
```

全部 26 条原有测试通过，无回归。

## ④ 结果图

本单为**验证欠账**类，不产出新物理场图。所有结论来自对已提交工件的机器化复核。
分辨率闭合研究的场量图（`docs/benchmarks/results/lax_resolution_closure/figures/`）
由原驱动 `benchmarks/lax_resolution_closure.py` 生成，本单只添加测试护栏。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **测试不重跑求解器**：本单 10 条测试全部读取已提交的 `metrics.json` 工件，
   不触发任何 DG 求解。这意味着测试速度极快（0.13 s），但也意味着
   如果工件本身有误，测试无法发现——工件的正确性由原分辨率闭合研究保证。
2. **N=800 通过 ≠ 台账已闭合**：台账登记的生产腿是 p3.N400，那条腿的诚实结论仍是 FAIL。
   本单只证明「同一配置在 N=800 上可达 3 % 门槛」，不改变台账原文的判定。
3. **限幅器不是补救手段**：测试确认 TVB/WENO 在 N=400 上使 L2 密度恶化 2.26×/2.07×，
   因为本仓限幅器将问题单元退化为线性重构，在 P3 上属于过度限制。
4. **工件路径**：测试从 `docs/benchmarks/results/lax_resolution_closure/metrics.json` 读取，
   该路径在 `docs/` 下（被 git 跟踪），而非 `runs/`（合并门禁不携带）。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | 是否 < 3 % |
|---|---|---|---|
| N=800 密度 L2 | 精确 Riemann 解 | 2.315 % | 是 |
| N=800 速度 L2 | 精确 Riemann 解 | 1.383 % | 是 |
| N=800 压力 L2 | 精确 Riemann 解 | 0.899 % | 是 |
| N=400 密度 L2 | 精确 Riemann 解 | 3.116 % | 否（honest FAIL） |

**结论**：分辨率闭合研究的 N=800 腿**全部三量 < 3 %**，闭合 verdict = PASS。
N=400 腿仍为 FAIL（3.116 % > 3 %），与算例池工件逐位一致。
本单新增的 10 条测试机器化锁死了这些结论，防止文档与工件漂移。
