# dev report · TensorDG：CASE-LAX-DG 工件清单漂移修复

**锚点**：`CASE-LAX-DG`（算例池·Lax via DG，验收原文「ref: 精确解｜容差 <3%」，`feature_ledger` status=open）
**分支**：`agent/m22307` **范围**：`scripts/sync_lax_report_numbers.py`、`docs/CASE-LAX-DG.md`

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Lax (1954) 激波管 / Toro (2009) Test 2：`[0,1]` 域内隔膜 `x₀ = 0.5` 两侧常状态，γ = 1.4，快照时刻 t = 0.13。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (0.445, 0.698, 3.528) | (0.5, 0.0, 0.571) |

精确 Riemann 解含四道波：左稀疏波（head → tail）、接触间断、右激波。

## ② 怎么做的（方法 / 流程 / 工具链）

**问题**：`tests/test_lax_case_pool.py::test_public_report_numbers_match_the_artifact` 失败，报错：
```
AssertionError: docs/CASE-LAX-DG.md drifted from the artifact:
['unlisted committed artifact: figures_binding.json']
```

根因：`docs/benchmarks/results/lax_case_pool/figures_binding.json` 是已提交的工件（图-工件-渲染器溯源绑定文件，含 sha256 校验链），但 `scripts/sync_lax_report_numbers.py` 的 `INVENTORY` 元组未包含该文件，导致 `_inventory_problems()` 将其报告为 "unlisted committed artifact"。

**修复**：在 `INVENTORY` 中新增一行：
```python
("figures_binding.json", "图-工件-渲染器溯源绑定（sha256 校验链）"),
```

然后重跑 `python scripts/sync_lax_report_numbers.py` 更新 `docs/CASE-LAX-DG.md` 的工件清单表。

## ③ 计算结果（程序化注入）

### 3.1 测试执行

```
python -m pytest tests/test_lax_case_pool.py -v
→ 26 passed, 20.57 s
```

修复前：1 failed (`test_public_report_numbers_match_the_artifact`)
修复后：26 passed

### 3.2 同步脚本自检

```
python scripts/sync_lax_report_numbers.py --check
→ exit 0（无漂移）
```

### 3.3 工件清单更新

| 文件 | 用途 | bytes | sha256[:16] |
|---|---|---|---|
| `metrics.json` | 实测值 + 判据 verdict + 全部容差配置 | 已更新 | 已更新 |
| `profiles.npz` | 最细腿原始量 vs 精确解逐点场 | 已更新 | 已更新 |
| `summary.md` | 由 metrics.json 渲染的表（禁手改） | 已更新 | 已更新 |
| **`figures_binding.json`** | **图-工件-渲染器溯源绑定（sha256 校验链）** | **新增** | **已记录** |
| `figures/lax_profiles.png` | 场量对比图（最细腿 vs 精确解） | 已更新 | 已更新 |
| `figures/lax_convergence.png` | 收敛图（全场与解析区 rel L1 vs N） | 已更新 | 已更新 |
| `figures/time_refinement.png` | dt 细化：固定网格上 L1 误差 vs dt | 已更新 | 已更新 |
| `figures/entropy_wave_order.png` | 熵波实测阶 vs 设计阶 p+1 | 已更新 | 已更新 |
| `figures/lax_acceptance_a0.png` | 验收 A0：逐腿 relative L2/L1 vs 容差 | 已更新 | 已更新 |

## ④ 结果图

本单为**验证欠账修复**类，不产出新物理场图。`docs/CASE-LAX-DG.md` 中的图清单由 sync 脚本自动更新。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **最小修复**：只修改了 `INVENTORY` 元组（加一行），未改动任何计算逻辑或测试断言。
2. **向后兼容**：`figures_binding.json` 是已有工件，只是未被清单收录。修复后清单与实际提交一致。
3. **不改变验收状态**：CASE-LAX-DG 的 A0 verdict 仍为 FAIL（P3/N400 密度 rel L2 = 3.116% > 3%），本单不改变这一诚实结论。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 实测 | 是否 < 3 % |
|---|---|---|---|
| 测试通过率 | 100% | 26/26 PASS | 是 |
| sync --check | 0 problems | exit 0 | 是 |
| A0 verdict（台账） | 精确 Riemann 解 | 3.116% (P3/N400) | 否（honest FAIL，非本单范围） |

**结论**：工件清单漂移已修复，`test_public_report_numbers_match_the_artifact` 恢复 PASS。
CASE-LAX-DG 的台账验收状态（A0 FAIL）保持不变，属求解器精度问题，非本单范围。
