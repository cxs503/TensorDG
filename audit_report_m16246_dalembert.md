# 交差审计报告：m16246 CASE-DG-DALEMBERT-ELASTIC-1D

> 审计人：board-audit (m16317) · 日期：2026-11-12 · 对象：m16246（TensorDG 一维弹性杆 d'Alembert 精确解验证）

## ① 取证摘要

| 验收项 | 状态 | 证据 |
|---|---|---|
| run_artifacts | ✅ 存在 | `validation/elastic_1d_dalembert/` 含 5 个工件（JSON×3, PNG×1, CSV×1），均有 sha256 |
| benchmark ref vs got | ✅ 9/9 PASS | 最大误差 0.602% << 2% 容差 |
| repro_cmd | ✅ 可执行 | `python -m benchmarks.elastic_1d_dalembert --output-dir validation/elastic_1d_dalembert` |
| 测试套件 | ✅ 21/21 PASS | test_elastic_1d_dalembert.py (12) + test_elastic_rod_case.py (9) |
| 报告六段 | ✅ 齐全 | docs/CASE-DG-DALEMBERT-ELASTIC-1D.md 含 ①~⑥ 全部 |

## ② 基准裁判（benchmark_ref_vs_got.json）

| # | 量 | ref | got | 误差% | 判定 |
|---|---|---|---|---|---|
| 1 | σ L2 误差 (N=400,P2) | 0 | 0.000284% | 0.000284% | PASS |
| 2 | 峰值应力 σ_max | 0.5 | 0.499999 | 0.000108% | PASS |
| 3 | 右行波位置 x_R | 0.7 | 0.700000 | 4.4e-07% | PASS |
| 4 | 左行波位置 x_L | 0.3 | 0.300000 | 1.0e-06% | PASS |
| 5 | 波速 c | 1.0 | 0.999923 | 0.0077% | PASS |
| 6 | 脉冲到达时间 (x=0.8) | 0.3 | 0.298194 | 0.602% | PASS |
| 7 | 一阶固有频率 f1 | 0.25 | 0.250147 | 0.0586% | PASS |
| 8 | 固定端应力放大比 | 2.0 | 2.000003 | 0.000159% | PASS |
| 9 | 弹性能守恒 | 0.031333 | 0.031333 | 3.2e-06% | PASS |

**最大误差 0.602% << 2% 容差，全部通过。**

## ③ 代码实质审查

- `tensordg/elastic_1d.py` (427 行)：完整 DG 求解器，含精确解、上风通量、特征边界、SSP-RK3/RK4 时间推进——**非 STUB/MOCK**
- `tensordg/cases/elastic_rod_1d.py` (278 行)：算例定义 + anchor_table + 参数表——**非 STUB/MOCK**
- `benchmarks/elastic_1d_dalembert.py` (554 行)：收敛扫描 + 驻波 + 脉冲 + 图 + JSON 输出
- 收敛阶验证：P2 → EOC≈3.003（理论 p+1=3），P1 → EOC≈2.0（理论 p+1=2）——**数学验证通过**

## ④ 判级（L0-L5）

| 级别 | 标准 | 是否满足 |
|---|---|---|
| L0 | 文档成果 | ✅ |
| L1 | 数学/单元验证 | ✅ (收敛阶、均匀态通量退化) |
| L2 | 独立功能实现 | ✅ (输入→计算→输出，427+278 行真代码) |
| L3 | 系统集成 | ✅ (tensordg/__init__.py 导入，4 个下游调用者) |
| L4 | 基准验证 | ✅ (9 锚点全通过 + 误差分析 + repro_cmd + 图) |
| L5 | 科研工程成果 | ❌ (单域 1D，无多物理场耦合/独立外部验证) |

** verdict: L4 **

## ⑤ 发现与建议

### 非阻塞问题
1. `tensordg/__init__.py` 中 elastic_1d 函数已 import 但未列入 `__all__`，导致 `from tensordg import solve_elastic_1d` 不可用（需 `from tensordg.elastic_1d import ...`）。建议补入 `__all__`。
2. 脉冲到达时间误差 0.602% 为最大项，源于抛物线插值细化（21 快照），属方法限制而非算法缺陷。

### 无阻塞正确性问题
- 所有 21 项测试通过
- 故障注入测试（边界特征对调→能量爆炸）证明门禁有牙齿
- 能量漂移 < 3.2e-06%（N=400, P2）

## ⑥ 落表

```sql
INSERT INTO tensor_maturity_verdicts
  (repo, scope, mission_id, level, evidence, verdict_by)
VALUES
  ('TensorDG', 'm16246 CASE-DG-DALEMBERT-ELASTIC-1D', 16317, 4, '<JSON>', 'board-audit');
```

**裁定：confirmed（消账）**
