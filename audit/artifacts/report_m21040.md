# 交叉审计：m21003 XVAL-LBM-MASS-EXCHANGE-SELECTOR（TensorDG / TensorLBM）

**裁定：L2（独立功能实现）** · 结论 `request changes`（reopened）· verdict id 872

## 1. 案例介绍

m21003 声称在 TensorDG 的 `TensorLBM` 子包里实现 Maarten-vd-Sande/lbm 的 mass-exchange-selector 模式：
D2Q9 格子上统一的 BGK / MRT / TRT 碰撞方案选择器，并保证质量/动量守恒（head `6000a43`，实现提交 `a43d251`）。
物理验证算例为二维 Poiseuille 通道流（常体力驱动、x 周期、上下无滑移）：

| 参数 | 值 |
|---|---|
| 格子 | D2Q9（9 速度，cs²=1/3） |
| 网格 | Nx × Ny = 64 × 32 |
| 松弛 | τ = 0.8（ω = 1.2500），ν = cs²(τ−½) = 0.1000 |
| 体力 | F = 1e-05（x 方向） |
| 解析解 | u(y) = F·y·(H−y)/(2ν)，H = 31，u_max = 1.201250e-02 |

## 2. 怎么做的（审计方法，非重做）

审计方未参与实现，独立取证：

1. **交付三件套核验**：在交付树 `find` 运行工件；`psql` 读 `tensor_benchmark_cases`（anchor 行数 = 0）；实跑入口 `python3 benchmarks/poiseuille.py` 记录退出码与栈。
2. **独立重跑单元测试**：在交付树的未改动副本上 `python3 -m pytest tests/ -q` → 8 passed, 1 warning in 4.25s。
3. **重算交付配置**：Nx=64, Ny=32, τ=0.8, F=1e-5, n_steps=2000，BGK / TRT / MRT 三方案。
4. **收敛探针**（`audit/probe_poiseuille_convergence.py`）：不改交付代码，只把步数从 2000 续到 16000，检验"11% 是格式误差还是欠收敛"。
5. **方案真伪检验**：float64 场下 BGK 与 MRT 各跑 6000 步，比对解析解（`audit/artifacts/scheme_parity.txt`）。
6. **机器复核预演**：以生态仓 `maturity_gate.judge()` 对 evidence 逐键核，确认 L2 全过、L3/L4 因 `integration_callers` 缺失与 `repro_cmd` 不成立而降级。

## 3. 计算结果

### 3.1 交付配置复算（n_steps=2000，与交付报告同参数）

| 方案 | 审计复算 rel-L2 | 交付报告 | 一致？ |
|---|---|---|---|
| BGK | 10.9985% | ~11% | 是 |
| TRT | 10.9452% | 未报 | 新增 |
| MRT | 入口崩溃 | 未报 | 缺陷 |
| 质量漂移 | 0.0e+00 | <1e-12 | 是 |

入口 `benchmarks/poiseuille.py` 的退出行为：`exit=1`（exit=1）——
根因 `poiseuille.py:79` 用 `torch.ones`（float32）构造 `f_init`，而 `collision.py:58` 的矩变换矩阵是 float64。

### 3.2 收敛探针（BGK，同一份交付代码，只加步数）

| steps | rel-L2 vs 节点中心解析解 | rel-L2 vs 半程反弹壁面解析解 | u_max/解析 |
|---|---|---|---|
| 2000 | 10.9985% | 5.4283% | 0.8525 |
| 5000 | 2.7870% | 3.5681% | 0.9693 |
| 10000 | 2.4492% | 3.8984% | 0.9745 |
| 16000 | 2.4475% | 3.9001% | 0.9745 |

扩散稳态时标 H²/ν ≈ 9610 步，交付的 2000 步只跑到约 20%，因此交付的 11% 主要来自**欠收敛**，不是报告归因的边界一阶精度/Guo 体力缺失。

### 3.3 三方案一致性（float64 场，6000 步）

| 方案 | rel-L2 | u_max/解析 |
|---|---|---|
| BGK | 2.5648% | 0.9727 |
| MRT | 2.5936% | 0.9724 |

两方案相对差 1.1229%，说明碰撞算子是真实现，非 MOCK/STUB。

## 4. 结果图

Poiseuille 速度剖面（交付配置 vs 续算收敛 vs 两种解析参考）+ 误差随步数收敛曲线：
交付点 11.0% 远在 3% 门槛之上，续算到 16000 步后落到 2.45%（其自设指标），曲线收敛平台可见。

## 5. 注意点

1. **基准表无此 anchor**：`tensor_benchmark_cases` 行数 = 0（`ILIKE '%MASS-EXCHANGE%'` 亦 0）→ 本单没有可机器复核的通过判据，L4 无从成立。
2. **交付零运行工件**：交付树无 `benchmarks/results/poiseuille_results.json`，`main()` 在 MRT 分支崩在写盘前；报告中 `figures: []`。
3. **精度结论指标敏感**：节点中心参考 2.45%（<3%）vs 半程反弹壁面参考 3.90%（>3%），跨门槛；交付报告未声明壁面位置约定，故 `<3%` 不可无条件宣称。
4. **无集成调用方**：全仓（`TensorLBM/` 外）对 `tensorlbm|MassExchangeSelector` 引用 0 处 → L3 拒绝。
5. **external anchor 未建立**：无上游 commit / 逐函数对照 / 数值 parity 工件，仅文档声明。
6. **未执行且不得视为通过**：上游 API 数值 parity、`_apply_guo_force`（求解器路径未使用）正确性、>1e5 步长时稳定性、D3Q19/D3Q27。

## 6. 误差判定（逐项 vs 3%）

| 关键量 | 基准 | 实测 | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| 质量守恒漂移 | 0 | 0.0e+00 | ~0 | ✅ 是 |
| Poiseuille rel-L2（交付配置 2000 步） | 解析解 | 10.9985% | 11.00% | ❌ 否 |
| Poiseuille rel-L2（收敛 16000 步，自设指标） | 节点中心解析解 | 2.4475% | 2.45% | ✅ 是（指标敏感） |
| Poiseuille rel-L2（收敛 16000 步，半程反弹参考） | 半程反弹解析解 | 3.9001% | 3.90% | ❌ 否 |
| 方案一致性 \|MRT−BGK\|/BGK | BGK | 1.1229% | 1.12% | ✅ 是 |

**显式回答**：交付配置下关键量（Poiseuille 精度）**不满足 <3%**（11.00%）；同等代码收敛后在其自设指标下满足（2.45%），但换用物理上更严格的壁面约定即不满足（3.90%）。因此"<3%"仅在声明的指标口径下成立——交付报告缺这一声明，是本次不给 L4 的直接原因之一。质量守恒与三方案一致性满足 <3%。

## 7. 判级与返修

L0 文档 < L1 单元验证（8 项物理不变量，独立重跑通过）< **L2 独立功能实现（本次裁定）** ；
L3 拒绝（零调用方）；L4 拒绝（无注册基准 + 工件缺失 + 入口崩溃 + 交付配置 11%）。

返修清单：修 `poiseuille.py:79` dtype 与 `main()` 的逐方案容错并落盘；基准改收敛判据并声明壁面约定；
在 `tensor_benchmark_cases` 注册该 anchor（metric=Poiseuille rel-L2, tolerance=0.03）；补 external anchor 对照；
接入一个真实调用方后方可申报 L3；补图。

## 复现命令

```bash
cd audit/refs/TensorLBM_m21003 && python3 -m pytest tests/ -q     # 8 passed, 1 warning in 4.25s
python3 benchmarks/poiseuille.py                                   # exit=1 (MRT dtype)
cd ../.. && python3 audit/probe_poiseuille_convergence.py           # 收敛探针
python3 audit/make_figures.py && python3 audit/insert_verdict.py
```
