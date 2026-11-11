# 交差审计报告：m15297 CASE-SHU-OSHER

> 审计任务 m15350 · 对象 m15297（CASE-SHU-OSHER·算例池·激波-熵波干扰·TensorDG）
> 审计时间：2026-11-11 · 审计人：board-audit

## 审计结论

**判级：L2（独立功能实现）**，gate_state=`reopened`，gate_level=2

**核心判定**：m15297 声称 succeeded，但关键验收测试 `test_ladder_artifacts_are_admissible_and_report_the_5pct_gate` **FAIL**（ladder 表为空），完整 DG 阶梯计算在提交时仍在执行（cluster jobs cj1054-1057 仍 queued）。基准算例 CASE-SHU-OSHER **未注册**于 `tensor_benchmark_cases` 表。

## 取证清单

### ① Run Artifacts 三件套核验

| 工件 | 状态 | 详情 |
|---|---|---|
| `shu_osher_results/shu_osher_metrics.json` | ✅ 存在 | 8 个配置完成，rel_L1 均 <3%（最优 0.55%） |
| `shu_osher_results/shu_osher_profiles.npz` | ✅ 存在 | 28KB 剖面数据 |
| `runs/shu_osher_case_pool/metrics.json` | ⚠️ 存在但空 | `ladder=[]`（空表），`dg_self_convergence=[]`（空表） |
| `runs/shu_osher_case_pool/raw/*.npz` | ✅ 存在 | 7 个原始结果文件 |
| `runs/shu_osher_case_pool/profiles.npz` | ✅ 存在 | 合并剖面 |

### ② Benchmark 容差核验

**关键发现**：`tensor_benchmark_cases` 表中 **无** `CASE-SHU-OSHER` 条目（全表仅 15 条，无 CASE- 前缀锚点）。

- 任务目标声明容差 `<5%`
- `shu_osher_metrics.json` 中 8 个配置 rel_L1 均 <3%（最优 0.55%，P1/无限制器/N=150）
- 但主验收测试因 ladder 为空而 FAIL

### ③ Repro_cmd 可执行性

报告列出了复现命令，但：
- `python -m benchmarks.shu_osher_case_pool all-ladder` 在提交时未完成
- cluster jobs cj1054-1057 递交时前方排队 ≈265 个作业，交付时仍 `queued`
- `logs/B.log` 显示部分 DG 运行出现 **负密度**（rho=[-0.0002, 40.8742]），物理违规

### ④ 测试执行结果

| 测试文件 | 结果 | 详情 |
|---|---|---|
| `tests/test_shu_osher_1d.py` | ✅ 9/9 PASSED | 初值/边界/限制器/激波位置等 |
| `tests/test_shu_osher_case_pool.py` | ❌ 6/7 FAILED | `test_ladder_artifacts` 断言 `ladder` 非空 → FAIL |

### ⑤ 报告六段完整性

| 段落 | 状态 |
|---|---|
| ① 案例介绍 | ✅ 完整（物理背景/参数表/RH 解析锚） |
| ② 方法 | ✅ 完整（DG 路径/FV 参考/比较口径） |
| ③ 计算结果 | ⚠️ DG 阶梯表为空（"共 0 个配置"） |
| ④ 收敛性 | ⚠️ 标注"未采集" |
| ⑤ 图 | ✅ 3 张图（但部分数据不完整） |
| ⑥ 误差判定 | ⚠️ 诚实声明了 WITHHELD 和未完成项 |

### ⑥ 实现代码核验

| 文件 | 状态 |
|---|---|
| `tensordg/euler_dg_1d.py` | ✅ 真实 DG 求解器（模态 Legendre + HLLC + SSP-RK3 + WENO 限制器） |
| `benchmarks/shu_osher_1d.py` | ✅ 真实驱动（solve_case 函数） |
| `benchmarks/shu_osher_case_pool.py` | ✅ 算例池驱动 |
| `benchmarks/fv_weno5_1d.py` | ✅ 独立 FV-WENO5 参考求解器 |

## 判级理由

| 级别 | 判定 | 理由 |
|---|---|---|
| L0 文档成果 | ✅ 通过 | 有完整报告 |
| L1 数学/单元验证 | ✅ 通过 | RH 解析锚验证、质量守恒、WENO5 阶数验证 |
| L2 独立功能实现 | ✅ 通过 | 真实代码：输入→计算→输出，8 个配置完成 |
| L3 系统集成 | ⚠️ 部分 | 接入 DG 求解器，但算例池不完整 |
| L4 基准验证 | ❌ 不通过 | 完整算例+误差分析+可复现 → ladder 为空，关键测试 FAIL |
| L5 科研工程 | ❌ 不通过 | 未达到 L4 |

**最终级别 = L2**（最高通过级别）

## 问题清单

| # | 严重度 | 问题 | 证据 |
|---|---|---|---|
| 1 | **BLOCKING** | 关键验收测试 FAIL | `test_ladder_artifacts`: ladder table is empty |
| 2 | **HIGH** | 基准算例未注册 | `tensor_benchmark_cases` 无 CASE-SHU-OSHER 条目 |
| 3 | **HIGH** | 负密度物理违规 | `logs/B.log`: rho=[-0.0002, 40.8742] |
| 4 | **MEDIUM** | 原文表值 WITHHELD | 无外网/无原文 PDF，无法引用 Shu-Osher 1988 表值 |
| 5 | **MEDIUM** | 完整阶梯未完成 | cluster jobs 仍 queued 时提交 |

## 处置建议

1. **reopened**：等待完整 DG 阶梯计算完成后重新提交
2. 修复负密度问题（positivity filter 可能未正确触发）
3. 将 CASE-SHU-OSHER 注册到 `tensor_benchmark_cases` 表
4. 确保 `test_ladder_artifacts` 通过后再标记 succeeded
