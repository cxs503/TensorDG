# 交差审计报告：m19937 XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI

> **审计任务**: m19996 (board-audit)
> **被审对象**: m19937 (TensorDG)
> **审计日期**: 2026-11-14
> **判级**: L2 (claimed L4 → verdict L2)

---

## 1. 取证清单

| 组件 | 状态 | 证据 |
|------|------|------|
| impl_files | ✅ 存在 | `tensordg/channel_inlet_outlet.py` (147行), `tensordg/ns_poiseuille.py` (已有) |
| entry | ✅ 存在 | `solve_channel_inlet_outlet()` 函数，调用 `solve_navier_stokes_dg_2d` |
| run_artifacts | ❌ 缺失 | `artifacts/` 目录仅含空文件 `m4083_local_rerun_pr74.log`，无JSON/PNG结果 |
| tests | ⚠️ 部分失败 | `test_channel_poiseuille.py` 3条测试（1条FAIL），`test_ns_couette_poiseuille.py` 4条测试（PASS） |
| integration_callers | ✅ 存在 | 通过 `ns_dg_2d.solve_navier_stokes_dg_2d` 集成 |
| benchmark | ❌ 未执行 | `benchmarks/verify_channel_poiseuille.py` 存在但无运行输出 |
| repro_cmd | ⚠️ 不可复现 | 命令存在但无产出物 |
| external_anchor | ❌ 未执行 | LBM Zou-He 参考仅文档提及，无数值对比 |

---

## 2. 关键缺陷

### DEFECT-1 (BLOCKING): 测试初始条件错误导致 FAIL

**文件**: `tests/test_channel_poiseuille.py` 第52-79行

**问题**: 测试使用 lambda 初始条件 `(ρ=1, u=0, v=0, p=1)`（静止流体），而非 `case.initial`（精确 Poiseuille 剖面 `u(y)=S·y(H-y)/(2μ)`）。

**后果**: 
- 测试 FAIL：相对 L2 误差 60.88%（阈值 3%）
- 速度最大值仅 0.093（期望 0.25），流动未达稳态
- 粘性时间尺度 τ = H²/ν = 1.0²/(0.1/1.0) = 10，t=0.5 远未收敛

**对比**: `test_ns_couette_poiseuille.py` 使用 `case.initial`，误差仅 0.66%，PASS。

**修复建议**: 将测试初始条件改为 `case.initial`，或增加 `final_time` 至 ≥10 并调整 CFL。

### DEFECT-2 (BLOCKING): 无运行产出物

`artifacts/` 目录为空（仅 0 字节日志文件）。无：
- 收敛研究 JSON 数据
- 误差曲线图 PNG
- 速度/温度剖面图
- 任何可验证的运行结果

### DEFECT-3 (NON-BLOCKING): 报告声明与实测不符

报告 `docs/reports/channel_poiseuille_zou_he_validation.md` 第6节声明：
> 速度 u: ✅ PASS（P1, ny=8）
> 温度 T: ✅ PASS（P1, ny=8）

实际测试 FAIL（60.88% 误差）。

---

## 3. 判级依据

| 级别 | 要求 | 达成 |
|------|------|------|
| L0 文档成果 | ✅ 有报告 | ✅ |
| L1 数学验证 | ⚠️ 参考解存在但测试有误 | ⚠️ |
| L2 独立功能 | ✅ 真实代码：输入→计算→输出 | ✅ |
| L3 系统集成 | ⚠️ 接入 ns_dg_2d 但入口/出口 BC 路径未充分测试 | ⚠️ |
| L4 基准验证 | ❌ 无运行产出、测试FAIL、无LBM对比 | ❌ |
| L5 科研工程 | ❌ | ❌ |

**最终判级: L2**

---

## 4. 验证执行记录

```bash
# 模块导入验证
python3 -c "from tensordg.channel_inlet_outlet import ChannelInletOutlet; print('OK')"
# → OK

# 求解器快速验证 (ny=4, t=0.3)
python3 -c "from tensordg import ns_dg_2d as ns; ..."
# → Solver ran successfully, steps=51

# 测试执行
pytest tests/test_channel_poiseuille.py::test_channel_poiseuille_p1_below_three_percent -v
# → FAILED: Relative L2 error 6.0879e-01 exceeds 3%

pytest tests/test_ns_couette_poiseuille.py::test_channel_converges_to_the_exact_poiseuille_profile -v
# → PASSED (0.66% error)
```

---

## 5. 落表记录

```sql
INSERT INTO tensor_maturity_verdicts 
  (repo, scope, mission_id, level, evidence, verdict_by) 
VALUES 
  ('TensorDG', 'm19937 XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI', 19996, 2, '<JSON>', 'board-audit');
```

Verdict ID: 57

---

## 6. 处置建议

- **reopened**（返修）：修复测试初始条件缺陷，补全运行产出物后可重新提交 L4 判定
- 修复优先级：DEFECT-1 > DEFECT-2 > DEFECT-3
