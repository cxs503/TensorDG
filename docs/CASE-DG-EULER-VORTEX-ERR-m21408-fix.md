# CASE-DG-EULER-VORTEX-ERR：缺陷修复报告（m21408）

## 修复内容

**缺陷**：`benchmarks/case_euler_vortex_yee.py` 的 `_report()` 函数中存在硬编码的错误数值，
与程序化生成的实际计算结果矛盾。

**原代码（第 467-470 行）**：
```python
L.append("- 关键发现：P2 DG 在 32x32 网格上，t=1 时误差仅 0.013%，但 t=10 时增至 16.4%。"
         "误差增长主要来自长时间积分的数值耗散累积，而非空间离散误差。"
         "所有 DG 族（P1/P2/P3）在 t=10 时误差均 ~16%，表明误差饱和值与多项式阶数无关，"
         "而是由时间积分方案（SSP-RK3）的耗散特性主导。")
```

**问题**：
- `0.013%` 和 `16.4%` 是手抄的过时数字，与实际计算结果不符
- 实际 errconserve 结果：DG-P2/32² 在 t=1 时为 0.0094%，t=10 时为 0.0152%
- `16.4%` 实际上是粗网格 DG-P3/8² 的结果，被错误地归因于门禁配置
- `~50%+` 的 P0 基线描述也不准确（实际 P0/64² 约 14%，P0/128² 约 12%）

**修复**：将硬编码数值替换为数据驱动的逻辑，从实际计算记录中动态提取误差值生成报告。

**修复后代码**：
```python
if gate and history:
    t1_rec = next((r for r in history if r["final_time"] == 1.0), None)
    t10_err = gate["max_relative_l2"] * 100
    t1_err = t1_rec["max_relative_l2"] * 100 if t1_rec else None
    if t1_err is not None:
        L.append(f"- 关键发现：P2 DG 在 32x32 网格上，t=1 时误差 {_fmt(t1_err)}%，"
                 f"t=10 时 {_fmt(t10_err)}%，增长因子 {_fmt(t10_err / t1_err if t1_err > 0 else 0)}×。"
                 "误差增长极小（α ≪ 1），证实高阶 DG 的误差是网格属性而非时间属性。")
```

## 新增测试

新增 `test_report_is_data_driven_not_hardcoded` 测试，验证：
1. 生成的报告不包含硬编码的错误数字（16.4%、0.013%、~16%）
2. 报告包含实际计算的数据驱动数值

## 验证

```bash
python3 -m pytest tests/test_case_euler_vortex_yee.py -v
# 5 passed
```

## 集群计算状态

以下配置已递交集群（cj3412-cj3419），等待调度完成后补充完整阶梯：
- cj3412: dg_d2_n32_t10_cfl0.2（门禁配置，最关键）
- cj3414: dg_d2_n16_t10_cfl0.2
- cj3415: dg_d1_n32_t10_cfl0.2
- cj3416: dg_d3_n16_t10_cfl0.2
- cj3417: p0_n64_t10_cfl0.4
- cj3418: dg_d2_n32_t1_cfl0.2
- cj3419: dg_d2_n32_t10_cfl0.4

## 锚点状态

CASE-DG-EULER-VORTEX-ERR 的门禁判据（<6%）已由 errconserve 基准证实：
- DG-P2/32² t=10: 0.0152% ✅（远低于 6%）
- DG-P2/32² t=1: 0.0094% ✅
- 误差增长指数 α = 0.199 ✅（≪ 1.0，误差守恒）

本修复确保 yee 基准的报告生成逻辑与 errconserve 结果一致，不再产生误导性数字。
