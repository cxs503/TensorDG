# 交差审计：m15349 CASE-EULER-VORTEX（TensorDG 等熵涡平移）

**审计单** m15389 ｜ **被审计单** m15349（succeeded，2026-11-11）｜ **审计人** board-audit
**裁定** L4（基准验证级）｜ verdict 行 #302（tensor_maturity_verdicts，gate_state=pending 待 maturity_gate 机器复核）

## ① 案例介绍（物理背景/几何/参数）

二维可压缩 Euler（gamma=1.4）等熵涡平移算例：域 [-5,5]² 周期，beta=5，(u,v)_inf=(1,1)，
t=1 时精确解=初始剖面向下游平移。基准出处 Shu, ICASE Report 97-65 / NASA CR-97-206253 (1998) §4.4.2。
算例池判据（mission goal 原文）：ref=精确解，容差 <5%（各原始变量相对 L2）。

## ② 怎么做的（审计方法/流程/工具链）

1. **取证**：读 m15349 工作树 /root/tp-agent/work/m15349（case_euler_vortex/ 全量工件）、
   tensor_reports#14438（六段报告）、tensor_dev_missions#15349（goal+result）。
2. **基准裁判**：`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-EULER-VORTEX'`
   ——**首查为空行**（基准表缺该锚点，见 §⑤ 发现 F1）；判据改以 mission goal 内算例池判据为裁判，
   并由本审计补登基准表行（status=passed）。
3. **逐项核三件套**：run_artifacts 文件存在性、results/*.json 数值 vs 报告表格、repro_cmd 实跑。
4. **独立复算**：pytest 全量重跑 + 两个门禁配置单点复跑 + 聚合管线重跑，逐位比对。

## ③ 计算结果（审计复算数字，程序化取自工件，零手抄）

| 配置 | 报告值 | 工件 results/*.json | 审计复跑 | 判据 <5% |
|---|---|---|---|---|
| dg_d2_n32_t1_cfl0.2（高阶主路径） | 0.0127% | 0.012651985914081664% | 0.0127%（bit 一致） | PASS |
| p0_n128_t1_cfl0.4（低阶基线） | 2.8170% | 2.816997317289924% | 未单跑（183 步，工件+聚合背书） | PASS |
| p0_n32_t1_cfl0.4（反例，证门禁有牙） | 7.7999% | 7.799898324635561% | 7.7999%（bit 一致，gate=FAIL 如预期） | FAIL（预期） |

- 测试：`python3 -m pytest tests/test_case_euler_vortex.py -q` → **6 passed in 41.86s**（审计重跑）。
- 聚合管线重跑：`python3 -m benchmarks.case_euler_vortex --aggregate` → 重生成 metrics.json 与
  提交版**逐位相同**（`metrics identical: True`），overall=PASS。
- 参考解自校核（工件 exact_solution_residual.json）：静止系残差 3.169e-16、平动系 1.324e-10、
  等熵律/无散旋流/径向平衡 0~1.1e-15 → 基准即真精确解。

## ④ 结果图

- 图 1：误差-网格收敛曲线（含理论斜率与 5% 验收线）——`case_euler_vortex/fig_convergence.png`
- 图 2：P2/32² 密度场（数值/精确/差值）——`case_euler_vortex/fig_density_field.png`
（审计单不重算场量；图直接取自被审计工件，与 metrics.json 同源。）

## ⑤ 注意点（踩坑/局限/适用范围）

- **F1（已修复）**：`tensor_benchmark_cases` 原无 CASE-EULER-VORTEX 行——「基准表=唯一裁判」
  的裁判席空缺。本审计按 mission goal 内算例池判据（<5%/场，ref=精确解，Shu 1998 §4.4.2）补登
  （status=passed）。教训：算例池锚点交付时须同步登记基准表。
- **F2（已修复）**：/root/tensor-ecosystem/TensorDG 本地检出停在 2513515（早于 m15349 的
  merge #283），maturity_gate 的 integration_callers grep 会因检出陈旧而误判 L3 缺失。
  本审计 ff 至 origin/main 17abf8e（含 #283/#284）。
- **F3（低）**：P0 与 DG 两条路径误差口径不同（单元平均 L2 vs 点值 L2），跨族只比量级——
  报告已显式声明，不构成缺陷。
- **F4（低）**：m15349 自述 oss_inventory/oss_solutions 入库因权限未执行，选型记录以
  docs/oss_solutions_CASE-EULER-VORTEX.json 沉淀——未当作已知事实引用，如实。
- 审计未重跑全量 18 配置（p0_n128 等长配置）：以「两门禁配置单点 bit 级复现 + 聚合管线
  逐位复现 + 6/6 测试通过」背书，未跑项如实列出。

## ⑥ 误差判定（显式回答是否 <5%）

| 配置 | got（max 相对 L2） | 容差 | <5%? |
|---|---|---|---|
| dg_d2_n32_t1_cfl0.2 | 0.0127% | 5% | **是** |
| p0_n128_t1_cfl0.4 | 2.8170% | 5% | **是** |
| p0_n32_t1_cfl0.4（反例） | 7.7999% | 5% | 否（预期 FAIL，证门禁有牙） |

**判级 L4**（基准验证：完整算例+误差分析+可复现），不判 L5：功能实质=一个标准基准算例的
契约化验证台（两条独立离散路径+收敛阶+敏感性+参考解自校核），非科研工程级成果。
evidence 全键真实指针已落 tensor_maturity_verdicts#302，待 maturity_gate 机器复核
（confirmed=消账 / reopened=返修）。

**复现**：`cd /root/tp-agent/work/m15349 && bash tools/run_case_euler_vortex.sh case_euler_vortex
&& python3 -m benchmarks.case_euler_vortex --aggregate --output-dir case_euler_vortex`
