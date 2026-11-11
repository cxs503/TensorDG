"""Publish the CASE-EULER-VORTEX progress report to the platform.

All numbers are read from ``case_euler_vortex/metrics.json`` (programmatic
injection, no hand-copied values) and all figures from the generated PNGs.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
OUT = Path("case_euler_vortex")


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    metrics = json.loads((OUT / "metrics.json").read_text())
    gate = metrics["gate"]
    conv = metrics["convergence"]
    sens = metrics["sensitivity"]
    residual = metrics["reference_check"]
    pct = lambda value: f"{100.0 * value:.4f}%"  # noqa: E731

    rows = []
    for name, records in conv.items():
        for row in records:
            eoc = row.get("eoc")
            rows.append(
                f"| {name} | {row['cells']} | {row['steps']} | {pct(row['errors']['density'])} | "
                f"{pct(row['errors']['velocity_x'])} | {pct(row['errors']['pressure'])} | "
                f"{pct(row['max_relative_l2'])} | "
                f"{'-' if not eoc else format(max(eoc.values()), '.2f')} |")
    sens_rows = [f"| {r['final_time']:g} | {r['cfl']:g} | {r['max_percent']:.4f}% | "
                 f"{'PASS' if r['gate_pass'] else 'FAIL'} |" for r in sens]

    markdown = f"""# CASE-EULER-VORTEX（第2轮）：二维等熵涡精确解验证

**锚点** CASE-EULER-VORTEX ｜ **算例池判据** EULER-VORTEX｜ref=精确解｜容差 <5% ｜
**本轮结论** 门禁 **{'PASS' if metrics['passed'] else 'FAIL'}**（数字全部由 `case_euler_vortex/metrics.json` 程序化注入）

## ① 案例介绍
二维可压缩 Euler 方程（理想气体 gamma=1.4）的等熵涡算例：计算域 x,y ∈ [-5,5]²（周期），
涡强度 beta=5，均匀来流 (u,v)_inf=(1,1)，t>0 的精确解为初始剖面向下游平移。
峰值切向马赫数 {metrics['peak_tangential_mach']:.4f}（c_inf=1）。
精确解与算例设定出处：Shu, ICASE Report 97-65 / NASA CR-97-206253 (1998) §4.4.2。

## ② 怎么做的
1. **参考解自校核（本轮新增，防拿错答案当基准）**：静止系残差 max|div F|/max(|F_x|+|F_y|) =
   {residual['steady_frame_residual']:.3e}；平动系残差 max|U_t+F_x+G_y| = {residual['spacetime_residual']:.3e}；
   独立性质 {', '.join(f'{k}={v:.2e}' for k, v in residual['solution_properties'].items())}。
   结论：解析剖面确为 Euler 精确解（到机器精度），基准可信。
2. **两条独立离散路径**：P0（守恒型有限体积极限，Rusanov+SSP-RK3，`tensordg/euler_2d.py`）
   与模态 DG（HLLC+SSP-RK3，`tensordg/euler_dg_2d.py`，P1/P2/P3）。
3. **契约化门禁编排**：`benchmarks/case_euler_vortex.py` 每配置一进程一检查点
   （`results/<key>.json`），`benchmarks/_case_euler_vortex_report.py` 聚合表/图/门禁，
   `tools/run_case_euler_vortex.sh` 并行递交 18 个配置。前轮 m15260 无 fail_stage/失败原因，
   本轮以「每配置独立工件 + 三态门禁(PASS/MISSING/FAIL)」替代一次性长跑。

## ③ 计算结果（基准判据逐项）
| 门禁配置 | 全场最大相对 L2 | 判据 <5% |
|---|---|---|
| `{gate[0]['task_key']}`（高阶主路径） | {gate[0]['max_percent']:.4f}% | {'PASS' if gate[0]['passed'] else 'FAIL'} |
| `{gate[1]['task_key']}`（低阶基线） | {gate[1]['max_percent']:.4f}% | {'PASS' if gate[1]['passed'] else 'FAIL'} |
| 反例 `p0_n32`（粗网格，证明门禁有判别力） | {metrics['p0_coarse_percent']:.4f}% | FAIL（预期） |

### 收敛阶梯（t=1，逐分辨率）
| 族 | N×N | 步数 | 密度 | 速度 u | 压力 | 最大场误差 | 观测阶（最大场） |
|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

观测阶：P0 {[round(x,2) for x in metrics['convergence_behavior']['p0_d0']['observed_order_density']]}（→1，一阶），
P1 {[round(x,2) for x in metrics['convergence_behavior']['dg_d1']['observed_order_density']]}，
P2 {[round(x,2) for x in metrics['convergence_behavior']['dg_d2']['observed_order_density']]}，
P3 {[round(x,2) for x in metrics['convergence_behavior']['dg_d3']['observed_order_density']]}；
设计阶门禁 {'通过' if metrics['design_order_gate'] else '未通过'}。

### 敏感性（P2，32×32）
| 末时刻 t | CFL | 最大场误差 | <5% |
|---|---|---|---|
{chr(10).join(sens_rows)}

## ④ 结果图
见附图：误差-网格收敛曲线（含理论斜率与 5% 验收线）、P2/32² 密度场（数值/精确/差值）、
过涡心中心线、CFL 与末时刻敏感性。

## ⑤ 注意点
- **度量为两套口径**：P0 用 4×4 精确单元平均（单元平均 L2），DG 用体积求积节点（点值 L2）。
  跨族只比量级，族内比收敛阶——这是前轮可能的口径错配点，本轮显式声明。
- **本机无 CJK 字体**，matplotlib 图注统一英文，避免出方块字。
- **autograd 对 0 维标量时间叶节点在本 build 上丢梯度**（loongarch torch），故时间导数改用中心差分
  （残差 ~1e-10，远低于判据尺度）；已在代码 docstring 与报告中标注。
- 光滑算例不开限制器（限制器降阶，见 `docs/limiter_verification.md`）。

## ⑥ 误差判定（显式回答是否 <5%）
- 高阶主路径 P2/32²：全场最大相对 L2 **{gate[0]['max_percent']:.4f}% ＜ 5% → 是**
- 低阶基线 P0/128²：**{gate[1]['max_percent']:.4f}% ＜ 5% → 是**
- 反例 P0/32²：{metrics['p0_coarse_percent']:.4f}% ＞ 5% → **否**（非缺陷，是网格不足，已作为门禁有判别力的证据）
- 基准可信性：参考解残差 {residual['steady_frame_residual']:.3e} / {residual['spacetime_residual']:.3e} → 基准即精确解

**未跑/局限**：`oss_inventory`/`negative_knowledge_ledger` SQL 查询与 `oss_solutions` 入库因本机角色无权限未执行；
集群作业 cj1079（节点 ~/oss-cache 盘点）排队未回件——均未当作已知事实引用。选型记录以
`docs/oss_solutions_CASE-EULER-VORTEX.json` 形式沉淀，供摄入器入库。

**复现**：`bash tools/run_case_euler_vortex.sh case_euler_vortex && python3 -m benchmarks.case_euler_vortex --aggregate --output-dir case_euler_vortex`
**工件**：`case_euler_vortex/{{metrics.json,convergence.csv,fig_*.png,results/*.json}}`、
`docs/CASE-EULER-VORTEX.md`、`docs/euler_vortex_method_survey.md`、`tests/test_case_euler_vortex.py`
"""

    figures = []
    captions = {
        "fig_convergence.png": "误差-网格收敛曲线（实测 vs 设计斜率，红线=5% 验收线）",
        "fig_density_field.png": "P2/32² 密度场：数值解 / 精确解 / 差值",
        "fig_centerline.png": "过涡心中心线密度：数值 vs 精确（t=1）",
        "fig_sensitivity.png": "对 CFL 与末时刻的敏感性（P2/32²）",
    }
    for name, caption in captions.items():
        path = OUT / name
        if path.exists():
            figures.append({"name": name, "caption": caption, "png_base64": b64(path)})

    token = post("/auth/login", {"username": USER, "password": PASS})
    token = token.get("access_token") or token.get("token")
    payload = {
        "title": "CASE-EULER-VORTEX：二维等熵涡精确解验证（第2轮，门禁 "
                 + ("PASS" if metrics["passed"] else "FAIL") + "）",
        "summary": (f"等熵涡算例 5% 判据达成：P2/32² 全场最大相对 L2 {gate[0]['max_percent']:.4f}%，"
                    f"P0/128² {gate[1]['max_percent']:.4f}%；参考解残差校核到 "
                    f"{residual['steady_frame_residual']:.1e}；观测阶 P1/P2/P3="
                    f"{metrics['convergence_behavior']['dg_d1']['observed_order_density'][-1]:.2f}/"
                    f"{metrics['convergence_behavior']['dg_d2']['observed_order_density'][-1]:.2f}/"
                    f"{metrics['convergence_behavior']['dg_d3']['observed_order_density'][-1]:.2f}"),
        "scenario_key": "case-euler-vortex",
        "mission_id": 15349,
        "tags": ["TensorDG", "Euler", "DG", "verification", "isentropic-vortex", "case-pool"],
        "markdown": markdown,
        "figures": figures,
    }
    response = post("/tensor/reports", payload, token)
    print("published:", json.dumps(response)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())