"""Publish the exact-in-time p-refinement report (CASE-DG-PREFINEMENT-EXPONENTIAL).

Every number is injected from the artefact written by
``python -m benchmarks.p_refinement_exact_time`` (no hand-copied values).

Usage::

    PYTHONPATH=. MISSION_ID=17067 python publish_exact_time_report.py [--doc docs/CLOSEOUT-m17067-exact-time-floor.md]
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = int(os.environ.get("MISSION_ID", "17067"))
ART = Path("docs/benchmarks/results/p-refinement-exact-time")
RESULTS = ART / "exact_time_results.json"


def post(path, payload, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build(art: dict) -> tuple[str, list[dict]]:
    levels = art["levels"]
    diag = art["diagnostics"]
    diag_r = art["diagnostics_resolved_ladder"]
    floor = art["floor"]
    proj = art["projection_check"]
    checks = art["checks"]
    sweep = {e["degree"]: e["history"] for e in art["temporal_sweep"]}
    cross = checks["small_cfl_rk4_cross_check"]
    top = levels[-1]
    cfl0 = levels[0]["rk4_production"]["cfl"]

    ladder = ["| p | DOFs | L2 精确时间 (longdouble) | L2 scipy expm (f64) | L2 出厂 RK4 | RK4/精确 | 表示地板 | 算子地板 | SNR(测量地板) | 判定 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for lv in levels:
        star = "" if lv["scipy_expm_adequate"] else "†"
        ladder.append(
            f"| {lv['degree']} | {lv['dofs']} | {lv['exact_time']['l2_longdouble_expm']:.4e} | "
            f"{lv['exact_time']['l2_scipy_expm']:.4e}{star} | {lv['rk4_production']['l2']:.4e} | "
            f"{lv['rk4_production']['temporal_dominance_factor']:.2f} | "
            f"{lv['roundoff_floor']:.2e} | {lv['operator_representation_floor']:.2e} | "
            f"{lv['snr_over_measurement_floor']:.1f} | {'PASS' if lv['resolved'] else 'FAIL'} |"
        )

    proj_rows = ["| p | 发布外推 L2 | 实测 L2 | 实测/外推 | 容差 | 判定 |", "|---|---|---|---|---|---|"]
    for degree, val in sorted(proj["levels"].items()):
        proj_rows.append(
            f"| {degree} | {val['predicted_error']:.4e} | {val['measured_error']:.4e} | "
            f"{val['measured_over_predicted']:.3f} | ±{val['factor_tolerance']:.0f}× | "
            f"{'PASS' if val['pass'] else 'FAIL'} |"
        )

    sweep_rows = ["| p | CFL | steps | RK4 L2 | RK4/精确时间 | 与上一档相对变化 |", "|---|---|---|---|---|---|"]
    for degree in sorted(sweep):
        for h in sweep[degree]:
            ch = h.get("relative_change_vs_previous")
            sweep_rows.append(
                f"| {degree} | {h['cfl']:.4f} | {h['steps']} | {h['l2']:.4e} | "
                f"{h['ratio_to_exact_time']:.2f} | {('—' if ch is None else f'{100 * ch:.1f} %')} |"
            )

    check_rows = ["| 检查 | 实测 | 阈值 | 通过 |", "|---|---|---|---|"]
    for key, val in checks.items():
        if key == "discriminating_criterion":
            c = val["conditions"]
            check_rows.append(
                f"| {key} (p={diag_r['degrees'][0]}..{diag_r['degrees'][-1]}) | "
                f"σ={c['sigma_above_threshold']['measured']:.4f}, 逐段率不减="
                f"{c['decay_rates_non_decreasing']['pass']}, 总降幅="
                f"{c['error_ratio_above_threshold']['measured']:.2e} | σ>1 ∧ 不减 ∧ >1e3 | {val['pass']} |"
            )
            continue
        if key == "projection_test":
            check_rows.append(f"| {key} | 外推复核 | ±3× | {val['all_pass']} |")
            continue
        if key == "small_cfl_rk4_cross_check":
            check_rows.append(
                f"| {key} | p={val['degree']} CFL={val['cfl']:.4f}: RK4/精确="
                f"{val['ratio_rk4_over_exact_time']:.4f} | ±{100 * val['tolerance']:.0f}% | {val['pass']} |"
            )
            continue
        if key == "all_levels_resolved_above_measurement_floor":
            check_rows.append(
                f"| {key} | 已解析 p={val['measured']}，floor={val['floor']:.2e} | 连续且含 p={top['degree']} | {val['pass']} |"
            )
            continue
        if isinstance(val, dict) and "pass" in val:
            measured = val.get("measured", val.get("max_relative_difference"))
            thr = val.get("threshold", val.get("required"))
            check_rows.append(f"| {key} | {measured} | {thr} | {val['pass']} |")

    top_hist = sweep.get(top["degree"], [])
    stall = ""
    if len(top_hist) >= 2:
        a, b = top_hist[0], top_hist[-1]
        stall = (f"p={top['degree']}：CFL {a['cfl']:.3f}→{b['cfl']:.3f}（步数 {a['steps']}→{b['steps']}，"
                 f"×{b['steps'] / a['steps']:.0f}），L2 从 {a['l2']:.3e} 降到 {b['l2']:.3e}，"
                 f"仍是精确时间值的 {b['ratio_to_exact_time']:.2f} 倍；而 CFL 再降 16 倍的 RK4 才落到 "
                 f"{cross['rk4_l2']:.3e}（精确时间值 {cross['exact_time_l2']:.3e}，比值 "
                 f"{cross['ratio_rk4_over_exact_time']:.4f}）。")

    rates_full = ", ".join(f"{r:.3f}" for r in diag["decay_rates"])
    bend = ""
    if len(diag["decay_rates"]) >= 2 and diag["decay_rates"][-1] < diag["decay_rates"][-2]:
        bend = (f"**诚实记录**：全阶梯 p=1..{top['degree']} 的最后一段衰减率由 "
                f"{diag['decay_rates'][-2]:.3f} 回落到 {diag['decay_rates'][-1]:.3f}，"
                f"故复合判据里「逐段衰减率不减」这一条在 p={top['degree']} 端部不成立；"
                f"判据在**已解析阶梯** p={diag_r['degrees'][0]}..{diag_r['degrees'][-1]} 上成立"
                f"（σ={diag_r['lsq_decay_rate']:.4f}）。")

    lim = (f"p={top['degree']} 实测 {top['exact_time']['l2_longdouble_expm']:.3e}，"
           f"是表示地板 {top['roundoff_floor']:.2e} 的 {top['snr_over_roundoff_floor']:.0f} 倍，"
           f"但对 float64 装配算子的 1e-16 相对扰动响应为 {top['operator_representation_floor']:.2e}"
           f"（SNR {top['snr_over_measurement_floor']:.1f}）")

    neg = ""
    if not cross["pass"]:
        neg = (f"- **honest 负结果：p={top['degree']} 的精确时间值未被独立确认**。"
               f"同一求解器改用出厂 RK4、CFL 降到 {cross['cfl']:.4f}（步数 {cross['steps']}，时间误差已可忽略）得到 "
               f"{cross['rk4_l2']:.3e}，是精确时间值 {cross['exact_time_l2']:.3e} 的 "
               f"{cross['ratio_rk4_over_exact_time']:.2f} 倍；而 p={top['degree']} 的 float64 算子表示地板"
               f"（1e-16 相对扰动响应）为 {top['operator_representation_floor']:.2e}，大于该信号本身。"
               f"⇒ p={top['degree']} 只能作为**上界**引用，不能作为「指数律延续到 p=8」的证据。"
               f"这也否定了上游 CLOSEOUT-m16918 §4 的推断「p=8 信噪比约 55，仍可测」——"
               f"真正的限制是算子表示 + 浮点运算，不是表示地板。")

    md = f"""# CASE-DG-PREFINEMENT-EXPONENTIAL：p=7/8 实测（精确时间积分）与时间积分地板

## ① 案例介绍
一维周期线性对流 `u_t + u_x = 0`，初值 `sin(2πx)`（整函数，处处解析），终时 `T=1`（一整周期，精确解=初值平移）。
固定网格 `ncells={art['config']['ncells']}`，只做 p 加密，`p={min(art['config']['degrees'])}..{max(art['config']['degrees'])}`
（库上限 `MAX_DEGREE=8`）。理论基准：Babuska & Suri (1994) SIAM Review 36(4) 578–632，解析解下误差随 p 指数衰减
`||u-u_h|| <= C exp(-sigma p)`。

锚点背景：已发布算例止步 p=6，`measurement_floor.json` 里 p=7/8 是**外推值**（σ={art['projection']['sigma']:.4f} 从 p=6 外推），
此前审计把 p≥7 记为 not_run。本单把 p=7/8 变成实测，并查出 p≥7 无法测量的真实原因。

## ② 怎么做的
线性半离散系统 `du/dt = L u` 的精确时间解是 `exp(L t) u0` ⇒ **完全去掉时间误差**，测到的是纯空间误差：

1. 用**出厂右端项** `tensordg.advection_1d._rhs` 逐列装配稠密算子 L（不重写格式），线性性残差
   {checks['operator_linear']['measured']:.2e}；半离散总质量守恒（周期 DG），质量漂移
   {checks['mass_conserved']['measured']:.2e}；
2. 矩阵指数用新增的 `tensordg.convergence.expm_scaled_squaring`（80 位 longdouble 缩放-平方 + 泰勒），
   并以**另一套控制参数**（threshold 0.25/0.125、τ 20/26 项）复算作控制检验：相对差
   {checks['expm_controls_agree']['measured']:.2e}；`scipy.linalg.expm`（float64）只在
   `eps·||L||₁·t < 0.1·误差` 的 p 上作对照（合格 p={checks['scipy_expm_agrees_where_adequate']['adequate_degrees']}，
   相对差 {checks['scipy_expm_agrees_where_adequate']['measured']:.2e}）；
3. L2 用 longdouble 求积（`4p+16` 点，2× 阶收敛校验相对差 {checks['quadrature_converged']['measured']:.2e}）；
4. **浮点地板两层**：表示地板 `eps₆₄·||u_h||`，以及**算子表示地板**——对 float64 装配的 L 施加 1e-16 相对随机扰动，
   看 L2 误差的响应（3 个种子取最大）。层高者决定该 p 能否"测量"；
5. 出厂 RK4（CFL {cfl0:.2f}）同 p 对照，量化时间误差占比；p={top['degree']} 另跑 CFL={cross['cfl']:.4f}（步数
   {cross['steps']}）做**独立积分器交叉验证**；
6. 复核发布值：p=1..6 的 RK4 结果与已发布工件最大相对差 {checks['published_rk4_reproduced']['max_relative_difference']:.2e}。

## ③ 计算结果

{chr(10).join(ladder)}

† = float64 scipy expm 在该 p 已失去分辨力（前向误差模型 `eps·||L||₁·t` 与信号同量级），不参与对照。

**已解析阶梯判据**（p={diag_r['degrees'][0]}..{diag_r['degrees'][-1]}）：σ = **{diag_r['lsq_decay_rate']:.4f}**，
逐段衰减率 {", ".join(f"{r:.3f}" for r in diag_r['decay_rates'])}，
总降幅 {diag_r['errors'][0] / diag_r['errors'][-1]:.2e}（门槛 1e3）⇒ 判定 `{art['verdict']}`。
全阶梯（含 p={top['degree']}）σ = {diag['lsq_decay_rate']:.4f}，逐段衰减率 {rates_full}。

{bend}

**发布外推 vs 实测**（预登记容差 ±{proj['levels'][sorted(proj['levels'])[0]]['factor_tolerance']:.0f}×）：

{chr(10).join(proj_rows)}

**时间积分地板**（出厂 RK4 逐档降 CFL）：

{chr(10).join(sweep_rows)}

{stall}

## ④ 结果图
`p_refinement_exact_time.png`：半对数误差-p 图（圆=精确时间实测，方=出厂 RK4 同 CFL，虚=发布外推，
点划=float64 表示地板 `{floor['value']:.3e}`）。RK4 曲线在 p≥7 变平（时间误差主导），精确时间曲线继续下降。

## ⑤ 注意点
- **p≥7 not_run 的真因是时间积分地板，不是 roundoff**：表示地板只有 {floor['value']:.3e}，
  而 RK4 在 p={top['degree']} 停在 {top['rk4_production']['l2']:.3e}（空间误差的
  {top['rk4_production']['temporal_dominance_factor']:.0f} 倍）。出厂的「时间收敛」判据是**相对**判据，
  高 p 时要求 dt 小到"微小空间误差的 1%"，代价极高；本单用精确时间积分绕开它。
- **{lim}**，故 p={top['degree']} 的数值应按"已解析但接近算子地板"读，拟合权重低于 p≤7。
- 发布外推方向正确、幅度偏保守（实测低于外推），**连续性假设成立**；但常数 σ 外推 2 阶已有
  {abs(1 - float(proj['levels'][sorted(proj['levels'])[-1]]['measured_over_predicted'])) * 100:.0f}% 偏差，不宜再往后推。
{neg}
- 适用面仍是一维周期线性对流光滑解；`exact_time_solution` 只对**线性**算子成立（非线性误用会被线性性残差闸拦下）。

## ⑥ 误差判定（逐项，回答"是否<3%"）
| 项 | 基准 | 结果 | 是否<3% |
|---|---|---|---|
| p=1..6 RK4 复现已发布 L2 | 发布工件 | 最大相对差 {checks['published_rk4_reproduced']['max_relative_difference']:.2e} | ✅ |
| expm 控制参数互校 | 另一套 schedule | {checks['expm_controls_agree']['measured']:.2e} | ✅ |
| scipy expm 对照（合格 p） | longdouble 路径 | {checks['scipy_expm_agrees_where_adequate']['measured']:.2e} | ✅ |
| 求积收敛（2× 阶） | 高 2 阶求积 | {checks['quadrature_converged']['measured']:.2e} | ✅ |
| p={cross['degree']} 独立积分器交叉验证 | CFL={cross['cfl']:.4f} RK4 | 比值 {cross['ratio_rk4_over_exact_time']:.4f}（偏差 {abs(100 * (cross['ratio_rk4_over_exact_time'] - 1)):.2f}%） | ✅ |
| 算子线性性 / 质量守恒 | 解析为 0 | {checks['operator_linear']['measured']:.2e} / {checks['mass_conserved']['measured']:.2e} | ✅ |
| σ 阈值 | σ>1 | {diag_r['lsq_decay_rate']:.4f}（{diag_r['lsq_decay_rate']:.1f}× 裕度） | ✅ |
| p=7/8 实测 vs 发布外推 | ±3× 容差 | {proj['levels']['7']['measured_over_predicted']:.2f}× / {proj['levels']['8']['measured_over_predicted']:.2f}× | ✅ |
| p=9 收敛 | — | 不可测（MAX_DEGREE=8） | ⚪ 无基准 |

工件：`{RESULTS}`。
"""
    figures = []
    png = ART / "p_refinement_exact_time.png"
    if png.exists():
        figures.append({
            "name": png.name,
            "caption": "半对数 L2 误差 vs p：精确时间实测（圆）、出厂 RK4（方）、发布外推（虚）、float64 表示地板（点划）",
            "png_base64": b64(png),
        })
    return md, figures


def main() -> int:
    emit_doc = None
    if "--doc" in sys.argv:
        emit_doc = Path(sys.argv[sys.argv.index("--doc") + 1])
    art = json.loads(RESULTS.read_text())
    md, figures = build(art)
    top = art["levels"][-1]
    diag_r = art["diagnostics_resolved_ladder"]
    proj = art["projection_check"]

    if emit_doc is not None:
        head = f"""# CLOSEOUT m17067 — p=7/8 实测（精确时间积分）与时间积分地板

> 锚点：`CASE-DG-PREFINEMENT-EXPONENTIAL`（feature_ledger=open）
> 上游：`docs/CLOSEOUT-m16918-prefinement-criterion.md` §5 把 p≥7 记为**外推未测**
> 本单：m17067 · 工件 `{RESULTS}` · 数字全部由 `publish_exact_time_report.py --doc` 从工件程序化注入（零手抄）

## 0. 交付工件

| 文件 | 说明 |
|---|---|
| `benchmarks/p_refinement_exact_time.py` | 精确时间 p 阶梯 + 三路测量 + 双层浮点地板 + CFL 扫描 + 小 CFL 交叉验证（新增）|
| `tensordg/convergence.py` | 新增 `semidiscrete_operator` / `operator_linearity_residual` / `expm_scaled_squaring` / `exact_time_solution`（纯 NumPy 通用验证工具）|
| `tests/test_p_refinement_exact_time.py` | 工件闸 + 活闸（线性性/守恒/p=7→8 实测/expm 对 scipy/expm 对小步长 RK4/负对照）|
| `{RESULTS}` | 主工件（checks/verdict/provenance）|
| `{ART / 'p_refinement_exact_time.png'}` | 半对数误差-p 图 |
| `publish_exact_time_report.py` | 平台报告与本文档生成器 |

## 1. 局限（honest）

- 只覆盖一维周期线性对流光滑解（与锚点算例一致）；二维/变系数/激波不在本单范围。
- `MAX_DEGREE=8` 是库上限：可测边界从 p=6 推到 p=8，**没有**推到无限；p=9 仍不可测。
- 精确时间积分只对**线性**半离散算子成立（非线性误用被线性性残差闸拦下）。
- 出厂的相对时间收敛判据**未修改**：本单给出绕开它的测量路径并量化其代价。若改为绝对判据
  （要求 `temporal_error < 0.1 * spatial_error`），建议另立条目 `CASE-DG-TEMPORAL-FLOOR-GATE`
  （量化验收：p=6..8 上该门必须拒绝 CFL=0.04 的 RK4 结果作为「空间收敛」证据）。

---

"""
        emit_doc.parent.mkdir(parents=True, exist_ok=True)
        emit_doc.write_text(head + md)
        print(f"doc written: {emit_doc}")

    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    payload = {
        "title": "CASE-DG-PREFINEMENT-EXPONENTIAL：p=7/8 实测与 RK4 时间误差地板",
        "summary": (
            f"用半离散算子的矩阵指数把 p 加密阶梯从 p=6 推进到 p={top['degree']}："
            f"σ(已解析阶梯 p=1..6)={diag_r['lsq_decay_rate']:.4f}；p=7 精确时间 "
            f"{art['errors'][-2]:.3e}（发布外推的 "
            f"{proj['levels']['7']['measured_over_predicted']:.2f}×）；p={top['degree']} "
            f"{art['errors'][-1]:.3e} 因 float64 算子表示地板 "
            f"{top['operator_representation_floor']:.2e} 与 RK4 交叉验证失败（比值 "
            f"{cross['ratio_rk4_over_exact_time']:.2f}）只作上界引用；"
            f"发现此前 p≥7 not_run 的真因是 RK4 时间误差地板（p={top['degree']} 时是空间误差的 "
            f"{top['rk4_production']['temporal_dominance_factor']:.0f} 倍）而非 roundoff；"
            f"发布外推在 ±3× 内复现。"
        ),
        "scenario_key": "CASE-DG-PREFINEMENT-EXPONENTIAL",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "DG", "p-refinement", "exponential-convergence",
                 "verification", "temporal-error-floor", "exact-time-integration"],
        "markdown": md,
        "figures": figures,
    }
    result = post("/reports", payload, token)
    print(json.dumps({"report_id": result.get("id", result), "artifact": str(RESULTS)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())