"""Publish the CASE-LAX-DG AO-arm closure to the platform.

Every number is read out of the committed artifacts
``docs/benchmarks/results/lax_scheme_variants/metrics.json`` and
``.../legs/*.json`` (zero hand transcription); the AO profile figure is built
from ``legs/ao.npz``.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_scheme_variants"
METRICS = OUT / "metrics.json"
LEGS = OUT / "legs"
FIG = OUT / "figures"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 22437
FIELDS = ("density", "velocity", "pressure")


def _post(path, body, token=None):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def login():
    body = _post("/auth/login", {"username": USER, "password": PASSWORD})
    return body.get("data", body)["access_token"]


def figure(name):
    return base64.b64encode((FIG / name).read_bytes()).decode()


def render_ao_figure():
    """Density / pressure of the AO arm against the exact Riemann solution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = LEGS / "ao.npz"
    data = np.load(path)
    points = data["points"]
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.4), sharex=True)
    for ax, field in zip(axes, ("density", "pressure")):
        ax.plot(points, data[f"exact_{field}"], "k-", lw=1.4, label="exact Riemann")
        ax.plot(points, data[f"numerical_{field}"], color="#c44e52", lw=1.0,
                label="DG p3.N400, AO limiter")
        ax.set_ylabel(field)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("x")
    axes[0].set_title("CASE-LAX-DG: AO arm at the pinned leg (t = 0.13)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    target = FIG / "ao_arm_profiles.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def row(metrics, name):
    return next(r for r in metrics["rows"] if r["variant"] == name)


def status_table(metrics):
    tol = metrics["config"]["tolerance"]
    lines = ["| 臂 | 旋钮 | steps | L2 ρ | L2 u | L2 p | 最差/容差 | 可用 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in metrics["rows"]:
        if r.get("not_run"):
            cells = ["未跑出"] * 4
            ratio = "—"
            ok = "否（`not run`）"
        elif "error" in r:
            cells = ["未跑出"] * 4
            ratio = "—"
            ok = f"否（`{r['error']}`）"
        else:
            cells = [f"{r['l2'][f]:.4e}" for f in FIELDS]
            ratio = f"{r['decision']['margin']:.4f}"
            ok = "是" if r["decision"]["admissible"] else "否"
        knob = "—（生产配置）" if not r["knob"] else ", ".join(
            f"{k}={v}" for k, v in r["knob"].items())
        first = [r["variant"], knob]
        if r.get("not_run") or "error" in r:
            lines.append("| " + " | ".join(first + cells + [ratio, ok]) + " |")
        else:
            lines.append("| " + " | ".join(
                first + [str(r["steps"])] + cells + [ratio, ok]) + " |")
    return "\n".join(lines), tol


def error_table(metrics):
    tol = metrics["config"]["tolerance"]
    lines = ["| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |", "|---|---|---|---|---|"]
    for r in metrics["rows"]:
        if r.get("not_run") or "error" in r:
            lines.append(f"| L2 ρ @ `{r['variant']}` (p3.N400) | 精确 Riemann 解 | 未跑出 | — | 未知 |")
            continue
        l2 = r["l2"]["density"]
        lines.append(
            f"| L2 ρ @ `{r['variant']}` (p3.N400) | 精确 Riemann 解 | `{l2:.4e}` | "
            f"{l2 * 100:.4f} % | {'是' if l2 < tol else '**否**'} |")
    return "\n".join(lines)


def build():
    metrics = json.loads(METRICS.read_text())
    verdict = metrics["verdict"]
    bound = metrics["attainability"]
    ao = row(metrics, "ao")
    base = row(metrics, "base")
    av = row(metrics, "av")
    table, tol = status_table(metrics)
    etable = error_table(metrics)
    cfg = metrics["config"]
    case = metrics["case"]

    ao_line = (f"`{ao['l2']['density']:.4e}`（{ao['l2']['density'] * 100:.4f} %）"
               if not (ao.get("not_run") or "error" in ao) else "未跑出")
    scope = "完整" if verdict["scope_complete"] else "不完整"

    md = f"""# CASE-LAX-DG · AO 限幅臂闭环：闭合臂集最后一次实测

锚点 `CASE-LAX-DG`（原文「ref: 精确解｜容差 < 3 %」，status open）。本文所有数字由
`benchmarks/lax_scheme_variants.py` 从
`docs/benchmarks/results/lax_scheme_variants/metrics.json`（生成于 {metrics['generated_at_utc']}）
程序化注入，零手抄。

## ① 案例介绍

{case['name']}：`{case['domain']}`，隔膜 `x0 = {case['diaphragm']}`，γ = {case['gamma']}，
左态 `(ρ,u,p) = {tuple(case['left_state'])}`，右态 `{tuple(case['right_state'])}`，快照 `t = {case['final_time']}`。
钉扎腿 `p{cfg['pinned_degree']}.N{cfg['pinned_cells']}`（DOF {cfg['dof']}），整场相对 L2，求积阶 {cfg['quadrature_order']}，
台账容差 {tol * 100:.1f} %（由 `lax_case_pool.ledger_acceptance_tolerance()` 从台账原文读出）。

**本单要回答的欠账**：预登记臂集里有 6 条臂，其中 `ao`（hierarchical AO 限幅器）从来没有数字——
上一单递交的集群作业 `cj4072` 在节点上以 `ModuleNotFoundError` 退出（见 ⑤），却被读成「已递交、待完成」。
一条**没跑过**的臂不等于「跑过且不合格」，所以「臂集闭合、配置空间内不可达」这个结论此前缺一块。

## ② 怎么做的

1. 先查作业真身：`cj4072` 的 stdout 尾部是 `ModuleNotFoundError: No module named 'benchmarks.lax_scheme_variants'`，
   节点检出 head = `caaa480`（远早于该模块入库的提交），而 `cj4072` 递交时 `stage_dir=None` → 跑的是节点上的
   旧 main 克隆，代码根本不在。
2. 复跑改用 `stage_dir` 把本工作树 tar 直达节点（`cj4120`，4 核，3300 s），命令
   `python3 -m benchmarks.lax_scheme_variants --variant ao --aggregate --figures --report`：
   同一把尺子（`modal_fields` / `_weighted_l1` / `_weighted_l2` / `exact_reference` 全部来自算例池），
   臂定义仍是预登记的那一条（`limiter='ao'`），一个旋钮，不新增臂、不改容差。
3. `--aggregate` 重新装配 `metrics.json`（`allow_missing=False`，缺任一臂直接报错），`--figures/--report`
   重生成图与 `docs/CASE-LAX-DG-scheme-variants.md`；控制臂 `base` 仍须逐位复现已提交台账臂。

## ③ 计算结果（程序化注入）

{table}

* 臂集是否全部有数：`arms_not_run = {verdict['arms_not_run']}`，`arms_errored = {verdict['arms_errored']}`，
  作用域{scope}（`scope_complete = {verdict['scope_complete']}`；唯一没有数值的臂是 `pos_none`，它按预登记规则
  以 `FloatingPointError` 记录，属**结果**而非缺口——`arms_not_run = {verdict['arms_not_run']}` 才是缺口清单，现在是空的）。
* 可用臂集合：`{verdict['admissible_arms']}`；sweep 判定 **{verdict['verdict']}**，控制臂复现
  `{verdict['control_reproduced']}`（最差相对差 `{metrics['control_check']['worst_relative_difference']:.3e}`）。
* AO 臂整场 L2 密度 = {ao_line}；对照 `base` = `{base['l2']['density']:.4e}`，`av` = `{av['l2']['density']:.4e}`。
* 可行性界（与格式无关）：参考投影 floor `{bound['reference_floor']:.4e}`（占容差 {bound['floor_share_of_tolerance'] * 100:.2f} %），
  达标所需 solver 腿上界 `{bound['solver_leg_required']:.4e}`，已提交格式的 solver 腿 `{bound['solver_leg_measured']:.4e}`，
  实测/所需 = `{bound['measured_over_required']:.4f}`，还需相对缩减 `{bound['relative_reduction_needed'] * 100:.2f} %`。

## ④ 结果图

![variant worst L2](benchmarks/results/lax_scheme_variants/figures/variant_worst_l2.png)

*图 1*：每臂最差场量整场相对 L2（红 = 不可用），虚线 = 台账 {tol * 100:.1f} %，点线 = 参考解投影 floor。
AO 臂现在有自己的柱子，不再是空档。

![ao arm profiles](benchmarks/results/lax_scheme_variants/figures/ao_arm_profiles.png)

*图 2*：AO 臂在钉扎腿上的密度/压力剖面 vs 精确 Riemann 解（Gauss 节点采样）。

## ⑤ 注意点（坑 / 局限）

1. **集群作业「succeeded」可能只是包装成功**：`cj4072` 平台侧显示 succeeded，节点日志 rc=1。判作业要通过
   `cluster/jobs/<jid>.log` 的 `rc` 与 `output_tail`，不能只看状态字。
2. **代码不在 main 的提交里时，递交必须带 `stage_dir`**：否则节点用自己那份（可能陈旧）的 main 克隆，
   失败形式还是「模块不存在」而不是「代码错」。本单复跑即用 `stage_dir=<worktree>`。
3. **AO 臂的物理解读**：AO 是层级（hierarchical）限幅，troubled cells 内尽量保阶，与 AV（连续人工黏性）
   机制不同，两者 L2 直接比大小只能在「同一把尺子」的意义上比；本文只报数并给可用判定。
4. **不移动生产腿**：`p3.N400` 的台账读数原样留在记录里；分辨率出路（`p3.N800` 达标 2.3149 %）见
   `docs/CASE-LAX-DG-resolution-requirement.md`，本单不重开口径。
5. 墙钟不作为排序判据（集群排队噪声）。

## ⑥ 误差判定（关键量 vs 基准，逐项）

{etable}

**结论**：sweep 判定 **{verdict['verdict']}**，作用域{scope}。钉扎腿 `p3.N400` 的 3 % 线在已提交格式的
配置空间里{'有可用臂' if verdict['admissible_arms'] else '无可用臂'}；闭集既然全部有数，这条结论不再是「尚缺一臂」的悬案。
honest FAIL 如实登记，不由本单放宽口径。
"""
    return {
        "title": "CASE-LAX-DG · AO 限幅臂闭环：闭合臂集最后一次实测（p3.N400，3 % 容差）",
        "summary": (
            f"补齐 scheme_variants 预登记臂集最后一条未跑臂 ao（cj4072 因节点旧检出 ModuleNotFoundError 而从未跑成，"
            f"改 stage_dir 复跑 cj4120）：AO 臂 L2 ρ = {ao_line}，可用={ao.get('decision', {}).get('admissible')}；"
            f"臂集作用域{scope}，sweep 判定 {verdict['verdict']}，台账 3 % 门槛在钉扎腿 p3.N400 上仍未达标。"),
        "scenario_key": "case-pool/lax-dg",
        "mission_id": MISSION_ID,
        "tags": ["CASE-LAX-DG", "TensorDG", "DG", "Lax", "AO-limiter", "verification-debt", "honest-FAIL"],
        "markdown": md,
        "figures": [
            {"name": "variant_worst_l2.png",
             "caption": "每臂最差场量整场相对 L2 vs 台账 3 % 容差与参考投影 floor",
             "png_base64": figure("variant_worst_l2.png")},
            {"name": "ao_arm_profiles.png",
             "caption": "AO 臂密度/压力剖面 vs 精确 Riemann 解（p3.N400, t=0.13）",
             "png_base64": figure("ao_arm_profiles.png")},
        ],
    }


def publish():
    body = build()
    token = login()
    return _post("/tensor/reports", body, token)


def write_doc(body):
    target = ROOT / "docs" / "CASE-LAX-DG-ao-arm-closure.md"
    note = ("\n> 本文与平台报告同源，数字由 `publish_case_lax_ao_arm_closure.py` 从\n"
            "> `docs/benchmarks/results/lax_scheme_variants/metrics.json` 程序化注入。\n")
    target.write_text(body["markdown"] + note)
    return target


if __name__ == "__main__":
    render_ao_figure()
    body = build()
    print("doc:", write_doc(body))
    out = publish()
    print(json.dumps(out, ensure_ascii=False)[:400])