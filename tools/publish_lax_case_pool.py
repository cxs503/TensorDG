"""Render ``docs/CASE-LAX-DG.md`` and publish the platform report from the artifact.

Every number in the document and in the report body is read out of
``docs/benchmarks/results/lax_case_pool/metrics.json`` at render time; nothing is transcribed by
hand.  Figures are the PNGs listed in that artifact, base64-inlined into the
report payload.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# tracked path: ``runs/`` never survives the merge gate
CASE = ROOT / "docs" / "benchmarks" / "results" / "lax_case_pool"
METRICS = CASE / "metrics.json"
SUMMARY = CASE / "summary.md"
DOC = Path("docs/CASE-LAX-DG.md")
MISSION_ID = int(os.environ.get("MISSION_ID", "15991"))


def f(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def pct(value, digits=3):
    return "n/a" if value is None else f"{100.0 * value:.{digits}f} %"


def build_markdown(metrics):
    waves = metrics["waves"]
    gates = metrics["gates"]
    ladder = metrics["ladder"]
    entropy = metrics["entropy_wave"]
    pos = waves["positions"]
    exact_pos = metrics["exact_positions"]

    lines = []
    add = lines.append
    add("# CASE-LAX-DG · Lax 激波管（Toro test 2）DG 算例池验证")
    add("")
    add(f"> 工件 `docs/benchmarks/results/lax_case_pool/metrics.json` 生成于 {metrics['generated_at_utc']}，"
        f"本文全部数字由 `benchmarks/lax_case_pool.py` 从该 JSON 注入（零手抄）。")
    add("")
    add("## ① 案例介绍")
    add("")
    add("Lax (1954) 激波管 / Toro (1999) test 2：`[0,1]` 域内隔膜 `x0 = 0.5` 两侧常状态，"
        f"γ = {waves['gamma']}，快照时刻 t = {waves['final_time']}。精确解含四道波："
        "左稀疏波（head → tail）、接触间断、右激波。四道波的位置与星区状态全部有闭式表达，"
        "因此该算例自带不需第三方求解器产生的参考值。")
    add("")
    add("| 参数 | 左态 | 右态 |")
    add("|---|---|---|")
    add(f"| (ρ, u, p) | ({waves['left_state'][0]}, {waves['left_state'][1]}, "
        f"{waves['left_state'][2]}) | ({waves['right_state'][0]}, "
        f"{waves['right_state'][1]}, {waves['right_state'][2]}) |")
    add("")
    add("解析波系（闭式，未经求解器回读）：")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    for key, label in (("pressure_star", "p*"), ("velocity_star", "u*"),
                       ("density_star_left", "ρ*_L（等熵稀疏）"),
                       ("density_star_right", "ρ*_R（Rankine–Hugoniot）")):
        add(f"| {label} | {f(waves[key], 6)} |")
    for key, label in (("rarefaction_head", "x 稀疏波头（解析边沿）"),
                       ("rarefaction_tail", "x 稀疏波尾（解析边沿）"),
                       ("contact", "x 接触间断"), ("shock", "x 右激波")):
        add(f"| {label} | {f(pos[key], 6)} |")
    add("")
    add("## ② 方法")
    add("")
    add(f"模态 DG（Legendre 基，P{waves.get('degree_hint', '1–3')}），HLLC 数值通量，SSP-RK3 "
        f"时间推进，CFL = {metrics['config']['cfl']}，位置与星区量在每单元 10 点 Gauss 节点上"
        "重建后测量。分辨率阶梯 N × p：" +
        ", ".join(f"N={c}/P{p}" for c, p in metrics["config"]["ladder"]) + "。")
    add("")
    add("五道预登记判据（容差在跑数之前写死在 `benchmarks/lax_case_pool.py`，未按结果回调）：")
    add("")
    add("| 判据 | 测量对象 | 容差 |")
    add("|---|---|---|")
    add(f"| C1 | 四道波位置（最细腿）vs 解析位置，归一化 `|Δx|/|x_exact-x0|` | < {pct(gates['c1']['tolerance'])} |")
    add(f"| C2 | 星区 (ρ*_L, u*, p*, ρ*_R) 平台值 vs 解析 | < {pct(gates['c2']['tolerance'])} |")
    add(f"| C3 | 全场相对 L1 收敛阶（密度，最细一对） | ≥ {gates['c3']['minimum_order']} |")
    add(f"| R3 | 解析区（波带 {gates['r3']['band_cells']} dx 剔除）相对 L1 收敛阶 | "
        f"≥ {gates['r3']['order_fraction']}·(p+1) |")
    add(f"| R5 | 光滑熵波设计阶（u = 1.0 / 2.0，周期域） | 实测阶与 p+1 相对偏差 < "
        f"{pct(gates['r5']['fraction'])} |")
    add("")
    add("## ③ 计算结果（程序化注入）")
    add("")
    add(SUMMARY.read_text().split("## resolution ladder", 1)[-1].join(
        ["## resolution ladder", ""]) if SUMMARY.exists() else "")
    add("")
    add("## ④ 图")
    add("")
    for image in metrics["figures"]:
        add(f"* `{image}`")
    add("")
    add("## ⑤ 注意点 / 局限")
    add("")
    head_offset = exact_pos["rarefaction_head"] - pos["rarefaction_head"]
    tail_offset = exact_pos["rarefaction_tail"] - pos["rarefaction_tail"]
    add(f"* **C1 的口径**：稀疏波头/尾按压力降的 "
        f"{100 * 0.1:.0f} % 水平线定位，因此它落在扇形区**内部**而不是解析边沿。"
        f"本例中该水平线与解析边沿的偏移为 {f(head_offset, 4)} / {f(tail_offset, 4)}（x 单位），"
        "故 C1 是「同一水平线在数值场与精确场中的位置差」，不是边沿分辨率。")
    add("* **接触间断/激波窗口**：搜索窗从相邻波回缩 `max(4 dx, 0.02)`，"
        "否则被抹宽的接触间断会被误认成激波（本单实测到的定位缺陷，已由 "
        "`tests/test_lax_case_pool.py::test_contact_and_shock_are_not_confused_by_the_search_windows` 锁死）。")
    add("* **宽谱 vs 解析区**：全场 L1 由间断主导、收敛阶被压到一阶附近；解析区把四道波各剔掉 "
        f"{gates['r3']['band_cells']} dx，剩余区域仍受相邻间断的数值抹平污染，因此解析区阶数不等于设计阶 p+1。")
    add(f"* **R5 的口径**：台账原文为「within 3% of 1.0/2.0」，本单读作「两种预登记波速 "
        "u = 1.0 / 2.0 下实测阶与设计阶 p+1 的相对偏差 < 3%」，两种波速与两条阶梯写在工件里，"
        "读数若有异议可直接按工件重判。")
    add("* 计算面：CPU / float64；本仓限幅器是局部正性回退而非 TVD 证明，"
        "因此本算例验证的是「该实现的实测精度」，不构成一般激波捕捉保证。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    add("| 判据 | 关键量 | 实测 | 容差 | 是否达标（<3% 口径） |")
    add("|---|---|---|---|---|")
    c1 = gates["c1"]
    worst_wave = max(c1["waves"].items(), key=lambda kv: kv[1]["relative_error"])
    add(f"| C1 | 最差波位置（{worst_wave[0]}） | {pct(worst_wave[1]['relative_error'])} | "
        f"< {pct(c1['tolerance'])} | {'是' if worst_wave[1]['relative_error'] < 0.03 else '**否**'}"
        f"（预登记容差 {pct(c1['tolerance'])} 下 {c1['status']}） |")
    c2 = gates["c2"]
    worst_star = max(c2["values"].items(), key=lambda kv: kv[1]["relative_error"])
    add(f"| C2 | 最差星区量（{worst_star[0]}） | {pct(worst_star[1]['relative_error'])} | "
        f"< {pct(c2['tolerance'])} | {'是' if worst_star[1]['relative_error'] < 0.03 else '**否**'}"
        f"（预登记容差 {pct(c2['tolerance'])} 下 {c2['status']}） |")
    add(f"| C3 | 密度全场 L1 阶 | {f(gates['c3']['gated_order'], 3)} | "
        f"≥ {gates['c3']['minimum_order']} | 阶数判据（非误差百分比）：{gates['c3']['status']} |")
    r3_rows = gates["r3"]["per_leg"]
    add(f"| R3 | 密度解析区阶（P1/P2/P3） | "
        + " / ".join(f(r3_rows.get(f'p{p}.density')) for p in (1, 2, 3))
        + f" | ≥ {gates['r3']['order_fraction']}·(p+1) | 阶数判据：{gates['r3']['status']} |")
    r5_rows = gates["r5"]["cases"]
    add(f"| R5 | 熵波阶偏差（最差） | "
        f"{pct(max(abs(v['relative_deviation']) for v in r5_rows.values()))} | "
        f"< {pct(gates['r5']['fraction'])} | {gates['r5']['status']} |")
    add("")
    add(f"**总判定：{metrics['verdict']}**。"
        "（判据逐条为 PASS/FAIL，未达标的条目按 honest-FAIL 原样记录，不重跑调参掩盖。）")
    add("")
    return "\n".join(lines)


def build_report_markdown(metrics):
    markdown = build_markdown(metrics)
    return markdown


def publish(markdown, figures):
    base = os.environ["TENSOR_API_BASE"]
    payload = {
        "title": "CASE-LAX-DG · Lax 激波管 DG 算例池：四波位置/星区/收敛阶五门验证",
        "summary": (f"Lax 算例池五门判据实测：C1 {metrics_gates('c1')}、C2 "
                    f"{metrics_gates('c2')}、C3 {metrics_gates('c3')}、R3 "
                    f"{metrics_gates('r3')}、R5 {metrics_gates('r5')}（总判定 "
                    f"{metrics_verdict()}）"),
        "scenario_key": "case-lax-dg",
        "mission_id": MISSION_ID,
        "tags": ["TensorDG", "CASE-LAX-DG", "verification", "Euler", "DG"],
        "markdown": markdown,
        "figures": figures,
    }
    token = _login(base)
    request = urllib.request.Request(
        base + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


_METRICS_CACHE = {}


def metrics_gates(name):
    return _METRICS_CACHE["gates"][name]["status"]


def metrics_verdict():
    return _METRICS_CACHE["verdict"]


def _login(base):
    request = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = json.loads(response.read())
    return body.get("access_token") or body["token"]


def main():
    metrics = json.loads(METRICS.read_text())
    _METRICS_CACHE.update(metrics)
    markdown = build_markdown(metrics)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown + "\n")
    print(f"wrote {DOC} ({len(markdown)} chars)")
    figures = []
    for path in metrics["figures"]:
        p = Path(path)
        figures.append({"name": p.stem,
                        "caption": p.stem.replace("_", " "),
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    response = publish(markdown, figures)
    print("report:", json.dumps(response)[:400])
    return response


if __name__ == "__main__":
    main()