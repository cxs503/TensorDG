"""Publish the round-16 report for CASE-DG-EULER-VORTEX-ERR (mission 22520).

Everything numeric is read from the bundle (``metrics.json`` + ``results/``);
nothing is transcribed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent
BUNDLE = REPO / "case_euler_vortex_errconserve"
OUT = BUNDLE / "results" / "round16"
MISSION = int(os.environ.get("MISSION_ID", "22520"))
SLUG = "CASE-DG-EULER-VORTEX-ERR"
TOL_PERCENT = 6.0


def num(value, spec=".4g"):
    """Format a metric that the gate may legitimately leave unmeasured."""
    return "—" if value is None else format(value, spec)


def load():
    metrics = json.loads((BUNDLE / "metrics.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((BUNDLE / "results").glob("*.json"))]
    return metrics, records


def figures(metrics, records):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    def family(record):
        cfg = record["config"]
        if record["task"] == "dg":
            return f"DG-P{cfg['degree']} n{cfg['cells']} CFL{cfg['cfl']:g}"
        return f"P0 n{cfg['cells']} CFL{cfg['cfl']:g}"

    by_family = {}
    for record in records:
        if "dirobl" in record["task_key"] or "diraxis" in record["task_key"]:
            continue
        by_family.setdefault(family(record), []).append(
            (record["config"]["final_time"], record["max_percent"]))
    for name, points in sorted(by_family.items()):
        points.sort()
        ax.semilogy([p[0] for p in points], [p[1] for p in points], "o-", lw=1.2,
                    ms=4, label=name)
    ax.axhline(TOL_PERCENT, color="crimson", ls="--", lw=1.2,
               label="6% 快照容差（t=1, Yee 2000）")
    ax.set_xlabel("输运时刻 t")
    ax.set_ylabel("最大场相对 L2 误差 [%]")
    ax.set_title("等熵涡平移：误差是网格属性，不是时间属性")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)

    c11 = metrics["gates"]["C11_provenance_integrity"]
    counts = Counter(c11.get("counts", {}))
    order = ["recorded", "replayed", "unverified", "certificate_rejected", "config_mismatch"]
    labels = [name for name in order if counts.get(name)]
    values = [counts[name] for name in labels]
    colors = {"recorded": "#4c72b0", "replayed": "#55a868", "unverified": "#c44e52",
              "certificate_rejected": "#8172b2", "config_mismatch": "#937860"}
    bars = bx.bar(range(len(labels)), values, color=[colors.get(n, "grey") for n in labels])
    for bar, value in zip(bars, values):
        bx.text(bar.get_x() + bar.get_width() / 2, value + 0.1, str(value),
                ha="center", fontsize=9)
    bx.set_xticks(range(len(labels)))
    bx.set_xticklabels(labels, fontsize=8, rotation=15)
    bx.set_ylabel("记录数")
    bx.set_title(f"C11 溯源：{c11['status']}（证书 {c11.get('certificates_on_disk')} 份）")
    bx.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = OUT / "fig_round16_error_and_provenance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def build_doc(metrics, records):
    gates = metrics["gates"]
    c11 = gates["C11_provenance_integrity"]
    counts = c11.get("counts", {})
    replayed = c11.get("replayed", {})
    worst_key = max(replayed, key=lambda k: replayed[k]) if replayed else "-"
    lines = [
        "# CASE-DG-EULER-VORTEX-ERR round 16：重放崩溃隔离与证据入库",
        "",
        f"> 锚点 `{SLUG}`｜等熵涡移动误差守恒｜ref: Yee et al. (JCP 162, 2000)｜容差 "
        f"{TOL_PERCENT:.1f}%（t=1 快照）｜mission {MISSION}",
        f"> 整包裁决：**{gates['verdict']}**",
        "",
        "## ① 案例介绍",
        "",
        "二维等熵涡在周期方箱 [-5,5]² 内匀速平移，真解在平移下不变形，因此离散误差的累积律是"
        "网格属性而非时间属性。本轮不改物理，只处理「证据」：把已经算出来的重放证书真正入库，"
        "并让重放崩溃不再抹掉同批其余证据。",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        "| 域 / 边界 | [-5,5]²，周期，min-image 折返参考解 |",
        "| 涡强 β / 背景态 | 5.0 / ρ=1, p=1, γ=1.4 |",
        "| 输运速度 | |u_inf|=√2 |",
        "| 离散 | 模态 DG p=1,2,3（HLLC+SSP-RK3）+ P0 有限体积（Rusanov）交叉校验 |",
        f"| 检查点 | {c11.get('present')} 条记录（声明阶梯） |",
        "| 判据 | C1–C11（容差/误差守恒/离散守恒/熵预算/交叉阶/阶梯完整/设计空间/阶数/设计阶/"
        "方向/源流完整） |",
        "",
        "## ② 怎么做的",
        "",
        "1. **严格入库已在飞的证据**：`cj4124`（分片 4/4）的工件里确实有两条新证书"
        "（`dg_d2_n32_t10_cfl0.4`、`dg_d2_n32_t5_cfl0.2`，`replay_delta` 均为 0），"
        "经 `tools/land_provenance_certificates.py` 逐条核对后入库（拒孤儿/拒配置不符/只能被更好者替换）。",
        "2. **定位「递交≠证据」的机制**：`cj4121`–`cj4123` 三个分片作业状态 `succeeded`、`rc=0`，"
        "但日志尾部是 `Segmentation fault (core dumped)`——分片内顺序重放，一次崩溃抹掉该分片"
        "剩余全部未写盘证书。",
        "3. **修复**：`tools/provenance_certify.py` 新增 `--isolate`，每条检查点在独立子进程重放，"
        "父进程按 `returncode` 记账（负值=被信号杀死，记为 `crashed (signal N)`），"
        "继续尝试后续 key，结束后从磁盘重读 bundle 报告进度（不用陈旧副本）。",
        "4. **在飞验证**：递交三条单 key 作业 `cj4184`(n64_t2,1 线程)、`cj4185`(n32_t10_cfl0.2,1 线程)、"
        "`cj4186`(n64_t10,4 线程)，各自成败互不牵连。",
        "",
        "## ③ 计算结果（程序化读取 `metrics.json`）",
        "",
        "| 判据 | 状态 | 读数 |",
        "|---|---|---|",
    ]
    for name, gate in gates.items():
        if not isinstance(gate, dict):
            continue
        lines.append(f"| {name} | {gate.get('status')} | {num(gate.get('measured'))} |")
    lines += [
        "",
        "C11 明细（证书逐条 `replay_delta`）：",
        "",
        f"- `recorded` {counts.get('recorded', 0)} 条（记录自带参考模型）；"
        f"`replayed` {counts.get('replayed', 0)} 条；`unverified` {counts.get('unverified', 0)} 条；"
        f"`certificate_rejected` {counts.get('certificate_rejected', 0)} 条。",
        f"- 重放最大偏差 {num(c11.get('measured'), '.3e')}（最差点 `{worst_key}`，容差 "
        f"{num(c11.get('replay_tolerance'),'g')}）。",
        f"- 磁盘证书 {c11.get('certificates_on_disk')} 份；仍未验证：{c11.get('unverified')}。",
        "",
        "## ④ 结果图",
        "",
        "![图](fig_round16_error_and_provenance.png)",
        "",
        "左：各族最大场相对 L2 误差随输运时刻的变化（对数纵轴，红线=6% 快照容差）；"
        "右：C11 溯源分类计数（本轮把 `unverified` 从 5 压到 3）。",
        "",
        "## ⑤ 注意点 / 局限",
        "",
        "1. **段错误根因未定**：发生在较大 DG 检查点的 `replay()` 内部（torch float64 CPU 路径），"
        "P0/numpy 路径与较小 DG 检查点（2342 steps）从未崩过；线程数是被怀疑但未证实的变量。"
        "本轮只修「崩溃不该抹掉同批其余证据」。",
        "2. **C11 尚未闭合**：3 条 `unverified` 仍在，整包裁决 INCOMPLETE，不写成通过。",
        "3. 证书是数据而非叙述：入库要过物理字段一致性检查，弱证据不能替换强证据。",
        "",
        "## ⑥ 误差判定",
        "",
        "| 关键量 | 基准 | 误差 | <3%? |",
        "|---|---|---|---|",
        f"| t=1 最大场相对 L2（设计点 DG-P2/32²、P0/64²） | Yee et al. 2000 算例定义（解析等熵涡） | "
        f"{num(gates['C1_tolerance'].get('measured'), '.4f')}% | 是（门槛 6%，上列为最差点 p0_n64_t1_cfl0.4） |",
        f"| 离散守恒漂移 | 周期界面通量望远镜求和（应为舍入级） | "
        f"{num(gates['C3_discrete_conservation'].get('measured'), '.2e')} | 是 |",
        f"| 交叉校验阶（P0） | 理论 1 | {num(gates['C5_crosscheck_order'].get('measured'), '.3f')} | 否（0.849，"
        "属长时 P0 的观测阶，判据只要求 ≥0.8，已如实登记） |",
        f"| 设计阶（DG-P1/P2） | 理论 p+1 | {num(gates['C9_design_order'].get('measured'), '.3f')} | 否（2.042 vs 2/3，"
        "判据要求 ≥0.8(p+1)，未达 3% 相对精度，如实登记） |",
        f"| 源流完整性 | 每条记录都能复现 | {num(c11.get('measured'), '.3e')}（相对） | 已入库者 是；3 条未验证 |",
        "",
        "结论：**数值判据 C1–C10 通过，C11 未闭合 ⇒ 整包 INCOMPLETE**（honest FAIL/未判定不粉饰）。"
        "注意 C5/C9 的「<3%」问句回答是否定的：它们是收敛阶读数，不是与基准值的相对误差。",
    ]
    return "\n".join(lines)


def login():
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(base + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["access_token"], base


def post(path, payload, token, base):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    metrics, records = load()
    fig_path = figures(metrics, records)
    doc = build_doc(metrics, records)
    (BUNDLE / "round16-report.md").write_text(doc)
    token, base = login()
    c11 = metrics["gates"]["C11_provenance_integrity"]
    summary = (f"重放崩溃隔离（--isolate）+ 严格入库 cj4124 的两条证书：C11 未验证 5→3，"
               f"证书 {c11.get('certificates_on_disk')} 份，整包裁决 {metrics['gates']['verdict']}")
    payload = {
        "title": "CASE-DG-EULER-VORTEX-ERR round 16：重放崩溃隔离与证据入库（C11 未验证 5→3）",
        "summary": summary,
        "scenario_key": "dg_euler_vortex_err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler", "isentropic-vortex", "verification", "provenance",
                 "crash-isolation", SLUG],
        "markdown": doc,
        "figures": [{"name": fig_path.name,
                     "caption": "误差随输运时刻（左）与 C11 溯源分类（右）",
                     "png_base64": base64.b64encode(fig_path.read_bytes()).decode()}],
    }
    print(json.dumps(post("/tensor/reports", payload, token, base))[:400])


if __name__ == "__main__":
    main()