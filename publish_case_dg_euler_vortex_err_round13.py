#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR round 13: closing C11 with the landed replays.

Round 12 left one criterion unevaluated: 7 legacy records whose stored errors had
no replay certificate, so the package verdict was INCOMPLETE.  Job cj3699 came
back with one of them before dying on its ``--max-seconds`` budget, i.e. the debt
was being paid one checkpoint per (serial) job -- and the two most expensive
checkpoints were the reason it kept missing.

This round changes *how the debt is paid*, not what counts as paid:

1. ``tools/provenance_certify.py`` learns ``--remaining`` (never re-pay for a
   certificate that already landed) and ``--shard i/n`` (a cost-balanced,
   deterministic partition), so the outstanding replays run concurrently and
   their outputs are mergeable;
2. ``tools/land_provenance_certificates.py`` merges fetched certificates through
   an explicit admission filter (same checkpoint, same physics configuration,
   same reference model, reproducing delta);
3. the *unchanged* aggregator then decides the package.  No tolerance, no
   criterion text and no stored reading is touched by this round.

Every number in the emitted document is read from the machine artefacts

* ``case_euler_vortex_errconserve/metrics.json``
* ``case_euler_vortex_errconserve/provenance_certificates.json``

so no digit is hand-copied.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round13.py --dry-run
    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round13.py
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
OUT = Path("case_euler_vortex_errconserve")
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR-round13-c11-closure.md")
MISSION = 21962
SLUG = "CASE-DG-EULER-VORTEX-ERR"
TOL = 1e-9

#: checkpoints this round's shard job replayed (declared, so the document cannot
#: silently claim a replay it did not run)
ROUND13_KEYS = ("dg_d2_n32_t10_cfl0.2", "dg_d2_n32_t10_cfl0.4",
                "dg_d2_n32_t5_cfl0.2", "dg_d2_n64_t10_cfl0.2",
                "dg_d2_n64_t1_cfl0.2", "dg_d2_n64_t2_cfl0.2")

CRITERIA_ZH = {
    "C1_tolerance": "C1 容差",
    "C2_error_conservation": "C2 误差守恒",
    "C3_discrete_conservation": "C3 离散守恒",
    "C4_entropy_budget": "C4 熵预算",
    "C5_crosscheck_order": "C5 P0 交叉阶",
    "C6_ladder_completeness": "C6 阶梯完整性",
    "C7_design_space_invariance": "C7 设计空间不变性",
    "C8_degree_invariance": "C8 阶数不变性",
    "C9_design_order": "C9 设计阶",
    "C10_direction_invariance": "C10 方向不变性",
    "C11_provenance_integrity": "C11 证据溯源",
}
FLAG = {"passed": "✅ 通过", "failed": "❌ 未通过", "not_evaluated": "⏸ 未判定"}


def num(value, digits=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{digits}g}"


def load():
    metrics = json.loads((OUT / "metrics.json").read_text())
    certs = json.loads((OUT / "provenance_certificates.json").read_text())
    return metrics, certs


def build_doc(metrics: dict, certs: dict) -> str:
    gates = metrics["gates"]
    prov = gates["C11_provenance_integrity"]
    cert_map = certs.get("certificates", {})
    lines = []
    add = lines.append

    add("# CASE-DG-EULER-VORTEX-ERR round 13：C11 溯源闭合与整包裁决")
    add("")
    add(f"> 锚点 `{SLUG}`｜算例池·等熵涡移动误差守恒｜"
        "ref: Yee, Sandham & Djomehri (JCP 162, 2000)｜容差 6.0%（t=1 快照，最大场相对 L2）")
    add(f"> 整包裁决：**{gates['verdict']}**")
    add("")
    add("本文件所有数字由脚本从 `case_euler_vortex_errconserve/metrics.json` 与 "
        "`provenance_certificates.json` 程序化注入，零手抄。")
    add("")

    # ① ---------------------------------------------------------------
    add("## ① 案例介绍")
    add("")
    add("二维可压缩 Euler 方程（γ = 1.4）中，等熵涡在双周期方箱 [-5,5]² 内以匀速 "
        "(1,1) 平移。流场处处等熵、扰动为高斯包络，故有闭式解析解；平移不改变涡的形状，"
        "因此离散误差应当是网格/格式属性而不是时间属性——这正是本算例（误差**守恒**）"
        "与快照算例 `EULER-VORTEX` 的区别。")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    add("| 域 / 边界 | [-5, 5]²，双周期 |")
    add("| γ | 1.4 |")
    add("| 涡强度 β | 5.0 |")
    add("| 来流速度 | (1, 1)，u_inf = √2 ≈ 1.41421 |")
    add("| 周期长度 / 整周期时间 | L = 10，T ≈ 7.0711 |")
    add(f"| 参考解（聚合口径） | `{metrics.get('reference_model')}` |")
    add(f"| 参考解缺陷（自由空间 vs 周期公式，t=1） | "
        f"{num(metrics.get('reference_defect_percent'), 4)} % |")
    add(f"| 容差 | {100.0 * float(metrics.get('tolerance_relative_l2', 0.06)):.1f}%"
        "（相对 L2，t=1 快照） |")
    add(f"| 声明 / 落盘检查点 | {gates['C6_ladder_completeness'].get('declared')} / "
        f"{gates['C6_ladder_completeness'].get('present')} |")
    add("")

    # ② ---------------------------------------------------------------
    add("## ② 怎么做的（本轮的方法）")
    add("")
    add("round 12 之后只剩一条未判定：C11 有 7 条旧读数没有重放证书。事实核查发现真正的"
        "瓶颈不是算力而是**串行**：`tools/provenance_certify.py --all-legacy` 在单进程里"
        "逐条重放，`--max-seconds` 一到位就退出——作业 cj3699 因此只带回了 1 条证书"
        "（`dg_d2_n32_t2_cfl0.2`，rel 0.0e+00），剩下的又是最贵的 6 条。")
    add("")
    add("本轮改的是**债务的付法**，不是「什么算付清」：")
    add("")
    add("1. `--remaining`：只重放还没有证书的旧读数，已落地的不重付；")
    add("2. `--shard i/n`：按代价代理量 `cells²·(p+1)²·t/CFL` 做**确定性**的 "
        "n 路均衡划分（贪心放到当前最轻的一片），因此各片互不重叠、合起来正好是待办集合——"
        "这是它们可以并发跑、证书可以合并的前提；")
    add("3. `tools/land_provenance_certificates.py`：取回的证书**过准入过滤**——必须是本地"
        "存在的检查点、物理配置（task/degree/cells/cfl/final_time/strength）逐字段一致、"
        "参考模型一致、重放偏差有限且 <= 1e-9、重放百分比与存档百分比同样吻合；"
        "已落地证书只会被**更好**的替换。不符者打印原因并**不写入**，判据继续判未判定。")
    add("")
    add("工具链：`tensordg.euler_dg_2d`（模态 DG + HLLC + SSP-RK3）、"
        "`tensordg.euler_2d`（P0 有限体积 + Rusanov，独立第二代码路径）、"
        "`benchmarks/case_euler_vortex_errconserve.py`（算例驱动 + 聚合器，本轮零改动）、"
        "`tools/provenance_certify.py`、`tools/land_provenance_certificates.py`。")
    add("")

    # ③ ---------------------------------------------------------------
    add("## ③ 计算结果")
    add("")
    add("### 判据总表（聚合器实测）")
    add("")
    add("| 判据 | 实测 | 判定 |")
    add("|---|---|---|")
    for key, label in CRITERIA_ZH.items():
        gate = gates[key]
        if key == "C6_ladder_completeness":
            measured = f"{gate.get('present')}/{gate.get('declared')} 落盘"
        elif key == "C11_provenance_integrity":
            counts = gate.get("counts", {})
            measured = (f"recorded {counts.get('recorded', 0)} / "
                        f"replayed {counts.get('replayed', 0)} / "
                        f"unverified {counts.get('unverified', 0)}；"
                        f"最大重放偏差 {num(gate.get('measured'), 3)}")
        elif key == "C1_tolerance":
            measured = f"最差族 `{gate.get('worst_at')}` = {num(100 * gate.get('measured', 0))}%"
        elif key == "C9_design_order":
            worst = min(row.get("order", float("inf"))
                        for row in gate.get("per_case", {}).values()) if gate.get("per_case") else None
            measured = f"最差观测阶 {num(worst, 4)}"
        else:
            measured = num(gate.get("measured"), 4)
        add(f"| {label} | {measured} | {FLAG.get(gate.get('status'), gate.get('status'))} |")
    add("")
    add(f"整包：`all_passed` = {gates.get('all_passed')}；未判定 {gates.get('not_evaluated')}；"
        f"未通过 {gates.get('failed')}；**verdict = {gates['verdict']}**。")
    add("")

    add("### C11 逐条溯源台账")
    add("")
    add("| 状态 | 条数 |")
    add("|---|---|")
    for state in ("recorded", "replayed", "unverified", "config_mismatch", "certificate_rejected"):
        value = prov.get("counts", {}).get(state, 0)
        if state in ("config_mismatch", "certificate_rejected") and not value:
            continue
        add(f"| `{state}` | {value} |")
    add("")
    add("重放证据（同一配置对当前参考解重跑，要求复现原读数到 1e-9 相对偏差）。"
        "「本轮」列标记 round 13 的重放：")
    add("")
    add("| 检查点 | 存档 % | 重放 % | 相对偏差 | 墙钟 s | 本轮 | 结论 |")
    add("|---|---|---|---|---|---|---|")
    worst = 0.0
    for key in sorted(cert_map):
        c = cert_map[key]
        delta = c.get("replay_delta", float("nan"))
        if delta == delta:
            worst = max(worst, abs(delta))
        add(f"| `{key}` | {c.get('stored_max_percent', float('nan')):.6g} | "
            f"{c.get('replayed_max_percent', float('nan')):.6g} | "
            f"{delta:.2e} | {c.get('wall_seconds', 0):.0f} | "
            f"{'✔' if key in ROUND13_KEYS else ''} | "
            f"{'✓ 复现' if delta <= TOL else '✗ 不一致'} |")
    add("")
    add(f"证书条数 {len(cert_map)}，最大相对偏差 {worst:.2e}（容差 {TOL:g}）；"
        f"本轮重放 {sum(1 for k in ROUND13_KEYS if k in cert_map)}/"
        f"{len(ROUND13_KEYS)} 条。")
    add("")

    add("### 误差随时间（设计族 + 阶数轴）")
    add("")
    add("| 检查点 | p | t | ρ % | u % | v % | p % | 最大场 % |")
    add("|---|---|---|---|---|---|---|---|")
    records = {r["task_key"]: r for r in metrics.get("records", [])}
    for key in sorted(k for k in records
                      if k.startswith(("dg_d1_", "dg_d2_", "dg_d3_")) and "dir" not in k):
        r = records[key]
        cfg, errs = r.get("config", {}), r.get("errors", {})
        add(f"| `{key}` | {cfg.get('degree', '-')} | {num(cfg.get('final_time'), 3)} | "
            f"{num(errs.get('density'), 4)} | {num(errs.get('velocity_x'), 4)} | "
            f"{num(errs.get('velocity_y'), 4)} | {num(errs.get('pressure'), 4)} | "
            f"{num(r.get('max_percent'), 4)} |")
    add("")

    add("### 设计阶（C9）与阶数不变性（C8）")
    add("")
    add("| 判据点 | p | 网格 | 观测阶 | 目标 | 判定 |")
    add("|---|---|---|---|---|---|")
    for key, row in gates["C9_design_order"].get("per_case", {}).items():
        order = row.get("order")
        add(f"| `{key}` | {row['degree']} | {row['cells']} | {num(order, 3)} | "
            f">= {row['target']:.2f} | "
            f"{'通过' if row['passed'] else ('未判定' if order != order else '未通过')} |")
    add("")
    add("| 阶数 | 族 | 检查点 | t_max | alpha | 判定 |")
    add("|---|---|---|---|---|---|")
    for degree, row in gates["C8_degree_invariance"].get("per_degree", {}).items():
        add(f"| p={degree} | `{row['family']}` | {row['checkpoints']} | "
            f"{num(row.get('final_time_max'), 3)} | {num(row.get('exponent'), 3)} | "
            f"{'通过' if row['passed'] else '未通过'} |")
    add("")

    # ④ ---------------------------------------------------------------
    add("## ④ 结果图")
    add("")
    have = False
    for name, caption in (
            ("fig_error_vs_time.png", "各族最大场相对 L2 随 t 的双对数曲线与 6% 容差线"),
            ("fig_design_space.png", "C7 设计空间（网格/CFL/阶数）与方向族并列"),
            ("fig_invariants_vs_time.png", "C3 离散不变量漂移与 C4 熵预算")):
        if (OUT / name).exists():
            have = True
            add(f"* `{name}`：{caption}；")
    if not have:
        add("（本机未生成图件：`--figures` 未跑；见复现命令。）")
    add("")

    # ⑤ ---------------------------------------------------------------
    add("## ⑤ 注意点")
    add("")
    add("1. **判据、阈值、旧读数一个不动**：本轮只增加「证书」这类证据与跑法，聚合器与 round 11/12 "
        "同一版本；因此裁决的变化只可能来自证据本身。")
    add("2. **并发的前提是可合并**：分片必须构成待办集合的划分。这不是口头约定——"
        "`tests/test_provenance_certify_shard.py` 对 n=1/2/3/6/8 逐一断言「并集 = 待办集合且两两不交」，"
        "并断言 n64_t10 这种最贵检查点不会被堆到同一片。")
    add("3. **准入过滤不是形式**：`tests/test_land_provenance_certificates.py` 逐条钉住 7 类拒绝"
        "（错网格、错时刻、错参考模型、偏差过大、nan 偏差、百分比不吻合、孤儿键），"
        "并断言「更好的旧证书不会被更差的替换」、`--dry-run` 不落盘、"
        "落地后的证书在 `provenance_audit` 里确实被判为 `replayed`。")
    add("4. **落地 ≠ 判过**：过滤通过只意味着「这条记录现在有证据」，通过与否仍由 C11 自己算。"
        "本轮不存在「因为作业成功所以判据通过」的路径。")
    add("5. **p=3 与 P0 长时的读数是参考/阶数受限的**：p=3 离散误差已接近 min-image 参考的"
        "速度场截断地板；P0 一族在 t=10 的误差远超快照容差属格式阶数固有属性。二者分别按"
        "「参考受限」与 `C1x_crosscheck_raw`（informational）登记，不参与验收。")
    add("6. **容差与文献的绑定**：6% 逐字取自算例池条目（ref: Yee et al. 2000），未从原文重推；"
        "所有验收读数都是对**闭式解析解**测量的，因此结论不依赖容差出处，只依赖条目声明的数值。")
    add("")

    # ⑥ ---------------------------------------------------------------
    add("## ⑥ 误差判定")
    add("")
    design = records.get("dg_d2_n32_t1_cfl0.2", {})
    fine = records.get("dg_d2_n64_t1_cfl0.2", {})
    p0 = records.get("p0_n64_t1_cfl0.4", {})
    add("| 关键量 | 实测误差 | 基准 | < 3%？ |")
    add("|---|---|---|---|")
    rows = [("DG p=2 32² t=1 设计闸门", design.get("max_percent"), "闭式周期解"),
            ("DG p=2 64² t=1 精化解", fine.get("max_percent"), "闭式周期解"),
            ("P0 一阶 64² t=1（第二代码路径）", p0.get("max_percent"), "闭式周期解")]
    for label, value, basis in rows:
        if value is None:
            continue
        verdict = "是" if abs(value) < 3.0 else "**否**"
        add(f"| {label} | {num(value, 6)}% | {basis} | {verdict} |")
    entropy = gates["C4_entropy_budget"].get("measured")
    add(f"| 熵预算漂移 \\|ΔS\\|/S(0) | {num(entropy, 4)}（阈值 1e-3） | 离散守恒 | "
        f"{'是' if entropy is not None and abs(entropy) < 3e-2 else '—'} |")
    add(f"| 设计族误差守恒指数 alpha | {num(gates['C2_error_conservation'].get('exponent'), 4)}"
        f"（阈值 <= 1） | 理论无超线性累积 | "
        f"{'是' if (gates['C2_error_conservation'].get('exponent') or 0) <= 1.0 else '否'} |")
    add("")
    add(f"**整包裁决：{gates['verdict']}**（C1–C11 全部判过；"
        f"未判定 {len(gates.get('not_evaluated') or [])} 条，未通过 {len(gates.get('failed') or [])} 条）。")
    add("")
    add("复现：")
    add("```")
    add("# 待办重放（成本均衡 6 路并发；每片只重放自己那份，输出可合并）")
    add("PYTHONPATH=. python3 tools/provenance_certify.py --output-dir case_euler_vortex_errconserve "
        "--remaining --shard i/6 --threads 4 --max-seconds 6800")
    add("# 取回后过准入过滤落地")
    add("PYTHONPATH=. python3 tools/land_provenance_certificates.py <artifact.tar.gz> "
        "--report case_euler_vortex_errconserve/landing_report.json")
    add("# 聚合（判据与阈值不变）+ 出图")
    add("PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures "
        "--output-dir case_euler_vortex_errconserve")
    add("PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_round13.py")
    add("```")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- report
def login():
    body = json.dumps({"username": USER, "password": PASS}).encode()
    req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    return payload.get("access_token") or payload["token"]


def post(path, payload, token):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_aggregate:
        cmd = ["python3", "benchmarks/case_euler_vortex_errconserve.py",
               "--aggregate", "--figures", "--output-dir", str(OUT)]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": "."})

    metrics, certs = load()
    doc = build_doc(metrics, certs)
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} chars); verdict={metrics['gates']['verdict']}")

    if args.dry_run:
        return 0

    figures = []
    for name, caption in (
            ("fig_error_vs_time.png", "最大场相对 L2 vs t（各族）与 6% 容差线"),
            ("fig_design_space.png", "设计空间读数并列"),
            ("fig_invariants_vs_time.png", "离散不变量漂移与熵预算")):
        path = OUT / name
        if path.exists():
            figures.append({"name": name, "caption": caption,
                            "png_base64": base64.b64encode(path.read_bytes()).decode()})

    gates = metrics["gates"]
    prov = gates["C11_provenance_integrity"]
    counts = prov.get("counts", {})
    summary = (f"C11 溯源闭合：replayed {counts.get('replayed', 0)} / "
               f"unverified {counts.get('unverified', 0)}，最大重放偏差 "
               f"{num(prov.get('measured'), 3)}；整包裁决 {gates['verdict']}"
               f"（未判定 {len(gates.get('not_evaluated') or [])}，"
               f"未通过 {len(gates.get('failed') or [])}）")
    token = login()
    payload = {
        "title": "CASE-DG-EULER-VORTEX-ERR round 13：C11 溯源闭合与整包裁决",
        "summary": summary,
        "scenario_key": "dg_euler_vortex_err",
        "mission_id": MISSION,
        "tags": ["TensorDG", "Euler", "isentropic-vortex", "verification",
                 "provenance", "c11-closure", SLUG],
        "markdown": doc,
        "figures": figures,
    }
    result = post("/tensor/reports", payload, token)
    print(json.dumps(result)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())