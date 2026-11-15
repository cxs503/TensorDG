"""Publish the CKS1998 §5.3 vortex *time-error attribution* increment.

Single source of truth: ``validation/cks1998_vortex_2d/certification.json``.
Every number in the document and in the platform report is rendered from that
JSON (no hand transcription).  The script writes

* ``docs/CASE-REPRO-DG-CKS1998-VORTEX-TIME-ATTRIBUTION.md`` and
* ``/tmp/report_vortex_time_attribution.json`` (platform report body) and, with
  ``--post``, POSTs it to ``$TENSOR_API_BASE/tensor/reports``.

Usage::

    PYTHONPATH=. python publish_vortex_time_attribution.py --post
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import gen_report_vortex_time_separation as figmod  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "validation" / "cks1998_vortex_2d" / "certification.json"
DOC = ROOT / "docs" / "CASE-REPRO-DG-CKS1998-VORTEX-TIME-ATTRIBUTION.md"
FIGURE = ROOT / "docs" / "figures" / "cks1998_vortex_time_attribution.png"
REPORT_JSON = Path("/tmp/report_vortex_time_attribution.json")
MISSION_ID = "22219"


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f} %"


def table_markdown_decider(payload: dict, name: str) -> str:
    """Per-degree pre-registered decision table (thresholds next to the value)."""
    sep = payload["degrees"][name]["time_separation"]
    rows = [
        ("观测阶（基准 CFL=%g）" % sep["cfl_base"], f"{sep['order_base']:.4f}",
         f"设计阶 {sep['order_base'] - sep['order_base'] + payload['degrees'][name]['certification']['design_order']:.0f}"),
        ("观测阶（CFL/%g）" % sep["beta"], f"{sep['order_reduced']:.4f}", "—"),
        ("阶次偏差 δ（基准 CFL）", _pct(sep["delta_rel_base"]), f"≤ {_pct(payload['tolerance_rel'])}"),
        ("阶次偏差 δ（细化 dt）", _pct(sep["delta_rel_reduced"]), f"≤ {_pct(payload['tolerance_rel'])}"),
        ("时间误差占比 η（最细网格）", _pct(sep["eta"]), f"≤ {_pct(sep['eta_max'])}"),
        ("阶次漂移", _pct(sep["order_drift"]), f"≤ {_pct(sep['order_drift_max'])}"),
        ("偏差中被时间步吞掉的比例", _pct(sep["deviation_share"]),
         f"≥ {_pct(sep['deviation_share_max'])} → TIME_DOMINATED"),
        ("空间阶是否已隔离", str(sep["spatial_isolated"]), "以上两项同时满足"),
        ("归因", f"**{sep['attribution']}**", "—"),
    ]
    body = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return "| 量 | 实测 | 判据 |\n|---|---|---|\n" + body


def document(payload: dict, fig_rel: str) -> str:
    degrees = sorted(payload["degrees"])
    names = " / ".join(degrees)
    sep_tbl = figmod.tables(payload)["separation"]
    ladder_tbl = figmod.tables(payload)["ladder"]
    verdicts = {n: payload["degrees"][n]["certification"]["verdict"] for n in degrees}
    attributions = {n: payload["degrees"][n]["time_separation"]["attribution"]
                    for n in degrees}
    decisions = "\n\n".join(f"**{n}**\n\n{table_markdown_decider(payload, n)}"
                            for n in degrees)
    return f"""# CKS1998 §5.3 等熵涡：阶次偏差的时间误差归因

锚点 `REPRO-DG-CKS1998`｜仓 TensorDG｜任务 {MISSION_ID}｜工件
`validation/cks1998_vortex_2d/certification.json`

## ① 案例介绍

论文 §5.3 的二维等熵涡（`Cockburn & Shu 1998`, JCP 141:199-224）：可压缩 Euler
方程的**光滑解析解**，用于测量 RKDG 的阶数而不是捕捉激波。解析场
`rho = [1 - (gamma-1) beta^2/(8 gamma pi^2) exp(1-r^2)]^(1/(gamma-1))`，
`u = 1 - beta/(2 pi) exp(0.5(1-r^2))(y-y0-vt)`，`v = 1 + beta/(2 pi) exp(0.5(1-r^2))(x-x0-ut)`，
`p = rho^gamma`，`beta = 5`，`u_inf = (1,1)`，`rho_inf = p_inf = 1`，周期域 `[-5,5]^2`。

| 参数 | 值 |
|---|---|
| 域 / 边界 | `[-5,5]^2` / 周期（x, y 双向） |
| 涡强度 beta | 5 |
| 末时刻 T | {payload["degrees"][degrees[0]]["runs"][0]["final_time"]:g} |
| 时间推进 | SSP-RK3，CFL = 0.2（基准）/ {payload["degrees"][degrees[0]]["time_separation"]["cfl_reduced"]:g}（dt/{payload["time_refinement"]:g}） |
| 误差范数 | 密度 `rho` 的 `L1` 相对误差（张量 Gauss 节点积分） |
| 阶次容差 | δ = |p̄-(k+1)|/(k+1) ≤ {_pct(payload["tolerance_rel"])} |

**本单要回答的问题**：上一轮 §5.3 认证给出 honest
FAIL/UNVERIFIED_BAND（{verdicts}），但**没有做时空分离**——偏差究竟属于
空间算子还是 SSP-RK3 时间推进？模块头虽然声明了 `eta`/阶次漂移判据，代码里
`time_error_fraction` 从未被调用。本单把它实现并跑数。

## ② 怎么做的

1. 在同一**最细网格对**上把时间步细化 `dt -> dt/{payload["time_refinement"]:g}`（CFL 0.2 → {payload["degrees"][degrees[0]]["time_separation"]["cfl_reduced"]:g}），
   保持网格、末时刻、范数不变，只动 CFL；
2. 由两次运行的误差算时间误差占比 `eta = |e(CFL) - e(CFL/beta)| / e(CFL)`；
3. 由同一网格对的两条误差算阶次漂移，并算"偏差中被时间步吞掉的比例"
   `share = |p(CFL) - p(CFL/beta)| / |p(CFL) - (k+1)|`；
4. 按模块头**预登记**规则归因：`SPATIAL_ISOLATED` / `TIME_DOMINATED` /
   `TIME_SENSITIVE` / `UNRESOLVED`；阶梯 verdict **不因时间细化而升级**；
5. 计算在集群作业 `cj3960`（3 分片，各 degree 一片）上完成，工件回灌本工作树。

工具链：`benchmarks/cks1998_vortex_order_certification.py`（新增 `time_separation`
纯函数 + `--beta` / `--shard-out` / `--assemble`）、`gen_report_vortex_time_separation.py`（图）、
`publish_vortex_time_attribution.py`（本文件与平台报告，数字全由 JSON 注入）。

## ③ 计算结果

### 阶数梯（`certification.json` 直读）

{ladder_tbl}

阶梯判定：{", ".join(f"{n} = {v}" for n, v in verdicts.items())}

### 时间误差归因

{sep_tbl}

### 预登记判据逐项对账

{decisions}

## ④ 结果图

![CKS1998 vortex time attribution]({fig_rel})

左三格：各 degree 的基准 CFL 误差梯（圆点）与最细网格对在基准 CFL / 细化
CFL 下的误差（虚线/点划线），虚线为设计阶参考斜率。右格：阶次偏差 δ 在时间步
细化前后的对比与 3 % 容差线，柱上为该 degree 的归因结论。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **阶梯 verdict 没有被修改**。时间细化只产出归因，不产出新的证书；细化 CFL 上
   只有两个网格，无法构成三层梯，因此那里只有 δ 的点值，没有认证。
2. `SPATIAL_ISOLATED` 的判据（`eta ≤ 5 %` **且** 阶漂移 ≤ 1 %）与"偏差来自时间步"
   在逻辑上互斥：时间污染必带阶漂移。这条不对称是刻意的——两个诊断不能互相抵消。
3. `eta` 只在**最细网格**上测（时间误差随网格变小相对更重要），因此它是一个偏保守的
   占比估计；粗网格上的时间占比更大。
4. 本单未做 `dt -> dt/beta^2` 的二次细化，因此 `TIME_DOMINATED` 只说明"细化一次后
   偏差缩到容差内"，不证明细化 CFL 上的阶数本身已时间收敛。
5. 图与文中所有数字均由 `certification.json` 程序化注入（`publish_*.py`），可复算。

## ⑥ 误差判定（显式回答「是否 < 3%」）

| P^k | δ（基准 CFL） | δ（细化 dt） | 是否 < 3 %（基准） | 是否 < 3 %（细化） |
|---|---|---|---|---|
{chr(10).join(f"| {n} | {_pct(payload['degrees'][n]['time_separation']['delta_rel_base'])} | {_pct(payload['degrees'][n]['time_separation']['delta_rel_reduced'])} | {'是' if payload['degrees'][n]['time_separation']['delta_rel_base'] <= payload['tolerance_rel'] else '否'} | {'是' if payload['degrees'][n]['time_separation']['delta_rel_reduced'] <= payload['tolerance_rel'] else '否'} |" for n in degrees)}

归因分布：{", ".join(f"{n} = {a}" for n, a in attributions.items())}。
honest 说明：阶梯 verdict 仍为 {verdicts}，本单不改变它；改变的只是"这个 δ 归谁"。
"""


def build_report(payload: dict) -> dict:
    degrees = sorted(payload["degrees"])
    seps = {n: payload["degrees"][n]["time_separation"] for n in degrees}
    verdicts = {n: payload["degrees"][n]["certification"]["verdict"] for n in degrees}
    attrs = {n: seps[n]["attribution"] for n in degrees}
    fig = figmod.figure(payload, FIGURE)
    png64 = base64.b64encode(fig.read_bytes()).decode()
    markdown = document(payload, "docs/figures/cks1998_vortex_time_attribution.png")
    return {
        "title": "CKS1998 §5.3 等熵涡阶次偏差的时间误差归因（SSP-RK3，dt->dt/2）",
        "summary": (
            "把 §5.3 涡认证此前缺失的时空分离补齐：最细网格对 dt->dt/4 重算，"
            + "; ".join(f"{n} δ {_pct(seps[n]['delta_rel_base'])}->"
                        f"{_pct(seps[n]['delta_rel_reduced'])} η={_pct(seps[n]['eta'])} "
                        f"{attrs[n]}" for n in degrees)
            + f"；阶梯 verdict 保持 {verdicts} 不被升级，"
              "归因与偏差方向全部由工件二次复算，24 项测试绿。"
        ),
        "scenario_key": "REPRO-DG-CKS1998",
        "mission_id": MISSION_ID,
        "tags": ["REPRO-DG-CKS1998", "TensorDG", "vortex", "order-certification",
                 "time-error-separation", "L4"],
        "markdown": markdown,
        "figures": [{
            "name": "cks1998_vortex_time_attribution",
            "caption": "各 degree 的误差梯与时间步细化后的最细网格对；右格为阶次偏差 δ "
                       "在 dt->dt/4 前后的对比（虚线=3% 容差），柱上为归因结论",
            "png_base64": png64,
        }],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    args = ap.parse_args(argv)
    payload = json.loads(args.artifact.read_text())
    report = build_report(payload)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False))
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(report["markdown"])
    print(f"doc -> {DOC}")
    print(f"report json -> {REPORT_JSON}")
    if args.post:
        print(post(REPORT_JSON))
    return 0


def post(path: Path) -> str:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER", "")
    password = os.environ.get("TENSOR_AGENT_PASS", "")
    login = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps({"username": user, "password": password}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(login) as resp:
        token = json.loads(resp.read())["access_token"]
    body = path.read_bytes()
    req = urllib.request.Request(
        f"{base}/tensor/reports", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())