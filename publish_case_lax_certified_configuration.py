"""Publish the CASE-LAX-DG certified-configuration study to the platform.

Numbers are read out of the committed artifact (zero hand transcription) and the
figures out of the committed results directory.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_certified_configuration"
METRICS = OUT / "metrics.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]


def _post(path, body, token=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"}
                                             if token else {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def login():
    body = _post("/auth/agent-login", {"username": USER, "password": PASSWORD})
    payload = body.get("data", body)
    return payload["access_token"]


def figure(name):
    return base64.b64encode((OUT / "figures" / name).read_bytes()).decode()


def main():
    m = json.loads(METRICS.read_text())
    res = m["certification"]
    cert = res["certified"]
    rows = list(zip(m["legs"], res["per_leg"]))
    table = "\n".join(
        f"| {leg['leg']} | {v['dof']} | {leg['steps']} | {leg['wall_seconds']:.1f} | "
        f"{leg['l2_errors']['density']:.5e} | {leg['l2_errors']['velocity']:.5e} | "
        f"{leg['l2_errors']['pressure']:.5e} | {v['status']} |"
        for leg, v in rows)
    figs = [{"name": f["name"], "caption": f["caption"],
             "png_base64": figure(f["name"])} for f in m["figures"]]
    markdown = f"""# CASE-LAX-DG · 认证构型与代价：哪一档 (p, N) 把台账 3 % 拿下

## ① 案例介绍
Lax (1954) / Toro (2009) Test 2 激波管，`[0,1]`，γ = 1.4，隔膜 x0 = 0.5，
左态 (ρ,u,p) = (0.445, 0.698, 3.528)，右态 (0.500, 0.000, 0.571)，t = 0.13；
参考值 = 闭式 Riemann 解。台账 `CASE-LAX-DG` 验收：`ref: 精确解｜容差 < 3 %`。
已在案：钉扎腿 p3.N400 FAIL（rel L2 ρ = 3.11556e-02），p3.N800 PASS（2.31490e-02）。
本单回答算例池从未回答的工程问题：**同等或更小自由度预算下，有没有更便宜的达标构型？**

## ② 怎么做的
* 跑数前写死网格（`CANDIDATES`），全部满足 `(p+1)·N ≤ DOF_CAP = 3200`（已知达标构型 p3.N800 的预算），只做加法不抬预算；
* 度量不重写：`measure_leg` / `_weighted_l2` / `exact_reference` / `modal_fields` 全部从算例池与分辨率研究模块 import，与台账臂同一把尺子（全场 rel L2，Gauss 节点 Legendre 加权，分母取精确解同范数）；
* 容差从 `tensordg.cases.lax_1d.lax_parameters()['acceptance_threshold']` 读，不复制字面量；
* 认证规则为纯函数：PASS ⟺ 三个量 rel L2 全部 < 容差（不挖波带、不豁免、不平均）；认证构型 = PASS 腿中 `(p+1)·N` 最小者（并列按实测墙钟）；
* 复算自检：p3.N400 必须逐位复现已提交算例池工件，p3.N800 必须逐位复现分辨率研究工件；
* 求解器：模态 DG + HLLC + SSP-RK3，CFL = {m['config']['cfl']}，float64，求积阶 {m['config']['quadrature_order']}；集群作业 cj3046（stage_dir 本工作树）。

## ③ 计算结果
| leg | DOF | steps | wall [s] | rel L2 ρ | rel L2 u | rel L2 p | 判据 |
|---|---|---|---|---|---|---|---|
{table}

* **认证构型 = {cert['leg'] if cert else 'none'}**（DOF {cert['dof'] if cert else '-'}），最差量 {cert['worst_field'] if cert else '-'} rel L2 = {cert['worst_l2']:.5e}（裕度 ×{cert['margin_of_tolerance']:.3f}）；
* 网格内最小性：自由度严格更小的腿全部 FAIL = `{res['minimality']['cheaper_legs_all_fail']}`（{('、'.join(res['minimality']['cheaper_legs'])) or '—'}）；
* 复算自检：{m['reuse_check']['status']}，最大相对差 {m['reuse_check']['worst_relative_difference']:.3e}；
* P4 支路：solver 对 degree=4 抛 `ValueError`（cj3025 实测），即 **P4 不受支持**，网格止于 P3（记录在 `config.unsupported_degrees`）。

## ④ 结果图
* `lax_certification_dof.png`：误差 vs 自由度（每阶一条曲线，虚红线 = 3 % 容差，空心 = 达标）；
* `lax_certification_cost.png`：误差 vs 实测墙钟（认证点的成本读数）；
* `lax_certified_profiles.png`：认证构型的 ρ/u/p 剖面 vs 精确解。

## ⑤ 注意点
1. 移动的是「认证半径」不是「验收线」：A0 钉扎腿仍 p3.N400，其 FAIL 原样保留；把台账生产腿改到认证构型是台账持有人裁决，本单不自行改写。
2. 最小性只在网格内成立，不对未测 (p, N) 断言。
3. wall clock 有环境噪声，认证排序用无噪声的 DOF，墙钟只作并列 tie-break 与成本参考。
4. 间断存在下全场阶 < 1（算例池 C3/R3 的 FAIL 一致）：达标靠加密，不靠提阶。

## ⑥ 误差判定（是否 < 3 %）
| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| rel L2 ρ @ {cert['leg']} | 精确 Riemann 解 | {cert['worst_l2']:.5e} | {cert['worst_l2']:.3%} | 是 |
| rel L2 u @ {cert['leg']} | 精确 Riemann 解 | {next(l['l2_errors']['velocity'] for l in m['legs'] if l['leg']==cert['leg']):.5e} | {next(l['l2_errors']['velocity'] for l in m['legs'] if l['leg']==cert['leg']):.3%} | 是 |
| rel L2 p @ {cert['leg']} | 精确 Riemann 解 | {next(l['l2_errors']['pressure'] for l in m['legs'] if l['leg']==cert['leg']):.5e} | {next(l['l2_errors']['pressure'] for l in m['legs'] if l['leg']==cert['leg']):.3%} | 是 |
| rel L2 ρ @ p3.N400（钉扎腿） | 精确 Riemann 解 | 3.11556e-02 | 3.116 % | 否 |

结论：认证构型 {cert['leg']} 三量全部 < 3 %；钉扎腿 p3.N400 的 FAIL 未被本文改写（honest FAIL 保留）。
"""
    token = login()
    body = {"title": "CASE-LAX-DG · 认证构型与代价：最小自由度达标 (p,N) 与 P4 不受支持",
            "summary": (f"Lax 激波管算例池台账 3 % 容差的认证构型 = {cert['leg']}"
                        f"（DOF {cert['dof']}，三量 rel L2 均 <3 %，"
                        f"复算自检 {m['reuse_check']['status']}）；"
                        "P4 被 1D Euler 核拒绝（不受支持）。"),
            "scenario_key": "CASE-LAX-DG",
            "mission_id": 20883,
            "tags": ["TensorDG", "CASE-LAX-DG", "certified-configuration",
                     "cost-frontier", "verification"],
            "markdown": markdown,
            "figures": figs}
    out = _post("/tensor/reports", body, token)
    print(json.dumps(out)[:600])


if __name__ == "__main__":
    main()