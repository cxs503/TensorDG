"""Publish the CASE-LAX-DG A0 production-leg closure to the platform."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_a0_production_leg_closure"
METRICS = OUT / "metrics.json"

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASSWORD = os.environ["TENSOR_AGENT_PASS"]
MISSION_ID = 23037


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
    payload = body.get("data", body)
    return payload["access_token"]


def pct(x):
    return f"{100.0 * x:.3f} %"


def build():
    m = json.loads(METRICS.read_text())
    o = m["old_production_leg"]
    n = m["new_production_leg"]
    markdown = f"""# CASE-LAX-DG · A0 生产腿闭合（production-leg closure）

## ① 案例介绍
Lax (1954) / Toro test 2 激波管：`[0,1]`、`x0=0.5`、γ=1.4、t=0.13；左态 `(ρ,u,p)=(0.445,0.698,3.528)`、右态 `(0.5,0,0.571)`。四道波：左稀疏头/尾、接触间断、右激波（后两者真跳变，全场 L2 收敛率被压到约 h^0.4）。台账 A0：10 点 Gauss 逐点对闭式 Riemann 解、全场相对 L2、取最差场量、与 3% 比较。

## ② 怎么做的
`gate_a0` 默认生产腿 = 阶梯最细腿 p3.N400（3.116%，超 3% 门槛，honest-FAIL）。验证综合报告推荐方案 1：把生产腿重钉到认证配置 p2.N800（DOF 2400，2.570%）。本单**不重跑求解器**，从两份已提交工件（case-pool 阶梯 + 认证配置枚举）重新裁决 A0 台账臂；p3.N400 的 honest-FAIL 独立保留。

## ③ 计算结果（程序化注入）
| 生产腿 | DOF | 最差 L2 | 判定 | 裕度 |
|---|---|---|---|---|
| 旧 p3.N400 | 1600 | {pct(o['worst_l2'])} | FAIL | x{o['margin']:.3f} |
| 新 p2.N800 | 2400 | {pct(n['worst_l2'])} | **PASS** | x{n['margin']:.3f} |

最小 DOF 通过腿：**p2.N800**（DOF 2400）。

## ④ 注意点
闭合是配置重读非精度提升；p3.N400 的 3.116%（间断主导）原样保留为 honest-FAIL，不被 p2.N800 的 PASS 改写；参考底限（精确解在 DG 空间的投影误差）非求解器缺陷。

## ⑤ 误差判定（<3%）
| 关键量 | 实测 | 是否<3% |
|---|---|---|
| p2.N800 密度 L2 | {pct(n['worst_l2'])} | ✅ |
| p3.N400 密度 L2 | {pct(o['worst_l2'])} | ❌（honest-FAIL 保留） |

**A0 闭合判定：PASS**（生产腿重钉到认证配置 p2.N800）。
"""
    return {
        "title": "CASE-LAX-DG · A0 生产腿闭合：认证配置 p2.N800 使台账 <3% 判据 PASS",
        "summary": "把 CASE-LAX-DG 的 A0 生产腿从 p3.N400（3.116%，honest-FAIL）重钉到认证配置 "
                   "p2.N800（2.570%），使台账 <3% 判据 PASS；不重跑求解器，从已提交工件重读裁决，"
                   "旧腿 honest-FAIL 独立保留。",
        "scenario_key": "CASE-LAX-DG",
        "mission_id": MISSION_ID,
        "tags": ["CASE-LAX-DG", "lax", "dg", "acceptance", "closure", "verification"],
        "markdown": markdown,
        "figures": [],
    }


def main():
    token = login()
    body = build()
    resp = _post("/tensor/reports", body, token=token)
    print(json.dumps(resp, indent=2)[:500])


if __name__ == "__main__":
    main()
