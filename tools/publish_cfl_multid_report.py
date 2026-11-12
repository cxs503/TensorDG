"""Publish the DG multid CFL study as a platform report (mission 16676).

Numbers come from ``validation/cfl_stability_multid/results.json``; the figure is
the one the study generated, quantised so the payload stays small (platform
rejects non-PNG payloads and chokes on >100 KB uploads).
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "validation/cfl_stability_multid"
MISSION = 16676


def token(base: str) -> str:
    resp = requests.post(f"{base}/auth/login", json={
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("access_token") or data["token"]


def png_payload(path: Path, width: int = 680, colors: int = 64) -> str:
    img = Image.open(path).convert("RGB")
    height = int(round(img.height * width / img.width))
    img = img.resize((width, height), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=colors)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    doc = (REPO / "docs/DG-CFL-STABILITY-LIMIT-MULTID.md").read_text(encoding="utf-8")
    fig = results["figures"][0]
    payload = {
        "title": "DG 多维显式 RK 稳定性上限实测：方向缩放精确、默认 cfl 有裕度",
        "summary": (
            "张量积 DG 的 dt=cfl/((2p+1)Σ|a_i|/h_i) 在多维下的两半此前未验证。实测："
            f"Kronecker 和谱给出 dt_max(d) = dt_max(1)/d（最大偏差 "
            f"{max(r['rel_error_vs_unity'] for r in results['structural_scaling']):.1e}），"
            "代码单位 cfl 方向不变（≤1e-15 量级）；2-D/3-D 生产求解器二分实测与同网格符号同序；"
            f"2-D Euler 爆破上限最小 {min(r['cfl_ceiling'] for r in results['euler_blow_up_ceiling']):.2f}，"
            f"默认 0.2 仍在界内。verdict={results['verdict']}。"
        ),
        "scenario_key": "tensordg-cfl-stability-multid",
        "mission_id": MISSION,
        "tags": ["TensorDG", "CFL", "stability", "verification", "multi-dimensional"],
        "markdown": doc,
        "figures": [{
            "name": fig,
            "caption": "左: dt_max vs p（d=1,2,3）；中: 2-D 对流实测 cfl_max vs 同网格符号；右: 2-D Euler 爆破 cfl",
            "png_base64": png_payload(OUT / fig),
        }],
    }
    text = json.dumps(payload)
    print("payload bytes:", len(text))
    r = requests.post(f"{base}/tensor/reports", json=payload,
                      headers={"Authorization": f"Bearer {token(base)}"}, timeout=120)
    print(r.status_code, r.text[:400])
    return 0 if r.status_code < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())