"""Publish the phase/shape-split report to the platform (numbers + figures injected)."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "case_euler_vortex_phase_shape"
NUMBERS = json.loads((BUNDLE / "report_numbers.json").read_text())
DOC = (ROOT / "docs" / "CASE-DG-EULER-VORTEX-ERR-phase-shape-split.md").read_text()
API = os.environ.get("TENSOR_API_BASE", "http://localhost:8001/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")


def token() -> str:
    body = json.dumps({"username": USER, "password": PASSWORD}).encode()
    for path in ("/auth/login", "/auth/token"):
        request = urllib.request.Request(API + path, data=body,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            for key in ("access_token", "token"):
                if key in payload:
                    return payload[key]
            if isinstance(payload.get("data"), dict):
                for key in ("access_token", "token"):
                    if key in payload["data"]:
                        return payload["data"][key]
        except urllib.error.HTTPError:
            continue
    raise RuntimeError("login failed")


def figures():
    out = []
    captions = {
        "fig_vortex_split_curves.png": "各族总误差（实线）与去掉平移滞后后的形状误差（虚线），右图为纯滞后本身能解释的误差。",
        "fig_vortex_split_lag.png": "最小二乘平移滞后 d(t)（单位为网格格数），虚线为 ±0.5 格搜索窗。",
        "fig_vortex_split_budget.png": "末点误差预算（形状 vs 传输）与仪器校准读数。",
    }
    for name, caption in captions.items():
        path = ROOT / "docs" / "figures" / name
        out.append({"name": name, "caption": caption,
                    "png_base64": base64.b64encode(path.read_bytes()).decode()})
    return out


def main() -> int:
    claims = NUMBERS["claims"]
    families = NUMBERS["families"]
    design = families.get("dg_p2_n32", {})
    p0 = families.get("p0_n64", {})
    summary = (
        "把锚点「移动误差守恒」的读数拆成传输滞后与形状两部分：新增带限 FFT 最小二乘平移仪器"
        f"（K1–K4 校准全过，平移回读误差 {NUMBERS['calibration']['k1_recovery']['worst_abs_error_cells']:.1e} 格）。"
        f"设计点 t=1 的误差里纯滞后只占 {claims['B_design_t1_is_shape']['value']:.2f}%；"
        f"P0 到 t=10 滞后占 {claims['C_p0_transport_share']['value']:.2f}%，"
        f"去掉滞后后的形状误差仍增长 {claims['D_p0_shape_growth']['value']:.2f} 倍 —— "
        "「误差增长=相位误差」这一常见解释在本算例被否证；设计族全梯子 alpha 因集群作业 "
        f"{NUMBERS['pending_job']} 未落而记 undecided。"
    )
    markdown = "\n".join([
        summary,
        "",
        "## 判据读数",
        "",
        "| 判据 | 读数 | 判定 |",
        "|---|---|---|",
    ] + [
        f"| {name}：{item['text']} | "
        f"{'—' if item['value'] is None else round(item['value'], 6) if isinstance(item['value'], float) else item['value']} | "
        f"{'undecided' if not item['decided'] else ('passed' if item['passed'] else 'FAILED')} |"
        for name, item in claims.items()
    ] + [
        "",
        f"设计族 alpha_shape = {design.get('alpha_shape')}（检查点 {design.get('checkpoints') and len([c for c in design['checkpoints'] if c['t']>0])} 个，未达 3 个 → undecided）",
        f"P0 各族 e_total/e_shape 见正文表 3.2。",
        "",
        "## 全文",
        "",
        DOC,
    ])
    payload = {
        "title": "CASE-DG-EULER-VORTEX-ERR：移动误差守恒的传输/形状分解（带校准仪器）",
        "summary": summary,
        "scenario_key": "case_dg_euler_vortex_err",
        "mission_id": 23160,
        "tags": ["TensorDG", "CASE-DG-EULER-VORTEX-ERR", "verification", "phase-error",
                 "error-conservation"],
        "markdown": markdown,
        "figures": figures(),
    }
    request = urllib.request.Request(
        API + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        print(response.status, json.load(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())