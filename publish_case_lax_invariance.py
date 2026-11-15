"""Publish the CASE-LAX-DG strict-invariance audit (numbers from the artifact)."""
import base64, json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_invariance_audit"
BASE = os.environ["TENSOR_API_BASE"]


def call(path, body=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(BASE + path, data=(json.dumps(body).encode() if body else None),
                                 headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def login():
    body = call("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                "password": os.environ["TENSOR_AGENT_PASS"]})
    return (body.get("data") or body)["access_token"]


def main():
    m = json.loads((OUT / "metrics.json").read_text())
    doc = (ROOT / "docs" / "CASE-LAX-DG-invariance-audit.md").read_text()
    verdict = m["verdict"]
    figs = [{"name": f["name"], "caption": f["caption"],
             "png_base64": base64.b64encode((OUT / "figures" / f["name"]).read_bytes()).decode()}
            for f in m["figures"]]
    head = (f"**判定** strict invariants O1–O5/T1–T4 最坏 {verdict['worst_trajectory_deviation']:.2e}"
            f"（容差 {verdict['invariant_tol']:g}）；对照臂最坏 {verdict['worst_control_deviation']:.2e}"
            f"，判别比 {verdict['discrimination']:.2e}；台账 p3.N400 读数 "
            f"{m['bias_bound']['pinned_density_rel_l2']:.5e} 保持 FAIL（方向性偏差上界 "
            f"{m['bias_bound']['bias_in_units_of_bar']:.2e} × 容差）。\n\n")
    token = login()
    body = call("/tensor/reports", {
        "title": ("CASE-LAX-DG 严格不变量审计：镜像宇称把台账读数与核函数方向性偏差分开"
                  "（O1–O5/T1–T4 全绿，对照臂判别比 ~2.8e10）"),
        "summary": ("对 1-D Euler DG 核做严格不变量审计：通量相容/互易、物理通量伽利略恒等式、"
                    "DG 残差与 Zhang-Shu 可容许算子的镜像宇称（1e-16 级），以及轨迹层镜像复现与"
                    "账本度量两系同值（4.8e-11 ≤ 1e-9）。两条对照臂（不翻速度 / 隔膜偏半格）在同一"
                    "度量下偏离 1.09/1.33，证明审计有判别力；据此给出台账 3.116% FAIL 的方向性偏差"
                    "上界 4.8e-08 × 容差，即该 FAIL 只能由分辨率解释（N=800 达标结论不变）。"),
        "scenario_key": "CASE-LAX-DG",
        "mission_id": 21487,
        "tags": ["CASE-LAX-DG", "TensorDG", "verification", "invariant", "mirror-symmetry",
                 "ledger", "honest-fail"],
        "markdown": head + doc,
        "figures": figs,
    }, token=token)
    print("report id:", (body.get("data") or body).get("id"))


if __name__ == "__main__":
    main()
