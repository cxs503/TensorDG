"""Publish the transcritical-bump verification report (no secrets on the command line)."""
import base64
import json
import pathlib
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
USER = __import__("os").environ["TENSOR_AGENT_USER"]
PASSWORD = __import__("os").environ["TENSOR_AGENT_PASS"]

HERE = pathlib.Path(__file__).resolve().parent
FIG = HERE / "fig_transcritical_bump.png"
DOC = HERE.parents[1] / "docs" / "CASE-DG-TRANSCRITICAL-BUMP-SHOCK.md"


def post(path, payload):
    request = urllib.request.Request(
        f"{BASE}/{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, response.read().decode()


def post_auth(path, payload, token):
    request = urllib.request.Request(
        f"{BASE}/{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.status, response.read().decode()


report = {
    "title": "TensorDG 跨临界驼峰定常流（驻激波）解析解与生产格式验证：良态平衡在间断处失效",
    "summary": ("构造 Goutal & Maurel test 3 (SWASHES 'Bump with shock') 的跨临界定常解析解"
                "（峰顶临界 + 驻激波，R-H 定位 x_s=11.665618 m），用独立 RK4 稳态 ODE 积分校验；"
                "生产 DG 格式的稳态缺陷在激波处 O(1) 且随加密增大（对照光滑亚临界档 ≈1 阶收敛），"
                "时间推进在 t≈6-7 s (N=100) 因 dt 下溢中止 —— honest FAIL。"),
    "scenario_key": "tensordg.swe_steady_bump.transcritical",
    "mission_id": 16859,
    "tags": ["TensorDG", "浅水方程", "well-balanced", "定常激波", "验证", "honest-FAIL"],
    "markdown": DOC.read_text(),
    "figures": [{
        "name": FIG.name,
        "caption": ("左：跨临界精确解（自由水面/深度/地形/驻激波位置/临界深度）；"
                    "中：稳态缺陷 vs dx（跨临界平坦–上升 vs 亚临界 ≈1 阶下降）；"
                    "右：从精确态时间推进的存活时间（红=dt 下溢中止）"),
        "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
    }],
}

status, body = post("auth/login", {"username": USER, "password": PASSWORD})
token = json.loads(body).get("access_token") or json.loads(body).get("token")
print("login", status, "token_len", len(token or ""))
status, body = post_auth("tensor/reports", report, token)
print("report", status, body[:300])