import base64, json, os, urllib.request
from pathlib import Path
BASE = os.environ["TENSOR_API_BASE"]
OUT = Path("docs/benchmarks/results/lax_certified_holdout")

def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                           "password": os.environ["TENSOR_AGENT_PASS"]})
TOKEN = tok.get("access_token") or tok.get("token")
rep = json.loads((OUT / "report_extrapolation.json").read_text())
payload = json.loads((OUT / "extrapolation.json").read_text())
caps = {
 "holdout_frontier.png": "图 1 DOF–误差平面：枚举腿（按 p=1/2/3 分形）、DOF 幂律、3% 台账线；★=本次未参与拟合的外样本腿（实测，误差棒=模型 LOO 窗口），×=点预测。p3.N1600 的 DOF 6400 超出枚举拟合区间（800–3200），故该点是外插检验。",
 "holdout_error_in_windows.png": "图 2 每条腿的 |实测−预测| / 预登记窗口（两模型并列）；1.0 虚线即窗口，越过即模型外推不足。4 个比值全部 <1。",
}
figs = [{"name": f, "caption": caps.get(f, f),
         "png_base64": base64.b64encode((OUT / "figures" / f).read_bytes()).decode()}
        for f in [Path(x).name for x in payload["figures"]]]
body = {
 "title": "CASE-LAX-DG 认证前沿能否外推：用两条从未参与拟合的腿做 out-of-sample 检验（盲测腿预登记、测量待集群）",
 "summary": ("枚举认证 p2.N800 的「最小 DOF」只对 8 条枚举腿成立。本单用分辨率阶梯里两条从未参与拟合的腿"
             "（p3.N200，DOF 800；p3.N1600，DOF 6400，超出拟合区间）检验误差模型的 out-of-sample 外推："
             "4/4 预测落在模型自身 LOO 窗口内，4/4 PASS/FAIL 判定复现，family 模型在 N=200 上偏差仅 0.007 个百分点；"
             "另预登记两条盲测腿（p2.N600 / p3.N600）的结构化预测文件，测量因集群排队未到账，如实记录为未完成。"),
 "scenario_key": "case-lax-dg-certified-holdout",
 "mission_id": 21904,
 "tags": ["CASE-LAX-DG", "TensorDG", "out-of-sample", "pre-registration",
          "verification-debt", "certification-frontier"],
 "markdown": rep["markdown"],
 "figures": figs,
}
res = post("/tensor/reports", body, TOKEN)
print(json.dumps(res, ensure_ascii=False)[:400])
