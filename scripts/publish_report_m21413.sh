#!/usr/bin/env bash
# Publish the m21413 progress report (payload + figure embedding + POST).
set -euo pipefail
cd "$(dirname "$0")/.."

python - <<'PY'
import base64, json, pathlib
png = base64.b64encode(pathlib.Path("docs/figures/cks1998_tvb_indicator_symmetry.png").read_bytes()).decode()
md = open("docs/CASE-REPRO-DG-CKS1998-TVB-INDICATOR-SYMMETRY.md").read()
payload = {
 "title": "TensorDG · REPRO-DG-CKS1998 TVB 指示器单侧化缺陷修复：方波无超调判据 1.40% FAIL → 0.00% PASS",
 "summary": ("定位并修复 TVB trouble-cell 判据的单侧化缺陷（deviation 只测 d_left，同号且 "
             "|d_left|<=|d_right| 时恒为 0），CKS1998 方波点值超调由 P1 2.58/1.40/1.19%"
             "（N=50/100/200）降为 0.000%，胞平均超调 0.15%→0.00%；限幅格式 L1 阶保持 0.65"
             "（>0.5），但 L1 绝对误差 +33%（如实登记）。"),
 "scenario_key": "tensordg-repro-cks1998",
 "mission_id": 21413,
 "tags": ["REPRO-DG-CKS1998", "TensorDG", "RKDG", "TVB limiter", "verification debt", "defect fix"],
 "markdown": md,
 "figures": [{"name": "cks1998_tvb_indicator_symmetry.png",
              "caption": ("(a) 方波点值越界 vs N（1% 门线），旧判据 P1 全越门且不随 h 收敛，"
                          "新判据贴零；(b) 胞平均 L1 vs h，两档判据阶均 ~0.65>0.5，"
                          "新判据抬高约 1/3。"),
              "png_base64": png}],
}
pathlib.Path("artifacts/report_m21413.json").write_text(json.dumps(payload))
print("payload bytes:", len(json.dumps(payload)))
PY

LOGIN_JSON=$(python -c 'import json,os;print(json.dumps({"username":os.environ["TENSOR_AGENT_USER"],"password":os.environ["TENSOR_AGENT_PASS"]}))')
TOKEN=$(curl -s -X POST "$TENSOR_API_BASE/auth/login" -H 'Content-Type: application/json' \
        -d "$LOGIN_JSON" | python -c 'import sys,json;d=json.load(sys.stdin);print(d.get("access_token") or d.get("token") or "")')
[ -n "$TOKEN" ] || { echo "LOGIN FAILED"; exit 1; }
curl -s -X POST "$TENSOR_API_BASE/tensor/reports" \
     -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d @artifacts/report_m21413.json