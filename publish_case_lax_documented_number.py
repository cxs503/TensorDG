"""Publish the CASE-LAX-DG documented-number audit to the platform report API."""
import base64, json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "benchmarks" / "results" / "lax_documented_number_audit"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1")


def post(path, body, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main():
    m = json.loads((OUT / "metrics.json").read_text())
    f, sweep, conv = m["findings"], m["metric_sweep"], m["converged"]
    tol, doc = m["tolerance"], m["documented_number"]
    rho = 100 * conv["640"]["density"]
    hist = m["historical_replay"]["readings"]["8"]
    fig = {name: base64.b64encode((OUT / name).read_bytes()).decode()
           for name in m["figures"]}
    sweep_tbl = "\n".join(
        f"| {q} | {100*r['density']:.4f} | {100*r['velocity']:.4f} | {100*r['pressure']:.4f} |"
        for q, r in sorted(sweep.items(), key=lambda kv: int(kv[0])))
    markdown = f"""# CASE-LAX-DG 文档精度号的溯源与去尺审计

## ① 案例介绍
Lax (1954)/Toro (2009) Test 2 激波管：`[0,1]`、隔膜 x0=0.5、γ=1.4、t=0.13；左态
(ρ,u,p)=(0.445,0.698,3.528)，右态 (0.5,0,0.571)；闭式星区 p*={m['waves']['closed_form']['pressure_star']:.6f}、
u*={m['waves']['closed_form']['velocity_star']:.6f}，四道波位置闭式给出。登记构型 p3·N=400·CFL=0.35，
台账 `CASE-LAX-DG` 验收「ref 精确解 | 容差 < 3%」。仓库公开口径为「max 2.769420%」，本单复核该数。

## ② 怎么做的
1. 当前 main 求解器跑登记构型（步数 {m['leg']['steps']}），用文档声明的度量模块
   `benchmarks.riemann_dg_error` 扫 Gauss 阶 q∈{m['config']['quadrature_orders']}（解固定，只动尺）；
2. 独立实现波对齐（按解析断点切单元、子段 Gauss 积分）的去尺评估器，子段 40/160/640 加密，
   断点另用细网格探测做灵敏度检查；与已提交 `lax_a0_quadrature_audit` 工件交叉核对（两个独立实现）；
3. 用 `git archive {m['documented_number_commit'][:9]}` 取出首次发布该数的提交，在子进程复跑同构型，
   把「2.769420% 是谁算的」钉死。

## ③ 计算结果（程序化注入）
固定解只动求积规则（当前 main）：

| q | ρ rel L2 % | u % | p % |
|---:|---:|---:|---|
{sweep_tbl}

- 文档公布值 {100*doc:.4f} % ；当前 main 默认 q=8 读数 {100*f['default_rule_value']:.4f} %（{f['default_rule_worst_field']}）；
  度量族内离公布值最近的读数仍差 {100*f['nearest_current_reading_gap']:.4f} 个百分点 ⇒
  `documented_reproducible_from_main = {f['documented_reproducible_from_main']}`。
- 发布提交 `{m['documented_number_commit'][:9]}` 的求解器在 q=8 下逐位复现 ρ={100*hist['density']:.6f} %、
  u={100*hist['velocity']:.6f} %、p={100*hist['pressure']:.6f} %（与文档三数一致）。
- 去尺收敛读数：ρ={rho:.6f} %、u={100*conv['640']['velocity']:.6f} %、p={100*conv['640']['pressure']:.6f} %；
  子段 40/640 差 <1e-9 个百分点；与已提交正交审计收敛值
  {100*m['cross_check_quadrature_audit']:.6f} % 相差 {100*f['converged_vs_committed_audit']:.2e} 个百分点。
- 密度读数在 q=4..32 间跨度 {f['quadrature_spread_density_pp']:.3f} 个百分点，判决在 PASS/FAIL 间翻转。

## ④ 结果图
- `documented_number_provenance.png`：左—密度 rel L2 对 Gauss 阶（当前 main 与发布提交两条线）+
  3% 台账门 + 文档值 + 波对齐收敛值；右—去尺评估器对子段加密的收敛性。
- `documented_number_verdict.png`：文档值 / 当前默认规则 / 去尺收敛 三个读数对 3% 门的 bar chart。

## ⑤ 注意点
1. 不改台账口径：钉扎腿 p3·N400 的诚实 FAIL（池子 10 点规则 3.1156%）原样保留；本单只查清文档 PASS 的来源。
2. 固定 Gauss 规则不是自由旋钮：精确解在单元内有四个解析断点，q=8 与 q=10 相差 0.21 个百分点，足以跨门。
3. 求解器演化本身也改变了读数（ρ 从 2.7694% 到 2.9058%），旧数+新码的组合在任何规则下都不成立。
4. 已提交文档（README/tutorial/expand audit/gate-power-audit）据此更正，新增测试
   `tests/test_lax_documented_number_audit.py`（24 passed 含回归）钉死「文档号≠当前读数」与「去尺 >3%」。

## ⑥ 误差判定（是否 < 3%）
| 判据 | 值 | 基准 | <3% ? |
|---|---:|---:|---|
| 文档公布（提交 {m['documented_number_commit'][:7]}） | {100*doc:.4f} % | 3.0 % | 是（但当前代码不可复现） |
| 当前 main + 文档默认规则 q=8 | {100*f['default_rule_value']:.4f} % | 3.0 % | 是（依赖尺） |
| 当前 main + 池子规则 q=10 | {100*sweep['10']['density']:.4f} % | 3.0 % | 否 |
| 去尺（波对齐收敛） | {rho:.4f} % | 3.0 % | **否** |

结论：文档号 2.769420% 属 2026-10-23 求解器 + 8 点规则的产物；去掉求积尺后密度相对 L2 为
{rho:.4f} % > 3.0 %，**不满足**台账容差（honest FAIL，未粉饰）。
"""
    token = post("/auth/agent-login", {"username": os.environ["TENSOR_AGENT_USER"],
                                       "password": os.environ["TENSOR_AGENT_PASS"]})
    token = token.get("data", token)["access_token"]
    body = {
        "title": "CASE-LAX-DG：文档 2.769420% 的溯源与去尺审计（ρ 收敛 3.0251% > 3%，honest FAIL）",
        "summary": ("文档公布的 Lax p3·N400 误差 2.769420% 由 2026-10-23 求解器 + 8 点 Gauss 规则产生，"
                    "当前 main 同规则 2.9058%，去尺（波对齐）ρ=3.0251% > 3% 台账门；已更正 README/tutorial/"
                    "expand audit/gate-power audit 并加 12 条测试钉死。"),
        "scenario_key": "case-lax-dg-documented-number-audit",
        "mission_id": 21424,
        "tags": ["TensorDG", "CASE-LAX-DG", "verification", "quadrature", "provenance", "honest-fail"],
        "markdown": markdown,
        "figures": [{"name": name, "caption": cap, "png_base64": fig[name]}
                    for name, cap in (
                        ("documented_number_provenance.png",
                         "密度 rel L2 随 Gauss 阶变化（当前 main vs 发布提交）与去尺收敛读数"),
                        ("documented_number_verdict.png",
                         "文档值 / 默认规则 / 去尺收敛 三个读数对 3% 台账门"))],
    }
    res = post("/tensor/reports", body, token)
    print(json.dumps(res)[:400])


if __name__ == "__main__":
    main()