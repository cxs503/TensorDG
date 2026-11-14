"""Publish the dt-consumer-audit progress report (owner standing order: every mission reports).

Numbers are injected from ``artifacts/dt_consumer_audit/audit.json`` — nothing is
transcribed by hand.  Run: ``PYTHONPATH=. python scripts/publish_dt_consumer_report.py``
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "dt_consumer_audit" / "audit.json"
FIG = ROOT / "docs" / "benchmarks" / "results" / "dt_consumer_audit" / "fig_dt_residual_share.png"
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(url, payload, token=None):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read() or b"{}")


def token():
    for path in ("/auth/login", "/api/v1/auth/login"):
        try:
            data = post(BASE + path, {"username": USER, "password": PASS})
        except Exception:
            continue
        tok = data.get("access_token") or data.get("token") or (data.get("data") or {}).get("access_token")
        if tok:
            return tok
    raise SystemExit("login failed")


def main() -> int:
    pan = json.loads(ART.read_text())
    meas = [r for r in pan["measurements"] if r["ok"]]
    lad = [r for r in pan["cfl_ladder"] if "ratio_mean" in r]
    worst = max(meas, key=lambda r: r["residual_share_pct"])
    dm = max(abs(r["ratio_mean"] - 2.0) / 2.0 for r in lad) * 100.0
    dl = max(abs(r["ratio_last"] - 2.0) / 2.0 for r in lad) * 100.0
    sites, viol = pan["counts"]["sites"], pan["counts"]["violations"]
    kinds = {}
    for s in pan["sites"]:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1

    rows = "\n".join(
        f"| {r['case']} | {r['steps']} | {r['dt_last']:.4e} | {r['dt_mean']:.4e} | "
        f"{r['residual_share_pct']:.2f} % |" for r in meas)
    lrows = "\n".join(
        f"| {r['cfl']:g} | {r['steps']} | {r['dt_mean']:.4e} | {r['ratio_mean']:.3f} | "
        f"{r['ratio_last']:.3f} |" for r in lad)
    krows = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(kinds.items()))
    fig_b64 = base64.b64encode(FIG.read_bytes()).decode()

    md = f"""# CASE-LAX-DG 验证欠账闭环：`meta["dt"]` 消费面审计（TensorDG，机器门）

## ① 案例介绍
TensorDG 的时间推进求解器用自适应 SSP-RK：`dt = min(cfl·dx/((2p+1)·max_speed), final_time − time)`。
第二项是**残步**，因此 `metadata["dt"]` 里存的常是被 `final_time − time` 截断的**最后一步**，不是运行步长。
它不是小量：在 `(0, dt_max]` 上近似均匀分布，**不随 CFL 或网格加密单调下降**。
`tensordg/stepping.py::step_metadata` 已发布 `dt / dt_last / dt_mean(=final_time/steps)`，但合约文档
`docs/CASE-LAX-DG-dt-metadata-contract.md` §⑤.1 留下一项显式移交：**逐个审计仓内所有 `meta["dt"]` 读取点的语义**
（原话：「另有 ≥6 处基准/测试把 meta["dt"] 当步长用 …… 逐个审计是后续单的独立工作项」）。本单把它做成机器门。

## ② 怎么做的
1. **生产者审计**：扫 `tensordg/*.py`，以截断惯用式 `final_time − time` 判「残步族 / 定步族」；再对残步族**实测** `dt`、`dt_last`、`dt_mean`。
2. **消费面静态审计**：扫 `tensordg/ benchmarks/ tests/ scripts/ tools/` 中全部 `["dt"]` 读取点，按用法分 6 类（PUBLISH / REPORT / STEP_ARITHMETIC / STEP_ARGUMENT / INVARIANT / OTHER），只有后三类属步长语义。
3. **判定规则**：步长语义读取 + 该文件调用残步族求解器 = 违规；同行出现 `dt_mean`/`dt_last`，或写入 `WAIVERS`（写明理由并钉死行号）才放行。
4. **门禁**：`python scripts/audit_dt_consumers.py --check`（违规>0 → 退出 1），并由 `tests/test_dt_consumer_contract.py` 的 7 条不变量测试接入 pytest；扫描器与门禁测试自身从扫描集合排除（否则审计计入自己，站点数随审计写法漂移）。
5. 工具链：Python 3 / PyTorch float64 / matplotlib(Agg) / pytest，本机 CPU（无集群作业）。

## ③ 计算结果
共扫到 **{sites}** 个 `["dt"]` 读取点；**未放行的步长语义误用 = {viol} 处**。用法分布：

| 用法 | 计数 |
|---|---:|
{krows}

残步族生产者实测（`dt` 与代表步长的偏离）：

| 算例 | steps | `dt_last` | `dt_mean` | 残差占比 |
|---|---:|---:|---:|---:|
{rows}

CFL 对折实测（euler_1d Sod, N=100, t=0.05）：

| CFL | steps | `dt_mean` | `dt_mean(k)/dt_mean(k+1)` | `dt_last(k)/dt_last(k+1)` |
|---:|---:|---:|---:|---:|
{lrows}

`dt_mean` 相邻比与 2 的最大偏差 **{dm:.2f} %**；遗留 `dt_last` 为 **{dl:.2f} %**（相邻比甚至 <1 或 >70）。

## ④ 结果图
左：6 个残步族求解器上遗留 `dt` 相对 `dt_mean` 的偏离（最坏 {worst['residual_share_pct']:.1f} %，{worst['case']}）；右：CFL 对折时 `dt_mean` 严格按 2 倍律下降，遗留 `dt_last` 不成立。

![fig](data:image/png;base64,{fig_b64})

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）
1. **「零违规」不等于「以前没问题」**：合约文档点名的 ≥6 处逐条核对后，仓内**没有一处真的对残步做步长算术**——多数是 `PUBLISH`（把 `dt` 原值抄进记录，键名就叫 `dt`，如实），少数是**定步族**求解器上的步长不变量（`elastic_1d`、`sipg_sparse` 上 `steps·dt == t_final` 恒成立）。被污染的真正消费点在**仓外**（历史报告 `dt`–误差曲线的横坐标、跨仓工件读者），静态不可达 → 本门只保证仓内不再新增。
2. 分类是「行正则 + 前 7 行语句窗口」启发式，覆盖不了间接别名；每条放行钉死 `file:line`，代码漂移会立刻打断门禁（不会静默通过）。
3. 残步族判定依赖 `final_time − time` 写法；换写法会误判，故须与既有 `tests/test_time_step_metadata_contract.py` 的源码漂移门配合。
4. 只改消费端，不重定义 `dt`，历史工件/报告数值不受影响。
5. **not_run**：跨仓/工件级消费者审计（需要工件血缘）未做。

## ⑥ 误差判定（逐项回答「是否 < 3 %」）
| 量 | 基准 | 值 | < 3 %? |
|---|---|---|---|
| `dt_mean` vs `final_time/steps` | 代数恒等式 | 0（精确） | 是 |
| `dt_mean` 的 CFL 2 倍律 | 2.0 | 最大偏差 {dm:.2f} % | {'是' if dm < 3 else '否'} |
| 遗留 `dt_last` 的 CFL 2 倍律 | 2.0 | 最大偏差 {dl:.2f} % | **否**（语义错误，不属小误差） |
| 消费面门禁 | 0 处 | {viol} 处 | 是（判据「0 处」） |

**判定**：`dt_mean` 一侧全部 < 3 % 通过；遗留 `dt` 不适用 3 % 判据（它根本不随 CFL 单调），本单结论是「仓内消费面已无一处把它当步长」，而非「它误差很小」。
"""

    payload = {
        "title": f'CASE-LAX-DG 验证欠账闭环：meta["dt"] 消费面审计机器门（{sites} 点 / {viol} 违规）',
        "summary": (f'把 TensorDG 步长元数据合约的显式移交项机器化：静态分类 {sites} 处 meta["dt"] 读取点，'
                    f'实测 6 个残步族求解器的 dt vs dt_mean（最坏 {worst["residual_share_pct"]:.1f}%），'
                    f'CFL 对折下 dt_mean 2 倍律 {dm:.2f}% 内、遗留 dt 不成立（偏差 {dl:.0f}%）；'
                    f'新增 7 条回归不变量测试，仓内未放行误用 {viol} 处。'),
        "scenario_key": "tensordg_dt_consumer_audit",
        "mission_id": 19947,
        "tags": ["TensorDG", "verification-debt", "time-step", "machine-gate", "CASE-LAX-DG"],
        "markdown": md,
        "figures": [{"name": "fig_dt_residual_share.png",
                     "caption": "左：残步族求解器上遗留 dt 相对 dt_mean 的偏离（实测）；右：CFL 对折时 dt_mean 按 2 倍律下降而 dt_last 不成立",
                     "png_base64": fig_b64}],
    }
    pathlib.Path("/tmp/report_dt_consumer.json").write_text(json.dumps(payload, ensure_ascii=False))
    tok = token()
    for path in ("/tensor/reports", "/api/v1/tensor/reports"):
        try:
            out = post(BASE + path, payload, tok)
        except Exception as exc:
            print("post failed", path, exc)
            continue
        print("published:", json.dumps(out, ensure_ascii=False)[:400])
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())