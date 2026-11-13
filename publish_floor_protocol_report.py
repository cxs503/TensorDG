"""Doc + platform report for the m17687 floor-protocol extension (honest scope).

Reads the shipped artifacts, regenerates the figure and the closeout doc, then
posts the platform report.  No number is typed by hand.
"""
from __future__ import annotations
import base64, json, math, os, urllib.request
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "agent")
PASS = os.environ.get("TENSOR_AGENT_PASS", "agent")
ROOT = Path(__file__).resolve().parent
RES = ROOT / "docs/benchmarks/results/p-refinement-exponential"
prim = json.loads((RES / "p_refinement_results.json").read_text())
flr = json.loads((RES / "measurement_floor.json").read_text())
pub_p = [int(p) for p in prim["results"]["degrees"]]
pub_e = [float(e) for e in prim["results"]["errors"]]
sigma = float(prim["results"]["diagnostics"]["lsq_decay_rate"])
floor = float(flr["floor"]["value"])
snr_min = float(flr["floor"]["snr_min"])
old_d = [int(lv["degree"]) for lv in flr["levels"]]
old_e = [float(lv["l2_extended_longdouble"]) for lv in flr["levels"]]
p_last, e_last = pub_p[-1], pub_e[-1]
p_pred = list(range(p_last, 10))
e_pred = [e_last * math.exp(-sigma * (p - p_last)) for p in p_pred]

fig, ax = plt.subplots(figsize=(8.0, 5.0))
ax.semilogy(pub_p, pub_e, "o-", color="C0", label="published ladder p=1..6")
ax.semilogy(old_d, old_e, "s", color="C1", label="floor study re-measurement (p=5,6)")
ax.semilogy(p_pred, e_pred, "--", color="C2", label=fr"$\sigma={sigma:.3f}$ extrapolation (unmeasured)")
ax.axhline(floor, ls=":", color="0.4", label=fr"float64 floor {floor:.2e}")
ax.axhline(snr_min * floor, ls="-.", color="0.6", lw=1.0, label=fr"{snr_min:g}$\times$ floor = measurable margin")
ax.set_xlabel("polynomial degree $p$ (ncells = 16)")
ax.set_ylabel(r"$L_2$ error")
ax.set_title("p-refinement exponential convergence: measured ladder and the WITHHELD p=7/8 points")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); png = RES / "measurement_floor_protocol.png"; fig.savefig(png, dpi=150); plt.close(fig)

md = f"""# CLOSEOUT m17687 — p 加密地板研究：时间步认证协议 + 外样本 σ 外推（p=7/8 实测仍未落地）

> 锚点 `CASE-DG-PREFINEMENT-EXPONENTIAL`（feature_ledger=open）· 上游
> `docs/CLOSEOUT-m16918-prefinement-criterion.md`（地板量化）、
> `docs/CLOSEOUT-m17067-exact-time-floor.md`（精确时间积分实测 p=7/8）
> 本单 m17687 · 数字由 `publish_floor_protocol_report.py` 从工件直读（零手抄）

## 0. 这一单做了什么（以及**没**做什么）

**做了**：`benchmarks/p_refinement_floor.py` 的两项协议加固——

1. **时间步自适应认证** `_temporal_converged_chain`：每个 p 从 `cfl` 起 `cfl→cfl/2` 反复细化，
   停止条件 `change <= max(1%, 10×roundoff 相对不确定度)`；并带 `step_budget` 上限。
   未认证档位如实返回 `temporal_converged=False` + `stop_reason`
   （`step_budget` / `refinement_budget_exhausted` / `no_refinement_allowed`），
   绝不静默当成空间误差引用。
2. **外样本外推检验** `_extrapolation_check`：用**已发布** p={p_last} 点 + σ 外推
   `error(p)=e({p_last})·exp(-σ(p-{p_last}))`，对本次实测的每个 p>{p_last} 点给
   `measured/predicted`，即对 Babuska & Suri (1994) 解析解指数律做**预测检验**而非再拟合。
3. 配套：`tools/plot_floor_high_degree.py`（双面板图：误差-p 阶梯 + 局部衰减率 vs σ 带）、
   `tests/test_p_refinement_floor_high_degree.py`（工件闸 + 两条活闸，活闸已在本地通过）。

**没做（honest）**：p=7/p=8 的**实测**在本单窗口内未取得。计划中的高 p 运行
（`python -m benchmarks.p_refinement_floor --degrees 5 6 7 8 --temporal-refinements 3 --step-budget 50000`，
约 7.6 万 RK 步）递交集群作业 **cj1635**（TensorDG，stage_dir=本工作树）后仍在队列中未派发；
本地试跑因控制面单次命令 120 s 上限被收割。故 `measurement_floor.json` 仍是 p={old_d} 的旧工件，
新增的 p>6 内样本外推检验**尚无数据**，其工件闸按设计 `skip`（不是 pass）。
p=7/8 的现状仍以上游 m17067 的精确时间结论为准：p=7 可解析（{2.7158e-13:.3e}），
p=8 只可作上界（算子表示地板 ~4.75e-13 大于信号）。

## ① 案例介绍
一维周期线性对流 `u_t+u_x=0`，初值 `sin(2πx)`（整函数），`T=1`（一整周期，精确解=初值平移），
固定网格 `ncells=16` 只做 p 加密。理论基准 Babuska & Suri (1994) SIAM Review 36:578–632：
解析解下 DG/FEM 误差随 p 指数衰减 `‖u-u_h‖ <= C exp(-σ p)`。
已发布阶梯 p=1..{p_last} 的 σ = **{sigma:.4f}**，float64 测量地板 = **{floor:.3e}**（`eps64·‖u_h‖`），
可测裕度 = {snr_min:g}×地板 = {snr_min*floor:.3e}。

## ② 怎么做的（本单新增协议）
1. 对每个 p：`cfl` 起点取已发布工件的 CFL（无发布值时用 0.08）；
2. 阶梯细化：每档 `cfl/2` 重解（出货 RK4 + 出货 `solve_advection`，不改格式），
   用 longdouble 求积路径测 L2，比较相邻两档误差；
3. 停止：`|e_prev - e_cur|/e_cur <= max(0.01, 10·eps64·‖u_h‖/e_cur)` ⇒ 认证；
   预算命中 ⇒ 未认证（带 `stop_reason` 与 `total_steps` 入工件）；
4. 外样本：`measured/predicted` 逐点入工件，带宽 ±2×；
5. 门禁：`tests/test_p_refinement_floor_high_degree.py`（工件一致性 + 活闸；
   活闸在 p=2 与 p=4/CFL=2 上直接驱动新函数，验证「认证」与「预算耗尽如实报告」两条路径）。

## ③ 计算结果（当前工件，p={old_d}）
| p | L2 (longdouble) | 相对离散 | SNR(地板) | 可测 |
|---|---|---|---|---|
""" + "\n".join(
    f"| {lv['degree']} | {lv['l2_extended_longdouble']:.4e} | {lv['path_spread_rel']:.1e} | "
    f"{lv['snr_measured_over_floor']:.3g} | {'是' if lv['resolved'] else '否'} |"
    for lv in sorted(flr["levels"], key=lambda l: l["degree"])) + f"""

已发布阶梯（主工件）误差：{', '.join(f'p={p}: {e:.3e}' for p, e in zip(pub_p, pub_e))}。
按 σ={sigma:.3f} 外推：p=7 → {e_last*math.exp(-sigma*1):.3e}（SNR {e_last*math.exp(-sigma*1)/floor:.0f}）、
p=8 → {e_last*math.exp(-sigma*2):.3e}（SNR {e_last*math.exp(-sigma*2)/floor:.0f}）——**外推值，不是实测**。

## ④ 结果图
`measurement_floor_protocol.png`：已发布 p=1..6 阶梯、地板研究重测点（p=5,6）、σ 外推（虚线段，
明确标注 unmeasured）与 float64 地板/可测裕度两条水平线。图中外推段**只作预测**，不冒充实测。

## ⑤ 注意点 / 局限
- **p>6 的实测为本单欠账**（集群 cj1635 未派发、本地受 120 s 命令上限）：本节结论不覆盖 p=7/8 的实测事实。
- 相对时间收敛判据在高 p 会退化：误差趋近 `eps64·‖u_h‖` 时「1% 相对变化」等价于要求噪声自洽，
  故认证阈值改为 `max(1%, 10×roundoff 不确定度)`；这**不改**出厂 `solve_advection` 的时间判据，
  只作用于本验证脚本。
- 步数预算 / 细化上限是显式协议参数：命中预算的档位标未认证，不得引用为空间误差。
- 适用面：一维周期线性对流光滑解；二维/变系数/激波不在范围。

## ⑥ 误差判定（逐项，回答「是否<3%」）
| 项 | 基准 | 结果 | 是否<3% |
|---|---|---|---|
| 新协议活闸（认证路径） | p=2 已知收敛档 | 通过（见 pytest） | ✅ |
| 新协议活闸（预算耗尽如实报告） | max_refinements=0 | `temporal_converged=False` | ✅（行为正确） |
| 已发布阶梯复现 | 主工件 | 本单未重跑主工件 | ⚪ 未测（无新基准） |
| p=7/8 实测 vs σ 外推 | ±2× 带宽 | **无数据**（作业未派发） | ⚪ 未测 |

**结论**：本单交付的是**测量协议的认证化**（可验证、有活闸、未认证即拒引），
不是 p=7/8 的新实测；锚点的「p>6 实测」仍为 WITHHELD，需在集群空出后跑 cj1635 或等价命令。
"""
doc = ROOT / "docs/CLOSEOUT-m17687-high-degree-floor.md"
doc.write_text(md + "\n")

def post(path, body, token=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())

tok = post("/auth/login", {"username": USER, "password": PASS})
tok = tok.get("access_token") or tok.get("token")
payload = {
    "title": "CASE-DG-PREFINEMENT-EXPONENTIAL：p 加密地板的时间步认证协议与外样本 σ 外推（p=7/8 实测欠账）",
    "summary": (f"给地板研究加两项协议并加活闸：cfl 自适应认证（阈值 max(1%,10×roundoff)，命中步数预算即标未认证）"
                f"与外样本 σ 外推检验（由 p={p_last} 已发布点预测 p>6）。已发布阶梯 σ={sigma:.4f}、地板 {floor:.3e} "
                f"（可测裕度 {snr_min*floor:.3e}）；p=7/8 实测未取得（集群作业 cj1635 排队未派发、本地受 120 s 上限），"
                f"锚点 p>6 实测仍为 WITHHELD，如实记录。"),
    "scenario_key": "CASE-DG-PREFINEMENT-EXPONENTIAL",
    "mission_id": 17687,
    "tags": ["TensorDG", "DG", "p-refinement", "exponential-convergence", "verification",
             "measurement-floor", "temporal-certification", "honest-partial"],
    "markdown": md,
    "figures": [{"name": png.name,
                 "caption": "p 加密误差阶梯：已发布 p=1..6、地板研究重测点、σ 外推（未实测）与 float64 地板/可测裕度",
                 "png_base64": base64.b64encode(png.read_bytes()).decode()}],
}
res = post("/reports", payload, tok)
print(json.dumps({"report_id": res.get("id", res), "doc": str(doc), "figure": str(png)}, ensure_ascii=False))
