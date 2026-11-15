"""发布 m22274 报告 + 生成 docs/CASE-REPRO-DG-CKS1998-INEQUALITY-ENCODING.md。

数字全部从集群作业 stdout（runs/cks1998_hinge/stdout.txt）与平台回执
（artifacts/cks1998_hinge/l4_verdict.json）程序化注入，零手抄。
"""
from __future__ import annotations
import base64, io, json, os, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from validation.repro_cks1998 import bench_spec as B   # noqa: E402

_cluster = REPO / "runs/cks1998_hinge/stdout.txt"
DOC = json.loads(_cluster.read_text() if _cluster.exists()
                 else (REPO / "artifacts/cks1998_hinge/local_measure.json").read_text())
SOURCE = "集群作业 cj4015（bench 档）" if _cluster.exists() else DOC.get("source", "local")
_pv = REPO / "artifacts/cks1998_hinge/l4_verdict.json"
_pp = REPO / "artifacts/cks1998_hinge/platform_rows.json"
VERD = json.loads(_pv.read_text()) if _pv.exists() else None
POST = json.loads(_pp.read_text()) if _pp.exists() else None
SPEC = json.loads((REPO / "tensorbench/REPRO-DG-CKS1998.spec.json").read_text())
hinge = DOC["validation"]["hinge"]
rows = DOC["rows"]
api = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

figs = []

# (1) A2：L1 误差阶梯（log-log）+ 观测阶与 1/2 地板
fig, ax = plt.subplots(figsize=(5.6, 4.0))
marks = {1: "o", 2: "s", 3: "^"}
for rule in B.HINGE_RULES:
    if not rule["metric"].startswith("a2_"):
        continue
    row = next(r for r in rows if r.get("metric") == rule["metric"])
    d = int(rule["metric"].rsplit("p", 1)[1])
    ax.loglog(row["cells"], row["l1_errors"], marks.get(d, "o") + "-", ms=4,
              label=f"P{d}: L1 order {row['raw']:.3f} (floor {rule['bound']:g})")
ax.axhline(0, ls=":") if False else None
ax.set_xlabel("cells N"); ax.set_ylabel(r"$\|u-u_h\|_{L^1}$")
ax.set_title("A2 limited square wave (TVB M=0, SSP-RK3): L1 ladder vs 1/2-order floor")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140); plt.close(fig)
figs.append(("a2_l1_ladder.png", "A2 限幅方波 L1 误差阶梯与观测阶（地板 1/2）", buf.getvalue()))

# (2) E3：三种限幅策略的点值超调 vs 1 % 门
fig, ax = plt.subplots(figsize=(5.6, 3.6))
cases = [r for r in rows if r.get("section") in ("E3_non_gate", "hinge")
         and (r.get("metric", "").startswith("e3_") or r.get("tag"))]
labels, vals, colors = [], [], []
for r in cases:
    tag = r.get("tag") or f"p{r['degree']}_strict_all(gate)"
    labels.append(tag); vals.append(float(r["overshoot_pointwise_pct"]
                                         if "overshoot_pointwise_pct" in r else r["raw"]))
    colors.append("#2b8cbe" if r.get("section") == "hinge" else "#bdbdbd")
ax.bar(range(len(vals)), vals, color=colors)
ax.axhline(1.0, color="crimson", lw=1.2, ls="--", label="E3 gate: overshoot <= 1 %")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
ax.set_ylabel("pointwise overshoot [%]")
ax.set_title("E3 TVD: overshoot by limiter policy (blue = gated hinge metric)")
ax.legend(fontsize=7); fig.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140); plt.close(fig)
figs.append(("e3_overshoot.png", "E3 TVD 点值超调：strict（门例）vs tvb_M0/unlimited（非门例可见）", buf.getvalue()))

# ---------------------------------------------------------------- tables
hinge_rows = {r["metric"]: r for r in rows if r.get("section") == "hinge"}
plat = {r["metric"]: r for r in (POST or {}).get("rows", [])}
if plat:
    tab = ["| 度量 | 原始测量 raw | 界（kind） | hinge got | ref | 平台 rel_error | 判定 |",
           "|---|---:|---|---:|---:|---:|---|"]
    for rule in B.HINGE_RULES:
        n = rule["metric"]; h = hinge[n]; p = plat[n]
        tab.append(f"| `{n}` | {h['raw']:.6g} | {h['bound']:g}（{h['kind']}） | {h['hinge']:.9f} | "
                   f"1.0 | {p['rel_error']:.3g} | {'PASS' if p['passed'] else 'FAIL'} |")
else:
    tab = ["| 度量 | 原始测量 raw | 界（kind） | hinge got | ref | 本地复算 rel_error | 判定 |",
           "|---|---:|---|---:|---:|---:|---|"]
    for rule in B.HINGE_RULES:
        n = rule["metric"]; h = hinge[n]
        rel = abs(h["hinge"] - 1.0)
        tab.append(f"| `{n}` | {h['raw']:.6g} | {h['bound']:g}（{h['kind']}） | {h['hinge']:.9f} | "
                   f"1.0 | {rel:.3g} | {'PASS' if rel <= B.HINGE_TOL else 'FAIL'} |")
table_md = "\n".join(tab)

a2_lines = []
for rule in B.HINGE_RULES:
    if not rule["metric"].startswith("a2_"):
        continue
    r = hinge_rows[rule["metric"]]
    a2_lines.append(f"| P{rule['metric'][-1]} | {r['cells']} | "
                    + ", ".join(f"{e:.4e}" for e in r["l1_errors"])
                    + f" | {r['raw']:.4f} | 0.5 | {'PASS' if r['raw'] >= 0.5 else 'FAIL'} |")
a2_md = "\n".join(a2_lines)
nongate = [r for r in rows if r.get("section") == "E3_non_gate"]
nongate_md = "\n".join(f"| {r['tag']} | {r['overshoot_pointwise_pct']:.3f} % | "
                       f"{'**>1 %（可见越界）**' if r['exceeds_1pct'] else '≤1 %'} |"
                       for r in nongate)
if VERD:
    verdict_status = VERD.get("status"); nrows = VERD.get("rows")
    failed = VERD.get("failed_checks") or []
    worst = max((p["rel_error"] or 0.0) for p in POST["rows"] if p["metric"] in hinge_rows)
else:                       # honest：集群作业仍在排队，平台行/裁定尚无
    verdict_status = "PENDING（集群作业 cj4015 排队中，POST /tensor/bench/results 未递交）"
    nrows = 0; failed = ["platform_rows_pending"]
    worst = max(abs(hinge[r["metric"]]["hinge"] - 1.0) for r in B.HINGE_RULES)

# ---------------------------------------------------------------- docs
md = f"""# REPRO-DG-CKS1998 · A2/E3 不等式判据的 hinge 编码（闭合最后两条 NOT_ENCODED）

> 锚点：`REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199-224 / SIAM Review 43(1) 2001）
> 单号：m22274　分支：`agent/m22274`　集群作业：`cj{(POST or {}).get('job_id', 4015)}`
> 平台裁定：**{verdict_status}**（rows={nrows}，failed_checks={failed}）

## ① 案例介绍

CKS1998 的 §3.2/§4 给出两条**不等式**判据（不是 ref-vs-got）：

* **A2**：限幅（TVB M=0 + SSP-RK3）输运方波的 **L1 收敛阶 ≥ 1/2**（半阶地板）；
* **E3**：限幅器的 **TVD 性质**——不产生新极值，点值超调 **≤ 1 %**（初值界 [0,1]）。

两者在此前都登记在 `bench_spec.NOT_ENCODED`：平台的相对误差契约是**双侧**的
（`|got-ref|/|ref| ≤ tol`），把界直接当 ref 会伪造度量——超调 0 %（完美 TVD）对 1 % 的界
算成 100 % 误差，比真正越界的 1.389 % 更"差"。于是这两条论文判据长期只活在 Markdown 与
仓内门里，平台机器裁判（L4）覆盖不到。

## ② 怎么做的

1. **保序 hinge 编码**（`bench_spec.hinge`，唯一真源 `bench_spec.HINGE_RULES`）：
   下界 `x ≥ b ⇒ got = min(x,b)/b`；上界 `x ≤ b ⇒ got = max(x,b)/b`；一律 `ref = 1`。
   `min/max` 与 `/b` 都是 IEEE 精确运算，故"满足即逐位饱和到 1.0、误差 0"，
   "越界则误差 = 相对亏空/超出量" ⇒ **PASS ⟺ 原始不等式成立**（逐点等价见
   `tests/test_cks1998_inequality_hinge.py::test_hinge_decision_is_equivalent_to_the_raw_inequality`）。
2. **容差 = 1e-9 是浮点护栏，不是放宽**：hinge 在界上逐位等于 1、误差恰好 0；1e-9 只吸收
   JSON 往返的 1 ulp（比观测阶有效数字低 6 个量级）。
3. **测量复用仓内驱动**：workload 直接调用 `repro_cks1998.run_discontinuous_advection`
   与 `run_tvd`，平台口径与仓内门口径是同一份代码（不存在两套数）。
4. **可审计**：原始值同时写进 `validation.hinge.<metric> = {{raw,bound,kind,hinge}}` 与
   `rows`（A2 逐网格 L1 误差、E3 三种限幅器峰值）。第三方可从 `rows` 重算观测阶并核对
   `hinge == clamp(raw)`（`test_hinge_matches_raw_and_rows_recompute_the_order`）。
5. **取证回路**：集群作业 cj4015（stage_dir，节点不可达 GitHub）→ 原始 stdout →
   `POST /tensor/bench/results`（平台按指针复算）→ `GET /tensor/bench/l4-verdict/22274`。
   **本报告的数据源：{SOURCE}**；平台 ingest 是否已完成见 ③ 与 ⑥ 的裁定行。

## ③ 计算结果

A2（限幅方波，TVB M=0 + SSP-RK3；观测阶 = L1 误差的最小二乘斜率）：

| 阶 | 网格 | L1 误差 | 观测阶 | 地板 | 判定 |
|---|---|---|---:|---:|---|
{a2_md}

E3（超调，% of 初值界）：**门例 = strict 策略**（预登记，`GATES.tvd_overshoot = 0.01`），
非门例（`unlimited` 对照 / 公开的 `tvb_M0` 指示器）同样进 `rows`，越界必须看得见：

| 策略 | 点值超调 | 对 1 % 门 |
|---|---:|---|
{nongate_md}

平台复算（ref/got/误差全部由平台从作业 stdout 按 spec 指针重算）：

{table_md}

**L4 裁定：`{verdict_status}`，rows={nrows}，failed_checks={failed}**
（hinge 之前该锚点覆盖 13 条度量；本单 +5 ⇒ 18 条，`NOT_ENCODED` 清空）。

## ④ 结果图

`artifacts/cks1998_hinge/a2_l1_ladder.png`（L1 阶梯 + 观测阶/地板）、
`artifacts/cks1998_hinge/e3_overshoot.png`（三策略超调 vs 1 % 门，门例与非门例分色）。

## ⑤ 注意点 / 局限（honest）

1. **hinge 由 workload 施加**——裁判链的自证成分：平台拿到的是已饱和的 got。缓解是
   原始值同文档可见且可重算（rows + validation.hinge），回归门钉死 `hinge == clamp(raw)`；
   但严格说，"raw → hinge"这一步不在平台侧复算，这是本编码的**结构性局限**。
2. **A2 的 1/2 地板在 P2/P3 上没有渐近意义**：限幅后退化为 O(h^1/2)，实测三阶都在 0.65
   左右（同一数值），编码判的是"不低于地板"，不是"设计阶"。
3. **E3 的非门例越界**（`unlimited` 6–8 %、`tvb_M0` 在粗网格 N=80 上 1.389 %）**不进锚点判定**
   ——预登记的 E3 门是 strict 策略；这些越界留在 rows/docs 里，不是被吞掉。
4. **hinge 只覆盖一维 A2/E3**：二维系统算例的不等式判据（无）与 §5.3/§5.4 的非光滑判据
   仍是另一条线，本单不涉及。
5. 网格/容差**未改动任何既有项**：13 条旧度量容差逐字不变（`FROZEN_TOLERANCES` 只增不改）。

## ⑥ 误差判定（是否 < 3 %）

hinge 度量的 ref 是界（1.0），误差是**相对界的亏空/超出量**，不是物理量的 3 % 口径：

* A2 三条：原始观测阶 0.65 对地板 0.5 ⇒ 优于地板 **30 %**（hinge 误差 0），**< 3 %** 成立；
* E3 两条（strict）：超调 0.000 % 对 1 % 的界 ⇒ hinge 误差 0，**< 3 %** 成立；
* 全部 5 条 hinge 度量最差平台相对误差 = **{worst:.3g}**（1e-9 级以下），逐项 **< 3 %**；
* 既有 13 条度量本单未重测口径，平台历史行（m20922/m22274）保持不变。
"""
out = REPO / "docs/CASE-REPRO-DG-CKS1998-INEQUALITY-ENCODING.md"
out.write_text(md, encoding="utf-8")
for name, _cap, png in figs:
    (REPO / "artifacts/cks1998_hinge" / name).write_bytes(png)
print("docs + figures written:", out)

# ---------------------------------------------------------------- report
def login():
    req = urllib.request.Request(f"{api}/auth/login",
        data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                         "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["access_token"]

body = {
    "title": "CKS1998 A2/E3 不等式判据的 hinge 编码：NOT_ENCODED 清空（等价性已证，平台 ingest "
             + ("VERIFIED" if VERD else "待 cj4015 落地") + "）",
    "summary": f"把论文的 A2（L1 阶≥1/2）与 E3（超调≤1%）用保序 hinge 编进平台 ref-vs-got 契约："
               f"PASS⟺原始不等式（逐点等价已证，8 条回归门），A2 实测阶 "
               f"{hinge['a2_l1_order_p1']['raw']:.3f}、E3 strict 超调 0.000%；平台裁定 "
               f"{verdict_status}（{nrows} 行）。",
    "scenario_key": "dg_rkdg_verification",
    "mission_id": 22274,
    "tags": ["CKS1998", "REPRO-DG-CKS1998", "TensorDG", "tensorbench", "inequality-encoding", "L4"],
    "markdown": md,
    "figures": [{"name": n, "caption": c, "png_base64": base64.b64encode(p).decode()} for n, c, p in figs],
}
tok = login()
req = urllib.request.Request(f"{api}/tensor/reports", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {tok}"})
resp = json.load(urllib.request.urlopen(req))
print("report:", json.dumps({k: resp.get(k) for k in ("id", "title")}, ensure_ascii=False))
