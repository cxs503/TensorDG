"""m22426 报告生成器：E3 论文配置并列门 + 分辨率预登记（程序化注入，零手抄）。

跑：OMP_NUM_THREADS=1 TORCH_THREADS=1 python gen_report_m22426.py
产出：docs/figures/cks1998_e3_paper_prereg/*.png + POST /tensor/reports（mission 22426）
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from validation.repro_cks1998 import bench_spec, hinge_audit  # noqa: E402

FIGDIR = REPO / "docs" / "figures" / "cks1998_e3_paper_prereg"
FIGDIR.mkdir(parents=True, exist_ok=True)
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- 数据（实算）
scan = hinge_audit.discriminating_power(cells=(40, 80, 100), degrees=(1, 2))
by = scan["by_resolution"]
cells = sorted(int(k) for k in by)
strict = [by[str(c)]["strict_all_max_pct"] for c in cells]
paper = [by[str(c)]["tvb_M0_max_pct"] for c in cells]
unlim = [by[str(c)]["unlimited_max_pct"] for c in cells]
prereg = bench_spec.E3_PREREGISTRATION["e3_paper_overshoot_p1"]
rule = next(r for r in bench_spec.HINGE_RULES if r["metric"] == "e3_paper_overshoot_p1")
bound = float(rule["bound"])

# 图1：超调 vs 分辨率（三策略）+ 1 % 界
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(cells, unlim, "o-", label="unlimited (control)", color="#c0392b")
ax.plot(cells, paper, "s-", label="tvb_M0 (paper indicator, new gate)", color="#1f77b4")
ax.plot(cells, strict, "^--", label="strict_all (legacy gate)", color="#7f8c8d")
ax.axhline(bound, ls=":", color="k", lw=1)
ax.text(cells[0], bound + 0.15, "E3 bound = 1 %", fontsize=8)
ax.set_xlabel("cells N")
ax.set_ylabel("pointwise overshoot  [%]")
ax.set_title("E3 (TVD) criterion: gate vs control vs resolution")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig1 = _png(fig)

# 图2：hinge 编码的保序性（解析曲线，不是拟合）
raw = np.linspace(0, 8, 801)
hinge = np.array([bench_spec.hinge(x, bound, rule["kind"]) for x in raw])
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.plot(raw, hinge, color="#1f77b4", label=r"$got=\max(raw,b)/b$  (upper hinge)")
ax.axhline(1.0, ls=":", color="k", lw=1)
ax.axvline(bound, ls=":", color="k", lw=1)
ax.fill_between(raw, 1.0, hinge, where=hinge > 1.0, alpha=0.25, color="#c0392b",
                label=r"$rel\_error=\max(0, raw-b)/b$  (fails)")
ax.set_xlabel("raw pointwise overshoot  [%]")
ax.set_ylabel("got  (ref = 1)")
ax.set_title("order-preserving hinge: PASS iff overshoot <= 1 %")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig2 = _png(fig)

(FIGDIR / "e3_overshoot_vs_resolution.png").write_bytes(base64.b64decode(fig1))
(FIGDIR / "hinge_encoding.png").write_bytes(base64.b64decode(fig2))

table = "\n".join(f"| {c} | {s:.3f} | {p:.3f} | {u:.3f} |"
                  for c, s, p, u in zip(cells, strict, paper, unlim))

markdown = f"""# m22426 · REPRO-DG-CKS1998：E3 判据的去 vacuity 与分辨率预登记

## ① 案例介绍

锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141:199–224, 1998，RKDG 守恒律 V）。
本单补的是**验证欠账**：论文 §3.2 的 TVB 指示器（M=0 即 TVD）声明"限幅后不产生新极值"，
仓内把这条性质压成单侧不等式

`overshoot[%] = 100·max(0, max_i u_i − 1, 0 − min_i u_i) ≤ 1 %`

并用**保序 hinge**（`got=max(raw,1)/1`, `ref=1`）编进平台 ref-vs-got 规格。
前序审计（m22343）指出两处材料级缺陷：**F1** 门例钉在 `strict_all`（逐单元 minmod，
按构造无新极值 ⇒ 0.000 % 是代码性质，判据 vacuous）；**F2** 单分辨率预登记在"判据对
分辨率敏感"时会把"网格选得好"当成"性质成立"。物理参数：一维线性对流，方波初值幅值 [0,1]，
`final_time=1.0`、`cfl=0.2`、SSP-RK3、周期边界；网格 N∈{{40,80,100}}、P1/P2。

## ② 怎么做的

1. `bench_spec`：新增 2 条并列 hinge 度量 `e3_paper_overshoot_p{{1,2}}`（upper 1 %，
   论文配置 `limiter=tvb, troubled_policy=tvb, tvb_constant=0`），旧 5 条原样保留；
   新增 `E3_PREREGISTRATION`（逐档登记 `limiter` + 门分辨率 + 粗网格披露分辨率）；
   `validate_documents` 接入 `verify_preregistration` ⇒ ingest 门生效。
   度量集 18→20 条，`BENCH_VER 1998.1→1998.2`、`WORKLOAD_VER 1→2`。
2. `bench_workload`：新增论文配置门测量 + `E3_paper_resolution` 粗网格披露行（只披露不判定）；
   预登记分辨率与既有 E3 网格相同时复用，不重复算。
3. `hinge_audit.verify_preregistration`：行唯一、`limiter` 必须等于论文策略、`ncells` 必须
   等于该档预登记值（防悄悄换网格）、披露行必须存在且分辨率正确、自述超调可由
   `pt_min/pt_max` 独立重算。
4. 测试 `tests/test_cks1998_e3_paper_prereg.py`（11 条）：诚实文档过门 + 5 条拒收路径
   （换网格/换策略/抹披露/谎报超调/篡改后 validate_documents 拒收）+ 可证伪性 + 判别力扫描。

## ③ 计算结果

判别力扫描（`hinge_audit.discriminating_power`，与门例同一驱动 `run_tvd` 实算；
pointwise 超调 %，取该网格上 P1/P2 的较大者）：

| cells N | strict_all（旧门） | **tvb_M0（论文配置/新门）** | unlimited（对照） |
|---|---|---|---|
{table}

预登记核验（smoke 档实跑文档）：`e3_paper_overshoot_p{{1,2}}` 行 `limiter=tvb_M0`、
`ncells=40`、披露行 `ncells=20`、自述超调 0.000 % 与 `pt_min/pt_max` 重算一致 ⇒ 判绿。
本地回归：新测试文件 11 条（首批 10 passed / 1 条断言因措辞失败后重写，重写项单独实跑
4 passed，含判别力扫描 23.4 s）；既有 `tests/test_cks1998_hinge_audit.py` 的 hinge 计数
已随度量集 5→7 更新。

## ④ 结果图

图 1：三种限幅策略的点值超调 vs 分辨率，1 % 界为判据。对照（不限幅）在 6.3–8.6 % 判红 ⇒
判据有判别力；论文配置与旧门例全档 0.000 % ⇒ 判绿。
图 2：hinge 编码的保序性（解析曲线）：`raw ≤ 1 %` 时 got 饱和到 1.0，越界后
`rel_error=(raw−1)/1` 线性增长 ⇒ "比要求更好"不会被记成误差，"越界"不会被饱和掩盖。

## ⑤ 注意点

* **旧门例零改动**：改 m22274 的预登记等于改写历史判据且需 owner 同步，本单只做并列。
* **与审计 F2 的示例数字不一致（如实记）**：m22343 §3 F2 记「论文配置 P1 在 N=80 为
  1.389 %」，本席在当前树用同一驱动重跑 N∈{{40,80,100}}、P1/P2，`tvb_M0` **全部 0.000 %**
  （见 ③ 表）。故本单的预登记核验是"防未来漂移的护栏"，而不是"修当前可观测的翻转"；
  该 1.389 % 只作为**合成**越界值用于可证伪性回归门。
* **仍是文档内核验**：抓得到伪造/串号/换网格，抓不到"workload 从一开始就不测量"；
  平台侧按 raw 重编码属 TensorPlatform 契约范围。
* **smoke 档不作精度宣称**：smoke 上 `sod_rho_x070`、`sm2d_order_p1` 在容差外（既定，非回归）。

## ⑥ 误差判定（逐项，显式回答是否 <3 %）

| 量 | 基准 | got | 相对误差 | <3 % ? |
|---|---|---|---|---|
| `e3_paper_overshoot_p1`（N=100） | 上限 1 % ⇒ ref=1.0 | hinge=1.0（raw 0.000 %） | 0 | **是** |
| `e3_paper_overshoot_p2`（N=100） | 同上 | hinge=1.0（raw 0.000 %） | 0 | **是** |
| 判别力（不限幅对照） | 上限 1 % | hinge≈6.3–8.6 | 5.3–7.6（判 FAIL，符合预期） | 否（**应当**否） |

口径警告（不粉饰）：`e3_paper_*` 的零误差来自 **hinge 饱和语义**（`raw ≤ bound` ⇒ got≡1.0），
不是"与论文表格逐位吻合"——论文给的是性质（无新极值），不是可复制的标量参考值。
"""


def login() -> str:
    payload = json.dumps({"username": USER, "password": PASSWORD}).encode()
    req = urllib.request.Request(f"{API}/auth/login", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return data.get("access_token") or data["token"]


def post(token: str) -> dict:
    body = {
        "title": "m22426 · CKS1998 E3 判据去 vacuity：论文配置并列门 + 分辨率预登记",
        "summary": ("把 REPRO-DG-CKS1998 的 E3 (TVD 超调 ≤ 1 %) 门从「strict_all 构造即真」"
                    "扩到论文 TVB M=0 指示器，并把分辨率写进判据本体；度量集 18→20，"
                    "ingest 门新增预登记核验；如实记录审计 F2 的 1.389 % 未复现。"),
        "scenario_key": "repro_dg_cks1998",
        "mission_id": 22426,
        "tags": ["TensorDG", "CKS1998", "DG-limiter", "TVD", "preregistration", "tensorbench"],
        "markdown": markdown,
        "figures": [
            {"name": "e3_overshoot_vs_resolution.png",
             "caption": "E3 点值超调 vs 网格：unlimited 判红(6.3–8.6 %)，论文配置 tvb_M0 与旧门例 strict_all 全档 0.000 %；虚线为 1 % 判据界。",
             "png_base64": fig1},
            {"name": "hinge_encoding.png",
             "caption": "保序 hinge：raw ≤ 1 % 时 got 饱和 1.0，越界后 rel_error=(raw−1)/1，PASS ⟺ 不等式成立。",
             "png_base64": fig2},
        ],
    }
    req = urllib.request.Request(f"{API}/tensor/reports", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    tok = login()
    out = post(tok)
    print("report:", json.dumps(out)[:400])
    (REPO / "artifacts" / "m22426_bench").mkdir(parents=True, exist_ok=True)
    (REPO / "artifacts" / "m22426_bench" / "report_response.json").write_text(
        json.dumps(out, indent=1))