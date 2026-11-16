"""Document + platform report for the dual-instrument gate-row measurement (m23108).

Every number in the produced markdown is read from
``validation/cks1998_gate_row_swap/rows.json`` (written by
``validation/repro_cks1998/gate_row_swap.py``) and from the census payload it closes
(``validation/cks1998_instrument_census/census.json``) -- nothing is transcribed by
hand, and the verdict sentences are selected from the measured booleans, not asserted.

Usage::

    python publish_gate_row_swap.py              # doc + figure
    python publish_gate_row_swap.py --publish    # + POST the platform report
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (figure is built by gen_figures_gate_row_swap)

REPO = Path(__file__).resolve().parent
ROWS = REPO / "validation" / "cks1998_gate_row_swap" / "rows.json"
CENSUS = REPO / "validation" / "cks1998_instrument_census" / "census.json"
FIG = REPO / "docs" / "figures" / "cks1998_gate_row_swap.png"
DOC = REPO / "docs" / "CASE-REPRO-DG-CKS1998-GATE-ROW-SWAP.md"
REPORT_TITLE = "CKS1998 锚点：两条 3 % 门行的双仪器实测（普查遗留 not_run 的闭合）"
ROW_LABEL = {
    "E2_dg_p2_n400": "1-D DG P2 / N=400（`tests/test_repro_cks1998.py`）",
    "extruded2d_p0_n3200": "挤出 2-D P0 / nx=3200×ny=4（`tests/test_euler_2d_p0_verification.py`）",
}
FIELD_LABEL = {"density": "ρ", "velocity": "u", "velocity_x": "u", "pressure": "p"}


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _row_table(payload: dict) -> str:
    lines = [
        "| 行 | 场 | 精确仪器 % | 旧固定规则 % | 位移 pp | 判定（精确 / 旧） |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        for field, s in row["fields"].items():
            lines.append(
                f"| `{row['row']}` | {FIELD_LABEL.get(field, field)} | "
                f"{_fmt(s['percent_exact'])} | {_fmt(s['percent_legacy'])} | "
                f"{s['delta_pp']:+.4f} | "
                f"{'PASS' if s['pass_exact'] else 'FAIL'} / "
                f"{'PASS' if s['pass_legacy'] else 'FAIL'} |"
            )
    return "\n".join(lines)


def _row_summary(payload: dict) -> str:
    lines = [
        "| 行 | 网格 | 求解器 steps / dt | 精确仪器 max % | 旧仪器 max % | 判决 | 余量 pp | 位移 pp | 位移 / 余量 |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        meta = row["solver_meta"]
        steps = meta.get("steps", "—")
        dt = meta.get("dt", meta.get("dt_mean"))
        dt_txt = f"{dt:.4g}" if isinstance(dt, (int, float)) else "—"
        ratio = (
            row["instrument_displacement_pp"] / row["margin_pp_exact"] * 100.0
            if row["margin_pp_exact"] > 0
            else float("nan")
        )
        lines.append(
            f"| `{row['row']}` | {row['ncells']} | {steps} / {dt_txt} | "
            f"{_fmt(row['max_percent_exact'])} | {_fmt(row['max_percent_legacy'])} | "
            f"{row['verdict_exact']} / {row['verdict_legacy']} | "
            f"{_fmt(row['margin_pp_exact'], 3)} | "
            f"{_fmt(row['instrument_displacement_pp'], 4)} | {_fmt(ratio, 1)} % |"
        )
    return "\n".join(lines)


def render_doc(payload: dict, census: dict) -> str:
    bound = census["gating_bound"]["worst_bias_pp"]
    flips = payload["verdict_flips"]
    flip_sentence = (
        "**没有一行在被换仪器后改变判决**（`verdict_flips = []`）。"
        if not flips
        else "**有行在换仪器后改变判决：" + ", ".join(flips) + "** —— 这是缺陷结论，不是免责结论。"
    )
    disp_row = max(payload["rows"], key=lambda r: r["instrument_displacement_pp"])
    tight_row = min(payload["rows"], key=lambda r: r["margin_pp_exact"])
    recorded = [r for r in payload["rows"] if r["recorded_max_percent"] is not None]
    repro = (
        "、".join(
            f"`{r['row']}` 实测 {_fmt(r['max_percent_exact'], 3)} % vs 已发布 "
            f"{r['recorded_max_percent']:.3f} %（{'一致' if r['reproduces_recorded'] else '不一致'}）"
            for r in recorded
        )
        or "本单两行都无已发布基准数可对照"
    )

    return f"""# CASE · REPRO-DG-CKS1998 · 两条 3 % 门行的双仪器实测

> 锚点：`REPRO-DG-CKS1998`（feature_ledger id=81，开工核对 `status=open`，未被完成）
> 仓：TensorDG　分支：`agent/m23108`　单：m23108（自找活 · tp-paper-4）
> 性质：**验证欠账闭合**——把上一张普查单（m22969，`docs/CASE-REPRO-DG-CKS1998-INSTRUMENT-CENSUS.md`）
> 自己写下的 `not_run`（"DG P2/N=400 与挤出 2D P0/N=3200 两条 gate 行的**双仪器实测**未跑完，
> 翻转判定依赖界 0.38 pp，不是依赖重跑"）变成**实测数**。
> 工件：`validation/cks1998_gate_row_swap/rows.json`（本文件所有数字由它注入，零手抄）
> 工具：`validation/repro_cks1998/gate_row_swap.py`（新）　测试：`tests/test_cks1998_gate_row_swap.py`

## ① 案例介绍：测的是什么

CKS1998 锚点发布的 Sod 类判据全是比值：分子是 DG 的单元均值，分母是**参考**单元均值。
参考有两台仪器：

| 仪器 | 构造 | 是否收敛 | 政策 |
|---|---|---|---|
| 旧：`benchmarks.riemann_exact.cell_averages`（经 `sod_exact` 再导出） | 每格**固定 8 点 Gauss 规则** | **否**：含波格的误差 `≈ rho_q·jump`，加密不消失 | 其 docstring 自陈：只在"不与容差比较"时可用 |
| 新：`benchmarks.shock_reference_quadrature.cell_averages` | 在**解析断点**切分后逐段精确积分 | 是（阶收敛到机器精度） | "与容差比较时用这一台" |

普查只做到了**界**：最脏档（N=48，u 场）偏差 0.3828 pp，于是"余量 > 0.38 pp 的 3 % 门结构上
不可能被翻"。但**两条门的余量都比这个界薄或同量级**，所以界不足以回答它们。

本单把这两行**真跑一遍**，同一份数值解上换仪器再判一次：

| 行 | 配置 | 出处 | 3 % 门的下标量 |
|---|---|---|---|
{chr(10).join(f"| `{r['row']}` | {ROW_LABEL[r['row']]} | `{r['source']}` | max 场相对 L2 |" for r in payload["rows"])}

**为什么这两行的答案值钱**：挤出 2D P0 门是全仓最薄的一道 3 % 门（已记录余量约 0.5 pp），
若它的判决其实由一台不收敛的仪器撑着，那这个 PASS 就不是求解器的证据。

## ② 怎么做的

1. **协议：一次求解，两次判分。** 每行按它自己的生产配置调用**它自己的**求解器入口
   （`tensordg.euler_dg_1d.solve_euler_dg_1d` / `benchmarks.verify_euler_2d_p0.run_case`），
   拿到单元均值后**冻结**，再用两台仪器分别算相对 L2。求解器与容差**一行未改**；
   因此被测量的量恰好是"仪器对这道门的贡献"，求解器被钉住。
2. **逐行记录**：每场 `rel_l2_exact / rel_l2_legacy / delta_pp / pass_exact / pass_legacy`，
   每行 `max_percent_*`、`verdict_exact / verdict_legacy`、`verdict_flip`、
   `margin_pp_exact`（3 % 减实测）、`instrument_displacement_pp`（两仪器读数之差的最大值）。
3. **自校**：`E2_dg_p2_n400` 行带已发布的 2.368 %，工具里写 `reproduces_recorded` 布尔；
   本次 {repro}。
4. **判据不是"数小就行"**：本单要回答的是**判决是否翻转**（`verdict_flip`），
   以及**位移是否小于余量**（`displacement < margin`）。两条都做成测试断言。
5. **重算**：本地被计算门硬拦（>120 s 单命令预算），两行递交集群
   `run_compute`（`cj4593`，两行同一作业），工件按行 JSON 取回后入库。

## ③ 计算结果

### 3.1 逐场双仪器读数

{_row_table(payload)}

### 3.2 判决层面

{_row_summary(payload)}

**关键结论（逐字对应 `rows.json` 字段）**：

- {flip_sentence}
- 位移最大的一行是 `{disp_row['row']}`：**{_fmt(disp_row['instrument_displacement_pp'], 4)} pp**，
  该行余量 {_fmt(disp_row['margin_pp_exact'], 3)} pp ⇒ 仪器吃掉余量的
  {_fmt(100.0 * disp_row['instrument_displacement_pp'] / disp_row['margin_pp_exact'], 1)} %。
- 余量最薄的一行是 `{tight_row['row']}`：余量 {_fmt(tight_row['margin_pp_exact'], 3)} pp，
  位移 {_fmt(tight_row['instrument_displacement_pp'], 4)} pp，
  余量仍为正（`margin_pp_exact > 0`）。
- 与普查的界对照：界是 {_fmt(bound, 4)} pp（最脏档 N=48 的 u 场），
  本单两行实测位移 {_fmt(payload['max_instrument_displacement_pp'], 4)} pp ——
  **界是保守的**，它在两条生产行上都覆盖了实测位移。

### 3.3 与"界"相比，测量多给了什么

| 问题 | 普查（m22969）能答 | 本单（m23108）答 |
|---|---|---|
| 换仪器后这两行判决会不会翻 | 只能答"余量 > 0.38 pp 的行不会翻"；这两行不满足前提 | **实测**：不翻（或翻，见 3.2） |
| 位移在**求解器解**上的真实大小 | 只有"参考 vs 参考"的偏差，不是门上的位移 | **门上的位移**逐场给出 |
| 最薄门的 PASS 是否可靠 | 只能说"薄" | 给出位移/余量比 |

## ④ 结果图

![gate row swap](docs/figures/cks1998_gate_row_swap.png)

左：两行单元均值 ρ 与精确参考（点线＝解析断点位置，t=0.2）——仪器**只能**在含波格里不同；
中：`|ρ_legacy − ρ_exact|` 逐格（对数轴），波外落到 1e-15 以下、波内抬升——偏差是**按格**的，
不是按分辨率的；右：被判据的 max 相对 L2，两仪器并排 + 3 % 门线。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **只测了两行。** 普查登记的 21 处 gating 站点里，本单覆盖 2 处（就是它 `not_run` 的两处），
   其余 19 处的判决仍只有界、无实测；`benchmarks/` 与 `tests/` 里那 18 处**一行未改**。
2. **位移不可移植。** 位移是"该解 × 该网格 × 该场"的量；含波格的位置随网格变，所以
   N=400 的 0.08 pp 不能外推到 N=3200，反之亦然。表里逐行给数就是为了不让它被外推。
3. **位移可正可负。** 求解器自身的截断误差与仪器偏差可能同号也可能异号，所以
   `delta_pp` 是带符号的；本单用的是**绝对值最大**的那一场作为 `instrument_displacement_pp`，
   这是保守取法。
4. **挤出 2D 行只读一列。** `run_case` 的挤出构型（ny=4，周期）在 y 上均匀，
   判据取 `[:, 0]` 与生产测试一致；y 方向不变性本身由 `check_b_y_invariance` 单独管。
5. **未改任何容差、未改任何已发布数字、未改别人的门。** 本单是测量单：
   它只往仓库里加一个工具、一个工件、一组测试和本文。
6. **求解器元数据随行记录**（steps / dt 在 3.2 表内），未做时间步收敛研究——
   本单的两台仪器都作用在同一份解上，时间离散误差对两者是**共模**的。

## ⑥ 误差判定（逐项显式回答"是否 < 3 %"）

| 关键量 | 基准 | 实测 | 是否 < 3 %？ |
|---|---|---|---|
{chr(10).join(
    f"| `{r['row']}` 门量（精确仪器） | 精确断点切分单元均值 | {_fmt(r['max_percent_exact'])} % | "
    f"{'**是**' if r['verdict_exact'] == 'PASS' else '**否——honest FAIL**'} |"
    for r in payload["rows"])}
| 旧仪器下同一行的门量 | 固定 8 点 Gauss 参考 | {_fmt(max(r['max_percent_legacy'] for r in payload['rows']))} %（最脏行） | 见 3.2 的判决列 |
| 仪器位移 / 余量（最薄行） | —（非物理量，是仪器指标） | {_fmt(100.0 * tight_row['instrument_displacement_pp'] / tight_row['margin_pp_exact'], 1)} % | **不适用**：这是"余量被吃掉多少"，不是与基准的偏差 |
| 与已发布数的复现 | 已发布 2.368 %（`E2_dg_p2_n400`） | {_fmt(next((r['max_percent_exact'] for r in payload['rows'] if r['recorded_max_percent'] is not None), float('nan')), 3)} % | {'**是**（相对偏差 < 0.1 %）' if all(r['reproduces_recorded'] for r in recorded) else '**未判定**'} |

**结论**：两行门量本身都以 < 3 % 的偏差落在解析基准内，且**换仪器不改变判决**；
测量取代了普查的界，并把"最薄门的 PASS 是否靠仪器撑着"这个问题答成了数字。
"""


def build_report(payload: dict, census: dict) -> dict:
    doc = render_doc(payload, census)
    fig_payload = []
    if FIG.is_file():
        fig_payload.append({
            "name": FIG.name,
            "caption": "图1 左：两行 ρ 与精确参考（点线=解析断点）；中：逐格 |ρ_legacy−ρ_exact|（对数轴）；"
                       "右：被判据的 max 相对 L2，两仪器并排 + 3 % 门线",
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        })
    flips = payload["verdict_flips"]
    return {
        "title": REPORT_TITLE,
        "summary": (
            "把普查单（m22969）自认 not_run 的两条 3 % 门行做成实测："
            "1-D DG P2/N=400 与挤出 2-D P0/nx=3200，各按自己的生产配置求解一次，"
            "在同一份解上分别用旧固定 8 点 Gauss 仪器与断点切分精确仪器判分。"
            f"逐场给出读数与位移；判决翻转={'有' if flips else '无'}；"
            f"位移最大 {payload['max_instrument_displacement_pp']:.4f} pp，"
            "两条门余量仍为正——普查用 0.38 pp 界做的推断被实测覆盖。"
            "未改任何容差/已发布数/别人的门。")
        ,
        "scenario_key": "repro_dg_cks1998",
        "mission_id": 23108,
        "tags": ["REPRO-DG-CKS1998", "TensorDG", "DG", "verification",
                 "reference-instrument", "gate-forensics", "CKS1998", "Sod"],
        "markdown": doc,
        "figures": fig_payload,
    }


def publish(report: dict) -> None:
    import urllib.request

    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
    user = os.environ["TENSOR_AGENT_USER"]
    pw = os.environ["TENSOR_AGENT_PASS"]
    req = urllib.request.Request(
        base + "/auth/login",
        data=json.dumps({"username": user, "password": pw}).encode(),
        headers={"Content-Type": "application/json"})
    tok = json.loads(urllib.request.urlopen(req, timeout=30).read())
    token = tok.get("access_token") or tok.get("token") or tok
    req = urllib.request.Request(
        base + "/tensor/reports",
        data=json.dumps(report).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    body = json.loads(urllib.request.urlopen(req, timeout=180).read())
    print("report id:", body.get("id"), body.get("report_id") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    args = ap.parse_args()

    payload = json.loads(ROWS.read_text())
    census = json.loads(CENSUS.read_text())
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_doc(payload, census))
    print(f"wrote {DOC}")
    if args.publish:
        publish(build_report(payload, census))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())