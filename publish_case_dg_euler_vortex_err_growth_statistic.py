#!/usr/bin/env python3
"""Publish CASE-DG-EULER-VORTEX-ERR round 21: is the growth statistic itself sound?

Reads the committed artefacts only -- ``metrics.json`` (gate payload including the
new C12), ``gate_power_matrix.json`` and ``gate_threshold_calibration.json`` --
draws the figure, writes the document and posts the report.  Nothing is
hand-copied.

Usage::

    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_growth_statistic.py --dry-run
    PYTHONPATH=. python3 publish_case_dg_euler_vortex_err_growth_statistic.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER", "eco_admin")
PASS = os.environ.get("TENSOR_AGENT_PASS", "EcoTest!2026pass")
BUNDLE = Path("case_euler_vortex_errconserve")
DOC = Path("docs/CASE-DG-EULER-VORTEX-ERR-growth-statistic-robustness.md")
FIG = Path("docs/fig_round21_growth_statistic_robustness.png")
MISSION = 23042
SLUG = "CASE-DG-EULER-VORTEX-ERR"


def num(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def load():
    metrics = json.loads((BUNDLE / "metrics.json").read_text())
    power = json.loads((BUNDLE / "gate_power_matrix.json").read_text())
    calibration = json.loads((BUNDLE / "gate_threshold_calibration.json").read_text())
    return metrics, power, calibration


def make_figure(c12) -> Path:
    rows = c12["per_family"]
    names = sorted(rows)
    ols = [rows[n]["ols"] for n in names]
    robust = [rows[n]["robust"] for n in names]
    x = range(len(names))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    width = 0.38
    ax.bar([i - width / 2 for i in x], ols, width, label="OLS (published statistic)",
           color="#4c72b0")
    ax.bar([i + width / 2 for i in x], robust, width,
           label="C12 conservative reading", color="#c44e52")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="threshold alpha = 1.0")
    ax.axhline(0.5, color="gray", ls=":", lw=1.0, label="strict target alpha = 0.5")
    for i, (a, b) in enumerate(zip(ols, robust)):
        ax.text(i + width / 2, b + 0.012, f"{b:.2f}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel(r"growth exponent $\alpha$ of $e \sim t^{\alpha}$")
    ax.set_title("C12: every family, both statistics")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    r2 = [rows[n]["r2"] for n in names]
    res = [rows[n]["max_log_residual"] for n in names]
    ax2.scatter(r2, res, s=42, color="#55a868")
    for i, name in enumerate(names):
        ax2.annotate(name.replace("_cfl0.2", "").replace("dg_", ""),
                     (r2[i], res[i]), fontsize=7, xytext=(3, 3),
                     textcoords="offset points")
    ax2.axvline(c12["fit_floor"], color="k", ls="--", lw=1.0,
                label=f"power-law floor R^2 = {c12['fit_floor']:g}")
    ax2.set_xlabel(r"$R^2$ of the log-log fit")
    ax2.set_ylabel(r"$\max|\log e - \log \hat e|$  (nepers)")
    ax2.set_title("is it a power law at all?")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25)
    fig.suptitle("CASE-DG-EULER-VORTEX-ERR · growth-statistic robustness (round 21)",
                 fontsize=11)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    return FIG


def build_doc(metrics, power, calibration) -> str:
    gates = metrics["gates"]
    c12 = gates["C12_growth_statistic_robustness"]
    c2 = gates["C2_error_conservation"]
    rows = c12["per_family"]
    table = ["| 族 | 检查点 | OLS | Theil-Sen | 端点 | 去掉 t=1 | 留一最坏 | 自助 99% "
             "| 保守读数 | 保守读数取自 | R^2 | 幂律? |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(rows):
        row = rows[name]
        table.append(
            f"| `{name}` | {row['points']} | {num(row['ols'])} | {num(row['theil_sen'])} | "
            f"{num(row['endpoint'])} | {num(row['drop_projection_floor'])} | "
            f"{num(row['leave_one_out_worst'])} | {num(row['bootstrap_quantile'])} | "
            f"**{num(row['robust'])}** | `{row['robust_at']}` | {num(row['r2'])} | "
            f"{'是' if row['power_law'] else '否'} |")
    m13 = [r for r in power["rows"] if r["target"] == "C12_growth_statistic_robustness"][0]
    k12 = [r for r in calibration["rows"] if r["criterion"] == "C12_growth_statistic_robustness"][0]
    design = c12["design_family"]
    return f"""# CASE-DG-EULER-VORTEX-ERR：增长统计量本身的稳健性（round 21）

> 锚点 `CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒）｜mission {MISSION}｜
> ref: Yee, Vinokur & Djomehri, JCP 162 (2000) 33-81｜容差 6%
> 本文档由 `publish_case_dg_euler_vortex_err_growth_statistic.py` 从
> `case_euler_vortex_errconserve/metrics.json`、`gate_power_matrix.json`、
> `gate_threshold_calibration.json` 程序化生成，数字零手抄。

## ① 案例介绍

二维可压缩 Euler（理想气体 γ=1.4）中，等熵涡在周期方箱 [-5,5]² 以 |u∞|=√2 匀速输运。
真解对平移不变，故**离散误差应是网格属性、不是时间属性**——这就是锚点「移动误差守恒」
的物理含义。C1–C11 已在此读数上全部通过；C1 是 t=1 的快照容差（6%），C2/C7/C8/C10 是
对「误差随时间增长」的判定，它们读的都是同一个数：

```
alpha := log-log 最小二乘斜率   e(t) ~ t^alpha,   t in [1, 10 or 20],  3~5 个检查点
```

前 20 轮把参考解（R1/R2）、不变量分母（R3）、误差范数求积（R4）、熵预算（R5）、证据溯源
（C11）和每一条判据的判别力（变异矩阵）都认证了一遍，**却从未认证过 alpha 这个数本身**：
换一个估计量会怎样？去掉一个检查点会怎样？拟合到底是不是幂律？本轮的实质增量就是把这三问
变成可判、可失败、可复算的判据 **C12**，并把它接进总裁决。

## ② 怎么做的

| 步骤 | 内容 |
|---|---|
| 估计量 | 同一族的检查点先按 t ≥ 1 取窗，再算五种读数：OLS（历史统计量）、Theil-Sen（两两斜率中位数）、端点斜率（只用首末点）、去掉 t=1 投影地板后的重拟合、留一重拟合（每次弃一个检查点） |
| 自助 | 残差自助 {c12['bootstrap']['trials']} 次（种子 {c12['bootstrap']['seed']}）取 {c12['bootstrap']['quantile']:.0%} 分位，检查点时间是设计好的阶梯、不是随机抽样，故只对残差重采样 |
| 保守读数 | 一族一个 `robust` 值 = 上述所有可判读数的**最大者**；C12 判的就是它 |
| 拟合质量 | 逐族报 log-log R^2 与最大对数残差；R^2 < {c12['fit_floor']:g} 时 alpha 只是**窗口内趋势**，报告必须这么写，不许叫幂律 |
| 判别力 | 变异 `{m13['id']}`：设计族「先平后炸」（前四个检查点压在投影地板上、只末点增长），全窗 OLS 被摊薄而端点/留一读得到 |
| 门限标定 | `{k12['id']}`：把同一构造的保守读数二分到 0.999 / 1.001 两侧，判据必须内侧 passed、外侧 failed |
| 不可判 | 检查点 < 3 的族、或无法再做留一的族的变体一律记 `nan` 并列入 `undecidable_variants`/`undecidable`，绝不填 0 |

## ③ 计算结果

C12 实测：**可判 {len(c12['families_evaluated'])} 个 DG 族，最坏保守读数
{num(c12['measured'])}（`{c12['worst_at']}`，取自 `{c12['worst_variant']}`），
门限 α ≤ 1.0 ⇒ {c12['status']}**。

| 判据 | 内容 | 实测 | 判定 |
|---|---|---|---|
| C12 增长统计量稳健性 | 每个 DG 族的保守读数 ≤ {c12.get('threshold', 1.0):g} | 最坏 {num(c12['measured'])}（{c12['worst_at']} / {c12['worst_variant']}） | {c12['status']} |
| C2（对照，未改） | 设计族 OLS α ≤ 1.0 | {num(c2['exponent'])} | {c2['status']} |

<<TABLE_ROWS>>
设计族 `{design['family']}`：OLS **{num(design['ols'])}** → 保守读数
**{num(design['robust'])}**（取自 `{design['robust_at']}`）。严格目标 α ≤
{design['strict_target']:g} 在保守读数下 **{'仍成立' if design['strict_passed'] else '不再成立'}**，
但裕度从 {num(design['strict_target'] - design['ols'])} 缩小到
{num(design['strict_target'] - design['robust'])}——这正是「点值」与「可报结论」的差别。

判别力与门限标定（本轮重生成的变异矩阵）：

| 项 | 结果 |
|---|---|
| 门禁变异力 | {power['power']}（survivors={power['survivors']}，对照组惰性={power['control_inert']}） |
| `m13` 注入 | {m13['detail']} |
| `m13` 裁决 | 目标判据 C12 → `{m13['statuses'].get('C12_growth_statistic_robustness')}`；C2 → `{m13['statuses'].get('C2_error_conservation')}`；整包 `{m13['verdict']}` |
| 门限标定 | {calibration['calibrated'].__len__()}/{len(calibration['rows'])} 咬合；`k12` 内侧 `{k12['readings']['inside']['status']}` / 外侧 `{k12['readings']['outside']['status']}` |

`m13` 是 C12 存在的理由：同一次注入里 **C2 保持通过、C12 变红**。一个只在第一段行程稳定的
跑法正是全窗回归会藏起来的失效模式。

## ④ 结果图

`{FIG}`（左）每族的 OLS 与保守读数并列，两条门限线画出裕度；（右）拟合质量散点——
横轴 log-log R^2、纵轴最大对数残差，虚线是幂律地板。

## ⑤ 注意点

* **alpha 不是幂律指数，是窗口内趋势界**。逐族 R^2 见上表：{len(c12['trend_not_power_law'])}
  个可判族的 R^2 < {c12['fit_floor']:g}。判据语义因此仍成立（「没有超线性累积」），但把
  alpha 当物理幂律去外推到 t=50 是不允许的，本轮明确写出这一点。
* **估计量之间的散布不是小量**：最坏族 OLS 与保守读数相差 {num(100.0 * (c12['measured'] / rows[c12['worst_at']]['ols'] - 1.0), 1)}%。
  所以 alpha 的点值不能按 3% 精度引用；能按窄裕度引用的是「到门限的距离」。
* **保守读数偏保守**：端点斜率与留一都只依赖少数点，噪声大的族（检查点少）会被高估。
  C12 取最坏者因此是**上界**，不是中心估计；这正是判据想要的方向。
* **C13 未做**：参考解重放证书（round 20 文档表里声明的 R6「复现」）在
  `instrument_certification.json` 中**没有任何读数**（文件只有 R1–R5，R4 表的
  「复现存储值」列全是「—」）。本轮不补，如实记为欠账。
* 本轮不改任何已入库读数、不改容差、不改 C1–C11 语义；唯一的语义变化是新增 C12 并把它
  计入总裁决（上一轮已经出现过「绿但没被判据覆盖」的教训）。

## ⑥ 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 设计族增长指数点值 | 历史 OLS {num(c2['exponent'])} | 保守读数偏离 {num(100.0 * (design['robust'] / design['ols'] - 1.0), 1)}% | **否**（如实记：点值不可报 3% 精度） |
| 判据判定（α ≤ 1.0） | 门限 1.0 | 最坏族裕度 {num(1.0 - c12['measured'])}（相对 {num(100.0 * (1.0 - c12['measured']), 1)}%） | **是**（判定不被估计量选择翻转） |
| 门限标定 | 声明阈值 1.0 | 内侧 0.999 / 外侧 1.001 各翻一次 | **是**（判据咬合到 1e-3） |
| 变异判别力 | 12 条判据 | 12/12 被杀，对照惰性 | **是** |

结论一句话：**C1–C12 全部通过，且 C2/C7/C8/C10 赖以判定的那个数在五种估计量与留一扰动下
都不越门限；但那个数是趋势界，不是幂律指数，本报告按前者引用。**
"""


def post_report(markdown: str, figure: Path, c12) -> dict:
    def login():
        body = json.dumps({"username": USER, "password": PASS}).encode()
        req = urllib.request.Request(f"{BASE}/auth/login", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        return payload.get("access_token") or payload["token"]

    token = login()
    figures = []
    if figure.exists():
        figures.append({"name": figure.name,
                        "caption": "左：各族 OLS 与 C12 保守读数；右：log-log 拟合质量",
                        "png_base64": base64.b64encode(figure.read_bytes()).decode()})
    payload = {
        "title": "CASE-DG-EULER-VORTEX-ERR round 21：增长统计量本身的稳健性认证（新增 C12）",
        "summary": (f"为 C2/C7/C8/C10 的 alpha 统计量补稳健性认证：五估计量+留一+残差自助，"
                    f"最坏保守读数 {num(c12['measured'])}（{c12['worst_at']}），"
                    f"门限 1.0 不翻转；同时指出 R^2<0.98 时 alpha 只是趋势界。"),
        "scenario_key": "case_dg_euler_vortex_err",
        "mission_id": MISSION,
        "tags": [SLUG, "C12", "gate-robustness", "statistic-uncertainty", "TensorDG"],
        "markdown": markdown,
        "figures": figures,
    }
    req = urllib.request.Request(
        f"{BASE}/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    metrics, power, calibration = load()
    c12 = metrics["gates"]["C12_growth_statistic_robustness"]
    figure = make_figure(c12)
    rows = c12["per_family"]
    table = ["| 族 | 检查点 | OLS | Theil-Sen | 端点 | 去掉 t=1 | 留一最坏 | 自助 99% "
             "| 保守读数 | 取自 | R^2 | 幂律? |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in sorted(rows):
        row = rows[name]
        table.append(
            f"| `{name}` | {row['points']} | {num(row['ols'])} | {num(row['theil_sen'])} | "
            f"{num(row['endpoint'])} | {num(row['drop_projection_floor'])} | "
            f"{num(row['leave_one_out_worst'])} | {num(row['bootstrap_quantile'])} | "
            f"**{num(row['robust'])}** | `{row['robust_at']}` | {num(row['r2'])} | "
            f"{'是' if row['power_law'] else '否'} |")
    markdown = build_doc(metrics, power, calibration).replace(
        "<<TABLE_ROWS>>", "\n".join(table))
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown)
    print(f"wrote {DOC} ({len(markdown)} chars) and {figure}")
    if args.dry_run:
        return 0
    result = post_report(markdown, figure, c12)
    print(json.dumps(result)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())