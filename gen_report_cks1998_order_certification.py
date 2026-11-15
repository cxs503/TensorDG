"""Figure + platform report for the CKS1998 order-certification increment.

Reads ``validation/cks1998_order_certification/certification.json`` (written by
``python -m benchmarks.cks1998_order_certification --assemble ...``) and

* draws the error-vs-h ladders with the fitted power laws, the estimator spread
  and the asymptotic-range / verdict annotations, and
* assembles the platform report body (metrics table is generated from the JSON,
  so no number is ever typed by hand).

Usage::

    PYTHONPATH=. python -m gen_report_cks1998_order_certification --figure \
        docs/figures/cks1998_order_certification.png
    PYTHONPATH=. python -m gen_report_cks1998_order_certification --report-json /tmp/report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "validation" / "cks1998_order_certification" / "certification.json"
VERDICT_COLOR = {"CERTIFIED": "#1b7f3b", "UNVERIFIED_BAND": "#b8860b", "FAIL": "#b22222"}


def load(path: Path = ARTIFACT) -> dict:
    return json.loads(path.read_text())


def figure(payload: dict, out: Path) -> Path:
    degrees = list(payload["degrees"])
    fig, axes = plt.subplots(1, len(degrees), figsize=(4.6 * len(degrees), 4.2))
    if len(degrees) == 1:
        axes = [axes]
    for ax, name in zip(axes, degrees):
        bundle = payload["degrees"][name]
        cert = bundle["certification"]
        levels = cert["levels"]
        h = [lv["h"] for lv in levels][::-1]
        e = [lv["error"] for lv in levels][::-1]
        ax.loglog(h, e, "o-", color="#1f4e79", label="DG 实测 $L_1(\\rho_h-\\rho)$")
        # reference slopes through the finest point: design order and observed order
        h_ref = [h[0], h[-1]]
        for slope, style, colour, label in (
            (cert["design_order"], "--", "#888888", f"设计阶 $p+1={cert['design_order']}$"),
            (cert["order_fine"], ":", "#b22222", f"观测阶 {cert['order_fine']:.3f}"),
        ):
            e_ref = [e[0] * (x / h[0]) ** slope for x in h_ref]
            ax.loglog(h_ref, e_ref, style, color=colour, lw=1.4, label=label)
        ax.invert_xaxis()
        ax.set_xlabel("h")
        ax.set_ylabel("$L_1$ 误差")
        title = (f"{name}: {cert['verdict']}\n"
                 f"$\\delta$={cert['delta_rel']*100:.2f}%, "
                 f"U={cert['uncertainty']:.3f}, "
                 f"$\\sigma$={cert['asymptotic_ratio']:.3f}")
        ax.set_title(title, fontsize=9,
                     color=VERDICT_COLOR.get(cert["verdict"], "#333333"))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("CKS1998 §5.1 光滑二维 Euler：阶数认证（L1 密度误差 vs h）", fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f} %"


def report(payload: dict) -> dict:
    rows = []
    for name, bundle in payload["degrees"].items():
        cert = bundle["certification"]
        sep = bundle.get("time_separation", {})
        rows.append(
            f"| {name} | {cert['design_order']} | {cert['order_fine']:.4f} | "
            f"{cert['uncertainty']:.4f} | {_fmt_pct(cert['delta_rel'])} | "
            f"{_fmt_pct(cert['delta_band_rel'])} | {cert['asymptotic_ratio']:.3f} "
            f"({'是' if cert['asymptotic_ok'] else '否'}) | "
            f"{_fmt_pct(cert['extrapolation_residual'])} | **{cert['verdict']}** | "
            f"{_fmt_pct(sep['eta']) if 'eta' in sep else 'n/a'} |")
    table = ("| P^k | 设计阶 | 最细对观测阶 | 估计子离散 U | δ | δ+U/(k+1) | σ | 渐近区 "
             "| ρ_R | verdict | 时间误差占比 η |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))
    ladder_lines = []
    for name, bundle in payload["degrees"].items():
        for lv in bundle["certification"].get("levels", []):
            ladder_lines.append(
                f"| {name} | {lv['ncells']} | {lv['h']:.5f} | {lv['error']:.6e} |")
    ladder = ("| P^k | ncells | h | L1(rho) |\n|---|---|---|---|\n" + "\n".join(ladder_lines))
    markdown = f"""## ③ 计算结果（程序化注入，零手抄）

### 阶数梯（`certification.json` 直读）

{ladder}

### 认证判定

{table}

CERTIFIED 三条件（全部预登记于 `benchmarks/cks1998_order_certification.py` 模块头）：
δ+U/(k+1) ≤ 3 % · 0.9 ≤ σ ≤ 1.1 · ρ_R ≤ 3 %。时间误差分离用同一网格对在
CFL=0.2 与 CFL=0.05 上重算，η 为最细网格误差的时间占比（阈值 5 %）。
"""
    return {"markdown_tables": markdown, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    payload = load(args.artifact)
    if args.figure:
        print(figure(payload, args.figure))
    if args.report_json:
        body = report(payload)
        args.report_json.write_text(json.dumps(body, indent=2))
        print(body["markdown_tables"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())