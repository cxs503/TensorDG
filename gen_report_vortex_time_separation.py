"""Figure + platform report for the CKS1998 §5.3 vortex time-error attribution.

Reads ``validation/cks1998_vortex_2d/certification.json`` (written by
``python -m benchmarks.cks1998_vortex_order_certification --assemble``) and

* draws, per degree, the base-CFL error ladder together with the refined-CFL
  (``dt -> dt/beta``) finest pair, and
* draws the deviation ``delta`` at both CFLs against the 3 % tolerance, and
* assembles the platform report body -- every number is read from the JSON, so
  nothing is typed by hand.

Usage::

    PYTHONPATH=. python -m gen_report_vortex_time_separation \
        --figure docs/figures/cks1998_vortex_time_attribution.png \
        --report-json /tmp/vortex_time_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "validation" / "cks1998_vortex_2d" / "certification.json"
ATTR_COLOUR = {"SPATIAL_ISOLATED": "#1b7f3b", "TIME_DOMINATED": "#b8860b",
               "TIME_SENSITIVE": "#7a4fb5", "UNRESOLVED": "#b22222"}
VERDICT_COLOUR = {"CERTIFIED": "#1b7f3b", "UNVERIFIED_BAND": "#b8860b",
                  "FAIL": "#b22222", "INSUFFICIENT_LEVELS": "#666666"}


def load(path: Path = ARTIFACT) -> dict:
    return json.loads(path.read_text())


def figure(payload: dict, out: Path) -> Path:
    degrees = sorted(payload["degrees"])
    fig, axes = plt.subplots(1, len(degrees) + 1,
                             figsize=(4.1 * (len(degrees) + 1), 4.1))
    for ax, name in zip(axes, degrees):
        bundle = payload["degrees"][name]
        cert = bundle["certification"]
        sep = bundle.get("time_separation") or {}
        levels = cert.get("levels", [])
        h = [lv["h"] for lv in levels]
        e = [lv["error"] for lv in levels]
        ax.loglog(h, e, "o-", color="#1f4e79", label="base-CFL ladder")
        if sep.get("status") == "OK":
            pair = sep["pairs"]
            h_base = [pair[f"cfl{sep['cfl_base']:g}"]["h_coarse"],
                      pair[f"cfl{sep['cfl_base']:g}"]["h_fine"]]
            e_base = [pair[f"cfl{sep['cfl_base']:g}"]["error_coarse"],
                      pair[f"cfl{sep['cfl_base']:g}"]["error_fine"]]
            e_red = [pair[f"cfl{sep['cfl_reduced']:g}"]["error_coarse"],
                     pair[f"cfl{sep['cfl_reduced']:g}"]["error_fine"]]
            ax.loglog(h_base, e_base, "o--", color="#1f4e79", lw=1.2,
                      label=f"CFL={sep['cfl_base']:g} finest pair")
            ax.loglog(h_base, e_red, "s-.", color="#b8860b", lw=1.2,
                      label=f"CFL={sep['cfl_reduced']:g} = dt/{sep['beta']:g}")
            design = cert["design_order"]
            ax.loglog([h_base[0], h_base[1]],
                      [e_base[0], e_base[0] * (h_base[1] / h_base[0]) ** design],
                      ":", color="#888888", lw=1.2, label=f"design order {design}")
        ax.invert_xaxis()
        ax.set_xlabel("h")
        ax.set_ylabel(r"relative $L_1$ error of $\rho$")
        ax.set_title(f"{name}: ladder {cert['verdict']} / attribution {sep.get('attribution', 'n/a')}",
                     fontsize=9, color=ATTR_COLOUR.get(sep.get("attribution"),
                                                       VERDICT_COLOUR.get(cert["verdict"], "#333")))
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")

    ax = axes[-1]
    width = 0.38
    xs = list(range(len(degrees)))
    base = [payload["degrees"][n]["time_separation"]["delta_rel_base"] * 100
            for n in degrees]
    reduced = [payload["degrees"][n]["time_separation"]["delta_rel_reduced"] * 100
               for n in degrees]
    ax.bar([x - width / 2 for x in xs], base, width, color="#1f4e79",
           label="base CFL")
    ax.bar([x + width / 2 for x in xs], reduced, width, color="#b8860b",
           label="CFL/beta (dt refined)")
    tol = payload["tolerance_rel"] * 100
    ax.axhline(tol, color="#b22222", ls="--", lw=1.2, label=f"tolerance {tol:g} %")
    for x, b, r, n in zip(xs, base, reduced, degrees):
        sep = payload["degrees"][n]["time_separation"]
        ax.text(x, max(b, r) + 0.15, sep["attribution"].replace("_", "\n"),
                ha="center", va="bottom", fontsize=7,
                color=ATTR_COLOUR.get(sep["attribution"], "#333333"))
    ax.set_xticks(xs)
    ax.set_xticklabels(degrees)
    ax.set_ylabel(r"$|\bar p - (k+1)|/(k+1)$  [%]")
    ax.set_title("order deviation before/after dt refinement", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(max(base), max(reduced), tol) * 1.45)

    fig.suptitle("CKS1998 5.3 isentropic vortex: time-error attribution of the order deviation (SSP-RK3, dt -> dt/beta)",
                 fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f} %"


def tables(payload: dict) -> dict:
    ladder_rows, sep_rows = [], []
    for name in sorted(payload["degrees"]):
        bundle = payload["degrees"][name]
        for lv in bundle["certification"].get("levels", []):
            ladder_rows.append(f"| {name} | {lv['ncells']} | {lv['h']:.5f} | "
                               f"{lv['error']:.6e} |")
        sep = bundle["time_separation"]
        sep_rows.append(
            f"| {name} | {sep['cfl_base']:g} / {sep['cfl_reduced']:g} "
            f"| {sep['order_base']:.4f} | {sep['order_reduced']:.4f} "
            f"| {_pct(sep['delta_rel_base'])} → {_pct(sep['delta_rel_reduced'])} "
            f"| {_pct(sep['eta'])} | {_pct(sep['order_drift'])} "
            f"| {_pct(sep['deviation_share'])} | {sep['spatial_isolated']} "
            f"| **{sep['attribution']}** |")
    ladder = ("| P^k | ncells | h | L1_rel(rho) |\n|---|---|---|---|\n"
              + "\n".join(ladder_rows))
    separation = ("| P^k | CFL 基准/细化 | 观测阶(基准) | 观测阶(细化) | δ | η | 阶漂移 "
                  "| 偏差中被时间步吞掉的比例 | 空间已隔离 | 归因 |\n"
                  "|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(sep_rows))
    return {"ladder": ladder, "separation": separation,
            "sep_rows": sep_rows}


def report(payload: dict) -> dict:
    tbl = tables(payload)
    return {"markdown_tables": tbl, "tables": tbl}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ARTIFACT)
    ap.add_argument("--figure", type=Path)
    ap.add_argument("--report-json", type=Path)
    args = ap.parse_args(argv)
    payload = load(args.artifact)
    if args.figure:
        print(figure(payload, args.figure))
    if args.report_json:
        args.report_json.write_text(json.dumps(report(payload), indent=2))
        print(tables(payload)["separation"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())