#!/usr/bin/env python
"""Run the T3 validation matrix and write the evidence bundle.

Usage
-----
    python benchmarks/t3_matrix_run.py --quick              # CI-sized, minutes
    python benchmarks/t3_matrix_run.py --full               # published resolutions
    python benchmarks/t3_matrix_run.py --only cavity-2d-ghia --full
    python benchmarks/t3_matrix_run.py --full --no-figures

Outputs (default ``outputs/t3_matrix/``)
---------------------------------------
``summary.json``   machine-readable matrix: reference, tolerance, measured
                   number, verdict, solver steps, wall time, gaps.
``MATRIX.md``      the same table in markdown (for docs and reports).
``fig_*.png``      comparison figures when matplotlib is available.

Exit code is 0 when the *specification* is valid and no executed case
failed, 1 otherwise -- so a CI job can gate on it.  ``NOT_SUPPORTED`` rows do
not fail the run: they are declared gaps, not silent passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.t3_matrix import (  # noqa: E402
    CASES,
    run_matrix,
    save_summary,
    to_markdown,
    validate_spec,
)

QUICK = {
    "cavity-2d-ghia": {"n": 65, "n_steps": 20000, "u_lid": 0.1},
    "taylor-green-2d": {"n": 32, "n_steps": 600},
    "static-droplet-laplace": {"n": 64, "radii": (8.0, 10.0, 12.0),
                               "n_equilibrate": 1500},
}

FULL = {
    "cavity-2d-ghia": {"n": 129, "n_steps": 120000, "u_lid": 0.1},
    "taylor-green-2d": {"n": 64, "n_steps": 2000},
    "static-droplet-laplace": {"n": 96, "radii": (12.0, 16.0, 20.0),
                               "n_equilibrate": 6000},
}


def figures(summary: dict, out_dir: Path) -> list:
    """Comparison figures: result vs reference, per executed case."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return []

    import numpy as np

    made = []
    rows = {r["key"]: r for r in summary["rows"]}

    cav = rows.get("cavity-2d-ghia")
    if cav and cav.get("detail"):
        from tensorlbm.cavity import GHIA_U, GHIA_V
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for re, color in ((100, "tab:blue"), (400, "tab:red")):
            d = cav["detail"].get(f"re{re}")
            if not d:
                continue
            p = d["profile"]
            axes[0].plot(p["u_over_U"], p["Y"], color=color, lw=2,
                         label=f"LBM N={d['N']} Re={re}")
            ref = GHIA_U[re]
            axes[0].plot(ref["u/U"], ref["y/H"], "o", color=color, ms=4,
                         mfc="none", label=f"Ghia 1982 Re={re}")
            axes[1].plot(p["X"], p["v_over_U"], color=color, lw=2,
                         label=f"LBM Re={re}")
            refv = GHIA_V[re]
            axes[1].plot(refv["x/H"], refv["v/U"], "s", color=color, ms=4,
                         mfc="none", label=f"Ghia 1982 Re={re}")
        axes[0].set_xlabel("u / U"); axes[0].set_ylabel("y / H")
        axes[0].set_title("vertical centreline")
        axes[1].set_xlabel("x / H"); axes[1].set_ylabel("v / U")
        axes[1].set_title("horizontal centreline")
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.suptitle("T3 cavity: LBM vs Ghia et al. (1982)")
        fig.tight_layout()
        f = out_dir / "fig_cavity_ghia.png"
        fig.savefig(f, dpi=140); plt.close(fig)
        made.append(f.name)

    tg = rows.get("taylor-green-2d")
    if tg and tg.get("detail"):
        d = tg["detail"]
        s = d["samples"]
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.semilogy(s["t"], s["amp"], "o", ms=3, label="LBM amplitude")
        t = np.asarray(s["t"], dtype=float)
        a0 = s["amp"][0]
        ax.semilogy(t, a0 * np.exp(-d["decay_rate_analytic"] * t), "-",
                    label=r"analytic $e^{-2\nu k^2 t}$")
        ax.set_xlabel("step"); ax.set_ylabel("modal amplitude |u|")
        ax.set_title(f"T3 Taylor-Green (n={d['n']}), rate err "
                     f"{tg['measured'] * 100:.2f}%")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
        fig.tight_layout()
        f = out_dir / "fig_taylor_green.png"
        fig.savefig(f, dpi=140); plt.close(fig)
        made.append(f.name)

    lap = rows.get("static-droplet-laplace")
    if lap and lap.get("detail"):
        d = lap["detail"]
        radii = np.asarray(d["radii"], dtype=float)
        dp = np.asarray([d["dp"][str(r)] if str(r) in d["dp"] else d["dp"][r]
                         for r in radii], dtype=float)
        fig, ax = plt.subplots(figsize=(5.6, 4.0))
        ax.plot(1.0 / radii, dp, "o", ms=6, label="LBM measured")
        ax.plot(1.0 / radii, d["sigma_implied"] / radii, "-",
                label=r"Laplace law $\Delta p=\sigma/R$, "
                      f"$\\sigma$={d['sigma_implied']:.4g}")
        ax.set_xlabel("1 / R  [lattice]"); ax.set_ylabel(r"$\Delta p$")
        ax.set_title("T3 static droplet (Laplace law)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        f = out_dir / "fig_laplace_droplet.png"
        fig.savefig(f, dpi=140); plt.close(fig)
        made.append(f.name)

    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="CI-sized resolutions")
    ap.add_argument("--full", action="store_true", help="published resolutions")
    ap.add_argument("--only", nargs="*", default=None, help="case keys")
    ap.add_argument("--out", default=str(REPO / "outputs" / "t3_matrix"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = validate_spec()
    if problems:
        print("SPEC INVALID:")
        for p in problems:
            print("  -", p)

    kwargs = FULL if args.full else QUICK
    summary = run_matrix(only=args.only, probe_kwargs=kwargs)
    summary["mode"] = "full" if args.full else "quick"
    save_summary(summary, out_dir / "summary.json")

    md = to_markdown(summary)
    (out_dir / "MATRIX.md").write_text(md + "\n", encoding="utf-8")
    print(md)

    if not args.no_figures:
        figs = figures(summary, out_dir)
        if figs:
            print("\nfigures:", ", ".join(figs))
            summary["figures"] = figs
            save_summary(summary, out_dir / "summary.json")

    failed = [r["key"] for r in summary["rows"]
              if r["status"] in ("FAIL", "ERROR")]
    if failed:
        print("\nFAILED:", ", ".join(failed))
    return 0 if (summary["spec_valid"] and not failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())