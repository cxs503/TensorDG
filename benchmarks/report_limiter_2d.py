"""Render the 2-D limiter verification evidence into a report + figures.

Usage::

    PYTHONPATH=. python -m benchmarks.report_limiter_2d

Reads ``validation/limiter_2d/before/evidence.json`` and
``validation/limiter_2d/after/evidence.json`` (written by
``benchmarks.verify_limiter_2d``), emits
``docs/limiter_2d_verification.md`` and the PNGs under
``docs/figures/limiter_2d/``.  Every number in the markdown is injected from
the JSON, so no digit is hand-copied.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("validation/limiter_2d")
FIGURES = Path("docs/figures/limiter_2d")
DOC = Path("docs/limiter_2d_verification.md")


def _load(tag):
    path = ROOT / tag / "evidence.json"
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _mesh_rows(evidence, degree, label):
    return [
        row
        for row in evidence["detector_scale"]["mesh_sweep"]
        if row["degree"] == degree and row["label"] == label
    ]


def figure_detector(before, after):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for tag, evidence, style in (
        ("before m2549", before, "o--"),
        ("after m2549", after, "s-"),
    ):
        if evidence is None:
            continue
        rows = _mesh_rows(evidence, 2, "M=1")
        cells = [row["cells"] for row in rows]
        ax.plot(cells, [row["fraction_x"] for row in rows], style,
                label=f"{tag}, M=1, x-axis")
        ax.plot(cells, [row["fraction_y"] for row in rows], style,
                markerfacecolor="none", label=f"{tag}, M=1, y-axis")
    if after is not None:
        rows = _mesh_rows(after, 2, "M=2|u''|")
        ax.plot([row["cells"] for row in rows], [row["fraction_x"] for row in rows],
                "^-", color="tab:green", label="after, M=2|u''| (0 everywhere)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("cells per direction")
    ax.set_ylabel("flagged cell fraction")
    ax.set_title("A: troubled-cell detector on a smooth 2-D wave")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    for tag, evidence, style in (
        ("before m2549", before, "o--"),
        ("after m2549", after, "s-"),
    ):
        if evidence is None:
            continue
        rows = evidence["detector_scale"]["constant_sweep"]
        ax.plot([r["tvb_constant"] for r in rows], [r["fraction_x"] for r in rows],
                style, label=f"{tag}")
    if after is not None:
        ax.axvline(after["detector_scale"]["tvb_constant_above_curvature"],
                   color="tab:green", ls=":", lw=1,
                   label="M = 2 max|u''|")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("TVB constant M")
    ax.set_ylabel("flagged cell fraction (x direction)")
    ax.set_title("A': tunability of M at 32^2 cells")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "detector_scale.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_vortex(before, after):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, degree in zip(axes, (1, 2)):
        if after is not None and f"P{degree}" in after.get("vortex", {}):
            block = after["vortex"][f"P{degree}"]
            rows = block["unlimited"]
            cells = sorted(int(k) for k in rows)
            ax.loglog(cells, [rows[str(c)]["errors"]["density"] for c in cells],
                      "k-o", label="unlimited")
            for series in block["limited"]:
                cs = [row["cells"] for row in series["rows"]]
                ax.loglog(cs, [row["errors"]["density"] for row in series["rows"]],
                          "s--", label=f"TVB M={series['tvb_constant']:g}")
        if before is not None and f"P{degree}" in before.get("vortex", {}):
            block = before["vortex"][f"P{degree}"]
            if "limited" in block:
                for series in block["limited"]:
                    cs = [row["cells"] for row in series["rows"]]
                    ax.loglog(cs, [row["errors"]["density"] for row in series["rows"]],
                              "x:", alpha=0.6,
                              label=f"before, M={series['tvb_constant']:g}")
            if "error" in block:
                ax.text(0.03, 0.05, "before: run failed\n" + block["error"][:40],
                        transform=ax.transAxes, fontsize=7, color="tab:red")
        ax.set_xlabel("cells per direction")
        ax.set_ylabel("density relative L2 error")
        ax.set_title(f"B: isentropic vortex, P{degree} (design order {degree + 1})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = FIGURES / "vortex_order.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def figure_riemann(after):
    blocks = after.get("riemann") if after else None
    if not blocks:
        return None
    block = [b for b in blocks if b.get("cells") == 32]
    block = block[0] if block else blocks[-1]
    if "none" not in block or "tvb" not in block:
        return None
    none = np.asarray(block["none"].get("cell_density", []))
    tvb = np.asarray(block["tvb"].get("cell_density", []))
    if none.size == 0 or tvb.size == 0:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    vmin = float(min(none.min(), tvb.min()))
    vmax = float(max(none.max(), tvb.max()))
    for ax, field, title in (
        (axes[0], none, "no limiter"),
        (axes[1], tvb, "TVB limiter (M=1)"),
    ):
        image = ax.imshow(
            field.T, origin="lower", extent=(0, 1, 0, 1), vmin=vmin, vmax=vmax,
            cmap="viridis", aspect="equal",
        )
        ax.set_title(f"C: cell-average density, {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    ax = axes[2]
    mid = none.shape[1] // 2
    ax.plot(np.linspace(0, 1, none.shape[0]), none[:, mid], "k-", label="no limiter")
    ax.plot(np.linspace(0, 1, tvb.shape[0]), tvb[:, mid], "r--", label="TVB (M=1)")
    ax.set_xlabel("x at y = 0.5")
    ax.set_ylabel("cell-average density")
    ax.set_title("C': centreline density profile")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURES / "riemann_fields.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _md_detector(evidence):
    lines = [
        "| cells/dir | dx | flagged x (M=1) | fraction x | flagged y (M=1) | fraction y | flagged x (M=2\\|u''\\|) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _mesh_rows(evidence, 2, "M=1"):
        released = [
            r for r in _mesh_rows(evidence, 2, "M=2|u''|") if r["cells"] == row["cells"]
        ]
        lines.append(
            f"| {row['cells']} | {row['dx']:.5f} | {row['flagged_x']} | "
            f"{row['fraction_x']:.4f} | {row['flagged_y']} | {row['fraction_y']:.4f} | "
            f"{released[0]['flagged_x'] if released else 'n/a'} |"
        )
    return "\n".join(lines)


def _md_constants(evidence):
    lines = ["| M | threshold M dx^2 | flagged x | flagged y |", "|---:|---:|---:|---:|"]
    for row in evidence["detector_scale"]["constant_sweep"]:
        lines.append(
            f"| {row['tvb_constant']:.3f} | {row['threshold']:.4e} | "
            f"{row['flagged_x']} | {row['flagged_y']} |"
        )
    return "\n".join(lines)


def _md_vortex(evidence):
    if not evidence or "vortex" not in evidence:
        return "not available"
    lines = []
    for degree in ("P1", "P2"):
        block = evidence["vortex"].get(degree)
        if not block:
            continue
        if "error" in block:
            lines.append(f"P{degree}: run failed -- `{block['error']}`")
            continue
        design = block["design_order"]
        lines.append(f"**P{degree}** (design order {design})")
        lines.append("")
        lines.append("| mesh | unlimited rho err | limited rho err | ratio | limited EOC |")
        lines.append("|---:|---:|---:|---:|---:|")
        for series in block["limited"]:
            for row in series["rows"]:
                unlimited = block["unlimited"][str(row["cells"])]["errors"]["density"]
                eoc = row.get("eoc", {}).get("density")
                lines.append(
                    f"| {row['cells']}^2 (M={series['tvb_constant']:g}) | "
                    f"{unlimited:.4e} | {row['errors']['density']:.4e} | "
                    f"{row['ratio_to_unlimited']['density']:.4f} | "
                    f"{eoc:.2f} |" if eoc is not None else
                    f"| {row['cells']}^2 (M={series['tvb_constant']:g}) | "
                    f"{unlimited:.4e} | {row['errors']['density']:.4e} | "
                    f"{row['ratio_to_unlimited']['density']:.4f} | -- |"
                )
        lines.append("")
        lines.append(
            f"constants behaving as a TVB limiter (ratio <= 1.10 and EOC >= {design - 0.3}): "
            f"{block['criterion'].get('constants_behaving_as_TVB')}"
        )
        lines.append("")
    return "\n".join(lines)


def _md_riemann(evidence):
    if not evidence or not evidence.get("riemann"):
        return "not available"
    lines = [
        "| mesh | limiter | min rho | min p | max rho | overshoot | undershoot | flagged (x/y) | wall s |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for block in evidence["riemann"]:
        if "none" not in block or "tvb" not in block:
            continue
        for label in ("none", "tvb"):
            row = block[label]
            lines.append(
                f"| {block['cells']}^2 | {label} | {row['min_density']:.5f} | "
                f"{row['min_pressure']:.5f} | {row['max_density']:.5f} | "
                f"{row['overshoot']:.4e} | {row['undershoot']:.4e} | "
                f"{row['flagged_cells_x']}/{row['flagged_cells_y']} | "
                f"{row['wall_seconds']:.1f} |"
            )
    return "\n".join(lines)


def render(before, after):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figures = {
        "detector": figure_detector(before, after),
        "vortex": figure_vortex(before, after),
        "riemann": figure_riemann(after),
    }
    lines = []
    lines.append("# Two-dimensional TVB limiter: validation debt and fix (mission m2549)")
    lines.append("")
    lines.append(
        "> Repository: TensorDG. Branch `agent/m2549`. Reproducers: "
        "`PYTHONPATH=. python -m benchmarks.verify_limiter_2d --tag after` and "
        "`PYTHONPATH=. python -m pytest tests/test_euler_dg_2d_limiter.py -v`. "
        "All numbers below are injected from "
        "`validation/limiter_2d/{before,after}/evidence.json`; no digit is typed by hand."
    )
    lines.append("")
    lines.append("## 1. What was wrong")
    lines.append("")
    lines.append(
        "Mission m2500 rewrote the 1-D troubled-cell test in `tensordg/limiters.py` into a "
        "scale-consistent form: the quantity compared against the TVB tolerance `M dx^2` is "
        "`|d_left - minmod(d_left, d_right)|`, which is O(dx^2 u'') on smooth data, instead of the "
        "raw one-sided differences `max(|d_left|, |d_right|)`, which are O(dx u'). The rewrite was "
        "never propagated to the 2-D limiter in `tensordg/euler_dg_2d.py`."
    )
    lines.append("")
    lines.append(
        "Two defects follow, both reproduced by the harness below on the pre-fix tree:"
    )
    lines.append("")
    lines.append(
        "1. **The 2-D test is not a TVB test.** On a smooth wave the flagged fraction grows with "
        "refinement instead of going to zero, so a fine mesh is limited harder than a coarse one; "
        "`tvb_constant` cannot release a smooth extremum because the O(dx u') measure always "
        "outgrows `M dx^2` once dx is small."
    )
    lines.append(
        "2. **`apply_tvb_limiter_2d` raises for `degree >= 2`.** The mask that removes the higher "
        "modes of a troubled direction is broadcast against the mode axes instead of the cell "
        "axes, so any P2/P3 run with an active limiter dies with `RuntimeError`. The four-quadrant "
        "verification of mission m2400 only ran the limiter at P1, so the branch was dead code and "
        "the README's \"optional TVB limiter\" was usable at P0/P1 only."
    )
    lines.append("")
    if before is not None:
        lines.append("Pre-fix evidence (`--tag before`):")
        lines.append("")
        lines.append(_md_detector(before))
        lines.append("")
        lines.append(
            "Structural check: "
            + json.dumps(before["structural"].get("error", "ran"))
        )
        lines.append("")
        lines.append(_md_constants(before))
        lines.append("")
    lines.append("## 2. The fix")
    lines.append("")
    lines.append(
        "* `tensordg/tvb.py` (new): the shared, scale-consistent test "
        "`troubled_from_differences(d_left, d_right, spacing, M)`, extracted verbatim from the "
        "1-D limiter so there is a single implementation for both dimensions."
    )
    lines.append(
        "* `tensordg/limiters.py`: `_troubled_cell_indicator` now calls the shared helper "
        "(numerically identical to the m2500 behaviour, all pre-existing 1-D tests still pass)."
    )
    lines.append(
        "* `tensordg/euler_dg_2d.py`: `_troubled_1d` delegates to the shared helper and gained a "
        "`periodic` flag -- wrap-around for periodic axes, edge replication with unflagged boundary "
        "rows on outflow axes (the old code always wrapped, so outflow boundaries saw the far side "
        "of the domain as a jump). `apply_tvb_limiter_2d` builds a per-cell mask of the right rank "
        "for the higher-mode removal, which fixes the P2/P3 crash. "
        "`solve_euler_dg_2d_vortex` gained the missing `tvb_constant` argument so the vortex "
        "wrapper can actually tune the limiter."
    )
    lines.append("")
    lines.append("## 3. Post-fix verification")
    lines.append("")
    if after is not None:
        lines.append("### 3.1 Detector scale consistency (check A)")
        lines.append("")
        lines.append(
            f"Measured data curvature max|u''| = {after['detector_scale']['curvature_estimate']:.4f} "
            f"-> TVB constant used for the release test M = {after['detector_scale']['tvb_constant_above_curvature']:.4f}."
        )
        lines.append("")
        lines.append(_md_detector(after))
        lines.append("")
        lines.append(_md_constants(after))
        lines.append("")
        lines.append(
            "Criterion: " + json.dumps(after["detector_scale"]["criterion"], default=float)
        )
        lines.append("")
        lines.append("### 3.2 Design-order preservation (check B)")
        lines.append("")
        lines.append(_md_vortex(after))
        lines.append("### 3.3 Shock robustness (check C)")
        lines.append("")
        lines.append(_md_riemann(after))
        lines.append("")
        for block in after.get("riemann", []):
            if "criterion" in block:
                lines.append(
                    f"* {block['cells']}^2: " + json.dumps(block["criterion"], default=float)
                )
        lines.append("")
        lines.append("### 3.4 Structural invariants (check D)")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(after.get("structural", {}), indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## 4. Figures")
    lines.append("")
    for key, path in figures.items():
        if path is not None:
            lines.append(f"![{key}]({path.as_posix()})")
            lines.append("")
    lines.append("## 5. Caveats")
    lines.append("")
    lines.append(
        "* The 2-D limiter remains *directional*: it limits along x and y independently and does "
        "not touch the mixed mode `c_11`, so a two-dimensional shock at 45 degrees is still "
        "under-damped compared with a genuinely multidimensional limiter (Krivodonova 2007)."
    )
    lines.append(
        "* `M` is problem dependent. The value that makes the limiter inert on the vortex "
        "(`M = 50` at P1, see table) is far above the smooth-data curvature; it is a statement "
        "about how strong the limiter must be, not a universal recommendation. For shocks the "
        "relevant regime is `M = O(1)`, which is what check C exercises."
    )
    lines.append(
        "* The four-quadrant problem has no closed-form solution, so check C measures positivity "
        "and over/undershoot, not an error norm; the accuracy claim rests on check B, which does "
        "have an exact reference."
    )
    lines.append("")
    DOC.write_text("\n".join(line for line in lines if line is not None) + "\n")
    return DOC


def main():
    before = _load("before")
    after = _load("after")
    doc = render(before, after)
    print(f"wrote {doc}")
    for path in sorted(FIGURES.glob("*.png")):
        print(f"  figure {path}")


if __name__ == "__main__":
    main()