"""Figures + machine-readable summary for the 2-D limiter order study.

Reads ``validation/limiter_2d_order/degree{1,2}.json`` (written by
``benchmarks.limiter_2d_order_study``), merges them into ``study.json`` and
produces the figures used by ``docs/limiter_2d_order_verification.md``:

* ``order_ladder.png``  -- relative L2 error vs mesh size on log-log axes for
  every limiter configuration, with the design-order ``h**(p+1)`` and the
  first-order reference slopes drawn as guides;
* ``fields_limit.png``  -- field cloud maps (density, |velocity| and the
  cell-average density error against the analytic vortex) for the unlimited
  run and for the strict ``M = 0`` limiter, i.e. a picture of *what* the
  order loss looks like: the smooth extremum is flattened;
* ``eoc_bars.png``      -- observed order per field and configuration.

Run with::

    PYTHONPATH=. python -m benchmarks.report_limiter_2d_order
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import tensordg.euler_dg_2d as dg2  # noqa: E402
from tensordg.euler_2d import vortex_primitive  # noqa: E402

VALIDATION = Path("validation/limiter_2d_order")
FIGURES = Path("docs/figures/limiter_2d_order")
DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
FINAL_TIME = 1.0
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")

COLORS = {
    "none": "#1f77b4",
    "tvb_M0": "#d62728",
    "tvb_M1": "#ff7f0e",
    "tvb_M10": "#2ca02c",
    "tvb_M100": "#9467bd",
    "weno_M0": "#8c564b",
    "weno_M10": "#17becf",
}


def merge():
    merged = {"degrees": {}}
    for degree in (1, 2):
        path = VALIDATION / f"degree{degree}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        bracket_path = VALIDATION / f"degree{degree}_bracket.json"
        if bracket_path.exists():
            bracket = json.loads(bracket_path.read_text())
            for key, entry in bracket["degrees"].items():
                entry["configs"].update(payload["degrees"][key]["configs"])
                entry["bit_identical_to_unlimited_on_finest_mesh"].update(
                    payload["degrees"][key]["bit_identical_to_unlimited_on_finest_mesh"]
                )
                payload["degrees"][key] = entry
        merged["degrees"].update(payload["degrees"])
        merged.setdefault("M_threshold_estimate", payload.get("M_threshold_estimate"))
        for key in ("final_time", "strength", "domain", "background"):
            merged.setdefault(key, payload.get(key))
    out = VALIDATION / "study.json"
    out.write_text(json.dumps(merged, indent=2))
    print("merged ->", out)
    return merged


def plot_ladder(study):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, field in zip(axes, ("density", "pressure")):
        for degree_key, degree_data in sorted(study["degrees"].items()):
            degree = int(degree_key)
            meshes = degree_data["meshes"]
            for label, row in degree_data["configs"].items():
                values = [row[str(n)]["relative_l2"][field] for n in meshes]
                if min(values) <= 0.0:
                    continue
                style = "-o" if degree == 2 else "--s"
                axis.loglog(
                    meshes, values, style, color=COLORS.get(label, "grey"),
                    ms=4, lw=1.4, alpha=0.9,
                    label=f"P{degree} {label}",
                )
            # reference slopes anchored at the coarse end of the unlimited run
            if "none" in degree_data["configs"]:
                row = degree_data["configs"]["none"]
                n0 = meshes[0]
                e0 = row[str(n0)]["relative_l2"][field]
                for order, colour in ((degree + 1, "0.35"), (1, "0.7")):
                    xs = [n0, meshes[-1]]
                    ys = [e0, e0 * (meshes[-1] / n0) ** (-order)]
                    axis.loglog(xs, ys, ":", color=colour, lw=1.0)
                    axis.annotate(f"h^{order}", (xs[1], ys[1]),
                                  textcoords="offset points", xytext=(-6, 4),
                                  fontsize=8, color=colour)
        axis.set_xlabel("cells per direction $n$")
        axis.set_ylabel(f"relative $L^2$ error ({field})")
        axis.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        axis.set_title(f"isentropic vortex, $T$ = {FINAL_TIME:g}: {field}")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("2-D limiter order study: is the limiter inert on smooth flow?")
    fig.tight_layout()
    path = FIGURES / "order_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def _fields_and_error(ncells=16, degree=2, limiter=None, tvb=1.0):
    """Cell-average fields plus the analytic reference on the same mesh."""
    nodes, _, basis, _, _ = dg2.legendre_operators(degree, max(degree + 2, 3))
    x, y, coeff, _ = dg2.solve_euler_dg_2d_vortex(
        ncells=ncells, degree=degree, final_time=FINAL_TIME,
        limiter=limiter, tvb_constant=tvb, domain=DOMAIN,
    )
    at_quad = torch.einsum("xyklv,ak,bl->xyabv", coeff, basis, basis)
    reference = vortex_primitive(x[:, None], y[None, :], FINAL_TIME)
    averages = dg2._primitive(coeff[..., 0, 0, :])
    speed = torch.sqrt(averages[1] ** 2 + averages[2] ** 2)
    return {
        "x": x.numpy(), "y": y.numpy(),
        "density": averages[0].numpy(),
        "speed": speed.numpy(),
        "density_error": (averages[0] - reference[0]).numpy(),
        "quad_energy": float((at_quad ** 2).mean().item()),
    }


def plot_fields(ncells=16, degree=2):
    unlimited = _fields_and_error(ncells, degree, limiter=None)
    strict = _fields_and_error(ncells, degree, limiter="tvb", tvb=0.0)
    rows = (
        ("density", "density $\\rho$", "%+.4f", "viridis"),
        ("speed", "speed $|\\mathbf{u}|$", "%+.4f", "viridis"),
        ("density_error", "$\\rho - \\rho_{exact}$", "%+.2e", "coolwarm"),
    )
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.0, 10.4))
    for row, (key, title, fmt, cmap) in enumerate(rows):
        for col, (data, label) in enumerate(
            ((unlimited, "no limiter"), (strict, "TVB $M$ = 0 (strict TVD)"))
        ):
            axis = axes[row, col]
            limit = max(abs(data[key]).max(), 1e-30)
            image = axis.imshow(
                data[key].T, origin="lower", cmap=cmap,
                extent=(data["x"][0], data["x"][-1], data["y"][0], data["y"][-1]),
                vmin=-limit if key == "density_error" else None,
                vmax=limit if key == "density_error" else None,
            )
            axis.set_title(f"{title} -- {label}", fontsize=9)
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle(
        f"Isentropic vortex, P{degree}, {ncells}$^2$ cells, T = {FINAL_TIME:g}: "
        "the strict limiter flattens the smooth extremum"
    )
    fig.tight_layout()
    path = FIGURES / "fields_limit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def plot_eoc(study):
    degree_data = study["degrees"].get("2") or study["degrees"]["1"]
    labels = list(degree_data["configs"])
    fig, axis = plt.subplots(figsize=(9.0, 4.2))
    width = 0.8 / len(FIELDS)
    for index, field in enumerate(FIELDS):
        values = [degree_data["configs"][label]["eoc"][field] for label in labels]
        positions = [k + index * width for k in range(len(labels))]
        axis.bar(positions, values, width=width, label=field)
    axis.axhline(degree_data["design_order"], color="k", ls="--", lw=1.0)
    axis.annotate(f"design order $p+1$ = {degree_data['design_order']}",
                  (0.02, degree_data["design_order"] + 0.05), fontsize=8)
    axis.set_xticks([k + 0.4 - width / 2 for k in range(len(labels))])
    axis.set_xticklabels(labels, rotation=15)
    axis.set_ylabel("observed order (finest interval)")
    axis.set_title("Observed convergence order per limiter configuration (P2 vortex)")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", ls=":", lw=0.5)
    fig.tight_layout()
    path = FIGURES / "eoc_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    study = merge()
    plot_ladder(study)
    plot_eoc(study)
    plot_fields()


if __name__ == "__main__":
    main()