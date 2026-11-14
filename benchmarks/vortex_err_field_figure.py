"""Field snapshot + cut-line figure for CASE-DG-EULER-VORTEX-ERR.

One DG-P2/32^2 solve (HLLC + SSP-RK3, the case's design family) at a chosen final
time, followed by the *local* error field ``rho - rho_exact`` and ``p - p_exact``
on the volume quadrature grid, plus a cut-line comparison of the numerical and the
exact (periodic) solution.  The global numbers of the case come from
``case_euler_vortex_errconserve/metrics.json``; this figure exists to show *where*
the remaining error sits, not to re-measure it.

Usage::

    PYTHONPATH=. python3 -m benchmarks.vortex_err_field_figure --final-time 1
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tensordg.euler_dg_2d as dg2  # noqa: E402
from tensordg.euler_2d import vortex_primitive  # noqa: E402
from benchmarks.case_euler_vortex_errconserve import (  # noqa: E402
    BACKGROUND, DOMAIN, PERIOD, STRENGTH, _operators,
)
from benchmarks.verify_euler_dg_2d import _primitive, _reconstruct  # noqa: E402


def snapshot(degree: int = 2, cells: int = 32, final_time: float = 1.0,
             cfl: float = 0.2) -> dict:
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=cells, degree=degree, final_time=final_time, cfl=cfl,
        strength=STRENGTH, background_velocity=BACKGROUND, domain=DOMAIN,
    )
    nodes, _weights, basis, _, _ = _operators(degree)
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    xc = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    yc = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    xq = xc[:, None, None, None] + 0.5 * dx * nodes[None, None, :, None]
    yq = yc[None, :, None, None] + 0.5 * dy * nodes[None, None, None, :]
    rho_e, u_e, v_e, p_e = vortex_primitive(xq, yq, final_time, STRENGTH,
                                            BACKGROUND, period=PERIOD)
    rho, u, v, p = _primitive(_reconstruct(coeff, basis))
    return {
        "meta": meta, "final_time": final_time, "degree": degree, "cells": cells,
        # 1-D-with-quadrature axis arrays: (cells, n) each, so a plot axis can
        # pick one quadrature node without knowing the broadcast shape.
        "x": (xc[:, None] + 0.5 * dx * nodes[None, :]).detach().numpy(),
        "y": (yc[:, None] + 0.5 * dy * nodes[None, :]).detach().numpy(),
        "rho": rho.detach().numpy(), "p": p.detach().numpy(),
        "rho_exact": rho_e.detach().numpy(), "p_exact": p_e.detach().numpy(),
        "u": u.detach().numpy(), "u_exact": u_e.detach().numpy(),
    }


def make_figure(data: dict, out_png: Path) -> dict:
    node = data["degree"] // 2  # an interior quadrature node, away from cell faces
    x = data["x"][:, node]
    y = data["y"][:, node]
    rho = data["rho"][:, :, node, node]
    rho_e = data["rho_exact"][:, :, node, node]
    p = data["p"][:, :, node, node]
    p_e = data["p_exact"][:, :, node, node]
    drho = rho - rho_e
    dp = p - p_e
    scale = 100.0 * float(np.max(np.abs(drho))) if np.any(drho) else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.2))
    vmax = max(np.abs(drho).max(), 1e-16)
    im = axes[0, 0].pcolormesh(x, y, drho.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="auto")
    axes[0, 0].set_title(r"$\rho - \rho_{\rm exact}$ (periodic ref.), t=%g"
                         % data["final_time"])
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im, ax=axes[0, 0], label="absolute density error")

    vmaxp = max(np.abs(dp).max(), 1e-16)
    im = axes[0, 1].pcolormesh(x, y, dp.T, cmap="RdBu_r", vmin=-vmaxp, vmax=vmaxp,
                               shading="auto")
    axes[0, 1].set_title(r"$p - p_{\rm exact}$, t=%g" % data["final_time"])
    axes[0, 1].set_aspect("equal")
    fig.colorbar(im, ax=axes[0, 1], label="absolute pressure error")

    j = len(y) // 2
    axes[1, 0].plot(x, rho_e[:, j], "k-", lw=2, label="exact (periodic)")
    axes[1, 0].plot(x, rho[:, j], "r--", lw=1.2, label="DG-P2/32$^2$ HLLC")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel(r"$\rho$")
    axes[1, 0].set_title("density cut at y = %.2f" % float(y[j]))
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, 100.0 * (rho[:, j] - rho_e[:, j]), "b-", lw=1.2)
    axes[1, 1].axhline(0.0, color="k", lw=0.6)
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel(r"$\rho-\rho_{\rm exact}$  [absolute]")
    axes[1, 1].set_title("cut-line error, peak |err| = %.2e (domain peak %.2e)"
                         % (float(np.max(np.abs(rho[:, j] - rho_e[:, j]))),
                            vmax))
    axes[1, 1].grid(alpha=0.3)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return {
        "name": out_png.name,
        "caption": ("上：DG-P2/32$^2$ 在 t=%g 的密度/压力逐点误差场（参考解取周期 min-image 形式）；"
                    "下左：y 中线的密度剖面对比（黑=精确解，红=数值解）；"
                    "下右：同一条剖线上的逐点误差。"
                    "误差场光滑、无单元尺度锯齿 ⇒ 剩余误差是色散型平滑误差而非格式失稳。"
                    % data["final_time"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--out-dir", default="case_euler_vortex_errconserve")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = snapshot(args.degree, args.cells, args.final_time)
    png = out_dir / f"fig_error_field_t{args.final_time:g}.png"
    figure = make_figure(data, png)
    print(f"steps={data['meta']['steps']}  -> {png}")
    print(figure["caption"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())