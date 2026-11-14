"""Corner probe: is the *free-space* vortex formula a solution of the periodic problem?

Pure analytics -- no solver, no time stepping.  At ``t = 5`` with
``u_inf = (1, 1)`` on ``[-5, 5]^2`` the vortex centre sits exactly on the box
corner ``(5, 5)``.  The free-space form then leaves the opposite corner
``(-5, -5)`` (radius ``sqrt(2)*10``) at the undisturbed state, whereas the
periodic (min-image) solution must carry a *quarter core* there, because the
nearest image of the centre is at distance 0 from that corner in the folded
metric (the corner is identified with the centre).

This script prints the two readings for each corner and their distance, which is
the local statement of the reference defect measured in relative L2 by
``scratch/diag_vortex_err.py``.

    PYTHONPATH=. python3 scratch/diag_vortex_corner.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from tensordg.euler_2d import vortex_primitive

BOX = 5.0
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
TIMES = (0.0, 1.0, 5.0, 7.5, 10.0)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")


def corner_probe(time: float, box: float = BOX) -> list[dict]:
    """Free-space vs periodic reference at the four corners of ``[-box, box]^2``."""
    rows = []
    corners = [(-box, -box), (box, -box), (-box, box), (box, box)]
    for cx, cy in corners:
        x = torch.tensor([cx], dtype=torch.float64)
        y = torch.tensor([cy], dtype=torch.float64)
        free = vortex_primitive(x, y, time, STRENGTH, BACKGROUND)
        periodic = vortex_primitive(x, y, time, STRENGTH, BACKGROUND, period=2.0 * box)
        row = {"time": time, "corner": [cx, cy]}
        for name, a, b in zip(FIELDS, free, periodic):
            row[f"{name}_free"] = float(a)
            row[f"{name}_periodic"] = float(b)
            row[f"{name}_delta"] = float(b - a)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    rows = []
    for time in TIMES:
        rows.extend(corner_probe(time))

    header = ("t", "corner", "rho_free", "rho_per", "|drho|_rel", "u_per")
    print("{:>5}  {:>12}  {:>8} {:>8}  {:>10} {:>8}".format(*header))
    for row in rows:
        rel = abs(row["density_delta"]) / abs(row["density_periodic"])
        print("{:5.1f}  {:>12}  {:8.4f} {:8.4f}  {:10.3e} {:8.4f}".format(
            row["time"], f"({row['corner'][0]:g},{row['corner'][1]:g})",
            row["density_free"], row["density_periodic"], rel,
            row["velocity_x_periodic"]))

    # The corner whose nearest folded image of the centre is closest must differ
    # most.  The velocity perturbation vanishes *at* the core, so density (not
    # velocity) is the discriminating quantity.
    peak = max(rows, key=lambda r: abs(r["density_delta"]))
    print(f"\nlargest corner defect: t={peak['time']:g} corner={peak['corner']} "
          f"rho_periodic={peak['density_periodic']:.4f} vs free-space "
          f"rho={peak['density_free']:.4f} (delta {peak['density_delta']:+.4f})")
    print("an undisturbed free-space corner reads rho=1, p=1, u=u_inf; the periodic "
          "reading shows the 1/4 core wound onto that corner")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(rows, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())