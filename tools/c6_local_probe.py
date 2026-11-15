"""Local (light) C6 probes: does the CKS1998 2-D Riemann case diverge here?

Each probe is a short DG P1 run of ``benchmarks.riemann_2d_cks1998``; the rows
are written to ``validation/riemann2d_c6/local_probe.json`` incrementally so a
capped run still leaves its finished rows on disk.  Nothing is typed by hand --
every number in the report's tables comes from this file.

The point of the probes is *falsification*: the previous mission recorded a
divergent configuration (32x32, CFL = 0.2, t = 0.25) and named two candidate
causes (CFL past the stability limit vs the M = 1 TVB indicator).  These runs
check the CFL axis at the meshes that are cheap enough to run locally.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.riemann_2d_cks1998 import (  # noqa: E402
    a_priori_bounds,
    c6_extremum_factor,
    c6_mass_bound_ratio,
    run_riemann_2d,
)

BUNDLE = Path("validation/riemann2d_c6")
PROBES = (
    (8, 0.1, 0.05, 0.0),
    (16, 0.1, 0.05, 1.0),
    (16, 0.2, 0.05, 1.0),
    (16, 0.1, 0.25, 1.0),
    (16, 0.2, 0.25, 1.0),
)


def main():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    payload = {"benchmark": "c6_local_probe", "case": "Lax_Liu_1998_config_3",
               "rows": []}
    target = BUNDLE / "local_probe.json"
    torch.set_num_threads(2)
    for ncells, cfl, final_time, tvb in PROBES:
        label = f"P1 {ncells}x{ncells} cfl={cfl:g} t={final_time:g} M={tvb:g}"
        started = time.time()
        try:
            record = run_riemann_2d(ncells=ncells, degree=1, final_time=final_time,
                                    cfl=cfl, tvb_constant=tvb)
            bounds = a_priori_bounds(final_time)
            row = {
                "label": label,
                "ncells": ncells,
                "cfl": cfl,
                "final_time": final_time,
                "tvb_constant": tvb,
                "steps": record["steps"],
                "mass_change": record["conservation"]["mass"]["change"],
                "mass_flux_bound": bounds["mass_flux_bound"],
                "c6_mass_bound_ratio": c6_mass_bound_ratio(record),
                "c6_extremum_factor": c6_extremum_factor(record),
                "rho_min": record["positivity"]["rho_min"],
                "rho_max": record["positivity"]["rho_max"],
                "p_max": record["positivity"]["p_max"],
                "finite": record["finite"],
                "wall_seconds": round(time.time() - started, 2),
                "diverged": not (
                    c6_mass_bound_ratio(record) <= 1.0
                    and c6_extremum_factor(record) <= 1.5
                ),
            }
        except Exception as exc:  # noqa: BLE001
            row = {"label": label, "ncells": ncells, "cfl": cfl,
                   "final_time": final_time, "tvb_constant": tvb,
                   "error": f"{type(exc).__name__}: {exc}",
                   "wall_seconds": round(time.time() - started, 2),
                   "diverged": True}
        payload["rows"].append(row)
        target.write_text(json.dumps(payload, indent=2, default=str))
        print(label, "->", {k: row.get(k) for k in
                            ("steps", "mass_change", "c6_mass_bound_ratio",
                             "c6_extremum_factor", "diverged", "error")},
              flush=True)
    return payload


if __name__ == "__main__":
    main()