"""Reproduce the PR #100 CI red (stale isothermal-wall pins) and the repaired pins.

The PR branch ``agent/m3714`` carried a pre-PR-#74 copy of
``tests/test_ns_heat_wall_closure.py``.  Two pins in that copy no longer hold on
the current tree:

* ``bottom_ratio < 1.1`` on the 16/32 cold-wall energy residual ratio -- the
  measured value is 1.1254, i.e. the bound sits 2.3 % *below* the truth;
* ``pytest.raises(FloatingPointError)`` around ``solve_heat_conduction_dg_2d``
  -- the crash stopped happening when the m3139 wall repair (one-sided DG wall
  derivative as the default plus a positivity filter) landed on main, so the
  entry point now completes instead of aborting.

Both numbers are recomputed here from scratch (float64/CPU) and checked against
the stale bounds and against the repaired bounds that main adopted in c051cef,
so the CI verdict is derived from arithmetic rather than quoted from a log.

Run: ``PYTHONPATH=. python validation/ns_heat_wall_closure/pr100_repro.py``
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tensordg import ns_dg_2d as ns
from tests.test_ns_heat_wall_closure import T_BOTTOM, T_TOP, _residual, _setup

OUT = Path(__file__).with_name("pr100_repro.json")


def _ratios():
    coarse = _residual(_setup(16)).abs().amax(dim=(2, 3))   # ny = 16
    mid = _residual(_setup(32)).abs().amax(dim=(2, 3))      # ny = 32
    fine = _residual(_setup(64)).abs().amax(dim=(2, 3))     # ny = 64
    return {
        "wall_ny16": float(coarse[:, 0, 3].max()),
        "wall_ny32": float(mid[:, 0, 3].max()),
        "wall_ny64": float(fine[:, 0, 3].max()),
        "interior_ny16": float(coarse[:, 1:-1, 3].max()),
        "interior_ny32": float(mid[:, 1:-1, 3].max()),
        "interior_ny64": float(fine[:, 1:-1, 3].max()),
    }, {
        "interior_ratio_16_32": float(coarse[:, 1:-1, 3].max() / mid[:, 1:-1, 3].max()),
        "wall_ratio_16_32": float(coarse[:, 0, 3].max() / mid[:, 0, 3].max()),
        "wall_ratio_32_64": float(mid[:, 0, 3].max() / fine[:, 0, 3].max()),
    }


def _entry_point():
    x, y, coeff, metadata = ns.solve_heat_conduction_dg_2d(
        ncells=(2, 8), final_time=0.5, dtype=torch.float64)
    fields = ns.cell_fields(coeff)
    pressure, density = fields["pressure"], fields["density"]
    temperature = pressure / density
    return {
        "finite": bool(torch.isfinite(coeff).all()),
        "p_min": float(pressure.min()),
        "p_max": float(pressure.max()),
        "rho_min": float(density.min()),
        "rho_max": float(density.max()),
        "theta_hot_wall": float(temperature[0, 0]),
        "theta_cold_wall": float(temperature[0, -1]),
        "monotone_in_y": bool((temperature[0][1:] <= temperature[0][:-1]).all()),
        "steps": metadata.get("steps"),
        # theta = p / rho along y at the first x-cell (R = 1 in this package)
        "theta_profile": [float(v) for v in temperature[0].tolist()],
        "y_cells": int(temperature.shape[1]),
    }, {"T_BOTTOM": T_BOTTOM, "T_TOP": T_TOP}


def main() -> int:
    residuals, ratios = _ratios()
    entry, _ = _entry_point()

    stale = {
        "wall_ratio_16_32_lt_1.1": ratios["wall_ratio_16_32"] < 1.1,
        "entry_point_raises_FloatingPointError": False,  # observed: completes
    }
    repaired = {
        "interior_ratio_16_32_gt_1.7": ratios["interior_ratio_16_32"] > 1.7,
        "wall_ratio_16_32_lt_1.2": ratios["wall_ratio_16_32"] < 1.2,
        "wall_ratio_32_64_lt_1.1": ratios["wall_ratio_32_64"] < 1.1,
        "wall_ratio_monotone": ratios["wall_ratio_32_64"] < ratios["wall_ratio_16_32"],
        "wall_floor_gt_1e_2": residuals["wall_ny64"] > 1e-2,
        "entry_point_completes": entry["finite"] and entry["steps"] is not None,
        "entry_point_positive": entry["p_min"] > 0.0 and entry["rho_min"] > 0.0,
        "entry_point_monotone": entry["monotone_in_y"],
        "entry_point_within_walls": (
            float(entry["theta_cold_wall"]) > T_TOP - 1e-2
            and float(entry["theta_hot_wall"]) < T_BOTTOM + 1e-2
        ),
    }

    report = {
        "mission": "m4361 / PR #100 (agent/m3714) CI red repair",
        "residuals": residuals,
        "ratios": ratios,
        "stale_pins_hold": stale,
        "stale_pins_all_hold": all(stale.values()),
        "repaired_pins_hold": repaired,
        "repaired_pins_all_hold": all(repaired.values()),
        "entry_point": entry,
        "margin_of_stale_wall_bound": ratios["wall_ratio_16_32"] / 1.1 - 1.0,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if (not report["stale_pins_all_hold"] and report["repaired_pins_all_hold"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())