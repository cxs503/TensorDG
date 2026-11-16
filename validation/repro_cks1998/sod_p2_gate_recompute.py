"""Re-run the DG P2 Sod gate from scratch and settle the 2.368 % vs 3.838 % dispute.

Background
----------
``docs/CASE-REPRO-DG-CKS1998-DUAL-INSTRUMENT-GATES.md`` §3.3 records a *hard*
mismatch: the repository publishes ``p2_n400`` worst-field relative L2 = **2.368 %**
(``docs/CKS1998-round2-results.md``, ``docs/AUDIT_m15345_REPRO-DG-CKS1998.md``), while
the committed solver profile ``sod-dg-p2-n400.npz`` scores **3.8384 %** on the same
configuration (DG P2, TVB M=1, ``limiter_stage="final"``, N=400, CFL 0.12, t=0.2).
The instrument is *not* the cause (dual-instrument delta ≤ 0.07 pp), so one of the two
numbers comes from a different code state or a different configuration.

This driver answers the two questions the previous round had to leave open:

1. **What does the current code read?** — the documented configuration is run from
   scratch (no cache reuse) and scored against both reference instruments, so the
   published number and the cached profile are compared with a third, independent
   reading produced on the tree that is being committed.
2. **Can the published 2.368 % be reproduced by any neighbouring configuration?**
   — a small pre-registered sweep over the configuration axes that the gate's own
   documentation leaves implicit (limiter stage, positivity fallback, TVB constant,
   CFL). If one of them lands on 2.368 %, the published number is a *configuration*
   mismatch; if none does, it is a code-state mismatch and must be recorded as such.

Nothing here changes a solver, a tolerance or a gate: the driver only measures.

Usage
-----
    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m validation.repro_cks1998.sod_p2_gate_recompute \
        --out docs/benchmarks/results/cks1998_dual_instrument/sod_p2_gate_recompute.json
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

#: the configuration the repository documents for this gate
DOCUMENTED = {"degree": 2, "ncells": 400, "cfl": 0.12, "limiter": "tvb",
              "tvb_constant": 1.0, "limiter_stage": "final", "positivity": "scale"}

#: the number this repository published for it (worst field = velocity)
PUBLISHED_VELOCITY_REL_L2 = 0.02368
#: the committed solver profile's reading (see metrics.json gate row)
CACHED_VELOCITY_REL_L2 = 0.03838350233770419
#: the gate tolerance the reading is judged against
TOLERANCE = 0.03

#: pre-registered sweep: every axis this gate's documentation leaves implicit
SWEEP = (
    {"tag": "documented", **DOCUMENTED},
    {"tag": "limiter_stage=every", **{**DOCUMENTED, "limiter_stage": "every"}},
    {"tag": "positivity=collapse", **{**DOCUMENTED, "positivity": "collapse"}},
    {"tag": "tvb_constant=0", **{**DOCUMENTED, "tvb_constant": 0.0}},
    {"tag": "cfl=0.3", **{**DOCUMENTED, "cfl": 0.3}},
    {"tag": "cfl=0.05", **{**DOCUMENTED, "cfl": 0.05}},
    {"tag": "p3_n400", **{**DOCUMENTED, "degree": 3}},
)


def _relative_l2(got, ref) -> float:
    import numpy as np

    diff = got - ref
    return float(np.sqrt(np.mean(diff ** 2)) / np.sqrt(np.mean(ref ** 2)))


def run_config(config: dict) -> dict:
    """Run one configuration and score all three fields against both instruments."""
    from benchmarks.sod_exact import GAMMA
    from tensordg.euler_dg_1d import solve_euler_dg_1d
    from validation.repro_cks1998.station_reference import exact_cell_averages
    from benchmarks.riemann_exact import cell_averages as legacy_cell_averages

    from benchmarks.shock_reference_quadrature_audit import (  # noqa: E402
        LEGACY_ORDER, SOD, SOD_TIME,
    )

    ncells = config["ncells"]
    started = time.time()
    _, coeff, meta = solve_euler_dg_1d(
        SOD[0], SOD[1], ncells=ncells, degree=config["degree"], final_time=SOD_TIME,
        cfl=config["cfl"], limiter=config["limiter"],
        tvb_constant=config["tvb_constant"], limiter_stage=config["limiter_stage"],
        positivity=config["positivity"],
    )
    average = coeff[:, 0]
    got = {
        "density": average[:, 0].numpy(),
        "velocity": (average[:, 1] / average[:, 0]).numpy(),
        "pressure": ((GAMMA - 1.0) * (average[:, 2] - 0.5 * average[:, 1] ** 2 / average[:, 0])).numpy(),
    }
    _, exact = exact_cell_averages(ncells, SOD_TIME)
    _, legacy = legacy_cell_averages(ncells, SOD_TIME, SOD[0], SOD[1],
                                     quadrature_order=LEGACY_ORDER)

    fields = {}
    for key in ("density", "velocity", "pressure"):
        e_legacy = _relative_l2(got[key], legacy[key])
        e_exact = _relative_l2(got[key], exact[key])
        fields[key] = {
            "legacy_rel_l2": e_legacy,
            "exact_rel_l2": e_exact,
            "delta_pp": 100.0 * (e_legacy - e_exact),
            "pass_legacy": bool(e_legacy < TOLERANCE),
            "pass_exact": bool(e_exact < TOLERANCE),
        }
    worst_legacy = max(fields, key=lambda k: fields[k]["legacy_rel_l2"])
    worst_exact = max(fields, key=lambda k: fields[k]["exact_rel_l2"])
    repro = 100.0 * abs(fields[worst_legacy]["legacy_rel_l2"] - PUBLISHED_VELOCITY_REL_L2)
    return {
        "config": config,
        "steps": meta["steps"],
        "seconds": round(time.time() - started, 2),
        "fields": fields,
        "worst_field_legacy": worst_legacy,
        "worst_field_exact": worst_exact,
        "worst_legacy": fields[worst_legacy]["legacy_rel_l2"],
        "worst_exact": fields[worst_exact]["exact_rel_l2"],
        # distance of this configuration's worst reading from the *published* number,
        # in percentage points; used only to ask "does any neighbour reproduce it?"
        "gap_to_published_pp": repro,
        "reproduces_published": bool(repro < 0.05),  # 0.05 pp ~ the precision published
    }


def git_identity() -> dict:
    def _git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except Exception:  # pragma: no cover
            return None

    return {"head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain"))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    import torch

    document = {
        "case": "Sod / Toro test 1, DG P2 TVB, N=400, t=0.2, gate tolerance 3 %",
        "question": "does the current tree read the published 2.368 % or the cached 3.8384 %?",
        "published_velocity_rel_l2": PUBLISHED_VELOCITY_REL_L2,
        "cached_velocity_rel_l2": CACHED_VELOCITY_REL_L2,
        "tolerance": TOLERANCE,
        "git": git_identity(),
        "environment": {"machine": platform.machine(), "python": platform.python_version(),
                        "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
        "runs": [],
    }
    for config in SWEEP:
        row = run_config(dict(config))
        document["runs"].append(row)
        print(f"{config['tag']}: worst legacy {100 * row['worst_legacy']:.4f} % "
              f"(exact {100 * row['worst_exact']:.4f} %), published {100 * PUBLISHED_VELOCITY_REL_L2:.3f} %",
              file=sys.stderr, flush=True)

    base = document["runs"][0]
    document["verdict"] = {
        "current_code_worst_legacy_pct": 100.0 * base["worst_legacy"],
        "current_code_gate_passes": bool(base["worst_exact"] < TOLERANCE),
        "current_code_margin_pp": 100.0 * (TOLERANCE - base["worst_exact"]),
        "agrees_with_cached_profile": bool(
            abs(base["worst_legacy"] - CACHED_VELOCITY_REL_L2) < 0.005),
        "configs_reproducing_published": [r["config"]["tag"] for r in document["runs"]
                                          if r["reproduces_published"]],
        "attribution": None,
    }
    reproducing = document["verdict"]["configs_reproducing_published"]
    document["verdict"]["attribution"] = (
        "configuration mismatch: " + ", ".join(reproducing) if reproducing
        else "no neighbouring configuration reproduces the published reading "
             "=> code-state mismatch, the published 2.368 % is stale")

    text = json.dumps(document, indent=2, ensure_ascii=False)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())