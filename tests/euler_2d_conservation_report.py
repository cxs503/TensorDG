"""Committed artifacts for the 2-D Euler system conservation audit (mission m4709).

Runs :func:`tests.conservation_study.study_euler_2d_conservation`, sweeps the
injected-leak negative control over four decades, and writes

* ``docs/benchmarks/results/conservation/euler_2d_system.json`` -- every number,
* ``docs/benchmarks/results/conservation/euler_2d_system_drift.png`` --
  per-invariant drift vs step count plus the leak-sweep response curve,
* ``docs/benchmarks/results/conservation/euler_2d_system_density.png`` --
  the density field of the audited run,
* ``docs/euler_2d_system_conservation_audit.md`` -- the human-readable summary.

Run from the repository root::

    python tests/euler_2d_conservation_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]
for _candidate in (_ROOT, Path(__file__).resolve().parent):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import tensordg.euler_2d as e2d  # noqa: E402
from tests.conservation_study import (  # noqa: E402
    CRITERIA,
    EULER2D_CASES,
    EULER2D_INVARIANTS,
    study_euler_2d_conservation,
)

RESULT_DIR = _ROOT / "docs" / "benchmarks" / "results" / "conservation"
DOC_PATH = _ROOT / "docs" / "euler_2d_system_conservation_audit.md"
LEAK_CASE = (("square_32", (32, 32), 1.0, ((-5.0, 5.0), (-5.0, 5.0))),)
LEAKS = (1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12)
LABELS = {
    "mass": "mass  $\\int\\rho$",
    "momentum_x": "x-momentum  $\\int\\rho u$",
    "momentum_y": "y-momentum  $\\int\\rho v$",
    "energy": "energy  $\\int E$",
}


def leak_sweep(leaks=LEAKS) -> list[dict]:
    """Measure the ledger response to a known non-conservative source."""
    rows = []
    for leak in leaks:
        result = study_euler_2d_conservation(cases=LEAK_CASE, leak=leak)
        row = result["rows"][0]
        mass = row["invariants"]["mass"]
        rows.append({
            "leak": float(leak),
            "steps": row["steps"],
            "expected_drift": row["expected_mass_drift"],
            "measured_drift": mass["scaled_drift"],
            "recovery_error": result["leak_recovery_error"],
            "bias_ratio": mass["bias_ratio"],
            "exceeds_criterion": bool(mass["scaled_drift"] > CRITERIA["euler2d_drift_max"]),
        })
    return rows


def _drift_figure(clean: dict, sweep: list[dict], path: Path) -> Path:
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    markers = {"mass": "o", "momentum_x": "s", "momentum_y": "^", "energy": "D"}
    for invariant in EULER2D_INVARIANTS:
        rows = sorted(clean["rows"], key=lambda row: row["steps"])
        steps = [row["steps"] for row in rows]
        drift = [row["invariants"][invariant]["scaled_drift"] for row in rows]
        left.semilogy(steps, np.maximum(drift, 1e-18), marker=markers[invariant],
                      label=LABELS[invariant], lw=1.2)
    left.axhline(CRITERIA["euler2d_drift_max"], color="crimson", ls="--", lw=1.4,
                 label=f"criterion {CRITERIA['euler2d_drift_max']:.0e}")
    left.set_xlabel("accepted time steps")
    left.set_ylabel("relative drift  $|I(t)-I(0)|/|I(0)|$")
    left.set_title("periodic 2-D Euler: invariants stay at float64 round-off")
    left.grid(True, which="both", alpha=0.25)
    left.legend(fontsize=8, loc="lower right")

    leaks = np.array([row["leak"] for row in sweep])
    measured = np.array([row["measured_drift"] for row in sweep])
    expected = np.array([row["expected_drift"] for row in sweep])
    right.loglog(leaks, expected, "k-", lw=1.2, label="analytic  $L\\,T/|M_0|$")
    right.loglog(leaks, measured, "o", color="tab:blue", label="measured drift")
    right.axhline(CRITERIA["euler2d_drift_max"], color="crimson", ls="--", lw=1.4,
                  label=f"criterion {CRITERIA['euler2d_drift_max']:.0e}")
    right.set_xlabel("injected mass source $L$  [mass / time]")
    right.set_ylabel("relative mass drift")
    right.set_title("negative control: ledger imbalance is recovered")
    right.grid(True, which="both", alpha=0.25)
    right.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return path


def _density_figure(path: Path, ncells=(64, 64), final_time: float = 1.0) -> dict:
    x, y, fields, metadata = e2d.solve_isentropic_vortex(
        ncells=ncells, final_time=final_time, cfl=0.4,
    )
    density = fields["density"].detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    mesh = axis.pcolormesh(
        x.detach().cpu().numpy(), y.detach().cpu().numpy(), density.T,
        shading="auto", cmap="viridis",
    )
    figure.colorbar(mesh, ax=axis, label="density $\\rho$")
    axis.set_aspect("equal")
    axis.set_title(f"isentropic vortex, {ncells[0]}x{ncells[1]} cells, t={final_time}")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return {"ncells": list(ncells), "final_time": final_time,
            "steps": metadata["steps"], "dt": metadata["dt"],
            "density_range": [float(density.min()), float(density.max())]}


def _markdown(clean: dict, sweep: list[dict], field: dict) -> str:
    lines = [
        "# 2-D Euler system conservation audit (per-step ledger)",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`python tests/euler_2d_conservation_report.py`.",
        "",
        "## Case",
        "",
        "Smooth isentropic vortex on a doubly periodic box "
        "$[-5,5]^2$, P0 finite volume with Rusanov fluxes and SSP-RK3, "
        "CFL 0.4, background velocity (1, 1).  Invariants sampled once per "
        "accepted step through the solver's `observer` hook.",
        "",
        "| case | cells | dx | dy | steps | $t_{final}$ |",
        "|---|---|---|---|---|---|",
    ]
    for row in clean["rows"]:
        dx, dy = row["cell_widths"]
        lines.append(
            f"| {row['case']} | {row['ncells'][0]}x{row['ncells'][1]} | {dx:.4f} | "
            f"{dy:.4f} | {row['steps']} | {row['final_time']:.2f} |"
        )
    lines += [
        "",
        "## Global invariants (relative to the initial integral)",
        "",
        "| case | mass | x-momentum | y-momentum | energy | max \\|mean\\|/rms |",
        "|---|---|---|---|---|---|",
    ]
    for row in clean["rows"]:
        cells = [f"{row['invariants'][inv]['scaled_drift']:.2e}" for inv in EULER2D_INVARIANTS]
        lines.append(
            f"| {row['case']} | " + " | ".join(cells)
            + f" | {row['max_bias_ratio']:.3f} |"
        )
    lines += [
        "",
        f"Criteria: drift <= {CRITERIA['euler2d_drift_max']:.0e}, "
        f"bias ratio <= {CRITERIA['euler2d_step_bias_max']}, "
        f"closure residual <= {CRITERIA['euler2d_closure_max']:.0e}.  "
        f"Measured maxima: drift {clean['max_scaled_drift']:.3e}, "
        f"bias {clean['max_bias_ratio']:.3f}, "
        f"closure {clean['max_closure_residual']:.1e}.",
        "",
        "## Negative control (injected non-conservative source)",
        "",
        "| leak L | steps | expected $LT/|M_0|$ | measured drift | recovery error | bias ratio | red? |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in sweep:
        lines.append(
            f"| {row['leak']:.0e} | {row['steps']} | {row['expected_drift']:.3e} | "
            f"{row['measured_drift']:.3e} | {row['recovery_error']:.1e} | "
            f"{row['bias_ratio']:.3f} | {'YES' if row['exceeds_criterion'] else 'no'} |"
        )
    lines += [
        "",
        "## Figures",
        "",
        "* `euler_2d_system_drift.png` -- invariant drift vs step count (left) and "
        "the leak response curve (right).",
        f"* `euler_2d_system_density.png` -- density field of the audited run "
        f"({field['ncells'][0]}x{field['ncells'][1]} cells, {field['steps']} steps, "
        f"rho in [{field['density_range'][0]:.3f}, {field['density_range'][1]:.3f}]).",
        "",
        "## Reproduce",
        "",
        "```bash",
        "TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_conservation_2d_system.py -q",
        "python tests/euler_2d_conservation_report.py            # regenerate this file",
        "```",
        "",
        "The in-suite negative control "
        "(`test_injected_cell_source_turns_the_ledger_red`, "
        "`test_the_criterion_catches_a_leak_far_below_the_clean_drift`) injects "
        "the source through `injected_mass_source`; the independent mutation "
        "evidence (a scaled periodic ghost state in `tensordg/euler_2d._pad`, "
        "which makes 5 of the 7 tests fail) is in "
        "`artifacts/euler2d_conservation_negative_control.log`.",
        "",
        "## Notes",
        "",
        "* The audit judges *conservation only*: an accuracy bug that keeps the "
        "scheme in flux-difference form (e.g. dropping the y flux entirely) leaves "
        "every invariant at round-off and is not caught here -- see "
        "`artifacts/euler2d_scope_probe.log`.  Accuracy is covered by the "
        "vortex/field-error tests.",
        "* Per-step statistics, not just the end-state difference, are what "
        "separate round-off noise (`|mean|/rms ~ 0.03-0.08`) from a one-sided "
        "leak (`|mean|/rms > 0.9`).",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    clean = study_euler_2d_conservation()
    sweep = leak_sweep()
    drift_png = _drift_figure(clean, sweep, RESULT_DIR / "euler_2d_system_drift.png")
    field = _density_figure(RESULT_DIR / "euler_2d_system_density.png")
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "cases": [list(case[:3]) for case in EULER2D_CASES],
        "criteria": {
            key: CRITERIA[key] for key in (
                "euler2d_drift_max", "euler2d_step_bias_max",
                "euler2d_closure_max", "euler2d_leak_recovery_max",
            )
        },
        "clean": clean,
        "leak_sweep": sweep,
        "field": field,
        "figures": [str(drift_png.relative_to(_ROOT))],
    }
    json_path = RESULT_DIR / "euler_2d_system.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    DOC_PATH.write_text(_markdown(clean, sweep, field))
    print(f"wrote {json_path.relative_to(_ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(_ROOT)}")
    print(f"wrote {drift_png.relative_to(_ROOT)}")
    print(f"wrote {RESULT_DIR.joinpath('euler_2d_system_density.png').relative_to(_ROOT)}")
    print(f"max drift {clean['max_scaled_drift']:.3e}  "
          f"max bias {clean['max_bias_ratio']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())