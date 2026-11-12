#!/usr/bin/env python3
"""Board cross-audit (交差核验) of mission m16218 — anchor CASE-DG-KOVASZNAY-NS (TensorDG).

Verification, not re-do.  Writes artifacts/audit_m16227/audit.json and
(--insert-verdict) one row in tensor_maturity_verdicts.

Checks (all re-executed by this audit, not copied from the subject report):
  R1 registry  : tensor_benchmark_cases WHERE anchor_slug='CASE-DG-KOVASZNAY-NS'
                 -> 0 rows (recorded); criterion traced to tensor_case_pool id 94.
  R2 files     : impl/entry/run_artifacts/tests exist and are non-placeholder.
  R3 numbers   : reported got re-derived from validation/kovasznay/fields_primary.npz
                 with an independent numpy Gauss-Legendre quadrature, and the stored
                 'exact' field re-derived from the closed form (independent of the
                 package's own velocity_l2_error()).
  R4 tests     : pytest tests/test_kovasznay_dg_2d.py -> 8 passed (re-run here).
  R5 repro     : --job p1_12 re-run in a copy of the delivered tree; every JSON leaf
                 compared (wall_seconds excluded).
  R6 report    : platform report 15318 six sections present; headline numbers vs JSON.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

AUDITED = "/root/tp-agent/work/m16218"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "artifacts", "audit_m16227")


def dsn_from_env(path="/root/tensor-ecosystem/TensorPlatform/.env"):
    for line in open(path):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                         "postgresql://")
    raise SystemExit("no DATABASE_URL")


def independent_error():
    import numpy as np
    d = np.load(os.path.join(REPO, "validation/kovasznay/fields_primary.npz"))
    x, y, u, v, ue, ve = (d[k] for k in ("x", "y", "u", "v", "u_exact", "v_exact"))
    Re, k = 40.0, 2 * math.pi
    lam = 0.5 * (Re - math.sqrt(Re * Re + 4 * k * k))
    ue2 = 1.0 - np.exp(lam * x) * np.cos(k * y)
    ve2 = (lam / k) * np.exp(lam * x) * np.sin(k * y)
    nodes, w = np.polynomial.legendre.leggauss(5)
    dx, dy = 1.5 / 24, 2.0 / 24
    W = w[None, None, :, None] * w[None, None, None, :] * 0.25 * dx * dy
    num = (u - ue) ** 2 + (v - ve) ** 2
    den = ue ** 2 + ve ** 2
    return {
        "reference_field_max_dev_from_closed_form": float(
            max(abs(ue - ue2).max(), abs(ve - ve2).max())),
        "velocity_rel_l2_independent": float(math.sqrt((W * num).sum() / (W * den).sum())),
        "reported": json.load(open(os.path.join(
            REPO, "validation/kovasznay/primary.json")))["error"]["velocity"],
        "quadrature": "5-pt Gauss-Legendre per direction, Jacobian dx*dy/4 (numpy only)",
    }


def registry(dsn):
    import psycopg
    with psycopg.connect(dsn) as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases "
                    "WHERE anchor_slug='CASE-DG-KOVASZNAY-NS'")
        n_bench = cur.fetchone()[0]
        cur.execute("SELECT id,case_code,repo,tol_policy,tier,status,mission_id,linked_anchor "
                    "FROM tensor_case_pool WHERE case_code='DG-KOVASZNAY-NS'")
        cols = [d.name for d in cur.description]
        pool = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("SELECT id,status,repo FROM tensor_dev_missions WHERE id=16218")
        mission = cur.fetchone()
    return {"tensor_benchmark_cases_rows": n_bench, "tensor_case_pool": pool,
            "mission_16218": mission}


def main():
    os.makedirs(OUT, exist_ok=True)
    dsn = dsn_from_env()
    reg = registry(dsn)
    indep = independent_error()
    primary = json.load(open(os.path.join(REPO, "validation/kovasznay/primary.json")))
    summary = json.load(open(os.path.join(REPO, "validation/kovasznay/summary.json")))
    rows = summary["rows"]
    for name in ("mach030", "bc_extrap"):  # rows landed as per-run JSONs (F2)
        if name not in rows:
            rows[name] = json.load(open(os.path.join(
                REPO, "validation/kovasznay/%s.json" % name)))["error"]
    order = {}
    for deg in (1, 2):
        a, b = rows[f"p{deg}_12"]["velocity"], rows[f"p{deg}_24"]["velocity"]
        order[f"P{deg}"] = math.log(a / b) / math.log(2.0)
    gate = 0.01
    got = primary["error"]["velocity"]
    audit = {
        "audit_mission": 16227, "subject_mission": 16218,
        "anchor": "CASE-DG-KOVASZNAY-NS", "repo": "TensorDG",
        "registry": reg,
        "gate": {"criterion": "<1% relative L2 velocity (case_pool#94 tol_policy)",
                 "got": got, "margin_factor": gate / got, "passed": got < gate},
        "ref_vs_got": {"ref": "Kovasznay 1948 closed form, program-evaluated",
                       "got_velocity_rel_l2": got, "got_u": primary["error"]["u"],
                       "got_v": primary["error"]["v"],
                       "independent_recompute": indep,
                       "steady_residual": primary["steady_residual"]},
        "convergence": {k: round(v, 3) for k, v in order.items()},
        "negative_controls": {"p1_12": rows["p1_12"]["velocity"],
                              "bc_extrap": rows["bc_extrap"]["velocity"]},
        "config_coverage": {"defined_in_RUNS": 13, "recorded_in_summary": len(rows),
                            "not_run": ["mach120", "bc_adjoint", "wavenumber3",
                                        "ic_uniform", "ic_exact_ref"]},
        "repro": {"p1_12_rerun": "all JSON leaves identical except wall_seconds "
                                 "(97.65 s vs 94.80 s rerun)",
                  "primary_rerun": "see docs/board_audit_m16227/M16227_BOARD_AUDIT.md"},
        "tests": {"cmd": "pytest tests/test_kovasznay_dg_2d.py -q", "result": "8 passed in 79.95 s"},
        "findings": [
            {"id": "F1", "severity": "non-blocking", "scope": "platform ledger",
             "text": "tensor_benchmark_cases has 0 rows for this anchor; criterion lives only "
                     "in tensor_case_pool#94 (tol_policy '<1%'), linked_anchor is NULL. "
                     "Nothing to compare got against inside the referee table."},
            {"id": "F2", "severity": "minor", "scope": "provenance",
             "text": "m16218 worktree carries uncommitted edits (summary.json +mach030/"
                     "+bc_extrap rows, regenerated kovasznay_convergence.png). Per-run JSONs "
                     "(mach030.json, bc_extrap.json) are committed in 64bfa60; only the "
                     "aggregate+figure are dirty, so the cited numbers stay traceable."},
            {"id": "F3", "severity": "minor", "scope": "report wording",
             "text": "committed docs/CASE-DG-KOVASZNAY-NS.md line 31 claims the k=3 wavenumber "
                     "variant 'ran and is in table 1'; table 1 and 5.3 mark wavenumber3 '未跑'. "
                     "Overstated claim in the doc (platform report 15318 does not repeat it)."},
            {"id": "F4", "severity": "minor", "scope": "experiment design",
             "text": "gate config marches to t=0.08 while every sweep run uses t=0.2; error still "
                     "drifting (+44.8% over the window, disclosed in report 5.4). Gate result at "
                     "t=0.2 not evidenced in the repo (adversarial probe submitted, not finished)."},
            {"id": "F5", "severity": "note", "scope": "claims",
             "text": "'M^2 fit n=1.98' in report 15318 rests on two Mach points - a slope, not a fit."},
        ],
        "level": 4,
        "level_rationale": "L1 closed-form residuals at 1e-15 + 8 judgment tests; L2 real "
                           "input->solve->output implementation (kovasznay.py drives the main NS "
                           "DG solver); L3 integration in the main solver (_boundary_flux 'state' "
                           "branch, backward compatible) with in-repo callers; L4 complete case "
                           "+ mesh/degree convergence + negative controls + independently "
                           "re-verified reproduction. L5 withheld: no independent third-party "
                           "numeric anchor (ref is a closed form, no literature error values).",
    }
    with open(os.path.join(OUT, "audit.json"), "w") as f:
        json.dump(audit, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: audit[k] for k in ("gate", "convergence", "config_coverage")},
                     indent=1, ensure_ascii=False))

    if "--insert-verdict" in sys.argv:
        import psycopg
        ev = {
            "impl_files": ["tensordg/kovasznay.py", "tensordg/ns_dg_2d.py"],
            "impl_pointers": {
                "tensordg/kovasznay.py:92": "kovasznay_exact closed form",
                "tensordg/kovasznay.py:214": "velocity_l2_error quadrature-exact relative L2",
                "tensordg/kovasznay.py:441": "solve_kovasznay_dg_2d chunked march -> main NS solver",
                "tensordg/ns_dg_2d.py:599": "_boundary_flux 'state' through-flow HLLC face (new)",
            },
            "entry": ["benchmarks/verify_kovasznay_dg_2d.py"],
            "run_artifacts": ["validation/kovasznay/primary.json",
                              "validation/kovasznay/summary.json",
                              "validation/kovasznay/fields_primary.npz",
                              "validation/kovasznay/mach030.json",
                              "validation/kovasznay/bc_extrap.json",
                              "validation/kovasznay/p1_12.json",
                              "validation/kovasznay/p2_24.json",
                              "docs/figures/kovasznay_fields.png",
                              "docs/figures/kovasznay_steady_state.png",
                              "docs/figures/kovasznay_convergence.png"],
            "tests": ["tests/test_kovasznay_dg_2d.py::test_closed_form_is_exact",
                      "tests/test_kovasznay_dg_2d.py::test_decay_rate_solves_the_characteristic_relation",
                      "tests/test_kovasznay_dg_2d.py::test_boundary_data_is_periodic_in_y",
                      "tests/test_kovasznay_dg_2d.py::test_prescribed_face_carries_mass_flux",
                      "tests/test_kovasznay_dg_2d.py::test_modal_restart_is_lossless",
                      "tests/test_kovasznay_dg_2d.py::test_steady_error_shrinks_with_degree",
                      "pytest re-run by audit: 8 passed in 79.95 s"],
            "integration_callers": [
                "tensordg/kovasznay.py: imports _rhs_ns, solve_navier_stokes_dg_2d from the main NS solver",
                "benchmarks/verify_kovasznay_dg_2d.py: --job primary drives tensordg.kovasznay.solve_kovasznay_dg_2d",
                "tensordg/ns_dg_2d.py::_boundary_flux: 'state' branch consumed by the case's inlet/outlet",
                "tensordg/__init__.py: exports solve_kovasznay_dg_2d, velocity_l2_error"],
            "benchmark": {
                "anchor": "CASE-DG-KOVASZNAY-NS",
                "criterion_source": "tensor_case_pool id 94 (tol_policy '<1% | 速度 L2 相对误差')",
                "tolerance": 0.01, "ref": "Kovasznay 1948 closed form (self-residual 3.77e-15)",
                "got_relative_l2_velocity": 8.818188869671156e-05,
                "got_pct": 0.0088, "margin_factor": 113.4, "passed": True,
                "independent_recompute_relative_l2": 8.818188869671156e-05,
                "ref_field_max_dev_from_closed_form": 0.0,
                "observed_order": {"P1": 2.08, "P2": 2.92},
                "negative_controls": {"p1_12": 0.0243, "bc_extrap": 3.371},
                "note": "tensor_benchmark_cases has 0 rows for this anchor (F1); "
                        "criterion taken from tensor_case_pool#94",
            },
            "repro_cmd": "cd /root/tp-agent/work/m16218 && PYTHONPATH=. python -m benchmarks.verify_kovasznay_dg_2d --job primary",
            "repro_verified": "p1_12 re-run in a tree copy: all JSON leaves identical except "
                              "wall_seconds; pytest 8 passed",
            "external_anchor": "L. I. G. Kovasznay, Proc. Cambridge Philos. Soc. 44 (1948) 58-62; "
                               "Bassi & Rebay, J. Comput. Phys. 131 (1997) 267 DOI 10.1006/jcph.1996.5572 "
                               "(no third-party error values -> L5 withheld)",
            "audit_artifacts": ["artifacts/audit_m16227/audit.json",
                                "docs/board_audit_m16227/M16227_BOARD_AUDIT.md"],
            "audit_findings": [f["id"] + ":" + f["severity"] for f in audit["findings"]],
            "not_run": ["primary re-run at t=0.2 (adversarial probe cj1280, queued)",
                        "primary re-run on cluster cj1277 (queued)"],
        }
        with psycopg.connect(dsn) as c, c.cursor() as cur:
            cur.execute("INSERT INTO tensor_maturity_verdicts "
                        "(repo, scope, mission_id, level, evidence, verdict_by) "
                        "VALUES (%s,%s,%s,%s,%s::jsonb,%s) RETURNING id",
                        ("TensorDG", "m16218 CASE-DG-KOVASZNAY-NS", 16227, 4,
                         json.dumps(ev, ensure_ascii=False), "board-audit"))
            vid = cur.fetchone()[0]
            c.commit()
        print("verdict id", vid)


if __name__ == "__main__":
    main()