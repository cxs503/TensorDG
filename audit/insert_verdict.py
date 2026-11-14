#!/usr/bin/env python3
"""Board-audit verdict for m21003 (XVAL-LBM-MASS-EXCHANGE-SELECTOR, TensorDG/TensorLBM).

Independence: m21040 auditor never touched m21003's implementation. Every number
below was recomputed by this audit:

  * pytest rerun on an unmodified copy of the delivered tree
    (audit/artifacts/pytest_rerun.log)
  * rerun of the delivered benchmark entry point, which aborts on MRT
    (audit/artifacts/repro_failure.log)
  * a self-written convergence probe on the delivered BGK configuration
    (audit/probe_poiseuille_convergence.py -> audit/artifacts/poiseuille_convergence.json)
  * a per-scheme parity run on the delivered collision operators
    (audit/artifacts/scheme_parity.txt)
  * an SQL read of tensor_benchmark_cases for this anchor (0 rows)

Level: L2 (独立功能实现). L3 withheld: zero callers of tensorlbm/ outside the
TensorLBM/ directory. L4 refused: no tensor_benchmark_cases row for the anchor,
no run artifacts shipped, repro_cmd exits 1 on the MRT branch, and the delivered
configuration measures 11.0% rel-L2 against its own 3% bar.

Idempotent: one row per (repo, scope, verdict_by) -> ON CONFLICT update.
Run: python3 audit/insert_verdict.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DSN = os.environ.get(
    "TP_PG_DSN",
    "host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
    "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
ROOT = Path(__file__).resolve().parent.parent          # audit worktree m21040
ART = ROOT / "audit" / "artifacts"
SUBJ = Path("/srv/tensor/agent/work/m21003")              # audited worktree (read-only)
REPO = "TensorDG"
SCOPE = "m21003 XVAL-LBM-MASS-EXCHANGE-SELECTOR"
MISSION = 21040
BY = "board-audit"
LEVEL = 2

REF = "audit/refs/TensorLBM_m21003"                    # byte-identical copy in this worktree


def psql(sql: str) -> str:
    env = dict(os.environ, PGPASSWORD="uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
    out = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-p", "5432", "-U", "foundation",
         "-d", "tensorplatform", "-At", "-c", sql],
        capture_output=True, text=True, env=env, check=True)
    return out.stdout.strip()


def evidence() -> dict:
    conv = json.loads((ART / "poiseuille_convergence.json").read_text())
    cps = {c["steps"]: c for c in conv["checkpoints"]}
    ncases = psql("SELECT count(*) FROM tensor_benchmark_cases "
                  "WHERE anchor_slug='XVAL-LBM-MASS-EXCHANGE-SELECTOR'")
    repro_tail = (ART / "repro_failure.log").read_text().strip().splitlines()[-1]
    pytest_tail = (ART / "pytest_rerun.log").read_text().strip().splitlines()[-2:]
    parity = (ART / "scheme_parity.txt").read_text().strip().splitlines()
    return {
        "impl_files": [
            f"{REF}/src/tensorlbm/mass_exchange_selector.py",
            f"{REF}/src/tensorlbm/collision.py",
            f"{REF}/src/tensorlbm/lattice.py",
            f"{REF}/src/tensorlbm/streaming.py",
        ],
        "impl_files_map": {
            f"{REF}/src/tensorlbm/mass_exchange_selector.py":
                ":80 collide, :112 _trt_collision, :146 step, :191 _apply_guo_force "
                "(never called by the shipped benchmark), :223 verify_mass_conservation",
            f"{REF}/src/tensorlbm/collision.py":
                ":7 bgk_collision, :28 mrt_collision, :58-68 M/M_inv (Lallemand-Luo basis, row-verified)",
            f"{REF}/src/tensorlbm/lattice.py":
                ":1-40 e/w/opp, :44 equilibrium, :72 macroscopic",
            f"{REF}/src/tensorlbm/streaming.py": ":7 stream_periodic",
            "subject (read-only, byte-identical, sha256 in audit/artifacts/sha256_subject.txt)":
                "/srv/tensor/agent/work/m21003/TensorLBM/src/tensorlbm/*.py @ head 6000a43 (impl a43d251)",
        },
        "entry": f"{REF}/benchmarks/poiseuille.py",
        "entry_map": {f"{REF}/benchmarks/poiseuille.py":
                      ":28 run_poiseuille, :79 float32 f_init (repro defect), :123 main, "
                      ":140-142 results write (never reached)"},
        "run_artifacts": [
            "audit/artifacts/poiseuille_convergence.json",
            "audit/artifacts/pytest_rerun.log",
            "audit/artifacts/scheme_parity.txt",
            "audit/artifacts/repro_failure.log",
            "audit/artifacts/fig_poiseuille_convergence.png",
        ],
        "run_artifacts_ownership": {
            "produced_by": "m21040 auditor, by running the delivered code unmodified "
                           "(probe_poiseuille_convergence.py / pytest / poiseuille.py)",
            "shipped_by_mission": "NONE",
            "detail": "m21003 shipped no run artifact: benchmarks/results/ does not exist in its "
                      "tree because main() aborts on the MRT branch before the write at "
                      "benchmarks/poiseuille.py:140-142; report_m21003.json has figures: []",
        },
        "tests": [
            f"{REF}/tests/test_mass_exchange_selector.py",
            "audit/artifacts/pytest_rerun.log",
        ],
        "tests_detail": {
            "audit_rerun": pytest_tail,
            "asserts_invariants":
                "yes - mass drift < 1e-12, x/y momentum drift < 1e-10, feq->macroscopic "
                "round trip 1e-12, periodic streaming mass conservation",
        },
        "integration_callers": [],
        "integration_callers_detail": {
            "status": "NONE",
            "detail": "grep -rn 'tensorlbm|MassExchangeSelector|mass_exchange_selector' over the "
                      "TensorDG worktree excluding TensorLBM/ and audit/refs/ -> 0 hits; not "
                      "imported by tensordg/ solver package nor by any platform route",
        },
        "benchmark": {
            "tensor_benchmark_cases_rows": int(ncases),
            "verdict": "no registered case for anchor -> no machine-checkable tolerance (L4 vacuous)",
            "audit_recompute_delivered_config": {
                "config": "BGK/TRT, Nx=64, Ny=32, tau=0.8, F=1e-5, n_steps=2000",
                "bgk_rel_l2": cps[2000]["rel_l2_node_centred"],
                "trt_rel_l2": 0.109452,
                "mrt": "RuntimeError dtype double != float (entry aborts)",
                "mass_drift": 0.0,
            },
            "audit_convergence_probe": {
                "rel_l2_vs_node_centred": {k: cps[k]["rel_l2_node_centred"] for k in sorted(cps)},
                "rel_l2_vs_halfway_bb": {k: cps[k]["rel_l2_halfway_bb"] for k in sorted(cps)},
                "u_max_ratio": {k: cps[k]["u_max_ratio"] for k in sorted(cps)},
                "finding": "the delivered 11% is dominated by under-convergence (2000 of ~9600 "
                           "diffusive steps); converged at 16000 steps the same code gives 2.45% "
                           "on its own node-centred metric but 3.90% on the physically stricter "
                           "half-way bounce-back reference -> '<3%' is metric-sensitive",
                "artifact": "audit/artifacts/poiseuille_convergence.json",
                "script": "audit/probe_poiseuille_convergence.py",
            },
            "scheme_parity": {"raw": parity, "artifact": "audit/artifacts/scheme_parity.txt"},
        },
        "repro_cmd": {
            "cmd": f"cd {REF} && python3 benchmarks/poiseuille.py",
            "result": repro_tail,
            "exit_code": 1,
            "log": "audit/artifacts/repro_failure.log",
            "root_cause": "benchmarks/poiseuille.py:79 builds f_init with torch.ones (float32) "
                          "while collision.py:58 M is float64",
        },
        "external_anchor": {
            "claim": "mass-exchange-selector pattern from Maarten-vd-Sande/lbm",
            "status": "UNVERIFIED",
            "detail": "no upstream commit id, no file-by-file mapping, no numerical parity artifact "
                      "in the delivered tree; upstream API/number comparison not_run",
        },
        "sha256": (ART / "sha256_subject.txt").read_text().strip().splitlines(),
        "audited_head": "6000a43 (impl a43d251)",
        "report_six_sections": {
            "present": True,
            "defect": "section 4 (结果图) is a placeholder, figures: [] despite a plottable case",
        },
        "independence": "auditor did not author m21003; all numbers recomputed in m21040",
        "checks_not_run": [
            "upstream Maarten-vd-Sande/lbm API/numerical parity",
            "Guo forcing _apply_guo_force correctness: defined at mass_exchange_selector.py:191 but no delivered caller passes force=, so that branch is never exercised",
            "long-time stability >1e5 steps, multi-core/GPU consistency, D3Q19/D3Q27",
        ],
        "findings_severity": {
            "F1_blocking": "no tensor_benchmark_cases row for anchor (0 rows)",
            "F2_blocking": "repro_cmd exits 1; run artifacts absent",
            "F3_high": "11% attributed to under-convergence, not to the causes stated in the report",
            "F4_medium": "accuracy verdict flips across wall-position conventions (2.45% vs 3.90%)",
            "F5_medium": "no integration callers -> L3 withheld",
            "F6_medium": "external anchor unestablished",
            "F7_low": "report ships no figure",
            "F8_low": "docs claim three usable schemes while MRT crashes on the default path",
            "F9_low": "package docstring advertises (BGK, MRT, entropic) but Scheme has only BGK/MRT/TRT - entropic unimplemented",
        },
        "report": "docs/AUDIT-m21003-XVAL-LBM-MASS-EXCHANGE-SELECTOR.md",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ev = evidence()
    (ART / "verdict_evidence.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    if a.dry_run:
        print(json.dumps(ev, indent=2, ensure_ascii=False)[:2000])
        return
    import psycopg  # real driver (psycopg2 shim in this sandbox is a stub)
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO tensor_maturity_verdicts
                (repo, scope, mission_id, level, evidence, verdict_by)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (repo, scope, verdict_by) DO UPDATE
               SET mission_id = EXCLUDED.mission_id,
                   level = EXCLUDED.level,
                   evidence = EXCLUDED.evidence,
                   created_at = now()
            RETURNING id, level, gate_state, gate_level
            """,
            (REPO, SCOPE, MISSION, LEVEL, json.dumps(ev, ensure_ascii=False), BY))
        row = cur.fetchone()
        conn.commit()
        print("verdict id=%s level=%s gate_state=%s gate_level=%s" % row)


if __name__ == "__main__":
    main()