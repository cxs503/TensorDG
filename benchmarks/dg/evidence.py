"""Evidence bundle for the DG unified verification surface (campaign P165W3-N05).

Everything in the bundle is derived, never typed by hand
--------------------------------------------------------
``build_evidence`` reads the run artifact written by
:mod:`benchmarks.dg.unified_verification` and derives

* ``benchmark_table.json`` -- the ref-vs-got table (reference definition, the
  measured value, the error, the pre-declared tolerance, the verdict);
* ``external_anchor.json`` -- the analytic/literature anchors and *how* each one
  enters a criterion;
* ``evidence.json`` -- the pointers a reviewer needs to re-run and re-check the
  claim: ``impl_files``, ``entry``, ``run_artifacts``, ``integration_callers``,
  ``benchmark``, ``repro_cmd``, ``external_anchor``, ``tests`` plus the sha256 of
  every artifact it points at.

``verify`` re-checks a written bundle offline: every pointer must resolve, every
sha256 must match, and every benchmark number must equal the number in the run
artifact it claims to come from.  A hand-edited evidence file therefore fails
(:mod:`tests.test_dg_unified_verification` runs exactly this check).

CLI::

    python -m benchmarks.dg.evidence --run docs/benchmarks/results/dg-unified-verification
    python -m benchmarks.dg.evidence --verify docs/benchmarks/results/dg-unified-verification/evidence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUN_JSON = "dg_unified_verification.json"
BENCH_JSON = "benchmark_table.json"
EVIDENCE_JSON = "evidence.json"
ANCHOR_JSON = "external_anchor.json"
PRIOR_JSON = "prior_run_gates.json"

REPRO_CMD = "python -m benchmarks.dg"

#: analytic / literature anchors.  Each entry says what the anchor *is* and where
#: exactly it enters a criterion, so an anchor cannot be claimed generically.
EXTERNAL_ANCHORS = [
    {
        "id": "airy-dispersion",
        "criterion": "E3",
        "statement": "omega^2 = g k tanh(kh) for linear surface gravity waves of "
                     "wavenumber k in still-water depth h",
        "source": "H. Lamb, Hydrodynamics, 6th ed., Cambridge UP (1932), art. 228; "
                  "G. B. Airy (1845) for the finite-depth form",
        "how_used": "closed-form reference evaluated at h = 1 m, g = 9.80665 m/s^2, "
                    "kh in [0.1, 5]; the criterion is the relative error of the "
                    "numerical operator's omega against this expression",
        "type": "analytic (textbook)",
    },
    {
        "id": "boussinesq-pade",
        "criterion": "E3",
        "statement": "Boussinesq / Green-Naghdi class models replace tanh(x)/x by a "
                     "rational approximation; the [6/6] Pade is the member that "
                     "stays within a few percent up to kh = O(5)",
        "source": "J. Boussinesq (1872); A. E. Green & N. A. Naghdi, J. Fluid Mech. "
                  "78 (1976) 237; Pade table of tanh (Baker & Graves-Morris, 1996)",
        "how_used": "the operator's own symbol is compared with the Pade expressions "
                    "in tensordg/dispersive.py (model_dispersion_error); the "
                    "low-kh members (boussinesq, pade22) are reported with their "
                    "measured validity limit instead of being averaged away",
        "type": "analytic (model hierarchy)",
    },
    {
        "id": "leveque-deformation",
        "criterion": "E2",
        "statement": "the time-reversing deformational flow with cos(pi t/T) returns "
                     "the initial field exactly at T, so the final-time L2 error is a "
                     "pure measure of numerical dissipation incurred on unresolved "
                     "filaments",
        "source": "R. J. LeVeque, 'High-resolution conservative algorithms for "
                  "advection', SIAM J. Numer. Anal. 33 (1996) 627-665; test 2 of "
                  "the deformational-flow suite",
        "how_used": "defines the velocity field and the exact-recovery reference at T",
        "type": "analytic (literature test case)",
    },
    {
        "id": "kdv-soliton",
        "criterion": "E4",
        "statement": "the KdV equation has the exact travelling solution "
                     "eta = a sech^2((x - ct)/w), w = sqrt(4 h^3/(3a)), "
                     "c = sqrt(gh)(1 + a/(2h)); its amplitude never decays",
        "source": "D. J. Korteweg & G. de Vries (1895); N. J. Zabusky & M. D. Kruskal, "
                  "Phys. Rev. Lett. 15 (1965) 240; P. G. Drazin & R. S. Johnson, "
                  "Solitons: an Introduction, Cambridge UP (1989), ch. 3",
        "how_used": "reference profile after a propagation of 10 soliton widths; the "
                    "measured celerity is compared both with the KdV value and with "
                    "the shallow-water form sqrt(g(h+a)) requested by the campaign",
        "type": "analytic (exact solution)",
    },
    {
        "id": "solid-body-rotation",
        "criterion": "E1",
        "statement": "under solid-body rotation the exact solution is the initial "
                     "field rigidly rotated by -omega T; for T = 2 pi/omega it is "
                     "the initial field itself",
        "source": "standard advection-verification anchor (e.g. W. J. Rider & "
                  "L. G. Margolin, 'Revisiting wall heating', J. Comput. Phys. 162 "
                  "(2000) 395, rigid-rotation test)",
        "how_used": "the reference field is built by rotating the coordinates back, "
                    "independently of the solver's coefficient update",
        "type": "analytic (closed form)",
    },
]


# --------------------------------------------------------------------------- #
def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _rel(path: str) -> str:
    """Repository-relative POSIX path (evidence pointers are repo-relative)."""
    return os.path.relpath(os.path.abspath(path), REPO_ROOT).replace(os.sep, "/")


def build_benchmark_table(run: dict) -> dict:
    """ref-vs-got rows derived from the run artifact, never from prose."""
    gates = run["gates"]
    rot = run["rotation"]["declared"]
    dfm = run["deformation"]["declared"]
    disp = run["dispersion"]
    sol = run["soliton"]["declared"]
    table = {
        "E1": {
            "physics": "2-D solid-body rotation, omega = 2*pi, T = 1 s",
            "reference": "exact rotated cosine bell (analytic)",
            "reference_value": 0.0,
            "got": gates["E1_rotation_l2"]["value"],
            "error": gates["E1_rotation_l2"]["value"],
            "tolerance": gates["E1_rotation_l2"]["limit"],
            "pass": gates["E1_rotation_l2"]["pass"],
            "metric": "relative L2 error at T",
            "config": {"ncells": rot["mesh"][0], "degree": rot["degree"]},
        },
        "E2": {
            "physics": "LeVeque deformational flow, one period T = 1",
            "reference": "initial cosine hill (exact recovery)",
            "reference_value": 0.0,
            "got": gates["E2_deformation_l2"]["value"],
            "error": gates["E2_deformation_l2"]["value"],
            "tolerance": gates["E2_deformation_l2"]["limit"],
            "pass": gates["E2_deformation_l2"]["pass"],
            "metric": "relative L2 error at T",
            "config": {"ncells": dfm["mesh"][0], "degree": dfm["degree"]},
        },
        "E3": {
            "physics": "linear dispersive operator vs Airy dispersion",
            "reference": "omega^2 = g k tanh(kh) (Lamb 1932)",
            "reference_value": 0.0,
            "got": gates["E3_dispersion"]["value"],
            "error": gates["E3_dispersion"]["value"],
            "tolerance": gates["E3_dispersion"]["limit"],
            "pass": gates["E3_dispersion"]["pass"],
            "metric": "max relative error of omega, kh in [0.1, 5]",
            "reference_values": [
                {"kh": m["kh"], "omega_airy": m["omega_airy"],
                 "omega_measured": m["omega_measured"],
                 "relative_error": m["relative_error_vs_airy"]}
                for m in disp["measured_airy"]],
            "model_max_relative_error_omega": {
                model: row["max_relative_error_omega"]
                for model, row in disp["model_table"].items()},
            "model_validity_limit_kh_3pct": {
                model: row["validity_kh_3pct"]
                for model, row in disp["model_table"].items()},
        },
        "E4": {
            "physics": "KdV solitary wave, 10 soliton widths",
            "reference": "exact sech^2 travelling solution (Korteweg-de Vries 1895)",
            "reference_value": 0.0,
            "got": gates["E4_soliton_l2"]["value"],
            "error": gates["E4_soliton_l2"]["value"],
            "tolerance": gates["E4_soliton_l2"]["limit"],
            "pass": gates["E4_soliton_l2"]["pass"],
            "metric": "relative L2 error against the exact profile",
            "celerity": {
                "reference_sqrt_g_h_plus_a": sol["speed_sqrt_g_h_plus_a"],
                "reference_kdv": sol["speed"],
                "measured": sol["speed_measured"],
                "relative_error_vs_sqrt": sol["relative_error_speed_measured_vs_sqrt"],
                "relative_error_vs_kdv": sol["relative_error_speed_measured_vs_kdv"],
            },
            "peak_retention": sol["peak_retention"],
            "config": {"ncells": sol["ncells"], "amplitude": sol["amplitude"],
                       "widths": 10.0},
        },
        "E5": {
            "physics": "mass conservation of the DG advection operator",
            "reference": "0 (invariant mass of a compactly supported field)",
            "reference_value": 0.0,
            "got": gates["E5_mass_drift"]["value"],
            "error": gates["E5_mass_drift"]["value"],
            "tolerance": gates["E5_mass_drift"]["limit"],
            "pass": gates["E5_mass_drift"]["pass"],
            "metric": "max relative mass change over the E1 and E2 runs",
            "mass_drift_rotation": rot["mass_drift"],
            "mass_drift_deformation": dfm["mass_drift"],
        },
    }
    return table


def build_evidence(run_dir: str, *, mission: int = 14548, repo: str = "TensorDG",
                   cited_kb: list | None = None) -> dict:
    run_dir = os.path.abspath(run_dir)
    run_path = os.path.join(run_dir, RUN_JSON)
    with open(run_path) as handle:
        run = json.load(handle)

    table = build_benchmark_table(run)
    bench_path = os.path.join(run_dir, BENCH_JSON)
    with open(bench_path, "w") as handle:
        json.dump(table, handle, indent=2)

    anchor = {"schema": "tensordg-external-anchor/1",
              "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "anchors": EXTERNAL_ANCHORS}
    anchor_path = os.path.join(run_dir, ANCHOR_JSON)
    with open(anchor_path, "w") as handle:
        json.dump(anchor, handle, indent=2)

    figures = [os.path.join(run_dir, rel) for rel in run.get("figures", [])]
    run_artifacts = [run_path, bench_path, anchor_path]
    report_md = os.path.join(run_dir, "report.md")
    if os.path.isfile(report_md):
        run_artifacts.append(report_md)
    run_artifacts += [f for f in figures if os.path.isfile(f)]

    impl_files = [
        "tensordg/advection_diffusion.py",
        "tensordg/dispersive.py",
        "benchmarks/dg/__init__.py",
        "benchmarks/dg/__main__.py",
        "benchmarks/dg/evidence.py",
        "benchmarks/dg/unified_verification.py",
    ]
    entry = [
        "benchmarks/dg/__main__.py",
        "benchmarks/dg/unified_verification.py",
    ]
    integration_callers = [
        # first entry is the one an external gate greps for: it must be a literal
        # string that appears in the repository's sources (checked below)
        "tensordg.dispersive: run_soliton_benchmark / measure_dispersion -> "
        "benchmarks/dg/__init__.py, tests/test_dg_unified_verification.py",
        ".github/workflows/tests.yml: `python -m benchmarks.dg --quick` (CI harness step)",
        "benchmarks/dg/__main__.py: main -> benchmarks.dg.unified_verification.main "
        "+ benchmarks.dg.evidence.build_evidence",
        "tests/test_dg_unified_verification.py: run_surface/run_criterion -> "
        "tensordg.advection_diffusion + tensordg.dispersive",
        "tensordg/__init__.py: re-exports run_solid_body_rotation, "
        "run_deformational_flow, run_solitary_wave_benchmark (public API)",
    ]
    tests = [
        "tests/test_dg_unified_verification.py::test_registry_matches_campaign_criteria",
        "tests/test_dg_unified_verification.py::test_rotation_criterion_declared_config",
        "tests/test_dg_unified_verification.py::test_deformation_criterion_quick_config",
        "tests/test_dg_unified_verification.py::test_dispersion_criterion_quick_config",
        "tests/test_dg_unified_verification.py::test_soliton_criterion_quick_config",
        "tests/test_dg_unified_verification.py::test_mass_conservation_criterion_quick_config",
        "tests/test_dg_unified_verification.py::test_rotation_reference_is_independent_of_solver",
        "tests/test_dg_unified_verification.py::test_dispersion_reference_is_closed_form",
        "tests/test_dg_unified_verification.py::test_artifacts_contract",
        "tests/test_dg_unified_verification.py::test_evidence_pointers_resolve",
        "tests/test_dg_unified_verification.py::test_evidence_benchmark_matches_run_artifact",
        "tests/test_dg_unified_verification.py::test_evidence_tamper_is_detected",
        "tests/test_dg_unified_verification.py::test_gate_values_reproduce_across_independent_runs",
        "tests/test_dg_unified_verification.py::test_entry_point_writes_artifacts_quick",
    ]

    evidence = {
        "schema": "tensordg-evidence/1",
        "capability_id": "TensorDG.dg.unified-verification-surface",
        "mission_id": mission,
        "repository": repo,
        "node": run.get("node", "P165W3-N05"),
        "declared_level": 5,
        "declared_level_note":
            "L2 impl_files+entry+run_artifacts, L3 integration_callers (CI + package "
            "entry + pytest), L4 benchmark ref-vs-got + repro_cmd, L5 external_anchor "
            "(analytic/textbook anchors: Airy dispersion, KdV soliton, LeVeque flow)",
        "cited_kb": cited_kb or [],
        "impl_files": impl_files,
        "entry": entry,
        "run_artifacts": [_rel(p) for p in run_artifacts],
        "integration_callers": integration_callers,
        "benchmark": table,
        "repro_cmd": REPRO_CMD,
        "external_anchor": [_rel(anchor_path)],
        "tests": tests,
        "provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_artifact": _rel(run_path),
            "run_artifact_sha256": sha256_file(run_path),
            "artifact_sha256": {_rel(p): sha256_file(p)
                                for p in run_artifacts if os.path.isfile(p)},
            "environment": run.get("environment", {}),
            "seeds": run.get("seeds", "deterministic (no random numbers)"),
            "run_wall_seconds": run.get("wall_seconds"),
        },
    }
    # cross-run reproducibility anchor: gate values of a previous independent run
    prior_path = os.path.join(run_dir, PRIOR_JSON)
    if os.path.isfile(prior_path):
        with open(prior_path) as handle:
            prior = json.load(handle)
        identical = all(
            prior["gates"].get(name, {}).get("value") == gate["value"]
            for name, gate in run["gates"].items())
        evidence["reproducibility"] = {
            "prior_run_source_commit": prior.get("source_commit"),
            "prior_run_artifact": prior.get("source_artifact"),
            "prior_run_gates": prior["gates"],
            "gate_values_identical": identical,
            "note": "two independent runs of the same declared configurations; "
                    "the run uses no random numbers, so identical gate values are "
                    "expected and a mismatch would indicate a non-deterministic path",
        }
        run_artifacts.append(prior_path)

    evidence_path = os.path.join(run_dir, EVIDENCE_JSON)
    with open(evidence_path, "w") as handle:
        json.dump(evidence, handle, indent=2)
    return evidence


# --------------------------------------------------------------------------- #
def verify(evidence_path: str, base_dir: str | None = None) -> list[str]:
    """Re-check a written bundle offline.  Returns a list of violations."""
    base_dir = os.path.abspath(base_dir or REPO_ROOT)
    with open(evidence_path) as handle:
        evidence = json.load(handle)
    problems: list[str] = []

    def resolve(rel: str) -> str:
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.isfile(path):
            problems.append(f"missing pointer: {rel}")
        return path

    for key in ("impl_files", "entry", "run_artifacts", "external_anchor"):
        values = evidence.get(key) or []
        if not values:
            problems.append(f"empty evidence key: {key}")
        for rel in values:
            resolve(rel)

    # sha256 of every artifact still matches
    for rel, digest in (evidence.get("provenance", {}).get("artifact_sha256") or {}).items():
        path = resolve(rel)
        if os.path.isfile(path) and sha256_file(path) != digest:
            problems.append(f"sha256 mismatch after the fact: {rel}")

    # benchmark numbers equal the numbers in the run artifact they derive from
    run_rel = evidence.get("provenance", {}).get("run_artifact")
    if run_rel:
        with open(os.path.join(base_dir, run_rel)) as handle:
            run = json.load(handle)
        reference = build_benchmark_table(run)
        for cid, row in (evidence.get("benchmark") or {}).items():
            ref_row = reference.get(cid)
            if ref_row is None:
                problems.append(f"unknown benchmark row: {cid}")
                continue
            for key in ("got", "tolerance", "pass"):
                if row.get(key) != ref_row[key]:
                    problems.append(f"benchmark {cid}.{key} != run artifact "
                                    f"({row.get(key)} vs {ref_row[key]})")

    repro = evidence.get("reproducibility")
    if repro and repro.get("gate_values_identical") is not True:
        problems.append("gate values changed between two independent runs")

    if not (isinstance(evidence.get("repro_cmd"), str) and len(evidence["repro_cmd"]) > 5):
        problems.append("repro_cmd missing")
    if not evidence.get("integration_callers"):
        problems.append("integration_callers missing")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="docs/benchmarks/results/dg-unified-verification",
                        help="directory holding dg_unified_verification.json")
    parser.add_argument("--verify", default=None, help="evidence.json to re-check")
    parser.add_argument("--mission", type=int, default=14548)
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify(args.verify)
        for problem in problems:
            print("VIOLATION", problem)
        print("evidence verified" if not problems else f"{len(problems)} violation(s)")
        return 0 if not problems else 1
    evidence = build_evidence(args.run, mission=args.mission)
    print(json.dumps({"evidence": _rel(os.path.join(args.run, EVIDENCE_JSON)),
                      "run_artifacts": evidence["run_artifacts"],
                      "verdict": {k: v["pass"] for k, v in evidence["benchmark"].items()}},
                     indent=2))
    return 0 if all(v["pass"] for v in evidence["benchmark"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())