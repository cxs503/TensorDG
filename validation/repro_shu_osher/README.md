# CASE-SHU-OSHER-DG — evidence bundle (TensorDG)

Shu & Osher (1988) shock / entropy-wave interaction: Mach-3 shock at `x=-4` in a
`gamma=1.4` gas sweeps a sinusoidal density wave `rho = 1 + 0.2 sin(5x)`, `t=1.8`.
Acceptance line: relative L1 of the density profile vs the reference `< 5%`.

## Layout

| path | what it is |
|---|---|
| `evidence.json` | machine-checkable evidence manifest (impl_files / entry / run_artifacts / integration_callers / benchmark / repro_cmd / tests / external_anchor) |
| `benchmark_ref_vs_got.json` / `.md` | ref vs got table, generated from `runs/shu_osher_case_pool/metrics.json` (no hand-copied number) |
| `evidence.md` | evidence chain + sha256 of every artifact |
| `artifacts/` | curated copy of the run outputs + `repro_smoke.json` + `platform_bench.json` |
| `smoke_repro.py` | ~25 s end-to-end re-run of one configuration vs the tabulated number |
| `platform_bench.py` | posts/reads the case on the platform bench API (`/tensor/bench/*`) |
| `make_evidence.py` | rebuilds this bundle from `runs/` |
| `run_all.sh` | one command: reference → ladder → merge → smoke → bundle |
| `bundle.py` | resolves the artifact directory and re-hashes the bundle (`--verify`) |

## Reproduce

```bash
bash validation/repro_shu_osher/run_all.sh          # full study, ~30 min, 8 procs
python validation/repro_shu_osher/smoke_repro.py    # one config, ~25 s
python -m tensordg.cases.shock_entropy --ncells 400 --degree 2 \
       --reference benchmarks/data/shu_osher_reference.npz   # library entry
python validation/repro_shu_osher/bundle.py --verify          # re-hash the bundle
```

The full study writes to `runs/shu_osher_case_pool/`, which is git-ignored and
therefore **absent on a clean checkout**. Every script in this directory asks
`bundle.resolve_run_dir()` for its input, so it reads the fresh study when it
exists and otherwise the tracked copy in `artifacts/`; the JSON they emit records
which one was used (`artifact_source`: `fresh` / `curated` / `env`). Point
`TENSORDG_RUN_DIR=<dir>` at another study to override. `--dry-run` on
`smoke_repro.py` / `platform_bench.py` resolves the artifacts and prints the plan
without solving or calling the API.

`artifacts.sha256.json` keys are repo-relative (`validation/...`,
`docs/figures/...`), so `bundle.py --verify` re-hashes all 19 entries from the
repository root and exits non-zero on any drift.

Threads are pinned to one per process inside `run_all.sh`: the DG kernel is small and
letting BLAS/torch spawn a thread per core on a 128-core host is 5–10x slower.

## Result

9 configurations (DG P1/P2/P3 × N=200/400/800, WENO limiter, CFL 0.4) against an
independent FV-WENO5/HLLC/SSP-RK3 reference at N=3200 (own grid convergence 0.101% L1):
relative L1 0.556%–2.462%, **all below the 5% gate**; best DG P2 N=800 at 0.556%
(also below 3%). Global mass closes to 1e-11 relative; the post-shock plateau matches
the Rankine–Hugoniot left state to 4.4e-11%; no negative density in any configuration.

## Caveat

The 1988/1989 papers present this case graphically and tabulate no reference values,
and no PDF is reachable from the execution environment: literature values are
**WITHHELD**, not transcribed. The numeric reference is an independent discretisation
plus closed-form anchors (see `evidence.json` → `external_anchor_note`).
