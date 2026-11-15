# AUDIT — XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO / Taylor-Green honest-FAIL close-out

Mission **m21276** (repo `TensorDG`, dir `TensorLBM/`; worktree branch `agent/m21276`).
Anchor: `XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO` (acceptance: cross-validation error **< 15 %**).
Predecessor: mission **m18373** ported the FluidX3D single-node multi-GPU host-staged halo scheme
(`src/tensorlbm/host_staged_halo.py`, doc `docs/XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO.md`)
and validated the transport, but **recorded an honest FAIL on its physics arm D3**: the container
solver `HostStagedD3Q19` missed the analytic Taylor–Green decay by 15–85 %, with the measured rate
*not scaling with `nu`*, and the follow-up path was left open ("diff collide+stream of the port
against the repo's collision kernels on an identical state").

This mission closes that open item. It is a verification-debt task: the honest FAIL is **not
overwritten**, it is re-examined with an independent oracle and given a root cause and a disposition.

**Disposition: the honest FAIL is a FALSE NEGATIVE of the cross-validation driver, not a defect of
the solver or of the host-staged transport.** The driver's Taylor–Green initial condition applies
the wavenumber twice (see ④). With the initial condition corrected, the same solver reproduces the
analytic decay to **0.46 % worst case** over 3 grids × 5 viscosities — inside the 3 % arm and far
inside the 15 % acceptance.

---

## ① Case: physics, geometry, parameters

Taylor–Green vortex on a periodic cube, D3Q19 BGK, lattice units.

* Exact solution: `u = u0 (sin kx cos ky, −cos kx sin ky, 0)`, `k = 2π/n`. It is an exact
  Navier–Stokes solution on the torus (`(u·∇)u = ∇p` pointwise), so
  * fundamental-mode amplitude `∝ exp(−2 ν k² t)`,
  * kinetic energy `0.5⟨ρu²⟩ ∝ exp(−4 ν k² t)`.
* Two observables are measured so that the two analytic laws can be checked separately:
  the **modal amplitude** `|û(kx,ky)|` (spectral projection on `(0,1,1)`) and the **integral kinetic
  energy**. The predecessor only used the integral, which mixes in compressible and mean-flow energy.

| quantity | values |
|---|---|
| lattices | D3Q19, BGK, standard store, float64 |
| grids | 16³, 24³, 32³ (periodic) |
| `nu` / `tau = 3nu + 1/2` | 0.005 / 0.515 … 0.10 / 0.80 |
| `u0` / `Ma = u0√3` | 0.05 / 0.0866 |
| steps / fit window | 600 / lattice steps 60–480 (stride 20) |
| device grids (port cross-check) | 1×1×1, 2×1×2, 1×2×2, 2×2×2 |
| acceptance / tight thresholds (declared before the runs) | 15 % / 3 % |

## ② How: independent oracle, not a re-read of the subject code

`TensorLBM/src/tensorlbm/tg_physics_audit.py` (new) re-implements a periodic D3Q19 BGK solver from
scratch in NumPy — single array, **push** streaming, no ghost layers, no torch, no code shared with
`host_staged_halo.py` (which uses per-device slabs, ghost layers and **pull** streaming). Agreement
between the two is therefore evidence about the *operator*, not about shared code.

Chain of evidence, in order of execution:

1. `tests/test_tg_physics_audit.py` (13 tests): initial-condition properties, decay laws, `nu`
   scaling, grid independence, the malformed-IC regression, and the port↔oracle identity.
2. Sweep (compute gate job **cj3306**, node n315, 8 cores): 3 grids × 5 `nu` × {corrected,
   malformed} IC = 30 cases → `outputs/tg_physics_audit/table_tg_decay.csv`.
3. Port cross-check (torch `HostStagedD3Q19`, host-staged transport, decomposed grids).
4. Figures + `outputs/tg_physics_audit/summary.json`.

Reproduce: `TG_AUDIT_PROFILE=full PYTHONPATH=src python benchmarks/tg_physics_audit.py`;
tests: `PYTHONPATH=src python -m pytest tests/test_tg_physics_audit.py -q` (13 passed, 19 s).

## ③ Results (programmatic, zero hand-copied numbers)

**Decay-rate error of the fundamental mode, corrected initial condition** (all values in
`table_tg_decay.csv`):

| grid | worst modal-rate error over `nu` ∈ {0.005…0.10} | worst kinetic-energy error |
|---|---|---|
| 16³ | 0.46 % | 0.46 % |
| 24³ | 0.35 % | 0.27 % |
| 32³ | 0.11 % | 0.11 % |

**Same solver, malformed initial condition (the predecessor's driver)**:

| observable | error range over the 15 cases |
|---|---|
| modal amplitude | 0.08 % – 111.7 % |
| kinetic energy | 60.9 % – 99.4 % |
| mean momentum of the "Taylor–Green" IC | 1.08e-2 (n=16) … 1.99e-2 (n=24) — *never* zero, and conserved exactly by the collision |

**Subject implementation (torch port) vs the oracle**:

| check | result |
|---|---|
| port 1×1×1, corrected IC, `nu`=0.02, KE rate error | 0.10 % (theory 1.2337e-2) |
| port 2×1×2 (host-staged halo), same case | 0.10 % |
| port 2×1×2 vs NumPy oracle after 50 steps (max&#124;Δu&#124;, max&#124;Δρ&#124;) | 7.0e-16, 1.4e-15 |
| port decomposition invariance (1×1×1 vs 2×2×1, 40 steps, corrected IC) | < 1e-12 |
| port with the malformed IC (KE rate error) | 65.8 % — reproduces the recorded FAIL |

So: solver **and** transport are correct; the whole 15–85 % FAIL is carried by the initial condition.

## ④ Root cause (single line defect, quoted from the predecessor driver)

`tools/xval_host_staged_halo/run_xval_host_staged_halo.py:101-110` (m18373) builds the field as

```python
z, y, x = torch.meshgrid(torch.arange(nz)*2*math.pi/nz,
                         torch.arange(ny)*ky, torch.arange(nx)*kx, indexing="ij")
ux = u0 * torch.sin(kx * x) * torch.cos(ky * y)     # kx applied a SECOND time
```

The meshgrid coordinates already carry `k`, so the sampled field is `sin(k² i)` with
`k² = (2π/n)²` — an irrational, incommensurate wavenumber. Consequences, all observed:

1. It is **not** a Taylor–Green vortex and **not** divergence-free.
2. It is **not periodic** and has a **non-zero mean velocity** (1.08e-2 at n=16). Mass and momentum
   are exact invariants of BGK + periodic streaming, so that drift survives forever.
3. The integral kinetic energy is therefore dominated by the conserved mean mode (94 % of the late
   time energy, `fig3_spectrum.png`), whose decay has nothing to do with `2νk²`.
   ⇒ large error **and** flat dependence on `nu` **and** no grid convergence: exactly the three
   symptoms the predecessor reported.
4. Secondary (measurement) defect: the driver hard-codes `k0 = u0**2/4` instead of measuring the
   initial kinetic energy. For the correct field the two agree to machine precision, so this only
   mattered because symptom 2 made them differ; still fixed here (`audit_case` reports both).

The ABC field of the same driver (`abc_velocity`) is built correctly, which is why the transport
arms (D1/D2/D4) of the predecessor were unaffected — consistent with the finding.

**Fix** (one line, both in the driver and in any downstream copy): drop the pre-scaling of the
meshgrid, i.e. `np.arange(nx)` then `sin(kx * x)`; see
`tg_physics_audit.taylor_green_velocity` (`sin(x)` with `x = arange(n)*k`).
The malformed form is kept as a regression fixture
(`taylor_green_velocity_double_scaled`) so the artefact cannot silently return.

## ⑤ Notes: pitfalls and scope

* The integral kinetic energy is **not** a valid observable for a periodic decay benchmark: any
  numerical mean-flow defect hides there and is conserved. Use the **modal** amplitude (this audit)
  — that also makes `nu`-scaling a real test rather than a coincidence.
* Discrete divergence-free check must be spectral (a central-difference check on a staggered-looking
  field can mask an incommensurate wavenumber).
* The port was executed only on CPU "devices" (this node has no CUDA device): the staging path is
  exercised as tensor views, not PCIe. That limitation is inherited unchanged from m18373 and is not
  affected by this fix (the defect is upstream of the transport).
* Scope of the claim: the transport + BGK container solver are validated for the periodic,
  single-phase, athermal D3Q19 case. Solids, free surface and forces remain outside the port.

## ⑥ Error verdict against acceptance

| quantity | reference | worst error | < 3 %? | < 15 %? |
|---|---|---|---|---|
| fundamental-mode decay rate, corrected IC (3 grids × 5 `nu`) | `2νk²` | **0.46 %** | **yes** | yes |
| kinetic-energy decay rate, corrected IC | `4νk²` | **0.46 %** | **yes** | yes |
| host-staged port KE decay rate (1×1×1 and 2×1×2), `nu`=0.02 | `4νk²` | **0.10 %** | **yes** | yes |
| port vs independent oracle field, 50 steps | NumPy oracle | 7.0e-16 (machine precision) | yes | yes |
| the *predecessor's* reported FAIL (malformed IC) | `4νk²` | 60.9–99.4 % (KE), 0.08–111.7 % (modal) | no | no — reproduced faithfully |

**Honest-FAIL ledger entry (kept, per discipline):** D3 of m18373 FAILED as recorded. This audit
shows the failure was caused by the benchmark's initial condition, not by the solver; with the
initial condition corrected the same code passes the 3 % arm. The transport validation of m18373
stands unchanged; nothing in the subject implementation was modified.

**Recommendation for the anchor:** `XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO` can be closed
with acceptance < 15 % (measured 0.46 % / 0.10 %); the residual open work is environmental (no CUDA
device on the fleet node ⇒ pinned-memory and PCIe timing still unmeasured), which is a different
anchor, not this one.

## Artifacts

```
TensorLBM/src/tensorlbm/tg_physics_audit.py            # independent oracle + IC + observables
TensorLBM/tests/test_tg_physics_audit.py               # 13 tests (IC, laws, scaling, port identity)
TensorLBM/benchmarks/tg_physics_audit.py               # sweep/cross-check/figure runner
TensorLBM/outputs/tg_physics_audit/summary.json        # worst cases, thresholds, root cause
TensorLBM/outputs/tg_physics_audit/table_tg_decay.csv  # 30 rows (3 grids x 5 nu x 2 ICs), job cj3306
TensorLBM/outputs/tg_physics_audit/figures/fig1_decay_curves.png
TensorLBM/outputs/tg_physics_audit/figures/fig2_rate_error_vs_nu.png
TensorLBM/outputs/tg_physics_audit/figures/fig3_spectrum.png
```