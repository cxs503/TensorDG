"""T3 international-standard validation matrix for the LBM domain.

What this module is
-------------------
The T3 line of the validation ledger asks for a *nine-case* standard matrix
(square cavity / cylinder / sphere / turbulent channel / Taylor-Green /
static droplet / Rayleigh-Taylor / dam break / ship-resistance waves), each
with (a) an analytic, experimental or reference-software comparison, (b) a
``repro_cmd``, (c) a **machine-checkable criterion**, and (d) CI
runnability.  The failure mode this module is designed against is the
classic one: a case is *declared* validated once, the number is copied into a
report, and nobody ever re-runs it -- or the tolerance is tuned after the
fact until the case "passes".

So the matrix is a data structure plus a runner:

* :data:`CASES` pins, for every one of the nine cases, the reference
  (citation and the reference numbers themselves), the observable, the
  tolerance **and** the reason for that tolerance.  Tolerances are registered
  here, before any run; :func:`validate_spec` refuses a tolerance looser than
  the ecosystem acceptance limit (15 %) and refuses a ``REAL`` claim for a
  case that has no executable probe.
* :func:`run_matrix` executes every probe that exists, records the measured
  number, the wall time and the verdict, and writes a machine-readable
  summary.  Cases without a probe are reported as ``NOT_SUPPORTED`` with the
  gap that blocks them -- never as "passed".
* :func:`validate_spec` is the CI gate on the *specification* (citations
  present, tolerances within the acceptance limit, every ``REAL`` claim
  backed by a registered probe, every ``repro_cmd`` runnable and pointing at
  the probe it claims).

Evidence levels
---------------
``REAL``           probe ran here, number compared against a pinned reference,
                   verdict computed by the stored criterion.
``NOT_SUPPORTED``  capability absent in this repository; the ``gap`` field
                   says exactly what is missing and what the criterion would
                   be, so the entry cannot silently become "validated" later.
``WITHHELD``       probe exists but could not produce a trustworthy number
                   (e.g. run failed to converge).  Never silently upgraded.
``STUB``/``MOCK``  reserved for probes that are hard-coded; the runner marks
                   a probe MOCK if it returns a value without running the
                   solver (``"solver_steps": 0``).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .cavity import (
    couette_error,
    ghia_error,
    poiseuille_error,
    run_cavity,
    run_couette,
    run_poiseuille,
)
from .lattice import D2Q9
from .collision import bgk_collision
from .streaming import stream_periodic

# The ecosystem acceptance standard: relative error above 15 % is not
# acceptable, therefore no T3 criterion may be registered looser than this.
ACCEPTANCE_LIMIT = 0.15


@dataclass(frozen=True)
class T3Case:
    """One row of the T3 validation matrix."""

    key: str
    zh: str
    en: str
    reference_kind: str          # analytic | literature | reference-software | none
    reference: str               # pinned citation + pinned reference values
    observable: str
    criterion: str               # human-readable verdict rule
    tolerance: float             # registered BEFORE running (relative)
    tolerance_basis: str         # why this number, not a tuned one
    repro_cmd: str
    probe: Optional[str] = None
    gap: Optional[str] = None


CASES: List[T3Case] = [
    T3Case(
        key="cavity-2d-ghia",
        zh="方腔 (lid-driven cavity)",
        en="Lid-driven square cavity, Re=100 and 400",
        reference_kind="literature",
        reference=(
            "Ghia, Ghia & Shin 1982, J. Comput. Phys. 48, 387-411, Tables I/II "
            "(u on the vertical centreline, v on the horizontal centreline)"
        ),
        observable="max |u_centre(Y) - Ghia| / U  and  max |v_centre(X) - Ghia| / U",
        criterion="both L-infinity profile errors <= tolerance at Re=100 and Re=400",
        tolerance=0.05,
        tolerance_basis=(
            "Re=100: BGK at 128 lattice units is a converged second-order case; "
            "published LB results sit at 1-2 %, 5 % leaves head-room only for "
            "the coarse-grid (N=129) discretisation, not for a wrong solver."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only cavity-2d-ghia",
        probe="cavity_ghia",
    ),
    T3Case(
        key="cylinder-2d-re100",
        zh="圆柱绕流 Re=100 (阻力/斯特劳哈尔)",
        en="Flow past a circular cylinder, Re=100 (Cd, St)",
        reference_kind="literature",
        reference=(
            "Cd = 1.32-1.38 and St = 0.164-0.166 for Re=100 "
            "(Park, Kwon & Choi 1998, J. Comput. Phys. 139, 401, and the "
            "benchmark set collected in Schaefer & Turek 1996)"
        ),
        observable="mean drag coefficient Cd and Strouhal number St = f D / U",
        criterion="|Cd - 1.35| / 1.35 <= 0.10 and |St - 0.165| / 0.165 <= 0.10",
        tolerance=0.10,
        tolerance_basis=(
            "literature spread itself is ~4 %; 10 % absorbs staircase-geometry "
            "error at 32 cells/diameter, still far inside the 15 % limit."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only cylinder-2d-re100",
        probe=None,
        gap=(
            "No curved/staircase solid support with a drag (momentum-exchange) "
            "diagnostic and no unsteady time-averaging harness: "
            "HalfwayBounceBack handles flat-axis walls only, and "
            "tensorlbm has no force-integral routine over an immersed body."
        ),
    ),
    T3Case(
        key="sphere-3d",
        zh="球体绕流 (3D)",
        en="Flow past a sphere (3D drag, Re 20-300)",
        reference_kind="literature",
        reference=(
            "Schiller-Naumann / Clift-Grace-Weber Cd-Re curve; "
            "Cd(Re=100) = 1.09, Cd(Re=300) = 0.66"
        ),
        observable="drag coefficient Cd(Re)",
        criterion="|Cd - reference| / reference <= 0.10 at Re=100 and Re=300",
        tolerance=0.10,
        tolerance_basis="correlation uncertainty ~5 %; 10 % for lattice geometry error",
        repro_cmd="python benchmarks/t3_matrix_run.py --only sphere-3d",
        probe=None,
        gap=(
            "The package is D2Q9 only -- there is no D3Q19/D3Q27 lattice, no 3D "
            "streaming and no 3D domain decomposition, so a sphere case cannot "
            "be discretised at all. Needs a 3D lattice module first."
        ),
    ),
    T3Case(
        key="turbulent-channel-re180",
        zh="湍流槽道 Re_tau=180",
        en="Turbulent plane channel, Re_tau = 180 (log law / u+ profile)",
        reference_kind="literature",
        reference=(
            "Kim, Moin & Moser 1987, J. Fluid Mech. 177, 133: u+(y+) log law "
            "1/kappa ln y+ + B with kappa=0.41, B=5.2; u_tau from the wall shear"
        ),
        observable="relative error of u+(y+) against the log law over 30 <= y+ <= 100",
        criterion="max |u+ - (ln(y+)/0.41 + 5.2)| / u+ <= 0.05",
        tolerance=0.05,
        tolerance_basis=(
            "log-law scatter of DNS data in that band is ~3 %; 5 % is the "
            "standard LB-with-subgrid acceptance band."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only turbulent-channel-re180",
        probe=None,
        gap=(
            "No subgrid/LES closure, no wall function and no time-averaging "
            "diagnostics; on top of that a 2D D2Q9 model cannot carry a "
            "turbulent channel at Re_tau=180 (needs 3D + near-wall modelling)."
        ),
    ),
    T3Case(
        key="taylor-green-2d",
        zh="Taylor-Green 涡衰减 (2D 解析)",
        en="Taylor-Green vortex decay, 2D analytic",
        reference_kind="analytic",
        reference=(
            "u = u0 sin(kx) cos(ky), v = -u0 cos(kx) sin(ky) decays as "
            "exp(-2 nu k^2 t) in the Navier-Stokes limit (Taylor 1923); "
            "kinetic energy as exp(-4 nu k^2 t)"
        ),
        observable="measured amplitude decay rate / (2 nu k^2)",
        criterion="relative error of the fitted decay rate <= tolerance",
        tolerance=0.02,
        tolerance_basis=(
            "the discrete BGK green-gauss/streaming error for k dx = 2*pi/64 is "
            "O((k dx)^2) ~ 1 %; 2 % covers it, and a 15 % error would be a real "
            "defect (as in the host-staged TG FAIL, see tg_physics_audit)."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only taylor-green-2d",
        probe="tg_decay_2d",
    ),
    T3Case(
        key="static-droplet-laplace",
        zh="静态液滴 Laplace 定律",
        en="Static droplet: Laplace law dp = sigma / R",
        reference_kind="analytic",
        reference=(
            "Laplace law: dp * R = sigma = const over radii; a straight line "
            "through the origin in dp vs 1/R (Rowlinson & Widom 1982)"
        ),
        observable="max deviation of dp*R from its mean over R = 12,16,20",
        criterion="max |dp*R - mean(dp*R)| / mean(dp*R) <= tolerance",
        tolerance=0.05,
        tolerance_basis=(
            "diffuse-interface lattice width is 3 cells, so dp carries an O(1/R) "
            "curvature-discretisation error of a few %; 5 % is the standard "
            "Shan-Chen acceptance band."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only static-droplet-laplace",
        probe="laplace_droplet",
    ),
    T3Case(
        key="rayleigh-taylor-2d",
        zh="Rayleigh-Taylor 不稳定性 (2D)",
        en="Rayleigh-Taylor instability, 2D (bubble/spike growth)",
        reference_kind="literature",
        reference=(
            "linear growth rate n = sqrt(A g k) with Atwood A = (rho_h-rho_l)/"
            "(rho_h+rho_l) (Chandrasekhar 1961); nonlinear benchmarks in "
            "He, Chen & Zhang 1999, J. Comput. Phys. 152, 642"
        ),
        observable="growth rate of the perturbation amplitude vs sqrt(A g k)",
        criterion="relative error of the fitted linear growth rate <= tolerance",
        tolerance=0.05,
        tolerance_basis="linear-regime fit over 1.5 decades of amplitude, 5 % band",
        repro_cmd="python benchmarks/t3_matrix_run.py --only rayleigh-taylor-2d",
        probe=None,
        gap=(
            "MCMP exists but there is no gravity/buoyancy body force and no "
            "Atwood-ratio initialisation; the interface amplitude diagnostic "
            "needed for the linear growth rate is also missing."
        ),
    ),
    T3Case(
        key="dam-break-ritter",
        zh="溃坝 (Ritter 解析解)",
        en="Dam break against the Ritter (1892) analytic solution",
        reference_kind="analytic",
        reference=(
            "Ritter 1892 parabolic dam-break solution for the shallow-water "
            "equations: h(x,t) = (1/9g)(2 sqrt(g h0) - x/t)^2 on -sqrt(g h0) t "
            "<= x <= 2 sqrt(g h0) t"
        ),
        observable="L2 error of the free-surface height against Ritter",
        criterion="relative L2 error of h(x) <= tolerance at t = 1",
        tolerance=0.05,
        tolerance_basis=(
            "Ritter is an exact solution of the shallow-water limit; a free-"
            "surface LBM at 256 cells resolves the rarefaction to ~2-3 %, so 5 % "
            "is a solver check rather than a modelling check."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only dam-break-ritter",
        probe=None,
        gap=(
            "No free-surface (mass-tracking) LBM; MCMP is a two-component "
            "model with a diffuse interface and cannot reproduce the dry bed. "
            "The Ritter reference itself already exists in the DG case pool "
            "(CASE-DG-RITTER-DAMBREAK-SWE), so only the LBM side is missing."
        ),
    ),
    T3Case(
        key="ship-resistance-waves",
        zh="船体阻力与兴波",
        en="Ship resistance and wave making (hull + free surface)",
        reference_kind="reference-software",
        reference=(
            "KCS/KVLCC2 model-scale resistance and wave-cut data "
            "(ITTC 2014 benchmark; Tokyo 2015 workshop).  In-house reference: "
            "TensorBEM potential-flow added resistance"
        ),
        observable="total resistance coefficient Ct and wave elevation along the hull",
        criterion="|Ct - Ct_ref| / Ct_ref <= 0.10 and wave-elevation L2 <= 0.10",
        tolerance=0.10,
        tolerance_basis=(
            "experimental Ct scatter across facilities is 3-5 %; 10 % is the "
            "usual CFD acceptance for a first-order method."
        ),
        repro_cmd="python benchmarks/t3_matrix_run.py --only ship-resistance-waves",
        probe=None,
        gap=(
            "Requires 3D, free surface, a moving no-slip hull and wave-absorbing "
            "far field.  None of these exist in tensorlbm (D2Q9 periodic + flat "
            "walls only); the ship domain belongs to the TensorBEM/FVM repos."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Probes -- each returns {"measured": float, "pass": bool, "solver_steps": int,
# "detail": {...}}.  A probe must run the solver: the runner flags a probe that
# reports zero solver steps as MOCK.
# ---------------------------------------------------------------------------


def probe_cavity_ghia(n: int = 129, u_lid: float = 0.1,
                      tolerance: float = 0.05,
                      n_steps: int = 80000) -> dict:
    """Lid-driven cavity vs Ghia et al. (1982) at Re = 100 and 400."""
    tol = tolerance
    out = {}
    worst = 0.0
    steps = 0
    for re in (100, 400):
        t0 = time.time()
        res = run_cavity(N=n, U=u_lid, re=re, n_steps=n_steps, tol=1e-7,
                         min_steps=2000, check_every=500)
        err = ghia_error(res, re)
        Yc, uc = res.centreline_u()
        Xc, vc = res.centreline_v()
        err["profile"] = {
            "Y": [float(v) for v in Yc],
            "u_over_U": [float(v) for v in uc],
            "X": [float(v) for v in Xc],
            "v_over_U": [float(v) for v in vc],
        }
        err["wall_seconds"] = time.time() - t0
        err["tau"] = res.tau
        err["u_lid"] = u_lid
        out[f"re{re}"] = err
        worst = max(worst, err["u_linf"], err["v_linf"])
        steps += res.steps
    return {
        "measured": worst,
        "pass": bool(worst <= tol) and all(
            out[k]["converged"] for k in out
        ),
        "solver_steps": steps,
        "detail": out,
    }


def probe_tg_decay_2d(n: int = 64, u0: float = 0.01, nu: float = 0.05,
                      n_steps: int = 2000, tolerance: float = 0.02) -> dict:
    """2D Taylor-Green vortex: fitted amplitude decay rate vs the analytic 2 nu k^2."""
    tol = tolerance
    k = 2.0 * math.pi / n
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    y, x = torch.meshgrid(
        torch.arange(n, dtype=torch.float64),
        torch.arange(n, dtype=torch.float64),
        indexing="ij",
    )
    ux = u0 * torch.sin(k * x) * torch.cos(k * y)
    uy = -u0 * torch.cos(k * x) * torch.sin(k * y)
    rho = torch.ones((n, n), dtype=torch.float64)
    f = D2Q9.equilibrium(rho, ux, uy)

    times, amps = [], []
    for step in range(n_steps + 1):
        rho, ux, uy = D2Q9.macroscopic(f)
        if step % 50 == 0:
            # modal amplitude of the fundamental mode of ux
            amp = float((ux * torch.sin(k * x) * torch.cos(k * y)).mean() * 4.0)
            times.append(step)
            amps.append(abs(amp))
        f = stream_periodic(bgk_collision(f, rho, ux, uy, omega))

    t = torch.tensor(times[2:], dtype=torch.float64)
    a = torch.log(torch.tensor(amps[2:], dtype=torch.float64))
    slope = float(-(len(t) * (t * a).sum() - t.sum() * a.sum())
                  / (len(t) * (t * t).sum() - t.sum() ** 2))
    analytic = 2.0 * nu * k * k
    rel = abs(slope - analytic) / analytic
    return {
        "measured": rel,
        "pass": bool(rel <= tol),
        "solver_steps": n_steps,
        "detail": {
            "n": n, "nu": nu, "tau": tau, "k": k, "u0": u0,
            "decay_rate_fit": slope,
            "decay_rate_analytic": analytic,
            "amplitude_first": amps[0], "amplitude_last": amps[-1],
            "kinetic_energy_analytic_rate": 4.0 * nu * k * k,
            "samples": {"t": times, "amp": amps},
        },
    }


def probe_laplace_droplet(n: int = 96, radii=(12.0, 16.0, 20.0),
                          n_equilibrate: int = 3000,
                          tolerance: float = 0.05) -> dict:
    """Static droplet: is dp * R constant (Laplace law) over several radii?

    Note: ``mcmp.measure_laplace_pressure`` compares dp against an *estimated*
    sigma (``-G cs^2 (d rho)^2 / 6``), which is itself an approximation, so it
    cannot serve as an independent reference.  The Laplace law proper is the
    ``1/R`` scaling, which is what is tested here.
    """
    from .mcmp import setup_laplace_test
    from .cross_force import pseudopotential_shan_chen

    tol = tolerance
    cs2 = float(D2Q9.cs2)
    measured_dp = {}
    steps = 0
    for R in radii:
        solver = setup_laplace_test(nx=n, ny=n, R=R)
        solver.run(n_equilibrate)
        steps += n_equilibrate
        solver._compute_macroscopic()
        yy, xx = torch.meshgrid(
            torch.arange(solver.ny, dtype=solver.dtype),
            torch.arange(solver.nx, dtype=solver.dtype),
            indexing="ij",
        )
        r = torch.sqrt((xx - solver.nx / 2) ** 2 + (yy - solver.ny / 2) ** 2)
        inside = r < (R - 5)
        outside = r > (R + 5)
        G = float(solver.cross_force.G_matrix[0, 1])

        def pressure(mask):
            r0 = float(solver.rho[0][mask].mean())
            r1 = float(solver.rho[1][mask].mean())
            return cs2 * (r0 + r1 + G * float(pseudopotential_shan_chen(
                torch.tensor(r0))) * float(pseudopotential_shan_chen(
                torch.tensor(r1))))

        measured_dp[R] = abs(pressure(inside) - pressure(outside))

    products = {R: measured_dp[R] * R for R in radii}
    mean = sum(products.values()) / len(products)
    worst = max(abs(v - mean) / mean for v in products.values())
    return {
        "measured": worst,
        "pass": bool(worst <= tol),
        "solver_steps": steps,
        "detail": {
            "n": n, "n_equilibrate": n_equilibrate,
            "dp": measured_dp,
            "dp_times_R": products,
            "radii": list(radii),
            "sigma_implied": mean,
            "linearity_rel_deviation": worst,
        },
    }


PROBES: Dict[str, Callable[..., dict]] = {
    "cavity_ghia": probe_cavity_ghia,
    "tg_decay_2d": probe_tg_decay_2d,
    "laplace_droplet": probe_laplace_droplet,
}


# ---------------------------------------------------------------------------
# Spec validation (CI gate on the matrix itself, not on the physics)
# ---------------------------------------------------------------------------


def validate_spec(cases: Optional[List[T3Case]] = None) -> List[str]:
    """Return a list of specification violations (empty list == valid)."""
    cases = CASES if cases is None else cases
    problems: List[str] = []
    keys = [c.key for c in cases]
    if len(set(keys)) != len(keys):
        problems.append("duplicate case keys")
    if len(cases) != 9:
        problems.append(f"T3 matrix must have 9 cases, found {len(cases)}")
    for c in cases:
        where = f"[{c.key}]"
        if c.reference_kind == "none" or not c.reference.strip():
            problems.append(f"{where} no reference pinned")
        if not (0.0 < c.tolerance <= ACCEPTANCE_LIMIT):
            problems.append(
                f"{where} tolerance {c.tolerance} outside (0, {ACCEPTANCE_LIMIT}]"
            )
        if not c.tolerance_basis.strip():
            problems.append(f"{where} tolerance has no registered basis")
        if not c.repro_cmd.startswith("python ") or c.key not in c.repro_cmd:
            problems.append(
                f"{where} repro_cmd must run from the repo root and name the case"
            )
        if c.probe is not None and c.probe not in PROBES:
            problems.append(f"{where} probe '{c.probe}' not registered")
        if c.probe is None and not (c.gap or "").strip():
            problems.append(f"{where} no probe and no documented gap")
        if c.probe is not None and (c.gap or "").strip():
            problems.append(f"{where} has both a probe and a gap")
    return problems


def run_matrix(only: Optional[List[str]] = None,
               probes: Optional[Dict[str, Callable[..., dict]]] = None,
               probe_kwargs: Optional[Dict[str, dict]] = None,
               cases: Optional[List[T3Case]] = None) -> dict:
    """Execute every probe that exists; return a JSON-serialisable summary."""
    cases = CASES if cases is None else cases
    probes = PROBES if probes is None else probes
    probe_kwargs = probe_kwargs or {}
    spec_problems = validate_spec(cases)

    rows = []
    for c in cases:
        if only and c.key not in only:
            continue
        row = {
            "key": c.key,
            "zh": c.zh,
            "en": c.en,
            "reference_kind": c.reference_kind,
            "reference": c.reference,
            "observable": c.observable,
            "criterion": c.criterion,
            "tolerance": c.tolerance,
            "repro_cmd": c.repro_cmd,
        }
        if c.probe is None:
            row.update(status="NOT_SUPPORTED", evidence="NOT_SUPPORTED",
                       measured=None, pass_=None, gap=c.gap, wall_seconds=None)
        else:
            fn = probes[c.probe]
            t0 = time.time()
            try:
                res = fn(tolerance=c.tolerance, **probe_kwargs.get(c.key, {}))
                wall = time.time() - t0
                steps = int(res.get("solver_steps", 0))
                evidence = "REAL" if steps > 0 else "MOCK"
                verdict = bool(res.get("pass", False)) and evidence == "REAL"
                row.update(
                    status="PASS" if verdict else "FAIL",
                    evidence=evidence,
                    measured=res.get("measured"),
                    pass_=verdict,
                    gap=None,
                    wall_seconds=wall,
                    solver_steps=steps,
                    detail=res.get("detail", {}),
                )
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                row.update(status="ERROR", evidence="WITHHELD", measured=None,
                           pass_=None, gap=f"probe raised: {exc!r}",
                           wall_seconds=time.time() - t0)
        rows.append(row)

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] in ("FAIL", "ERROR"))
    n_unsupported = sum(1 for r in rows if r["status"] == "NOT_SUPPORTED")
    return {
        "matrix": "T3",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "acceptance_limit": ACCEPTANCE_LIMIT,
        "spec_problems": spec_problems,
        "spec_valid": not spec_problems,
        "rows": rows,
        "summary": {
            "n_cases": len(rows),
            "n_real_pass": n_pass,
            "n_real_fail": n_fail,
            "n_not_supported": n_unsupported,
            "coverage_real": n_pass / len(rows) if rows else 0.0,
        },
    }


def to_markdown(summary: dict) -> str:
    """Render a matrix summary as a markdown table."""
    lines = [
        "| case | 案例 | evidence | measured | tolerance | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary["rows"]:
        measured = "-" if r["measured"] is None else f"{r['measured']:.4g}"
        verdict = {None: "n/a"}.get(r["pass_"], "PASS" if r["pass_"] else "FAIL")
        lines.append(
            f"| `{r['key']}` | {r['zh']} | {r['evidence']} | {measured} | "
            f"{r['tolerance']:.3f} | {verdict} |"
        )
    s = summary["summary"]
    lines.append("")
    lines.append(
        f"REAL & passing: {s['n_real_pass']}/{s['n_cases']} "
        f"(coverage {s['coverage_real']:.0%}); "
        f"honest NOT_SUPPORTED: {s['n_not_supported']}"
    )
    return "\n".join(lines)


def save_summary(summary: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)