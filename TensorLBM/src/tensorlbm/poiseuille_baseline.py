"""Single-phase plane-Poiseuille validation baseline skeleton.

Anchor: ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO``

Cross-validation target
-----------------------
``lettucecfd/lettuce`` — ``lettuce/ext/_flows/poiseuille.py:30`` (force-driven
plane channel, streamwise periodic, analytic parabola as reference) plus
``lettuce/ext/_reporter/error_reporter.py:11`` (a *reusable error reporter* with
a pre-registered tolerance gate, not a one-off print).  The pattern worth
copying is the combination:

1. a **canonical force-driven single-phase case** that any D2Q9 kernel can run,
2. an **analytic reference expressed in the solver's own wall convention**,
3. an **error reporter** that returns machine-readable numbers against a
   tolerance fixed *before* the run,
4. a **steady-state criterion** (residual), so the reported error is a
   converged-flow number rather than a snapshot.

Why a skeleton and not just another benchmark
---------------------------------------------
The delivered ``benchmarks/poiseuille.py`` in this repo reports ~11 % relative
L2 error against its analytic parabola at its delivered configuration.  Two
independent defects produce that number, and both are invisible unless the
reference and the forcing are separated from the run:

* **forcing order** — the delivered body force is added as
  ``f_i += w_i * 3 * e_ix * F`` (Schmolz/naive term).  This is only first order
  in ``F``; it does not reproduce the discrete steady state exactly, and the
  momentum it injects is not the ``F/2`` half-step that the collision has to
  see.  Guo, Zheng & Shi (1993) give the second-order consistent term
  ``F_i = w_i (1 - w/2) [3 (e_i·F) + 9 (e_i·u)(e_i·F) - 3 (u·F)]`` with the
  momentum definition ``rho u = sum_i e_i f_i + F/2``.
* **wall convention** — half-way bounce-back puts the *no-slip surface halfway
  between the last fluid node and the solid node*.  With solid nodes at
  ``y = 0`` and ``y = Ny-1`` that surface is at ``y = 0.5`` and ``y = Ny-1.5``,
  so the channel width is ``Ny-2`` lattice units and the first fluid node sits
  ``0.5`` from the wall.  The delivered reference parabola is written for walls
  at ``y = 0`` and ``y = Ny-1`` (width ``Ny-1``), i.e. half a cell off on each
  side; that alone is a boundary-layer-sized error no number of time steps can
  remove.

Both are quantified by :func:`run` in this module, and both are pinned by
``tests/test_poiseuille_baseline.py``.

Contents
--------
``PoiseuilleConfig``      frozen run specification (case + tolerance gate input)
``analytic_profile``      the parabola in a *declared* wall convention
``guo_forcing``           Guo et al. (1993) forcing terms
``run``                   one baseline run -> machine-readable evidence dict
``ReL2Gate``              pre-registered tolerance gate (refuses > 15 %)
``BASELINE_SPEC``         the pre-registered baseline: case, reference, gate

All quantities are in lattice units.  Nothing here is hard-coded to a flow
case outside this module's declared scope (single-phase plane Poiseuille).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .boundaries import HalfwayBounceBack
from .lattice import D2Q9

#: Hard ceiling on any acceptance tolerance (ecosystem rule: a tolerance above
#: 15 % cannot be a gate, it is an admission that the model is wrong).
MAX_TOLERANCE = 0.15


@dataclass(frozen=True)
class PoiseuilleConfig:
    """Frozen specification of one plane-Poiseuille baseline run.

    Parameters
    ----------
    Nx, Ny : int
        Grid size.  ``x`` is the streamwise (periodic) direction, ``y`` the
        wall-normal direction.  Nodes ``y = 0`` and ``y = Ny-1`` are solid.
    tau : float
        BGK relaxation time; the kinematic viscosity is ``cs2 (tau - 1/2)``.
    force : float
        Constant body force per unit mass, in the ``+x`` direction.
    forcing : {"guo", "simple"}
        ``"guo"`` = Guo et al. (1993) second-order consistent forcing and the
        ``F/2`` momentum definition.  ``"simple"`` = the first-order
        ``w_i 3 e_ix F`` term used by the legacy benchmark (kept only so the
        defect can be measured, never for a passing run).
    wall_origin : {0.5, 0.0}
        Wall position convention for the analytic reference.
        ``0.5`` = half-way bounce-back (walls at ``y=0.5`` and ``y=Ny-1.5``,
        width ``Ny-2``).  ``0.0`` = legacy node-centred parabola (walls at
        ``y=0`` and ``y=Ny-1``, width ``Ny-1``).
    residual_tol : float
        Steady-state criterion: stop when the max velocity change over
        ``check_every`` steps, normalised by ``u_max``, drops below this.
    max_steps, check_every : int
        Iteration budget and residual sampling interval.
    """

    Nx: int = 8
    Ny: int = 32
    tau: float = 0.8
    force: float = 1e-5
    forcing: str = "guo"
    wall_origin: float = 0.5
    residual_tol: float = 1e-12
    max_steps: int = 400_000
    check_every: int = 100
    init: str = "rest"
    """Initial condition: ``"rest"`` (u = 0, honest cold start) or
    ``"analytic"`` (start from the reference parabola — the steady state is
    unique, so this only shortens the transient; it is used for the grid
    ladder where the cold-start cost would dominate)."""

    # -- derived quantities -------------------------------------------------
    @property
    def omega(self) -> float:
        return 1.0 / self.tau

    @property
    def nu(self) -> float:
        return float(D2Q9.cs2) * (self.tau - 0.5)

    @property
    def width(self) -> float:
        """Channel width in lattice units implied by ``wall_origin``."""
        if self.wall_origin == 0.5:
            return float(self.Ny - 2)
        if self.wall_origin == 0.0:
            return float(self.Ny - 1)
        raise ValueError(f"unsupported wall_origin {self.wall_origin!r}")

    @property
    def u_max_analytic(self) -> float:
        return self.force * self.width ** 2 / (8.0 * self.nu)

    def __post_init__(self):
        if self.forcing not in ("guo", "simple"):
            raise ValueError(f"unknown forcing {self.forcing!r}")
        if self.wall_origin not in (0.0, 0.5):
            raise ValueError(f"unsupported wall_origin {self.wall_origin!r}")
        if self.init not in ("rest", "analytic"):
            raise ValueError(f"unknown init {self.init!r}")
        if self.tau <= 0.5:
            raise ValueError("tau must exceed 0.5")
        if self.Ny < 4:
            raise ValueError("Ny must be >= 4 (at least two fluid rows)")
        if self.Nx < 1:
            raise ValueError("Nx must be >= 1")


def solid_mask(Ny: int, Nx: int) -> torch.Tensor:
    """Solid mask for a plane channel: rows 0 and Ny-1 are walls."""
    solid = torch.zeros((Ny, Nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    return solid


def analytic_profile(cfg: PoiseuilleConfig) -> torch.Tensor:
    """Analytic parabola at the fluid-node ``y`` positions, same convention.

    With ``wall_origin = 0.5`` the bottom wall is at ``y = 0.5 - 1 = 0`` in
    *wall-distance* coordinates: node ``i`` sits ``i - 0.5`` from the wall, and
    the profile vanishes at the two wall positions ``y = 0.5`` and
    ``y = Ny - 1.5`` — exactly where half-way bounce-back puts the no-slip
    surface.  The returned tensor is full-grid sized (``Ny`` rows); solid rows
    carry the value of the parabola extrapolated to them and are excluded from
    every error norm.
    """
    y = torch.arange(cfg.Ny, dtype=torch.float64)
    if cfg.wall_origin == 0.5:
        # distance from bottom wall for a node at index y is (y - 0.5) + 0.5 = y
        # (bottom wall at index 0.5), so wall-to-wall span is Ny - 2.
        s = y - 0.5
    else:
        s = y
    L = cfg.width
    return (cfg.force / (2.0 * cfg.nu)) * s * (L - s)


def guo_forcing(u_x, u_y, force: float, omega: float) -> torch.Tensor:
    """Guo, Zheng & Shi (1993) D2Q9 forcing source terms.

    ``F_i = w_i (1 - w/2) [ 3 (e_i·F) + 9 (e_i·u)(e_i·F) - 3 (u·F) ]`` with
    ``F = (force, 0)``.  Requires the momentum definition
    ``rho u = sum_i e_i f_i + F/2`` (see :func:`macroscopic_with_force`).
    """
    Q = D2Q9.Q
    w = D2Q9.w
    e = D2Q9.e
    pref = 1.0 - 0.5 * omega
    ex = e[:, 0].view(Q, 1, 1)
    ey = e[:, 1].view(Q, 1, 1)
    wv = w.view(Q, 1, 1)
    eu = ex * u_x + ey * u_y
    eF = ex * force                      # e_i · F with F = (force, 0)
    uF = u_x * force                     # u · F
    return pref * wv * (3.0 * eF + 9.0 * eu * eF - 3.0 * uF)


def simple_forcing(force: float) -> torch.Tensor:
    """First-order ``w_i 3 e_ix F`` term (legacy benchmark form) as (Q,1,1)."""
    Q = D2Q9.Q
    return (D2Q9.w * 3.0 * D2Q9.e[:, 0]).view(Q, 1, 1) * force


def macroscopic_with_force(f, force: float, rho=None):
    """Moments with the ``F/2`` half-step: ``rho u = sum e_i f_i + F/2``."""
    rho = f.sum(dim=0)
    jx = (D2Q9.e[:, 0].view(-1, 1, 1) * f).sum(dim=0)
    jy = (D2Q9.e[:, 1].view(-1, 1, 1) * f).sum(dim=0)
    u_x = (jx + 0.5 * force) / rho
    u_y = jy / rho
    return rho, u_x, u_y


def _raw_moments(f):
    rho = f.sum(dim=0)
    u_x = (D2Q9.e[:, 0].view(-1, 1, 1) * f).sum(dim=0) / rho
    u_y = (D2Q9.e[:, 1].view(-1, 1, 1) * f).sum(dim=0) / rho
    return rho, u_x, u_y


def _relative_errors(profile: torch.Tensor, reference: torch.Tensor,
                     fluid: torch.Tensor, u_max: float) -> Dict[str, float]:
    """Rel-L2 and rel-Linf of the profile on fluid nodes, normalised by u_max.

    ``rel_l2`` / ``rel_linf`` are normalised by the *exact* analytic peak
    ``u_max``; ``rel_l2_shape`` removes the uniform slip first, and
    ``rel_u_max`` compares the peak value node-for-node (see below).
    """
    p = profile[fluid]
    r = reference[fluid]
    diff = p - r
    l2 = float(torch.sqrt(torch.mean(diff ** 2)))
    linf = float(torch.max(torch.abs(diff)))
    # A constant offset is a *global momentum* (wall-slip) effect, not a shape
    # error: report it separately so the profile shape can be judged on its own.
    offset = float(torch.mean(diff))
    shape = diff - offset
    return {
        "rel_l2": l2 / u_max,
        "rel_linf": linf / u_max,
        "offset": offset,
        "rel_l2_shape": float(torch.sqrt(torch.mean(shape ** 2))) / u_max,
        "rel_linf_shape": float(torch.max(torch.abs(shape))) / u_max,
        "abs_l2": l2,
        "abs_linf": linf,
        # peak comparison is made *on the same nodes*: the exact peak of the
        # parabola falls between nodes for odd/even widths, and that sampling
        # artefact is not a discretisation error of the solver.
        "u_max_numeric": float(torch.max(profile[fluid])),
        "u_max_reference": float(torch.max(reference[fluid])),
        "u_max_analytic": u_max,
        "rel_u_max": float(torch.max(profile[fluid]))
        / float(torch.max(reference[fluid])) - 1.0,
    }


def run(cfg: PoiseuilleConfig) -> Dict:
    """Run one plane-Poiseuille baseline and return machine-readable evidence.

    The returned dict contains the configuration actually used, the converged
    residual, the iteration count, the full ``y`` profile and the analytic
    reference, plus relative L2 / Linf errors normalised by the analytic peak
    velocity.  Nothing is hard-coded: every number comes out of the run.
    """
    Q = D2Q9.Q
    Ny, Nx = cfg.Ny, cfg.Nx
    omega = cfg.omega
    u_max = cfg.u_max_analytic

    rho0 = 1.0
    rho = torch.full((Ny, Nx), rho0, dtype=torch.float64)
    zero = torch.zeros((Ny, Nx), dtype=torch.float64)
    if cfg.init == "rest":
        f = D2Q9.equilibrium(rho, zero, zero)
    else:
        u0 = analytic_profile(cfg).view(Ny, 1).expand(Ny, Nx).contiguous()
        f = D2Q9.equilibrium(rho, u0, zero)

    walls = HalfwayBounceBack(solid_mask(Ny, Nx))
    fluid = ~solid_mask(Ny, Nx)

    if cfg.forcing == "guo":
        fsrc = None  # depends on u, computed per step
    else:
        fsrc = simple_forcing(cfg.force)

    u_prev = None
    residual = float("inf")
    steps = 0
    residual_history: List[List[float]] = []
    mass0 = float(rho.sum())

    while steps < cfg.max_steps:
        for _ in range(cfg.check_every):
            if cfg.forcing == "guo":
                rho, u_x, u_y = macroscopic_with_force(f, cfg.force)
            else:
                rho, u_x, u_y = _raw_moments(f)
            feq = D2Q9.equilibrium(rho, u_x, u_y)
            f_post = f - omega * (f - feq)
            if cfg.forcing == "guo":
                f_post = f_post + guo_forcing(u_x, u_y, cfg.force, omega)
            else:
                f_post = f_post + fsrc
            f = walls.stream(f_post)
        steps += cfg.check_every

        if cfg.forcing == "guo":
            rho, u_x, u_y = macroscopic_with_force(f, cfg.force)
        else:
            rho, u_x, u_y = _raw_moments(f)
        if u_prev is not None:
            residual = float(torch.max(torch.abs(u_x - u_prev))) / u_max
            residual_history.append([steps, residual])
            if residual < cfg.residual_tol:
                u_prev = u_x
                break
        u_prev = u_x

    if cfg.forcing == "guo":
        rho, u_x, u_y = macroscopic_with_force(f, cfg.force)
    else:
        rho, u_x, u_y = _raw_moments(f)

    mid = Nx // 2
    profile = u_x[:, mid]
    reference = analytic_profile(cfg)
    fluid_y = fluid[:, 0]
    errs = _relative_errors(profile, reference, fluid_y, u_max)
    errs["offset_over_force"] = errs["offset"] / cfg.force

    return {
        "case": {
            "Nx": Nx,
            "Ny": Ny,
            "tau": cfg.tau,
            "omega": omega,
            "nu": cfg.nu,
            "force": cfg.force,
            "forcing": cfg.forcing,
            "wall_origin": cfg.wall_origin,
            "channel_width": cfg.width,
            "init": cfg.init,
            "u_max_analytic": u_max,
            "reynolds": u_max * cfg.width / cfg.nu,
        },
        "convergence": {
            "steps": steps,
            "residual_tol": cfg.residual_tol,
            "final_residual": residual,
            "converged": bool(residual < cfg.residual_tol),
            "residual_history": residual_history[-20:],
        },
        "errors": errs,
        "mass_drift": float(rho.sum()) - mass0,
        "profile_y": torch.arange(Ny, dtype=torch.float64).tolist(),
        "profile_u": profile.tolist(),
        "reference_u": reference.tolist(),
        "fluid": fluid[:, 0].tolist(),
    }


@dataclass(frozen=True)
class GateResult:
    """Outcome of a pre-registered tolerance gate."""

    quantity: str
    value: float
    tolerance: float
    basis: str
    passed: bool

    def as_dict(self) -> Dict:
        return {
            "quantity": self.quantity,
            "value": self.value,
            "tolerance": self.tolerance,
            "basis": self.basis,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ReL2Gate:
    """Pre-registered analytic-profile gate (the lettuce ``error_reporter`` idea).

    The tolerance is an *input to the run*, never a post-hoc choice: the gate
    refuses to be constructed with a tolerance above 15 % (ecosystem rule) or
    at or below zero, and it refuses to accept a value that is ``None`` /
    ``NaN`` (a run that did not produce a number is ``WITHHELD``, not a pass).
    """

    tolerance: float = 0.01
    quantity: str = "rel_l2_profile_vs_analytic"
    basis: str = "L2, wall-normal profile on fluid nodes, normalised by u_max_analytic"

    def __post_init__(self):
        if not (0.0 < self.tolerance <= MAX_TOLERANCE):
            raise ValueError(
                f"tolerance {self.tolerance!r} outside (0, {MAX_TOLERANCE}]: "
                "a gate must be pre-registered and cannot exceed 15 %"
            )

    def check(self, value: Optional[float]) -> GateResult:
        if value is None or not torch.isfinite(torch.tensor(float(value))):
            return GateResult(self.quantity, float("nan"), self.tolerance,
                              self.basis, False)
        return GateResult(self.quantity, float(value), self.tolerance,
                          self.basis, bool(float(value) <= self.tolerance))


#: The pre-registered baseline.  Frozen before any run; the figure and the
#: report consume this dict, so a tolerance cannot be tuned after the fact.
BASELINE_SPEC: Dict = {
    "slug": "XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETO",
    "reference": {
        "oss": "lettucecfd/lettuce",
        "paths": [
            "lettuce/ext/_flows/poiseuille.py:30",
            "lettuce/ext/_reporter/error_reporter.py:11",
        ],
        "adopted_pattern": "force-driven single-phase channel + analytic parabola "
                           "+ reusable error reporter with pre-registered gate",
    },
    "canonical_case": {
        "Nx": 8,
        "Ny": 32,
        "tau": 0.8,
        "force": 1e-5,
        "forcing": "guo",
        "wall_origin": 0.5,
        "residual_tol": 1e-12,
    },
    "gate": {
        "quantity": "rel_l2_profile_vs_analytic",
        "tolerance": 0.01,
        "basis": "L2 over fluid nodes, normalised by u_max_analytic "
                 "(lettuce-style 1 % analytic-profile gate; ecosystem cap 15 %)",
    },
    "grid_ladder": [16, 32, 64],
    "reported_quantities": [
        "rel_l2", "rel_linf", "rel_u_max", "steps", "final_residual", "mass_drift",
    ],
}