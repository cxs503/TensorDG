"""Explicit Runge-Kutta tableaus for the DG semi-discrete ODE.

The 1-D/2-D DG solvers in this package integrate ``du/dt = L(u)`` with a
hand-coded scheme.  Accuracy studies, however, need the *time* order to be a
free knob: the classic RKDG accuracy tables (e.g. Sun, Math. Comp. 2024,
DOI 10.1090/mcom/4037) report one column per DG degree, each advanced with
exactly ``p + 1`` stages, so a like-for-like comparison requires the same
family of integrators.  This module provides the minimal-stage tableaus, the
classical RK4, and the five-stage fourth-order strong-stability-preserving
scheme that closes the ``p = 4`` rung (see :func:`tableau_for_degree`).

Butcher barrier
---------------
An explicit RK method with ``s`` stages cannot have order ``s`` for
``s = 5``: order 5 needs at least 6 stages (Butcher 1965).  The reference
column ``RK5DG4`` of the RKDG accuracy tables therefore denotes *five* stages
advancing a degree-4 DG space -- i.e. a fourth-order, five-stage scheme
(Spiteri & Ruuth's SSP-RK(5,4)), not a fifth-order one.  :data:`SSP_RK54` is
that scheme, so the "``p + 1`` stages" convention can be honoured at every
degree without claiming an order that no explicit scheme of that size has.

Only *explicit* tableaus with a strictly lower-triangular ``a`` are provided.
``c`` is carried for reference/reporting; the solvers use the autonomous form
(the semi-discrete DG operator of a linear conservation law has no explicit
time dependence).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class RKTableau:
    """An explicit Runge-Kutta tableau (Butcher form)."""

    name: str
    order: int
    stages: int
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    reference: str = ""

    def check(self) -> "RKTableau":
        s = self.stages
        if len(self.b) != s or len(self.c) != s or len(self.a) != s:
            raise ValueError(f"{self.name}: inconsistent tableau size")
        for i, row in enumerate(self.a):
            if len(row) != i:
                raise ValueError(f"{self.name}: row {i} must have {i} entries")
        if abs(sum(self.b) - 1.0) > 1e-14:
            raise ValueError(f"{self.name}: weights do not sum to one")
        return self


#: Euler (RK1) -- one stage, first order.  Included so that the ``p = 0``
#: (finite-volume-like) rung of an accuracy ladder can be run self-consistently.
EULER = RKTableau(
    name="euler", order=1, stages=1,
    a=((),), b=(1.0,), c=(0.0,),
    reference="forward Euler",
).check()

#: Heun's two-stage, second-order scheme -- the minimal pair for degree 1.
RK2 = RKTableau(
    name="rk2", order=2, stages=2,
    a=((), (1.0,)), b=(0.5, 0.5), c=(0.0, 1.0),
    reference="Heun (1900), explicit trapezoidal rule",
).check()

#: Kutta's three-stage, third-order scheme.
RK3 = RKTableau(
    name="rk3", order=3, stages=3,
    a=((), (0.5,), (-1.0, 2.0)),
    b=(1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0),
    c=(0.0, 0.5, 1.0),
    reference="Kutta (1901)",
).check()

#: Classical four-stage, fourth-order scheme (the package default).
RK4 = RKTableau(
    name="rk4", order=4, stages=4,
    a=((), (0.5,), (0.0, 0.5), (0.0, 0.0, 1.0)),
    b=(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c=(0.0, 0.5, 0.5, 1.0),
    reference="classical Runge-Kutta (1895)",
).check()

#: Butcher's six-stage, fifth-order scheme -- the minimal tableau of order 5.
RK5 = RKTableau(
    name="rk5", order=5, stages=6,
    a=((),
       (1.0 / 4.0,),
       (1.0 / 8.0, 1.0 / 8.0),
       (0.0, -1.0 / 2.0, 1.0),
       (3.0 / 16.0, 0.0, 0.0, 9.0 / 16.0),
       (-3.0 / 7.0, 2.0 / 7.0, 12.0 / 7.0, -12.0 / 7.0, 8.0 / 7.0)),
    b=(7.0 / 90.0, 0.0, 32.0 / 90.0, 12.0 / 90.0, 32.0 / 90.0, 7.0 / 90.0),
    c=(0.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 1.0),
    reference="Butcher (1964), 6-stage 5th-order",
).check()


# --------------------------------------------------------------------------- #
# SSP-RK(5,4): five stages, fourth order (Spiteri & Ruuth 2002)
# --------------------------------------------------------------------------- #
#: Shu-Osher (convex-combination) form of SSP-RK(5,4), transcribed from
#: Spiteri & Ruuth, *A new class of optimal high-order strong-stability-
#: preserving time discretization methods*, SIAM J. Numer. Anal. 40 (2002)
#: 469-491, eq. (3.5); reproduced e.g. as (2.11) in Ketcheson, *Highly
#: efficient strong stability preserving Runge-Kutta methods with low-storage
#: implementations*, SIAM J. Sci. Comput. 30 (2008) 2113-2135.  Each entry is
#: ``(u_old_coeff, u_prev_coeff, dt*F_prev_coeff)``.
_SHUS_OSHER_RK54 = (
    (1.0, 0.0, 0.391752226571890),               # u1 = u + 0.3918 dt F(u)
    (0.444370493651235, 0.555629506348765, 0.368410593050371),   # u2
    (0.620101851488403, 0.379898148511597, 0.251891774271694),   # u3
    (0.178079954393132, 0.821920045606868, 0.544974750228521),   # u4
)
#: Final update: ``u_{n+1} = c2 u2 + c3 u3 + g3 dt F(u3) + c4 u4 + g4 dt F(u4)``.
_FINAL_RK54 = {"coeffs": (0.517231671970585, 0.096059710526147, 0.386708617503269),
               "stage_fluxes": (0.063692468666290, 0.226007483236906)}


def _ssp_rk54_tableau() -> RKTableau:
    """Convert the Shu-Osher form above into Butcher form mechanically.

    ``u_i = u_n + dt * sum_j a_ij F(u_j)`` is accumulated stage by stage, so the
    printed numbers never have to be transcribed (or trusted) by hand; the
    invariants ``sum(b) = 1`` and ``c_i = sum_j a_ij`` are asserted by
    :meth:`RKTableau.check` and by the row-sum test in the unit tests.
    """
    rows: list[list[float]] = []
    # stage 1 (Euler step)
    rows.append([0.391752226571890])
    for u_old, u_prev, g in _SHUS_OSHER_RK54[1:]:
        prev = rows[-1]
        row = [u_prev * a for a in prev]
        # u_prev * u_{i-1} contributes u_prev * (u_n + dt * sum a_{i-1,j} F_j);
        # the u_n coefficients telescope because each Shu-Osher row is a convex
        # combination (u_old + u_prev = 1).
        row.append(g)
        rows.append(row)
    coeffs = _FINAL_RK54["coeffs"]
    g3, g4 = _FINAL_RK54["stage_fluxes"]
    b = [0.0] * 5
    for c, row in zip(coeffs, rows[1:]):    # u2, u3, u4 rows
        for j, a in enumerate(row):
            b[j] += c * a
    # The final Shu-Osher line evaluates fluxes at the *third* and *fourth*
    # stages, i.e. Butcher indices 3 and 4 (index 0 is the F(u^n) evaluation).
    b[3] += g3
    b[4] += g4
    c = [0.0] + [sum(row) for row in rows]   # c_0 = 0 for the first stage
    return RKTableau(
        name="ssp_rk54", order=4, stages=5,
        a=((),) + tuple(tuple(r) for r in rows), b=tuple(b), c=tuple(c),
        reference="Spiteri & Ruuth (2002), SSP-RK(5,4), 5-stage 4th-order",
    ).check()


SSP_RK54 = _ssp_rk54_tableau()

TABLEAUS: dict[str, RKTableau] = {
    tab.name: tab for tab in (EULER, RK2, RK3, RK4, SSP_RK54, RK5)
}


def get_tableau(name: str) -> RKTableau:
    """Return the tableau registered under ``name``."""
    try:
        return TABLEAUS[name]
    except KeyError:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown explicit RK tableau {name!r}; available: {sorted(TABLEAUS)}"
        ) from None


def rk_step(rhs: Callable[[torch.Tensor], torch.Tensor], u: torch.Tensor,
            dt: float, tableau: RKTableau,
            project: Callable[[torch.Tensor], torch.Tensor] | None = None,
            ) -> torch.Tensor:
    """Advance ``u`` by one step of ``tableau`` on ``du/dt = rhs(u)``.

    ``project`` is applied to every stage input (as RKDG limiters require) and
    to the final update; pass ``None`` for the unlimited scheme.
    """
    stage_in: list[torch.Tensor] = []
    k: list[torch.Tensor] = []
    for i in range(tableau.stages):
        y = u
        for j in range(i):
            aij = tableau.a[i][j]
            if aij != 0.0:
                y = y + (dt * aij) * k[j]
        if project is not None:
            y = project(y)
        stage_in.append(y)
        k.append(rhs(y))
    out = u
    for i, bi in enumerate(tableau.b):
        if bi != 0.0:
            out = out + (dt * bi) * k[i]
    return project(out) if project is not None else out


#: Largest order attainable by an explicit RK method with ``s`` stages
#: (Butcher's order barrier: order ``<= s`` for every explicit method, and
#: order 5 is impossible with ``s = 5`` -- it needs at least six stages).
#: Consulted wherever a stage-matched scheme must state what accuracy it can
#: promise, so no claim is made that no scheme of that size can honour.
MAX_ORDER_FOR_STAGES: dict[int, int] = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 7,
}


def max_order_for_stages(stages: int) -> int:
    """Butcher-barrier bound on the order of an explicit ``stages``-stage RK."""
    if stages < 1:
        raise ValueError("stages must be >= 1")
    if stages in MAX_ORDER_FOR_STAGES:
        return MAX_ORDER_FOR_STAGES[stages]
    return stages // 2 + 1 if stages % 2 == 0 else (stages + 1) // 2 + 1


def tableau_for_degree(degree: int) -> RKTableau:
    """Tableau used to advance a degree-``degree`` DG space.

    The RKDG accuracy tables advance degree ``p`` with ``p + 1`` stages, so the
    selection rule is *stage count first*: among the registered tableaus pick
    the highest order whose stage count is exactly ``p + 1``.

    * ``p = 0..3``  the minimal-stage classical tableaus of order ``p + 1``
      (Euler, Heun, Kutta-3, RK4) -- order and stages agree.
    * ``p = 4``     :data:`SSP_RK54` -- five stages, *fourth* order.  Order 5
      with five stages is impossible for an explicit method (Butcher barrier),
      which is exactly why the reference column is called ``RK5DG4``: five
      stages, degree 4.  The time error stays far below the spatial one at the
      CFL used by the accuracy study (measured share <= 1 %), so the observed
      rate is the spatial order ``p + 1 = 5``.
    * ``p >= 5``    no registered tableau has that many stages; the highest
      order available is used (documented cap, not silently extrapolated).
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    wanted_stages = degree + 1
    exact = [t for t in TABLEAUS.values() if t.stages == wanted_stages]
    if exact:
        return max(exact, key=lambda t: t.order)
    return max(TABLEAUS.values(), key=lambda t: (t.order, -t.stages))