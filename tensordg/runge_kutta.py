"""Explicit Runge-Kutta tableaus for the DG semi-discrete ODE.

The 1-D/2-D DG solvers in this package integrate ``du/dt = L(u)`` with a
hand-coded scheme.  Accuracy studies, however, need the *time* order to be a
free knob: the classic RKDG accuracy tables (e.g. Sun, Math. Comp. 2024,
DOI 10.1090/mcom/4037) report ``p``-th degree columns advanced with exactly
``p + 1`` stages, so a like-for-like comparison requires the same family of
integrators.  This module provides the minimal-stage classical tableaus plus
classical RK4, in a form the solvers can drive generically.

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

#: Butcher's six-stage, fifth-order scheme -- needed for a degree-4 DG rung
#: whose spatial order is 5.
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

TABLEAUS: dict[str, RKTableau] = {
    tab.name: tab for tab in (EULER, RK2, RK3, RK4, RK5)
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


def tableau_for_degree(degree: int, cap: int = 5) -> RKTableau:
    """Minimal-stage tableau whose order is ``degree + 1`` (capped at ``cap``).

    ``degree + 1`` is the spatial order of a degree-``degree`` DG scheme for a
    smooth solution, so this is the tableau that makes the time error share the
    spatial order -- the convention used by the RKDG accuracy tables in the
    literature.  Degrees beyond ``cap - 1`` are served by the highest-order
    (``cap``) tableau available.
    """
    if degree < 0:
        raise ValueError("degree must be >= 0")
    order = min(degree + 1, cap)
    for tab in TABLEAUS.values():
        if tab.order == order:
            return tab
    raise ValueError(f"no tableau of order {order}")