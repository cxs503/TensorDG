"""CASE-DG-EULER-VORTEX-ERR: error *conservation* of the transported isentropic vortex.

Sibling of :mod:`benchmarks.case_euler_vortex` (case-pool entry ``EULER-VORTEX``,
"等熵涡平移"), which fixes one transport time ``t = 1`` and one question: is the
field error below tolerance *at that instant*?  That snapshot check cannot
distinguish a scheme whose error is a fixed discretisation residue from one
whose error keeps accumulating while the vortex crosses the periodic box.

This entry (``EULER-VORTEX-ERR``, 等熵涡移动误差守恒) asks the transport
question.  The exact problem has a symmetry: uniform translation in a periodic
box leaves the *shape* of the vortex invariant, hence the discretisation error
is a property of the mesh/flux pair, not of the elapsed time.  Four
machine-decidable criteria:

============ ==========================================================
C1 tolerance at the design checkpoint ``t = 1`` every family's max
             primitive-field relative L2 < 6 % (case-pool tolerance, ref. Yee
             et al. 2000).  The tolerance is a *snapshot* tolerance -- that is
             how sibling entry ``EULER-VORTEX`` uses it (``GATE_SET`` = DG-P2/32^2
             and P0/128^2 at t = 1) -- so long-time checkpoints are graded by C2,
             not by it; their raw numbers stay visible as C1x.
C2 error     at the design gate config the error must not accumulate faster
   conservation   than linearly: the least-squares exponent of ``e ~ t^alpha``
             over the whole ladder satisfies ``alpha <= 1``.  (The stricter
             sub-linear target ``alpha <= 0.5`` is *reported*, not gated: no
             alpha threshold appears in Yee et al. 2000.)
C3 discrete  total mass / x-momentum / y-momentum / energy drift by
   conservation < 1e-10 relative to the *t=0 discrete* state (periodic
             interface fluxes telescope, so this is a round-off-level
             identity, not a modelling assumption).
C4 entropy   ``S = int p rho^{-gamma} dA`` (exact value = box area, the
   budget    vortex is isentropic) drifts by < 1e-3 relative; physical
             dissipation may only *decrease* it.
C5 cross-    the P0 second code path must converge (observed order >= 0.8 at
   check     t = 1), so the case is not certified by a single implementation.
C6 ladder    every checkpoint declared in ``LADDER`` has a result record --
   complete- a declared-but-missing run is a hard FAIL, never a silent PASS.
   ness
C7 design-   C2's conclusion must survive a change of *design point*: every DG
   space     family (mesh x CFL x degree) with >= 3 checkpoints has alpha <= 1,
   invariance and at least two such families must exist.  Without it C2 is a
             statement about one 32^2 / CFL 0.2 run rather than about the scheme.
C8 degree    the same bound must survive a change of *polynomial degree*: every
   invariance degree in (1, 2, 3) with >= 3 checkpoints has alpha <= 1 and at
             least two degrees are decidable.  C7 only moves the mesh and the
             CFL, so without C8 "error conservation" is a claim about p = 2.
C9 design    the long-time error must still fall at the design order: over
   order     n = 32/64 the observed spatial order reaches 0.8 * (p + 1) for
             p = 1 and p = 2 at both t = 1 and t = 10.  This is what turns
             "the error is a grid property" into a measurement instead of a fit.
============ ==========================================================

Two independent discrete families are measured, so the criteria are not
self-certified by one code path:

* ``dg`` -- modal DG (:func:`tensordg.euler_dg_2d.solve_euler_dg_2d_vortex`,
  HLLC + SSP-RK3, L2-projected initial state, volume-quadrature metric);
* ``p0`` -- P0 finite volume (:func:`tensordg.euler_2d.solve_isentropic_vortex`,
  Rusanov + SSP-RK3, exact cell-average metric).

Both are compared with the *exact* solution (:func:`tensordg.euler_2d.
vortex_primitive`), whose exactness this package already verifies independently
(``case_euler_vortex.py --task residual``: ``U_t + F_x + G_y = 0`` to 1e-10).

One process per configuration, so the ladder is embarrassingly parallel::

    python benchmarks/case_euler_vortex_errconserve.py --shard 3/16 --output-dir DIR
    python benchmarks/case_euler_vortex_errconserve.py --aggregate --output-dir DIR
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import tensordg.euler_dg_2d as dg2
from tensordg.euler_1d import GAMMA
from tensordg.euler_2d import (_cell_averages, minimum_images,
                               solve_isentropic_vortex, vortex_error,
                               vortex_primitive)

from benchmarks.verify_euler_dg_2d import _reconstruct, _vortex_error as dg_vortex_error

DOMAIN = ((-5.0, 5.0), (-5.0, 5.0))
STRENGTH = 5.0
BACKGROUND = (1.0, 1.0)
FIELDS = ("density", "velocity_x", "velocity_y", "pressure")
BOX_AREA = (DOMAIN[0][1] - DOMAIN[0][0]) * (DOMAIN[1][1] - DOMAIN[1][0])
#: Box length handed to the *exact* solution.  Both solvers close their faces
#: periodically, so the reference must be the periodic solution: the free-space
#: vortex formula is not periodic and stops being the solution as soon as the
#: centre is advected out of the box (``t > 5`` for ``u_inf = (1, 1)`` on this
#: box).  Without this the long-time "error" is a reference defect, not a
#: discretisation defect -- the numbers of both readings are kept in
#: ``docs/CASE-DG-EULER-VORTEX-ERR.md``.
PERIOD = (DOMAIN[0][1] - DOMAIN[0][0], DOMAIN[1][1] - DOMAIN[1][0])

#: Which periodic reference to measure against.
#: ``min_image`` -- the min-image fold, i.e. only the nearest image of the
#:   free-space vortex (the reading published through round 6);
#: ``lattice``   -- the full lattice sum, which also carries the tails of the
#:   neighbouring images and is therefore the exact periodic solution up to
#:   O(tail^2) nonlinearity.  The two differ by a *velocity-only* truncation
#:   floor of ~1.5e-6 relative L2 (density/pressure differ by ~5e-13), see
#:   ``docs/CASE-DG-EULER-VORTEX-ERR-reference-floor.md``.
REFERENCE_MIN_IMAGE = "min_image"
REFERENCE_LATTICE = "lattice"
REFERENCE_MODELS = (REFERENCE_MIN_IMAGE, REFERENCE_LATTICE)
REFERENCE_MODEL = REFERENCE_MIN_IMAGE


def reference_images(final_time: float, strength: float = STRENGTH,
                     background=BACKGROUND) -> int:
    """Lattice half-width that resolves the periodic reference at ``final_time``.

    One ring more than :func:`minimum_images` requires, so the truncated tail can
    be compared against a converged sum.
    """
    del strength
    half_box = 0.5 * max(abs(DOMAIN[0][1] - DOMAIN[0][0]),
                         abs(DOMAIN[1][1] - DOMAIN[1][0]))
    travel = max(abs(background[0]), abs(background[1])) * final_time
    return 1 + int(math.ceil((half_box + travel) / min(PERIOD)))

#: C1 -- case-pool tolerance for this entry (ref: Yee et al. 2000).
TOLERANCE = 0.06
#: C2 -- error-accumulation bound.  On a fixed mesh a dissipative scheme commits an
#: error proportional to the elapsed transport time, ``e ~ t`` (``alpha = 1``), in
#: the pre-asymptotic regime, so ``alpha <= 1`` is the physical bound for "no
#: secular amplification": ``alpha > 1`` means the error is amplified faster than
#: the accumulated dissipation can explain.  No ``alpha`` threshold appears in
#: Yee et al. (2000) -- the case-pool entry cites that paper for the vortex
#: problem and the 6 % tolerance only.  The stricter sub-linear reading is kept as
#: a *reported* target (``STRICT_ALPHA_TARGET``) rather than a silent gate.
MAX_GROWTH_EXPONENT = 1.0
STRICT_ALPHA_TARGET = 0.5
#: C2 -- informational error ratio between the last checkpoint and ``T_REF``.
GROWTH_RATIO_MAX = 2.0
T_REF = 1.0
#: C3 -- round-off level for a global invariant (1e-10 ~ 4.5e5 eps).
INVARIANT_TOLERANCE = 1e-10
#: Gauss points per direction of ``solve_euler_2d``'s cell-average projection.
#: Must mirror that solver's default, otherwise the t=0 reference state (and
#: hence every conservation drift) would be measured against a different state.
P0_QUADRATURE = 4
#: C4 -- entropy-functional drift at the gate configuration.
ENTROPY_TOLERANCE = 1e-3

#: C10 -- the advection *direction* axis.  Every family above transports the vortex
#: along the grid diagonal, ``u_inf = (1, 1)``: the direction in which a structured
#: DG discretisation is flattered by symmetry (both coordinate fluxes carry the same
#: sign and the same cell crossing pattern).  A direction-invariant statement about
#: "the error is a grid property, not a time property" cannot be certified on one
#: direction, exactly as C7 could not certify it on one mesh or C8 on one degree.
#:
#: The three directions carry the *same* speed ``|u_inf| = sqrt(2)``, so the axis
#: moves the direction alone: a rotated family travels the same distance per unit
#: time, and any change in the accumulation exponent is a change of direction, not
#: of transport length.  ``diag45`` is the historical family (its key is unchanged,
#: so every earlier result file stays in the same family); the two rotated labels
#: get a ``_dir<label>`` suffix in their task/family keys.
DIRECTION_SPEED = math.sqrt(2.0)
DIRECTION_BASELINE = "diag45"
DIRECTION_AXIS = (
    ("diag45", (1.0, 1.0)),
    ("axis0", (DIRECTION_SPEED, 0.0)),
    ("obl30", (DIRECTION_SPEED * math.cos(math.radians(30.0)),
               DIRECTION_SPEED * math.sin(math.radians(30.0)))),
)
#: C10 requires at least this many directions to be decidable, and at least one of
#: them to be a *rotated* direction: a criterion whose only decidable family is the
#: historical ``diag45`` group is a re-reading of C2, not a direction test.
MIN_DECIDABLE_DIRECTIONS = 2


def direction_label(background) -> str:
    """Label of the declared direction whose background velocity matches."""
    vector = (float(background[0]), float(background[1]))
    for label, declared in DIRECTION_AXIS:
        if (abs(vector[0] - declared[0]) < 1e-12
                and abs(vector[1] - declared[1]) < 1e-12):
            return label
    raise ValueError(f"background velocity {vector!r} is not on the direction axis "
                     f"{[lab for lab, _ in DIRECTION_AXIS]}")


def direction_suffix(label: str) -> str:
    """Key suffix for a direction; the historical diagonal keeps an empty one."""
    return "" if label == DIRECTION_BASELINE else f"_dir{label}"


def direction_family(label: str, degree: int = 2, cells: int = 32,
                     cfl: float = 0.2) -> str:
    """Family key of one direction at the design mesh / CFL."""
    return f"dg_d{degree}_n{cells}_cfl{cfl:g}{direction_suffix(label)}"


#: Long-time transport ladder.  With |u_inf| = sqrt(2) and a period of 10,
#: ``t = 10`` is one full box crossing per axis; the P0 family and (round 7) the
#: design DG family ``dg_d2_n32_cfl0.2`` reach ``t = 20`` (two crossings), the
#: latter so C2's fit window is not silently pinned to a single crossing.  The
#: remaining DG families are capped at ``t = 10`` by the per-run budget of the
#: compute gate; the report states that scope explicitly instead of extrapolating.
LADDER = (
    # design family: three checkpoints, so C2's accumulation exponent is decidable
    # (the earlier two-point ladder returned ``nan`` and left the whole case at
    # INCOMPLETE -- see docs/AUDIT-m20486-CASE-DG-EULER-VORTEX-ERR.md).
    *({"task": "dg", "degree": 2, "cells": 32, "final_time": t, "cfl": 0.2}
      for t in (1.0, 2.0, 5.0, 10.0)),
    *({"task": "p0", "cells": 128, "final_time": t, "cfl": 0.4}
      for t in (1.0, 2.0, 5.0, 10.0)),
    # -- design-space replication of C2 (mesh axis + CFL axis) --------------
    # Round 5 measured alpha on one mesh at one CFL only, which leaves the
    # criterion open to the objection that it is a property of that pair.  The
    # two families below close it: same degree, one axis moved at a time.
    *({"task": "dg", "degree": 2, "cells": 32, "final_time": t, "cfl": 0.4}
      for t in (1.0, 2.0, 5.0, 10.0)),
    *({"task": "dg", "degree": 2, "cells": 64, "final_time": t, "cfl": 0.2}
      for t in (1.0, 2.0, 10.0)),
    # cross-check resolution ladder at the design checkpoint t = 1:
    # n = 64/128/256 gives three points for an observed convergence order (C5).
    *({"task": "p0", "cells": cells, "final_time": 1.0, "cfl": 0.4}
      for cells in (64, 256)),
    *({"task": "p0", "cells": 64, "final_time": t, "cfl": 0.4}
      for t in (5.0, 10.0)),
    # -- degree axis (round 7) -----------------------------------------------
    # Every design point above holds the polynomial degree fixed at p = 2, so
    # round 5/6 could only claim that error conservation holds *for that degree*.
    # The two families below move the degree axis alone (mesh and CFL untouched),
    # which turns "the error is a grid property, not a time property" into a
    # statement about the DG scheme family instead of about one p = 2 run.
    *({"task": "dg", "degree": degree, "cells": 32, "final_time": t, "cfl": 0.2}
      for degree in (1, 3) for t in (1.0, 2.0, 5.0, 10.0)),
    # spatial design order (C9): a second resolution per degree, at the snapshot
    # and after one full box crossing, so the design order p+1 is measured rather
    # than assumed.  Degree 2 reads its cells from the two families above; the
    # p = 1 pair is given all four checkpoints so it is also a C7 design point
    # (a two-point family would leave C7 blind on the refined p = 1 mesh).
    *({"task": "dg", "degree": 1, "cells": 64, "final_time": t, "cfl": 0.2}
      for t in (1.0, 2.0, 5.0, 10.0)),
    # double box crossing on the design family: extends C2's fit window past the
    # single crossing the previous rounds stopped at.
    {"task": "dg", "degree": 2, "cells": 32, "final_time": 20.0, "cfl": 0.2},
    # -- direction axis (round 9, C10) ---------------------------------------
    # Degree, mesh, CFL *and* speed are held fixed; only the advection direction
    # moves.  ``axis0`` is grid aligned (the direction in which a structured
    # tensor-product DG operator has no cross-derivative coupling at all) and
    # ``obl30`` sits between it and the diagonal, so the axis brackets the
    # historical 45 deg reading instead of extending it in one direction only.
    *({"task": "dg", "degree": 2, "cells": 32, "final_time": t, "cfl": 0.2,
       "background_velocity": direction}
      for label, direction in DIRECTION_AXIS if label != DIRECTION_BASELINE
      for t in (1.0, 2.0, 5.0, 10.0)),
)

#: Checkpoints that are declared in ``LADDER`` and whose run has been *submitted*
#: to the compute gate but has not landed in ``results/`` yet.  They are absent
#: evidence, so the verdict is INCOMPLETE -- never PASS -- until they land; the
#: archive keeps them visible (a hole that is merely quiet is exactly the defect
#: C6 was written for).  When a record arrives the entry is ignored automatically.
PENDING_RUNS = {
    # round 7 (degree axis + design order + double box crossing).  History, kept
    # because it is the reason C6 stayed undecided for so long: cj3605-cj3608 all
    # died in staging (github unreachable / ref lock / wrong checkout path), while
    # job cj3577 actually ran the whole ladder on n330 and was never landed.  The
    # artifacts of cj3577 are now in ``results/`` (landed and config-rechecked by
    # ``tools/land_cluster_artifacts.py``), so these ids are inert; the entry is
    # kept so the dead jobs are not silently forgotten.
    "dg_d1_n32_t5_cfl0.2": "cj3605",
    "dg_d1_n32_t10_cfl0.2": "cj3606",
    "dg_d3_n32_t1_cfl0.2": "cj3606",
    "dg_d3_n32_t2_cfl0.2": "cj3607",
    "dg_d3_n32_t5_cfl0.2": "cj3607",
    "dg_d3_n32_t10_cfl0.2": "cj3608",
    "dg_d1_n64_t2_cfl0.2": "cj3605",
    "dg_d1_n64_t5_cfl0.2": "cj3606",
    "dg_d1_n64_t10_cfl0.2": "cj3607",
    "dg_d2_n32_t20_cfl0.2": "cj3608",
}

#: Provenance of the *result files* (not of the paper): ``run_dg``/``run_p0``
#: record ``reference_model`` inside every record, but the bundle also contains
#: records written before that field existed.  The aggregator used to default a
#: missing field to ``min_image`` -- an *assumption*, and exactly the assumption
#: that was wrong in rounds 1-3 (the pre-referencing runs were measured against
#: the free-space formula, which is not the periodic solution for t > 5).  A
#: record without a recorded model is therefore only trusted when a replay
#: certificate says the same checkpoint was re-run against the current reference
#: and reproduced the stored numbers.
PROVENANCE_CERTIFICATES = "provenance_certificates.json"
#: Accepted relative deviation between a stored error and its replay.  A genuine
#: replay reproduces to round-off (measured 0 .. 1e-15), so 1e-9 leaves three
#: orders of magnitude of slack for BLAS/threading non-determinism while still
#: rejecting any record that was produced by a different reference or scheme.
PROVENANCE_REPLAY_TOLERANCE = 1e-9
#: Fields of a configuration that change the *physics or the discretisation*.
#: They must agree between the declaration and the record: a result file whose
#: own config says another CFL/degree/mesh is evidence for a different problem.
PROVENANCE_PHYSICS_FIELDS = ("task", "degree", "cells", "final_time", "cfl", "strength")

#: Configuration carrying C2/C4 (same pair of meshes as the sibling entry's gate).
#: The CFL is part of the family identity: ``dt`` follows it, so two CFLs on the
#: same mesh are two different discretisations and must never share one fit.
GATE_FAMILY = "dg_d2_n32_cfl0.2"
#: C8 -- the degrees whose conservation conclusion must be decidable.  Round 6
#: certified C2 on p = 2 only, which is a statement about one polynomial degree;
#: at least ``MIN_DECIDABLE_DEGREES`` distinct degrees must clear the same bound.
DEGREE_AXIS = (1, 2, 3)
MIN_DECIDABLE_DEGREES = 2
#: C9 -- observed spatial order must reach this fraction of the design order p+1.
#: The 80 % convention is the one already used by C5 (0.8 of the theoretical 1.0),
#: declared here so the threshold is a convention, not a number fitted to the run.
ORDER_SLACK = 0.8
#: C9 checkpoints: the snapshot, and one full box crossing (the long-time state
#: whose error conservation C2 certifies).
ORDER_CHECKPOINTS = (1.0, 10.0)
#: C9 meshes per degree: the design mesh and one refinement.
ORDER_CELLS = (32, 64)
#: C9 holds the CFL at its design value: ``dt`` follows the CFL, so mixing two
#: CFLs in one order fit would measure the time step, not the mesh.
ORDER_CFL = 0.2
#: C9 degrees: p = 1 (design order 2) and the design degree p = 2 (order 3).
ORDER_FAMILIES = (1, 2)
#: C7 -- minimum number of *decidable* DG design points (mesh x CFL) that must
#: show error conservation.  One point is an anecdote: C2 lives on a single
#: (32^2, CFL 0.2) run, so without C7 the criterion could be an artefact of that
#: one mesh and that one time step.
MIN_DECIDABLE_DESIGN_FAMILIES = 2
#: One full box crossing per axis: L / u_inf = 10 / 1.
BOX_CROSSING_TIME = 10.0
#: Informational, predeclared refinement prediction: a second-order scheme's
#: spatial error falls faster than its projection floor, so the amplification
#: A = e(10)/e(0) must not grow under mesh refinement.  Registered as a
#: prediction (supported / falsified), not as a gate: no literature threshold
#: exists for A, and inventing one is out of bounds for this case pool.
AMPLIFICATION_PREDICTION = (
    "A(t=10) = e(10)/e(0) on the refined mesh (dg_d2_n64_cfl0.2) must not exceed "
    "A(t=10) on the design mesh (dg_d2_n32_cfl0.2)")




def task_key(config: dict) -> str:
    background = tuple(config.get("background_velocity", BACKGROUND))
    suffix = direction_suffix(direction_label(background))
    if config["task"] == "dg":
        return (f"dg_d{config['degree']}_n{config['cells']}"
                f"_t{config['final_time']:g}_cfl{config['cfl']:g}{suffix}")
    if config["task"] == "p0":
        return (f"p0_n{config['cells']}_t{config['final_time']:g}"
                f"_cfl{config['cfl']:g}{suffix}")
    raise ValueError(f"unknown task {config['task']!r}")


def declared_task_keys() -> list:
    """Task keys of every configuration the module *claims* to have run."""
    return [task_key(config) for config in LADDER]


def ladder_completeness(records: list) -> dict:
    """Reconcile the declared ladder against the records that actually exist.

    A checkpoint that is declared in ``LADDER`` but has no result file is a hole
    in the evidence bundle, and holes must be *reported*, never absorbed: the
    pre-C6 aggregator graded 11 of 12 declared checkpoints as a complete PASS
    because the missing ``dg_d2_n32_t10`` run simply was not in ``results/``.
    """
    declared = declared_task_keys()
    present = {record["task_key"] for record in records}
    missing = [key for key in declared if key not in present]
    pending = {key: PENDING_RUNS[key] for key in missing if key in PENDING_RUNS}
    unexpected = sorted(present - set(declared))
    return {
        "declared": len(declared),
        "present": len(present),
        "missing": missing,
        "pending": pending,
        "unexpected": unexpected,
    }


def canonical_config(config: dict) -> dict:
    """Configuration in a comparison-normal form.

    JSON round-trips tuples into lists, so a naive ``dict == dict`` marks every
    direction family as mismatched.  Normalising here keeps the audit about the
    physics instead of about the serialisation.
    """
    out = {key: config[key] for key in PROVENANCE_PHYSICS_FIELDS if key in config}
    if "background_velocity" in config:
        out["background_velocity"] = [float(v) for v in config["background_velocity"]]
    if "reference" in config:
        out["reference"] = config["reference"]
    return out


def config_mismatch(record: dict, declared: dict) -> dict:
    """Fields where a record's *own* configuration contradicts its declaration."""
    got = canonical_config(record.get("config") or {})
    want = canonical_config(declared)
    diffs = {}
    for key in PROVENANCE_PHYSICS_FIELDS:
        if key in want and got.get(key) != want[key]:
            diffs[key] = {"declared": want[key], "record": got.get(key)}
    if "background_velocity" in want:
        have = got.get("background_velocity")
        if have is None or any(abs(a - b) > 1e-12
                               for a, b in zip(have, want["background_velocity"])) \
                or len(have) != len(want["background_velocity"]):
            diffs["background_velocity"] = {"declared": want["background_velocity"],
                                            "record": have}
    return diffs


def certificate_mismatch(record: dict, certificate: dict) -> dict:
    """Fields where a replay certificate describes a *different* checkpoint.

    ``tools/land_provenance_certificates.py`` applies the same filter before it
    writes a certificate, but the gate must not depend on that: the certificate
    bundle is a plain JSON file, so anything can put a claim in it.  C11
    therefore re-derives the check from the record and the certificate alone.

    A certificate is bound to a checkpoint by three things and all three are
    re-checked here:

    * the physics/discretisation fields (task/degree/cells/cfl/final_time/
      strength) -- a certificate for another mesh is evidence for another problem;
    * the checkpoint name the certificate says it replayed;
    * the stored percentage it claims to reproduce -- a matching ``replay_delta``
      attached to different absolute numbers is a mismatched pairing, not a
      reproduction.
    """
    diffs = {}
    got = canonical_config(record.get("config") or {})
    want = canonical_config(certificate.get("config") or {})
    for key in PROVENANCE_PHYSICS_FIELDS:
        if key in want and got.get(key) != want[key]:
            diffs[key] = {"record": got.get(key), "certificate": want[key]}
    named = certificate.get("task_key")
    if named is not None and named != record.get("task_key"):
        diffs["task_key"] = {"record": record.get("task_key"), "certificate": named}
    stored = certificate.get("stored_max_percent")
    record_percent = record.get("max_percent")
    if stored is not None and record_percent is not None:
        relative = abs(stored - record_percent) / max(abs(record_percent), 1e-30)
        if relative > PROVENANCE_REPLAY_TOLERANCE:
            diffs["stored_max_percent"] = {"record": record_percent,
                                           "certificate": stored,
                                           "relative": relative}
    return diffs


def load_provenance_certificates(output_dir) -> dict:
    """Replay certificates on disk (``{}`` when none were produced yet)."""
    path = Path(output_dir) / PROVENANCE_CERTIFICATES
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return data.get("certificates", data)


def provenance_audit(records: list, certificates: dict | None = None) -> dict:
    """Classify every declared checkpoint by the *provenance* of its evidence.

    C6 asks "is there a file?", which a stale file answers just as well as a
    fresh one.  This audit asks the question C6 cannot:

    * ``recorded``   -- the record carries the reference model it was measured
      against (and it is the one being aggregated);
    * ``replayed``   -- a legacy record (no recorded model) whose checkpoint was
      re-run against the current reference and reproduced the stored errors
      within ``PROVENANCE_REPLAY_TOLERANCE``;
    * ``unverified`` -- legacy record, no certificate: unknown provenance;
    * ``config_mismatch`` -- the record's own config contradicts the ladder;
    * ``missing``    -- declared, no file at all.
    """
    certificates = certificates or {}
    by_key = {record["task_key"]: record for record in records}
    ladder = {task_key(declared): declared for declared in LADDER}
    # Audit every record on disk, not just the declared ones: an undeclared
    # record still contributes numbers to the gates (C6 reports it as
    # ``unexpected``), so its provenance has to be graded as well.
    rows = {}
    for key, record in by_key.items():
        declared = ladder.get(key)
        row = {"family": record.get("family"),
               "recorded_reference_model": record.get("reference_model"),
               "wall_seconds": record.get("wall_seconds")}
        mismatch = config_mismatch(record, declared) if declared else {}
        if mismatch:
            row["status"] = "config_mismatch"
            row["config_mismatch"] = mismatch
            rows[key] = row
            continue
        recorded = record.get("reference_model")
        if recorded is not None:
            row["status"] = "recorded"
            rows[key] = row
            continue
        certificate = certificates.get(key)
        if certificate is None:
            row["status"] = "unverified"
            rows[key] = row
            continue
        delta = certificate.get("replay_delta")
        problems = []
        if certificate.get("reference_model") != REFERENCE_MODEL:
            problems.append(f"replayed against {certificate.get('reference_model')!r}, "
                            f"aggregate is {REFERENCE_MODEL!r}")
        if delta is None or not np.isfinite(delta) or delta > PROVENANCE_REPLAY_TOLERANCE:
            problems.append(f"replay delta {delta!r} > {PROVENANCE_REPLAY_TOLERANCE:g}")
        # The certificate must be about *this* checkpoint: the landing step checks
        # it, and the gate cannot assume the landing step ran.
        diffs = certificate_mismatch(record, certificate)
        if diffs:
            problems.append(f"certificate does not describe this checkpoint: {diffs}")
        row["certificate"] = {"reference_model": certificate.get("reference_model"),
                             "replay_delta": delta,
                             "wall_seconds": certificate.get("wall_seconds")}
        if problems:
            row["status"] = "certificate_rejected"
            row["problems"] = problems
            row["certificate_mismatch"] = diffs
        else:
            row["status"] = "replayed"
        rows[key] = row
    for key in ladder:
        rows.setdefault(key, {"status": "missing"})
    counts = {}
    for row in rows.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    present = {k: v for k, v in rows.items() if v["status"] != "missing"}
    return {"rows": rows, "counts": counts, "present": len(present),
            "certificates_used": len(certificates),
            "replay_tolerance": PROVENANCE_REPLAY_TOLERANCE}


def family_key(record: dict) -> str:
    """Family identity = scheme x degree x mesh x CFL x transport direction.

    The direction has to be part of the identity: two directions on one mesh are
    two discretisations of two *different* problems, and a single ``e ~ t^alpha``
    fit through both would measure the direction mix rather than the accumulation.
    """
    config = record.get("config") or {}
    label = direction_label(tuple(config.get("background_velocity", BACKGROUND)))
    suffix = direction_suffix(label)
    if record["task"] == "dg":
        return (f"dg_d{record['degree']}_n{record['cells']}"
                f"_cfl{record['cfl']:g}{suffix}")
    return f"p0_n{record['cells']}{suffix}"


# ---------------------------------------------------------------------------
# exact-solution invariants
# ---------------------------------------------------------------------------

def exact_invariants(points: int = 320, strength: float = STRENGTH,
                     background=BACKGROUND):
    """Global invariants of the exact initial state via Gauss-Legendre quadrature.

    The vortex is smooth (core scale O(1) in a 10x10 box), so 320 points per
    direction are converged far below the 1e-12 resolution of the drift columns.
    """
    nodes, weights = np.polynomial.legendre.leggauss(points)
    (xmin, xmax), (ymin, ymax) = DOMAIN
    nodes_t = torch.as_tensor(nodes, dtype=torch.float64)
    weights_t = torch.as_tensor(weights, dtype=torch.float64)
    x = 0.5 * (xmin + xmax) + 0.5 * (xmax - xmin) * nodes_t
    y = 0.5 * (ymin + ymax) + 0.5 * (ymax - ymin) * nodes_t
    rho, u, v, p = vortex_primitive(x[None, :], y[:, None], 0.0, strength, background)
    energy = p / (GAMMA - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    jacobian = 0.25 * (xmax - xmin) * (ymax - ymin)
    weight = weights_t[:, None] * weights_t[None, :] * jacobian
    return {
        "mass": float((rho * weight).sum()),
        "momentum_x": float((rho * u * weight).sum()),
        "momentum_y": float((rho * v * weight).sum()),
        "energy": float((energy * weight).sum()),
        "entropy": float(((p / rho ** GAMMA) * weight).sum()),
        "density_min": float(rho.min()),
    }


def exact_core_density(points: int = 4001, strength: float = STRENGTH):
    """Minimum core density, sampled from ``vortex_primitive`` on a fine ray.

    No closed form is hand-transcribed: the value comes from the same analytic
    function the solvers are tested against.
    """
    radius = torch.linspace(0.0, 3.0, points, dtype=torch.float64)
    rho, _, _, _ = vortex_primitive(radius, torch.zeros_like(radius), 0.0,
                                   strength, (0.0, 0.0))
    return float(rho.min())


def reference_defect_sweep(times, points: int = 200):
    """Relative L2 distance between the *periodic* and the *free-space* reference.

    A pure function of the analytic solution -- no time stepping, no solver.  It
    is the quantitative reason this case was re-referenced: the free-space form
    of :func:`vortex_primitive` stops being the solution of the periodic problem
    as soon as the centre leaves the box, so any error measured against it is
    dominated by the reference, not by the scheme.  Reported per checkpoint time
    so the historical readings stay explainable from a reproducible number.
    """
    axis = torch.linspace(DOMAIN[0][0], DOMAIN[0][1], points, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
    sweep = {}
    for time in times:
        periodic = vortex_primitive(grid_x, grid_y, time, STRENGTH, BACKGROUND,
                                    period=PERIOD)
        free = vortex_primitive(grid_x, grid_y, time, STRENGTH, BACKGROUND)
        sweep[f"{time:g}"] = {
            name: 100.0 * float(torch.sqrt(torch.mean((periodic[i] - free[i]) ** 2))
                                / torch.sqrt(torch.mean(periodic[i] ** 2)))
            for i, name in enumerate(FIELDS)}
    return sweep


# ---------------------------------------------------------------------------
# per-configuration measurement
# ---------------------------------------------------------------------------

def _dg_measurements(coeff, basis, nodes, weights):
    """Invariants of one DG state, on the same quadrature as the case gate."""
    nx, ny = coeff.shape[0], coeff.shape[1]
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=coeff.dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=coeff.dtype) + 0.5) * dy
    state = _reconstruct(coeff, basis)
    weight = (weights[:, None] * weights[None, :]) * (0.25 * dx * dy)
    rho, _, _, p = dg2._primitive(state)
    entropy = p / rho ** GAMMA
    field = lambda component: float((state[..., component] * weight).sum())
    s_total = float((entropy * weight).sum())
    return {
        "invariants": {
            "mass": field(0),
            "momentum_x": field(1),
            "momentum_y": field(2),
            "energy": field(3),
            "entropy": s_total,
        },
        "entropy_mean": s_total / float(weight.sum()),
        "density_min": float(rho.min()),
        "pressure_min": float(p.min()),
    }


def _p0_measurements(fields):
    """Invariants of one P0 (cell-average) state; quadrature-free and exact."""
    nx, ny = fields.shape[0], fields.shape[1]
    (xmin, xmax), (ymin, ymax) = DOMAIN
    cell_area = ((xmax - xmin) / nx) * ((ymax - ymin) / ny)
    rho = fields[..., 0]
    mom_x = fields[..., 1]
    mom_y = fields[..., 2]
    energy = fields[..., 3]
    u = mom_x / rho
    v = mom_y / rho
    p = (GAMMA - 1.0) * (energy - 0.5 * rho * (u ** 2 + v ** 2))
    entropy = p / rho ** GAMMA
    total = float(entropy.sum())
    return {
        "invariants": {
            "mass": float(rho.sum() * cell_area),
            "momentum_x": float(mom_x.sum() * cell_area),
            "momentum_y": float(mom_y.sum() * cell_area),
            "energy": float(energy.sum() * cell_area),
            "entropy": float(total * cell_area),
        },
        "entropy_mean": float(entropy.mean()),
        "density_min": float(rho.min()),
        "pressure_min": float(p.min()),
    }


def _operators(degree: int):
    return dg2.legendre_operators(degree, max(degree + 2, 3))


def _initial_dg_state(config: dict):
    """The exact t=0 discrete state the solver starts from (same code path)."""
    degree, cells = config["degree"], config["cells"]
    nodes, weights, basis, _, _ = _operators(degree)
    nx = ny = cells
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    x = xmin + (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(ny, dtype=torch.float64) + 0.5) * dy
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))

    def initial(xx, yy):
        return vortex_primitive(xx, yy, 0.0, strength, background)

    coeff = dg2._initial_coefficients(initial, x, y, dx, dy, nodes, weights, basis,
                                      torch.device("cpu"), torch.float64)
    signs = _operators(degree)[4]
    coeff = dg2._positivity_filter(coeff, basis, signs)
    return coeff, basis, nodes, weights


def _initial_p0_state(config: dict):
    cells = config["cells"]
    (xmin, xmax), (ymin, ymax) = DOMAIN
    dx, dy = (xmax - xmin) / cells, (ymax - ymin) / cells
    x = xmin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dx
    y = ymin + (torch.arange(cells, dtype=torch.float64) + 0.5) * dy
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))

    def initial(xx, yy):
        return vortex_primitive(xx, yy, 0.0, strength, background)

    return _cell_averages(initial, x, y, dx, dy, P0_QUADRATURE,
                          torch.device("cpu"), torch.float64)


def _reference_kwargs(config: dict, final_time: float) -> dict:
    """Keyword arguments selecting the periodic reference for one checkpoint."""
    model = config.get("reference", REFERENCE_MODEL)
    if model == REFERENCE_LATTICE:
        return {"period": PERIOD, "images": reference_images(final_time)}
    if model == REFERENCE_MIN_IMAGE:
        return {"period": PERIOD}
    raise ValueError(f"unknown reference model {model!r}")


def run_dg(config: dict) -> dict:
    started = time.time()
    degree = config["degree"]
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))
    final_time = config.get("final_time", 1.0)
    _x, _y, coeff, meta = dg2.solve_euler_dg_2d_vortex(
        ncells=config["cells"],
        degree=degree,
        final_time=final_time,
        cfl=config.get("cfl", 0.2),
        strength=strength,
        background_velocity=background,
        domain=DOMAIN,
    )
    nodes, weights, basis, _, _ = _operators(degree)
    reference_kwargs = _reference_kwargs(config, final_time)
    errors = dg_vortex_error(coeff, basis, nodes, DOMAIN, final_time, strength,
                            background, **reference_kwargs)
    final = _dg_measurements(coeff, basis, nodes, weights)
    coeff0, basis0, nodes0, weights0 = _initial_dg_state(config)
    initial = _dg_measurements(coeff0, basis0, nodes0, weights0)
    return {
        "task": "dg",
        "degree": degree,
        "cells": config["cells"],
        "final_time": final_time,
        "cfl": config.get("cfl", 0.2),
        "steps": meta["steps"],
        # ``dt`` is the *last* step, already clamped by ``final_time - time`` in
        # the solver loop, so ``steps * dt != final_time`` by construction.
        "dt": meta["dt"],
        "dt_note": "final (clamped) step; steps*dt is not the elapsed time",
        "dofs": meta["dofs"],
        "errors": {name: float(errors[name]) for name in FIELDS},
        "metric": "volume-quadrature relative L2 (L2-projected modal state)",
        "reference_model": config.get("reference", REFERENCE_MODEL),
        "reference_images": reference_kwargs.get("images"),
        "final": final,
        "initial": initial,
        "wall_seconds": time.time() - started,
    }


def run_p0(config: dict) -> dict:
    started = time.time()
    strength = config.get("strength", STRENGTH)
    background = tuple(config.get("background_velocity", BACKGROUND))
    final_time = config.get("final_time", 1.0)
    x, y, fields, meta = solve_isentropic_vortex(
        ncells=config["cells"],
        final_time=final_time,
        cfl=config.get("cfl", 0.4),
        domain=DOMAIN,
        strength=strength,
        background_velocity=background,
    )
    conservative = fields["conservative"] if isinstance(fields, dict) else fields
    reference_kwargs = _reference_kwargs(config, final_time)
    raw = vortex_error(x, y, conservative, final_time, strength, background,
                       **reference_kwargs)
    errors = {name: raw[name]["relative_l2"] for name in FIELDS}
    final = _p0_measurements(conservative)
    initial = _p0_measurements(_initial_p0_state(config))
    return {
        "task": "p0",
        "degree": 0,
        "cells": config["cells"],
        "final_time": final_time,
        "cfl": config.get("cfl", 0.4),
        "steps": meta["steps"],
        "dt": meta.get("dt"),
        "dofs": int(conservative.numel()),
        "errors": {name: float(errors[name]) for name in FIELDS},
        "metric": "exact cell-average relative L2 (P0 finite volume)",
        "reference_model": config.get("reference", REFERENCE_MODEL),
        "reference_images": reference_kwargs.get("images"),
        "final": final,
        "initial": initial,
        "wall_seconds": time.time() - started,
    }


def run_task(config: dict) -> dict:
    record = (run_dg if config["task"] == "dg" else run_p0)(config)
    record["task_key"] = task_key(config)
    # the config is the authority for the family key (it carries the direction),
    # so it has to be attached before the key is derived
    record["config"] = dict(config)
    record["family"] = family_key(record)
    record["max_relative_l2"] = max(record["errors"].values())
    record["max_percent"] = 100.0 * record["max_relative_l2"]
    drift = {}
    for name, value in record["final"]["invariants"].items():
        reference = record["initial"]["invariants"][name]
        drift[name] = abs(value - reference) / abs(reference)
    record["invariant_drift"] = drift
    record["entropy_production"] = (
        record["final"]["invariants"]["entropy"] / record["initial"]["invariants"]["entropy"]
        - 1.0)
    record["entropy_drift"] = abs(record["entropy_production"])
    record["density_min_relative_error"] = (
        abs(record["final"]["density_min"] - record["initial"]["density_min"])
        / abs(record["initial"]["density_min"]))
    record["gate_pass"] = record["max_relative_l2"] < TOLERANCE
    return record


# ---------------------------------------------------------------------------
# aggregation / gates / tables
# ---------------------------------------------------------------------------

def _fit_exponent(times, errors, t_min=T_REF):
    """Least-squares exponent of ``e ~ t^alpha`` over ``t >= t_min``.

    Fewer than three checkpoints is reported as ``nan``: a two-point "fit" through
    an error curve that starts at the projection floor says nothing about
    accumulation, and turning it into a gate would be a vacuous claim.
    """
    points = [(t, e) for t, e in zip(times, errors) if t >= t_min and e > 0.0]
    if len(points) < 3:
        return float("nan")
    lt = np.log([p[0] for p in points])
    le = np.log([p[1] for p in points])
    return float(np.polyfit(lt, le, 1)[0])


def _group(records):
    families = {}
    for record in records:
        families.setdefault(record["family"], []).append(record)
    for rows in families.values():
        rows.sort(key=lambda r: r["final_time"])
    return families


def _p0_cells(record) -> int:
    """Cell count of a P0 record, from the record or from its family key."""
    cells = record.get("cells")
    if cells is None:
        cells = int(record["family"].split("_n")[-1])
    return int(cells)


def _observed_order(records, checkpoint: float):
    """Observed convergence order of the P0 cross-check family at one checkpoint.

    Uses the cell counts / errors actually present, so the fit degrades gracefully
    (``nan``) when fewer than three resolutions have been run.  This is the
    *cross-check* integrity measure: it asks whether the second code path is
    converging at all, not whether it meets the design-mesh tolerance.
    """
    rows = [r for r in records if r["family"].startswith("p0_n")
            and abs(r["final_time"] - checkpoint) < 1e-12]
    rows.sort(key=_p0_cells)
    cells = np.array([_p0_cells(r) for r in rows], dtype=float)
    err = np.array([r["max_relative_l2"] for r in rows], dtype=float)
    if len(rows) < 3 or np.any(err <= 0.0):
        return float("nan"), [int(c) for c in cells]
    return float(-np.polyfit(np.log(cells), np.log(err), 1)[0]), [int(c) for c in cells]


def record_direction(record: dict) -> str:
    """Declared advection direction of one record (``diag45`` when unrecorded).

    Result files written before the direction axis existed carry no
    ``background_velocity``; those are the historical ``diag45`` legs, so an
    absent field means the baseline rather than "unknown".
    """
    background = (record.get("config") or {}).get("background_velocity")
    if background is None:
        return DIRECTION_BASELINE
    return direction_label(tuple(background))


def _dg_order(records, degree: int, checkpoint: float, cells=ORDER_CELLS):
    """Observed spatial order of one DG degree family at one checkpoint.

    Least-squares slope of ``log e`` against ``log h`` over the meshes present.
    Two resolutions are enough here (unlike C2's exponent, where a two-point fit
    through a curve that starts at the projection floor says nothing): the slope
    of a straight line through two points *is* the measured order, and the gate
    asks only whether it reaches a predeclared fraction of ``p + 1``.  The CFL is
    pinned to ``ORDER_CFL``: two CFLs on one mesh are two discretisations, and a
    fit through both would measure the time step rather than the mesh.  The
    advection direction is pinned for the same reason -- and that pin was a real
    defect: once the C10 axis landed, the p = 2 fit ran through *four* points (the
    three 32-cell legs ``diag45``/``axis0``/``obl30`` plus one 64-cell leg), so the
    published slope (4.09 at t = 1, 4.64 at t = 10) was mostly the spread between
    directions, not the mesh.  DG-P2 cannot beat p + 1 = 3 in the asymptotic range,
    which is what gave it away.  Rotated legs are graded by C10, not here.
    """
    rows = [r for r in records if r.get("task") == "dg"
            and int(r.get("degree", -1)) == int(degree)
            and abs(float(r.get("cfl", float("nan"))) - ORDER_CFL) < 1e-12
            and record_direction(r) == DIRECTION_BASELINE
            and int(r.get("cells", -1)) in set(cells)
            and abs(r["final_time"] - checkpoint) < 1e-12]
    rows.sort(key=lambda r: r["cells"])
    used = np.array([r["cells"] for r in rows], dtype=float)
    err = np.array([r["max_relative_l2"] for r in rows], dtype=float)
    if len(rows) < 2 or np.any(err <= 0.0) or len(set(used)) < 2:
        return float("nan"), [int(c) for c in used]
    return float(-np.polyfit(np.log(used), np.log(err), 1)[0]), [int(c) for c in used]


def design_space_spread(records: list) -> dict:
    """Cross-design-point errors at one box crossing (informational, C7x).

    C7 decides *whether* conservation survives a change of design point; this
    block records *how much* the long-time error moves when the mesh or the time
    step is changed.  It is informational because no literature threshold exists
    for either ratio -- the numbers are here so a reader can judge them, and so a
    future round cannot claim "design-space invariant" without them on the page.
    """
    at_crossing = {}
    for record in records:
        if (record["family"].startswith("dg_d")
                and abs(record["final_time"] - BOX_CROSSING_TIME) < 1e-9):
            at_crossing[record["family"]] = record["max_percent"]
    design_point, cfl_point, refined = ("dg_d2_n32_cfl0.2",
                                        "dg_d2_n32_cfl0.4",
                                        "dg_d2_n64_cfl0.2")
    ratio = lambda num, den: (at_crossing[num] / at_crossing[den]
                             if num in at_crossing and den in at_crossing
                             and at_crossing[den] > 0.0 else float("nan"))
    return {
        "criterion": ("INFORMATIONAL: max-field error at t = "
                      f"{BOX_CROSSING_TIME:g} per DG design point, and the two "
                      "one-at-a-time ratios"),
        "e_at_crossing_percent": at_crossing,
        "cfl_ratio_0p4_over_0p2": ratio(cfl_point, design_point),
        "refinement_ratio_n64_over_n32": ratio(refined, design_point),
        "passed": True,   # informational: never gates the verdict
        "status": "informational",
    }


def amplification(records: list) -> dict:
    """A(t) = e(t)/e(0) per design point, e(0) from the projection baseline (C7x).

    The predeclared prediction (:data:`AMPLIFICATION_PREDICTION`) is falsifiable:
    if the refined mesh amplifies the *same* transport error more than the design
    mesh does, then the "error is a property of the mesh/flux pair, not of the
    elapsed time" reading would be wrong in the one direction that matters, and
    this flag would read ``False``.  Reported, never gated.
    """
    from benchmarks.vortex_err_projection_baseline import dg_baseline

    values = {}
    for record in records:
        config = record.get("config") or {}
        cells, degree = config.get("cells"), config.get("degree")
        if (not record["family"].startswith("dg_d")
                or cells is None or degree is None
                or abs(record["final_time"] - BOX_CROSSING_TIME) > 1e-9):
            continue
        e0 = dg_baseline(degree=degree, cells=cells)["max_relative_l2"]
        values[record["family"]] = {
            "cells": int(cells),
            "cfl": float(record["cfl"]),
            "e0_percent": 100.0 * e0,
            "e_crossing_percent": record["max_percent"],
            "amplification": (record["max_relative_l2"] / e0 if e0 > 0.0
                              else float("nan")),
        }
    design_point, refined = "dg_d2_n32_cfl0.2", "dg_d2_n64_cfl0.2"
    supported = None
    if design_point in values and refined in values:
        supported = bool(values[refined]["amplification"]
                         <= values[design_point]["amplification"])
    return {
        "prediction": AMPLIFICATION_PREDICTION,
        "values": values,
        "supported": supported,
        "passed": True,   # a falsified prediction is a finding, not a gate failure
        "status": "informational",
    }


def _normalise_families(records: list) -> list:
    """Re-derive the family key from each record's own configuration.

    Result files written before the CFL became part of the family identity carry
    the old mesh-only key (``dg_d2_n32``).  Trusting the stored string would
    split one family into two and silently hide the design family from C2, so the
    key is always recomputed from ``(task, degree, cells, cfl)`` -- the config is
    the authority, not the label.
    """
    for record in records:
        config = record.get("config") or {}
        if (record.get("task") and record.get("cfl") is not None
                and config.get("cells") is not None):
            record["family"] = family_key(record)
    return records


def evaluate_gates(records: list, exact: dict, certificates: dict | None = None) -> dict:
    records = _normalise_families(records)
    families = _group(records)
    gate_rows = [r for r in families.get(GATE_FAMILY, []) if r["final_time"] >= T_REF]
    gates = {}

    # -- C1: snapshot tolerance at the design checkpoint, *every* family. ------
    # The case-pool tolerance (6 %, Yee et al. 2000) is a snapshot tolerance: the
    # sibling entry EULER-VORTEX applies it to its GATE_SET at t = 1 only.  The
    # previous release applied it to *all* long-time checkpoints, which graded a
    # deliberately first-order scheme against a design-mesh tolerance and failed
    # the gate at 8.97 % while the t = 1 checkpoint of the same run reads 2.82 %.
    # Long-time checkpoints are therefore graded by C2, and their raw worst value
    # is kept verbatim in C1x below -- nothing is dropped.
    snapshot = [r for r in records if abs(r["final_time"] - T_REF) < 1e-12]
    snapshot_worst = max((r["max_relative_l2"] for r in snapshot),
                         default=float("nan"))
    snapshot_worst_row = (max(snapshot, key=lambda r: r["max_relative_l2"])
                          if snapshot else None)
    c1_ok = bool(np.isfinite(snapshot_worst) and snapshot_worst < TOLERANCE)
    gates["C1_tolerance"] = {
        "criterion": (f"every family at the design checkpoint t={T_REF:g}: "
                      f"max-field relative L2 < {TOLERANCE:.0%} (snapshot tolerance)"),
        "measured": snapshot_worst,
        "worst_at": snapshot_worst_row["task_key"] if snapshot_worst_row else None,
        "families": sorted({r["family"] for r in snapshot}),
        "passed": c1_ok,
        "status": ("passed" if c1_ok else
                   ("failed" if np.isfinite(snapshot_worst) else "not_evaluated")),
    }

    # -- C1x: raw cross-check numbers (informational, never softened). ---------
    cross = [r for r in records if r["final_time"] >= T_REF]
    cross_worst = max((r["max_relative_l2"] for r in cross), default=float("nan"))
    worst_row = max(cross, key=lambda r: r["max_relative_l2"]) if cross else None
    gates["C1x_crosscheck_raw"] = {
        "criterion": (f"INFORMATIONAL: raw max-field relative L2 over *every* "
                      f"long-time checkpoint, vs {TOLERANCE:.0%}"),
        "measured": cross_worst,
        "worst_at": worst_row["task_key"] if worst_row else None,
        "note": ("reported for transparency only: the first-order P0 family exceeds "
                 "the snapshot tolerance during long-time transport; accumulation is "
                 "what C2 measures, and C5 keeps the cross-check honest"),
        "per_reference_time": {
            f"{t:g}": {r["task_key"]: r["max_percent"] / 100.0
                       for r in cross if abs(r["final_time"] - t) < 1e-12}
            for t in sorted({r["final_time"] for r in cross})},
        "passed": bool(np.isfinite(cross_worst) and cross_worst < TOLERANCE),
    }

    # -- C2: error accumulation on the design family. --------------------------
    exponents = {name: _fit_exponent([r["final_time"] for r in rows],
                                    [r["max_relative_l2"] for r in rows])
                 for name, rows in families.items()}
    ratio = float("nan")
    if gate_rows:
        reference = [r for r in gate_rows if abs(r["final_time"] - T_REF) < 1e-12]
        if reference:
            ratio = gate_rows[-1]["max_relative_l2"] / reference[0]["max_relative_l2"]
    gate_exponent = exponents.get(GATE_FAMILY, float("nan"))
    alpha_decidable = bool(np.isfinite(gate_exponent))
    # 1e-9 slack: the least-squares exponent of an exactly linear curve lands on
    # 1.0 +/- 1 eps, and a gate must not hinge on the sign of that round-off.
    alpha_ok = bool(alpha_decidable
                    and gate_exponent <= MAX_GROWTH_EXPONENT + 1e-9)
    gates["C2_error_conservation"] = {
        "criterion": (f"{GATE_FAMILY}: least-squares exponent of e ~ t^alpha over "
                      f"t = {T_REF:g}..{gate_rows[-1]['final_time']:g} satisfies "
                      f"alpha <= {MAX_GROWTH_EXPONENT} (no super-linear accumulation)"
                      if gate_rows else
                      f"{GATE_FAMILY}: alpha <= {MAX_GROWTH_EXPONENT}"),
        "growth_ratio": ratio,
        "growth_ratio_window": [T_REF, gate_rows[-1]["final_time"] if gate_rows else None],
        "strict_target": {"alpha_max": STRICT_ALPHA_TARGET,
                          "passed": bool(alpha_decidable
                                         and gate_exponent
                                         <= STRICT_ALPHA_TARGET + 1e-9)},
        "exponent": gate_exponent,
        "checkpoint_count": len(gate_rows),
        "exponents_all_families": exponents,
        # A decidable sub-criterion that fails must read as ``failed``, never as
        # ``not_evaluated``: the previous release hid the 19.5x growth ratio
        # behind an INCOMPLETE verdict.
        "status": ("passed" if alpha_ok else
                   ("failed" if alpha_decidable else "not_evaluated")),
        "passed": alpha_ok,
    }

    # -- C3: discrete conservation, on a scale that cannot vanish. -----------
    # A relative drift needs a non-zero reference.  For the *cross-stream*
    # momentum of a straight transport that reference is exactly zero (the swirl
    # has no net circulation, so <rho v> = 0 when u_inf is along x), and the
    # relative drift of a round-off sized integral then reads O(1) -- the
    # axis-aligned direction added by C10 exposed exactly that: the old metric
    # reported ``dg_d2_n32_t2_cfl0.2_diraxis0:momentum_y`` as a conservation
    # failure while the absolute drift was at round-off.  The denominator is
    # therefore the *transported amount* of the invariant,
    # ``max(|reference|, mass * |u_inf|)`` for a momentum component and
    # ``|reference|`` otherwise (mass, energy and entropy cannot cancel).
    worst_drift, worst_at = 0.0, None
    for record in records:
        config = record.get("config") or {}
        background = tuple(config.get("background_velocity", BACKGROUND))
        speed = float(math.hypot(*background))
        initial_inv = (record.get("initial") or {}).get("invariants")
        final_inv = (record.get("final") or {}).get("invariants")
        mass = abs(initial_inv["mass"]) if initial_inv else 0.0
        for name in ("mass", "momentum_x", "momentum_y", "energy"):
            if not initial_inv or not final_inv or name not in initial_inv:
                # records that only carry the precomputed drift (test bundles)
                value = record["invariant_drift"][name]
                if value > worst_drift:
                    worst_drift, worst_at = value, f"{record['task_key']}:{name}"
                continue
            reference = initial_inv[name]
            scale = abs(reference)
            if name in ("momentum_x", "momentum_y"):
                scale = max(scale, mass * speed)
            value = abs(final_inv[name] - reference)
            value = value / scale if scale > 0.0 else float("inf")
            if value > worst_drift:
                worst_drift, worst_at = value, f"{record['task_key']}:{name}"
    c3_ok = bool(worst_drift < INVARIANT_TOLERANCE)
    gates["C3_discrete_conservation"] = {
        "criterion": ("drift of mass/x-mom/y-mom/energy vs the t=0 discrete state, "
                      "scaled by the transported amount "
                      f"max(|reference|, mass*|u_inf|) < {INVARIANT_TOLERANCE:g}"),
        "worst_drift": worst_drift,
        "worst_at": worst_at,
        "passed": c3_ok,
        "status": "passed" if c3_ok else "failed",
    }

    # -- C4: entropy budget (unchanged). --------------------------------------
    entropy_drift = max((r["entropy_drift"] for r in gate_rows), default=float("nan"))
    c4_ok = bool(np.isfinite(entropy_drift) and entropy_drift < ENTROPY_TOLERANCE)
    gates["C4_entropy_budget"] = {
        "criterion": (f"{GATE_FAMILY}: |Delta S|/S(0) < {ENTROPY_TOLERANCE:g} "
                      "(S = int p rho^-gamma)"),
        "measured": entropy_drift,
        "exact_entropy": exact["entropy"],
        "passed": c4_ok,
        "status": ("passed" if c4_ok else
                   ("failed" if np.isfinite(entropy_drift) else "not_evaluated")),
    }

    # -- C5: cross-check convergence (the integrity check for C1x). -----------
    order, cells = _observed_order(records, T_REF)
    c5_ok = bool(np.isfinite(order) and order >= 0.8)
    gates["C5_crosscheck_order"] = {
        "criterion": (f"P0 cross-check observed order at t={T_REF:g} over "
                      f"n=64/128/256 "
                      ">= 0.8 (first order; confirms the second code path converges)"),
        "order": order,
        "cells": cells,
        "passed": c5_ok,
        "status": ("passed" if c5_ok else
                   ("failed" if np.isfinite(order) else "not_evaluated")),
    }

    # -- C6: ladder completeness (no declared checkpoint may be silently absent).
    completeness = ladder_completeness(records)
    # A missing checkpoint that has a *submitted job* behind it is still missing
    # evidence, but it is a known, self-resolving hole rather than a silent one:
    # it reads as ``not_evaluated`` (=> INCOMPLETE), never as ``passed``.  Any
    # other missing checkpoint stays a hard failure.
    unfinished = [key for key in completeness["missing"]
                  if key not in completeness["pending"]]
    if not completeness["missing"]:
        c6_status = "passed"
    elif unfinished:
        c6_status = "failed"
    else:
        c6_status = "not_evaluated"
    gates["C6_ladder_completeness"] = {
        "criterion": ("every checkpoint declared in LADDER has a result record "
                      f"({completeness['declared']} declared)"),
        "declared": completeness["declared"],
        "present": completeness["present"],
        "missing": completeness["missing"],
        "pending_jobs": completeness["pending"],
        "unexpected": completeness["unexpected"],
        "note": ("unexpected keys are listed for transparency but are not a "
                 "failure: extra evidence is allowed, missing evidence is not. "
                 "A missing checkpoint with a submitted job reads not_evaluated "
                 "(INCOMPLETE), never passed"),
        "passed": c6_status == "passed",
        "status": c6_status,
    }

    # -- C7: does the conservation conclusion survive a change of design point?
    # C2 is bound to one (mesh, CFL) pair.  A reviewer is entitled to ask whether
    # alpha <= 1 there is a property of the scheme or of that one run, so every DG
    # design point with enough checkpoints for a decidable exponent is graded by
    # the same bound, and at least ``MIN_DECIDABLE_DESIGN_FAMILIES`` of them must
    # exist -- otherwise the criterion is reported as *not evaluated* rather than
    # silently satisfied by a single family.
    design_families = {name: rows for name, rows in families.items()
                       if name.startswith("dg_d")}
    per_family = {}
    for name, rows in sorted(design_families.items()):
        exponent = exponents.get(name, float("nan"))
        per_family[name] = {
            "checkpoints": len(rows),
            "final_time_max": max(r["final_time"] for r in rows),
            "cfl": sorted({float(r["cfl"]) for r in rows}),
            "exponent": exponent,
            "passed": bool(np.isfinite(exponent)
                           and exponent <= MAX_GROWTH_EXPONENT + 1e-9),
        }
    evaluated = {k: v for k, v in per_family.items() if np.isfinite(v["exponent"])}
    offenders = {k: v["exponent"] for k, v in evaluated.items() if not v["passed"]}
    if offenders:
        c7_status = "failed"
    elif len(evaluated) < MIN_DECIDABLE_DESIGN_FAMILIES:
        c7_status = "not_evaluated"
    else:
        c7_status = "passed"
    gates["C7_design_space_invariance"] = {
        "criterion": (f"every DG design point (mesh x CFL x degree) with >= 3 "
                      f"checkpoints satisfies alpha <= {MAX_GROWTH_EXPONENT}, and at "
                      f"least {MIN_DECIDABLE_DESIGN_FAMILIES} such points exist"),
        "measured": max((v["exponent"] for v in evaluated.values()),
                        default=float("nan")),
        "worst_at": (max(evaluated, key=lambda k: evaluated[k]["exponent"])
                     if evaluated else None),
        "per_family": per_family,
        "evaluated_families": sorted(evaluated),
        "offenders": offenders,
        "passed": c7_status == "passed",
        "status": c7_status,
    }

    # -- C8: does the conservation conclusion survive a change of *degree*? ---
    # C7 moves the mesh and the CFL, both of which leave the spatial operator's
    # order untouched.  A reviewer can therefore still read C2 as "p = 2 happens
    # to conserve its error".  C8 asks the same bound of every degree in
    # ``DEGREE_AXIS`` that has a decidable exponent, and requires at least
    # ``MIN_DECIDABLE_DEGREES`` such degrees: one degree is the anecdote C7 was
    # written against, three is a statement about the DG family.
    degree_rows = {}
    for degree in DEGREE_AXIS:
        family = f"dg_d{degree}_n32_cfl0.2"
        rows = families.get(family, [])
        exponent = exponents.get(family, float("nan"))
        degree_rows[degree] = {
            "family": family,
            "checkpoints": len(rows),
            "final_time_max": max((r["final_time"] for r in rows), default=None),
            "exponent": exponent,
            "passed": bool(np.isfinite(exponent)
                           and exponent <= MAX_GROWTH_EXPONENT + 1e-9),
        }
    degrees_evaluated = [d for d, row in degree_rows.items()
                         if np.isfinite(row["exponent"])]
    degree_offenders = {d: row["exponent"] for d, row in degree_rows.items()
                        if np.isfinite(row["exponent"]) and not row["passed"]}
    if degree_offenders:
        c8_status = "failed"
    elif len(degrees_evaluated) < MIN_DECIDABLE_DEGREES:
        c8_status = "not_evaluated"
    else:
        c8_status = "passed"
    gates["C8_degree_invariance"] = {
        "criterion": (f"every DG degree in {list(DEGREE_AXIS)} with >= 3 checkpoints "
                      f"satisfies alpha <= {MAX_GROWTH_EXPONENT}, and at least "
                      f"{MIN_DECIDABLE_DEGREES} distinct degrees are decidable"),
        "measured": max((row["exponent"] for row in degree_rows.values()
                         if np.isfinite(row["exponent"])), default=float("nan")),
        "worst_at": (max((d for d in degrees_evaluated),
                         key=lambda d: degree_rows[d]["exponent"])
                     if degrees_evaluated else None),
        "per_degree": degree_rows,
        "degrees_evaluated": degrees_evaluated,
        "offenders": degree_offenders,
        "passed": c8_status == "passed",
        "status": c8_status,
    }

    # -- C10: does the bound survive a change of *advection direction*? ------
    # Every family above is transported along the grid diagonal.  C7 moved the mesh
    # and the CFL, C8 the degree; the direction was never moved, so "the error is a
    # grid property" was still a statement about one advection direction.  C10 asks
    # the same alpha bound of every declared direction and requires that at least
    # one *rotated* direction (not ``diag45``) be decidable -- otherwise the
    # criterion would only re-read C2.
    direction_rows = {}
    for label, direction in DIRECTION_AXIS:
        family = direction_family(label)
        rows_d = families.get(family, [])
        exponent_d = exponents.get(family, float("nan"))
        direction_rows[label] = {
            "family": family,
            "background_velocity": [float(direction[0]), float(direction[1])],
            "speed": float(math.hypot(*direction)),
            "angle_deg": float(math.degrees(math.atan2(direction[1],
                                                       direction[0]))),
            "checkpoints": len(rows_d),
            "final_time_max": max((r["final_time"] for r in rows_d),
                                  default=None),
            "exponent": exponent_d,
            "passed": bool(np.isfinite(exponent_d)
                           and exponent_d <= MAX_GROWTH_EXPONENT + 1e-9),
        }
    directions_evaluated = [label for label, row in direction_rows.items()
                            if np.isfinite(row["exponent"])]
    rotated_evaluated = [label for label in directions_evaluated
                         if label != DIRECTION_BASELINE]
    direction_offenders = {label: row["exponent"]
                           for label, row in direction_rows.items()
                           if np.isfinite(row["exponent"]) and not row["passed"]}
    if direction_offenders:
        c10_status = "failed"
    elif (len(directions_evaluated) < MIN_DECIDABLE_DIRECTIONS
            or not rotated_evaluated):
        c10_status = "not_evaluated"
    else:
        c10_status = "passed"
    gates["C10_direction_invariance"] = {
        "criterion": (f"every declared advection direction at "
                      f"{direction_family(DIRECTION_BASELINE)} "
                      f"(mesh x CFL x degree held fixed) satisfies alpha <= "
                      f"{MAX_GROWTH_EXPONENT}, at least "
                      f"{MIN_DECIDABLE_DIRECTIONS} directions are decidable and at "
                      f"least one of them is rotated off the grid diagonal"),
        "measured": max((row["exponent"] for row in direction_rows.values()
                         if np.isfinite(row["exponent"])), default=float("nan")),
        "worst_at": (max(directions_evaluated,
                         key=lambda label: direction_rows[label]["exponent"])
                     if directions_evaluated else None),
        "per_direction": direction_rows,
        "directions_evaluated": directions_evaluated,
        "rotated_evaluated": rotated_evaluated,
        "offenders": direction_offenders,
        "passed": c10_status == "passed",
        "status": c10_status,
    }

    # -- C9: is the long-time error a *grid* property?  Measure the order. ----
    # The claim C2 makes ("the error is a mesh property, not a time property") is
    # only credible if the error still falls at the design order p + 1 after the
    # vortex has crossed the box: a discretisation that accumulates a non-grid
    # residue loses that slope.  Every (degree, checkpoint) pair with two meshes
    # is graded against ``ORDER_SLACK * (p + 1)``.
    order_rows = {}
    order_failures = {}
    for degree in ORDER_FAMILIES:
        for checkpoint in ORDER_CHECKPOINTS:
            order, cells = _dg_order(records, degree, checkpoint)
            target = ORDER_SLACK * (degree + 1)
            ok = bool(np.isfinite(order) and order + 1e-12 >= target)
            key = f"dg_d{degree}_t{checkpoint:g}"
            order_rows[key] = {
                "degree": degree,
                "checkpoint": checkpoint,
                "cells": cells,
                "order": order,
                "target": target,
                "passed": ok,
            }
            if np.isfinite(order) and not ok:
                order_failures[key] = order
    decidable = {k: v for k, v in order_rows.items() if np.isfinite(v["order"])}
    if order_failures:
        c9_status = "failed"
    elif len(decidable) < len(ORDER_FAMILIES) * len(ORDER_CHECKPOINTS):
        c9_status = "not_evaluated"
    else:
        c9_status = "passed"
    gates["C9_design_order"] = {
        "criterion": (f"observed spatial order over n={list(ORDER_CELLS)} reaches "
                      f"{ORDER_SLACK:g} * (p + 1) for every degree in "
                      f"{list(ORDER_FAMILIES)} at t in {list(ORDER_CHECKPOINTS)}"),
        "measured": min((v["order"] for v in decidable.values()), default=float("nan")),
        "worst_at": (min(decidable, key=lambda k: decidable[k]["order"])
                     if decidable else None),
        "per_case": order_rows,
        "failures": order_failures,
        "passed": c9_status == "passed",
        "status": c9_status,
    }

    # -- C7x: informational spread / amplification prediction. ----------------
    gates["C7x_design_space_spread"] = design_space_spread(records)
    gates["C7x_amplification_prediction"] = amplification(records)

    # -- C11: provenance of the evidence bundle. ------------------------------
    # C6 counts files; a stale file counts too.  Nine of the ten gates below are
    # computed from records that were measured against an *exact solution*, so
    # the reference used is part of the claim: this gate refuses to let a record
    # of unknown provenance stand in for a verified one unless a replay
    # certificate reproduces it against the current reference.
    audit = provenance_audit(records, certificates)
    statuses = {k: v["status"] for k, v in audit["rows"].items()}
    unverified = sorted(k for k, v in statuses.items() if v == "unverified")
    mismatched = sorted(k for k, v in statuses.items() if v == "config_mismatch")
    rejected = sorted(k for k, v in statuses.items()
                      if v == "certificate_rejected")
    replayed = {k: audit["rows"][k]["certificate"]["replay_delta"]
                for k, v in statuses.items() if v == "replayed"}
    if mismatched or rejected:
        c11_status = "failed"
    elif unverified or audit["present"] == 0:
        c11_status = "not_evaluated"
    else:
        c11_status = "passed"
    gates["C11_provenance_integrity"] = {
        "criterion": (
            "every record on disk matches its declared configuration (physics "
            "fields) and either records the reference model it was measured "
            "against or carries a replay certificate that reproduces its errors "
            f"to {PROVENANCE_REPLAY_TOLERANCE:g} against the aggregated model"),
        "measured": (max(replayed.values()) if replayed else 0.0),
        "worst_at": (max(replayed, key=replayed.get) if replayed else None),
        "counts": audit["counts"],
        "present": audit["present"],
        "replayed": replayed,
        "unverified": unverified,
        "config_mismatch": mismatched,
        "certificate_rejected": rejected,
        "certificates_on_disk": audit["certificates_used"],
        "passed": c11_status == "passed",
        "status": c11_status,
    }

    gates["informational"] = ["C1x_crosscheck_raw", "C7x_design_space_spread",
                              "C7x_amplification_prediction"]
    acceptance = {k: v for k, v in gates.items()
                  if isinstance(v, dict) and k not in gates["informational"]}
    gates["all_passed"] = all(v["passed"] for v in acceptance.values())
    not_evaluated = [k for k, v in acceptance.items() if v.get("status") == "not_evaluated"]
    failed = [k for k, v in acceptance.items() if v.get("status") == "failed"]
    gates["not_evaluated"] = not_evaluated
    gates["failed"] = failed
    gates["verdict"] = ("PASS" if gates["all_passed"]
                        else ("FAIL" if failed else
                              ("INCOMPLETE" if not_evaluated else "FAIL")))
    return {"gates": gates, "families": {k: len(v) for k, v in families.items()}}


def aggregate(records: list, exact: dict, output_dir: Path) -> dict:
    models = {r.get("reference_model", REFERENCE_MIN_IMAGE) for r in records}
    if len(models) > 1:
        raise SystemExit(
            "results/ mixes reference models "
            f"{sorted(models)}: the published numbers must be measured against one "
            "reference, use a separate --output-dir per model")
    model = models.pop() if models else REFERENCE_MODEL
    certificates = load_provenance_certificates(output_dir)
    evaluation = evaluate_gates(records, exact, certificates)
    provenance = provenance_audit(records, certificates)
    metrics = {
        "case": "EULER-VORTEX-ERR",
        "ledger_slug": "CASE-DG-EULER-VORTEX-ERR",
        "reference": (
            "Case-pool entry EULER-VORTEX-ERR cites Yee, Sandham & Djomehri (2000) "
            "with a <6 % tolerance; the analytic isentropic-vortex solution itself "
            "is the standard form documented in Shu (1998) sec. 4.4.2 "
            "(domain [-5,5]^2, beta=5, u_inf=(1,1), gamma=1.4, periodic). The "
            "tolerance is taken from the case-pool entry verbatim and is NOT "
            "re-derived from the paper text (the paper is not available in-repo); "
            "the acceptance numbers below are therefore measured against the "
            "analytic solution, which is independent of which paper introduced it."),
        "tolerance_relative_l2": TOLERANCE,
        "reference_model": {
            "period": list(PERIOD),
            "model": model,
            "images_per_checkpoint": {
                r["task_key"]: r.get("reference_images")
                for r in records if r.get("reference_images") is not None},
            "note": ("both solvers close their faces periodically, so the exact "
                     "solution is evaluated through the periodic map; "
                     "`min_image` folds the displacement onto the nearest image "
                     "only and drops the neighbouring tails, `lattice` sums them "
                     "(exact up to O(tail^2)) -- see "
                     "docs/CASE-DG-EULER-VORTEX-ERR-reference-floor.md"),
            "truncated_images": (
                "min_image: next image shell at radial distance >= box - |shift| "
                ">= 5 -> envelope ratio ~6e-6 -> measured 4e-13 relative L2 in "
                "density/pressure (second order in the envelope) but 1.5e-6 in "
                "velocity (first order, displacement-weighted)"),
        },
        "reference_defect_percent": reference_defect_sweep(
            sorted({r["final_time"] for r in records})),
        "ladder": list(LADDER),
        "ladder_completeness": ladder_completeness(records),
        "provenance": {
            "certificates_file": PROVENANCE_CERTIFICATES,
            "replay_tolerance": PROVENANCE_REPLAY_TOLERANCE,
            "counts": provenance["counts"],
            "rows": provenance["rows"],
            "note": ("`recorded` = the result file carries the reference model it "
                     "was measured against; `replayed` = legacy record (no model "
                     "in the file) certified by a re-run of the same checkpoint "
                     "against the current reference; `unverified` = legacy record "
                     "with no certificate, i.e. its numbers are not shown to "
                     "belong to the current reference"),
        },
        "exact_invariants": exact,
        "gates": evaluation["gates"],
        "family_count": evaluation["families"],
        "records": [
            {
                "task_key": r["task_key"],
                "family": r["family"],
                "config": r["config"],
                "final_time": r["final_time"],
                "cfl": r["cfl"],
                "steps": r["steps"],
                "errors": r["errors"],
                "max_percent": r["max_percent"],
                "invariant_drift": r["invariant_drift"],
                "entropy_drift": r["entropy_drift"],
                "entropy_production": r["entropy_production"],
                "density_min_relative_error": r["density_min_relative_error"],
                "initial_invariants": r.get("initial", {}).get("invariants"),
                "final_invariants": r.get("final", {}).get("invariants"),
                "gate_pass": r["gate_pass"],
                "wall_seconds": r["wall_seconds"],
            }
            for r in sorted(records, key=lambda r: (r["family"], r["final_time"]))
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "error_conservation.csv").write_text(_csv(records))
    (output_dir / "summary.md").write_text(_markdown(records, metrics, exact))
    return metrics


def _csv(records: list) -> str:
    header = ("family,final_time,steps,max_percent,mass_drift,momentum_x_drift,"
              "momentum_y_drift,energy_drift,entropy_drift,density_min_relative_error\n")
    lines = []
    for r in sorted(records, key=lambda r: (r["family"], r["final_time"])):
        d = r["invariant_drift"]
        lines.append(
            f"{r['family']},{r['final_time']:g},{r['steps']},{r['max_percent']:.6f},"
            f"{d['mass']:.3e},{d['momentum_x']:.3e},{d['momentum_y']:.3e},"
            f"{d['energy']:.3e},{r['entropy_drift']:.3e},"
            f"{r['density_min_relative_error']:.6e}\n")
    return header + "".join(lines)


def _markdown(records: list, metrics: dict, exact: dict) -> str:
    gates = metrics["gates"]

    def flag(name):
        return "通过" if gates[name]["passed"] else "未通过"

    lines = [
        "# CASE-EULER-VORTEX-ERR：等熵涡移动的误差守恒",
        "",
        "本文件由 `benchmarks/case_euler_vortex_errconserve.py --aggregate` 从 "
        "`results/*.json` 程序化生成，数字零手抄。",
        "",
        f"参考解模型：`{metrics['reference_model'].get('model', REFERENCE_MODEL)}`"
        + ("" if metrics['reference_model'].get('model', REFERENCE_MODEL)
           == REFERENCE_MIN_IMAGE else
           "（格点镜像求和；每检查点的像半宽见 `metrics.json`）"),
        "",
        "## 门禁",
        "",
        "| 判据 | 内容 | 实测 | 判定 |",
        "|---|---|---|---|",
        f"| C1 容差（t={T_REF:g} 快照，各族） | 各族在 t={T_REF:g} 的最大场相对 L2 "
        f"< {TOLERANCE:.0%} | {gates['C1_tolerance']['measured']:.4%}"
        f"（{gates['C1_tolerance']['worst_at']}） | {flag('C1_tolerance')} |",
        f"| C1x 长时原始值（仅参考） | 全部长时检查点最大场 vs {TOLERANCE:.0%} | "
        f"{gates['C1x_crosscheck_raw']['measured']:.4%}"
        f"（{gates['C1x_crosscheck_raw']['worst_at']}） | 不作判据 |",
        f"| C2 误差守恒 | alpha <= {MAX_GROWTH_EXPONENT}"
        f"（{gates['C2_error_conservation']['checkpoint_count']} 个检查点） | alpha "
        f"{gates['C2_error_conservation']['exponent']:.3f}，"
        f"e(t_last)/e(t={T_REF:g})="
        f"{gates['C2_error_conservation']['growth_ratio']:.3f}"
        f"（参考），严格目标 alpha<={STRICT_ALPHA_TARGET} "
        f"{'通过' if gates['C2_error_conservation']['strict_target']['passed'] else '未达'} | "
        f"{flag('C2_error_conservation')} |",
        f"| C5 交叉校核收敛阶 | P0 在 t=5 的观测阶 >= 0.8（n=64/128/256） | "
        f"{gates['C5_crosscheck_order']['order']:.3f} | "
        f"{flag('C5_crosscheck_order')} |",
        f"| C3 离散守恒 | 质量/动量/能量漂移（分母 = max(\\|参考值\\|, mass*\\|u_inf\\|)，"
        f"截面动量参考值为零时不失效）< {INVARIANT_TOLERANCE:g} | "
        f"{gates['C3_discrete_conservation']['worst_drift']:.3e} | "
        f"{flag('C3_discrete_conservation')} |",
        f"| C4 熵预算 | \\|dS\\|/S(0) < {ENTROPY_TOLERANCE:g} | "
        f"{gates['C4_entropy_budget']['measured']:.3e} | "
        f"{flag('C4_entropy_budget')} |",
        f"| C6 阶梯完整性 | LADDER 声明的 {gates['C6_ladder_completeness']['declared']} "
        f"个检查点全部有结果文件 | 已有 {gates['C6_ladder_completeness']['present']} 个，"
        f"缺失 {gates['C6_ladder_completeness']['missing'] or '无'}"
        + (f"（已递交 {'/'.join(sorted(set(gates['C6_ladder_completeness']['pending_jobs'].values())))}，"
           "未落地 => 判据 INCOMPLETE）"
           if gates['C6_ladder_completeness']['pending_jobs'] else "") + " | "
        f"{flag('C6_ladder_completeness')} |",
        f"| C7 设计空间不变性 | 每个有 >= 3 检查点的 DG 设计点 "
        f"(网格 x CFL) 都满足 alpha <= {MAX_GROWTH_EXPONENT}，且至少 "
        f"{MIN_DECIDABLE_DESIGN_FAMILIES} 个设计点可判 | 可判 "
        f"{len(gates['C7_design_space_invariance']['evaluated_families'])} 个族，"
        f"最大 alpha {gates['C7_design_space_invariance']['measured']:.3f}"
        f"（{gates['C7_design_space_invariance']['worst_at']}） | "
        f"{flag('C7_design_space_invariance')} |",
        "",
        "",
        f"| C8 阶数不变性 | {list(DEGREE_AXIS)} 中每个有 >= 3 检查点的阶数都满足 "
        f"alpha <= {MAX_GROWTH_EXPONENT}，且至少 {MIN_DECIDABLE_DEGREES} 个阶数可判 | 可判 "
        f"{gates['C8_degree_invariance']['degrees_evaluated']}，最大 alpha "
        f"{gates['C8_degree_invariance']['measured']:.3f}"
        f"（p={gates['C8_degree_invariance']['worst_at']}） | "
        f"{flag('C8_degree_invariance')} |",
        f"| C9 设计阶 | n={list(ORDER_CELLS)} 的观测空间阶达到 "
        f"{ORDER_SLACK:g}*(p+1)，对 p in {list(ORDER_FAMILIES)}、"
        f"t in {list(ORDER_CHECKPOINTS)} 全部成立 | 最差 {gates['C9_design_order']['measured']:.3f}"
        f"（{gates['C9_design_order']['worst_at']}） | "
        f"{flag('C9_design_order')} |",
        f"| C10 方向不变性 | 网格 x CFL x 阶数固定、只换对流方向（同速 "
        f"|u_inf|={DIRECTION_SPEED:.4f}），每个可判方向的 alpha <= "
        f"{MAX_GROWTH_EXPONENT}，且至少 {MIN_DECIDABLE_DIRECTIONS} 个方向可判、"
        f"其中至少一个是旋转方向 | 可判 "
        f"{gates['C10_direction_invariance']['directions_evaluated']}，最大 alpha "
        f"{gates['C10_direction_invariance']['measured']:.3f}"
        f"（{gates['C10_direction_invariance']['worst_at']}） | "
        f"{flag('C10_direction_invariance')} |",
        f"| C11 证据溯源 | 每个结果文件与其声明的配置（任务/阶数/网格/CFL/beta/方向）"
        f"一致，且文件里记了所用参考解模型、或有重放证书把同一检查点对照当前模型复现"
        f"到 {PROVENANCE_REPLAY_TOLERANCE:g} | 记录 "
        f"{gates['C11_provenance_integrity']['counts'].get('recorded', 0)}，"
        f"重放证书 {gates['C11_provenance_integrity']['counts'].get('replayed', 0)}，"
        f"未证实 {len(gates['C11_provenance_integrity']['unverified'])}，"
        f"配置矛盾 {len(gates['C11_provenance_integrity']['config_mismatch'])}，"
        f"证书被拒 {len(gates['C11_provenance_integrity']['certificate_rejected'])}"
        + (f"（未证实：{', '.join(gates['C11_provenance_integrity']['unverified'][:4])}"
           + ("…" if len(gates['C11_provenance_integrity']['unverified']) > 4 else "") + "）"
           if gates['C11_provenance_integrity']['unverified'] else "") + " | "
        f"{flag('C11_provenance_integrity')} |",
        "",
        "## 阶数轴（C8）与设计阶（C9）",
        "",
        "| 阶数 | 族 | 检查点数 | t_max | alpha | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for degree, row in gates["C8_degree_invariance"]["per_degree"].items():
        lines.append(
            f"| p={degree} | {row['family']} | {row['checkpoints']} | "
            f"{row['final_time_max'] if row['final_time_max'] is not None else '-'} | "
            f"{row['exponent']:.3f} | {'通过' if row['passed'] else '未通过'} |")
    lines += [
        "",
        "| 判据点 | 阶数 | 网格 | 观测阶 | 目标 | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for key, row in gates["C9_design_order"]["per_case"].items():
        lines.append(
            f"| {key} | p={row['degree']} | {row['cells']} | {row['order']:.3f} | "
            f">= {row['target']:.1f} | {'通过' if row['passed'] else '未通过'} |")
    lines += [
        "",
        "## 设计空间（C7：网格轴 / CFL 轴 / 阶数轴，一轮动一个）",
        "",
        "| 设计点 | 检查点数 | CFL | alpha | t_max | 判定 |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in gates["C7_design_space_invariance"]["per_family"].items():
        lines.append(
            f"| {name} | {row['checkpoints']} | "
            f"{'/'.join(f'{c:g}' for c in row['cfl'])} | {row['exponent']:.3f} | "
            f"{row['final_time_max']:g} | {'通过' if row['passed'] else '未通过'} |")
    spread = gates["C7x_design_space_spread"]
    amp = gates["C7x_amplification_prediction"]
    lines += [
        "",
        f"C7x（仅登记）：t = {BOX_CROSSING_TIME:g} 的最大场误差 —— "
        + "，".join(f"{k} {v:.4f}%" for k, v in
                    sorted(spread["e_at_crossing_percent"].items()))
        + f"；CFL 0.4/0.2 比 {spread['cfl_ratio_0p4_over_0p2']:.3f}，"
        f"网格 64/32 比 {spread['refinement_ratio_n64_over_n32']:.3f}。"
        f"放大因子预测（{AMPLIFICATION_PREDICTION}）："
        + ("支持" if amp["supported"] else
           ("被否定" if amp["supported"] is False else "未评估"))
        + ("（" + "，".join(f"{k} A={v['amplification']:.3f}"
                            for k, v in sorted(amp["values"].items())) + "）"
           if amp["values"] else ""),
        "",
        "## 方向轴（C10）",
        "",
        "| 标签 | u_inf | 角度 | 族 | 检查点数 | t_max | alpha | 判定 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, row in gates["C10_direction_invariance"]["per_direction"].items():
        lines.append(
            f"| {label} | ({row['background_velocity'][0]:.4f}, "
            f"{row['background_velocity'][1]:.4f}) | {row['angle_deg']:.1f} deg | "
            f"`{row['family']}` | {row['checkpoints']} | "
            f"{row['final_time_max'] if row['final_time_max'] is not None else '-'} | "
            f"{row['exponent']:.3f} | {'通过' if row['passed'] else '未通过'} |")
    lines += [
        "",
        f"三个方向的速度模相同（|u_inf| = {DIRECTION_SPEED:.4f}），因此该轴只移动方向、"
        "不移动输运长度：任何 alpha 之差都归因于方向，而不是行程。",
        "",
        "## 误差随时间（族内逐检查点）",
        "",
        "| 族 | t | 步数 | rho % | u % | p % | 最大场 % | 族 alpha |",
        "|---|---|---|---|---|---|---|---|",
    ]
    families = _group(records)
    for name, rows in families.items():
        exponent = _fit_exponent([r["final_time"] for r in rows],
                                 [r["max_relative_l2"] for r in rows])
        for r in rows:
            e = r["errors"]
            lines.append(
                f"| {name} | {r['final_time']:g} | {r['steps']} | "
                f"{100 * e['density']:.4f} | {100 * e['velocity_x']:.4f} | "
                f"{100 * e['pressure']:.4f} | {r['max_percent']:.4f} | "
                f"{exponent:.3f} |")
    lines += [
        "",
        "## 守恒与熵",
        "",
        "| 配置 | t | 质量漂移 | x 动量漂移 | 能量漂移 | 熵漂移 | 核密度相对误差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: (r["family"], r["final_time"])):
        d = r["invariant_drift"]
        lines.append(
            f"| {r['task_key']} | {r['final_time']:g} | {d['mass']:.2e} | "
            f"{d['momentum_x']:.2e} | {d['energy']:.2e} | {r['entropy_drift']:.2e} | "
            f"{r['density_min_relative_error']:.3e} |")
    lines += [
        "",
        "精确解初始不变量（320 点 Gauss 每方向）：",
        "",
        "| 量 | 值 |",
        "|---|---|",
        f"| 质量 | {exact['mass']:.12f} |",
        f"| x 动量 | {exact['momentum_x']:.12f} |",
        f"| 能量 | {exact['energy']:.12f} |",
        f"| 熵 S = ∫p·rho^-gamma（应等于面积 {BOX_AREA:g}）| {exact['entropy']:.12f} |",
        f"| 核最小密度 | {exact['density_min']:.9f} |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_figures(records: list, output_dir: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    families = _group(records)
    figures = []

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for name, rows in sorted(families.items()):
        times = [r["final_time"] for r in rows]
        values = [100.0 * r["max_relative_l2"] for r in rows]
        ax.loglog(times, values, "o-", label=name)
    ax.axhline(100.0 * TOLERANCE, color="crimson", ls="--",
               label=f"tolerance {TOLERANCE:.0%} (Yee et al. 2000 / case pool)")
    ax.set_xlabel("dimensionless time t  (|u_inf|t = travel distance, period 10)")
    ax.set_ylabel("max primitive-field relative L2 error [%]")
    ax.set_title("Isentropic vortex, long-time transport: does the error accumulate?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_error_vs_time.png", dpi=150)
    plt.close(fig)
    figures.append({"name": "fig_error_vs_time.png",
                    "caption": "各离散族最大场相对 L2 误差随无量纲时间的变化（双对数轴，"
                               "斜率即累积指数 alpha）；红虚线为算例池 6% 容差。"})

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for name, rows in sorted(families.items()):
        times = [r["final_time"] for r in rows]
        axes[0].semilogy(times, [max(r["invariant_drift"]["mass"], 1e-18) for r in rows],
                         "o-", label=f"{name} mass")
        axes[0].semilogy(times, [max(r["invariant_drift"]["energy"], 1e-18) for r in rows],
                         "s--", label=f"{name} energy")
        axes[1].semilogy(times, [max(r["entropy_drift"], 1e-18) for r in rows],
                         "o-", label=f"{name} |dS|/S")
        axes[1].semilogy(times, [r["density_min_relative_error"] for r in rows],
                         "s--", label=f"{name} core-density error")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("relative drift of a global invariant")
    axes[0].set_title("C3 discrete conservation (periodic interface fluxes telescope)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("relative measure")
    axes[1].set_title("C4 entropy budget and core-density dissipation")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_invariants_vs_time.png", dpi=150)
    plt.close(fig)
    figures.append({"name": "fig_invariants_vs_time.png",
                    "caption": "左：C3 质量/能量守恒漂移（应处于舍入水平）；"
                               "右：C4 熵泛函漂移与涡核最小密度的相对误差随时间的增长。"})

    # -- design space (C7): one axis moved at a time --------------------------
    design = {name: rows for name, rows in families.items() if name.startswith("dg_d")}
    if design:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
        panels = (
            (axes[0], "dg_d2_n32_cfl0.2", "dg_d2_n64_cfl0.2",
             "mesh axis (CFL 0.2): 32^2 vs 64^2"),
            (axes[1], "dg_d2_n32_cfl0.2", "dg_d2_n32_cfl0.4",
             "time-step axis (32^2): CFL 0.2 vs 0.4"),
        )
        for ax, base, other, title in panels:
            for name in (base, other):
                rows = design.get(name)
                if not rows:
                    continue
                exponent = _fit_exponent([r["final_time"] for r in rows],
                                         [r["max_relative_l2"] for r in rows])
                label = f"{name} (alpha={exponent:.3f})" if np.isfinite(exponent) else name
                ax.loglog([r["final_time"] for r in rows],
                          [100.0 * r["max_relative_l2"] for r in rows], "o-", label=label)
            ax.axhline(100.0 * TOLERANCE, color="crimson", ls="--",
                       label=f"tolerance {TOLERANCE:.0%}")
            ax.set_xlabel("time t")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, loc="lower right")
        axes[0].set_ylabel("max primitive-field relative L2 error [%]")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_design_space.png", dpi=150)
        plt.close(fig)
        figures.append({"name": "fig_design_space.png",
                        "caption": "C7 设计空间：左=网格轴（32² vs 64²，同 CFL 0.2），"
                                   "右=时间步轴（同 32²，CFL 0.2 vs 0.4）；"
                                   "两条直线的斜率即各自的累积指数 alpha，"
                                   "误差守恒（alpha <= 1）在两条轴上都必须成立。"})
    return figures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def select_shard(index: int, total: int) -> list:
    """Configurations of shard ``index`` out of ``total``.

    The comparison uses ``index % total`` so that *both* 0-based (``0..n-1``) and
    1-based (``1..n``) shard conventions used by different schedulers partition
    the ladder exactly once -- neither convention can silently drop config 0.
    """
    if total < 1:
        raise ValueError("total must be >= 1")
    if index < 0:
        raise ValueError("index must be non-negative")
    return [config for position, config in enumerate(LADDER)
            if position % total == index % total]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("dg", "p0"), default=None)
    parser.add_argument("--cells", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=None)
    parser.add_argument("--strength", type=float, default=STRENGTH)
    parser.add_argument("--output-dir", default="case_euler_vortex_errconserve")
    parser.add_argument("--shard", default=None, help="i/n: run every n-th ladder entry")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--direction", choices=[label for label, _ in DIRECTION_AXIS],
                        default=DIRECTION_BASELINE,
                        help="advection direction of the C10 axis (speed is held "
                             "fixed at |u_inf| = sqrt(2) by construction)")
    parser.add_argument("--reference", choices=REFERENCE_MODELS,
                        default=REFERENCE_MODEL,
                        help="periodic reference: min_image fold (published "
                             "through round 6) or the exact lattice sum")
    args = parser.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    output_dir = Path(args.output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    if args.task is not None:
        config = {"task": args.task, "cells": args.cells or 32,
                  "final_time": args.final_time if args.final_time is not None else 1.0,
                  "cfl": args.cfl if args.cfl is not None else (0.2 if args.task == "dg" else 0.4),
                  "strength": args.strength}
        config["reference"] = args.reference
        config["background_velocity"] = dict(DIRECTION_AXIS)[args.direction]
        if args.task == "dg":
            config["degree"] = args.degree if args.degree is not None else 2
        record = run_task(config)
        path = output_dir / "results" / f"{record['task_key']}.json"
        path.write_text(json.dumps(record, indent=2))
        print(f"{record['task_key']}: max {record['max_percent']:.4f}% "
              f"({record['wall_seconds']:.1f} s) -> {path}")
        return 0

    if args.shard:
        index, total = (int(part) for part in args.shard.split("/"))
        for config in select_shard(index, total):
            config = dict(config, reference=args.reference)
            path = output_dir / "results" / f"{task_key(config)}.json"
            if path.exists():
                print(f"skip existing {path.name}", flush=True)
                continue
            record = run_task(config)
            path = output_dir / "results" / f"{record['task_key']}.json"
            path.write_text(json.dumps(record, indent=2))
            print(f"{record['task_key']}: max {record['max_percent']:.4f}% "
                  f"({record['wall_seconds']:.1f} s) -> {path}", flush=True)
        return 0

    # aggregate whatever checkpoints exist
    records = []
    for path in sorted((output_dir / "results").glob("*.json")):
        records.append(json.loads(path.read_text()))
    if not records:
        raise SystemExit(f"no records in {output_dir / 'results'}")
    exact = exact_invariants()
    exact["core_density"] = exact_core_density()
    metrics = aggregate(records, exact, output_dir)
    if args.figures:
        make_figures(records, output_dir)
    print(json.dumps(metrics["gates"], indent=2))
    return 0 if metrics["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())