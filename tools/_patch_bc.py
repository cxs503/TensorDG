"""One-shot patch: add wall / general ghost-state boundary conditions to
``tensordg/euler_dg_2d.py`` (mission m16803).

Run once: ``python tools/_patch_bc.py`` (idempotent, refuses to patch twice).
"""
from __future__ import annotations

from pathlib import Path

TARGET = Path("tensordg/euler_dg_2d.py")
text = TARGET.read_text()

MARK = "# --- m16803 boundary conditions ---"
if MARK in text:
    raise SystemExit("already patched")

OLD_FLUXES = '''def _axis_fluxes(coeff, basis, signs, weights, periodic, axis):
    """Integrated interface fluxes ``(F_right, F_left)`` per cell.

    ``F_right[x, y, l, v]`` (x direction) is the numerical flux of the face
    between cell ``x`` and cell ``x+1``, projected onto the test functions of
    the remaining direction.  Non-periodic boundaries use the interior trace on
    both sides, which makes HLLC collapse to the physical flux (outflow).
    """
    trace_right, trace_left = _traces(coeff, basis, signs, axis)
    if axis == 0:
        # ``neighbour[x]`` is the *left* trace of the cell to the right of
        # face ``x``; the roll is required for both the periodic and the
        # outflow case (m2400: omitting it made HLLC see the cell's own state
        # and silently degrade to the physical flux).
        neighbour = torch.roll(trace_left, shifts=-1, dims=0)
        if not periodic:
            neighbour[-1] = trace_right[-1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xybv,b,bl->xylv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=0)
        if not periodic:
            outer = _hllc(trace_left[:1], trace_left[:1], axis)
            flux_left[0] = torch.einsum("ybv,b,bl->ylv", outer[0], weights, basis)
    else:
        neighbour = torch.roll(trace_left, shifts=-1, dims=1)
        if not periodic:
            neighbour[:, -1] = trace_right[:, -1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xyav,a,ak->xykv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=1)
        if not periodic:
            outer = _hllc(trace_left[:, :1], trace_left[:, :1], axis)
            flux_left[:, 0] = torch.einsum(
                "xav,a,ak->xkv", outer[:, 0], weights, basis
            )
    return flux_right, flux_left
'''

NEW_FLUXES = '''# --- m16803 boundary conditions ---
#
# ``solve_euler_dg_2d`` historically exposed only periodic faces and a
# zero-gradient ("outflow") outer face, which made the Woodward--Colella double
# Mach reflection impossible: the reflecting floor and the tracked post-shock
# inflow have no representation.  The boundary is now described per face as
#
#   * ``"outflow"`` / ``None`` -- interior trace on both sides (old behaviour),
#   * ``"wall"`` -- impermeable inviscid slip wall, implemented with the exact
#     wall flux ``(0, p, 0, 0)`` normal to the face (no mass, momentum or
#     energy leak; free-stream preservation is exact, not O(M^2)),
#   * ``callable(X, Y, t) -> state`` or ``-> (state, wall_mask)`` -- a general
#     ghost state given at the face quadrature points ``X, Y`` (physical
#     coordinates, shapes ``(n_face, q)``) and the stage time ``t``.  ``state``
#     is a *conservative* ``(rho, rho u, rho v, E)`` array broadcastable to
#     ``(n_face, q, 4)``; a ``True`` entry in ``wall_mask`` switches that point
#     to the exact wall flux.  ``conservative_state`` builds the array from
#     primitive variables.  This covers supersonic inflow, tracked moving
#     post-shock states and piecewise wall/inflow segments on one face.


def conservative_state(rho, velocity_x, velocity_y, pressure):
    """Conservative variables ``(rho, rho u, rho v, E)`` from primitives."""
    return _conservative(rho, velocity_x, velocity_y, pressure)


def _wall_flux(state, axis):
    """Exact inviscid slip-wall flux: only the normal pressure does work."""
    pressure = _pressure(state).clamp_min(0.0)
    flux = torch.zeros_like(state)
    flux[..., 1 if axis == 0 else 2] = pressure
    return flux


def _face_points(geom, axis, high):
    """Physical coordinates of the quadrature points of a boundary face."""
    nodes = geom["nodes"]
    count = nodes.numel()
    if axis == 0:
        x_face = geom["xmax"] if high else geom["xmin"]
        X = torch.full((geom["ny"], count), x_face, device=nodes.device, dtype=nodes.dtype)
        Y = geom["y"][:, None] + 0.5 * geom["dy"] * nodes[None, :]
    else:
        y_face = geom["ymax"] if high else geom["ymin"]
        X = geom["x"][:, None] + 0.5 * geom["dx"] * nodes[None, :]
        Y = torch.full((geom["nx"], count), y_face, device=nodes.device, dtype=nodes.dtype)
    return X, Y


def _ghost_state(bc, interior, X, Y, time, axis):
    """Exterior trace of a boundary face and its optional wall mask."""
    if bc is None or bc == "outflow":
        return interior, None
    if bc == "wall":
        ghost = interior.clone()
        ghost[..., 1 if axis == 0 else 2] = -ghost[..., 1 if axis == 0 else 2]
        return ghost, torch.ones(interior.shape[:-1], device=interior.device, dtype=torch.bool)
    if bc == "periodic":  # pragma: no cover - guarded in solve_euler_dg_2d
        raise ValueError("a periodic axis must not carry a boundary condition")
    out = bc(X, Y, time)
    wall = None
    if isinstance(out, tuple):
        out, wall = out
    state = torch.as_tensor(out, device=interior.device, dtype=interior.dtype)
    state = torch.broadcast_to(state, interior.shape)
    if wall is not None:
        wall = torch.as_tensor(wall, device=interior.device, dtype=torch.bool)
        wall = torch.broadcast_to(wall, interior.shape[:-1])
    return state, wall


def _boundary_flux(bc, interior, weights, basis, geom, axis, high, time):
    """Integrated numerical flux through one boundary face."""
    X, Y = _face_points(geom, axis, high)
    ghost, wall = _ghost_state(bc, interior, X, Y, time, axis)
    if high:
        flux = _hllc(interior, ghost, axis)
    else:
        flux = _hllc(ghost, interior, axis)
    if wall is not None:
        flux = torch.where(wall.unsqueeze(-1), _wall_flux(interior, axis), flux)
    if axis == 0:
        return torch.einsum("ybv,b,bl->ylv", flux, weights, basis)
    return torch.einsum("xav,a,ak->xkv", flux, weights, basis)


def _axis_fluxes(coeff, basis, signs, weights, periodic, axis, geom, stage_time=0.0):
    """Integrated interface fluxes ``(F_right, F_left)`` per cell.

    ``F_right[x, y, l, v]`` (x direction) is the numerical flux of the face
    between cell ``x`` and cell ``x+1``, projected onto the test functions of
    the remaining direction.  Non-periodic boundaries take their exterior trace
    from ``geom["bcs"]`` (default ``None`` = outflow, i.e. the interior trace on
    both sides, which makes HLLC collapse to the physical flux).
    """
    trace_right, trace_left = _traces(coeff, basis, signs, axis)
    if axis == 0:
        # ``neighbour[x]`` is the *left* trace of the cell to the right of
        # face ``x``; the roll is required for both the periodic and the
        # outflow case (m2400: omitting it made HLLC see the cell's own state
        # and silently degrade to the physical flux).
        neighbour = torch.roll(trace_left, shifts=-1, dims=0)
        if not periodic:
            neighbour[-1] = trace_right[-1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xybv,b,bl->xylv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=0)
        if not periodic:
            bcs = geom["bcs"]
            flux_left[0] = _boundary_flux(
                bcs.get("left"), trace_left[0], weights, basis, geom, 0, False, stage_time
            )
            flux_right[-1] = _boundary_flux(
                bcs.get("right"), trace_right[-1], weights, basis, geom, 0, True, stage_time
            )
    else:
        neighbour = torch.roll(trace_left, shifts=-1, dims=1)
        if not periodic:
            neighbour[:, -1] = trace_right[:, -1]
        interface = _hllc(trace_right, neighbour, axis)
        flux_right = torch.einsum("xyav,a,ak->xykv", interface, weights, basis)
        flux_left = torch.roll(flux_right, shifts=1, dims=1)
        if not periodic:
            bcs = geom["bcs"]
            flux_left[:, 0] = _boundary_flux(
                bcs.get("bottom"), trace_left[:, 0], weights, basis, geom, 1, False, stage_time
            )
            flux_right[:, -1] = _boundary_flux(
                bcs.get("top"), trace_right[:, -1], weights, basis, geom, 1, True, stage_time
            )
    return flux_right, flux_left


def normalise_bcs(bcs):
    """Validate and normalise the ``bcs`` mapping of ``solve_euler_dg_2d``."""
    if bcs is None:
        return {}
    if not isinstance(bcs, dict):
        raise ValueError("bcs must be a dict keyed by 'left'/'right'/'bottom'/'top'")
    allowed = {"left", "right", "bottom", "top"}
    unknown = set(bcs) - allowed
    if unknown:
        raise ValueError(f"unknown boundary faces {sorted(unknown)}")
    for face, value in bcs.items():
        if value is None or value in ("outflow", "wall"):
            continue
        if not callable(value):
            raise ValueError(
                f"boundary {face!r} must be 'outflow', 'wall', None or a callable"
            )
    return {face: value for face, value in bcs.items() if value not in (None, "outflow")}
# --- end m16803 boundary conditions ---
'''

assert OLD_FLUXES in text
text = text.replace(OLD_FLUXES, NEW_FLUXES)

OLD_RHS_SIG = '''def _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic):
    """DG residual of the 2-D Euler equations on a Cartesian mesh."""'''
NEW_RHS_SIG = '''def _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic, geom,
         stage_time=0.0):
    """DG residual of the 2-D Euler equations on a Cartesian mesh."""'''
assert OLD_RHS_SIG in text
text = text.replace(OLD_RHS_SIG, NEW_RHS_SIG)

OLD_CALLS = '''    right_x, left_x = _axis_fluxes(coeff, basis, signs, weights, periodic[0], 0)
    right_y, left_y = _axis_fluxes(coeff, basis, signs, weights, periodic[1], 1)'''
NEW_CALLS = '''    right_x, left_x = _axis_fluxes(
        coeff, basis, signs, weights, periodic[0], 0, geom, stage_time
    )
    right_y, left_y = _axis_fluxes(
        coeff, basis, signs, weights, periodic[1], 1, geom, stage_time
    )'''
assert OLD_CALLS in text
text = text.replace(OLD_CALLS, NEW_CALLS)

# --- solver signature --------------------------------------------------------
OLD_SIG = '''    quadrature=None,
    initial_projection="l2",'''
NEW_SIG = '''    quadrature=None,
    bcs=None,
    initial_projection="l2",'''
assert OLD_SIG in text
text = text.replace(OLD_SIG, NEW_SIG, 1)

OLD_DOC = '''    periodic : (bool, bool)
        Periodicity in ``x`` and ``y``; non-periodic faces are outflow.'''
NEW_DOC = '''    periodic : (bool, bool)
        Periodicity in ``x`` and ``y``; non-periodic faces are outflow unless
        ``bcs`` overrides them.
    bcs : dict or None
        Boundary conditions for the non-periodic faces, keyed by ``"left"`` /
        ``"right"`` / ``"bottom"`` / ``"top"``.  Each value is ``"outflow"``
        (default), ``"wall"`` (impermeable inviscid slip wall, exact wall flux
        ``(0, p, 0, 0)``) or ``callable(X, Y, t)`` returning a conservative
        ghost state, or ``(state, wall_mask)`` where ``True`` switches that
        point to the wall flux.  ``X``/``Y`` are the physical coordinates of
        the face quadrature points, shape ``(n_face, quadrature)``; ``t`` is the
        RK stage time.  Entries of a periodic axis are ignored.'''
assert OLD_DOC in text
text = text.replace(OLD_DOC, NEW_DOC)

OLD_TAIL = '''    if not periodic:
        bc_low = '''  # not present yet; only used as a guard below
assert OLD_TAIL not in text

OLD_BUILD = '''    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy
'''
NEW_BUILD = '''    x = xmin + (torch.arange(nx, device=dev, dtype=dtype) + 0.5) * dx
    y = ymin + (torch.arange(ny, device=dev, dtype=dtype) + 0.5) * dy

    face_bcs = normalise_bcs(bcs)
    for face, axis in (("left", 0), ("right", 0), ("bottom", 1), ("top", 1)):
        if face in face_bcs and periodic_flags_pre[axis]:
            face_bcs.pop(face)  # periodic axis wins; documented above
    geom = {
        "nodes": nodes,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "nx": nx,
        "ny": ny,
        "bcs": face_bcs,
    }
'''
assert OLD_BUILD in text
text = text.replace(OLD_BUILD, NEW_BUILD)

# ``periodic_flags`` is defined after the projection block today; hoist a copy.
OLD_FLAGS = '''    periodic_flags = (bool(periodic[0]), bool(periodic[1]))

    limit_each_stage'''
NEW_FLAGS = '''    limit_each_stage'''
assert OLD_FLAGS in text
text = text.replace(OLD_FLAGS, NEW_FLAGS)

OLD_PRE = '''    dev = torch.device(device)
    order = quadrature if quadrature is not None else max(degree + 2, 3)'''
NEW_PRE = '''    periodic_flags_pre = (bool(periodic[0]), bool(periodic[1]))
    periodic_flags = periodic_flags_pre
    dev = torch.device(device)
    order = quadrature if quadrature is not None else max(degree + 2, 3)'''
assert OLD_PRE in text
text = text.replace(OLD_PRE, NEW_PRE)

OLD_K1 = '''        k1 = _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic)'''
NEW_K1 = '''        k1 = _rhs(coeff, basis, derivative, weights, signs, dx, dy, periodic, geom, time)'''
assert OLD_K1 in text
text = text.replace(OLD_K1, NEW_K1)

OLD_K2 = '''        k2 = _rhs(u1, basis, derivative, weights, signs, dx, dy, periodic)'''
NEW_K2 = '''        k2 = _rhs(
            u1, basis, derivative, weights, signs, dx, dy, periodic, geom, time + dt
        )'''
assert OLD_K2 in text
text = text.replace(OLD_K2, NEW_K2)

OLD_K3 = '''        k3 = _rhs(u2, basis, derivative, weights, signs, dx, dy, periodic)'''
NEW_K3 = '''        k3 = _rhs(
            u2, basis, derivative, weights, signs, dx, dy, periodic, geom, time + 0.5 * dt
        )'''
assert OLD_K3 in text
text = text.replace(OLD_K3, NEW_K3)

OLD_META = '''        "periodic": tuple(periodic),'''
NEW_META = '''        "periodic": tuple(periodic),
        "bcs": {
            face: (value if isinstance(value, str) else getattr(value, "__name__", "callable"))
            for face, value in face_bcs.items()
        },'''
assert OLD_META in text
text = text.replace(OLD_META, NEW_META)

OLD_ALL = '''    "cell_fields",
    "legendre_operators",
]'''
NEW_ALL = '''    "cell_fields",
    "legendre_operators",
    "conservative_state",
    "normalise_bcs",
]'''
assert OLD_ALL in text
text = text.replace(OLD_ALL, NEW_ALL)

TARGET.write_text(text)
print("patched", TARGET, len(text.splitlines()), "lines")