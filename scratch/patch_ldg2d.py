import pathlib, textwrap

p = pathlib.Path("tensordg/poisson_2d_ldg.py")
src = p.read_text()

# ---------------------------------------------------------------- docstring
old_start = src.index("STATUS: PROTOTYPE -- NOT VALIDATED, NOT EXPORTED.")
old_end = src.index("Original module description follows.")
new_status = '''STATUS: VALIDATED (2-D extension of the verified 1-D LDG closure).
    The Dirichlet closure that carried the structural rank deficiency reported
    in ``docs/ldg_poisson_status.md`` is fixed here with the two ingredients
    derived and verified in one dimension (``tensordg.ldg_1d``,
    ``docs/CASE-DG-LDG-1D-MIXED.md``):

    1. *face-oriented* alternating traces -- on every face the element on the
       lower-index ("-") side supplies ``u_hat`` and the element on the
       higher-index ("+") side supplies ``q_hat`` (``trace_policy="ldg"``);
    2. a *discrete Dirichlet flux* ``q_hat = q_int + beta/h (u_int - g_D)`` on
       boundary faces instead of the bare interior trace ``q_int``.

    With those two the 2-D scheme is algebraically non-singular and converges at
    order ``k + 1`` in the L2 norm for smooth solutions; the h-convergence
    table, the closure ablation, the conditioning audit and the
    polynomial-exactness residual are recorded in ``docs/CASE-DG-LDG-2D.md``
    (artifacts under ``validation/ldg2d/``).  The legacy mirror closure
    ``u_hat = 2 g_D - u_int`` and the bare closure are kept only as ablation
    controls (``boundary_closure="mirror"`` / ``"bare"``) and are *not*
    convergent.

'''
src = src[:old_start] + new_status + src[old_end:]

# ---------------------------------------------------------------- signature
old_sig = '''    exact_grad: tuple[Callable, Callable] | None = None,
) -> Dict[str, object]:'''
new_sig = '''    exact_grad: tuple[Callable, Callable] | None = None,
    trace_policy: str = "ldg",
    boundary_closure: str = "flux",
    dirichlet_flux_beta: float = 4.0,
    return_matrix: bool = False,
) -> Dict[str, object]:'''
assert src.count(old_sig) == 1
src = src.replace(old_sig, new_sig)

old_doc = '''        ``l2_relative_error``, ``grad_relative_error``, ``h``, ``ndofs``,
        ``ncells``, ``degree`` and the element-centre coordinates ``xc``/``yc``.
    """'''
new_doc = '''        ``l2_relative_error``, ``grad_relative_error``, ``h``, ``ndofs``,
        ``ncells``, ``degree`` and the element-centre coordinates ``xc``/``yc``.
        With ``return_matrix=True`` the assembled sparse matrix and right-hand
        side are returned (``matrix``/``rhs``) instead, which is what the rank
        and conditioning audits use.

    trace_policy : {"ldg", "reversed"}
        Interior trace orientation.  ``"ldg"`` (default) is the verified
        face-oriented alternating choice; ``"reversed"`` is the ablation
        control in which ``u_hat`` and ``q_hat`` are taken from the other side.
    boundary_closure : {"flux", "mirror", "bare"}
        Dirichlet-face closure for ``q_hat``.  ``"flux"`` (default) uses the
        discrete flux ``q_int + beta/h (u_int - g_D)``; ``"mirror"`` is the
        legacy prototype closure ``u_hat = 2 g_D - u_int, q_hat = q_int`` and
        ``"bare"`` takes ``u_hat = g_D, q_hat = q_int``.  The last two are
        singular (or catastrophically inaccurate) and exist only as controls.
    dirichlet_flux_beta : float
        Stabilisation strength ``beta > 0`` of the discrete Dirichlet flux; it
        does not affect the convergence order (default 4).
    """'''
assert src.count(old_doc) == 1
src = src.replace(old_doc, new_doc)

# ---------------------------------------------------------------- validation
old_val = '''    if degree < 0:
        raise ValueError("degree must be non-negative")

    default = manufactured_solution()'''
new_val = '''    if degree < 0:
        raise ValueError("degree must be non-negative")
    if trace_policy not in ("ldg", "reversed"):
        raise ValueError("trace_policy must be 'ldg' or 'reversed'")
    if boundary_closure not in ("flux", "mirror", "bare"):
        raise ValueError("boundary_closure must be 'flux', 'mirror' or 'bare'")
    if boundary_closure == "flux" and not dirichlet_flux_beta > 0.0:
        raise ValueError("dirichlet_flux_beta must be positive for the flux closure")

    default = manufactured_solution()'''
assert src.count(old_val) == 1
src = src.replace(old_val, new_val)

# ---------------------------------------------------------------- x faces
x_start = src.index("            # ---- x-faces ---")
x_end = src.index("            # ---- y-faces ---")
new_x = '''            # ---- x-faces ---------------------------------------------------
            for side, sign, mfx in (
                ("left", -1.0, mfx_left),
                ("right", +1.0, mfx_right),
            ):
                nbr = ix - 1 if side == "left" else ix + 1
                if 0 <= nbr < nx:
                    en = nbr * ny + iy
                    # Face-oriented alternating traces: on a face the element on
                    # the lower-index ("-") side contributes u_hat, the element
                    # on the higher-index ("+") side contributes q_hat.  This is
                    # the orientation verified in 1-D (tensordg.ldg_1d,
                    # docs/CASE-DG-LDG-1D-MIXED.md); "reversed" is an ablation.
                    if trace_policy == "ldg":
                        u_elem, q_elem = (en, e) if side == "left" else (e, en)
                    else:
                        u_elem, q_elem = (e, en) if side == "left" else (en, e)
                    _block(e, 1, u_elem, 0, -sign * mfx)
                    _block(e, 0, q_elem, 1, -sign * mfx)
                else:
                    x_face = 0.0 if side == "left" else 1.0
                    y_pts = ((iy + 0.5) + 0.5 * nodes) * hy
                    gvals = np.asarray(
                        dirichlet(np.full_like(y_pts, x_face), y_pts), dtype=float,
                    )
                    endpoint = left if side == "left" else right
                    gf = (0.5 * hy) * np.einsum("q,q,jq->j", weights, gvals, phi)
                    # load vector of the trace: int_face g phi_ij ds
                    trace = np.outer(endpoint, gf).reshape(block)
                    rhs_qx = e * 3 * block + block
                    rhs_u = e * 3 * block
                    if boundary_closure == "mirror":
                        # legacy prototype closure: u_hat = 2 g_D - u_int
                        _block(e, 1, e, 0, sign * mfx)
                        rhs_vec[rhs_qx: rhs_qx + block] += 2.0 * sign * trace
                    else:
                        # u_hat = g_D on the Dirichlet face (flux and bare)
                        rhs_vec[rhs_qx: rhs_qx + block] += sign * trace
                    # q_hat = q_int (+ discrete Dirichlet flux for "flux")
                    _block(e, 0, e, 1, -sign * mfx)
                    if boundary_closure == "flux":
                        beta_x = dirichlet_flux_beta / hx
                        _block(e, 0, e, 0, -sign * beta_x * mfx)
                        rhs_vec[rhs_u: rhs_u + block] += -sign * beta_x * trace

'''
src = src[:x_start] + new_x + src[x_end:]

# ---------------------------------------------------------------- y faces
y_start = src.index("            # ---- y-faces ---")
y_end = src.index("    matrix = sp.coo_matrix(")
new_y = '''            # ---- y-faces ---------------------------------------------------
            for side, sign, mfy in (
                ("left", -1.0, mfy_left),
                ("right", +1.0, mfy_right),
            ):
                nbr = iy - 1 if side == "left" else iy + 1
                if 0 <= nbr < ny:
                    en = ix * ny + nbr
                    if trace_policy == "ldg":
                        u_elem, q_elem = (en, e) if side == "left" else (e, en)
                    else:
                        u_elem, q_elem = (e, en) if side == "left" else (en, e)
                    _block(e, 2, u_elem, 0, -sign * mfy)
                    _block(e, 0, q_elem, 2, -sign * mfy)
                else:
                    y_face = 0.0 if side == "left" else 1.0
                    x_pts = ((ix + 0.5) + 0.5 * nodes) * hx
                    gvals = np.asarray(
                        dirichlet(x_pts, np.full_like(x_pts, y_face)), dtype=float,
                    )
                    endpoint = left if side == "left" else right
                    gf = (0.5 * hx) * np.einsum("q,q,iq->i", weights, gvals, phi)
                    trace = np.outer(gf, endpoint).reshape(block)
                    rhs_qy = e * 3 * block + 2 * block
                    rhs_u = e * 3 * block
                    if boundary_closure == "mirror":
                        _block(e, 2, e, 0, sign * mfy)
                        rhs_vec[rhs_qy: rhs_qy + block] += 2.0 * sign * trace
                    else:
                        rhs_vec[rhs_qy: rhs_qy + block] += sign * trace
                    _block(e, 0, e, 2, -sign * mfy)
                    if boundary_closure == "flux":
                        beta_y = dirichlet_flux_beta / hy
                        _block(e, 0, e, 0, -sign * beta_y * mfy)
                        rhs_vec[rhs_u: rhs_u + block] += -sign * beta_y * trace

'''
src = src[:y_start] + new_y + src[y_end:]

# ---------------------------------------------------------------- return matrix
old_mat = '''    matrix = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndofs, ndofs),
    ).tocsr()
    solution = spla.spsolve(matrix, rhs_vec)'''
new_mat = '''    matrix = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ndofs, ndofs),
    ).tocsr()
    if return_matrix:
        return {
            "matrix": matrix,
            "rhs": rhs_vec,
            "ndofs": int(ndofs),
            "nnz": int(matrix.nnz),
            "ncells": (nx, ny),
            "degree": degree,
            "h": max(hx, hy),
            "trace_policy": trace_policy,
            "boundary_closure": boundary_closure,
            "dirichlet_flux_beta": float(dirichlet_flux_beta),
        }
    solution = spla.spsolve(matrix, rhs_vec)'''
assert src.count(old_mat) == 1
src = src.replace(old_mat, new_mat)

# ---------------------------------------------------------------- result keys
old_ret = '''        "ncells": (nx, ny),
        "degree": degree,
        "xc": (np.arange(nx) + 0.5) * hx,'''
new_ret = '''        "ncells": (nx, ny),
        "degree": degree,
        "trace_policy": trace_policy,
        "boundary_closure": boundary_closure,
        "dirichlet_flux_beta": float(dirichlet_flux_beta),
        "xc": (np.arange(nx) + 0.5) * hx,'''
assert src.count(old_ret) == 1
src = src.replace(old_ret, new_ret)

p.write_text(src)
print("patched", p, len(src.splitlines()), "lines")