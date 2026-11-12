import pathlib

p = pathlib.Path("tensordg/poisson_2d_ldg.py")
s = p.read_text()

# ---- add a mixed-endpoint face-operator helper next to _face_mass ----------
anchor = '''# ---------------------------------------------------------------------------
# Manufactured solution (default Dirichlet benchmark)
# ---------------------------------------------------------------------------'''
helper = '''def _face_operator(
    test_endpoint: np.ndarray,
    coef_endpoint: np.ndarray,
    mass: np.ndarray,
    scale: float,
    direction: str = "x",
) -> np.ndarray:
    """Face term for a trace taken from *another* element.

    The test function is evaluated at the endpoint of the element that owns the
    row, while the traced coefficient comes from the element that owns the
    trace; on a shared face those two endpoints are generally *different*
    (``+1`` for the left neighbour, ``-1`` for the right one).  Using the same
    endpoint vector for both factors silently replaces ``L_k(-1)=(-1)^k`` by
    ``L_k(+1)=1`` on half of the modes -- a defect that leaves the scheme
    non-singular but order-destroying.
    """
    outer = np.outer(test_endpoint, coef_endpoint)
    return scale * (
        _kron2(outer, mass) if direction == "x" else _kron2(mass, outer)
    )


''' + anchor
assert s.count(anchor) == 1
s = s.replace(anchor, helper)

# ---- rewrite the interior x-face block --------------------------------------
old_x = '''            for side, sign, mfx in (
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
                    _block(e, 0, q_elem, 1, -sign * mfx)'''
new_x = '''            for side, sign, endpoint_self in (
                ("left", -1.0, left),
                ("right", +1.0, right),
            ):
                nbr = ix - 1 if side == "left" else ix + 1
                endpoint_nbr = right if side == "left" else left
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
                    u_ep = endpoint_nbr if u_elem == en else endpoint_self
                    q_ep = endpoint_nbr if q_elem == en else endpoint_self
                    _block(e, 1, u_elem, 0,
                           -sign * _face_operator(endpoint_self, u_ep, mass, 0.5 * hy))
                    _block(e, 0, q_elem, 1,
                           -sign * _face_operator(endpoint_self, q_ep, mass, 0.5 * hy))'''
assert s.count(old_x) == 1
s = s.replace(old_x, new_x)

# ---- rewrite the interior y-face block --------------------------------------
old_y = '''            for side, sign, mfy in (
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
                    _block(e, 0, q_elem, 2, -sign * mfy)'''
new_y = '''            for side, sign, endpoint_self in (
                ("left", -1.0, left),
                ("right", +1.0, right),
            ):
                nbr = iy - 1 if side == "left" else iy + 1
                endpoint_nbr = right if side == "left" else left
                if 0 <= nbr < ny:
                    en = ix * ny + nbr
                    if trace_policy == "ldg":
                        u_elem, q_elem = (en, e) if side == "left" else (e, en)
                    else:
                        u_elem, q_elem = (e, en) if side == "left" else (en, e)
                    u_ep = endpoint_nbr if u_elem == en else endpoint_self
                    q_ep = endpoint_nbr if q_elem == en else endpoint_self
                    _block(e, 2, u_elem, 0,
                           -sign * _face_operator(endpoint_self, u_ep, mass, 0.5 * hx, "y"))
                    _block(e, 0, q_elem, 2,
                           -sign * _face_operator(endpoint_self, q_ep, mass, 0.5 * hx, "y"))'''
assert s.count(old_y) == 1
s = s.replace(old_y, new_y)

p.write_text(s)
print("patched face operators")