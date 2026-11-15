import pathlib
p = pathlib.Path("benchmarks/dmr_three_shock.py"); s = p.read_text()

start = s.index("    # de-duplicate (each physical root shows up from many seeds)")
end = s.index("# ------------------------------------------------- Galilean (independent) form")
new = '''    # de-duplicate (each physical root shows up from many seeds)
    uniq = []
    for cand in solutions:
        if not any(abs(cand[0] - u[0]) < 1e-6 and abs(cand[1] - u[1]) < 1e-6 for u in uniq):
            uniq.append(cand)
    uniq.sort(key=lambda c: c[0])
    branches = [_branch(root[0], root[1], omega, mach, gamma) for root in uniq]
    primary = branches[0]
    return {
        "case": "double_mach_reflection · three-shock (von Neumann) theory",
        "mach": mach,
        "wall_angle_deg": wall_angle_deg,
        "gamma": gamma,
        "n_roots_found": len(branches),
        "roots_chi_deg": sorted(b["chi_deg"] for b in branches),
        "branch_labels": [b["label"] for b in branches],
        "branches": {b["label"]: b for b in branches},
        # convenience aliases for the lowest-chi branch (never the only answer)
        "chi_deg": primary["chi_deg"],
        "q": primary["q"],
        "states_triple_point_frame": primary["states_triple_point_frame"],
        "slip_line_properties": primary["slip_line_properties"],
        "rh_flux_residuals": primary["rh_flux_residuals"],
        "rh_worst": primary["rh_worst"],
        "shock_angles_deg": primary["shock_angles_deg"],
        "mach_numbers": primary["mach_numbers"],
    }


def _branch(chi: float, phi: float, omega: float, mach: float,
            gamma: float) -> Dict[str, object]:
    """Full description of one polar-intersection branch."""
    up, post_inc, post_stem, q = _frame_states(chi, omega, mach, gamma)
    post_refl = stationary_shock(post_inc, (math.cos(phi), math.sin(phi)), gamma)
    n_inc = _incident_normal(omega)
    n_refl = (math.cos(phi), math.sin(phi))
    n_stem = _stem_normal_preferred()
    residuals = {
        "incident": rh_flux_residuals(up, post_inc, n_inc, gamma),
        "reflected": rh_flux_residuals(post_inc, post_refl, n_refl, gamma),
        "stem": rh_flux_residuals(up, post_stem, n_stem, gamma),
    }
    worst = max(abs(v) for r in residuals.values() for v in r.values())
    # normal-component Mach downstream of the reflected shock (<1 => strong shock)
    vn3 = post_refl.u * n_refl[0] + post_refl.v * n_refl[1]
    chi_deg = math.degrees(chi)
    branch = {
        "label": "chi_%05.2fdeg" % chi_deg,
        "chi_deg": chi_deg,
        "q": q,
        "triple_point_speed_lab": [q * math.cos(chi), q * math.sin(chi)],
        "slip_line_direction_deg": math.degrees(math.atan2(post_stem.v, post_stem.u)),
        "states_triple_point_frame": {
            "1_upstream": up.as_dict(),
            "2_post_incident": post_inc.as_dict(),
            "3_post_reflected": post_refl.as_dict(),
            "3p_post_stem": post_stem.as_dict(),
        },
        "slip_line_properties": {
            "p_post_reflected": post_refl.p,
            "p_post_stem": post_stem.p,
            "pressure_mismatch_rel": (post_refl.p - post_stem.p) / post_stem.p,
            "velocity_cross_norm": _cross(post_refl, post_stem)
            / max(post_refl.speed() * post_stem.speed(), 1e-12),
            "velocity_dot_sign": float(np.sign(post_refl.u * post_stem.u
                                               + post_refl.v * post_stem.v)),
        },
        "shock_angles_deg": {
            "incident_normal_from_y_axis": 180.0 - math.degrees(omega),
            "reflected_normal_from_x_axis": math.degrees(phi) % 360.0,
            "stem_normal_from_x_axis": 180.0,
            "stem_plane_from_wall": 90.0,
        },
        "mach_numbers": {
            "M2_post_incident": post_inc.speed() / post_inc.sound(gamma),
            "M1n_incident": mach,
            "M1n_stem": q * math.cos(chi) / PRE.sound(gamma),
            "Mn_downstream_reflected": vn3 / post_refl.sound(gamma),
        },
        "rh_flux_residuals": residuals,
        "rh_worst": worst,
    }
    branch["trajectory"] = triple_point_trajectory(branch)
    return branch


'''
s = s[:start] + new + s[end:]

a = s.index("def summary(mach: float = SHOCK_MACH")
b = s.index("def main() -> None:  # pragma: no cover - CLI")
new_summary = '''def summary(mach: float = SHOCK_MACH, wall_angle_deg: float = WALL_ANGLE_DEG,
            gamma: float = GAMMA, times: Sequence[float] = (0.1, 0.15, 0.2),
            galilean: bool = True) -> Dict[str, object]:
    """Theory bundle: every polar-intersection branch plus the Galilean cross-check."""
    primary = solve_three_shock(mach, wall_angle_deg, gamma)
    branches = {}
    for label, br in primary["branches"].items():
        br = dict(br)
        br["trajectory"] = triple_point_trajectory(br, times)
        branches[label] = br
    out: Dict[str, object] = {
        "theory": primary,
        "branches": branches,
        "branch_labels": primary["branch_labels"],
        "detachment_angle_deg": math.degrees(detachment_angle(mach, gamma)),
        "times": list(times),
    }
    if galilean:
        gal = solve_three_shock_galilean(mach, wall_angle_deg, gamma)
        out["galilean_cross_check"] = gal
        out["chi_agreement_deg"] = min(abs(br["chi_deg"] - gal["chi_deg"])
                                       for br in branches.values())
    return out


'''
s = s[:a] + new_summary + s[b:]
p.write_text(s)
print("patched solve_three_shock tail + summary")