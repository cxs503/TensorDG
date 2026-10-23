import torch

from tensordg.euler_2d import solve_isentropic_vortex, vortex_error


def test_vortex_initial_projection_and_conservation():
    x, y, fields, _ = solve_isentropic_vortex(ncells=32, final_time=0.0)
    initial_errors = vortex_error(x, y, fields["conservative"], 0.0)
    assert initial_errors["density"]["relative_l2"] < 1e-8
    initial_mass = fields["conservative"][..., 0].sum()
    x, y, fields, _ = solve_isentropic_vortex(ncells=32, final_time=0.1)
    assert torch.isfinite(fields["conservative"]).all()
    assert (fields["density"] > 0).all()
    assert (fields["pressure"] > 0).all()
    final_mass = fields["conservative"][..., 0].sum()
    assert abs((final_mass / initial_mass).item() - 1.0) < 1e-11


def test_documented_vortex_case_is_below_three_percent_for_every_field():
    x, y, fields, _ = solve_isentropic_vortex(ncells=192, final_time=1.0)
    errors = vortex_error(x, y, fields["conservative"], 1.0)
    for key, metrics in errors.items():
        assert metrics["relative_l2"] < 0.03, (key, metrics)
