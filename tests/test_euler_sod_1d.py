import torch
from benchmarks.euler_sod_1d import solve_sod


def test_sod_state_remains_physical_and_finite():
    x, fields, meta = solve_sod(ncells=100, final_time=0.05, cfl=0.3)
    assert torch.isfinite(fields["conservative"]).all()
    assert (fields["density"] > 0).all()
    assert (fields["pressure"] > 0).all()
    assert meta["steps"] > 0
    assert x.numel() == 100


def test_sod_early_time_conservation_and_wave_direction():
    _, fields, _ = solve_sod(ncells=120, final_time=0.02, cfl=0.3)
    u = fields["conservative"]
    mass = u[:, 0].mean().item()
    assert abs(mass - 0.5625) < 2e-3
    assert fields["velocity"].max().item() > 0.01
    assert fields["density"].max().item() > fields["density"].min().item()
