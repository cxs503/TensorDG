import numpy as np
import torch

from benchmarks.euler_sod_1d import solve_sod
from benchmarks.sod_exact import cell_averages, star_state


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


def test_sod_respects_requested_tensor_dtype():
    _, fields, _ = solve_sod(ncells=40, final_time=0.005, dtype=torch.float64)
    assert fields["conservative"].dtype == torch.float64
    _, fields32, _ = solve_sod(ncells=40, final_time=0.005, dtype=torch.float32)
    assert fields32["conservative"].dtype == torch.float32


def test_exact_sod_star_state():
    pressure, velocity = star_state()
    assert abs(pressure - 0.30313017805064685) < 1e-12
    assert abs(velocity - 0.9274526200489499) < 1e-12


def test_documented_sod_case_matches_exact_solution_within_three_percent():
    _, fields, _ = solve_sod(ncells=3200, final_time=0.2, cfl=0.4)
    _, reference = cell_averages(3200, final_time=0.2)
    for key in ("density", "velocity", "pressure"):
        numerical = fields[key].detach().cpu().numpy()
        error = np.sqrt(np.mean((numerical - reference[key]) ** 2))
        norm = np.sqrt(np.mean(reference[key] ** 2))
        assert error / norm < 0.03, (key, error / norm)
