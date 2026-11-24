"""Sod shock tube using TVB detection and monotonized central reconstruction."""
from benchmarks.sod_mc_refinement import measure


def test_mc_sod_matches_wave_split_riemann_reference():
    result = measure(400)
    for field, error in result["relative_l2"].items():
        assert error < .03, (field, error)
    for field, residual in result["balance_residual"].items():
        assert residual < 1e-12, (field, residual)
