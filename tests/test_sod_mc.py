"""Sod shock tube using TVB detection and monotonized central reconstruction."""
from benchmarks.sod_mc_refinement import measure


def test_mc_sod_matches_wave_split_riemann_reference():
    result = measure(400)
    for field, error in result["relative_l2"].items():
        assert error < .03, (field, error)
    for field, residual in result["balance_residual"].items():
        assert residual < 1e-12, (field, residual)


def test_minmod_sod_primitives_of_conserved_averages():
    result = measure(400, reconstruction="minmod")
    for field, error in result["matched_relative_l2"].items():
        assert error < .03, (field, error)
