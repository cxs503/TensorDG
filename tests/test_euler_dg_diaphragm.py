"""Conservative cell averages for a Riemann discontinuity inside a DG cell."""
import numpy as np
import pytest

from tensordg.euler_1d import solve_riemann
from tensordg.euler_dg_1d import solve_euler_dg_1d


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_cut_cell_initial_conserved_integrals(degree):
    # Left and right conserved states for gamma=1.4 are (2, 4, 9)
    # and (1, -1, 3). The left state occupies a quarter of cell 8.
    options = dict(ncells=20, domain=(-1., 1.), diaphragm=-.175, final_time=0.)
    states = (2., 2., 2.), (1., -1., 1.)
    if degree == 0:
        _, fields, _ = solve_riemann(*states, **options)
        averages = fields["conservative"].numpy()
    else:
        _, coeff, _ = solve_euler_dg_1d(*states, degree=degree, **options)
        averages = coeff[:, 0].numpy()
    np.testing.assert_allclose(averages[8], [1.25, .25, 4.5], atol=2e-14, rtol=0)
    np.testing.assert_allclose(.1 * averages.sum(axis=0),
                               [2.825, 2.125, 10.95], atol=2e-14, rtol=0)


def test_half_cell_stationary_contact_stays_in_equilibrium():
    _, coeff, _ = solve_euler_dg_1d(
        (2., 0., 1.), (1., 0., 1.), ncells=21, degree=2,
        diaphragm=.5, final_time=.02,
    )
    expected = np.zeros(coeff.shape)
    expected[:10, 0, 0] = 2.
    expected[10, 0, 0] = 1.5
    expected[11:, 0, 0] = 1.
    expected[:, 0, 2] = 2.5
    np.testing.assert_allclose(coeff.numpy(), expected, atol=3e-13, rtol=0)
