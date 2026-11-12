"""WENO-Z nonlinear weights for the Qiu--Shu limiter (absorption of Borges et al. 2008).

Motivation (external practice)
------------------------------
The WENO limiter in :mod:`tensordg.limiters` uses the classical Jiang--Shu (JS)
weights

    alpha_k = d_k / (eps + beta_k)**r,   w_k = alpha_k / sum(alpha),

with ``eps = 1e-6`` and ``r = 2``.  Borges, Carmona, Costa & Don (JCP 227,
2008, 3101--3111) showed that this normalisation is *not* invariant under a
rescaling of the smoothness indicators: when every ``beta_k`` is much smaller
than ``eps`` (which happens for small-amplitude fields, or for fine meshes,
because ``beta ~ dx * (du/dx)**2``), all ``alpha_k`` collapse to ``d_k / eps**r``
and the weights degenerate to the *ideal* linear weights ``d = (1/6, 2/3, 1/6)``
regardless of the data.  The limiter then loses the ability to discard the
stencil that crosses a discontinuity.  WENO-Z replaces the normalisation by

    alpha_k = d_k * (1 + (tau / (beta_k + eps))**q),   tau = |beta_0 - beta_2|,

which depends only on *ratios* of the smoothness indicators, so the weights are
invariant under a uniform rescaling of ``beta`` (and hence under amplitude or
mesh rescaling) and recover ``w -> d`` only where the data really is smooth.

The default variant stays ``"js"`` so that every verified number in ``docs/``
is bit-for-bit reproducible; ``"z"`` is opt-in.

What is pinned here
-------------------
1. ``"js"`` reproduces the historical formula bit for bit (no silent drift);
2. the Z weights are invariant under a rescaling of the smoothness indicators
   (``beta -> c * beta``) while the JS weights are not -- the defect above;
3. both variants return the ideal weights ``d`` when the data is uniform
   (``beta_0 = beta_1 = beta_2``) and concentrate on the smooth stencil when one
   stencil crosses a jump;
4. inside the limiter, ``weno_variant="z"`` (a) preserves the cell average
   exactly, (b) never increases the reconstruction error on smooth data --
   measured on a sine wave over a mesh sequence, the Z weights give a slightly
   *smaller* L2 error because they stay closer to the ideal weights for the
   stencil that carries the cell -- and (c) removes the spurious limited slope
   in a cell whose own average is uniform when the jump amplitude is small
   enough to push ``beta`` below ``eps`` (the JS degeneracy in practice).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensordg.limiters import _weno_weights, apply_limiter, apply_weno_limiter

torch.set_default_dtype(torch.float64)

IDEAL = (1 / 6, 2 / 3, 1 / 6)


def _project(f, ncells: int, degree: int = 2) -> torch.Tensor:
    """Modal Legendre coefficients ``[ncells, degree+1, 1]`` of ``f`` by L2 projection."""
    nodes, _ = np.polynomial.legendre.leggauss(degree + 1)
    centres = (np.arange(ncells) + 0.5) / ncells
    vander = np.polynomial.legendre.legvander(nodes, degree)
    # Legendre nodes live on [-1, 1]; the cell is dx = 1/ncells wide.
    values = f(centres[:, None] + 0.5 * nodes[None, :] / ncells)
    return torch.tensor(values @ np.linalg.inv(vander).T)[:, :, None]


def _reconstruct(coeff: torch.Tensor, ncells: int, order: int = 8) -> np.ndarray:
    """Nodal values of the modal representation at ``order`` Gauss points per cell."""
    nodes, _ = np.polynomial.legendre.leggauss(order)
    vander = np.polynomial.legendre.legvander(nodes, coeff.shape[1] - 1)
    return coeff[:, :, 0].detach().cpu().numpy() @ vander.T


def _l2_error(coeff: torch.Tensor, f, ncells: int, order: int = 8) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    centres = (np.arange(ncells) + 0.5) / ncells
    exact = f(centres[:, None] + 0.5 * nodes[None, :] / ncells)
    residual = _reconstruct(coeff, ncells, order) - exact
    return float(np.sqrt((weights[None, :] * residual**2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# 1. weight functions
# ---------------------------------------------------------------------------

def test_js_variant_reproduces_the_historical_formula_bitwise():
    """Regression: the default path must not move any verified number."""
    torch.manual_seed(0)
    beta = torch.rand(3, 64) * 10.0 ** torch.randint(-8, 2, (3, 64)).to(torch.float64)
    legacy = tuple(
        (IDEAL[k] / (1e-6 + beta[k]) ** 2)
        / sum(IDEAL[j] / (1e-6 + beta[j]) ** 2 for j in range(3))
        for k in range(3)
    )
    got = _weno_weights(beta[0], beta[1], beta[2], variant="js")
    for reference, value in zip(legacy, got):
        assert torch.equal(reference, value)


@pytest.mark.parametrize("scale", [1e-8, 1e-4, 1e4, 1e8])
def test_z_weights_are_invariant_under_rescaling_of_the_indicators(scale):
    """WENO-Z depends on beta only through ratios: w(c*beta) == w(beta).

    The JS weights are *not* invariant (``eps`` fixes a scale), which is exactly
    the degeneracy this variant removes.
    """
    torch.manual_seed(1)
    beta = (torch.rand(3, 32) + 0.05) * 1e-3
    reference = _weno_weights(beta[0], beta[1], beta[2], variant="z")
    scaled = _weno_weights(*(scale * b for b in beta), variant="z")
    for reference_k, scaled_k in zip(reference, scaled):
        assert torch.allclose(reference_k, scaled_k, rtol=1e-10, atol=1e-12)

    js_reference = _weno_weights(beta[0], beta[1], beta[2], variant="js")
    js_scaled = _weno_weights(*(scale * b for b in beta), variant="js")
    drift = max(
        float((a - b).abs().max()) for a, b in zip(js_reference, js_scaled)
    )
    assert drift > 1e-3, "expected the JS weights to break scale invariance"
    assert _weno_weights(beta[0], beta[1], beta[2], variant="js")[0].shape == beta[0].shape
    del js_reference, js_scaled


@pytest.mark.parametrize("variant", ["js", "z"])
def test_uniform_data_returns_the_ideal_weights(variant):
    beta = torch.full((64,), 0.37)
    weights = _weno_weights(beta, beta, beta, variant=variant)
    for ideal_k, weight_k in zip(IDEAL, weights):
        assert torch.allclose(weight_k, torch.full_like(weight_k, ideal_k), rtol=1e-12)


@pytest.mark.parametrize("variant", ["js", "z"])
def test_stencil_crossing_a_jump_is_rejected(variant):
    """``beta_0`` smooth, ``beta_1 = beta_2`` huge: the smooth stencil must win."""
    beta0 = torch.full((16,), 1e-6)
    beta1 = torch.full((16,), 50.0)
    weights = _weno_weights(beta0, beta1, beta1, variant=variant)
    assert (weights[0] > 0.9).all()


def test_unknown_variant_raises():
    beta = torch.ones(4)
    with pytest.raises(ValueError):
        _weno_weights(beta, beta, beta, variant="weno5")


# ---------------------------------------------------------------------------
# 2. behaviour inside the limiter
# ---------------------------------------------------------------------------

def _sine(x):
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)


@pytest.mark.parametrize("ncells", [20, 40, 80])
def test_z_variant_preserves_cell_averages_exactly(ncells):
    coeff = _project(_sine, ncells)
    limited = apply_limiter(
        coeff, 1.0 / ncells, kind="weno", tvb_constant=1.0, weno_variant="z"
    )
    assert torch.equal(limited[:, 0], coeff[:, 0])


@pytest.mark.parametrize("ncells", [20, 40, 80])
def test_z_variant_never_increases_the_smooth_reconstruction_error(ncells):
    """Measured on a sine: the Z weights stay closer to the ideal weights."""
    coeff = _project(_sine, ncells)
    js = apply_limiter(coeff, 1.0 / ncells, kind="weno", tvb_constant=1.0,
                       weno_variant="js")
    z = apply_limiter(coeff, 1.0 / ncells, kind="weno", tvb_constant=1.0,
                      weno_variant="z")
    js_error = _l2_error(js, _sine, ncells)
    z_error = _l2_error(z, _sine, ncells)
    assert z_error <= js_error * (1.0 + 1e-12), (ncells, js_error, z_error)


def _jump_weights(ncells: int, amplitude: float, fraction: float = 0.3) -> tuple:
    """Weights of the three limiter stencils in the cell that holds a jump.

    The jump sits *inside* a cell so that all three candidate stencils see
    different curvature; this is the configuration in which the JS
    ``epsilon``-degeneracy shows up as weights that stop depending on the data.
    """
    # The jump sits at the *same relative position* inside a cell for every
    # mesh, so a sweep over ``ncells`` changes only the resolution, not the
    # geometry (and a sweep over ``amplitude`` changes only the scale).
    cell = ncells // 2
    x0 = (cell + fraction) / ncells
    coeff = _project(lambda x: 1.0 + amplitude * np.where(x < x0, -0.5, 0.5), ncells)
    dx = 1.0 / ncells
    avg = coeff[:, 0, 0]
    padded = torch.cat((avg[:1], avg, avg[-1:]))
    d_left = avg - padded[:-2]
    d_right = padded[2:] - avg
    beta0 = dx * (d_left / dx) ** 2
    beta1 = dx * (d_right / dx) ** 2
    beta2 = (1 / 3) * (padded[:-2] - 2 * avg + padded[2:]) ** 2 / dx + dx * (
        (padded[2:] - padded[:-2]) / (2 * dx)
    ) ** 2
    return (
        torch.stack(_weno_weights(beta0[cell:cell + 1], beta1[cell:cell + 1],
                                  beta2[cell:cell + 1], variant="js")),
        torch.stack(_weno_weights(beta0[cell:cell + 1], beta1[cell:cell + 1],
                                  beta2[cell:cell + 1], variant="z")),
    )


def test_js_weights_are_amplitude_dependent_while_z_weights_are_not():
    """The measurable face of the ``epsilon`` degeneracy.

    For a fixed jump geometry, ``beta ~ amplitude**2``.  As the amplitude
    shrinks below ``sqrt(epsilon * dx)`` the JS weights collapse to the ideal
    ``(1/6, 2/3, 1/6)`` -- the limiter silently stops being data-adaptive and
    becomes a fixed linear weighting.  The Z weights are unchanged because they
    only see ratios of the indicators.
    """
    reference_js, reference_z = _jump_weights(40, 1.0)
    for amplitude in (1e-4, 1e-5, 1e-6):
        js, z = _jump_weights(40, amplitude)
        assert float((z - reference_z).abs().max()) < 1e-9, (amplitude, z)
        assert float((js - reference_js).abs().max()) > 0.1, (amplitude, js)
    # ... and the same degeneration is reached by mesh refinement alone at
    # fixed amplitude, because ``beta ~ dx * (du/dx)**2`` shrinks with dx.
    js_fine, z_fine = _jump_weights(200, 1e-4)
    js_coarse, z_coarse = _jump_weights(40, 1e-4)
    assert float((z_fine - z_coarse).abs().max()) < 1e-9
    assert float((js_fine - js_coarse).abs().max()) > 0.05, (js_coarse, js_fine)
