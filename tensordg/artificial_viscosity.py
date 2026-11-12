"""Sensor-gated modal artificial viscosity (exponential/SVV filter) for modal DG.

Motivation (the gap this module closes)
--------------------------------------
Every shock-capturing device in :mod:`tensordg.limiters` (TVB/minmod,
``"weno"``, ``"ao"``) is a *reconstruction replacement*: once the troubled-cell
detector fires, the polynomial of that cell is rebuilt from cell averages (to a
linear polynomial, or to a WENO combination of stencils).  The replacement is a
**switch**: cells are either fully limited or untouched, so the scheme is
discontinuous in the data, smooth extrema are clipped when the detector fires
there, and the strength cannot be ramped.

The classical alternative -- Persson & Peraire, *"Sub-cell shock capturing for
discontinuous Galerkin methods"*, AIAA Paper 2006-0112 (also the modal
spectral-viscosity filter of Maday, Kaber & Tadmor, J. Sci. Comput. 1993) -- keeps
the polynomial basis and instead **damps the highest modes with a strength that
varies continuously in space and time**.  Nothing in this repository implemented
it; ``grep -ri "artificial viscosity" tensordg/ benchmarks/ tests/`` returned no
hit before this module.

Two ingredients
---------------
1. **Smoothness indicator** (:func:`modal_smoothness`, :func:`av_sensor`).  In a
   modal Legendre basis the energy of mode ``m`` is ``c_m^2 / (2m+1)`` on the
   reference element, so the fraction of the element energy carried by the top
   mode

       s_e = log10( (c_p^2/(2p+1)) / sum_m c_m^2/(2m+1) )

   is a *scale-free* measure of how badly the element is resolved.  Its two
   limits are what makes it usable as a sensor: for a smooth field the top-mode
   fraction decays like ``h^{2p}`` (so ``s_e`` decreases at rate ``2p`` per
   decade of ``h``) while for a jump it saturates at ``O(1)`` independent of
   ``h``.  Both exponents are **measured** by
   ``benchmarks/artificial_viscosity_verification.py`` -- none of them is quoted.
2. **Exponential (SVV) filter** (:func:`exponential_filter_factor`).  Mode ``m``
   of a flagged element is multiplied by

       sigma_m = exp( -alpha_e * (m / P)^{2*order} ),   P = p + 1,

   with ``alpha_e`` a quadratic ramp of the sensor between ``s0 - kappa`` and
   ``s0 + kappa`` (Persson--Peraire's ramp shape).  ``mode 0`` is **never**
   touched, so mass, momentum and energy are conserved to the roundoff of the
   sum that formed them -- the filter is conservative by construction, which no
   cell-average limiter can claim (all of them preserve it too, but by
   re-deriving the mean; here it is left bitwise intact).

Deliberate limitations
----------------------
* ``p = 0`` has a single mode, hence no sub-cell content and no sensor; the
  filter is a no-op and :func:`modal_smoothness` rejects ``nmodes < 2``.
* The filter is *dissipative but not provably positive*.  It is designed to run
  together with the existing Zhang--Shu positivity scaling
  (``solve_euler_dg_1d(..., positivity="scale")``), which is what keeps the
  pressure admissible; the module never claims an entropy or TVD bound.
* ``s0`` is mesh dependent for smooth data by construction (the sensor decays
  like ``h^{2p}``).  The default is a *measured* value on the documented
  configurations (see ``docs/CASE-DG-AV-SHOCK-CAPTURE.md``); for a new
  configuration the honest usage is to run :func:`sensor_reference` on the
  smooth part of the same mesh and place ``s0`` above it.
"""
from __future__ import annotations

import torch

__all__ = [
    "modal_smoothness",
    "av_sensor",
    "sensor_reference",
    "exponential_filter_factor",
    "av_strength",
    "apply_av_filter",
]

_TINY = 1e-300


def modal_smoothness(
    coeff: torch.Tensor,
    top_mode: int | None = None,
    floor_rel: float = 1e-12,
    normalisation: str = "fraction",
) -> torch.Tensor:
    """Per-element, per-variable log10 top-mode energy fraction.

    Parameters
    ----------
    coeff : (ncells, nmodes, nvar) modal Legendre coefficients (as assembled by
        the solvers in this package: ``mode 0`` is the cell average).
    top_mode : int, optional
        Mode index treated as "top"; defaults to the last one (degree ``p``).
    normalisation : {"fraction", "global"}
        ``"fraction"`` (default) divides the top-mode energy of an element by
        **that same element's** total modal energy (plus ``floor``); it
        separates shocks from smooth data by ~5 decades on the documented
        configuration.  The local ratio is ill-conditioned where the field
        passes through zero (``E_e -> 0`` with the top mode still finite):
        measured on a resolved sine, ``p = 1`` gives a *mesh independent*
        ``s_e = -0.603`` for 20..160 cells instead of the theoretical
        ``2p = 2`` per-decade decay, because the element at the zero crossing
        dominates the max.  ``"global"`` divides the top-mode energy by the
        **mesh-mean** element energy of the same variable; it is sign-robust
        and reproduces the ``2p`` law for both degrees (measured 4.00 at
        ``p = 2``, 2.00 at ``p = 1``).
    floor_rel : float
        The fraction is taken against ``E_e + floor_rel * max_e E_e`` (per
        variable).  Without a floor the ratio is **not scale aware**, and a
        variable that is (numerically) zero in an element -- the momentum
        ``rho u`` in a gas at rest is the standard example -- has
        ``E_e ~ 1e-34`` built out of roundoff while its top mode is a
        comparable fraction of it, so the sensor fires at ``s_e ~ O(10^-1)``
        in an exactly uniform region (measured: ``s = -0.21`` on the upstream
        state of the Sod problem before this floor was added).  The floor is
        relative to the strongest element of the same variable, so it is
        scale-invariant in the state.

    Returns
    -------
    (ncells, nvar) tensor of ``log10`` fractions in ``[-inf, 0]``.  ``0`` means
    all the element energy sits in the top mode (a jump at that mesh size);
    very negative means the element is resolved.
    """
    if coeff.dim() != 3:
        raise ValueError("coeff must be (ncells, nmodes, nvar)")
    nmodes = coeff.shape[1]
    if nmodes < 2:
        raise ValueError(
            "modal_smoothness needs at least two modes (p >= 1); "
            "the p=0 scheme has no sub-cell content to sense"
        )
    if top_mode is None:
        top_mode = nmodes - 1
    if not 0 < top_mode < nmodes:
        raise ValueError("top_mode must satisfy 0 < top_mode < nmodes")
    # L2 weights of the unnormalised Legendre basis on [-1, 1]: 2/(2m+1).
    weights = 1.0 / torch.arange(1, 2 * nmodes, 2, dtype=coeff.dtype, device=coeff.device)
    energy = (coeff * coeff * weights.view(1, -1, 1)).sum(dim=1)          # (cells, nvar)
    top = coeff[:, top_mode, :] ** 2 * weights[top_mode]                   # (cells, nvar)
    # Floor on the *global* state scale, not on the per-variable one: a
    # variable that vanishes everywhere (momentum in a gas at rest) would
    # otherwise be normalised by its own roundoff and fire (measured
    # s = -4e-13 before this change).
    mean_energy = energy.mean(dim=0)
    floor = floor_rel * mean_energy.amax().clamp_min(_TINY)
    if normalisation == "fraction":
        denominator = energy + floor
    elif normalisation == "global":
        denominator = (mean_energy + floor).view(1, -1)
    else:
        raise ValueError("normalisation must be 'global' or 'fraction'")
    return torch.log10(top / denominator.clamp_min(_TINY))


def _neighbour_max(values: torch.Tensor, passes: int, periodic: bool) -> torch.Tensor:
    """Widen a cell-wise indicator by a max over the face neighbours.

    Persson & Peraire smooth the viscosity field so that neighbouring cells do
    not carry viscously inconsistent states; a max (rather than an average)
    keeps the shock fully covered, which is the conservative choice.
    """
    if passes <= 0:
        return values
    out = values
    for _ in range(passes):
        if periodic:
            padded = torch.cat((out[-1:], out, out[:1]))
        else:
            padded = torch.cat((out[:1], out, out[-1:]))
        out = torch.maximum(torch.maximum(padded[:-2], out), padded[2:])
    return out


def av_sensor(
    coeff: torch.Tensor,
    variables=None,
    smooth: int = 1,
    periodic: bool = False,
    normalisation: str = "fraction",
) -> torch.Tensor:
    """Element-wise sensor value ``s_e`` (max over the selected variables).

    ``variables`` restricts the sensor to a subset of the ``nvar`` components
    (e.g. density only); the default is *all* components, which is the safe
    choice because a jump in any conserved variable produces sub-cell
    oscillation.
    """
    s = modal_smoothness(coeff, normalisation=normalisation)
    if variables is None:
        s = s.max(dim=1).values
    else:
        index = torch.as_tensor(list(variables), dtype=torch.long, device=s.device)
        s = s.index_select(1, index).max(dim=1).values
    return _neighbour_max(s, smooth, periodic)


def sensor_reference(coeff: torch.Tensor, quantile: float = 1.0, **kwargs) -> float:
    """Reference sensor level of a **smooth** field at the same mesh/degree.

    Returns the ``quantile`` (default: the maximum) of ``av_sensor``.  Placing
    ``s0`` a margin above this number is the reproducible way to keep the filter
    inert on the resolved part of a flow instead of guessing a literature
    constant (the paper's constant is resolution independent only for the
    *discontinuous* branch of the sensor).
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    s = av_sensor(coeff, **kwargs)
    if quantile == 1.0:
        return float(s.max().item())
    k = max(1, int(round(quantile * s.numel())))
    return float(torch.sort(s.flatten()).values[k - 1].item())


def exponential_filter_factor(
    nmodes: int,
    alpha: torch.Tensor,
    order: float = 2.0,
    dtype=None,
    device=None,
) -> torch.Tensor:
    """``sigma[m] = exp(-alpha * (m / P)^{2*order})`` for ``m = 0..P-1``.

    ``sigma[0] == 1`` exactly, so cell averages are conserved bitwise.
    """
    if order <= 0:
        raise ValueError("order must be positive")
    dtype = dtype or alpha.dtype
    device = device or alpha.device
    m = torch.arange(nmodes, dtype=dtype, device=device)
    scaled = (m / float(nmodes)) ** (2.0 * order)
    sigma = torch.exp(-alpha.reshape(-1, 1) * scaled.reshape(1, -1))
    sigma[:, 0] = 1.0
    return sigma


def av_strength(
    sensor: torch.Tensor,
    s0: float = -2.0,
    kappa: float = 0.5,
    alpha_max: float = 20.0,
) -> torch.Tensor:
    """Quadratic ramp of the filter strength, Persson--Peraire shape.

    ``s <= s0 - kappa``  -> 0 (filter inert)
    ``s >= s0 + kappa``  -> ``alpha_max``
    in between           -> ``alpha_max * t^2``, ``t`` linear in ``s``.
    """
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if alpha_max < 0:
        raise ValueError("alpha_max must be non-negative")
    t = ((sensor - (s0 - kappa)) / (2.0 * kappa)).clamp(0.0, 1.0)
    return alpha_max * t * t


def apply_av_filter(
    coeff: torch.Tensor,
    dx: float,
    s0: float = -2.0,
    kappa: float = 0.5,
    alpha_max: float = 20.0,
    order: float = 2.0,
    smooth: int = 1,
    periodic: bool = False,
    variables=None,
    normalisation: str = "fraction",
    tvb_constant: float = 1.0,  # accepted for apply_limiter() dispatch symmetry
    diagnostics: bool = False,
):
    """Sensor-gated exponential (SVV) filter; drop-in for :func:`apply_limiter`.

    Parameters
    ----------
    coeff : (ncells, nmodes, nvar) modal coefficients.
    dx : cell width.  Accepted for interface symmetry with the limiters; the
        sensor is dimensionless in ``dx`` *only through the coefficients they
        already carry*, so it is not used in the filter itself.
    s0, kappa, alpha_max, order, smooth, periodic, variables :
        see :func:`av_strength`, :func:`av_sensor`,
        :func:`exponential_filter_factor`.
    tvb_constant : ignored (present so that ``apply_limiter(kind="av")`` can
        forward the generic limiter keyword).
    diagnostics : when true, return ``(filtered, info)`` with the measured
        sensor statistics of *this* call.

    Returns
    -------
    filtered : same shape as *coeff*; mode 0 untouched.
    """
    del dx, tvb_constant
    if coeff.dim() != 3 or coeff.shape[1] < 2:
        return (coeff, {"flagged": 0, "sensor_max": float("nan")}) if diagnostics else coeff
    sensor = av_sensor(
        coeff, variables=variables, smooth=smooth, periodic=periodic,
        normalisation=normalisation,
    )
    alpha = av_strength(sensor, s0=s0, kappa=kappa, alpha_max=alpha_max)
    sigma = exponential_filter_factor(
        coeff.shape[1], alpha, order=order, dtype=coeff.dtype, device=coeff.device,
    )
    filtered = coeff * sigma.unsqueeze(-1)
    if diagnostics:
        info = {
            "flagged": int((alpha > 0).sum().item()),
            "saturated": int((alpha >= alpha_max).sum().item()),
            "sensor_max": float(sensor.max().item()),
            "sensor_min": float(sensor.min().item()),
            "alpha_max": float(alpha.max().item()),
        }
        return filtered, info
    return filtered