"""Scalar L2 energy and upwind losses on uniform tensor-product DG boxes."""
import math

import torch


def _mean_weights(coeff, dimension):
    if coeff.ndim != 2*dimension or min(coeff.shape) < 1 or not coeff.is_floating_point():
        raise ValueError(f"provide a nonempty {2*dimension}-dimensional real modal field")
    return [(2*torch.arange(n, dtype=torch.float64, device=coeff.device)+1).reciprocal()
            for n in coeff.shape[dimension:]]


def _weighted_mean_square(modes, cell_axes, weights):
    value = modes.square().mean(dim=cell_axes)
    for weight in reversed(weights):
        value = value @ weight
    return value.item()


def modal_energy(coeff, lengths, reference):
    dimension = len(lengths)
    weights = _mean_weights(coeff, dimension)
    if reference is not None and not math.isfinite(reference):
        raise ValueError("energy reference must be finite or None")
    modes = coeff.double()
    if reference != 0.:
        # Shift only the cell-average mode. Never reconstruct the large
        # background or subtract two nearly equal integrated energies.
        modes = modes.clone()
        averages = modes[(Ellipsis,) + (0,)*dimension]
        if reference is None:
            # The absolute mean can lie between adjacent floats. Center in
            # offsets from one cell so that its fractional part is retained.
            offsets = averages - averages[(0,)*dimension]
            averages.copy_(offsets - offsets.mean())
        else:
            averages.sub_(reference)
    mean_square = _weighted_mean_square(modes, tuple(range(dimension)), weights)
    return .5*math.prod(lengths)*mean_square


def upwind_losses(coeff, lengths, speeds):
    dimension = len(lengths)
    weights = _mean_weights(coeff, dimension)
    modes = coeff.double()
    losses = []
    for axis, speed in enumerate(speeds):
        normal = modes.movedim(dimension+axis, -1)
        neighbor = normal.roll(-1, dims=axis)
        # Subtract like even modes before summation so a large uniform
        # background cannot round away a jump carried by higher modes.
        jump = (normal[..., ::2] - neighbor[..., ::2]).sum(dim=-1)
        jump += (normal[..., 1::2] + neighbor[..., 1::2]).sum(dim=-1)
        tangential = weights[:axis] + weights[axis+1:]
        mean_square = _weighted_mean_square(jump, tuple(range(dimension)), tangential)
        area = math.prod(lengths) * coeff.shape[axis] / lengths[axis]
        losses.append(.5*abs(speed)*area*mean_square)
    return tuple(losses)
