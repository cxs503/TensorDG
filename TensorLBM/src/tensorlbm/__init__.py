"""TensorLBM — PyTorch-based Lattice Boltzmann solver with mass-exchange selector.

Implements the mass-exchange-selector approach from Maarten-vd-Sande/lbm:
a unified interface for selecting among mass-conserving exchange schemes
(BGK, MRT, entropic) with verified conservation properties.
"""

__version__ = "0.1.0"

from .mass_exchange_selector import MassExchangeSelector, Scheme
from .collision import bgk_collision, mrt_collision
from .streaming import stream_periodic
from .lattice import D2Q9

__all__ = [
    "MassExchangeSelector",
    "Scheme",
    "bgk_collision",
    "mrt_collision",
    "stream_periodic",
    "D2Q9",
]
