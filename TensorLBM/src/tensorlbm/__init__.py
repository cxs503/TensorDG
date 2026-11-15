"""TensorLBM — PyTorch-based Lattice Boltzmann solver.

Modules:
    - D2Q9 lattice definition
    - BGK, MRT, TRT collision operators
    - Periodic streaming
    - Mass-exchange selector (unified collision interface)
    - MCMP (Multi-Component Multi-Phase) solver with Shan-Chen cross-force
    - Cross-force operator with symbolic verification and spectral analysis
    - Sparse list-based AOS LBM (lbmpy sparse-list-based-aos-lbm pattern)
"""

__version__ = "0.3.0"

from .lattice import D2Q9
from .collision import bgk_collision, mrt_collision
from .streaming import stream_periodic
from .mass_exchange_selector import MassExchangeSelector, Scheme
from .cross_force import (
    CrossForceOperator,
    pseudopotential_rho,
    pseudopotential_shan_chen,
    pseudopotential_carnahan_starling,
    symbolic_cross_force_verification,
    spectral_analysis_cross_force,
)
from .mcmp import MCMPSolver, setup_laplace_test, measure_laplace_pressure
from .sparse_aos import (
    SparseAOSLBM,
    SparseDomain,
    build_sparse_domain,
    channel_mask,
)

__all__ = [
    "D2Q9",
    "bgk_collision",
    "mrt_collision",
    "stream_periodic",
    "MassExchangeSelector",
    "Scheme",
    "CrossForceOperator",
    "pseudopotential_rho",
    "pseudopotential_shan_chen",
    "pseudopotential_carnahan_starling",
    "symbolic_cross_force_verification",
    "spectral_analysis_cross_force",
    "MCMPSolver",
    "setup_laplace_test",
    "measure_laplace_pressure",
    "SparseAOSLBM",
    "SparseDomain",
    "build_sparse_domain",
    "channel_mask",
]
