# TensorLBM: Mass-Exchange Selector Implementation

## Overview

This module implements the **mass-exchange-selector** pattern from [Maarten-vd-Sande/lbm](https://github.com/Maarten-vd-Sande/lbm), providing a unified interface for selecting among mass-conserving LBM collision schemes with verified conservation properties.

## Architecture

```
TensorLBM/
├── src/tensorlbm/
│   ├── __init__.py              # Public API
│   ├── lattice.py               # D2Q9 lattice definition
│   ├── collision.py             # BGK and MRT collision operators
│   ├── streaming.py             # Periodic streaming operator
│   └── mass_exchange_selector.py # Core selector implementation
├── benchmarks/
│   └── poiseuille.py            # Poiseuille flow validation
├── tests/
│   └── test_mass_exchange_selector.py
└── docs/
    └── mass_exchange_selector.md
```

## Mass-Exchange Selector

The `MassExchangeSelector` class provides:

1. **Scheme Selection**: Choose between BGK, MRT, and TRT collision models
2. **Mass Conservation Verification**: Built-in verification that mass is conserved to machine precision
3. **Momentum Conservation**: Verified momentum conservation (no force) or correct momentum addition (with force)

### Supported Schemes

| Scheme | Description | Relaxation Parameters |
|--------|-------------|----------------------|
| BGK | Single-relaxation-time Bhatnagar-Gross-Krook | `omega` (1/tau) |
| MRT | Multi-relaxation-time in moment space | `s_vec` (Q-vector) |
| TRT | Two-relaxation-time (symmetric/anti-symmetric) | `omega`, `magic_number` |

### Conservation Properties

All schemes guarantee:
- **Mass conservation**: `sum(f_out) == sum(f_in)` to machine precision (< 1e-12 relative drift)
- **Momentum conservation**: `sum(e_i * f_out) == sum(e_i * f_in)` (no force case)

## Validation Results

### Unit Tests (8/8 PASS)

| Test | Description | Status |
|------|-------------|--------|
| test_bgk_mass_conservation | BGK mass conservation | PASS |
| test_mrt_mass_conservation | MRT mass conservation | PASS |
| test_trt_mass_conservation | TRT mass conservation | PASS |
| test_bgk_momentum_conservation | BGK momentum conservation | PASS |
| test_scheme_selection | Scheme differentiation | PASS |
| test_verify_mass_conservation | Verification routine | PASS |
| test_equilibrium_reconstruction | feq macroscopic recovery | PASS |
| test_streaming_periodic | Streaming mass conservation | PASS |

### Poiseuille Flow Benchmark

The Poiseuille flow benchmark validates velocity profile accuracy against the analytical solution:

```
u(y) = (F / (2 * nu)) * y * (H - y)
```

**Current status**: BGK scheme achieves ~11% relative L2 error on 64×32 grid with 2000 steps.
This exceeds the 3% target, indicating the need for:
1. Longer simulation time for steady-state convergence
2. Finer grid resolution
3. Improved bounce-back boundary condition implementation

## Usage

```python
from tensorlbm import D2Q9, MassExchangeSelector, Scheme

# Initialize
rho = torch.ones(64, 128, dtype=torch.float64)
ux = torch.zeros(64, 128, dtype=torch.float64)
uy = torch.zeros(64, 128, dtype=torch.float64)
f = D2Q9.equilibrium(rho, ux, uy)

# Create selector
selector = MassExchangeSelector(scheme=Scheme.BGK, omega=0.8)

# Collision step
f_post = selector.collide(f, rho, ux, uy)

# Verify mass conservation
result = selector.verify_mass_conservation(f, n_steps=100)
print(f"Mass conservation: {result['pass']}")
```

## Relation to Maarten-vd-Sande/lbm

This implementation follows the mass-exchange-selector pattern from the Maarten-vd-Sande/lbm repository:
- Registry-based scheme selection
- Unified collision interface
- Conservation verification built-in

Key differences:
- PyTorch-based (vs original TensorFlow/NumPy)
- D2Q9 lattice (vs D2Q9/D3Q19 in original)
- Added TRT scheme with magic number optimization

## Limitations

1. **Poiseuille accuracy**: Current implementation shows ~11% error vs analytical solution; target is <3%
2. **Boundary conditions**: Simple bounce-back; curved boundaries not yet implemented
3. **Forcing scheme**: Simplified body force; full Guo forcing scheme needs implementation
4. **3D lattices**: Only D2Q9 implemented; D3Q19/D3Q27 pending

## References

- Maarten-vd-Sande/lbm: https://github.com/Maarten-vd-Sande/lbm
- Ginzburg, I., & d'Humières, D. (2003). Multiple-relaxation-time lattice Boltzmann models in three dimensions. Phil. Trans. R. Soc. Lond. A, 360(1792), 445-456.
- Ginzburg, I., Verhaeghe, F., & d'Humières, D. (2008). Two-relaxation-time lattice Boltzmann scheme: About parametrization for velocity, pressure and mixed boundary conditions. Communications in Computational Physics, 3(2), 427-478.
