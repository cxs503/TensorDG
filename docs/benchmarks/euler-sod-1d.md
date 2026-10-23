# Benchmark: 1D Euler Sod shock tube (PyTorch, DG P0 baseline)

## Purpose

The Sod shock tube is a canonical compressible Euler benchmark containing a left-going rarefaction, a contact discontinuity, and a right-going shock. It tests conservative fluxes, wave-speed estimates, time integration, and physical admissibility.

## Governing equations

The conservative Euler equations are U_t + F(U)_x = 0, with U=(rho, rho*u, E) and F=(rho*u, rho*u^2+p, u*(E+p)). Pressure is p=(gamma-1)*(E-0.5*rho*u^2), gamma=1.4.

Domain: [0,1]. The initial discontinuity is at x=0.5:
- left: rho=1, u=0, p=1
- right: rho=0.125, u=0, p=0.1

Default final time: t=0.2.

## Numerical method and scope

- PyTorch tensor state and conservative Euler flux.
- Local Lax-Friedrichs / Rusanov interface flux.
- SSP-RK3 time stepping with a CFL-limited step.
- Transmissive boundary conditions.
- The first implementation is DG polynomial degree zero (P0), equivalent to a first-order finite-volume method. It is a baseline for flux and conservation checks, not yet a high-order shock-capturing DG solver. Higher-order modes, limiting and positivity-preserving treatment are future work.

## Run

    python -m pip install torch numpy pytest matplotlib
    python benchmarks/euler_sod_1d.py --cells 400 --time 0.2 --output sod_results.csv
    python benchmarks/plot_euler_sod_1d.py --input sod_results.csv --output sod_profiles.png
    pytest -q

The CSV contains x, density, velocity, and pressure. The plotting utility writes a three-panel density/velocity/pressure profile image to `sod_profiles.png`; it uses a non-interactive Matplotlib backend so it also runs in CI.

## Acceptance checks

1. Density and pressure remain positive.
2. State and output fields remain finite.
3. The density profile develops the expected expansion/contact/shock structure.
4. Early-time mean density remains close to the initial 0.5625 before boundary waves can affect the domain.

## Status

Implementation and regression tests are added on a feature branch. Do not treat this as validated until CI has executed successfully and the output profile has been inspected against a trusted reference.
