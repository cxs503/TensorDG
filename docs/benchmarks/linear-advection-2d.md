# Benchmark: 2D periodic linear advection (tensor-product DG)

## Problem

This benchmark solves `u_t + ax*u_x + ay*u_y = 0` on the periodic unit square.
The initial condition is `1 + sin(2*pi*x)*sin(2*pi*y)`, so the exact solution
is the same profile translated by `(ax*t, ay*t)`. The nonzero mean makes the
mass-conservation check nontrivial.

## Method

- Uniform Cartesian mesh with tensor-product modal Legendre bases P0--P3.
- Dimension-by-dimension weak DG volume terms and upwind face fluxes.
- Classical RK4 by default, with `dt = CFL / ((2*p+1)*(|ax|/hx+|ay|/hy))`.
- Gauss--Legendre initial projection and L2-error quadrature.
- PyTorch tensors for the state and semi-discrete operator.

The solver also accepts `time_integrator="euler"`, `"rk2"`, `"rk3"`,
`"ssp_rk3"`, `"ssp_rk54"` and `"rk5"`. The returned metadata records the
selected method, its order and its number of stages.

The documented case uses P1 on a 24x24 mesh, velocity `(1,0.5)`, CFL 0.08,
and final time 1. Its relative L2 error must be below 3%; the regression
suite also checks second-order convergence and negative velocity components.

For a physical rectangle, pass the same side lengths to the solver and the
error integral. A custom initial profile receives physical coordinates.

```python
from tensordg import error_2d, solve_advection_2d

lengths = (3.0, 0.5)
velocity = (-0.75, 0.25)
x, y, coeff, meta = solve_advection_2d((24, 26), 2, 0.125, velocity, lengths=lengths)
metrics = error_2d(x, y, coeff, 0.125, velocity, lengths=meta["lengths"])
```

To continue on the same mesh, pass the returned modal coefficients and end
time into the next solve. The solver copies the coefficients. Splitting a run
can change the time step because each call lands on its requested end time.

```python
x, y, continued, next_meta = solve_advection_2d(
    meta["ncells"], meta["degree"], 0.25, velocity, lengths=meta["lengths"],
    start_time=meta["final_time"], initial=coeff,
)
metrics = error_2d(x, y, continued, 0.25, velocity, lengths=lengths)
```

For an exact translation check, pass the elapsed time since the analytic
initial profile was prescribed to `error_2d`.

A spatially uniform `velocity(t)` may vary in time. Each Runge–Kutta stage
uses its absolute time. Set `max_speed=(ax_max, ay_max)` for known absolute
component bounds and `max_dt` to resolve short pulses. Without explicit
bounds, the solver samples each smooth interval at five times and checks the
CFL at the stages. For right-continuous jumps pass the interior switch times
in `velocity_jump_times`. To verify a known displacement, use its components
as the velocity in `error_2d` with elapsed time set to one.

Use `fx, fy = advection_face_fluxes_2d(coeff, velocity, lengths=lengths)`
from `tensordg` to obtain instantaneous transport rates. The two entries
at `[i,j]` are the face integrals on the right and top of that cell, signed
in the positive coordinate directions. Summing `fx[i,:]` gives the rate
across the entire vertical section. For a velocity callback, supply its value
at the output time.

For cumulative transport, pass `return_integrated_fluxes=True` to
`solve_advection_2d` and unpack `x, y, coeff, meta, (Fx, Fy)`. Both face
arrays cover the full interval from `start_time` to `final_time`. They use
the same stage velocities, upwind states and Runge–Kutta weights as the cell
update. The cell mass gain equals
`Fx.roll(1, 0) - Fx + Fy.roll(1, 1) - Fy`, including across periodic seams.

For the quadratic scalar energy, call `energy_2d(coeff, lengths=lengths)`.
It integrates half the squared field including all Legendre modes.
`upwind_dissipation_2d(coeff, velocity, lengths=lengths)` returns the
nonnegative x and y contributions to the instantaneous energy loss.
Their sum equals the negative semidiscrete energy derivative for periodic
transport with a spatially uniform velocity. For a velocity callback pass its
value at the sample time.

## Run

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.linear_advection_2d --cells 24 --degree 1 \
  --output-dir docs/benchmarks/results/linear-advection-2d
```

The command writes exact/numerical/error contours, a non-degenerate y=0.25
centerline comparison, CSV data, and a JSON metrics record. See the
[Chinese classic-benchmark tutorial](../tutorials/classic-dg-benchmarks.md)
for equations, measured results, and images.
