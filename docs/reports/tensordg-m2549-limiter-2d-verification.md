# Two-dimensional TVB limiter: validation debt and fix (mission m2549)

> 注：本报告原路径为 `docs/limiter_2d_verification.md`。该路径在 main 0ec9e1ea 上已被
> m2503/PR#18 的同名工件占用（与本分支平行开发的限制器线），重建时整体移至本路径，
> 内容未改动（除本注与文内运行命令的模块名同步）。


> Repository: TensorDG. Branch `agent/m2549`. Reproducers: `PYTHONPATH=. python -m benchmarks.verify_limiter_2d_m2549 --tag after` and `PYTHONPATH=. python -m pytest tests/test_euler_dg_2d_limiter.py -v`. All numbers below are injected from `validation/limiter_2d/{before,after}/evidence.json`; no digit is typed by hand.

## 1. What was wrong

Mission m2500 rewrote the 1-D troubled-cell test in `tensordg/limiters.py` into a scale-consistent form: the quantity compared against the TVB tolerance `M dx^2` is `|d_left - minmod(d_left, d_right)|`, which is O(dx^2 u'') on smooth data, instead of the raw one-sided differences `max(|d_left|, |d_right|)`, which are O(dx u'). The rewrite was never propagated to the 2-D limiter in `tensordg/euler_dg_2d.py`.

Two defects follow, both reproduced by the harness below on the pre-fix tree:

1. **The 2-D test is not a TVB test.** On a smooth wave the flagged fraction grows with refinement instead of going to zero, so a fine mesh is limited harder than a coarse one; `tvb_constant` cannot release a smooth extremum because the O(dx u') measure always outgrows `M dx^2` once dx is small.
2. **`apply_tvb_limiter_2d` raises for `degree >= 2`.** The mask that removes the higher modes of a troubled direction is broadcast against the mode axes instead of the cell axes, so any P2/P3 run with an active limiter dies with `RuntimeError`. The four-quadrant verification of mission m2400 only ran the limiter at P1, so the branch was dead code and the README's "optional TVB limiter" was usable at P0/P1 only.

Pre-fix evidence (`--tag before`):

| cells/dir | dx | flagged x (M=1) | fraction x | flagged y (M=1) | fraction y | flagged x (M=2\|u''\|) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.25000 | 12 | 0.1875 | 0 | 0.0000 | 0 |
| 16 | 0.12500 | 84 | 0.3281 | 36 | 0.1406 | 24 |
| 32 | 0.06250 | 404 | 0.3945 | 284 | 0.2773 | 260 |
| 64 | 0.03125 | 1788 | 0.4365 | 1504 | 0.3672 | 1436 |

Structural check: "RuntimeError: The expanded size of the tensor (1) must match the existing size (24) at non-singleton dimension 2.  Target sizes: [24, 24, 1, 3, 4].  Tensor sizes: [24, 24, 24, 3, 4]"

| M | threshold M dx^2 | flagged x | flagged y |
|---:|---:|---:|---:|
| 0.000 | 0.0000e+00 | 512 | 512 |
| 0.500 | 1.9531e-03 | 464 | 388 |
| 1.000 | 3.9062e-03 | 404 | 284 |
| 2.467 | 9.6366e-03 | 260 | 76 |
| 9.868 | 3.8546e-02 | 0 | 0 |

## 2. The fix

* `tensordg/tvb.py` (new): the shared, scale-consistent test `troubled_from_differences(d_left, d_right, spacing, M)`, extracted verbatim from the 1-D limiter so there is a single implementation for both dimensions.
* `tensordg/limiters.py`: `_troubled_cell_indicator` now calls the shared helper (numerically identical to the m2500 behaviour, all pre-existing 1-D tests still pass).
* `tensordg/euler_dg_2d.py`: `_troubled_1d` delegates to the shared helper and gained a `periodic` flag -- wrap-around for periodic axes, edge replication with unflagged boundary rows on outflow axes (the old code always wrapped, so outflow boundaries saw the far side of the domain as a jump). `apply_tvb_limiter_2d` builds a per-cell mask of the right rank for the higher-mode removal, which fixes the P2/P3 crash. `solve_euler_dg_2d_vortex` gained the missing `tvb_constant` argument so the vortex wrapper can actually tune the limiter.

## 3. Post-fix verification

### 3.1 Detector scale consistency (check A)

Measured data curvature max|u''| = 1.2335 -> TVB constant used for the release test M = 2.4670.

| cells/dir | dx | flagged x (M=1) | fraction x | flagged y (M=1) | fraction y | flagged x (M=2\|u''\|) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.25000 | 4 | 0.0625 | 0 | 0.0000 | 0 |
| 16 | 0.12500 | 12 | 0.0469 | 8 | 0.0312 | 0 |
| 32 | 0.06250 | 64 | 0.0625 | 24 | 0.0234 | 0 |
| 64 | 0.03125 | 256 | 0.0625 | 56 | 0.0137 | 0 |

| M | threshold M dx^2 | flagged x | flagged y |
|---:|---:|---:|---:|
| 0.000 | 0.0000e+00 | 512 | 512 |
| 0.500 | 1.9531e-03 | 236 | 28 |
| 1.000 | 3.9062e-03 | 64 | 24 |
| 2.467 | 9.6366e-03 | 0 | 12 |
| 9.868 | 3.8546e-02 | 0 | 0 |

Criterion: {"M_sweep_must_change_the_flagged_set": true, "fraction_at_M1_bounded_by_0_10": true, "fraction_at_M1_no_upward_trend": true, "fraction_at_M_above_curvature_must_be_zero": true, "observed_fraction_M1": [0.0625, 0.046875, 0.0625, 0.0625, 0.0625, 0.046875, 0.0625, 0.0625], "passed": true, "strict_test_must_flag_interior_cells": true, "tvb_constant_used": 2.466967615255271}

### 3.2 Design-order preservation (check B)

**PP1** (design order 2)

| mesh | unlimited rho err | limited rho err | ratio | limited EOC |
|---:|---:|---:|---:|---:|
| 12^2 (M=1) | 6.8816e-03 | 1.3795e-02 | 2.0046 | -- |
| 16^2 (M=1) | 3.8091e-03 | 9.5147e-03 | 2.4979 | 1.29 |
| 24^2 (M=1) | 1.6740e-03 | 6.1633e-03 | 3.6817 | 1.07 |
| 12^2 (M=10) | 6.8816e-03 | 6.8816e-03 | 1.0000 | -- |
| 16^2 (M=10) | 3.8091e-03 | 3.8091e-03 | 1.0000 | 2.06 |
| 24^2 (M=10) | 1.6740e-03 | 1.6740e-03 | 1.0000 | 2.03 |
| 12^2 (M=50) | 6.8816e-03 | 6.8816e-03 | 1.0000 | -- |
| 16^2 (M=50) | 3.8091e-03 | 3.8091e-03 | 1.0000 | 2.06 |
| 24^2 (M=50) | 1.6740e-03 | 1.6740e-03 | 1.0000 | 2.03 |

constants behaving as a TVB limiter (ratio <= 1.10 and EOC >= 1.7): [10.0, 50.0]

**PP2** (design order 3)

| mesh | unlimited rho err | limited rho err | ratio | limited EOC |
|---:|---:|---:|---:|---:|
| 8^2 (M=1) | 3.2751e-03 | 1.7749e-02 | 5.4195 | -- |
| 12^2 (M=1) | 9.1389e-04 | 1.5069e-02 | 16.4894 | 0.40 |
| 16^2 (M=1) | 3.9756e-04 | 1.0526e-02 | 26.4762 | 1.25 |
| 8^2 (M=10) | 3.2751e-03 | 3.2751e-03 | 1.0000 | -- |
| 12^2 (M=10) | 9.1389e-04 | 9.1389e-04 | 1.0000 | 3.15 |
| 16^2 (M=10) | 3.9756e-04 | 3.9756e-04 | 1.0000 | 2.89 |
| 8^2 (M=50) | 3.2751e-03 | 3.2751e-03 | 1.0000 | -- |
| 12^2 (M=50) | 9.1389e-04 | 9.1389e-04 | 1.0000 | 3.15 |
| 16^2 (M=50) | 3.9756e-04 | 3.9756e-04 | 1.0000 | 2.89 |

constants behaving as a TVB limiter (ratio <= 1.10 and EOC >= 2.7): [10.0, 50.0]

### 3.3 Shock robustness (check C)

| mesh | limiter | min rho | min p | max rho | overshoot | undershoot | flagged (x/y) | wall s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| 24^2 | none | 0.11124 | 0.16325 | 2.29449 | 7.9449e-01 | -2.6762e-02 | 417/417 | 39.5 |
| 24^2 | tvb | 0.12987 | 0.23905 | 1.86930 | 3.6930e-01 | -8.1281e-03 | 406/406 | 39.4 |
| 32^2 | none | 0.11290 | 0.22699 | 2.38504 | 8.8504e-01 | -2.5102e-02 | 721/721 | 70.3 |
| 32^2 | tvb | 0.12981 | 0.25144 | 1.98578 | 4.8578e-01 | -8.1916e-03 | 720/720 | 69.6 |

* 24^2: {"initial_density_range": [0.138, 1.5], "initial_min_pressure": 0.3, "limiter_active_at_shocks": true, "overshoot_not_worse": true, "passed": true, "positivity_kept": true, "undershoot_not_worse": true}
* 32^2: {"initial_density_range": [0.138, 1.5], "initial_min_pressure": 0.3, "limiter_active_at_shocks": true, "overshoot_not_worse": true, "passed": true, "positivity_kept": true, "undershoot_not_worse": true}

### 3.4 Structural invariants (check D)

```json
{
  "cell_average_drift_jump": 0.0,
  "cell_average_drift_smooth": 0.0,
  "cells": 24,
  "criterion": {
    "bounded_reconstruction": true,
    "cell_averages_exactly_preserved": true,
    "jump_is_flagged": true,
    "y_invariant_data_not_flagged_in_y": true
  },
  "degree": 2,
  "fraction_of_quadrature_points_inside_neighbour_bound": 1.0,
  "jump_flagged_x": 48,
  "jump_flagged_y": 0
}
```

## 4. Figures

![detector](docs/figures/limiter_2d/detector_scale.png)

![vortex](docs/figures/limiter_2d/vortex_order.png)

![riemann](docs/figures/limiter_2d/riemann_fields.png)

## 5. Caveats

* The 2-D limiter remains *directional*: it limits along x and y independently and does not touch the mixed mode `c_11`, so a two-dimensional shock at 45 degrees is still under-damped compared with a genuinely multidimensional limiter (Krivodonova 2007).
* `M` is problem dependent. The value that makes the limiter inert on the vortex (`M = 50` at P1, see table) is far above the smooth-data curvature; it is a statement about how strong the limiter must be, not a universal recommendation. For shocks the relevant regime is `M = O(1)`, which is what check C exercises.
* The four-quadrant problem has no closed-form solution, so check C measures positivity and over/undershoot, not an error norm; the accuracy claim rests on check B, which does have an exact reference.

