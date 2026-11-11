# WENO limiter smooth-flow adjudication — machine tables


| ladder | deg | cells | M | rho rel-L2 | rho x unr. | worst field x unr. | flagged (x/y/either) | fraction | bitwise |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| isotropic | 1 | 8x8 | 1 | 0.0147 | 1.04 | 1.04 | 0/0/0 | 0.000 | no |
| isotropic | 1 | 12x12 | 1 | 0.0179 | 2.6 | 2.71 | 4/2/5 | 0.035 | no |
| isotropic | 1 | 16x16 | 1 | 0.0116 | 3.04 | 3.05 | 6/4/9 | 0.035 | no |
| isotropic | 1 | 8x8 | 2 | 0.0141 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 12x12 | 2 | 0.0080 | 1.16 | 1.17 | 0/0/0 | 0.000 | no |
| isotropic | 1 | 16x16 | 2 | 0.0077 | 2.02 | 2.02 | 1/3/4 | 0.016 | no |
| isotropic | 1 | 8x8 | 10 | 0.0141 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 12x12 | 10 | 0.0069 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 16x16 | 10 | 0.0038 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 8x8 | 1e+06 | 0.0141 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 12x12 | 1e+06 | 0.0069 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 1 | 16x16 | 1e+06 | 0.0038 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 8x8 | 1 | 0.0216 | 6.61 | 7.3 | 0/0/0 | 0.000 | no |
| isotropic | 2 | 12x12 | 1 | 0.0198 | 21.6 | 23.1 | 4/2/5 | 0.035 | no |
| isotropic | 2 | 16x16 | 1 | 0.0125 | 31.5 | 31.5 | 7/5/10 | 0.039 | no |
| isotropic | 2 | 8x8 | 2 | 0.0033 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 12x12 | 2 | 0.0073 | 7.98 | 9.07 | 0/0/0 | 0.000 | no |
| isotropic | 2 | 16x16 | 2 | 0.0077 | 19.3 | 19.3 | 2/3/5 | 0.020 | no |
| isotropic | 2 | 8x8 | 10 | 0.0033 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 12x12 | 10 | 9.1389e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 16x16 | 10 | 3.9756e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 8x8 | 1e+06 | 0.0033 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 12x12 | 1e+06 | 9.1389e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| isotropic | 2 | 16x16 | 1e+06 | 3.9756e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 8x16 | 1 | 0.0125 | 1.18 | 1.21 | 0/2/2 | 0.016 | no |
| anisotropic | 1 | 12x24 | 1 | 0.0142 | 2.77 | 2.77 | 8/7/13 | 0.045 | no |
| anisotropic | 1 | 16x32 | 1 | 0.0090 | 3.19 | 3.19 | 11/13/21 | 0.041 | no |
| anisotropic | 1 | 8x16 | 2 | 0.0107 | 1.01 | 1.05 | 0/1/1 | 0.008 | no |
| anisotropic | 1 | 12x24 | 2 | 0.0077 | 1.5 | 1.5 | 2/4/6 | 0.021 | no |
| anisotropic | 1 | 16x32 | 2 | 0.0053 | 1.87 | 1.87 | 4/4/8 | 0.016 | no |
| anisotropic | 1 | 8x16 | 10 | 0.0105 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 12x24 | 10 | 0.0051 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 16x32 | 10 | 0.0028 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 8x16 | 1e+06 | 0.0105 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 12x24 | 1e+06 | 0.0051 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 1 | 16x32 | 1e+06 | 0.0028 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 8x16 | 1 | 0.0124 | 5.2 | 5.2 | 0/2/2 | 0.016 | no |
| anisotropic | 2 | 12x24 | 1 | 0.0150 | 22.7 | 22.9 | 8/7/13 | 0.045 | no |
| anisotropic | 2 | 16x32 | 1 | 0.0092 | 32.3 | 36.4 | 11/14/22 | 0.043 | no |
| anisotropic | 2 | 8x16 | 2 | 0.0070 | 2.93 | 2.93 | 0/1/1 | 0.008 | no |
| anisotropic | 2 | 12x24 | 2 | 0.0081 | 12.2 | 12.2 | 3/3/6 | 0.021 | no |
| anisotropic | 2 | 16x32 | 2 | 0.0049 | 17.2 | 17.2 | 3/3/6 | 0.012 | no |
| anisotropic | 2 | 8x16 | 10 | 0.0024 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 12x24 | 10 | 6.6156e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 16x32 | 10 | 2.8501e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 8x16 | 1e+06 | 0.0024 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 12x24 | 1e+06 | 6.6156e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |
| anisotropic | 2 | 16x32 | 1e+06 | 2.8501e-04 | 1 | 1 | 0/0/0 | 0.000 | yes |


| ladder | deg | M | design order | worst EOC (fields) | worst error ratio | bitwise all |
|---|---:|---:|---:|---|---:|---|
| isotropic | 1 | 1 | 2 | 0.294 | 3.05 | no |
| isotropic | 1 | 2 | 2 | 0.915 | 2.02 | no |
| isotropic | 1 | 10 | 2 | 1.877 | 1 | yes |
| isotropic | 1 | 1e+06 | 2 | 1.877 | 1 | yes |
| isotropic | 2 | 1 | 3 | 0.361 | 31.5 | no |
| isotropic | 2 | 2 | 3 | -1.567 | 19.3 | no |
| isotropic | 2 | 10 | 3 | 2.738 | 1 | yes |
| isotropic | 2 | 1e+06 | 3 | 2.738 | 1 | yes |
| anisotropic | 1 | 1 | 2 | 0.416 | 3.19 | no |
| anisotropic | 1 | 2 | 2 | 0.994 | 1.87 | no |
| anisotropic | 1 | 10 | 2 | 1.891 | 1 | yes |
| anisotropic | 1 | 1e+06 | 2 | 1.891 | 1 | yes |
| anisotropic | 2 | 1 | 3 | -0.094 | 36.4 | no |
| anisotropic | 2 | 2 | 3 | -0.018 | 17.2 | no |
| anisotropic | 2 | 10 | 3 | 2.758 | 1 | yes |
| anisotropic | 2 | 1e+06 | 3 | 2.758 | 1 | yes |


| deg | cells | unlimited rho | limited rho | ratio | flagged | mode-0 changed |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 8x8 | 0.0051 | 0.0226 | 4.46 | 0.984 | 0 |
| 1 | 16x16 | 0.0022 | 0.0066 | 3.01 | 0.945 | 0 |
| 1 | 32x32 | 5.5595e-04 | 0.0016 | 2.87 | 0.942 | 0 |
| 1 | 64x64 | 1.3955e-04 | 3.4651e-04 | 2.48 | 0.924 | 0 |
| 2 | 8x8 | 0.0025 | 0.0228 | 9.1 | 0.984 | 0 |
| 2 | 16x16 | 2.3906e-04 | 0.0065 | 27.3 | 0.945 | 0 |
| 2 | 32x32 | 3.0743e-05 | 0.0016 | 51.2 | 0.942 | 0 |
| 2 | 64x64 | 3.8615e-06 | 3.4141e-04 | 88.4 | 0.924 | 0 |
| 3 | 8x8 | 8.9697e-05 | 0.0228 | 254 | 0.984 | 0 |
| 3 | 16x16 | 2.4182e-05 | 0.0065 | 270 | 0.945 | 0 |
| 3 | 32x32 | 1.5019e-06 | 0.0016 | 1.05e+03 | 0.942 | 0 |
| 3 | 64x64 | 9.4422e-08 | 3.4141e-04 | 3.62e+03 | 0.924 | 0 |


| deg | EOC rho | EOC u | EOC v | EOC p | worst limited/unlimited on finest |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.013 | 2.080 | 1.984 | 2.039 | 2.49 |
| 2 | 2.024 | 2.089 | 1.983 | 2.049 | 102 |
| 3 | 2.023 | 2.089 | 1.983 | 2.049 | 3.62e+03 |


| cells | M | limiter | rho rel-L2 | flagged either |
|---|---:|---|---:|---:|
| 8x8 | 1 | tvb | 0.0177 | 0 |
| 8x8 | 1 | weno | 0.0216 | 0 |
| 16x16 | 1 | tvb | 0.0105 | 8 |
| 16x16 | 1 | weno | 0.0125 | 10 |
| 8x8 | 10 | tvb | 0.0033 | 0 |
| 8x8 | 10 | weno | 0.0033 | 0 |
| 16x16 | 10 | tvb | 3.9756e-04 | 0 |
| 16x16 | 10 | weno | 3.9756e-04 | 0 |


| criterion | value | verdict |
|---|---|---|
| `inert_M_bitwise_matches_unlimited` | True | PASS |
| `flagged_fraction_bounded_at_M1` | 0.04513888888888889 | PASS |
| `flagged_count_monotone_in_M` | True | PASS |
| `design_order_recovered_when_M_above_data_bound` | isotropic-p1-M10: 1.884, isotropic-p2-M10: 3.049, anisotropic-p1-M10: 1.891, anisotropic-p2-M10: 3.067 | PASS |
| `forced_limiter_representation_order` | p1: {'density': 2.013, 'velocity_x': 2.08, 'velocity_y': 1.984, 'pressure': 2.039}, p2: {'density': 2.024, 'velocity_x': 2.089, 'velocity_y': 1.983, 'pressure': 2.049}, p3: {'density': 2.023, 'velocity_x': 2.089, 'velocity_y': 1.983, 'pressure': 2.049} | PASS |
| `forced_limiter_preserves_cell_average` | 0 | PASS |
| `forced_limiter_flags_all_cells` | 0.923583984375 | FAIL |

