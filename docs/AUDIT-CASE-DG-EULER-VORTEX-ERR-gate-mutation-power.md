| 变异 | 目标判据 | 注入的缺陷 | 裁决 | 目标判据读数 | 被杀 | 附带翻转 |
|---|---|---|---|---|---|---|
| `m01` | C1_tolerance | checkpoint above the 6% acceptance | FAIL | failed | 是 | C11_provenance_integrity |
| `m02` | C2_error_conservation | super-linear accumulation, design family | FAIL | failed | 是 | C10_direction_invariance, C11_provenance_integrity, C7_design_space_invariance, C8_degree_invariance |
| `m03` | C3_discrete_conservation | mass leak of 1% | FAIL | failed | 是 | 无 |
| `m04` | C4_entropy_budget | entropy budget overrun | FAIL | failed | 是 | 无 |
| `m05` | C5_crosscheck_order | cross-check path stops converging | FAIL | failed | 是 | C11_provenance_integrity, C1_tolerance |
| `m06` | C6_ladder_completeness | checkpoint evidence missing | FAIL | failed | 是 | C9_design_order |
| `m07` | C7_design_space_invariance | growth off the design point | FAIL | failed | 是 | C11_provenance_integrity, C9_design_order |
| `m08` | C8_degree_invariance | growth at a non-design degree | FAIL | failed | 是 | C7_design_space_invariance |
| `m09` | C9_design_order | refinement stops buying accuracy | FAIL | failed | 是 | C11_provenance_integrity, C7_design_space_invariance |
| `m10` | C10_direction_invariance | growth on a rotated direction | FAIL | failed | 是 | C7_design_space_invariance |
| `m11` | C11_provenance_integrity | certificate no longer reproduces | FAIL | failed | 是 | 无 |
| `m12` | C11_provenance_integrity | legacy record left unverified | INCOMPLETE | not_evaluated | 是 | 无 |
| `c01` | —（对照） | control: timing metadata | PASS | — | — | 无 |
| `c02` | —（对照） | control: iteration counters | PASS | — | — | 无 |

### 全判据 x 全变异状态矩阵

| 判据 | 基线 | m01 | m02 | m03 | m04 | m05 | m06 | m07 | m08 | m09 | m10 | m11 | m12 | c01 | c02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1_tolerance | passed | failed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C2_error_conservation | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C3_discrete_conservation | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C4_entropy_budget | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C5_crosscheck_order | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed | passed |
| C6_ladder_completeness | passed | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | passed |
| C7_design_space_invariance | passed | passed | failed | passed | passed | passed | passed | failed | failed | failed | failed | passed | passed | passed | passed |
| C8_degree_invariance | passed | passed | failed | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed | passed | passed |
| C9_design_order | passed | passed | passed | passed | passed | passed | not_evaluated | failed | passed | failed | passed | passed | passed | passed | passed |
| C10_direction_invariance | passed | passed | failed | passed | passed | passed | passed | passed | passed | passed | failed | passed | passed | passed | passed |
| C11_provenance_integrity | passed | failed | failed | passed | passed | failed | passed | failed | passed | failed | passed | failed | not_evaluated | passed | passed |

### 门限标定（阈值两侧各一针）

| 标定 | 判据 | 声明阈值 | 来源 | 失败方向 | 内侧读数 | 外侧读数 | 咬合 |
|---|---|---|---|---|---|---|---|
| `k01` | C1_tolerance | 6.000e-02 | `TOLERANCE` | fails above | 5.994e-02 → passed | 6.006e-02 → failed | 是 |
| `k02` | C2_error_conservation | 1.000e+00 | `MAX_GROWTH_EXPONENT` | fails above | 9.990e-01 → passed | 1.001e+00 → failed | 是 |
| `k03` | C3_discrete_conservation | 1.000e-10 | `INVARIANT_TOLERANCE` | fails above | 9.990e-11 → passed | 1.001e-10 → failed | 是 |
| `k04` | C4_entropy_budget | 1.000e-03 | `ENTROPY_TOLERANCE` | fails above | 9.990e-04 → passed | 1.001e-03 → failed | 是 |
| `k05` | C5_crosscheck_order | 8.000e-01 | `0.8 (literal in evaluate_gates)` | fails below | 8.008e-01 → passed | 7.992e-01 → failed | 是 |
| `k09` | C9_design_order | 2.400e+00 | `ORDER_SLACK*(p+1), p=2` | fails below | 2.402e+00 → passed | 2.398e+00 → failed | 是 |
| `k11` | C11_provenance_integrity | 1.000e-09 | `PROVENANCE_REPLAY_TOLERANCE` | fails above | 9.990e-10 → passed | 1.001e-09 → failed | 是 |