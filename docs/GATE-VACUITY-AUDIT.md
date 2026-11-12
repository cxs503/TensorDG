# TensorDG 验收门「空转」审计（gate vacuity）

> 生成器 `tools/audit_gate_vacuity.py`（`--check` 防漂移）· 机器门 `tests/test_gate_vacuity_policy.py`
> 扫描面：`tests/test_*.py` 全量静态分析（零算力）

## 判定口径

| 判级 | 含义 |
|---|---|
| VACUOUS | 「工件缺失即 skip」守护验收断言，且该工件**此刻不在树里** → CI 绿但零断言 |
| LATENT | 有缺失即 skip 的写法，但工件已提交 → 该 skip 是永不触发的安全网，离 VACUOUS 只差一次合并门过滤 |
| HARDENED | 用 `assert … .exists()` 硬断言工件，缺失即红 |
| UNRESOLVED | 有工件类 skip，但路径静态不可解析（诚实登记：未裁决，不算通过） |
| PLAIN | 不依赖已提交工件 |

## 统计

- 扫描测试模块 **129**：VACUOUS **2** · LATENT **5** · HARDENED **7** · UNRESOLVED **8** · PLAIN **107**
- skip 位点：工件类 **21** · 环境类 **2**
- 缺失工件（skip 会真的触发）：**2**
- cwd 依赖（相对路径）工件常量：**7**

## 逐模块清单（非 PLAIN）

| 模块 | 判级 | 工件类 skip | 硬断言 | cwd 依赖 |
|---|---|---|---|---|
| `tests/test_case_dg_solid_body_rotation_2d.py` | UNRESOLVED | 1 | 0 | — |
| `tests/test_dg_advection_bench.py` | VACUOUS | 1 | 1 | — |
| `tests/test_euler_dg_2d_temporal_order.py` | LATENT | 1 | 0 | — |
| `tests/test_evidence_integrity.py` | UNRESOLVED | 1 | 0 | — |
| `tests/test_gate_vacuity_policy.py` | HARDENED | 0 | 1 | — |
| `tests/test_in31_hp_ladder.py` | UNRESOLVED | 2 | 0 | — |
| `tests/test_lax_case_pool.py` | UNRESOLVED | 2 | 0 | — |
| `tests/test_ns_heat_conduction.py` | LATENT | 1 | 0 | WALL_ROW_LAW |
| `tests/test_ns_wall_heat_flux_order.py` | LATENT | 1 | 0 | EVIDENCE |
| `tests/test_oblique_shock_theta_beta.py` | LATENT | 1 | 0 | ANCHOR_FILE |
| `tests/test_oss_inventory_dg_registry.py` | HARDENED | 0 | 4 | — |
| `tests/test_p_refinement_exponential.py` | LATENT | 1 | 4 | — |
| `tests/test_repro_cks1998.py` | UNRESOLVED | 1 | 0 | — |
| `tests/test_riemann_vacuum_1d.py` | UNRESOLVED | 1 | 0 | — |
| `tests/test_shu_osher_case_pool.py` | VACUOUS | 2 | 0 | COMMITTED_REFERENCE |
| `tests/test_sipg_backend_comparison.py` | HARDENED | 0 | 1 | — |
| `tests/test_solid_body_rotation_case_pool.py` | HARDENED | 0 | 1 | — |
| `tests/test_swe_manning_friction.py` | HARDENED | 0 | 1 | — |
| `tests/test_swe_wb_drift_law_artifact.py` | HARDENED | 0 | 3 | — |
| `tests/test_upstream_delta.py` | UNRESOLVED | 4 | 0 | — |
| `tests/test_verification_ladder.py` | UNRESOLVED | 1 | 0 | — |
| `tests/test_weno_limiter_policy.py` | HARDENED | 0 | 1 | — |

## VACUOUS 明细（当下零验证）

### `tests/test_dg_advection_bench.py`
- 缺失工件：`/root/tp-agent/work/m16367/artifacts/dg_advection_bench.json`
- L147: `not ARTIFACT.exists()` → skip("artifacts/dg_advection_bench.json not generated yet")

### `tests/test_shu_osher_case_pool.py`
- 缺失工件：`/root/tp-agent/work/m16367/runs/shu_osher_case_pool/metrics.json`
- L65: `not COMMITTED_REFERENCE.exists()` → skip("committed reference missing")
- L113: `not (RUN / 'metrics.json').exists()` → skip("case artefacts not generated")


## 为什么这比「测试绿」重要

本生态的合并门会过滤分支文件（提交里可见「N 个文件被门禁滤除」）。工件一旦被滤掉，`skip` 型验收门就从「断言数字」退化为「断言无」——CI 仍然绿。本审计把这种退化变成可数的数字，并由 `tests/test_gate_vacuity_policy.py` 阻止新增：新出现的 VACUOUS 模块除非在 `ALLOWED_VACUOUS` 里带理由登记（且条目过期即红），否则测试直接红。
