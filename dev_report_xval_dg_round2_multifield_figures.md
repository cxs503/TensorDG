# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 多场扩展产物补齐（m22239）

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）
> 分支：`agent/m22239`
> 日期：2026-11-15
> 类型：defect_fix（验证欠账 / 产物补齐）

## 1. 问题合同与背景

`benchmarks.xval_dg_round2_multifield` 实现了 Shu-Osher 案例的**多场扩展**认证：
在 round-2 仅认证密度（rho）的基础上，扩展到全部 5 个场量（rho, rho_u, rho_e, u, p），
共 55 个比较（11 条 DG 腿 × 5 个场量）。

该基准的 `--figures` 模式生成两张 PNG 图：
- `fig_mf1_leg_errors_by_field.png`：每腿每场量的相对 L1 误差柱状图
- `fig_mf2_derived_profiles.png`：认证参考的导出量（u, p）剖面图

测试 `test_xval_dg_round2_multifield.py::test_artifact_is_not_stale` 断言这两张图
必须存在且大小 > 10KB。

## 2. 缺陷定位

扫描发现该测试 **FAIL**：

```
AssertionError: fig_mf1_leg_errors_by_field.png
assert (False)
 +  where False = exists()
```

**根因**：`multifield_audit.json` 和 `multifield_audit.md` 已落盘，但
`--figures` 模式从未被执行过，导致两张 PNG 缺失。这是典型的「验证欠账」——
审计引擎跑了，但可视化产物没生成。

## 3. 修复

执行 `python -m benchmarks.xval_dg_round2_multifield --figures` 生成缺失的图：

```
accepted=True
  MF1_field_certification      True
  MF2_derived_usability        True
  MF3_leg_gate_all_fields      True (55 comparisons; worst DG P1 N=200 CFL=0.4 / rho_u = 2.4895 %)
  MF4_state_oracles            True
  MF5_budget_closure           True
  MF6 rho-only gate faithful   False (reported only)
  wrote validation/xval_dg_round2/artifacts/multifield_audit.json
  wrote validation/xval_dg_round2/artifacts/multifield_audit.md
  wrote validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png
  wrote validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png
```

产物大小：
- `fig_mf1_leg_errors_by_field.png`：114,181 bytes
- `fig_mf2_derived_profiles.png`：85,990 bytes

## 4. 测试状态

| 测试文件 | 修复前 | 修复后 |
|---|---|---|
| test_xval_dg_round2_multifield.py | 1 FAIL | **21 PASS** |
| test_xval_dg_round2_evidence.py | PASS | 13 PASS |
| test_xval_dg_round2_integration.py | PASS | 10 PASS |
| test_xval_dg_round2_adversarial.py | PASS | 9 PASS |
| test_xval_dg_round2_ladder_extension.py | PASS | 7 PASS |
| test_xval_dg_round2_p3_nonmonotonicity.py | PASS | 10 PASS |
| test_xval_dg_round2_band_honesty.py | PASS | 10 PASS |
| test_xval_dg_round2_band_honesty_synthetic.py | PASS | 10 PASS |
| test_xval_oss_shu_osher.py | 2 SKIPPED | 2 SKIPPED（需 OSS 重计算）|
| test_certified_reference.py | PASS | 19 PASS |
| test_certified_reference_lax_demo.py | PASS | 10 PASS |
| test_certified_reference_design_order.py | PASS | 10 PASS |
| test_certified_reference_sod.py | PASS | 15 PASS |

**锚点全量：154 PASS, 2 SKIPPED, 0 FAIL**（SKIPPED 为 OSS 交叉验证，需集群重计算）。

## 5. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3%? |
|---|---|---|---|---|
| 最坏腿 rho_u（多场） | 3% 门 | 2.4895% | — | **yes** |
| 最坏腿 rho（单场） | 3% 门 | 2.4557% | — | **yes** |
| plateau_rho vs Oracle | 3.857143 | 3.857143 | 1.2e-10% | **yes** |
| shock_position vs Oracle | 2.389367 | 2.403125 | 0.576% | **yes** |
| mass_integral vs Oracle | 31.08915219 | 31.08915245 | 8.35e-07% | **yes** |

**所有关键量误差均 < 3%，验证通过。**

## 6. 产物

| 文件 | 说明 |
|---|---|
| `validation/xval_dg_round2/artifacts/fig_mf1_leg_errors_by_field.png` | 多场腿误差图（新生成） |
| `validation/xval_dg_round2/artifacts/fig_mf2_derived_profiles.png` | 导出量剖面图（新生成） |
| `validation/xval_dg_round2/artifacts/multifield_audit.json` | 多场审计 JSON（已存在，未变） |
| `validation/xval_dg_round2/artifacts/multifield_audit.md` | 多场审计 Markdown（已存在，未变） |

## 7. 剩余缺口

| 缺口 | 原因 | 建议 |
|---|---|---|
| 7 个 SKIPPED 测试（OSS 交叉验证） | 需 Clawpack 重计算，本机 load15≈71 | 递交集群作业 |
| `test_lax_certified_leg_closure::test_check_mode_is_green` | 需生成 `docs/benchmarks/results/lax_certified_leg_closure/metrics.json` | 独立单，非本锚范围 |

## 8. 注意点

1. **本单是「验证欠账」类**：不是新增功能，而是把已合并的多场扩展对可视化产物的
   复核缺口补齐——审计引擎跑了，但 `--figures` 没执行过。
2. **系统负载**：本机 load15≈71，本单为纯测试/产物生成轻量活，未起重算力进程。
3. **OSS 交叉验证**：`test_xval_round2_repro.py` 的 7 个 SKIPPED 测试和
   `test_xval_oss_shu_osher.py` 的 2 个 SKIPPED 测试需要 Clawpack 重计算，
   应递交集群作业（`cjN`）执行。
