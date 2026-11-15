# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 多场扩展（实质增量交付）

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger acceptance：对标误差 <15%）
> 分支：`agent/m21915`；平台报告：`report_id 19761`（submission `156ed558b5ef4ccfa23390a2b9f778b2`）
> 完整论文式文档：`docs/CASE-XVAL-DG-ROUND2-MULTIFIELD.md`（由审计 JSON 程序化生成）

## 1. 一句话

round-2 认证参考只认证了密度 ρ，并用它给 11 条 DG 腿判 "rel L1 < 3%"。
本单把认证与门限扩展到全部三支守恒场（ρ、ρu、ρE）与两个派生量（u、p），
把 11 腿的判决从 11 次比较变成 55 次，并补上两个密度看不见的闭式预算
（动量、能量）。结论：55/55 全部 PASS（最坏 ρu @ P1 N=200 = 2.4895% < 3%），
但如实记录两件事：密度门对 ρu 是**略偏乐观**的代理（最大 1.1848 倍），
以及 P3 在 N=800 上对 ρ/ρu **不单调**。

## 2. 为什么这不是重做（与既有工件的关系）

| 既有 | 覆盖 | 本单增量 |
|---|---|---|
| `benchmarks/xval_dg_round2_certified.py`（D1–D4） | 只 ρ：`load_ladder` 只读 `rho` | 未改动；本单另起 `xval_dg_round2_multifield.py`，读 `rho_u`/`rho_e` |
| `benchmarks/xval_dg_round2_ladder_extension.py`（L0–L4） | 参考梯次是否可扩展到 N=6400 | 未改动；本单扩的是「场的维度」，不是梯次 |
| `validation/xval_dg_round2/artifacts/xval_round2_audit.json` | ρ 的带与 11 条腿 | 未改动；本单产物 `multifield_audit.{json,md}` |

三个方案决策（MF1–MF3）都用**源码内预声明**的阈值与带宽规则，规则在看到结果前写死
（`BAND_RULE` = 加性三角不等式；严格门 3%，Oracle 1%，预算 0.01%）。

## 3. 关键数字（全部从 `multifield_audit.json` 直读）

| 量 | ρ | ρu | ρE | u | p |
|---|---|---|---|---|---|
| 观测阶 p | 1.2825 | 1.3132 | 1.3013 | — | — |
| GCI 带 U % | 0.0828 | 0.0789 | 0.0416 | 0.1617 | 0.2033 |
| 最坏腿 rel L1 % | 2.4557 | 2.4895 | 1.1804 | 0.7411 | 0.8300 |

闭式 Oracle（认证参考 vs 解析）：平台 ρ 1.2e-10%、u 9.4e-11%、p 1.7e-10%；
熵波后未扰动区 u ≡ 0、p ≡ 1（逐位）。预算闭合：质量 8.4e-07%、动量 3.0e-11%、能量 3.0e-11%。

## 4. 验证（跑了什么）

```bash
python -m benchmarks.xval_dg_round2_multifield --figures      # accepted=True, 出 2 图
python -m pytest tests/test_xval_dg_round2_multifield.py -q   # 21 passed
python -m pytest tests/test_xval_dg_round2_*.py -q            # 77 passed（含既有 56 项，无回归）
```

测试里有 3 个**负控制**，证明门不是摆设：篡改某腿 `rho_u × 1.05` → MF3 必须红；
篡改 `rho × 1.10` → `--require-accepted` 必须返回 1；DG 家族自认证 ρu/ρE 必须被拒。
另有 anti-staleness 测试：仓库里的审计 JSON 必须与现算结果逐位一致，否则测试红。

## 5. 未完成 / 下一步线索

1. **跨族参考**：五个量仍都在 FV-WENO5 家族上自认证，界定不了同族系统偏差。
   真正的下一步是用第二种离散（不同通量/不同限制器的 FV，或 DG P1 在 N=6400 上）
   在 N=1600 网格上做「跨族一致性 ≤ U」的检验——需要一次集群全量重算（本机 load15
   已 >70，未在本单强跑）。
2. **P3 非单调**值得单独立案：`dg_p3_n800` 的 ρ/ρu 误差大于 P2 的同网格值，
   可能是 WENO 限制器作用于 P3 模态时的截断点选择问题（`tensordg/limiters`）。
3. 动量/能量预算的闭式性依赖本算例几何，若要做通用工具，
   应把 `closed_form_budgets()` 提升为「边界通量账」框架并按案例注入。
