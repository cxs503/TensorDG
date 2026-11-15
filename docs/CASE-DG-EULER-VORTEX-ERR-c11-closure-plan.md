# CASE-DG-EULER-VORTEX-ERR：C11 证据溯源收口计划

> 锚点 `CASE-DG-EULER-VORTEX-ERR` | 算例池·等熵涡移动误差守恒
> ref: Yee, Sandham & Djomehri JCP 162 (2000) | 容差 <6%

## 当前状态（m21897 快照）

| 判据 | 状态 | 备注 |
|---|---|---|
| C1 容差 | ✅ 通过 | 最差 4.9260%（p0_n64_t1_cfl0.4）< 6% |
| C2 误差守恒 | ✅ 通过 | alpha = 0.313 <= 1.0 |
| C3 离散守恒 | ✅ 通过 | 最大漂移 3.911e-13 < 1e-10 |
| C4 熵预算 | ✅ 通过 | 6.336e-07 < 0.001 |
| C5 交叉校核 | ✅ 通过 | P0 观测阶 0.849 >= 0.8 |
| C6 阶梯完整性 | ✅ 通过 | 40/40 检查点已落盘 |
| C7 设计空间不变性 | ✅ 通过 | 8 个设计点，最大 alpha 0.440 |
| C8 阶数不变性 | ✅ 通过 | [1,2,3] 均可判 |
| C9 设计阶 | ✅ 通过 | 最差 2.042 >= 1.6 |
| C10 方向不变性 | ✅ 通过 | 3 个方向，最大 alpha 0.313 |
| C11 证据溯源 | ⏸ 未判定 | 7 个记录待重放证书 |

**整包裁决：INCOMPLETE**（仅因 C11 待重放）

## C11 缺口明细

以下 7 个 DG-P2 检查点缺少重放证书（`reference_model` 字段缺失且无 replay certificate）：

| 检查点 | 网格 | CFL | t | 存储误差 % |
|---|---|---|---|---|
| dg_d2_n32_t2_cfl0.2 | 32 | 0.2 | 2.0 | 0.00969 |
| dg_d2_n32_t5_cfl0.2 | 32 | 0.2 | 5.0 | 0.01113 |
| dg_d2_n32_t10_cfl0.2 | 32 | 0.2 | 10.0 | 0.01516 |
| dg_d2_n32_t10_cfl0.4 | 32 | 0.4 | 10.0 | 0.01517 |
| dg_d2_n64_t1_cfl0.2 | 64 | 0.2 | 1.0 | 0.00115 |
| dg_d2_n64_t2_cfl0.2 | 64 | 0.2 | 2.0 | 0.00115 |
| dg_d2_n64_t10_cfl0.2 | 64 | 0.2 | 10.0 | 0.00122 |

## 收口方法

使用 `tools/provenance_certify.py` 重放上述 7 个检查点，与当前 `min_image` 参考模型比对，
生成重放证书（replay delta 预期 < 1e-9）。

已有 12 个证书的先例表明：DG-P2 的重放 delta 在 1e-13 ~ 1e-12 量级，远低于 1e-9 容差。

### 执行命令

```bash
PYTHONPATH=. python3 tools/provenance_certify.py \
  --keys dg_d2_n32_t2_cfl0.2 dg_d2_n32_t5_cfl0.2 dg_d2_n32_t10_cfl0.2 \
         dg_d2_n32_t10_cfl0.4 dg_d2_n64_t1_cfl0.2 dg_d2_n64_t2_cfl0.2 dg_d2_n64_t10_cfl0.2 \
  --output-dir case_euler_vortex_errconserve --max-seconds 1800
```

然后重新聚合：

```bash
PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py \
  --aggregate --output-dir case_euler_vortex_errconserve
```

### 预期结果

- 7 个证书全部生成，replay delta < 1e-9
- C11 状态从 `not_evaluated` 变为 `passed`
- 整包裁决从 `INCOMPLETE` 变为 `PASS`

## 物理结论（不受 C11 影响）

等熵涡在周期方箱中平移的误差守恒已获充分验证：

1. **误差是网格属性，不是时间属性**：所有 DG 家族的累积指数 alpha << 1（最差 0.440），
   证实离散误差由网格/通量对决定，不随输运时间累积。

2. **设计阶保持**：P1/P2 在 t=1 和 t=10 均达到 0.8*(p+1) 的观测阶，
   证明长时误差仍按设计阶收敛。

3. **方向不变性**：三个对流方向（45°/0°/30°）的 alpha 均 <= 0.313，
   误差守恒结论不依赖于输运方向。

4. **一阶格式的局限**：P0 格式在 t=10 时误差达 14.3%（n=64），
   这是耗散抹平涡核的固有属性，非失稳。

## 参考

- Yee, Sandham & Djomehri, "Low-dissipative high-order shock-capturing methods using
  characteristic-based filters", JCP 162 (2000) 100-125.
- TensorDG `benchmarks/case_euler_vortex_errconserve.py` — 11 判据实现
- `case_euler_vortex_errconserve/metrics.json` — 当前聚合结果
- `case_euler_vortex_errconserve/provenance_certificates.json` — 12 个已有证书
