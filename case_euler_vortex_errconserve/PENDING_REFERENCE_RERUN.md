# 已解决：周期参考重跑 + 全图生成（m21091 收口）

本文件原为待办标记（旧 `metrics.json` 用自由空间参考）。**m20824 已完成重跑**，
`metrics.json` / `summary.md` / `error_conservation.csv` / `fig_*.png` 全部由
`benchmarks/case_euler_vortex_errconserve.py --aggregate --figures` 从
`results/*.json`（周期参考，`PERIOD`）程序化重生成，可引用。

## 全量检查点覆盖（12/12，m21091 补齐）

| 族 | t | 步数 | rho % | u % | p % | 最大场 % | 熵漂移 |
|---|---|---|---|---|---|---|---|
| dg_d2_n32 | 1 | 469 | 0.0050 | 0.0094 | 0.0070 | 0.0094 | 2.34e-08 |
| dg_d2_n32 | 2 | 937 | 0.0051 | 0.0097 | 0.0071 | 0.0097 | 5.25e-08 |
| dg_d2_n32 | 5 | 2342 | 0.0054 | 0.0111 | 0.0075 | 0.0111 | 1.42e-07 |
| dg_d2_n32 | 10 | 4684 | 0.0061 | 0.0152 | 0.0085 | 0.0152 | 2.98e-07 |
| p0_n128 | 1 | 183 | 1.2797 | 2.8022 | 2.0258 | 2.8169 | 3.51e-03 |
| p0_n128 | 2 | 357 | 2.5325 | 4.8617 | 3.9561 | 4.8755 | 5.50e-03 |
| p0_n128 | 5 | 854 | 4.5337 | 8.8749 | 7.0069 | 8.8764 | 8.19e-03 |
| p0_n128 | 10 | 1631 | 5.4216 | 12.0836 | 7.9004 | 12.0846 | 9.73e-03 |
| p0_n256 | 1 | 370 | 0.6933 | 1.5043 | 1.1039 | 1.5186 | 2.01e-03 |
| p0_n64 | 1 | 90 | 2.2038 | 4.9191 | 3.4388 | 4.9260 | 5.52e-03 |
| p0_n64 | 5 | 408 | 5.7257 | 12.0235 | 8.4758 | 12.0235 | 9.56e-03 |
| p0_n64 | 10 | 778 | 6.0187 | 14.3043 | 8.3587 | 14.3047 | 1.04e-02 |

**全部 12 个检查点已落地**，C6 门禁 `missing = []`。

## 全量图件（4/4，m21091 补齐）

| 图件 | 内容 | 状态 |
|---|---|---|
| `fig_error_vs_time.png` | 各族最大场相对 L2 随时间的双对数曲线 + 6% 容差线 | ✓ |
| `fig_invariants_vs_time.png` | C3 质量/能量漂移 + C4 熵泛函漂移 + 核密度误差 | ✓ |
| `fig_amplification.png` | A(t) = e(t)/e(0) 误差放大因子（投影基线） | ✓ |
| `fig_error_field_t1.png` | DG-P2/32² t=1 密度/压力误差场 + y 中线剖面 | ✓ |

## 关键结论（参考缺陷修复后的读数）

* **DG-P2/32² 的"误差放大"消失了**：t=1/2/5/10 的最大场误差 0.0094% / 0.0097% / 0.0111% / 0.0152%
  —— 累积指数 alpha = 0.199（修复前同一族 alpha≈3.3–4.4），
  e(t=10)/e(t=1) = 1.608。这就是锚点「等熵涡**移动误差守恒**」
  的量测含义：高阶族的误差是网格属性，不是时间属性。
* **P0 一阶族的累积依旧存在且被如实保留**：n=64 的 4.926% → 12.024% → 14.305%
  （耗散抹平涡核，`density_min_relative_error` 同步上升），n=128 亦有 2.817% → 12.085%。
  C1x 长时最差 14.30%（p0_n64_t10_cfl0.4）原样保留。
* **C1 快照容差 PASS**：t=1 四族最差 4.9260%（p0_n64_t1_cfl0.4）< 6%。
* **守恒/熵/收敛阶**：C3 最坏 1.95e-13（dg_d2_n32_t10_cfl0.2:energy）、
  C4 2.98e-07、C5 0.849 —— 全部 PASS。
* **误差放大因子**：DG 族 A(10) = 2.761（量级 1 ⇒ 误差守恒），P0 族 A 达 10³ 量级
  （一阶格式的时间累积，机制已知、非失稳）。

## 顺带推翻的机制假设（M1）

`benchmarks/case_euler_vortex_err_mechanism.py` 的 M1 把旧读数里「u 在 t=2 跳 19.5 倍、rho/p 平坦」
读成无散度 alias 模，并计划 Helmholtz 分解。参考缺陷扫掠（`reference_defect_percent`）显示该签名
**正是非周期参考的签名**：t=2 处 u 缺陷 0.2632% 而 rho 缺陷 0.0004%（比 658 倍），且 u 缺陷本身
就覆盖了整个 0.2467% 的"跳变"。故 M1 的证据不具判别力，机制结论不得建立在修复前读数上；
该判据已固化为回归测试 `test_reference_defect_mimics_the_velocity_only_mechanism_signature`。
errmechanism 工件（`case_euler_vortex_errmechanism`，仅 base_t1）保持原样、不再推进。

## 复现

    PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --shard i/n --output-dir case_euler_vortex_errconserve
    PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate --figures --output-dir case_euler_vortex_errconserve
    PYTHONPATH=. python3 benchmarks/vortex_err_projection_baseline.py --results-dir case_euler_vortex_errconserve/results --json case_euler_vortex_errconserve/projection_baseline.json
    PYTHONPATH=. python3 benchmarks/vortex_err_field_figure.py --final-time 1 --degree 2 --cells 32 --out-dir case_euler_vortex_errconserve
