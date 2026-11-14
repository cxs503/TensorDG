# 已解决：周期参考重跑（m20824）

本文件原为待办标记（旧 `metrics.json` 用自由空间参考）。**m20824 已完成重跑**，
`metrics.json` / `summary.md` / `error_conservation.csv` / `fig_*.png` 全部由
`benchmarks/case_euler_vortex_errconserve.py --aggregate --figures` 从
`results/*.json`（周期参考，`PERIOD`）程序化重生成，可引用。

## 重跑覆盖（9/12 检查点，诚实范围）

| 族 | t | 步数 | rho % | u % | p % | 最大场 % | 熵漂移 |
|---|---|---|---|---|---|---|---|
| dg_d2_n32 | 1 | 469 | 0.0050 | 0.0094 | 0.0070 | 0.0094 | 2.34e-08 |
| dg_d2_n32 | 2 | 937 | 0.0051 | 0.0097 | 0.0071 | 0.0097 | 5.25e-08 |
| dg_d2_n32 | 5 | 2342 | 0.0054 | 0.0111 | 0.0075 | 0.0111 | 1.42e-07 |
| p0_n128 | 1 | 183 | 1.2797 | 2.8022 | 2.0258 | 2.8169 | 3.51e-03 |
| p0_n128 | 10 | 1631 | 5.4216 | 12.0836 | 7.9004 | 12.0846 | 9.73e-03 |
| p0_n256 | 1 | 370 | 0.6933 | 1.5043 | 1.1039 | 1.5186 | 2.01e-03 |
| p0_n64 | 1 | 90 | 2.2038 | 4.9191 | 3.4388 | 4.9260 | 5.52e-03 |
| p0_n64 | 5 | 408 | 5.7257 | 12.0235 | 8.4758 | 12.0235 | 9.56e-03 |
| p0_n64 | 10 | 778 | 6.0187 | 14.3043 | 8.3587 | 14.3047 | 1.04e-02 |

**未完成 3 个检查点**：`dg_d2_n32_t10`、`p0_n128_t2`、`p0_n128_t5` —— 集群作业
`cj2969`（6 分片）17 min 只回 2 个检查点、`cj3010`（10 分片）另回 2 个后被拥塞拖住，
本单按预算收口；它们在 `LADDER` 中保留，重跑命令见下，位置一旦补齐 C2/C1x 会自动纳入。
t=1 的四个族、DG 设计族 t=1/2/5、P0 n=64 族 t=1/5/10 均已落地，所以 C1/C2/C3/C4/C5
**全部可判定**（无 not_evaluated）。

## 关键结论（参考缺陷修复后的读数）

* **DG-P2/32² 的"误差放大"消失了**：t=1/2/5 的最大场误差 0.0094% / 0.0097% / 0.0111%
  —— 累积指数 alpha = 0.106（修复前同一族 alpha≈3.3–4.4），
  e(t=5)/e(t=1) = 1.180。这就是锚点「等熵涡**移动误差守恒**」
  的量测含义：高阶族的误差是网格属性，不是时间属性。
* **P0 一阶族的累积依旧存在且被如实保留**：n=64 的 4.926% → 12.024% → 14.305%
  （耗散抹平涡核，`density_min_relative_error` 同步上升），n=128 亦有 2.817% → 12.085%。
  C1x 长时最差 14.30%（p0_n64_t10_cfl0.4）原样保留。
* **C1 快照容差 PASS**：t=1 四族最差 4.9260%（p0_n64_t1_cfl0.4）< 6%。
* **守恒/熵/收敛阶**：C3 最坏 1.1e-13（p0_n128_t10_cfl0.4:energy）、
  C4 1.42e-07、C5 0.849 —— 全部 PASS。

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
