# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — L4 机器门闭环（集群作业 cj3111）
## ① 案例
Shu–Osher (1988) 激波-熵波相互作用，1D Euler，M=3，γ=1.4，x∈[-5,5]，t=1.8。本单不是复述物理，而是补上此前两轮审计反复挂账的一件制度性缺口：锚点已在 `tensor_benchmark_cases`(id 79) 登记，但 `tensorbench_results` **零行**，平台 L4 判据 `tools/maturity_gate/benchmark_source.py` 因而恒返 `CLAIMED_NOT_PROVEN`。
## ② 做法
新增 `validation/xval_dg_round2/cluster_regen.py`：在集群 n330 上**从零重解整条阶梯**（3 个 FV-WENO5 参考层 N=800/1600/3200 + 11 条 DG 腿 P1/P2/P3×N=200/400/800 + CFL=0.1/0.2 两条），不复用任何已提交 npz；随后 `benchmarks.xval_dg_round2_certified.cross_validate` 对这批新剖面做 D1–D4 判据，stdout 只输出一份平台 workload 文档。新增 `validation/xval_dg_round2/close_l4.py` 把该作业 stdout 与同批剖面提交 `POST /tensor/bench/results`，绑定作业号（平台反造假绑定：作业须 succeeded 且 mid 与 mission 一致）。
作业 cj3111：8 workers，14 个配置墙钟 21.38–442.24 s，作业总时 ~10 min，rc=0。
## ③ 结果
| 量 | 值 | 判据 |
|---|---|---|
| D1 参考解自身不确定带 U | 0.082797 % | U/gate=0.0166 ≤ 0.3333 |
| D2 最差 DG 腿 rel L1 | 2.455701 % (DG P1 N=200 CFL=0.4) | < 3 % |
| D3 闭式锚 plateau_rho | err=1.225e-10 % | < 1 % |
| D3 闭式锚 shock_position | err=5.758e-01 % | < 1 % |
| D3 闭式锚 mass_integral | err=8.350e-07 % | < 1 % |
| D3 闭式锚 preshock_sine_linf_percent | err=8.637e-04 % | < 1 % |
| D4 | WITHHELD_ORDER_TOO_LOW | 拒自证 |

平台复算行（`tensorbench_results`）：

| metric | ref | got | rel_error | tol | passed |
|---|---|---|---|---|---|
| plateau_rho | 3.857143 | 3.857143 | 1.225e-12 | 0.01 | True |
| shock_position | 2.389366587 | 2.403125 | 5.758e-03 | 0.01 | True |
| mass_integral | 31.08915219 | 31.08915245 | 8.350e-09 | 0.01 | True |

`GET /tensor/bench/l4-verdict/20997` → **VERIFIED**（rows=3，failed_checks=[]）。
## ④ 图
图 1（下图）为 cj3111 从零重解的 11 条 DG 腿 rel L1 与 3 %/5 % 双门对照；与已提交参考审计（最差 2.4557 %，同为 `DG P1 N=200 CFL=0.4`）逐位一致 → 参考解与阶梯均可由代码重生。
## ⑤ 注意点/局限
1. 平台复算的是文档内 ref/got 指针与误差，不是独立第三方数据；独立性由闭式锚（Hugoniot 平台密度、激波位置、质量预算、未扰正弦）+ 从零重解共同提供。
2. 作业 stdout 经调度器包装后只保留 `output_tail`，本单的 `workload_stdout` 因此由**作业自身产出的 `cluster_artifacts/cluster_audit.json`** 重排生成（sha256 已记入 evidence），不是手工改数；这一点如实登记。
3. `shock_position` 误差 0.576 % 最大（网格分辨激波位置的固有量化，N=3200 时 Δx≈0.003 对应 ~0.14 %，叠加激波抹平）。
4. 未做多算例重复（单算例 Shu–Osher + 已发布的 Lax 第二算例各自独立）。
## ⑥ 误差判定（是否 < 3 %）
| 关键量 | 误差 | <3 %? |
|---|---|---|
| 平台复算 plateau_rho | 1.225e-10 % | 是 |
| 平台复算 shock_position | 0.575818 % | 是 |
| 平台复算 mass_integral | 8.350e-07 % | 是 |
| 最差 DG 腿 rel L1 | 2.455701 % | 是（对 3 % 门余量 18.1 %）|

L4 机器门：`CLAIMED_NOT_PROVEN` → **VERIFIED**（此前多单挂账的阻断项 B2 关闭）。
