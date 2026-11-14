# REPRO-DG-CKS1998 — L4 机器判定闭合（mission 19918）

- **锚点**：`REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199-224 / SIAM Review 43(1) 2001）
- **上一单（m17924）留下的唯一阻断欠账**：判据已在 Markdown 与仓内 JSON 里，但
  `tensorbench_results` 对 TensorDG **零行** ⇒ `GET /tensor/bench/l4-verdict/<mid>`
  恒答 `CLAIMED_NOT_PROVEN`（「自证，非实测」）。原因不是算法，而是**平台反造假绑定**：
  `POST /tensor/bench/results` 要求同一 mission 的 **succeeded** 集群作业号，
  上一单递交的 cj1706 在收尾时仍在排队。
- **本单**：把这条回路**跑通并落盘**——集群作业 → 原始 stdout → 平台按指针复算 ref/got →
  8 条实证行入库 → L4 裁定 `VERIFIED`。**不改数值内核、不改任何容差**。

## 1. 怎么做（工具链，全部仓内既有件）

| 步骤 | 命令 | 产出 |
|---|---|---|
| 1 集群实跑 | `run_compute submit`：`bench_workload.py --profile bench`（stage_dir=本工作树） | 作业 **cj2385**（node n330，rc=0），原始 stdout 4399 B，sha256 `b2892d833603428436af97beeeae00764289639b03dda9e49c156b2dae536f8a` |
| 2 契约预检 + 递交 | `tensorbench_ingest.py --stdout ... --post --job-id 2385 --mission 19918` | 8 条 `tensorbench_results` 行（`tbr-3d5d2ceab…` 等） |
| 3 L4 裁定 | `GET /tensor/bench/l4-verdict/19918` | `status = VERIFIED`，`l4_benchmark_pass = true`，`failed_checks = []` |

第一次递交（cj2371/cj2372，ref=main）在节点上失败：`Failed to connect to github.com port 443`
——这是**调度侧拿不到 GitHub**，不是代码问题；改用 `stage_dir`（把本地工作树 tar 到节点）后一次通过。
**这条踩坑对后续所有 TensorDG 取证作业都成立：集群节点不可达 GitHub，必须用 stage_dir。**

## 2. 平台复算结果（数字由 `receipt.json` 程序化注入，零手抄）

| 度量 | ref（平台复算） | got（DG 实测） | 相对误差 | 容差 | 判定 | 平台 result_id |
|---|---:|---:|---:|---:|---|---|
| adv_order_p1 | 2 | 2.00424 | 0.2119 % | 5.00 % | PASS | `tbr-3d5d2ceab…` |
| adv_order_p2 | 3 | 2.99993 | 0.0024 % | 5.00 % | PASS | `tbr-04bf3e0b4…` |
| adv_order_p3 | 4 | 3.99989 | 0.0029 % | 5.00 % | PASS | `tbr-ae70c3c08…` |
| sod_rho_x030 | 0.873497 | 0.869379 | 0.4715 % | 3.00 % | PASS | `tbr-12957cee6…` |
| sod_rho_x070 | 0.265574 | 0.269618 | 1.5229 % | 3.00 % | PASS | `tbr-2e7037f45…` |
| sod_vel_x055 | 0.927453 | 0.927491 | 0.0041 % | 3.00 % | PASS | `tbr-a84e21062…` |
| sod_mass_integral | 0.5625 | 0.5625 | 0.0000 % | 0.10 % | PASS | `tbr-43b157910…` |
| sod_energy_integral | 1.375 | 1.375 | 0.0000 % | 0.10 % | PASS | `tbr-700a28154…` |

`run_id = REPRO-DG-CKS1998-bench-p2_n400`；`implementation.revision = unknown`；
作业号 `cj2385`；判定 `VERIFIED`（8/8）。

## 3. 误差判定（显式回答「是否 < 3 %」）

- 3 个收敛阶（A1，理论 k+1 = 2/3/4）：相对误差 0.2119 %（最差），**全部 < 3 %**。
- 3 个 Sod 台站（E2，精确 Riemann 解为 ref）：最差 **sod_rho_x070 = 1.5229 %**，**< 3 %**（余量
  1.48 个百分点）。
- 2 个解析守恒积分：1.07e-10 %（机器精度），**< 3 %**。
- **逐项结论：8/8 度量 < 3 %（最差 1.5229 %）。** 门线本身是**预登记**的
  （`bench_spec.METRICS` 容差表在任何结果被看到之前写定），本单未调任何容差。

## 4. 证据的可复核性（本单新增的机器裁判）

`validation/repro_cks1998/artifacts/l4/` 入库了**字节级**回执：`stdout.txt`（集群作业原始 stdout）、
`spec.json`/`result.json`（平台规格与结果文档）、`platform_response.json`（平台返回的 8 行）、
`l4_verdict_before.json`（闭合前 `CLAIMED_NOT_PROVEN` / rows=0）、`l4_verdict.json`（闭合后 `VERIFIED`）、
`receipt.json`（上述内容的绑定摘要）。

新增回归门 `tests/test_cks1998_platform_receipt.py`（5 个测试）**不采信回执里写的数字**：

1. `stdout_sha256` 与入库 stdout 字节一致，且 stdout 是纯 JSON（平台 `json_stdout` 提取器的前提）；
2. `spec.evaluation.metrics` 与 `bench_spec.METRICS` 逐项同名同容差（防事后放宽）；
3. 用 `bench_spec.evaluate_metrics` **从 stdout 本地重算** ref/got/误差/判定，逐项与回执行比对
   （洗白型的回执在这里失败）；
4. 8 行全部 `passed`、全部绑定 `cj2385`、`rel_error` 非空（ref/got/误差三件齐全）；
5. 闭合后 `l4_benchmark_pass is True` 且 `failed_checks == []`，并断言**闭合前**是
   `CLAIMED_NOT_PROVEN`/rows=0——把「增量」写成断言，而不是散文。

## 5. 局限（honest）

1. **只覆盖 1-D 判据**：A1/E2/E3 都是 1-D 标量或 1-D Euler；JCP 141 的招牌算例（Double Mach
   reflection、2-D Riemann）**仍未做**，也不在本锚点 `quantitative_targets` 内。平台行里 E3 的
   不等式判据（超调 ≤ 1 %）仍未编码（见 `bench_spec.NOT_ENCODED`，理由：不等式与相对误差编码不等价）。
2. **`sod_rho_x070`（激波后平台）1.52 % 是 8 项里最差项**，来源是 DG 单元均值 vs 点值口径 + 平台
   内侧陡梯度，不是求解器缺陷；它 < 3 % 但余量仅 1.48 个百分点。
3. **`docs/repro_cks1998.md` §3 引用的 full 档工件**另行重生（作业 cj2386），与本条的 L4 判定**无关**；
   若该作业未回，M1 仍按未闭合记账，不以本地未跟踪文件充当证据。
4. 平台 `contract.evaluate_metrics` 只支持「相对误差 ≤ 容差」，A1 的不等式门（order ≥ p+0.5）用
   `ref=p+1, tol=0.5/(p+1)` 无损等价编码——语义等价性由 `spec` 注释与测试 2 共同钉住。

## 6. 复现（一条命令一段）

```bash
# 集群（注意：节点不可达 GitHub，必须 stage_dir）
run_compute submit repo=TensorDG stage_dir=<worktree> \
  cmd='OMP_NUM_THREADS=1 python validation/repro_cks1998/bench_workload.py --profile bench \
       --out runs/cks1998_tensorbench/doc.json > runs/cks1998_tensorbench/stdout.txt'
python validation/repro_cks1998/tensorbench_ingest.py \
  --stdout validation/repro_cks1998/artifacts/l4/stdout.txt --post --job-id <cjN> --mission <mid>
python -m pytest tests/test_cks1998_platform_receipt.py -q
```
