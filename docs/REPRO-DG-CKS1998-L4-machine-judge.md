# REPRO-DG-CKS1998 · 把「自述」接进平台机器裁判（TensorBench 取证回路）

> 仓 TensorDG · 分支 `agent/m17011` · 单号 m17011（自找活 · 锚点 REPRO-DG-CKS1998）
> 产物：`validation/repro_cks1998/bench_spec.py`（规格/结果文档唯一真源）、
> 　　　`validation/repro_cks1998/bench_workload.py`（集群 workload，stdout = JSON 文档）、
> 　　　`validation/repro_cks1998/tensorbench_ingest.py`（自检 + `POST /tensor/bench/results`）、
> 　　　`tensorbench/REPRO-DG-CKS1998.spec.json`（平台 `evaluator_command` 约定路径的规格件）
> 　　　`tests/test_cks1998_tensorbench.py`（11 条，全绿）
> 本单**不改**任何求解器内核；只补证据链。

## ① 案例介绍（背景 / 判据 / 参数）

`REPRO-DG-CKS1998 = Cockburn & Shu, JCP 141 (1998) "…conservation laws V: multidimensional
systems" / SIAM Review 43(1) (2001)` 的 1-D RKDG 基准集。round-1/round-2（m15309、m15347 族）
已把四条判据跑绿并落了仓内工件，但**三次独立审计**（`docs/audit_m15292_*.md`、
`docs/AUDIT_m15345_*.md`、`docs/REPRO-DG-CKS1998-round2.md` §6）指向同一处欠账：

> `tensor_benchmark_cases` / `tensorbench_results` 对该锚点 **零行** ⇒
> `GET /tensor/bench/l4-verdict/<mid>` 只能答 `CLAIMED_NOT_PROVEN`——
> 数字再漂亮也是**自述**，不是平台复算过的实证。

平台判据（`tools/maturity_gate/benchmark_source.py`）要求每行满足
`execution_status='success'` ∧ `source='cluster_job'` ∧ `cluster_job_id` 非空 ∧
`|got-ref|/|ref| ≤ tolerance_rel`。本单把 CKS1998 判据翻译成满足该契约的度量表。

| 组 | 度量 | ref（独立来源） | 容差 |
|---|---|---|---|
| A1 | `adv_order_p1/p2/p3` | 理论阶 k+1 = 2/3/4（论文 §2 渐近阶） | 5 % |
| E2 | `sod_rho_x030`、`sod_rho_x070`、`sod_vel_x055` | 精确 Riemann 解（`benchmarks.sod_exact`，Toro 求解器，**独立于 DG**）在固定台站的单元均值 | 3 %（锚点门） |
| E2 | `sod_mass_integral`、`sod_energy_integral` | 精确守恒积分 0.5625 / 1.375 | 1e-3 |

**不进入规格的两条（诚实声明）**：`A2`（限幅方波 L1 阶 ≥ 1/2）与 `E3`（超调 ≤ 1 %）
是**不等式**判据，没有可比的非零 ref；硬编码成 `|got-ref|/|ref|` 会伪造度量
（got=0 % vs ref=1 % ⇒ 100 % 误差），故仍留在仓内门 `tests/test_repro_cks1998.py`。

## ② 怎么做的

1. **规格即真源**：`bench_spec.METRICS` 一处定义 → `build_spec()` 生成
   `tensorbench-eval/1` 规格（键集合与 `contract.parse_spec` 白名单逐字对齐）、
   `build_result()` 生成 `benchmark-result.schema.json` v1.0.0 结果文档。
2. **测量与判据分离**：`bench_workload.py --profile bench` 只负责测量（A1 三级×5 网格 +
   E2 Sod P2/N400），把 got 与 ref 一起打成**一份 JSON 打印到 stdout**，日志走 stderr；
   ref 一律由理论阶/精确 Riemann 解给出，脚本里不出现任何硬编码的"答案"。
3. **仓内等价校验器**：`bench_spec.validate_documents()` 复刻平台
   `parse_spec`/`validate_result`/`evaluate_metrics` 的算式，可在无网无库环境全量自检；
   平台侧收到 `POST /tensor/bench/results` 后会**再从原始 stdout 字节独立复算**。
4. **反造假绑定不可绕**：`tensorbench_ingest.py` 不提供 bypass 开关——`--post` 必须给
   `--job-id`，平台校验 `cluster_jobs` 该作业存在、终态 `succeeded`、mission 一致。

## ③ 计算结果（程序化打印，零手抄）

`--profile smoke`（秒级，只验证契约通路，**不用于判精度**；`OMP_NUM_THREADS=1`）：

| metric | ref | got | rel_err | tol | 判定 |
|---|---:|---:|---:|---:|---|
| adv_order_p1 | 2 | 2.06221 | 3.111 % | 5 % | PASS |
| adv_order_p2 | 3 | 2.99864 | 0.045 % | 5 % | PASS |
| adv_order_p3 | 4 | 3.98391 | 0.402 % | 5 % | PASS |
| sod_rho_x030 | 0.87103 | 0.84672 | 2.791 % | 3 % | PASS |
| sod_rho_x070 | 0.265574 | 0.298661 | **12.46 %** | 3 % | **FAIL（N=48 粗网格，如实记录）** |
| sod_vel_x055 | 0.927453 | 0.92602 | 0.155 % | 3 % | PASS |
| sod_mass_integral | 0.5625 | 0.5625 | 1.3e-8 | 0.1 % | PASS |
| sod_energy_integral | 1.375 | 1.375 | 9e-9 | 0.1 % | PASS |

守恒误差 ~1e-8 与 float32 舍入一致（求解器默认 float32，非 float64）——这条正是
"workload 真在解 PDE、不是回填数字"的判据。`sod_rho_x070` 在 N=48 上 12.5 % 失败是
分辨率的诚实后果；`--profile bench`（N=400）才是判定档，其数值由集群作业 cj1393
的 stdout 给出（见 §5 状态栏）。

## ④ 结果图

本文件无内嵌图：判定档数值仍在集群作业 `cj1393`（排队中，递交时 `queued_ahead=537`），
图必须由**实测观测量**生成，不做无数据的示意图；round-1/round-2 的收敛图已在
`validation/repro_cks1998/artifacts/{quick,full}/` 与平台报告 14372 内。

## ⑤ 注意点 / 局限

- **L4 仍未开证（honest FAIL）**：`tensor_benchmark_cases` 的锚点行需平台侧数据注入，
  本单只交付了规格件与取证驱动；`tensorbench_results` 的行必须等 `cj1393` 成功后排入，
  未成功前平台仍会答 `CLAIMED_NOT_PROVEN`。
- 平台 `store.ingest_result` 对 `cluster_jobs.mid` 为 NULL 的作业不做 mission 比对
  （代码：`if job.get("mid") is not None and ...`），故用 `run_cluster` 递交的作业
  也可入账；本单不利用此宽松点做任何"自选裁判"，只在文档中标明来源作业号。
- 观测阶对 ref=k+1 的 5 % 容差在粗档（smoke 的 8/16 网格）会因拟合样本太少而偏大
  （p1 3.1 %），bench 档用 5 级网格 20…320。

## ⑥ 误差判定（显式回答"是否 <3 %"）

| 关键量 | 基准 | 实测（判定档） | <3 %？ |
|---|---|---|---|
| A1 P1/P2/P3 观测阶 | 2 / 3 / 4 | 待 cj1393；smoke 档 2.062/2.999/3.984 | 相对理论阶偏差 smoke 3.1 % / 0.05 % / 0.40 %（P1 3.1 %>3 %，如实记） |
| Sod 台站 ρ(x=0.30) | 精确解 | 待 cj1393；smoke 2.79 % | smoke: 是 |
| Sod 台站 ρ(x=0.70) | 精确解 | 待 cj1393；smoke 12.46 % | smoke: **否**（粗网格） |
| Sod 台站 u(x=0.55) | 精确解 | 待 cj1393；smoke 0.155 % | 是 |
| 质量/能量积分 | 0.5625 / 1.375 | 1e-8 级 | 是（严 5 个量级） |

round-1/round-2 已发布的可比数字（full 档、P2/N400）：Sod 相对 L2 密度 1.11 %、
速度 2.37 %、压力 0.59 %（全部 <3 %），见 `validation/repro_cks1998/artifacts/full/summary.md`。

## 复现命令

```bash
OMP_NUM_THREADS=1 python validation/repro_cks1998/bench_workload.py --profile smoke
OMP_NUM_THREADS=1 python validation/repro_cks1998/tensorbench_ingest.py \
    --stdout runs/cks1998_tensorbench/stdout.txt --validate-only
# 判定档（集群）：bench_workload.py --profile bench → 拿到 stdout 后
python validation/repro_cks1998/tensorbench_ingest.py \
    --stdout runs/cks1998_tensorbench/stdout.txt --post --job-id <cjN> --mission 17011
OMP_NUM_THREADS=1 python -m pytest tests/test_cks1998_tensorbench.py -q
```