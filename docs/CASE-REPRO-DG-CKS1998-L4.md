# REPRO-DG-CKS1998 — 把「自证」换成「机器裁判」+ 审计 M2 修复（m17924）

- **锚点**：`REPRO-DG-CKS1998`（Cockburn & Shu, SIAM Review 43(1) 2001 / JCP 141 (1998)）
- **起点**：审计 m15345（`docs/AUDIT_m15345_REPRO-DG-CKS1998.md`）裁 L3，两条阻断/中危缺陷：
  - **M2**：`benchmark_ref_vs_got` 把**越界行**印成 `informative`（Sod N=200 速度 rel-L2 = 4.17 % vs
    自述容差 < 3 %），且 `assert n_fail == 0` 让 FAIL 列**结构上不可能非空**；
  - **L4 阻断**：`tensor_benchmark_cases` 对本锚点 0 行、`tensorbench_results` 0 行 ⇒
    `GET /tensor/bench/l4-verdict/<mid>` 恒答 `CLAIMED_NOT_PROVEN`——判据只活在 Markdown 里。
- **本单**：修 M2（判据/门例解耦，越界不得洗白）+ 补锚点登记 + 走平台取证回路落
  `tensorbench_results`。不改锚点预登记阈值，不改数值内核。

## 1. M2 修复：verdict 只由容差规则决定，门例另列 `gated`

`benchmarks/cks1998_benchmark.py`：

| 旧行为 | 新行为 |
|---|---|
| `verdict` 把「门例选择」和「是否越界」混在一列，越界的非门例行写 `informative` | `verdict ∈ {PASS, FAIL, UNSCORED}` **只由 (ref, got, tolerance) 决定**；门例身份写进独立布尔列 `gated` |
| `assert n_fail == 0`（FAIL 列结构上不可能非空） | `assert gated_fail == 0`（只约束**预登记门例**）；非门例越界照印 `FAIL` |
| 表头 "PASS 43 / FAIL 0 / informative 11" 把一次真实越界藏进 `informative` | 表头 "gated PASS 11/11 · non-gated out-of-tolerance 7 · unscored 4 · rows 54" |
| 退出码看 `FAIL == 0` | 退出码看 `gated_fail == 0`（门例语义，非门例违规不能反噬锚点，但一定看得见） |

`UNSCORED` 只给**纯零参考**（如 x=0.10 处精确速度 = 0，rel 误差无标度可言）——旧版把这些也归进
`informative`，与新语义混在一处。

修复后 quick 档（`artifacts/quick/metrics.json`，仓内已入库）重生成为：
`gated PASS 11/11`，`non-gated out-of-tolerance 7`，`unscored 4`，共 54 行。
旧版 11 个 `informative` 的**逐行归属**（这就是被洗白的内容）：

| 行 | 度量 | ref → got | 容差 | 新判定 | 门例 |
|---|---|---|---|---|---|
| E2_sod_p2_n200 | 速度 rel-L2 | — → 4.173 % | < 3 % | **FAIL** | no（粗网格） |
| E2_sod_p2_n200 | x=0.302 速度 | 0.16310 → 0.17290 | < 3 % 相对 | **FAIL**（6.01 %） | no（台站为补充证据） |
| E2_sod_p2_n200 | x=0.703 密度 | 0.26557 → 0.27570 | < 3 % 相对 | **FAIL**（3.81 %） | no |
| E2_sod_p2_n400 | x=0.301 速度 | 0.15789 → 0.16324 | < 3 % 相对 | **FAIL**（3.39 %） | no |
| E3_tvd | p1/p2 `unlimited` 点值 | 1 → 1.0685 / 1.0852 | ≤ 1 % 超调 | **FAIL** | no（对照组：证明门非平凡） |
| E3_tvd | p1 `tvb_M0`（论文指示器） | 1 → 1.01389 | ≤ 1 % 超调 | **FAIL** | no |
| E2_sod | x=0.102 / 0.902 速度 | 0 → ≤1.8e-12 | < 3 % 相对 | **UNSCORED**（纯零参考） | no |

**台站行不是门例**（这一点旧版没写清）：预登记的 E2 门是**积分**相对 L2（P2/N=400 < 3 %），
台站比较的是「DG 单元平均 vs 精确解点值」，在稀疏波陡段天然有 3–6 % 的垂向差。
新表把它们按同一条 3 % 规则判（越界就是 FAIL），但 `gated=false`，
所以 x=0.301 速度 3.39 % 是**如实可见的局限**，而不是被吞掉或被抬成锚点失败。

回归测试（`tests/test_repro_cks1998.py`）：
- `test_benchmark_ref_vs_got_table_from_committed_artifacts`：断言 `gated_fail == 0`、
  verdict 取值集 ⊆ {PASS, FAIL, UNSCORED}、**被洗白的那一行（N=200 速度 rel-L2）必须是 FAIL 且 gated=false**；
- 新增 `test_ref_vs_got_cannot_launder_an_out_of_tolerance_non_gate_row`：用合成 metrics
  （非门例 10 % > 3 %）证明「越界非门例 → FAIL」这条规则本身，不依赖仓内产物。

## 2. 锚点登记（`tensor_benchmark_cases`，此前 0 行）

审计给的机械化升 L4 路径第一步——登记 3 条判据（实测 id 76/77/78，`repo=TensorDG`）：

| id | 判据 | 容差 | ref 出处 |
|---|---|---|---|
| 76 | A1 光滑对流 L2 观测阶（P1/P2/P3，20→320 网格） | `order >= p+0.5` | 论文定理 p+1 阶（平台编码 ref = p+1，tolerance_rel = 0.5/(p+1)，两者**等价**于 order ≥ p+0.5） |
| 77 | E2 Sod（γ=1.4，(1,0,1)/(0.125,0,0.1)，T=0.2，P2/400，TVB M=1.0） | 相对 L2 < 3 %/分量 | 精确 Riemann 解（`benchmarks.sod_exact`，Toro 采样） |
| 78 | E3 TVD/TVB（方波，80 单元，P1/P2，strict minmod） | 点值超调 ≤ 1 % | 论文「不产生新极值」+ 初值界 [0,1] |

**平台编码的一个诚实说明**：平台 `contract.evaluate_metrics` 只支持「相对误差 ≤ 容差」这一种判据，
而 A1 的门是**不等式**（order ≥ p+0.5）。上面的等价编码把不等式无损写成相对误差形式
（`|order-(p+1)|/(p+1) ≤ 0.5/(p+1)` ⟺ `order ≥ p+0.5`），不是把门放宽：
P1 档 order = 2.0131 → 相对误差 0.65 % ≤ 25 %（= 门线 1.5），余量语义与纸面门一致。

## 3. 平台取证回路（把 got 交给平台复算）

三件套均已在仓内（m17011 建）：workload `validation/repro_cks1998/bench_workload.py`、
规格 `bench_spec.py`、驱动 `tensorbench_ingest.py`。本单真正把它**跑起来**：
集群作业（`bench` 档：A1 三级 × 5 网格 + Sod P2/N=400）的**原始 stdout** →
`--stdout ... --validate-only`（平台同算式预检）→ `--post --job-id ...`（平台按指针复算 ref/got）→
`GET /tensor/bench/l4-verdict/17924`。

实测结果见 §5（数字由命令输出程序化抄录）。

## 4. 局限（honest）

1. **full 档工件仍为 WITHHELD**：`docs/repro_cks1998.md` 第 3 节引用
   `validation/repro_cks1998/artifacts/full/`，而 git 只跟踪 `quick/`（4 CSV + metrics + summary）。
   本工作树内确有 `artifacts/full/`（6 件，未跟踪）且其 `metrics.json` 自洽、`all_pass=true`，
   但**未入库、非本单产出**，故本单不据其下结论，也不改那行声明（改声明属另一单的交付面）。
   记 **M1 未闭合**。
2. **台站 3–6 % 偏差**已在上表如实列出：这是「单元平均 vs 点值」的口径差 + 稀疏波内陡梯度，
   不是求解器 bug；但它在旧表里以 `informative` 出现，读表的人看不出。
3. **2-D 系统算例（JCP 141 主算例）未做**：锚点 `quantitative_targets` 未要求，仍属范围外。
4. 论文 PDF 表值逐格对照仍 **WITHHELD**（仓内无 PDF，未冒充 REAL）。

### 5.1 本单实跑结果（`runs/cks1998_tensorbench/`，程序化抄录）

`bench_workload.py --profile bench` 本机实跑完成，stdout sha256 =
`5ceaa3f8185fc185a6d208b27de8e7417cce55cbbef7579c504a6a97017b2a6b`（4438 B）；
`tensorbench_ingest.py --validate-only` 用平台同算式逐度量复算，**8/8 PASS**：

| 度量 | ref | got | 相对误差 | 容差 | 判定 |
|---|---:|---:|---:|---:|---|
| adv_order_p1（L2 观测阶，理论 p+1=2） | 2 | 2.00424 | 0.212 % | 5 % | PASS |
| adv_order_p2 | 3 | 2.99993 | 0.002 % | 5 % | PASS |
| adv_order_p3 | 4 | 3.99990 | 0.003 % | 5 % | PASS |
| sod_rho_x030（精确 Riemann） | 0.873497 | 0.869379 | 0.472 % | 3 % | PASS |
| sod_rho_x070 | 0.265574 | 0.269618 | 1.523 % | 3 % | PASS |
| sod_vel_x055 | 0.927453 | 0.927491 | 0.004 % | 3 % | PASS |
| sod_mass_integral（解析积分 0.5625） | 0.5625 | 0.5625 | 0.000 % | 0.1 % | PASS |
| sod_energy_integral（解析积分 1.375） | 1.375 | 1.375 | 0.000 % | 0.1 % | PASS |

**误差判定 vs 基准（<3 % 逐项）**：6 个物理量的相对误差全部 < 3 %（最差 1.523 %，
Sod 激波后平台密度），3 个收敛阶的相对误差全部 < 0.25 %；质量/能量解析积分精确到 0.000 %。
E2 的整场相对 L2 = **2.3677 %** < 3 %（工作日志 `workload.err`）。

### 5.2 L4 机器判定：**未闭合（honest not_run → 已被外部队列阻塞）**

`POST /tensor/bench/results` 由平台强制要求「同一 mission 的 **succeeded** 集群作业」绑定
（`store.ingest_result` 反造假绑定，本驱动无 bypass）。本单递交的集群作业 **cj1706 在收尾时
仍为 `queued`（队列前有 287 个作业，等待 9 min 未派发）**，因此：

- `tensorbench_results` 对本 mission 仍是 **0 行**；`l4-verdict/17924` 仍会答 `CLAIMED_NOT_PROVEN`；
- 契约面已全部预检通过（`--validate-only` 8/8 PASS），**只差一次带作业号的 POST**。

后继单一条命令即可闭合（不要重算，直接复用产物）：

```bash
run_cluster ps --job-id cj1706        # 等 succeeded
python validation/repro_cks1998/tensorbench_ingest.py \
    --stdout runs/cks1998_tensorbench/stdout.txt --post --job-id <cj1706 的号> --mission 17924
curl -s -H "Authorization: Bearer $TOKEN" \
    http://127.0.0.1:8010/api/v1/tensor/bench/l4-verdict/17924
```

`runs/` 为本地工作档（不入库），上面的 stdout sha256 是入库复用的锚。

## 5. 复现

```bash
# 1) 回归（M2 语义）
OMP_NUM_THREADS=1 python -m pytest tests/test_repro_cks1998.py -q -k benchmark
# 2) 重生 ref-vs-got 表（quick 档，仓内入库工件）
OMP_NUM_THREADS=1 python -m benchmarks.cks1998_benchmark --profiles quick
# 3) 平台回路（本单实跑）
OMP_NUM_THREADS=1 python validation/repro_cks1998/bench_workload.py --profile bench \
    --out runs/cks1998_tensorbench/doc.json > runs/cks1998_tensorbench/stdout.txt
python validation/repro_cks1998/tensorbench_ingest.py \
    --stdout runs/cks1998_tensorbench/stdout.txt --validate-only
python validation/repro_cks1998/tensorbench_ingest.py \
    --stdout runs/cks1998_tensorbench/stdout.txt --post --job-id <cjN> --mission 17924
curl -s -H "Authorization: Bearer $TOKEN" \
    http://127.0.0.1:8010/api/v1/tensor/bench/l4-verdict/17924
```