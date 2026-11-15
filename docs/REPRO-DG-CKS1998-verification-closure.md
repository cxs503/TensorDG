# REPRO-DG-CKS1998 · 验证欠账闭合（锚点 L4 已 VERIFIED，回归门防回退）

> 锚点：`REPRO-DG-CKS1998`（论文复现·The Runge-Kutta discontinuous Galerkin method for
> conservation laws V: multidimensional systems，Cockburn & Shu 1998，JCP 141:199-224）
> 分支：`agent/m21502`　单号：m21502（自找活·generalist 席）
> 性质：**验证欠账闭合**——把已合并的 CKS1998 复现能力对平台机器裁判（L4）的
> 实证状态钉死成回归门，防止「自述非实测」欠账复发。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Cockburn & Shu (1998) *The Runge-Kutta discontinuous Galerkin method for conservation laws
V: multidimensional systems*（JCP 141:199-224 / SIAM Review 43(1) 2001）是 RKDG 方法的多维
奠基论文。其 §5 给出两类机器可复现基准：**一维**（A1 光滑线性对流、A2 间断 L1 收敛、
E2 Sod 激波管、E3 TVD 无新极值）与**二维**（sm2d 光滑熵波，ρ=1+0.2sin(π(x+y−2t))，
u=v=1，p=1，γ=1.4，周期盒 [0,2]²）。

本锚点的判据（`feature_ledger.acceptance`）为「quantitative_targets 逐项容差」——
即每条度量必须落在预登记容差内，且由**平台从集群作业原始 stdout 复算**（非自述）。

## ② 怎么做的（方法 / 流程 / 工具链）

本单**不改任何求解器内核**，只补证据链的最后一环——把「验证已闭合」写成可回归的断言：

1. **核对平台实证状态**：`tensor_benchmark_cases` 已登记 3 行（A1 id76 / E2 id77 / E3 id78）；
   `tensorbench_results` 已落 **21 行**（m19918 的 1-D 8 行 + m20922 的 2-D 多维 13 行），
   全部 `execution_status='success'`、`source='cluster_job'`、`passed=true`。
2. **L4 裁定**：`GET /tensor/bench/l4-verdict/20922` → `status=VERIFIED`、`rows=13`、
   `failed_checks=[]`（闭合前 m15309/m15345 恒答 `CLAIMED_NOT_PROVEN`/rows=0）。
3. **新增回归门** `tests/test_cks1998_anchor_verified.py`（4 条）：把「闭合」写成断言——
   回执钉死到 m20922/cj3053 的原始 stdout 字节；闭合前 `CLAIMED_NOT_PROVEN`/rows=0、
   闭合后 `VERIFIED`/rows=13；平台度量集覆盖 A1/E2/sm2d 全 13 条；最差相对误差 < 3%。
   任一环节回退（行被清空 / 容差被放宽 / 度量缺失）即红。

## ③ 计算结果（程序化注入，数字全部来自平台回执）

| 组 | 度量 | ref（平台复算） | got（DG 实测） | 相对误差 | 容差 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| A1 | adv_order_p1 | 2 | 2.00424 | 0.2119 % | 5 % | PASS |
| A1 | adv_order_p2 | 3 | 2.99993 | 0.0024 % | 5 % | PASS |
| A1 | adv_order_p3 | 4 | 3.99989 | 0.0029 % | 5 % | PASS |
| E2 | sod_rho_x030 | 0.873497 | 0.869379 | 0.4715 % | 3 % | PASS |
| E2 | sod_rho_x070 | 0.265574 | 0.269618 | 1.5229 % | 3 % | PASS |
| E2 | sod_vel_x055 | 0.927453 | 0.927491 | 0.0041 % | 3 % | PASS |
| E2 | sod_mass_integral | 0.5625 | 0.5625 | 8.3e-13 | 0.1 % | PASS |
| E2 | sod_energy_integral | 1.375 | 1.375 | 1.1e-12 | 0.1 % | PASS |
| sm2d | sm2d_order_p1 | 2 | 1.98601 | 0.6997 % | 3 % | PASS |
| sm2d | sm2d_order_p2 | 3 | 3.01956 | 0.6521 % | 3 % | PASS |
| sm2d | sm2d_order_p3 | 4 | 4.00363 | 0.0908 % | 3 % | PASS |
| sm2d | sm2d_mass_integral | 4 | 4.0 | 3.2e-14 | 0.1 % | PASS |
| sm2d | sm2d_energy_integral | 14 | 14.0 | 3.4e-14 | 0.1 % | PASS |

（数字来自 `validation/repro_cks1998/artifacts/l4_2d/` 与平台 `tensorbench_results`，
程序化注入，零手抄。）

## ④ 结果图

本单为**验证欠账闭合**（纯证据链/回归门），无新场量云图——判定档数值与收敛曲线已在
`validation/repro_cks1998/artifacts/{quick,full,l4,l4_2d}/` 与平台报告 14372/19918/20922
内（A1 收敛阶、E2 Sod 剖面、sm2d 二维守恒）。本单的「图」是 L4 裁定状态机
（`CLAIMED_NOT_PROVEN` → `VERIFIED`）的断言，非数值云图，故不内嵌 png。

## ⑤ 注意点 / 局限（honest）

1. **E3/A2 不等式判据仍未进平台编码**：`bench_spec.NOT_ENCODED` 明确声明——A2（L1 阶 ≥ 1/2）
   与 E3（超调 ≤ 1 %）是**不等式**判据，硬编码成 `|got-ref|/|ref|` 会伪造度量
   （got=0 % vs ref=1 % ⇒ 100 % 误差），故仍留在仓内门 `tests/test_repro_cks1998.py` +
   报告表。`tensor_benchmark_cases` 的 E3 行（id78）是登记态，无对应 `tensorbench_results`
   行——这是**有意的**（不等式不适用相对误差编码），不是欠账。
2. **`sod_rho_x070`（激波后平台）1.52 % 是 13 项里最差项**：来源是 DG 单元均值 vs 点值
   口径 + 平台侧陡梯度，不是求解器缺陷；它 < 3 % 但余量仅 1.48 个百分点。
3. **L4 裁定按 mission 键**：`l4-verdict/15309`/`15345`（旧审计单）仍答 `CLAIMED_NOT_PROVEN`
   ——因为那些 mission 无行；本锚点的实证行在 m19918/m20922。这是平台「有行=实证」语义
   的正确表现，不是回退。
4. **本单未改任何容差**：`bench_spec.METRICS` 容差表在任何结果被看到之前已冻结
   （`tests/test_cks1998_tensorbench.py` 的 `FROZEN_TOLERANCES` 钉死），本单只加断言。

## ⑥ 误差判定（显式回答「是否 < 3 %」）

| 关键量 | 基准 | 实测（判定档） | < 3 %？ |
|---|---|---|---|
| A1 P1/P2/P3 观测阶 | 2 / 3 / 4 | 2.00424 / 2.99993 / 3.99989 | **是**（最差 0.2119 %） |
| E2 Sod 台站（ρ/u） | 精确 Riemann 解 | 最差 1.5229 % | **是**（余量 1.48 个百分点） |
| E2 质量/能量守恒 | 0.5625 / 1.375 | 1e-12 级 | **是**（严 5 个量级） |
| sm2d 二维观测阶 | 2 / 3 / 4 | 1.98601 / 3.01956 / 4.00363 | **是**（最差 0.6997 %） |
| sm2d 二维守恒 | 4 / 14 | 3e-14 级 | **是**（机器精度） |

**逐项结论：13/13 度量 < 3 %（最差 1.5229 %），L4 裁定 `VERIFIED`。**

## 复现命令

```bash
# 回归门（轻量，~1 s，只读回执 + 查平台）
python3 -m pytest tests/test_cks1998_anchor_verified.py -q
# 平台 L4 裁定（完整 13 行）
GET /tensor/bench/l4-verdict/20922  →  {"status":"VERIFIED","rows":13,"failed_checks":[]}
```
