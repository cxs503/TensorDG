# CASE-LAX-DG · 认证构型：哪一档（p, N）把台账 3 % 拿下，代价是多少

> 本文全部数字由 `benchmarks/lax_certified_configuration.py` 从 `docs/benchmarks/results/lax_certified_configuration/metrics.json`（生成于 2026-11-14T21:01:31Z）程序化注入，零手抄；`--check` 在漂移时失败。metrics sha256 `fb2cf719d93722f7e1415fb4be5359175c0cfdbff5fa8e6d41612494e194c214`。

## ① 案例介绍

Lax (1954) / Toro (2009) Test 2 激波管：`[0,1]` 域，γ = 1.4，隔膜 `x0 = 0.5`，左态 (ρ, u, p) = (0.445, 0.698, 3.528)，右态 (0.500, 0.000, 0.571)，快照 t = 0.13。解含左稀疏波、接触间断、右激波，参考值由闭式 Riemann 解给出（`benchmarks/lax_case_pool.py` 的 `exact_reference`，本驱动的误差全对标它）。

台账验收原文：`ref: 精确解｜容差 < 3 %`。已在案的两件事：钉扎腿 `p3.N400` **未达**（rel L2 ρ = 3.11556e-02），而 `p3.N800` 达标（rel L2 ρ = 2.31490e-02）。本文回答第三个问题：**在同等或更小自由度预算下，是否还有更便宜的达标构型？**

## ② 怎么做的

* 网格（跑数前写死，`CANDIDATES`）：p1.N400、p1.N800、p1.N1600、p2.N400、p2.N800、p2.N1000、p3.N400、p3.N800。
* 自由度上限 `DOF_CAP = 3200`，即已知达标构型 `p3.N800` 的 `(p+1)·N`；每个候选腿都满足该上限（只做加法，不抬预算）。
* 度量**不重写**：`measure_leg`、`_weighted_l1/_weighted_l2`、`exact_reference`、`modal_fields` 全部从算例池/分辨率研究模块 import，保证与台账臂是同一把尺子（全场 rel L2，Gauss 节点 Legendre 加权，分母取精确解同范数）。
* 容差**不复制**：来自 `tensordg.cases.lax_1d.lax_parameters()['acceptance_threshold']`。
* 认证规则（纯函数 `certification`）：a configuration is PASS iff max_field rel L2 < tolerance (whole field, every field, no wave band removed, no averaging); the certified configuration is the PASS leg with the smallest (p+1)*N, tie-broken by measured wall clock。
* 复算自检：`p3.N400` 必须逐位复现已提交的算例池工件，`p3.N800` 必须逐位复现分辨率研究工件（判据 rel ≤ 1e-06）。
* 求解器/参数：模态 DG + HLLC 界面通量 + SSP-RK3，CFL = 0.35，float64，求积阶 10，终止时刻 0.13。

## ③ 计算结果

| leg | DOF | steps | wall [s] | rel L2 ρ | rel L2 u | rel L2 p | 最差 | 判据 |
|---|---|---|---|---|---|---|---|---|
| p1.N400 | 800 | 2199 | 29.6 | 4.93486e-02 | 1.78668e-02 | 1.01644e-02 | 4.93486e-02 | FAIL |
| p1.N800 | 1600 | 4390 | 67.7 | 3.77126e-02 | 1.14182e-02 | 7.39018e-03 | 3.77126e-02 | FAIL |
| p1.N1600 | 3200 | 8773 | 169.2 | 2.95401e-02 | 9.89871e-03 | 5.40336e-03 | 2.95401e-02 | PASS |
| p2.N400 | 1200 | 3662 | 50.2 | 3.52587e-02 | 1.57638e-02 | 8.57558e-03 | 3.52587e-02 | FAIL |
| p2.N800 | 2400 | 7341 | 118.4 | 2.57033e-02 | 9.10287e-03 | 5.49864e-03 | 2.57033e-02 | PASS |
| p2.N1000 | 3000 | 9183 | 158.2 | 2.36827e-02 | 1.05645e-02 | 5.75614e-03 | 2.36827e-02 | PASS |
| p3.N400 | 1600 | 5109 | 73.9 | 3.11556e-02 | 2.24805e-02 | 1.30501e-02 | 3.11556e-02 | FAIL |
| p3.N800 | 3200 | 10250 | 176.4 | 2.31490e-02 | 1.38257e-02 | 8.99402e-03 | 2.31490e-02 | PASS |

### 认证结论

* **认证构型 = `p2.N800`**（DOF 2400），最差量 `density` rel L2 = 2.57033e-02，容差裕度 ×0.857（< 1 即达标）。
* 墙钟最省的达标候选（本机实测，仅供参考）：`p2.N800`。
* 网格内最小性：自由度严格小于认证构型的腿全部 FAIL → `True`（这些腿：p1.N400、p1.N800、p2.N400、p3.N400）。

## ④ 结果图

* `docs/benchmarks/results/lax_certified_configuration/figures/lax_certification_dof.png`：whole-field relative L2 (worst field) against degrees of freedom, per polynomial order; dashed line = registered 3 % tolerance; hollow marker = configuration certifies
* `docs/benchmarks/results/lax_certified_configuration/figures/lax_certification_cost.png`：same error measure against measured wall clock; the certified configuration is the cheapest point under the dashed tolerance line
* `docs/benchmarks/results/lax_certified_configuration/figures/lax_certified_profiles.png`：density/velocity/pressure at the certified configuration against the closed-form Riemann solution

## ⑤ 注意点

1. **移动的是「认证半径」不是「验收线」**：A0 钉扎腿仍是 `p3.N400`，它的 FAIL 原样保留；本文只声明「算例在哪个构型上达标、代价多少」。若要把台账 `CASE-LAX-DG` 翻成 done，需要台账持有人按本文证据把生产腿改到认证构型——那是验收口径变更，不是本驱动可以自己做的裁决。
2. **最小性是网格内的**：只声明「网格内自由度更小的腿都 FAIL」，不对未测的 (p, N) 作断言。
3. **wall clock 有环境噪声**：DOF 是无噪声的记账量，认证用 DOF 排序，墙钟只作并列时的 tie-break 与成本参考。
4. **阶数不进 3 %不代表族达设计阶**：间断存在下全场阶 < 1 是 DG 的已知属性（见算例池 C3/R3 的 FAIL），达标靠加密不靠提阶。

## ⑥ 误差判定（是否 < 3 %）

| 关键量 | 基准 | 数值 | 相对误差 | < 3 % ? |
|---|---|---|---|---|
| rel L2 density @ p2.N800 | 精确 Riemann 解 | 2.57033e-02 | 2.570% | 是 |
| rel L2 velocity @ p2.N800 | 精确 Riemann 解 | 9.10287e-03 | 0.910% | 是 |
| rel L2 pressure @ p2.N800 | 精确 Riemann 解 | 5.49864e-03 | 0.550% | 是 |

**结论**：认证构型 `p2.N800` 三个量全部 < 3%；钉扎腿 `p3.N400` 的 FAIL 未被本文改写。
