# CASE · REPRO-DG-CKS1998 — TVB troubled-cell indicator symmetry defect (mission m21413)

**锚点**：`REPRO-DG-CKS1998`（feature_ledger status=open；验收原文
「论文复现|CKS1998｜ref: SIAM Review / JCP 系列｜判据: quantitative_targets 逐项容差」，
其中第三条判据 = **TVD 界无振荡，无超调 > 1 %**）
**变更范围**：`tensordg/tvb.py`（判据）、`tests/_limiter_reference.py`（oracle 同步）、
`tests/test_limiter_policy.py`（一条被旧判据语义绑定的断言）、
`tests/test_tvb_indicator_two_sided.py`（新增 10 条回归）、
`benchmarks/cks1998_tvb_two_sided_indicator.py`（证据生成器）、本文。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Cockburn & Shu 1998（JCP 141:199–224）§5 的一维标量算例：周期域 `[0,1]` 上的线性对流

```
u_t + u_x = 0,   u0(x) = 1 for x ∈ [0.25,0.75],  0 otherwise,   periodic BC
```

| 参数 | 值 |
|---|---|
| 计算域 | `[0,1]`，周期边界 |
| 终态 | `t = 1.0`（方波恰好输运一个周期回到原位） |
| CFL | 0.2 |
| 时间积分 | SSP-RK3（`time_integrator="ssp_rk3"`，CKS1998 的 TVD-RK） |
| 限幅器 | TVB/minmod，`M = 0`（严格 TVD 档） |
| 多项式阶 | p = 1, 2 |
| 网格 | N = 50 / 100 / 200 / 400（均匀，h = 1/N） |
| 精确解 | 初值的刚体平移；无耗散，方波跳变幅度恒为 1 |

判据：`max(u_h) ≤ 1`、`min(u_h) ≥ 0`（点值层面）以及胞平均层面同界，越界量以初值幅度
（1）的百分比给出——即台账的「无超调 > 1 %」门。参考值不用查表：跳变幅度、平台值
（0/1）、以及胞平均 L1 误差都由解析式程序化给出（`benchmarks/cks1998_tvb_two_sided_indicator.py`
内 `square_wave` / 精确胞平均），零手抄。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **从判据反推被测对象**：`REPRO-DG-CKS1998` 的第三条判据从来只在
   `troubled_policy="all"`（逐单元 minmod）下通过（0.000 %）；论文自己的 TVB 指示器档
   （`troubled_policy="tvb"`, `M = 0`）一直是 1.40 % 的 honest FAIL（见
   `docs/audit_m15292_REPRO-DG-CKS1998.md` §3）。本单不重跑台账，而是查这个 FAIL 的根因。
2. **定位**：用 41 点 Legendre 节点还原 `u_h`，逐胞打印 `d_left, d_right, minmod, deviation`。
   在 N=100 的最终态上找点值越界胞（例如 j=20：
   `d_left = +3.40e-3`、`d_right = +6.44e-2`、`minmod = +3.40e-3`、
   `deviation = |d_left − minmod| = 0.00`），发现**判据恒为零**。
   代数上立刻可证：`deviation = |d_left − minmod(d_left,d_right)|`
   在同号且 `|d_left| ≤ |d_right|` 时**恒等于 0**——即任何右向波前的左半侧永远不会被判为
   trouble cell，与 `M`、与 `dx` 都无关。这是一处**单侧化缺陷**，不是精度/分辨率问题。
3. **修复**：把偏差改成双侧
   `max(|d_left − mm|, |d_right − mm|)`（`tensordg/tvb.py`）。可证该量仅在
   `d_left == d_right`（胞平均局部仿射，此时 minmod 斜率恰等于仿射斜率，限幅是恒等操作）
   时为零，故不丢任何可检测构型；量纲仍是 O(h²·u'')（光滑）/ O(h·u')（跳变），
   `M` 的 O(dx²) 语义与「释放光滑极值」性质不变（`tests/test_limiter_policy.py` 以
   N=20/40/80/160 逐档断言 `M=10 → 0 个 trouble cell`）。
4. **证据生成**：`benchmarks/cks1998_tvb_two_sided_indicator.py` 用同一驱动、同一求解器、
   同一度量跑两档——`--modes one_sided` 用 monkey-patch 恢复旧判据作基线，
   `--modes two_sided` 跑出厂实现——避免「旧数据 vs 新代码」的混比。
   产物：`artifacts/cks1998_tvb_indicator_two_sided.json`、
   `artifacts/cks1998_tvb_indicator_one_sided.json`。
5. **工具链**：torch float64、`OMP_NUM_THREADS=1`；本机仅跑 <110 s 短测，其它回归递交
   `run_compute`（TensorDG，stage_dir=本工作树）。

## ③ 计算结果（程序化注入，数字全部来自上述 JSON）

点值越界（%）——门 = 1 %：

| 阶 | N | one-sided（基线，旧判据） | two-sided（出厂） |
|---:|---:|---:|---:|
| 1 | 50 | **2.5788** ❌ | 0.0000 ✅ |
| 1 | 100 | **1.4036** ❌ | 0.0000 ✅ |
| 1 | 200 | **1.1895** ❌ | 0.0000 ✅ |
| 1 | 400 | —（超时单次预算，见⑤） | 0.0000 ✅ |
| 2 | 50 | 0.5644 ✅ | 0.0000 ✅ |
| 2 | 100 | 0.7791 ✅ | 0.0000 ✅ |
| 2 | 200 | 0.4668 ✅ | 0.0000 ✅ |

胞平均越界（%，TVD 理论真正控制的量）：

| 阶 | N | one-sided | two-sided |
|---:|---:|---:|---:|
| 1 | 50 / 100 / 200 | 0.1219 / 0.1483 / 0.1626 | 0.0000 / 0.0000 / 0.0000 |
| 2 | 50 / 100 / 200 | 0.0808 / 0.0751 / 0.0944 | 0.0000 / 0.0000 / 0.0000 |

胞平均 L1 误差与网格阶（限幅格式的代价，如实登记）：

| 阶 | N | one-sided L1 | two-sided L1 | 相对变化 |
|---:|---:|---:|---:|---:|
| 1 | 50 | 0.076056 | 0.099918 | +31.4 % |
| 1 | 100 | 0.047470 | 0.063143 | +33.0 % |
| 1 | 200 | 0.029914 | 0.040155 | +34.2 % |
| 1 | 400 | — | 0.025483 | — |
| 2 | 50 / 100 / 200 | 0.075347 / 0.047050 / 0.029688 | 0.100090 / 0.063143 / 0.040155 | +32.8 / +34.2 / +35.3 % |

L1 逐对阶（h 加密 N=50→100→200→400）：

| 阶 | one-sided | two-sided |
|---:|---|---|
| 1 | 0.6800, 0.6662 | 0.6621, 0.6530, 0.6560 |
| 2 | 0.6794, 0.6643 | 0.6646, 0.6530, 0.6560 |

`M` 扫描（P1, N=100，点值越界 %）：`M=0/0.1/1/10` → one-sided
1.404 / 1.416 / 1.522 / **2.375**；two-sided 0.000 / 0.000 / 0.003 / 0.027。
旧判据下加大 `M` 反而**恶化**（释放更多波前单元），新判据下单调整调。

## ④ 结果图

`docs/figures/cks1998_tvb_indicator_symmetry.png`
- (a) 点值越界 vs N（P1/P2，两档判据，含 1 % 门线）：旧判据 P1 全部越门且几乎不随 h 收敛
  （2.58→1.40→1.19 %），新判据贴零。
- (b) 胞平均 L1 误差 vs h（双对数，含 1/2 阶参考线）：两档判据的阶都 ≈0.65 > 0.5，
  新判据整条曲线抬高约 1/3（额外耗散，见⑤）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **代价是真实的**：修好单调性后 L1 误差抬高 ~33 %（表格 ③ 第三块）。这是限幅格式固有的
   「单调 vs 精度」折中——旧判据的 1.40 % 超调与更低 L1 是同一个来源（未限幅的陡峭波前）。
   本单没有粉饰：`REPRO-DG-CKS1998` 的 A2 半阶判据（≥0.5）两档都通过，而 L1 绝对值变差
   1/3，报告与本文都把它写成代价而不是收益。
2. **仍有残余盲区（诚实登记）**：任何「差值型」判据在胞平均局部仿射时偏差恒零。若一个胞
   *内部*含跳变而左右胞平均恰好落在一条直线上（例如 N=50 时方波的跳变落在
   `0.25/0.02 = 12.5` 这种半胞位置），投影阶段就能造出超调，且不会被本判据识别；
   这种算例必须用 `troubled_policy="all"`。本单没有把该情形的保护做进 `tvb` 判据。
3. **范围只到 1-D 标量**：修复经 `troubled_from_differences` 同时对 1-D Euler
   （`limiters.py`）与 2-D 方向限幅（`euler_dg_2d.py`）生效；2-D 的多维后果未做新的
   收敛/超调阶梯（2-D 侧只跑既有回归）。
4. **`M` 的语义变严了**：`M` 必须大于局部 `|u''|` 上限才彻底释放；对
   `1 + 0.2 sin(2πx)`（`|u''| ≈ 3.95`）这类波，`M = 1` 会判为陡峭区而限幅，
   `M = 10` 才全释放。这是正确语义，但会让「小 `M`」在不同算例上的表现与旧文档描述不同，
   `tests/test_limiter_policy.py` 的相应断言已按新语义改写并注明理由。
5. `tests/_limiter_reference.py` 是「逐位等价」测试的 oracle，必须同步改；否则
   `tests/test_limiter_kernel_budget.py` 会以「与 head 逐位相同」为由失败——它保护的是
   *重构不改算术*，不是*策略不得演进*。
6. N=400 的 one-sided P1 单次运行超过本机 110 s 短测预算，未纳入 JSON（不影响结论：
   N=50..200 已全部越门）。

## ⑥ 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测（出厂） | 是否 <3 % |
|---|---|---|---|
| 方波点值越界 P1, N=100 | ≤1 %（台账门） | **0.0000 %** | ✅（0 %） |
| 方波点值越界 P2, N=100 | ≤1 % | **0.0000 %** | ✅（0 %） |
| 方波胞平均越界 P1/P2 | ≤1 % | **0.0000 %** | ✅（0 %） |
| 限幅格式 L1 阶（P1/P2） | ≥0.5（CKS1998 半阶律） | 0.653–0.666 | ✅（相对余量 ≥0.15 阶） |
| 光滑波 `M=10` 释放性 | 0 个 trouble cell | N=20/40/80/160 全 0 | ✅ |
| 光滑波 A1 阶（`REPRO-DG-CKS1998` 第一条判据） | ≥p+0.5 | 不受影响（该算例无限幅） | ✅（未触及） |
| 方波 L1 绝对误差（N=100, P1） | 与旧判据比较 | 0.0475 → 0.0631（**+33.0 %**） | ❌ 变差，如实登记 |

**结论（honest）**：第三条判据（无超调 > 1 %）在论文自己的 TVB 指示器档下由
**FAIL（1.40 %）变为 PASS（0.00 %）**，根因是一处可证的单侧化判据缺陷，不是调参；
代价是限幅格式 L1 误差 +33 %，阶不变。残余盲区（胞平均局部仿射 + 胞内跳变）未修，
需 `troubled_policy="all"`。第二条判据（Sod <3 %）与第一条（阶 ≥p+0.5）未在本单重测，
沿用 `docs/audit_m15292_REPRO-DG-CKS1998.md` 与 `docs/repro_cks1998_m21122.md` 的结论。

## 复现命令

```bash
# 证据（两档判据，同一驱动）
python benchmarks/cks1998_tvb_two_sided_indicator.py --modes two_sided \
    --out artifacts/cks1998_tvb_indicator_two_sided.json
python benchmarks/cks1998_tvb_two_sided_indicator.py --modes one_sided --cells 50 100 200 \
    --out artifacts/cks1998_tvb_indicator_one_sided.json
# 回归
python -m pytest tests/test_tvb_indicator_two_sided.py tests/test_limiter_policy.py \
    tests/test_limiter_units.py tests/test_limiter_kernel_budget.py -q
```