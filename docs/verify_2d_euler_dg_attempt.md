# TensorDG 高阶二维 Euler DG：验证欠账首轮复核（honest FAIL）

- 任务：m2143（models 席自找活），锚点 = `expand_audit_tensordg.md` §6 backlog 第 3 项
  「2D Euler 高阶（非 P0）」，README 也把 "high-order multidimensional Euler" 明确列为 out of scope。
- 分支：`agent/m2143`　日期：2026-11-02　Python 3.11 / torch 2.7.1 / pytest
- 结论一句话：**新增的二维高阶 DG 装配未通过验证（误差 100%，远超 3% 门），按 honest FAIL 归档**；
  三条子缺陷已定位并修复，剩一条已精确收敛到「面通量投影」这一步，可直接派单续做。

---

## 1. 案例介绍（要验证什么）

已合并能力盘点（读 `expand_audit_tensordg.md` / `validation_report.json` / git log）：

| backlog # | 项 | 状态 |
|---|---|---|
| 1 | TVB/WENO 限制器 | ✅ 已合并（m274，PR #6） |
| 2 | Matrix-free / sparse SIPG | ✅ 已合并（m275，PR #7），与 dense 对比 1e-13 |
| **3** | **2D Euler 高阶（非 P0）** | ❌ 空白：`tensordg/euler_2d.py` 只有 P0/FV，`metadata["degree"]=0` 写死 |
| 4 | 非结构网格 | 未做（P2） |
| 5 | NS 粘性通量 | 未做（P2） |

因此本轮选 **#3**：为二维 Euler 增加 tensor-product 模态 DG（P0–P3）+ HLLC 法向通量 + SSP-RK3
+ positivity 投影回退 + 可选方向性 minmod/TVB 限制，并给出可判定的验证方案：

- **B1 光滑** 等熵涡（解析解）：设计要求 EOC = p+1，误差 < 3%；
- **B2 挤压一致性** Sod 管挤成二维（y 不变、单层网格）：二维残差必须逐格等于独立实现的
  一维 DG 残差，且二维解 vs Sod 精确解 < 3%；
- **B3 真二维** Lax–Liu 配置 3 四象限 Riemann 问题：正性 + 网格自收敛（无解析解，如实声明）。

物理/几何/参数：理想气体 γ=1.4；涡：定义域 [-5,5]²，强度 β=5，背景速度 (1,1)，
t∈{0.5,1.0}；Sod：域 [0,1]，隔膜 x=0.5，左 (1,0,1)、右 (0.125,0,0.1)，t=0.2；
四象限：域 [0,1]²，t=0.25，状态取自 Lax & Liu (1998) 配置 3。

## 2. 怎么做的（方法 / 工具链）

新增 `tensordg/euler_dg_2d.py`（tensor-product Legendre 模态基，`P_m(+1)=1`、
`P_m(-1)=(-1)^m`；面通量逐面积分点 HLLC；体积项五操作数 `einsum`；SSP-RK3；
`apply_tvb_limiter_2d` 方向性 minmod；`_positivity_filter` 在**格平均 + 体积积分点 + 两个方向迹**
上判正性）。验证脚本 `benchmarks/diagnose_euler_dg_2d.py`（一键复现缺陷）、
出图 `benchmarks/plot_euler_dg_2d_diagnosis.py`、回归位 `tests/test_euler_dg_2d_experimental.py`。
交叉参照对象是仓库里**独立实现**的一维高阶 DG（`tensordg/euler_dg_1d.py`，已合并并通过 Sod 2.5% 验证）。

推导（二维弱形式，用于定标）：

```
(c_kl)_t = -[(2k+1)(2l+1)/(2 dx)] * ( phi_k(+1) F_r[l] - phi_k(-1) F_l[l] - V[k,l] )
V[k,l] = sum_ab w_a w_b F(a,b) dphi[a,k] phi[b,l],  F_r[l] = sum_b w_b Fhat(+1,eta_b) phi[b,l]
```

当 l=0 时 `F_r[0]=2*Fhat`、`V[k,0]=2*V_1D`，故二维残差**必须**逐格等于一维残差——这正是 B2 的判据。

## 3. 计算结果（全部程序化读出，见 `benchmarks/diagnose_euler_dg_2d.py` 输出）

### 3.1 装配正确性的两条"必过"检查（均 PASS）

| 检查 | 判据 | deg 0 | deg 1 | deg 2 |
|---|---|---:|---:|---:|
| 均匀流残差 | 应 = 0 | 0.0 | 3.3e-16 | 3.0e-14 |
| y 不变数据不激发 y 模态 | < 1e-12 | — | — | 1e-16 |

均匀流残差严格为 0，说明几何/标定/守恒权重这一层没错（否则恒流也会产伪残差）。

### 3.2 B2 挤压一致性（FAIL，误差 100%）

网格 nx=100、ny=1，dx=0.01，dy=0.25，P1，隔膜在 cell 49/50 之间：

| 量 | 数值 | 说明 |
|---|---:|---|
| HLLC 界面质量通量（面 49，q=0） | **0.430260348** | 与一维参考解一致 |
| 该通量经 `_axis_fluxes` 投影后 | **0.0** | 通量在投影一步完全丢失 |
| 一维 DG 密度残差 max | **-43.0260348** | = -0.4302603/dx，逐位吻合 |
| 二维 DG 密度残差 max | **0.0** | 密度（和能量）方程恒为 0 |
| 二维 DG 能量残差 max | **0.0** | 一维同量 116.170294 |
| 投影损失比例 | **100%** | `projection_lost_fraction = 1.0` |

关键读数：**Riemann 求解器是对的**（0.430260348 与一维参考 43.0260348·dx 完全一致），
**错在把界面通量投影到面试探函数那一步**，导致 x 方向残差恒为零、物质完全不输运。

### 3.3 挤压 Sod 的非物理化过程（FAIL 的动力学证据）

P1、nx=100、cfl=0.2，逐步最小格平均压力：

| step | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| min p̄ | 9.483e-2 | 8.687e-2 | 7.167e-2 | 5.680e-2 | 3.745e-2 | 1.637e-2 | **-6.211e-3** |

7 步后触发 `FloatingPointError`。格平均压力从 1.0e-1 单调崩塌，与 3.2 的"残差恒零 +
动量/能量方程错误耦合"一致：质量不流动但动量和能量被强行推动，等价于凭空加热/抽空。

### 3.4 已定位并修复的三条子缺陷（每条都有最小复现）

| # | 缺陷 | 症状 | 修法 |
|---|---|---|---|
| a | HLLC 星区 `speed - normal_velocity` 被 `clamp_min(1e-14)` | 左声波该项本应为 **负**（= -a），钳正后能量通量爆到 2.5e12 | 只钳幅值、保留符号 |
| b | y 面迹取样用错表面求积指标（把 y 积分点当成了面求积点） | y 模态残差非零（1e-1 量级），y 方向伪源 | `_traces(..., axis)` 分方向取样 |
| c | positivity 只查格平均 | HLLC 在迹点收到负压状态，NaN 扩散 | 改为查格平均 + 体积积分点 + 两个方向迹（与一维实现对齐） |

三条修完后，3.1 的两条必过检查转绿、y 伪源消失，但 3.2 的投影缺陷仍在 → 本轮判定 FAIL。

## 4. 结果图

`outputs/euler_dg_2d_diagnosis.png`（由 `benchmarks/plot_euler_dg_2d_diagnosis.py` 生成）：

- 左：柱状图，界面 HLLC 通量 0.4303 / 投影后 0.0 / 一维参考残差 43.0262(坐标轴同量纲下比较投影损失)；
- 右：挤压 Sod 的二维（P1, 64 格, t=0.001）与一维 DG 密度剖面对比。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **单机环境陷阱**：本机 torch 默认 128 线程，32² P2 的 `_rhs` 高达 0.75 s；`torch.set_num_threads(1)`
   后 0.058 s（12 倍差）。二维 DG 小网格必须限制线程，否则误判为"算法太慢"。
2. **/tmp 共享**：同机多席并行，`/tmp/<script>.py` 会被别的席覆盖（本轮实测踩到 traceback 指向别人的代码）；
   一律用 `/tmp/<mission>/` 私有目录。
3. 本轮所有超时/失败都在**单层（ny=1）**挤压算例上暴露，光滑涡算例反而不会暴露——说明
   **光滑算例不能替代挤压/激波算例做装配验证**（光滑解下 x 方向残差为 0 时"解看起来还在动"）。
4. 未完成的部分（如实）：B1 光滑涡收敛阶、B3 四象限算例均**未取得可信数据**（上游装配缺陷未修，
   跑出的数没有意义，故不报告、不粉饰）；二维高阶 DG 未导出到 `tensordg/__init__.py`，不进入公开 API。

## 6. 误差判定（关键量逐项 vs 基准）

| 关键量 | 基准 | 实测 | 相对误差 | 是否 < 3% |
|---|---|---:|---:|---|
| 二维/一维残差一致性（密度，cell 49） | -43.0260348（一维 DG） | 0.0 | **100%** | ❌ 否 |
| 二维/一维残差一致性（能量，max） | 116.170294 | 0.0 | **100%** | ❌ 否 |
| y 伪源（挤压数据） | 0 | 1e-16 | 1.4e-16 | ✅ 是 |
| 恒流残差 | 0 | 0.0 / 3.0e-14 | — | ✅ 是 |
| 挤压 Sod vs 精确解 | Sod 精确解 | 解在 t≈0.0024 非物理化 | 不可定义 | ❌ 失败（未达判定点） |

**判定：本轮新增能力不满足 3% 门，FAIL。** 保留的三条修复与两条必过检查是真实增量（把缺陷从
"整体不可用"收敛到"单点投影 bug"），但不构成可发布能力。

## 7. 给下一次自找活的建议（可直接派单）

缺陷已收敛到一步：`_axis_fluxes` 中
`torch.einsum("xybv,b,bl->xylv", interface, weights, basis)` 在本构数据上返回 0，而同一 einsum
在最小复现里（常数 `interface=1`、`weights` 和为 2、`basis[:,0]=1`）返回 2.0 —— 说明问题在
**传入 `_axis_fluxes` 的 `interface` 与脚本里单独算的 `interface` 不是同一个张量**
（最可疑处：`neighbour` 的克隆/末尾覆盖与 `_hllc` 的 `degenerate` 分支），
建议下一步：(i) 在 `_axis_fluxes` 内断言 `interface[...,0]` 与外部一致；
(ii) 或直接把整条面通量投影改成 `torch.tensordot`/显式 `matmul`，绕开多操作数 einsum。
判据现成：`tests/test_euler_dg_2d_experimental.py::test_extruded_sod_reproduces_the_1d_dg_solution`
（`xfail(strict=True)`），修好后会自动 XPASS 并要求摘掉 pin。
