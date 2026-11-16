# Sod DG-P2 门 `2.368 %` 的归属：登记配置在当前代码上读到什么

> 本文件由 `python3 -m validation.cks1998_dual_instrument.dg_p2_positivity_attribution --doc`
> 从工件 `docs/benchmarks/results/shock_reference_quadrature/dg_p2_positivity_attribution.json` 渲染（零手抄）。
> 上游欠账：`docs/REFERENCE-QUADRATURE-INSTRUMENT-AUDIT.md` §未闭合项、
> `docs/CASE-REPRO-DG-CKS1998-DUAL-INSTRUMENT-GATES.md` §5-1（`not_run`）。

## ① 案例介绍（物理背景 / 几何 / 参数）

Sod (Toro test 1) 激波管：`[0,1]` 域，隔膜 `x0 = 0.5`，γ = 1.4，快照 `t = 0.2`；解含左稀疏波、接触间断、右激波。

| 参数 | 左态 | 右态 |
|---|---|---|
| (ρ, u, p) | (1.0, 0.0, 1.0) | (0.125, 0.0, 0.1) |

被审的公开数字（`docs/CKS1998-round2-results.md` E2 `p2_n400`）：

| 项 | 值 | 出处 |
|---|---|---|
| 已公布最差场量相对 L2 | **2.3680 %** | docs/CKS1998-round2-results.md (E2 p2_n400); docs/AUDIT_m15345_REPRO-DG-CKS1998.md |
| 判据 | < 3.0000 % | 同一文件 |
| 已公布读数时的代码 | `103e778f8c77` | `docs/CKS1998-round2-results.md` |

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

登记门配置（`benchmarks/shock_reference_quadrature_audit.GATES`）：DG P2，N = 400，CFL = 0.12，t = 0.2，limiter = `tvb`（M = 1.0），`limiter_stage = final`。

**只放开一个变量**：`positivity` 策略。`tensordg.euler_dg_1d` 在公开读数之后把
瞬时性处理从「整单元坍缩到一阶平均」换成了默认的 Zhang–Shu 缩放（`positivity="scale"`），
而历史策略仍可通过 `positivity="collapse"` 复现。于是候选集 = 4 条：

| id | 配置 | 角色 |
|---|---|---|
| `registered_scale` | positivity=`scale`, stage=`final` | registered |
| `historic_collapse` | positivity=`collapse`, stage=`final` | historic |
| `every_scale` | positivity=`scale`, stage=`every` | control |
| `every_collapse` | positivity=`collapse`, stage=`every` | control |

每一条腿用**两台仪器**各测一次（同一份场，只换基准）：
* 旧基准：`benchmarks.riemann_exact.cell_averages`（每单元 8 点 Gauss，非收敛）；
* 精确基准：`benchmarks.shock_reference_quadrature.cell_averages`（按解析折点切段，收敛）。

预登记规则（跑之前写在驱动里）：

* **R1** registered gate configuration executed as declared; positivity is the only free variable
* **R2** a candidate reproduces the published number iff |delta| <= 0.5 pp
* **R3** attributed iff exactly one candidate reproduces it
* **R4** no tolerance, published number or registered expectation is edited

## ③ 计算结果（程序化注入）

环境：`ec6ae0b42a0c` / loongarch64 / python 3.11.6 / git `db04255c94b5`。

### 3.1 四条腿 × 两台仪器

| 腿 | steps | 最差场量 (旧基准) | 旧基准 rel L2 | 与已公布差 [pp] | 精确基准 rel L2 | 旧判决 | 精确判决 |
|---|---:|---|---:|---:|---:|---|---|
| `registered_scale` | 7281 | velocity | 3.8384 % | +1.4704 | 3.7702 % | FAIL | FAIL |
| `historic_collapse` | 7281 | velocity | 3.8384 % | +1.4704 | 3.7702 % | FAIL | FAIL |
| `every_scale` | 7282 | velocity | 3.8329 % | +1.4649 | 3.7640 % | FAIL | FAIL |
| `every_collapse` | 7282 | velocity | 3.8329 % | +1.4649 | 3.7640 % | FAIL | FAIL |

### 3.2 归属判定

* 判定：**UNATTRIBUTED**；复现已公布读数的候选 = 无。
* 规则：R3: exactly one candidate must reproduce the published number; zero -> unattributed, several -> ambiguous；指纹容差 ±0.5 pp。

**读法（未归属）**：候选集内没有任何配置复现已公布读数——登记门在当前代码上读 3.8384 %（两台仪器一致，见 3.3）。历史策略 `positivity=collapse` 的读数 3.8384 % 与默认策略同值，故**「瞬时性策略切换导致 1.47 pp 位移」这一假设被否定**。因此已公布读数与当前代码之间是**候选集未覆盖的代码状态差**；本单如实登记 `unattributed`，不指定解释，也不改写已公布数字。下一步应把候选扩展到 `limiter` 实现版本（TVB 限幅器在公开读数之后被重写过）与限幅器关闭/点值范数等口径，属于独立的取证单。

### 3.3 登记门在当前代码上的读数

| 量 | 旧基准 rel L2 | 精确基准 rel L2 | 仪器位移 [pp] | < 3 % (精确基准) |
|---|---:|---:|---:|:--:|
| density | 1.1799 % | 1.1657 % | +0.0142 | 是 |
| velocity | 3.8384 % | 3.7702 % | +0.0681 | 否 |
| pressure | 0.8536 % | 0.8433 % | +0.0103 | 是 |

### 3.4 历史策略（`positivity=collapse`）

| 量 | 旧基准 rel L2 | 精确基准 rel L2 | < 3 % (精确基准) |
|---|---:|---:|:--:|
| density | 1.1799 % | 1.1657 % | 是 |
| velocity | 3.8384 % | 3.7702 % | 否 |
| pressure | 0.8536 % | 0.8433 % | 是 |

## ④ 结果图

* `docs/figures/cks1998_dg_p2_positivity_profiles.png` —— 四条腿的 ρ/u/p 单元平均
  vs 精确基准（左），以及「最差场量 rel L2 vs 已公布 2.368 % / 3 % 门」的柱状对账（右）。

## ⑤ 注意点 / 局限（honest）

1. **这不是「求解器变好/变坏」的判决**：两条腿都过/不过 3 % 各自成表，本单只回答「已公布的那个数由哪个代码状态产生」。
2. **候选集不完备**：只枚举了 `positivity` 一个开关（公开读数之后唯一改变数值路径的默认值）。若上游还有未被登记的实现变更，本归属可能只是最接近的解释——`attributed` 的前提是其它候选都不在指纹带内，见 3.2。
3. **两台仪器的差 ≤ 0.07 pp**，远小于 1.47 pp 的分歧，所以分歧不能归因于基准。
4. **不改台账、不改容差、不改已公布数字**：本单只增加归属与实测读数，已公布读数原样挂在 `docs/CKS1998-round2-results.md`。

## ⑥ 误差判定（逐项回答「是否 < 3 %」）

| 关键量 | 基准 | 读数 | 相对误差 | < 3 % |
|---|---|---:|---:|:--:|
| `registered_scale` 最差场量 (velocity) | 精确单元平均 | 3.7702 % | 3.7702 % | 否 |
| `historic_collapse` 最差场量 (velocity) | 精确单元平均 | 3.7702 % | 3.7702 % | 否 |
| `every_scale` 最差场量 (velocity) | 精确单元平均 | 3.7640 % | 3.7640 % | 否 |
| `every_collapse` 最差场量 (velocity) | 精确单元平均 | 3.7640 % | 3.7640 % | 否 |
| 已公布读数（103e778f8c77） | 旧基准 | 2.3680 % | 2.3680 % | 是 |

**结论**：见 §3.2 的归属判定；两条读数各自与 3 % 门的距离写在 §3.3/§3.4，不做四舍五入式的粉饰。

