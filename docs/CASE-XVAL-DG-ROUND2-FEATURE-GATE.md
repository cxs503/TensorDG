# 集合级功能量门 + 单元平均极值反卷积（`XVAL-DG-ROUND2-CERTIFIED-REFERENCE` 的收口增量）

锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`。本单把该锚上一轮自己留下的、**唯一还没做成机器可检规则**的
遗留项落地：

> `docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md` §6 结论：
> 「全局 L1 门槛对该算例不充分，**建议在 CSV/平台验收中补一条功能量级门槛（≥1 个积分型 + 1 个极值型，均带宽感知）**」

---

## ① 案例介绍

* 物理背景：1-D Euler（γ=1.4）M=3 激波 + 其后的熵波列（Shu–Osher 问题），输出时刻 t=1.8。
  激波后的**波列幅度/位置**是工程上最容易被「全局 L1 合格」掩盖的特征量。
* 数据：仓库内已提交的 cell-average 剖面 `validation/repro_shu_osher/artifacts/raw/*.npz`
  （FV-WENO5 阶梯 N=800/1600/3200/6400 作认证者；DG P1/P2/P3 × N=200/400/800 + CFL 三档共 11 条腿）。
  **本单不跑任何新仿真**，全部数字可从已提交剖面复算。
* 参数：门 τ = 3 %（owner 精度标准），带宽上限 U/τ ≤ 1/3（引擎自带非空洞规则），
  统计窗 x ∈ [−2.7031, 2.2894]（解析头波/激波内缩 0.05/0.10）。

## ② 怎么做的（方法/流程/关键设置）

1. **新模块 `tensordg/feature_gate.py`**（算例无关，无任何域语义硬编码）：
   * `cell_average_extremum(values, x=…, mode="max"|"min")`：单元平均极值的**反卷积估计器**。
     对光滑场，单元均值 `<f>_i = f(x_i) + h²/24·f''(x_i) + O(h⁴)`，因此「取单元均值最大」天然带
     O(h²) 偏置、且位置被量化到单元中心。该函数用三点抛物线拟合求顶点（消量化）+ 曲率求
     `h²/24·f''` 并扣除（消偏置）。**它只看 cell-average 数组，看不到精确解**，因此不是把
     答案塞进测量。边界极值 / 非均匀网格 / 退化抛物线一律**拒绝并回落到原始值**，显式返回 `status`。
   * `judge_feature_set(candidate, references=…, kinds=…, required=…)`：集合级门。
     对每个**必需功能量类型**要求 ≥`required[kind]` 个功能量在带宽感知容差下 `PASS`
     （`PASS_MARGINAL` 默认不计）。证据不足（缺该类型 / 参考被拒 / 参考值为零）→ `GATE_WITHHELD`，
     **绝不当作通过**；`required={}` 直接抛错（空要求即空洞门）。
   * `gate_teeth_check(...)`：反空洞自检——同一门必须接受「候选=认证值」、并拒绝被扰动
     `perturbation_percent` 的候选；两半任一不成立即判该电池无齿。
2. **驱动 `benchmarks/xval_dg_round2_feature_gate.py`**：对已提交剖面同时用两种估计器
   （raw / deconvolved）认证 6 个功能量、测偏置衰减指数、跑集合级门、与已发布的 profile 门逐腿对照。
3. **测试**：`tests/test_feature_gate.py`（15 项，全部对**解析真值**断言）+ 
   `tests/test_xval_dg_round2_feature_gate.py`（8 项，对已提交数据断言不变式）。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 反卷积估计器的阶（解析真值：`f=sin 2πt` 的精确单元均值）

| N | raw max 误差 | 抛物线顶点误差 | **反卷积误差** | 反卷积观测阶 |
|---|---|---|---|---|
| 100 | 6.58e-04 | 1.65e-04 | 5.19e-07 | — |
| 200 | 1.65e-04 | 4.12e-05 | 3.25e-08 | 4.00 |
| 400 | 4.11e-05 | 1.03e-05 | 2.03e-09 | 4.00 |
| 800 | 1.03e-05 | 2.57e-06 | 1.27e-10 | 4.00 |

位置：对称峰下三点均值对称 ⇒ 顶点位置**精确到舍入（<1e-12）**，而 raw argmax 永远是 h/2 阶梯。

### 3.2 真实剖面上的偏置律（认证者阶梯）

| N | h | 偏置 `h²/24 f''` | f'' |
|---|---|---|---|
| 800 | 1.250e-02 | −1.4135e-03 | −2.1711e+02 |
| 1600 | 6.250e-03 | −3.6716e-04 | −2.2558e+02 |
| 3200 | 3.125e-03 | −9.2271e-05 | −2.2676e+02 |
| 6400 | 1.562e-03 | −2.4483e-05 | −2.4068e+02 |

实测衰减指数 **1.945 / 1.992 / 1.914**（期望 2）→ 偏置项形式在实测数据上得到验证。
**关键量级**：最粗级偏置 1.41e-03 **是**最细级相邻差（2.45e-05）的 **57 倍**——偏置并非可忽略项。

### 3.3 功能量认证：两种估计器对照（Q1）

| 功能量 | 类型 | raw 判决 | raw p | raw 带 % | 反卷积判决 | 反卷积 p | 反卷积带 % | 判决改变 |
|---|---|---|---|---|---|---|---|---|
| amplitude | extremum | CERTIFIED | 1.601 | 0.1169 | CERTIFIED | 1.455 | 0.1616 | 否 |
| crest | extremum | CERTIFIED | **4.703** | 2.6e-05 | **WITHHELD_STAGNANT** | −0.961 | — | **是** |
| l1 | integral | WITHHELD_STAGNANT | −0.715 | — | WITHHELD_STAGNANT | −0.715 | — | 否 |
| peak_x | phase | **WITHHELD_STAGNANT** | −0.585 | — | **CERTIFIED** | 0.965 | 0.0634 | **是** |
| rms | integral | CERTIFIED | 1.512 | 0.0762 | CERTIFIED | 1.512 | 0.0762 | 否 |
| trough | extremum | CERTIFIED | 1.684 | 0.0593 | CERTIFIED | 1.568 | 0.0734 | 否 |

* **crest 撤回**：raw 阶梯给 p=4.703、带 2.6e-05 %，这比 WENO5 的可信阶高一个量级——
  原因是 O(h²) 偏置本身随 h 单调衰减，把真实阶梯的非单调部分「抹平」成假收敛。扣除偏置后
  最细两级差值变号（+1.23e-4 → −2.39e-4），引擎按停滞拒证。**上一轮发布的「crest 认证」是测量伪影**，
  本单如实撤回（honest-FAIL，不粉饰）。
* **peak_x 转正**：raw 的峰值位置是 h/2 阶梯（1.843750→1.840625→1.842188→1.839844，非单调）→ 拒证；
  反卷积后差值 −3.31e-3/−1.73e-3/−8.88e-4 单调、阶 0.965，带 0.063 % —— 一个**位置型**功能量
  首次可认证，补上了功能量门的相位置轴。
* 积分型功能量（rms、l1）两估计器**逐位一致**（构造上如此），l1 的停滞是已知的激波边缘窗口抖动（§⑤）。

### 3.4 集合级门 vs profile 门（11 条 DG 腿，Q2）

| 腿 | profile rel L1 % | profile 门 | 功能量门（raw） | 功能量门（反卷积） |
|---|---|---|---|---|
| P1 N=200 | 2.456 | PASS | GATE_FAIL | GATE_FAIL |
| P1 N=400 | 1.548 | PASS | GATE_FAIL | GATE_FAIL |
| P1 N=800 | 0.5786 | PASS | GATE_FAIL | GATE_FAIL |
| P2 N=200 | 2.420 | PASS | GATE_FAIL | GATE_FAIL |
| P2 N=400 CFL 0.1/0.2/0.4 | 1.520/1.518/1.516 | PASS×3 | GATE_FAIL×3 | GATE_FAIL×3 |
| P2 N=800 | 0.5667 | PASS | GATE_FAIL | GATE_FAIL |
| P3 N=200/400/800 | 2.427/1.522/0.6699 | PASS×3 | GATE_FAIL×3 | GATE_FAIL×3 |

* **11/11 通过 profile 门；0/11 通过功能量门**（要求 {integral≥1, extremum≥1}）。
  「通过 L1 却掉功能量」的反例数 = **11**，profile 门充分性 = **False**（机器判定）。
* 门有齿自检：候选=认证值 → `GATE_PASS`；候选统一扰动 10 % → `GATE_FAIL`（`has_teeth=True`）。
* 失败轴：最粗 P1 腿的 amplitude 误差 67.9 %、rms 53.6 %、trough 19.1 %、peak_x 3.59 %；
  随 N 加密单调下降（P1 N=800: amplitude 18.4 %、rms 12.0 %），与上一轮「DG+WENO 限制器过度耗散」的
  归因一致，且**不是时间积分**（CFL 0.1/0.2/0.4 极差 0.19 pp）。

## ④ 结果图

* `fig1_ladder_estimators.png`：crest / amplitude 的认证者阶梯（raw vs 反卷积）+ 副轴 |bias| 的 log-log 衰减。
* `fig2_feature_errors.png`：11 条腿 × 4 个功能量的相对误差（对数轴）+ 3 % 门线。
* `fig3_extremum_zoom.png`：波列极值放大——raw argmax 与反卷积顶点的位置差异可视化。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **本单最硬的事实是「撤回」，不是「新增」**：crest 的认证判决被反卷积推翻。任何继续引用
   `docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md` §3 中 crest=CERTIFIED(p=4.70) 的下游结论，
   都应在 estimator 意义上重新判定。该文档未改动，本单在 §3.3 显式登记冲突。
2. **反卷积不等于「修好功能量」**：它只消除表示层的 O(h²) 偏置；真实阶梯若仍非单调（crest），
   引擎照样拒证。本单不做任何 `stagnant → 平均掉` 的处理。
3. **l1 的停滞未解决**：窗口右边界固定在 x_shock−0.1，陡峭段格子进出带来 O(h) 抖动（上一轮已披露）。
   本单保持原状，仅在测试中把它当作「withheld ≠ pass」的实例。
4. **门是「集合级」而非「新物理判据」**：`required` 配额由调用方给定；本单用文档推荐的
   {integral:1, extremum:1}，另有 {…, phase:1} 的更严变体跑通（peak_x 认证后可用）。
5. **适用范围**：1-D Euler cell-average 剖面 + 任一「有认证参考的标量场」。不对多维/粘性/湍流作声明。
6. **不涉及新算例**：所有数据取自仓库已提交剖面，因此「与文献值对标」仍为 WITHHELD
   （Shu–Osher 原文只给图不给数值表），基准是 Richardson 认证参考解本身。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 < 3 %」）

| 关键量 | 基准 | 测量 | 误差 | < 3 %？ |
|---|---|---|---|---|
| 反卷积极值估计器（解析 sin） | 峰值 1 | 1 + 1.27e-10 | 1.27e-10 % | **是** |
| 偏置律指数（实测剖面） | 2 | 1.945 / 1.992 / 1.914 | 最大偏差 4.3 % | **否（4.3 %）**，但指数在 1.914–1.992，偏置量级 1.4e-3…2.4e-5 与 `h²/24 f''` 闭式一致（见 §3.2） |
| amplitude 认证值 raw vs 反卷积 | 0.827972（raw） | 0.827710（反卷积） | 0.0317 % | **是** |
| rms 认证值 raw vs 反卷积 | 0.29992366843862206 | 同值（逐位） | 0 | **是** |
| crest 原发布带 U | — | 2.6e-05 %（raw） | 判决被撤回 = 不可用 | **否（该量不再可引用）** |
| peak_x 反卷积带 U | — | 0.0634 %（占 3 % 门 2.1 %） | — | **是** |
| 11 条腿 amplitude 误差 | 认证值 | 18.4 %–67.9 % | — | **否（11/11 超 3 %，这就是门要拦的）** |

**结论**：本单新增/校正的两类量——估计器精度（1.27e-10 %）与可用参考带（0.03–0.16 %）——均 < 3 %；
偏置衰减指数的 4.3 % 偏差属于「指数拟合对最粗级不渐近」的正常范围，如实列出不粉饰；
而 DG 腿的功能量误差 18–68 % 是**被门拦下的不合格证据**，不是本单的精度指标。

## ⑦ 产物与复现

| 产物 | 路径 |
|---|---|
| 集合级门 + 极值反卷积 | `tensordg/feature_gate.py`（并在 `tensordg/__init__.py` 导出） |
| 驱动 | `benchmarks/xval_dg_round2_feature_gate.py` |
| 审计工件 | `validation/xval_dg_round2/artifacts/feature_gate/feature_gate_audit.{json,md}` + 3 图 |
| 测试 | `tests/test_feature_gate.py`（15）、`tests/test_xval_dg_round2_feature_gate.py`（8） |

```bash
PYTHONPATH=. python -m benchmarks.xval_dg_round2_feature_gate --figures
PYTHONPATH=. python -m pytest tests/test_feature_gate.py tests/test_xval_dg_round2_feature_gate.py -q
```

## ⑧ 给平台验收的接口（落实 §6 建议）

`judge_feature_set(candidate, references=…, kinds=…, required={"integral":1,"extremum":1})`
返回 `verdict ∈ {GATE_PASS, GATE_PASS_MARGINAL, GATE_FAIL, GATE_WITHHELD}` 与逐功能量明细；
`GATE_WITHHELD` 必须映射为平台侧的「证据不足」（不得记为通过），这正是 CSV/平台验收里那条
「带宽感知功能量门槛」可直接调用的形态。