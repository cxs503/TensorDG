# CLOSEOUT m16918 — p-refinement 判据收紧与测量地板量化（关闭 m16499 F4/F5）

> 锚点：`CASE-DG-PREFINEMENT-EXPONENTIAL`（feature_ledger=open）  
> 上游审计：`docs/AUDIT-m16499-CASE-DG-PREFINEMENT-EXPONENTIAL.md`（F1–F3 已修，F4「判据非鉴别」/F5「p≥7 未测」遗留）  
> 本单：m16918（verifier 席）· 基线 commit `ac49329` · 数字全部由脚本从工件直读（`tools/render_precision_closeout.py`），零手抄

## 1. 做了什么（两项遗留关闭）

**① F4 判据收紧（代码级，不再是“建议”）** — `benchmarks/p_refinement_exponential.py` 的 verdict 由复合预登记判据门控：

```
sigma > 1   AND   逐段衰减率不减(相对容差1e-9)   AND   e(p_min)/e(p_max) > 1e3   AND   误差单调下降
```

旧的 `sigma > 0.5` 保留为 `checks.decay_rate` 子项（可比性），但不再驱动结论；每次跑基准都会内嵌一张**证伪对照表**（已知衰减律 × 判据裁决），门 `falsification_control.gate_pass` 必须为真。

**② F5 测量地板量化（从 not_run 变成数字）** — 新增 `benchmarks/p_refinement_floor.py`：同一算例三路独立测量（生产 float64 / 另一求积阶 float64 / `np.longdouble` 80 位累加），与可表示解地板 `eps64*||u_h||` 比较，给出信噪比与“最大可证实误差降幅”。

**③ 前置条件修复** — 主工件重跑后 `independent_check.json` 的 `primary_sha256` 必须同步（否则溯源门红），已重跑复核脚本。

## 2. 证伪对照表（程序化从工件 `falsification_control.laws` 生成）

| 已知衰减律 | σ(最小二乘) | 总降幅 e(p1)/e(p6) | 旧判据 σ>0.5 | 新复合判据 | 期望 | 相符 |
|---|---|---|---|---|---|---|
| constant | 0.0000 | 1.000e+00 | fail | fail | fail | ✅ |
| algebraic_p^-1 | 0.3427 | 6.000e+00 | fail | fail | fail | ✅ |
| algebraic_p^-2 | 0.6854 | 3.600e+01 | PASS | fail | fail | ✅ |
| algebraic_p^-3 | 1.0282 | 2.160e+02 | PASS | fail | fail | ✅ |
| algebraic_p^-4 | 1.3709 | 1.296e+03 | PASS | fail | fail | ✅ |
| algebraic_p^-5 | 1.7136 | 7.776e+03 | PASS | fail | fail | ✅ |
| exponential_sigma_0.5 | 0.5000 | 1.218e+01 | fail | fail | fail | ✅ |
| exponential_sigma_1.0 | 1.0000 | 1.484e+02 | PASS | fail | fail | ✅ |
| exponential_sigma_1.5 | 1.5000 | 1.808e+03 | PASS | PASS | PASS | ✅ |
| exponential_sigma_3.93 | 3.9270 | 3.368e+08 | PASS | PASS | PASS | ✅ |

`gate_pass = True`；旧判据假阳性 = ['algebraic_p^-2', 'algebraic_p^-3', 'algebraic_p^-4', 'algebraic_p^-5', 'exponential_sigma_1.0']。
**关键发现（新增，超出原审计）**：σ>1 本身也不能分家——代数律 p⁻³/p⁻⁴/p⁻⁵ 的 σ 分别为 1.028, 1.371, 1.714，且 p⁻⁴/p⁻⁵ 的总降幅 (1296×/7776×) 也越过 1e3 门；**唯一决定性条件是逐段衰减率单调不减**（代数律必递减）。故该条件不可选。

## 3. 真实求解器上的反例（判据有牙，不是只拒合成序列）

| 初值 | 光滑性 | p 阶梯 | σ | 总降幅 | 旧判据 | 新判据 | 说明 |
|---|---|---|---|---|---|---|---|
| sin(2πx) | 整函数 | 1–6 | 3.9271 | 3.35e+08 | PASS | **PASS** | 正例（本算例） |
| 三角波 2x/(2−2x) | C⁰ 折点 | 1–4 | 0.544 | 1.7e+01 | **PASS** | **拒** | 旧判据在真实非光滑解上假阳性 |
| 方波 ±1 | 间断 | 1–3 | −0.008 | 0.97 | fail | **拒** | Gibbs 主导，误差不降 |
（复现：`pytest -q tests/test_p_refinement_criterion.py::test_real_kinked_solution_passes_the_legacy_gate_but_is_rejected` 等；三角波 σ>0.5 而逐段衰减率 1.053→0.648→0.466 递减，正是代数律指纹）

## 4. 测量地板（F5 量化）

| p | DOFs | L2 (生产 float64) | L2 (longdouble) | 路径相对离散 | 解范数 ||u_h|| | 地板 eps·||u_h|| | 信噪比 | 可用 |
|---|---|---|---|---|---|---|---|---|
| 5 | 96 | 1.592692e-09 | 1.592692e-09 | 5.47e-09 | 0.707107 | 1.570e-16 | 1.014e+07 | 是 |
| 6 | 112 | 2.228252e-11 | 2.228254e-11 | 8.42e-07 | 0.707107 | 1.570e-16 | 1.419e+05 | 是 |

地板 = **1.570e-16**（= eps64 × max||u_h||），snr_min = 10.0。按已发布 σ=3.927 外推：

| 预测 p | 预测 L2 | 预测信噪比 | 结论 |
|---|---|---|---|
| 7 | 4.390e-13 | 2795.9 | 可测 |
| 8 | 8.648e-15 | 55.1 | 可测 |
| 9 | 1.704e-16 | 1.1 | **进入地板** |

即 `MAX_DEGREE=8` 仍在可测区间（p=8 信噪比约 55），**p=9 起不可测**——这是量化结论，取代原审计的 “p≥7 not_run”。

## 5. 注意点 / 局限

- **未跑 p=7/8 实测**：p=7 在 cfl 自动细化回路下单点 >120 s（本机 load15≈93，容量纪律），故 p≥7 是**外推**而非实测；外推前提是本算例 σ 在 p=6→7 不塌（连续性假设，未证）。
- 判据的 `e(p_min)/e(p_max)>1e3` 是**尺度相关**的：只对 p 阶梯跨度 ≥5 阶且 σ≳1.5 成立；σ≈1 的真指数律也判拒（属保守设计，已在证伪表钉住）。
- 地板 “eps·||u_h||” 是表示层地板；四路测量间的实际离散（p=6 相对 8.4e-07）更大，故真实可用域比地板估计更保守。
- 与第三方 DG 实现交叉仍 not_run（无本地依赖）。
## 6. 误差判定（逐项，回答“是否<3%”）

| 项 | 基准 | 结果 | 是否<3% |
|---|---|---|---|
| 重跑主工件 vs 已发布 JSON（6 个 p 的 L2） | 逐位 | 最大相对差 ≤ 1e-12（同 σ=3.9271） | ✅ |
| σ vs 预登记阈值 1.0 | 阈值 | 3.9271（3.9 倍裕度） | ✅ |
| 判据鉴别力 vs 合成已知律（10 条） | 期望裁决 | 10/10 相符 | ✅（门 gate_pass=True）|
| 三角波真实反例 vs 旧判据 | 期望“拒” | σ=0.544 旧判据误判 PASS，新判据拒 | ✅（判据有效）|
| 地板量化 vs 独立 longdouble 路径 | 相对 | 最大路径离散 8.42e-07 | ✅ |
| p=7/8 实测 vs 外推 | — | 未跑 | ⚪ 无基准（声明外推）|

## 7. 交付工件

| 文件 | 说明 |
|---|---|
| `benchmarks/p_refinement_exponential.py` | 复合预登记判据 + 证伪对照套件 + verdict 门控（改动）|
| `benchmarks/p_refinement_floor.py` | 测量地板/信噪比/外推脚本（新增）|
| `tensordg/convergence.py` | `roundoff_floor` / `signal_to_noise` / `resolved_levels` 通用验证地板工具（末尾新增）|
| `tests/test_p_refinement_criterion.py` | 24 道新闸（22 passed, 2 依赖工件 skip 条件）|
| `docs/benchmarks/results/p-refinement-exponential/*` | 重跑主工件（含 criterion/falsification）、`measurement_floor.json`、重跑 independent_check、provenance |
| `docs/CASE-DG-PREFINEMENT-EXPONENTIAL.md` | §7 增补：判据收紧落地与地板量化 |
