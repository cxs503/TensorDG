# 开发报告：系统化收敛阶与 GCI 网格无关性验收

- 任务：求解器开发波 S1·TensorDG（mission 2091），分支 `agent/m2091`
- 输入锚：`expand_audit_tensordg.md`（首波审计）、`dev_report_tensordg-matrix-free-sipg.md`（m275）、`dev_report_tensordg-tvb-weno-limiter.md`
- 日期：2026-11-02
- 环境：CPU、Python 3.11、PyTorch（float64）、pytest；宿主机 128 核高负载（load ≈ 150–300），套件用 `OMP_NUM_THREADS=1`

## 1. 缺口定位（来自缺口账 + 代码实况）

首波审计结论：8 个 benchmark 各带单点误差门，21 条测试覆盖守恒/收敛/可接受性/精确参考/3% 门；m275 补了 sparse + matrix-free SIPG 后端，m282 补了 TVB/WENO 限制器。逐条核对既有收敛断言后发现：

| 现状 | 问题 |
|---|---|
| `tests/test_poisson_2d.py`、`test_advection_1d.py`、`test_heat_1d.py` 的"收敛"断言 | 只用 **2 个网格** 估阶、单侧下界（如 `rate > 1.8`），无渐近性证据，无 GCI，无 p 加密 |
| 无 Richardson 外推 / 网格无关性工具 | 无法回答"最细网格离收敛极限多远" |
| 各 benchmark 的阶计算散落在脚本内联代码 | 无统一、可单测的估计器 |
| matrix-free / sparse 后端 | 只验过与 dense 的 max_diff，未验"阶是否保持" |

因此选题：**建立收敛阶 + GCI 的系统化验收层**（验证面优先），而不是继续加物理算例。

## 2. 交付物

| 文件 | 内容 | 行数 |
|---|---|---|
| `tensordg/convergence.py` | 验证内核（纯 NumPy，与被测求解器解耦）：`pairwise_orders`、`observed_order`、`richardson_extrapolate`、`gci_study`（Roache 阶 + GCI + 渐近比）、`convergence_table`、`p_decay_diagnostics`、模态系数 QoI 提取（点值/域平均/L2 范数） | 373 |
| `benchmarks/convergence_study.py` | 验收驱动：8 组 h 研究 + 3 组 p 加密 + 1 组 CN 时间收敛；输出 `metrics.json` / `h_studies.csv` / 4 张图 / 程序化渲染的 `summary.md`；26 条机器判据 | 499 |
| `tests/test_verification.py` | 12 条回归测试 | 235 |
| `docs/tutorials/convergence-verification.md` | 中文方法教程（公式、判据、局限、三态结论） | — |
| `docs/benchmarks/convergence-verification.md` | 英文 benchmark 规格页 | — |
| `docs/benchmarks/results/convergence/` | 生成产物：`metrics.json`、`summary.md`、`h_studies.csv`、`h_convergence.png`、`order_vs_theory.png`、`gci_check.png`、`p_convergence.png` | — |
| `README.md`、`tensordg/__init__.py` | 新增 "Convergence order and GCI verification" 节与公共导出 | — |

## 3. 判据与实测（数字全部由 `metrics.json` 程序化注入，见 `summary.md`）

判据：`|p_obs − p_theory| ≤ 0.2`；`GCI(fine) ≥ 解析真误差`；渐近比 ∈ [0.8, 1.25]；p 加密单调且超线性。

| study | 理论阶 | 实测阶（LSQ） | 结论 |
|---|---:|---:|---|
| poisson2d_p1 (4,8,16,32) | 2 | 1.998 | 已证实 |
| poisson2d_p2 (2,4,8,16) | 3 | 2.958 | 已证实 |
| poisson2d_p3 (2,4,8) | 4 | 3.958 | 已证实 |
| advection1d_p1 (20,40,80) | 2 | 1.997 | 已证实 |
| advection1d_p2 (10,20,40) | 3 | 3.000 | 已证实 |
| heat1d_p1 (8,16,32), dt 固定 1000 步 | 2 | 2.001 | 已证实 |
| heat1d_p2 (8,16,32), dt 固定 1000 步 | 3 | 2.996 | 已证实 |
| poisson2d_p1_matrixfree (4,8,12) | 2 | 1.995 | 已证实 |

时间方向：固定 32 cells、dt 加密 (250,500,1000,2000 步)，CN 实测阶 2.051（理论 2）。
p 加密：三组算例误差逐级下降且每步下降 30–85 倍（超线性）。

机器判决：**25 PASS / 1 NOT_APPLICABLE / 0 FAIL**（26 条）。

## 4. 诚实记录

- **NOT_APPLICABLE（1 条）**：`heat1d_p2` 的 GCI 覆盖判据。h 研究固定 dt，CN 时间误差构成地板，h-GCI 只有真误差的约 3%。这不是放宽判据：驱动把该条标为 `NOT_APPLICABLE` 并附原因，另用 dt 加密研究量化地板（时间阶 2.05）。GCI 只估计"被加密参数"对应的误差分量，这是方法边界。
- **QoI 超收敛**：Poisson 点值 QoI 实测阶约 2p（P1≈2、P2≈4、P3≈6），高于 L2 阶；因此"是否达理论阶"只用 L2 范数判，QoI 仅走 GCI 流程。
- **未验证范围**：GPU、float32、非结构网格、p>3、三维；Euler 激波算例（Sod/Lax/涡）不做收敛阶断言（间断解无经典阶），其验收仍走精确 Riemann 参考解误差门。

## 5. 验证

```
# 新增回归（12 条）
OMP_NUM_THREADS=2 python -m pytest -q tests/test_verification.py
12 passed in 18.73s

# 既有套件基线（不含本单新增文件，34 条）
OMP_NUM_THREADS=1 python -m pytest -q --ignore=tests/test_verification.py
34 passed in 175.14s

# 全量套件（46 条）
OMP_NUM_THREADS=1 python -m pytest -q
（见本单 finish 中的实测记录）

# 验收驱动（26 条机器判据）
python -m benchmarks.convergence_study
overall: PASS
```

## 6. 复现

```bash
python -m pip install -e ".[dev]"
python -m benchmarks.convergence_study          # 写 docs/benchmarks/results/convergence
pytest -q tests/test_verification.py
```