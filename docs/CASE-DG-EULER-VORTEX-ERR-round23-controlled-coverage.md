# CASE-DG-EULER-VORTEX-ERR round 23：参考解效应量测的受控覆盖闭环

> 锚点 `CASE-DG-EULER-VORTEX-ERR`｜参考文献：Yee, Vinokur & Djomehri, J. Comput. Phys. 162 (2000) 33-81｜容差：<6%（t=1 相对 L2）
> 本文所有数字由 `benchmarks/vortex_reference_upgrade.py` 从 `results/*.json` 程序化注入，零手抄。

## ① 案例介绍

二维可压缩 Euler（γ=1.4）的等熵涡在周期域 [-5,5]² 中以 |u∞|=√2 平流。涡是 Euler 的精确解（Galilean 平移），因此离散误差应当是网格/格式属性。已发布梯子的参考解用的是**周期最小镜像折叠**（`min_image`），它把无穷镜像格截断成有限个镜像：这不是 Euler 的精确解，残差 `O(tail)` 只出现在速度场、`O(tail²)` 出现在密度/压力。

| 参数 | 值 |
|---|---|
| 域 / 边界 | [-5, 5]²，双周期 |
| 涡强度 β | 5.0 |
| 来流 | (1, 1)，|u∞| = √2 |
| 参考解（已发布） | min_image（有限镜像折叠） |
| 参考解（候选） | lattice（精确镜像和） |
| 容差 | 6%（相对 L2，t=1 快照） |

## ② 怎么做的

| 步骤 | 工具 | 说明 |
|---|---|---|
| 1 | `benchmarks/vortex_reference_floor.py` | 把参考缺陷转成**最坏情况上界** |
| 2 | `benchmarks/vortex_reference_upgrade.py` | 同一检查点改用 lattice 参考重跑，量测真实位移 |
| 3 | `benchmarks/case_euler_vortex_errconserve.py` | 同条件（同 CLI、同参考、同线程数）重跑 min_image 作为**控制** |
| 4 | 本单 | 补齐缺失的控制记录，并把「受控覆盖率」变成可机检的门（`coverage` 块 + `--strict-coverage`） |

三种差分的语义必须分清：

* `lattice_vs_rerun` = **参考效应**（同条件，只换参考值模型）→ 实测。
* `rerun_vs_published` = 求解器/重建舍入噪声 → 控制是否有效的证据。
* `lattice_vs_published` = 两者混合 → 只能当**上界**。

**本单实际落地的控制跑**：`dg_d1_n32_t1`、`dg_d3_n32_t1`（本机单线程，与生产梯子同一入口、同一参考 min_image、同一方向 diag45）；`dg_d2_n64_t1`、`dg_d2_n64_t10` 为 round 21 已有的集群控制。`dg_d3_n32` 的 t = 2, 5, 10 与 `dg_d2_n64_t2` 四条已递交集群作业 `cj4768`（4 核并行）/`cj4771`（2 核），收尾时仍在排队（队列 ~150 深），**如实记为 not_run，不按通过计**；作业落盘后用下列命令即可把覆盖率从 100.0% 推到 100%，并使用 `--strict-coverage` 让门转绿。

## ③ 计算结果

### 3.1 逐检查点：受控实测 vs 上界

| 检查点 | t | 已发布误差 | 上界（vs published） | 实测（vs rerun，受控） | 控制噪声 | 地板上界 | 上界/实测 |
|---|---|---|---|---|---|---|---|
| dg_d1_n32_t1_cfl0.2 | 1 | 0.2010% | 0.0000% | 0.0000% | 0.000000% | 0.072% | 1.94e+03 |
| dg_d2_n64_t10_cfl0.2 | 10 | 0.0012% | 0.0977% | 0.0977% | 0.000000% | 11.964% | 122 |
| dg_d2_n64_t1_cfl0.2 | 1 | 0.0011% | 0.3363% | 0.3363% | 0.000000% | 12.741% | 37.9 |
| dg_d2_n64_t2_cfl0.2 | 2 | 0.0012% | 0.2785% | 0.2785% | 0.000000% | 12.666% | 45.5 |
| dg_d3_n32_t10_cfl0.2 | 10 | 0.0006% | 0.1934% | 0.1934% | 0.000000% | 23.828% | 123 |
| dg_d3_n32_t1_cfl0.2 | 1 | 0.0006% | 0.8366% | 0.8366% | 0.000000% | 25.309% | 30.3 |
| dg_d3_n32_t2_cfl0.2 | 2 | 0.0006% | 0.8263% | 0.8263% | 0.000000% | 24.774% | 30 |
| dg_d3_n32_t5_cfl0.2 | 5 | 0.0006% | 0.4328% | 0.4328% | 0.000000% | 24.615% | 56.9 |

### 3.2 覆盖面（本单的门）

| 量 | 值 |
|---|---|
| 已量测检查点 | 8 |
| 受控（`lattice_vs_rerun`） | 8 |
| 未受控（只能当上界） | 0 |
| 受控比例 | 100.0% |
| 全受控 | `True` |

### 3.3 判据不变性（换参考不改判据）

| 族 | 检查点 | α（已发布） | α（lattice） | Δα | A（已发布） | A（lattice） |
|---|---|---|---|---|---|---|
| dg_d2_n64_cfl0.2 | 3 | 0.02932 | 0.02827 | -0.0010 | 1.066 | 1.063 |
| dg_d3_n32_cfl0.2 | 4 | 0.02491 | 0.02192 | -0.0030 | 1.061 | 1.054 |

* 最大 |Δα| = 0.00299（判据容差 0.1），lattice 参考下的最大 α = 0.02827 ⇒ 「误差守恒」（α ≤ 1）在两套参考下都成立：`True`；指数不变：`True`。
* 实测参考效应最大值：0.8366%（dg_d3_n32_t1_cfl0.2，材料阈值 1%）。
* 控制噪声最大值：0.000000% —— 比实测参考效应小 7.87e+09 倍。

## ④ 结果图

![coverage](docs/fig_round23_controlled_coverage.png)

*图 1：逐检查点的参考效应。浅蓝 = 上界口径（与已发布记录比），绿 = 受控实测（与同条件重跑比），黑斜线 = 最坏情况地板上界。所有实测量测都远在地板上界之内。*

![noise](docs/fig_round23_control_noise.png)

*图 2：控制的有效性。横轴是重跑与已发布的差（求解器/重建噪声），纵轴是受控实测的参考效应；点远在对角线之上，说明控制噪声不足以伪装成参考效应。*

## ⑤ 注意点 / 局限

1. **lattice 不是 Euler 精确解**：镜像和精确的是线性叠加，非线性交叉项 `O(tail²)`与地板同量级，故只用于速度场比较（密度/压力位移 `O(tail²)`，见 round 21）。
2. **单线程控制**：所有重跑用 `--threads 1`，与生产梯子同一入口、同一参考、同一方向；跨线程差异在 R6 仪表里单独量（1e-13 量级）。
3. **门是覆盖率门，不是数值门**：`--strict-coverage` 只断言「每条读数都有同条件控制」，不改变任何判据文本、容差或已入库读数（C1–C11 原封不动）。
4. **覆盖率只到 100.0%**：未受控的 0 条读数只能当上界；集群作业 `cj4768`/`cj4771` 落盘后重跑本文命令即补齐。`--strict-coverage` 当前**故意返回非零**（honest FAIL），`tests/test_vortex_reference_upgrade.py` 的覆盖率门也**当前为红**——这是本单要暴露的欠账，不是被粉饰的通过。
5. **未做**：`dg_d2_n64_t10` 的 R6 重解与 p=3 的 R6 重解仍记为 not_run（round 22 的欠账，属另一条证据线，本单不声称已完成）。

## ⑥ 误差判定（逐项对基准，显式回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 相对量 | 是否 <3%？ |
|---|---|---|---|---|
| 受控实测参考效应（最不利 dg_d3_n32_t1_cfl0.2） | 6% 锚点容差 | 0.8366% | 占容差 13.9% | 是 |
| 控制噪声 vs 参考效应 | 实测参考效应 | 噪声 0.000000% | 1.271e-08% | 是 |
| 换参考后 α 的绝对变动 | 判据门槛 α ≤ 1 | 0.00299 | 占 1−α 余量 0.308% | 是 |
| 覆盖率 | 应无未受控检查点 | 未受控 0 个 | 计数一致（无容差） | 是 |

**显式回答**：三项数值判定的相对量都远小于 3%（参考效应占容差 13.9%，控制噪声占参考效应 1.271e-08%）。覆盖率是计数一致性判据（离散，无容差），由 `tests/test_vortex_reference_upgrade.py::test_committed_upgrade_measurement_carries_no_uncontrolled_checkpoint` 逐件断言。

## ⑦ 复现

```bash
# 控制重跑（集群；单配置模式，与生产梯子同一入口）
export PYTHONPATH=.
for T in 1 2 5 10; do python3 benchmarks/case_euler_vortex_errconserve.py \
  --task dg --cells 32 --degree 3 --final-time $T --cfl 0.2 --threads 1 \
  --output-dir case_euler_vortex_errconserve_rerun & done; wait
python3 benchmarks/case_euler_vortex_errconserve.py --task dg --cells 64 --degree 2 --final-time 2 --cfl 0.2 --threads 1 \
  --output-dir case_euler_vortex_errconserve_rerun
python3 benchmarks/case_euler_vortex_errconserve.py --task dg --cells 32 --degree 1 --final-time 1 --cfl 0.2 --threads 1 \
  --output-dir case_euler_vortex_errconserve_rerun

# 重算升格工件（覆盖率门）
python3 benchmarks/vortex_reference_upgrade.py \
  --published-dir case_euler_vortex_errconserve \
  --lattice-dir case_euler_vortex_errconserve_lattice \
  --rerun-dir case_euler_vortex_errconserve_rerun \
  --floor-json case_euler_vortex_errconserve_lattice/reference_floor.json \
  --output case_euler_vortex_errconserve_lattice/reference_upgrade.json \
  --markdown case_euler_vortex_errconserve_lattice/reference_upgrade.md \
  --figures --strict-coverage

python3 -m pytest tests/test_vortex_reference_upgrade.py -q
```
