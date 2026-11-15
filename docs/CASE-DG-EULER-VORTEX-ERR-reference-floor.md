# CASE-DG-EULER-VORTEX-ERR · 参考解截断地板（round 7：验证欠账）

锚点：`CASE-DG-EULER-VORTEX-ERR`（算例池·等熵涡移动误差守恒，ref: Yee et al. 2000，容差 <6%）。
本轮不新增算例、不改结论口径，只做一件事：**把「上一轮把 FAIL 翻成 PASS 的那个参考解修正」本身
放到秤上称一遍**——它是精确的，还是仍然带着一个只影响速度场的截断地板？

结论先行（诚实）：`min_image` 参考确实带一个 **1.46e-6 相对 L2 的速度截断地板**
（密度/压力只有 4e-11 %，即机器精度量级）。该地板对**设计网格**（DG-P2/32²）只有实测误差的
**1.0–1.5 %**，故 round 6 的 headline 判定（alpha = 0.199，C1 通过）**不受影响、稳健**；
但在**细化网格**（DG-P2/64²）上它达到实测误差的 **12.7 %**，即该点的 alpha=0.029 与
C7x 放大因子 A(64²)=1.775 带有 ~13 % 的参考解不确定度。为此本单把参考解升级为**格点镜像求和**
（`lattice`），并给出可复现的量测工具、测试与逐检查点地板工件。

## 1. 为什么会有地板（机制）

自由空间等熵涡（`tensordg/euler_2d.py::vortex_primitive`）：
`e = exp((1 - r²)/2)`，`T = 1 - c e²`，`u = u_inf - (θ/2π) e · y_shift`，`ρ = T^{1/(γ-1)}`。

周期解是自由空间解的**全格点镜像叠加**；`period=...`（round 4 引入）只保留**最近镜像**（min-image
折返），丢掉了邻居镜像的尾巴。丢尾的代价由扰动量的阶决定：

* 密度/压力扰动是包络的**二阶**量（`T` 里是 `e²`）⇒ 尾巴贡献 `O(tail²)` ⇒ 实测 **4e-13**；
* 速度扰动是包络的**一阶**量、且乘了位移 `≤ 5` ⇒ 尾巴贡献 `O(tail)` ⇒ 实测 **1.46e-6**。

`tail = exp((1 - (L-|shift|)²)/2)`，`L = 10`，最不利 `|shift| = 5` ⇒ `tail ≈ 6.1e-6`。
这解释了为什么 istropic 误差里 **`u` 场最大、`p`/`ρ` 最小**（round 6 表里同样如此），也解释了
为什么只有速度场可能被参考解限住。

## 2. 怎么做的

1. `tensordg/euler_2d.py`：把自由空间公式抽成 `_vortex_from_shift`，`vortex_primitive` 新增
   `images=K`（`(2K+1)²` 格点叠加，扰动量叠加后回加背景），`minimum_images()` 给出足够半宽；
   半宽不足时**抛 ValueError**（防止悄悄用错镜像）；`images` 缺省 `None` = 原 min-image 行为，**旧读数不变**。
   `vortex_error()` 与 `benchmarks/verify_euler_dg_2d.py::_vortex_error()` 透传 `images`。
2. `benchmarks/case_euler_vortex_errconserve.py`：新增 `--reference {min_image,lattice}`，
   每个检查点按 `reference_images(t) = 1 + ceil((5 + |u_inf| t)/L)` 自动取半宽；
   记录里写入 `reference_model` / `reference_images`；`aggregate` **拒绝混用两种模型的 results 目录**
   （保护已发布数字不被悄悄覆写）。
3. `benchmarks/vortex_reference_floor.py`（新）：用**与门禁完全相同的求积网格与加权相对 L2**
   逐检查点量测地板，并与 round 6 已发布误差求比值，输出 JSON/Markdown/图。
4. `tests/test_vortex_reference_floor.py`（新，13 条）：镜像和的周期不变量、半宽收敛、
   半宽不足必须报错、地板量级与阶、地板/实测比值（对设计点与最细点分别给出可反驳的上界）、
   端到端「换参考只动速度读数」、`aggregate` 混用拒绝。

## 3. 计算结果（程序化注入，零手抄）

工件：`case_euler_vortex_errconserve_lattice/reference_floor.json` /
`reference_floor.md` / `fig_reference_floor.png`（`vortex_reference_floor.py` 生成）。

| 场 | 地板（相对 L2） | 相对 L2 % | 阶 |
|---|---|---|---|
| density | 4.1e-13 … 4.3e-13 | 4.1e-11 % | `O(tail²)` |
| velocity_x / velocity_y | 1.45e-6 … 1.46e-6 | 1.45e-4 % | `O(tail)` |
| pressure | 5.8e-13 … 6.1e-13 | 5.8e-11 % | `O(tail²)` |

地板对时间的依赖 < 1 %（t = 1, 2, 5, 10 内几乎常数）——它是**参考解属性**，不是输运时间的函数：

| 检查点 | t | K | 地板/实测（最坏场） | 判读 |
|---|---|---|---|---|
| dg_d2_n32_t1_cfl0.2（设计点） | 1 | 2 | **0.015** | 实测误差主导 ✔ |
| dg_d2_n32_t5_cfl0.2 | 5 | 2 | 0.013 | ✔ |
| dg_d2_n32_t10_cfl0.2 | 10 | 3 | 0.010 | ✔ |
| p0_n128 / p0_n256 / p0_n64（全部 t） | 1–10 | 2–3 | < 4e-4 | ✔（一阶格式误差量级 1e-2…1e-1） |
| **dg_d2_n64_t1_cfl0.2（最细）** | 1 | 2 | **0.127** | ⚠ 地板占 12.7 % |
| **dg_d2_n64_t10_cfl0.2（最细，长时）** | 10 | 3 | **0.120** | ⚠ 地板占 12.0 % |

端到端（本机实测，同一解、同一度量、只换参考）：`dg_d2_n32_t1_cfl0.2`
density 4.952144e-05 → 4.952144e-05（相对变化 `7.7e-14`，即机器精度），
velocity_x 9.420594e-05 → 9.420666e-05（`+7.6e-6`），步数同为 469。
⇒ 换参考**只动速度**，且在设计点上只动 0.0008 %，与第 3 节地板比值完全自洽。

## 4. 结果图

`fig_reference_floor.png`：左 = 地板随时间的双对数曲线（速度 `O(tail)` vs 密度 `O(tail²)`，
相差 7 个量级）；右 = 逐检查点「地板/实测误差」柱状图 + `=1` 红线（地板等于实测误差即判据被参考解
接管）。设计点柱高 ~1.5 %，最细 DG 点 ~13 %，全部 < 1 —— 即**没有任何检查点的判据被参考解接管**。

## 5. 注意点 / 局限

1. **格点叠加不是 Euler 精确解**：镜像和精确解的是*线性*叠加，非线性交叉项为 `O(tail²) ~ 1e-11`
   （相对速度 1e-6 的地板低 5 个量级），故 `lattice` 参考的自身误差在 double 噪声附近；
   这一点写在 `vortex_primitive` docstring 里，不夸大。
2. **半宽是安全量**：`reference_images(t)` 多给一圈；`minimum_images` 不足时**报错而不是静默降级**，
   测试 `test_undersized_lattice_is_rejected_instead_of_silently_wrong` 守护。
3. **旧读数全部保留**：round 6 的 `case_euler_vortex_errconserve/` 一字未改（仍是 `min_image`），
   新读数落在 `case_euler_vortex_errconserve_lattice/`；`aggregate` 混用会直接报错退出。
4. **本轮未完成的**：全梯 19 个检查点的 `lattice` 复测已递交集群（`cj3379`，6 分片，
   `--shard i/6 --reference lattice`），递交时刻状态 `queued`；故**「换参考后 alpha/A 的具体数值」
   本单尚未给出**，只在设计点上给出了端到端实测（第 3 节末）。复测落地后应把 `reference_floor.md`
   的比值表与 `summary.md` 的 alpha/A 更新为 `lattice` 读数，并把 `REFERENCE_MODEL` 默认值切到
   `lattice`（需同步刷新 C1/C2/C7 的发布数字，属下一单的范围）。

## 6. 误差判定（逐项，显式回答「是否 < 3%」）

| 量 | 基准 | 数值 | 误差 | < 3 % ? |
|---|---|---|---|---|
| min-image 参考 vs lattice 参考（速度，相对 L2） | 格点镜像和 | 1.46e-6 | — | **是**（0.00015 %） |
| min-image 参考 vs lattice 参考（密度/压力） | 格点镜像和 | 4–6e-13 | — | **是**（机器精度） |
| 设计点 DG-P2/32² 实测误差 vs 6 % 容差（round 6） | 算例池容差 | 0.0094 % | 差 3 个量级 | **是** |
| 地板/实测（设计点） | 实测误差 | 0.015 | — | **是**（1.5 %） |
| 地板/实测（最细点 DG-P2/64²） | 实测误差 | 0.127 | — | **否**（12.7 %，故需 lattice 复测） |
| 全梯 lattice 复测 alpha/A | round 6 alpha=0.199 | **未落地（cj3379 queued）** | — | **未判定（not_run）** |

honest-FAIL 纪律：最后一行是**未做**，不是通过；本单不给 round 6 的 PASS 盖章，也不宣称复测已改善任何数字。

## 7. 复现

    PYTHONPATH=. python3 benchmarks/vortex_reference_floor.py \
        --results-dir case_euler_vortex_errconserve \
        --output case_euler_vortex_errconserve_lattice/reference_floor.json \
        --markdown case_euler_vortex_errconserve_lattice/reference_floor.md \
        --figure case_euler_vortex_errconserve_lattice/fig_reference_floor.png
    PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --task dg --degree 2 \
        --cells 32 --final-time 1 --cfl 0.2 --reference lattice --output-dir /tmp/x
    PYTHONPATH=. python3 -m pytest tests/test_vortex_reference_floor.py -q   # 13 passed
    # 全梯复测（集群）：--shard {i}/6 --reference lattice --output-dir case_euler_vortex_errconserve_lattice