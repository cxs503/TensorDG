# REPRO-DG-CKS1998 · 2-D Riemann 算例的守恒判据重订 + 发散发现（honest FAIL）

- **锚点**：`REPRO-DG-CKS1998`（Cockburn & Shu, JCP 141 (1998) 199-224, Part V；四象限算例源自
  Lax & Liu, SIAM J. Sci. Comput. 19(2) 1998, configuration 3）
- **上一单留下的两条诚实 FAIL**（`docs/CASE-DG-CKS1998-2D-RIEMANN.md`）：
  1. 质量变化 25.16 % ≫ 1 % 门，归因为「TVB 限幅器的模态缩放不严格守恒」；
  2. 自收敛 `not_run`（`run_convergence_study` 用「两个网格总质量之差」当自收敛度量）。
- **本单结论（三句话）**：
  1. **归因是错的**：限幅器与正性过滤器**都不动单元均值**，格式在周期域上把质量/动量/能量
     守恒到 **6.1e-15**；在透射边界下 `Q(t)-Q(0) == ∫边界通量` 闭合到 **6.4e-15**。
  2. 因此旧门的「25 % 质量变化」既不是限幅器缺陷、也不是判据可用的守恒度量，而是**透射边界的
     物理通量 + 后面这条**：
  3. **新发现的真缺陷（honest FAIL，未粉饰）**：算例声明的设置（`final_time=0.25, cfl=0.2`）下
     解**发散** —— n=16/32 总质量跑到 2.2e7 / 4.9e26，n=64 直接
     `FloatingPointError: non-finite 2-D DG state`。旧文档的 25.16 %（t=0.1）是该增长过程的早期读数。
     守恒恒等式在发散的解上**依然闭合到 1e-15**，正说明"守恒"与"解是否有意义"是两件事。

## 1. 缺陷一：守恒判据把「边界通量」当「守恒误差」

CKS1998/Lax–Liu 的四象限算例是**全场四个片段的初值问题**，标准做法在单位方形的四条边上用
透射（零梯度）边界 —— `solve_euler_dg_2d_riemann` 里就是 `periodic=(False, False)`。
而旧门 `C4 mass conservation < 1 %` 的注释写的是 "periodic BCs"。判据与执行路径不一致：

| | 旧门声明 | 旧门实际执行 |
|---|---|---|
| 边界 | periodic | transmissive (`periodic=(False,False)`) |
| 应测的量 | 守恒误差 | 物理流出/流入 + 任何格式误差（不可区分） |

**正确的守恒判据（对任何边界条件都成立）**：弱形式 DG 的单元均值变化只经由界面通量差分，
内部界面在求和时逐项相消，于是

```
Σ_cells mean(t) - Σ_cells mean(0)  ==  ∫_0^t Σ_boundary_faces F·n ds dτ
```

左侧 = 域积分的变化，右侧 = 时间积分的边界通量。`solve_euler_dg_2d(..., track_boundary_flux=True)`
用 SSP-RK3 的同一组权重 (1/6, 1/6, 2/3) 累加该积分（`_rhs(ledger=...)`），残差即为判据。
**只有**当限幅器 / 正性过滤器改动了单元均值时残差才会非零 —— 这就是这条恒等式的诊断力。

### 1.1 实测（程序化注入，见 `artifacts/stable_window/metrics.json` + 集群 cj2915 日志）

| 场景 | n | Δt 设置 | 质量变化 | 边界通量积分 | 残差/尺度 | 判定 |
|---|---:|---|---:|---:|---:|---|
| 周期控制（无边界） | 16 | t=0.05, CFL=0.1 | −1.1e-15 | 0（不可能有） | **6.1e-15** | **PASS** |
| 透射（算例真身） | 8 | t=0.05, CFL=0.1 | +4.300e-02 | +4.300e-02 | **6.4e-15** | **PASS** |
| 透射（算例真身） | 16 | t=0.05, CFL=0.1 | +4.626e-02 | +4.626e-02 | **6.4e-15** | **PASS** |
| 透射 | 16 | t=0.25, CFL=0.2 | **+2.209e+07** | +2.209e+07 | 1.25e-15 | 恒等式闭合，但**解已发散** |
| 透射 | 32 | t=0.25, CFL=0.2 | **+4.846e+26** | +4.846e+26 | 5.80e-16 | 同上 |
| 透射 | 64 | t=0.25, CFL=0.2 | — | — | — | **作业失败**：`non-finite 2-D DG state` |

四分量（质量/x 动量/y 动量/能量）的残差全部 ≤6.4e-15；`x/y` 动量在对称初值下相等到 1e-15，
是一条独立的自洽性旁证。

**被否定假设（写进账，不惩罚）**：「TVB 限幅器的模态缩放不严格守恒」。
直接单测：对随机模态状态 + 人为跳变，`apply_tvb_limiter_2d` 后 mode (0,0) **逐位不变**
（`tests/test_riemann_2d_conservation.py::test_tvb_limiter_preserves_cell_averages`）；
`_positivity_filter` 的 `c <- mean + θ(c-mean)` 形式同样**逐位保留均值**
（`...::test_positivity_filter_preserves_cell_averages_and_restores_admissibility`）。

## 2. 缺陷二：自收敛度量测的是标量

旧 `run_convergence_study` 里

```python
mass_diff = abs(fine["total_mass"] - coarse["total_mass"])   # 当成 L1 自收敛
```

总质量对固定末时刻是**网格无关的标量**（守恒格式下更是同一个数），它衡量不了收敛性；
而且它把"守恒"和"收敛"两件事混在一起。替换为真正的度量：
细网格单元均值按 `factor×factor` 块平均降阶后与粗网格比较相对 L1：

`self_convergence_l1 = Σ|m_coarse − ⟨m_fine⟩_block| / Σ|⟨m_fine⟩_block|`

实测（t=0.05, CFL=0.1, P1）：`8² vs 16² = 0.0274`（< 0.05 门，PASS）。

## 3. 缺陷三（本单新发现，honest FAIL）：声明设置下解发散

| n | 步数 | 总质量变化 | rho_min（Gauss 点） | p_min（Gauss 点） |
|---:|---:|---:|---:|---:|
| 16 | 540 | +2.209e+07 | 0.1305 | 0.1757 |
| 32 | 1810 | +4.846e+26 | 0.1299 | 0.1748 |
| 64 | — | — | — | `FloatingPointError` |

三点观察，**不超额断言**：

1. 质量变化随末时刻**单调爆炸**：t=0.05 → 4.6e-2（≈6.8 % 初质量，物理通量量级）；
   t=0.1 → 25 %（旧文档读数）；t=0.25 → 1e7…1e26。这是**不稳定增长**的特征，不是物理流出。
2. 与此同时 `rho_min/p_min` 仍 >0、四分量恒等式仍闭合到 1e-15 ——
   **现有 C1/C4/C5 三条门都抓不到发散**。这是本单暴露的判据覆盖缺口（见 §5 待办 C6）。
3. 根因**未在本单内隔离**。候选：(i) `CFL=0.2` 对 2-D P1 DG + SSP-RK3 已越过稳定界；
   (ii) TVB 指示器 M=1 释放激波单元后多项式鼓出（旧文档 §5 的机制）。判据上要区分两者需要
   CFL×M 的二维扫描，属下一单范围；本文件不给未测过的结论。

## 4. 代码改动（最小、可验证、默认行为位不变）

| 文件 | 改动 | 向后兼容性证明 |
|---|---|---|
| `tensordg/euler_dg_2d.py` | `_rhs(..., ledger=None)`：把边界界面通量投影到均值模 (k=l=0) 并求和；`solve_euler_dg_2d(..., track_boundary_flux=False)`：按 SSP-RK3 权重累加，metadata 新增 `boundary_flux` / `conservation_residual` | `test_boundary_flux_tracking_does_not_change_the_solution` 断言开/关该开关的**末态系数逐位相等**；`ledger=None` 时 `_rhs` 路径不变 |
| `benchmarks/riemann_2d_cks1998.py` | 门重订：C4 = 通量平衡残差 <1e-10，C5 = 周期控制漂移 <1e-12，C2 = 真自收敛 <0.05；`judge()` 变成纯函数，`NOT_RUN` 不再计 PASS（旧代码里 `verdicts["C4_mass_conservation"] = True  # single run, no reference` 是硬编码 PASS）；物理质量变化照旧打印但不再冒充守恒 | 新的 `judge` 敏感性由 `test_judge_fails_on_a_broken_flux_balance` / `test_judge_fails_on_negative_pressure` / `test_judge_reports_missing_checks_as_not_run_not_pass` 覆盖 |
| `validation/riemann2d_conservation/audit.py`（新） | 守恒/正性/自收敛三件套的审计驱动，产出 `metrics.json` + 3 张图 | — |
| `tests/test_riemann_2d_conservation.py`（新，10 条） | 上述所有断言 | 本地 10 passed；集群 cj2926 复跑 |

**未改**：任何容差、任何数值内核、`solve_euler_dg_2d_riemann` 的边界设置（它用的是本算例的
标准边界，改它才是错的）。

## 5. 注意点 / 局限 / 待办（诚实清单）

1. **判据缺口 C6（建议新增）**：`|质量(t)|` 应落在 `mass(0)` 的同一量级内（例如
   `[0.2, 5]×mass0`）才能说明解还有物理意义。本单**没有**加，因为阈值必须先预登记、
   且要以 CFL/M 扫描为基线；已记录为下一单的首选。
2. **发散根因未隔离**（见 §3.3）；本单只给出"恒等式闭合但解非物理"的分层证据。
3. **P2 档未取到数**：cj2916（P2 32²/64², t=0.25）在收尾时仍在排队，本文件不含其数字，
   不以占位值充数。
4. **集群取证踩坑（对后续单成立）**：`run_compute fetch` 的 tar 只含**仓内已跟踪**路径，
   本单新目录 `validation/riemann2d_conservation/artifacts/` 未被回收；集群数字取自
   `/root/tp-agent/cluster/jobs/cj2915.log` 的 `output_tail`，图与本文件表格由本地同一脚本
   同一配置重生（`artifacts/stable_window/`）。
5. **`tests/test_euler_dg_2d_wall_bc.py` 在集群收集期报错**（本单未改该文件；本地
   `--collect-only` 正常），因此回归复跑把该模块排除并记为 **not_run**，不当 PASS。
6. 算例 t>0.1 的结果**不可用于任何文献比对**（解已发散），旧文档里的密度范围
   `[0.13, 8.94]` 之类的读数同样不可用作定量证据。

## 6. 复现

```bash
# 守恒/正性/自收敛审计（稳定窗口，~30 s）
OMP_NUM_THREADS=1 PYTHONPATH=. python validation/riemann2d_conservation/audit.py \
    --meshes 8 16 --periodic-meshes 8 --final-time 0.05 --cfl 0.1 \
    --out validation/riemann2d_conservation/artifacts/stable_window
# 声明设置下的发散复现（集群，~39 min，节点 n315）
OMP_NUM_THREADS=1 PYTHONPATH=. python validation/riemann2d_conservation/audit.py \
    --meshes 16 32 64 --periodic-meshes 16 32 --final-time 0.25 --cfl 0.2 \
    --out validation/riemann2d_conservation/artifacts/p1
# 回归门（10 条 + 既有 2D Euler 门）
OMP_NUM_THREADS=1 PYTHONPATH=. python -m pytest tests/test_riemann_2d_conservation.py -q
```

图：`artifacts/stable_window/density_fields.png`（8²/16² 密度场，稳定窗口）、
`flux_balance_residual.png`（四分量残差 vs 1e-12 门）、`self_convergence.png`（8² vs 16²）。