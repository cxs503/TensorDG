# 交差核验：m16075 CASE-DG-KOVASZNAY-NS（TensorDG）— board audit m16174

审对象：mission 16075，succeeded，分支 `agent/m16075`，matured commit `c24426f`
（2026-11-12 10:21，作者 tp-pusher，含 8 个交付文件 + 12 个工件 JSON）。
审计者：board-audit 席位（m16174），**非重做**，只核验与独立复算。审于 2026-11-12 10:14–10:33 UTC。

判定：**L4（基准验证）/ 处置：approve with notes**。门禁量（速度相对 L2 误差
8.818e-05 = 0.008818 % vs 判据 <1 %，余量 113×）经独立复算与干净副本复跑双重确认，
**逐位重合**。不阻塞的缺陷 2 项、证据缺口 3 项、平台登记缺口 1 项，见 §4。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

Kovasznay (1948) 的定常不可压 Navier–Stokes 二维精确解——均匀来流叠加衰减涡列，
是二维定常 NS 唯一已知的闭式非平行解：

```
lam = (Re - sqrt(Re^2 + 4 k^2)) / 2,   u = 1 - e^{lam x} cos(k y),
v = (lam/k) e^{lam x} sin(k y),        p = p_ref - e^{2 lam x}/2,  rho = 1
```

| 项 | 值 | 来源 |
|---|---|---|
| 域 | x∈[-0.5,1], y∈[-0.5,1.5]（y 向 2 个周期）| `tensordg/kovasznay.py::default_domain` |
| Re / ν | 40 / 0.025 | 算例卡 |
| 涡波数 k | 6.283185 (=2π)，λ=-0.9637405 | `decay_rate()` |
| 离散 | 张量积模态 Legendre DG，P3，24×24（36864 自由度）| `ns_dg_2d` HLLC + BR1/SIPG + SSP-RK3 |
| 低马赫嵌入 | ρ=ρ0=1，c=60 ⇒ M=0.0437，p=p0+p_dynp0=c²/γ=2571.43）| `kovasznay_thermodynamic_state` |
| 边界 | 流向两端给定状态（HLLC 通流面）+ Dirichlet 黏性罚；y 向周期 | `ns_dg_2d._boundary_flux` 的 `"state"` 分支（本单新增）|
| 推进/步数 | t=0.08，8 个重启块，2592 步 | `validation/kovasznay/primary.json` |

判据（**预先声明**，取自 mission goal 原文）：「容差判据：<1 %；指标：速度 L2 相对误差
vs 精确解 (Re=40, k=3, 网格 24x24)」。

## ② 怎么做的（审计流程 / 工具链）

1. **取证**：mission 16075 的 goal/result、`tensor_reports` 中 id=15231/15244 两份报告、
   工作树 `/root/tp-agent/work/m16075` 与 matured commit `c24426f` 的文件清单。
2. **基准裁判查表**：`psql -U postgres -d tensorplatform "SELECT ... FROM
   tensor_benchmark_cases WHERE anchor_slug='CASE-DG-KOVASZNAY-NS'"` → **0 行**（§4 F3），
   故容差以 mission goal 原文为准。
3. **三件套核验**：run_artifacts / ref-vs-got / repro_cmd 逐项对照；`git ls-tree HEAD
   validation/kovasznay` 确认 11 个工件已入库。
4. **独立复算**（`runs/m16174/recheck_kovasznay.py`，不调用 case 模块的误差度量完成 A/B/C）：
   - A 用原始节点场 `fields_primary.npz` + 自写 Gauss–Legendre 积分（w/2、cell Jacobian）
     重算相对 L2；
   - B 用中心差分（h=1e-3，400 随机点）验证闭式解满足三条定常 NS 方程；
   - C 复算衰减率特征关系 λ²−Reλ−k²=0；
   - D 把精确解 L2 投影到 DG 空间（各网格/阶）算「纯表示误差地板」，与实测误差对照；
   - E 逐 run 的 JSON 数值 vs 文档表格 / 「未跑」标注一致性。
5. **干净副本复跑**：把交付代码拷到 `/tmp/kovaudit`（不污染证据目录），
   `KOVASZNAY_THREADS=4 PYTHONPATH=. python -m benchmarks.verify_kovasznay_dg_2d --job primary`
   （636 s）与 `--job summary`（图 + 汇总）。
6. **测试**：`tests/test_kovasznay_dg_2d.py`、`tests/test_ns_dg_2d.py` 实跑。
7. **缺陷复现**：`--job wavenumber3` 在干净副本中复现崩溃（§4 F2）。

## ③ 计算结果（数字表，全部程序化取数）

### 3.1 门禁量：ref vs got（唯一裁判项）

| 项 | ref | got | 门限 | 判定 |
|---|---|---|---|---|
| 速度矢量相对 L2（P3/24²/Re=40/c=60）| Kovasznay 闭式解（同一求积节点求值）| **8.818188869671156e-05**（0.008818 %）| <1 % | **PASS**，余量 113× |
| u 分量 | 同上 | 8.452181340142625e-05 | — | — |
| v 分量 | 同上 | 3.1207594975243293e-04 | — | — |
| 定常残差（模态 2-范数比）| 0 | 9.9308e-05 | 未声明 | 未收敛到机器零（声学振荡，文档已述）|

**独立复算**：从 `fields_primary.npz` 原始节点场用自写求积得
`velocity_rel_l2 = 8.818188869671157e-05`（与 artifact 相对差 <1e-15）；
**干净副本复跑**：`vec=8.818188869671156e-05, steps=2592, residual=9.930843041485108e-05`
→ 与原始 artifact **逐位相同**（残差相对差 1.4e-16）。ref 自身：FD 残差连续性 1.0e-5、
x/y 动量 1.5e-5/1.8e-6（h=1e-3，round-off/截断量级），λ 特征关系残差 8.5e-14。

### 3.2 网格/阶数阶梯：实测误差 vs 纯表示误差地板（审计侧新增）

| 配置 | 实测相对 L2 | 投影地板（L2 projection）| 实测/地板 | 观测阶（文档）| 理论阶 |
|---|---|---|---|---|
| P1/12² | 2.4296e-02 | 2.2254e-02 | 1.09 | 2.08 | 2 |
| P2/12² | 2.9679e-03 | 1.9733e-03 | 1.50 | 2.92 | 3 |
| P3/12² | 2.2829e-04 | 1.3038e-04 | 1.75 | — | 4 |
| P1/24² | 5.7384e-03 | —（未算）| — | — | — |
| P2/24² | 3.9141e-04 | 2.4885e-04 | 1.57 | — | — |
| **P3/24²（门禁）** | **8.8182e-05** | 8.2112e-06 | **10.7** | — | — |

结论：实测误差一致地高于纯插值地板（1.1–10.7×），门禁配置上高出 10.7×，说明本算例
检验的不只是「把解析解投影进 DG 空间」；但同时也说明 P3/24² 的误差不是由 DG 插值精度
主导（§4 F6 的措辞修正）。

### 3.3 敏感性工件（存在，但文档未回填——见 F1）

| run | 相对 L2 | 判定 | 步数 | 说明 |
|---|---|---|---|---|
| mach030（c=30，M=0.0873）| 3.4870e-04（0.0349 %）| PASS | 6470 | / 门禁 = 3.954，M² 预期 4.0 → 支持「误差地板 ∝ M²」|
| bc_extrap（出口外推）| **3.3710e+00（337 %）** | **FAIL** | 6407 | 残差 2.56，解被出口污染——诚实负结果，文档标「未跑」|
| wavenumber3（k=3 变体）| — | 崩溃 | — | F2 |

### 3.4 报告六要素齐全性

①物理/几何/参数表 ✓；②方法/工具链/一行复现 ✓；③数字表（程序化注入，抽查 14 个数
与 JSON 完全一致，含 primary history 8 行）✓；④图 3 张（场量云图 vs 解析解、误差/残差
vs t、阶梯+马赫）✓；⑤注意点 5 条（声学振荡不归零、M² 地板、k 的两读法、无文献逐位比对、
单机单线程）✓；⑥误差判定显式给出「是否 <3 %」的同类回答（<1 % 判据 → 0.0088 %，通过）✓。
`tensor_reports` id=15231/15244 已发布（qa_status=pending）。

## ④ 注意点（审计发现，按严重度排序）

**F1（中，报告与工件不同步）** 文档与已发布报告把 `mach030`、`bc_extrap` 标为「未跑」，
但两个工件 `validation/kovasznay/mach030.json`（10:04 产出）、`bc_extrap.json`（09:59 产出）
**就在同一个 matured commit c24426f 里**。文档由 `tools/gen_case_kovasznay_report.py`
在 09:58 前生成后未重跑。后果：a) 表格与入库工件自相矛盾；b) 一条诚实的 FAIL
（外推出口 337 % 误差）被漏报，而它恰好是「为何必须用给定状态出口」的关键证据。
不影响门禁结论（方向是少报不是虚标）。

**F2（中，可复现缺陷）** `RUNS["wavenumber3"]` 永远跑不起来：
`_case()` 已传 `wavenumber=wavenumber`，`extra={"wavenumber": 3.0}` 再展开一次 →
`TypeError: solve_kovasznay_dg_2d() got multiple values for keyword argument 'wavenumber'`。
在干净副本中复现（`runs/m16174/defect_wavenumber3.log`）。因此 §① 声称的「k 的两种读法
都给了」实际只有 DG 阶数读法有数；涡波数 k=3 读法**没有**数据支撑。

**F3（中，平台侧登记缺口）** `tensor_benchmark_cases` 中该锚点 **0 行**
（同表已有 16–26 号其他 CASE-* 行），容差 <1 % 只存在于 mission goal。
按「基准判据是唯一裁判」的口径，本锚点缺裁判行 → 机器无法独立复核容差/指标。
建议补登记：anchor_slug=CASE-DG-KOVASZNAY-NS, metric=速度相对 L2, tolerance=<1%, repo=TensorDG。

**F4（低，证据缺口）** 初场无关性（`ic_uniform`/`ic_exact_ref`）未跑：`scratch/job_ic_uniform.log`
为空，文档如实写「未跑」。故本算例证明的是**离散定常算子的一致性/稳态保持**（初场=精确解），
不是「任意初场收敛到该定常解」。属标准做法但须在级别说明里写明。

**F5（低，口径）** `_order_table()` 用 `log(e1/e2)/log(h1/h2)`（h 递增）→ 观测阶显示为
−2.08/−2.92（量值正确、符号为负），文档与图例直接沿用负数。建议取绝对值或统一定义。

**F6（低，措辞过严/过宽）** 文档 §⑥ 局限 2 说「P3/24² 的 DG 离散误差（~1e-5）远低于
O(M²) 下限」。审计实测：纯投影地板 8.2e-6（确为 ~1e-5 量级），但**行进的稳态误差是
8.8e-5（10.7×地板）**；马赫扫描比 3.954 ≈ 4 支持「误差由 M² 嵌入地板主导」的结论，
但把「离散误差」说成 1e-5 会低估实际误差一个量级。建议改写为「投影地板 8.2e-6、
实测 8.8e-5、M² 标度由 c=30 运行确认」。

## ⑤ 未做的检查（明示为 not_run，不冒充通过）

- `mach120`、`bc_adjoint`（墙面对称伴随）未跑；`bc_extrap` 有工件但未进文档。
- `ic_uniform`/`ic_exact_ref` 未跑（F4）。
- 未上集群：全部在控制面单机跑（primary 636 s/4 线程，与原 597 s/1 线程一致为开销主导）。
- 未与文献数值逐位比对：该配置下文献无「速度 L2 误差应为多少」的表，ref 只能是闭式解
  （闭式解已由 FD 残差独立校核）。
- 独立性局限：A/D/E 的积分/投影为新写代码，但被评的**求解器与参考解实现**是原交付代码，
  审计未独立重写 DG-NS 求解器；复跑只证明可复现，不证明求解器物理正确性（后者由
  ref 残差 + 阶数阶梯 + M² 标度三面交叉支持）。

## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）

| 关键量 | 基准 | 误差 | 是否 <3 % |
|---|---|---|---|
| 门禁：速度相对 L2（P3/24²）| Kovasznay 闭式解 | 0.008818 % | **是**（余量 113×）|
| 门禁：u 分量 | 闭式解 | 0.008452 % | 是 |
| 门禁：v 分量 | 闭式解 | 0.031208 % | 是 |
| P2/24² 速度 | 闭式解 | 0.039141 % | 是 |
| P1/24² 速度 | 闭式解 | 0.573840 % | 是 |
| —— 以下为诚实负结果/不达标项 —— |
| P1/12² 速度 | 闭式解 | 2.4296 % | 是（<3 %）但**不达 <1 % 门禁** |
| bc_extrap 速度（出口外推）| 闭式解 | 337.1 % | **否**（FAIL，未报入文档）|
| 观测收敛阶 P1 / P2 | 理论 2 / 3 | 4 % / 2.7 % 偏差 | 是 |

## ⑦ 结论与产物

- **级别 L4**：真代码（输入→DG 离散→定常推进→误差度量→JSON/图）+ 接入既有 `ns_dg_2d`
  主求解器（`_rhs_ns`→`_boundary_flux`，向后兼容 6 项测试通过）+ 完整算例/误差分析/
  可复现（干净副本逐位重合）+ 外部锚（Kovasznay 1948）。
- **处置**：approve with notes。F1/F2 建议返修单（重生成文档含 mach030/bc_extrap、修
  `wavenumber3` 重复关键字）；F3 建议平台补评测登记行。均不改变门禁 PASS 的事实。
- 审计工件：
  - `runs/m16174/recheck_kovasznay.py` / `recheck_kovasznay.json`（A–E 五组独立复算）
  - `runs/m16174/repro_primary_compare.json`（原始 vs 干净副本复跑，逐位比较）
  - `runs/m16174/repro_summary_job.log`（`--job summary` 可执行，生成 3 图）
  - `runs/m16174/pytest_kovasznay.log`（8 passed / 82.95 s）、`pytest_ns_dg_backcompat.log`（6 passed / 53.92 s）
  - `runs/m16174/defect_wavenumber3.log`（F2 崩溃复现）
  - `docs/board_audit/m16174/figures/kovasznay_audit_ladder.png`（实测 vs 投影地板）
- 报告图说明：本审计为核验任务，无新物理场；图 1 用原始 `fields_primary.npz`（场量）
  已由被审方提供，审计侧新增图为「实测误差 vs 纯表示地板 + 实测/地板比值」。