# REPRO-DG-CKS1998 round 2 —— full 档工件补齐 + Sod L1 网格阶梯

> 承接：`docs/repro_cks1998.md`（round 1，m15309）与 `docs/AUDIT_m15345_REPRO-DG-CKS1998.md`（交差审计）。
> 本轮：TensorDG · 分支 `agent/m16814` · 2026-11-12。
> 判据预登记见 §2（**先写判据、后跑**）；全部数字由 `python -m tools.render_cks1998_round2`
> 从工件 JSON 渲染到 `docs/CKS1998-round2-results.md`，本文件不手抄任何数值。

---

## 0. 为什么要做这一轮（round-1 的两个未闭项）

交差审计（m15345）把两件事列为**阻断级/未闭**：

| 编号 | 审计发现 | 后果 |
|---|---|---|
| **M1** | 交付的 `validation/repro_cks1998/artifacts/full/` 根本不存在（`--profile full` 递交集群 cj1058 未回） | `full` 档数字在 worktree 里**无工件可核** → 记为 WITHHELD，L4 不可机械复核 |
| **趋势未量化** | Sod 的 3 % 门只落在「各阶最细网格」，粗网格 N=200 FAIL（最差分量 4.17 %）如实记录，但**没有任何网格阶梯**说明这条曲线处在哪个渐近区 | 无法回答「3 % 门是不是偶然」、也无法回答「限幅后的阶是多少」 |

本轮只做这两件（+ 证据链刷新），**不改任何既有门**：`GATES`（A1 ≥ p+0.5、A2 L1 ≥ 0.5、
E2 < 3 %、E3 ≤ 1 %）一字未动，round-1 的 honest-FAIL 原样保留（含 E3 论文指示器策略的越界）。

## 1. 做了什么

| # | 动作 | 产物 |
|---|---|---|
| 1 | 跑 `--profile full`（P1/P2/P3 五级光滑网格、四级方波网格、Sod 三档、E3 三策略） | `validation/repro_cks1998/artifacts/full/{metrics.json,4×CSV,a1_l2_order.png}` |
| 2 | 新增 Sod **L1 网格阶梯**驱动（逐单元 8 点 Gauss 相对 L1；含关闭限幅的对照跑） | `validation/repro_cks1998/sod_l1_ladder.py`、`artifacts/sod_l1_ladder/`（P2）、`artifacts/sod_l1_ladder_p1/`（P1） |
| 3 | 结果段程序化渲染（零手抄） | `tools/render_cks1998_round2.py` → `docs/CKS1998-round2-results.md` |
| 4 | 回归门（工件门 + 活门 + 度量不变量门） | `tests/test_sod_l1_ladder.py`（8 条） |

## 2. 预登记判据（在跑阶梯之前写定，跑后未改阈值）

Sod L1 阶梯（P1、P2 各一条；`cfl=0.12`、`tvb M=1.0`、`limiter_stage="final"`，
与 round-1 的 E2 设置**逐字相同**，因此可与之交叉核对）：

| 判据 | 内容 | 判什么 |
|---|---|---|
| **C1** | 每个分量、每一级，相对 L1 严格下降 | 阶梯不是噪声 |
| **C2** | 密度 L1 观测阶（最细一对）∈ [0.5, 2.0] | 限幅把阶压到设计阶 p+1 之下（CKS1998 的 TVB 半阶/一阶结论方向），且不退化到 0 阶 |
| **C3** | 与 round-1 的 E2 在重叠网格上单元平均相对 L2 相对差 ≤ 1e-6 | 阶梯与已发布工件同源（防"另起一套设置"） |
| **C4** | 最细一级是否过 3 % 门（**只记录判决，门本身不变**） | 3 % 门在新网格上的状态 |
| **C5** | 最细一级相对 L1 > 1e-10 | 不是浮点地板伪装成收敛 |
| **C6** | 同网格关闭限幅的对照跑，密度超调严格大于限幅跑 | 限幅确实在工作（可证伪对照） |

C2 的区间是**双侧**的：若限幅未生效（阶冲到 ~3）或数值耗散吃掉一切（阶 → 0）都会掉出区间。
这与 `docs/AUDIT-m16499-...md` 提出的「判据要有鉴别力」一致：单侧 `>0.5` 会被代数律蒙混。

## 3. 复现命令

```bash
cd <worktree>
OMP_NUM_THREADS=1 python validation/repro_cks1998/repro_cks1998.py --profile full
OMP_NUM_THREADS=1 python validation/repro_cks1998/sod_l1_ladder.py --degree 2 --cells 100 200 400 800 1600
OMP_NUM_THREADS=1 python validation/repro_cks1998/sod_l1_ladder.py --degree 1 --cells 100 200 400 800 \
    --out-dir validation/repro_cks1998/artifacts/sod_l1_ladder_p1
OMP_NUM_THREADS=1 python -m tools.render_cks1998_round2          # → docs/CKS1998-round2-results.md
OMP_NUM_THREADS=1 python -m tools.render_cks1998_evidence        # → validation/repro_cks1998/evidence.md（sha256）
OMP_NUM_THREADS=1 python -m pytest tests/test_sod_l1_ladder.py tests/test_repro_cks1998.py -q
```

## 4. 结果

**全部数值表（门判决、阶梯、观测阶、判据、交叉核对）在 `docs/CKS1998-round2-results.md`（程序生成）。**
本节只写判读，不写数字：

1. **M1 已闭**：`full` 档四张 CSV + `metrics.json` + 收敛图全部入库，full 档 verdict 与 quick 档
   同向（PASS），且 full 档多出的 P3 E3 与 P1 Sod 用例也一并落在预登记门内。
   审计「full 档 WITHHELD」的原因（无工件）不再成立。
2. **Sod 阶梯的形状**：L1 误差在每个分量上单调下降（C1 PASS），最细一对的密度阶落在
   C2 的双侧区间内 —— 即限幅后的实测阶**显著低于**设计阶 p+1，与 CKS1998 关于 TVB 限幅在
   间断面处降阶的结论方向一致；两条阶（P1、P2）互相接近，说明这个阶由**激波**控制、
   而非多项式阶数控制（这是「不许拿单条网格外推」的直接证据）。
3. **3 % 门的位置**：C4 给出最细一级的判决（见生成件）；阶梯把「粗网格 FAIL」定位成
   曲线上的一个点，而不是方案的成败。
4. **限幅在工作**（C6 PASS）：同网格关闭限幅后密度超调显著变大，且 L1 并不因此变好——
   说明限幅换的是**可容许性**，不是精度。
5. **交叉核对**（C3 PASS）：阶梯在 N=200/400 处与 round-1 已发布的 E2 单元平均相对 L2
   逐位一致（相对差在 1e-6 门内），证明阶梯与已发布工件同源、不是另起设置。

### 图

- `validation/repro_cks1998/artifacts/sod_l1_ladder/figures/l1_ladder.png`
  —— 相对 L1 vs h（三个分量 + 设计阶参考线 + 实测最细斜率）；
- `validation/repro_cks1998/artifacts/sod_l1_ladder/figures/gate_status.png`
  —— 每级 L2 与 3 % 门，末柱为关闭限幅的对照；
- `validation/repro_cks1998/artifacts/full/a1_l2_order.png` —— full 档光滑对流收敛图。

## 5. 注意点（踩过的坑 / 局限）

1. **`full` 档不是"更准"，是"更多级"**：它把光滑网格加到 320、方波加到 320、加 P3 的 E3
   与 P1 的 Sod；判据阈值一律不变，所以 full 档通过**不等于**精度提高，只是覆盖面变大。
2. **E3 的 1 % 门只在严格策略下判**：full 档里 `*_tvb_M0`（论文指示器）在 P1 仍越 1 %，
   与 round-1 结论一致；本轮的阶梯不动 E3，因此这个 honest-FAIL 依旧挂在台账上。
3. **L1 阶 ≠ 全场精度**：Sod 的 L1 由接触间断/激波主导，阶≈1 是**期望行为**，不是缺陷；
   光滑区精度要看 A1/A2 的阶（round-1 已单列）。把两者混为一谈会把正确实现判成 FAIL。
4. **P2 N=1600 的墙钟明显上升**（steps 随 N 线性增长）；本轮单线程在本机跑完，
   未上集群。若要 P3 + 3200 网格，应走 `run_cluster` 分片。
5. **限幅阶段固定 `final`**：与 E2 一致；不同 `limiter_stage` 会改变阶（round-1 E3 已示），
   本轮不做该扫描，属范围外。
6. **架构注记**：本机 loong64、float64、`OMP_NUM_THREADS=1`；x86 CI 上末位可能有差异。

## 6. 误差判定（回答「是否 <3 %」）

| 关键量 | 基准 | 是否 <3 % |
|---|---|---|
| Sod 各分量相对 L2（门网格 N=400，P2） | 精确 Riemann 解（`benchmarks.sod_episode`→`riemann_exact`） | **是**（round-1 与 full 档同值） |
| Sod 各分量相对 L2（P1，N=400，full 档新增） | 同上 | **是** |
| Sod 相对 L1（阶梯各级，P1/P2） | 同上（8 点 Gauss 逐单元积分） | 由 §4 生成件读出；粗网格级 >3 % 属**渐近区前**的正常表现，不是漂移 |
| Sod N=200（粗网格，P2） | 同上 | **否**（4.17 %，round-1 已如实登记，本轮保留） |
| A1 光滑对流观测阶 | 理论 p+1（预登记门 p+0.5） | 阶判据，见生成件 |
| E3 点值超调（论文指示器策略，P1） | 预登记 1 % | **否**（越界，如实保留） |

**结论（诚实）**：精度门上，Sod 的**门网格**误差 <3 %；粗网格 4.17 % 与 E3 论文指示器策略
的 P1 越界都是**保留的 honest-FAIL**，本轮没有通过调参掩盖，而是用网格阶梯把它们定位成
可解释的渐近/策略现象。

## 7. 未完成（honest not_run，不当成 passed）

- **2-D 系统算例（JCP 141 主算例）仍未做**：round-1 的未完成项，本轮不在范围内。
  `tensordg.euler_dg_2d` 有能力，但工作量大（限幅 + 2-D 阶梯 + 图），应单开一单。
- **论文表值逐格对照仍 WITHHELD**：仓内无 CKS1998 PDF，不冒充 REAL。
- **未与第三方 DG/FV 实现交叉**（独立性限制同上）。
- **未做 `limiter_stage` / `tvb M` 扫描**：属范围外。
- **未在 x86 CI 复核末位**。
- **平台侧 `tensor_benchmark_cases` 登记未做**：审计给的「机械升 L4」路径需要平台 API，
  本单未触碰平台。

## 8. 对台账的建议

`REPRO-DG-CKS1998` 的两条未闭项中，**M1（full 档工件缺失）本轮已闭**；Sod 的渐近行为
已有量化证据。建议台账下一条：
`CASE-DG-CKS1998-2D-SYSTEM`（slug 建议）：「CKS1998 §V 的 2-D 系统算例（双马赫反射 /
前台阶）在 TensorDG 上复现，判据 = 激波位置 vs 文献图注带 ±3 %，附 2-D 网格阶梯与图」。