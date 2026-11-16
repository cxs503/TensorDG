# XVAL-DG-ROUND2-CERTIFIED-REFERENCE — 时间离散污染审计的证据完整性修复

> 锚点：`XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open，开工前已核对）
> 分支：`agent/m23178`　日期：2026-11-16
> 类型：defect_fix（验证欠账：已发布判决与其证据脱钩）

## 1. 案例介绍（物理背景 / 参数）

**Shu–Osher 激波/熵波相互作用**（Shu & Osher 1988）是 DG 求解器的经典一维 Euler 基准，
本锚点把它的参考解从「一条细网格」升级为 **round-2 认证参考**（Roache GCI 三级/四级梯次，
`Fs=1.25`，Richardson 外推到连续极限，参考解存在次细层网格上）。

| 参数 | 值 |
|---|---|
| 方程 / γ | 1D Euler / 1.4 |
| 域 / 终态 | x∈[-5,5]，t=1.8 |
| 左状态 | ρ=3.857143, u=2.629369, p=10.33333（Mach-3 Hugoniot） |
| 右状态 | ρ=1+0.2sin(5x), u=0, p=1 |
| 参考求解器 | FV-WENO5(JS)/HLLC/SSP-RK3，CFL=0.3 |
| 空间梯次 | N = 800 / 1600 / 3200 / 6400 |
| DG 腿 | 11 条（P1/P2/P3 × N=200/400/800, CFL=0.4；P2 N=400, CFL=0.1/0.2） |
| 精度门 | 3 %（owner 标准）/ 5 %（认证使用门） |

**本单对象**：`validation/xval_dg_round2/artifacts/time_pollution.{json,md}` —— 该锚点
「已发布的参考解是否被时间步污染」这一审计的发布件。

## 2. 怎么做的（独立重建执行路径）

### 2.1 缺陷定位

在**当前检出**上重跑审计，`tp.audit()` 直接抛

```
FileNotFoundError: grid 800 has 2 of 3 CFL levels (need 3); missing
validation/repro_shu_osher/artifacts/raw/fv_cfl0p075_n800.npz
```

即：发布件声称的 CFL 序列（0.3/0.15/0.075 × 800/1600/3200/6400）中有 **6 个剖面从未入库**，
审计所依赖的时间腿证据在仓库里不存在。与之并存的发布件却是

```
Verdict: TIME_CLEAN  (accepted = True)   T0..T4 全过
```

三处硬矛盾（都可用仓库内文件复核）：

| 项 | 发布件声称 | 已入库数据实测 |
|---|---|---|
| 三级参考观测阶 p | 1.376846 | **1.2825091761164593** |
| 三级参考带 U % | 0.445327 | **0.08279718833081827** |
| 四级参考带 U % | 0.192169 | **0.04162173159427086** |
| 11 条 DG 腿误差 | 4.961733 / 2.139138 / 0.910532（**全部相同**，P 与 CFL 都不分辨） | 2.455701 / 1.547920 …（11 条互不相同） |

**来源判定（决定性证据）**：把该模块的**合成夹具**（`_synthetic_rawdir(k_time=1.0)`，
严格 CFL^2 功率律）跑一遍，得到

```
raw_band_percent           = 0.4453274366117238      ← 与发布件 0.445327 相同
time_corrected_band       = 0.45079257267357276     ← 与发布件 0.450793 相同
pollution_ratio           = 0.04643949427538337     ← 与发布件 0.046439 相同
p_t                       = 2.0（精确）              ← 与发布件 1.9999999999998446 相同
```

即 **`TIME_CLEAN` 是合成数据的自检结果，被当作真实参考解的回答发布了**。合成夹具的 11 条
DG 腿按 `ncells` 生成、与 P/CFL 无关 —— 这正是发布件里「11 条腿塌成 3 个误差值」的来源，
也使 T4「无判决翻转」在那份数据上是**空转**（它只比较了 3 个不同剖面）。

### 2.2 修复（最小、可验证）

**(a) 缺输入不再是异常，而是可渲染的判决** —— `audit_or_withheld()`
新增 `WITHHELD_MISSING_INPUTS`：缺少 CFL 剖面时返回结构化结果（逐条列出缺失文件 +
生成命令 + 成本），`markdown_withheld()` 渲染成「不主张任何污染判决」。理由写在代码里：
函数抛异常是对的，但**发布件**在异常后保留上一次判决，读者无法区分「实测 CLEAN」与
「陈旧残留」。

**(b) 新增 T5（腿间可分辨 / T4 非空转）** —— `leg_profiles()` + `leg_distinctness()`
同一网格上任意两条腿的相对 L1 必须 > `LEG_DISTINCT_EPS = 1e-12`；重复剖面存在则不通过，
并把 **T4 降级**（`downgraded_by="T5"`），判决走 `AUDIT_INVALID_LEGS`。
若只加 T5 而不降级 T4，判决仍可能被一个空转的 T4 带着走。

**(c) 输入溯源戳** —— `input_provenance()` 对审计消费的每个 npz 记 `sha256` + 字节数，
写进 `time_pollution.json` 的 `provenance.files`，使「发布件 vs 现算」的陈旧检测成为可能。

**(d) 发布件改正** —— 用 `audit_or_withheld()` 重新生成
`time_pollution.{json,md}`：verdict = `WITHHELD_MISSING_INPUTS`，逐条列出 6 个缺失剖面
与生成命令。**旧的 `TIME_CLEAN` 不予保留**（它是合成数据的自检结果，不是真实参考解的答案）。

**(e) 测试层加闸**（`tests/test_xval_dg_round2_time_pollution.py`）：
`test_committed_artifact_is_not_stale`（发布件必须等于现算的判决；withheld 时禁止出现
`TIME_CLEAN`）、`test_committed_dg_legs_are_pairwise_distinct`（11 条腿 16 个同网格对
全部可分辨）、`test_leg_distinctness_flags_duplicate_profiles`（负控制）、
`test_audit_or_withheld_never_crashes_and_claims_nothing`、
`test_missing_inputs_carry_their_producer_command`、
`test_provenance_detects_a_changed_profile`。
合成夹具同时修正为「每条腿各自一块权重」——否则夹具本身就在制造 T5 要拒的空转。

## 3. 计算结果（程序化注入）

### 3.1 已入库 FV 梯次认证（复核，与既有报告一致）

| 梯次 | 观测阶 p | 带 U % | 参考层 N | 判决 |
|---|---|---|---|---|
| 三级 800/1600/3200 | 1.2825091761 | 0.08279719 | 1600 | CERTIFIED |
| 四级 800/1600/3200/6400 | 1.1767348864 | 0.04162173 | 3200 | CERTIFIED |

### 3.2 11 条 DG 腿对四级参考（5 % 门，全部 PASS，与既有报告一致）

| 配置 | rel L1 % | 配置 | rel L1 % |
|---|---|---|---|
| P1 N=200 | 2.455701 | P2 N=800 | 0.566719 |
| P1 N=400 | 1.547928 | P3 N=200 | 2.426617 |
| P1 N=800 | 0.578643 | P3 N=400 | 1.522033 |
| P2 N=200 | 2.420304 | P3 N=800 | 0.669867 |
| P2 N=400 | 1.516026 | P2 N=400 CFL=0.1 / 0.2 | 1.519713 / 1.518092 |

T5 实测：`n_legs=11, n_groups=3, n_pairs=16`，`min_pair_rel_l1 > 1e-6`，无重复 → T4 有证据。

### 3.3 时间腿审计

**无实测数字**：6 个 CFL 剖面不在仓库内，判决 = `WITHHELD_MISSING_INPUTS`（不再是
`TIME_CLEAN`）。生成作业已递交集群：`cj4661`（6 分片，`gen_cfl_ladder.py --shard i --nshards 6`，
单核，上限 3600 s；递交时排队位次 111）。

## 4. 结果图

无图，理由：本单交付的是**证据完整性闸门 + 撤销一个无证据支持的判决**，真实时间腿数字
尚不存在（`WITHHELD`），画任何「时间腿 vs 空间腿」曲线都会把合成数据当实测发布，
正是本单要修的缺陷。图件生成路径（`tp.figures`，3 张）保留，待剖面入库后自动出图。

## 5. 注意点

1. **不要把 `WITHHELD` 读成「审计失败」**：它是「答案还没算出来」，与 `TIME_POLLUTED`
   是不同结论；旧发布件把前者写成了后者之外的一个假 `TIME_CLEAN`。
2. **T5 只在同网格内比较**：N=200 与 N=800 的 rel L1 需要插值，审计不引入这个自由度。
3. **合成夹具是双刃剑**：它能证明引擎正确（本单正是靠它反推出旧发布件的真实来源），
   但也极易被当成实测数字抄走。T5 + provenance + 陈旧闸门是三道防线。
4. **适用范围**：一维 Euler；不声明多维激波捕获的时间污染结论。
5. **未跑项（honest not_run）**：集群作业 `cj4661` 在本单结束前仍在排队，未取回工件，
   因此 T0–T4 的真实数字仍为 `unknown`（不是 passed）。

## 6. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3 %? |
|---|---|---|---|---|
| 三级参考 p | 既有报告 1.282509 | 1.2825091761 | 0（逐位） | **yes** |
| 四级参考 p | 既有报告 1.176735 | 1.1767348864 | 0（逐位） | **yes** |
| 四级参考带 U | 既有报告 0.041622 % | 0.04162173 % | 0（逐位） | **yes** |
| 旧发布件 band vs 实测 band | 0.445327 %（发布） | 0.082797 %（实测三级） | **+437.8 %** | **no** |
| 旧发布件 legs vs 实测 legs | 4.9617 / 2.1391 / 0.9105 | 2.4557 / 1.5479 / 0.5786 | **+102.0 %** | **no** |

→ 已入库数据的复核全部逐位一致（< 3 %）；**旧发布件的数字全部超出 15 % 不可接受线**，
本单据此撤销该判决，而不是保留它。合成夹具复现旧数字（逐位相同）是撤销的依据。

## 7. 产物

| 文件 | 说明 |
|---|---|
| `benchmarks/xval_dg_round2_time_pollution.py` | T5、`WITHHELD_MISSING_INPUTS`、`audit_or_withheld`、`input_provenance`、被降级的 T4、withheld 渲染、CLI 走 withhold 通道 |
| `validation/xval_dg_round2/artifacts/time_pollution.{json,md}` | 改为 `WITHHELD_MISSING_INPUTS` + 6 条缺失输入与生成命令 |
| `tests/test_xval_dg_round2_time_pollution.py` | 6 个新测试（含 2 个负控制）+ 合成夹具改为腿间可分辨；31 passed / 4 skipped |

复现：

```bash
python -m benchmarks.xval_dg_round2_time_pollution            # → WITHHELD_MISSING_INPUTS
python -m pytest tests/test_xval_dg_round2_time_pollution.py -q   # 31 passed, 4 skipped
```