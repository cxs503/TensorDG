# AUDIT m16938 — CASE-DG-AV-SHOCK-CAPTURE（传感器门控 SVV 滤波）独立复核

- **锚点**：`KPI-QC-RECHECK`（已合并功能独立复核；本次对象=已合并功能 #368/m16874 的 AV/SVV 激波捕捉）
- **被审对象**：commit `f9489be`（m16874 matured deliverables，含 `tensordg/artificial_viscosity.py`、
  `benchmarks/artificial_viscosity_verification.py`、`tests/test_artificial_viscosity.py`、
  `validation/artificial_viscosity/*`、`docs/CASE-DG-AV-SHOCK-CAPTURE.md`）
- **复核方**：m16938（generalist 席自找活），分支 `agent/m16938`

## ① 独立性限制（先声明，避免过度解读）

| 维度 | 是否独立 | 说明 |
|---|---|---|
| 求解器实现 | 否 | 与作者同一 `solve_euler_dg_1d`（同仓同架构 loong64，torch 见 `metrics.json.environment`） |
| 参考解 | **是（部分）** | `benchmarks/av_independent_check.py` 自带一套从 Rankine–Hugoniot 关系重写的精确 Riemann 解（独立 Newton 迭代 + 独立分支采样）；与仓内 `benchmarks/riemann_exact` 的密度单元均值最大偏差 **2.09e-2**（N=24），**不足以**独立重算 1e-3 量级的 L1，故 L1 复算只主张「跨工件一致性」而非实现独立 |
| 测量路径 | **是** | L1/TV 由 `profiles.csv` 原始剖面重算（不读 `metrics.json` 的结论字段）；守恒预算由边界通量重推而不是初始均值 |
| 第三方 DG | 否 | 未与 PyFR/Trixi/UCNS3D 交叉（见 ④ checks not run） |

## ② 本次做了什么（逐项命令）

| # | 检查 | 手段 | 结果 |
|---|---|---|---|
| C1 | 已发布数字复现 | `python -m benchmarks.artificial_viscosity_verification --output-dir /tmp/av_repro --cells 64 --degree 2 --quick` | 四个数逐位一致：L1 0.329 % / 0.830 %，TV 超出 0.05024 / 0.00186，min p 9.998e-2 / 1.000e-1 |
| C2 | 门禁有效性（核心发现） | `pytest tests/test_artificial_viscosity.py -q -k sensor_scaling` | **红**：`AssertionError: ('fraction', …) … 8.003 < 1.5` → 见 F1 |
| C3 | 工件证据链 | 读 `validation/artificial_viscosity/summary.md` + `metrics.json` | `conservation drift = inf`（每个 run）→ 见 F2；§④ 引用的 4 张 PNG 在树内不存在 → 见 F4 |
| C4 | 守恒预算独立重推 | 直接跑 N=24 无限制解，打印单元均值 | 动量均值 **+0.18000087** vs 边界通量预期 `(p_L-p_R)t = 0.180000`（相对差 4.8e-7）→ 旧度量把「物理上必须变化」误报为漂移 |
| C5 | 滤波强度语义 | `exponential_filter_factor(p+1, α=20)` | σ_p = 0.2865（p=1）/ 1.92e-2（p=2）/1.79e-3（p=3）/2.77e-4（p=4）→ α_max 不代表满阻尼，见 F3 |
| C6 | 网格阶梯（作者标 not_run） | `--cells 64 96 128`（含 TVB 头对头，265.7 s） | N=64/96/128：AV L1 0.830/0.571/0.437 %，TV 超出 0.00186/0.00529/0.00954；同网格无限制 0.329/0.210/0.152 %、0.0502/0.0530/0.0609 |
| C7 | 跨工件一致性 | `python benchmarks/av_independent_check.py`（由 `profiles.csv` 重算 7 个 run 的 L1/TV） | 最大偏差 L1 2.1e-10、TV 2.0e-8 → **PASS** |
| C8 | 复算后门禁 | `pytest tests/test_artificial_viscosity.py -q` | **18 passed**（原 12 条 + 本单 6 条），73.5 s |

## ③ 发现（按严重度排序，含证据与复现）

### F1（高，阻塞「合入即绿」的声明）门禁结构上不可能通过：符号约定错位
`sensor_law()` 返回 `smooth_slope = polyfit(log10(cells), log10(sensor))[0]`，对衰减的传感器**必然为负**
（实测 −4.003 / −3.987，作者自己在 `docs/CASE-DG-AV-SHOCK-CAPTURE.md` §③ 表格里也印的是 −4.002 / −3.987），
而原门禁断言 `abs(law["smooth_slope"] - 4.0) < 1.5`——要求斜率 ≈ **+4**。
复现：`pytest tests/test_artificial_viscosity.py -k sensor_scaling` → `abs(-4.003 - 4.0) = 8.003 > 1.5`。
结论：这条门在**任何机器、任何版本**上都红；m16874 的「测试绿」不成立（`main` 上是红门）。
处置：比较幅值并把容差从 ±1.5 收紧到 ±0.35（实测偏差 0.003/0.013，p=1 global 1.987→0.013），修后绿。

### F2（高，证据完整性）守恒度量同时「无定义」且「参照错」
原式 `drift = max(|mean - initial_mean| / |initial_mean|)`：
1. 动量分量 `initial_mean[1] = 0` → **每个 run 都报 `inf`**（`summary.md` 原样打印 `inf`）。
   一个恒为 `inf` 的列无法区分「守恒」与「坏掉」，而这是它唯一的存在理由。
2. 更根本：Sod 是**开域**且 `p_L != p_R`，边界压差持续注入动量，**精确解**的域平均动量也必须变
   `(p_L-p_R)t = 0.18`（C4 实测 +0.18000087）。用初始均值做参照，把物理输运记成格式误差。
处置：新增 `expected_state_mean()`（初始均值 + 边界通量贡献）与 `conservation_drift()`（零均值分量走绝对分支，
有限值），`summary.md` 拆 `drift rel` / `drift abs` 两列。修后：N=64 相对漂移 1.46e-10、N=96/128 约 1e-13，
动量分支有限（绝对 5.7e-11）。另加「注入 +1e-3 非守恒必须被抓到」的牙齿测试。

### F3（中）「饱和强度」不是满阻尼，且随阶数漂移（未文档化）
`exponential_filter_factor` 用 `P = nmodes = p+1` 归一，最高模 `m=p` 的滤波因子
`σ_p = exp(-α_max (p/(p+1))^4)`：α_max=20 时 p=1 仍保留 **28.7%** 的模幅，p=2 保留 1.9%，p=4 保留 0.028%。
即 α_max 是**与 p 强耦合的旋钮**，不是阻尼上界；p=1 上该机制近似失效。这与常见的「以最高模为参考」写法不同，
跨 p 迁移参数会失真。本次不擅自改数值语义（会作废已发布标定），改为：文档 §⑤ 明写 + 新增知识门禁
`test_top_mode_damping_is_degree_dependent_as_documented`（改归一化必须同时改文档，否则红）。

### F4（中，证据完整性）§④ 引用的 4 张图在树内不存在
`docs/CASE-DG-AV-SHOCK-CAPTURE.md` §④ 指向 `validation/artificial_viscosity/figures/{sensor_law,sod_profiles,
sensor_profile,error_tv}.png`，但 `git ls-files validation/artificial_viscosity` 只有 4 个文本工件——
合并门禁在 m16874 滤掉了这 3 个二进制（commit 标题自证「3 个文件被门禁滤除」）。
处置：图由命令再生（`--output-dir validation/artificial_viscosity`），文档 §④ 改写为「图=可再生产物 + 其数据源
（`metrics.json`/`profiles.csv`）已入树且被 sha256 钉住」；新增 `test_committed_metrics_artifact_matches_its_sha256`
把「发布的数字 = 被哈希的数字」变成硬门禁（工件被滤掉/被改写即红，不再是静默 skip）。

### F5（低）`modal_smoothness` 文档串自相矛盾
原文把 `"fraction"` 描述为「除以 **mesh-mean**」，又说 `"global"` 除以…（句子被截断）。
与实现（fraction=本单元总能量+floor；global=网格均值+floor）不符。已改写为与代码一致的两句。

### F6（中，门禁牙齿）原文件只「跑过代码」，没断言能力
原 Sod 门禁只断言 `tv_av < tv_none`（无余量、1e-9 的改善也能过），对已发布工件零断言。新增 6 条：
零强度 AV 与无限制解**逐位相同**（端到端惰性，覆盖 `limiter_kwargs` 透传 + RK 子步路径）、
守恒漂移有限且有牙齿、AV 不得让守恒变差（≤2× 无限制解）、度相关阻尼语义、`metrics.sha256` 溯源、
边界通量预算（`expected mean momentum = 0.18`）。

## ④ 未做（如实登记，不得当成通过）

- 与第三方 DG 实现（PyFR / Trixi.jl / UCNS3D）交叉：**not_run**（本机无这些栈，集群未装）。
- 二维/三维与强激波（Blast wave、双爆轰）上的同一机制：**not_run**。
- 熵稳定性/可容许性证明：**不存在**（作者已诚实声明；本次也未做）。
- x86 CI 架构复核：**not_run**（本机 loong64；浮点求和顺序可能改动末位，不影响 1e-2 级结论）。
- WENO/AO 在 N=96/128 的头对头：**not_run**（TVB 已补到 N=64，见下）。

## ⑤ 被夸大的声明（逐条）

| 原声明 | 实际 | 判定 |
|---|---|---|
| 「TV 超出量 0.050 → 0.002（25×）」 | 只有 N=64 成立：N=64/96/128 分别为 **27.0× / 10.0× / 6.4×** | **缩减倍数随加密下降**，不能当作网格无关能力 |
| 「可判定量全部达标」 | 守恒列曾是 `inf`（无量纲意义下的失败），且源文件有一条红门 | 不成立（本单已修） |
| §④ 4 张图 | 树内不存在 | 不成立（本单改为可再生产物 + 哈希门禁） |
| 「TVB 头对头未完成，不得宣称优于 TVB」 | N=64 已有：AV 0.830 % / TV 0.00186 vs TVB 1.685 % / 0.0119（同网格同格式） | 现在**可以**说 AV 在 N=64 上两项都优于 TVB；仍不得外推到 N≥96 或 WENO/AO |
| 「L1 劣化 2.5×」 | 2.52× / 2.71× / 2.88×（N=64/96/128） | 量级正确，轻微随加密变差 |

## ⑥ 精度判定（vs 基准，显式回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| Sod L1(ρ)，AV 默认（N=64/96/128） | 精确 Riemann 单元均值 | 0.830 / 0.571 / 0.437 % | **是** |
| Sod L1(ρ)，无限制（同三档） | 同上 | 0.329 / 0.210 / 0.152 % | **是** |
| 传感器光滑支标度指数（p=2） | 2p = 4 | 4.003（fraction）/3.987（global） | **是**（0.08 % / 0.33 %） |
| 压力正性 | p > 0（精确右态 0.1） | AV 1.000e-1；无限制 N=128 降到 9.59e-2 | **是**（正性） |
| 守恒（对边界通量预算） | 0 | 1.46e-10（N=64）/≈1e-13（N≥96） | **是** |
| σ_p（阻尼语义） | 公式 exp(-20(p/(p+1))^4) | 逐位相等（<1e-12） | **是**（但 α_max 非满阻尼，见 F3） |

## ⑦ disposition

**approve with changes —— 变更已在本分支落地并通过验收**：
F1/F2/F4/F5 已修（红门转绿、度量重定义、溯源硬化、文档纠正），F3/F6 以门禁+文档方式固化。
**阻塞项已清零**，但「AV 优于 WENO/AO」与「网格无关的振荡削减倍数」仍**不可宣称**（见 ⑤）。
建议后续：把 AV 机制接到二维 Euler（`euler_dg_2d`）+ Blast wave（`CASE-DG-BLASTWAVE-PERSSON` 已有锚点）上复测。
