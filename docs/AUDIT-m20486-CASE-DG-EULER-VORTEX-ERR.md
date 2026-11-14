# AUDIT-m20486：CASE-DG-EULER-VORTEX-ERR 证据包独立复核（adversarial review）

- **复核人 / 单号**：m20486（generalist 席，自驱工时；技能库 adversarial-review）
- **被复核提交**：`0558046` "m20264: matured mission deliverables"（引入 `case_euler_vortex_errconserve/` 与 `docs/CASE-DG-EULER-VORTEX-ERR.md`）
- **被复核工件**（commit 内身份，sha256）：
  - `case_euler_vortex_errconserve/metrics.json` → `6912fcfa9e95f9a791493cd7012d5a9aadd397524d72396427cba57aa03afb67`
  - `docs/CASE-DG-EULER-VORTEX-ERR.md` → `42489197b4fd3b6af6755683c3b9ed92ade9683730967b6a4f5d79e2430cfc59`
  - `case_euler_vortex_errconserve/results/*.json`（6 个检查点）、`error_conservation.csv`、`summary.md`
- **被复核结论**：`verdict = INCOMPLETE`（C1 未通过 8.97% > 6%；C2 `status = not_evaluated`）
- **总账锚点**：`CASE-DG-EULER-VORTEX-ERR`（复核时 `status = open`）→ 本单做的是「让该锚可判定」而非重跑既有检查点

## 1. 独立性限制（必须先声明）

| 限制 | 说明 |
|---|---|
| 同一工作树 | 复核在同一 checkout 内进行，未做独立第三方环境复现；已用重跑 2 个既有检查点来替代（见 §3） |
| 未重跑全部 6 个既有记录 | `--shard` 语义会跳过已存在结果，本轮新增检查点与新跑记录混合；既有 6 条按其提交身份引用，未全部重新执行 |
| 参考文献未取原文 | Yee, Sandham & Djomehri (2000) JCP 150:199 全文不在库内 → 6% 容差与「误差守恒」的原文表述标 **WITHHELD（未核）**，不得声称已复核文献 |
| 不改平台 | 遵 owner 控仓令，本单不触碰 TensorPlatform |

## 2. 审查范围与所做检查

1. 读入 `metrics.json` / `summary.md` / `error_conservation.csv` / `results/*.json`，用原始记录**独立重算**每一条门禁量（不信任聚合产物）。
2. 从 `steps` / `dt` / `final_time` 三个字段反推时间推进账，检查是否自洽。
3. 逐条对照 `benchmarks/case_euler_vortex_errconserve.py` 的门禁实现与 `docs/`、平台报告正文的**声明一致性**。
4. 对照姊妹算例 `benchmarks/case_euler_vortex.py` 的 `GATE_SET` / `TOLERANCE`，确认「算例池容差」的作用域先例。
5. 检查测试是否断言不变量，还是仅执行代码（vacuity 检查）；检查门禁反向证据（能否拒绝坏证据）。
6. 复现性：重跑 2 个既有检查点比对。

## 3. 独立重算与复现性结论

- 原始记录 → 聚合量重算一致：C3 最坏漂移 `1.078e-13`（`p0_n128_t10_cfl0.4:energy`）、C4 `5.245e-08`、精确不变量 `质量 = x 动量 = 98.241743560191`、`熵 = 100.0 = 箱面积`。这些**都成立**：质量与 x 动量相等是 `u_inf = 1` 且密度扰动在全域积分为零的必然结果，熵等于箱面积是等熵条件的必然结果 —— 精确解侧无缺陷。
- **复现性（本单新做）**：`--task p0 --cells 128 --final-time 1.0` 与 `--task dg --degree 2 --cells 32 --final-time 1.0` 重跑，与已发布记录**逐位一致**（`max_percent` 相对差 0.00e+00，步数 183 / 469 相同）。⇒ 随机性不是本包的问题，所有分歧都来自判据与作用域。
- `steps × dt ≠ final_time`：经代码核对，`dt` 是**末步被 `final_time - time` 截断**的值（`euler_dg_2d.py:927`），故乘积天然不等于末时刻。**不是积分错**，但字段无标注会误导读者（F5）。

## 4. 分级发现

### F1（阻塞级，判据可判定性）：C2 结构上不可判定，且用 `INCOMPLETE` 掩盖了一个已可判定的失败

- **证据**：`metrics.json → gates.C2_error_conservation`：`exponent = NaN`、`status = "not_evaluated"`、`growth_ratio = 19.510`、`verdict = "INCOMPLETE"`。
- **机制**：`C2` 绑定到 `GATE_FAMILY = dg_d2_n32`，而该族阶梯只有 `t = 1, 2` **两个**检查点；`_fit_exponent()` 明确要求 ≥3 点，于是 alpha 恒为 `NaN`，整包判为 `INCOMPLETE`。与此同时 `growth_ratio = 19.5` 已经**硬性违反**同一条判据里写的 `<= 2.0` 界，却被降格成「未评估」。
- **复现**：`PYTHONPATH=. python3 benchmarks/case_euler_vortex_errconserve.py --aggregate`（用 commit 内的 `results/`）。
- **影响**：该锚点的主判据（误差守恒）从未被有效测量；`INCOMPLETE` 比 `FAIL` 更「好看」，属**健康度过度声明**。
- **修复**：设计族阶梯扩到 `t = 1, 2, 5, 10`（4 点，alpha 可判定）；门禁状态语义改为「可判定的子判据失败 ⇒ `status = failed` ⇒ `verdict = FAIL`，不得写成 `not_evaluated`」，并新增回归测试 `test_c2_gate_is_decidable_only_with_three_checkpoints`、`test_c2_gate_rejects_superlinear_error_growth`。

### F2（阻塞级，判据作用域）：6% 快照容差被施加到全部长时检查点

- **证据**：聚合 `C1` 取「所有检查点」最大场误差 ⇒ `8.965%`（`p0_n128_t5_cfl0.4`）；但**同一条运行**在 `t=1` 是 `2.817%`，设计族 `dg_d2_n32` 全部检查点 ≤ `0.247%`。
- **先例**：姊妹算例 `benchmarks/case_euler_vortex.py` 的 `GATE_SET = (DG-P2/32² @ t=1, P0/128² @ t=1)`、`TOLERANCE = 0.05` ⇒ 算例池容差是**快照容差**（单点），不是「任意时刻」容差。把长时检查点也拿它卡，等于用设计网格的容差去审一个**故意一阶**的交叉校核格式。
- **影响（verdict 相关，必须公开）**：这一条是 `INCOMPLETE/FAIL` 与 `PASS` 的分水岭，属**改变结论**的修正，故必须同时满足：①原始数字**不得删除**；②依据必须来自既有工件自述（改包自己写的「两条独立离散路径防单一实现自证」= 交叉校核目的）与姊妹算例先例。
- **修复**：`C1` 改为「各族在**设计检查点 `t = 1`** 的最大场相对 L2 < 6%」（两族都被包含，交叉校核仍在验收集内，未剔除）；长时原始最差值以 `C1x_crosscheck_raw`（`informational`，不进 `all_passed`）**逐字保留**，并新增守护测试 `test_long_time_checkpoints_exceed_tolerance_but_stay_visible`（断言 t=20 的 10% 违规必须仍在 `C1x` 里可见、`worst_at` 指向它）。

### F3（重要，判据无出处）：`alpha <= 0.5` 阈值没有任何文献来源

- **证据**：`metrics.json → reference` 自述「容差按条目 verbatim 采用、未从原文复核」；而 `alpha <= 0.5` 既不在算例池条目的验收串（`【算例池|DG-EULER-VORTEX-ERR｜ref: Yee et al. 2000｜容差: <6%】`）里，也不在 `docs/` 的出处说明里 —— 它是上一轮**自拟**的。
- **物理判断**：固定网格上耗散格式的误差在预渐近区随输送时间线性累积（`e ∝ t`，即 `alpha = 1`），`alpha <= 1` 才是「不超线性放大」的物理上界；`alpha <= 0.5` 会判**任何**收敛格式的长时间平移为失败，作为验收门限不成立。
- **修复**：门限改为 `alpha <= 1`（加 `1e-9` 浮点余量，避免恰好 1.0 的线性曲线卡在舍入上）；`alpha <= 0.5` 保留为**报告项** `strict_target`（其达标/未达标在报告与门禁表里显式写出），不参与 `all_passed`。并新增边界测试 `test_c2_linear_accumulation_is_the_boundary_and_passes`。
- **另注**：`growth_ratio <= 2.0` 同时降级为参考量——`t=1` 的 `dg` 误差（`1.27e-4`）落在投影/耗散地板上，用它做比值的分母没有物理意义。此改动**不偏袒**结论：比值与指数两种读法在本包实测下给出同一方向（见 §5 复测）。

### F4（重要，已发布报告过度声明）：`docs/` 与平台报告声明的算例范围超出工件实际内容

| 声明原文（`docs/CASE-DG-EULER-VORTEX-ERR.md`） | 工件实际 | 判定 |
|---|---|---|
| `离散族 \| DG-P2/32²、DG-P1/32²、DG-P3/16²（HLLC+SSP-RK3）；P0/128²` | `metrics.json.family_count = {dg_d2_n32: 2, p0_n128: 4}` → **1 个 DG 族 + 1 个 P0 族** | 过度声明（无 P1/P3 记录） |
| `无量纲末时刻 \| t ∈ {1,2,5,10,20}`、摘要 `t 到 20，每轴 2 个周期箱` | 检查点 `t ≤ 10`（**无 t=20 记录**） | 过度声明 |
| `6 个界面的周期通量相消` | 二维周期箱是 **2 对（4 条）边** | 描述错误 |
| `集群递交（2 核/片，OMP_NUM_THREADS=2）` | 本轮与既有递交均为 1 核/片 | 描述与运行不符 |

- **影响**：这些是**对外发布**的报告正文（平台 `reports` 条目同源），读者会据此认为验证覆盖了 4 个族与 2 个周期箱。属可复现的夸大陈述。
- **修复**：①表的「末时刻」「离散族」两行改为**从 `records` 程序化生成**（`t_list` / `family_list` / `crossings`），摘要同样程序化，杜绝手写；「6 个界面」改「2 对（4 条）边」；递交描述改为实际 1 核/片；测试项数从 `tests/` 文件实时统计（改前 12、改后 16）。

### F5（次要，可读性）：`dt` 字段无标注

`dt` 是末步截断值，`steps × dt` 与 `final_time` 无关。已加 `dt_note` 字段说明，避免下一位复核者（本单即曾）误判为积分未跑到末时刻。

### F6（次要，测试设计）：测试把无出处的阈值写成了断言

改前 `test_c2_gate_rejects_secular_error_growth` 用 `e = 1e-4·t`（恰好 `alpha = 1` 的**线性**累积）断言「必须判失败」——它守护的是 F3 里那个无来源的 0.5 阈值，而不是物理不变量。已改为「超线性（`e ∝ t²`）必须失败」，并补「线性 = 边界必须通过」与「<3 点必须 `not_evaluated`」两条。

## 5. 修复后的复测（程序化注入，无手抄）

新增检查点（`run_compute` cj2738 分片 + 本机 t=1 分辨率点），全部由 `metrics.json` 注入：

| 族 | t | 步数 | 最大场误差 % | 质量漂移 | 熵漂移 |
|---|---|---|---|---|---|
| dg_d2_n32 | 1 | 469 | 0.0127 | 1.7e-14 | 2.34e-08 |
| dg_d2_n32 | 2 | 937 | 0.2468 | 3.5e-14 | 5.25e-08 |
| dg_d2_n32 | 5 | 2342 | 14.4302 | 8.8e-14 | 1.42e-07 |
| p0_n128 | 1 | 183 | 2.8170 | 1.1e-14 | 3.51e-03 |
| p0_n128 | 2 | 357 | 4.8977 | 2.2e-14 | 5.50e-03 |
| p0_n128 | 5 | 854 | 8.9651 | 5.1e-14 | 8.19e-03 |
| p0_n128 | 10 | 1631 | 5.8941 | 9.6e-14 | 9.73e-03 |
| p0_n256 | 1 | 370 | 1.5187 | 2.3e-14 | 2.01e-03 |
| p0_n64 | 1 | 90 | 4.9262 | 5.5e-15 | 5.52e-03 |
| p0_n64 | 5 | 408 | 8.0939 | 2.4e-14 | 9.56e-03 |

| 门禁 | 实测 | 判定 |
|---|---|---|
| C1 快照容差（各族 @ t=1） | 4.9262%（p0_n64_t1_cfl0.4）< 6% | passed |
| C1x 长时最差（仅参考） | 14.4302%（dg_d2_n32_t5_cfl0.2） | 不作判据 |
| C2 累积指数（3 点） | alpha = **4.377** vs 上界 1.0；严格目标 alpha<=0.5 未达 | failed |
| C3 离散守恒 | 最坏 1.08e-13 | passed |
| C4 熵预算 | 1.42e-07 < 1e-3 | passed |
| C5 交叉校核观测阶（t=1，n=[64, 128, 256]） | 0.849 >= 0.8 | passed |
| **总判定** | 无不可判定门禁（`not_evaluated = []`） | **FAIL** |

**结果解读（honest FAIL，不粉饰）**：设计族 DG-P2/32² 的误差随输送时间**超线性放大**
（t=1 `0.0127%` → t=2 `0.2468%` → t=5 `14.43%`，最小二乘 alpha = `4.377`），
即在 32² 网格上该 DG 离散**不满足**「误差守恒」——这两条读法（alpha<=0.5 与 alpha<=1）**同向失败**，
故 F3 的阈值修正是**与结论无关**的（不构成 p-hacking）。
t=2→t=5 的 58 倍跳变远超线性累积，**强烈提示 DG 长时积分的放大机制**（可能是界面项/SSP-RK3 或投影初值在长时间后的色散聚焦），
已列入下次自找活线索，本单不下结论。
同时 C1 快照（t=1，两族 4 族点全 <6%）通过、C3 守恒贴舍入、C4 熵预算 `1.4e-07` 通过、
C5 交叉校核以观测阶 `0.849` 独立确认第二条代码路径确实一阶收敛。

## 6. 未执行的检查（unknown / not_run，不计为通过）

| 检查 | 状态 | 原因 |
|---|---|---|
| Yee et al. (2000) 原文的容差与「误差守恒」表述逐字复核 | **not_run / WITHHELD** | 论文全文不在库内、无网络取原文；6% 仍按算例池条目 verbatim 采用 |
| 独立第三方环境（另一 checkout / 另一机器的 bit-level 复现） | **not_run** | 单干范围；已用同机重跑 2 个检查点替代（逐位一致） |
| 既有 6 个检查点全部重跑核对 | **部分 not_run** | `--shard` 跳过已存在结果；仅重跑 2 个（`p0_n128_t1`、`dg_d2_n32_t1`） |
| 激波 / 限制器 / 边界层算例 | **not applicable** | 本算例为光滑等熵涡，判据不覆盖间断问题 |
| DG-P1 / DG-P3 族与 t=20 检查点 | **not_run（已从报告中删除该声明）** | 从声明中移除，而非补算；如需覆盖应另开条目 |

## 7. 处置（disposition）

**request changes → 已在本单完成修改**，随后按复核后结论重新发布报告。
本单**不**把该锚点标记为 verified：验收由平台总账流程决定，本单只提交「可判定的门禁 + 如实数字」。

修复清单（全部落在白名单路径）：

| 文件 | 改动 |
|---|---|
| `benchmarks/case_euler_vortex_errconserve.py` | 阶梯扩到 4 个设计检查点 + P0 分辨率梯 n=64/128/256；C1 作用域（快照）；C1x 信息态原始值；C2 阈值与状态语义；新增 C5 交叉校核收敛阶；`_p0_cells`/`_observed_order`；`dt_note`；`summary.md` 门禁表同步 |
| `publish_case_euler_vortex_errconserve_report.py` | 作用域与族列表程序化；门禁表 6 行（C1/C1x/C2/C3/C4/C5）；⑥ 误差表重写（含 `<3%` 显式回答）；摘要重写 |
| `tests/test_case_euler_vortex_errconserve.py` | 12 → 16 项：超线性拒绝、线性边界通过、2 点不可判定、C1x 可见性守护、C5 非收敛拒绝、合成 P0 梯 |