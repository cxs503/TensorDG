# TensorDG 验收门空转普查 —— 收口（GATE-VACUITY-CLOSURE）

- **锚点**：`【锚点=NEW:gate_vacuity 漂移守卫在作者工作树之外恒红（schema 1 存绝对路径）；RED-ON-CLEAN 尚余 1 例（positivity_scaling_limiter 记录未生成）；ALLOWED_RED_CLEAN 有 1 条过期申报】`
- 仓：TensorDG · 工具 `tools/audit_gate_vacuity.py` · 政策门 `tests/test_gate_vacuity_policy.py` · 上游 `docs/AUDIT-m16815-gate-vacuity-red-main.md`、`docs/AUDIT-m16875-gate-vacuity-tracked-artifacts.md`
- 图解：`docs/reports/figs/gate_vacuity_closure.png`（判级分布 前/后）

## ① 案例介绍（问题）

m16815/m16875 建成了两层能力：静态空转普查（`tools/audit_gate_vacuity.py`）与政策门（`tests/test_gate_vacuity_policy.py`）。二者都在 `main` 上，但**在作者工作树之外**，它们自己就是红的。「管到底」这一轮要回答的是：这套守卫到底能不能在别人的检出上运行？

只读阶段在**当前 main 的干净检出**（`git archive HEAD` → `/tmp/gv_pre`）上实测到 4 个缺陷：

| # | 现象 | 现场证据（HEAD 内容、干净检出） |
|---|---|---|
| D1 | 普查 JSON 存的是**生成者的绝对工作树路径**（`repo_root` 与每个 `artifacts[].path`/`skip_sites[].artifact` 都是 `/root/tp-agent/work/m16993/...`）→ `--check` 比较两个绝对路径，只能在生成它的那台机器上相等 | `python tools/audit_gate_vacuity.py --check` → `gate vacuity census drifted` · exit **1**（树内容完全一致） |
| D2 | 同一原因让政策门 3 道在别人的检出上直接红 | `pytest tests/test_gate_vacuity_policy.py -q` → **3 failed, 11 passed** |
| D3 | `ALLOWED_RED_CLEAN` 里 `tests/test_dg_advection_bench.py` 已过期（记录早已落在被跟踪路径） | `ALLOWED_RED_CLEAN entries no longer red: ['tests/test_dg_advection_bench.py']` |
| D4 | 最后 1 例 `RED-ON-CLEAN`：`tests/test_positivity_scaling_limiter.py` 硬断言 `docs/benchmarks/results/positivity_limiter/metrics.json`，**该记录从未生成** → 干净检出上 3 道门必红 | `pytest tests/test_positivity_scaling_limiter.py -q` → **3 failed, 7 passed**；`ls docs/benchmarks/results/positivity_limiter` → 不存在 |

D1 的性质要点：这不是「快照过期」，而是**守卫本身空转**——一个只能在作者工作树里通过的漂移守卫，等价于没有守卫。这与本普查统计的失败模式同源（「绿而零断言」的镜像：**红而无缺陷**）。

## ② 方法

1. **普查 schema 2：路径可移植**（`tools/audit_gate_vacuity.py`）
   - 新增 `_portable()`：仓库内路径一律序列化为**相对 `ROOT` 的 POSIX 路径**，仓库外路径保留绝对形式（并由政策门拒绝，因为无法从检出复现）；
   - `repo_root` 换成 `schema: 2`（显式声明路径语义，schema 1 = 绝对路径）；
   - 序列化面全覆盖：`modules[].artifacts[].path`、`modules[].skip_sites[].artifact`、`vacuous_artifacts` / `untracked_artifacts` / `red_artifacts`、汇总键 `missing_artifacts` / `untracked_artifacts`。
2. **`--check` 从布尔升级为可执行诊断**
   - 比较**全部载荷**（schema/统计/模块逐条）而不只是 `modules`；
   - 同时校验 `docs/GATE-VACUITY-AUDIT.md` 与当前扫描的渲染一致（手改 MD 也算漂移）；
   - 漂移时打印 `added module … (VERDICT)` / `changed module …: A -> B` / 逐键新旧值——把「漂移」变成可执行的下一步，而不是让人猜。
3. **关掉最后一例 RED-ON-CLEAN**：生成被跟踪记录
   `python -m benchmarks.positivity_limiter --mesh 60,120 --degrees 1,2,3`
   → `docs/benchmarks/results/positivity_limiter/metrics.json`（18 条记录，含门所需的 N=60 一档）+ `profile_collapse.npz`。
4. **申报表清空**：`ALLOWED_RED_CLEAN` 的两条（`test_dg_advection_bench` 过期、`test_positivity_scaling_limiter` 已修）全部删除，仅保留「空表 = 零申报」的说明。
5. **政策门补 2 道（回归）**（`tests/test_gate_vacuity_policy.py`）
   - `test_census_carries_no_checkout_specific_path`：载荷中不得出现任何绝对路径字符串，且 `schema >= 2`——D1 的直接回归；
   - `test_drift_guard_passes_in_a_foreign_checkout`：在 `tmp_path` 构造**另一个绝对路径下的忠实检出**（git 携带、且普查会读的 `tests/`、工具、普查 JSON、以及所有被跟踪工件），`git init/add` 后要求 `--check` **绿**——D1/D2 的行为级回归（不是「路径看起来相对」，而是「换检出真的能过」）。
6. **不新增依赖、不碰求解器**：全部改动落在工具/政策门/证据工件/文档四类，`tensordg/` 求解器与其它门模块零改动。

## ③ 计算结果

判级分布（同一工具，同一树内容，一次只变一个因素）：

| 判级 | main @ HEAD (151 模块) | 本分支 (165 模块) | 变化归因 |
|---|---|---|---|
| VACUOUS | 0 | 0 | — |
| **RED-ON-CLEAN** | **1** | **0** | 本单修（positivity 记录落地） |
| LATENT | 7 | 8 | main 前进新增 1 模块（`test_p_refinement_criterion`） |
| UNRESOLVED | 8 | 10 | main 前进新增 2 模块 |
| HARDENED | 11 | 17 | main 前进 +5（`cks1998_tensorbench`/`lax_case_pool_limiter_arms`/`lax_reference_crosscheck`/`swe_wb_drift_law_time_scaling`/`artificial_viscosity`）+ 本单 +1（positivity 修复后由 RED 升 HARDENED） |
| PLAIN | 124 | 130 | main 前进 +6 |
| 扫描模块数 | 151 | 165 | main 前进 +14（非本单） |
| skip 位点（工件/环境） | 22 / 4 | 27 / 4 | main 前进 |
| 缺失工件（skip 会触发） | 0 | 0 | — |
| cwd 依赖工件常量 | 8 | 8 | 未在本单范围（见 ⑤） |
| 钉在 git 不携带工件上的 skip 门 | 0 | 0 | — |

门行为的受控对照（**同一棵干净检出** `/tmp/gv_pre`，只换工具+普查）：

| 实验 | 命令 | 修前 | 修后 |
|---|---|---|---|
| E1 漂移守卫在别人的检出 | `python tools/audit_gate_vacuity.py --check`（cwd=`/tmp/gv_pre`，HEAD 内容） | exit **1** `census drifted` | exit **0** `census fresh` |
| E2 政策门在作者工作树 | `pytest tests/test_gate_vacuity_policy.py -q` | **3 failed, 11 passed** | **16 passed** |
| E3 最后一例红门 | `pytest tests/test_positivity_scaling_limiter.py -q` | **3 failed, 7 passed** | **10 passed** |
| E4 守卫仍有牙（故障注入：往检出里放一个 skip 型门） | `--check` | — | exit **1**，并点名 `added module tests/test_zz_fault_injection_fixture.py (VACUOUS)` |
| E5 相邻门未被波及 | `pytest tests/test_dg_advection_bench.py -q` | — | **15 passed**（142.5 s） |

E4 的意义：修 D1 后守卫变绿，但必须证明它不是变成「恒绿」。故障注入后它按预期转红并给出模块名与判级。

## ④ 结果图

`docs/reports/figs/gate_vacuity_closure.png`：六个判级的模块数，main @ HEAD 与本分支并列（数据直接取自两份普查 JSON，零手抄）。

## ⑤ 注意点（坑 / 局限 / 适用范围）

1. **「跟踪」= 在 git 索引里**，不是「文件存在」。生成记录后必须先 `git add` 再 `--write`，否则普查仍把它判成 `RED-ON-CLEAN`（第一次重算就踩到：记录已在树里但未跟踪，判级不变）。这是工具语义，不是缺陷；写在这里是为了下一个人不再花 10 分钟。
2. **`--check` 的通过性还依赖「干净检出」等价性**：带未生成工件的本地工作树与干净检出会给出不同 `exists`。本单的政策门用「只复制被跟踪文件」的临时检出复现干净检出语义，因此该测试在 CI 与本地含义一致。
3. **未在范围**：8 个 `cwd` 依赖工件常量（`Path("validation/…")` 形态，从非仓根启动 pytest 会静默 skip）与 10 个 `UNRESOLVED`（路径静态不可解析）仍然存在。普查把它们**登记为未裁决**，不当作通过；它们不是红门，但也不等于已修。下一步的自然靶子是把 `cwd` 依赖降为 0（机械替换为 `Path(__file__).resolve().parents[1]`），并逐个把 `UNRESOLVED` 解析出来。
4. **本单不做**：不改 `tensordg/` 求解器、不改其它门模块、不动 `ALLOWED_VACUOUS`（空表保持空表）。
5. **精度判据不适用**：本单产出为静态普查与门行为，无数值解，故无「<3% 误差」项；下文 ⑥ 逐项给出可机器裁的定量判据替代。

## ⑥ 误差判定（逐项，机器可裁）

| 判据 | 目标 | 实测 | 是否达标 |
|---|---|---|---|
| 漂移守卫在异地检出 exit code | = 0 | E1: 0 | ✅ |
| 政策门失败数 | = 0 | E2: 16 passed / 0 failed | ✅ |
| RED-ON-CLEAN 模块数 | = 0 | 0（HEAD 为 1） | ✅ |
| VACUOUS 模块数 | = 0 | 0 | ✅ |
| 缺失工件数（skip 会真触发） | = 0 | 0 | ✅ |
| 普查中绝对路径条目 | = 0 | 0（HEAD：每个被解析工件都是绝对路径） | ✅ |
| 守卫故障注入灵敏度 | 注入即红且点名 | E4: exit 1 + `added module … (VACUOUS)` | ✅ |
| `--check` 与渲染 MD 一致性 | 手改即红 | 已实现（E4 中报 `… is not the rendering of a fresh scan`） | ✅ |
| 数值精度（vs 解析解/文献） | < 3% | 本单无数值解，不适用；证据是门行为（exit code / 判级计数） | 声明原因 |

## 复现（全部在本仓根执行）

```bash
python tools/audit_gate_vacuity.py --check          # 期望 exit 0, "census fresh"
python tools/audit_gate_vacuity.py --write          # 树变动后重算（先 git add 新工件）
python -m pytest tests/test_gate_vacuity_policy.py -q
python -m benchmarks.positivity_limiter --mesh 60,120 --degrees 1,2,3   # 重生成记录
python -m pytest tests/test_positivity_scaling_limiter.py -q
```