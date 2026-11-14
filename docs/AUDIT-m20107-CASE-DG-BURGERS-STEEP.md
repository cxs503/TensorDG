# 交差审计 · m20053 `CASE-DG-BURGERS-STEEP`（TensorDG）→ **L3**

- **被审对象**：任务 m20053（worktree `/root/tp-agent/work/m20053`，HEAD `1a2e303c87138056ebb21febb825e6e97f7a5f35`
  `CASE-DG-BURGERS-STEEPEN: benchmark the steepening gradient peak against the exact characteristic solution`），
  状态 `succeeded`，报告 id 18283 / 18282（重复发布），case pool id=199。
- **审计任务**：m20107（board-audit 交差核验，非重做；不动被审代码）
- **审计者独立设备**：`artifacts/audit_m20107/recompute.py`（只用 numpy + csv，**不 import tensordg**）
- **裁判规则（任务单给定）**：`tensor_benchmark_cases` 是唯一裁判；报告好看不算数。

## 0. 结论摘要

| 项 | 结论 |
|---|---|
| 三件套（run_artifacts / ref-vs-got / repro_cmd） | **真存在**：7 件工件全部 `git ls-files` 跟踪（§2 sh­a256 表）；ref 为**闭式**解而非查表 |
| 判据数值 | 15 腿阶梯**逐腿审计侧独立重算 = 交付值**（`all_rows_recomputed_match=true`）；闭式 vs 审计侧稠密特征线求极大差 `2.5e-11` |
| **基准裁判（唯一裁判）** | **0 行**：`tensor_benchmark_cases` 无 `CASE-DG-BURGERS-STEEP`（也无任何 BURGERS 行，全库 0 条匹配）；`tensorbench_results` mission=20053 **0 行**；`GET /tensor/bench/l4-verdict/20053` = `CLAIMED_NOT_PROVEN` |
| 判级 | **L3（系统集成）**。L2/L3 实证达标；**L4 不开证**（裁判零判据） |
| 缺陷 | 1 中（测试门例配置 ≠ 交付门例配置）· 3 中低（自诊断未达 + 台账未回填 + 命名三写）· 1 低（报告重复发布） |

## 1. 基准裁判实况（决定级别上限）

任务单指定的查询，在平台库 `tensorplatform` 上执行（`root` 角色对 `tensor_benchmark_cases`
无 SELECT 权限，改以可读该表的角色复核，两种连接方式结果一致）：

```
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-DG-BURGERS-STEEP';
→ (0 rows)

SELECT count(*) FROM tensor_benchmark_cases
 WHERE anchor_slug ILIKE '%BURGERS%' OR case_name ILIKE '%BURGERS%';
→ 0                      -- 全库无任何 Burgers 锚点行

GET /api/v1/tensor/bench/results?mission_id=20053   → {"total": 0, "rows": []}
GET /api/v1/tensor/bench/l4-verdict/20053
→ {"status":"CLAIMED_NOT_PROVEN","l4_benchmark_pass":false,"rows":0,
   "failed_checks":["execution_success","source_cluster_job","cluster_job_bound",
                    "ref_got_error_present","within_tolerance"],
   "reason":"no tensorbench_results row for this mission: the L4 benchmark claim is
             self-declared, not measured"}
```

即：机器裁判面 **0 判据**。本单容差 `<5%` 只写在 mission 正文与 case pool `tol_policy`
（id=199 `<5%`）里，**没有**变成 `tensor_benchmark_cases` 的一行，也没有 `tensorbench_results`
的 ref/got/within_tolerance 三列。→ **L4 结构性不可能开证**，与报告质量无关。

台账侧同源缺口：`tensor_case_pool` id=199 `case_code='DG-BURGERS-STEEP'` 至今
`status='dispatched'`（`updated_at=2026-11-14 08:11:35`，即派单时刻），mission 已 `succeeded`
但**未回填 `passed`**。命名还存在三写：任务锚点 `CASE-DG-BURGERS-STEEP` / case pool
`DG-BURGERS-STEEP` / 交付 `case_key=DG-BURGERS-STEEPEN` —— 即便登记裁判行，按锚点字面匹配也会落空。

## 2. 三件套取证（真指针）

| 件 | 指针 | 核验结果 |
|---|---|---|
| impl_files | `tensordg/cases/burgers_steepening.py:275` `solve_case()`；`:301` 调**生产内核** `solve_burgers`（`:19` 明示复用 `burgers_1d.solve_burgers`：模态 Legendre + Rusanov + SSP-RK3，`:89` 导入）；`:224` `exact_gradient`、`:231` `exact_peak_gradient`（闭式 `G=S/(1-tS)`，`t≥t_b` 抛 `ValueError`） | **REAL**：算例模块只有参照是新的，求解器是既有生产代码（非桩、非硬编码） |
| entry | `benchmarks/burgers_steepening_peak.py:497` `main()`，`:604` `--render-doc`，`:320` `render_doc()`（文档由 `metrics.json` 渲染，零手抄）；`:236/:254` 直连 `tensordg.burgers_1d.{_operators,solve_burgers}` | **REAL，可执行**（§4 实跑） |
| run_artifacts | `artifacts/burgers_steepening_peak/{metrics.json, ladder.csv, time_sweep.csv, cache.json, fig_profiles_T.png, fig_peak_vs_time.png, fig_convergence.png}`（7/7 `git ls-files` 跟踪） | 存在且内容自洽（§3） |
| tests | `tests/test_burgers_steepening.py`（11 个 test 函数 / 18 个用例） | 审计实跑 **18 passed in 18.06 s**（§4） |
| integration_callers | `tensordg/cases/__init__.py:8` `from . import burgers_steepening`、`:62` 导出 `burgers_steepening_parameters` 等；`tensordg/cases/burgers_steepening.py:89` 反向依赖生产求解器 `tensordg.burgers_1d`；驱动 `benchmarks/burgers_steepening_peak.py` 仅经包 API 调用 | **REAL（库级集成）**：算例挂在包内 case 命名空间、复用生产 kernel；**平台级挂载缺**（§1） |
| benchmark | 仓内 `metrics.json.criteria` 三条 + `docs/CASE-DG-BURGERS-STEEPEN.md §④` 15 腿对照表 | 自建对照真实且可独立复算；**未登记平台裁判表** → 不构成 L4 证书 |
| repro_cmd | `PYTHONPATH=. python -m benchmarks.burgers_steepening_peak --render-doc`；契约测试 `PYTHONPATH=. python -m pytest tests/test_burgers_steepening.py -q` | **可执行**（测试侧实跑；driver 侧因 `cache.json` 命中会跳过重算，审计未重刷被审工作树，见 §6 not_run） |
| external_anchor | Whitham, *Linear and Nonlinear Waves*, Wiley 1974, ch.2（特征线 `u=u_0(ξ), ξ=x-tu_0`）→ 求导得 `du/dx = u_0'/(1+t u_0')`，`t_b=1/(1.6π)=0.1989437`，`T=0.19` 时 `G=111.8108`（`T/t_b=0.955`） | **REAL 外部锚点**：公式级引用（章级），无数值抄录；审计侧用**独立的稠密特征线求极大**复核到 `2.5e-11`（§3） |
| 工件 sha256（被审 HEAD `1a2e303`） | `metrics.json 91e34bc8…`、`ladder.csv 14c8de43…`、`time_sweep.csv 288fe7d7…`、`cache.json 96ec0cdc…`、`fig_profiles_T 9ff05c70…`、`fig_peak_vs_time 4050a0a5…`、`fig_convergence e9125455…` | 7/7 记录在案，可回归比对 |

## 3. 逐项核 got vs 容差（审计侧独立重算，不采信交付 verdict 块）

`artifacts/audit_m20107/recompute.py`（**不 import tensordg**）：以 2^21 点稠密特征线参数化
`x=ξ+t·u_0(ξ)`、`g=u_0'/(1+t u_0')` 直接求 `max|g|`，再与 `ladder.csv` 的 15 腿 `got` 逐腿比较。

- 参照自检：闭式 `G=111.81081243066606` vs 审计侧稠密数值极大 `111.81081242787543`
  → 相对差 **−2.50e-11**；峰位 `x=0.538000` vs 交付 `0.538023`（半格内）。
- 阶梯自洽：**15/15 腿** `relative_peak_error` 与交付值差 < 5e-9（`all_rows_recomputed_match=true`）。

| 判据（交付 `criteria`） | 阈值 | got | **审计独立重算** | 判定 |
|---|---|---|---|---|
| 峰值相对误差 @ 最细档 `N=1024 p=3` | < 5 % | +0.5607 % | **+0.5607 %**（112.43771 vs 111.81081） | PASS（且 <3%） |
| 同档全场 L2 相对误差 | — | 9.58e-07 | 由 CSV 复核一致 | PASS |
| 均值守恒误差（同档） | < 1e-12 | 1.66e-13 | CSV 复核一致 | PASS |
| 非平凡性：带外配置数 | ≥1 | 7 / 15 | **7**（p1: −43.60/−43.36/−18.00；p2: −28.33/+8.44；p3: −26.24/+14.44） | PASS（门不是空洞过门） |
| 时间步敏感性（CFL 0.2/0.1/0.05 @ 同一网格） | 可忽略 | 极差 5.8e-07 | `time_sweep.csv` 复核一致 | PASS |

**但峰值误差不是单调的**：p3 依次 −26.24 % → **+14.44 %** → −3.16 % → −0.41 % → **+0.56 %**；
p2 依次 −28.33 % → +1.13 % → +8.44 % → +0.90 % → +2.81 %。最细两档峰值变化 **0.97 %**，
交付自己的 `diagnostics.peak_change_between_two_finest = {value: 0.00975, declared_target: 0.005,
met: false}` —— 即**交付方自诊断未达**（如实写在工件里，未粉饰）。结论：`+0.561 %` 是一个
**非渐近采样点上的单点读数**（符号在 N=512/−0.41 % 与 N=1024/+0.56 % 之间翻转），
不能读作"离散误差已收敛到 0.56 %"。这削弱的是误差的*可外推性*，不削弱"该档在 5 % 带内"这一事实。

## 4. 复现性（审计实跑）

| 命令 | 结果 |
|---|---|
| `OMP_NUM_THREADS=1 python -m pytest tests/test_burgers_steepening.py -q`（在 `/root/tp-agent/work/m20053`） | **18 passed, 1 warning in 18.06 s**（wall 20.4 s）—— 与交付自陈「18 passed（19.5 s）」一致 |
| `python3 artifacts/audit_m20107/recompute.py`（本 worktree） | 15/15 腿逐腿一致，产出 `artifacts/audit_m20107/recompute.json` |
| driver `--render-doc` | **未在被审工作树里重刷**（`cache.json` 命中会重写 `docs/` 与工件，属改被审对象）。可执行性由 `main()`/`render_doc()` 存在 + 测试实跑 + 工件与 `git ls-files` 一致间接证明 → 记为 **partial**，见 §6 |

## 5. 缺陷（按严重度；均不推翻 L3 主结论）

**M1（中 · 测试所锁的门例配置 ≠ 交付声明的门例配置）**
交付 `criteria.peak_within_tolerance_at_finest` 与报告 ③/⑥′ 都把判据绑在
`N=1024 p=3`（+0.561 %）；但 `tests/test_burgers_steepening.py:107`
`test_documented_run_reproduces_the_peak_within_the_five_percent_band` 实际断言的是
**`cells=256, degree=3`**（该档交付误差 **−3.164 %**，距 5 % 带边只剩 1.84 pp）。
即：被测试锁住的是另一档；`N=1024 p=3` 这一行**只由工件自证**，无任何测试在回归时看守它。
若最细档劣化到 −4.9 %，测试仍全绿。→ 建议加一条以 `(1024,3)` 为键、从 `metrics.json` 读回的
回归断言（或让测试直接跑该档并比对工件）。

**M2（中低 · 交付自诊断未达，PASS 头衔未加限定）** 见 §3 末：`peak_change_between_two_finest`
`met=false`（0.97 % vs 声明目标 0.5 %）、p2/p3 峰值误差非单调（p3 在 N=128 处 +14.4 % 带外）。
`metrics.json.verdict` 仍是裸 `"PASS"`，未带 `converged: false` 之类的限定词。文档 §⑤ 有如实讨论，
但机器可读的 `verdict` 字段会误导只看该字段的下游（例如将来的裁判脚本）。

**M3（中低 · 台账未回填 + 锚点命名三写）** 见 §1：case pool id=199 `status='dispatched'`；
`tensor_benchmark_cases` 0 行；`CASE-DG-BURGERS-STEEP` / `DG-BURGERS-STEEP` /
`DG-BURGERS-STEEPEN` 三种写法并存。

**L1（低 · 报告重复发布）** `/tensor/reports` mission=20053 有 **2 条**内容相同的 published 记录
（id 18282 @08:36:03、18283 @08:36:18，标题/summary/figures 相同）。属发布器重试未去重，
不影响取证（两报告六段齐全，附图经 `/tensor/reports/18283/figures/fig_convergence` 可取到
92 212 字节 PNG，非空壳）。

**L2（低 · 口径命名）** 工件里有三套峰值读数（`polynomial` / `quadrature` / `naive_ratio`）与
三条参照（`closed` / `sampled` / `at_quadrature`），命名依赖读者读代码才知道谁是谁；
`reference_at_quadrature_bias = −17.81 %` 这类量建议在 `metrics.json` 里给出显式 `protocol` 标签。

## 6. 审计未做（honest not_run）

- **driver 全量重跑**：`--render-doc` 会重写被审工作树的 `docs/`+`artifacts/`（cache 命中时也是），
  审计遵"不改被审对象"未执行 → **WITHHELD**（测试侧已覆盖 kernel 与两档算例）。
- **N=1024 p=3 全新（非 cache）单腿重算**：未在本审计内跑（交付该腿墙钟 49.65 s，无 cache 亦可行，
  但会写被审工作树）→ 该档 got 值仅有"CSV/JSON 自洽 + 独立参照"支撑，**未独立重算求解器输出**。
- `time_sweep` 5 点时刻扫描的物理正确性：只做 CSV 内部一致性核对，未逐点独立复算 → **partial**。
- 平台侧 L4 通道（`tensorbench_results` 写入路径）：本审计不触碰写入端，无法验证登记后能否通过 → **not_run**。
- 我的 `recompute.py` 里"参照场的点采样口径"一项自建实现失败（x→ξ 映射在陡化后非单调，
  朴素 `np.interp` 反演得到 2.56，与交付 91.9/111.8 不可比）——该字段在 `recompute.json` 中保留但
  **标记无效**，结论**不依赖**它。

## 7. 级别裁定（L0–L5）

- **L1 · 达标**：参照闭式与破断时刻有单元级可证伪测试（`:26` 由数据导出 `t_b`/`S`、`:100` 破断后
  拒答、`:34` 闭式 vs 稠密数值极大）。
- **L2 独立功能实现 · 达标**：`solve_case` 输入（`cells/degree/final_time/cfl`）→ 调生产 kernel →
  输出峰值/误差/守恒量，真代码非桩；审计侧独立复算逐腿一致。
- **L3 系统集成 · 达标**：算例挂 `tensordg.cases`（`__init__.py:8,62` 导出），反向复用生产 solver
  `tensordg.burgers_1d.solve_burgers`（`:89/:301`），驱动脚本只经包 API 调用；工件由驱动
  程序化生成、文档由工件渲染（零手抄）。**平台挂载层面缺**：case pool 未回填成功、裁判表 0 行。
- **L4 基准验证 · 不开证（阻断）**：①唯一裁判 0 判据（§1：`tensor_benchmark_cases` 0 行、
  `tensorbench_results` 0 行、`l4-verdict` = `CLAIMED_NOT_PROVEN`）；②即便登记，"通过"仍需绑定
  门例说明（本单只在 `N=1024 p=3` 过 5 %，`N=256 p=3` = −3.16 % 余量 1.84 pp，p3@N=128 = +14.4 %
  带外），且最细两档仍在 0.97 % 漂移 → 属**条件达标**而非无条件达标。
- **L5 · 不适用**：无新方法/新物理；实质增量是"用梯度峰值补住 L2 均值范数看不见的陡化抹平"
  这一**门禁设计**（判据设计有价值，但不是新算法）。

**升 L4 的机械化路径（给后继单，非本单职责）**：在 `tensor_benchmark_cases` 为
`CASE-DG-BURGERS-STEEP`（并统一 case pool `case_code` 拼写）登记 1 行：
`metric='gradient peak max|du/dx| at T=0.19 vs closed form S/(1-TS), DG p=3 N=1024'`,
`tolerance=0.05`；并由平台 tensorbench 产 ≥1 条 `tensorbench_results`（ref/got/within_tolerance）。
同时把 `tests` 的门例改成 `(1024,3)` 以看守交付声明的那一档。

## 8. 复现本审计

```bash
python3 artifacts/audit_m20107/recompute.py            # 15 腿独立重算 + 参照自检
# 被审对象侧（只读）：
cd /root/tp-agent/work/m20053
OMP_NUM_THREADS=1 python -m pytest tests/test_burgers_steepening.py -q     # 18 passed
psql -U ship -d tensorplatform -tAc \
  "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug ILIKE '%BURGERS%'"   # 0
```