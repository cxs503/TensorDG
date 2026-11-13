# 交差审计：m17539 · XVAL-DG-ROUND2-CERTIFIED-REFERENCE（TensorDG）

- 审计单：m17580（board-audit 席）
- 被审对象：m17539（status=succeeded，claimed_by=tp-ver-3，auto-approve=dev_acceptance）
- 被审分支/提交：`agent/m17539` @ `92b2c41`（"TensorDG: reusable certified-reference engine (Roache GCI + gate-validity rule)"，9 files / +2151）
- 独立复算脚本：`artifacts/audit_m17580/independent_recheck.py`
- 复算结果：`artifacts/audit_m17580/independent_recheck.json`
- 结论：**L2 认定**（不消 L3/L4）；数值实质达 L4 强度，但集成与基准登记两道机器门均未过。

---

## ① 取证：三件套逐项核验

| 项 | 判据 | 核验结果 |
|---|---|---|
| run_artifacts 存在 | 交付自述 5 个工件 | **5/5 真实存在且非空**：`certified_reference_audit.json` 21131 B、`.md` 1455 B、`certified_reference_fv.npz` 10360 B、`fig1…png` 157306 B、`fig2…png` 77864 B。全部已在 `92b2c41` 落盘（`git ls-files` 命中，`git status` 干净） |
| benchmark ref vs got | 交付数字 vs 独立复算 | **逐位一致**：我用自写 `coarsen`/`rel_l1`（未 import 交付模块）复算 11 条 DG 腿，与交付 `rel_l1_percent` 的 `max_abs_delta = 0.0`（<1e-4 阈值） |
| repro_cmd 可执行 | 命令行能跑且能复现 | **通过**：`python -m tensordg.certified_reference --rawdir validation/repro_shu_osher/artifacts/raw --outdir /tmp/repro_m17539` 退出 0、打印完整 11 腿表；与仓内工件**逐字节相同**：`certified_reference_audit.md` sha256 `879c790f…` 两侧相同，`certified_reference_fv.npz` sha256 `ffcbca01…` 两侧相同，JSON 叶子级 diff = 0 |

### 基准裁判（唯一裁判）＝ tensor_benchmark_cases

```
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='XVAL-DG-ROUND2-CERTIFIED-REFERENCE';
--> 0 rows
```

该表全表仅 55 行，DG 相关仅 3 行（`CASE-SHU-OSHER-DG` id=19 passed、`CASE-DG-SOLID-BODY-ROTATION-2D` id=32、
`CAP-LEDGER-EVIDENCE` id=13）。**本锚点零登记**，故「按本锚点逐项核 got/容差」在制度上无法执行——
这不是「通过」，是**登记缺口**（同类锚点在 m17443 审计中已作为 F2 记录，本轮同一缺口仍在，未修）。

平台侧同向证据：

| 探针 | 结果 |
|---|---|
| `tensorbench_results WHERE mission_id=17539` | **0 行**（全表仅 4 行） |
| `GET /api/v1/tensor/bench/l4-verdict/17539` | `CLAIMED_NOT_PROVEN`，`l4_benchmark_pass=false`，`rows=0`，`failed_checks=[execution_success, source_cluster_job, cluster_job_bound, ref_got_error_present, within_tolerance]` |

### 退而用已登记的同算例裁判（CASE-SHU-OSHER-DG id=19，tol `<5%` rel L1，DG ladder N=200/400/800 × P1/P2/P3 CFL=0.4）

| 项 | 数值 | 判定 |
|---|---|---|
| 11 腿 rel L1（独立复算） | 最差 `DG P1 N=200 CFL=0.4` = **2.4557 %** | **11/11 < 5 %** ✓（且 11/11 < 3 %） |
| 参考解自身不确定带 U（Roache GCI，Fs=1.25） | **0.0828 %**，`U/τ = 1.66 % ≤ 1/3` | 门有效（非空判）✓ |
| FV 认证阶梯观测阶 | p = **1.282509**（`p_min=0.5, p_max=6`） | 可为认证者 ✓ |
| DG 阶梯自证 | dg_p1/p2/p3 分别在 `U/τ = 2.27/2.22/10.33 > 1/3` | 一律 `WITHHELD_ORDER_TOO_LOW`（**拒绝自证**，正确）✓ |

### 外部锚点（我独立实现 RH 与闭式质量预算，不复用交付代码）

| 锚 | 闭式/独立值 | 认证参考解实测 | 误差 | <1 % ? |
|---|---|---|---|---|
| Rankine–Hugoniot 平台区密度（Ms=3, γ=1.4） | 3.857142857 | 3.857143000（x ≤ −3.0 均值） | 3.7e-06 % | ✓ |
| 全局质量预算（仅 x=−5 入流） | 31.089226（= 初值 12.833894 + ρ₂u₂t = 18.255332） | 31.074065（梯形积分） | **0.049 %** | ✓ |
| 激波位置（解析 x₀+Wt = −4+3.54965×1.8） | 2.389367 | 2.396875（最陡负梯度） | **0.314 %** | ✓ |
| 未扰正弦区（x ≥ 2.6） | 1+0.2 sin 5x | max&#124;err&#124; = **1.04e-05** | — | ✓ |

文献数值（Shu & Osher, JCP 77 (1988) 439）原文只给图不给表 → **WITHHELD**，交付文档亦如实声明，无粉饰。

---

## ② 判级：L2

**级别＝功能实质，附报告不改变级别。**

| 级 | 判据 | 本单 | 依据 |
|---|---|---|---|
| L0 文档 | — | 已越 | — |
| L1 数学/单元验证 | 解析不变量被断言 | **达标** | `tests/test_certified_reference.py` 18 passed / 16 test 函数 / 46 assert，皆为真不变量：幂律阶梯阶数复原 `rel=2e-3`、Richardson 外推对解析误差 `<1e-10`、带/误差恒等 `rel=1e-6`、60 例覆盖率 `cov=1.0`、三条拒证路径（停滞/舍入地板/级数不足/带吃门）各自返回 `WITHHELD_*` |
| L2 独立功能实现 | 真代码 输入→计算→输出 | **达标（定为 L2）** | `tensordg/certified_reference.py` 617 行：`classify:206` / `certify:241` / `judge:364` / `audit_ladder:431` / `band_coverage_study:451` / `main:597`；CLI 入口可跑；工件在盘；审计员**自写度量**从原始 npz 复现全部关键数字 |
| L3 系统集成 | 接入主求解器/平台 | **不达（阻断）** | ①`from tensordg import certify` → `ImportError: cannot import name 'certify'`；②仓自契约门 `tests/test_public_api_contract.py` **RED，coverage 0.9614**，15 个未导出符号中 **11 个属本模块**（`coarsen/rel_l1/observed_order/richardson_extrapolate/…`）；③全仓调用方仅 `tests/` 与本模块自带 `tools/`，`tensordg/` 与 `benchmarks/` 零引用；④未并入 `main`（`main` 上只有另一单的 `docs/XVAL-DG-ROUND2-CERTIFIED-REFERENCE.md`，无本引擎） |
| L4 基准验证 | 完整算例+误差分析+可复现 **且** 平台基准行 | **不达（阻断）** | 锚点登记 0 行 + `tensorbench_results` 0 行 → 机器门 `CLAIMED_NOT_PROVEN`。**注**：其数值实质（完整算例、11 腿逐项误差、GCI 不确定带、门有效性规则、逐字节可复现、三族闭式外锚）已达 L4 强度，但级别=功能实质+机器门，不越级 |
| L5 科研工程成果 | — | 不达 | 单一算例工程化，未形成多算例可迁移流水线 |

**与先例一致**：同锚点前次审计（verdict id=560，m17443 审 m17365）亦判 **L2 / request changes**，两轮判据相同、结论相同；本单情况优于先例之处仅在于**交付已提交**（先例 F1 是 untracked + head_commit=null）。

---

## ③ 发现（按严重度，附证据与复现）

### B1 阻断 L3（高）：引擎不可从包根导出，且使仓自契约门新增 11 个缺口

- 复现：`cd /root/tp-agent/work/m17539 && python -c "from tensordg import certify"`
  → `ImportError: cannot import name 'certify' from 'tensordg'`；
  `python -m pytest tests/test_public_api_contract.py -q` → `1 failed`，
  `AssertionError: 15 public symbols are not exported; first offenders: [('tensordg.certified_reference', ['coarsen','rel_l1','observed_order','richardson_extrapolate']), ('tensordg.convergence', [...])]`，`assert 0.961439588688946 == 1.0`。
- 定量：15 个未导出符号中 11 个来自本模块，4 个为既有 `tensordg.convergence` 缺口 → **契约门恶化 11 个符号**。
- 影响：外部用户无法 `from tensordg import certify`；只有自带测试与渲染器能用，属「库内孤岛」。

### B2 阻断 L4（高）：锚点零登记 + 平台零基准行

- 复现：`tensor_benchmark_cases` 按本锚点查询 0 行；`tensorbench_results` mission 17539 = 0 行；
  `GET /api/v1/tensor/bench/l4-verdict/17539` → `CLAIMED_NOT_PROVEN`。
- 影响：无论数值多好，L4 机器门无法确认；且后续任何单都得重复人工裁决。**这是可修的制度缺口**（补登记锚点 + 落 `tensorbench_results` 一行）。

### M1（中）：因果表述低估了本模块对 RED 门的贡献

- 原文（`tensordg/certified_reference.py:80-86`）：`tests/test_public_api_contract.py is RED on this branch **for reasons that predate this module**`。
- 实测：新增前缺口 4 个，新增后 15 个；**73 % 的缺口由本模块引入**。下一句虽自陈「adds its 11 public names to that list」，但首句的成功归因（全部归因于历史）不成立。
- 严重度中（非造假，属表述失衡）：诚实披露存在，但读者会误判该模块对契约门是中性。

### M2（中）：result 台账与实物不一致，审计被迫回 git 求解

- 实物：交付已提交为 `92b2c41`（`git ls-files` 命中 5 个工件 + 3 个源文件，`git status` 干净）。
- 台账：`tensor_dev_missions.result.evidence.head_commit = null`、`server_quality_audit.gate_valid=false / key_present=false`；
  `result.on_disk_changes` 漏列已提交的 `fig2_band_share_and_legs.png`。
- 影响：若审计只读台账会误判为「未提交」（先例 m17365 正是此种）；本单从 git 侧纠正为「已提交但未合并 main」。台账漂移是可复现的系统性风险。

### L1 级（低，不阻断）

- 引擎的「与算例无关」仅由一个物理算例（Shu–Osher）+ 合成幂律阶梯支撑，**无第二个物理算例**；通用性主张按现有证据偏强。
- 文献数值对比 WITHHELD（原文只给图），已如实声明，无过度主张。

---

## ④ 未跑项（如实登记，记 unknown/not_run，不记 passed）

1. DG 求解器原始产出**未从零重跑**（本轮只复算其落盘剖面；先例 m17443 曾重跑 OSS 腿 N=800 逐位相同，但那是 OSS 而非本单 DG 腿）。
2. FV-WENO5 认证阶梯**未从零重跑**（只复算落盘 npz；单级 wall_time 记录 117.7 s，集群重跑未做）。
3. `test_public_api_contract.py` 的 RED **未做 main 基线对照**（未在 `main` 上跑同一测试取 before/after 覆盖率；缺口分解为静态计数）。
4. 文献数值逐点对比 WITHHELD（无表可对）。

## ⑤ 处置（disposition）

**request changes**：L2 认定，**不消账 L3/L4**。返修三件（按性价比排序）：

1. 在 `tensordg/__init__.py` 走别名表导出 `certify/judge/classify/audit_ladder/…`，令 `test_public_api_contract.py` 不再新增缺口；
   至少把 11 个新符号中的公共 API 面（`certify`、`judge`、`classify`、`audit_ladder`）导出并加 `from tensordg import certify` 的测试。
2. 给 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE` 补 `tensor_benchmark_cases` 登记（建议：`case_name=CASE-SHU-OSHER-DG / metric=rel L1 of 11 DG legs vs certified FV-WENO5 reference at same cell count / tolerance=<5%`），
   并把 11 腿 ref/got/error 落一行 `tensorbench_results`（含 repro 命令与工件 sha256）→ 过 `l4-verdict`。
3. 修正 `tensordg/certified_reference.py:80-86` 的因果表述，写明「本模块新增 11 个未导出符号」；
   同时修 `result.evidence.head_commit` 与 `on_disk_changes` 的台账漂移。

## ⑥ 审计独立性声明与限制

- 独立性：全部数值由本席自写公式（`coarsen` 单元平均、`rel_l1`、RH 密度比、`u₂`、闭式质量预算）从原始 npz 复算，**未 import 交付引擎**；交付引擎仅在「复现其 CLI 输出」与「逐字节比对工件」时运行。
- 限制：未重跑求解器本身（见④）；DG 腿误差与认证参考解同源于同一批落盘剖面，若落盘剖面被伪造则本审计不能发现（仅能证明「交付数字 ↔ 落盘剖面 ↔ 公式」三者自洽）；针对伪造的独立证据来自闭式外锚（RH/质量/激波位置/未扰正弦四项均在 1 % 内），已显著提高伪造门槛。
- 一处自查纠错（留痕）：本席最初把 Shu–Osher 初值质量误取为 ∫(1+0.2 sin5x)=10，得「质量错 3 倍」的假发现；
  实为初值在 x≤−4 处已处于激波后状态（ρ=3.857），且 t=1.8 内有 x=−5 入流。改用闭式预算 31.089226 后实测 31.074065（0.049 %）。
  **陷阱结论：Shu–Osher 的「质量守恒」锚不是 10，而是 mass₀+ρ₂u₂t；不做入流项会错判参考解为假。**