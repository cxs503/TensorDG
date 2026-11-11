# 证据完整性审计：陈旧证据与孤儿证据（TensorDG）

> 任务 m14698（verifier 席自找活，锚点=NEW:证据完整性欠账）。
> 引擎 `tools/audit_evidence_integrity.py`；闸门 `tests/test_evidence_integrity.py`；
> 棘轮注册表 `validation/evidence_registry.json`；机器报告 `artifacts/evidence_integrity_audit.json`。
> 本页所有数字由脚本从上述 JSON 直读渲染，零手抄。

## 1. 案例：验证证据会自己烂掉

机器可读工件（`validation/**`、`artifacts/**`、`docs/benchmarks/results/**` 下的 `*.json`）
只有在**被读**且**可溯源到产出它的代码**时才是证据。两种失效模式不会触发任何测试红灯：

- **孤儿证据（orphan）**：没有任何测试/工具/基准/文档读它。求解器演进了，文件照旧宣称旧数字。
- **陈旧证据（stale）**：文件被读——甚至被文档称作“唯一真值源”——但由已不存在的代码产出。

本任务在 TensorDG 里找到了第二种的真实例：`validation/swe_dam_break/metrics.json` 的
`ritter` 块。`docs/swe_verification.md` 把它声明为“所有数字都从这里读回”，而那里写着
**28.546%**（dry-guard 之前的求解器在 t≈0.076 s 静默停步，测的是初值误差），
当前求解器实测 **0.2395%**——两者差 **119.2 倍**，
而闸门 `tests/test_swe_dam_break.py`（<3%）一直绿。

## 2. 怎么做的

1. **引擎**：扫描三类工件根目录，判定“被引用”（路径/文件名/父目录出现在 tests、tools、benchmarks、
   tensordg、validation、docs 的文本语料中；过于通用的文件名如 `metrics.json` 需父目录名同时出现），
   再判定 provenance 等级（`block` / `per_block` / `none`）。引擎**不把自己产出**（注册表、审计报告）
   算作消费者——否则一注册就把孤儿洗成“被引用”，测量能力归零（`SELF_OUTPUTS`）。
2. **棘轮注册表**：已知孤儿逐条登记 `status` + `reason`；新增未登记孤儿使 `--check` 退出码 1，
   注册表行消失（工件被删）同样报错。
3. **修复**：`validation/swe_dam_break/study.py` 增加 provenance 印章（UTC 时间/命令行/git commit/
   dirty/版本/机器）与 `--only ritter` 分块刷新；旧值**不删除**，降级为 `ritter.legacy_pre_dry_guard`
   （仍是 `docs/swe_verification.md` 第 2 节 before 栏的来源），未被重测的块显式列入
   `blocks_not_remeasured_in_this_run`。
4. **绑定闸门**：`tests/test_evidence_integrity.py` 断言工件带 provenance、`time_reached == final_time`、
   rel L1 < 3%、legacy 值保留且 ≥10× 当前值、**文档引用的数字与其声明的真值源一致**（doc↔artifact 绑定）。

## 3. 结果（数字直读 `artifacts/evidence_integrity_audit.json`）
| 指标 | 值 |
|---|---|
| artifacts | 121 |
| orphans | 13 |
| provenance_block | 8 |
| provenance_env_only | 0 |
| provenance_none | 113 |
| provenance_per_block | 0 |
| referenced | 108 |
| unreadable | 0 |

已登记孤儿 13 条（见表 3.1）。**provenance 覆盖率 8/121** 是本次审计暴露的最大欠账：
113 个工件无法证明由当前代码产出（其中多数被闸门读取，数字本身有测试背书，但“这份文件是哪次运行写的”不可考）。

### 3.1 已登记孤儿（棘轮）
| 工件 | 状态 | 理由 |
|---|---|---|
| `validation/euler_dg_2d/probe_galilean.json` | candidate_retirement | 树内无写入者也无读取者的遗留伽利略不变性探针记录（探针脚本未入库）；数字不可溯源，仅作历史对照，不得引用为当前证据。 |
| `validation/euler_dg_2d/probe_mesh.json` | candidate_retirement | 树内无写入者也无读取者的遗留网格收敛探针记录；同上，不可溯源。 |
| `validation/gate_power/advection_1d_p1.json` | archival | 增量写盘的逐算例原始记录（benchmarks/audit_gate_power.py）；pytest 闸门（tests/test_gate_power.py）读聚合件 validation/gate_power_audit.json，逐算例文件是聚合件的溯源。 |
| `validation/gate_power/advection_2d_p1.json` | archival | 同上：逐算例原始记录，聚合件 gate_power_audit.json 才是闸门输入。 |
| `validation/gate_power/heat_1d_p2.json` | archival | 同上：逐算例原始记录，聚合件 gate_power_audit.json 才是闸门输入。 |
| `validation/gate_power/lax_p3.json` | archival | 同上：逐算例原始记录，聚合件 gate_power_audit.json 才是闸门输入。 |
| `validation/gate_power/poisson_2d_p1.json` | archival | 同上：逐算例原始记录，聚合件 gate_power_audit.json 才是闸门输入。 |
| `validation/gate_power/sod_p0.json` | archival | 同上：逐算例原始记录，聚合件 gate_power_audit.json 才是闸门输入。 |
| `validation/gate_power/vortex_p0.json` | archival | 同上：逐算例原始记录，聚合件 gate_power_audit.json 才是闸门输入。 |
| `validation/limiter_2d_order/degree1.json` | archival | benchmarks/limiter_2d_order_study.py 的研究输出；闸门 tests/test_limiter_2d_order.py 直接从模块重算，不读该文件。 |
| `validation/limiter_2d_order/degree1_bracket.json` | archival | 同上：研究输出，闸门重算不读盘。 |
| `validation/limiter_2d_order/degree2_bracket.json` | archival | 同上：研究输出，闸门重算不读盘。 |
| `validation/ns_wall_diag/tool_convention_baseline.json` | archival | 壁面诊断工具约定的历史基线；被闸门的约定审计是 validation/ns_wall_diag/convention_audit.json（tests/test_ns_wall_diag_binding.py）。 |

### 3.2 Ritter 干床溃坝（before / after，同一 200 格 P1 配置）
| 量 | 旧（legacy_pre_dry_guard） | 新（当前求解器） |
|---|---|---|
| rel L1(h) | 28.546 % | 0.2395 % |
| h_min | 6.780e-13 | 9.154e-07 |
| 实际到达时刻 | ≈0.076000 s（请求 0.5） | 0.500000 s |
| 时间步数 | — | 5711 |
| provenance commit | 无（旧文件无 provenance 块） | 1414a2e（dirty=True） |

## 4. 图

本任务为**证据元数据审计**（文件级引用关系与 provenance 字段），无场量、无基准曲线，故不产出云图/比较曲线；
替代可视化是上面两张表 + 机器报告 JSON（可 diff、可机器复核）。这是显式说明的无图理由，不是遗漏。

## 5. 注意点与局限

- **引用判据是文本级的**：模板式引用（`validation/gate_power/<case>.json`）会被判为孤儿，这是刻意的
  保守方向（宁报可疑，不放过真孤儿）；13 条登记项已逐条人工核对写入理由。
- **provenance 覆盖率不是闸门**：`113` 个无 provenance 工件不会被本闸门判红，
  否则一次性红满屏反而失去信号；本任务选择“新产出必带、旧产出逐批偿还”的路线，
  已偿还有据的是 `validation/swe_2d_wb`、`validation/swe_thacker`、`validation/swe_dam_break`。
- **未重测的块**：`--only ritter` 未重测 `stoker` / `lake_rest_1d` / `lake_rest_2d`（重测需十余分钟
  算力，本机 load15 已 >100，按容量纪律只做轻量活）；这些块已在 `blocks_not_remeasured_in_this_run`
  中显式声明，闸门要求该声明存在且不含 `ritter`。
- **文档绑定容差 10%**：`docs/swe_verification.md` 的 0.249% 来自 `benchmarks/ritter_dam_break.py`
  （另一条独立实现，干床播种/质量口径不同），本次 study 口径给出 0.2395%；
  两者相对差约 4.0%，在容差内，但它们是**两个独立估计量**，不应互相替代。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| Ritter 干床 rel L1(h)（当前求解器, 200 格 P1, t=0.5） | Ritter (1892) 精确自相似解 | 0.2395 % | **是** |
| 同一量（旧工件所称） | 同上 | 28.546 % | 否（且该值测的是初值误差，非求解器误差） |
| 文档-工件一致性（0.249% vs 0.2395%） | doc↔artifact 绑定 | 4.0 % 相对 | 是（<10% 容差） |

结论：本次修复把“声明为真值源却相差 100 倍”的陈旧证据消掉，并把该绑定做成机器闸门；
残余欠账（113 个工件无 provenance）如实登记，未粉饰。
