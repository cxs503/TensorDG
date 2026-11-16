# 交差核验 m24010 · CAP-REG-DG TensorDG 能力注册表（双试点规范）

审计单：m24035（board-audit）· 对象：m24010（succeeded，PR #786 已并 main=be21cda1，commit 0ff7732e）
基准裁判：`tensor_benchmark_cases WHERE anchor_slug='CAP-REG-DG'` → **0 rows**（注册表类交付无文献/实验锚，裁判行未注册；L4 不可达，同 CAP-REG-DEM 先例）。

## ① 验收三件套逐项核（独立复算，非自声明）

| 件 | 声明/期望 | 审计复算 | 判定 |
|---|---|---|---|
| run_artifacts 文件 | 应有 runs/ 运行工件 | m24010 worktree 无 `runs/`；mission result `git_status='?? tensordg/registry.py'`（finish 时未跟踪）；无持久化 stdout 工件 | **缺失** |
| benchmark ref vs got | 容差内数值对照 | `tensor_benchmark_cases` anchor_slug='CAP-REG-DG' → 0 rows；无 got 数值可核 | **缺失（无裁判行）** |
| repro_cmd 可执行 | 一键复现 | `python3 tensordg/registry.py --static` exit 0、stdout json.loads OK、total_count=0；`--parity` exit 0、parity 0=0（空集平凡真）；`python -m tensordg.registry --help` exit 0 | **可执行（但输出为空）** |

## ② 判级（L0-L5）

- 交付物=单文件 `tensordg/registry.py`（455 行，真码：@capability 装饰器 + --static AST 扫描 + import 扫描 + --parity + 本地纪律校验 + 单 JSON 输出）。
- 但**核心功能（全仓能力盘点）为空**：全仓无任何模块使用 `@capability` 装饰器 → static/import 双模均 0 条目；C003「条目数>0」不成立。
- 无 tests（tests/ 无 registry 测试）、无 `docs/m24010/REGISTRY_AUDIT.md`（⑤ 审计脚本+审计文档缺失）、无 tensor_reports（mission_id=24010 → 0 rows）、平台 `/tensor/capabilities?repo=TensorDG` → count 0（/sync 未获数据）、feature_ledger CAP-REG-DG(id=612) 仍 in_progress。
- 规范偏差：装饰器签名偏离双试点规范（name/category 而非 id/kind；缺 module_path/entry/params/wired/tested/test_path 字段）；无 REGISTRY 单例。
- 裁定：**L1**（代码真实、CLI 级可运行=单元/单元验证级；非 L2——run_artifacts 缺失且功能主体（能力清单）为空，C003 失败）。

## ③ 验收四判据

| 判据 | 结果 |
|---|---|
| C001 --static exit 0 且 stdout 可 json.loads | **PASS**（独立复跑验证） |
| C002 --parity 双模一致或差异如实 | **PASS（空集平凡真，0=0）** |
| C003 条目数>0 如实；本地纪律全过；REGISTRY_AUDIT.md 在案 | **FAIL**（条目数=0；REGISTRY_AUDIT.md 不存在） |
| C004 PR 并 main | **PASS**（PR #786 merged） |

## ④ 结论

verdict 落表 level=1（board-audit）；maturity_gate 机器复核。三件套 2/3 缺失（run_artifacts、benchmark 数值），报告六段不适用（报告不存在）。honest-FAIL 如实入账。
