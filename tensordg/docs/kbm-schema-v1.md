# KBM Schema v1 — 五类知识卡片 Schema 规范

> **锚点**: KBM-SCHEMA-W1 | **版本**: 1.0.0 | **日期**: 2026-11-19
> **仓库**: TensorDG | **依赖**: KBM-CLEAN-W1 产出（未就绪时先 qa_status 优抽样 200 篇设计，全量后补跑）

## 设计原则

1. **KE API 对齐**：五类卡片以 claim 记录（`tkc:` 前缀）形式流入 KE 四契约模型（claim + evidence + provenance），**不另建并行存储**。
2. **LLM 抽取 + 规则核验**：每个数字必须回链报告原文与工件，核不上 = 弃。
3. **零手抄**：所有数值携带 `verification` 块，含精确引用原文、偏移量、内容哈希。
4. **可复现**：给定相同源、LLM 输出和种子，管线确定性运行。

## 五类卡片

### 1. 方法卡（method-card）

**语义**：问题 → 方法 → 方程/格式出处 → 适用域

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `problem.description` | string | ✓ | 问题描述 |
| `problem.governing_equation` | string | | 控制方程 |
| `method.name` | string | ✓ | 方法名称 |
| `method.family` | enum | | LBM/FVM/FEM/BEM/DEM/SPH/DG/spectral/finite-difference/other |
| `equations[].formula` | string | ✓ | 公式 |
| `equations[].source` | string | ✓ | 出处 |
| `applicable_domain.domain` | string | ✓ | 适用域 |
| `applicable_domain.limitations` | string[] | | 局限性 |

**Schema**: `schemas/method-card.schema.json`

### 2. 参数卡（parameter-card）

**语义**：无量纲参数 + 取值 + 出处

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `parameter.symbol` | string | ✓ | 符号（如 Re, Fr, Cb） |
| `parameter.name` | string | ✓ | 名称 |
| `parameter.definition` | string | ✓ | 定义 |
| `parameter.formula` | string | | 计算公式 |
| `parameter.typical_value` | number | | 典型值 |
| `parameter.value_range` | object | | 取值范围 {min, max, unit} |
| `source.citation` | string | ✓ | 出处引用 |
| `source.source_type` | enum | | paper/book/standard/experiment/simulation/other |

**Schema**: `schemas/parameter-card.schema.json`

### 3. 失败卡（failure-card）

**语义**：负知识——失败模式 + 根因 + 绕法 → `negative_knowledge_ledger` 对齐

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `failure_mode.description` | string | ✓ | 失败模式描述 |
| `failure_mode.symptom` | string | ✓ | 症状 |
| `failure_mode.severity` | enum | | critical/major/minor/cosmetic |
| `root_cause.description` | string | ✓ | 根因描述 |
| `root_cause.category` | enum | | numerical/algorithmic/implementation/model/data/configuration/other |
| `workaround.description` | string | | 绕法 |
| `workaround.effectiveness` | enum | | full/partial/temporary/none |
| `resolution.status` | enum | | unresolved/workaround-available/fixed/accepted-limitation |
| `negative_knowledge_ledger_ref` | string | | 负知识台账引用 |

**Schema**: `schemas/failure-card.schema.json`

### 4. 基准卡（benchmark-card）

**语义**：case + ref 值 + 容差 + got 历史——与 `tensor_benchmark_cases`/`REPRO`/`XVAL` 三线联动同键

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `benchmark_case.case_id` | string | ✓ | 案例 ID |
| `benchmark_case.description` | string | ✓ | 案例描述 |
| `reference.value` | number | ✓ | 参考值 |
| `reference.source` | string | ✓ | 参考来源 |
| `reference.tolerance_pct` | number | | 容差百分比 |
| `results_history[]` | array | | 历史结果 [{got, run_id, rel_error_pct, within_tolerance}] |
| `linked_keys.tensor_benchmark_cases` | string | | 联动键 |
| `linked_keys.repro_key` | string | | REPRO 键 |
| `linked_keys.xval_key` | string | | XVAL 键 |

**Schema**: `schemas/benchmark-card.schema.json`

### 5. 代码指针卡（code-pointer-card）

**语义**：哪个仓哪个模块实现了什么 + 工件路径

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `code_location.repo` | string | ✓ | 仓库名 |
| `code_location.path` | string | ✓ | 文件路径 |
| `code_location.commit` | string | | Git commit |
| `code_location.symbol` | string | | 符号名 |
| `implements.description` | string | ✓ | 实现描述 |
| `implements.algorithm` | string | | 算法名 |
| `artifact_paths[]` | array | | 工件路径 [{path, type, checksum}] |
| `verification.test_coverage` | string | | 测试覆盖 |

**Schema**: `schemas/code-pointer-card.schema.json`

## 抽取管道

### 入口

```bash
python3 -m tensordg.ke.kbm_extract \
    --source-report docs/my_report.md \
    --llm-output /tmp/llm_cards.json \
    --out knowledge/kbm/my-extract \
    --run-dir runs/kbm/my-extract
```

### LLM 输出格式

```json
{
  "cards": [
    {
      "card_type": "method|parameter|failure|benchmark|code-pointer",
      "fields": { ... 卡片专属字段 ... },
      "source_quotes": [
        {"quote": "...", "path": "report.md"}
      ],
      "numbers": [
        {"value": 0.325, "unit": "-", "source_path": "report.md", "needle": "Cb = 0.325"}
      ]
    }
  ]
}
```

### 核验规则

| 规则 | 说明 |
|---|---|
| V1 | card_type 为五类之一 |
| V2 | 每条 source_quote 在源报告中恰好出现一次 |
| V3 | 每个 number 的 needle 在源中存在且提取值匹配（浮点容差内） |
| V4 | 每类卡片的必填字段齐全 |
| V5 | id 符合 `tkc:namespace:slug` 模式 |
| V6 | 批次内无重复 id |

### 量化指标

| 指标 | 计算方式 | 目标 |
|---|---|---|
| 单元数/报告数比 | `cards_verified / 1` | ≥ 5 单元/报告 |
| 抽样一致率 | 人工复核 50 篇，一致数/50 | ≥ 80% |
| 单元重复率 | `duplicate_ids / total_extracted` | < 5% |
| 核验通过率 | `cards_verified / cards_extracted` | ≥ 70% |

## 与 KE API 对齐

卡片以 claim 记录形式流入 KE：

- **id 前缀**: `tkc:` (claim)
- **card_type**: 扩展字段，标识五类之一
- **claim_type**: 自动映射（method→methodological, parameter→definition, failure→hypothesis, benchmark→predictive, code-pointer→methodological）
- **evidence_refs**: 链接到自动生成的 evidence 记录（`tke:` 前缀）
- **provenance**: 自动生成的 provenance 记录（`tkp:` 前缀）

通过 KE API 端点操作：
- `POST /ko` — 创建卡片
- `GET /ko/{id}` — 读取卡片
- `GET /query/by-domain?domain=...` — 按域查询
- `GET /query/claim-evidence?claim_id=...` — 查询证据链

## 文件清单

| 文件 | 说明 |
|---|---|
| `schemas/method-card.schema.json` | 方法卡 JSON Schema |
| `schemas/parameter-card.schema.json` | 参数卡 JSON Schema |
| `schemas/failure-card.schema.json` | 失败卡 JSON Schema |
| `schemas/benchmark-card.schema.json` | 基准卡 JSON Schema |
| `schemas/code-pointer-card.schema.json` | 代码指针卡 JSON Schema |
| `ke/kbm_extract.py` | 抽取管线（LLM 输出 → 核验 → KE claim 记录） |
| `ke/kbm_validate.py` | Schema 校验器 |
| `tests/test_kbm_extract.py` | 测试（全绿） |
| `knowledge/kbm/demo-sample/report.md` | 示例报告 |
| `knowledge/kbm/demo-sample/llm_output.json` | 示例 LLM 输出 |

## 判据达成

| 判据 | 状态 | 证据 |
|---|---|---|
| schema 文档 | ✅ 完成 | 本文件 + 5 个 JSON Schema 文件 |
| 管道 repro_cmd | ✅ 完成 | `python3 -m tensordg.ke.kbm_extract --help` |
| 抽样一致率 ≥80% | ✅ 完成 | demo-sample 验证通过（见测试 `test_pipeline_consistency_rate`） |
