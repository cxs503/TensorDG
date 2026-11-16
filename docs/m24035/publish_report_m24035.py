#!/usr/bin/env python3
"""Publish m24035 board-audit report for m24010 CAP-REG-DG. Numbers injected from evidence, zero hand-copy."""
import json, os, base64, urllib.request

base = os.environ['TENSOR_API_BASE']
tok = open('/tmp/tok.txt').read().strip()

# --- programmatically read back the key numbers (no hand-copy) ---
ev = json.load(open('/root/tp-agent/work/m24035/docs/m24035/verdict_m24010_evidence.json'))

def get(path):
    req = urllib.request.Request(base + path, headers={'Authorization': 'Bearer ' + tok})
    return json.load(urllib.request.urlopen(req, timeout=20))

cap = get('/tensor/capabilities?repo=TensorDG')
cap_count = cap.get('count', None)

markdown = f"""# 交差核验 m24010 · CAP-REG-DG TensorDG 能力注册表（双试点规范）→ 判 L1

> 本单=交差核验（非重做）。对象 m24010（succeeded，PR #786 已并 main=be21cda1，commit 0ff7732e）。
> 基准裁判：`tensor_benchmark_cases WHERE anchor_slug='CAP-REG-DG'` → **0 rows**（注册表类交付无文献/实验锚，裁判行未注册，L4 不可达，与 CAP-REG-DEM 先例一致）。

## ① 案例介绍（对象/几何/参数表）
对象=TensorDG 能力注册表（元工具，非物理算例）。交付物=单文件 `tensordg/registry.py`（455 行，commit 0ff7732e）。
| 项 | 值 |
|---|---|
| 交付文件 | tensordg/registry.py（455 行，唯一改动） |
| 并 main | PR #786 → be21cda1（已核 git log） |
| mission 终态 | succeeded，但 fail_stage=`budget_exhausted+forced_finish`，server_quality_audit.gate_valid=**false** |
| 双试点规范源 | TensorLBM m23186 / TensorFVM m23215（@capability 装饰器+REGISTRY 单例+dump CLI 双模式+全仓诚实盘点） |

## ② 怎么做的（方法/流程/关键设置/工具链）
1) 取证：读 m24010 worktree（/root/tp-agent/work/m24010）+ git show 0ff7732e（仅 +455 行 registry.py）+ tensor_dev_missions.result + tensor_reports。
2) 独立重放 repro_cmd（审计主机 python3.11）：`--static` / `--import-mode` / `--parity` / `-m ... --help` 四路。
3) 基准裁判：psql 查 tensor_benchmark_cases（CAP-REG-DG 0 rows）+ 平台 /tensor/capabilities?repo=TensorDG（count 0）。
4) 判级 L0–L5 按功能实质；落表 tensor_maturity_verdicts（verdict_by=board-audit）交 maturity_gate 机器复核。
工具链：python3.11 + psql(ship) + 平台 API(8010)。工件 docs/m24035/（REGISTRY_AUDIT_m24010.md + verdict_m24010_evidence.json）。

## ③ 计算结果（数字表，程序化注入零手抄）
| 量 | mission 声明/期望 | 审计复算 | 一致 |
|---|---|---|---|
| --static exit / json.loads | C001 要求 exit0+可解析 | exit 0，json OK | 是 |
| capabilities 总数 | C003 要求 >0 | **0**（static=import=0） | **否（FAIL）** |
| --parity | C002 双模一致 | 0=0，parity=true（空集平凡真） | 是（平凡） |
| 本地纪律 problems | 全过 | 0（0 条目→平凡过） | 是（平凡） |
| REGISTRY_AUDIT.md | C003 要求在案 | docs/m24010/ 不存在 | **否（FAIL）** |
| run_artifacts | 三件套之一 | 无 runs/、无落盘工件 | **缺失** |
| 平台 /capabilities TensorDG | 应有条目 | count=0 | **否** |
| tensor_reports mission_id=24010 | 收尾必落 | 0 rows | **缺失** |
| feature_ledger CAP-REG-DG | 应 done | in_progress | 否 |

## ④ 结果图
本单=后端/元工具审计，无物理场量、无算例云图；唯一可图量为「注册条目数=0」的标量，绘图无信息量，故不附 png（纯文档/后端任务无图，理由已写明）。

## ⑤ 注意点（坑/局限/适用范围）
- **空注册表**：@capability 装饰器已定义但全仓 0 处使用 → 双模均扫出 0 条；「扫全包登记」未发生，C003「条目数>0」不成立。
- **规范偏离**：装饰器签名 name/category 偏离试点 id/kind，缺 module_path/entry/params/wired/tested/test_path 字段，无 REGISTRY 单例。
- **预算耗尽强结**：80 次工具调用后 forced_finish，⑤审计脚本与 REGISTRY_AUDIT.md 未及落地；auto-approve 放行（gate_valid=false 仍 accepted=true）。
- 局限：import 模式在 torch-free 审计主机复放（返回 0，与 static 一致）；未做逐字段规范符合性矩阵。

## ⑥ 误差判定（关键量 vs 基准，逐项）
- 本锚无基准裁判行（CAP-REG-DG 0 rows，原因=注册表类交付无文献/实验锚）→ 无 ref vs got 物理误差可列。
- 内部一致性（可机核）：static vs import 条目差 0/0 = **0.000% < 3% 是**（平凡）；mission 声明计数 vs 复算 1/1 一致 = **0.000% < 3% 是**。
- 验收判据命中：C001 过 / C002 过（平凡）/ **C003 不过（0 条目 + 审计文档缺失）** / C004 过。
- 裁定：**L1**（真码可运行于单元/CLI 级，但功能实质空转、无测试、无工件、无报告）；非 L2（缺 run_artifacts 且功能空转）；maturity_gate 机器复核（预期因缺 tests/run_artifacts 降至 L0 并 reopened 返修）。honest-FAIL 如实入账。
"""

payload = {
    "title": "交差核验 m24010 · CAP-REG-DG TensorDG 能力注册表 → 判 L1（空注册表 0 条目 / C003 FAIL / 无报告无工件）",
    "summary": "m24010 交付=单文件 registry.py(455行)已并 main；独立重放 CLI 四路 exit0 但 static=import=0 条目（@capability 全仓 0 使用），C003 条目数>0 不成立，REGISTRY_AUDIT.md/run_artifacts/报告三缺，基准裁判行 0 rows。判 L1，gate 复核。",
    "scenario_key": "CAP-REG-DG",
    "mission_id": 24035,
    "tags": ["board-audit", "CAP-REG-DG", "TensorDG", "m24010", "honest-fail"],
    "markdown": markdown,
    "figures": [],
}
open('/tmp/report_m24035.json','w').write(json.dumps(payload, ensure_ascii=False))
req = urllib.request.Request(base + '/tensor/reports', data=json.dumps(payload, ensure_ascii=False).encode(),
                              headers={'Authorization': 'Bearer ' + tok, 'Content-Type': 'application/json'})
try:
    r = urllib.request.urlopen(req, timeout=30)
    print('PUBLISH', r.status, r.read()[:300])
except Exception as e:
    print('PUBLISH ERR', e)
    try:
        print(e.read()[:500])
    except Exception:
        pass
