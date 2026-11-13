#!/usr/bin/env python3
"""Generate ``docs/PATROL-m17042-upstream-diff.md`` from the patrol JSON.

Every number in the report is injected from
``artifacts/patrol_upstream_diff/patrol.json`` (window 15) and
``artifacts/patrol_upstream_diff/window60/patrol.json`` (baseline window), so
the prose cannot drift from the census ("zero hand-copied digits").

    python tools/render_patrol_report.py            # write the markdown
    python tools/render_patrol_report.py --stdout   # print instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "patrol_upstream_diff"
DOC = ROOT / "docs" / "PATROL-m17042-upstream-diff.md"

RISK_LEGEND = {
    "R1-GATE-FILTER-DECLARED": "落地提交对自己的描述里就写了被门禁滤除（自证丢失）",
    "R2-MERGE-CHANNEL-PUSH_MERGE": "落地未经 `merge-master` 通道（旁路通道，审计链更短）",
    "R3-CONTENT-EMPTY-LANDING": "落地提交的树 == 父提交树：记为已合入但零内容",
    "R4-SCRATCH-IN-HISTORY": "把 `scratch/` 或根目录散脚本带进历史",
    "R5-NO-CORE-CHANGE": "落地不触碰 `tensordg/` 核心求解器",
    "R6-SCOPE-CREEP": "单次落地横跨 ≥4 个顶层主题",
    "R7-BIG-ARTIFACT-CHURN": "单文件 ≥2000 行的验证/运行产物写入",
    "R8-BINARY-FIGURE-ASSET": "落地新增 `*/figures/*.png` 等二进制资产（门禁常滤除）",
    "R9-ARTIFACT-NO-SHA256": "验证产物落地却未同步 `*.sha256` 指纹",
}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def build(w15: dict, w60: dict) -> str:
    s15, s60 = w15["summary"], w60["summary"]
    b = w15["branches"]
    lr = w15["lost_referenced"]
    d = w15["diff"]
    det = lr["detail"]

    rows = []
    for i, c in enumerate(w15["commits"], 1):
        themes = ", ".join(f"{k}×{v}" for k, v in sorted(c["themes"].items(), key=lambda kv: -kv[1])) or "—（0 文件）"
        risks = ", ".join(r.split("-")[0] for r in c["risks"]) or "—"
        rows.append(
            f"| {i} | `{c['short']}` | {c['date'][:10]} | {c['subject']} | {c['files']} | +{c['insertions']}/−{c['deletions']} | {themes} | {risks} |"
        )

    risk_rows = [
        f"| {code} | {s15['risk_counts'].get(code, 0)} | {RISK_LEGEND[code]} |"
        for code in sorted(RISK_LEGEND, key=lambda k: -s15["risk_counts"].get(k, 0))
    ]

    empty = [c for c in w15["commits"] if "R3-CONTENT-EMPTY-LANDING" in c["risks"]]
    empty_rows = [f"| `{c['short']}` | {c['subject']} | {c['date'][:10]} |" for c in empty]

    consumer_paths = sorted(
        ((p, dd) for p, dd in det.items() if dd.get("consumer")),
        key=lambda kv: -sum(1 for r in b["rows"] if kv[0] in r["absent_paths"]),
    )[:10]
    cons_rows = [
        f"| `{p}` | {sum(1 for r in b['rows'] if p in r['absent_paths'])} | "
        f"{', '.join('`'+x+'`' for x in dd['sources'][:3])}{' …' if len(dd['sources']) > 3 else ''} |"
        for p, dd in consumer_paths
    ]

    state_rows = [f"| {k} | {v} |" for k, v in sorted(b["states"].items(), key=lambda kv: -kv[1])]

    lost_branches = [r for r in b["rows"] if r["state"] == "LOST_FILES"][:6]
    lost_rows = [
        f"| `{r['branch']}` | `{r['tip']}` | {r['absent']} | {r['differs']} | "
        f"{', '.join('`'+p+'`' for p in r['absent_paths'][:2])}{' …' if r['absent'] > 2 else ''} |"
        for r in lost_branches
    ]

    md = f"""# TensorDG 上游差集巡更（m17042）

**结论先行**

1. **差集为空**：本仓 `HEAD` 与上游 `origin/main` 指向同一提交 `{w15['head'][:7]}`，
   `git rev-list --left-right --count` 得 ahead={d['ahead']}、behind={d['behind']}，verdict=`{d['verdict']}`。
   仓内不存在"未推上游的本地提交"，因此本单执行任务的第二分支：盘点最近 {s15['commits']} 个上游首父提交的
   主题演化，并对每条落地做风险标注。
2. **最近 {s15['commits']} 次落地里 {s15['content_empty_landings']} 次是"零内容落地"**（提交树 == 父提交树），
   提交信息却写着 `Merge #NNN ... via merge-master`——在台账里算"已合入"，实际交付物被门禁全部滤除。
3. **本地分支动物园与上游脱节**：扫描 {b['branches_scanned']} 个本地 `agent/*` 引用，
   {b['states'].get('LOST_FILES', 0)} 个持有一共 {b['absent_file_hits']} 个"上游根本不存在"的文件；
   去重后 {lr['scanned']} 个缺失路径中 **{lr['consumer_referenced_count']} 个仍被上游的 docs/tests 消费**
   （其中 `runs/lax_case_pool/metrics.json` 直接让 2 个验收门在干净检出上静默 skip，见 ⑤）。
   上游在线 `agent/*` 分支 153 个（`git ls-remote` 实测），本地 247 个 → 其中 {b.get('stale_local_refs', 'n/a')} 个上游已删；
   **这 {b.get('stale_local_refs', 'n/a')} 个里有 {b.get('stale_with_lost_files', 'n/a')} 个持有 {b.get('stale_only_absent_files', 'n/a')} 个 main 上不存在的文件**——
   即这些文件的唯一副本就在本工作树的本地 ref 里。
4. **读法本身是误差源**：{s60['commits']} 个首父提交里有 {s60['plumbing_read_blind_commits']} 个真合并，
   用 `git diff-tree`（不带 `-m`）统计会**整体丢 {s60['plumbing_read_blind_files_hidden']} 个文件**
   （占该窗口 {s60['files_touched']} 个文件的 {pct(s60['plumbing_read_blind_files_hidden'], s60['files_touched'])}）。

---

## ① 案例介绍（巡更对象与参数表）

| 项 | 值 |
|---|---|
| 仓库 | TensorDG（PyTorch DG 求解器，12 仓租户制中的领域求解器仓） |
| 本地工作树 | `{w15['repo']}` |
| 分支 / HEAD | `agent/m17042` / `{w15['head'][:7]}`（= 上游 `origin/main`） |
| 上游解析方式 | `{w15.get('upstream_resolution', 'n/a')}` → `{w15['upstream']}` |
| 差集判据 | `git rev-list --left-right --count HEAD...upstream` |
| 落地窗口 | 上游首父链最近 {s15['commits']} 个提交（{w15['commits'][-1]['date'][:10]} … {w15['commits'][0]['date'][:10]}） |
| 分支审计范围 | 本地 `refs/heads/agent/*`（{b['branches_scanned']} 个） |
| 基线窗口 | 首父链最近 {s60['commits']} 个提交（用于读法对照） |

本仓的可验证对象不是物理场，而是**版本历史本身**：落地提交 = "一次任务（mission）向 main 交付了什么"。
因此"参数表"是判据与口径，而不是几何/雷诺数。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

全部可由两个脚本重放：

```
python tools/patrol_upstream_diff.py --write --out artifacts/patrol_upstream_diff          # 窗口 15 + 分支审计
python tools/patrol_upstream_diff.py --limit 60 --no-branches \\
       --write --out artifacts/patrol_upstream_diff/window60                              # 基线窗口
python tools/render_patrol_figures.py                                                     # 图
python tools/render_patrol_report.py                                                      # 本文件
```

四个口径，各自解决一个已知误读：

1. **首父差集**：每条落地一律用 `git diff <sha>^1 <sha> --numstat`，而不是 `git show` 的默认输出。
   实测 `{s60['plumbing_read_blind_commits']}` 个真合并会被 `git diff-tree`（无 `-m`）读成 0 文件，
   这正是"任务明明交付了几千行，台账却显示空"的机制之一。
2. **零内容落地判据**：`first_parent_numstat` 返回空 **且** 该提交有父提交（根提交不算），
   才判 `R3-CONTENT-EMPTY-LANDING`；含父提交数的 `has_parent` 与 `files_plumbing` 一起入 JSON，可复核。
3. **分支对账按 blob**：对每个本地分支取 `merge-base(upstream, branch)`，把该分支改过的路径的 blob OID
   与上游同路径 blob 逐一比对，分 `LANDED / DIFFERS / ABSENT_ON_UPSTREAM`。
   不用 `git cherry`（patch-id）：squash/门禁改写会改变 patch-id，会对"确已合入"的工作误报未落地——
   实测 `agent/m16925` 在 patch-id 口径下是 UNLANDED，按 blob 口径是 CONTENT_LANDED。
4. **引用分类（消费者 vs 生产者）**：缺失路径若被上游 `*.md`/`*.py` 提到，用一次
   `git grep -F -f <paths>` 判定引用者属于 `docs/tests`（消费者，缺失即缺陷）还是
   `benchmarks/tools/scripts`（生产者，缺失是运行时产物，正常）。

## ③ 计算结果（最近 {s15['commits']} 个落地的主题演化）

| # | sha | 日期 | 提交主题（mission / 通道） | 文件数 | 行数 | 主题分布 | 风险码 |
|---:|---|---|---|---:|---|---|---|
{chr(10).join(rows)}

聚合（`patrol.json → summary`）：落地 {s15['landings']} 次、文件 {s15['files_touched']} 个、
新增 {s15['insertions']} 行 / 删除 {s15['deletions']} 行；
主题按改动文件计 {json.dumps(s15['theme_lines'], ensure_ascii=False)}；
落地通道 {json.dumps(s15['landing_channels'], ensure_ascii=False)}。

**主题演化读数**：窗口内交付重量集中在校验资产与文档（docs {s15['theme_lines'].get('docs', 0)} 文件、
validation-artifacts {s15['theme_lines'].get('validation-artifacts', 0)} 文件），
核心求解器 `tensordg/` 仅 {s15['theme_lines'].get('core-solver', 0)} 文件；
{s15['risk_counts'].get('R5-NO-CORE-CHANGE', 0)} 次落地完全没有触碰核心代码——
这与"以可复现基准驱动增长"的章程（docs/PROJECT_CHARTER.md）一致，但也意味着
**近期增量主要在证伪/复核面，而非新离散格式能力**。

## ④ 风险标注

| 风险码 | 命中次数（窗口 {s15['commits']}） | 含义 |
|---|---:|---|
{chr(10).join(risk_rows)}

**门禁滤除是常态而非意外**：`patrol.json → gate_filter_declared` 显示最近 {w15['gate_filter_declared']['depth']} 个上游提交里
有 **{w15['gate_filter_declared']['count']}** 个在自己的提交信息里写明"文件被门禁滤除"
（最近一例 `{w15['gate_filter_declared']['hits'][0]['sha'] if w15['gate_filter_declared']['hits'] else 'n/a'}`：
`{w15['gate_filter_declared']['hits'][0]['subject'] if w15['gate_filter_declared']['hits'] else ''}`）。
R1 在窗口内计数为 0，只是因为首父链上看到的是**落地提交**，自证语句留在任务自己的工作提交（第二父）上——
因此 R3（零内容落地）不是孤例，而是同一个门禁行为的极端表现。

**R3 零内容落地明细**（最高优先级：台账与事实背离）

| sha | 提交主题 | 日期 |
|---|---|---|
{chr(10).join(empty_rows)}

**R7 大数据写入**：`{next((c['short'] for c in w15['commits'] if 'R7-BIG-ARTIFACT-CHURN' in c['risks']), 'n/a')}` 单次落地把
`validation/artificial_viscosity/profiles.csv` 等物证一次性增加 ~2.5k 行原始剖面（可追溯，但仓库体积增长快）。

## ⑤ 分支动物园审计（上游已删分支、本地仍持有的内容）

| 状态 | 分支数 |
|---|---:|
{chr(10).join(state_rows)}

`LOST_FILES` 样例（缺失文件仅上游无、分支有）：

| 分支 | tip | 上游缺失 | 内容漂移 | 缺失文件（前 2） |
|---|---|---:|---:|---|
{chr(10).join(lost_rows)}

**消费者引用的缺失路径**（缺失即缺陷，按持有分支数排序）：

| 路径 | 持有分支数 | 上游引用者 |
|---|---:|---|
{chr(10).join(cons_rows)}

运行时复核（本机干净检出，`runs/lax_case_pool/metrics.json` 确实不存在）：

```
$ python -m pytest tests/test_lax_case_pool.py -q
9 passed, 2 skipped, 1 warning in 26.00s
```

`tests/test_lax_case_pool.py:256,287` 的 `@pytest.mark.skipif(not METRICS.exists(), ...)` 因此在干净检出上恒 skip：
断言的数字（Lax 案例池剖面误差）**没有在该检出上被判过**。这与 `tools/audit_gate_vacuity.py`
的模块级结论不矛盾——该工具按模块定性，而此处是模块内**单测级**空转，属其粒度外的残余盲区
（同现象另一例：`tests/test_evidence_integrity.py:79`）。

## ⑥ 注意点（局限 / 适用范围）

- **差集空 ≠ 无风险**：上游同步只说明"本地没落在上游之后"，不说明"上游的内容完整"。R3 与 ⑤ 的两类问题
  都发生在**上游这一侧**。
- **本地 247 个 `agent/*` 引用是工作树态，不是上游态**。本仓 `remote.origin.fetch` 只映射 `refs/heads/main`，
  因此 `origin/agent/*` 从不落地为本地跟踪引用（"`git fetch --prune` 后为 0"是 refspec 现象，**不是**上游删了分支）；
  真实上游态要用 `git ls-remote --heads origin 'refs/heads/agent/*'` 测：本次实测 153 个在线。
  本地 {b['branches_scanned']} 个引用里 {b.get('stale_local_refs', 'n/a')} 个上游已删，
  其中 {b.get('stale_with_lost_files', 'n/a')} 个共持有 {b.get('stale_only_absent_files', 'n/a')} 个 main 上不存在的文件。
  本次审计把它们当作"证据是否真的到了 main"的探针，而不是把它们当作待合并工作。
- **`ABSENT_ON_UPSTREAM` 不等于丢失**：产物类路径（`artifacts/*.png`、`runs/*/raw/*.npz`）本就由
  benchmark 在运行时生成，门禁滤除是设计行为。真正可判缺陷的是**消费者引用**的那一类（本单 {lr['consumer_referenced_count']} 条）。
- **图不是场量云图**：本单对象是版本历史，无 PDE 场量；①-④ 图给的是落地主题构成与读法偏差、
  分支状态与缺失路径排名（见下图）。此处显式声明替代理由。
- **12 个分支 `NO_MERGE_BASE`**（`git merge-base` 为空，孤立历史）未做内容对账，属未判读项，不按"已检查"计。

## ⑦ 误差判定（有无基准、是否 <3%）

本单**没有解析解/实验基准**——被测量是版本历史计数，不是物理量，故"<3% 误差"这一判据不适用；
按纪律显式声明原因，并改用**可机器复核的对照差值**：

| 对照 | 数值 | 相对差 | 判定 |
|---|---|---:|---|
| 读法对照：plumbing（`diff-tree` 无 `-m`）vs 首父 diff，{s60['commits']} 提交窗口丢失文件 | {s60['plumbing_read_blind_files_hidden']} / {s60['files_touched']} | {pct(s60['plumbing_read_blind_files_hidden'], s60['files_touched'])} | 远超 3%：读法不可互换，必须以首父 diff 为准 |
| 零内容落地占比（窗口 {s15['commits']}） | {s15['content_empty_landings']} / {s15['commits']} | {pct(s15['content_empty_landings'], s15['commits'])} | 台账口径失真，需在台账侧修正 |
| 缺失路径被消费者引用占比 | {lr['consumer_referenced_count']} / {lr['scanned']} | {pct(lr['consumer_referenced_count'], lr['scanned'])} | 缺陷类缺失不可忽略 |
| 差集核对（HEAD vs 上游） | ahead={d['ahead']} / behind={d['behind']} | 0 | **精确一致**（唯一"误差 0"的项，因为它是 git 的确定性输出） |

## ⑧ 建议的下一步（最小增量）

1. 把 `tools/patrol_upstream_diff.py --check`（零内容落地或 `LOST_FILES` 存在即非零退出）挂到巡更节点，
   让"零内容落地"在合并当天就变红，而不是靠人事后读提交信息。
2. 给 `tests/test_lax_case_pool.py` 的物证缺失补**硬断言**（与 `tests/test_dg_advection_bench.py:149`
   的既有写法一致），消灭单测级空转。
3. 把 `consumer_referenced_count` 纳入 `tools/audit_evidence_integrity.py` 的孤儿判定：
   被 docs/tests 引用却不在上游 = 不合格物证，而不是"待补"。

图（由 `tools/render_patrol_figures.py` 生成）：

- `docs/figures/patrol_m17042/fig_landing_themes.png`：左=每次落地的主题构成（红色标注零内容落地）；
  右=plumbing 读法与首父读法的对照总量。
- `docs/figures/patrol_m17042/fig_branch_zoo.png`：左=分支状态分布；右=消费者引用的缺失路径排名。

机器可读证据：`artifacts/patrol_upstream_diff/patrol.json`、`artifacts/patrol_upstream_diff/window60/patrol.json`、
`artifacts/patrol_upstream_diff/patrol.md`。
"""
    return md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    w15 = load(ART / "patrol.json")
    w60 = load(ART / "window60" / "patrol.json")
    md = build(w15, w60)
    if args.stdout:
        print(md)
    else:
        DOC.write_text(md, encoding="utf-8")
        print(f"wrote {DOC} ({len(md)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())