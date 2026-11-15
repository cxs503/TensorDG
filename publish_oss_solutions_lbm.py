#!/usr/bin/env python3
"""Publish the FEAT-OSS-INVENTORY-LBM wave-2 report (scenario_key=oss-solutions).

_numbers are read from docs/oss_solutions_lbm_registry.json (programmatic, no
hand transcription); figures are rendered here and embedded base64.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "tools"))
import oss_solutions_lbm_taxonomy as T  # noqa: E402

REG = json.loads((ROOT / "docs" / "oss_solutions_lbm_registry.json").read_text(encoding="utf-8"))
FIG = ROOT / "artifacts" / "oss_solutions_lbm" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def fig_stacked() -> pathlib.Path:
    cats = [c for c in REG["category_stats"] if c["n_entries"]]
    names = [c["name"] for c in cats]
    levels = ["direct", "adapt", "reference", "none", "unlabelled"]
    level_zh = T.MIGRATION_NAME
    per = {lv: [] for lv in levels}
    for c in cats:
        sub = [s for fp in REG["functional_points"] if fp["category"] == c["key"]
               for s in fp["solutions"]]
        for lv in levels:
            per[lv].append(sum(1 for s in sub if s["migration_level"] == lv))
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bottom = np.zeros(len(cats))
    colors = {"direct": "#2f7d31", "adapt": "#c98a00", "reference": "#3b6ea5",
              "none": "#a33a3a", "unlabelled": "#999999"}
    for lv in levels:
        vals = np.array(per[lv], dtype=float)
        if vals.sum() == 0:
            continue
        ax.bar(names, vals, bottom=bottom, label=level_zh[lv], color=colors[lv])
        bottom += vals
    ax.set_ylabel("entries")
    ax.set_title("LBM OSS method set: entries per functional-point category by migration level")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.xticks(rotation=25, ha="right", fontsize=8)
    fig.tight_layout()
    p = FIG / "lbm_oss_category_migration.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def fig_coverage() -> pathlib.Path:
    fps = REG["functional_points"]
    st = REG["stats"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    n_sol = sorted((fp["n_solutions"] for fp in fps), reverse=True)
    ax.plot(range(1, len(n_sol) + 1), n_sol, "o-", ms=3.5, color="#3b6ea5")
    ax.axhline(1, color="#a33a3a", lw=0.8, ls="--")
    ax.set_xlabel("functional points (ranked)")
    ax.set_ylabel("OSS implementations per functional point")
    ax.set_title(f"{len(fps)} functional points, {st['n_entries']} entries")

    ax = axes[1]
    labels = ["repos cited", "inventory repos", "entries w/ file ptr", "entries w/ line ptr"]
    vals = [st["n_repos_cited"], st["inventory_repos"], st["n_entries_with_file"],
            st["n_upstream_pointers"]]
    ax.barh(labels, vals, color=["#2f7d31", "#999999", "#3b6ea5", "#c98a00"])
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.set_title("coverage / pointer completeness")
    fig.tight_layout()
    p = FIG / "lbm_oss_coverage.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def md_report() -> str:
    st = REG["stats"]
    mig = st["migration_level_counts"]
    cov = REG["coverage"]
    L = []
    L.append("## ① 案例介绍")
    L.append("")
    L.append("LBM 是本生态里唯一已完成「24+ 开源项目代码级盘点」的求解器域（波 1 落 "
             f"`oss_inventory` {st['inventory_repos']} 仓）。但盘点条目按各任务当时的锚点散落：同一个功能点"
             "（例：插值 bounce-back、接触角边界、融合 stream+collide）的实现被写在 pylbm / lettuce / LUMA / "
             "walberla 等仓的不同 slug 下，开发单开工前无法一眼回答「这个功能点别人怎么做的、指针在哪、能不能搬」。"
             "本单（波 2）把这批条目按 **LBM 功能点** 重新归集为可检索的实现方式集，并落库供 PANORAMA 每单注入。")
    L.append("")
    L.append("参数表（全部由注册表程序化注入）：")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 源快照 `oss_solutions` 行数 | {REG['provenance']['source_rows']['n_rows']} |")
    L.append(f"| 源快照 sha256 | `{REG['provenance']['source_rows']['sha256']}` |")
    L.append(f"| 归集功能点 | {st['n_func_points']} |")
    L.append(f"| 方式集条目 | {st['n_entries']}（实现 {st['n_solution_entries']} + 反例 {st['n_gap_entries']}）|")
    L.append(f"| 涉及上游仓 | {st['n_repos_cited']} |")
    L.append(f"| 带文件指针 / 带行号指针 | {st['n_entries_with_file']} / {st['n_upstream_pointers']} |")
    L.append("")
    L.append("## ② 怎么做的")
    L.append("")
    L.append("1. `tools/build_oss_solutions_lbm.py --dump` 用 psql 把 `oss_solutions`(全量) 与 "
             "`oss_inventory`(33 仓) 导出为快照，sha256 入册（禁止手抄数字）。")
    L.append("2. 人工层只有一处：`tools/oss_solutions_lbm_taxonomy.py` 定义 "
             f"{st['n_func_points']} 个规范功能点（10 类，含 7 条「反例/不可迁移」）与每条目的源行选择器 "
             "`(repo, tag 子串[, slug 子串])`；选择器必须唯一命中一行，否则构建失败（防止归集漂移）。")
    L.append("3. 程序化提取：`upstream_ref` 由正则从源行 `files` 抽出 `path:line`；`migration_level` 由 "
             "`transfer` 前缀分类（direct/adapt/reference/none）；`method`/`files`/`transfer` **逐字保留**。")
    L.append("4. 产物：`docs/oss_solutions_lbm_registry.json`（注册表）+ `docs/OSS-SOLUTIONS-LBM.md`（对照报告）"
             "+ `artifacts/oss_solutions_lbm/insert.sql`（入库注入，`feature_slug=FEAT-OSS-INVENTORY-LBM`）。")
    L.append("5. 校验：`tests/test_oss_solutions_lbm_registry.py` 13 项断言（规模、选区唯一、逐字一致、"
             "指针格式、迁移级别枚举、统计自洽、SQL 字段数、再生确定性）。")
    L.append("6. 注入：SQL 已执行，`oss_solutions` 中 `feature_slug='FEAT-OSS-INVENTORY-LBM'` 现有 "
             f"{st['n_entries']} 行 / {st['n_repos_cited']} 仓；PANORAMA「开源实现方式集」表按 feature_slug "
             "聚合，故对全员自动可见。")
    L.append("")
    L.append("## ③ 计算结果")
    L.append("")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 功能点数 | {st['n_func_points']} |")
    L.append(f"| 条目数 | {st['n_entries']} |")
    L.append(f"| 覆盖 inventory 仓 | {st['inventory_repos_covered_by_solutions']}/{st['inventory_repos']} |")
    L.append(f"| 迁移级别 direct | {mig['direct']} |")
    L.append(f"| 迁移级别 adapt | {mig['adapt']} |")
    L.append(f"| 迁移级别 reference | {mig['reference']} |")
    L.append(f"| 迁移级别 none | {mig['none']} |")
    L.append(f"| 迁移级别 unlabelled | {mig['unlabelled']} |")
    L.append("")
    L.append("分类明细：")
    L.append("")
    L.append("| 类别 | 功能点 | 条目 | 仓 |")
    L.append("|---|---|---|---|")
    for c in REG["category_stats"]:
        L.append(f"| {c['name']} | {c['n_func_points']} | {c['n_entries']} | "
                 f"{', '.join(c['repos']) if c['repos'] else '—'} |")
    L.append("")
    L.append("## ④ 结果图")
    L.append("")
    L.append("图 1：分类 × 迁移级别堆叠柱（每条目按 `migration_level` 计数）；"
             "图 2：左=每功能点的 OSS 实现数排序（≥1 全部满足），右=覆盖/指针完备度。")
    L.append("")
    L.append("## ⑤ 注意点 / 局限")
    L.append("")
    L.append("1. 条目文本是**搬运**不是重写：`method/files/transfer` 逐字取自源行，本单不保证源行摘录本身的语义"
             "完全正确；因此新增指针探测工具 `tools/probe_oss_pointers.py`（浅克隆 + 回读 `path:line` 行内容）。")
    L.append("2. **指针未被实测**（not_run）：本机 `github.com` 出网被重置，探测须在有网计算节点跑；"
             "探测输入已固化在 `artifacts/oss_solutions_lbm/pointer_probe_input.json`（含 "
             f"{st['n_upstream_pointers']} 条行号指针）。未运行不等于通过。")
    L.append(f"3. inventory 中 {len(cov['inventory_repos_with_empty_summary'])} 仓 summary 为空且无任何 "
             f"oss_solutions 条目：{', '.join(cov['inventory_repos_without_solution_rows'])} —— "
             "这批仓的代码级盘点仍是欠账，本单未覆盖（不粉饰）。")
    L.append("4. 7 条 `LBM-GAP-*` 是反例（记录「该仓没有这个东西」），不计入新增能力，仅用于防误引。")
    L.append("5. SPHinXsys 的 SPH 条目属 SPH 域锚点，未纳入本 LBM 方式集。")
    L.append("")
    L.append("## ⑥ 误差判定")
    L.append("")
    L.append("本任务为代码级盘点，没有物理量基准，故按机械可验指标逐项判定（不宣称无意义的物理误差）：")
    L.append("")
    L.append("| 指标 | 目标 | 实测 | 是否达标 |")
    L.append("|---|---|---|---|")
    L.append(f"| 功能点数 | ≥40 | {st['n_func_points']} | {'是' if st['n_func_points'] >= 40 else '否'} |")
    L.append(f"| 条目数 | ≥40 | {st['n_entries']} | {'是' if st['n_entries'] >= 40 else '否'} |")
    L.append(f"| 覆盖项目数 | ≥24 | {st['n_repos_cited']} | {'是' if st['n_repos_cited'] >= 24 else '否'} |")
    L.append(f"| 每条目带文件指针 | 100% | "
             f"{round(100.0 * st['n_entries_with_file'] / st['n_entries'], 2)}% | "
             f"{'是' if st['n_entries_with_file'] == st['n_entries'] else '否'} |")
    L.append(f"| 其中带行号 | 越高越好 | "
             f"{round(100.0 * st['n_upstream_pointers'] / st['n_entries'], 2)}%"
             f"（{st['n_file_only_pointers']} 条源行只给文件级，未伪造行号）| — |")
    L.append(f"| 每条目带迁移评估 | 100% | 100%（构建强制非空）| 是 |")
    L.append("| 行号指针实测命中 | 高覆盖 | **not_run**（本机无外网）| 未知 |")
    L.append("| 再生确定性 | 逐字节一致 | `--check` 通过 + pytest 再生断言 | 是 |")
    L.append("")
    L.append("结论：**达到锚点量化验收（≥40 功能点 / ≥40 条 / 24 仓）**；"
             "唯一未满足项是「行号指针实测」，属 not_run 而非通过，已留工具+输入待有网节点补跑。")
    return "\n".join(L)


def main() -> int:
    f1 = fig_stacked()
    f2 = fig_coverage()
    body = md_report()
    import urllib.request

    tok_req = urllib.request.Request(
        f"{API}/auth/login",
        data=json.dumps({"username": os.environ.get("TENSOR_AGENT_USER", "eco_admin"),
                         "password": os.environ.get("TENSOR_AGENT_PASS", "")}).encode(),
        headers={"Content-Type": "application/json"},
    )
    tok = json.loads(urllib.request.urlopen(tok_req, timeout=60).read())
    token = tok.get("access_token") or tok.get("token") or tok.get("data", {}).get("access_token")
    payload = {
        "title": "FEAT-OSS-INVENTORY-LBM 波 2：LBM 开源实现方式集（按功能点归集 + 落库注入）",
        "summary": f"{REG['stats']['n_func_points']} 个 LBM 功能点 × "
                   f"{REG['stats']['n_entries']} 条方式集条目（{REG['stats']['n_repos_cited']} 仓，"
                   f"{REG['stats']['n_upstream_pointers']} 条文件:行指针），已注入 oss_solutions 供 PANORAMA 全员可见",
        "scenario_key": "oss-solutions",
        "mission_id": 21437,
        "tags": ["oss-inventory", "lbm", "method-set", "FEAT-OSS-INVENTORY-LBM", "reusable-knowledge"],
        "markdown": body,
        "figures": [
            {"name": "lbm_oss_category_migration.png",
             "caption": "分类 × 迁移级别堆叠柱：条目按 transfer 前缀程序化分类",
             "png_base64": base64.b64encode(f1.read_bytes()).decode()},
            {"name": "lbm_oss_coverage.png",
             "caption": "左：每功能点 OSS 实现数排序；右：覆盖与指针完备度",
             "png_base64": base64.b64encode(f2.read_bytes()).decode()},
        ],
    }
    req = urllib.request.Request(
        f"{API}/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=180).read())
    rid = resp.get("id") or resp.get("data", {}).get("id")
    print(json.dumps({"report_id": rid, "figs": [str(f1), str(f2)],
                      "resp_keys": sorted(resp)[:8]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())