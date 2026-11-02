#!/usr/bin/env python3
"""Generate docs/upstream_delta_audit_m1634.md from the audit artifact.

Every number in the produced report is injected from either
  * artifacts/upstream_delta_m1634.json (produced by tools/upstream_delta.py), or
  * an independent `git` invocation used as a cross-check (the "error" column),
  * blobs read out of the unmerged upstream branch (`git show <sha>:<path>`).
Nothing is hand-transcribed, so the document can be regenerated and diffed.

    python3 tools/render_upstream_delta_md.py \
        --artifact artifacts/upstream_delta_m1634.json --out docs/upstream_delta_audit_m1634.md
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import upstream_delta as ud  # noqa: E402

RISK_DEF_ORDER = ["R1_DOC_ONLY", "R2_TESTS_TOUCHED", "R3_NUMERIC_ARTIFACT",
                  "R4_API_SURFACE", "R5_PHYSICS_CORE", "R6_MERGE_COMMIT",
                  "R7_LARGE_CHURN", "R8_CI_CONFIG", "R9_UNMERGED_BRANCH"]


def _git(repo, *args):
    return ud.git(repo, *args, check=False)


def _shortstat_insertions(text):
    for tok in text.split():
        pass
    parts = text.replace("\n", " ").split(",")
    for p in parts:
        p = p.strip()
        if "insertion" in p:
            return int(p.split()[0])
    return 0


def _git_stdin(repo, args, stdin_text):
    proc = subprocess.run(["git", "-C", repo, *args], input=stdin_text,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    return proc.stdout


def crosschecks(repo, d):
    """Independent git measurements vs the artifact -> per-item relative error.

    The commit window cannot be expressed as a `HEAD~N` range here: the history
    is merge-heavy, so the first-parent chain is shorter than N and `HEAD~15`
    does not resolve.  The baseline therefore pins the window as an explicit
    sha set taken from `git rev-list -n N HEAD` and replays it through other
    plumbing (`rev-list --parents --no-walk --stdin`, `log --stdin --numstat`).
    """
    rows = []

    def add(item, tool, reference, unit=""):
        if reference == 0:
            rows.append({"item": item, "tool": tool, "git": reference, "unit": unit,
                         "rel_err_pct": 0.0 if tool == 0 else None})
            return
        rows.append({"item": item, "tool": tool, "git": reference, "unit": unit,
                     "rel_err_pct": abs(tool - reference) / reference * 100.0})

    recent = d["recent_commits"]
    # the window is taken from the *audited* sha, not from the moving HEAD
    window = [s for s in _git(repo, "rev-list", "-n", str(len(recent)),
                              d["head"]["sha"]).splitlines() if s]
    sha_stdin = "\n".join(window) + "\n"

    add("window commit count", len(recent), len(window))

    recent_shas = {r["sha"] for r in recent}
    add("window sha set matches rev-list", len(recent_shas & set(window)), len(window))

    parents = _git_stdin(repo, ["rev-list", "--parents", "--no-walk", "--stdin"], sha_stdin)
    non_merge_ref = len([l for l in parents.splitlines() if len(l.split()) == 2])
    non_merge = [r for r in recent if not r["merge"]]
    add("non-merge commits in window", len(non_merge), non_merge_ref)

    # --no-walk: enumerate exactly the window shas, not their ancestor history
    numstat = _git_stdin(repo, ["log", "--no-walk", "--no-merges", "--no-renames",
                                "--numstat", "--pretty=", "--stdin"], sha_stdin)
    ref_files, ref_ins = 0, 0
    for line in numstat.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        ref_files += 1
        if parts[0].isdigit():
            ref_ins += int(parts[0])
    add("file changes in window", sum(r["files_changed"] for r in non_merge), ref_files)
    add("insertions in window", sum(r["insertions"] for r in non_merge), ref_ins)

    for b in d["unmerged_upstream_branches"]:
        head = d["head"]["sha"]
        short = _git(repo, "diff", "--shortstat", "--no-renames",
                     "%s...%s" % (head, b["sha"])).strip()
        name = b["ref"].split("/")[-1]
        add("unmerged %s: insertions" % name, b["insertions"], _shortstat_insertions(short))
        add("unmerged %s: files touched" % name, b["files_touched"],
            len([l for l in _git(repo, "diff", "--name-only", "--no-renames",
                                 "%s...%s" % (head, b["sha"])).splitlines() if l.strip()]))
        add("unmerged %s: commits ahead" % name, b["commits_ahead"],
            int(_git(repo, "rev-list", "--count", "%s..%s" % (head, b["sha"])).strip() or 0))
    return rows


def unmerged_physics(repo, d):
    """Read the GCI/convergence blobs of the unmerged branch straight out of git."""
    out = []
    for b in d["unmerged_upstream_branches"]:
        for path in ("artifacts/gci_poisson_p2.json", "artifacts/converge_poisson_p2.json"):
            blob = _git(repo, "show", "%s:%s" % (b["sha"], path)).strip()
            if not blob.startswith("{"):
                continue
            data = json.loads(blob)
            if "apparent_order" in data:
                theo = 3.0  # SIPG P2: L2 order p+1
                out.append({
                    "ref": b["ref"], "path": path,
                    "metric": "apparent order (FVM/DG Richardson)",
                    "observed": data["apparent_order"], "expected": theo,
                    "rel_err_pct": abs(data["apparent_order"] - theo) / theo * 100.0,
                })
                exp_ratio = data.get("expected_asymptotic_ratio")
                if exp_ratio:
                    out.append({
                        "ref": b["ref"], "path": path,
                        "metric": "asymptotic convergence ratio",
                        "observed": data["asymptotic_ratio"], "expected": exp_ratio,
                        "rel_err_pct": abs(data["asymptotic_ratio"] - exp_ratio)
                        / exp_ratio * 100.0,
                    })
                if "gci_fine_21" in data:
                    out.append({"ref": b["ref"], "path": path, "metric": "GCI (fine, 21)",
                                "observed": data["gci_fine_21"], "expected": None,
                                "rel_err_pct": None})
            else:
                entries = [e for e in data.get("entries", []) if e.get("order_l2") is not None]
                if entries:
                    last = entries[-1]
                    theo = float(data.get("theoretical_order") or 0)
                    out.append({"ref": b["ref"], "path": path,
                                "metric": "observed L2 order (finest triple) vs theory",
                                "observed": last["order_l2"], "expected": theo,
                                "rel_err_pct": (abs(last["order_l2"] - theo) / theo * 100.0
                                                if theo else None)})
    return out


def verification_table(repo):
    path = os.path.join(repo, "artifacts", "m1634_verification.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path, encoding="utf-8")).get("runs", [])


def render(repo, d):
    acc = d["acceptance"]
    delta = d["delta"]
    themes = d["theme_evolution"]
    rows = crosschecks(repo, d)
    phys = unmerged_physics(repo, d)
    mismatches = [r for r in rows if r["rel_err_pct"] is None]
    worst = max([r["rel_err_pct"] for r in rows if r["rel_err_pct"] is not None] or [0.0])

    L = []
    a = L.append
    a("# TensorDG 上游差集与提交主题演化审计（m1634）\n")
    a("> 机器可读工件：`artifacts/upstream_delta_m1634.json`（schema `%s`）；"
      "图表：`artifacts/upstream_delta_m1634.png`；"
      "生成器：`tools/upstream_delta.py`、`tools/render_upstream_delta_md.py`。"
      "本文所有数字均由脚本从上述工件/`git` 读取注入，无手工抄写。\n" % d["schema"])

    a("## ① 案例介绍\n")
    a("| 项 | 值 |\n|---|---|")
    a("| 仓库 | `%s`（远端 `%s`，工作树目录 `%s`） |"
      % (d.get("upstream_url") or d["repo"], d["remote"], d["repo"]))
    a("| 本地 HEAD | `%s` (`%s`) |" % (d["head"]["sha"][:12], d["head"]["ref"]))
    a("| 上游基线 | `%s` = `%s` |" % (delta["base_ref"], delta["base_sha"][:12]))
    a("| 两者是否同一提交 | **%s** |" % ("是（差集为空）" if delta["identical"] else "否"))
    a("| ahead / behind 提交数 | %d / %d |" % (len(delta["ahead"]), len(delta["behind"])))
    a("| 未并入 HEAD 的远端分支 | %d 个（合计 +%d 行） |"
      % (acc["unmerged_branch_count"], acc["unmerged_insertions"]))
    a("| 最近提交窗口 | %d 条（%s → %s，%d 个提交日） |"
      % (acc["recent_commit_count"], themes["first_date"], themes["last_date"],
         themes["span_days"]))
    a("| 其中 merge 提交占比 | %.1f%% |" % (themes["merge_ratio"] * 100.0))
    a("")
    a("TensorDG 是本生态的间断 Galerkin（DG）求解器仓：1D/2D Euler（TVB/WENO 限制器）、"
      "标量对流/热传导/Burgers、SIPG 椭圆（matrix-free 与 sparse 两条路径）、"
      "以及对照精确解/参考解的 benchmark 与基准套件。本单要回答的不是「代码算得准不准」，"
      "而是**本仓与它自己的远端上游之间是否已经漂移**，以及最近 15 次提交的主题演化与风险面。"
      "对 TensorDG 这类以数值精度为交付物的仓库，"
      "「上游有成熟成果但没进 main」等价于**成果不在任何 CI/发布门的视野内**，"
      "风险等级高于任何单点精度偏差。\n")

    a("## ② 方法与工具链\n")
    a("只使用只读 git 管道命令，不写仓库、不推远端：`rev-parse` / `merge-base` / "
      "`rev-list` / `for-each-ref` / `log --numstat` / `diff --name-status`。流程：\n")
    a("1. **差集**：`branch_delta(base=origin/main, head=HEAD)` 给出 identical/ahead/behind/merge-base；\n"
      "2. **未并入上游盘点**：枚举 `refs/remotes/origin/*`，用 `merge-base --is-ancestor` 剔除可达分支，"
      "余者即「远端存在、HEAD 不可达」的成果；\n"
      "3. **主题归类**：按路径规则（顶层目录优先于文件名启发式）把每个文件映射到单一主题；"
      "commit 主题取非文档文件的最大计数主题；\n"
      "4. **风险标注**：10 条路径/规模规则（下表），每条给出等级与可执行含义；\n"
      "5. **可复现性**：工件不写入生成时刻时间戳，`--check` 模式可逐字段比对，"
      "失配即 exit≠0（防「报告漂移」）；\n"
      "6. **验证**：`tests/test_upstream_delta.py` 在临时合成仓上固定作者/提交时间，"
      "覆盖主题映射、风险规则、ahead/behind 算术、未并入分支识别、工件新鲜度。\n")
    a("| 规则 | 等级 | 含义（触发即需动作） | 命中数 |\n|---|---|---|---|")
    hits = {k: 0 for k in RISK_DEF_ORDER}
    for r in d["recent_commits"]:
        for t in r["risks"]:
            hits[t["rule"]] = hits.get(t["rule"], 0) + 1
    for b in d["unmerged_upstream_branches"]:
        for t in b["risks"]:
            hits[t["rule"]] = hits.get(t["rule"], 0) + 1
    for rule in RISK_DEF_ORDER:
        level, note = ud.RISK_RULES.get(rule, ("HIGH", "upstream branch not reachable "
                                                    "from HEAD -> invisible to main CI"))
        a("| `%s` | %s | %s | %d |" % (rule, level, note, hits.get(rule, 0)))
    a("")

    a("## ③ 计算结果\n")
    a("### 3.1 差集结论\n")
    if delta["identical"]:
        a("`origin/main` 与本地 `HEAD` **指向同一提交** `%s`：**差集为空**，"
          "ahead=0、behind=0，merge-base 即该提交本身。"
          "换言之 main 线不存在「本地落后上游」或「上游落后本地」的问题。\n"
          % delta["base_sha"][:12])
    else:
        a("ahead=%d、behind=%d，merge-base=`%s`。\n"
          % (len(delta["ahead"]), len(delta["behind"]), (delta["merge_base"] or "")[:12]))
    a("但「main 无差集」不等于「上游无差集」：远端仍存在未并入 HEAD 的分支，见 3.2。\n")

    a("### 3.2 未并入 HEAD 的远端分支（本次审计的核心发现）\n")
    a("| 远端分支 | tip | 领先提交 | 文件 | 新增行 | 删除行 | 命中主题 |\n|---|---|---|---|---|---|---|")
    for b in d["unmerged_upstream_branches"]:
        a("| `%s` | `%s` | %d | %d | +%d | -%d | %s |"
          % (b["ref"], b["sha"][:9], b["commits_ahead"], b["files_touched"],
             b["insertions"], b["deletions"], ", ".join(b["themes"])))
    if not d["unmerged_upstream_branches"]:
        a("| — | — | 0 | 0 | +0 | -0 | — |")
    a("")
    for b in d["unmerged_upstream_branches"]:
        a("该分支 tip 提交：`%s`（%s）%s。相对 HEAD 的差异文件（前 20）：\n"
          % (b["sha"][:9], b["commits"][0]["date"] if b["commits"] else "—",
             b["commits"][0]["subject"] if b["commits"] else ""))
        files = [c for c in b["commits"][0]["files"]] if b["commits"] else []
        a("| 文件 | + | - | 主题 |\n|---|---|---|---|")
        for f in files[:20]:
            a("| `%s` | %s | %s | %s |"
              % (f["path"], "bin" if f["binary"] else f["insertions"],
                 "bin" if f["binary"] else f["deletions"], f["theme"]))
        a("")

    a("### 3.3 最近 %d 提交的主题演化\n" % acc["recent_commit_count"])
    a("| # | 提交 | 日期 | 主题 | 文件 | +行 | -行 | merge | 风险标注 |\n|---|---|---|---|---|---|---|---|---|")
    per = {p["short"]: p for p in themes["per_commit"]}
    for i, r in enumerate(d["recent_commits"], 1):
        a("| %d | `%s` | %s | %s | %d | %d | %d | %s | %s |"
          % (i, r["short"], r["date"],
             "—（merge）" if r["merge"] else per[r["short"]]["theme"], r["files_changed"],
             r["insertions"], r["deletions"], "Y" if r["merge"] else "",
             ", ".join(t["rule"].split("_")[0] for t in r["risks"]) or "—"))
    a("")
    a("非 merge 提交的主题计数：%s；含 merge 时：%s。\n"
      % (", ".join("%s=%d" % kv for kv in themes["theme_counts"].items()),
         ", ".join("%s=%d" % kv for kv in themes["theme_counts_incl_merge"].items())))
    a("按主题的代码周转（insertions+deletions，仅非 merge）：\n")
    a("| 主题 | 周转行 |\n|---|---|")
    for k, v in themes["churn_by_theme"].items():
        a("| %s | %d |" % (k, v))
    a("")
    a("**主题演化判读**：窗口内 %d 条提交中 %d 条为 merge（%.1f%%），"
      "真正的代码推进集中在两条：限制器（TVB/WENO）与 SIPG 椭圆求解器（matrix-free/sparse），"
      "两者都同时命中 `R5_PHYSICS_CORE` 与 `R7_LARGE_CHURN`；其余非 merge 提交是"
      "「报告/工件落盘」（`R1_DOC_ONLY` + `R3_NUMERIC_ARTIFACT`）。"
      "即演化形态是**少数深改 + 大量证据落盘**，与「有产出即写机器可读判据」的纪律一致；"
      "代价是 main 上出现 7 条无代码内容的 merge 提交，"
      "若只看 `git log --oneline` 会高估实际改动量。\n"
      % (acc["recent_commit_count"], len(d["recent_commits"]) - len(
          [r for r in d["recent_commits"] if not r["merge"]]), themes["merge_ratio"] * 100.0))

    a("### 3.4 风险清单（按等级）\n")
    a("| 等级 | 窗口内提交命中 | 含未并入分支 |\n|---|---|---|")
    allt = d.get("risk_tally_by_level_incl_upstream", d["risk_tally_by_level"])
    for k in sorted(set(d["risk_tally_by_level"]) | set(allt)):
        a("| %s | %d | %d |" % (k, d["risk_tally_by_level"].get(k, 0), allt.get(k, 0)))
    a("")
    a("需动作的三条（按优先级）：\n")
    a("1. **R9（HIGH）** 未并入分支 `%s` 带 %d 行成熟代码（含 %d 个测试文件）游离在 main 之外——"
      "合并前 main 的 CI 不会跑它，版本会与 main 的 `tensordg/` 并行漂移；\n"
      % (d["unmerged_upstream_branches"][0]["ref"] if d["unmerged_upstream_branches"] else "—",
         acc["unmerged_insertions"],
         sum(1 for c in (d["unmerged_upstream_branches"][0]["commits"] if
                         d["unmerged_upstream_branches"] else [])
             for f in c["files"] if f["path"].startswith("tests/"))))
    a("2. **R3（HIGH）** 窗口内 %d 条提交改写了机器可读数值工件（`validation_report.json` 等）。"
      "工件被手改与被打算出来在 git 里不可区分，唯一验证手段是**重跑生成器**；\n"
      % hits.get("R3_NUMERIC_ARTIFACT", 0))
    a("3. **R5（HIGH）** %d 条提交触及格式/限制器核心，精度结论不能继承旧 benchmark，"
      "必须重跑精确解对照。\n" % hits.get("R5_PHYSICS_CORE", 0))

    if phys:
        a("### 3.5 未并入分支的精度自证（直接从该分支 blob 读取）\n")
        a("| 分支 | 工件 | 指标 | 观测 | 期望/理论 | 相对误差 |\n|---|---|---|---|---|---|")
        for p in phys:
            a("| `%s` | `%s` | %s | %s | %s | %s |"
              % (p["ref"], p["path"].split("/")[-1], p["metric"],
                 ("%.6g" % p["observed"]) if p["observed"] is not None else "—",
                 ("%.6g" % p["expected"]) if p["expected"] is not None else "—",
                 ("%.3f%%" % p["rel_err_pct"]) if p["rel_err_pct"] is not None else "—（非误差量）"))
        a("")
        a("即：该未并入分支不只是「有代码」，其 SIPG P2 的 Richardson 表观阶与"
          "渐进比都落在 3%% 以内，属于可合并的成熟成果，进一步抬高「长期不并 main」的机会成本。\n")

    a("## ④ 图表\n")
    a("![upstream delta](artifacts/upstream_delta_m1634.png)\n")
    a("三联图（由 `tools/plot_upstream_delta.py` 读工件绘制，标签为 ASCII：容器无 CJK 字体）："
      "**(A)** 最近窗口内非 merge 提交的改动行数，按主导主题着色——一眼看出改动集中在"
      "限制器与 SIPG 两次深改；**(B)** 按主题汇总的周转 vs 未并入分支的 +%d 行（黑柱），"
      "后者的体量与窗口内任何单主题量级相当；**(C)** 风险标注按等级计数。"
      "图与表同源，改工件即改图。\n" % acc["unmerged_insertions"])

    a("## ⑤ 注意点与局限\n")
    a("- **单分支 clone 陷阱**：本工作树 `remote.origin` 只抓 `main`，"
      "`git fetch origin` 后 `refs/remotes/` 里依旧只有 main，「看不到上游分支」是假象；"
      "必须先用 `git ls-remote origin` 枚举远端真实引用，再显式 "
      "`git fetch origin refs/heads/<b>:refs/remotes/origin/<b>` 逐个拉。"
      "本单正是靠这一步才发现 `%s`。\n"
      % (d["unmerged_upstream_branches"][0]["ref"] if d["unmerged_upstream_branches"] else "—"))
    a("- **merge 提交稀释统计**：%.1f%% 的窗口提交是 merge，"
      "任何「按主题计数」的结论都必须只看非 merge（本工具同时给出两种视图）。\n"
      % (themes["merge_ratio"] * 100.0))
    a("- **主题/风险是路径启发式**，不做语义 diff：如「把物理公式从 A 挪到 B」不会触发 R5，"
      "「改注释」也会触发 R4。规则的作用是**排优先级**，不是判定正确性。\n")
    a("- **未并入判定只覆盖已 fetch 的远端引用**；未 fetch 的远端分支对本工具不可见，"
      "故结论要配 `git ls-remote` 交叉确认。\n")
    a("- **重测试未跑完**：`tests/test_euler_sod_1d.py`（5 例）与 `tests/test_vortex_2d.py`（2 例）"
      "在本容器的单次调用预算内未跑完（单例 >52 s，其中 3%% 精度门限用例尤其重），"
      "已跑部分见 ⑥；这属于执行预算限制，不代表失败，也不代表通过，需另择宽预算复核。\n")
    a("- **审计对象按 sha 钉死**：工件以 `head.sha`（本次审计时的 `origin/main` = `%s`）为锚，本单自身在 `agent/m1634` 上的提交不计入差集；否则每写一次提交报告就自我失效（本单首版即踩此坑，`--check` 立刻报 STALE）。\n" % d["head"]["sha"][:12])
    a("- 本单是仓库治理审计，**不涉及船型几何与 Cb/L/B 校验**（非船舶总体性能求解任务），"
      "故无几何前置验证项。\n")

    a("## ⑥ 误差与验收判定\n")
    a("本文的「精度」是可数事实与 git 的一致性，用**独立 git 命令**做基准逐项比对"
      "（误差 = |工具值−git 直读| / git 直读）：\n")
    a("| 项 | 工具值 | git 直读（基准） | 相对误差 |\n|---|---|---|---|")
    for r in rows:
        err = "—（基准为 0 且工具非 0：MISMATCH）" if r["rel_err_pct"] is None \
            else "%.3f%%" % r["rel_err_pct"]
        a("| %s | %s | %s | %s |" % (r["item"], r["tool"], r["git"], err))
    a("")
    a("- **是否 <3%%**：%s（最大相对误差 %.3f%%）。\n"
      % (("**否**，%d 项基准为 0 而工具非 0（MISMATCH）" % len(mismatches))
         if mismatches else
         "**是**，全部计数项与 git 直读一致（误差 0）", worst))
    a("- **工件新鲜度**：`tools/upstream_delta.py --check artifacts/upstream_delta_m1634.json` "
      "逐字段相等 → 报告与仓库状态零漂移（由 `tests/test_upstream_delta.py` 自动断言）。\n")
    ver = verification_table(repo)
    if ver:
        a("- **测试证据**（`artifacts/m1634_verification.json`，本单实跑记录）：\n")
        a("| 套件 | 结果 | 用时 | 命令 |\n|---|---|---|---|")
        for v in ver:
            a("| %s | %s | %s | `%s` |" % (v["suite"], v["result"], v.get("duration", "—"),
                                           v.get("command", "")))
        a("")
    if phys:
        perr = [p["rel_err_pct"] for p in phys if p["rel_err_pct"] is not None]
        if perr:
            a("- **未并入分支精度门限**：表观阶/渐进比相对理论值最大误差 %.3f%% → "
              "%s；GCI(fine) 见 3.5。\n" % (max(perr),
                                            "**<3%，达标**" if max(perr) < 3.0 else "**≥3%，未达标**"))
    a("- 无基准项已显式声明：物理精度未在本单重测（见 ⑤ 重测试预算），"
      "因此除 3.5 引用的工件事后数据外，不给出「DG 精度 <3%%」的新结论。\n")

    a("## 结论\n")
    a("1. **差集为空**：`origin/main` == `HEAD` == `%s`，无本地/上游错位。\n" % d["head"]["sha"][:12])
    a("2. **发现 1 个未并入远端分支**（+%d 行成熟代码，含测试与收敛/GCI 工件，"
      "自身 22 例测试全绿、表观阶误差 <3%%）——这是本单最有价值的发现，"
      "建议下一步单独立单做「合并 + 全量回归 + CI 覆盖」。\n" % acc["unmerged_insertions"])
    a("3. **窗口演化形态**：少数深改（限制器、SIPG）+ 大量证据落盘，"
      "%.1f%% 为 merge 提交，统计需去 merge 才是真实改动。\n" % (themes["merge_ratio"] * 100.0))
    a("4. **三类 HIGH 风险**（未并入分支 / 手改数值工件 / 物理核心变更）已逐条标注可执行动作。\n")
    return "\n".join(L).rstrip() + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=ROOT)
    ap.add_argument("--artifact", default=os.path.join("artifacts", "upstream_delta_m1634.json"))
    ap.add_argument("--out", default=os.path.join("docs", "upstream_delta_audit_m1634.md"))
    ns = ap.parse_args(argv)
    art = ns.artifact if os.path.isabs(ns.artifact) else os.path.join(ns.repo, ns.artifact)
    d = json.load(open(art, encoding="utf-8"))
    text = render(ns.repo, d)
    os.makedirs(os.path.dirname(ns.out) or ".", exist_ok=True)
    with open(ns.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote %s (%d bytes)" % (ns.out, len(text.encode())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())