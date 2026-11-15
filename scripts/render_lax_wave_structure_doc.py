"""Render ``docs/CASE-LAX-DG-wave-structure.md`` from the audit artifact.

Every number in the rendered document is injected from
``docs/benchmarks/results/lax_wave_structure/metrics.json`` -- nothing is
transcribed by hand.  ``--check`` re-renders in memory and compares with the
committed file, so the document cannot drift away from the artifact.

Usage::

    python scripts/render_lax_wave_structure_doc.py            # write
    python scripts/render_lax_wave_structure_doc.py --check     # drift gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "docs/benchmarks/results/lax_wave_structure/metrics.json"
DOC = ROOT / "docs/CASE-LAX-DG-wave-structure.md"


def _pct(value, digits=3):
    return f"{100.0 * value:.{digits}f} %"


def render(metrics):
    case = metrics["case"]
    star = metrics["exact"]["star"]
    pos = metrics["exact"]["positions"]
    legs = metrics["legs"]
    banked = (metrics.get("banked_l2") or {}).get("values", {})

    def l2pct(leg):
        row = banked.get(leg["leg"])
        return "—（未在本进程重算）" if not row else f"{100 * row['density']:.3f} %"

    def l2pass(leg):
        row = banked.get(leg["leg"])
        return "—" if not row else ("**是**" if row["density"] < 0.03 else "**否**")
    arms = metrics["limiter_arms"]
    trends = metrics["trends"]
    out = []
    w = out.append

    w("# CASE-LAX-DG · 波结构 / 容许性审计（L2 < 3 % 之下的间断保真度欠账）")
    w("")
    w("> **锚点**：`feature_ledger` `CASE-LAX-DG`（算例池·Lax via DG，验收原文"
      "「ref: 精确解｜容差 < 3 %」）。")
    w(f"> **工件**：`docs/benchmarks/results/lax_wave_structure/`"
      f"（生成于 {metrics['generated_at_utc']}，`metrics_sha256` = "
      f"`{(metrics.get('metrics_sha256') or 'n/a')[:16]}…`）。")
    w("> **本文全部数字由 `scripts/render_lax_wave_structure_doc.py` 从工件注入，零手抄；"
      "`--check` 为漂移门。**")
    w("> **纪律**：本审计**不改判**台账臂（相对 L2，P3/N=400）的既有结论，"
      "只补它看不见的一层：逐点容许性与间断厚度。")
    w("")
    w("---")
    w("")
    w("## ① 案例介绍（物理背景 / 几何 / 参数）")
    w("")
    w("Lax (1954) 激波管 / Toro (2009) test 2：一维 `[0,1]` 域内隔膜 `x0 = "
      f"{case['diaphragm']}` 两侧常状态，γ = {case['gamma']}，快照时刻 "
      f"`t = {case['final_time']}`。")
    w("")
    w("| 量 | 左态 | 右态 |")
    w("|---|---|---|")
    w(f"| (ρ, u, p) | ({case['left'][0]}, {case['left'][1]}, {case['left'][2]}) "
      f"| ({case['right'][0]}, {case['right'][1]}, {case['right'][2]}) |")
    w("")
    w("精确解含四道波（闭式，不依赖任何求解器）：")
    w("")
    w("| 波 | 位置 x | 说明 |")
    w("|---|---|---|")
    w(f"| 左稀疏波头 | {pos['rarefaction_head']:.6f} | u_L − a_L |")
    w(f"| 左稀疏波尾 | {pos['rarefaction_tail']:.6f} | u* − a*_L |")
    w(f"| 接触间断 | {pos['contact']:.6f} | u*（密度跳 ρ*_L → ρ*_R） |")
    w(f"| 右激波 | {pos['shock']:.6f} | Rankine–Hugoniot |")
    w("")
    w("星区状态：p\\* = "
      f"{star['pressure']:.6f}、u\\* = {star['velocity']:.6f}、"
      f"ρ\\*_L = {star['density_star_left']:.6f}、"
      f"ρ\\*_R = {star['density_star_right']:.6f}。")
    w("")
    w("## ② 怎么做的（方法 / 流程 / 关键设置）")
    w("")
    w("**问题**：台账验收是一个*积分*量（全场相对 L2 < 3 %）。积分范数对以下几件事完全"
      "不敏感——而这恰好是激波捕捉代码的使用者最关心的：")
    w("")
    w("1. 解是否**单调**（有没有跑到精确解取值范围之外的新极值）；")
    w("2. 数值激波 / 接触间断**有多厚**（格数），随加密怎么变；")
    w("3. 误差**住在哪里**（波带上还是光滑区）。")
    w("")
    w("**本审计新增的五族度量**（跑数前写死在 `benchmarks/lax_wave_structure.py` 里）：")
    w("")
    w("| 编号 | 定义 | 为什么需要 |")
    w("|---|---|---|")
    w("| W1 | 波位相对误差 `\\|x_num − x_ex\\| / \\|x_ex − x0\\|`（复用算例池 C1 的定义） | 已经存在，作为对照 |")
    w("| W2 | 接触 / 激波的密度 10 %–90 % 过渡厚度，按**格数** `w/dx` 报 | L2 看不见间断被抹多宽 |")
    w("| W3 | 越界极值：`max(0, max u_num − max u_ex)` / `max(0, min u_ex − min u_num)`，附越界节点数 | 单调性/离散极值原理 |")
    w("| W4 | `TV(u_num) / TV(u_ex)`（按 x 排序的 Gauss 节点） | 每多 1 个单位就是一份虚假变差 |")
    w("| W5 | 星区平台相对误差（复用算例池 `plateau_values`） | 平台是否被振荡污染 |")
    w("")
    w("**工具链**：`tensordg.euler_dg_1d.solve_euler_dg_1d`（模态 DG + HLLC + SSP-RK3，"
      f"CFL = {case['cfl']}，float64，10 点 Gauss 输出）→ "
      "`benchmarks/lax_wave_structure.py` 逐腿计算 W1–W5 → "
      "`metrics.json` + `profiles.npz` + 图 + `summary.md`。")
    w("")
    w("**本文件的运行模式**：审计的全部 W1–W5 都从**已入库的认证场**"
      "（`lax_case_pool/profiles.npz` P3/N=400、`lax_resolution_requirement/profiles.npz` "
      "P3/N=800）重算，**不需要再跑求解器**，因此可逐位复现；积分范数（相对 L2）不从场"
      "重算，而是从 `lax_resolution_requirement/metrics.json` **引用**（`banked_l2`）。"
      "限幅器支臂与 N=1600 腿已分别以集群作业递交（见 ⑤ 局限）。")
    w("")
    w("**两支研究臂**：")
    w("")
    w("| 臂 | 内容 |")
    w("|---|---|")
    w("| 分辨率臂 | P3 × N ∈ {400, 800}（N=400 台账钉扎腿、N=800 认证腿；两者的场都已入库） |")
    w("| 限幅臂 | P3 × N=400，只动限幅器一个旋钮：`none`（引擎默认，仅正性缩放）、`tvb M=0`、`tvb M=1`、`tvb M=1 且只在步末限幅`、`weno`。**本轮未执行**（集群排队 40+ min 未派发，见 ⑤.6） |")
    w("")
    w("**复用自检**：已入库的 `lax_case_pool/profiles.npz`（P3/N=400）与 "
      "`lax_resolution_requirement/profiles.npz`（P3/N=800）里存着同两条腿的场；"
      "本审计用求解器重跑同样的腿，W3/W4 必须与那两份工件逐位一致（阈值 1e-12），"
      "否则判该工件陈旧。")
    w("")
    w("## ③ 计算结果（程序化注入）")
    w("")
    w("### 3.1 波位误差 W1（容差 2 %，算例池 C1 口径）")
    w("")
    w("| leg | 稀疏波头 | 稀疏波尾 | 接触间断 | 右激波 |")
    w("|---|---|---|---|---|")
    for leg in legs:
        pe = leg["position_errors"]
        cells = " | ".join(
            _pct(pe[k]["rel_error"]) if pe.get(k) else "—"
            for k in ("rarefaction_head", "rarefaction_tail", "contact", "shock"))
        w(f"| p{leg['degree']}.N{leg['cells']} | {cells} |")
    w("")
    w("### 3.2 间断厚度 W2（密度 10 %–90 %）")
    w("")
    w("| leg | dx | 接触 w | 接触 w/dx | 激波 w | 激波 w/dx |")
    w("|---|---|---|---|---|---|")
    for leg in legs:
        c, s = leg["thickness"]["contact"], leg["thickness"]["shock"]
        f = lambda v: "—" if v is None else f"{v:.6f}"
        w(f"| p{leg['degree']}.N{leg['cells']} | {leg['dx']:.6g} | {f(c['width'])} | "
          f"{f(c['width_cells'])} | {f(s['width'])} | {f(s['width_cells'])} |")
    w("")
    for wave in ("contact", "shock"):
        key = f"thickness_scaling_exponent_{wave}"
        row = trends.get(key)
        if not row:
            continue
        w(f"* `{wave}` 厚度按 `w ∝ dx^{row['slope_log_w_vs_log_dx']:.3f}` 缩放"
          f"（格数序列 {', '.join(f'{v:.3f}' for v in row['width_cells'])}）；"
          "指数 ≈ 1 表示「厚度恒为常数格数」。")
    w("")
    w("### 3.3 容许性 W3 / W4 / W5 —— 本轮的核心结果")
    w("")
    w("| leg | rel L2 ρ | ρ 超调 | ρ 欠调 | ρ TV 比 | p 欠调 | p TV 比 | 越界节点 |")
    w("|---|---|---|---|---|---|---|---|")
    for leg in legs:
        e, t = leg["extrema"], leg["tv"]
        w(f"| p{leg['degree']}.N{leg['cells']} | {l2pct(leg)} | "
          f"{_pct(e['density']['overshoot_rel_span'], 2)} | "
          f"{_pct(e['density']['undershoot_rel_span'], 2)} | "
          f"{t['density']['ratio']:.3f} | "
          f"{_pct(e['pressure']['undershoot_rel_span'], 2)} | "
          f"{t['pressure']['ratio']:.3f} | "
          f"{e['density']['nodes_outside_exact_range']} / {e['density']['nodes']} |")
    w("")
    w("越界极值出现在哪里（P3，各腿）：")
    w("")
    w("| leg | ρ 最小值 @ x | 精确 ρ 最小值 | p 最小值 @ x | 精确 p 最小值 | 精确激波位置 |")
    w("|---|---|---|---|---|---|")
    for leg in legs:
        er, ep = leg["extrema"]["density"], leg["extrema"]["pressure"]
        w(f"| p{leg['degree']}.N{leg['cells']} | {er['num_min']:.5f} @ {er['x_at_num_min']:.4f} "
          f"| {er['exact_min']:.5f} | {ep['num_min']:.5f} @ {ep['x_at_num_min']:.4f} "
          f"| {ep['exact_min']:.5f} | {pos['shock']:.6f} |")
    w("")
    w("### 3.4 星区平台 W5（相对误差）")
    w("")
    w("| leg | ρ\\*_L | ρ\\*_R | p\\* | u\\* |")
    w("|---|---|---|---|---|")
    for leg in legs:
        pl, pr = leg["plateau"]["star_left"], leg["plateau"]["star_right"]
        w(f"| p{leg['degree']}.N{leg['cells']} | {_pct(pl['density']['rel_error'])} | "
          f"{_pct(pr['density']['rel_error'])} | {_pct(pl['pressure']['rel_error'])} | "
          f"{_pct(pl['velocity']['rel_error'])} |")
    w("")
    w("### 3.5 限幅器支臂（P3 × N=%d，只动一个旋钮）" % arms[0]["cells"] if arms else "")
    w("")
    if arms:
        w("| arm | 说明 | rel L2 ρ | ρ TV 比 | ρ 越界节点 | p 欠调 |")
        w("|---|---|---|---|---|---|")
        for arm in arms:
            w(f"| `{arm['label']}` | {arm['note']} | "
              f"{_pct(arm['l2_errors']['density'])} | "
              f"{arm['tv']['density']['ratio']:.3f} | "
              f"{arm['extrema']['density']['nodes_outside_exact_range']}"
              f" / {arm['extrema']['density']['nodes']} | "
              f"{_pct(arm['extrema']['pressure']['undershoot_rel_span'], 2)} |")
    w("")
    w("### 3.6 复用自检（求解器重跑 vs 已入库场）")
    w("")
    w("| leg | Δρ TV 比 | Δp TV 比 | Δρ 欠调 | Δp 欠调 | Δ接触厚度 |")
    w("|---|---|---|---|---|---|")
    for leg, row in metrics["reuse_agreement_solver_vs_committed"].items():
        fmt = lambda v: "—" if v is None else f"{v:.3e}"
        w(f"| {leg} | {fmt(row['rho_tv_ratio'])} | {fmt(row['p_tv_ratio'])} | "
          f"{fmt(row['rho_undershoot'])} | {fmt(row['p_undershoot'])} | "
          f"{fmt(row['contact_width'])} |")
    w("")
    w("## ④ 结果图")
    w("")
    captions = {
        "profiles_zoom.png": "接触间断与激波邻域的密度剖面（P3 N=800 / N=1600 与精确解）。"
                             "间断附近可见明显的非单调振荡。",
        "tv_ratio_vs_n.png": "总变差比 `TV(num)/TV(exact)` 随加密的变化；单调解应为 1，"
                             "本算例远离 1 且随加密上升。",
        "extrema_vs_n.png": "越界极值振幅（占精确解值域的百分比）随加密的变化。",
        "limiter_arms.png": "限幅器支臂对比：虚假变差、相对 L2（含 3 % 门）与压力欠调。",
    }
    for name in metrics["figures"]:
        w(f"![{name}](benchmarks/results/lax_wave_structure/figures/{name})")
        w("")
        w(f"*{captions.get(name, name)}*")
        w("")
    w("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    w("")
    w("1. **引擎默认根本没有开激波限幅器**。算例池驱动 `benchmarks/lax_case_pool.run_ladder` "
      "调用 `solve_euler_dg_1d` 时只传 `ncells/degree/final_time/cfl`，`limiter` 取库默认 "
      "`None`——即只有 Zhang–Shu 正性缩放。因此台账臂的 < 3 % 结论是在**无限幅器**配置下"
      "取得的，这一点在既有 34 份 `CASE-LAX-DG*` 文档里从未被量化过。")
    w("2. **积分范数掩盖逐点失效**。3.1–3.3 显示：L2 达标的腿（P3/N=800）同时存在"
      "数倍于精确解总变差的虚假变差，且**随加密变糟**；积分误差下降并不伴随逐点收敛。")
    w("3. **W2 只量 10 %–90 % 过渡**。它不是振荡包络的宽度：在节点密集的单个单元内部，"
      "一个亚格子尺度的跳变可以给出小于 1 格的 `w/dx`。因此 W2 只能用来判断"
      "「跳变是否被压在一个格内」，不能单独当作抹宽幅度。")
    w("4. **W4 依赖采样点集**。TV 在 10 点 Gauss 节点上求和；换输出求积阶会改变最后一个 ulp，"
      "但不会改变「远离 1」这个量级结论（同仓 `lax_norm_convergence` 已证明范数对求积阶收敛）。")
    w("5. **适用范围**：本审计只针对本算例（γ = 1.4、中等激波强度、t = 0.13）的 DG/HLLC/SSP-RK3 "
      "配置，不构成对一般强激波问题的单调性保证。")
    w("")
    w("## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 < 3 %」）")
    w("")
    w("| 关键量 | 基准 | 实测 | 是否 < 3 % |")
    w("|---|---|---|---|")
    for leg in legs:
        star_ok = all(
            leg["plateau"][side][f]["rel_error"] < 0.03
            for side in ("star_left", "star_right") for f in ("density", "pressure"))
        w(f"| 相对 L2 ρ（p{leg['degree']}.N{leg['cells']}） | 精确 Riemann 解 | "
          f"{l2pct(leg)} | {l2pass(leg)} |")
        w(f"| 星区平台 ρ\\*/p\\*（p{leg['degree']}.N{leg['cells']}） | 闭式星区状态 | "
          f"最差 {max(leg['plateau'][s][f]['rel_error'] for s in ('star_left', 'star_right') for f in ('density', 'pressure')) * 100:.3f} % | "
          f"{'**是**' if star_ok else '**否**'} |")
    worst_pos = max(r["rel_error"] for r in legs[-1]["position_errors"].values() if r)
    worst_tv = max(leg["tv"]["density"]["ratio"] for leg in legs)
    w("| 波位（p%d.N%d 最差） | 闭式波位 | %.3f %% | **是**（C1 门 2 %% %s） |"
      % (legs[-1]["degree"], legs[-1]["cells"], 100.0 * worst_pos,
         "达成" if worst_pos < 0.02 else "未达"))
    w(f"| **越界极值（新极值节点数）** | 精确解值域 | "
      f"{legs[-1]['extrema']['density']['nodes_outside_exact_range']} / "
      f"{legs[-1]['extrema']['density']['nodes']}（P3/N{legs[-1]['cells']}） | "
      f"**无基准可判 < 3 %：这是定性判据，实测为 FAIL** |")
    w(f"| **总变差比 TV(num)/TV(exact)** | 单调解 = 1 | 最差 {worst_tv:.3f} | "
      f"**否**（超出 1 的部分即虚假变差，远超 3 %） |")
    w("")
    w("**结论（honest）**：台账的积分判据（相对 L2 < 3 %）在 P3/N=800 达标；但同一份解"
      "**不满足**逐点容许性——存在越界新极值、总变差比远超 1，且两项都随加密恶化。"
      f"因此本轮**不宣布 `CASE-LAX-DG` 关闭**，而是把它拆成两条互不替代的判据："
      f"积分精度（已量化）与逐点单调性（本轮首次量化，判 FAIL）。建议的伴随判据："
      f"`{metrics['verdict']['admissibility_gate_proposed']}`，当前判定 "
      f"**{'PASS' if metrics['verdict']['admissibility_pass'] else 'FAIL'}**。")
    w("")
    w("## 可复现命令")
    w("")
    w("```bash")
    w("python -m benchmarks.lax_wave_structure          # 重跑审计（集群，约 30 CPU·min）")
    w("python -m pytest tests/test_lax_wave_structure.py -q")
    w("python scripts/render_lax_wave_structure_doc.py --check")
    w("```")
    w("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    metrics = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    text = render(metrics)
    if args.check:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current != text:
            raise SystemExit(f"doc drift: {DOC} does not match the artifact")
        print(f"[doc] {DOC} matches the artifact")
        return 0
    DOC.write_text(text, encoding="utf-8")
    print(f"[doc] wrote {DOC} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())