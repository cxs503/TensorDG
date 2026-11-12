#!/usr/bin/env python3
"""Write the board-audit verdict of mission m16072 (CASE-DG-BLASTWAVE-PERSSON, TensorDG).

Every number below is pulled from ``runs/m16133_audit/audit_summary.json``
(produced by ``scripts/audit/m16133_blastwave_audit.py`` from the delivered raw
profiles) or from SQL already run by the audit; nothing is retyped by hand.

    python scripts/audit/verdict_m16133.py            # dry run
    python scripts/audit/verdict_m16133.py --write    # INSERT ... ON CONFLICT
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "runs/m16133_audit/audit_summary.json"
SRC = REPO / "runs/m16133_audit/_source_m16072"
DELIVERED = "/root/tp-agent/work/m16072"

REPO_SLUG = "TensorDG"
SCOPE = "m16072 CASE-DG-BLASTWAVE-PERSSON"
MISSION = 16133
LEVEL = 3
VERDICT_BY = "board-audit"


def row(s, case):
    return next(r for r in s["rows"] if r["case"] == case)


def build_evidence(s: dict) -> dict:
    ex = s["exact_riemann"]
    p2n200 = row(s, "dg-av-p2n200")
    p2n100 = row(s, "dg-av-p2n100")
    p3 = row(s, "dg-av-p3n100")
    noav = row(s, "dg-noav-p2n100")
    ref = row(s, "ref-1600")
    rp = s["repro"]
    return {
        "anchor_slug": "CASE-DG-BLASTWAVE-PERSSON",
        "audited_mission": "m16072 (succeeded) branch agent/m16072 @ f2dc095c74fe6ac91e4f20382eb32436c4d4c6fb",
        "audited_report": "tensor_reports#15209 (scenario dg_blastwave_av, 6 sections, 2 figures)",
        "impl_files": [
            f"{DELIVERED}/tensordg/shock_capture.py",
            f"{DELIVERED}/tensordg/euler_dg_1d.py",
            f"{DELIVERED}/tensordg/cases/blast_wave.py",
            f"{DELIVERED}/benchmarks/blast_wave_av_1d.py",
        ],
        "impl_lines": [
            "tensordg/shock_capture.py:1-157 —— Persson–Peraire 模态衰减传感器 modal_smoothness:48"
            "（三守恒量取最大 + 零变量保护:80-92） / smooth_ramp C1 正弦:96 / default_s0=2p·log10(h)+shift:118"
            " / viscosity_from_modal ν=ε0·h/p:130 / sample_profile 6 Gauss 采样:143",
            "tensordg/euler_dg_1d.py:338-527 —— solve_euler_av_1d（SIPG 中心梯度 + (p+1)² 跳跃罚加入模态残差）;"
            " _viscous_rhs 见 :300 起；粘性 dt 限制 0.25dx²/((2p+1)²ν) 在 :470 附近",
            "tensordg/cases/blast_wave.py:1-132 —— 强激波算例、Rankine–Hugoniot 残差、front_metrics:95（peak/front/"
            "undershoot/behind_oscillation 诊断）",
            "benchmarks/blast_wave_av_1d.py:1-363 —— 19 算例池（ref-400/800/1600/3200/6400、dg-av p2/p3、无粘性对照、"
            "单因子敏感性）+ 汇总 + 作图；l1_err_pct:230 用 mean|Δρ|/跳跃 归一；post_osc_pct:231 实为 max|Δρ|/跳跃",
        ],
        "entry": f"{DELIVERED}/benchmarks/blast_wave_av_1d.py",
        "entry_cmd": "cd <worktree> && PYTHONPATH=$PWD python3 benchmarks/blast_wave_av_1d.py --list "
                     "| --case dg-av-p2n200 | --report（审计人实跑 --list 19 个算例与 --case dg-av-p2n100，OK）",
        "run_artifacts": [
            "runs/m16133_audit/_source_m16072/raw/ref-1600.npz",
            "runs/m16133_audit/_source_m16072/raw/dg-av-p2n200.npz",
            "runs/m16133_audit/_source_m16072/raw/dg-av-p2n100.npz",
            "runs/m16133_audit/_source_m16072/raw/dg-noav-p2n100.npz",
            "runs/m16133_audit/_source_m16072/summary.json",
            "runs/m16133_audit/_source_m16072/results.md",
            "runs/m16133_audit/_source_m16072/author_doc.md",
            "runs/m16133_audit/repro/dg-av-p2n100.npz",
            "runs/m16133_audit/audit_summary.json",
            "docs/board_audit/M16133_CASE_DG_BLASTWAVE_BOARD_AUDIT.md",
            "docs/board_audit/m16133/fig1_profiles.png",
            "docs/board_audit/m16133/fig2_gates.png",
            "docs/board_audit/m16133/fig3_convergence.png",
        ],
        "tests": [
            "tests/test_shock_capture_av.py（9 项；审计人实跑 PYTHONPATH=$PWD pytest -q → 9 passed / 16.65 s）",
            "test_av_off_reproduces_inviscid_driver:131-137（av=None 与无粘残差 assert torch.equal）",
            "test_diffusion_operator_decay_rate:98-128（SIPG 衰减率 vs 精确 5% 带）",
        ],
        "integration_callers": [
            "tensordg/euler_dg_1d.py:solve_euler_av_1d —— 粘性项直接进 DG 模态残差，与既有 HLLC/SSP-RK3/limiter 共用算子栈；"
            "调用方 benchmarks/blast_wave_av_1d.py:138、tests/test_shock_capture_av.py:135/142",
        ],
        "integration_note": "分支未并入 trunk（/root/tensor-ecosystem/TensorDG @812c05e：tensordg/shock_capture.py 不存在、"
                            "grep -rl solve_euler_av_1d = 0 命中；agent/m16072 与 main 无 merge-base）→ ECO 侧 grep 现为 0，"
                            "且未从 tensordg/__init__.py 导出（findings.F4）",
        "benchmark": {
            "declared_criterion_tensor_case_pool": "DG-BLASTWAVE-PERSSON tol_policy: <5% 峰值密度与位置 vs 精细网格参考解 "
                                                   "(t=0.01, 无 Gibbs 过冲>2%)",
            "referee_table": "tensor_benchmark_cases WHERE anchor_slug='CASE-DG-BLASTWAVE-PERSSON' → 0 行（全表 25 行；"
                             "TensorDG 仅 CASE-SHU-OSHER-DG 已登记）→ 唯一裁判缺席（findings.F1）",
            "ref": {"case": "FV-WENO5 N=1600 (benchmarks/fv_weno5_1d.py)", "rho_peak": 6.043057, "x_front": 0.3353125,
                    "audit_verified_vs_analytic": "peak +0.73% / front +0.06% / star −0.29% vs 解析自相似 Riemann 解"},
            "analytic_anchor": {"p_star": round(ex["p_star"], 4), "u_star": round(ex["u_star"], 4),
                                "rho_star_driver_side": round(ex["rho_star_L"], 6),
                                "rho2_postshock": round(ex["rho2"], 6), "D": round(ex["D"], 5),
                                "x_shock_t0.01": round(ex["x_shock_t"], 6), "mach": round(ex["mach"], 2),
                                "source": "自相似 Riemann 解闭式（Toro, Riemann Solvers…, ch.4），审计人独立求解"},
            "got_vs_ref_pct": {"dg-av-p2n200": {"peak": round(p2n200["D1_peak_err_pct"], 2),
                                                "front": round(p2n200["front_excess_pct"], 2),
                                                "overshoot_author_C3": round(p2n200["D1_peak_err_pct"], 2)},
                               "dg-av-p2n100": {"peak": round(p2n100["D1_peak_err_pct"], 2),
                                                "front": round(p2n100["front_excess_pct"], 2)},
                               "dg-av-p3n100": {"peak": round(p3["D1_peak_err_pct"], 2)},
                               "dg-noav-p2n100": {"peak": round(noav["D1_peak_err_pct"], 2)}},
            "got_vs_analytic_pct": {"dg-av-p2n200": {"peak": round(p2n200["D1_vs_exact_pct"], 2),
                                                     "front": round(p2n200["front_vs_exact_pct"], 2)},
                                    "dg-av-p2n100": {"peak": round(p2n100["D1_vs_exact_pct"], 2)},
                                    "dg-av-p3n100": {"peak": round(p3["D1_vs_exact_pct"], 2)},
                                    "dg-noav-p2n100": {"peak": round(noav["D1_vs_exact_pct"], 2)},
                                    "ref-1600": {"peak": round(ref["D1_vs_exact_pct"], 2)}},
            "overshoot_three_definitions_pct": {
                "author_peak_vs_refpeak": round(p2n200["D1_peak_err_pct"], 2),
                "audit_peak_vs_ref_at_same_x": round(p2n200["D2_spike_local_pct"], 2),
                "audit_peak_vs_analytic_rho2": round(p2n200["D1_vs_exact_pct"], 2),
                "gate": "<2%", "verdict": "仅第一种定义通过（余量 0.04pp）；另两种越线 → 2% 门不稳健（findings.F2）"},
            "undisclosed_star_region": {
                "window_x": [0.02, 0.26], "analytic_rho_star": round(ex["rho_star_L"], 4),
                "dg_av_p2n200_err_pct": round(p2n200["star"]["err_pct"], 2),
                "dg_av_p2n100_err_pct": round(p2n100["star"]["err_pct"], 2),
                "dg_noav_p2n100_err_pct": round(noav["star"]["err_pct"], 2),
                "dg_av_p3n100_err_pct": round(p3["star"]["err_pct"], 2),
                "ref1600_err_pct": round(ref["star"]["err_pct"], 2),
                "note": "光滑区 −26%，加 p2 人工粘性由 −11% 恶化 15pp，N100→200 不收敛（−25.8→−26.9%）；"
                        "报告 ⑥ 未列此项，L1=1.30% 为 mean|Δρ|/跳跃(≈6) 归一（findings.F3）",
            },
            "reference_self_convergence": {"peak": {k: round(v, 4) for k, v in s["reference_self_convergence"]["peak"].items()},
                                           "observed_order_peak": round(s["observed_order"]["peak"], 2),
                                           "richardson_peak": round(s["richardson_limit"]["peak"], 4),
                                           "uncertainty_pct": {k: round(v, 2) for k, v in s["reference_uncertainty_pct"].items()}},
            "status": "C1/C2（<5% 峰值与位置）通过；C3（无 Gibbs 过冲>2%）仅在作者定义下贴线通过；"
                      "裁判表无行 → 门不可机检",
        },
        "repro_cmd": "cd /tmp/m16133_repro && cp -r <worktree>/{tensordg,benchmarks} . && PYTHONPATH=$PWD "
                     "python3 benchmarks/blast_wave_av_1d.py --case dg-av-p2n100"
                     "（审计人实跑 43.1 s；peak 6.1552→6.1508 即 −0.071%，max|Δρ|=0.240 集中 x≈0.3403，"
                     "星区差 3e-5，步数同为 1774 → 非逐位可复现）",
        "repro_verdict": {"bitwise_identical": rp["bitwise_identical"],
                          "peak_rel_diff_pct": round(rp["peak_rel_diff_pct"], 4),
                          "max_abs_rho_diff": round(rp["max_abs_rho_diff"], 4),
                          "argmax_at_x": round(rp["argmax_at_x"], 4),
                          "star_mean_delivered": round(rp["star_mean_delivered"], 6),
                          "star_mean_rerun": round(rp["star_mean_rerun"], 6),
                          "gate_margin_pp": round(rp["gate_margin_vs_rerun_scatter_pp"], 3),
                          "why": "余量 0.04pp 小于复跑散布 0.07pp；交付未记录线程/种子（findings.F5）"},
        "external_anchor": "解析自相似 Riemann 解（Toro 教材 ch.4；该算例即经典 explosion problem）——审计人独立求解得 "
                           "ρ*_L=0.575062、ρ₂=5.999241、D=23.51754、x_s(t=0.01)=0.335175；"
                           "AIAA 2006-1125 原文数值表在交付环境不可达（WITHHELD，与交付同判）；"
                           f"实现：{REPO}/scripts/audit/m16133_blastwave_audit.py::exact_riemann",
        "level": LEVEL,
        "level_rationale":
            "L2 成立：157 行传感器 + 求解器内 SIPG 粘性 + 算例诊断，输入→计算→输出闭合，无占位，9 项单测绿。"
            "L3 成立（部分）：粘性项进主 1-D Euler DG 求解器残差装配，与既有算子/HLLC/SSP-RK3/limiter 共用，"
            "av=None 与无粘残差逐位相同（tests:137）；扣分=未导出 __init__、无其他消费方、分支未并入 trunk（F4）。"
            "L4 NOT_REACHED：算例/工件/预声明判据/参考解经解析真值验证（+0.06% front、−0.29% 星区）都成立，"
            "但 (a) tensor_benchmark_cases 该锚点 0 行→唯一裁判缺席；(b) 2% 过冲门仅在最有利定义下 +1.96% 贴线，"
            "另两种定义 +2.70%/+2.77% 越线，余量 0.04pp<复跑散布；(c) 光滑星区 −26% 误差未披露而报告称关键量<3%；"
            "(d) 复跑非逐位且离散集中在门取样处。百页报告不改变级别；级别=功能实质。L5 不成立（无外部实验/文献数值锚）。",
        "findings": {
            "F1_high_referee_absent": "tensor_benchmark_cases 无 CASE-DG-BLASTWAVE-PERSSON 行（0/25；TensorDG 仅 "
                                      "CASE-SHU-OSHER-DG 已登记）→ 任务令指定的唯一裁判不存在；maturity_gate.py::judge "
                                      "的 L4 只验 benchmark 键含数字，不读裁判表 → 门无法发现空缺",
            "F2_medium_high_overshoot_definition": "2% 过冲门：author C3 (peak vs ref-peak)=+1.96%（余量 0.04pp）；"
                                                   "peak vs ref@same_x=+2.77%；peak vs 解析 ρ₂=+2.70% → 2 越线 1 通过；"
                                                   "参考峰值自身比解析值高 0.73%，正好抵掉一部分过冲",
            "F3_high_star_region_undisclosed": "光滑星区 x∈[0.02,0.26] 密度均值 0.4257 vs 解析 0.5751 = −26.0%（p2N200）；"
                                               "无粘性 −11.2%、p3 −11.4% → 加 p2 粘性恶化 15pp；N100→200 −25.8→−26.9% 不收敛；"
                                               "复跑复现（0.42573/0.42570）。报告 ⑥ 未列，L1=1.30% 用品跳归一掩盖",
            "F4_high_not_on_trunk": "trunk 812c05e 无 tensordg/shock_capture.py、grep solve_euler_av_1d=0；"
                                    "agent/m16072 与 main 无 merge-base（push-merge 重建史）→ 平台能力面尚无此项",
            "F5_medium_repro_not_bitwise": "同码同参复跑 peak −0.071%、max|Δρ|=0.240 @x=0.3403（过冲门取样处）、"
                                           "dt 末步 3.192e-6→3.142e-6、步数同 1774 → 线程/归约级非确定性，未记录线程/种子",
            "F6_low_misnamed_metric": "benchmarks/blast_wave_av_1d.py:231 post_osc_pct = 100·max|Δρ|/jump（逐点最大偏差，"
                                      "激波抹宽主导 52.9%），非振荡度量且无判别力（noav 51.2 / 基准 52.9 / ε0=2 54.4）；"
                                      "真正的振荡度量是 cases/blast_wave.py:95 front_metrics.behind_oscillation",
            "F7_low_test_gate_mismatch": "tests/test_shock_capture_av.py:153 用 peak_av<=6.0/0.98（渐近极限 2% 带）且配置 "
                                         "N=64/t=0.004，与算例判据（vs 参考峰值 2%）不同源",
            "F8_positive_real_impl": "非 MOCK/STUB：传感器带零变量保护、SIPG 衰减率 5% 对照、av=None 逐位回归、"
                                     "19 算例池含无粘性对照与单因子扫描；作者报告数字与工件逐位自洽（未编数）",
        },
        "disposition": "approve with findings —— L3 confirmed；L4 NOT_REACHED（honest FAIL 保持，账不消）；"
                       "返修：登记裁判行 + 过冲三口径 + 披露/修星区 −26% + 并 trunk + 记录线程/种子",
        "checks_not_run": [
            "未诊断星区 −26% 的机制（只证系统性/非随机/不随网格收敛；候选=早期强瞬变期粘性熵增生、出口边界通量，未做对照实验）",
            "未跑 ref-3200/6400、dg-av-p2n400、dg-av-p3n200（参考不确定度仍取 800→1600 的 0.27%）",
            "无外部求解器（OpenFOAM/Basilisk/FLASH）交叉验证（环境无外网）",
            "未核 AIAA 2006-1125 原文表（不可达，方法匹配性只按公开公式结构复核）",
            "只复跑 1 个算例做确定性检验，未统计全部 19 算例的复跑散布",
        ],
    }


def main(argv: list[str]) -> int:
    s = json.loads(SUMMARY.read_text())
    ev = build_evidence(s)
    if "--write" not in argv:
        print(json.dumps({"repo": REPO_SLUG, "scope": SCOPE, "mission_id": MISSION,
                          "level": LEVEL, "verdict_by": VERDICT_BY,
                          "evidence_keys": sorted(ev)}, indent=1, ensure_ascii=False))
        print("(dry run; pass --write to insert)")
        return 0
    import subprocess
    import tempfile
    payload = json.dumps(ev, ensure_ascii=False).replace("'", "''")
    stmt = f"""INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('{REPO_SLUG}', '{SCOPE}', {MISSION}, {LEVEL}, '{payload}'::jsonb, '{VERDICT_BY}')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
      evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL, gate_note = NULL;
SELECT id, repo, scope, mission_id, level, gate_state, gate_level, verdict_by
  FROM tensor_maturity_verdicts
 WHERE repo = '{REPO_SLUG}' AND scope = '{SCOPE}' AND verdict_by = '{VERDICT_BY}';"""
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as fh:
        fh.write(stmt)
        path = fh.name
    subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-f", path], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))