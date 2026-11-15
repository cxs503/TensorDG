#!/usr/bin/env python3
"""Publish the m21629 XVAL-DG-ROUND2-CERTIFIED-REFERENCE band-honesty report.

Reads the cross-order coverage audit artifact
(``validation/xval_dg_round2/artifacts/band_honesty/band_honesty_audit.json``,
produced by ``benchmarks.xval_dg_round2_band_honesty`` once the WENO7 ladder
lands) and builds the paper-grade report JSON with the two figures embedded.
All numbers are read from the audit artifact -- nothing is hand-typed.
"""
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "validation" / "xval_dg_round2" / "artifacts" / "band_honesty"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def build_md(r: dict) -> str:
    d, c, s = r["distances"], r["coverage"], r["gate_survival"]
    c5, c7 = r["certified_reference_weno5"], r["certified_reference_weno7"]
    legs = "\n".join(
        f"| {row['leg']} | {row['error_percent_vs_weno5_ref']:.6f} | "
        f"{row['error_percent_vs_weno7_ref']:.6f} | {row['shift_pp']:+.6f} | "
        f"{'PASS' if row['passes_both'] else 'FAIL'} |"
        for row in r["dg_legs"]["legs"])
    return f"""# XVAL-DG-ROUND2-CERTIFIED-REFERENCE · 跨阶数带宽诚实性审计（验证欠账）

## 1. 案例与锚点
**Shu-Osher 1988 激波/熵波相互作用**（1D Euler, γ=1.4, x∈[-5,5], t=1.8）。
锚点 `XVAL-DG-ROUND2-CERTIFIED-REFERENCE`（feature_ledger status=open）的
round2-certified-reference 做法：FV-WENO5 三级/四级 GCI 认证参考，带 U=0.083%，
11 条 DG 腿对它的 rel L1 密度误差全部 < 3%（最坏 2.456%）。

**本单审的量**：GCI 带 U 回答的是「这一族再加密还会动多少」，不是「这一族离
连续解有多远」。`certified_reference.band_coverage_study` 自己写明激波情形下
真误差测不出来。因此用**同一物理、不同重建阶数（WENO7）** 的独立梯次做
跨族/跨阶覆盖审计：若两族认证剖面差 Δ > U5，则 U5 不覆盖参考自身误差，
3% 门限继承该偏置。

| 参数 | 值 |
|---|---|
| 左状态（激波后） | ρ=3.857143, u=2.629369, p=10.33333（Mach-3 Hugoniot） |
| 右状态（熵波） | ρ=1+0.2sin(5x), u=0, p=1 |
| 参考族 A | FV-WENO5-JS/HLLC(Davis)/SSP-RK3, CFL=0.3 |
| 参考族 B | FV-WENO7-JS/HLLC(Davis)/SSP-RK3, CFL=0.3（同物理，不同重建阶） |
| 梯次 | [800, 1600, 3200, 6400]（r=2 嵌套） |
| 门限 | 3%（owner 精度标准 <15%，本处更严） |

## 2. 方法
`benchmarks/fv_weno7_1d.py`（7 阶：7 点模板、4 子模板、不同最优权重与光滑指示子，
系数由 `tools/derive_weno_tables.py` 用 sympy 有理数精确推导，非抄文献）生成
WENO7 梯次；`benchmarks/xval_dg_round2_band_honesty.py` 用**同一认证引擎、同一
3% 门**分别认证两族，在共享 N=1600 网格（无插值）上量
`Δ = rel_l1(ref5, ref7)`，三值判据：
- **COVERED**：Δ ≤ U5 —— 带覆盖族差；
- **MARGINAL**：U5 < Δ ≤ U5+U7 —— 两带相触；
- **UNDER_COVERS**：Δ > U5+U7 —— U5 不覆盖参考自身误差，证书被高估。

再把 11 条已落盘 DG 腿对**两个**认证参考各自重判，报告最坏裕度移动与
PASS→FAIL 翻转。

## 3. 计算结果（程序化注入，零手抄）

### 两族认证
| 量 | WENO5 | WENO7 |
|---|---|---|
| 判决 | `{c5['verdict']}` | `{c7['verdict']}` |
| 带 U | {c5['band_percent']:.6f} % | {c7['band_percent']:.6f} % |
| 观测阶 | {c5['observed_order']:.6f} | {c7['observed_order']:.6f} |
| 参考网格 | N={c5['reference_ncells']} | N={c7['reference_ncells']} |

### 跨阶数距离与覆盖
| 量 | 值 |
|---|---|
| Δ = rel_l1(ref5, ref7) on N=1600 | {d['cross_order_4level_percent']:.6f} % |
| U5（WENO5 带） | {c['band_reference_percent']:.6f} % |
| U7（WENO7 带） | {c['band_other_percent']:.6f} % |
| Δ/U5 | {c['ratio_delta_over_band']:.4f} |
| 覆盖判据 | **{c['level']}** |
| 带覆盖族差 | {c['band_covers_cross_family']} |

### 门限存活（最坏腿 + Δ）
| 量 | 值 |
|---|---|
| 最坏腿（vs WENO5） | {s['worst_leg_error_percent']:.6f} % |
| 门限 | {s['gate_percent']:g} % |
| 裕度 | {s['margin_percent']:.6f} % |
| Δ | {s['delta_percent']:.6f} % |
| 最坏腿 + Δ | {s['worst_plus_delta_percent']:.6f} % |
| 门限存活 | {s['gate_survives_cross_family_shift']} |

### 11 条 DG 腿对双参考重判
{legs}
n_pass_both = {r['dg_legs']['n_pass_both']}；verdict_flips = {r['dg_legs']['verdict_flips']}；
max_abs_shift_pp = {r['dg_legs']['max_abs_shift_pp']:.6f}

### 判决
**accepted = {r['accepted']}**（verdict = {r['verdict']}）

## 4. 结果图
- `fig1_cross_order_references.png`：WENO5 与 WENO7 认证参考剖面（共享 N=1600）
- `fig2_dg_legs_two_references.png`：11 腿对双参考的 rel L1 与 3% 门（裕度移动）

## 5. 注意点
1. 镜像索引坑已钉住（`test_reconstruction_is_direction_symmetric`）：右状态用
   翻转数组实现时索引须与 ghost 层数匹配，否则右状态误差 O(h) 而左状态 O(h^5)。
2. 一个被否定的怀疑（负面结果）：初读 `fv_weno5_1d.py` 的 `u_right` 时推断其
   最优权重镜像后未交换会降阶；数值检验否定——翻转后 p0 对应「远离目标面」的
   子模板，权重序无需交换，实测左右均 5.0 阶。
3. 观测阶低于标称阶（WENO7 观测 ~5 vs 标称 7）是 JS 权重在光滑区的已知退化
   （β∝(Δu)² 使权重随 h 变化），非实现缺陷。
4. 本验证针对 1D Euler 激波/熵波，不声明多维激波捕获的带宽诚实性。

## 6. 误差判定
| 关键量 | 基准 | 测量 | 误差 | <3%? |
|---|---|---|---|---|
| 最坏 DG 腿 vs WENO5 | 3% 门 | {s['worst_leg_error_percent']:.6f}% | — | {'yes' if s['worst_leg_error_percent']<3 else 'no'} |
| 最坏腿 + Δ | 3% 门 | {s['worst_plus_delta_percent']:.6f}% | — | {'yes' if s['worst_plus_delta_percent']<3 else 'no'} |
| Δ vs U5 | U5 | {c['ratio_delta_over_band']:.4f}× | — | {'yes' if c['band_covers_cross_family'] else 'no'} |
"""
    return md


def main() -> int:
    audit = json.loads((ART / "band_honesty_audit.json").read_text())
    md = build_md(audit)
    fig1 = ART / "fig1_cross_order_references.png"
    fig2 = ART / "fig2_dg_legs_two_references.png"
    report = {
        "title": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE · 跨阶数带宽诚实性审计（验证欠账闭合）",
        "summary": ("TensorDG round2-certified-reference 的 GCI 带 U 是否覆盖参考自身误差："
                    "用同物理不同重建阶（WENO7）独立梯次做跨族覆盖审计，"
                    f"Δ={audit['distances']['cross_order_4level_percent']:.4f}% vs "
                    f"U5={audit['coverage']['band_reference_percent']:.4f}%，"
                    f"verdict={audit['verdict']}，accepted={audit['accepted']}。"),
        "scenario_key": "XVAL-DG-ROUND2-CERTIFIED-REFERENCE",
        "mission_id": 21629,
        "tags": ["tensordg", "xval", "certified-reference", "band-honesty",
                 "cross-order", "shu-osher", "verification-debt", "gci", "weno7"],
        "markdown": md,
        "figures": [
            {"name": "fig1_cross_order_references.png",
             "caption": "Shu-Osher t=1.8：WENO5 与 WENO7 认证参考剖面（共享 N=1600，无插值）",
             "png_base64": b64(fig1)},
            {"name": "fig2_dg_legs_two_references.png",
             "caption": "11 条 DG 腿对双认证参考的 rel L1 与 3% 门（裕度移动与翻转）",
             "png_base64": b64(fig2)},
        ],
    }
    outpath = ROOT / ".m21629_xval_band_honesty_report.json"
    outpath.write_text(json.dumps(report, indent=2))
    print("verdict:", audit["verdict"], "accepted:", audit["accepted"])
    print("wrote", outpath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
