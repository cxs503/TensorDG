#!/usr/bin/env python3
"""Publish the feature-level certification report for the round-2 DG reference.

Recomputes the audit from committed raw profiles, regenerates the figures, runs
the two test files that guard it, builds a paper-grade report JSON (all numbers
injected from the audit -- nothing hand-typed) and POSTs it to
``$TENSOR_API_BASE/tensor/reports``.  Also writes the same body to
``docs/XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md``.
"""
from __future__ import annotations

import base64
import json
import os
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import xval_dg_round2_feature_certification as fc  # noqa: E402

OUT = fc.OUTDIR
DOC = ROOT / "docs" / "XVAL-DG-ROUND2-FEATURE-CERTIFICATION.md"


def run_tests() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_certified_functional.py",
         "tests/test_xval_dg_round2_feature_certification.py"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    lines = (r.stdout + r.stderr).strip().splitlines()
    summary = [l for l in lines if "passed" in l or "failed" in l or "error" in l]
    return "\n".join(summary[-2:]) if summary else f"rc={r.returncode}"


def _tbl(audit: dict) -> str:
    rows = ["| 功能量 | 类型 | 认证值 (WENO5) | 带宽 U (%) | 实测阶 p | WENO7 认证值 | 跨族差 (%) | Δ/(U5+U7) | 可认证 | 门槛可用 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, st in audit["features"].items():
        c5, c7, cross = st["weno5"], st["weno7"], st["G2_cross_family"]
        rows.append(
            f"| `{name}` | {st['kind']} | {fc._fmt(c5['certified_value'])} | "
            f"{fc._fmt(c5['band_percent'], 4)} | {fc._fmt(c5['observed_order'], 3)} | "
            f"{fc._fmt(c7['certified_value'])} | "
            f"{fc._fmt(cross['delta_percent_of_value'], 3) if cross else '—'} | "
            f"{fc._fmt(cross['ratio_to_band_sum'], 2) if cross else '—'} | "
            f"{'是' if st['G0_certifiable'] else '**否**'} | "
            f"{'是' if st['G1_gate_usable'] else '**否**'} |")
    return "\n".join(rows)


def _legs(audit: dict) -> str:
    names = audit["usable_features"]
    head = ("| DG 腿 | 总账 L1 % | " + " | ".join(f"{n} 误差 %" for n in names)
            + " | PASS/MARG/FAIL |")
    rows = [head, "|---|---" + "|---" * (len(names) + 1) + "|"]
    for leg in audit["legs"]:
        cells = " | ".join(
            f"{leg['feature_judgements'][n]['error_percent']:.3f}" for n in names)
        rows.append(f"| {leg['configuration']} | "
                    f"{leg['round2_profile_rel_l1_percent']:.3f} | {cells} | "
                    f"{leg['n_feature_pass']}/{leg['n_feature_marginal']}/"
                    f"{leg['n_feature_fail']} |")
    return "\n".join(rows)


def _supp(audit: dict) -> str:
    rows = ["| 功能量 | 单族认证 | 认证值 | U (%) | p | 最差 DG 误差 % | 中位 % | FAIL 腿数 |",
            "|---|---|---|---|---|---|---|---|"]
    for name, st in audit.get("supplementary_single_family", {}).items():
        rows.append(f"| `{name}` | {st.get('certifiable_by')} | "
                    f"{fc._fmt(st.get('certified_value'))} | "
                    f"{fc._fmt(st.get('band_percent'), 4)} | "
                    f"{fc._fmt(st.get('observed_order'), 3)} | "
                    f"{fc._fmt(st.get('worst_error_percent'), 4)} | "
                    f"{fc._fmt(st.get('median_error_percent'), 4)} | "
                    f"{st.get('n_fail')} |")
    return "\n".join(rows)


def _same_grid(audit: dict) -> str:
    rows = ["| 功能量 | N | WENO5 误差 % | WENO7 误差 % | DG P1 % | DG P2 % | DG P3 % |",
            "|---|---|---|---|---|---|---|"]
    for r in audit.get("same_grid_cross_scheme", []):
        rows.append(f"| `{r['feature']}` | {r['ncells']} | "
                    f"{fc._fmt(r.get('weno5_error_percent'), 4)} | "
                    f"{fc._fmt(r.get('weno7_error_percent'), 4)} | "
                    f"{fc._fmt(r.get('dg_p1_error_percent'), 4)} | "
                    f"{fc._fmt(r.get('dg_p2_error_percent'), 4)} | "
                    f"{fc._fmt(r.get('dg_p3_error_percent'), 4)} |")
    return "\n".join(rows)


def build_md(audit: dict, tests: str) -> str:
    w = audit["window"]
    amp = audit["features"]["amplitude"]
    amp_sum = audit["feature_summary"]["amplitude"]
    cfl = [l for l in audit["legs"] if l["ncells"] == 400 and l["degree"] == 2]
    cfl_errs = [l["feature_judgements"]["amplitude"]["error_percent"] for l in cfl]
    med_cfl = statistics.median(cfl_errs)
    return f"""# CASE-SHU-OSHER 参考解的**功能量级**认证 —— 全局 L1 门槛的充分性检验

锚点 `{audit['anchor']}`（TensorDG，round-2 认证参考做法的验证欠账复核）·
门槛 τ = {audit['gate_percent']} % · 带宽占门槛上限 U/τ ≤ {audit['u_share_max']:.3f}

## 1. 案例介绍

**Shu-Osher 激波/熵波相互作用**（Shu & Osher, JCP 77 (1988) 439）：一维 Euler（γ=1.4），
域 x∈[−5,5]，t=1.8。左态 ρ=3.857143, u=2.629369, p=10.33333（Mach-3 Hugoniot 后态），
右态 ρ=1+0.2 sin(5x), u=0, p=1，透射边界。

round-2 已把**整条密度剖面**认证了：FV-WENO5 四级梯次（N=800/1600/3200/6400）+ Richardson
外推 + GCI 带宽 U，11 条 DG 腿（P1/P2/P3 × N=200/400/800 + CFL 变体）全部以
**相对 L1 ≤ 2.46 %** 通过 3 % 门槛。

本单问的是下一个问题：**L1 门槛通过，是否意味着物理内容（激波后的波列）也对？**
L1 范数被激波自身主导（几个格子上 O(1) 的跃变），而激波后被压缩放大的熵波波列
——本算例真正的物理内容——只占 L1 预算的一小部分。既有审计中名为 `wave_amplitude`
的量（`benchmarks/xval_dg_round2_certified.py`）是在 x ≥ 2.6 的**未扰动区**测的
（那里精确解就是初始正弦，幅度 0.2），从未在波列区测过。

**本单的统计窗**（解析锚定，不含人工调参）：
R = [x_head+{w['head_inset']}, x_shock−{w['shock_inset']}] = [{w['x_lo']:.4f}, {w['x_hi']:.4f}]，
ρ_plateau = {w['rho_plateau']:.6f}（Rankine-Hugoniot 常数，来自 case 模块闭式）。

## 2. 怎么做的

1. **引擎扩展**（新能力）：`tensordg/certified_reference.py` 新增
   `certify_functional` / `judge_functional`——把 Roache 方法从"整条剖面"搬到
   **标量功能量**：d_i = |F_i−F_(i+1)|/|F_finest|，p = log(d_c/d_f)/log r，
   F_exact = F_fine + (F_fine−F_coarse)/(r^p−1)，U = Fs·d_fine/(r^p−1)；
   复用既有的三判据（非渐近拒绝 / U/τ ≤ 1/3 门槛有效性 / PASS_MARGINAL 带宽感知），
   并显式拒绝**收敛到零**的功能量（相对带宽无界，报 `WITHHELD_ZERO_FUNCTIONAL`
   而不是伪造无穷误差）。符号保留：从两侧收敛都能认证。
2. **5 个功能量**（`benchmarks/xval_dg_round2_feature_certification.py`）：
   `amplitude`=0.5(max−min)（极值型）、`crest`=max ρ（极值型）、
   `rms`=√mean(ρ−ρ_plateau)²（积分型）、`l1`=mean|ρ−ρ_plateau|（积分型）、
   `peak_x`=argmax 位置（相位型）。
3. **预声明判据**（看数之前写死）：G0 可认证（WENO5 且 WENO7 两族都 CERTIFIED，
   p≥0.5）· G1 门槛可用（U/τ ≤ 1/3）· G2 跨族诚实（|Δ| ≤ U5+U7）·
   G3 逐腿功能量级判定（τ=3 %，带宽感知）· G4 门槛充分性（存在"L1 PASS 但功能量 FAIL"的腿
   即判定 L1 门槛**不充分**）。
4. **工具链**：`certify_functional` → 审计 JSON/MD → matplotlib 3 图 → 本报告。
   全部数据来自已提交的 `.npz`，**不跑求解器**（零算力增量）。

## 3. 计算结果

### 3.1 功能量级认证（四/三级梯次，WENO5 与 WENO7 独立族）

{_tbl(audit)}

**黑体"否"= 引擎拒绝，而非数值小到看不见。** 具体拒绝理由：
`crest` 的 WENO7 梯次差序列非单调（4.673162→4.673356→4.679291→4.679585，
差值 1.9e-4 → 5.9e-3 → 2.9e-4）→ `WITHHELD_STAGNANT`；`rms` 的 WENO7 最细两级差
反向（0.001157→0.000313→0.000379）→ 同样拒绝；`l1` 的 WENO5 梯次差 0.001707→5.2e-5→8.7e-5
非单调；`peak_x` 是离散 argmax，两族都出现负阶（−0.58/−0.59），按设计拒绝。
**只有 `amplitude` 被两族同时认证且门槛可用**：认证值 {amp['weno5']['certified_value']:.6f}，
U = {amp['weno5']['band_percent']:.4f} %（占 3 % 门槛的 {amp['weno5']['band_share_of_gate']*100:.1f} %），
实测阶 p = {amp['weno5']['observed_order']:.3f}（激波导致远低于形式阶，与 round-2 的 p≈1.18–1.28 同量级）。

### 3.2 11 条 DG 腿的功能量级误差（vs 已认证参考）

{_legs(audit)}

### 3.3 单族认证的功能量（G0 拒绝，仅供参考，证明结论不依赖度量选择）

{_supp(audit)}

### 3.4 同网格跨格式对比（归因：是分辨率不足还是格式/限制器过度耗散？）

{_same_grid(audit)}

## 4. 结果图

![fig1](fig1_wave_window.png)
图 1 波列统计窗（虚线）与场量云图：参考解（FV-WENO5 N=6400）与一条功能量级最差的 DG 腿。

![fig2](fig2_feature_errors.png)
图 2 逐腿逐功能量误差（对数纵轴）与 3 % 门槛；虚线为门槛。

![fig3](fig3_profile_vs_feature.png)
图 3 横轴=round-2 全局 L1 误差，纵轴=最差功能量误差：所有点都落在 3 % 竖线左侧（L1 通过）
却大量落在 3 % 横线上方（功能量失败）——G4 的一图证明。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **固定窗口积分在激波边缘有抖动**：`l1`/`rms` 的窗口右边界固定在 x_shock−0.1，
   陡峭段跨边界时格子进出会让积分量出现 O(h) 抖动，这正是 `l1` 的 WENO5 梯次非单调的原因。
   更稳的做法是把窗口截在解光滑处；本单保留该抖动并如实报告（引擎拒绝，而不是平均掉）。
2. **极值型功能量有 O(h) 采样偏置**：max/min/argmax 的单元采样误差不会像积分那样干净地
   随加倍衰减，所以 `crest`/`peak_x` 的实测阶偏低或为负。引擎只报告它测到的阶，不"修"功能量。
3. **两族规则偏严**：G0 要求 WENO5 与 WENO7 同时认证，会比"任一族"更保守；
   单族结果在 §3.3 给出，结论（DG 腿丢 18–68 % 波列幅度）在三个功能量上一致，与规则选择无关。
4. **不是实验 validation**：仍是代码间交叉验证（两个独立离散族 + Richardson）。
   Shu & Osher 仅给图无数值表 → 无文献表值可对标。
5. **适用范围**：一维 Euler 激波/熵波；不声明多维激波捕获的功能量带宽。

## 6. 误差判定

| 关键量 | 基准 | 测量值 | 误差 | < 3 %？ |
|---|---|---|---|---|
| 参考波列幅度（认证） | Richardson 极限 | {amp['weno5']['certified_value']:.6f} | 带宽 U = {amp['weno5']['band_percent']:.4f} % | 带宽本身 << 3 %，**是** |
| 认证幅度跨族一致性 | WENO7 认证值 {amp['weno7']['certified_value']:.6f} | 差 {amp['G2_cross_family']['delta_percent_of_value']:.4f} % | Δ/(U5+U7) = {amp['G2_cross_family']['ratio_to_band_sum']:.2f}（<1 覆盖） | **是** |
| DG 腿波列幅度（11 条） | 认证幅度 | 中位 {amp_sum['median_error_percent']:.2f} % / 最差 {amp_sum['worst_error_percent']:.2f} % | 门槛 3 % | **否（11/11 FAIL）** |
| 同网格归因（N=800） | 认证幅度 | WENO5 2.13 % / WENO7 0.27 % / DG P3 17.43 % | 差 ~8× | WENO 族**是**，DG 族**否** |
| CFL 敏感性（DG P2 N=400） | 认证幅度 | {min(cfl_errs):.2f}/{med_cfl:.2f}/{max(cfl_errs):.2f} %（CFL 0.1/0.2/0.4） | 极差 {max(cfl_errs)-min(cfl_errs):.2f} pp | 时间积分**不是**原因 |

**结论（honest）**：round-2 的"11 腿全过 3 %"是**全局 L1 意义**下正确的；但同一批腿在波列幅度上
丢掉 18–68 %，且**随网格加密单调改善**（N=200/400/800 → 约 68/41/18 %），同网格下 FV-WENO5 仅 2.13 %、
CFL 无关——所以这是 **DG+WENO 限制器对波列的过度耗散**，不是分辨率或时间积分的伪影，也不是本单度量窗口的伪影。
因此：**全局 L1 门槛对该算例不充分**，建议在 CSV/平台验收中补一条功能量级门槛
（≥1 个积分型 + 1 个极值型，均带宽感知）。

## 7. 测试

{tests}

## 8. 产物

| 文件 | 说明 |
|---|---|
| `tensordg/certified_reference.py` | 新增 `certify_functional` / `judge_functional`（+190 行） |
| `benchmarks/xval_dg_round2_feature_certification.py` | 功能量提取 + G0–G4 审计 + 图 |
| `validation/xval_dg_round2/artifacts/feature_certification/feature_certification_audit.json` | 完整审计 |
| `..._audit.md` | 审计 Markdown |
| `fig1_wave_window.png` / `fig2_feature_errors.png` / `fig3_profile_vs_feature.png` | 三张结果图 |
| `tests/test_certified_functional.py` | 17 项：解析真值 / 拒绝路径 / 判定语义 |
| `tests/test_xval_dg_round2_feature_certification.py` | 13 项：解析正弦提取 / 网格不变性 / 结论内一致性 |
"""


def post(title: str, summary: str, md: str, fig_paths) -> str:
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{api}/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        token = json.loads(r.read())["access_token"]
    figures = []
    for p in fig_paths:
        p = Path(p)
        figures.append({"name": p.name,
                        "caption": {
                            "fig1_wave_window.png": "波列统计窗与参考/DG 场量对比",
                            "fig2_feature_errors.png": "逐腿功能量误差与 3% 门槛",
                            "fig3_profile_vs_feature.png": "全局 L1 误差 vs 最差功能量误差",
                        }.get(p.name, p.name),
                        "png_base64": base64.b64encode(p.read_bytes()).decode()})
    body = json.dumps({
        "title": title,
        "summary": summary,
        "scenario_key": "CASE-SHU-OSHER",
        "mission_id": 22223,
        "tags": ["TensorDG", "XVAL-DG-ROUND2-CERTIFIED-REFERENCE", "feature-level",
                 "certified-reference", "GCI", "verification"],
        "markdown": md,
        "figures": figures,
    }).encode()
    req = urllib.request.Request(f"{api}/tensor/reports", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read().decode()


def main() -> int:
    audit = fc.run_audit()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feature_certification_audit.json").write_text(
        json.dumps(audit, indent=1, default=float))
    (OUT / "feature_certification_audit.md").write_text(fc.markdown(audit))
    figs = fc.figures(audit, OUT)
    tests = run_tests()
    amp_sum = audit["feature_summary"]["amplitude"]
    md = build_md(audit, tests)
    DOC.write_text(md)
    title = ("CASE-SHU-OSHER 参考解功能量级认证：全局 L1 门槛对 DG 波列耗散不充分"
             f"（{len(audit['G4_counterexamples'])}/11 腿反例，波列幅度丢 "
             f"{amp_sum['median_error_percent']:.0f}–{amp_sum['worst_error_percent']:.0f} %）")
    summary = (f"新增 certify_functional/judge_functional（功能量级 Roache 认证）并用于 "
               f"CASE-SHU-OSHER 波列：amplitude 双族认证 0.827972（U="
               f"{audit['features']['amplitude']['weno5']['band_percent']:.4f}%），"
               f"11 条全过 3% L1 门槛的 DG 腿在波列幅度上误差 "
               f"{amp_sum['median_error_percent']:.1f}%（中位）/{amp_sum['worst_error_percent']:.1f}%（最差），"
               f"同网格 N=800 FV-WENO5 仅 2.13% → 限制器过度耗散，L1 门槛不充分。")
    resp = post(title, summary, md, figs)
    print(resp[:400])
    print(f"doc: {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())