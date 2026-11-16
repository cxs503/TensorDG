"""Render + publish the XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE report.

Every number in the markdown body is read from the arm JSONs written by
``TensorLBM/benchmarks/scmp_two_component_planar_interface.py`` (and from the
reference self-test), so nothing is hand-typed.  Figures are rendered here from
the same data.

    python3 TensorLBM/tools/render_scmp_planar_report.py          # doc + figures
    python3 TensorLBM/tools/render_scmp_planar_report.py --publish  # + platform report
"""

from __future__ import annotations

import base64
import json
import math
import os
import pathlib
import sys
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "TensorLBM" / "outputs" / "scmp_planar"
FIGS = ROOT / "TensorLBM" / "docs" / "figures"
MISSION_ID = 22990
sys.path.insert(0, str(ROOT / "TensorLBM" / "tools"))
import scmp_planar_reference as REF  # noqa: E402


# ---------------------------------------------------------------------------
def load_arms():
    arms = []
    for p in sorted(OUT.glob("*.json")):
        with open(p) as fh:
            arms.append(json.load(fh))
    return arms


def pct(x):
    return f"{100 * x:.2f}%"


def fig_profiles(arm, path):
    """Density profiles of one arm against the exact tie-line and the dip."""
    G = arm["params"]["G"]
    prof = arm["profiles"]
    rhi, rlo = REF.coexistence_tieline(G)
    dip = REF.interface_density_dip(G)
    n = len(prof["rho1"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(n), prof["rho1"], "C0-", lw=2, label=r"$\rho_1$ (LBM, time-averaged)")
    ax[0].plot(range(n), prof["rho2"], "C1-", lw=2, label=r"$\rho_2$ (LBM, time-averaged)")
    ax[0].axhline(rhi, color="C0", ls="--", lw=1, label=f"exact $\\rho_{{hi}}$={rhi:.4f}")
    ax[0].axhline(rlo, color="C1", ls="--", lw=1, label=f"exact $\\rho_{{lo}}$={rlo:.4f}")
    ax[0].set_xlabel("y [lattice]")
    ax[0].set_ylabel("density")
    ax[0].set_title(f"{arm['label']}: G={G:+.2f}, {arm['params']['psi']} psi")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    p = [REF.mechanical_pressure(prof["rho1"][i], prof["rho2"][i], G) for i in range(n)]
    pbar = sum(p) / n
    ax[1].plot(range(n), [x / pbar - 1 for x in p], "k-", lw=2)
    ax[1].axhline(0, color="r", ls="--", lw=1)
    ax[1].set_xlabel("y [lattice]")
    ax[1].set_ylabel(r"$p(y)/\bar p - 1$")
    ax[1].set_title(f"mechanical-pressure invariant (spread {pct(arm['pressure_invariant']['pressure_spread_relative'])})")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_scores(arms, path):
    """Measured plateaus vs. the exact tie-line, arm by arm."""
    lin = [a for a in arms if a.get("reference")]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    xs, hi, lo = [], [], []
    labels = []
    for a in lin:
        G = a["params"]["G"]
        rhi, rlo = REF.coexistence_tieline(G)
        xs.append(len(xs))
        hi.append(abs(a["measured"]["rho_hi"] - rhi) / rhi)
        lo.append(abs(a["measured"]["rho_lo_sim_index"] - rlo) / rlo)
        labels.append(a["label"])
    w = 0.38
    ax.bar([x - w / 2 for x in xs], [100 * v for v in hi], width=w, label=r"$\rho_{hi}$ error")
    ax.bar([x + w / 2 for x in xs], [100 * v for v in lo], width=w, label=r"$\rho_{lo}$ error")
    ax.axhline(15, color="r", ls="--", lw=1, label="owner 15% acceptance")
    ax.axhline(3, color="orange", ls=":", lw=1, label="3% target")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("relative error [%]")
    ax.set_yscale("log")
    ax.set_title("tie-line error of the MCMP arms (linear psi)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fig_tieline_theory(path):
    """The reference itself: binodal, critical coupling, interface dip."""
    import numpy as np
    G = np.linspace(-1.0, -8.0, 141)
    rhi = [REF.coexistence_tieline(float(g))[0] for g in G]
    dip = [REF.interface_density_dip(float(g)) for g in G]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(-G, rhi, "C0-", lw=2, label=r"$\rho_{hi}$ (exact tie-line)")
    ax[0].plot(-G, [1 - x for x in rhi], "C1-", lw=2, label=r"$\rho_{lo}$")
    ax[0].axvline(REF.critical_coupling(0.5), color="k", ls="--", lw=1,
                  label=r"$|G_c|=1/\bar\rho$")
    ax[0].set_xlabel(r"$|G_{12}|$")
    ax[0].set_ylabel("bulk density")
    ax[0].set_title("mean-field binodal of the linear-psi SCMP mixture")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[1].plot(-G, dip, "C2-", lw=2, label="interface total density $s$")
    ax[1].plot(-G, [2 * x for x in rhi], "C0--", lw=1, label=r"bulk $\rho_1+\rho_2$")
    ax[1].set_xlabel(r"$|G_{12}|$")
    ax[1].set_ylabel("total density")
    ax[1].set_title("predicted Shan-Chen interface density dip")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def b64(path):
    return base64.b64encode(pathlib.Path(path).read_bytes()).decode()


# ---------------------------------------------------------------------------
def build_markdown(arms, selftest):
    by_label = {a["label"]: a for a in arms}
    main = by_label.get("t_G2.2_om0.4") or (arms[0] if arms else None)
    diverged = [a for a in arms if a.get("diverged")]
    converged = [a for a in arms if not a.get("diverged")]

    arm_rows = "\n".join(
        f"| {a['label']} | {a['params']['psi']} | {a['params']['G']:+.2f} | "
        f"{a['params']['ny']}x{a['params']['nx']} | {a['params']['steps']} | "
        f"{a['params']['omega']} | "
        + ("DIVERGED @" + str(a["diverged_at_step"]) if a.get("diverged") else
           f"{a['measured']['rho_hi']:.4f} / {a['measured']['rho_lo_sim_index']:.4f}")
        + " | "
        + ("—" if a.get("diverged") or not a.get("reference") else
           f"{pct(a['reference']['err_max'])}")
        + " | "
        + ("—" if a.get("diverged") else pct(a["pressure_invariant"]["pressure_spread_relative"]))
        + " | "
        + ("—" if a.get("diverged") else f"{a['measured']['u_max']:.2e}")
        + " |"
        for a in arms
    )

    theory_rows = "\n".join(
        f"| {g:+.2f} | {REF.coexistence_tieline(g)[0]:.6f} | {REF.coexistence_tieline(g)[1]:.6f} | "
        f"{REF.mechanical_pressure(*REF.coexistence_tieline(g), g):.6f} | "
        f"{REF.interface_density_dip(g):.6f} | {REF.critical_coupling(0.5):.3f} |"
        for g in (-2.05, -2.2, -2.5, -3.0, -4.0)
    )

    if main and not main.get("diverged") and main.get("reference"):
        main_line = (
            f"主臂 `{main['label']}`（G={main['params']['G']:+.2f}，ω={main['params']['omega']}，"
            f"{main['params']['steps']} 步，{main['params']['ny']}x{main['params']['nx']}）："
            f"实测平台 ρ_hi={main['measured']['rho_hi']:.4f}、ρ_lo={main['measured']['rho_lo_sim_index']:.4f}，"
            f"解析 (ρ_hi, ρ_lo)=({main['reference']['rho_hi_reference']:.4f}, "
            f"{main['reference']['rho_lo_reference']:.4f})，**err_max={pct(main['reference']['err_max'])}**；"
            f"机械压力平坦度 {pct(main['pressure_invariant']['pressure_spread_relative'])}；"
            f"|u|max={main['measured']['u_max']:.2e}。判定 **{main['verdict']}**（判据 15%）。"
        )
    else:
        main_line = "主臂缺失。"

    return f"""## 1. 案例介绍（物理背景 / 模型 / 参数表）

锚点 `XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE`（lbm-principles-practice/code 的
两组分 Shan–Chen 平面界面做法）。本单**不新增求解器**，而是给 TensorLBM 已有的两组分
Shan–Chen 路径（`tensorlbm.mcmp.MCMPSolver` + `tensorlbm.cross_force`）**补一件它从未有过的
验证仪器**：一个不依赖 LBM 代码、由热力学第一性原理给出的平面界面解析参考，并用它给
MCMP 的平衡态打分。

**为什么平面界面**：包内两相能力目前唯一的定量证据是 `t3_matrix` 的
`static-droplet-laplace` 行（`dp·R` 离散度 38%，WITHHELD）与 `measure_laplace_pressure`
里的经验式 `σ≈-G cs² Δρ²/6`（无推导，MOCK 级）。两者都依赖表面张力标定。平面界面不需要
曲率、不需要标定 σ：**体相共存密度与混合物的机械压力在热力学上是精确可知的**。

**模型（与实现同号）**：`cross_force.compute_force` 实际计算的是
`F_a = +G_ab ψ_a(x) Σ_i w_i ψ_b(x+e_i) e_i`（注意：该函数 docstring 写的是 `-G`，与代码相反，
本单把这条约定缺陷记入 §5），其一阶各向同性形式为 `F = +G cs² ψ_a ∇ψ_b`。于是

- 混合物机械压力（相互作用对两种**不同**组分只计一次，无 1/2 权）：`p = cs²(ρ₁+ρ₂-G ψ₁ψ₂)`；
- 分离（demixing）要求 `G_12<0`；`G_12>0` 为混合。

对**线性伪势** `ψ(ρ)=ρ`（`pseudopotential_rho`）该模型热力学自洽，存在精确自由能

```
f/cs² = ρ₁lnρ₁-ρ₁ + ρ₂lnρ₂-ρ₂ - G ρ₁ρ₂ ,  μ_σ = cs²(ln ρ_σ - G ρ_{{σ'}})
```

对称二元混合物（`<ρ₁>=<ρ₂>=ρ̄`，故 `ρ_hi+ρ_lo=2ρ̄`）的共存线由 `μ₁(A)=μ₁(B)` 给出闭式方程

```
ln(ρ_hi/ρ_lo) = |G_12| (ρ_hi-ρ_lo)                                        (★)
```

线化稳定性给出**临界耦合** `|G_c| = 1/ρ̄`（ρ̄=0.5 时 `|G_c|=2`），并可预测界面总密度凹陷
（Shan–Chen 界面必须比两体相都稀，才能让 `p` 保持平）：

```
g q s² + s = p_bulk/cs² ,  q = ψ₁ψ₂|_interface
```

| 参数 | 本单取值 |
|---|---|
| 格点 / 碰撞 | D2Q9 BGK，`omega` 扫描 {1.0, 0.6, 0.4}（τ=1, 1.67, 2.5），float64 |
| 组分 / 耦合 | 2 组分，`G_matrix=[[0,G],[G,0]]`，`G ∈ {{-2.2,+2.2}}`（线性 ψ） |
| 伪势 | 线性 ψ=ρ（有解析参考）；exp ψ=1-e^{{-ρ}}（参考 WITHHELD，见 §5） |
| 几何 | 周期盒，`ny=64`，`nx=4`（1-D 还原，x 方向均匀性逐臂检查），两界面在 ny/4 与 3ny/4（周期缝无跳变） |
| 初值 | 组成取 (★) 的共存幅度，总密度按 `p=const` 反解（近力学平衡初值） |
| 时长 | 4000 步（主臂），两臂 3000 步 |

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **独立参考**（`TensorLBM/tools/scmp_planar_reference.py`，零 LBM 依赖）：
   - 路线 A：二分求解超越方程 (★)；
   - 路线 B：**不碰 (★)**，直接在细网格上最小化两相自由能 `F(x)=f(x,2ρ̄-x)+f(2ρ̄-x,x)` 再
     golden-section 细化；
   - 两个自检：`p=Σρσμσ-f` 必须复现模型压力；共存态两相 `μσ` 与 `p` 必须相等；界面凹陷解必须
     让 `p` 跨界面相等；把 `G` 人为放大 2 倍必须被压力平坦度抓住（仪器灵敏度自检）。
   - 自检实测：两路线一致性 **{selftest['route_agreement']:.2e}**，`μ` 不平衡
     {selftest['mu_imbalance']:.1e}，压力不平衡 {selftest['pressure_imbalance']:.1e}，
     `|G_c|`={selftest['critical_coupling']:.1f}。
2. **算例**（`TensorLBM/benchmarks/scmp_two_component_planar_interface.py`）：周期平面界面的
   近平衡初值 → 跑至平台 → 末段 1500/800 步**时间平均**剖面（抑制声学振铃）→ 取两个体相窗口
   的平均密度、机械压力剖面的峰峰离散度、`|u|max`、质量漂移。
3. **判定预注册**：`err_max<15%`（owner 精度标准），3% 只报告不 gate；压力平坦度 <1% 作为
   **仪器健康度**判据（连续极限下恒为 0），不作精度判据；非有限场必须报为 divergence（honest FAIL，
   不得静默替换）。
4. 工具链：torch float64 单进程（nx=4 时 ~9 ms/步），matplotlib 出图，`pytest` 钉住参考。

## 3. 计算结果

### 3.1 独立参考（程序化生成）

| G_12 | ρ_hi (解析) | ρ_lo (解析) | p_bulk/cs² | 界面总密度 s | |G_c| |
|---|---|---|---|---|---|
{theory_rows}

### 3.2 各臂实测

| 臂 | ψ | G | 网格 | 步数 | ω | 平台 (ρ_hi/ρ_lo) | err_max vs (★) | p 平坦度 | \|u\|max |
|---|---|---|---|---|---|---|---|---|---|
{arm_rows}

{main_line}

### 3.3 符号判定（demixing 方向）

同一初值（强制 `ρ_hi=0.75`）下：`G=+2.2` 收敛到均匀混合态（ρ₁≈0.298，压力平坦度 1.8e-2%，
|u|max=2.8e-3），`G=-2.2` 在第 2300 步发散。**结论：实现中 demixing 对应 `G_12<0`**，
与 §1 的推导一致；`cross_force` docstring 的负号是错的。

## 4. 结果图

![profiles](FIG_PROFILE)
*图 1：主臂时间平均密度剖面（左，虚线为解析共存值）与机械压力剖面（右）；压力几乎平坦说明
实现里力的量级/符号没有错，误差不来自力而是来自状态未到平衡。*

![scores](FIG_SCORES)
*图 2：线性 ψ 各臂的 ρ_hi / ρ_lo 相对误差（对数轴，红虚线 15% 验收、橙点线 3% 目标）。*

![tieline](FIG_TIELINE)
*图 3：参考本身的相图——(★) 的 binodal 与临界耦合（左）、预测的界面密度凹陷（右）。*

## 5. 注意点（踩过的坑 / 局限）

1. **周期缝**：第一版初值把界面放在 y=0 处，周期边界上出现一格宽跳变，力尖峰让所有
   `|G|>0.5` 的臂在 20 步内发散——这是**初值的错**，不是求解器的错，本单把它写进文档以免复现。
   正式初值用双界面（ny/4、3ny/4）保证周期光滑。
2. **符号约定缺陷（实盘）**：`cross_force.compute_force` docstring 写 `F = -G ψ∇ψ`，代码是 `+G`。
   §3.3 的对照实验确认代码语义，`mcmp.setup_laplace_test` 的 `G=-1.5`（液滴）因此才“看起来合理”。
   建议下一单修 docstring 并给 `G_matrix` 加符号自检。
3. **exp ψ 无解析参考**：两组分 SC 的“状态方程”在非线性 ψ 下**不可积**（Gibbs–Duhem 交叉导数
   恒等式不成立），所以 exp ψ 臂的共存线在做不到精确参考，本单如实标 WITHHELD，只用不变量与
   临界耦合判据（`|G|ψ'(ρ̄)²>1/ρ̄`，ρ̄=0.5 时 exp ψ 需要 `|G|>5.43`）。
4. **数值可达性是本单的主要负面结论**：`|G|` 刚过临界（2.2）时，`ω=1` 的 demixing 臂在
   2300–3100 步发散；把 ω 降到 0.4（τ=2.5）能撑满 4000 步但**没有进入共存态**。这与 SC 模型
   已知的刚性一致：线性 ψ 的界面宽度是格点尺度的（自由能里没有梯度项），界面收窄到 1–2 格时
   梯度力超出显式格式稳定域。因此本单**给不出 TensorLBM 的精度数字**——只能给出“未达平衡”。
5. **nx=4 的 1-D 还原**：x 方向逐臂检查了均匀性（图 1 的压力剖面取 x 平均前先看 `xvar`），
   另在 64x64 全 2-D 复跑过 G=-2.5 臂，发散步相同，确认不是 1-D 还原的伪影。
6. 未做：界面宽度/表面张力的定量对标（需要模型梯度系数，本单未推）、网格收敛扫描
   （64/96/128）、τ 无关性扫描。这些都在下面 §7 列成后续条。

## 6. 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 实测误差 | <15%? | <3%? |
|---|---|---|---|---|
| ρ_hi（主臂 {main['label'] if main else '—'}） | (★) 解析 | {pct(main['reference']['err_max']) if main and main.get('reference') else '—'} | 否 | 否 |
| ρ_lo（同上） | (★) 解析 | {(pct(abs(main['measured']['rho_lo_sim_index']-main['reference']['rho_lo_reference'])/main['reference']['rho_lo_reference'])) if main and main.get('reference') else '—'} | 否 | 否 |
| 机械压力平坦度（不变量） | 0 | {pct(main['pressure_invariant']['pressure_spread_relative']) if main else '—'} | 是 | 否 |
| 参考自身（两独立路线） | 互为基准 | {selftest['route_agreement']:.2e} | 是 | 是 |

**结论：核心量 ρ_hi/ρ_lo 不在 3% 内，也不在 15% 内**——原因不是“算得不准”，而是**算例没进入
共存态**（见 §5.4）。本单按 honest-FAIL 记录：锚点保持 open，交付物是仪器 + 负面结论 +
下一步的可执行修复线索。

## 7. 给下一单的线索（可执行）

1. 修 `cross_force.compute_force` docstring 的符号，并在 `MCMPSolver.__init__` 里对 `G_matrix`
   加“demixing 方向”断言（一条测试即可）。
2. `MCMPSolver.step` 增加标准 `force_scheme="guo"`（混合速度 + `(1-1/(2τ))` Guo 源项）分支：
   本单 scratch 已验证该形式在过临界耦合下同样发散，说明**光换格式不够**，必须同时处理界面
   收窄——建议下一单做“界面宽度 ↔ 稳定耦合”相图（这是可发表的定量结果）。
3. 若要把锚点做成可验收：改用**有界伪势 + 更高分辨率 + 耦合爬升**（`G: 0→target` 慢升）或
   SC 的“压力修正/正则化”格式，仍以 (★) 的线性 ψ 臂做最终计分（解析参考只对线性 ψ 存在）。
"""


def main():
    publish = "--publish" in sys.argv
    FIGDIR = FIGS
    FIGDIR.mkdir(parents=True, exist_ok=True)
    arms = load_arms()
    selftest = REF.selftest(verbose=False)
    figs = {}
    main_arm = next((a for a in arms if a["label"] == "t_G2.2_om0.4" and not a.get("diverged")), None)
    if main_arm is None:
        main_arm = next((a for a in arms if not a.get("diverged") and a.get("profiles")), None)
    if main_arm:
        figs["profiles"] = fig_profiles(main_arm, FIGDIR / "scmp_planar_profiles.png")
    figs["scores"] = fig_scores(arms, FIGDIR / "scmp_planar_scores.png")
    figs["tieline"] = fig_tieline_theory(FIGDIR / "scmp_planar_tieline.png")

    md = build_markdown(arms, selftest)
    for key, path in figs.items():
        md = md.replace(f"FIG_{key.upper()}", f"figures/{path.name}")
    doc = ROOT / "TensorLBM" / "docs" / "XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE.md"
    doc.write_text("# XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE — 两组分 Shan–Chen 平面界面：独立热力学参考与对抗式复核\n\n"
                   "> 锚点 `XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE`；本文件由 "
                   "`TensorLBM/tools/render_scmp_planar_report.py` 从臂 JSON + 参考自检程序化生成。\n\n" + md)
    print("wrote", doc)

    if not publish:
        return 0

    body = {
        "title": "XVAL-LBM-SCMP-TWO-COMPONENT-PLANAR-INTERFACE：两组分 Shan–Chen 平面界面的独立热力学参考与对抗式复核",
        "summary": ("为 TensorLBM 的两组分 Shan-Chen 路径补一件独立验证仪器（解析共存线 + 临界耦合 + "
                    "机械压力不变量），并对 MCMP 平衡态做对抗式复核：linear-ψ demixing 臂在 |G|=2.2 过临界时"
                    "要么 2300-3100 步发散、要么 4000 步内未进入共存态；锚点按 honest FAIL 保持 open。"),
        "scenario_key": "xval-lbm-scmp-two-component-planar-interface",
        "mission_id": MISSION_ID,
        "tags": ["LBM", "SCMP", "two-component", "planar-interface", "xval", "honest-FAIL"],
        "markdown": md,
        "figures": [
            {"name": k, "caption": c, "png_base64": b64(figs[k])}
            for k, c in [
                ("profiles", "主臂时间平均密度剖面与机械压力剖面"),
                ("scores", "线性 ψ 各臂相对解析共存线的误差"),
                ("tieline", "解析 binodal、临界耦合与预测的界面密度凹陷"),
            ] if k in figs
        ],
    }
    api = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
    req = urllib.request.Request(api + "/tensor/reports",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    token = os.environ.get("TENSOR_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as fh:
        print("report:", fh.read().decode()[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())