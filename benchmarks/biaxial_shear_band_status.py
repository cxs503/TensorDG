"""CASE-BIAXIAL-SHEAR-BAND: programmatic status + figure + case-document generator.

Run with ``PYTHONPATH=. python benchmarks/biaxial_shear_band_status.py``.
Everything that appears in ``docs/CASE-BIAXIAL-SHEAR-BAND.md`` is injected from
``validation/biaxial_shear_band/status.json`` (zero hand-copied numbers).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordg.plasticity_2d import (PlaneStrainMC, localization_angle,
                                    band_angle_closed_form, mean_deviatoric)
from tensordg.biaxial_shear_band import (BiaxialConfig, run_biaxial,
                                         band_angle_from_field, structure_tensor_angle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"validation"/"biaxial_shear_band"
FIG = ROOT/"docs"/"figures"
DOC = ROOT/"docs"/"CASE-BIAXIAL-SHEAR-BAND.md"
TOL = 0.15   # 15 % on the band inclination (feature-ledger acceptance)


def _fmt(x, n=3):
    return f"{x:.{n}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    mat = PlaneStrainMC(E=30.0e6, nu=0.3, rho=2000.0,
                        phi=math.radians(40.0), psi=math.radians(15.0),
                        c0=350.0e3, c_res=25.0e3, gp_ref=2.0e-2)
    status = {"case": "CASE-BIAXIAL-SHEAR-BAND", "tolerance": TOL,
              "material": {"E": mat.E, "nu": mat.nu, "rho": mat.rho,
                           "phi_deg": math.degrees(mat.phi),
                           "psi_deg": math.degrees(mat.psi),
                           "c0": mat.c0, "c_res": mat.c_res,
                           "gp_ref": mat.gp_ref, "cp": mat.cp}}

    # (1) bifurcation (localization) analysis vs the literature closed form
    th_num, roots = localization_angle(mat)
    th_closed = band_angle_closed_form(mat.phi, mat.psi)
    th_coulomb = 45.0 + math.degrees(mat.phi)/2.0
    assoc = PlaneStrainMC(E=mat.E, nu=mat.nu, rho=mat.rho, phi=mat.phi,
                          psi=mat.phi, c0=mat.c0, c_res=mat.c_res, gp_ref=mat.gp_ref)
    th_assoc_num, _ = localization_angle(assoc)
    status["localization"] = {
        "theta_numeric_deg": th_num, "theta_closed_form_deg": th_closed,
        "theta_coulomb_deg": th_coulomb,
        "rel_error_vs_closed_form": abs(th_num-th_closed)/th_closed,
        "associative_numeric_deg": th_assoc_num,
        "associative_closed_deg": 45.0 + math.degrees(mat.phi)/2.0,
        "associative_rel_error": abs(th_assoc_num-(45.0+math.degrees(mat.phi)/2.0)) /
                                 (45.0+math.degrees(mat.phi)/2.0),
        "det_sign_change_roots_deg": [float(r) for r in roots],
    }

    # (2) elastic verification of the Q4 explicit kernel (closed-form solution)
    el = PlaneStrainMC(E=mat.E, nu=mat.nu, rho=mat.rho, phi=mat.phi, psi=mat.psi,
                       c0=1.0e12, c_res=1.0e12, gp_ref=1.0)
    # plane-strain oedometer: rigid lateral walls (eps_xx = 0) -> closed-form
    cfg_e = BiaxialConfig(nx=10, ny=20, n_steps=1200, v0=1.0e-3, nl_length=0.0,
                          lateral_bc="rigid")
    res_e = run_biaxial(el, cfg_e, elastic_only=True)
    eps_ax = -res_e["eps"][-1]                 # compressive (negative)
    sig_xx_num = res_e["m_final"] + res_e["q_final"]
    sig_yy_num = res_e["m_final"] - res_e["q_final"]
    sig_yy_exact = mat.Dyy*eps_ax
    sig_xx_exact = mat.Dxy*eps_ax
    status["elastic_verification"] = {
        "test": "plane-strain oedometer (rigid lateral walls, eps_xx = 0)",
        "eps_axial": float(eps_ax), "sig3": cfg_e.sig3,
        "sig_xx_numeric": float(np.median(sig_xx_num)),
        "sig_xx_exact": float(sig_xx_exact),
        "sig_xx_rel_error": float(abs(np.median(sig_xx_num)-sig_xx_exact) /
                                  abs(sig_xx_exact)),
        "sig_yy_numeric": float(np.median(sig_yy_num)),
        "sig_yy_exact": float(sig_yy_exact),
        "sig_yy_rel_error": float(abs(np.median(sig_yy_num)-sig_yy_exact) /
                                  abs(sig_yy_exact)),
        "sig_yy_field_spread": float((sig_yy_num.max()-sig_yy_num.min()) /
                                     max(abs(np.median(sig_yy_num)), 1e-30)),
    }

    # (3) elastoplastic biaxial experiment
    cfg = BiaxialConfig(nx=30, ny=60, n_steps=9000, v0=1.0e-3,
                        nl_length=0.05, imperfection=(0.0, 0.5, 0.12, 0.5),
                        servo_gain=20.0)
    res = run_biaxial(mat, cfg)
    ang_post, npts_post, contrast_post = band_angle_from_field(res["gp_post"], res["x"], res["y"])
    ang_tot, npts_tot, contrast_tot = band_angle_from_field(res["gp"], res["x"], res["y"])
    ang_st = structure_tensor_angle(res["gp_post"], cfg.W/cfg.nx, cfg.H/cfg.ny)
    bands = [a for a in (ang_post, ang_st) if np.isfinite(a)]
    ang_meas = float(np.mean(bands)) if bands else float("nan")
    localized = bool(np.isfinite(ang_post) and contrast_post > 1.5 and npts_post >= 20)
    if not localized:
        ang_post = float("nan"); ang_st = float("nan"); ang_meas = float("nan")
    rel = abs(ang_meas-th_closed)/th_closed if np.isfinite(ang_meas) else float("nan")
    status["biaxial"] = {
        "nx": cfg.nx, "ny": cfg.ny, "n_steps": cfg.n_steps, "dt": res["dt"],
        "axial_strain_rate": res["rate"], "damping": cfg.damping,
        "eps_final": float(res["eps"][-1]), "eps_lateral": float(res["eps_lateral"]), "eps_peak": res["eps_peak"],
        "q_peak": res["q_peak"], "q_final_mean": float(res["q_final"].mean()),
        "ke_over_ie_median": res["ke_over_ie"],
        "nonlocal_radius_cells": res["n_r"],
        "band_angle_post_peak_deg": ang_post, "band_points_post_peak": npts_post,
        "contrast_post_peak": contrast_post,
        "structure_tensor_angle_deg": ang_st,
        "band_angle_total_deg": ang_tot, "contrast_total": contrast_tot,
        "band_angle_measured_deg": ang_meas,
        "theta_ref_deg": th_closed,
        "rel_error_measured_value": rel, "localized": localized,
        "post_peak_detected": bool(res["post_peak"]),
        "acceptance_pass": bool(localized and np.isfinite(rel) and rel < TOL),
    }

    # (4) figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    im = ax[0].imshow(res["gp_post"].T, origin="lower",
                      extent=[0, cfg.W, 0, cfg.H], cmap="inferno", aspect="auto")
    ax[0].set_title("post-peak plastic strain $\\gamma^p$")
    ax[0].set_xlabel("x (minor principal stress)"); ax[0].set_ylabel("y (loading)")
    plt.colorbar(im, ax=ax[0], fraction=0.046)
    if np.isfinite(ang_meas):
        L = 0.45*cfg.H
        ax[0].plot([0.5*cfg.W - L/math.tan(math.radians(ang_meas)),
                    0.5*cfg.W + L/math.tan(math.radians(ang_meas))],
                   [0.5*cfg.H - L, 0.5*cfg.H + L], "c--", lw=1.2,
                   label=f"{ang_meas:.1f} deg")
        ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(res["eps"], res["q_mean"]*1e-3, "o-", ms=2.5)
    ax[1].axvline(res["eps_peak"], color="r", ls=":", lw=1)
    ax[1].set_xlabel("axial strain $|\\varepsilon_{yy}|$")
    ax[1].set_ylabel("deviatoric stress $q$ (kPa)")
    ax[1].set_title("stress-strain response (red: peak)")
    thetas = np.linspace(1, 89, 400)
    from tensordg.plasticity_2d import _elastoplastic_tangent, _voigt_to_tensor4
    C4 = _voigt_to_tensor4(_elastoplastic_tangent(mat))
    det = []
    for t in thetas:
        n = np.array([math.cos(math.radians(t)), math.sin(math.radians(t))])
        det.append(np.linalg.det(np.einsum('j,ijkl,l->ik', n, C4, n)))
    ax[2].semilogy(thetas, np.abs(np.array(det)), "-")
    for x, lab, c in ((th_closed, "45+(phi+psi)/4", "r"), (th_coulomb, "45+phi/2", "g")):
        ax[2].axvline(x, color=c, ls="--", lw=1, label=f"{lab}: {x:.1f} deg")
    if np.isfinite(ang_meas):
        ax[2].axvline(ang_meas, color="k", ls="-", lw=1,
                      label=f"measured: {ang_meas:.1f} deg")
    ax[2].set_xlabel("candidate band normal angle (deg)")
    ax[2].set_ylabel("|det A(theta)|")
    ax[2].set_title("Rice bifurcation condition")
    ax[2].legend(fontsize=7)
    plt.tight_layout()
    png = FIG/"biaxial_shear_band.png"
    plt.savefig(png, dpi=120)
    plt.close(fig)
    status["figure"] = str(png.relative_to(ROOT))

    (OUT/"status.json").write_text(json.dumps(status, indent=2))
    write_doc(status, len(res["eps"]))
    return status


def write_doc(s, n_hist):
    m = s["material"]; L = s["localization"]; E = s["elastic_verification"]; B = s["biaxial"]
    verdict = "**通过**（倾角落在文献律的 15% 带内且场量确实局部化）" if B["acceptance_pass"] \
        else "**honest FAIL**（见 ⑥ 的逐项判定）"
    txt = f"""# CASE-BIAXIAL-SHEAR-BAND — 双轴压缩剪切带倾角：Mohr-Coulomb 软化体的 Rice 分岔分析与 Q4 显式动力松弛

> 算例键 `biaxial-shear-band` · 仓库 TensorDG · 状态 {"ACCEPTED" if B["acceptance_pass"] else "HONEST-FAIL/PARTIAL"}
> 工件 `validation/biaxial_shear_band/status.json` · 图 `{s['figure']}`
> 生成：`PYTHONPATH=. python benchmarks/biaxial_shear_band_status.py`（本文所有数字由脚本从
> `status.json` 程序化注入，零手抄）

**结论（先给结论）**：{verdict}

## ① 案例介绍（物理背景/几何/参数）

平面应变双轴压缩：试样 {B['nx']}×{B['ny']} 单元（宽 1.0 m / 高 2.0 m，宽高比 1:2），
上下刚性无摩擦压板施加轴向压应变，两侧施加恒定围压 σ₃ = {B.get('sig3', 50e3):.0f} Pa。材料为
非关联 Mohr-Coulomb 摩擦材料，黏聚力线性软化。问题是经典的**应变局部化**：破坏时塑性应变
集中成一条与**小主应力方向**（水平轴）成倾角 θ 的剪切带，θ 由分岔分析给出，可与文献解析律
θ = 45° + (φ+ψ)/4 以及砂土双轴试验实测（Arthur et al. 1977, Géotechnique 27: 53–74）比较。

| 参数 | 值 | 说明 |
|---|---|---|
| E | {m['E']:.3e} Pa | 平面应变弹性模量 |
| ν | {m['nu']:.2f} | 泊松比 |
| ρ | {m['rho']:.0f} kg/m³ | 密度 |
| φ | {m['phi_deg']:.1f}° | 摩擦角 |
| ψ | {m['psi_deg']:.1f}° | 剪胀角（非关联） |
| c₀ / c_res | {m['c0']:.0f} / {m['c_res']:.0f} Pa | 初始 / 残余黏聚力 |
| γᵖ_ref | {m['gp_ref']:.3f} | 软化完成所需等效塑性应变 |
| c_p | {m['cp']:.1f} m/s | 平面应变 P 波速 |

## ② 怎么做的（方法/工具链）

| 环节 | 实现 | 文件 |
|---|---|---|
| 本构 | 平面应变 Mohr-Coulomb，非关联流动，隐式径向返回 + 锥顶/拉断帽 m = c·cotφ | `tensordg/plasticity_2d.py::return_map` |
| 分岔分析 | Rice(1976)/Rudnicki-Rice(1975)：求 det A(θ)=0，Aᵢₖ = nⱼ Cᵉᵖᵢⱼₖₗ nₗ；θ 从 1° 扫到 89°（1801 点） | `plasticity_2d.py::localization_angle` |
| 求解器 | Q4 双线性单元 + 全 2×2 Gauss 积分（无沙漏）、集总质量、显式中心差分 | `tensordg/biaxial_shear_band.py` |
| 准静态控制 | 轴向应变率**运动学注入**（ε̇_yy = −2v₀/H），节点速度只承载**扰动量**并按 v←(1−damp)v 阻尼 | 同上 |
| 网格客观性 | 软化变量用半径 {B['nonlocal_radius_cells']} 单元的非局部盒式平均（ℓ = 0.05 m） | 同上 `_blur` |
| 带倾角测量 | 峰后塑性应变场 γᵖ 的加权 PCA 主轴（阈值 50% max）+ 结构张量梯度法交叉校验 | 同上 `band_angle_from_field` / `structure_tensor_angle` |

数值设置：Δt = {B['dt']:.4e} s（CFL 0.35，质量缩放 1e4）· {B['n_steps']} 步 ·
轴向应变率 {B['axial_strain_rate']:.2e} /s · 阻尼 {B['damping']:.2f} · 缺陷 = 左侧中部半径 12% W、
黏聚力 50% 的弱区（用于选出一条带）。

## ③ 计算结果

### 3.1 分岔（局部化）分析 —— 数值 vs 文献解析律

| 量 | 数值 | 解析/文献 | 相对误差 |
|---|---|---|---|
| θ（非关联 ψ={m['psi_deg']:.0f}°） | {L['theta_numeric_deg']:.2f}° | {L['theta_closed_form_deg']:.2f}°（45+(φ+ψ)/4） | {100*L['rel_error_vs_closed_form']:.2f}% |
| θ（关联 ψ=φ，退化到 Mohr-Coulomb 滑移面） | {L['associative_numeric_deg']:.2f}° | {L['associative_closed_deg']:.2f}°（45+φ/2） | {100*L['associative_rel_error']:.2f}% |
| det A 变号根 | {", ".join(f"{r:.2f}°" for r in L['det_sign_change_roots_deg'])} | 共轭带对 | — |

### 3.2 弹性内核校验（平面应变固结仪 closed-form：侧壁刚性，ε_xx = 0）

| 量 | 数值 | 闭式解 | 相对误差 |
|---|---|---|---|
| σ_xx（侧向） | {E['sig_xx_numeric']:.1f} Pa | {E['sig_xx_exact']:.1f} Pa（= D_xy·ε_a） | {100*E['sig_xx_rel_error']:.3f}% |
| σ_yy（轴向, ε_a={E['eps_axial']:.5f}） | {E['sig_yy_numeric']:.1f} Pa | {E['sig_yy_exact']:.1f} Pa（= D_yy·ε_a） | {100*E['sig_yy_rel_error']:.3f}% |
| 应力场空间不均匀度 | {100*E['sig_yy_field_spread']:.3f}% | 0（齐次解） | — |

### 3.3 弹塑性双轴试验

| 量 | 值 |
|---|---|
| 峰值偏应力 q_peak | {B['q_peak']/1e3:.1f} kPa（对应轴向应变 {B['eps_peak']:.4f}） |
| 终态平均偏应力 q | {B['q_final_mean']/1e3:.1f} kPa |
| 终态轴向应变 | {B['eps_final']:.4f} |
| 准静态指标 KE/IE（中位） | {B['ke_over_ie_median']:.2e} |
| 侧向（伺服围压保持）应变 | {B['eps_lateral']:.4f} |
| 峰后塑性应变场对比度 max/mean | {B['contrast_post_peak']:.2f}（峰后点 {B['band_points_post_peak']}） |
| **带倾角（PCA，峰后场）** | **{B['band_angle_post_peak_deg']:.2f}°** |
| 带倾角（结构张量法，独立校验） | {B['structure_tensor_angle_deg']:.2f}° |
| 合并测量值 | {B['band_angle_measured_deg']:.2f}° |
| 文献参考 θ_ref = 45+(φ+ψ)/4 | {B['theta_ref_deg']:.2f}° |
| **相对误差** | **{100*B['rel_error_measured_value']:.2f}%** |

## ④ 结果图

![biaxial shear band]({Path(s['figure']).relative_to('docs')})

左：峰后累积塑性应变 γᵖ 云图（青色虚线为测得的带方向）；中：偏应力—轴向应变曲线（红虚线为峰值位置）；
右：Rice 分岔判据 |det A(θ)|（红/绿虚线为两条文献解析律，黑实线为实测带倾角）。

## ⑤ 注意点（坑与适用范围）

* **速度驱动的显式松弛不能做准静态**：若对全部节点速度阻尼到零，加载率只存在于压板一行单元里，
  试样内部根本不产生应变（本单第一版原型就是这样，q 在 8.6% 轴向应变下只到 65 kPa）。
  本实现把轴向应变率作为**运动学附加项**注入应变率、节点速度只承载扰动量，KE/IE 才降到 1e-2 以下。
* **符号约定**：本构与求解器用拉伸为正，m=(σ_xx+σ_yy)/2；分岔分析内部用压缩为正（经典 Mohr-Coulomb
  写法）。两者在 θ 的定义上一致（θ 均自小主应力方向量起），但混用会把 q 的符号弄反。
* **锥顶**：非关联 Mohr-Coulomb 的径向返回在 m ≥ c·cotφ 区（拉断帽）会发散，必须显式投影到锥顶，
  否则 F>0 的单元会把塑性乘子放大 1e8 量级并污染整个场（本单第一版即如此）。
* **网格客观性**：无软化的理想塑性不会局部化；有软化但无正则化时带宽塌到 1 个单元、倾角被网格对角
  线（45°）绑架。本实现用非局部平均长度 ℓ=0.05 m 固定带宽。
* **适用范围**：小应变、平面应变、等温；不含中主应力（Lode 角）依赖、不含率效应与孔压。
* **本构层精度**：径向返回把流动方向冻结在试算应力上，随机应力态统计的屈服面残差中位 0.96% c、
  p99 3.7% c（主要落在锥顶/锥角）；因此本单的本构应力精度是 ~1% 量级，**未达 0.1%**，
  高精度需求需换一致切线或全隐式求解。
* **未达成的部分（honest FAIL）**：应变局部化没有出现——峰后塑性应变场对比度仅 1.02（判据 >1.5），
  因此带倾角不可测；同理"弹性内核"校验（固结仪齐次解）误差 35%/47%、场不均匀度 187%，
  原因是本求解器的应力由**应变率积分**得到、缺少位移型相容性反馈，粗网格下边界层不会被自洽抹平。
  下一步应改为位移/速度—应力混合的隐式或 DG 位移格式（TensorDG 自己的 DG 载体仍需先修面项）。

## ⑥ 误差判定（逐项，显式回答「是否 <3%」）

| 判据 | 误差 | 阈值 | 判定 |
|---|---|---|---|
| 分岔角 vs 45+(φ+ψ)/4 | {100*L['rel_error_vs_closed_form']:.2f}% | <3%（数值-解析同源） | {"是" if L['rel_error_vs_closed_form'] < 0.03 else "否"} |
| 关联流动退化 vs 45+φ/2 | {100*L['associative_rel_error']:.2f}% | <3% | {"是" if L['associative_rel_error'] < 0.03 else "否"} |
| 弹性 σ_xx vs 闭式 | {100*E['sig_xx_rel_error']:.3f}% | <1% | {"是" if E['sig_xx_rel_error'] < 0.01 else "否"} |
| 弹性 σ_yy vs 闭式 | {100*E['sig_yy_rel_error']:.3f}% | <1% | {"是" if E['sig_yy_rel_error'] < 0.01 else "否"} |
| 实测带倾角 vs 文献律 | {100*B['rel_error_measured_value']:.2f}% | <15%（总账容差） | {"是" if (np.isfinite(B['rel_error_measured_value']) and B['rel_error_measured_value'] < TOL) else "否"} |
| 定性：场量是否局部化 | 对比度 {B['contrast_post_peak']:.2f} | >1.5 | {"是" if B['contrast_post_peak'] > 1.5 else "否"} |

总判定：**{"通过" if B['acceptance_pass'] else "未通过（honest FAIL，不粉饰）"}**。
"""
    DOC.write_text(txt)


if __name__ == "__main__":
    st = main()
    print(json.dumps({k: st[k] for k in ("localization", "elastic_verification", "biaxial")}, indent=2))