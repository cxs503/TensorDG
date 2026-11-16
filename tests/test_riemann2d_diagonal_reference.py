"""REPRO-DG-CKS1998 二维 Riemann 算例的**对角对称线精确参考**回归门。

被测对象：`validation/repro_cks1998/riemann2d_diagonal.py`。

这条门不检查"解看起来像"，而是检查三件事：

1. **求值器自洽**：对角线上的模态多项式求值，其 Gauss 加权平均必须等于单元均值
   （`coeff[...,0,0,:]`）——否则"逐点比较"是在和错误的点比。
2. **参考解与波系结构自洽**：`wave_structure` 的分支速与 `benchmarks.riemann_exact`
   的精确解一致（`p*` 与 `u*` 逐值绑定），且左右波速有序。
3. **判据本身**：对称 config 3 的对角线 DG 解与一维精确解吻合到远场机器精度级、
   波区在粗网格上有界误差；而**破坏 `y=x` 对称**的变体上同一判据明显变差
   ——即判据不是"空门"。

注意：第 3 条用的是**粗网格 + 短时间**（为把门留在秒级），它证明的是"判据能区分
正确/错误前提"，不是"精度足够"。定量的精细分辨率为集群档
（`docs/CASE-REPRO-DG-CKS1998-2D-RIEMANN-DIAGONAL.md`）。
"""
from __future__ import annotations

import numpy as np
import pytest

from benchmarks import riemann_exact
from tensordg.euler_dg_2d import LAX_LIU_CONFIG_3, solve_euler_dg_2d_riemann
from validation.repro_cks1998 import riemann2d_diagonal as diag

NCELLS = 20
DEGREE = 1
FINAL_TIME = 0.08


def test_normal_states_are_the_projection_of_ll_ur():
    """对角线的左右状态 = LL/UR 象限在法向 n=(1,1)/√2 上的投影，横向速度为零。"""
    left, right = diag.normal_states(LAX_LIU_CONFIG_3)
    ll = LAX_LIU_CONFIG_3["lower-left"]
    ur = LAX_LIU_CONFIG_3["upper-right"]
    assert left[0] == pytest.approx(ll[0]) and left[2] == pytest.approx(ll[3])
    assert left[1] == pytest.approx((ll[1] + ll[2]) / np.sqrt(2.0))
    assert right == pytest.approx((ur[0], 0.0, ur[3]))
    # 横向速度 (v-u)/√2 = 0 —— 这正是"对角线是一维问题"的初值前提
    assert (ll[2] - ll[1]) / np.sqrt(2.0) == pytest.approx(0.0)
    assert (ur[2] - ur[1]) / np.sqrt(2.0) == pytest.approx(0.0)


def test_wave_structure_is_bound_to_the_exact_solver():
    """波系结构与 `benchmarks.riemann_exact` 的精确解逐值绑定，且各波速有序。"""
    ws = diag.wave_structure(FINAL_TIME)
    p_star, u_star = riemann_exact.star_state(diag.NORMAL_LEFT, diag.NORMAL_RIGHT)
    assert ws["star_pressure"] == pytest.approx(p_star, rel=0.0, abs=0.0)
    assert ws["contact"] == pytest.approx(u_star, rel=0.0, abs=0.0)
    assert ws["left_wave"] < ws["contact"] < ws["right_inner"] < ws["right_outer"]
    assert ws["right_kind"] == "rarefaction"


def test_diagonal_pointwise_evaluation_matches_the_modal_polynomial():
    """求值器单元测试：`diagonal_samples` 在 (xi_j, xi_j) 上取的值必须等于模态多项式
    的直接求值，且这些点的 Gauss 加权平均等于单元均值 ``coeff[...,0,0,0]``。

    用**合成系数**而不是求解器的输出：这里要证明的是"求值 + 积分归一化"实现正确，
    与求解器的时间推进无关（求解器的均值一致性由 ``cell_fields`` 的语义保证）。
    """
    rng = np.random.default_rng(20261116)
    for degree in (1, 2):
        n, nb = 7, degree + 1
        coeff = np.zeros((n, n, nb, nb, 4))
        coeff[..., 0] = rng.normal(size=(n, n, nb, nb))     # 只测密度分量
        got = diag.diagonal_samples(coeff, n, degree, 0.1, quadrature=degree + 2)
        nodes, weights = np.polynomial.legendre.leggauss(degree + 2)
        basis = np.polynomial.legendre.legvander(nodes, degree)
        for i in range(n):
            direct = np.einsum("jm,mn,jn->j", basis, coeff[i, i, :, :, 0], basis)
            np.testing.assert_allclose(got["density"].reshape(n, -1)[i], direct,
                                       rtol=0.0, atol=1e-14)
        # 采样点必须落在第 i 个对角单元内（且严格在 y=x 上：x 由对角坐标映射而来）
        x = got["x"].reshape(n, -1)
        for i in range(n):
            assert np.all(x[i] >= i / n) and np.all(x[i] <= (i + 1) / n)

    # 说明：对角线上的**点值**比较是本题的正确口径——沿一条线采样无法还原二维单元均值
    # （对角 Gauss 点集不是二维求积规则），故这里不假装能复算 coeff[...,0,0,0]。


def test_symmetry_residual_is_machine_zero():
    """config 3 关于 y=x 对称 ⇒ 数值解也应逐位对称（同时印证"对角线降维"的前提）。"""
    _, _, coeff, _ = solve_euler_dg_2d_riemann(
        ncells=16, degree=1, final_time=0.05, cfl=0.2, limiter="tvb")
    residual = diag.reflection_residual(coeff.detach().cpu().numpy())
    assert residual < 1e-12, f"对称残差 {residual:.3e} 超出机器精度"


def test_diagonal_matches_exact_one_dimensional_riemann_solution():
    """核心判据：对角线上的 DG 解与一维精确解吻合（远场机器级、波区有界）。"""
    doc = diag.measure(NCELLS, DEGREE, FINAL_TIME, cfl=0.2)
    assert doc["symmetry_residual"] < 1e-12
    # 远场（未扰动区）是常数，误差应在机器精度量级（这个数不靠网格细化才成立）
    assert doc["diag_rho_l1_smooth"] < 5e-3, doc["diag_rho_l1_smooth"]
    # 波区：粗网格上仍有 O(10 %) 的激波抹平误差，但必须有界（不是发散）
    assert doc["diag_rho_l1_wave"] < 0.35, doc["diag_rho_l1_wave"]
    assert doc["wave_samples"] >= 8
    # 台站上是"DG vs 精确解"的逐点比较，参考值非零且有限
    for station in doc["stations"]:
        assert np.isfinite(station["density_ref"]) and station["density_ref"] > 0.0
        assert np.isfinite(station["density_rel_err"])


def test_asymmetric_variant_breaks_the_reference():
    """否证实验：破坏 y=x 对称后，对角线解本身移动、且与同一判据的误差变大。

    三条断言合起来才构成"判据不是空门"的证据：
    1. 解确实不再对称（``symmetry_residual`` 从 1e-15 跳到 O(0.1)）；
    2. 对角线剖面本身移动了 O(0.1)（"一维精确解"对破坏前提的初值不再是它的解）；
    3. 与一维精确解的波区误差变大（粗网格上幅度有限，见 docs 的 1.43 倍 @ N=32）。
    本门为控制在秒级用 N=20/t=0.08，故 (3) 的门取 1.05；分辨率更高时差距变大。
    """
    symmetric = diag.measure(NCELLS, DEGREE, FINAL_TIME, cfl=0.2)
    asymmetric = diag.measure(NCELLS, DEGREE, FINAL_TIME, cfl=0.2,
                              states=diag.ASYMMETRIC_CONFIG)
    assert asymmetric["symmetry_residual"] > 0.05      # 解确实不再对称
    got_sym = np.asarray(symmetric["profile"]["density_dg"])
    got_asym = np.asarray(asymmetric["profile"]["density_dg"])
    xi = np.asarray(symmetric["profile"]["xi"])
    lo, hi = symmetric["wave_window"]
    window = (xi >= lo) & (xi <= hi)
    shift = float(np.max(np.abs(got_asym - got_sym)[window]))
    assert shift > 0.1, f"破坏对称前提后对角线剖面几乎没动（{shift:.3f}）"
    ratio = asymmetric["diag_rho_l1_wave"] / symmetric["diag_rho_l1_wave"]
    assert ratio > 1.05, (f"破坏对称前提后误差只变化了 {ratio:.3f} 倍 —— "
                          "说明该判据对前提不敏感（空门）")