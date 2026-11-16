"""CKS1998 §5.5 二维 Riemann 问题的**对角对称线**精确参考（验证欠账闭合）。

问题
----
Lax & Liu (1998) config 3 的四象限 Riemann 问题（`tensordg.euler_dg_2d.LAX_LIU_CONFIG_3`）
是 CKS1998 论文里二维系统算例的代表。它**没有闭式解**，所以仓内既有验证只能做
结构性判据（有限性、守恒漂移、正性、发散集成门），拿不到"数值解 vs 精确解"的误差。

本模块给出该算例**沿对角线的严格精确参考**，从而把这条二维算例变成一个可判定的
定量度量（误差 = DG 解 vs 解析解，不是自己的另一种网格）。

数学依据（不是拟合、不是高分辨率自比）
--------------------------------------
1. config 3 的初值关于对角线 ``y = x`` **对称**：交换 ``x <-> y`` 并交换 ``u <-> v``
   后四个象限的数据逐项不变（LL/UR 自身 u=v；LR(0,1.206) <-> UL(1.206,0)）。
2. 旋转到法向/横向坐标 ``X = (x+y-1)/√2``、``Y = (y-x)/√2`` 后，该对称性意味着
   ``rho, p, u_X`` 是 ``Y`` 的偶函数、``u_Y`` 是奇函数。在 ``Y = 0`` 上逐条代入二维
   Euler 方程：连续方程与能量方程的 ``∂_Y(rho u_Y)``、``∂_Y(u_Y(E+p))`` 项是"奇函数
   在 0 处的导数 = 0"；X 动量方程同理；Y 动量方程里 ``∂_Y(rho u_Y² + p) = 0``（偶函数）
   且 ``u_Y(0) = 0``，恒等成立。于是**限制到对角线上的解精确满足一维 Euler 方程**。
3. 对角线上的初值只有两个状态：``X<0`` 是 LL 象限 ``(0.138, 1.206, 1.206, 0.3)``
   投影到法向（``u_n = (u+v)/√2``，横向速度 ``(v-u)/√2 = 0``）；``X>0`` 是 UR 象限
   ``(1.5, 0, 0, 1.5)``。故对角线上的二维解 = 这两个状态的一维 Riemann 问题的
   **精确解**，自变量为自相似坐标 ``xi = X/t``；参考值由 ``benchmarks.riemann_exact``
   （Toro 三波精确 Riemann 求解器，独立于 DG 求解器）给出。

参考因此是：
* **解析的**（不是另一种数值格式的高分辨率自比）；
* **可被否证的**：把初值改成"对角不对称"的变体（``ASYMMETRIC_CONFIG``，只把 LL 象限
  的 v 从 1.206 改成 0.3），同一个"对角线 = 一维精确解"的判据立刻大幅失效
  （``tests/test_riemann2d_diagonal_reference.py`` 里把这条写成断言）。

测量口径
--------
* 采样点**恰好落在对角线上**：第 ``i`` 个对角单元的参考坐标 ``xi_j``（Gauss 点）映射到
  物理点 ``x = y = x_left + (xi_j+1) h / 2``；DG 解在该点由模态多项式逐点求值
  （``Σ_mn c_mn P_m(xi_j) P_n(xi_j)``，与 ``tensordg.euler_dg_2d`` 的原始 Legendre 基
  一致），**不做单元平均、不做插值**。
* 参考值在同一物理点上取 ``xi = X/t`` 的精确一维 Riemann 解，故误差是纯离散误差
  （无数值参考的插值误差、无口径差）。
* 度量：对角线上密度的相对 L1 误差、若干 ``xi`` 台站的 rho/p 相对误差、以及
  ``y = x`` 对称残差（精确为 0；用来暴露破对称的离散化）。
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from benchmarks.riemann_exact import sample as exact_sample, star_state
from tensordg.euler_dg_2d import LAX_LIU_CONFIG_3, solve_euler_dg_2d_riemann

GAMMA = 1.4
S2 = float(np.sqrt(2.0))

def normal_states(config: Mapping[str, Tuple[float, float, float, float]]
                  ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """把 LL/UR 象限投影到法向 ``n = (1,1)/√2``：``u_n = (u+v)/√2``（见 docstring 第 3 条）。"""
    ll = config["lower-left"]
    ur = config["upper-right"]
    return ((ll[0], (ll[1] + ll[2]) / S2, ll[3]),
            (ur[0], (ur[1] + ur[2]) / S2, ur[3]))


#: 法向状态（默认 config 3）：对角线一维 Riemann 问题的左右状态。
NORMAL_LEFT, NORMAL_RIGHT = normal_states(LAX_LIU_CONFIG_3)

#: 否证用变体（**强**）：把 LL 象限的 ``v`` 取反号，使对角附近出现强横向剪切
#: （``u_Y = (v-u)/√2 = -1.706``），``y=x`` 对称被彻底破坏，从而"对角线 = 一维精确解"
#: 这一前提对它是**假的**。判据若仍说"吻合"，说明判据是空门（vacuous）。
ASYMMETRIC_CONFIG = {
    "lower-left": (0.138, 1.206, -1.206, 0.3),
    "lower-right": LAX_LIU_CONFIG_3["lower-right"],
    "upper-left": LAX_LIU_CONFIG_3["upper-left"],
    "upper-right": LAX_LIU_CONFIG_3["upper-right"],
}

#: 台站（自相似坐标 ``xi = X/t``）**由精确解自身的波系结构定义**，不手挑数字：
#: 左行激波速 ``s_shock``、接触速 ``u*``、右侧稀疏扇头 ``u_r + a_r`` 都从
#: ``benchmarks.riemann_exact`` 的精确解算出来，台站取这些区间的中点/四分点。
#: 这样换初值（例如否证用的不对称变体）台站跟着走，不会"挑一个好看的点"。
#: 波区 L1 的窗口外扩容差（自相似坐标单位）。**先定后跑**：0.5 约等于最外波一个
#: 数值厚度的量级（``sqrt(2) h / t`` 在 N=64、t=0.2 时约 0.11），只是把窗口边界推出
#: 波面之外，不参与任何判据调参。
WAVE_MARGIN = 0.5

STATION_FRACTIONS: Tuple[Tuple[str, float, str], ...] = (
    ("star_left", 0.5, "left_plateau"),   # 左波与接触之间（左侧星区平台）中点
    ("star_right", 0.5, "right_plateau"),  # 接触与右侧稀疏扇尾之间中点
    ("fan_mid", 0.5, "right_fan"),         # 稀疏扇中点（只有右波是稀疏扇时存在）
)


def wave_structure(final_time: float,
                   left: Tuple[float, float, float] = NORMAL_LEFT,
                   right: Tuple[float, float, float] = NORMAL_RIGHT,
                   gamma: float = GAMMA) -> Dict[str, float]:
    """精确解的波系速度：左行/右行激波速、接触间断速、稀疏扇头速。"""
    rho_l, u_l, p_l = left
    rho_r, u_r, p_r = right
    a_l = float(np.sqrt(gamma * p_l / rho_l))
    a_r = float(np.sqrt(gamma * p_r / rho_r))
    p_star, u_star = star_state(left, right, gamma)
    if p_star > p_l:      # 左波是激波
        ratio = p_star / p_l
        left_speed = u_l - a_l * float(np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)))
    else:                 # 左波是稀疏扇
        left_speed = u_l - a_l
    if p_star > p_r:
        ratio = p_star / p_r
        right_speed = u_r + a_r * float(np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * ratio + (gamma - 1.0) / (2.0 * gamma)))
        right_inner = right_speed          # 激波：内外同速
        right_kind = "shock"
    else:
        right_speed = u_r + a_r            # 稀疏扇头（外）
        rho_star_r = rho_r * (p_star / p_r) ** (1.0 / gamma)
        a_star_r = float(np.sqrt(gamma * p_star / rho_star_r))
        right_inner = u_star + a_star_r    # 稀疏扇尾（内）
        right_kind = "rarefaction"
    return {"left_wave": float(left_speed), "contact": float(u_star),
            "right_inner": float(right_inner), "right_outer": float(right_speed),
            "right_kind": right_kind, "star_pressure": float(p_star),
            "sound_left": a_l, "sound_right": a_r}


def discontinuities(**kwargs: Any) -> List[float]:
    """波系里"误差集中"的位置（自相似坐标）：度量与 smooth 掩码都从这里取。"""
    ws = wave_structure(0.0, **kwargs)
    positions = [ws["left_wave"], ws["contact"]]
    if ws["right_kind"] == "shock":
        positions.append(ws["right_inner"])
    else:                       # 稀疏扇两端是连续转折，但仍是误差集中处
        positions.extend([ws["right_inner"], ws["right_outer"]])
    return sorted(float(p) for p in positions)


def station_positions(**kwargs: Any) -> List[Tuple[str, float]]:
    """按波系结构给出台站的 ``xi``：每个台站落在**一个确定的波系区间内**。"""
    ws = wave_structure(0.0, **kwargs)
    out = []
    for name, frac, region in STATION_FRACTIONS:
        if region == "left_plateau":
            lo, hi = ws["left_wave"], ws["contact"]
        elif region == "right_plateau":
            lo, hi = ws["contact"], ws["right_inner"]
        else:                                        # 右侧稀疏扇
            if ws["right_kind"] != "rarefaction":
                continue
            lo, hi = ws["right_inner"], ws["right_outer"]
        out.append((name, lo + frac * (hi - lo)))
    return out


#: 兼容旧名（文档/测试里用的台站列表）
def STATIONS(**kwargs: Any) -> List[float]:  # noqa: N802 - 保留为大写访问器
    return [xi for _, xi in station_positions(**kwargs)]


def _normal_coordinate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``X = ((x-0.5)+(y-0.5))/√2``（域中心为四象限交点）。"""
    return ((np.asarray(x, dtype=np.float64) - 0.5)
            + (np.asarray(y, dtype=np.float64) - 0.5)) / S2


def exact_diagonal(xi: Iterable[float], final_time: float,
                   left: Tuple[float, float, float] = NORMAL_LEFT,
                   right: Tuple[float, float, float] = NORMAL_RIGHT,
                   gamma: float = GAMMA) -> Dict[str, np.ndarray]:
    """沿对角线的一维精确 Riemann 解；``xi = X/t`` 是自相似坐标（diaphragm 在 0）。"""
    xi = np.asarray(list(xi), dtype=np.float64)
    fields = exact_sample(xi * final_time, final_time, left, right,
                          diaphragm=0.0, gamma=gamma)
    fields = dict(fields)
    fields["xi"] = xi
    return fields


def station_table(stations: Iterable[Tuple[str, float]],
                  **kwargs: Any) -> List[Dict[str, Any]]:
    """台站上的精确参考值（供测试与文档交叉核对）。"""
    named = list(stations)
    fields = exact_diagonal([xi for _, xi in named], 1.0, **kwargs)
    out = []
    for k, (name, xi) in enumerate(named):
        out.append({"name": name, "xi": float(xi),
                    "density": float(fields["density"][k]),
                    "velocity": float(fields["velocity"][k]),
                    "pressure": float(fields["pressure"][k])})
    return out


def _legendre_row(xi: float, degree: int) -> np.ndarray:
    return np.polynomial.legendre.legvander(np.asarray([xi], dtype=np.float64), degree)[0]


def diagonal_samples(coeff: np.ndarray, ncells: int, degree: int,
                     final_time: float, quadrature: int = 3) -> Dict[str, np.ndarray]:
    """在对角线上逐点求值 DG 解，并返回同一批点的物理坐标与参考自变量 ``xi``。

    ``coeff`` 形状 ``[nx, ny, nbasis_x, nbasis_y, nvars]``（``tensordg.euler_dg_2d`` 的
    模态系数，原始 Legendre 基）。采样点为 ``quadrature`` 个 Gauss 点/单元，落在
    ``y = x`` 上（张量积基在同号节点上取值 ⇒ ``x`` 与 ``y`` 的参考坐标相同）。
    """
    coeff = np.asarray(coeff, dtype=np.float64)
    if coeff.ndim != 5:
        raise ValueError(f"coeff 形状应为 [nx, ny, nbx, nby, nvars]，得到 {coeff.shape}")
    if coeff.shape[0] != coeff.shape[1]:
        raise ValueError("对角线求值要求方形网格")
    h = 1.0 / ncells
    nodes = np.polynomial.legendre.leggauss(quadrature)[0]
    basis = np.polynomial.legendre.legvander(nodes, degree)          # [q, degree+1]
    xs = []
    values = []
    for i in range(ncells):
        c = coeff[i, i]                                              # [nbx, nby, nvars]
        # 同号节点 ⇒ (xi_j, eta_j) 落在对角线上；物理点 x = y = (i + (xi_j+1)/2) h
        kernel = np.einsum("jm,mn,jn->j", basis, c[..., 0], basis)    # 密度
        xs.append((i + 0.5 * (nodes + 1.0)) * h)
        values.append(kernel)
    x = np.concatenate(xs)
    rho = np.concatenate(values)
    if not final_time > 0.0:
        raise ValueError("final_time 必须为正（自相似坐标 xi = X/t）")
    X = _normal_coordinate(x, x)
    return {"x": x, "X": X, "xi": X / float(final_time), "density": rho}


def _pressure_profile(coeff: np.ndarray, ncells: int, degree: int,
                      quadrature: int = 3) -> List[float]:
    """对角线采样点上的 DG 压力（由模态系数逐点求值；与密度同一批点）。"""
    nodes = np.polynomial.legendre.leggauss(quadrature)[0]
    basis = np.polynomial.legendre.legvander(nodes, degree)
    out = []
    for i in range(ncells):
        c = coeff[i, i]
        vals = [np.einsum("jm,mn,jn->j", basis, c[..., var], basis)
                for var in range(4)]
        rho, mom_x, mom_y, energy = vals
        u = mom_x / rho
        v = mom_y / rho
        pressure = (GAMMA - 1.0) * (energy - 0.5 * rho * (u ** 2 + v ** 2))
        out.append(pressure)
    return np.concatenate(out).tolist()


def reflection_residual(coeff: np.ndarray) -> float:
    """``max|rho(x,y) - rho(y,x)| / max rho``（单元均值口径；对称解上精确为 0）。"""
    coeff = np.asarray(coeff, dtype=np.float64)
    rho = coeff[..., 0, 0, 0]
    scale = float(np.max(np.abs(rho)))
    if scale == 0.0:
        raise ValueError("密度全零，无法归一化对称残差")
    return float(np.max(np.abs(rho - rho.T)) / scale)


def measure(ncells: int, degree: int, final_time: float, *, cfl: float = 0.2,
            limiter: str = "tvb", tvb_constant: float = 1.0,
            states: Mapping[str, Tuple[float, float, float, float]] | None = None,
            quadrature: int = 3) -> Dict[str, Any]:
    """跑一次二维算例并给出对角线误差与台站误差。"""
    left, right = normal_states(states) if states is not None else (NORMAL_LEFT, NORMAL_RIGHT)
    x, y, coeff, meta = solve_euler_dg_2d_riemann(
        ncells=ncells, degree=degree, final_time=final_time, cfl=cfl,
        limiter=limiter, tvb_constant=tvb_constant, states=states,
    )
    coeff_np = coeff.detach().cpu().numpy()
    got = diagonal_samples(coeff_np, ncells, degree, final_time, quadrature=quadrature)
    ref = exact_diagonal(got["xi"], final_time, left, right)
    diff = got["density"] - ref["density"]
    # 相对 L1：以参考解的均值水平为尺度（避开激波处 |rho| 过小引起的比值爆炸）
    scale = float(np.mean(np.abs(ref["density"])))
    l1 = float(np.mean(np.abs(diff)) / scale)
    # 光滑区 L1：剔除各波位置 ±SMOOTH_EXCLUDE 的窗口（离散误差在那里是 O(jump)，不是
    # 精度问题）。窗口宽度在**看到结果之前**写定，按"波的数值厚度 ~4 单元"折算成
    # 自相似坐标的单位：h_xi = sqrt(2) h / t（对角线方向每单元跨 sqrt(2) h）。
    h_xi = np.sqrt(2.0) * (1.0 / ncells) / final_time
    mask = np.ones_like(got["xi"], dtype=bool)
    for xi_d in discontinuities(left=left, right=right):
        mask &= np.abs(got["xi"] - xi_d) > 2.0 * h_xi
    smooth_l1 = (float(np.mean(np.abs(diff[mask])) / scale) if mask.any() else float("nan"))
    smooth_samples = int(mask.sum())
    # 波区 L1：只在"解真正在变"的自相似窗口里量（默认窗口 = 最外两波各外扩 WAVE_MARGIN）。
    # 远场样本是常数、误差近似为 0，会把 L1 稀释掉；波区口径才能暴露波系的离散误差。
    ws_now = wave_structure(final_time, left, right)
    lo = ws_now["left_wave"] - WAVE_MARGIN
    hi = ws_now["right_outer"] + WAVE_MARGIN
    wave_mask = (got["xi"] >= lo) & (got["xi"] <= hi)
    wave_l1 = (float(np.mean(np.abs(diff[wave_mask])) / float(np.mean(np.abs(ref["density"][wave_mask]))))
               if wave_mask.any() else float("nan"))
    stations = []
    for name, xi_s in station_positions(left=left, right=right):
        idx = int(np.argmin(np.abs(got["xi"] - xi_s)))
        rho_ref = float(ref["density"][idx])
        stations.append({
            "name": name,
            "xi_request": float(xi_s),
            "xi_got": float(got["xi"][idx]),
            "density_got": float(got["density"][idx]),
            "density_ref": rho_ref,
            "density_rel_err": abs(rho_ref - float(got["density"][idx])) / abs(rho_ref),
        })
    return {
        "ncells": int(ncells),
        "degree": int(degree),
        "final_time": float(final_time),
        "cfl": float(cfl),
        "limiter": limiter,
        "tvb_constant": float(tvb_constant),
        "diagonal_samples": int(got["x"].size),
        "diag_rho_l1_rel": l1,
        "diag_rho_l1_wave": wave_l1,
        "wave_window": [float(lo), float(hi)],
        "wave_samples": int(wave_mask.sum()),
        "diag_rho_l1_smooth": smooth_l1,
        "smooth_samples": smooth_samples,
        "smooth_window_h_xi": float(h_xi),
        "symmetry_residual": reflection_residual(coeff_np),
        "stations": stations,
        "reference": {"left": list(left), "right": list(right),
                      "star_pressure_velocity": list(star_state(left, right))},
        "wave_structure": ws_now,
        "solver_meta": {k: meta[k] for k in ("nx", "ny", "dofs") if k in meta},
        "profile": {"xi": got["xi"].tolist(),
                    "density_dg": got["density"].tolist(),
                    "density_exact": ref["density"].tolist(),
                    "pressure_dg": _pressure_profile(coeff_np, ncells, degree, quadrature),
                    "pressure_exact": ref["pressure"].tolist()},
    }


def convergence(ladder: Iterable[int], degree: int, final_time: float,
                **kwargs: Any) -> Dict[str, Any]:
    """网格梯子上的对角线 L1 误差与观测阶（相邻比值按 ``log(e1/e2)/log(h1/h2)``）。"""
    rows = [measure(n, degree, final_time, **kwargs) for n in ladder]
    order = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        ratio = coarse["diag_rho_l1_rel"] / fine["diag_rho_l1_rel"]
        order.append(float(np.log(ratio) / np.log(fine["ncells"] / coarse["ncells"])))
    return {"rows": rows, "observed_order": order}


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CKS1998 二维 Riemann 对角线精确参考")
    parser.add_argument("--ncells", type=int, default=64)
    parser.add_argument("--degree", type=int, default=1)
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--ladder", type=int, nargs="*", default=None)
    parser.add_argument("--asymmetric", action="store_true",
                        help="用破坏 y=x 对称的初值跑同一判据（否证实验）")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    states = ASYMMETRIC_CONFIG if args.asymmetric else None
    if args.ladder:
        doc = convergence(args.ladder, args.degree, args.time, cfl=args.cfl, states=states)
        doc["asymmetric"] = bool(args.asymmetric)
    else:
        doc = measure(args.ncells, args.degree, args.time, cfl=args.cfl, states=states)
        doc["asymmetric"] = bool(args.asymmetric)
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())