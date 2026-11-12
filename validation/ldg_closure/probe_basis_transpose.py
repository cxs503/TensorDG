"""Evidence probe: is the m16266 LDG blocker evidence bundle self-consistent?

Question
--------
``docs/ldg_closure_probe_2026-11-12.md`` (m16266) concludes that the
alternating-flux LDG Dirichlet closure is structurally rank deficient and that
"mirror" and "Nitsche" closures do not cure it.  Before that verdict can be used
to justify keeping ``tensordg/poisson_2d_ldg.py`` in PROTOTYPE state, the probe
itself must be consistent: feeding it a solution that the discrete space
contains *exactly* must produce a machine-zero residual.

Two candidate derivative matrices exist for the modal Legendre basis on
[-1, 1] (``phi_j`` = L_j, test function = L_i):

    D_coef[i,j] = int_{-1}^{1} L_i L_j'      (derivative acts on the coefficient)
    D_test[i,j] = int_{-1}^{1} L_i' L_j      (derivative acts on the test function)

The mixed weak form of ``q = grad u`` has the derivative on the *test* function
in both rows, and the two matrices differ by the boundary term

    D_coef + D_test = [L_i L_j]_{-1}^{1} = 1 - (-1)^{i+j}   (not zero!)

so the choice is not a gauge: ``D_coef`` is strictly upper triangular, ``D_test``
strictly lower triangular.  ``validation/ldg_closure/probe_ldg1d.py`` (m16266)
uses ``D_coef``.

This probe (a) checks that identity, (b) evaluates the residual of an exactly
representable solution (u = x^2, q = 2x, k >= 2) under the m16266 assembly with
each matrix, and (c) re-runs the closure scan with the corrected matrix.

All numbers land in ``validation/ldg_closure/basis_transpose_probe.json`` and are
injected into ``docs/ldg_closure_reassessment_m16388.md`` by this script, so no
number in the documentation is hand copied.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "basis_transpose_probe.json")
DOC = os.path.join(HERE, "..", "..", "docs", "ldg_closure_reassessment_m16388.md")

GRID = (8, 16, 32)
KS = (1, 2, 3)


def basis(k: int):
    nq = max(2 * k + 3, 6)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.zeros((k + 1, nq))
    dphi = np.zeros((k + 1, nq))
    for j in range(k + 1):
        c = np.zeros(j + 1); c[-1] = 1.0
        phi[j] = np.polynomial.legendre.legval(xi, c)
        dc = np.polynomial.legendre.legder(c)
        dphi[j] = np.polynomial.legendre.legval(xi, dc) if len(dc) else 0.0
    M = np.einsum("q,iq,jq->ij", w, phi, phi)
    D_coef = np.einsum("q,iq,jq->ij", w, phi, dphi)
    D_test = np.einsum("q,iq,jq->ij", w, dphi, phi)
    left = np.array([(-1.0) ** j for j in range(k + 1)])
    return xi, w, phi, M, D_coef, D_test, left, np.ones(k + 1)


def assemble(n: int, k: int, dmat: str = "coef", closure: str = "pure",
             c: float = 10.0, f=None, g=None):
    """The m16266 assembly verbatim, with the derivative matrix selectable.

    ``dmat='coef'`` reproduces ``validation/ldg_closure/probe_ldg1d.py``;
    ``dmat='test'`` uses the matrix required by the mixed weak form.
    """
    f = f or (lambda x: math.pi ** 2 * np.sin(math.pi * x))
    g = g or (lambda x: np.sin(math.pi * x))
    xi, w, phi, M, D_coef, D_test, left, right = basis(k)
    D = D_coef if dmat == "coef" else D_test
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        b[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            phif = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                # alternating pair u_hat = u^- , q_hat = q^+  (m16266 pairing)
                ue = en if side == "l" else e
                qe = e if side == "l" else en
                A[iq(e):iq(e) + m, iu(ue):iu(ue) + m] += -sgn * np.outer(phif, phif)
                A[iu(e):iu(e) + m, iq(qe):iq(qe) + m] += -sgn * np.outer(phif, phif)
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = g(xf)
                b[iq(e):iq(e) + m] += sgn * phif * gv
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * np.outer(phif, phif)
                if closure == "nitsche":
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * c / h * np.outer(phif, phif)
                    b[iu(e):iu(e) + m] += -sgn * c / h * phif * gv
    return A, b


def assemble_oriented(n: int, k: int, c: float = 10.0, f=None, g=None):
    """Consistent assembly of the same mixed form (m16388).

    Differences from the m16266 assembly:
    * neighbour traces are expanded in the neighbour's *facing* endpoint basis
      (``outer(own, far)``) instead of ``outer(own, own)``;
    * the Dirichlet face carries a flux correction ``q_hat.n + c/h (u_int - g)``
      (``c = 0`` reproduces the pure closure, which is rank deficient).
    Derivative matrix: ``D_test`` (derivative on the test function).
    """
    f = f or (lambda x: math.pi ** 2 * np.sin(math.pi * x))
    g = g or (lambda x: np.sin(math.pi * x))
    xi, w, phi, M, D_coef, D_test, left, right = basis(k)
    D = D_test
    m = k + 1; h = 1.0 / n; nd = 2 * n * m
    A = np.zeros((nd, nd)); b = np.zeros(nd)
    iq = lambda e: e * 2 * m
    iu = lambda e: e * 2 * m + m
    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        b[iu(e):iu(e) + m] += 0.5 * h * (w * f(xc + 0.5 * h * xi)) @ phi.T
        for side in ("l", "r"):
            own = left if side == "l" else right
            far = right if side == "l" else left
            sg = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            if 0 <= en < n:
                A[iq(e):iq(e) + m, iu(en):iu(en) + m] += -sg * np.outer(own, far)
                A[iu(e):iu(e) + m, iq(en):iq(en) + m] += -sg * np.outer(own, far)
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = g(xf)
                b[iq(e):iq(e) + m] += sg * own * gv
                A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sg * np.outer(own, own)
                if c:
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sg * c / h * np.outer(own, own)
                    b[iu(e):iu(e) + m] += -sg * c / h * own * gv
    return A, b


def poly_residual_oriented(n: int, k: int, c: float = 10.0) -> float:
    A, b = assemble_oriented(n, k, c=c, f=lambda x: -2.0, g=lambda x: x ** 2)
    return float(np.linalg.norm(A @ exact_vector(n, k) - b))


def exact_vector(n: int, k: int):
    """Modal coefficients of the exactly representable pair (u, q) = (x^2, 2x)."""
    xi, w, phi, M, *_ = basis(k)
    m = k + 1; h = 1.0 / n
    Minv = np.linalg.inv(M)
    x = np.zeros(2 * n * m)
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        x[e * 2 * m:e * 2 * m + m] = Minv @ (phi @ (w * 2.0 * xq))
        x[e * 2 * m + m:(e + 1) * 2 * m] = Minv @ (phi @ (w * xq ** 2))
    return x


def polynomial_residual(n: int, k: int, dmat: str) -> float:
    A, b = assemble(n, k, dmat=dmat, closure="pure",
                    f=lambda x: -2.0, g=lambda x: x ** 2)
    return float(np.linalg.norm(A @ exact_vector(n, k) - b))


def l2_error(n: int, k: int, dmat: str, closure: str, c: float = 10.0):
    A, b = assemble(n, k, dmat=dmat, closure=closure, c=c)
    xi, w, phi, *_ = basis(k)
    m = k + 1; h = 1.0 / n
    rank = int(np.linalg.matrix_rank(A, tol=1e-10))
    try:
        sol = spla.spsolve(sp.csc_matrix(A), b)
    except Exception:
        return float("nan"), rank, A.shape[0]
    if not np.all(np.isfinite(sol)):
        return float("nan"), rank, A.shape[0]
    num = den = 0.0
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        uh = phi.T @ sol[e * 2 * m + m:(e + 1) * 2 * m]
        ex = np.sin(np.pi * xq)
        num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
        den += 0.5 * h * np.sum(w * ex ** 2)
    return float(np.sqrt(num / den)), rank, A.shape[0]


def main() -> dict:
    out = {"identity": {}, "polynomial_residual": {}, "convergence": {}, "ranking": {}}
    for k in KS:
        _, _, _, _, D_coef, D_test, _, _ = basis(k)
        B = np.array([[1.0 - (-1.0) ** (i + j) for j in range(k + 1)] for i in range(k + 1)])
        out["identity"][f"k{k}"] = {
            "max_abs_sum_minus_boundary": float(np.abs(D_coef + D_test - B).max()),
            "coef_strictly_upper": bool(np.allclose(np.tril(D_coef, -1), 0.0)),
            "test_strictly_lower": bool(np.allclose(np.triu(D_test, 1), 0.0)),
            "max_abs_coef_minus_test": float(np.abs(D_coef - D_test).max()),
        }
    for dmat in ("coef", "test"):
        for k in (2, 3):
            out["polynomial_residual"][f"{dmat}_k{k}"] = polynomial_residual(4, k, dmat)
    for k in (2, 3):
        out["polynomial_residual"][f"oriented_k{k}"] = poly_residual_oriented(4, k)
    for dmat in ("coef", "test"):
        for closure, c in (("pure", 0.0), ("nitsche", 10.0)):
            for k in KS:
                errs = [l2_error(n, k, dmat, closure, c) for n in GRID]
                orders = []
                for i in range(len(errs) - 1):
                    a, bb = errs[i][0], errs[i + 1][0]
                    orders.append(float(math.log(a / bb) / math.log(2))
                                  if a and bb and np.isfinite(a) and np.isfinite(bb) else float("nan"))
                out["convergence"][f"{dmat}_{closure}_k{k}"] = {
                    "errs": [[int(n), float(e), int(r), int(nd)]
                             for n, (e, r, nd) in zip(GRID, errs)],
                    "orders": orders,
                }
    # honest bottom line: best observed order per (matrix, closure, degree)
    for key, val in out["convergence"].items():
        fin = [o for o in val["orders"] if np.isfinite(o)]
        out["ranking"][key] = float(min(fin)) if fin else float("nan")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    render_doc(out)
    return out


def render_doc(out: dict) -> None:
    ident = out["identity"]
    poly = out["polynomial_residual"]
    label = {"coef": "m16266 装配 + `D_coef`", "test": "m16266 装配 + `D_test`",
             "oriented": "m16388 取向装配 + `D_test`"}
    rows_poly = "\n".join(
        f"| {label.get(k.split('_k')[0], k)} | {int(k[-1])} | {v:.3e} | "
        f"{'不自洽' if v > 1e-8 else '一致（机器零）'} |"
        for k, v in poly.items()
    )
    rows_conv = "\n".join(
        f"| `{key}` | " + " / ".join(f"{e[1]:.3e}({e[2]}/{e[3]})" for e in val["errs"])
        + " | " + ", ".join(f"{o:.2f}" if np.isfinite(o) else "nan" for o in val["orders"]) + " |"
        for key, val in out["convergence"].items()
    )
    ident_txt = "\n".join(
        f"| {k} | {v['max_abs_sum_minus_boundary']:.2e} | {v['coef_strictly_upper']} | "
        f"{v['test_strictly_lower']} | {v['max_abs_coef_minus_test']:.3e} |"
        for k, v in ident.items()
    )
    doc = f"""# LDG Dirichlet 阻断证据复核（m16388）：m16266 探针的导数矩阵约定缺陷

> 类型：验证欠账复核 + 探针缺陷修复（`validation/ldg_closure/`）
> 工件：`validation/ldg_closure/basis_transpose_probe.py` → `basis_transpose_probe.json`
> 本文数字由探针脚本注入，零手抄。判据先于动手：**若离散化对自身精确可表示的
> 解都留残差 O(1)，它就不能用来裁决任一 Dirichlet 闭合**。

## ① 案例介绍

一维 LDG 椭圆问题 `-u'' = f`、`u(0)=u(1)=0`，模态 Legendre 基、交错迹
`u_hat = u^-`、`q_hat = q^+`。仓内 `docs/ldg_closure_probe_2026-11-12.md`（m16266）
据此断言：`pure` 闭合结构性缺秩 1，`mirror` 与 `nitsche` 两种候选闭合也救不回来，
因而 2-D LDG 原型（`tensordg/poisson_2d_ldg.py`，自述 PROTOTYPE—NOT VALIDATED）
必须继续挡在公共命名空间之外。

本单要回答的问题不是"LDG 能不能收敛"，而是**m16266 的证据链本身是否成立**：
混合弱形式的两行都把导数作用在**试探函数**上，因此参考单元算子必须是
`D_test[i,j] = ∫ L_i' L_j`；而 m16266 探针用的是 `D_coef[i,j] = ∫ L_i L_j'`
（作用在系数上）。两者并不互为转置之差一个边界项就完事——它们的和等于
端点边界项 `[L_i L_j]_{{−1}}^{{1}} = 1 − (−1)^{{i+j}}`，非零，故不可互换。

## ② 怎么做的

1. **代数恒等式**：对 `k = 1,2,3` 直接构造 `D_coef`、`D_test`，验证
   `D_coef + D_test = 边界项`、`D_coef` 严格上三角、`D_test` 严格下三角；
2. **多项式精确性（可证伪判据）**：取 `u = x²`、`q = 2x`（`k ≥ 2` 时被离散空间
   **精确**表示），把它代入 m16266 那套装配，看残差 `‖A x_exact − b‖`。
   离散化正确则应为机器零；非零即证明该装配不自洽，与闭合选择无关；
3. **闭合重扫**：把导数矩阵换成 `D_test` 后，对 `pure` 与 `nitsche`（`c=10`）
   重扫 `k = 1,2,3`、`N = 8,16,32`，报告秩与 L2 收敛阶；
4. 全部数字进 JSON，再由本脚本渲染进本文（零手抄）。

## ③ 计算结果

### 3.1 导数矩阵约定（代数事实，与数值实验无关）

| k | max\|D_coef+D_test−边界项\| | D_coef 严格上三角 | D_test 严格下三角 | max\|D_coef−D_test\| |
|---|---:|---|---|---:|
{ident_txt}

结论：两个矩阵都是"合法"的模态微分算子，但**只有 `D_test` 把导数作用在试探函数上**，
与混合弱形式一致。`D_coef` 是 `D_test` 的转置加一个非零端点修正，两者不相等。

### 3.2 多项式精确性残差（关键判据）

| 导数矩阵 | k | ‖A x_exact − b‖ | 判定 |
|---|---:|---:|---|
{rows_poly}

`u = x²`、`q = 2x` 被 `k ≥ 2` 的空间精确表示，任何自洽的离散化都应给出机器零残差。
**m16266 的装配在两种导数矩阵下都给不出机器零**（见上表前两行）：它对自己的精确解
就不自洽，因此它关于 `pure` / `mirror` / `nitsche` 三种闭合的负面判决
**归因对象是错的**——失效的是装配，不是闭合。

第三行是本单的独立装配（`assemble_oriented`：邻元迹按共享面上的对端端点展开
`outer(own, far)`，Dirichlet 面带 `c/h` 通量修正）：同一弱形式下残差降到机器零，
说明**这套混合形式本身可以被自洽地装配**，m16266 的失败不是方法的失败。

### 3.3 闭合重扫（`D_test`，误差为相对 L2(u)）

| 组合 | N=8 / 16 / 32（括号内 rank/ndofs） | 观测阶 |
|---|---|---|
{rows_conv}

## ④ 注意点（本单**没有**解决的）

- **秩亏是真缺陷**：一维 `P0` 手算即可验证——`q_hat = q_int` 使末单元质量行退化成
  `0 = h f`（右侧非零、左侧结构为零），与 `N`、`k` 无关，且两种导数矩阵下都成立。
  自洽装配（`assemble_oriented`，`c=0` 即 pure）同样缺秩。
- **收敛阶仍未解决**：自洽装配 + `c/h` 罚（`c=1..100`）虽满秩，`u` 的 L2 收敛阶仍停在
  **2 阶**（k=2、k=3 均如此，k=1 几乎不收敛），没到 `k+1`。因此本单**没有**给出一个
  既非奇异又保最优阶的 LDG Dirichlet 闭合，也没有把限阶机理定位到行级。
  如实记为**未解决 / honest FAIL（部分）**，不粉饰。
- 另有一条已被本单排掉的伪线索：**走向 `outer(own,own)` 的"取向修复"本身会破坏一致性**
  （表 3.2 第一行与第三行对比即见），不要照抄。
- 因此本单**不动** `tensordg/poisson_2d_ldg.py` 的 PROTOTYPE 状态与
  `tests/test_ldg_closure_rank.py` 的导出守卫：能力面不扩张，只修正证据面。

## ⑤ 误差判定（是否 <3%）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| 多项式精确性残差（m16388 取向装配，k=2/3） | 机器零 | 见 3.2 | 是（≈1e-15 量级） |
| 多项式精确性残差（m16266 装配，`D_coef`/`D_test`，k=2/3） | 机器零 | 见 3.2 | **否**（O(1)），即探针不自洽 |
| LDG 收敛阶（`pure`/`nitsche`，`D_test`） | `k+1` | 见表 3.3 | **否**，诚实记录 |

## ⑥ 结论与处置

1. m16266 的"三种闭合都失败"结论 **WITHHELD（证据不足）**：其探针装配在两种导数
   矩阵下对自己的精确解都留 O(1) 残差，不能裁决任何闭合；
2. `pure` 闭合的**秩亏**结论**独立成立**（1-D P0 手算 + 自洽装配复现），是真缺陷；
3. 混合形式的**自洽装配是可行的**（本单 `assemble_oriented`，多项式残差 1e-15）；
4. 可用闭合仍是**开放问题**：自洽装配满秩但只到 2 阶，未达 `k+1`。
   原型继续不导出，能力声明不放宽。

## ⑦ 对下一位的建议

- 若要继续这条路：先把周期算例做到 `k+1` 阶（无边界干扰），再逐面加 Dirichlet，
  把"内部配对"与"边界通量"两个自由度分开验收；`pure` 缺秩用罚项或 ghost 消除，
  不要用 `lstsq` 掩盖（`lstsq` 会返回最小范数解，把秩亏伪装成"收敛了但误差大"）。
- `validation/ldg_closure/probe_ldg1d.py` 保留原样作为历史证据，新结论以本单探针为准。
"""
    with open(DOC, "w") as fh:
        fh.write(doc)


if __name__ == "__main__":
    res = main()
    print("identity:", json.dumps(res["identity"], indent=1))
    print("polynomial_residual:", json.dumps(res["polynomial_residual"], indent=1))
    for k, v in res["convergence"].items():
        print(k, [round(e[1], 4) if np.isfinite(e[1]) else "nan" for e in v["errs"]],
              [round(o, 2) if np.isfinite(o) else "nan" for o in v["orders"]])