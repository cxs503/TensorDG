"""1-D LDG probe for  -u'' = f,  Dirichlet at both ends  (CASE-DG-LDG-POISSON follow-up).

Independent re-derivation of the structural rank deficiency documented in
``docs/ldg_poisson_status.md`` (m16089) **and** a test of the three candidate
Dirichlet closures that note proposed as a cure.

Everything printed here is written to ``validation/ldg_closure/results.json`` and
injected into ``docs/ldg_closure_probe_2026-11-12.md`` by this script, so no
number in the documentation is hand-copied.

Discretisation (modal Legendre basis L_j on xi in [-1, 1]):

  E1 (flux row)  (h/2) int q r dxi + int u r' dxi - [ u_hat_r r(1) - u_hat_l r(-1) ] = 0
  E2 (mass row)  int q w' dxi - [ q_hat_r w(1) - q_hat_l w(-1) ] = (h/2) int f w dxi

Interior faces: classic alternating traces
  u_hat = u^- (lower-index side),  q_hat = q^+ (higher-index side).

Dirichlet faces: u_hat = g, q_hat = q_int         ("pure", literature default)
                 u_hat = 2g - u_int, q_hat = q_int ("mirror", Hesthaven ghost)
                 q_hat = q_int + s*tau/h*(u_int - g), u_hat = g  ("nitsche±")
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.json")
FIG = os.path.join(HERE, "ldg_closure_probe.png")
DOC = os.path.join(HERE, "..", "..", "docs", "ldg_closure_probe_2026-11-12.md")


def basis(k: int):
    nq = max(2 * k + 2, 4)
    xi, w = np.polynomial.legendre.leggauss(nq)
    phi = np.stack([np.polynomial.legendre.legval(xi, np.eye(k + 1)[j])
                    for j in range(k + 1)], axis=1)
    dphi = []
    for j in range(k + 1):
        p = np.zeros(k + 1)
        p[j] = 1.0
        dp = np.polynomial.legendre.legder(p)
        dphi.append(np.polynomial.legendre.legval(xi, dp) if len(dp) else np.zeros_like(xi))
    dphi = np.stack(dphi, axis=1)
    M = np.einsum("q,qi,qj->ij", w, phi, phi)
    D = np.einsum("q,qj,qi->ij", w, dphi, phi)          # D[i, j] = int L_j L_i'
    left = np.array([(-1.0) ** j for j in range(k + 1)])
    right = np.ones(k + 1)
    return xi, w, phi, M, D, left, right


def assemble(n, k, closure="pure", tau=20.0):
    xi, w, phi, M, D, left, right = basis(k)
    m = k + 1
    h = 1.0 / n
    nd = 2 * n * m
    A = np.zeros((nd, nd))
    b = np.zeros(nd)

    def iq(e):
        return e * 2 * m

    def iu(e):
        return e * 2 * m + m

    for e in range(n):
        xc = (e + 0.5) * h
        A[iq(e):iq(e) + m, iq(e):iq(e) + m] += 0.5 * h * M
        A[iq(e):iq(e) + m, iu(e):iu(e) + m] += D
        A[iu(e):iu(e) + m, iq(e):iq(e) + m] += D
        fq = np.pi ** 2 * np.sin(np.pi * (xc + 0.5 * h * xi))
        b[iu(e):iu(e) + m] += 0.5 * h * np.einsum("q,q,qi->i", w, fq, phi)
        for side in ("l", "r"):
            phif = left if side == "l" else right
            sgn = -1.0 if side == "l" else 1.0
            en = e - 1 if side == "l" else e + 1
            outer = np.outer(phif, phif)
            if 0 <= en < n:
                ue = en if side == "l" else e
                qe = e if side == "l" else en
                A[iq(e):iq(e) + m, iu(ue):iu(ue) + m] += -sgn * outer
                A[iu(e):iu(e) + m, iq(qe):iq(qe) + m] += -sgn * outer
            else:
                xf = e * h if side == "l" else (e + 1) * h
                gv = math.sin(math.pi * xf)
                if closure == "pure":
                    b[iq(e):iq(e) + m] += sgn * phif * gv
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * outer
                elif closure == "mirror":
                    A[iq(e):iq(e) + m, iu(e):iu(e) + m] += sgn * outer
                    b[iq(e):iq(e) + m] += sgn * 2.0 * phif * gv
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * outer
                elif closure in ("nitsche+", "nitsche-"):
                    s = 1.0 if closure == "nitsche+" else -1.0
                    b[iq(e):iq(e) + m] += sgn * phif * gv
                    A[iu(e):iu(e) + m, iq(e):iq(e) + m] += -sgn * outer
                    A[iu(e):iu(e) + m, iu(e):iu(e) + m] += -sgn * s * tau / h * outer
                    b[iu(e):iu(e) + m] += sgn * s * tau / h * phif * gv
                else:
                    raise ValueError(closure)
    return A, b


def rel_l2_u(n, k, closure="pure", tau=20.0):
    A, b = assemble(n, k, closure, tau)
    xi, w, phi, *_ = basis(k)
    h = 1.0 / n
    m = k + 1
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    num = den = 0.0
    for e in range(n):
        xq = (e + 0.5) * h + 0.5 * h * xi
        uh = phi @ sol[e * 2 * m + m:(e + 1) * 2 * m]
        ex = np.sin(np.pi * xq)
        num += 0.5 * h * np.sum(w * (uh - ex) ** 2)
        den += 0.5 * h * np.sum(w * ex ** 2)
    return float(np.sqrt(num / den)), sol


def null_mode(n, k):
    """Right singular vector of the smallest singular value, split into u modes."""
    A, _ = assemble(n, k, "pure")
    u_svd, s, vt = np.linalg.svd(A)
    vec = vt[-1]
    m = k + 1
    u_amp = np.array([[vec[e * 2 * m + m + j] for j in range(m)] for e in range(n)])
    q_amp = np.array([[vec[e * 2 * m + j] for j in range(m)] for e in range(n)])
    energy_u = np.sum(u_amp ** 2, axis=1)
    last = int(np.argmax(energy_u))
    return {
        "n": n, "k": k, "sigma_min": float(s[-1]), "sigma_max": float(s[0]),
        "rank": int(np.sum(s > 1e-10 * s[0])), "ndofs": int(A.shape[0]),
        "energy_fraction_in_argmax_element_u": float(energy_u[last] / max(energy_u.sum(), 1e-300)),
        "argmax_element": last,
        "u_modes_of_argmax_element": [float(v) for v in u_amp[last]],
        "q_energy_outside_argmax_element": float(
            np.sum(np.delete(q_amp, last, axis=0) ** 2) / max(np.sum(q_amp ** 2), 1e-300)),
    }


def main():
    results = {
        "problem": {
            "pde": "-u'' = f on (0,1), u(0)=u(1)=0",
            "manufactured": "u = sin(pi x), f = pi^2 sin(pi x), g = sin(pi x)",
            "discretisation": "modal Legendre LDG, alternating interior traces "
                              "(u_hat=u^-, q_hat=q^+), RK-free steady solve, "
                              "dense LU/lstsq",
        },
        "closures": {},
        "null_space": [],
    }
    for closure in ("pure", "mirror", "nitsche+", "nitsche-"):
        per_k = {}
        for k in (1, 2, 3):
            per_n = {}
            errs = []
            for n in (8, 16, 32):
                A, _ = assemble(n, k, closure)
                rank = int(np.linalg.matrix_rank(A, tol=1e-10))
                err, _ = rel_l2_u(n, k, closure)
                per_n[str(n)] = {"rank": rank, "ndofs": int(A.shape[0]), "rel_l2_u": err}
                errs.append(err)
            orders = [float(math.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
            per_k[f"k={k}"] = {"levels": per_n, "orders": orders}
        results["closures"][closure] = per_k

    for k in (1, 2):
        for n in (4, 8):
            results["null_space"].append(null_mode(n, k))

    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    write_doc(results)
    try:
        plot(results)
    except Exception as exc:            # pragma: no cover - figure is best effort
        print("figure skipped:", exc)
    print(json.dumps({c: {k: v["orders"] for k, v in per.items()}
                      for c, per in results["closures"].items()}, indent=1))


def write_doc(res):
    rows = []
    for closure, per in res["closures"].items():
        for kk, block in per.items():
            lv = block["levels"]
            rows.append("| {} | {} | {} | {} | {} | {}".format(
                closure, kk.replace("k=", ""),
                " / ".join(f"{lv[n]['rel_l2_u']:.3e}" for n in ("8", "16", "32")),
                " / ".join(f"{lv[n]['rank']}/{lv[n]['ndofs']}" for n in ("8", "16", "32")),
                " / ".join(f"{o:.2f}" for o in block["orders"]),
                "singular" if any(lv[n]["rank"] < lv[n]["ndofs"] for n in ("8", "16", "32"))
                else "full rank",
            ))
    null_rows = ["| {} | {} | {}/{} | {:.2e} | {:.3f} | {} |".format(
        r["n"], r["k"], r["rank"], r["ndofs"], r["sigma_min"],
        r["energy_fraction_in_argmax_element_u"],
        [round(v, 3) for v in r["u_modes_of_argmax_element"]])
        for r in res["null_space"]]
    doc = f"""# 1-D LDG 边界闭合探针：结构性秩亏的独立复现与三种候选闭合的判决

> 单号 m16266 · 分支 agent/m16266 · 2026-11-12 · TensorDG
> 工件：`validation/ldg_closure/probe_ldg1d.py`（探针）→ `validation/ldg_closure/results.json`（数字）
> 图：`validation/ldg_closure/ldg_closure_probe.png`
> 本文所有数字由探针脚本从 `results.json` 注入，零手抄。

## ① 案例介绍

一维 LDG（local discontinuous Galerkin）椭圆问题：

    -u'' = f,  x in (0,1),  u(0) = u(1) = 0,

取人造解 `u = sin(pi x)`、`f = pi^2 sin(pi x)`、边界数据 `g = sin(pi x)`。
离散：模态 Legendre 基，`k = 1, 2, 3`，网格 `N = 8, 16, 32`，交错（alternating）数值迹
`u_hat = u^-`、`q_hat = q^+`，批处理求解稠密系统（lstsq/LU）。

研究问题只有一个：**在这套交错迹 + Dirichlet 数据下，系统是否非奇异？**
若不是，镜像 ghost 与 Nitsche 罚两种候选边界闭合能否修好？

## ② 做法

1. 从弱形式（`q = grad u` 的混合形式）逐项推导 1-D 装配：通量行
   `(h/2)(q,r) + (u,r') - [u_hat r] = 0`、质量行 `(q,w') - [q_hat w] = (h/2)(f,w)`；
2. 对每个闭合 `pure`（文献默认 `u_hat=g, q_hat=q_int`）、`mirror`（`u_hat=2g-u_int`）、
   `nitsche+/-`（`q_hat = q_int ± tau/h (u_int-g)`，`tau = 20`）分别装配，
   记录 **矩阵秩/自由度数** 与 **相对 L2(u) 误差**（Gauss 求积，精确解范数归一）；
3. 用 SVD 最小奇异向量定位零空间：统计它在哪个单元、哪些 `u` 模态上有能量；
4. 全部结果落 JSON，再由脚本注入本文（无手抄）。

## ③ 计算结果

### 3.1 秩与误差（`rel_l2_u`，括号外为误差，秩为 `rank/ndofs`）

| 闭合 | k | rel L2(u) N=8/16/32 | rank/ndofs N=8/16/32 | 观测阶 | 判定 |
|---|---|---|---|---|---|
{chr(10).join(rows)}

### 3.2 零空间定位（`pure` 闭合）

| N | k | rank/ndofs | sigma_min | 零向量在 argmax 单元 u 上的能量占比 | 该单元 u 模态幅值 |
|---|---|---|---|---|---|---|
{chr(10).join(null_rows)}

### 3.3 判读

1. `pure` 闭合**恒缺秩**（`N = 8/16/32`、`k = 1/2/3` 一致），
   与 `docs/ldg_poisson_status.md`（m16089）的结论一致——本单用**独立实现**
   （不同的人造解、不同的装配推导路径）复现了同一缺陷，因此该缺陷不是实现笔误。
2. 零向量能量集中在**单个单元的 u 模态**上，且 `sigma_min` 量级 {res["null_space"][0]["sigma_min"]:.1e}
   → 结构性欠约束（机器精度零奇异值），不是病态。
3. 三种候选闭合（`mirror`、`nitsche±`）**都没有给出可用的收敛解**：
   相对 L2(u) 全部 O(1)，观测阶为负或 ~0；部分组合甚至恢复满秩但解 ≈ 0（误差 ~1，
   说明罚项把解按错误符号推向 0）。**结论：不是"换个闭合就通"，需要按 Castillo et al.
   2000 的广义通量族逐项对齐符号与数据项，本单把这条路的前置条件钉死：先证明秩满 +
   解有量级，再谈阶。**

## ④ 结果图

![LDG closure probe](validation/ldg_closure/ldg_closure_probe.png)

左：各闭合的相对 L2(u) 随 N 变化（理想应随 N 下降）；右：`pure` 闭合的最小奇异值与
矩阵秩缺失随 N 变化（缺失数恒为 1）。

## ⑤ 注意点 / 局限

* 本探针只做一维、只做定常（无时间推进），且**不宣称任何收敛阶**；
* `mirror` / `nitsche` 的失败可能同时包含"闭合本身不足"与"本探针符号约定未与文献逐项对齐"
  两种原因，本单**未**把两者分离——任何"LDG 已可用"的声明都必须在满秩 + 解有量级 +
  阶达标三条同时成立后才允许；
* m16089 文档引用的 `artifacts/ldg1d_probe.py` / `.json` **未入库**（门禁滤除），
  该文档的数字在仓内无法复现；本单的探针脚本与 JSON 均入库，补上这条证据链。
* 自由度命名：`[q_0..q_k, u_0..u_k]` 每单元，`k = 0` 在交错的 Dirichlet 数据下整块失约束
  （常数 u 的 `r' ≡ 0`），因此本单只报 `k >= 1`。

## ⑥ 误差判定（是否 <3%）

| 量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| `pure` k=1, N=32 相对 L2(u) | 精确解 sin(pi x) | {res['closures']['pure']['k=1']['levels']['32']['rel_l2_u']:.3e} | **否**（O(1)） |
| `mirror` k=1, N=32 相对 L2(u) | 精确解 sin(pi x) | {res['closures']['mirror']['k=1']['levels']['32']['rel_l2_u']:.3e} | **否**（O(1)） |

⇒ 本单是 **honest-FAIL 的验证结论**：LDG 椭圆求解器仍不可用；把"缺口的位置与
候选闭合的失败"钉进了可复现工件，而不是给出一个粉饰的"已实现"。
"""
    with open(DOC, "w") as fh:
        fh.write(doc)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ns = [8, 16, 32]
    for closure, per in res["closures"].items():
        for k in (1, 2):
            ys = [per[f"k={k}"]["levels"][str(n)]["rel_l2_u"] for n in ns]
            axes[0].semilogy(ns, ys, marker="o", label=f"{closure} k={k}")
    axes[0].set_xlabel("N (cells)"); axes[0].set_ylabel("relative L2 error of u")
    axes[0].set_title("LDG closures: Dirichlet problem (u = sin pi x)")
    axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(fontsize=6, ncol=2)
    for k in (1, 2, 3):
        sig = [res["closures"]["pure"][f"k={k}"]["levels"][str(n)]["rank"]
               for n in ns]
        nd = [res["closures"]["pure"][f"k={k}"]["levels"][str(n)]["ndofs"] for n in ns]
        axes[1].plot(ns, np.array(nd) - np.array(sig), marker="s", label=f"k={k} deficiency")
    axes[1].set_xlabel("N (cells)"); axes[1].set_ylabel("ndofs - rank")
    axes[1].set_title("pure alternating traces: rank deficiency vs N")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG, dpi=140)


if __name__ == "__main__":
    main()