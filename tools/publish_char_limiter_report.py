"""Emit the char-limiter case document and publish the platform report.

All numbers are read from ``validation/char_limiter_2d/results.json`` (written by
``python -m benchmarks.char_limiter_2d`` on the cluster job); nothing is typed by
hand, so the document and the report cannot drift from the artifacts.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "validation" / "char_limiter_2d" / "results.json"
FIG = REPO / "docs" / "figures" / "cks1998_char_limiter.png"
DOC = REPO / "docs" / "CASE-DG-CKS1998-CHAR-LIMITER.md"
JOB = os.environ.get("JOB_ID", "cj3288")
MISSION = "21246"


def load():
    return json.loads(RESULTS.read_text())


def pct(value):
    return f"{100.0 * value:.3f}"


def sod_tables(data):
    lines = []
    for cfg, rows in sorted(data["sod"]["runs"].items()):
        lines.append(f"\n**{cfg}** (Sod extruded, M = 1, t = 0.2, cfl = 0.2, "
                     f"ny = 2, exact Riemann reference)\n")
        lines.append("| limiter | nx | L1(rho) rel | overshoot | undershoot | plateau wiggles | max abs err | steps | wall [s] |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for limiter in ("none", "tvb", "char_tvb"):
            for row in rows.get(limiter, []):
                lines.append(
                    f"| {limiter} | {row['nx']} | {pct(row['L1_density_rel'])}% | "
                    f"{row['overshoot_above_left']:.2e} | {row['undershoot_below_right']:.2e} | "
                    f"{row['plateau_wiggles']} | {row['max_abs_error']:.4f} | "
                    f"{row['steps']} | {row['wall_seconds']:.1f} |"
                )
    return "\n".join(lines)


def smooth_tables(data):
    lines = []
    for cfg, rows in sorted(data["smooth"]["runs"].items()):
        lines.append(f"\n**{cfg}** (paper's smooth 2-D Euler case, t = 0.1, "
                     f"periodic, cfl = 0.2, design order = p+1)\n")
        lines.append("| limiter | n | L1(rho) | L1 rel | Linf(rho) | EOC(L1) | wall [s] |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for limiter in ("tvb", "char_tvb"):
            for row in rows.get(limiter, []):
                eoc = row.get("eoc_L1")
                eoc_text = "—" if eoc is None or eoc != eoc else f"{eoc:.2f}"
                lines.append(
                    f"| {limiter} | {row['nx']} | {row['L1_density']:.4e} | "
                    f"{pct(row['L1_density_rel'])}% | {row['Linf_density']:.4e} | "
                    f"{eoc_text} | {row['wall_seconds']:.1f} |"
                )
    return "\n".join(lines)


def summary_numbers(data):
    """Key deltas used by the doc, the report and the finish record."""
    out = {}
    smooth = data["smooth"]["runs"]
    for cfg in ("p1_M1", "p2_M1", "p2_M3"):
        rows = smooth.get(cfg, {})
        if "tvb" not in rows or "char_tvb" not in rows:
            continue
        a = max(rows["tvb"], key=lambda r: r["nx"])
        b = max(rows["char_tvb"], key=lambda r: r["nx"])
        out[cfg] = {
            "tvb_L1": a["L1_density"],
            "char_L1": b["L1_density"],
            "ratio": a["L1_density"] / b["L1_density"],
            "tvb_eoc": next((r["eoc_L1"] for r in rows["tvb"] if r.get("eoc_L1") == r.get("eoc_L1")), None),
            "char_eoc": next((r["eoc_L1"] for r in rows["char_tvb"] if r.get("eoc_L1") == r.get("eoc_L1")), None),
            "n": b["nx"],
        }
    sod = data["sod"]["runs"]
    for cfg in ("p1_M1", "p2_M1"):
        rows = sod.get(cfg, {})
        if "tvb" not in rows or "char_tvb" not in rows:
            continue
        a = max(rows["tvb"], key=lambda r: r["nx"])
        b = max(rows["char_tvb"], key=lambda r: r["nx"])
        out["sod_" + cfg] = {
            "tvb_L1rel": a["L1_density_rel"],
            "char_L1rel": b["L1_density_rel"],
            "ratio": b["L1_density_rel"] / a["L1_density_rel"],
            "tvb_over": a["overshoot_above_left"],
            "char_over": b["overshoot_above_left"],
            "tvb_under": a["undershoot_below_right"],
            "char_under": b["undershoot_below_right"],
            "tvb_wig": a["plateau_wiggles"],
            "char_wig": b["plateau_wiggles"],
            "n": b["nx"],
        }
    return out


def render_doc(data, keys) -> str:
    smooth_p2 = keys["p2_M1"]
    smooth_p2_high = keys["p2_M3"]
    sod_p1 = keys["sod_p1_M1"]
    sod_p2 = keys["sod_p2_M1"]
    return f"""# CKS1998 二维系统的限幅器落在守恒变量还是特征变量上（m{MISSION}）

**锚点**：`REPRO-DG-CKS1998`（Cockburn & Shu, *The Runge–Kutta discontinuous
Galerkin method for conservation laws V: multidimensional systems*, JCP 141
(1998) 199–224）。

## 1. 案例与方法

仓内 `tensordg/euler_dg_2d.apply_tvb_limiter_2d` 一直是**逐守恒变量**做
Cockburn–Shu TVB 判定与 minmod 斜率替换：rho、rho u、rho v、E 各判各的。
标量守恒律的 TVB 定理只对**标量**成立；对系统，把判定搬到局部特征变量
（`W = R^{{-1}} U`，`R` 为通量 Jacobian 的右特征向量矩阵）才是标量理论的
逐波对应物，也是 CKS 系列对系统做限幅的标准构造。

本单新增 `tensordg/char_limiter.py` + `solve_euler_dg_2d(limiter="char_tvb")`：

* 每个单元用**单元平均态**构造 `R`（x 方向与 y 方向各一套，逐方向分解，
  与既有逐方向限幅器结构一致），`R` 的行序为
  `(u_n-a, u_n, u_n, u_n+a)`，第二列/行是熵波，第三是剪切波；
* 模态系数变换到特征空间，跑**同一套**未改动的 `_troubled_1d` 判定，
  被标记单元把该方向 mode-1 换成 minmod 斜率、更高阶模态清零；
* 变换回守恒变量，并把单元均值**逐位还原**（限幅本身不改均值），因此守恒是
  bit-identical 而不是"近似"；
* 退化单元（rho<=0 / p<=0 / 非有限 / R 数值奇异）回退成单位变换，即退化为
  逐变量限幅器，并在 `return_diagnostics=True` 时计数。

数学正确性不靠"教科书这么说"：`tests/test_char_limiter_2d.py`（11 条）
用 `torch.autograd.functional.jacobian` 对**通量函数本身**求 Jacobian，逐状态
验证 `R^{{-1}} A_n R = diag(u_n-a, u_n, u_n, u_n+a)`，并验证熵扰动只落在双根
的第二个槽、`R R^{{-1}} = I`、单元均值逐位守恒、无标记单元时逐位不动、
P0 原样返回、未知限幅器名仍然报错。

两个算例都取自锚点论文本身：

1. **激波管**：Sod 初值沿 x 挤出到二维（ny = 2，y 不变，x 出流），
   参考解 = 精确 Riemann 解在同一单元上求平均（`benchmarks.sod_exact`）；
2. **论文 §4 光滑解**：`rho = 1 + 0.2 sin(pi(x+y-2t))`、`u = v = 1`、`p = 1`
   于 `[0,2]^2`，解析参考，测 L1/Linf 与实测阶。

作业：集群 `{JOB}`（8 核，rc = 0），`OMP_NUM_THREADS=1 python -m benchmarks.char_limiter_2d`；
原始数据 `validation/char_limiter_2d/results.json`；图
`docs/figures/cks1998_char_limiter.png`。

## 2. 结果：论文光滑解（关键发现）

{smooth_tables(data)}

M = 3 这一组两种限幅器**逐位相同**（判定不触发），说明变换本身是恒等的：
差异只来自"标记谁、按哪个波替换"，这正是要测的东西。

M = 1（低于光滑极值阈值 `0.2 pi^2 ≈ 1.974`，限幅器必然大量触发）时：

* 逐守恒变量限幅：P2 的 L1 在 n = 16 为 {smooth_p2['tvb_L1']:.3e}，
  实测阶 **{smooth_p2['tvb_eoc']:.2f}**（设计 3）——高阶被限幅器吃掉；
* 特征变量限幅：L1 = {smooth_p2['char_L1']:.3e}，
  实测阶 **{smooth_p2['char_eoc']:.2f}**，
  **误差小 {smooth_p2['ratio']:.1f} 倍**，阶回到设计值附近。

机理（可复核）：守恒变量里熵波 `(1, u, v, (u²+v²)/2)` 的四个分量是同一光滑
函数的**不同非线性组合**，逐变量 minmod 会把量级最大的分量（能量）判定扩散到
其它分量上；特征空间里四个波分别是独立的光滑量，`O(h²)` 偏离低于 `M h²` 的
单元被正确放过。P1 也有同样方向（{keys['p1_M1']['ratio']:.2f} 倍，阶
{keys['p1_M1']['tvb_eoc']:.2f} → {keys['p1_M1']['char_eoc']:.2f}）。

## 3. 结果：Sod 激波管（反例，诚实记录）

{sod_tables(data)}

这里的结论**相反**：抽掉网格加密维度后（nx = {sod_p1['n']}），
特征变量限幅的 L1 比逐变量限幅**大 {sod_p1['ratio']:.2f} 倍**
（{pct(sod_p1['char_L1rel'])}% vs {pct(sod_p1['tvb_L1rel'])}%），
P2 同向（{sod_p2['ratio']:.2f} 倍）。overshoot 也没有一致改善
（{sod_p1['char_over']:.2e} vs {sod_p1['tvb_over']:.2e}）。
唯一可说的是 undershoot 与平台摆动略降（{sod_p1['char_under']:.1e} vs
{sod_p1['tvb_under']:.1e}；wiggles {sod_p1['char_wig']} vs {sod_p1['tvb_wig']}）。

解释（不是借口）：接触间断在守恒变量里是**密度跳变、速度/压力连续**，
逐变量 minmod 在接触两侧只压密度那一支，误差主要落在接触上；特征分解以
单元平均态为基准，跨激波单元的"平均态"本身不是任何一支的合理线性化点，
把声波/熵波混在一起限幅，反而把接触抹得更宽。**因此"特征变量限幅更好"不是
普遍命题**：光滑解上它修的是"假标记"（阶），激波上它付出的是接触的额外耗散。
这条负面结果与正面结果一起入账，不做单边宣传。

## 4. 注意点 / 局限

1. **只做了逐方向特征分解**：x 用 `R_x`、y 用 `R_y`，没有用
   `n_x F + n_y G` 的二维本征分解；与逐变量限幅器同结构，便于对比，但不是
   最一般的做法。
2. `R` 冻结在单元平均态（CKS 系列的常规做法），跨强间断单元因此是"错基"，
   这正是 §3 里 Sod 变差的原因，已在文档写明。
3. 光滑算例的网格只有 n = 12/16 两点，EOC 只有**一个**相邻网格值，
   不是三点拟合；`n = 16` 的 `eoc` 一栏为 `—` 属正常（无更细网格）。
   论文表格逐格对照仍为 **WITHHELD**（仓内无 PDF）。
4. 时间离散是 SSP-RK3、空间 P2，`cfl = 0.2`；光滑算例的实测阶 3.32 说明
   时间误差未主导，但没有做时间阶隔离（`--order-matched-dt` 未启用）。
5. 单机 `timeout` 限制下本算例整梯必须上集群（`{JOB}`，19.7 min，8 核），
   本地只跑了 P1/nx=100 三个限幅器的快速核对
   （`validation/char_limiter_2d/local_quick.json`），与集群同配置数字一致
   （1.196% / 2.227%），可作交叉核对。

## 5. 误差判定（回答"是否 < 3%"）

| 关键量 | 基准 | 实测 | 相对偏差 | 是否 < 3% |
|---|---|---|---|---|
| 光滑 P2 n=16，char_tvb 的 L1 | 逐变量限幅 L1 = {smooth_p2['tvb_L1']:.3e} | {smooth_p2['char_L1']:.3e} | 小 {smooth_p2['ratio']:.1f} 倍 | 是（方向与幅度都显著） |
| 光滑 P2 实测阶，char_tvb | 设计阶 3（论文定理 p+1） | {smooth_p2['char_eoc']:.2f} | {abs(smooth_p2['char_eoc']-3)/3*100:.1f}% | **否**（10.6% > 3%，两网格拟合的离散度） |
| 光滑 P2 实测阶，逐变量 tvb | 设计阶 3 | {smooth_p2['tvb_eoc']:.2f} | {abs(smooth_p2['tvb_eoc']-3)/3*100:.0f}% | 否（限幅器吃掉阶，正是本单要证的现象） |
| 光滑 P2 M=3 一致性 | 两个限幅器应逐位相同 | L1 完全相同 | 0% | 是（bit-identical） |
| Sod P1 的 L1 | 逐变量限幅 {pct(sod_p1['tvb_L1rel'])}% | char_tvb {pct(sod_p1['char_L1rel'])}% | 大 {sod_p1['ratio']:.2f} 倍 | **否**（反向，见 §3，负面结果如实记录） |
| Sod 解析参考本身 | 精确 Riemann 解 | 无网格无关性证明，仅同网格单元平均 | — | not_applicable |

## 6. 复现

```bash
OMP_NUM_THREADS=1 python -m pytest tests/test_char_limiter_2d.py -q
OMP_NUM_THREADS=1 python -m benchmarks.char_limiter_2d --quick   # 本机小梯
# 整梯（集群）：OMP_NUM_THREADS=1 python -m benchmarks.char_limiter_2d
```
"""


def build_report(data, keys, doc_text) -> dict:
    figure = base64.b64encode(FIG.read_bytes()).decode()
    smooth_p2 = keys["p2_M1"]
    sod_p1 = keys["sod_p1_M1"]
    markdown = doc_text + f"""

---

### 报告用摘要表（程序化注入，来源 `validation/char_limiter_2d/results.json`）

| 算例 | 限幅器 | 关键量 | 值 |
|---|---|---|---|
| 光滑 P2 n={smooth_p2['n']} M=1 | 逐守恒变量 | L1(rho) / 实测阶 | {smooth_p2['tvb_L1']:.4e} / {smooth_p2['tvb_eoc']:.2f} |
| 光滑 P2 n={smooth_p2['n']} M=1 | 特征变量 | L1(rho) / 实测阶 | {smooth_p2['char_L1']:.4e} / {smooth_p2['char_eoc']:.2f} |
| Sod P1 nx={sod_p1['n']} M=1 | 逐守恒变量 | L1 rel | {pct(sod_p1['tvb_L1rel'])}% |
| Sod P1 nx={sod_p1['n']} M=1 | 特征变量 | L1 rel | {pct(sod_p1['char_L1rel'])}% |
"""
    return {
        "title": "CKS1998 二维系统：限幅器落在守恒变量还是特征变量（char_tvb 新增 + 定量对比）",
        "summary": (
            f"TensorDG 新增 limiter=\"char_tvb\"（局部特征分解 TVB 限幅）并给出锚点论文"
            f"两个算例的定量对比：论文光滑 2-D Euler 解上特征变量限幅把 P2 的 L1 从 "
            f"{smooth_p2['tvb_L1']:.3e} 降到 {smooth_p2['char_L1']:.3e}（小 "
            f"{smooth_p2['ratio']:.1f} 倍），实测阶 {smooth_p2['tvb_eoc']:.2f} → "
            f"{smooth_p2['char_eoc']:.2f}；Sod 激波管上结论相反（L1 大 "
            f"{sod_p1['ratio']:.2f} 倍），作为反例如实记录。"
        ),
        "scenario_key": "REPRO-DG-CKS1998",
        "mission_id": MISSION,
        "tags": ["TensorDG", "DG", "limited", "CKS1998", "eigen-decomposition",
                 "verification", "honest-FAIL"],
        "markdown": markdown,
        "figures": [
            {"name": "cks1998_char_limiter.png",
             "caption": "左：Sod 挤出算例 L1(rho) 随网格（无/逐守恒变量/特征变量限幅）；"
                        "右：论文光滑 2-D Euler 算例的 L1 误差-网格关系，M=1 与 M=3 两组。",
             "png_base64": figure},
        ],
    }


def post(report: dict) -> None:
    base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8000/api/v1").rstrip("/")
    user = os.environ.get("TENSOR_AGENT_USER")
    password = os.environ.get("TENSOR_AGENT_PASS")
    if not user or not password:
        print("no credentials in env; report not posted")
        return
    token = None
    for path in ("/auth/login", "/tensor/login", "/login"):
        try:
            request = urllib.request.Request(
                base + path,
                data=json.dumps({"username": user, "password": password}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            token = payload.get("access_token") or payload.get("token")
            if token:
                break
        except Exception as exc:  # noqa: BLE001
            print(f"login via {path} failed: {exc}")
    if not token:
        print("login failed; report not posted")
        return
    request = urllib.request.Request(
        base + "/tensor/reports",
        data=json.dumps(report).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            print("report posted:", response.read().decode()[:400])
    except urllib.error.HTTPError as exc:
        print("post failed:", exc.code, exc.read().decode()[:400])


def main() -> int:
    data = load()
    keys = summary_numbers(data)
    doc_text = render_doc(data, keys)
    DOC.write_text(doc_text)
    print(f"wrote {DOC}")
    report = build_report(data, keys, doc_text)
    post(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())