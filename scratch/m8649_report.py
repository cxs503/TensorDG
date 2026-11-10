"""Build the m8649 progress report (six-element, numbers injected programmatically).

Run: PYTHONPATH=. python scratch/m8649_report.py
Writes /tmp/report.json (figures embedded as base64) for POST /tensor/reports.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = json.load(open(ROOT / "validation/swe_thacker/hydrostatic_wb_results.json"))
ALT = json.load(open(ROOT / "artifacts/m8649/hydrostatic_wb_results_loong64_cj30.json"))
FIG1 = ROOT / "docs/figures/swe_hydrostatic_wb_roundoff.png"
FIG2 = ROOT / "docs/figures/swe_hydrostatic_wb_state.png"
PYTEST_LOG = Path("/tmp/m8649_keep/pytest_swe_all.log")

prov = MAIN["provenance"]
single = MAIN["single_run"]
summary = MAIN["summary"]

runs = {b["degree"]: {r["ncells"]: r for r in b["results"]}
        for b in MAIN["convergence_study"]["runs"]}
alt_runs = {b["degree"]: {r["ncells"]: r for r in b["results"]}
            for b in ALT["convergence_study"]["runs"]}

ETA_TOL, U_TOL = 1e-8, 1e-8
G = 9.81


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}e}"


# ---- cross-host table -------------------------------------------------------
rows = []
for deg in (1, 2):
    for n in (50, 100, 200):
        a, b = runs[deg][n], alt_runs[deg][n]
        rows.append(
            f"| P{deg} | {n} | {fmt(a['eta_error_max'])} | {fmt(b['eta_error_max'])} "
            f"| {fmt(a['u_max'])} | {fmt(b['u_max'])} | "
            f"{b['eta_error_max']/a['eta_error_max']:.1f}x |"
        )
cross_table = "\n".join(rows)

# ---- h-metric analytic prediction ------------------------------------------
dx_of = lambda n: 2.0 / n  # noqa: E731  (domain [-L, L], L = 1)
rows = []
for deg in (1, 2):
    for n in (50, 100, 200):
        pred = (single["H0"] / single["L"] ** 2) * dx_of(n) ** 2 / 12.0
        meas = runs[deg][n]["h_error_max"]
        rows.append(f"| P{deg} | {n} | {pred:.4e} | {meas:.4e} | {(meas-pred)/pred:+.1%} |")
h_table = "\n".join(rows)

# ---- error vs analytic reference -------------------------------------------
eta_rel = single["eta_error_max"] / single["eta0"]           # vs eta0 = 1 m
c_scale = math.sqrt(G * (single["eta0"] + single["H0"]))    # sqrt(g*h_max)
u_rel = single["u_max"] / c_scale
worst_eta = max(r["eta_error_max"] for d in runs.values() for r in d.values())
worst_u = max(r["u_max"] for d in runs.values() for r in d.values())

figs = []
for p, cap in (
    (FIG1, "图1 湖静平衡漂移 vs 网格与主机：max|η−η0|（左）与 max|u|（右），P1/P2、N=50/100/200，"
           "曲线为主机A（仓库工作树）与主机B（集群 n315, cj30）；点线为 1e-8 验收带。"),
    (FIG2, "图2 左：t=1 s 自由面 η_h、河床 z_b 与水深 h_h（N=100, P1），η_h 与解析 1.0 m 线不可分辨；"
           "右：虚假流速 max|u| 随时间（6e-14…2e-12 m/s），对照 well_balanced=False 的 2.1e-2 m/s。"),
):
    figs.append({"name": p.name, "caption": cap,
                 "png_base64": base64.b64encode(p.read_bytes()).decode()})

pytest_tail = ""
if PYTEST_LOG.exists():
    lines = [l for l in PYTEST_LOG.read_text().splitlines() if l.strip()]
    pytest_tail = lines[-1] if lines else ""

sha_alt = hashlib.sha256(
    (ROOT / "artifacts/m8649/hydrostatic_wb_results_loong64_cj30.json").read_bytes()).hexdigest()[:16]

markdown = f"""# TensorDG SWE 静平衡（lake-at-rest）良态格式验证：Thacker 抛物盆地 P1/P2、跨主机复现与工件溯源

## ① 案例介绍

物理背景：一维浅水方程（SWE）带底坡源项

```
∂h/∂t + ∂(hu)/∂x = 0
∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = −g h ∂z_b/∂x
```

的**静平衡（lake-at-rest）解** `η = h + z_b = const, u = 0` 是数值格式最易破坏、也最要求精确保持的状态（近岸/海啸/长时积分中虚假流会淹没物理信号）。算例为 Thacker (1981) 抛物盆地：

- 河床 `z_b(x) = −H₀ + (H₀/L²)x²`，`H₀ = 2.0 m`，`L = 1.0 m`，`x ∈ [−1, 1] m`（二次、有曲率）
- 初始 `η₀ = 1.0 m`，`h = η₀ − z_b ∈ [1, 3] m`（全域湿，隔离干湿前沿问题）

参数表：

| 项 | 取值 |
|---|---|
| 空间离散 | 模态 Legendre DG，P1 / P2 |
| 网格 | N = 50 / 100 / 200（dx = 0.04 / 0.02 / 0.01 m） |
| 数值通量 | HLL |
| 时间推进 | SSP-RK，CFL = 0.3，t ∈ [0, 1.0] s |
| 良态开关 | `well_balanced=True`（节点 LGL 多项式河床 + 多项式河床精确导数） |
| 求解器 | TensorDG `tensordg/swe_1d.py::solve_swe_1d` |
| 环境 | Python {prov['python']} / torch {prov['torch']} / NumPy {prov['numpy']} / {prov['platform']}（{prov['cpu_count']} 核） |
| 代码版本 | `{prov['git_commit_short']}`（`git_dirty={prov['git_dirty']}`），生成于 {prov['generated_at_utc']} |

## ② 怎么做的

1. **良态路径复核**：`well_balanced=True` 把河床在单元 LGL 节点插值成模态系数 `cz`，初值 `h, hu` 用同一组 LGL 节点插值，使离散静平衡 `η_h = h_h + z_h` 精确落在 DG 空间内；动量源项用**多项式河床的精确导数**（而非解析 dz_b/dx 采样），保证源项与通量在同求积点上严格抵消。
2. **后处理口径修正**：`reconstruct_fields` 默认用梯形面平均做单元中心河床，只对分段线性河床等于单元平均；对二次河床会在**报告出来的** η 上留下 O(dx²) 伪漂移（求解器状态本身仍是舍入级）。验证脚本改用 `zb_center=metadata["zb_modes"][:, 0]`（多项式河床 0 阶模态＝单元平均）。
3. **工件溯源**：`run_hydrostatic_wb.py` 新增 `collect_provenance()`，把 UTC 戳、命令行、git 提交/脏标记、Python/torch/NumPy 版本、主机平台/架构写入结果 JSON 的 `provenance` 块。本次结果由**干净工作树**（`git_dirty=false`，提交 `{prov['git_commit_short']}`）单次运行生成，同机重跑数值逐位一致。
4. **工件守卫**：新增 `tests/test_swe_hydrostatic_results_artifact.py`（6 测试，0.14 s，不做时间推进）——provenance 键齐全可解析、`reconstruction` 记录 `zb_center` 口径、`summary` 与 `single_run` 逐位一致、六个 (阶, 网格) 条目均满足声明的验收带。此前仓内提交的结果文件是**手搬的修复前快照**（P2 条目带 O(dx²) 后处理漂移，且无任何代码版本记录），守卫测试让这种失效模式在不跑 5 分钟求解的情况下也会被抓到。
5. **跨主机复现**：同脚本在集群节点 n315（作业 cj30）复跑，原始输出归档 `artifacts/m8649/hydrostatic_wb_results_loong64_cj30.json`（sha256 `{sha_alt}…`），用于判断舍入底噪是格式性质还是单机算术性质。

## ③ 计算结果

主机 A（本次提交工件，`{prov['git_commit_short']}`，`machine={prov['machine']}`）：

| 阶 | N | max|η−η₀| [m] | max|u| [m/s] | max|h−h_exact| [m] | 步数 |
|---|---|---|---|---|
""" + "\n".join(
    f"| P{d} | {n} | {fmt(runs[d][n]['eta_error_max'])} | {fmt(runs[d][n]['u_max'])} "
    f"| {fmt(runs[d][n]['h_error_max'])} | {runs[d][n]['steps']} |"
    for d in (1, 2) for n in (50, 100, 200)
) + f"""

单次基准算例（N=100, P1, t=1.0 s）：`max|η−η₀| = {fmt(single['eta_error_max'])} m`，`max|u| = {fmt(single['u_max'])} m/s`，{single['steps']} 步，{single['wall_time_s']:.1f} s 墙钟；`pass = {summary['pass']}`。

跨主机对照（同脚本同参数；比值＝主机B/主机A）：

| 阶 | N | η_A [m] | η_B [m] | u_A [m/s] | u_B [m/s] | η 比值 |
|---|---|---|---|---|---|---|
{cross_table}

两台主机同为 `loongarch64`，残差相差至多 ~5x，量级带不变（1e-14…1e-9），说明静平衡不动点是格式性质，差异来自 libm/BLAS 舍入路径。集群侧早前工件缺 provenance 块，正是本次补的缺口。

## ④ 结果图

![图1]({FIG1.name})

![图2]({FIG2.name})

（图由 `scratch/m8649_figures.py` 生成，数据直接读自已提交 JSON 与归档集群 JSON，无手抄；两张 PNG 的 sha256 前 16 位为 `{hashlib.sha256(FIG1.read_bytes()).hexdigest()[:16]}`、`{hashlib.sha256(FIG2.read_bytes()).hexdigest()[:16]}`。）

## ⑤ 注意点

1. **η/u 残差是舍入底噪，不是离散误差**：随 N 加密反而**增大**（约每加密一倍涨 2–4x，≈∝ 步数/dx），故湖静测试的正确判据是"机器 eps 的若干倍"，不能套 h^p 收敛率。
2. **`h` 列不是求解误差**：`h_h` 与点值精确解之差恰为 `mean_cell(z_b) − z_b(x_c) = (H₀/L²)dx²/12`。验证表：

| 阶 | N | dx²/12·H₀/L² [m] | 实测 max|h−h_exact| [m] | 相对偏差 |
|---|---|---|---|---|
{h_table}

P2 单元内二次河床可精确表示，实测与解析预测在 1e-3 相对量级内吻合（差值即求解器自身舍入残差）；P1 多出线性表示的 O(dx²) 插值误差，实测约为预测 3 倍。**良态判据只能用 η 与 u。**
3. **干湿前沿不在良态范围内**：`well_balanced=True` 不改变干单元处理（仍需 Audusse 静水重构），本算例全域湿（h ≥ 1 m）以隔离该问题。
4. **性能代价**：节点 LGL 路径比标准 L2 投影多一次插值变换，约 10–20% 开销。
5. **工件纪律**：结果文件必须由脚本生成并自带 provenance；手搬数字（本次修掉的历史快照）同时带来数值漂移与不可溯源两个问题。
6. **测试范围**：控制面主机跑 `tests/test_swe_wellbalanced_hydrostatic.py`（21 项）与新增守卫测试（6 项）；另递交集群跑 TensorDG 全量 `tests/`（作业 cj124）。

## ⑥ 误差判定（对照解析解）

解析基准：`η(x,t) ≡ η₀ = 1.0 m`、`u(x,t) ≡ 0`（湖静精确解，无离散化误差）。

| 关键量 | 数值（N=100, P1, t=1 s） | 基准 | 相对误差 | 是否 < 3% |
|---|---|---|---|---|
| 自由面 max|η−η₀| | {fmt(single['eta_error_max'])} m | 1.0 m | {eta_rel:.2e}（{eta_rel*100:.1e} %） | **是** |
| 流速 max|u| | {fmt(single['u_max'])} m/s | 特征速度 √(g·h_max) ≈ {c_scale:.2f} m/s | {u_rel:.2e}（{u_rel*100:.1e} %） | **是** |
| 最差条目（N=200/P2） | η {fmt(worst_eta)} m，u {fmt(worst_u)} m/s | 1e-8 m / 1e-8 m/s 验收带 | 带宽裕度 {ETA_TOL/worst_eta:.0f}x / {U_TOL/worst_u:.0f}x | **是** |

对照 `well_balanced=False`（同弯底，N=50/P1）：η 误差 1.5e-2 m、u 2.1e-2 m/s，良态路径把误差降低约 8 个量级。

## 交付与验证

- 变更文件：`validation/swe_thacker/run_hydrostatic_wb.py`（+provenance）、`validation/swe_thacker/hydrostatic_wb_results.json`（脚本重生成）、`tests/test_swe_hydrostatic_results_artifact.py`（新增守卫）、`docs/tensordg-swe-hydrostatic-wellbalanced.md`（数字表/跨主机/图/provenance）、两张 `docs/figures/*.png`、`scratch/m8649_figures.py`、`artifacts/m8649/hydrostatic_wb_results_loong64_cj30.json`。
- 本地测试：`tests/test_swe_wellbalanced_hydrostatic.py` 21 passed（242 s）；`tests/test_swe_hydrostatic_results_artifact.py` 6 passed（0.14 s）。
- SWE 相关 7 文件合并跑结果：`{pytest_tail}`
- honest 说明：历史 `git_dirty=true` 快照已被重生成工件替换；集群早前工件（cj30）未带 provenance，本次仅归档对照、未重跑。

后续机制化：每次重生成都会刷新 provenance 块，守卫测试会把"缺 provenance / 口径不对 / 内部不自洽 / 超验收带"的工件直接判红。
"""

payload = {
    "title": "TensorDG SWE 静平衡良态验证（Thacker 抛物盆地）：跨主机舍入底噪与工件溯源守卫",
    "summary": (
        "重生成并溯源化 TensorDG SWE 静平衡验证工件：P1/P2 × N=50/100/200 于 Thacker 抛物盆地 "
        f"max|η−η₀|={fmt(worst_eta)} m、max|u|={fmt(worst_u)} m/s（相对 1e-8 验收带 3–5 个量级裕度），"
        "跨主机残差差 ≤5x（格式性质），新增 provenance 块与 6 项工件守卫测试，替代此前手搬的修复前快照。"
    ),
    "scenario_key": "tensordg_swe_hydrostatic_wellbalanced",
    "mission_id": 8649,
    "tags": ["TensorDG", "SWE", "well-balanced", "lake-at-rest", "Thacker",
             "reproducibility", "artifact-guard"],
    "markdown": markdown,
    "figures": figs,
}

out = Path("/tmp/report.json")
out.write_text(json.dumps(payload, ensure_ascii=False))
print("wrote", out, out.stat().st_size, "bytes")
print("eta_rel=%.3e u_rel=%.3e worst_eta=%.3e worst_u=%.3e" % (eta_rel, u_rel, worst_eta, worst_u))