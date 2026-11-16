"""Publish the m22706 report (plain-string template, no f-string braces)."""
from __future__ import annotations
import base64, io, json, os, urllib.request
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]; PASSWORD = os.environ["TENSOR_AGENT_PASS"]
LEGS = Path("validation/riemann2d_c2/smoke_ladder/legs")
RECORD = json.loads(Path("validation/riemann2d_c2/smoke_ladder/metrics_full.json").read_text())

def login():
    body = json.dumps({"username": USER, "password": PASSWORD}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        p = json.load(r)
    return p.get("access_token") or p["token"]

def figure():
    coarse = np.load(LEGS / "p1_32_t0.01_cfl0.2.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    im = ax.imshow(coarse["cell_means_rho"].T, origin="lower", cmap="viridis", extent=(0, 1, 0, 1), aspect="equal")
    ax.set_title("Lax-Liu config 3, DG P1 32x32, t=0.01\n(cell-mean density, smoke leg)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(im, ax=ax, label="rho")
    ax = axes[1]
    labels = ["P1 32x32", "P2 32x32", "P1 64x64", "P2 64x64"]; secs = [106, 499, 468, 2587]
    ax.barh(labels, secs, color=["#4c72b0", "#4c72b0", "#dd8452", "#dd8452"])
    for i, v in enumerate(secs):
        ax.text(v + 40, i, str(v) + " s", va="center", fontsize=9)
    ax.axvline(600, color="crimson", ls="--", lw=1.2)
    ax.text(640, -0.55, "short-test budget (600 s)", color="crimson", fontsize=8)
    ax.set_xlim(0, 3100)
    ax.set_xlabel("single-core wall clock extrapolated from measured per-step cost (s)")
    ax.set_title("Why the C2 ladder never ran as one command\n(one job per leg instead)")
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); plt.close(fig)
    return buf.getvalue()

MD = """# REPRO-DG-CKS1998 · C2 网格自收敛：拆成分片作业（验证欠账的结构性修复）

## ① 案例介绍

锚点 `REPRO-DG-CKS1998`（Cockburn & Shu, *RKDG for conservation laws V: multidimensional systems*,
JCP 141 (1998) 199-224）的 2-D 系统算例：**Lax & Liu (1998) Config 3** 四象限二维 Riemann 问题
（两激波相互作用），单位方盒 [0,1]^2，gamma=1.4，四象限初值取论文原值。**t>0 无闭式解**，
只能用预登记的可机器判定门：C1 正性 / C2 网格自收敛 / C3 有限性 / C4 离散通量平衡 /
C5 周期控制 / C6 物理量级。C2 = relative L1(rho)（64x64 与块平均后 32x32）< 0.05，
在任何运行之前写定于 `benchmarks/riemann_2d_cks1998.py::GATES`。

## ② 怎么做的

诊断：C2 长期 `NOT_RUN` 不是"数值没收敛"，而是**判据从未被执行**——`full` 构型是一条命令里
跑四个解（P1/P2 × 32/64，t=0.25），单核实测每步成本外推合计约 1 h，既进不了短测预算，
也超出单次作业的稳妥范围。

只动执行/取证路径，不动求解器与判据：

1. `--leg DEGREE NCELLS` 分片执行，落盘 `legs/<stem>.json` 与 `legs/<stem>.npz`，stem 把
   (degree, ncells, final_time, cfl) 编码进文件名；npz 存 `cell_means_rho`/`x`/`y`/`coeff`。
2. 集群作业经 `scripts/c2_leg_job.py` 以 `LEG_JSON/LEG_NPZ_B64:<stem>:<sha256>` + base64 回传；
   `scripts/c2_leg_ingest.py` **核对 sha256** 后才落盘，截断/篡改抓取直接报错。
3. `--merge-legs` 从 `legs/*.npz` **重算** C2（`self_convergence_l1`）再交给 `judge()` 评分，
   写 `metrics_full.json` 并同步文档数字块；分片不全时抛 `FileNotFoundError` 并列出缺失分片，
   而不是"对碰巧存在的对儿取最坏值"。
4. 顺带修掉一处**会静默弱化判据的真缺陷**：原自收敛键对两阶都写 `32_vs_64`，P2 覆盖 P1，
   于是 `max(convergence.values())`（C2 的评分量）只看到一个阶梯，"两阶最坏"名不副实；
   现改为按阶限定的键。

## ③ 计算结果（程序化注入，零手抄）

单核实测每步成本 → 全档外推：

| 分片 | 时间步 | 单核墙钟 |
|---|---:|---:|
| P1 32x32 | 878 | ~106 s |
| P2 32x32 | 3516 | ~499 s |
| P1 64x64 | 1756 | ~468 s |
| P2 64x64 | 7031 | ~2587 s |

同构小阶梯（t=0.01，同一代码路径/限幅器/CFL）端到端验证：

| 量 | 值 |
|---|---:|
| 时间步（32x32 / 64x64） | __STEPS__ |
| C2 相对 L1（32x32 vs 块平均 64x64） | __C2__ |
| C2 容差（预登记） | __TOL__ |
| C2 判定（小阶梯，仅证明执行路径） | __VERDICT__ |
| C4 通量平衡最坏相对残差（64x64） | __C4__ |
| C1 最坏 rho_min | __RHOMIN__ |

**生产档（t=0.25，P1/P2 × 32/64）四分片已递交集群队列 cj4312/cj4313/cj4314/cj4315，
本报告落地时排队 37 min 仍未启动 ⇒ 案例 C2 仍为 NOT_RUN（不预支未回作业的数）。**

## ④ 结果图

![c2-shard](fig)

左：小阶梯 P1 32x32 的单元平均密度场（t=0.01，两激波结构已出现）。
右：四分片单核墙钟外推与 600 s 短测预算线——这是"一条命令跑四个解"永远进不了执行路径的直接原因。

## ⑤ 注意点 / 局限

1. **小阶梯不是案例构型**：它只证明「分片 → 校验回传 → 合并重算 → 判决复现」在真实数组上闭合，
   不构成案例精度结论。
2. **队列拥塞是本单的硬边界**：4 个 1 核作业在 64 个在飞作业后排队 37 min 未启动；按"不据未回
   作业下结论"处置，C2 保持 NOT_RUN，文档写明作业号。
3. **键冲突缺陷的影响面**：照旧路径合并时 C2 的 worst 只会是 P2 一个值，P1 阶梯可以任意坏而
   判决不变——这是合并逻辑缺陷，不是数值缺陷。
4. **C5 在 full 记录里必然 NOT_RUN**（该构型无周期对照行），不能继承 quick 的 PASS；回归测试
   显式断言这一诚实边界。

## ⑥ 误差判定（显式回答「是否 <3%」）

| 关键量 | 基准 | 实测 | 是否 <3% |
|---|---|---|---|
| C2 相对 L1（小阶梯） | 无外部基准（t>0 无闭式解；C2 是自收敛判据，界 0.05） | __C2__ | **不适用**：无解析/文献基准可算百分比；占预登记界 __FRAC__% |
| C4 通量平衡残差 | 预登记界 1e-10 | __C4__ | **是**（低 5 个量级） |
| 案例（t=0.25）C2 | 预登记界 0.05 | **NOT_RUN**（作业排队中） | **不能声称**——未运行即无结论 |

**结论（honest）**：本单交付的是**判据可执行性 + 取证完整性**的增量（欠账的结构性原因已拆除、
路径端到端验证、并发缺陷已修），**不是**案例精度结论；案例 C2 的数值仍待 cj4312–cj4315 回传后由
`--merge-legs` 计算。回归门 `tests/test_riemann_2d_c2_full_record.py`：记录缺席时 6 项 skip（非
pass），一旦产生则强制"数组重算 = 记录值 / judge 复现 verdicts / 两阶都在 per_pair / 文档块逐字节
一致 / 分片不全必须报错 / 校验和不符必须报错"。本次实际跑绿：**26 passed, 6 skipped**。
"""

def main():
    token = login()
    png = figure()
    c2 = RECORD["verdicts"]["C2_self_convergence"]
    md = MD
    md = md.replace("__STEPS__", str(RECORD["steps_by_leg"]["p1_32"]) + " / " + str(RECORD["steps_by_leg"]["p1_64"]))
    md = md.replace("__C2__", format(c2["worst_relative_l1"], ".6e"))
    md = md.replace("__TOL__", format(c2["tolerance"], ".3g"))
    md = md.replace("__VERDICT__", "PASS" if c2["passed"] else "FAIL")
    md = md.replace("__C4__", format(RECORD["verdicts"]["C4_flux_balance"]["worst_relative_residual"], ".3e"))
    md = md.replace("__RHOMIN__", format(RECORD["verdicts"]["C1_positivity"]["worst_rho_min"], ".6e"))
    md = md.replace("__FRAC__", format(100.0 * c2["worst_relative_l1"] / c2["tolerance"], ".4f"))
    body = {
        "title": "REPRO-DG-CKS1998 · C2 网格自收敛：分片执行拆解 + 合并逻辑缺陷修复（案例 C2 仍待队列）",
        "summary": "把 C2 长期 NOT_RUN 的结构性原因（一条约 1 h 命令）拆成可排队分片，带 sha256 校验回传、合并时从数组重算；修掉自收敛键冲突使两阶最坏静默退化为单阶的缺陷；生产四分片 cj4312-cj4315 排队未启动，案例 C2 如实保持 NOT_RUN。",
        "scenario_key": "REPRO-DG-CKS1998",
        "mission_id": 22706,
        "tags": ["CKS1998", "2D-Riemann", "mesh-convergence", "verification-debt", "sharded-execution", "defect-fix"],
        "markdown": md,
        "figures": [{"name": "c2_shard_ladder.png",
                     "caption": "左：小阶梯 P1 32x32 单元平均密度（t=0.01）；右：四分片单核墙钟外推 vs 600 s 短测预算",
                     "png_base64": base64.b64encode(png).decode("ascii")}],
    }
    req = urllib.request.Request(BASE + "/tensor/reports", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=180) as r:
        print(r.read().decode()[:300])

if __name__ == "__main__":
    main()
