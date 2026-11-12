#!/usr/bin/env python3
"""Publish the m16228 board-audit report to the platform.

Numbers are read from the artifacts at publish time (zero hand-copy):
  runs/m16228/case_dg_solid_body_rotation_2d_p*.json  (subject, audited copies)
  runs/m16228/auditor_repro_p2n25.json                (auditor reproduction)
  runs/m16228/auditor_novel_p1n20.json                (auditor anti-hardcode probe)
Figure: docs/figs/m16228_sbr_audit.png (base64-embedded).
"""
from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "m16228"
sys.path.insert(0, str(ROOT / "tools"))
from tp_api import post  # noqa: E402  (token from /root/tp-agent/agent.env)

TOL = 0.05


def case(name: str) -> dict:
    return json.loads((RUNS / name).read_text())["case"]


def rows() -> tuple[str, list[str]]:
    lines, gates = [], []
    items = []
    for deg in (1, 2):
        for p in sorted(RUNS.glob(f"case_dg_solid_body_rotation_2d_p{deg}_n*.json")):
            items.append(case(p.name))
    novel = case("auditor_novel_p1n20.json")
    items.append({**novel, "ncells": 20, "degree": 1})
    for c in items:
        l1 = c["l1_relative"]
        ok1, ok2 = l1 < TOL, c["peak_retention_nodal"] > 1 - TOL
        gates.append(f"P{c['degree']} n={c['ncells']}: L1 {'PASS' if ok1 else 'FAIL'}")
        lines.append(
            f"| P{c['degree']} | {c['ncells']} | {c['dofs']:,} | {c['steps']} | "
            f"{c['l1_relative']*100:.3f}% | {c['l2_relative']*100:.3f}% | "
            f"{c['peak_retention_nodal']*100:.3f}% | {c['mass_drift_rel']:.2e} | "
            f"{'✓' if ok1 else '✗'} | {'✓' if ok2 else '✗'} |")
    return "\n".join(lines), gates


def eoc() -> str:
    out = []
    for deg in (1, 2):
        pts = sorted((case(p.name)["ncells"], case(p.name)["l1_relative"])
                     for p in RUNS.glob(f"case_dg_solid_body_rotation_2d_p{deg}_n*.json"))
        for (n0, e0), (n1, e1) in zip(pts, pts[1:]):
            out.append(f"P{deg} {n0}→{n1}: EOC = {math.log(e0/e1)/math.log(n1/n0):.3f}")
    return "; ".join(out)


def main() -> None:
    table, gates = rows()
    repro = case("auditor_repro_p2n25.json")
    subj = case("case_dg_solid_body_rotation_2d_p2_n25.json")
    same = all(abs(repro[k] - subj[k]) < 1e-18 for k in
               ("l1_relative", "l2_relative", "peak_retention_nodal", "mass_drift_rel"))
    novel = case("auditor_novel_p1n20.json")
    cov = subj["l1_reference"] / subj["mass_initial"]
    png = base64.b64encode((ROOT / "docs/figs/m16228_sbr_audit.png").read_bytes()).decode()

    md = f"""# 交差核验：m16200 · CASE-DG-SOLID-BODY-ROTATION-2D（判级 L3，L4 否决）

**结论**：代码真、可跑、可复现、非硬编码；但**被声明的验收算例 100×100（CFL=0.5, L1<5%）从未执行**——
无工件、3 条验收断言 `pytest.skip`、集群作业 cj1265 递交 30 min 仍 `queued`；平台基准表无本锚点行；三个交付文件在工作树里全部未入 git。
判 **L3**（功能实质 + 主解算器集成），**L4 不成立**（基准验证缺最后一环）。verdict id=452。

## ① 案例介绍
- 物理背景：周期单位方域上的标量刚体旋转输运 `u_t + ∇·(a u)=0`，`a = ω(-(y-½), x-½)`，`ω=2π`（一周 T=1）。
- 几何/初值：unit square [0,1]²，高斯包 center (0.5, 0.35)，σ=0.075，峰值 1.0。
- 参数：CFL=0.5；方法 modal Legendre DG（p∈{{1,2}}）+ upwind 通量 + RK4；误差范数 disk r=0.28 @ (0.5,0.5)。
- 声明判据（m16200 goal）：`网格 100×100、CFL=0.5、旋转一周后 L1 误差与极值衰减 vs 初值、容差 <5%`。
- 外部锚：LeVeque 2002 *Finite Volume Methods for Hyperbolic Problems* Ch.19, DOI 10.1017/cbo9780511791253（引用，未对照其数值表）。

## ② 怎么做的（审计方法，非重做）
1. 工件取证 + 全部交付文件 sha256；`git ls-files` / `git log --all` / `git status` 核落地。
2. 照抄被审报告 §5.2 的 repro_cmd 独立重跑 → 与被审工件 7/7 指标逐位比对（结论：`identical={same}`）。
3. 反硬编码探针：跑工件集中不存在的新配置 P1 n=20 → 得到新值 `{novel['l1_relative']*100:.3f}%`（不在任何被审工件中）。
4. 非回归：改动后模块跑既有 `tests/test_advection_2d_var.py`（进程内打印模块路径自证）→ 14/14 绿。
5. 直查库：`tensor_benchmark_cases` 锚点 0 行；集群 `cj1265/cj1271` 状态 `queued`。
6. 审计侧重算 EOC 与 disk 范数的质量覆盖面。

## ③ 计算结果（数字由脚本从工件注入，零手抄）
| deg | n | DOFs | steps | L1 rel | L2 rel | peak 保留 | mass drift | L1<5% | peak>95% |
|---|---|---|---|---|---|---|---|---|---|
{table}

- 审计侧重算观测阶：{eoc()}
- 复现一致性：`l1/l2/peak/mass` 四项与 P2 n=25 被审工件逐位相同 → `{same}`
- 守恒性：所有档位 `|mass drift| ≤ {max(case(p.name)['mass_drift_rel'] for p in RUNS.glob('case_dg_solid_body_rotation_2d_p*.json')):.2e}`

## ④ 结果图
误差-网格双对数图（含 5% 声明门线、审计侧复现点、反硬编码点，以及 **100×100 声明算例 NOT RUN 阴影带**）+ 极值保留柱图。
![audit](fig)

## ⑤ 注意点 / 局限
1. **验收算例未执行**（F1）：n=100 需 ~1500–2000 s，超出本单墙钟；集群排队未派发 → 锚点判据只能记 NOT_RUN，不能记 PASS。
2. **判据自定**（F2）：`<5%` 只写在 goal 文本与被审自己脚本的 `TOLERANCE=0.05` 里，`tensor_benchmark_cases` 无登记行。
3. **交付未入仓**（F3）：1 个 M + 9 个 untracked，`head_commit=null`；`succeeded` 来自 dev_acceptance 自动批准，不是 PR 合并。
4. **测试形同虚设**（F4）：`_load_artifact` 缺文件即 `pytest.skip` → `4 passed` 里 0 条覆盖声明判据。
5. **指标口径**（F5）：disk r=0.28 只覆盖初值 L1 质量的 `{cov*100:.2f}%`，另有 `{(1-cov)*100:.2f}%` 场域误差不计入。
6. **声明未落地**（F6）：`cone_profile`/`slotted_cylinder` 零调用方、`smooth_bump` 零工件 → 关于**间断**输运（LeVeque/Zalesak 类）本次无任何新证据。
7. **外推当结论**（F7）：EOC 实测高于设计阶（预渐近区），不能外推到 n=100。**repro_cmd 依赖 PYTHONPATH**（F8），否则命中别的 worktree 的包报 ImportError。

## ⑥ 误差判定（逐项，显式回答）
| 量 | 声明容差 | got | 判 |
|---|---|---|---|
| L1 rel @100×100（声明验收算例） | <5% | **未执行**（无工件/断言 skip/作业 queued） | **NOT_RUN** |
| L1 rel @P2 n=25 | <5% | {subj['l1_relative']*100:.3f}% | PASS（粗网格） |
| peak 保留 @P2 n=25 | >95% | {subj['peak_retention_nodal']*100:.3f}% | PASS（粗网格） |
| mass drift @P2 n=25 | 机器精度 | {subj['mass_drift_rel']:.2e} | PASS |
| L1 rel @P1 n=20（审计新配置） | <5% | {novel['l1_relative']*100:.3f}% | FAIL（P1 粗网格不达标，符合预期） |

**是否 <3%？** 在被审实际跑过的算例里，P2 n=25 的 L1 相对误差 {subj['l1_relative']*100:.3f}% < 3%；
但**声明的那一个算例（100×100）没有数字**，因此「本锚点判据 <3%/<5% 是否满足」的正确答案是 **未判定（NOT_RUN）**，不得含糊记为通过。
诚实面：被审报告三处如实标注 n=100 pending，反硬编码探针未发现回填。

## 处置
Approve with notes：核定 L3，拒绝 L4 主张；算例池该锚点保持 open。修补顺序=①入仓 commit ②补跑 n=100 并落工件 ③`_load_artifact` 改 fail ④登记 `tensor_benchmark_cases` 行 ⑤把 slotted_cylinder 接进 `--profile` 再谈间断能力。
"""

    payload = {
        "title": "m16228 交差核验：m16200 CASE-DG-SOLID-BODY-ROTATION-2D（判级 L3，L4 否决）",
        "summary": ("审计 m16200：代码真且逐位可复现（7/7 指标）、非硬编码（新配置产生新值）、既有 14 条解算器测试改动后全绿；"
                    "但声明的 100×100 验收算例从未执行（无工件/3 断言 skip/cj1265 queued）、基准登记表无本锚点行、交付未入 git。判 L3，L4 否决。"),
        "scenario_key": "CASE-DG-SOLID-BODY-ROTATION-2D",
        "mission_id": 16228,
        "tags": ["audit", "TensorDG", "solid-body-rotation", "maturity-L3", "m16200", "honest-pending"],
        "markdown": md,
        "figures": [{"name": "m16228_sbr_audit.png",
                     "caption": "L1 误差-网格（含 5% 门线与 100×100 NOT RUN 阴影带）+ 极值保留柱图",
                     "png_base64": png}],
    }
    resp = post("/tensor/reports", payload)
    print(json.dumps(resp, ensure_ascii=False)[:600] if not isinstance(resp, str) else resp[:600])


if __name__ == "__main__":
    main()