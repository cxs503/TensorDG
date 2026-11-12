#!/usr/bin/env python3
"""Build the TensorDG open-source DG registry (FEAT-OSS-INVENTORY-DG, wave 1).

Design rule: **all numeric/temporal metadata (stars, licence, language,
last push, archived flag, description) is copied programmatically from the raw
GitHub API probe** in ``artifacts/oss_inventory_dg/github_probe_raw.json``.
Only human judgement (tier, role, focus, capability tags, gap anchoring) lives
in this file, so no number in the published registry is hand-typed.

Usage::

    python3 tools/build_oss_inventory_dg.py            # writes registry json
    python3 tools/build_oss_inventory_dg.py --check    # exit 1 if stale

The output is validated by ``tests/test_oss_inventory_dg_registry.py``.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "oss_inventory_dg" / "github_probe_raw.json"
OUT = ROOT / "docs" / "oss_inventory_dg_registry.json"

SCHEMA_VERSION = "1.0"
MIN_REPOS = 20
MIN_GAPS = 4

TIER_RUBRIC = {
    "1": "DG/FR 主体求解器：算法谱系与验证 oracle 可直接对标（可作 wave-2 代码级深盘首选）",
    "2": "DG 使能库/含 DG 组件的框架：提供 DG 离散组件、矩阵自由/小矩阵核、时间积分或 DG 参照实现",
    "3": "DG 生态基础设施：AMR/性能可移植/多物理宿主/大规模 HPC 支撑（不提供 DG 离散本身）",
}

# capability taxonomy used by the coverage matrix (kept small and mechanical)
CAPABILITIES = [
    "basis-modal",
    "basis-nodal",
    "matrix-free",
    "curved-isoparametric",
    "h-adaptivity",
    "p-adaptivity",
    "ader-lts",
    "gpu-portability",
    "shock-limiter",
    "plasma-kinetic",
    "elastic-seismic",
    "fsi-coupling",
    "verification-suite",
]

# ---------------------------------------------------------------------------
# Curated layer (judgement, not numbers).  `focus` states why the repo matters
# for TensorDG; `caps` lists capability tags; `gaps` anchors to GAP-DG-*.
# ---------------------------------------------------------------------------
CURATION: dict[str, dict] = {
    # ---- tier 1: DG / FR solvers -----------------------------------------
    "PyFR/pyfr": dict(
        tier=1, role="core-dg-solver",
        focus="通量重构(FR/CPR)型 DG 的 GPU 参考实现：同一高阶离散在多种后端的可移植写法，"
              "是 TensorDG「矩阵自由 + 可移植后端」演进路线的直接对照",
        caps=["basis-nodal", "gpu-portability", "curved-isoparametric"],
        gaps=["GAP-DG-MATRIXFREE-GPU", "GAP-DG-CURVED-EULER"],
    ),
    "flexi-framework/flexi": dict(
        tier=1, role="core-dg-solver",
        focus="高阶 DG 双曲-抛物框架（Fortran/MPI）：面通量、粘性项与限幅器的工程化组织方式，"
              "对 TensorDG 的 NS 面通量/BR1 路径是同级参照",
        caps=["basis-nodal", "shock-limiter", "curved-isoparametric"],
        gaps=["GAP-DG-CURVED-EULER", "GAP-DG-3D-EULER"],
    ),
    "exadg/exadg": dict(
        tier=1, role="core-dg-solver",
        focus="矩阵自由高阶 DG 的 exa-scale 实现（基于 deal.II）：算子求值不组装矩阵的组织范式，"
              "对应 TensorDG 的 block-Jacobi/矩阵自由后端欠账",
        caps=["matrix-free", "h-adaptivity", "gpu-portability"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
    "trixi-framework/Trixi.jl": dict(
        tier=1, role="core-dg-solver",
        focus="Julia 自适应高阶 DG 守恒律框架：树结构 AMR + 多种数值通量/限幅器组合，"
              "其 elixir 组织方式与收敛验证范式可移植",
        caps=["h-adaptivity", "p-adaptivity", "shock-limiter", "verification-suite"],
        gaps=["GAP-DG-AMR", "GAP-DG-3D-EULER"],
    ),
    "horses-framework/horses3d": dict(
        tier=1, role="core-dg-solver",
        focus="高阶 DG 流动求解器（含曲线网格与湍流模型）：三维 DG 的工程化流程参照，"
              "覆盖 TensorDG 尚未触及的三维可压路径",
        caps=["basis-nodal", "curved-isoparametric", "h-adaptivity"],
        gaps=["GAP-DG-3D-EULER", "GAP-DG-CURVED-EULER"],
    ),
    "ucns3d-team/UCNS3D": dict(
        tier=1, role="core-dg-solver",
        focus="非结构可压 NS 三维 DG：非结构网格上的 DG 数据组织（面-单元邻接表），"
              "是 TensorDG 从张量积笛卡尔网格走向非结构的关键参照",
        caps=["basis-nodal", "shock-limiter", "curved-isoparametric"],
        gaps=["GAP-DG-3D-EULER", "GAP-DG-CURVED-EULER"],
    ),
    "SeisSol/SeisSol": dict(
        tier=1, role="core-dg-solver",
        focus="ADER-DG + 局部时间步进的代表实现（地震波）：TensorDG 仅有 RK 时间积分，"
              "时空预测校正路径的方法学来源",
        caps=["ader-lts", "elastic-seismic", "h-adaptivity"],
        gaps=["GAP-DG-ADER-TIME"],
    ),
    "gkeyllorg/gkeyll": dict(
        tier=1, role="core-dg-solver",
        focus="DG 等离子体动理学框架：DG 在非流体系统（Vlasov/矩系统）上的应用与守恒维护，"
              "可作为 DG 通用性的对照证据",
        caps=["basis-modal", "ader-lts", "plasma-kinetic"],
        gaps=["GAP-DG-ADER-TIME"],
    ),
    "mfem/mfem": dict(
        tier=1, role="core-dg-solver",
        focus="含成熟 DG 离散组件的高性能 FEM 库：DG 弱形式、面/边界积分与 HDG 的权威参照，"
              "TensorDG 的 SIPG/LDG 面项实现可直接对照",
        caps=["basis-nodal", "matrix-free", "h-adaptivity", "verification-suite", "gpu-portability"],
        gaps=["GAP-DG-HDG-POISSON", "GAP-DG-MATRIXFREE-GPU"],
    ),
    # ---- tier 2: DG-capable frameworks / enabling kernels ----------------
    "dealii/dealii": dict(
        tier=2, role="dg-capable-framework",
        focus="通用 FEM 框架，其 matrix-free DG 与几何（等参曲线单元）基础设施被 exadg 等 DG 代码复用；"
              "曲线网格几何守恒律实现参照",
        caps=["matrix-free", "curved-isoparametric", "h-adaptivity", "p-adaptivity"],
        gaps=["GAP-DG-CURVED-EULER", "GAP-DG-MATRIXFREE-GPU"],
    ),
    "libMesh/libmesh": dict(
        tier=2, role="dg-capable-framework",
        focus="支持 DG 的通用有限元框架（面-邻接与悬挂节点处理成熟）：DG 网格数据结构的工程参照",
        caps=["h-adaptivity", "curved-isoparametric"],
        gaps=["GAP-DG-AMR"],
    ),
    "FEniCS/dolfinx": dict(
        tier=2, role="dg-capable-framework",
        focus="支持 DG 单元与 UPWIND 面通量的现代 FEM 前端：DG 弱形式声明式表达参照，"
              "对 TensorDG 的 API 组织有借鉴",
        caps=["matrix-free", "h-adaptivity"],
        gaps=["GAP-DG-HDG-POISSON"],
    ),
    "firedrakeproject/firedrake": dict(
        tier=2, role="dg-capable-framework",
        focus="含 DG/HDG 的生成式有限元系统：HDG 静态凝聚（trace 变量）的实现谱系参照",
        caps=["matrix-free", "h-adaptivity"],
        gaps=["GAP-DG-HDG-POISSON"],
    ),
    "feelpp/feelpp": dict(
        tier=2, role="dg-capable-framework",
        focus="含 DG 组件的 C++ FEM 库：多物理宿主中 DG 的集成方式参照",
        caps=["h-adaptivity"],
        gaps=["GAP-DG-HDG-POISSON"],
    ),
    "gismo/gismo": dict(
        tier=2, role="dg-capable-framework",
        focus="等几何/B 样条求解库：TensorDG 已有 B 样条基导数矩阵工具，等几何曲线几何可作扩展参照",
        caps=["basis-modal", "curved-isoparametric"],
        gaps=["GAP-DG-CURVED-EULER"],
    ),
    "libxsmm/libxsmm": dict(
        tier=2, role="enabling-kernel",
        focus="小矩阵/张量收缩专用库：DG 单元局部算子（(p+1)^d 量级小矩阵）的性能关键核，"
              "是 TensorDG 局部算子加速的直接参照",
        caps=["matrix-free"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
    "paranumal/libparanumal": dict(
        tier=2, role="enabling-kernel",
        focus="GPU 加速高阶有限元/DG 流求解器与核集合：高性能后端组织（OCCA/并行核）参照",
        caps=["matrix-free", "gpu-portability"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
    # ---- tier 3: ecosystem infrastructure -------------------------------
    "DLR-AMR/t8code": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="树结构并行 AMR 数据结构（含 DG 悬挂节点/2:1 平衡约定）：h-自适应的数据层参照",
        caps=["h-adaptivity"],
        gaps=["GAP-DG-AMR"],
    ),
    "AMReX-Codes/amrex": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="块结构 AMR 框架：自适应网格与负载平衡的工程范式（被多物理代码复用）",
        caps=["h-adaptivity"],
        gaps=["GAP-DG-AMR"],
    ),
    "kokkos/kokkos": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="性能可移植编程模型：TensorDG 若需多后端（CPU/GPU）算子，其抽象层是业界标准参照",
        caps=["gpu-portability", "matrix-free"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
    "idaholab/moose": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="多物理应用宿主框架（含 DG 内核）：DG 在大型多物理应用中的集成与验证组织参照",
        caps=["h-adaptivity"],
        gaps=["GAP-DG-AMR"],
    ),
    "petsc/petsc": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="线性/非线性求解器与 DMPlex 网格抽象：DG 面项装配后的求解层参照",
        caps=["matrix-free"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
    "Trilinos/Trilinos": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="大规模并行数值栈（含 DG 离散包与代数多重网格）：exa-scale 求解层参照",
        caps=["matrix-free", "gpu-portability"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
    "Nek5000/NekRS": dict(
        tier=3, role="ecosystem-infrastructure",
        focus="谱元 GPU 大规模流动求解器：高阶连续谱元 + GPU 的工程实现，"
              "与高阶 DG 在后端/并行策略上可交叉借鉴",
        caps=["gpu-portability", "matrix-free"],
        gaps=["GAP-DG-MATRIXFREE-GPU"],
    ),
}

# Verified but deliberately outside the 24-entry registry (kept for wave 2).
RESERVE = {
    "Nek5000/Nek5000": "谱元（连续）HPC 代码，与 DG 离散非同族，本轮仅作历史参照",
    "spectralDNS/spectralDNS": "谱方法（Fourier/Chebyshev），自 2023 起未更新，暂不占册位",
    "KratosMultiphysics/Kratos": "以 FEM 为主，DG 组件非核心能力，待 wave-2 核验后再定",
    "Precice/precice": "多物理耦合库，非 DG 离散，暂缓",
    "trixi-framework/TrixiParticles.jl": "SPH（无网格）实现，属 TensorSPH 域，非本册",
    "firedrakeproject/loopy": "代码生成工具（1 star），DSL 生成 DG 核的潜在方向，暂缓",
}

# ---------------------------------------------------------------------------
# GAP-DG-* proposals: each anchored to real TensorDG scope gaps and backed by
# >=2 registry entries as evidence.
# ---------------------------------------------------------------------------
GAP_PROPOSALS = [
    dict(
        slug="GAP-DG-ADER-TIME",
        title="ADER-DG 时空预测校正时间积分（对照 RK 基线）",
        rationale="TensorDG 时间推进仅 RK（README「Current scope」与 tensordg/ 实现），"
                  "而 ADER-DG 是 DG 社区主流的高阶单步替代路径；SeisSol/Gkeyll 提供方法谱系证据。",
        acceptance=("1-D Euler Sod 与 2-D isentropic vortex 在 p=1,2,3 下：观测收敛阶 >= p+1-0.15；"
                    "vortex L2 密度误差 <1%；同一 CFL 下与 RK 基准解差 <1e-8；"
                    "守恒量漂移 <1e-12；新增单测数 >=3"),
        domain="TensorDG/time-integration",
        priority="P1",
        repos_evidence=["SeisSol/SeisSol", "gkeyllorg/gkeyll"],
    ),
    dict(
        slug="GAP-DG-3D-EULER",
        title="三维可压 Euler DG（现仅三维周期 advection）",
        rationale="tensordg/advection_3d.py 存在但无三维 Euler；Euler 仅 1-D/2-D。"
                  "HORSES3D/UCNS3D/Trixi 提供三维 DG 工程参照。",
        acceptance=("3-D isentropic vortex 在 p=1,2,3 上 h-收敛阶 >= p+0.85；L2 密度误差 <1%；"
                    "8^3 单元 p=2 单节点运行 <600 s；无负密度/负压；守恒漂移 <1e-12"),
        domain="TensorDG/compressible",
        priority="P1",
        repos_evidence=["horses-framework/horses3d", "ucns3d-team/UCNS3D", "trixi-framework/Trixi.jl"],
    ),
    dict(
        slug="GAP-DG-AMR",
        title="h-自适应（树结构 AMR + 2:1 平衡）",
        rationale="TensorDG 全部算例使用均匀张量积网格，无 AMR；"
                  "t8code/AMReX/Trixi 提供树结构与负载平衡参照。",
        acceptance=("2-D vortex advection 上：2:1 平衡成立（机器检查）、悬挂节点面通量守恒（漂移 <1e-11）；"
                    "等自由度下误差 <= 0.7x 均匀网格误差；提供回归测试"),
        domain="TensorDG/mesh",
        priority="P2",
        repos_evidence=["DLR-AMR/t8code", "AMReX-Codes/amrex", "trixi-framework/Trixi.jl"],
    ),
    dict(
        slug="GAP-DG-HDG-POISSON",
        title="HDG（静态凝聚）与现有 LDG/SIPG 的精度-自由度效率对照",
        rationale="TensorDG 有 poisson_2d_ldg.py 与 sipg_sparse/matrixfree，但无 HDG 的 trace 凝聚路径；"
                  "MFEM/Firedrake/dolfinx 提供 HDG 参照实现。",
        acceptance=("2-D Poisson p=1..3：HDG L2 误差 <= 1.2x SIPG 同自由度误差；"
                    "全局未知量（凝聚后）相对 SIPG 减少 >=30%；报告自由度表与收敛阶；新增单测 >=2"),
        domain="TensorDG/diffusion",
        priority="P2",
        repos_evidence=["mfem/mfem", "firedrakeproject/firedrake", "FEniCS/dolfinx"],
    ),
    dict(
        slug="GAP-DG-CURVED-EULER",
        title="曲线/等参网格上的 Euler（现曲线几何仅覆盖 advection）",
        rationale="tensordg/curved_geometry.py 与 docs/DG_CURVED_GEOMETRY.md 只覆盖标量 advection；"
                  "曲线网格上的几何守恒律（GCL）与 Euler 尚未验证。",
        acceptance=("曲线网格上自由流保持误差 <1e-12（GCL 机器检查）；warped mesh 上 vortex 收敛阶 >= p+1-0.2；"
                    "与笛卡尔网格同算例误差比 <=2x"),
        domain="TensorDG/geometry",
        priority="P2",
        repos_evidence=["PyFR/pyfr", "dealii/dealii", "gismo/gismo"],
    ),
    dict(
        slug="GAP-DG-MATRIXFREE-GPU",
        title="矩阵自由/多后端（GPU）DG 算子路径",
        rationale="TensorDG 为 NumPy 张量实现 + block-Jacobi 预条件，无 GPU 后端与算子融合；"
                  "PyFR/libxsmm/libparanumal/Kokkos 提供后端抽象与小矩阵核参照。",
        acceptance=("2-D Euler p=2 上：新后端吞吐 >=5x 现有 CPU 基线；与基线解差 <1e-10；"
                    "后端可切换且默认路径不变（回归全绿）"),
        domain="TensorDG/performance",
        priority="P3",
        repos_evidence=["PyFR/pyfr", "libxsmm/libxsmm", "paranumal/libparanumal", "kokkos/kokkos"],
    ),
]


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    raw = json.loads(RAW.read_text())
    cand = raw["candidates"]
    verified = {v["full_name"]: v for v in cand.values() if v.get("ok")}
    # GitHub canonicalises owner/repo case (PyFR/PyFR, Nek5000/nekRS,
    # trilinos/Trilinos); match case-insensitively and emit the API spelling.
    by_lower = {k.lower(): k for k in verified}
    canon = {k.lower(): by_lower[k.lower()] for k in CURATION if k.lower() in by_lower}

    missing = sorted(k for k in CURATION if k.lower() not in by_lower)
    if missing:
        raise SystemExit(f"curated repos missing from probe: {missing}")

    repos = []
    for name, cur in CURATION.items():
        name = canon[name.lower()]
        meta = verified[name]
        repos.append({
            "repo": name,
            "url": f"https://github.com/{name}",
            "tier": cur["tier"],
            "dg_role": cur["role"],
            "focus": cur["focus"],
            "capabilities": sorted(cur["caps"]),
            "tensordg_gaps": sorted(cur["gaps"]),
            # --- numeric/temporal metadata: copied from probe, never typed ---
            "stars": meta["stars"],
            "forks": meta["forks"],
            "language": meta["language"],
            "license": meta["license"],
            "pushed_at": meta["pushed_at"],
            "archived": meta["archived"],
            "size_kb": meta["size_kb"],
            "description_en": meta["description"],
            "topics": meta.get("topics", []),
            # -----------------------------------------------------------------
            "provenance": {"source": "api.github.com/repos/" + name,
                           "raw_file": str(RAW.relative_to(ROOT))},
        })
    repos.sort(key=lambda r: (r["tier"], -r["stars"], r["repo"]))

    reserve = []
    for name, why in RESERVE.items():
        if name.lower() not in by_lower:
            continue
        name = by_lower[name.lower()]
        meta = verified[name]
        reserve.append({"repo": name, "url": f"https://github.com/{name}",
                        "reason_excluded": why, "stars": meta["stars"],
                        "license": meta["license"], "pushed_at": meta["pushed_at"]})

    gap_slugs = {g["slug"] for g in GAP_PROPOSALS}
    for r in repos:
        unknown = set(r["tensordg_gaps"]) - gap_slugs
        if unknown:
            raise SystemExit(f"{r['repo']} anchors unknown gaps {unknown}")

    # canonicalise repo spellings inside gap evidence (GitHub case rules)
    for g in GAP_PROPOSALS:
        g["repos_evidence"] = [by_lower[e.lower()] if e.lower() in by_lower else e
                               for e in g["repos_evidence"]]

    tiers = {str(t): 0 for t in (1, 2, 3)}
    for r in repos:
        tiers[str(r["tier"])] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "feature_anchor": "FEAT-OSS-INVENTORY-DG",
        "wave": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "LBM wave-1 协议接力：普查（本册）→ tier 分级 → GAP-DG-* 锚点 → wave-2 逐仓代码级深盘",
        "provenance": {
            "api": "api.github.com (unauthenticated, read-only)",
            "raw_file": str(RAW.relative_to(ROOT)),
            "raw_sha256": _sha256(RAW),
            "probe_time": raw.get("probed_at"),
            "note": "stars/license/language/pushed_at/archived/description 均由脚本从 raw 探针拷贝，零手抄",
        },
        "tier_rubric": TIER_RUBRIC,
        "capability_taxonomy": CAPABILITIES,
        "coverage": {
            "repos_in_registry": len(repos),
            "repos_verified_total": len(set(verified)),
            "reserve_excluded": len(reserve),
            "tier_counts": tiers,
            "gap_proposals": len(GAP_PROPOSALS),
            "capability_coverage": {
                c: sum(1 for r in repos if c in r["capabilities"]) for c in CAPABILITIES
            },
        },
        "repos": repos,
        "reserve": reserve,
        "gap_proposals": GAP_PROPOSALS,
        "limitations": [
            "波1 只做公开元数据 + 领域判断，未逐仓 clone 做代码级方法核查（wave-2 按 LBM 协议逐仓深盘）",
            "capabilities 为保守标签（依据仓描述与公开文档），未做逐仓源码定位，wave-2 需给出文件级证据",
            "tier 分级为启发式判断，非度量；跨档边界仓（如 libMesh/feelpp）已在 focus 中说明",
            "stars/pushed_at 为探针时刻快照，会随时间漂移；再生脚本 tools/build_oss_inventory_dg.py 可刷新",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify committed registry matches regeneration")
    args = ap.parse_args(argv)
    reg = build()
    text = json.dumps(reg, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists():
            print("MISSING", OUT, file=sys.stderr)
            return 1
        cur = json.loads(OUT.read_text())
        # generated_at legitimately changes; compare everything else
        cur.pop("generated_at", None)
        new = dict(reg)
        new.pop("generated_at", None)
        if cur != new:
            print("STALE: docs/oss_inventory_dg_registry.json out of date, rerun builder", file=sys.stderr)
            return 1
        print("OK registry up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}: {reg['coverage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())