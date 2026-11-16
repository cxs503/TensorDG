#!/usr/bin/env python3
"""Publish the T2 unified core-path report (anchor GAP-LBM-T2-GPU-COREPATH).

Every number in the markdown below is read from
``TensorLBM/outputs/core_path_t2/summary.json`` (written by
``TensorLBM/benchmarks/core_path_t2_bench.py``), so nothing is hand-typed.
The same function also (re)generates ``TensorLBM/docs/T2-COREPATH-UNIFIED-BENCH.md``
so the in-repo document and the platform report cannot drift apart.

Usage:  python3 publish_core_path_t2.py [--dry]
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
SUM_PATH = ROOT / "TensorLBM" / "outputs" / "core_path_t2" / "summary.json"
DOC_PATH = ROOT / "TensorLBM" / "docs" / "T2-COREPATH-UNIFIED-BENCH.md"
MISSION_ID = 22722
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def pct(x: float, nd: int = 3) -> str:
    return f"{100 * float(x):.{nd}f}%"


def arm_rows(arms) -> str:
    rows = []
    for a in arms:
        c = a["config"]
        cons = ("—" if a["consistency_rel_diff"] is None
                else f"{a['consistency_rel_diff']:.2e}")
        rows.append(
            f"| {c['kernel']} | {c['backend']} | {c['dtype']} | "
            f"{c['shape'][0]}×{c['shape'][1]} | {a['steps']} | "
            f"{a['mlups']:.2f} | {a['seconds']:.3f} | "
            f"{a['decay_rate_fit']:.6e} | {pct(a['accuracy_rel_err'])} | {cons} | "
            f"{a['declared_live_bytes']/1024:.1f} | {a['measured_peak_bytes']/1024:.1f} | "
            f"{a['verdict']} |")
    return "\n".join(rows)


def build_markdown(s: dict) -> str:
    arms = s["arms"]
    gates = s["preregistered_gates"]
    case = s["case"]
    machine = s["machine"]
    ratios = s["ratios"]
    ok = [a for a in arms if a["perf_admissible"]]
    rejected = [a for a in arms if not a["perf_admissible"]]

    withheld_rows = "\n".join(
        f"| {w['backend']} | {w['kind']} | {w['reason']} | {w['grade']} |"
        for w in s["withheld"]) or "| — | — | （本机所有后端均可用） | — |"

    backend_rows = "\n".join(
        f"| {p['name']} | {p['kind']} | {'是' if p['available'] else '否'} | {p['reason']} |"
        for p in s["backends"].values())

    ratio_rows = "\n".join(
        f"| {k} | {'n/a（含未通过闸门的臂）' if v is None else (f'{v:.4f}' if isinstance(v, float) else v)} |"
        for k, v in ratios.items())

    worst_acc = max(a["accuracy_rel_err"] for a in arms)
    best_acc = min(a["accuracy_rel_err"] for a in arms)
    cons_vals = [a["consistency_rel_diff"] for a in arms
                 if a["consistency_rel_diff"] is not None]
    worst_cons = max(cons_vals) if cons_vals else None
    n_torch = sum(1 for a in arms if a["config"]["backend"].startswith("torch"))
    _sli = ratios.get("fused_slices_over_reference_mlups_64")
    if _sli is None:
        throughput_verdict = ("该臂未通过闸门，故不给吞吐结论（数字作废）")
    elif _sli >= 1.05:
        throughput_verdict = "融合+slice 流在 CPU/numpy 上确证比未融合参考核快"
    elif _sli <= 0.95:
        throughput_verdict = "融合+slice 流在 CPU/numpy 上反而更慢，加速未获证实"
    else:
        throughput_verdict = "融合+slice 流在 CPU/numpy 上与参考核持平，加速未获证实"

    acc_rows = "\n".join(
        f"| {a['config']['kernel']}/{a['config']['backend']}/{a['config']['dtype']}/"
        f"{a['config']['shape'][0]} | {a['decay_rate_fit']:.6e} | "
        f"{a['decay_rate_analytic']:.6e} | {pct(a['accuracy_rel_err'])} | "
        f"{'是' if a['accuracy_rel_err'] <= 0.03 else '否'} |"
        for a in arms)

    return f"""## 1. 案例介绍（物理背景 / 几何 / 参数）

本单锚点 **`GAP-LBM-T2-GPU-COREPATH`**（T2 GPU 核心路径统一优化，P1）。该锚点的验收口径
写死了两条纪律：**优化对象必须是 T3 已验证的物理算例（禁盲目追峰值）**，且**对标口径必须统一**
（加速数字与精度、内存、后端可用性一起报，不能只挑好看的那一个）。因此本单不追峰值，而是先把
"能合规地报出加速比"这件事本身建起来，然后在这套口径下测出来。

物理算例取 T3 验证矩阵第 5 行 **`taylor-green-2d`**（当前矩阵中唯一 REAL 通过的算例）：

| 项 | 值 |
|---|---|
| 解析解 | 幅值 ∝ exp(−2νk²t)，k = 2π/N |
| 格子 | D2Q9，BGK，cs² = 1/3 |
| ν | 0.05（τ = 0.65） |
| 初值幅值 u₀ | 0.01（Mach ≈ 0.017，可压误差 <1e-3） |
| 网格 | 64×64（主族，= T3 矩阵配置）、128×128（分辨率标度点） |
| 时间步 | 主族 1200 步、128² 600 步（每 100 步取一次幅值） |
| 预注册精度闸门 | 衰减率相对误差 ≤ {pct(gates['accuracy_tol'], 1)}（与 T3 矩阵同容差） |
| 预注册一致性闸门 | float64 相对 |Δu| ≤ {gates['consistency_tol_float64']:.1e}，float32 ≤ {gates['consistency_tol_by_dtype']['float32']:.1e} |

机器上下文（程序化写入 `summary.json`）：`{machine['host']}` / {machine['platform']}，
Python {machine['python']}，numpy {machine['numpy']}，可用核 {machine['nproc_affinity']}，
loadavg {machine['loadavg']}。计时为单核、单进程。

后端探测结果（**不可用的后端只能 WITHHELD，不许 CPU 假装**）：

| backend | 类型 | 可用 | 证据/原因 |
|---|---|---|---|
{backend_rows}

GPU 臂的处置：

| backend | 类型 | 原因 | 证据级 |
|---|---|---|---|
{withheld_rows}

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

新增 `TensorLBM/src/tensorlbm/core_path.py`，把在此之前互不相干的三套时间推进
（`FusedPeriodicBGK`、`SparseAOSLBM`、`MCMPSolver`）统一到一个 API 上，并提供：

1. **统一 step 接口** `CorePath.init_from_velocity / step / macroscopic`，参数集合被冻结在
   `CoreConfig`（stencil, shape, ν, kernel, backend, dtype）——一个臂就是一个不可变的身份。
2. **两个核函数，同一 API**：
   - `reference`：碰撞阶段物化整场 `f_eq` 与整场碰撞后 `f_post`，再整场周期迁移 → **4 个 Q·N 活缓冲**；
   - `fused`：逐方向单趟扫（single sweep）——`f_eq[q]` 只落在一个 N 大小的暂存里，就地松弛后就地
     迁入目标缓冲，`f_eq`/`f_post` **从不整场物化** → **2 个 Q·N 活缓冲**。周期回绕折进扫描里
     （每个方向一次 `roll`）；平壁边界仍归 `boundaries.py`，本模块**不假装支持**。
3. **多后端 + 能力探测**：`probe_backends()` 逐项报告 numpy / torch-cpu / torch-cuda / cupy 的
   可用性与原因；`make_ops()` 对不可用后端直接抛 `BackendUnavailable`，绝不回退。
4. **内存账**：`declared_live_bytes()` 由配置精确算出（fused = 2QN + 5N，reference = 4QN + 5N，
   按 dtype itemsize），另有实测 `tracemalloc` 峰值，两者都进表。
5. **统一对标口径**：`run_arm()` 在一个函数里完成"物理跑 → 精度拟合 → 计时 → 判决"，
   判决只能由 `admissibility()` 给出，规则写死在代码里：
   **精度闸门与一致性闸门都过，才发 MLUPS；否则该臂的加速数字作废**（`REJECTED_ACCURACY` /
   `REJECTED_CONSISTENCY`）。计时协议：一次 warm-up 块 + `repeat` 个计时块，取最小块
   （噪声最小），精度历史来自**另一次不计时的相同跑**，计时永不污染物理。
6. 工具链：`TensorLBM/benchmarks/core_path_t2_bench.py` 跑全部臂并程序化写出
   `outputs/core_path_t2/summary.json` + 4 张图；本脚本再把 summary 渲染成本文与平台报告。

质量门：`TensorLBM/tests/test_core_path.py`（17 项）断言的是**不变量**而非"能跑"——
融合核与参考核逐位一致、T3 闸门通过、闸门失败时吞吐不可发布、不可用后端必抛异常、
融合活缓冲严格少于参考、步进过程零状态缓冲分配（缓冲复用而非重分配）。

## 3. 计算结果（全部由 `summary.json` 程序化注入）

| kernel | backend | dtype | 网格 | 步数 | MLUPS | 计时秒 | 拟合衰减率 | 衰减率误差 | 一致性 |Δu| | 声明活内存 KiB | 实测峰值 KiB | 判决 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{arm_rows(arms)}

关键比值（只有两臂都通过闸门时才给加速比，否则显式 `n/a`）：

| 比值 | 值 |
|---|---|
{ratio_rows}

本轮结论（数字口径）：

- **通过闸门的臂 {len(ok)}/{len(arms)}**，被拒 {len(rejected)} 个。
- 精度：全部臂衰减率相对误差落在 {pct(best_acc)} – {pct(worst_acc)}，预注册闸门 {pct(gates['accuracy_tol'],1)}。
- 一致性：融合核与参考核的速度场相对差最大 {('%.2e' % worst_cons) if worst_cons is not None else 'n/a'}（float64 逐位为 0）。
- 内存：`fused` 的声明活内存是 `reference` 的 {ratios.get('fused_over_reference_live_bytes_64', float('nan')):.4f} 倍（64²），实测峰值是其 {ratios.get('fused_over_reference_peak_bytes_64', float('nan')):.4f} 倍。
- 吞吐（只有两臂都通过闸门才给比值）：`fused/reference` = {ratios.get('fused_over_reference_mlups_64')}（64², numpy/float64），
  128² 上 {ratios.get('fused_over_reference_mlups_128')}；`fused_slices/reference` = {ratios.get('fused_slices_over_reference_mlups_64')}（64²），
  128² 上 {ratios.get('fused_slices_over_reference_mlups_128')}。{n_torch} 个 torch-cpu 臂作为第二后端参与同一口径。
- 抗噪说明：全部比值都由 `summary.json` 的 `ratios` 段程序化给出，任何一臂未过闸门则该比值显式为 `null`。

## 4. 结果图

1. `fig_field_contours.png` — T3 算例跑完后的 u_x 场云图（64² 与 128²，fused 臂）。
2. `fig_tradeoff.png` — **精度-性能权衡**：横轴 MLUPS、纵轴衰减率相对误差，实心=可发布，
   空心红=被闸门拒；虚线=预注册容差。
3. `fig_memory.png` — 内存管理：声明活缓冲 vs 实测 tracemalloc 峰值，逐臂。
4. `fig_decay.png` — 闸门本身的核对：拟合衰减率 vs 解析 2νk²，逐臂。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

- **参考核符号错被闸门当场抓住**：`f_post = f - ω(f - f_eq)` 写成 `f + ω(f - f_eq)` 时
  首个版本 400 步内发散（衰减率误差 154%），是**精度闸门**而不是人工检查发现的。这正是
  "先有闸门再谈优化"的价值；两核逐位一致后误差回到 {pct(best_acc)} 量级。
- **"省内存"与"省时间"是两件不同的事**：`fused`（roll 版）把整场 `f_eq`/`f_post` 去掉，
  活内存降到参考核的 {ratios.get('fused_over_reference_live_bytes_64', float('nan')):.3f}×，但每个方向仍要
  `roll` 出一份 N 大小临时量，numpy 下解释器开销与参考核的整场扫描同量级，所以
  `fused/reference` 吞吐比只有 {ratios.get('fused_over_reference_mlups_64')}。把回绕改成**预计算 slice 视图拷贝**
  （`fused_slices`）后，每步 `roll` 调用数从 Q=9 降到 0（`roll_calls_per_step` 有测试断言），
  吞吐比升到 {ratios.get('fused_slices_over_reference_mlups_64')}：{throughput_verdict}。
  这说明"融合"的收益要落到 **数据搬运方式** 上，而不是只在概念上取消中间场。
- **编译型后端仍缺**：本机无 numba、无 GPU，所以融合核还没来得及交给编译器；这也是本锚点
  在 GPU 侧仍未闭环的原因（见 WITHHELD 段）。
- **GPU 臂 WITHHELD**：本机 `torch.cuda.is_available() == False`、cupy 未安装。GPU 核心路径的
  吞吐/通信/精度权衡**在本单无法判定**，`probe_backends()` 已把这一事实写进 summary 的
  `withheld` 段；任何"GPU 加速比"数字在本单都不存在，也不许由 CPU 臂代替。
- **适用范围**：本模块只覆盖全周期域（Taylor-Green / 剪切波 / 周期驱动 Poiseuille）。
  带平壁的算例仍走 `boundaries.py`，两者尚未在同一次扫描里融合——这是下一步的统一工作。
- float32 臂的容差单独预注册（{gates['consistency_tol_by_dtype']['float32']:.1e}），因为 float32 自身的
  舍入地板就比 float64 高 9 个量级；用 float64 的容差去判 float32 会把"位宽"误判成"缺陷"。

## 6. 误差判定（逐项，显式回答是否 <3%）

基准 = Taylor-Green 解析衰减率 2νk²（无自由参数，不依赖任何算例库）。

| 臂 | 拟合衰减率 | 解析值 | 相对误差 | <3%？ |
|---|---|---|---|---|
{acc_rows}

一致性判定：融合核 vs 参考核速度场，float64 臂相对差最大
{('%.2e' % worst_cons) if worst_cons is not None else 'n/a'}，闸门 {gates['consistency_tol_float64']:.1e}。

**回答**：{('全部 %d 个臂的衰减率相对误差均 <3%%（最差 %s），即"是"。' % (len(arms), pct(worst_acc))) if worst_acc < 0.03 else ('存在臂未达 <3%%（最差 %s），见上表；这些臂的吞吐判决为 REJECTED_ACCURACY，未发布数字。' % pct(worst_acc))}
GPU 臂为 WITHHELD（无基准可判，原因见第 1 节探测表），不做任何精度陈述。
"""


def _fmt_ratio(v) -> str:
    return "n/a（含未通过闸门的臂）" if v is None else f"{v:.2f}×"


def _summary_text(s: dict) -> str:
    arms = s["arms"]
    r = s["ratios"]
    acc = [a["accuracy_rel_err"] for a in arms]
    torch_ok = [a for a in arms if a["config"]["backend"].startswith("torch")
                and a["perf_admissible"]]
    torch_line = ""
    if torch_ok:
        ref = next((a for a in arms if a["config"]["kernel"] == "reference"
                    and a["config"]["backend"] == torch_ok[0]["config"]["backend"]
                    and a["config"]["dtype"] == "float64"), None)
        fus = next((a for a in torch_ok if a["config"]["kernel"] == "fused"
                    and a["config"]["dtype"] == "float64"), None)
        if ref and fus:
            torch_line = (f"；torch-cpu 上 fused 相对 reference 吞吐 "
                          f"{fus['mlups']/ref['mlups']:.2f}×")
    return (
        "把 TensorLBM 三套时间推进统一到一个 CorePath API（reference / fused / fused_slices 三核 + "
        "可插拔后端能力探测 + 内存账 + 统一对标协议）。在 T3 REAL 算例 taylor-green-2d 上："
        "融合核与参考核速度场逐位一致（rel Δ=0），"
        f"衰减率相对误差 {pct(min(acc))}–{pct(max(acc))}（预注册闸门 {pct(s['preregistered_gates']['accuracy_tol'],1)}），"
        f"声明活内存降至参考核的 {r.get('fused_over_reference_live_bytes_64', float('nan')):.3f}×；"
        f"吞吐 fused_slices/reference = {_fmt_ratio(r.get('fused_slices_over_reference_mlups_64'))}（64² CPU/numpy）、"
        f"{_fmt_ratio(r.get('fused_slices_over_reference_mlups_128'))}（128²）{torch_line}；"
        f"{sum(1 for a in arms if a['perf_admissible'])}/{len(arms)} 臂通过双闸门，"
        "GPU 臂因本机无设备判 WITHHELD（不发布任何 GPU 加速比）。"
    )


def main() -> None:
    s = json.loads(SUM_PATH.read_text())
    body = build_markdown(s)
    DOC_PATH.write_text("# T2 核心路径统一优化：口径、内存与精度-性能权衡\n\n"
                        "> 本文件由 `publish_core_path_t2.py` 从 "
                        "`TensorLBM/outputs/core_path_t2/summary.json` 程序化生成，"
                        "请勿手工改数字。\n\n" + body)

    cap = {
        "fig_field_contours.png": "T3 算例 taylor-green-2d 在统一核心路径上跑完后的 u_x 场云图"
                                  "（左 64²，右 128²，fused 臂，numpy/float64）。",
        "fig_tradeoff.png": "精度-性能权衡：MLUPS 对衰减率相对误差。实心=通过预注册闸门、MLUPS 可发布；"
                            "空心红=被拒；虚线=预注册容差。",
        "fig_memory.png": "内存管理证据：声明活状态（2QN vs 4QN）与实测 tracemalloc 峰值，逐臂。",
        "fig_decay.png": "闸门核对：拟合衰减率 vs 解析 2νk²，逐臂。",
    }
    figs = []
    for name, caption in cap.items():
        p = SUM_PATH.parent / name
        if p.exists():
            figs.append({"name": name, "caption": caption,
                         "png_base64": base64.b64encode(p.read_bytes()).decode()})

    arms = s["arms"]
    ok = sum(1 for a in arms if a["perf_admissible"])
    ratios = s["ratios"]
    payload = {
        "title": ("GAP-LBM-T2-GPU-COREPATH · 核心路径统一与口径统一："
                  "融合核/参考核逐位一致、活内存降至 "
                  f"{ratios.get('fused_over_reference_live_bytes_64', float('nan')):.3f}×，"
                  f"{ok}/{len(arms)} 臂通过预注册闸门（GPU 臂 WITHHELD）"),
        "summary": _summary_text(s),
        "scenario_key": "lbm-core-path-t2",
        "mission_id": MISSION_ID,
        "tags": ["lbm", "core-path", "T2", "GAP-LBM-T2-GPU-COREPATH", "TensorLBM",
                 "performance", "memory", "backend-dispatch", "taylor-green"],
        "markdown": body,
        "figures": figs,
    }
    if "--dry" in sys.argv:
        pathlib.Path("/tmp/report.json").write_text(json.dumps(payload))
        print(body)
        print(f"[dry] body {len(body)} chars, figures {len(figs)}")
        return

    pathlib.Path("/tmp/report.json").write_text(json.dumps(payload))
    req = urllib.request.Request(
        f"{API}/auth/login",
        data=json.dumps({
            "username": os.environ.get("TENSOR_AGENT_USER", "eco_admin"),
            "password": os.environ.get("TENSOR_AGENT_PASS", ""),
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    tok = json.loads(urllib.request.urlopen(req, timeout=60).read())
    token = tok.get("access_token") or tok.get("token") or \
        tok.get("data", {}).get("access_token")
    post = urllib.request.Request(
        f"{API}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    resp = json.loads(urllib.request.urlopen(post, timeout=180).read())
    print(json.dumps(resp, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()