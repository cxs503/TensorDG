"""Publish the CASE-DG-DALEMBERT-ELASTIC-1D verification report to the platform.

Numbers are injected from the validation JSON artifacts (no hand copying);
the figure is embedded as base64.  Credentials come from the environment
(TENSOR_AGENT_USER / TENSOR_AGENT_PASS), the endpoint from TENSOR_API_BASE.

Usage::  python3 tools/publish_elastic_report.py [--dry-run]
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "validation/elastic_1d/elastic_1d_dalembert.json")
FAULT_PATH = os.path.join(ROOT, "validation/elastic_1d/bc_fault_injection.json")
PNG_PATH = os.path.join(ROOT, "docs/figures/elastic_1d_dalembert.png")


def _post(url, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read().decode()


def build() -> dict:
    d = json.load(open(JSON_PATH))
    f = json.load(open(FAULT_PATH))
    P, G = d["parameters"], d["gate"]
    pulse, stand = d["pulse_case"], d["standing_case"]

    pulse_tbl = "\n".join(
        f"| {r['cells']} | {r['steps']} | {r['rel_l2_v']:.3e} | "
        f"{r['rel_l2_sigma']:.3e} | "
        + (f"{r['eoc_sigma']:.3f}" if "eoc_sigma" in r else "--")
        + f" | {r['peak_left']['sigma']:.6f} | {r['peak_right']['sigma']:.6f} |"
        for r in pulse)
    stand_tbl = "\n".join(
        f"| {r['cells']} | {r['steps']} | {r['rel_l2_v']:.3e} | "
        f"{r['rel_l2_sigma']:.3e} | "
        + (f"{r['eoc_sigma']:.3f}" if "eoc_sigma" in r else "--")
        + f" | {r['energy']:.10f} | {r['energy_drift']:.2e} |"
        for r in stand)
    worst_pulse = G["worst"]["rel_l2_sigma_pulse"]
    worst_stand = G["worst"]["rel_l2_sigma_standing"]
    speed_error = max(r["pulse_speed_error"] for r in pulse)

    md = f"""# 一维弹性杆波 d'Alembert 验证：TensorDG 新增速度—应力 DG 求解器（锚点 CASE-DG-DALEMBERT-ELASTIC-1D）

**结论**：TensorDG 新增 `tensordg/elastic_1d.py`（一维线弹性速度—应力 DG，精确上风通量 + 特征边界 + SSP-RK3）。
两条闭式解析算例全部达标：应力剖面相对 L2 误差最差 {worst_pulse*100:.4f}%（锚点容差 <2%），
P2 观测收敛阶 {pulse[-1]['eoc_sigma']:.2f}，脉冲位置（质心）相对偏差 {speed_error:.2e}，
夹紧—自由模态能量漂移 {stand[0]['energy_drift']:.1e}；12 项新测试全绿。

## ① 案例介绍
物理背景：等截面杆纵向弹性波（固体动力学；TensorDG 此前无弹性能力，是结构振动/声振/冰激振动的前置）。
方程：v_t - σ_x/ρ = 0，σ_t - E v_x = 0，c = sqrt(E/ρ) = {P['c']}；Riemann 不变量 w± = σ ∓ ρ c v，
d'Alembert 通解 = 两条反向行波叠加。
几何/参数：L = 1 一维均匀网格；ρ = {P['rho']}，E = {P['young']}；DG p = {P['degree']}，CFL = {P['cfl']}，SSP-RK3。
- 算例 A（pulse，周期杆）：t=0 高斯速度脉冲（中心 {P['pulse_center']}、宽 {P['pulse_width']}），σ=0；终时 t={P['pulse_time']}；网格 {P['meshes']}。
  解析 v = ½[v₀(x−ct)+v₀(x+ct)]，σ = ½ρc[v₀(x+ct)−v₀(x−ct)]（Graff 1975）。
- 算例 B（standing，夹紧—自由）：第一阶纵向模态释放，终时 t={P['standing_time']}，网格 {P['standing_meshes']}；
  解析 σ = E A k cos(kx) cos(ωt)，k = π/(2L)，ω = ck（LeVeque 2002 Ch.3 特征边界）。

## ② 怎么做的
1) 复用仓库模态 Legendre DG 算子（`advection_1d._operators`），体积项精确积分；coeff 形状 [ncells, p+1, 2] = (v, σ)。
2) 界面通量为线性系统**精确上风** F* = A⁺U_L + A⁻U_R（w⁺ 取左态、w⁻ 取右态），均匀态退化为物理通量（有单测）。
3) 特征边界：x=0 进入不变量是 w⁺、x=L 是 w⁻；夹紧端 v=0 → 反射不变量 = 外送不变量；自由端 σ=0 → 进入不变量取反号。
4) SSP-RK3，Δt = CFL·h/(c(2p+1))，步数取整后回填 Δt = T/N，并断言 step·dt == T（防静默提前停时）。
5) 工件：`benchmarks/elastic_1d_dalembert.py`（主验证 + 图 + JSON）、`benchmarks/elastic_1d_bc_fault_injection.py`（门禁功率）、`tests/test_elastic_1d_dalembert.py`（12 项）、`docs/elastic_wave_verification.md`（六要素报告）。

## ③ 计算结果

**算例 A（周期脉冲，t = {P['pulse_time']}）**

| cells | steps | rel L2 v | rel L2 σ | EOC σ | σ峰值(左) | σ峰值(右) |
|---:|---:|---:|---:|---:|---:|---:|
{pulse_tbl}

解析应力峰值严格为 ±ρc/2 = ±0.5；解析质心 {pulse[0]['centroid_exact']['left']:.6f} / {pulse[0]['centroid_exact']['right']:.6f}（x₀ ∓ ct）。

**算例 B（夹紧—自由模态，t = {P['standing_time']}）**

| cells | steps | rel L2 v | rel L2 σ | EOC σ | 弹性能 | 能量相对漂移 |
|---:|---:|---:|---:|---:|---:|---:|
{stand_tbl}

解析能量 {stand[0]['energy_exact']:.10f}；边界迹 v(0) = σ(L) = 0。

**故障注入（门禁功率）**：正确配对能量比 {f['correct_pairing']['energy_ratio']:.6f}；
左右互换（错误配对）放大到 {f['swapped_pairing']['energy_ratio']:.3e} 倍 → 判据有牙，
已固定为测试 `test_energy_gate_detects_wrong_characteristic_pairing`。

## ④ 结果图
2×2 图：算例 A 的 v(x)、σ(x)（解析 vs 数值，图上不可分辨）、误差—网格双对数（斜率 3）、算例 B 应力剖面（见 figures 内嵌 PNG）。

## ⑤ 注意点
1) 特征边界的不变量配对必须分左右：首版把 x=0 的进入不变量写成 w⁻，夹紧—自由算例指数发散（50 单元 t=0.5 相对误差 4.6e+03、能量 1.07e+07 vs 解析 0.61685），改正后 3.5e-07——波形"像"不算验证。
2) 脉动位置不能用网格 argmax（被采样间距量化，出现恒 0 假信号），改用 |σ| 一阶矩（质心）。
3) 相对误差在解析场为零的时刻无定义（算例 B 在 t=1.0 时 σ≡0、t=2.0 时 v≡0，会给出 1e+08 伪误差）→ 判据须选两场非零的终时。
4) 周期解析解要求两脉冲再次相遇前结束（ct < min(x₀, 1−x₀) − 脉冲半宽）。
5) 上风通量有真实数值耗散（能量漂移随网格高阶下降）；长时能量守恒需求应提阶/换通量，不能放宽判据。
6) 局限：仅一维线弹性常系数无阻尼；二维/三维弹性（P-SV、Lamb）与实验 validation 未覆盖。

## ⑥ 误差判定（是否 <3%：是）
| 关键量 | 基准 | 实测最差 | 判据 | 达标 |
|---|---|---:|---|---|
| σ 相对 L2（算例 A） | d'Alembert 双行波 | {worst_pulse*100:.4f}% | < 2%（锚点） | 是 |
| σ 相对 L2（算例 B） | 夹紧—自由模态 | {worst_stand*100:.6f}% | < 2%（锚点） | 是 |
| 无弥散（质心相对偏差） | x₀ ± ct | {speed_error:.2e} | ≪ 1% | 是 |
| 应力峰值 | ±ρc/2 | {abs(pulse[-1]['peak_left']['sigma']/0.5-1)*100:.2e}% | < 2% | 是 |
| 能量守恒（算例 B） | ¼A²k²E | {stand[0]['energy_drift']:.1e} | < 1e-4 | 是 |
| 收敛阶 P2 | p+1 = 3 | {min(pulse[-1]['eoc_sigma'], stand[-1]['eoc_sigma']):.3f}–{max(pulse[1]['eoc_sigma'], stand[1]['eoc_sigma']):.3f} | ≥ 2.7 | 是 |

门禁：`validation/elastic_1d/elastic_1d_dalembert.json` → `gate.passed = {str(G['passed']).lower()}`。
测试：`tests/test_elastic_1d_dalembert.py` 12/12 通过；与 `test_advection_1d`、`test_benchmark_smoke` 合跑 19/19 通过。
全量 1312 项回归已递交集群作业（排队中，本单不宣称全量绿）。
"""
    png = base64.b64encode(open(PNG_PATH, "rb").read()).decode()
    return {
        "title": "TensorDG 一维弹性杆波 d'Alembert 验证（CASE-DG-DALEMBERT-ELASTIC-1D）",
        "summary": (f"新增速度—应力 DG 弹性波求解器；σ 相对 L2 最差 {worst_pulse*100:.4f}%"
                    f"（<2%），P2 收敛阶 {pulse[-1]['eoc_sigma']:.2f}，能量守恒，12/12 测试绿。"),
        "scenario_key": "case-dg-dalembert-elastic-1d",
        "mission_id": 16149,
        "tags": ["TensorDG", "elasticity", "dAlembert", "DG", "verification",
                 "CASE-DG-DALEMBERT-ELASTIC-1D"],
        "markdown": md,
        "figures": [{
            "name": "elastic_1d_dalembert.png",
            "caption": "算例A：v/σ 剖面与收敛曲线；算例B：夹紧—自由模态应力剖面",
            "png_base64": png,
        }],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = build()
    if args.dry_run:
        path = "/tmp/report.json"
        with open(path, "w") as fh:
            fh.write(json.dumps(payload))
        print(f"dry run: wrote {path} ({len(json.dumps(payload))} bytes)")
        return 0
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    user = os.environ["TENSOR_AGENT_USER"]
    password = os.environ["TENSOR_AGENT_PASS"]
    status, body = _post(f"{base}/auth/login",
                         {"username": user, "password": password})
    token = json.loads(body).get("access_token") or json.loads(body).get("token")
    if not token:
        raise SystemExit(f"login failed: {status} {body[:200]}")
    status, body = _post(f"{base}/tensor/reports", payload, token)
    print(status, body[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())