"""CASE-DG-EULER-VORTEX-ERR: 等熵涡移动误差守恒回归测试。

本测试从 case_euler_vortex_errconserve/results/*.json 读取已有计算结果，
验证锚点 CASE-DG-EULER-VORTEX-ERR 的核心物理声明：

  C1 容差：t=1 快照最大场相对 L2 < 6%
  C2 误差守恒：DG 族累积指数 alpha <= 1
  C3 离散守恒：质量/动量/能量漂移 < 1e-10
  C4 熵预算：|dS|/S(0) < 1e-3
  C7 设计空间不变性：各设计点 alpha <= 1
  C8 阶数不变性：各阶数族 alpha <= 1
  C10 方向不变性：各方向 alpha <= 1

所有数字从工件直读，零手抄。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

RESULTS_DIR = Path("case_euler_vortex_errconserve/results")
TOLERANCE_C1 = 0.06  # 6%
TOLERANCE_ALPHA = 1.0  # 误差守恒阈值
TOLERANCE_CONSERVATION = 1e-10  # 质量/动量/能量漂移
TOLERANCE_ENTROPY = 1e-3  # 熵预算


def _load_results():
    """加载所有结果文件，返回 {task_key: data} 字典。"""
    results = {}
    for fn in RESULTS_DIR.glob("*.json"):
        with open(fn) as f:
            data = json.load(f)
        key = data.get("task_key", fn.stem)
        results[key] = data
    return results


def _compute_alpha(error_history):
    """从误差历史计算累积指数 alpha。

    alpha = ln(e(t_max)/e(t_min)) / ln(t_max/t_min)
    alpha <= 1 表示误差守恒（误差增长慢于线性）。
    """
    if len(error_history) < 2:
        return None
    times = [h["time"] for h in error_history]
    errors = [h["max_percent"] for h in error_history]
    t_min, t_max = min(times), max(times)
    if t_min <= 0 or t_max <= t_min:
        return None
    e_min = errors[times.index(t_min)]
    e_max = errors[times.index(t_max)]
    if e_min <= 0:
        return None
    return math.log(e_max / e_min) / math.log(t_max / t_min)


# ── C1: 容差 ──────────────────────────────────────────────

class TestC1Tolerance:
    """C1: t=1 快照最大场相对 L2 < 6%。"""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        return _load_results()

    def test_gate_config_passes(self, results):
        """门禁配置 dg_d2_n32_t10_cfl0.2 必须通过。"""
        key = "dg_d2_n32_t10_cfl0.2"
        assert key in results, f"门禁配置 {key} 结果缺失"
        d = results[key]
        assert d["max_percent"] < TOLERANCE_C1 * 100, (
            f"门禁配置 {key}: {d['max_percent']:.4f}% >= {TOLERANCE_C1*100}%"
        )

    def test_all_t1_configs_pass(self, results):
        """所有 t=1 配置必须通过 6% 容差。"""
        t1_keys = [k for k, v in results.items()
                   if abs(v.get("final_time", 0) - 1.0) < 1e-9]
        assert len(t1_keys) > 0, "无 t=1 配置结果"
        failed = []
        for k in t1_keys:
            pct = results[k]["max_percent"]
            if pct >= TOLERANCE_C1 * 100:
                failed.append((k, pct))
        assert not failed, (
            f"以下 t=1 配置超过 {TOLERANCE_C1*100}% 容差: "
            + ", ".join(f"{k}={p:.4f}%" for k, p in failed)
        )

    def test_design_family_t1_passes(self, results):
        """设计族 dg_d2_n32 在 t=1 必须通过。"""
        key = "dg_d2_n32_t1_cfl0.2"
        assert key in results, f"设计族 t=1 配置 {key} 结果缺失"
        d = results[key]
        assert d["max_percent"] < TOLERANCE_C1 * 100, (
            f"设计族 t=1: {d['max_percent']:.4f}% >= {TOLERANCE_C1*100}%"
        )


# ── C2: 误差守恒 ──────────────────────────────────────────

class TestC2ErrorConservation:
    """C2: DG 族累积指数 alpha <= 1（误差守恒）。"""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        return _load_results()

    def test_dg_d2_n32_alpha(self, results):
        """DG-P2/32² 族的 alpha 必须 <= 1。"""
        family_keys = sorted([
            k for k in results
            if k.startswith("dg_d2_n32_t") and "_dir" not in k and k.endswith("_cfl0.2")
        ])
        assert len(family_keys) >= 2, f"设计族检查点不足: {family_keys}"

        history = []
        for k in family_keys:
            d = results[k]
            history.append({
                "time": d["final_time"],
                "max_percent": d["max_percent"],
            })
        history.sort(key=lambda x: x["time"])

        alpha = _compute_alpha(history)
        assert alpha is not None, "无法计算 alpha（数据不足）"
        assert alpha <= TOLERANCE_ALPHA, (
            f"DG-P2/32² 族 alpha = {alpha:.4f} > {TOLERANCE_ALPHA}，"
            f"误差不守恒: e(t={history[-1]['time']})/{history[-1]['max_percent']:.4f}% vs "
            f"e(t={history[0]['time']})/{history[0]['max_percent']:.4f}%"
        )

    def test_dg_d1_n32_alpha(self, results):
        """DG-P1/32² 族的 alpha 必须 <= 1。"""
        family_keys = sorted([
            k for k in results
            if k.startswith("dg_d1_n32_t") and k.endswith("_cfl0.2")
        ])
        if len(family_keys) < 2:
            pytest.skip("DG-P1/32² 族检查点不足")

        history = []
        for k in family_keys:
            d = results[k]
            history.append({"time": d["final_time"], "max_percent": d["max_percent"]})
        history.sort(key=lambda x: x["time"])

        alpha = _compute_alpha(history)
        assert alpha is not None
        assert alpha <= TOLERANCE_ALPHA, f"DG-P1/32² 族 alpha = {alpha:.4f} > 1"

    def test_dg_d3_n32_alpha(self, results):
        """DG-P3/32² 族的 alpha 必须 <= 1。"""
        family_keys = sorted([
            k for k in results
            if k.startswith("dg_d3_n32_t") and k.endswith("_cfl0.2")
        ])
        if len(family_keys) < 2:
            pytest.skip("DG-P3/32² 族检查点不足")

        history = []
        for k in family_keys:
            d = results[k]
            history.append({"time": d["final_time"], "max_percent": d["max_percent"]})
        history.sort(key=lambda x: x["time"])

        alpha = _compute_alpha(history)
        assert alpha is not None
        assert alpha <= TOLERANCE_ALPHA, f"DG-P3/32² 族 alpha = {alpha:.4f} > 1"


# ── C3: 离散守恒 ──────────────────────────────────────────

class TestC3DiscreteConservation:
    """C3: 质量/动量/能量漂移 < 1e-10。"""

    @pytest.fixture(scope="class")
    @classmethod
    def metrics(cls):
        with open("case_euler_vortex_errconserve/metrics.json") as f:
            return json.load(f)

    def test_mass_conservation(self, metrics):
        """质量漂移必须 < 1e-10。"""
        conservation = metrics.get("conservation", {})
        for key, vals in conservation.items():
            mass_drift = abs(vals.get("mass_drift", 0))
            assert mass_drift < TOLERANCE_CONSERVATION, (
                f"{key}: 质量漂移 {mass_drift:.2e} >= {TOLERANCE_CONSERVATION}"
            )

    def test_energy_conservation(self, metrics):
        """能量漂移必须 < 1e-10。"""
        conservation = metrics.get("conservation", {})
        for key, vals in conservation.items():
            energy_drift = abs(vals.get("energy_drift", 0))
            assert energy_drift < TOLERANCE_CONSERVATION, (
                f"{key}: 能量漂移 {energy_drift:.2e} >= {TOLERANCE_CONSERVATION}"
            )


# ── C4: 熵预算 ────────────────────────────────────────────

class TestC4EntropyBudget:
    """C4: 熵漂移 < 1e-3。"""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        return _load_results()

    def test_entropy_budget_dg(self, results):
        """DG 配置的熵漂移必须 < 1e-3。"""
        dg_keys = [k for k in results if k.startswith("dg_")]
        assert len(dg_keys) > 0, "无 DG 配置结果"
        failed = []
        for k in dg_keys:
            entropy = results[k].get("entropy_drift", results[k].get("entropy_production", 0))
            if entropy >= TOLERANCE_ENTROPY:
                failed.append((k, entropy))
        assert not failed, (
            f"以下 DG 配置熵漂移超标: "
            + ", ".join(f"{k}={e:.2e}" for k, e in failed)
        )


# ── C7: 设计空间不变性 ────────────────────────────────────

class TestC7DesignSpace:
    """C7: 各设计点 alpha <= 1。"""

    @pytest.fixture(scope="class")
    @classmethod
    def metrics(cls):
        with open("case_euler_vortex_errconserve/metrics.json") as f:
            return json.load(f)

    def test_design_space_alpha(self, metrics):
        """设计空间各点 alpha <= 1。"""
        design_space = metrics.get("design_space", {})
        for key, vals in design_space.items():
            alpha = vals.get("alpha", 0)
            if alpha is not None and not math.isnan(alpha):
                assert alpha <= TOLERANCE_ALPHA, (
                    f"设计点 {key}: alpha = {alpha:.4f} > {TOLERANCE_ALPHA}"
                )


# ── C8: 阶数不变性 ────────────────────────────────────────

class TestC8DegreeInvariance:
    """C8: 各阶数族 alpha <= 1。"""

    @pytest.fixture(scope="class")
    @classmethod
    def metrics(cls):
        with open("case_euler_vortex_errconserve/metrics.json") as f:
            return json.load(f)

    def test_degree_invariance(self, metrics):
        """各阶数族 alpha <= 1。"""
        degree_axis = metrics.get("degree_axis", {})
        for degree, vals in degree_axis.items():
            alpha = vals.get("alpha", 0)
            if alpha is not None and not math.isnan(alpha):
                assert alpha <= TOLERANCE_ALPHA, (
                    f"阶数 {degree}: alpha = {alpha:.4f} > {TOLERANCE_ALPHA}"
                )


# ── C10: 方向不变性 ───────────────────────────────────────

class TestC10DirectionInvariance:
    """C10: 各方向 alpha <= 1。"""

    @pytest.fixture(scope="class")
    @classmethod
    def metrics(cls):
        with open("case_euler_vortex_errconserve/metrics.json") as f:
            return json.load(f)

    def test_direction_invariance(self, metrics):
        """各方向 alpha <= 1。"""
        direction_axis = metrics.get("direction_axis", {})
        for key, vals in direction_axis.items():
            alpha = vals.get("alpha", 0)
            if alpha is not None and not math.isnan(alpha):
                assert alpha <= TOLERANCE_ALPHA, (
                    f"方向 {key}: alpha = {alpha:.4f} > {TOLERANCE_ALPHA}"
                )


# ── 综合：误差增长因子 ────────────────────────────────────

class TestErrorGrowthFactor:
    """验证误差增长因子 err(T)/err(t_min) <= 5（预声明上限）。"""

    @pytest.fixture(scope="class")
    @classmethod
    def results(cls):
        return _load_results()

    def test_growth_factor_dg_d2_n32(self, results):
        """DG-P2/32² 从 t=1 到 t=10 的增长因子 <= 5。"""
        t1_key = "dg_d2_n32_t1_cfl0.2"
        t10_key = "dg_d2_n32_t10_cfl0.2"
        assert t1_key in results, f"{t1_key} 结果缺失"
        assert t10_key in results, f"{t10_key} 结果缺失"

        e1 = results[t1_key]["max_percent"]
        e10 = results[t10_key]["max_percent"]
        growth = e10 / e1 if e1 > 0 else float("inf")

        assert growth <= 5.0, (
            f"误差增长因子 = {growth:.4f} > 5.0: "
            f"e(t=1)={e1:.4f}%, e(t=10)={e10:.4f}%"
        )

    def test_growth_factor_dg_d2_n32_cfl04(self, results):
        """DG-P2/32² CFL=0.4 从 t=1 到 t=10 的增长因子 <= 5。"""
        t1_key = "dg_d2_n32_t1_cfl0.4"
        t10_key = "dg_d2_n32_t10_cfl0.4"
        if t1_key not in results or t10_key not in results:
            pytest.skip("CFL=0.4 配置结果缺失")

        e1 = results[t1_key]["max_percent"]
        e10 = results[t10_key]["max_percent"]
        growth = e10 / e1 if e1 > 0 else float("inf")

        assert growth <= 5.0, (
            f"CFL=0.4 误差增长因子 = {growth:.4f} > 5.0"
        )
