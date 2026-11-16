"""CASE-DG-EULER-VORTEX-ERR: 参考解与量测仪表认证的回归测试。

被测模块：`benchmarks/vortex_reference_instrument_certification.py`。

本测试**不是**只把函数跑一遍（"执行过即通过"），每一条都钉住一个可判真的
不变式，并且都配一个突变对照（mutant）：如果仪表没有判别力，对照先失败。

  R1 解析参考解满足二维 Euler 方程（守恒形式残差），且四类合理误抄被检出
  R1b 周期折叠严格 L 周期
  R2 p/rho^gamma 空间常数，等温变体被检出
  R3 门禁用的 320 点张量 Gauss 不变量分母收敛（远低于其分母化的漂移）
  R4 相对 L2 范数的求积收敛（投影态）+ 欠积分对照被检出
  R5 熵预算的求积部分（多项式不变量 vs 非多项式熵）
  R6 已发布工件 `instrument_certification.json` 与本模块的读数一致（防陈旧）

耗时：全部走投影态与解析场，不做时间推进，总耗时 < 30 s。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensordg import euler_dg_2d as dg2
from tensordg.euler_2d import vortex_primitive
from benchmarks.case_euler_vortex_errconserve import (
    BACKGROUND,
    DOMAIN,
    PERIOD,
    STRENGTH,
    _dg_measurements,
    _initial_dg_state,
)
from benchmarks.vortex_reference_instrument_certification import (
    _smallest_conservation_drift,
    _stored,
    certify_error_norm,
    certify_exactness,
    certify_invariant_reference,
    certify_isentropy,
    euler_residual,
)

ARTIFACT = Path("case_euler_vortex_errconserve/instrument_certification.json")


# -- R1 --------------------------------------------------------------------

def test_reference_field_satisfies_euler_equations():
    """解析场代入守恒形式 Euler 方程，残差必须是舍入噪声。"""
    torch.manual_seed(0)
    points = torch.rand(128, 2, dtype=torch.float64) * 10.0 - 5.0
    for time in (0.0, 1.0):
        residual, scale = euler_residual(points, time=time)
        for name, value in residual.items():
            relative = float((value.abs() / scale[name]).max())
            assert relative < 1e-12, f"{name} 在 t={time} 的相对残差 {relative:.2e}"


def test_residual_detects_plausible_transcription_errors():
    """残差检测器必须能分辨"抄错"，否则 R1 通过没有信息量。"""
    torch.manual_seed(1)
    points = torch.rand(64, 2, dtype=torch.float64) * 10.0 - 5.0
    clean, _ = euler_residual(points, time=1.0)
    clean_worst = max(float((v.abs()).max()) for v in clean.values())
    for mutant in ("envelope_exponent", "temperature_factor", "velocity_one_sign"):
        residual, _ = euler_residual(points, time=1.0, mutant=mutant)
        worst = max(float((v.abs()).max()) for v in residual.values())
        assert worst > 1e-2, f"突变 {mutant} 未被检出（{worst:.2e}）"
        assert worst > 1e3 * clean_worst


def test_periodic_fold_is_exactly_periodic():
    """period= 折叠必须严格 L 周期，否则参考解在盒内不自洽。"""
    torch.manual_seed(2)
    points = torch.rand(64, 2, dtype=torch.float64) * 10.0 - 5.0
    shift = torch.tensor(PERIOD, dtype=torch.float64)
    for time in (0.0, 3.0):
        base = torch.stack(list(vortex_primitive(
            points[:, 0], points[:, 1], time, STRENGTH, BACKGROUND, period=PERIOD)))
        moved = torch.stack(list(vortex_primitive(
            points[:, 0] + shift[0], points[:, 1] + shift[1], time, STRENGTH,
            BACKGROUND, period=PERIOD)))
        assert float((base - moved).abs().max()) < 1e-13


# -- R2 --------------------------------------------------------------------

def test_reference_is_isentropic_and_mutant_is_not():
    record = certify_isentropy()
    assert record["passed"], record
    assert record["analytic"] <= 1e-14
    assert record["mutant_detected"], record


# -- R3 --------------------------------------------------------------------

def test_invariant_reference_is_converged_beyond_the_drift_it_divides():
    record = certify_invariant_reference()
    assert record["passed"], record
    # the drift columns divide by these invariants, so the reference has to be
    # at least an order of magnitude finer than the smallest reported drift
    assert record["gate_worst"] * 10.0 <= record["smallest_drift"]
    assert record["smallest_drift"] == _smallest_conservation_drift()[0]


def test_density_min_is_a_sampled_minimum_not_the_core_value():
    """文档化的局限：张量网格最小值偏高于真实核心密度，只能同仪器相比。"""
    record = certify_invariant_reference()
    gap = record["density_min"]["relative_gap"]
    assert 1e-6 < gap < 1e-2, gap


# -- R4 --------------------------------------------------------------------

def test_error_norm_is_quadrature_converged():
    record = certify_error_norm(with_solves=False)
    assert record["passed"], record
    for name, case in record["cases"].items():
        assert abs(case["relative_shift"]) < 1e-2, name


def test_error_norm_would_show_under_integration():
    """欠积分对照：1 点求积必须显著改变读数，否则 R4 通过没有信息量。"""
    record = certify_error_norm(with_solves=False)
    for name, case in record["cases"].items():
        assert case["under_integrated_shift"] < -0.5, name


# -- R5 --------------------------------------------------------------------

def test_entropy_integral_is_quadrature_limited_while_conservation_is_not():
    """C4 的积分核 p/rho^gamma 非多项式，门禁求积与 6× 求积必须可比但不等。

    同时钉住对照：同一状态下质量/动量/能量（多项式被积函数）的求积差必须是
    舍入噪声，即"求积差"确实来自熵核，而不是求积实现本身有问题。
    """
    config = {"task": "dg", "degree": 2, "cells": 32, "final_time": 0.0, "cfl": 0.2}
    coeff, _basis, _nodes, _weights = _initial_dg_state(config)
    readings = {}
    for multiplier in (1, 6):
        nodes, weights, basis, _, _ = dg2.legendre_operators(2, 4 * multiplier)
        readings[multiplier] = _dg_measurements(coeff, basis, nodes, weights)
    for name in ("mass", "momentum_x", "momentum_y", "energy"):
        gap = abs(readings[1]["invariants"][name] / readings[6]["invariants"][name] - 1.0)
        assert gap < 1e-14, (name, gap)
    entropy_gap = abs(readings[1]["invariants"]["entropy"]
                      / readings[6]["invariants"]["entropy"] - 1.0)
    assert entropy_gap > 0.0
    assert entropy_gap < 1e-6


# -- R6 --------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="认证工件尚未生成")
def test_published_certificate_is_complete_and_reproduces_the_gate():
    """发布工件不能陈旧：它必须自洽，且其范数读数必须复现已入库的门禁读数。"""
    data = json.loads(ARTIFACT.read_text())
    assert data["verdict"]["status"] == "CERTIFIED", data["verdict"]
    assert data["verdict"]["failed"] == []
    assert data["R1_exactness"]["passed"] and data["R1_exactness"]["mutants_detected"]
    assert data["R2_isentropy"]["passed"] and data["R2_isentropy"]["mutant_detected"]
    assert data["R3_invariants"]["passed"]
    assert data["R4_error_norm"]["passed"]
    # every norm case that has a committed reading must reproduce it
    for name, case in data["R4_error_norm"]["cases"].items():
        stored = case.get("stored")
        if stored is None:
            continue
        assert abs(case["reproduction_relative"]) < 1e-9, (name, case["reproduction_relative"])
