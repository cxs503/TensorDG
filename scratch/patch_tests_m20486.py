"""m20486 patch 3: robust cross-check order + tests re-bound to the corrected gates."""
from pathlib import Path

# ---------------------------------------------------------------- source
p = Path("benchmarks/case_euler_vortex_errconserve.py")
src = p.read_text("utf-8")

old = '''    rows = [r for r in records
            if r["task"] == "p0" and abs(r["final_time"] - checkpoint) < 1e-12]
    rows.sort(key=lambda r: r["cells"])'''
new = '''    rows = [r for r in records if r["family"].startswith("p0_n")
            and abs(r["final_time"] - checkpoint) < 1e-12]
    rows.sort(key=_p0_cells)'''
assert old in src
src = src.replace(old, new, 1)

old = '''def _observed_order(records, checkpoint: float):'''
new = '''def _p0_cells(record) -> int:
    """Cell count of a P0 record, from the record or from its family key."""
    cells = record.get("cells")
    if cells is None:
        cells = int(record["family"].split("_n")[-1])
    return int(cells)


def _observed_order(records, checkpoint: float):'''
assert old in src
src = src.replace(old, new, 1)

src = src.replace('''    cells = np.array([r["cells"] for r in rows], dtype=float)''',
                  '''    cells = np.array([_p0_cells(r) for r in rows], dtype=float)''', 1)
p.write_text(src, "utf-8")

# ---------------------------------------------------------------- tests
t = Path("tests/test_case_euler_vortex_errconserve.py")
ts = t.read_text("utf-8")

old = '''def test_c2_gate_rejects_secular_error_growth():
    exact = ec.exact_invariants(points=80)
    accumulating = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
                    for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(accumulating, exact)["gates"]
    assert gates["C2_error_conservation"]["exponent"] == pytest.approx(1.0, abs=1e-6)
    assert gates["C2_error_conservation"]["passed"] is False
    assert gates["all_passed"] is False


def test_c2_gate_accepts_conserved_error():
    exact = ec.exact_invariants(points=80)
    conserved = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * (1.0 + 0.01 * math.log(t)))
                 for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(conserved, exact)["gates"]
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["C3_discrete_conservation"]["passed"] is True
    assert gates["C1_tolerance"]["passed"] is True
    assert gates["C4_entropy_budget"]["passed"] is True
    assert gates["all_passed"] is True'''
new = '''def _p0_ladder(times=(1.0, 5.0)):
    """Synthetic first-order cross-check ladder (n=64/128/256)."""
    return [_synthetic_record(f"p0_n{n}", t, 0.4 / n * (1.0 + 0.01 * math.log(t)))
            for n in (64, 128, 256) for t in times]


def test_c2_gate_rejects_superlinear_error_growth():
    exact = ec.exact_invariants(points=80)
    accumulating = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t * t)
                    for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(accumulating, exact)["gates"]
    assert gates["C2_error_conservation"]["exponent"] == pytest.approx(2.0, abs=1e-6)
    assert gates["C2_error_conservation"]["passed"] is False
    assert gates["C2_error_conservation"]["status"] == "failed"
    assert gates["all_passed"] is False


def test_c2_linear_accumulation_is_the_boundary_and_passes():
    """alpha == 1 is the physical bound (error ∝ elapsed time) -> accepted.

    The stricter ``alpha <= 0.5`` reading has no literature source (see
    docs/AUDIT-m20486-CASE-DG-EULER-VORTEX-ERR.md) and is reported only.
    """
    exact = ec.exact_invariants(points=80)
    linear = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
              for t in (1.0, 2.0, 5.0, 10.0, 20.0)]
    gates = ec.evaluate_gates(linear, exact)["gates"]
    assert gates["C2_error_conservation"]["exponent"] == pytest.approx(1.0, abs=1e-6)
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["C2_error_conservation"]["strict_target"]["passed"] is False


def test_c2_gate_is_decidable_only_with_three_checkpoints():
    exact = ec.exact_invariants(points=80)
    two_points = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t) for t in (1.0, 2.0)]
    gate = ec.evaluate_gates(two_points, exact)["gates"]["C2_error_conservation"]
    assert gate["status"] == "not_evaluated"
    assert math.isnan(gate["exponent"])


def test_c2_gate_accepts_conserved_error():
    exact = ec.exact_invariants(points=80)
    conserved = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * (1.0 + 0.01 * math.log(t)))
                 for t in (1.0, 2.0, 5.0, 10.0, 20.0)] + _p0_ladder()
    gates = ec.evaluate_gates(conserved, exact)["gates"]
    assert gates["C2_error_conservation"]["passed"] is True
    assert gates["C3_discrete_conservation"]["passed"] is True
    assert gates["C1_tolerance"]["passed"] is True
    assert gates["C4_entropy_budget"]["passed"] is True
    assert gates["C5_crosscheck_order"]["passed"] is True
    assert gates["all_passed"] is True'''
assert old in ts
ts = ts.replace(old, new, 1)

old = '''def test_c1_gate_rejects_a_checkpoint_above_tolerance():
    exact = ec.exact_invariants(points=80)
    bad = [_synthetic_record(ec.GATE_FAMILY, 1.0, 1e-4),
           _synthetic_record(ec.GATE_FAMILY, 20.0, 0.10)]
    gates = ec.evaluate_gates(bad, exact)["gates"]
    assert gates["C1_tolerance"]["passed"] is False'''
new = '''def test_c1_gate_rejects_a_checkpoint_above_tolerance():
    """The 6 % case-pool tolerance is a snapshot tolerance at t = 1."""
    exact = ec.exact_invariants(points=80)
    bad = [_synthetic_record(ec.GATE_FAMILY, 1.0, 0.10),
           _synthetic_record(ec.GATE_FAMILY, 20.0, 1e-4)]
    gates = ec.evaluate_gates(bad, exact)["gates"]
    assert gates["C1_tolerance"]["passed"] is False
    assert gates["C1_tolerance"]["worst_at"].endswith("t1")


def test_long_time_checkpoints_exceed_tolerance_but_stay_visible():
    """Scope correction, guarded: a t=20 violation must NOT vanish.

    It leaves C1 (snapshot) untouched -- that is the corrected scope -- but the
    number must still be reported verbatim in C1x, otherwise the fix would be
    evidence laundering.
    """
    exact = ec.exact_invariants(points=80)
    records = [_synthetic_record(ec.GATE_FAMILY, 1.0, 1e-4),
               _synthetic_record(ec.GATE_FAMILY, 20.0, 0.10)] + _p0_ladder()
    gates = ec.evaluate_gates(records, exact)["gates"]
    assert gates["C1_tolerance"]["passed"] is True
    assert gates["C1x_crosscheck_raw"]["measured"] == pytest.approx(0.10)
    assert gates["C1x_crosscheck_raw"]["worst_at"].endswith("t20")
    assert "C1x_crosscheck_raw" in gates["informational"]


def test_c5_crosscheck_order_flags_a_non_converging_second_path():
    exact = ec.exact_invariants(points=80)
    flat = [_synthetic_record(f"p0_n{n}", 5.0, 3e-3) for n in (64, 128, 256)]
    flat += [_synthetic_record(f"p0_n{n}", 1.0, 1e-3) for n in (64, 128, 256)]
    gates = ec.evaluate_gates(flat, exact)["gates"]
    assert gates["C5_crosscheck_order"]["status"] == "failed"
    assert gates["verdict"] == "FAIL"'''
assert old in ts
ts = ts.replace(old, new, 1)

old = '''    records = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t)
               for t in (1.0, 2.0, 5.0)]
    exact = ec.exact_invariants(points=80)
    metrics = ec.aggregate(records, exact, tmp_path)'''
new = '''    records = [_synthetic_record(ec.GATE_FAMILY, t, 1e-4 * t * t)
               for t in (1.0, 2.0, 5.0)]
    exact = ec.exact_invariants(points=80)
    metrics = ec.aggregate(records, exact, tmp_path)'''
assert old in ts
ts = ts.replace(old, new, 1)

t.write_text(ts, "utf-8")
print("patched3")