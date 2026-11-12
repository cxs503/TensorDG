"""Machine gate for the 1-D LDG Dirichlet-closure probe (m16266).

Pins the reproducible facts that block any "LDG is available" claim:

* the classic alternating traces + Dirichlet data give a rank deficiency of
  exactly one, independent of the mesh, for k = 1, 2, 3;
* the null direction lives in the u block of a single element;
* the mirror / Nitsche closure variants that the status note proposed do not
  produce a convergent solution (relative L2 error stays O(1)).

The probe writes ``validation/ldg_closure/results.json``; these tests recompute
the facts directly from the probe's own assembly so they fail if the probe or
its conclusion is edited without re-deriving the numbers.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = ROOT / "validation" / "ldg_closure" / "probe_ldg1d.py"


def _load():
    spec = importlib.util.spec_from_file_location("ldg_closure_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PROBE_MOD = _load()


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("n", [4, 8, 16])
def test_pure_closure_is_rank_deficient_by_exactly_one(n, k):
    A, _ = PROBE_MOD.assemble(n, k, "pure")
    rank = np.linalg.matrix_rank(A, tol=1e-10)
    assert A.shape[0] - rank == 1


@pytest.mark.parametrize("k", [1, 2])
def test_null_direction_is_confined_to_one_element_u_block(k):
    rec = PROBE_MOD.null_mode(8, k)
    assert rec["energy_fraction_in_argmax_element_u"] > 0.999
    assert rec["sigma_min"] < 1e-12


@pytest.mark.parametrize("closure", ["pure", "mirror", "nitsche+", "nitsche-"])
def test_no_candidate_closure_converges_at_first_order(closure):
    coarse, _ = PROBE_MOD.rel_l2_u(8, 1, closure)
    fine, _ = PROBE_MOD.rel_l2_u(16, 1, closure)
    # A working LDG would shrink like h^(k+1); every candidate stays O(1).
    assert coarse > 0.3 and fine > 0.3
    assert fine / coarse > 0.9


def test_probe_results_json_is_in_repo_and_consistent():
    import json

    data = json.loads((ROOT / "validation" / "ldg_closure" / "results.json").read_text())
    for closure in ("pure", "mirror", "nitsche+", "nitsche-"):
        for k in ("k=1", "k=2", "k=3"):
            block = data["closures"][closure][k]
            for n in ("8", "16", "32"):
                lv = block["levels"][n]
                assert 0.0 < lv["rel_l2_u"] < 1e3
        assert set(data["closures"][closure]) == {"k=1", "k=2", "k=3"}
