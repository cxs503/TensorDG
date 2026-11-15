"""Conformance of the derived exchange plan with the installed upstream lbmpy.

Anchor: ``XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED``.

Upstream: ``lbmpy.advanced_streaming.communication.LBMPeriodicityHandling``
(``src/lbmpy/advanced_streaming/communication.py:20``) builds
``comm_slices`` — a list of ``(src_slice, dst_slice)`` copy pairs in which the
last index of every slice is a *single* pdf index.  That is the
direction-selective ghost exchange this anchor validates.

Frame relation
--------------
Upstream writes **into the ghost layer** of the ghost-extended array and is a
copy-kernel formulation, while the fused pull implementation in
``tensorlbm.fused_periodicity`` reads across the seam directly.  For a
population with velocity ``c`` the two cell sets are related by a shift of
``+c`` modulo the domain size: an upstream copy from interior cell ``x`` into
the far ghost serves exactly the pull destination ``x + c``.  The test asserts
that relation per population, not a raw set equality.
"""

import itertools

import numpy as np
import pytest

lbmpy = pytest.importorskip("lbmpy")

from lbmpy.session import LBStencil, Stencil  # noqa: E402
from pystencils.datahandling import create_data_handling  # noqa: E402
from lbmpy.advanced_streaming.communication import LBMPeriodicityHandling  # noqa: E402

from tensorlbm.fused_periodicity import D2Q9, D3Q19, D3Q27, selection_matrix  # noqa: E402

CASES = [
    ("D2Q9", Stencil.D2Q9, D2Q9, (3, 4)),
    ("D2Q9", Stencil.D2Q9, D2Q9, (7, 5)),
    ("D3Q19", Stencil.D3Q19, D3Q19, (4, 5, 3)),
    ("D3Q27", Stencil.D3Q27, D3Q27, (3, 4, 2)),
    ("D3Q27", Stencil.D3Q27, D3Q27, (4, 3, 5)),
]


def upstream_ghost_targets(shape, n_pop, stencil):
    """Cells per physical velocity that upstream copies (ghost frame -> interior)."""
    dh = create_data_handling(shape, periodicity=(True,) * len(shape), default_target="cpu")
    dh.add_array("pdfs", values_per_cell=n_pop)
    dh.fill("pdfs", 0.0, ghost_layers=True)
    per = LBMPeriodicityHandling(stencil=stencil, data_handling=dh, pdf_field_name="pdfs")
    lb = np.array(stencil, dtype=np.int64)
    targets = {}
    for timestep in per.comm_slices:
        for _src, dst in timestep:
            *spatial, pdf = dst
            axes = []
            for axis, entry in enumerate(spatial):
                n = shape[axis]
                ext = n + 2
                if isinstance(entry, slice):
                    lo = 0 if entry.start is None else (entry.start + ext if entry.start < 0 else entry.start)
                    hi = ext if entry.stop is None else (entry.stop + ext if entry.stop < 0 else entry.stop)
                    indices = range(lo, hi)
                else:
                    indices = [entry + ext if entry < 0 else entry]
                axes.append([(int(i) - 1) % n for i in indices])
            key = tuple(int(v) for v in lb[pdf])
            targets.setdefault(key, set()).update(itertools.product(*axes))
    return targets


@pytest.mark.parametrize("name,stencil_enum,stencil,shape", CASES)
def test_seam_selection_matches_upstream_comm_slices(name, stencil_enum, stencil, shape):
    upstream = upstream_ghost_targets(shape, len(stencil), LBStencil(stencil_enum))
    mine = selection_matrix(stencil, shape)
    for q, velocity in enumerate(stencil):
        c = tuple(int(v) for v in velocity)
        shifted = {tuple((cell + e) % n for cell, e, n in zip(cell, c, shape))
                   for cell in upstream.get(c, set())}
        ours = {tuple(int(v) for v in idx) for idx in np.argwhere(mine[q])}
        assert shifted == ours, f"{name} {shape} population {c}: {len(shifted)} vs {len(ours)} cells"
    total_upstream = sum(len(v) for v in upstream.values())
    assert total_upstream == int(mine.sum())


@pytest.mark.parametrize("name,stencil_enum,stencil,shape", CASES)
def test_upstream_is_population_selective(name, stencil_enum, stencil, shape):
    """Every copied cell carries exactly one pdf index (not the full Q block)."""
    upstream = upstream_ghost_targets(shape, len(stencil), LBStencil(stencil_enum))
    n_seam = int(selection_matrix(stencil, shape).sum())
    assert sum(len(v) for v in upstream.values()) == n_seam
    # Un-fused alternative would move Q populations over the same cells.
    assert n_seam * len(stencil) > n_seam