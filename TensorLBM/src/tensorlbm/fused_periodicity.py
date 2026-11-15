"""Fused, direction-selective periodic ghost exchange for LBM streaming.

Anchor
------
``XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED`` — cross-validation of
``lssfau/lbmpy`` ``mpi-ghost-exchange-kernel-fused``
(``src/lbmpy/advanced_streaming/communication.py``, classes
``LBMPeriodicityHandling`` / helper ``get_communication_slices``).

What the upstream做法 actually is (verbatim pointers, registry LBM-FP-58)
-----------------------------------------------------------------------
The upstream module does **no** domain decomposition and no asynchronous
communication.  Instead it *fuses* the periodic ghost exchange into the
generated streaming/communication kernels: for every communication direction
``comm_dir`` it copies **only the ghost slab** (thickness 1) and **only the
one population whose streaming direction crosses that slab** (``write_index``
is a single constant pdf index in ``_compile_copy_kernels``), using plain
``arr[dst] = arr[src]`` on the CPU or a generated copy kernel on the GPU.
The comment in ``LBMPeriodicityHandling.__init__`` claims those generated GPU
copy kernels are "almost twice as fast in execution as cupy array copying".

Two properties therefore have to be validated, not assumed:

1. **Selectivity** — the set of cells whose values are fetched across the
   periodic seam, and the population index used for each of them, must equal
   the upstream ``comm_slices`` set.  Being "selective" is worthless if the
   selection is wrong.
2. **No correctness loss** — a fused/shell-only exchange must reproduce the
   whole-field-copy result bit-for-bit, and the resulting LBM solution must
   still match analytic decay rates.

This module provides both: the exchange plan is *derived* from the stencil
(no hand-written tables), three streaming back-ends (fused shell copy,
full-field ``roll`` copy, separate ghost-array pass) for the comparison, and
a small NumPy BGK solver used by the benchmark arms.

Conventions
-----------
``f`` has shape ``(Q, N0, N1[, N2])``; axis ``a + 1`` of ``f`` is physical
component ``a`` of the stencil, i.e. ``stencil[q, a]`` is read along spatial
axis ``a``.  All quantities are in lattice units.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

__all__ = [
    "D2Q9",
    "D3Q19",
    "D3Q27",
    "weights_cs",
    "opposite_index",
    "equilibrium",
    "macroscopic",
    "wrap_plan",
    "selection_matrix",
    "FusedPeriodicStreamer",
    "ExchangeStats",
    "FusedPeriodicBGK",
    "taylor_green_2d",
    "shear_wave",
    "mode_amplitude",
    "fit_exponential_rate",
]


# ---------------------------------------------------------------------------
# Stencils (component order (x, y[, z]); row q = population q)
# ---------------------------------------------------------------------------

D2Q9: np.ndarray = np.array(
    [(0, 0),
     (1, 0), (-1, 0), (0, 1), (0, -1),
     (1, 1), (-1, 1), (-1, -1), (1, -1)],
    dtype=np.int64,
)

D3Q19: np.ndarray = np.array(
    [(0, 0, 0),
     (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
     (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0),
     (1, 0, 1), (-1, 0, 1), (-1, 0, -1), (1, 0, -1),
     (0, 1, 1), (0, -1, 1), (0, -1, -1), (0, 1, -1)],
    dtype=np.int64,
)

D3Q27: np.ndarray = np.array(
    list(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64
)


def weights_cs(stencil: np.ndarray) -> Tuple[np.ndarray, float]:
    """Standard BGK weights for the supported stencils (cs^2 = 1/3)."""
    stencil = np.asarray(stencil)
    q = len(stencil)
    speed = np.abs(stencil).sum(axis=1)
    mapping = {
        (9, 2): {0: 4 / 9, 1: 1 / 9, 2: 1 / 36},
        (19, 3): {0: 1 / 3, 1: 1 / 18, 2: 1 / 36},
        (27, 3): {0: 8 / 27, 1: 2 / 27, 2: 1 / 54, 3: 1 / 216},
    }
    if (q, stencil.shape[1]) not in mapping:
        raise ValueError(f"no BGK weights known for a {q}Q{stencil.shape[1]} stencil")
    table = mapping[(q, stencil.shape[1])]
    return np.array([table[int(s)] for s in speed], dtype=np.float64), 1.0 / 3.0


def opposite_index(stencil: np.ndarray) -> np.ndarray:
    """Index of the population with the reversed velocity."""
    stencil = np.asarray(stencil)
    index = {tuple(c): i for i, c in enumerate(stencil.tolist())}
    return np.array([index[tuple(-c)] for c in stencil.tolist()], dtype=np.int64)


# ---------------------------------------------------------------------------
# Kinetic layer (NumPy, generic dimension)
# ---------------------------------------------------------------------------

def equilibrium(stencil: np.ndarray, rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Second-order incompressible equilibrium; ``u`` has shape (D, *spatial)."""
    stencil = np.asarray(stencil)
    w, cs2 = weights_cs(stencil)
    rho = np.asarray(rho, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    eu = np.tensordot(stencil, u, axes=(1, 0))          # (Q, *spatial)
    usq = np.einsum("a...,a...->...", u, u)             # (*spatial,)
    shape = (len(stencil),) + (1,) * usq.ndim
    wq = w.reshape(shape)
    return rho * wq * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * usq)


def macroscopic(stencil: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Density and velocity fields from ``f`` of shape (Q, *spatial)."""
    stencil = np.asarray(stencil)
    rho = f.sum(axis=0)
    u = np.einsum("qa,q...->a...", stencil.astype(np.float64), f) / rho
    return rho, u


# ---------------------------------------------------------------------------
# Exchange plan derived from the stencil
# ---------------------------------------------------------------------------

def _axis_boxes(n: int, e: int) -> List[Tuple[slice, slice, bool]]:
    """Per-axis decomposition of a wrapped shift by ``e`` cells.

    Returns ``(dst_slice, src_slice, crosses_seam)`` triples that partition the
    destination axis exactly once.
    """
    if e == 0:
        return [(slice(0, n), slice(0, n), False)]
    if e > 0:
        return [(slice(e, n), slice(0, n - e), False),
                (slice(0, e), slice(n - e, n), True)]
    return [(slice(0, n + e), slice(-e, n), False),
            (slice(n + e, n), slice(0, -e), True)]


def wrap_plan(stencil: np.ndarray, shape: Sequence[int]) -> Dict[int, List[Tuple[Tuple[slice, ...], Tuple[slice, ...], bool, int]]]:
    """Streaming plan for one periodic shift per population.

    For every population ``q`` the returned list partitions the destination
    cells into boxes ``(dst_slices, src_slices, crosses_seam, nelem)``.  Boxes
    with ``crosses_seam=False`` are the plain interior copy; boxes with
    ``crosses_seam=True`` are exactly the cells whose value has to come from
    the far side of the periodic seam — the ghost values.  The plan is derived
    from the stencil alone (no per-lattice hard-coding).
    """
    stencil = np.asarray(stencil)
    shape = tuple(int(n) for n in shape)
    if stencil.shape[1] != len(shape):
        raise ValueError("stencil dimension and field rank disagree")
    plan: Dict[int, List[Tuple[Tuple[slice, ...], Tuple[slice, ...], bool, int]]] = {}
    for q in range(len(stencil)):
        per_axis = [_axis_boxes(shape[a], int(stencil[q, a])) for a in range(len(shape))]
        boxes = []
        for combo in itertools.product(*per_axis):
            nelem = 1
            for dst_s, _, _ in combo:
                nelem *= max(0, dst_s.stop - dst_s.start)
            if nelem == 0:
                continue
            dst = tuple(c[0] for c in combo)
            src = tuple(c[1] for c in combo)
            seam = any(c[2] for c in combo)
            boxes.append((dst, src, seam, nelem))
        plan[q] = boxes
    return plan


def selection_matrix(stencil: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Boolean mask ``(Q, *shape)`` of cells fetched across the periodic seam.

    This is the discrete object that has to agree with the upstream
    ``LBMPeriodicityHandling.comm_slices`` (see
    ``tests/test_fused_periodicity_lbmpy_conformance.py``).
    """
    stencil = np.asarray(stencil)
    plan = wrap_plan(stencil, shape)
    mask = np.zeros((len(stencil),) + tuple(shape), dtype=bool)
    for q, boxes in plan.items():
        for dst, _src, seam, _n in boxes:
            if seam:
                mask[(q,) + dst] = True
    return mask


# ---------------------------------------------------------------------------
# Streaming back-ends
# ---------------------------------------------------------------------------

@dataclass
class ExchangeStats:
    """Per-call counters, in float64 elements (8 bytes each)."""

    mode: str
    n_elem_total: int = 0        # every element written by the streaming pass
    n_elem_seam: int = 0         # elements read across the periodic seam
    n_field_copies: int = 0      # full-field temporary copies materialised
    n_ghost_arrays: int = 0      # ghost-extended buffers allocated

    @property
    def seam_fraction(self) -> float:
        return 0.0 if self.n_elem_total == 0 else self.n_elem_seam / self.n_elem_total

    @property
    def bytes_written(self) -> int:
        return 8 * self.n_elem_total

    @property
    def bytes_seam(self) -> int:
        return 8 * self.n_elem_seam

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "n_elem_total": int(self.n_elem_total),
            "n_elem_seam": int(self.n_elem_seam),
            "seam_fraction": float(self.seam_fraction),
            "bytes_written": int(self.bytes_written),
            "bytes_seam": int(self.bytes_seam),
            "n_field_copies": int(self.n_field_copies),
            "n_ghost_arrays": int(self.n_ghost_arrays),
        }


class FusedPeriodicStreamer:
    """Three interchangeable periodic-streaming implementations.

    ``stream_fused``      pull streaming with the seam values written **in the
                          same pass** as the interior values, straight from the
                          post-collision array (upstream's fused copy kernel).
    ``stream_fullcopy``   ``np.roll`` per population — the whole-field array
                          copy the upstream comment measures against.
    ``stream_ghost``      separate ghost-array pass: pad the population with a
                          wrapped halo, then pull from the padded array — the
                          un-fused "exchange then stream" arrangement.

    All three must agree bit-for-bit; they differ only in data movement.
    """

    def __init__(self, stencil: np.ndarray, shape: Sequence[int], dtype=np.float64):
        self.stencil = np.asarray(stencil)
        self.shape = tuple(int(n) for n in shape)
        self.dtype = dtype
        self.plan = wrap_plan(self.stencil, self.shape)
        self.n_q = len(self.stencil)
        self.stats = ExchangeStats(mode="fused")

    # -- helpers ---------------------------------------------------------
    def _seam_elements(self) -> int:
        return sum(n for boxes in self.plan.values() for _d, _s, seam, n in boxes if seam)

    def _total_elements(self) -> int:
        return sum(n for boxes in self.plan.values() for _d, _s, _seam, n in boxes)

    # -- back-end 1: fused -------------------------------------------------
    def stream_fused(self, f_post: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        f_post = np.asarray(f_post)
        if f_post.shape != (self.n_q,) + self.shape:
            raise ValueError(f"expected shape {(self.n_q,) + self.shape}, got {f_post.shape}")
        f_new = np.empty_like(f_post, dtype=self.dtype) if out is None else out
        for q, boxes in self.plan.items():
            src_f = f_post[q]
            dst_f = f_new[q]
            for dst, src, _seam, _n in boxes:
                dst_f[dst] = src_f[src]
        self.stats = ExchangeStats(
            mode="fused",
            n_elem_total=self._total_elements(),
            n_elem_seam=self._seam_elements(),
            n_field_copies=0,
            n_ghost_arrays=0,
        )
        return f_new

    # -- back-end 2: full-field copy ---------------------------------------
    def stream_fullcopy(self, f_post: np.ndarray) -> np.ndarray:
        f_post = np.asarray(f_post)
        f_new = np.empty_like(f_post, dtype=self.dtype)
        for q in range(self.n_q):
            shifts = tuple(int(c) for c in self.stencil[q])
            f_new[q] = np.roll(f_post[q], shift=shifts, axis=tuple(range(len(self.shape))))
        n = self.n_q * int(np.prod(self.shape))
        self.stats = ExchangeStats(
            mode="fullcopy",
            n_elem_total=2 * n,              # roll temporary + assignment
            n_elem_seam=self._seam_elements(),
            n_field_copies=self.n_q,
            n_ghost_arrays=0,
        )
        return f_new

    # -- back-end 3: separate ghost-array pass -----------------------------
    def stream_ghost(self, f_post: np.ndarray) -> np.ndarray:
        f_post = np.asarray(f_post)
        pad_width = [(0, 0)] + [(1, 1)] * len(self.shape)
        f_new = np.empty_like(f_post, dtype=self.dtype)
        for q in range(self.n_q):
            padded = np.pad(f_post[q], [(1, 1)] * len(self.shape), mode="wrap")
            inner = tuple(slice(1 - int(e), 1 - int(e) + n)
                          for e, n in zip(self.stencil[q], self.shape))
            f_new[q] = padded[inner]
        n = self.n_q * int(np.prod(self.shape))
        self.stats = ExchangeStats(
            mode="ghost",
            n_elem_total=2 * n,              # halo padding + extraction
            n_elem_seam=self._seam_elements(),
            n_field_copies=self.n_q,
            n_ghost_arrays=self.n_q,
            )
        return f_new


# ---------------------------------------------------------------------------
# Minimal BGK solver (physics arm of the cross-validation)
# ---------------------------------------------------------------------------

class FusedPeriodicBGK:
    """BGK collision + fused periodic streaming, NumPy, periodic in all axes."""

    def __init__(self, stencil: np.ndarray, shape: Sequence[int], nu: float,
                 mode: str = "fused", dtype=np.float64):
        self.stencil = np.asarray(stencil)
        self.shape = tuple(int(n) for n in shape)
        self.dtype = dtype
        self.nu = float(nu)
        self.omega = 1.0 / (3.0 * self.nu + 0.5)
        self.streamer = FusedPeriodicStreamer(self.stencil, self.shape, dtype=dtype)
        if mode not in ("fused", "fullcopy", "ghost"):
            raise ValueError(f"unknown streaming mode {mode!r}")
        self.mode = mode
        self.f = np.zeros((len(self.stencil),) + self.shape, dtype=dtype)

    # -- state -----------------------------------------------------------
    def init_from_velocity(self, u: np.ndarray, rho: float | np.ndarray = 1.0) -> None:
        u = np.asarray(u, dtype=np.float64)
        rho_arr = np.full(self.shape, float(rho), dtype=np.float64) if np.isscalar(rho) else np.asarray(rho, dtype=np.float64)
        self.f = equilibrium(self.stencil, rho_arr, u).astype(self.dtype)

    def macroscopic(self) -> Tuple[np.ndarray, np.ndarray]:
        return macroscopic(self.stencil, self.f)

    # -- one time step ---------------------------------------------------
    def step(self, n: int = 1) -> "FusedPeriodicBGK":
        for _ in range(n):
            rho, u = macroscopic(self.stencil, self.f)
            f_eq = equilibrium(self.stencil, rho, u)
            f_post = self.f - self.omega * (self.f - f_eq)
            if self.mode == "fused":
                self.f = self.streamer.stream_fused(f_post, out=self.f)
            elif self.mode == "fullcopy":
                self.f = self.streamer.stream_fullcopy(f_post)
            else:
                self.f = self.streamer.stream_ghost(f_post)
        return self

    # -- observables -----------------------------------------------------
    def total_mass(self) -> float:
        return float(self.f.sum())

    def momentum(self) -> np.ndarray:
        rho, u = self.macroscopic()
        return rho * u


# ---------------------------------------------------------------------------
# Analytic initial conditions / diagnostics
# ---------------------------------------------------------------------------

def taylor_green_2d(shape: Sequence[int], amp: float = 0.04,
                    k: float | None = None) -> np.ndarray:
    """2-D Taylor-Green: exact decaying NS solution, velocity ∝ exp(-2 nu k^2 t)."""
    ny, nx = int(shape[0]), int(shape[1])
    k = 2.0 * np.pi / nx if k is None else float(k)
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    ux = amp * np.sin(k * X) * np.cos(k * Y)
    uy = -amp * np.cos(k * X) * np.sin(k * Y)
    return np.stack([ux, uy])


def shear_wave(shape: Sequence[int], axis: int = 0, varying: int = 1,
               amp: float = 0.04, k: float | None = None) -> np.ndarray:
    """Kolmogorov shear wave ``u_axis = amp cos(k n_varying)``.

    The convective term vanishes identically, so this is an exact decaying
    solution of the incompressible Navier-Stokes equations with velocity
    amplitude ∝ exp(-nu k^2 t) for any dimension and any BGK stencil.
    """
    shape = tuple(int(n) for n in shape)
    k = 2.0 * np.pi / shape[varying] if k is None else float(k)
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    u = np.zeros((len(shape),) + shape)
    u[axis] = amp * np.cos(k * grids[varying])
    return u


def mode_amplitude(field: np.ndarray, mode: Tuple[int, ...]) -> float:
    """|DFT| at a single mode index, normalised by the number of cells."""
    spec = np.fft.fftn(field)
    return float(np.abs(spec[mode]) / field.size)


def fit_exponential_rate(times: Sequence[float], values: Sequence[float]) -> float:
    """Least-squares slope of ``ln(values)`` against ``times`` (negative = decay)."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if np.any(v <= 0):
        raise ValueError("log fit needs positive values")
    return float(np.polyfit(t, np.log(v), 1)[0])