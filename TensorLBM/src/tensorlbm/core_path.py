"""T2 unified LBM core path: one API, two kernels, pluggable back-ends.

Anchor
------
``GAP-LBM-T2-GPU-COREPATH`` — "T2 GPU 核心路径统一优化（P1）":
collision+streaming fusion / memory management / multi-back-end /
performance-accuracy trade-off, under the explicit discipline

    **optimisation object == T3-validated physics case, no blind peak chasing;
    one unified comparison protocol (对标口径统一纪律)**

Why this module exists
----------------------
Before this module ``tensorlbm`` had *three* unrelated time-stepping
implementations (``FusedPeriodicBGK`` in ``fused_periodicity.py``,
``SparseAOSLBM``, ``MCMPSolver``), **no** back-end abstraction, **no** memory
accounting, and **no** place where an acceleration claim could be checked
against a physics reference.  A speed number produced that way is not
comparable to anything — exactly the failure mode the anchor forbids.

What is unified here
--------------------
1. **One step API** (:class:`CorePath`: ``init_from_velocity`` / ``step`` /
   ``macroscopic``) for every (kernel, backend, dtype) combination.
2. **Two kernels on the same API**:
   ``reference`` — collision materialises a full ``f_eq`` and a full
   post-collision field, then full-field periodic streaming (4 live Q*N
   buffers);
   ``fused`` — per-population single sweep: ``f_eq[q]`` is formed in one
   N-sized scratch buffer, relaxed in place and streamed straight into the
   destination, so neither ``f_eq`` nor ``f_post`` is ever materialised
   (2 live Q*N buffers).
   Periodic wrap is folded into the sweep (one ``roll`` per population).
   Planar walls stay in ``boundaries.py`` — out of scope here, declared, not
   silently wrong.
3. **Pluggable back-ends** (``numpy``, ``torch-cpu``, ``torch-cuda``,
   ``cupy``) with capability probing: an unavailable back-end is reported
   ``available=False`` with a reason and can never produce a number, so a GPU
   arm on a GPU-less machine is ``WITHHELD`` instead of silently emulated.
4. **Memory accounting**: the exact live-buffer inventory implied by each
   kernel (:meth:`CorePath.declared_live_bytes`) plus the measured
   ``tracemalloc`` peak, so "fusion saves memory" is a measured statement.
5. **Unified comparison protocol** (:func:`run_arm` + :class:`ArmResult`):
   every arm runs the T3-validated case ``taylor-green-2d`` (T3 matrix row 5,
   REAL, analytic decay rate ``2 nu k^2``) and its throughput is only marked
   ``perf_admissible`` when both pre-registered gates pass.  The verdict is
   computed by :func:`admissibility`, so it cannot be edited after the run.

Pre-registered gates (fixed in code, not per run)
-------------------------------------------------
``ACCURACY_TOL = 2e-2``       relative error of the fitted decay rate vs
                              ``2 nu k^2`` (same tolerance as T3 matrix row 5)
``CONSISTENCY_TOL = 1e-10``   relative max |fused - reference| velocity
                              difference, float64 arms
``DTYPE_CONSISTENCY_TOL``     the same, per storage dtype (float32 arms are
                              limited by their own round-off, declared up front)

Conventions
-----------
``f`` has shape ``(Q, *shape)``; ``stencil[q, a]`` is the lattice velocity of
population q along spatial axis a.  Lattice units, ``cs^2 = 1/3``.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fused_periodicity import (
    D2Q9,
    D3Q19,
    D3Q27,
    weights_cs,
    wrap_plan,
    equilibrium,
    macroscopic,
    mode_amplitude,
    fit_exponential_rate,
    taylor_green_2d,
)

__all__ = [
    "ACCURACY_TOL",
    "CONSISTENCY_TOL",
    "DTYPE_CONSISTENCY_TOL",
    "T3_CASE",
    "STENCILS",
    "BackendUnavailable",
    "probe_backends",
    "CoreConfig",
    "CorePath",
    "ArmResult",
    "admissibility",
    "run_arm",
]

# ---------------------------------------------------------------------------
# Pre-registered gates (do not tune after seeing a run)
# ---------------------------------------------------------------------------

ACCURACY_TOL = 2.0e-2
CONSISTENCY_TOL = 1.0e-10
DTYPE_CONSISTENCY_TOL = {"float64": CONSISTENCY_TOL, "float32": 1.0e-5}
T3_CASE = "taylor-green-2d"

STENCILS: Dict[str, np.ndarray] = {"D2Q9": D2Q9, "D3Q19": D3Q19, "D3Q27": D3Q27}


# ---------------------------------------------------------------------------
# Back-ends
# ---------------------------------------------------------------------------

class BackendUnavailable(RuntimeError):
    """Raised when a caller asks for a back-end the machine cannot provide."""


class _NumpyOps:
    name = "numpy"
    kind = "cpu"

    def __init__(self, dtype: str):
        self.dtype_name = dtype
        self.dtype = np.dtype(dtype)

    def empty(self, shape: Sequence[int]):
        return np.empty(tuple(int(s) for s in shape), dtype=self.dtype)

    def zeros(self, shape: Sequence[int]):
        return np.zeros(tuple(int(s) for s in shape), dtype=self.dtype)

    def roll(self, a, shift: Tuple[int, ...]):
        return np.roll(a, shift=tuple(int(s) for s in shift),
                       axis=tuple(range(a.ndim)))

    def copyto(self, dst, src):
        np.copyto(dst, src, casting="unsafe")

    def copy_slices(self, dst, src, dst_slices, src_slices):
        dst[dst_slices] = src[src_slices]

    def from_numpy(self, a):
        return np.array(a, dtype=self.dtype, copy=True)

    def to_numpy(self, a):
        return np.asarray(a)

    def sync(self):
        return None

    def fill(self, a, value):
        a[...] = value


class _TorchOps:
    def __init__(self, device: str, dtype: str):
        import torch

        self._torch = torch
        self.device = device
        self.name = f"torch-{device}"
        self.kind = "gpu" if device.startswith("cuda") else "cpu"
        self.dtype_name = dtype
        self.dtype = getattr(torch, dtype)

    def empty(self, shape: Sequence[int]):
        return self._torch.empty(tuple(int(s) for s in shape), dtype=self.dtype,
                                 device=self.device)

    def zeros(self, shape: Sequence[int]):
        return self._torch.zeros(tuple(int(s) for s in shape), dtype=self.dtype,
                                 device=self.device)

    def roll(self, a, shift: Tuple[int, ...]):
        return self._torch.roll(a, shifts=tuple(int(s) for s in shift),
                                dims=tuple(range(a.dim())))

    def copyto(self, dst, src):
        dst.copy_(src)

    def copy_slices(self, dst, src, dst_slices, src_slices):
        dst[dst_slices].copy_(src[src_slices])

    def from_numpy(self, a):
        return self._torch.as_tensor(np.ascontiguousarray(a), dtype=self.dtype,
                                     device=self.device).clone()

    def to_numpy(self, a):
        return a.detach().cpu().numpy()

    def sync(self):
        if self.kind == "gpu":
            self._torch.cuda.synchronize()

    def fill(self, a, value):
        a.fill_(float(value))


@dataclass
class BackendProbe:
    """Result of probing one back-end.  ``available=False`` means "no number"."""

    name: str
    kind: str
    available: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "available": bool(self.available), "reason": self.reason}


def probe_backends() -> Dict[str, BackendProbe]:
    """Probe every candidate back-end without hard-importing optional ones."""
    out: Dict[str, BackendProbe] = {}
    out["numpy"] = BackendProbe("numpy", "cpu", True, f"numpy {np.__version__}")

    try:
        import torch

        out["torch-cpu"] = BackendProbe("torch-cpu", "cpu", True,
                                        f"torch {torch.__version__}")
        if torch.cuda.is_available():  # pragma: no cover - no GPU on this host
            out["torch-cuda"] = BackendProbe(
                "torch-cuda", "gpu", True, torch.cuda.get_device_name(0))
        else:
            out["torch-cuda"] = BackendProbe(
                "torch-cuda", "gpu", False, "torch.cuda.is_available() == False")
    except Exception as exc:  # noqa: BLE001 - a probe must never raise
        out["torch-cpu"] = BackendProbe("torch-cpu", "cpu", False, repr(exc))
        out["torch-cuda"] = BackendProbe("torch-cuda", "gpu", False,
                                         "torch import failed")

    try:
        import cupy

        try:
            n_dev = int(cupy.cuda.runtime.getDeviceCount())
            reason = f"cupy {cupy.__version__}, {n_dev} device(s)"
        except Exception as exc:  # noqa: BLE001
            n_dev, reason = 0, f"cupy installed but no device: {exc!r}"
        out["cupy"] = BackendProbe("cupy", "gpu", n_dev > 0, reason)
    except Exception:
        out["cupy"] = BackendProbe("cupy", "gpu", False, "cupy not installed")

    return out


def make_ops(backend: str, dtype: str):
    """Array-ops object for ``backend``; raises if the machine cannot provide it."""
    probes = probe_backends()
    if backend not in probes:
        raise BackendUnavailable(
            f"unknown backend {backend!r}; probed: {sorted(probes)}")
    if not probes[backend].available:
        raise BackendUnavailable(
            f"backend {backend!r} unavailable on this host: {probes[backend].reason}")
    if backend == "numpy":
        return _NumpyOps(dtype)
    if backend == "torch-cpu":
        return _TorchOps("cpu", dtype)
    raise BackendUnavailable(
        f"backend {backend!r} is probed but has no ops implementation yet")


# ---------------------------------------------------------------------------
# Configuration / state
# ---------------------------------------------------------------------------

KERNELS = ("reference", "fused", "fused_slices")


@dataclass(frozen=True)
class CoreConfig:
    """Everything that defines a core-path arm.  Frozen: an arm is an identity."""

    stencil: str = "D2Q9"
    shape: Tuple[int, ...] = (64, 64)
    nu: float = 0.05
    kernel: str = "fused"
    backend: str = "numpy"
    dtype: str = "float64"

    def __post_init__(self):
        if self.stencil not in STENCILS:
            raise ValueError(f"unknown stencil {self.stencil!r}")
        if STENCILS[self.stencil].shape[1] != len(tuple(self.shape)):
            raise ValueError("stencil dimension and shape disagree")
        if self.kernel not in KERNELS:
            raise ValueError(f"unknown kernel {self.kernel!r}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    @property
    def n_q(self) -> int:
        return int(len(STENCILS[self.stencil]))

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.shape))

    @property
    def itemsize(self) -> int:
        return 4 if self.dtype == "float32" else 8

    def as_dict(self) -> dict:
        return {"stencil": self.stencil, "shape": list(self.shape), "nu": self.nu,
                "kernel": self.kernel, "backend": self.backend,
                "dtype": self.dtype, "n_nodes": self.n_nodes}

    def ident(self) -> str:
        return (f"{self.kernel}/{self.backend}/{self.dtype}/"
                f"{self.stencil}/{'x'.join(str(s) for s in self.shape)}")


class CorePath:
    """Unified collision+streaming driver over (kernel, backend, dtype).

    Periodic in every axis.  ``step(n)`` advances n lattice steps without
    reallocating the *state* buffers: :attr:`steady_state_allocations` counts
    allocations made through the back-end layer after construction (0 for both
    kernels once the workspace exists).
    """

    def __init__(self, cfg: CoreConfig):
        self.cfg = cfg
        self.ops = make_ops(cfg.backend, cfg.dtype)
        self.stencil = STENCILS[cfg.stencil]
        self.shape = tuple(int(s) for s in cfg.shape)
        self.n_q = cfg.n_q
        self.dim = len(self.shape)
        self.w, _ = weights_cs(self.stencil)
        self.cs2 = 1.0 / 3.0
        self.omega = 1.0 / (3.0 * cfg.nu + 0.5)
        self._alloc_count = 0
        self._roll_calls = 0
        self._plan = (wrap_plan(self.stencil, self.shape)
                      if cfg.kernel == "fused_slices" else None)
        self._build_buffers()
        self._alloc_at_init = self._alloc_count
        self.n_steps = 0

    # -- buffers -----------------------------------------------------------
    def _state_buffer(self, shape):
        """Allocate one Q*N state buffer and account for it."""
        self._alloc_count += 1
        return self.ops.zeros(shape)

    def _build_buffers(self) -> None:
        qn = (self.n_q,) + self.shape
        self.f = self._state_buffer(qn)
        self._dst = self._state_buffer(qn)
        self._rho = self.ops.zeros(self.shape)
        self._u = self.ops.zeros((self.dim,) + self.shape)
        self._usq = self.ops.zeros(self.shape)
        self._cu = self.ops.zeros(self.shape)
        self._tmp = self.ops.zeros(self.shape)
        if self.cfg.kernel == "reference":
            self._feq = self._state_buffer(qn)
            self._fpost = self._state_buffer(qn)

    def declared_live_bytes(self) -> int:
        """Bytes of *state* the kernel keeps live, computed from the config.

        fused     : f + dst                 = 2 Q N
        reference : f + dst + f_eq + f_post  = 4 Q N
        plus the (dim + 3) N moment workspace both need.
        """
        n, qn = self.cfg.n_nodes, self.n_q * self.cfg.n_nodes
        state = 4 * qn if self.cfg.kernel == "reference" else 2 * qn
        return int((state + (self.dim + 3) * n) * self.cfg.itemsize)

    @property
    def steady_state_allocations(self) -> int:
        """Q*N state buffers allocated by ``step`` after construction (must be 0).

        Per-population ``roll`` temporaries are *not* state buffers; they are
        accounted for by the measured ``tracemalloc`` peak instead of hidden.
        """
        return self._alloc_count - self._alloc_at_init

    @property
    def roll_calls_per_step(self) -> float:
        """Full-field/N-temporary ``roll`` calls per step (deterministic count).

        ``fused_slices`` must be 0: it streams through slice views of the
        precomputed wrap plan instead of materialising a rolled copy.
        """
        return self._roll_calls / self.n_steps if self.n_steps else 0.0

    def state_buffer_ids(self) -> set:
        """Identities of the Q*N state buffers currently held (buffer-reuse proof)."""
        ids = {id(self.f), id(self._dst)}
        for name in ("_feq", "_fpost"):
            if hasattr(self, name):
                ids.add(id(getattr(self, name)))
        return ids

    # -- moments / equilibrium --------------------------------------------
    def _moments(self) -> None:
        f, ops = self.f, self.ops
        rho, u = self._rho, self._u
        ops.copyto(rho, f[0])
        for q in range(1, self.n_q):
            rho += f[q]
        for a in range(self.dim):
            if not bool(np.any(self.stencil[:, a] != 0)):
                continue
            ua = u[a]
            ops.fill(ua, 0.0)
            for q in range(self.n_q):
                c = int(self.stencil[q, a])
                if c:
                    ua += (c * f[q])
            ua /= rho
        usq = self._usq
        ops.copyto(usq, u[0] * u[0])
        for a in range(1, self.dim):
            usq += u[a] * u[a]

    def _equilibrium_into(self, q: int, out) -> None:
        """Write ``f_eq[q]`` into the N-sized buffer ``out`` (no Q*N temporary)."""
        ops = self.ops
        cu = self._cu
        ops.fill(cu, 0.0)
        for a in range(self.dim):
            c = int(self.stencil[q, a])
            if c:
                cu += (c * self._u[a])
        ops.copyto(out, self._rho)
        out *= float(self.w[q])
        out *= (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * self._usq)

    # -- kernels -----------------------------------------------------------
    def _stream_shifts(self, q: int) -> Tuple[int, ...]:
        return tuple(int(self.stencil[q, a]) for a in range(self.dim))

    def _step_fused(self) -> None:
        self._moments()
        f, dst, ops = self.f, self._dst, self.ops
        omega = self.omega
        for q in range(self.n_q):
            tmp = self._tmp
            self._equilibrium_into(q, tmp)
            fq = f[q]
            tmp -= fq          # eq - f
            tmp *= omega
            tmp += fq          # post-collision population q, never stored as Q*N
            ops.copyto(dst[q], ops.roll(tmp, self._stream_shifts(q)))
            self._roll_calls += 1
        self.f, self._dst = dst, f

    def _step_fused_slices(self) -> None:
        """Fused sweep that streams through slice views (no per-population roll copy).

        Same arithmetic as ``_step_fused``; the only difference is that the
        periodic wrap is applied with the precomputed ``wrap_plan`` box copies,
        so no Q*N/N roll temporary is materialised per step.
        """
        self._moments()
        f, dst, ops = self.f, self._dst, self.ops
        omega = self.omega
        for q in range(self.n_q):
            tmp = self._tmp
            self._equilibrium_into(q, tmp)
            fq = f[q]
            tmp -= fq
            tmp *= omega
            tmp += fq
            dq = dst[q]
            for dst_s, src_s, _seam, _n in self._plan[q]:
                ops.copy_slices(dq, tmp, dst_s, src_s)
        self.f, self._dst = dst, f

    def _step_reference(self) -> None:
        self._moments()
        ops = self.ops
        feq, fpost, dst, f = self._feq, self._fpost, self._dst, self.f
        for q in range(self.n_q):
            self._equilibrium_into(q, feq[q])
        ops.copyto(fpost, f)
        fpost -= feq
        fpost *= -self.omega
        fpost += f
        for q in range(self.n_q):
            ops.copyto(dst[q], ops.roll(fpost[q], self._stream_shifts(q)))
            self._roll_calls += 1
        self.f, self._dst = dst, f

    # -- public API --------------------------------------------------------
    def init_from_velocity(self, u, rho: float = 1.0) -> "CorePath":
        u = np.asarray(u, dtype=np.float64)
        rho_arr = np.full(self.shape, float(rho))
        feq = equilibrium(self.stencil, rho_arr, u)
        # re-use the existing state buffers: no reallocation of Q*N memory
        self.ops.copyto(self.f, self.ops.from_numpy(
            feq.reshape((self.n_q,) + self.shape)))
        self.ops.fill(self._dst, 0.0)
        self.n_steps = 0
        return self

    def step(self, n: int = 1) -> "CorePath":
        if self.cfg.kernel == "fused":
            kernel = self._step_fused
        elif self.cfg.kernel == "fused_slices":
            kernel = self._step_fused_slices
        else:
            kernel = self._step_reference
        for _ in range(int(n)):
            kernel()
        self.n_steps += int(n)
        self.ops.sync()
        return self

    def macroscopic(self) -> Tuple[np.ndarray, np.ndarray]:
        return macroscopic(self.stencil,
                           self.ops.to_numpy(self.f).astype(np.float64))

    def velocity(self) -> np.ndarray:
        return self.macroscopic()[1]


# ---------------------------------------------------------------------------
# Unified comparison protocol
# ---------------------------------------------------------------------------

@dataclass
class ArmResult:
    cfg: dict
    steps: int
    seconds: float
    steps_per_second: float
    mlups: float
    decay_rate_fit: float
    decay_rate_analytic: float
    accuracy_rel_err: float
    consistency_rel_diff: Optional[float]
    consistency_tol: Optional[float]
    declared_live_bytes: int
    measured_peak_bytes: int
    steady_state_allocations: int
    accuracy_pass: bool
    consistency_pass: Optional[bool]
    perf_admissible: bool
    verdict: str
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["config"] = d.pop("cfg")
        d["declared_live_bytes"] = int(self.declared_live_bytes)
        d["measured_peak_bytes"] = int(self.measured_peak_bytes)
        d["steady_state_allocations"] = int(self.steady_state_allocations)
        return d


def admissibility(accuracy_rel_err: float,
                  consistency_rel_diff: Optional[float] = None,
                  consistency_tol: Optional[float] = None) -> dict:
    """Apply the pre-registered gates.  Only this function may set the verdict.

    * accuracy: ``|rel_err| <= ACCURACY_TOL`` on the T3 case, else the arm has
      no validated physics and its timing is meaningless (``REJECTED_ACCURACY``).
    * consistency: a fused arm must reproduce the reference arm's velocity field
      to ``consistency_tol`` — a "fused" kernel that changes the solution is a
      different solver, not an optimisation (``REJECTED_CONSISTENCY``).
    * only arms passing both are ``ADMISSIBLE`` and may publish a MLUPS number.
    """
    reasons: List[str] = []
    accuracy_pass = abs(float(accuracy_rel_err)) <= ACCURACY_TOL
    if not accuracy_pass:
        reasons.append(f"accuracy |rel_err|={abs(float(accuracy_rel_err)):.3e} "
                       f"> {ACCURACY_TOL:.1e} on T3 case {T3_CASE}")

    consistency_pass: Optional[bool] = None
    if consistency_rel_diff is not None:
        tol = CONSISTENCY_TOL if consistency_tol is None else float(consistency_tol)
        consistency_pass = abs(float(consistency_rel_diff)) <= tol
        if not consistency_pass:
            reasons.append(f"consistency |Δu|/|u|={abs(float(consistency_rel_diff)):.3e}"
                           f" > {tol:.1e}")

    if not accuracy_pass:
        verdict = "REJECTED_ACCURACY"
    elif consistency_pass is False:
        verdict = "REJECTED_CONSISTENCY"
    else:
        verdict = "ADMISSIBLE"
    return {"accuracy_pass": accuracy_pass, "consistency_pass": consistency_pass,
            "perf_admissible": verdict == "ADMISSIBLE", "verdict": verdict,
            "reasons": reasons}


def tg_decay_rate(ux_history: Sequence[np.ndarray], times: Sequence[float],
                  mode: Tuple[int, ...]) -> float:
    """Fitted decay rate of the Taylor-Green fundamental mode (positive number)."""
    amps = [mode_amplitude(np.asarray(h, dtype=np.float64), mode) for h in ux_history]
    return fit_exponential_rate(times, amps)


def run_arm(cfg: CoreConfig, steps: int = 1200, u0: float = 0.01,
            record_every: int = 100, repeat: int = 3,
            reference_field: Optional[np.ndarray] = None,
            measure_memory: bool = True) -> Tuple[ArmResult, np.ndarray]:
    """Run one arm of the unified protocol on the T3 case and judge it.

    Timing protocol: one warm-up block, then ``repeat`` timed blocks of
    ``steps`` steps; the reported time is the **minimum** block (least noise).
    The amplitude history used for the accuracy fit comes from a separate,
    untimed run, so timing never contaminates the physics.
    """
    if cfg.stencil != "D2Q9" or len(cfg.shape) != 2:
        raise ValueError("the T3 taylor-green-2d reference is D2Q9 in 2-D only")
    n = int(cfg.shape[0])
    if int(cfg.shape[1]) != n:
        raise ValueError("taylor-green-2d needs a square grid")
    k = 2.0 * np.pi / n
    u_field = taylor_green_2d((n, n), amp=u0, k=k)
    analytic = 2.0 * cfg.nu * k * k
    n_rec = max(1, steps // record_every)
    times = [i * float(record_every) for i in range(n_rec + 1)]

    peak = 0
    if measure_memory:
        tracemalloc.start()
    solver = CorePath(cfg).init_from_velocity(u_field)
    hist = [solver.velocity()[0].copy()]
    for _ in range(n_rec):
        solver.step(record_every)
        hist.append(solver.velocity()[0].copy())
    if measure_memory:
        peak = int(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    u_final = solver.velocity()
    fit = abs(tg_decay_rate(hist, times, (1, 1)))
    rel_err = abs(fit - analytic) / analytic

    perf = CorePath(cfg).init_from_velocity(u_field)
    perf.step(record_every)                       # warm-up
    perf.ops.sync()
    best = float("inf")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        perf.step(steps)
        perf.ops.sync()
        best = min(best, time.perf_counter() - t0)
    sps = steps / best

    consistency_rel: Optional[float] = None
    consistency_tol: Optional[float] = None
    if reference_field is not None:
        scale = max(float(np.max(np.abs(reference_field))), 1e-30)
        consistency_rel = float(np.max(np.abs(u_final - reference_field))) / scale
        consistency_tol = DTYPE_CONSISTENCY_TOL[cfg.dtype]
    judge = admissibility(rel_err, consistency_rel, consistency_tol)

    res = ArmResult(
        cfg=cfg.as_dict(), steps=steps, seconds=best, steps_per_second=sps,
        mlups=sps * cfg.n_nodes / 1e6,
        decay_rate_fit=fit, decay_rate_analytic=analytic, accuracy_rel_err=rel_err,
        consistency_rel_diff=consistency_rel, consistency_tol=consistency_tol,
        declared_live_bytes=solver.declared_live_bytes(),
        measured_peak_bytes=peak,
        steady_state_allocations=solver.steady_state_allocations,
        accuracy_pass=judge["accuracy_pass"],
        consistency_pass=judge["consistency_pass"],
        perf_admissible=judge["perf_admissible"],
        verdict=judge["verdict"], reasons=judge["reasons"],
    )
    return res, u_final