"""Stack fingerprinting and thread pinning for reproducible numerical artifacts.

Numerical artifacts in this repository (``docs/benchmarks/results/*/metrics.json``)
quote errors at the last-bit level, but a rerun of the same leg on another
machine lands a few ulp -- or, once a wave field starts to steepen, a few
1e-4 -- away, because torch/numpy kernels reduce in a different order and use
different vector widths on every stack.

Two small tools are enough to keep an artifact's claim honest:

* :func:`environment_fingerprint` -- stamped into the artifact so the stack the
  numbers belong to is part of the record (``python``/``numpy``/``torch``
  versions, machine, CPU model, thread count);
* :func:`pinned_threads` -- a context manager that fixes the reduction topology
  for the duration of a measurement, so *same stack* reruns are bit-identical
  and a drift can only be blamed on the stack, not on an ambient thread setting.

Neither tool changes a number by itself; they make the numbers attributable.
"""
from __future__ import annotations

import contextlib
import platform

import numpy as np
import torch

__all__ = ["environment_fingerprint", "pinned_threads", "cpu_model"]


def cpu_model() -> str:
    """Best-effort CPU model string (empty when the platform does not expose it)."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith(("model name", "cpu model", "processor")) and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value and not value.isdigit():
                        return value
    except OSError:
        pass
    return platform.processor()


def environment_fingerprint() -> dict:
    """The stack a measured number belongs to."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "torch_threads": int(torch.get_num_threads()),
    }


@contextlib.contextmanager
def pinned_threads(threads: int = 1):
    """Pin torch's intra-op thread count for the duration of a measurement."""
    threads = max(1, int(threads))
    previous = int(torch.get_num_threads())
    torch.set_num_threads(threads)
    try:
        yield threads
    finally:
        torch.set_num_threads(previous)