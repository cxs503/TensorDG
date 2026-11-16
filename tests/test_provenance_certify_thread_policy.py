"""Tests for the replay thread policy in ``tools/provenance_certify.py``.

Round 18 of CASE-DG-EULER-VORTEX-ERR isolated the last blocking failure of the
C11 provenance gate: the checkpoint ``dg_d2_n64_t10_cfl0.2`` was killed with
SIGSEGV inside ``torch.einsum`` (``tensordg/euler_dg_2d.py::_traces``) when the
replay ran with ``OMP_NUM_THREADS`` >= 4, twice, on two different nodes/counts
(``cj4121`` shard 1/4 at 8 threads, ``cj4186`` at 4 threads), while every replay
attempted with one thread completed.

The two claims that make a single-thread retry *legitimate evidence* rather than
a workaround are tested here as far as a unit test can:

1. the retry policy itself -- a key killed by a signal under n > 1 threads is
   retried exactly once on one thread, a *clean* failure is never retried, and a
   crash at one thread stays a crash (no infinite retry, no silent success);
2. the certificate must declare the thread count it was produced with, so a
   replay whose reduction order is unknown cannot pass as evidence.

The numerical claim -- that the replay is thread-invariant, so a one-thread
certificate certifies the same stored error as a four-thread one -- is not a
unit-testable property; it is measured in
``docs/CASE-DG-EULER-VORTEX-ERR-round18-c11-thread-policy.md``
(``dg_d2_n32_t2_cfl0.2`` replayed to ``replay_delta`` 8.14e-13 at both 1 and 4
threads, wall 86.9 s vs 90.2 s).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import provenance_certify as pc  # noqa: E402


def test_signal_death_on_many_threads_is_retried_once_on_one_thread():
    """The observed cluster failure: SIGSEGV at 4 threads, clean at 1."""
    attempts = []

    def run(key, threads):
        attempts.append((key, threads))
        return -11 if threads > 1 else 0

    status = pc.isolated_batch(["dg_d2_n64_t10_cfl0.2"], run, threads=4,
                               fallback_threads=1, echo=lambda *a, **k: None)

    assert attempts == [("dg_d2_n64_t10_cfl0.2", 4), ("dg_d2_n64_t10_cfl0.2", 1)]
    assert status["dg_d2_n64_t10_cfl0.2"] == "ok (retried on threads=1)"
    assert status["dg_d2_n64_t10_cfl0.2"].startswith("ok"), \
        "a completed retry is landed evidence, but it must say how it was produced"


def test_crash_at_one_thread_is_reported_as_a_crash_not_certified():
    """No retry ladder downwards: a dead single-threaded replay stays dead."""
    calls = []

    def run(key, threads):
        calls.append(threads)
        return -11

    status = pc.isolated_batch(["dg_d2_n64_t10_cfl0.2"], run, threads=1,
                               fallback_threads=1, echo=lambda *a, **k: None)

    assert calls == [1], "single-threaded replay must not be retried"
    assert status["dg_d2_n64_t10_cfl0.2"] == "crashed (signal 11)"


def test_clean_failure_is_not_retried():
    """A non-zero exit is a real error (bad config, budget, exception) -- not a
    threading accident, so retrying it would only hide the message."""
    calls = []

    def run(key, threads):
        calls.append(threads)
        return 2

    status = pc.isolated_batch(["k"], run, threads=4, fallback_threads=1,
                               echo=lambda *a, **k: None)

    assert calls == [4]
    assert status["k"] == "failed (rc 2)"


def test_retry_can_be_disabled():
    calls = []

    def run(key, threads):
        calls.append(threads)
        return -11

    status = pc.isolated_batch(["k"], run, threads=4, fallback_threads=None,
                               echo=lambda *a, **k: None)

    assert calls == [4]
    assert status["k"] == "crashed (signal 11)"


def test_ok_at_requested_thread_count_is_not_relabelled():
    status = pc.isolated_batch(["k"], lambda key, threads: 0, threads=4,
                               fallback_threads=1, echo=lambda *a, **k: None)
    assert status["k"] == "ok", "a first-attempt success must not claim a retry"


def test_replay_certificate_declares_the_thread_count(tmp_path, monkeypatch):
    """The thread count is part of the evidence, so it must be in the record."""
    record = {
        "task_key": "dg_d2_n32_t1_cfl0.2",
        "config": {"task": "dg", "cells": 32, "degree": 2,
                   "final_time": 1.0, "cfl": 0.2, "strength": 5.0},
        "errors": {"density": 0.1, "velocity_x": 0.1,
                   "velocity_y": 0.1, "pressure": 0.1},
        "max_percent": 0.05,
    }
    fresh = dict(record, errors=dict(record["errors"]), max_percent=0.05,
                 reference_model="min_image", steps=7)

    monkeypatch.setattr(pc.ec, "run_task", lambda config: fresh)

    certificate = pc.replay(record, threads=1)
    assert certificate["threads"] == 1
    assert certificate["replay_delta"] == 0.0
    assert certificate["tolerance"] == pc.ec.PROVENANCE_REPLAY_TOLERANCE


def test_default_replay_thread_count_is_one():
    """Default policy: replays run single-threaded (thread-invariance measured,
    no speed-up from threads, and the only crash mode observed needs threads>1)."""
    args = pc.build_parser().parse_args([])
    assert args.threads == 1
    assert args.fallback_threads == 1