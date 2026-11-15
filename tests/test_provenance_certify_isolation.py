"""Tests for crash-tolerant isolated replay in ``tools/provenance_certify.py``.

The pointed question: when one replay dies, does the batch keep the evidence it
already produced?  Round 15 answered it the hard way -- three of five replays
were lost while the shard jobs still reported ``succeeded``, because a single
in-process crash ends the whole invocation.  ``--isolate`` puts every replay in
its own subprocess; these tests segfault a *real* child to prove the parent
survives and that a dead key is reported as dead, never as certified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools import provenance_certify as pc  # noqa: E402


def _script(kill):
    body = "os.kill(os.getpid(), signal.SIGSEGV)" if kill else "print('child done')"
    return f"import os, signal, sys\n{body}\n"


def test_isolated_status_survives_a_segfaulting_child(capsys):
    """A key killed by a signal must be reported ``crashed`` and not stop the rest."""
    attempted = []

    def run(key):
        attempted.append(key)
        # A real child process, really killed by SIGSEGV -- this is the failure
        # mode observed on the cluster (rc reported as success by the job wrapper).
        import subprocess
        proc = subprocess.run([sys.executable, "-c", _script(key == "boom")])
        return proc.returncode

    status = pc.isolated_status(["ok_a", "boom", "ok_b"], run=run)

    assert attempted == ["ok_a", "boom", "ok_b"], "the batch must continue past a crash"
    assert status["ok_a"] == "ok" and status["ok_b"] == "ok"
    assert status["boom"].startswith("crashed (signal "), status
    assert status["boom"] == "crashed (signal 11)", "SIGSEGV is signal 11"
    out = capsys.readouterr().out
    assert "isolated boom: crashed (signal 11)" in out


def test_isolated_status_reports_nonzero_exit_as_failure():
    status = pc.isolated_status(["a", "b"], run=lambda key: 0 if key == "a" else 3)
    assert status == {"a": "ok", "b": "failed (rc 3)"}


def test_key_command_targets_exactly_one_checkpoint():
    cmd = pc.key_command("dg_d2_n64_t2_cfl0.2", Path("out"), threads=2, max_seconds=60)
    assert cmd[1:3] == ["-m", "tools.provenance_certify"]
    assert cmd[cmd.index("--keys") + 1] == "dg_d2_n64_t2_cfl0.2"
    assert cmd[cmd.index("--threads") + 1] == "2"
    # one key per child: the parent owns the batch, the child owns one replay
    assert cmd.count("--keys") == 1


def test_isolate_batch_writes_nothing_itself_and_reloads_from_disk(tmp_path, monkeypatch):
    """The parent must report progress from the bundle the children wrote."""
    results = tmp_path / "results"
    results.mkdir()
    keys = ["dg_d2_n32_t1_cfl0.2", "dg_d2_n32_t2_cfl0.2"]
    for key, final_time in zip(keys, (1.0, 2.0)):
        (results / f"{key}.json").write_text(json.dumps({
            "task_key": key, "reference_model": None,
            "config": {"task": "dg", "cells": 32, "degree": 2,
                       "final_time": final_time, "cfl": 0.2, "strength": 5.0},
            "errors": {"density": 0.1}, "max_percent": 0.01}))

    certificate_path = tmp_path / "provenance_certificates.json"

    def fake_run_key(key, output_dir, threads, max_seconds):
        """First key certifies itself, second one crashes before writing."""
        if key == keys[1]:
            return -11
        bundle = (json.loads(certificate_path.read_text())
                  if certificate_path.exists() else {"certificates": {}})
        bundle["certificates"][key] = {"task_key": key, "replay_delta": 0.0}
        certificate_path.write_text(json.dumps(bundle))
        return 0

    monkeypatch.setattr(pc, "run_key", fake_run_key)
    rc = pc.main(["--output-dir", str(tmp_path), "--remaining", "--isolate"])
    assert rc == 0
    bundle = json.loads(certificate_path.read_text())
    assert set(bundle["certificates"]) == {keys[0]}, "only the completed replay is evidence"