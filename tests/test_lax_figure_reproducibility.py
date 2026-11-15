"""Gate: the committed ``CASE-LAX-DG`` figures are bound to the committed records.

The record says (in ``figures_provenance``) that its figures were rendered from
its own numbers.  That is a claim, and a claim about binary files is worth
nothing unless something checks bytes.  Five gates, cheapest first:

``G1`` record / profiles / renderer / figure bytes all agree with the sidecar
      ``figures_binding.json``, and record / disk / digest sets are equal
      (no matplotlib needed);
``G2`` a one-byte tamper in a *copy* of a figure must turn G1 to FAIL;
``G3`` a one-field edit in a *copy* of the record (a re-run whose figures were
      never re-rendered) must also turn G1 to FAIL;
``G4`` the renderer's writer reproduces a record byte for byte (no cosmetic
      churn, so a no-op render shows no diff);
``G5`` a fresh render from the committed record + ``profiles.npz`` reproduces
      every committed PNG byte for byte (needs matplotlib).

Run: ``python -m pytest tests/test_lax_figure_reproducibility.py -q``
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.render_lax_figures import (  # noqa: E402
    BINDING, PROVENANCE, RECORDS, check_binding, render_arms_record,
    render_pool, sha256_file, write_record,
)
from scripts import render_lax_figure_doc  # noqa: E402

LABELS = [label for _, label in RECORDS]
RENDERERS = {"pool": render_pool, "arms": render_arms_record}


def _rec_dir(label: str) -> Path:
    for rec, name in RECORDS:
        if name == label:
            return ROOT / rec
    raise KeyError(label)


def _copy(label: str, tmp_path: Path, *, records: bool = True) -> Path:
    """Copy one record (sidecar, figures, profiles) under its repo path."""
    src = _rec_dir(label)
    dst = tmp_path / src.relative_to(ROOT)
    (dst / "figures").mkdir(parents=True)
    shutil.copy(src / "metrics.json", dst / "metrics.json")
    shutil.copy(src / "profiles.npz", dst / "profiles.npz")
    shutil.copy(src / BINDING, dst / BINDING)
    for png in (src / "figures").glob("*.png"):
        shutil.copy(png, dst / "figures" / png.name)
    if not records:
        (dst / "metrics.json").unlink()
    return dst


def _mirror(label: str, tmp_path: Path) -> Path:
    """Copy only the inputs a render needs (no figures, no sidecar)."""
    src = _rec_dir(label)
    dst = tmp_path / src.relative_to(ROOT)
    dst.mkdir(parents=True)
    shutil.copy(src / "metrics.json", dst / "metrics.json")
    shutil.copy(src / "profiles.npz", dst / "profiles.npz")
    return tmp_path


# --------------------------------------------------------------- G1 (cheap) --
@pytest.mark.parametrize("label", LABELS)
def test_sidecar_binds_every_committed_figure(label):
    rec = _rec_dir(label)
    metrics = json.loads((rec / "metrics.json").read_text())
    binding = json.loads((rec / BINDING).read_text())
    names = metrics.get("figures") or []
    assert names, f"{label}: the record must list the figures it produced"
    assert sorted(binding["figures_sha256"]) == sorted(names)
    assert sorted(binding["figures"]) == sorted(names)
    on_disk = sorted(p.name for p in (rec / "figures").glob("*.png"))
    assert on_disk == sorted(names), (
        f"{label}: record names {sorted(names)} but disk holds {on_disk}")
    for name in names:
        assert binding["figures_sha256"][name] == sha256_file(rec / "figures" / name)


@pytest.mark.parametrize("label", LABELS)
def test_committed_artifacts_pass_the_binding_gate(label):
    ok, rows = check_binding(_rec_dir(label))
    bad = [r for r in rows if r[1] is None or r[1] != r[2]]
    assert ok, f"{label}: binding mismatch on {bad}"


@pytest.mark.parametrize("label", LABELS)
def test_sidecar_pins_the_record_and_the_renderer(label):
    rec = _rec_dir(label)
    binding = json.loads((rec / BINDING).read_text())
    assert binding["record_sha256"] == sha256_file(rec / "metrics.json")
    assert binding["profiles_sha256"] == sha256_file(rec / "profiles.npz")
    assert binding["renderer_sha256"] == sha256_file(ROOT / binding["renderer"]), (
        "the renderer changed after the figures were drawn -- re-render them")


# ------------------------------------------------------------ G2 / G3 control --
@pytest.mark.parametrize("label", LABELS)
def test_a_one_byte_figure_tamper_is_caught(label, tmp_path):
    """Negative control: the gate must be able to say FAIL."""
    rec = _copy(label, tmp_path)
    assert check_binding(rec)[0] is True, "the clean copy must pass G1"

    target = sorted((rec / "figures").glob("*.png"))[0]
    blob = bytearray(target.read_bytes())
    blob[-1] ^= 0xFF                # last byte of the IEND chunk
    target.write_bytes(bytes(blob))
    ok, rows = check_binding(rec)
    assert ok is False, "a flipped byte in a committed figure went unnoticed"
    assert any(r[1] != r[2] for r in rows if r[0].startswith("figures/"))


@pytest.mark.parametrize("label", LABELS)
def test_a_record_change_without_a_re_render_is_caught(label, tmp_path):
    """The direction an in-record digest cannot see: record moved, figures did not."""
    rec = _copy(label, tmp_path)
    assert check_binding(rec)[0] is True
    metrics = json.loads((rec / "metrics.json").read_text())
    metrics["generated_at_utc"] = "1999-01-01T00:00:00Z"
    write_record(rec / "metrics.json", metrics)
    ok, rows = check_binding(rec)
    assert ok is False, "a stale record/figures pair passed the gate"
    assert any(r[0] == "record_sha256" and r[1] != r[2] for r in rows)


@pytest.mark.parametrize("label", LABELS)
def test_missing_orphan_and_unbound_figures_fail(label, tmp_path):
    """Three ways for the binding to rot; each must be a FAIL, not a skip."""
    missing = _copy(label, tmp_path / "missing", records=False)
    missing.mkdir(parents=True, exist_ok=True)
    ok, _ = check_binding(missing)
    assert ok is False, "a record that is not there passed the gate"

    orphan = tmp_path / "orphan" / _rec_dir(label).relative_to(ROOT)
    (orphan / "figures").mkdir(parents=True)
    shutil.copy(_rec_dir(label) / "metrics.json", orphan / "metrics.json")
    shutil.copy(_rec_dir(label) / "profiles.npz", orphan / "profiles.npz")
    shutil.copy(_rec_dir(label) / BINDING, orphan / BINDING)
    for png in (_rec_dir(label) / "figures").glob("*.png"):
        shutil.copy(png, orphan / "figures" / png.name)
    assert check_binding(orphan)[0] is True
    shutil.copy(next((_rec_dir(label) / "figures").glob("*.png")),
                orphan / "figures" / "not_in_the_record.png")
    assert check_binding(orphan)[0] is False, "an unlisted figure passed"

    unbound = _copy(label, tmp_path / "unbound")
    (unbound / BINDING).unlink()
    assert check_binding(unbound)[0] is False, (
        "a record with no sidecar passed the gate silently")


# ------------------------------------------------------------ G4 (writer) ---
@pytest.mark.parametrize("label", LABELS)
def test_writer_round_trip_leaves_the_record_untouched(label, tmp_path):
    """Re-serialising a record must not touch a byte (no cosmetic churn)."""
    src = _rec_dir(label) / "metrics.json"
    copy = tmp_path / "metrics.json"
    copy.write_text(src.read_text())
    write_record(copy, json.loads(src.read_text()))
    assert copy.read_bytes() == src.read_bytes(), (
        f"{label}: re-rendering rewrites the record without any change")


# ------------------------------------------------------------- G5 (render) ---
@pytest.mark.parametrize("label", LABELS)
def test_fresh_render_reproduces_the_committed_figures(label, tmp_path):
    """The strongest statement: same record in, same pixels out."""
    pytest.importorskip("matplotlib")
    root = _mirror(label, tmp_path)
    names = RENDERERS[label](root)
    rec = _rec_dir(label)
    committed = json.loads((rec / "metrics.json").read_text())
    assert names == sorted(committed["figures"])
    for name in names:
        fresh = root / rec.relative_to(ROOT) / "figures" / name
        assert fresh.is_file(), f"{name} was not rendered"
        assert sha256_file(fresh) == sha256_file(rec / "figures" / name), (
            f"{name}: a fresh render from the committed record differs from the "
            "committed bytes -- the figure is not the one this record explains")
    # the fresh run also wrote the record back unchanged and a matching sidecar
    assert (root / rec.relative_to(ROOT) / "metrics.json").read_bytes() == (
        rec / "metrics.json").read_bytes(), "a re-render rewrote the record"
    fresh_binding = json.loads(
        (root / rec.relative_to(ROOT) / BINDING).read_text())
    assert fresh_binding["figures_sha256"] == json.loads(
        (rec / BINDING).read_text())["figures_sha256"]


def test_provenance_names_the_renderer():
    for rec, label in RECORDS:
        metrics = json.loads((ROOT / rec / "metrics.json").read_text())
        assert metrics.get("figures_provenance") == PROVENANCE, label


def test_note_is_a_pure_function_of_the_records():
    assert render_lax_figure_doc.main(["--check"]) == 0