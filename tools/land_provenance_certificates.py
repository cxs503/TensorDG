"""Land replay certificates produced on another machine into the local bundle.

Why this is a separate, strict step
-----------------------------------
A certificate is a *claim* that a legacy record's stored errors were reproduced
against the current reference.  The claim is only usable by gate C11 if it is
about the same checkpoint the bundle actually aggregates.  Copying a
``provenance_certificates.json`` from a fetched artifact therefore has to be
filtered, not trusted:

* the checkpoint must exist locally (an orphan certificate is not evidence);
* the certificate's configuration must match the local record's configuration on
  the physics fields (task/degree/cells/cfl/final_time/strength) -- a certificate
  for a *different* mesh is the exact failure mode this case was built to catch;
* it must name the aggregated reference model, and its replay delta must be
  finite and within ``PROVENANCE_REPLAY_TOLERANCE``;
* the replayed percentage must also agree with the stored percentage to the same
  tolerance (a matching delta on different absolute numbers is suspicious);
* an existing certificate is only replaced by a strictly better one, so a
  re-run can never weaken evidence that already landed.

Rejected certificates are printed with the reason and are *not* written, so the
gate keeps reporting them ``unverified`` -- the honest outcome.

Usage::

    PYTHONPATH=. python3 tools/land_provenance_certificates.py \\
        /root/tp-agent/cluster/fetch/cj3699_n315.tar.gz --dry-run
    PYTHONPATH=. python3 tools/land_provenance_certificates.py <artifact...>
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import case_euler_vortex_errconserve as ec  # noqa: E402
from tools.provenance_certify import load_records  # noqa: E402

CERT_NAME = ec.PROVENANCE_CERTIFICATES
PHYSICS_FIELDS = ("task", "degree", "cells", "cfl", "final_time", "strength")


def certificates_from(path: Path) -> dict:
    """Read a certificate bundle from a JSON file or from a fetched tarball."""
    if path.is_dir():
        candidate = path / CERT_NAME
        if not candidate.exists():
            raise SystemExit(f"no {CERT_NAME} under {path}")
        return json.loads(candidate.read_text()).get("certificates", {})
    if path.suffix in (".tar", ".gz", ".tgz", ".tar.gz"):
        with tarfile.open(path) as archive:
            member = next((name for name in archive.getnames()
                           if name.endswith("/" + CERT_NAME)), None)
            if member is None:
                raise SystemExit(f"no {CERT_NAME} inside {path}")
            handle = archive.extractfile(member)
            return json.loads(handle.read().decode()).get("certificates", {})
    return json.loads(path.read_text()).get("certificates", {})


def check(certificate: dict, record: dict) -> str | None:
    """Return a rejection reason, or ``None`` when the certificate is usable."""
    if certificate.get("reference_model") != ec.REFERENCE_MODEL:
        return (f"replayed against {certificate.get('reference_model')!r}, "
                f"aggregate is {ec.REFERENCE_MODEL!r}")
    cert_config = certificate.get("config") or {}
    for field in PHYSICS_FIELDS:
        want = record.get("config", {}).get(field)
        have = cert_config.get(field)
        if have is None and want is None:
            continue
        if have != want:
            return f"config mismatch on {field}: record {want!r} vs certificate {have!r}"
    delta = certificate.get("replay_delta")
    if delta is None or delta != delta or abs(delta) > ec.PROVENANCE_REPLAY_TOLERANCE:
        return f"replay delta {delta!r} > {ec.PROVENANCE_REPLAY_TOLERANCE:g}"
    stored, replayed = certificate.get("stored_max_percent"), certificate.get("replayed_max_percent")
    if stored is None or replayed is None:
        return "certificate does not carry both stored and replayed percentages"
    scale = max(abs(stored), abs(replayed)) or 1.0
    if abs(stored - replayed) / scale > ec.PROVENANCE_REPLAY_TOLERANCE:
        return f"stored {stored!r}% vs replayed {replayed!r}% exceeds tolerance"
    return None


def land(certificates: dict, output_dir: Path, dry_run: bool = False,
         source: str = "artifact") -> dict:
    records = {record["task_key"]: record for record in load_records(output_dir / "results")}
    path = output_dir / CERT_NAME
    bundle = json.loads(path.read_text()) if path.exists() else {"certificates": {}}
    landed, rejected, kept = [], [], []

    for key in sorted(certificates):
        certificate = certificates[key]
        record = records.get(key)
        if record is None:
            rejected.append((key, "no such record in the local bundle"))
            continue
        reason = check(certificate, record)
        if reason is not None:
            rejected.append((key, reason))
            continue
        existing = bundle["certificates"].get(key)
        if existing is not None:
            old, new = existing.get("replay_delta"), certificate.get("replay_delta")
            if old is not None and new is not None and old <= new:
                kept.append((key, f"existing certificate already at least as good "
                                  f"({old:.3e} <= {new:.3e})"))
                continue
        bundle["certificates"][key] = certificate
        landed.append((key, certificate.get("replay_delta"),
                       certificate.get("stored_max_percent")))

    bundle["reference_model"] = ec.REFERENCE_MODEL
    bundle["tolerance"] = ec.PROVENANCE_REPLAY_TOLERANCE
    if not dry_run and landed:
        path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    report = {"source": source, "landed": landed, "rejected": rejected,
              "kept_existing": kept, "certificates_on_disk": len(bundle["certificates"]),
              "dry_run": dry_run}
    for key, reason in rejected:
        print(f"REJECTED {key}: {reason}")
    for key, delta, stored in landed:
        print(f"landed   {key}: stored {stored:.12g}% rel {delta:.3e}")
    for key, reason in kept:
        print(f"kept     {key}: {reason}")
    print(f"{source}: landed {len(landed)}, rejected {len(rejected)}, "
          f"kept {len(kept)}; certificates on disk {len(bundle['certificates'])}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", nargs="+", help="JSON file, directory or fetched tarball")
    parser.add_argument("--output-dir", default="case_euler_vortex_errconserve")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=None, help="write the landing report as JSON")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    reports = []
    for artifact in args.artifacts:
        certificates = certificates_from(Path(artifact))
        reports.append(land(certificates, output_dir, dry_run=args.dry_run, source=artifact))
    if args.report:
        Path(args.report).write_text(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())