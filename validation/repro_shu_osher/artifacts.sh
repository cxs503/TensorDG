#!/usr/bin/env bash
# Generate and validate the Shu--Osher case-pool artifacts.
#
# This script is the *artifact entry point* for CI and quick reproducibility:
# it runs a minimal configuration (one FV reference + one DG leg), merges the
# metrics, and validates the sha256 manifest. For the full study use
# ``run_all.sh`` instead.
#
# Usage:
#   bash validation/repro_shu_osher/artifacts.sh          # generate + validate
#   bash validation/repro_shu_osher/artifacts.sh --validate-only   # skip generation
#
# Outputs:
#   runs/shu_osher_case_pool/raw/<key>.npz      one file per configuration
#   runs/shu_osher_case_pool/metrics.json       merged comparison tables
#   runs/shu_osher_case_pool/profiles.npz       profiles for plotting
#   validation/repro_shu_osher/artifacts.sha256.json   sha256 manifest
#
# Exit code 0 iff every artifact is present and the manifest matches.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

VALIDATE_ONLY=0
if [[ "${1:-}" == "--validate-only" ]]; then
    VALIDATE_ONLY=1
fi

RUN="runs/shu_osher_case_pool"
ART="validation/repro_shu_osher/artifacts"
MANIFEST="validation/repro_shu_osher/artifacts.sha256.json"

# ----------------------------------------------------------- generation (full)
if [[ "$VALIDATE_ONLY" -eq 0 ]]; then
    echo "[artifacts.sh] generating minimal artifact set..."

    # independent FV-WENO5 reference at two resolutions (self-convergence pair)
    python -m benchmarks.shu_osher_case_pool reference --ncells 1600
    python -m benchmarks.shu_osher_case_pool reference --ncells 3200

    # one DG ladder leg (P2, N=400 - divides the reference grid)
    python -m benchmarks.shu_osher_case_pool ladder --degree 2 --ncells 400

    # merge -> metrics.json + profiles.npz
    python -m benchmarks.shu_osher_case_pool merge --reference-cells 3200

    # copy the curated artifacts into the validation bundle
    mkdir -p "$ART/raw"
    cp "$RUN/metrics.json" "$ART/metrics.json"
    cp "$RUN/profiles.npz" "$ART/profiles.npz"
    for f in "$RUN"/raw/*.npz; do
        [[ -f "$f" ]] && cp "$f" "$ART/raw/"
    done

    echo "[artifacts.sh] generation complete."
fi

# ----------------------------------------------------------- validation
echo "[artifacts.sh] validating artifacts..."

# check required files exist
required=(
    "$ART/metrics.json"
    "$ART/profiles.npz"
    "$MANIFEST"
)
for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing required artifact: $f" >&2
        exit 1
    fi
done

# check raw directory has at least the expected files
raw_count=$(find "$ART/raw" -name "*.npz" 2>/dev/null | wc -l)
if [[ "$raw_count" -lt 3 ]]; then
    echo "ERROR: expected at least 3 raw .npz files in $ART/raw, found $raw_count" >&2
    exit 1
fi

# validate sha256 manifest (format: {"filename": "sha256hex", ...})
echo "[artifacts.sh] checking sha256 manifest..."
python3 - "$ART" "$MANIFEST" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

art_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])

if not manifest_path.exists():
    print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
    sys.exit(1)

manifest = json.loads(manifest_path.read_text())
errors = 0
checked = 0

# manifest is a flat dict: {"filename": "sha256", ...}
# files live either in art_dir/ or art_dir/raw/
for filename, expected_sha in manifest.items():
    # try art_dir/filename first, then art_dir/raw/filename
    candidate = art_dir / filename
    if not candidate.exists():
        candidate = art_dir / "raw" / filename
    if not candidate.exists():
        # also try docs/figures for png files
        if filename.endswith(".png"):
            candidate = Path("docs/figures") / filename
    if not candidate.exists():
        print(f"  MISSING: {filename}", file=sys.stderr)
        errors += 1
        continue
    actual_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        print(f"  MISMATCH: {filename} (expected {expected_sha[:12]}..., got {actual_sha[:12]}...)",
              file=sys.stderr)
        errors += 1
    else:
        checked += 1

if errors:
    print(f"ERROR: {errors} artifact(s) failed validation", file=sys.stderr)
    sys.exit(1)
print(f"  OK: {checked} artifact(s) validated")
PYEOF

echo "[artifacts.sh] all artifacts valid."
exit 0
