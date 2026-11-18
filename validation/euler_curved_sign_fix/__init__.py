"""Model directory: curvilinear Euler DG interface orientation fix.

This directory is a self-contained CI unit for the validation case
``CASE-DG-CKS1998-CURVED-OPERATOR-SIGN-FIX``.

Structure
---------
- ``summary.json``  -- case description, physics, acceptance criteria, status
- ``metrics.json``  -- raw numbers (spectrum, convergence, conservation, wall_march)
- ``audit.md``      -- human-readable audit table
- ``regenerate.py`` -- script to regenerate all artifacts from scratch
- ``test_ci.py``    -- pytest tests that verify the directory is self-consistent

Usage
-----
    # Run CI tests
    pytest validation/euler_curved_sign_fix/test_ci.py -v

    # Regenerate all artifacts
    python validation/euler_curved_sign_fix/regenerate.py

The directory follows the CFD-GO/TCLB "model-dir-as-build-and-ci-unit" pattern:
each validation case is a self-contained unit that can be independently built,
tested, and verified.
"""
from __future__ import annotations

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def load_summary():
    """Load the case summary."""
    return json.loads((HERE / "summary.json").read_text())


def load_metrics():
    """Load the raw metrics."""
    return json.loads((HERE / "metrics.json").read_text())


def verify():
    """Verify that all acceptance criteria are met.
    
    Returns True if all criteria pass, False otherwise.
    """
    m = load_metrics()
    s = load_summary()
    
    # Check spectrum
    for key, val in m["linearised_spectrum"].items():
        if val["fixed"]["max_real"] >= 1e-8:
            return False
    
    # Check convergence
    for key in ("straight_p1", "warped_p1"):
        if key in m["convergence"]:
            orders = m["convergence"][key]["orders"]
            if orders and min(orders) < 1.5:
                return False
    
    # Check conservation
    for key, val in m["conservation"].items():
        if val["relative_drift"] >= 1e-10:
            return False
    
    return True


if __name__ == "__main__":
    if verify():
        print("PASS: All acceptance criteria met")
    else:
        print("FAIL: Some acceptance criteria not met")
        exit(1)
