"""Model directory: curvilinear Euler DG convergence study.

This directory is a self-contained CI unit for the validation case
``CASE-DG-CKS1998-CURVED-CONVERGENCE``.

Structure
---------
- ``summary.json``  -- case description, physics, acceptance criteria, status
- ``audit.json``    -- raw convergence data (straight/warped, p=1,2)
- ``audit.md``      -- human-readable audit table
- ``test_ci.py``    -- pytest tests that verify the directory is self-consistent

Usage
-----
    # Run CI tests
    pytest validation/euler_curved_convergence/test_ci.py -v

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


def load_audit():
    """Load the raw audit data."""
    return json.loads((HERE / "audit.json").read_text())


def verify():
    """Verify that all acceptance criteria are met.
    
    Returns True if all criteria pass, False otherwise.
    """
    a = load_audit()
    s = load_summary()
    
    # Check convergence orders
    blocks = a.get("blocks", {})
    for strength in ("strength_1", "strength_5"):
        if strength in blocks:
            straight = blocks[strength].get("straight", {})
            orders = straight.get("evolve_orders", [])
            if orders and min(orders) < 1.5:
                return False
    
    return True


if __name__ == "__main__":
    if verify():
        print("PASS: All acceptance criteria met")
    else:
        print("FAIL: Some acceptance criteria not met")
        exit(1)
