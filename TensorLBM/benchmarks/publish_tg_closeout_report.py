"""Publish the Taylor-Green honest-FAIL close-out report to the platform API.

Builds ``/tmp/report.json`` (markdown + three figures, base64) from the audit
doc and artifacts, logs in with ``$TENSOR_AGENT_USER`` / ``$TENSOR_AGENT_PASS``
and POSTs it to ``$TENSOR_API_BASE/tensor/reports``.

Usage: python benchmarks/publish_tg_closeout_report.py
"""

import base64
import json
import os
import pathlib
import urllib.request

BASE = pathlib.Path(__file__).resolve().parent.parent
CAPS = [
    ("fig1_decay_curves.png",
     "Taylor-Green decay: fundamental-mode amplitude and integral kinetic energy, corrected vs "
     "malformed initial condition, with the analytic laws exp(-2 nu k^2 t) / exp(-4 nu k^2 t)."),
    ("fig2_rate_error_vs_nu.png",
     "Decay-rate error of the fundamental mode vs viscosity: the corrected IC stays inside 3 % at "
     "every grid/viscosity, the malformed IC of the predecessor driver reaches >100 %."),
    ("fig3_spectrum.png",
     "Late-time spectral content: the malformed IC parks ~94 % of its energy in the conserved mean "
     "mode (0,0,0), which is why the predecessor's total-KE test failed and was flat in nu."),
]


def main() -> dict:
    figs = []
    for name, cap in CAPS:
        png = BASE / "outputs" / "tg_physics_audit" / "figures" / name
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(png.read_bytes()).decode()})
    md = (BASE / "docs" / "XVAL-LBM-HOST-STAGED-TG-MODAL-CLOSEOUT.md").read_text()
    report = {
        "title": ("XVAL-LBM host-staged halo: Taylor-Green honest-FAIL closed -- the defect is the "
                  "driver initial condition, not the solver (worst error 0.46 %)"),
        "summary": ("An independent NumPy D3Q19 BGK oracle isolates mission m18373's honest FAIL to "
                    "the cross-validation driver's Taylor-Green initial condition (wavenumber applied "
                    "twice -> sin(k^2 i), net momentum 1.08e-2 that BGK conserves exactly). With the "
                    "IC corrected, the FluidX3D-style host-staged port reproduces the analytic decay "
                    "to 0.10 % (kinetic energy, single device and 2x1x2) and matches the independent "
                    "oracle field to 7e-16 after 50 steps. Sweep over 3 grids x 5 viscosities: worst "
                    "fundamental-mode rate error 0.46 % (16^3) down to 0.11 % (32^3) -- inside the "
                    "3 % arm and far inside the 15 % acceptance. Transport validation of m18373 "
                    "unchanged; no subject code modified."),
        "scenario_key": "xval_lbm_single_node_multi_gpu_host_staged_halo_tg_closeout",
        "mission_id": 21276,
        "tags": ["LBM", "XVAL", "host-staged-halo", "FluidX3D", "Taylor-Green", "D3Q19",
                 "honest-FAIL-closeout", "defect-fix"],
        "markdown": md,
        "figures": figs,
    }
    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    creds = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                        "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(f"{api}/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"})
    token = json.load(urllib.request.urlopen(req))["access_token"]
    body = json.dumps(report).encode()
    req = urllib.request.Request(f"{api}/tensor/reports", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    out = json.load(urllib.request.urlopen(req))
    print("payload MB", round(len(body) / 1e6, 2), "->", json.dumps(out)[:300])
    return out


if __name__ == "__main__":
    main()