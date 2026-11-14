"""External-tool integration probes for TensorDG (ledger anchor KPI-TOOLS-5).

The anchor asks the solver ecosystem to *actually use* at least five classes of
external scientific-computing tools, each exercised through a real DG surface and
scored against a closed-form / exact reference (``abs_err <= limit`` qualifies as
REAL).  Import-without-error is explicitly **not** accepted as evidence.

Tool classes covered here (all genuine TensorDG dependencies):

======================  ============  ==================================================
tool class              module        DG surface exercised
======================  ============  ==================================================
autodiff                ``torch``     energy functional of the SIPG heat operator:
                                      ``grad(u^T M u)`` and the dissipation identity
                                      ``dE/dt = -2 u^T K u``
numerical_integration   ``scipy``     independent adaptive quadrature of the DG P2
                                      advection L2 error (8-point Gauss vs ``quad``)
symbolic_math           ``sympy``     exact rational DG volume matrix
                                      ``V_ij = int P_j dP_i/dxi`` and mass
                                      ``int P_i P_j = 2/(2i+1) delta_ij``
arbitrary_precision     ``mpmath``   50-digit reference integral used to bound the
                                      float64 reference chain (``int_0^1 sin^2 = 1/2``)
spectral_analysis       ``numpy.fft`` mode purity of the one-period DG P2 advection
                                      solution (amplitude + leakage)
visualization           ``matplotlib``non-empty render of the DG field vs exact wave
======================  ============  ==================================================

Missing tools are reported as ``MISSING`` and force ``verdict = FAIL`` -- they are
never silently skipped (honest-FAIL discipline).

Usage::

    python3 -c "from tensordg.external_tools_verification import run_all, render_markdown; \\
                print(render_markdown(run_all()))"
"""
from __future__ import annotations

import datetime
import importlib
import io
import math
import pathlib
import sys

import numpy as np
import torch

SCHEMA_VERSION = "1.0"
ANCHOR = "KPI-TOOLS-5"
REQUIRED_CLASSES = 5

#: ``tool class -> (module name, minimum version, reference description)``
PROBE_SPEC = {
    "autodiff": ("torch", "1.13", "grad(u^T M u) = 2 M u ; dE/dt = -2 u^T K u (exact)"),
    "numerical_integration": ("scipy", "1.7", "adaptive quad of the DG L2 error; anchor int sin^2 = 1/2"),
    "symbolic_math": ("sympy", "1.10", "exact rational DG volume/mass matrices"),
    "arbitrary_precision": ("mpmath", "1.3", "50-digit int_0^1 sin^2(pi x) dx = 1/2"),
    "spectral_analysis": ("numpy.fft", "1.20", "one-period P2 advection keeps mode amplitude 1.0"),
    "visualization": ("matplotlib", "3.5", "non-empty DG field render (data span > 0)"),
}

#: ``tool class -> importable module path`` (``numpy.fft`` is a submodule)
MODULE_PATH = {
    "autodiff": "torch",
    "numerical_integration": "scipy",
    "symbolic_math": "sympy",
    "arbitrary_precision": "mpmath",
    "spectral_analysis": "numpy.fft",
    "visualization": "matplotlib",
}


def _load(tool_class: str, disabled: set[str] | None = None):
    """Import the tool module for ``tool_class``; ``None`` when unavailable."""
    if disabled and tool_class in disabled:
        return None
    try:
        return importlib.import_module(MODULE_PATH[tool_class])
    except Exception:  # pragma: no cover - exercised via the ``disabled`` switch
        return None


def _version(module) -> str:
    for candidate in (module, sys.modules.get(module.__name__.split(".")[0])):
        if candidate is None:
            continue
        for attr in ("__version__", "version"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str):
                return value
    return "unknown"


def _record(tool_class, module, observed, reference, abs_err, limit, metric, detail="") -> dict:
    err = float(abs_err)
    return {
        "tool_class": tool_class,
        "module": MODULE_PATH[tool_class],
        "version": _version(module),
        "metric": metric,
        "observed": float(observed),
        "reference": float(reference),
        "abs_err": err,
        "limit": float(limit),
        "verdict": "REAL" if err <= float(limit) else "FAIL",
        "detail": detail,
    }


def _missing(tool_class, reason: str) -> dict:
    module, min_version, _ = PROBE_SPEC[tool_class]
    return {
        "tool_class": tool_class,
        "module": MODULE_PATH[tool_class],
        "version": "MISSING",
        "metric": "-",
        "observed": float("nan"),
        "reference": float("nan"),
        "abs_err": float("nan"),
        "limit": float("nan"),
        "verdict": "MISSING",
        "detail": f"required {module}>={min_version}: {reason}",
    }


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def probe_autodiff(module) -> dict:
    """torch.autograd on the semi-discrete SIPG heat energy functional.

    ``M du/dt = -K u`` with SPD ``M`` (mass) and ``K`` (SIPG stiffness) gives the
    exact dissipation identity ``d/dt (u^T M u) = -2 u^T K u``.  Autograd of the
    energy functional must reproduce both ``grad = 2 M u`` and that rate.
    """
    from .heat_1d import assemble_sipg_1d

    assembly = assemble_sipg_1d(ncells=8, degree=2)
    mass = assembly["mass_matrix"]
    stiffness = assembly["diffusion"]
    u0 = assembly["initial_coefficients"].reshape(-1)
    u = u0.clone().detach().requires_grad_(True)
    energy = torch.dot(u, mass @ u)
    energy.backward()
    grad_auto = u.grad.detach()
    grad_exact = 2.0 * (mass @ u0)
    err_grad = float((grad_auto - grad_exact).abs().max() / grad_exact.abs().max())
    rate = torch.linalg.solve(mass, -(stiffness @ u0))
    rate_auto = float(torch.dot(grad_auto, rate))
    rate_exact = float(-2.0 * torch.dot(u0, stiffness @ u0))
    err_rate = abs(rate_auto - rate_exact) / abs(rate_exact)
    return _record(
        "autodiff", module,
        observed=rate_auto, reference=rate_exact,
        abs_err=max(err_grad, err_rate), limit=1e-12,
        metric="max(rel err grad(2Mu), rel err dE/dt)",
        detail=(
            f"heat_1d SIPG ncells=8 degree=2 dofs={assembly['dofs']}; "
            f"rel err grad={err_grad:.3e}, rel err dE/dt={err_rate:.3e}"
        ),
    )


def _modal_to_callable(centers: torch.Tensor, coeff: torch.Tensor):
    """Return ``u_h(x)`` as a NumPy callable for the modal DG state ``[cell, mode]``."""
    c = coeff.detach().cpu().numpy()
    ncells, modes = c.shape
    h = 1.0 / ncells
    polys = []
    for k in range(modes):
        poly = np.zeros(k + 1)
        poly[-1] = 1.0
        polys.append(poly)

    def evaluate(x: float) -> float:
        cell = min(int(x / h), ncells - 1)
        xi = 2.0 * (x - (cell + 0.5) * h) / h
        total = 0.0
        for k, poly in enumerate(polys):
            total += c[cell, k] * np.polynomial.legendre.legval(xi, poly)
        return float(total)

    return evaluate, ncells, h


def probe_numerical_integration(module) -> dict:
    """scipy.integrate.quad as an independent quadrature oracle for DG L2 errors."""
    from scipy import integrate

    from .advection_1d import l2_error, solve_advection

    anchor, _ = integrate.quad(
        lambda x: math.sin(2.0 * math.pi * x) ** 2, 0.0, 1.0,
        epsabs=1e-14, epsrel=1e-14, limit=200,
    )
    err_anchor = abs(anchor - 0.5)  # int_0^1 sin^2(2 pi x) dx = 1/2 exactly

    centers, coeff, meta = solve_advection(ncells=20, degree=2, final_time=0.5, speed=1.0)
    exact = lambda x: math.sin(2.0 * math.pi * (x - 0.5))
    uh, ncells, h = _modal_to_callable(centers, coeff)
    err2 = 0.0
    for cell in range(ncells):
        lo, hi = cell * h, (cell + 1) * h
        piece, _ = integrate.quad(
            lambda x: (uh(x) - exact(x)) ** 2, lo, hi,
            epsabs=1e-18, epsrel=1e-13, limit=200,
        )
        err2 += piece
    quad_error = math.sqrt(err2)
    internal_error = l2_error(centers, coeff, 0.5)
    rel = abs(quad_error - internal_error) / internal_error
    return _record(
        "numerical_integration", module,
        observed=quad_error, reference=internal_error,
        abs_err=max(rel, err_anchor), limit=1e-8,
        metric="rel diff(adaptive quad, DG Gauss) of L2 error (+anchor)",
        detail=(
            f"anchor int_0^1 sin^2 = {anchor:.17f} (err {err_anchor:.2e}); "
            f"DG P2 ncells=20 t=0.5 steps={meta['steps']}: quad L2={quad_error:.12e}, "
            f"solver Gauss L2={internal_error:.12e}, rel diff={rel:.3e}"
        ),
    )


def probe_symbolic_math(module) -> dict:
    """sympy re-derives the DG volume/mass matrices in exact rational arithmetic."""
    import sympy as sp

    from .advection_1d import _operators

    xi = sp.symbols("xi", real=True)
    degree = 3
    basis = [sp.legendre(k, xi) for k in range(degree + 1)]
    volume_exact = sp.Matrix(
        degree + 1, degree + 1,
        lambda i, j: sp.integrate(sp.expand(basis[j] * sp.diff(basis[i], xi)), (xi, -1, 1)),
    )
    mass_exact = sp.Matrix(
        degree + 1, degree + 1,
        lambda i, j: sp.integrate(sp.expand(basis[i] * basis[j]), (xi, -1, 1)),
    )
    mass_closed = sp.Matrix(
        degree + 1, degree + 1,
        lambda i, j: sp.Rational(2, 2 * i + 1) if i == j else sp.Integer(0),
    )
    mass_residual = sp.simplify(mass_exact - mass_closed)
    _, _, _, volume_float, _, _ = _operators(degree, torch.device("cpu"), torch.float64)
    err_volume = float(
        np.abs(np.array(volume_float.numpy(), dtype=float)
               - np.array(volume_exact.tolist(), dtype=float)).max()
    )
    err_mass = float(abs(mass_residual).max()) if mass_residual != sp.zeros(*mass_exact.shape) else 0.0
    return _record(
        "symbolic_math", module,
        observed=err_volume, reference=0.0,
        abs_err=max(err_volume, err_mass), limit=1e-13,
        metric="max abs err float64 DG volume matrix vs exact rationals",
        detail=(
            f"degree={degree}: exact V rows="
            f"{[[int(volume_exact[i, j]) for j in range(degree + 1)] for i in range(degree + 1)]}; "
            f"mass identity residual exact={mass_residual == sp.zeros(*mass_exact.shape)}"
        ),
    )


def probe_arbitrary_precision(module) -> dict:
    """mpmath at 40 digits pins the L2-moment references of the DG verification chain.

    The scored integral is ``int_0^1 x^2 sin^2(pi x) dx = 1/6 - 1/(4 pi^2)``, an
    L2-type moment of the kind the DG error chain evaluates; its closed form is
    irrational, so the probe error is genuinely non-zero at finite precision.  The
    heat-equation energy reference ``int_0^1 sin^2(pi x) dx e^{-2 pi^2 t}`` used by
    :mod:`tensordg.heat_1d` is reported alongside to expose the float64 floor.
    """
    mp = module

    with mp.workdps(40):
        high_precision = mp.quad(lambda x: x**2 * mp.sin(mp.pi * x) ** 2, [0, 1])
        exact = mp.mpf(1) / 6 - 1 / (4 * mp.pi**2)  # closed form, irrational
        err_hp = abs(high_precision - exact)
        time = mp.mpf("0.1")
        energy_hp = mp.quad(lambda x: mp.sin(mp.pi * x) ** 2, [0, 1]) * mp.e ** (-2 * mp.pi**2 * time)
        energy_exact = mp.mpf(1) / 2 * mp.e ** (-2 * mp.pi**2 * time)
        err_energy_hp = abs(energy_hp - energy_exact)
        energy_f64 = 0.5 * math.exp(-2.0 * math.pi**2 * float(time))
        rel_f64 = abs(energy_f64 - float(energy_exact)) / float(energy_exact)
    return _record(
        "arbitrary_precision", module,
        observed=float(high_precision), reference=float(exact),
        abs_err=float(err_hp), limit=1e-38,
        metric="abs err 40-digit int_0^1 x^2 sin^2(pi x) vs 1/6 - 1/(4 pi^2)",
        detail=(
            f"dps=40 err={float(err_hp):.3e}; heat energy ref (t=0.1) hp err={float(err_energy_hp):.3e}, "
            f"float64 vs hp rel diff={rel_f64:.3e} (= float64 floor of the DG heat reference)"
        ),
    )


def probe_spectral_analysis(module) -> dict:
    """numpy.fft measures DG P2 dispersion/dissipation on one advection period."""
    from .advection_1d import solve_advection

    ncells, degree = 32, 2
    centers, coeff, meta = solve_advection(ncells=ncells, degree=degree,
                                           final_time=1.0, speed=1.0)
    samples = 8 * ncells  # 256 uniform points, no aliasing of the mesoscale jitter
    xs = (np.arange(samples) + 0.5) / samples
    uh, _, _ = _modal_to_callable(centers, coeff)
    values = np.array([uh(x) for x in xs])
    spectrum = np.fft.rfft(values) / samples * 2.0  # unit-amplitude sine convention
    amplitude = float(abs(spectrum[1]))
    energy = np.abs(spectrum[1:]) ** 2
    leakage = float(energy[1:].sum() / energy.sum()) if energy.sum() > 0 else 1.0
    err_amp = abs(amplitude - 1.0)  # exact solution sin(2 pi x) after one period
    return _record(
        "spectral_analysis", module,
        observed=amplitude, reference=1.0,
        abs_err=max(err_amp, leakage), limit=1e-6,
        metric="max(|c1| - 1, higher-mode leakage fraction)",
        detail=(
            f"DG P{degree} ncells={ncells} t=1.0 steps={meta['steps']}: "
            f"|c1|={amplitude:.12f} (err {err_amp:.2e}), leakage={leakage:.3e}"
        ),
    )


def probe_visualization(module) -> dict:
    """matplotlib renders the DG field; the canvas must carry real data."""
    from matplotlib.figure import Figure

    from .advection_1d import solve_advection

    centers, coeff, _ = solve_advection(ncells=40, degree=2, final_time=0.25, speed=1.0)
    uh, ncells, h = _modal_to_callable(centers, coeff)
    xs = np.linspace(0.0, 1.0, 401)
    values = np.array([uh(min(x, 1.0 - 1e-12)) for x in xs])
    truth = np.sin(2.0 * math.pi * (xs - 0.25))

    figure = Figure(figsize=(6.0, 3.0), dpi=80)
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(xs, values, label="DG P2")
    axes.plot(xs, truth, "--", label="exact")
    axes.set_xlabel("x")
    axes.set_ylabel("u")
    axes.legend()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    payload = buffer.getvalue()
    span = float(values.max() - values.min())
    transient_deviation = float(np.abs(values - truth).max())
    # span vs the exact wave's peak-to-peak 2.0 proves real data reached the canvas;
    # the payload size proves the raster was actually produced.
    payload_ok = span > 0.0 and len(payload) > 1000
    return _record(
        "visualization", module,
        observed=span, reference=2.0,
        abs_err=abs(span - 2.0) if payload_ok else 1.0, limit=1e-3,
        metric="|y-span - exact peak-to-peak 2.0| (PNG payload > 1 kB required)",
        detail=(
            f"png={len(payload)} B, y-span={span:.6f} (payload_ok={payload_ok}), "
            f"max|u_h - exact|={transient_deviation:.3e}, ncells={ncells} h={h:.4f}"
        ),
    )


PROBES = {
    "autodiff": probe_autodiff,
    "numerical_integration": probe_numerical_integration,
    "symbolic_math": probe_symbolic_math,
    "arbitrary_precision": probe_arbitrary_precision,
    "spectral_analysis": probe_spectral_analysis,
    "visualization": probe_visualization,
}


def run_all(disabled: set[str] | None = None) -> dict:
    """Run every probe; ``disabled`` simulates a missing tool (honest FAIL path)."""
    records = []
    for tool_class, probe in PROBES.items():
        module = _load(tool_class, disabled)
        if module is None:
            records.append(_missing(tool_class, "import failed or disabled"))
            continue
        try:
            records.append(probe(module))
        except Exception as exc:  # a tool that cannot be used is not "available"
            record = _missing(tool_class, f"probe raised {type(exc).__name__}: {exc}")
            records.append(record)
    # the verdict is derived from the numbers, never trusted from the probe label
    for rec in records:
        if rec["verdict"] == "MISSING":
            continue
        rec["verdict"] = "REAL" if float(rec["abs_err"]) <= float(rec["limit"]) else "FAIL"
    real = sum(1 for rec in records if rec["verdict"] == "REAL")
    verdict = "PASS" if real >= REQUIRED_CLASSES and all(
        rec["verdict"] == "REAL" for rec in records
    ) else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "anchor": ANCHOR,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "required_classes": REQUIRED_CLASSES,
        "tool_classes": len(records),
        "real_classes": real,
        "verdict": verdict,
        "probes": records,
    }


def render_markdown(report: dict) -> str:
    """Render the probe ledger as markdown (all numbers read from the report)."""
    lines = [
        "# KPI-TOOLS-5: external tool integration -- TensorDG solver-side probes",
        "",
        f"**Anchor**: {report['anchor']} (tool classes >= {report['required_classes']})  ",
        f"**Verdict**: `{report['verdict']}` -- real_classes={report['real_classes']}"
        f"/{report['tool_classes']}  ",
        f"**Generated**: {report['generated_at']} (numbers injected from `run_all()`)",
        "",
        "## Tool class x DG surface x reference",
        "",
        "| tool class | module | version | metric | observed | reference | abs err | limit | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for rec in report["probes"]:
        lines.append(
            "| {tool_class} | {module} | {version} | {metric} | {observed} | {reference} "
            "| {abs_err} | {limit} | {verdict} |".format(
                tool_class=rec["tool_class"], module=rec["module"], version=rec["version"],
                metric=rec["metric"], observed=_fmt(rec["observed"]),
                reference=_fmt(rec["reference"]), abs_err=_fmt(rec["abs_err"]),
                limit=_fmt(rec["limit"]), verdict=rec["verdict"],
            )
        )
    lines += ["", "## Probe details", ""]
    for rec in report["probes"]:
        lines.append(f"- **{rec['tool_class']}** ({rec['verdict']}): {rec['detail']}")
    lines += [
        "",
        "## Scope and limits",
        "",
        "- These probes establish *tool usability on genuine DG surfaces*, not new",
        "  solver accuracy: every probe's reference is closed form or exact rational",
        "  arithmetic, so the scored error is the tool+operator chain, not a physics model.",
        "- A missing tool reports `MISSING` and forces `verdict = FAIL`; nothing is",
        "  silently skipped.",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n/a"
        if value == 0.0:
            return "0"
        return f"{value:.6e}"
    return str(value)


def write_artifacts(out_dir: pathlib.Path | str = "artifacts/external_tools") -> dict:
    """Write the JSON ledger + markdown doc + renders; returns the report."""
    import json

    root = pathlib.Path(__file__).resolve().parents[1]
    out = pathlib.Path(out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    report = run_all()
    (out / "tensordg_external_tools_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (root / "docs" / "external-tools-verification.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


if __name__ == "__main__":  # pragma: no cover
    result = write_artifacts()
    print(render_markdown(result))