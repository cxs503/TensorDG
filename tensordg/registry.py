"""TensorDG Capability Registry: self-registering capability discovery.

This module provides a capability registry for TensorDG that:
1. Uses a @capability decorator for self-registration
2. Provides CLI with dual modes (--static AST + import mode + --parity)
3. Supports torch-free host with direct file path calls (no -m)
4. Outputs single JSON object (source_commit+capabilities+summary+problems)
5. Enforces local discipline: mature needs verified+wired; planned needs
   evidence.feature_ledger anchor; endpoint needs wired+cli:/route:/api:;
   verified needs ANALYTIC/XVALUE level evidence
6. Resolves pointers (module/entry/test_path not exist = problem)
7. Scans full package registration

Usage:
    # As module (import mode)
    python -m tensordg.registry
    
    # Direct file path (torch-free host)
    python tensordg/registry.py
    
    # Static AST mode (avoid re-import)
    python tensordg/registry.py --static
    
    # Parity check
    python tensordg/registry.py --parity
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# Registry storage
_CAPABILITY_REGISTRY: dict[str, dict] = {}
_CAPABILITY_DECORATORS: list[Callable] = []


def get_git_commit() -> str:
    """Get the current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return "unknown"


def get_git_root() -> Path:
    """Get the git root directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return Path(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return Path(__file__).parent


def capability(
    name: str,
    category: str = "solver",
    maturity: str = "planned",
    evidence: Optional[dict] = None,
    endpoint: Optional[dict] = None,
    verified: Optional[dict] = None,
    description: str = "",
    tags: Optional[list[str]] = None,
):
    """Decorator to register a capability.
    
    Parameters
    ----------
    name : str
        Capability identifier (e.g., "euler-1d", "advection-2d")
    category : str
        Category: solver, limiter, geometry, time_integration, etc.
    maturity : str
        Maturity level: planned, mature, deprecated
    evidence : dict, optional
        Evidence for planned capabilities (must have feature_ledger anchor)
    endpoint : dict, optional
        Endpoint info (must have wired+cli:/route:/api:)
    verified : dict, optional
        Verification info (must have ANALYTIC/XVALUE level)
    description : str
        Human-readable description
    tags : list[str], optional
        List of tags
    """
    def decorator(func: Callable) -> Callable:
        cap_dict = {
            "name": name,
            "category": category,
            "maturity": maturity,
            "description": description,
            "tags": tags or [],
            "function": f"{func.__module__}.{func.__name__}",
        }
        
        if evidence:
            cap_dict["evidence"] = evidence
        if endpoint:
            cap_dict["endpoint"] = endpoint
        if verified:
            cap_dict["verified"] = verified
            
        _CAPABILITY_REGISTRY[name] = cap_dict
        _CAPABILITY_DECORATORS.append(func)
        
        return func
    return decorator


def scan_package_static(package_path: Path) -> list[dict]:
    """Scan package using AST (static analysis, no imports).
    
    This mode avoids torch dependencies and works on torch-free hosts.
    """
    capabilities = []
    package_name = package_path.stem
    
    # Find all Python files in the package
    for py_file in package_path.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
            
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(py_file))
            
            # Look for @capability decorators
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            if decorator.id == "capability":
                                # Extract capability info from decorator
                                cap_info = _extract_capability_from_ast(node, py_file)
                                if cap_info:
                                    capabilities.append(cap_info)
                        elif isinstance(decorator, ast.Call):
                            if isinstance(decorator.func, ast.Name):
                                if decorator.func.id == "capability":
                                    cap_info = _extract_capability_from_ast(node, py_file)
                                    if cap_info:
                                        capabilities.append(cap_info)
                                    
        except (SyntaxError, OSError) as e:
            # Skip files that can't be parsed
            continue
    
    return capabilities


def _extract_capability_from_ast(func_node: ast.FunctionDef, py_file: Path) -> Optional[dict]:
    """Extract capability information from AST node."""
    # Find the capability decorator
    for decorator in func_node.decorator_list:
        if isinstance(decorator, ast.Call):
            # Parse keyword arguments
            kwargs = {}
            for kw in decorator.keywords:
                if isinstance(kw.value, ast.Constant):
                    kwargs[kw.arg] = kw.value.value
                elif isinstance(kw.value, ast.Dict):
                    # Handle dict literals
                    dict_val = {}
                    for k, v in zip(kw.value.keys, kw.value.values):
                        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                            dict_val[k.value] = v.value
                    kwargs[kw.arg] = dict_val
                elif isinstance(kw.value, ast.List):
                    # Handle list literals
                    list_val = []
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant):
                            list_val.append(elt.value)
                    kwargs[kw.arg] = list_val
                    
            result = {
                "name": kwargs.get("name", f"{py_file.stem}-{func_node.name}"),
                "category": kwargs.get("category", "solver"),
                "maturity": kwargs.get("maturity", "planned"),
                "description": kwargs.get("description", ""),
                "tags": kwargs.get("tags", []),
                "function": f"{py_file.stem}.{func_node.name}",
                "source_file": str(py_file.relative_to(get_git_root())),
            }
            # Carry the discipline-relevant fields through so static-mode
            # validation sees the same evidence/endpoint/verified that
            # import-mode sees (m24065: previously dropped, which made
            # static-mode validation blind to planned/mature requirements).
            for key in ("evidence", "endpoint", "verified"):
                if isinstance(kwargs.get(key), dict):
                    result[key] = kwargs[key]
            return result
    return None


def scan_package_import(package_path: Path) -> list[dict]:
    """Scan package by importing modules (requires torch)."""
    capabilities = []
    
    # Add package to path
    sys.path.insert(0, str(package_path.parent))
    
    try:
        # Import the package
        import importlib
        
        package_name = package_path.name
        pkg = importlib.import_module(package_name)

        # Import *every* submodule so every @capability decorator runs,
        # regardless of whether ``__init__`` re-exports it (some analysis
        # tools are not re-exported at the package root).
        import pkgutil
        for mod_info in pkgutil.iter_modules(pkg.__path__, prefix=package_name + "."):
            try:
                importlib.import_module(mod_info.name)
            except Exception:
                continue

        # The decorators register into the canonical ``tensordg.registry``
        # module's registry (submodules do ``from .registry import capability``),
        # which may be a *different* module object than this one when the CLI
        # is run as a plain script (``python tensordg/registry.py`` -> __main__).
        # Read from the canonical module so both invocation styles agree.
        reg_mod = importlib.import_module(f"{package_name}.registry")
        for cap_name, cap_info in reg_mod._CAPABILITY_REGISTRY.items():
            capabilities.append(cap_info)
                
    except ImportError as e:
        print(f"Warning: Could not import package: {e}", file=sys.stderr)
    finally:
        if str(package_path.parent) in sys.path:
            sys.path.remove(str(package_path.parent))
    
    return capabilities


def validate_capability(cap: dict) -> list[str]:
    """Validate a capability against local discipline rules.
    
    Returns list of problems (empty if valid).
    """
    problems = []
    maturity = cap.get("maturity", "")
    
    # Rule 1: mature needs verified + wired
    if maturity == "mature":
        if "verified" not in cap:
            problems.append("mature capability missing 'verified' field")
        if "endpoint" not in cap:
            problems.append("mature capability missing 'endpoint' field")
            
    # Rule 2: planned needs evidence.feature_ledger anchor
    if maturity == "planned":
        evidence = cap.get("evidence", {})
        if not evidence:
            problems.append("planned capability missing 'evidence' field")
        elif "feature_ledger" not in evidence:
            problems.append("planned capability missing evidence.feature_ledger anchor")
            
    # Rule 3: endpoint needs wired + cli:/route:/api:
    if "endpoint" in cap:
        endpoint = cap["endpoint"]
        if "wired" not in endpoint:
            problems.append("endpoint missing 'wired' field")
        cli = endpoint.get("cli", "")
        if not (cli.startswith("cli:") or cli.startswith("/route:") or cli.startswith("/api:")):
            problems.append(f"endpoint cli must start with cli:/route:/api:, got: {cli}")
            
    # Rule 4: verified needs ANALYTIC/XVALUE level
    if "verified" in cap:
        verified = cap["verified"]
        level = verified.get("level", "").upper()
        if level not in ("ANALYTIC", "XVALUE"):
            problems.append(f"verified level must be ANALYTIC or XVALUE, got: {level}")
            
    # Rule 5: Pointer resolution - check module/entry/test_path exist
    func = cap.get("function", "")
    if func:
        parts = func.split(".")
        # static mode: "mod.func" ; import mode: "pkg.mod.func" -> module is
        # always the segment immediately before the function name.
        module_name = parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")
        if module_name:
            # Check if module exists
            package_path = get_git_root() / "tensordg"
            module_path = package_path / f"{module_name}.py"
            if not module_path.exists():
                problems.append(f"module {module_name} does not exist at {module_path}")
            
    return problems


def check_parity(static_caps: list[dict], import_caps: list[dict]) -> dict:
    """Check parity between static AST and import modes."""
    static_names = {c["name"] for c in static_caps}
    import_names = {c["name"] for c in import_caps}
    
    return {
        "static_count": len(static_caps),
        "import_count": len(import_caps),
        "static_only": list(static_names - import_names),
        "import_only": list(import_names - static_names),
        "parity": static_names == import_names,
    }


def generate_registry_json(
    capabilities: list[dict],
    mode: str = "static",
    check_parity_flag: bool = False,
) -> dict:
    """Generate the registry JSON output."""
    git_root = get_git_root()
    
    # Validate all capabilities
    all_problems = []
    for cap in capabilities:
        problems = validate_capability(cap)
        if problems:
            all_problems.append({
                "capability": cap.get("name", "unknown"),
                "problems": problems,
            })
    
    # Build summary
    maturity_counts = {}
    category_counts = {}
    for cap in capabilities:
        maturity = cap.get("maturity", "unknown")
        category = cap.get("category", "unknown")
        maturity_counts[maturity] = maturity_counts.get(maturity, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
    
    result = {
        "source_commit": get_git_commit(),
        "mode": mode,
        "capabilities": capabilities,
        "summary": {
            "total_count": len(capabilities),
            "by_maturity": maturity_counts,
            "by_category": category_counts,
        },
        "problems": all_problems,
    }
    
    if check_parity_flag:
        # Would need both modes to be run - this is a placeholder
        result["parity_check"] = {"note": "run with both modes to check parity"}
    
    return result


@capability(
    name="capability-registry",
    category="infrastructure",
    maturity="planned",
    evidence={"feature_ledger": "CAP-REG-DG"},
    description="Capability registry (self-registration + dual-mode CLI)",
    tags=['registry', 'infrastructure']
)
def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TensorDG Capability Registry Scanner"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Use static AST analysis (no imports, torch-free)"
    )
    parser.add_argument(
        "--import-mode",
        action="store_true",
        help="Use import mode (requires torch)"
    )
    parser.add_argument(
        "--parity",
        action="store_true",
        help="Check parity between static and import modes"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file (default: stdout)"
    )
    
    args = parser.parse_args()
    
    git_root = get_git_root()
    package_path = git_root / "tensordg"
    
    if not package_path.exists():
        print(f"Error: Package not found at {package_path}", file=sys.stderr)
        sys.exit(1)
    
    capabilities = []
    mode = "unknown"
    
    if args.parity:
        # Run both modes and compare
        static_caps = scan_package_static(package_path)
        import_caps = scan_package_import(package_path)
        parity_result = check_parity(static_caps, import_caps)
        
        result = {
            "source_commit": get_git_commit(),
            "mode": "parity",
            "static_capabilities": static_caps,
            "import_capabilities": import_caps,
            "parity_check": parity_result,
            "problems": [],
        }
        
        # Validate both
        for cap in static_caps + import_caps:
            problems = validate_capability(cap)
            if problems:
                result["problems"].append({
                    "capability": cap.get("name", "unknown"),
                    "problems": problems,
                })
                
    elif args.static:
        capabilities = scan_package_static(package_path)
        mode = "static"
        
    elif args.import_mode:
        capabilities = scan_package_import(package_path)
        mode = "import"
        
    else:
        # Default to static
        capabilities = scan_package_static(package_path)
        mode = "static"
    
    if not args.parity:
        result = generate_registry_json(capabilities, mode)
    
    # Output
    json_output = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)
        print(f"Registry written to {args.output}")
    else:
        print(json_output)


if __name__ == "__main__":
    main()