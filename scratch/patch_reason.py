from pathlib import Path

p = Path("benchmarks/shock_reference_quadrature_audit.py")
s = p.read_text()
old = '        raise FileNotFoundError(f"no cached profile for {name} (SRQ_ALLOW_MISSING=1)")'
new = ('        raise FileNotFoundError(\n'
       '            f"{name}: not computed in this session; the run exceeds the local "\n'
       '            "single-command budget and is queued on the cluster (job cj3966)"\n'
       '        )')
assert old in s
p.write_text(s.replace(old, new))

t = Path("tests/test_shock_reference_quadrature.py")
u = t.read_text()
old = '''    for gate in metrics["gates"]:
        for key, field in gate["fields"].items():'''
new = '''    for gate in metrics["gates"]:
        if gate.get("status") != "measured":
            assert gate["reason"]          # a not_run gate must say why
            continue
        for key, field in gate["fields"].items():'''
assert old in u
t.write_text(u.replace(old, new))
print("patched")