# CASE-LAX-DG · certified configuration and its cost

> Rule: a configuration is PASS iff max_field rel L2 < tolerance (whole field, every field, no wave band removed, no averaging); the certified configuration is the PASS leg with the smallest (p+1)*N, tie-broken by measured wall clock
> Tolerance (imported from the registered case): 0.0300
> Certified: **p2.N800** · worst rel L2 2.57033e-02 (density, margin ×0.857)
> metrics sha256 `fb2cf719d93722f7e1415fb4be5359175c0cfdbff5fa8e6d41612494e194c214`

| leg | DOF | steps | wall [s] | rel L2 rho | rel L2 u | rel L2 p | worst | status |
|---|---|---|---|---|---|---|---|---|
| p1.N400 | 800 | 2199 | 29.6 | 4.93486e-02 | 1.78668e-02 | 1.01644e-02 | 4.93486e-02 | FAIL |
| p1.N800 | 1600 | 4390 | 67.7 | 3.77126e-02 | 1.14182e-02 | 7.39018e-03 | 3.77126e-02 | FAIL |
| p1.N1600 | 3200 | 8773 | 169.2 | 2.95401e-02 | 9.89871e-03 | 5.40336e-03 | 2.95401e-02 | PASS |
| p2.N400 | 1200 | 3662 | 50.2 | 3.52587e-02 | 1.57638e-02 | 8.57558e-03 | 3.52587e-02 | FAIL |
| p2.N800 | 2400 | 7341 | 118.4 | 2.57033e-02 | 9.10287e-03 | 5.49864e-03 | 2.57033e-02 | PASS |
| p2.N1000 | 3000 | 9183 | 158.2 | 2.36827e-02 | 1.05645e-02 | 5.75614e-03 | 2.36827e-02 | PASS |
| p3.N400 | 1600 | 5109 | 73.9 | 3.11556e-02 | 2.24805e-02 | 1.30501e-02 | 3.11556e-02 | FAIL |
| p3.N800 | 3200 | 10250 | 176.4 | 2.31490e-02 | 1.38257e-02 | 8.99402e-03 | 2.31490e-02 | PASS |

Reuse check: PASS (worst relative difference 0.000e+00)
Minimality (grid scope): True
Wall time: 843.4 s
