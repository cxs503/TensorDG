# Demo Sample Report — Wigley Hull Resistance

## 1. Hull Geometry
The Wigley hull is a canonical test case for ship resistance prediction.
- Length L = 2.0 m
- Beam B = 0.3 m
- Draft T = 0.125 m
- Block coefficient Cb = 0.467
- Wetted surface area S = 1.85 m²

## 2. Method
Resistance is computed using the Holtrop-Mennen empirical method.
The method decomposes total resistance into frictional, residual, and appendage components.
Total resistance coefficient Ct = 3.25e-3 at Fr = 0.25.

## 3. Results
At Froude number Fr = 0.25:
- Computed Ct = 3.28e-3
- Reference Ct_ref = 3.25e-3 (Holtrop & Mennen, 1982)
- Relative error = 0.92%

At Froude number Fr = 0.35:
- Computed Ct = 4.12e-3
- Reference Ct_ref = 4.08e-3
- Relative error = 0.98%

## 4. Known Issues
When the Froude number exceeds 0.45, the Holtrop-Mennen method tends to
overpredict resistance by 5-10%. This is a known limitation of the empirical
formulation for high-speed displacement hulls.

## 5. Code Reference
The implementation is in TensorFVM, module resistance/holtrop_mennen.py,
function compute_ct().

## 6. Dimensionless Parameters
- Froude number Fr = V / sqrt(g * L), typical range 0.1 to 0.5
- Reynolds number Re = V * L / nu, typical value 2.8e6
- Block coefficient Cb = V_displaced / (L * B * T) = 0.467
