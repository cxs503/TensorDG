import torch, importlib.util, tensordg.ns_dg_2d as ns
spec = importlib.util.spec_from_file_location("t", "tests/test_ns_heat_conduction.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ny, degree, nx = 32, 1, 2
dx, dy = 1.0/nx, 1.0/ny
nodes, weights, basis, derivative, signs = ns.legendre_operators(degree, degree+2, torch.device("cpu"), torch.float64)
x = (torch.arange(nx, dtype=torch.float64)+0.5)*dx
y = (torch.arange(ny, dtype=torch.float64)+0.5)*dy
coeff = ns._initial_coefficients(m._steady_initial, x, y, dx, dy, nodes, weights, basis, torch.device("cpu"), torch.float64)
axis=1; h=dy
for side in ("right","left"):
    fr = ns._face_primitives(coeff, basis, derivative, signs, axis, side)
    idx = -1 if side=="right" else 0
    print(side, "T trace:", fr["temperature"][0,idx].tolist(), "T_n(ref):", fr["T_n"][0,idx].tolist())
recon = ns._wall_normal_derivative(coeff, basis, signs, h, axis)
print("recon right T_n:", recon["right"]["T_n"][0,-1].tolist() if recon["right"]["T_n"].dim()==3 else recon["right"]["T_n"][0,-1])
print("recon left  T_n:", recon["left"]["T_n"][0,0].tolist())
print("2/h =", 2.0/h)

MU=0.1; PRANDTL=0.71; KAPPA=MU*ns.CP/PRANDTL
boundary = {"ymin": {"velocity": (0.0,0.0), "temperature": 1.0},
            "ymax": {"velocity": (0.0,0.0), "temperature": 0.5}}
recon = ns._wall_normal_derivative(coeff, basis, signs, h, axis)
right = ns._face_primitives(coeff, basis, derivative, signs, axis, "right")
out = ns._boundary_flux(right, boundary["ymax"], axis, h, MU, KAPPA, 8.0, 2.0/h, 2.0/dx,
                        normal_derivative=recon["right"], side_name="right", outward_sign=1.0)
print("top wall flux comps (cell 0):", out[0,-1].tolist())
print("kappa:", KAPPA)
print("full out shape", out.shape)
print("out[...,3]:", out[...,3].tolist())
print("u_n trace at top:", right["u_n"][...,0].tolist() if right["u_n"].dim()==3 else "?")
print("T_n raw at top:", right["T_n"].tolist())
