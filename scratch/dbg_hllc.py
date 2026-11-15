import numpy as np, torch
from tensordg import euler_dg_2d as E
from tensordg.euler_curved import hllc_normal
rng = np.random.default_rng(3)
UL = rng.standard_normal((5,4)); UR = rng.standard_normal((5,4))
UL[:,0] = 1.0+np.abs(UL[:,0]); UR[:,0] = 1.0+np.abs(UR[:,0])
UL[:,3] = 2.0+np.abs(UL[:,3]); UR[:,3] = 2.0+np.abs(UR[:,3])
for axis in (0,1):
    ref = E._hllc(torch.as_tensor(UL), torch.as_tensor(UR), axis).numpy()
    n = np.zeros(2); n[axis] = 1.0
    mine = hllc_normal(UL, UR, np.full(5, n[0]), np.full(5, n[1]))
    print("axis", axis, "max|diff|", np.abs(ref-mine).max(), "max|ref|", np.abs(ref).max())
    print("  ref ", np.round(ref[0],5))
    print("  mine", np.round(mine[0],5))
