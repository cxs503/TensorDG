import numpy as np, math
from tensordg.ldg_1d import solve_ldg_1d, convergence_study, assemble_1d

f = lambda x: np.pi**2*np.sin(np.pi*x)
ue = lambda x: np.sin(np.pi*x)
qe = lambda x: np.pi*np.cos(np.pi*x)
rows = convergence_study([0,1,2,3],[8,16,32,64],f,ue,qe)
for r in rows:
    print("k=%d order_u=%.3f order_q=%.3f expect=%d" % (r["degree"], r["order_u"], r["order_q"], r["expected_order"]),
          ["%.2e"%e for e in r["l2_u"]], "cond=%.1e"%r["condition_numbers"][-1])
# periodic
fp = lambda x: 4*np.pi**2*np.sin(2*np.pi*x)
up = lambda x: np.sin(2*np.pi*x)
qp = lambda x: 2*np.pi*np.cos(2*np.pi*x)
print("periodic:", [(d, round(convergence_study([d],[8,16,32],fp,up,qp,boundary="periodic")[0]["order_u"],3)) for d in (1,2,3)])
# exactness: u = x(1-x) with k=2, dirichlet g=0
ex = lambda x: x*(1-x)
r = solve_ldg_1d(4, 2, lambda x: 2.0+0*x, g0=0.0, g1=0.0)
print("quadratic exactness L2=%.2e" % r.l2_error_u(ex))
print("beta=0 dirichlet cond:", [ "%.1e"%assemble_1d(4,d,f)[0].toarray().astype(float).__abs__()[0,0] ] if False else "")
import scipy.sparse.linalg as spla
for beta in (0.0, 2.0):
    A,b,_ = assemble_1d(4,1,f,beta=beta)
    sv = np.linalg.svd(A.toarray(), compute_uv=False)
    print("beta=%.1f nd=%d nsing(<1e-10)=%d cond=%.2e" % (beta, A.shape[0], int(np.sum(sv<1e-10*sv[0])), sv[0]/sv[-1]))
