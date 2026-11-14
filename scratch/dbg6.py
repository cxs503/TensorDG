import numpy as np, math
src=open('scratch/proto4.py').read().split('nx, ny =')[0]
exec(src)
rng=np.random.default_rng(1)
N=5000
S=[rng.uniform(-4e5,4e5,N),rng.uniform(-4e5,4e5,N),rng.uniform(-4e5,4e5,N)]
c=25e3
o=return_map(S[0],S[1],S[2],np.full(N,c))
m,q=mq(*o[:3]); F=q+m*sp-c*cpn
fin=np.isfinite(o[3])&np.isfinite(o[0])
print('n nonfinite',(~fin).sum(),' n F>tol',(F>1e-3*c).sum(),' F max',F[fin].max())
idx=np.argsort(-F)[:5]
for i in idx: print('  in',S[0][i],S[1][i],S[2][i],'F',F[i],'dgp',o[3][i])
# monotone biaxial path
sxx=-50e3; syy=-50e3
for k in range(8):
    sxx-=50e3
    o=return_map(np.array([sxx]),np.array([syy]),np.array([0.0]),np.array([25e3]))
    m,q=mq(*o[:3])
    print(f'step {k}: in sxx={sxx:.3e} out m={m[0]:.4e} q={q[0]:.4e} F={q[0]+m[0]*sp-25e3*cpn:.3e} dgp={o[3][0]:.3e}')
