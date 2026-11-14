import numpy as np, math
src=open('scratch/proto4.py').read().split('nx, ny =')[0]
exec(src)
rng=np.random.default_rng(1)
bad=0
for i in range(20000):
    sxx=rng.uniform(-4e5,4e5); syy=rng.uniform(-4e5,4e5); sxy=rng.uniform(-4e5,4e5)
    c=25e3
    o=return_map(sxx,syy,sxy,c)
    m,q=mq(*o[:3]); F=q+m*sp-c*cpn
    if not np.all(np.isfinite(o[3])) or F>1e-3*c:
        bad+=1
        if bad<6: print('bad',sxx,syy,sxy,'->',o[0],o[1],o[2],'F',F,'dgp',o[3])
print('bad count',bad)
# biaxial loading path check
m0=-50e3; q=0.0
for i in range(12):
    sxx=m0-q*3; syy=m0+q*1; sxy=0.0
    sxx-=60e3
    o=return_map(sxx,syy,sxy,25e3); mm,qq=mq(*o[:3])
    print(f'in=({sxx:.3e},{syy:.3e}) out m={mm:.4e} q={qq:.4e} F={qq+mm*sp-25e3*cpn:.3e} dgp={o[3]:.3e}')
    break
