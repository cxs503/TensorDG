import numpy as np
# plane-strain elasticity, compression-positive convention for stresses
def D_plane(E, nu):
    a=E/((1+nu)*(1-2*nu))
    return np.array([[a*(1-nu), a*nu, 0.0],[a*nu, a*(1-nu),0.0],[0.0,0.0,a*(1-2*nu)/2]])

def tensor_from_voigt(v):  # v = (xx,yy,xy)
    return np.array([[v[0], v[2]],[v[2], v[1]]])
def voigt_from_tensor(T):
    return np.array([T[0,0], T[1,1], T[0,1]])

def C_ep(D, P, Q, h):
    # D 3x3 voigt, P,Q 3x3 voigt; C_ep = D - (D Q)(P D)/(h + P D Q)
    DQ = D@Q; PD = P@D
    denom = h + P@D@Q
    return D - np.outer(DQ, PD)/denom

def acout(n2, Cep):
    # A_ik = n_j Cep_ijkl n_l  (2D, voigt->tensor)
    def C4(C):
        C4 = np.zeros((2,2,2,2))
        idx=[(0,0),(1,1),(0,1)]
        for a,(i,j) in enumerate(idx):
            for b,(k,l) in enumerate(idx):
                val=C[a,b]
                for (ii,jj) in {(i,j),(j,i)}:
                    for (kk,ll) in {(k,l),(l,k)}:
                        pass
        # simpler: build via convention C_ijkl = C[a,b] with pairs (ij)->a, (kl)->b, and mapping factors
        return None
    # use formulation: A_ik = n_j C_ijkl n_l ; with C from voigt by summing over basis
    B = []  # basis tensors for voigt components
    pairs=[(0,0),(1,1),(0,1)]
    C = np.zeros((2,2,2,2))
    for a,(i,j) in enumerate(pairs):
        for b,(k,l) in enumerate(pairs):
            # distribute symmetric components
            fac = 1.0
            if i!=j: fac*=1.0
            for (ii,jj),fi in [((i,j),1.0)]+([((j,i),1.0)] if i!=j else []):
                for (kk,ll),fk in [((k,l),1.0)]+([((l,k),1.0)] if k!=l else []):
                    w = fi*fk
                    C[ii,jj,kk,ll] += Cep[a,b]*w
    return C

def A_mat(theta, Cep):
    C = acout(None, Cep)
    n = np.array([np.cos(theta), np.sin(theta)])
    A = np.einsum('j,ijkl,l->ik', n, C, n)
    return A

def band_angle(E,nu,phi,psi,s1,s3,h=-1e6, deg_range=(0,90)):
    D=D_plane(E,nu)
    P = np.array([1.0-np.sin(phi), -(1.0+np.sin(phi)), 0.0])
    Q = np.array([1.0-np.sin(psi), -(1.0+np.sin(psi)), 0.0])
    Cep=C_ep(D,P,Q,h)
    best=None
    thetas=np.linspace(np.deg2rad(deg_range[0])+1e-6, np.deg2rad(deg_range[1])-1e-6, 90001)
    dets=[]
    for t in thetas:
        A=A_mat(t,Cep); dets.append(np.linalg.det(A))
    dets=np.array(dets)
    # find sign changes
    sc=np.where(np.sign(dets[:-1])!=np.sign(dets[1:]))[0]
    print('sign changes at angles(deg):', np.rad2deg(thetas[sc]) if len(sc) else None)
    return thetas[dets.argmin()], thetas, dets

for phi,psi in [(40,40),(40,15),(30,0),(35,10)]:
    t,th,d=band_angle(30e6,0.3,np.deg2rad(phi),np.deg2rad(psi),1.0,0.0)
    print(f'phi={phi} psi={psi}: argmin-det theta={np.rad2deg(t):.2f} deg, closed-form 45+(phi+psi)/4={45+(phi+psi)/4:.2f}, 45+phi/2={45+phi/2:.2f}')
