# Appendix C and D: checks the closed forms of the overlaps (Bessel functions) against numerical integrals,
# and recomputes the worked example of Appendix D with and without the minus sign (a few seconds).
import numpy as np
from scipy.special import i0, i1
from scipy.integrate import quad
lam=50.0
bump=lambda th,d: np.exp(lam/2*(np.cos(th-d)-1))
# closed forms vs numerical integrals
rng=np.random.default_rng(0); err=0
for _ in range(200):
    dm,dl=rng.uniform(-np.pi,np.pi,2)
    W=quad(lambda t: bump(t,dm)*bump(t,dl),0,2*np.pi,limit=200)[0]
    Ox=quad(lambda t: bump(t,dm)*np.cos(t)*bump(t,dl),0,2*np.pi,limit=200)[0]
    D=np.angle(np.exp(1j*(dm-dl))); x=lam*np.cos(D/2); db=np.angle(np.exp(1j*dm)+np.exp(1j*dl))
    c=2*np.pi*np.exp(-lam)
    err=max(err,abs(W-c*i0(x))/max(W,1e-300),abs(Ox-c*i1(x)*np.cos(db))/max(abs(W),1e-300))
print("closed form max rel err", err)
# worked example
d2r=np.deg2rad
own=0; heads=d2r([10,5,20,-10,0,15]); bear=np.arctan2([1,-1,3,-3,0,4],[3,4,2,3,5,-1])
print("bearings", np.round(np.rad2deg(bear),1))
dirs=np.linspace(0,2*np.pi,120,endpoint=False)
def P(sign):
    a=np.r_[1,np.sqrt(.8)*np.ones(6),np.sqrt(.2)*sign]; a/=np.linalg.norm(a)
    d=np.r_[own,heads,bear]
    psi=(a[:,None]*bump(dirs[None,:],d[:,None])).sum(0); p=psi**2; return p/p.sum(), a
for name,sg in [("phase",np.array([1,1,1,-1,-1,-1])),("nophase",np.ones(6))]:
    p,a=P(sg); deg=np.rad2deg(np.angle(np.exp(1j*dirs)))
    v=(p*np.cos(dirs)).sum(),(p*np.sin(dirs)).sum()
    print(name,"coef",np.round(a,3)[:2],np.round(a[7:],3),"peak",deg[p.argmax()],round(p.max(),3),
      "P[-6,21]",round(p[(deg>=-6)&(deg<=21)].sum(),3),"P[-60,-9]",round(p[(deg>=-60)&(deg<=-9)].sum(),3),
      "exp heading",round(np.rad2deg(np.arctan2(v[1],v[0])),2), "v",np.round(v,6))
