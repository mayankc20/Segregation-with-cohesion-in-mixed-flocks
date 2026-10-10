# Appendix B ("Choice of the measures"): does the mixing entropy depend on the number of neighbours k, and
# does cohesion depend on the link distance? Final states of 5 runs per case (about 10 minutes on 4 cores).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flock_model import *
from joblib import Parallel, delayed
def one(c, s):
    r = run(c, s, BASE, 5000, keep_final=True)
    X, th, kind = r["final"]
    return c, s, [mixing_entropy(X, kind, 3, m=k) for k in (5, 10, 20, 40)], \
           [float(np.mean([mixing_entropy(X, np.random.default_rng(1).permutation(kind), 3, m=k)])) for k in (5, 10, 20, 40)], \
           [cohesion(X, L) for L in (10.0, 15.0, 25.0)]
res = Parallel(n_jobs=N_JOBS)(delayed(one)(c, s) for c in ["full", "relational", "no_identity"] for s in range(3000, 3005))
for c in ["full", "relational", "no_identity"]:
    v = np.array([r[2] for r in res if r[0] == c]); rnd = np.array([r[3] for r in res if r[0] == c]); C = np.array([r[4] for r in res if r[0] == c])
    print(c, "M for k=5,10,20,40:", np.round(np.median(v, 0), 3), "random:", np.round(rnd.mean(0), 2), "C with link 10,15,25:", np.round(C.mean(0), 2))
