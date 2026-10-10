# Section 5.5: do the subgroups drift apart over time? Distance between the centres of the nearest and the
# farthest subgroups at steps 1000-5000, and cohesion for longer link distances, for the signed vector sum,
# the vector sum with other kinds' bearings ignored, and the full model (4 runs each; about 10 minutes on 4 cores).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flock_model import *
from joblib import Parallel, delayed
def one(c, s):
    r = run(c, s, BASE, 5000, keep_final=True)
    X, th, kind = r["final"]
    t = r["rec"]["t"]; g = r["rec"]["gap_min"]; G2 = r["rec"]["gap_max"]
    pick = [np.argmin(abs(t - x)) for x in (1000, 2000, 3000, 4000, 5000)]
    return c, s, g[pick], G2[pick], [cohesion(X, L) for L in (25.0, 50.0, 100.0, 1e6)]
res = Parallel(n_jobs=N_JOBS)(delayed(one)(c, s) for c in ["vector", "vector_zero", "full"] for s in range(3000, 3004))
for c in ["vector", "vector_zero", "full"]:
    a = np.array([r[2] for r in res if r[0] == c]); b = np.array([r[3] for r in res if r[0] == c]); C = np.array([r[4] for r in res if r[0] == c])
    print(c, "nearest-subgroup gap at t=1000..5000:", np.round(a.mean(0), 0), "| farthest:", np.round(b.mean(0), 0), "| C with link 25, 50, 100, inf:", np.round(C.mean(0), 2))
