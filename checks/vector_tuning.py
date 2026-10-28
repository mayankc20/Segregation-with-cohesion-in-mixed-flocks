# ESM table S1: are the vector sums weak only because they use the model's parameters? The signed vector sum
# with a weaker minus sign c on the bearings of other kinds (c = 1: signed vector sum; c = 0: zero weight),
# and both rules at other weights of headings w. 10 runs each (seeds 3000-3009); about 2 hours on 4 cores.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flock_model import *
import flock_model
from joblib import Parallel, delayed
JOBS = [(c, 0.8) for c in (1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0)] + [(c, w) for c in (1, 0) for w in (0.5, 0.65, 0.95)]
def one(c, w, s):
    flock_model.CASES["tuned"] = dict(near=0.0, similarity=True, phase=PI, rule="vector", other_bearing=c)
    r = run("tuned", s, dict(BASE, w=w), T_RUN)
    return dict(w=w, c=c, M=final(r, "M"), Phi=final(r, "Phi"), one_group=float(r["rec"]["C"][-1]) > C_ONE)
d = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(one)(c, w, s) for c, w in JOBS for s in range(3000, 3010)))
d["below_005"] = d.M < 0.05
print(d.groupby(["w", "c"]).agg(M=("M", "mean"), below_005=("below_005", "sum"), Phi=("Phi", "mean"), one_group=("one_group", "sum")).round(2).to_string())
