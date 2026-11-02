# Section 4: what if the weight of a partner falls with its distance? Full model with the partner-draw weight
# s_ij / r_ij^alpha instead of s_ij, for alpha = 0 (the paper), 1 and 2; 30 runs each (seeds 3000-3029), about 1 hour on 4 cores.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flock_model import *
import flock_model
from joblib import Parallel, delayed
def one(alpha, s):
    flock_model.CASES["decay"] = dict(CASES["full"], alpha=alpha)
    r = run("decay", s, BASE, T_RUN)
    return dict(alpha=alpha, M=final(r, "M"), Phi=final(r, "Phi"), one_group=float(r["rec"]["C"][-1]) > C_ONE)
d = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(one)(a, s) for a in (0.0, 1.0, 2.0) for s in range(3000, 3030)))
d["below_005"] = d.M < 0.05
print(d.groupby("alpha").agg(M=("M", "mean"), below_005=("below_005", "sum"), Phi=("Phi", "mean"), one_group=("one_group", "sum")).round(2).to_string())
