# Section 3.8 (table 5): how wide must the visual cone be? Full model with the visual cone from 90 to 330 degrees,
# 10 runs each (seeds 3000-3009); everything else as in the paper (about 1.5 hours on 4 cores).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flock_model import *
from joblib import Parallel, delayed
ANGLES = [90, 120, 150, 180, 210, 240, 270, 300, 330]
def one(a, s):
    r = run("full", s, dict(BASE, fov_deg=float(a)), T_RUN)
    return dict(cone=a, M=final(r, "M"), Phi=final(r, "Phi"), one_group=float(r["rec"]["C"][-1]) > C_ONE)
d = pd.DataFrame(Parallel(n_jobs=N_JOBS)(delayed(one)(a, s) for a in ANGLES for s in range(3000, 3010)))
print(d.groupby("cone").agg(M=("M", "mean"), Phi=("Phi", "mean"), one_group=("one_group", "sum")).round(2).to_string())
