# ESM table S2: Couzin et al. (2002) zonal model in open space, with one identity knob c that scales the
# attraction toward individuals of other kinds (c = 1: the published model, kinds ignored; c = 0: no
# attraction to other kinds; c < 0: pushed away from them). Orientation and repulsion ignore kinds.
# Published values (Couzin et al. 2002, J. Theor. Biol. 218:1-11): rr = 1, field of perception 270 deg
# (blind zone for orientation and attraction only), turning rate 40 deg/s, speed 3 units/s, tau = 0.1 s,
# noise sigma = 0.05 rad. Measures are those of the paper (flock_model.py): M (10 nearest neighbours),
# C with link = ra (the range of perception), Phi. Start as in the paper: N = 240, G = 3, disc of radius 17.5.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import flock_model as fm

RR, ALPHA, THETA, SPEED, TAU, SIGMA = 1.0, 270.0, 40.0, 3.0, 0.1, 0.05

def run(c, dro, dra, seed, T=8000, N=240, G=3, every=50):
    rng = np.random.default_rng(seed)
    r = 17.5 * np.sqrt(rng.random(N)); a = rng.random(N) * 2 * np.pi
    X = np.c_[r * np.cos(a), r * np.sin(a)]
    th = rng.random(N) * 2 * np.pi
    kind = rng.permutation(np.arange(N) % G)
    ro, ra = RR + dro, RR + dro + dra
    other = kind[None, :] != kind[:, None]
    wa = np.where(other, c, 1.0)                     # attraction weight per pair
    cos_half = np.cos(np.deg2rad(ALPHA) / 2)
    maxturn = np.deg2rad(THETA) * TAU
    rec = []
    for t in range(T):
        dx = X[None, :, 0] - X[:, None, 0]; dy = X[None, :, 1] - X[:, None, 1]
        d = np.hypot(dx, dy); np.fill_diagonal(d, np.inf)
        ux, uy = dx / d, dy / d
        hx, hy = np.cos(th), np.sin(th)
        seen = (ux * hx[:, None] + uy * hy[:, None]) >= cos_half
        rep = d < RR
        ori = seen & (d >= RR) & (d < ro)
        att = seen & (d >= ro) & (d < ra)
        # orientation: sum of neighbours' headings, including own (Couzin 2002)
        ox = (ori * hx[None, :]).sum(1) + np.where(ori.any(1), hx, 0)
        oy = (ori * hy[None, :]).sum(1) + np.where(ori.any(1), hy, 0)
        ax = (att * wa * ux).sum(1); ay = (att * wa * uy).sum(1)
        on, an = np.hypot(ox, oy), np.hypot(ax, ay)
        o_ok, a_ok = ori.any(1) & (on > 0), att.any(1) & (an > 0)
        dxs = np.where(o_ok, ox / np.where(on > 0, on, 1), 0) * np.where(o_ok & a_ok, 0.5, 1) \
            + np.where(a_ok, ax / np.where(an > 0, an, 1), 0) * np.where(o_ok & a_ok, 0.5, 1)
        dys = np.where(o_ok, oy / np.where(on > 0, on, 1), 0) * np.where(o_ok & a_ok, 0.5, 1) \
            + np.where(a_ok, ay / np.where(an > 0, an, 1), 0) * np.where(o_ok & a_ok, 0.5, 1)
        rx, ry = -(rep * ux).sum(1), -(rep * uy).sum(1)
        has_rep = rep.any(1)
        dxs = np.where(has_rep, rx, dxs); dys = np.where(has_rep, ry, dys)
        zero = np.hypot(dxs, dys) < 1e-12
        desired = np.where(zero, th, np.arctan2(dys, dxs)) + rng.normal(0, SIGMA, N)
        diff = (desired - th + np.pi) % (2 * np.pi) - np.pi
        th = (th + np.clip(diff, -maxturn, maxturn)) % (2 * np.pi)
        X = X + SPEED * TAU * np.c_[np.cos(th), np.sin(th)]
        if t % every == 0 or t == T - 1:
            rec.append((t, fm.mixing_entropy(X, kind, G), fm.cohesion(X, ra),
                        float(np.hypot(np.cos(th).sum(), np.sin(th).sum()) / N)))
    rec = np.array(rec)
    last = rec[rec[:, 0] >= 0.8 * T]
    return dict(c=c, dro=dro, dra=dra, seed=seed, M=float(last[:, 1].mean()), C=float(last[:, 2].mean()),
                Phi=float(last[:, 3].mean()), C_end=float(rec[-1, 2]), M_end=float(rec[-1, 1]))

if __name__ == "__main__":
    # c from 1 to -1 at the zone widths chosen with identity off (dro = 14, dra = 6), 10 runs each (about 30 min on 4 cores)
    import pandas as pd
    from joblib import Parallel, delayed
    CS = [1.0, 0.5, 0.25, 0.0, -0.02, -0.05, -0.1, -0.25, -0.5, -1.0]
    d = pd.DataFrame(Parallel(n_jobs=fm.N_JOBS)(delayed(run)(c, 14, 6, s) for c in CS for s in range(3000, 3010)))
    d["one_group"] = d.C_end > 0.9; d["below_005"] = d.M < 0.05
    print(d.groupby("c").agg(M=("M", "mean"), below_005=("below_005", "sum"), Phi=("Phi", "mean"), one_group=("one_group", "sum")).round(2).to_string())
