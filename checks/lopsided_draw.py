# Section 3.3: does the minus sign make the draw lopsided, away from the other kind?
# Snapshots of the full model (seeds 3000-3002, every 100 steps up to step 1500). For each agent with a partner of
# another kind whose bearing is within BAND degrees of the peak, the percept (without memory) is computed with and
# without the minus sign, with the same partners. Reported: the probability within 45 degrees of the peak on the
# other kind's side minus that on the opposite side, and its change caused by the minus sign.
# Usage: python checks/lopsided_draw.py [BAND]      (the paper uses BAND = 60; about 2 minutes)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from flock_model import *
BAND = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
p = BASE; n = p["n"]
dirs = np.linspace(0, 2*np.pi, p["n_dir"], endpoint=False)
out = {"pi": [], "zero": []}
for seed in [3000, 3001, 3002]:
    r = run("full", seed, p, 1500, frame_every=100)
    _, _, kind, _ = start(p, seed)
    rng = np.random.default_rng(seed + 99)
    S = np.where(kind[:, None] == kind[None, :], 1.0, similarity_s(p["G"], p["delta0"]))
    for X, theta in r["frames"][1:]:
        N = len(theta)
        dx = X[None, :, 0] - X[:, None, 0]; dy = X[None, :, 1] - X[:, None, 1]
        dist = np.hypot(dx, dy); np.fill_diagonal(dist, np.inf)
        vis = (dist <= p["R"]) & (dx*np.cos(theta)[:, None] + dy*np.sin(theta)[:, None] >= np.cos(np.deg2rad(p["fov_deg"])/2)*dist)
        with np.errstate(divide="ignore"):
            score = np.where(vis, -rng.standard_exponential((N, N)) / S, -np.inf)
        part = np.argpartition(-score, n-1, axis=1)[:, :n]
        ok = np.isfinite(np.take_along_axis(score, part, 1)).all(1)
        rows = np.arange(N)[:, None]
        bear = np.arctan2(dy[rows, part], dx[rows, part])
        other = kind[part] != kind[:, None]
        use = ok & other.any(1)
        direction = np.c_[theta, theta[part], bear]
        states = np.exp(0.5*p["lam"]*(np.cos(dirs[None, None, :] - direction[:, :, None]) - 1))
        # side of the other kinds: mean direction of the other-kind bearings
        ob = np.where(other, 1.0, 0.0)
        for key, sgn in [("pi", -1.0), ("zero", 1.0)]:
            a = np.c_[np.ones(N), np.sqrt(p["w"])*np.ones((N, n)), np.sqrt(1-p["w"])*np.where(other, sgn, 1.0)]
            P = np.abs(np.einsum("im,imk->ik", a, states))**2; P /= P.sum(1, keepdims=True)
            pk = dirs[P.argmax(1)]
            off = np.angle(np.exp(1j*(bear - pk[:, None])))                    # other-kind bearings relative to the peak
            off = np.where(other, off, np.inf); j = np.abs(off).argmin(1)
            near_off = off[np.arange(N), j]
            side = np.sign(np.sin(near_off))
            use_k = use & (np.abs(near_off) < np.deg2rad(BAND))
            rel = np.sin(dirs[None, :] - pk[:, None]) * side[:, None]          # >0: toward the other kinds' side
            near = np.cos(dirs[None, :] - pk[:, None]) > np.cos(np.deg2rad(45))
            toward = (P*((rel > 0) & near)).sum(1); away = (P*((rel < 0) & near)).sum(1)
            out[key].append(np.c_[toward[use_k], away[use_k], np.abs(near_off[use_k])])
for key in out:
    v = np.concatenate(out[key]); t, a = v[:, 0], v[:, 1]
    print(f"phase {key:4s}: agents {len(v)}, mean P(toward other side) {t.mean():.3f}, P(away) {a.mean():.3f}, "
          f"ratio of means {t.mean()/a.mean():.2f}, share of agents with toward<away {(t < a).mean():.2f}")
a = np.concatenate(out["pi"]); b = np.concatenate(out["zero"])
if len(a) == len(b):
    d = (a[:, 0] - a[:, 1]) - (b[:, 0] - b[:, 1])
    print(f"paired change in P(toward) - P(away) from the minus sign: {d.mean():.4f} +- {d.std()/np.sqrt(len(d)):.4f} (SE), agents {len(d)}")
    print(f"P(toward) with minus sign / without: {a[:,0].mean()/b[:,0].mean():.3f}; P(away): {a[:,1].mean()/b[:,1].mean():.3f}")
