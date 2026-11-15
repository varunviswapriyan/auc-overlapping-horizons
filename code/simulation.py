"""
simulation.py -- Monte Carlo study of inference for paired Delta-AUC in a
panel with common-timestamp shocks and overlapping-horizon event targets.

Design (ADEMP, Morris, White & Crowther 2019) -- see the manuscript, Section 4.

Usage
  python simulation.py --reps 1000 --reps_alt 500 --B 499 --out ../results/sim
  python truth.py --out ../results/sim     # population Delta for each scenario
  python simulation.py --smoke --out ../results/smoke   # 20 reps, quick check

Every replication r of scenario s uses numpy.random.default_rng([SEED, s, r]),
so results do not depend on the number of worker processes or run order.
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auc_inference import all_methods, METHODS, placement_values, analytic_tests, z_summary

SEED = 20261113
N_ASSETS = 10
T_TEST = 1000          # test timestamps per replication
PHI = 0.9              # AR(1) persistence of all predictors
RHO = 0.5              # correlation of each candidate signal with the base predictor
KAPPA = 0.5            # weight of the candidate signal in each score
BETA = 0.5             # loading of the event latent on the base predictor (AUC about 0.73-0.75)
BASE_RATE = 0.05       # target unconditional event rate of y^(h)
BURN = 200

TAUS = [0.0, 0.5]
HS = [1, 3, 6, 21]          # = horizons (in 8h steps) of the application's targets
GAMMAS = [0.0, 0.05, 0.10]


def scenarios():
    return [dict(sid=i, tau=t, h=h, gamma=g) for i, (t, h, g) in
            enumerate(itertools.product(TAUS, HS, GAMMAS))]


def bandwidth(T, h):
    """Bartlett bandwidth / block length in timestamp steps."""
    return int(np.ceil(4 * (T / 100) ** (2 / 9))) + h - 1


def _ar1(rng, shape, phi):
    """Stationary unit-variance AR(1) along axis 0."""
    e = rng.standard_normal(shape) * np.sqrt(1 - phi ** 2)
    x = np.empty(shape)
    x[0] = rng.standard_normal(shape[1:])
    for t in range(1, shape[0]):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _factor(rng, T, N, tau, phi):
    """sqrt(tau) * common AR(1) + sqrt(1-tau) * idiosyncratic AR(1); unit variance."""
    c = _ar1(rng, (T, 1), phi)
    u = _ar1(rng, (T, N), phi)
    return np.sqrt(tau) * c + np.sqrt(1 - tau) * u


def simulate_panel(rng, T, N, tau, h, gamma, mu):
    """Returns arrays of shape (T, N): y (event within h steps), score_a, score_b."""
    Tt = T + h - 1 + BURN
    x = _factor(rng, Tt, N, tau, PHI)          # base predictor
    xa = _factor(rng, Tt, N, tau, PHI)         # candidate A innovation (signal if gamma>0)
    xb = _factor(rng, Tt, N, tau, PHI)         # candidate B innovation (never signal)
    za = RHO * x + np.sqrt(1 - RHO ** 2) * xa
    zb = RHO * x + np.sqrt(1 - RHO ** 2) * xb
    g = rng.standard_normal((Tt, 1))           # common event shock (serially independent)
    e = rng.standard_normal((Tt, N))
    eps = np.sqrt(tau) * g + np.sqrt(1 - tau) * e
    d = (mu + BETA * x + gamma * xa + eps) > 0  # one-step event indicator
    # y_t = 1{any event in t..t+h-1}
    y = np.zeros((Tt - h + 1, N), bool)
    for s in range(h):
        y |= d[s:Tt - h + 1 + s]
    sl = slice(BURN, BURN + T)
    return y[sl], (x + KAPPA * za)[sl], (x + KAPPA * zb)[sl]


def calibrate_mu(tau, h, gamma, T=20000, N=10, seed=0):
    """Intercept giving P(y=1) = BASE_RATE (bisection with common random numbers)."""
    lo, hi = -6.0, 0.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        rng = np.random.default_rng([SEED, 999, seed])
        y, _, _ = simulate_panel(rng, T, N, tau, h, gamma, mid)
        if y.mean() > BASE_RATE: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def one_rep(args):
    sc, r, mu, B = args
    rng = np.random.default_rng([SEED, sc["sid"], r])
    y, sa, sb = simulate_panel(rng, T_TEST, N_ASSETS, sc["tau"], sc["h"], sc["gamma"], mu)
    t_idx = np.repeat(np.arange(T_TEST), N_ASSETS)
    y, sa, sb = y.ravel(), sa.ravel(), sb.ravel()
    L = bandwidth(T_TEST, sc["h"])
    res = all_methods(y, sa, sb, t_idx, T_TEST, L, B, rng)
    # bandwidth sensitivity for the analytic HAC estimator (no extra randomness)
    for lab, LL in (("hac_if_halfL", max(1, L // 2)), ("hac_if_2L", 2 * L)):
        d, se = analytic_tests(y, sa, sb, t_idx, T_TEST, LL)["hac_if"]
        res[lab] = (d, *z_summary(d, se))
    rows = []
    for m in METHODS + ["hac_if_halfL", "hac_if_2L"]:
        d, p, lo, hi = res[m]
        rows.append(dict(sid=sc["sid"], rep=r, method=m, delta_hat=d, p=p, lo=lo, hi=hi,
                         n_pos=int(y.sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000, help="replications per null scenario")
    ap.add_argument("--reps_alt", type=int, default=500, help="replications per non-null scenario")
    ap.add_argument("--B", type=int, default=499)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "sim"))
    ap.add_argument("--only", type=str, default="", help="comma-separated scenario ids")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.reps, a.reps_alt, a.B = 20, 20, 99
    os.makedirs(a.out, exist_ok=True)
    scs = scenarios()
    if a.only: scs = [s for s in scs if str(s["sid"]) in a.only.split(",")]
    meta_path = os.path.join(a.out, "scenarios.csv")
    meta = []
    for sc in scs:
        mu = calibrate_mu(sc["tau"], sc["h"], sc["gamma"])
        meta.append(dict(**sc, mu=mu, L=bandwidth(T_TEST, sc["h"])))
        print(f"scenario {sc} mu={mu:.4f}", flush=True)
    pd.DataFrame(meta).to_csv(meta_path if not a.only else meta_path.replace(".csv", f"_{a.only.replace(',', '_')}.csv"), index=False)
    for m in meta:
        sc = {k: m[k] for k in ("sid", "tau", "h", "gamma")}
        f = os.path.join(a.out, f"reps_s{sc['sid']:02d}.csv")
        R = a.reps if sc["gamma"] == 0 else a.reps_alt
        if os.path.exists(f) and len(pd.read_csv(f)) == R * (len(METHODS) + 2):
            print("skip", f); continue
        t0 = time.time()
        jobs = [(sc, r, m["mu"], a.B) for r in range(R)]
        with Pool(a.jobs) as pool:
            rows = [row for out in pool.imap(one_rep, jobs, chunksize=4) for row in out]
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"done s{sc['sid']} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
