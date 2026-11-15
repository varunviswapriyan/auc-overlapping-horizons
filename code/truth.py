"""
truth.py -- population Delta-AUC for each non-null simulation scenario.

For gamma = 0 the two scores are exchangeable, so Delta = 0 exactly.
For gamma > 0, Delta is approximated by the average over M independent large
panels (N = 10 assets, T_BIG timestamps each); the Monte Carlo standard error
of that average is reported alongside.

Usage: python truth.py --out ../results/sim  (reads scenarios.csv written by simulation.py)
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulation as S
from auc_inference import placement_values

T_BIG, M = 200_000, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "sim"))
    a = ap.parse_args()
    sc = pd.read_csv(os.path.join(a.out, "scenarios.csv"))
    rows = []
    for _, r in sc.iterrows():
        d, aa = [], []
        for m in range(M):
            rng = np.random.default_rng([S.SEED, 555, int(r.sid), m])
            y, sa, sb = S.simulate_panel(rng, T_BIG, S.N_ASSETS, r.tau, int(r.h), r.gamma, r.mu)
            y, sa, sb = y.ravel(), sa.ravel(), sb.ravel()
            A = placement_values(y, sa)[0]; Bv = placement_values(y, sb)[0]
            d.append(A - Bv); aa.append(A)
        if r.gamma == 0:   # exact by exchangeability; the simulated mean is kept only as a check
            rows.append(dict(sid=int(r.sid), true_delta=0.0, true_delta_mcse=0.0, check_mean_delta=np.mean(d),
                             auc_a=np.mean(aa), base_rate=y.mean()))
        else:
            rows.append(dict(sid=int(r.sid), true_delta=np.mean(d), true_delta_mcse=np.std(d, ddof=1) / np.sqrt(M),
                             check_mean_delta=np.mean(d), auc_a=np.mean(aa), base_rate=y.mean()))
        print(int(r.sid), rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "truth.csv"), index=False)


if __name__ == "__main__":
    main()
