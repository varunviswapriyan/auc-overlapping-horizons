"""
application.py -- re-assess the 20 incremental-AUC comparisons of the
funding-dispersion application (Table T3 of the pipeline) with six inference
methods, verify that the fast weighted bootstrap reproduces the pipeline's
paired_boot_auc exactly, and benchmark run time.

Usage
  python application.py --pred ../fd_project/outputs/predictions.parquet --out ../results/application
"""
from __future__ import annotations
import argparse, os, sys, time, platform
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auc_inference import (analytic_tests, z_summary, weights_iid, weights_cluster,
                           weights_mbb, boot_delta, boot_summary,
                           paired_boot_auc_reference)

HORIZON_STEPS = {"dd5_8h": 1, "dd5_24h": 3, "dd10_24h": 3, "dd10_48h": 6, "dd10_7d": 21}
TARGETS = ["dd5_8h", "dd5_24h", "dd10_24h", "dd10_48h", "dd10_7d"]
B = 1000


def holm(p):
    p = np.asarray(p, float); m = p.size; o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def bandwidth(T, h):
    return int(np.ceil(4 * (T / 100) ** (2 / 9))) + h - 1


def prepare(C, target, model_a, model_b):
    """Merge and order exactly as funding_dispersion.paired_boot_auc does."""
    a = C[(C.target == target) & (C.model == model_a)]
    b = C[(C.target == target) & (C.model == model_b)]
    m = a.merge(b, on=["date", "asset"], suffixes=("_a", "_b"))
    m = m.sort_values("date").reset_index(drop=True)
    codes, _ = pd.factorize(m.date, sort=False)
    t_idx = ((m.date - m.date.min()) / pd.Timedelta(hours=8)).round().astype(int).to_numpy()
    return m.y_a.to_numpy().astype(bool), m.p_a.to_numpy(), m.p_b.to_numpy(), codes, t_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results", "application"))
    ap.add_argument("--skip_reference", action="store_true", help="skip the slow loop benchmark")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    P = pd.read_pickle(a.pred) if a.pred.endswith(".pkl") else pd.read_parquet(a.pred)
    C = P[P.target.isin(TARGETS)]
    rows = []
    for t in TARGETS:
        h = HORIZON_STEPS[t]
        for base in ("logit", "lgbm"):
            for alt in ("FULL", "STALE"):
                y, pa, pb, codes, t_idx = prepare(C, t, f"{base}_{alt}", f"{base}_BASE")
                T_grid = t_idx.max() + 1; K = codes.max() + 1
                L = bandwidth(T_grid, h)
                an = analytic_tests(y, pa, pb, t_idx, T_grid, L)
                res = {k: (d, se, *z_summary(d, se)) for k, (d, se) in an.items()}
                for lab, LL in (("hac_if_halfL", max(1, L // 2)), ("hac_if_2L", 2 * L)):
                    d, se = analytic_tests(y, pa, pb, t_idx, T_grid, LL)["hac_if"]
                    res[lab] = (d, se, *z_summary(d, se))
                delta = an["delong"][0]
                # boot_cl uses seed 7 and the same draw sequence as the pipeline (T3)
                for lab, W in (("boot_cl", weights_cluster(np.random.default_rng(7), codes, K, B)),
                               ("boot_iid", weights_iid(np.random.default_rng(8), y.size, B)),
                               ("boot_mbb", weights_mbb(np.random.default_rng(9), t_idx, T_grid, L, B))):
                    diffs = boot_delta(y, pa, pb, W)
                    res[lab] = (delta, diffs.std(ddof=1), *boot_summary(delta, diffs))
                for meth, (d, se, p, lo, hi) in res.items():
                    rows.append(dict(target=t, h=h, model=base, comparison=f"{alt}_minus_BASE", method=meth,
                                     dAUC=d, se=se, ci_lo=lo, ci_hi=hi, p_one_sided=p, n=y.size,
                                     n_pos=int(y.sum()), K=int(K), L=L))
                print(t, base, alt, "done", flush=True)
    R = pd.DataFrame(rows)
    for (meth, cmp_), g in R.groupby(["method", "comparison"]):
        R.loc[g.index, "p_holm"] = holm(g.p_one_sided.clip(1e-4, 1).to_numpy())
    R.to_csv(os.path.join(a.out, "A1_methods_all_comparisons.csv"), index=False)

    # se ratio relative to the pipeline's timestamp-cluster bootstrap
    piv = R.pivot_table(index=["target", "model", "comparison"], columns="method", values="se")
    (piv.div(piv["boot_cl"], axis=0)).round(3).to_csv(os.path.join(a.out, "A2_se_ratio_vs_boot_cl.csv"))

    # --- exact-equivalence check and run-time benchmark on the main comparison
    y, pa, pb, codes, t_idx = prepare(C, "dd10_24h", "lgbm_FULL", "lgbm_BASE")
    T_grid = t_idx.max() + 1; K = codes.max() + 1; L = bandwidth(T_grid, 3)
    bench = []
    t0 = time.perf_counter(); analytic_tests(y, pa, pb, t_idx, T_grid, L); bench.append(("analytic (DeLong, cluster, HAC)", time.perf_counter() - t0))
    t0 = time.perf_counter()
    W = weights_cluster(np.random.default_rng(7), codes, K, B); diffs = boot_delta(y, pa, pb, W)
    fast = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean()))
    bench.append(("weighted cluster bootstrap, B=1000", time.perf_counter() - t0))
    t0 = time.perf_counter(); W = weights_mbb(np.random.default_rng(9), t_idx, T_grid, L, B); boot_delta(y, pa, pb, W)
    bench.append(("weighted moving-block bootstrap, B=1000", time.perf_counter() - t0))
    eq = {}
    if not a.skip_reference:
        t0 = time.perf_counter(); ref = paired_boot_auc_reference(y, pa, pb, codes, B=B, seed=7)
        bench.append(("loop cluster bootstrap (pipeline code), B=1000", time.perf_counter() - t0))
        eq = dict(ref_lo=ref[1], ref_hi=ref[2], ref_p=ref[3], fast_lo=fast[0], fast_hi=fast[1], fast_p=fast[2],
                  max_abs_diff=max(abs(ref[1] - fast[0]), abs(ref[2] - fast[1]), abs(ref[3] - fast[2])))
        pd.Series(eq).to_csv(os.path.join(a.out, "A3_equivalence_check.csv"), header=["value"])
    bt = pd.DataFrame(bench, columns=["task", "seconds"])
    bt["n"] = y.size; bt["K"] = int(K); bt["machine"] = f"{platform.system()} {platform.machine()}, numpy {np.__version__}"
    bt.round(3).to_csv(os.path.join(a.out, "A4_runtime.csv"), index=False)
    print(bt); print(eq)


if __name__ == "__main__":
    main()
