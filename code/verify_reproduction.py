"""
verify_reproduction.py -- regenerate results and compare them with the shipped ones.

  python verify_reproduction.py                 # application + two simulation scenarios + tables (~20 min)
  python verify_reproduction.py --sim all       # all 24 simulation scenarios (about 1.5 h on 2 cores)
  python verify_reproduction.py --pipeline      # also re-run the original pipeline's build,
                                                # evaluate and economics stages and diff its tables

Every check prints PASS/FAIL with the largest absolute difference found.
Run from the replication/code directory.
"""
import argparse, os, shutil, subprocess, sys, tempfile, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPL = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(REPL, "results")
PIPE = os.path.join(REPL, "pipeline")
PRED = os.path.join(PIPE, "fd_project", "outputs", "predictions.parquet")
TOL = 1e-10
fails = []


def cmp_csv(a, b, label, keys=None, tol=TOL):
    if open(a, "rb").read() == open(b, "rb").read():
        print(f"PASS {label}: byte-identical"); return
    A, B = pd.read_csv(a), pd.read_csv(b)
    if A.shape != B.shape or list(A.columns) != list(B.columns):
        print(f"FAIL {label}: shape/columns differ {A.shape} vs {B.shape}"); fails.append(label); return
    worst = 0.0
    for c in A.columns:
        if pd.api.types.is_numeric_dtype(A[c]):
            x, y = A[c].to_numpy(float), B[c].to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0, np.abs(x - y)); worst = max(worst, np.nanmax(d) if d.size else 0)
            if np.any(np.isnan(x) != np.isnan(y)): worst = np.inf
        elif not (A[c].astype(object).fillna("").astype(str).to_numpy() == B[c].astype(object).fillna("").astype(str).to_numpy()).all():
            worst = np.inf
    ok = worst <= tol
    print(f"{'PASS' if ok else 'FAIL'} {label}: max abs diff {worst:.3g}")
    if not ok: fails.append(label)


def run(cmd, cwd=HERE):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="0,21", help="comma-separated scenario ids, or 'all'")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--pipeline", action="store_true")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="cssc_verify_")
    py = sys.executable

    # 1. application (also checks fast bootstrap == original loop)
    run([py, "application.py", "--pred", PRED, "--out", os.path.join(tmp, "application")])
    for f in ("A1_methods_all_comparisons.csv", "A2_se_ratio_vs_boot_cl.csv"):
        cmp_csv(os.path.join(tmp, "application", f), os.path.join(RES, "application", f), "application " + f)
    eq = pd.read_csv(os.path.join(tmp, "application", "A3_equivalence_check.csv"), index_col=0)["value"]
    ok = eq["max_abs_diff"] < 1e-12
    print(f"{'PASS' if ok else 'FAIL'} weighted cluster bootstrap equals original loop: max abs diff {eq['max_abs_diff']:.3g}")
    if not ok: fails.append("equivalence")
    # the pipeline's own T3 table (rounded to 4 d.p.) against the recomputed cluster bootstrap
    A1 = pd.read_csv(os.path.join(tmp, "application", "A1_methods_all_comparisons.csv"))
    T3 = pd.read_csv(os.path.join(PIPE, "fd_project", "outputs", "tables", "T3_auc_gain_from_dispersion.csv"))
    m = A1[A1.method == "boot_cl"].merge(T3, on=["target", "model", "comparison"], suffixes=("", "_t3"))
    d = max(np.abs(m[c].round(4) - m[c + "_t3"]).max() for c in ("dAUC", "ci_lo", "ci_hi", "p_one_sided"))
    print(f"{'PASS' if d == 0 and len(m) == 20 else 'FAIL'} cluster bootstrap reproduces pipeline table T3 (20 rows): max abs diff {d:.3g}")
    if not (d == 0 and len(m) == 20): fails.append("T3")

    # 2. simulation
    sims = os.path.join(tmp, "sim")
    only = [] if a.sim == "all" else ["--only", a.sim]
    run([py, "simulation.py", "--jobs", str(a.jobs), "--out", sims] + only)
    for f in sorted(glob.glob(os.path.join(sims, "reps_s*.csv"))):
        cmp_csv(f, os.path.join(RES, "sim", os.path.basename(f)), "simulation " + os.path.basename(f))
    sc_new = sorted(glob.glob(os.path.join(sims, "scenarios*.csv")))[0]
    new, old = pd.read_csv(sc_new), pd.read_csv(os.path.join(RES, "sim", "scenarios.csv"))
    old = old[old.sid.isin(new.sid)].reset_index(drop=True)
    dd = np.abs(new[["mu", "L"]].to_numpy() - old[["mu", "L"]].to_numpy()).max()
    print(f"{'PASS' if dd <= TOL else 'FAIL'} calibrated intercepts and bandwidths: max abs diff {dd:.3g}")
    if dd > TOL: fails.append("scenarios")
    if a.sim == "all":
        run([py, "truth.py", "--out", sims])
        cmp_csv(os.path.join(sims, "truth.csv"), os.path.join(RES, "sim", "truth.csv"), "simulation truth.csv")

    # 3. tables and figures from the shipped results
    tabs = os.path.join(tmp, "tables")
    run([py, "make_outputs.py", "--res", RES, "--tab", tabs, "--fig", os.path.join(tmp, "figures")])
    for f in sorted(glob.glob(os.path.join(tabs, "*.tex"))):
        ref = os.path.join(REPL, "..", "manuscript", "tables", os.path.basename(f))
        if os.path.exists(ref):
            same = open(f).read() == open(ref).read()
            print(f"{'PASS' if same else 'FAIL'} manuscript table {os.path.basename(f)} identical")
            if not same: fails.append(f)

    # 4. optional: original pipeline stages that do not need LightGBM/SHAP/arch
    if a.pipeline:
        pcopy = os.path.join(tmp, "pipeline"); shutil.copytree(PIPE, pcopy)
        for st in ("build", "evaluate", "economics"):
            run([py, "funding_dispersion.py", "--stage", st], cwd=pcopy)
        for f in sorted(glob.glob(os.path.join(PIPE, "fd_project", "outputs", "tables", "*.csv"))):
            g = os.path.join(pcopy, "fd_project", "outputs", "tables", os.path.basename(f))
            cmp_csv(g, f, "pipeline " + os.path.basename(f), tol=1e-9)

    print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
    print("scratch directory:", tmp)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
