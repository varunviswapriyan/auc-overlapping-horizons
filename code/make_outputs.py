"""
make_outputs.py -- builds every simulation/application table (LaTeX + CSV) and
figure used in the manuscript from the raw result files.

Usage: python make_outputs.py --res ../results --tab ../manuscript/tables --fig ../manuscript/figures
"""
import argparse, os, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ["delong", "boot_iid", "cluster_if", "boot_cl", "hac_if", "boot_mbb"]
LABEL = {"delong": "DeLong", "boot_iid": "IID boot.", "cluster_if": "Cluster IF",
         "boot_cl": "Cluster boot.", "hac_if": "HAC IF", "boot_mbb": "Block boot.",
         "hac_if_halfL": "HAC IF, $L/2$", "hac_if_2L": "HAC IF, $2L$"}
# categorical slots 1-6 of the validated reference palette, fixed order; markers as secondary encoding
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARKER = dict(zip(METHODS, ["o", "s", "^", "v", "D", "P"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)


def load_sim(res):
    sc = pd.read_csv(os.path.join(res, "sim", "scenarios.csv"))
    tr = pd.read_csv(os.path.join(res, "sim", "truth.csv"))
    sc = sc.drop(columns=[c for c in sc.columns if c == "true_delta"]).merge(tr, on="sid")
    reps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(res, "sim", "reps_s*.csv")))])
    d = reps.merge(sc, on="sid")
    d["rej"] = d.p < 0.05
    d["cov"] = (d.lo <= d.true_delta) & (d.hi >= d.true_delta)
    d["width"] = d.hi - d.lo
    return sc, d


def summarise(d):
    g = d.groupby(["sid", "tau", "h", "gamma", "method"])
    S = g.agg(R=("rej", "size"), rej=("rej", "mean"), cov=("cov", "mean"), width=("width", "mean"),
              mean_delta=("delta_hat", "mean"), true_delta=("true_delta", "first"),
              n_pos=("n_pos", "mean")).reset_index()
    sd = d[d.method == "delong"].groupby("sid").delta_hat.std(ddof=1).rename("emp_sd")
    S = S.merge(sd, on="sid")
    S["width_ratio"] = S.width / (2 * 1.959964 * S.emp_sd)
    S["mcse_rej"] = np.sqrt(S.rej * (1 - S.rej) / S.R)
    S["mcse_cov"] = np.sqrt(S["cov"] * (1 - S["cov"]) / S.R)
    return S


def f3(x): return f"{x:.3f}"


def latex_null_table(S, path):
    N = S[S.gamma == 0]
    lines = [r"\begin{tabular}{rr" + "r" * 6 + "}", r"\toprule",
             r"$\tau$ & $h$ & " + " & ".join(LABEL[m] for m in METHODS) + r" \\", r"\midrule",
             r"\multicolumn{8}{l}{\textit{Panel A: rejection rate of the one-sided 5\% test ($\Delta=0$)}} \\"]
    for (t, h), g in N.groupby(["tau", "h"]):
        r = g.set_index("method")
        lines.append(f"{t:.1f} & {h} & " + " & ".join(f3(r.loc[m, 'rej']) for m in METHODS) + r" \\")
    lines += [r"\midrule", r"\multicolumn{8}{l}{\textit{Panel B: coverage of the 95\% confidence interval}} \\"]
    for (t, h), g in N.groupby(["tau", "h"]):
        r = g.set_index("method")
        lines.append(f"{t:.1f} & {h} & " + " & ".join(f3(r.loc[m, 'cov']) for m in METHODS) + r" \\")
    lines += [r"\midrule", r"\multicolumn{8}{l}{\textit{Panel C: mean CI width relative to $2\times1.96\times$ Monte Carlo s.d.\ of $\hat\Delta$}} \\"]
    for (t, h), g in N.groupby(["tau", "h"]):
        r = g.set_index("method")
        lines.append(f"{t:.1f} & {h} & " + " & ".join(f"{r.loc[m, 'width_ratio']:.2f}" for m in METHODS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def latex_alt_table(S, path):
    A = S[S.gamma > 0]
    lines = [r"\begin{tabular}{rrrr" + "r" * 6 + "}", r"\toprule",
             r"$\tau$ & $h$ & $\gamma$ & $\Delta$ & " + " & ".join(LABEL[m] for m in METHODS) + r" \\", r"\midrule",
             r"\multicolumn{10}{l}{\textit{Panel A: rejection rate of the one-sided 5\% test}} \\"]
    for (t, h, gm), g in A.groupby(["tau", "h", "gamma"]):
        r = g.set_index("method")
        lines.append(f"{t:.1f} & {h} & {gm:.2f} & {r.true_delta.iloc[0]:.4f} & " + " & ".join(f3(r.loc[m, 'rej']) for m in METHODS) + r" \\")
    lines += [r"\midrule", r"\multicolumn{10}{l}{\textit{Panel B: coverage of the 95\% confidence interval}} \\"]
    for (t, h, gm), g in A.groupby(["tau", "h", "gamma"]):
        r = g.set_index("method")
        lines.append(f"{t:.1f} & {h} & {gm:.2f} & {r.true_delta.iloc[0]:.4f} & " + " & ".join(f3(r.loc[m, 'cov']) for m in METHODS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def latex_bw_table(S, path):
    N = S[S.gamma == 0]
    ms = ["hac_if_halfL", "hac_if", "hac_if_2L"]
    lines = [r"\begin{tabular}{rrr" + "rrr" + "rrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Rejection rate} & \multicolumn{3}{c}{Coverage} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"$\tau$ & $h$ & $L$ & $L/2$ & $L$ & $2L$ & $L/2$ & $L$ & $2L$ \\", r"\midrule"]
    for (t, h), g in N.groupby(["tau", "h"]):
        r = g.set_index("method")
        lines.append(f"{t:.1f} & {h} & {{L{h}}} & " + " & ".join(f3(r.loc[m, 'rej']) for m in ms) + " & "
                     + " & ".join(f3(r.loc[m, 'cov']) for m in ms) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def fig_null(S, path):
    N = S[S.gamma == 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    hs = sorted(N.h.unique()); xpos = np.arange(len(hs))
    for ax, t in zip(axes, sorted(N.tau.unique())):
        g = N[N.tau == t]
        R = int(g.R.iloc[0]); band = 1.96 * np.sqrt(0.05 * 0.95 / R)
        ax.axhspan(0.05 - band, 0.05 + band, color=GRID, lw=0)
        ax.axhline(0.05, color=INK2, lw=0.8, ls=(0, (3, 2)))
        for m in METHODS:
            r = g[g.method == m].set_index("h").loc[hs]
            ax.plot(xpos, r.rej, color=COLOR[m], lw=2, marker=MARKER[m], ms=6,
                    markeredgecolor="white", markeredgewidth=0.8, label=LABEL[m])
        ax.set_xticks(xpos, [str(h) for h in hs]); style(ax)
        ax.set_xlabel("Target horizon $h$ (8-hour steps)", color=INK, fontsize=9)
        ax.set_title(fr"$\tau={t:.1f}$" + (" (no common shock)" if t == 0 else " (common shock)"), color=INK, fontsize=9)
    axes[0].set_ylabel("Rejection rate under $\\Delta=0$", color=INK, fontsize=9)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(path); fig.savefig(path.replace(".pdf", ".png"), dpi=200); plt.close(fig)


def app_tables(res, tab, fig_path):
    R = pd.read_csv(os.path.join(res, "application", "A1_methods_all_comparisons.csv"))
    tnames = {"dd5_8h": r"5\% / 8h", "dd5_24h": r"5\% / 24h", "dd10_24h": r"10\% / 24h",
              "dd10_48h": r"10\% / 48h", "dd10_7d": r"10\% / 7d"}
    order = ["dd5_8h", "dd5_24h", "dd10_24h", "dd10_48h", "dd10_7d"]
    show = ["boot_cl", "hac_if", "boot_mbb"]
    lines = [r"\begin{tabular}{llrr" + "rrr" * 3 + "}", r"\toprule",
             r" & & & & \multicolumn{3}{c}{Cluster boot.\ (original)} & \multicolumn{3}{c}{HAC IF} & \multicolumn{3}{c}{Block boot.} \\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
             r"Target & Comp. & $h$ & $\hat\Delta$ & CI low & CI high & $p$ & CI low & CI high & $p$ & CI low & CI high & $p$ \\", r"\midrule"]
    for cmp_, lab in (("FULL_minus_BASE", r"\textsc{Full}"), ("STALE_minus_BASE", r"\textsc{Stale}")):
        for t in order:
            for mdl, ml in (("logit", "Logit"), ("lgbm", "LGBM")):
                g = R[(R.target == t) & (R.model == mdl) & (R.comparison == cmp_)].set_index("method")
                cells = []
                for m in show:
                    cells += [f"{g.loc[m, 'ci_lo']:.4f}", f"{g.loc[m, 'ci_hi']:.4f}", f"{g.loc[m, 'p_one_sided']:.3f}"]
                cells = [c.replace("-", "$-$") for c in cells]
                d = f"{g.loc['delong', 'dAUC']:.4f}".replace("-", "$-$")
                lines.append(f"{tnames[t]} {ml} & {lab} & {int(g.h.iloc[0])} & {d} & " + " & ".join(cells) + r" \\")
        if cmp_ == "FULL_minus_BASE": lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tab, "tab_app_methods.tex"), "w").write("\n".join(lines) + "\n")

    # se ratio summary by horizon
    piv = R.pivot_table(index=["target", "model", "comparison", "h"], columns="method", values="se").reset_index()
    for m in METHODS: piv[m + "_ratio"] = piv[m] / piv["boot_cl"]
    summ = piv.groupby("h")[[m + "_ratio" for m in METHODS]].agg(["min", "median", "max"])
    summ.to_csv(os.path.join(tab, "tab_app_se_ratio_by_h.csv"))
    hs = sorted(piv.h.unique())
    lines = [r"\begin{tabular}{l" + "c" * len(hs) + "}", r"\toprule",
             r"Procedure & " + " & ".join(f"$h={h}$ ({(piv.h == h).sum()})" for h in hs) + r" \\", r"\midrule"]
    for m in METHODS:
        cells = []
        for h in hs:
            g = piv[piv.h == h][m + "_ratio"]
            cells.append(f"{g.median():.2f} [{g.min():.2f}, {g.max():.2f}]")
        lines.append(LABEL[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(tab, "tab_app_se_ratio.tex"), "w").write("\n".join(lines) + "\n")

    # figure: se ratio vs h
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    hs = sorted(piv.h.unique()); xpos = {h: i for i, h in enumerate(hs)}
    for j, m in enumerate(METHODS):
        x = np.array([xpos[h] for h in piv.h]) + (j - 2.5) * 0.08
        ax.scatter(x, piv[m + "_ratio"], s=22, color=COLOR[m], marker=MARKER[m], edgecolor="white", linewidths=0.4, label=LABEL[m], zorder=3)
    ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(range(len(hs)), [str(h) for h in hs]); style(ax)
    ax.set_xlabel("Target horizon $h$ (8-hour steps)", color=INK, fontsize=9)
    ax.set_ylabel("s.e. relative to cluster bootstrap", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_path); fig.savefig(fig_path.replace(".pdf", ".png"), dpi=200); plt.close(fig)
    return R


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--res", default=os.path.join(here, "..", "results"))
    ap.add_argument("--tab", default=os.path.join(here, "..", "manuscript", "tables"))
    ap.add_argument("--fig", default=os.path.join(here, "..", "manuscript", "figures"))
    a = ap.parse_args()
    os.makedirs(a.tab, exist_ok=True); os.makedirs(a.fig, exist_ok=True)
    sc, d = load_sim(a.res)
    S = summarise(d)
    S.round(5).to_csv(os.path.join(a.tab, "sim_summary.csv"), index=False)
    latex_null_table(S, os.path.join(a.tab, "tab_sim_null.tex"))
    latex_alt_table(S, os.path.join(a.tab, "tab_sim_alt.tex"))
    bw = os.path.join(a.tab, "tab_sim_bandwidth.tex"); latex_bw_table(S, bw)
    txt = open(bw).read()
    for h, L in sc.groupby("h").L.first().items(): txt = txt.replace(f"{{L{h}}}", str(L))
    open(bw, "w").write(txt)
    fig_null(S, os.path.join(a.fig, "fig_sim_size.pdf"))
    sc.round(6).to_csv(os.path.join(a.tab, "sim_scenarios.csv"), index=False)
    app_tables(a.res, a.tab, os.path.join(a.fig, "fig_app_se_ratio.pdf"))
    print("outputs written to", a.tab, "and", a.fig)


if __name__ == "__main__":
    main()
