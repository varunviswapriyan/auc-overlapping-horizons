"""
auc_inference.py -- inference for the difference of two paired AUCs under
cross-sectional (common-timestamp) and serial dependence.

Methods (all for Delta = AUC(score_a) - AUC(score_b) on the same observations):

  Analytic (influence-function / placement-value based)
    delong      DeLong, DeLong & Clarke-Pearson (1988) variance (independent obs.)
    cluster_if  cluster-robust version: influence values summed within timestamps
    hac_if      Driscoll-Kraay-type version: timestamp sums + Bartlett kernel over lags

  Resampling (implemented as multiplicity weights on a pre-sorted score vector)
    boot_iid    paired bootstrap of observations
    boot_cl     paired bootstrap of timestamps (clusters)
    boot_mbb    paired moving-block bootstrap of consecutive timestamps

The weighted-AUC kernel reproduces sklearn.metrics.roc_auc_score on a resampled
data set exactly (ties are counted 1/2), because the AUC of a resampled data set
depends only on how many times each original observation was drawn.

Only numpy (and scipy.stats.norm for p-values) is required.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

# ----------------------------------------------------------------------------
# basic AUC objects
# ----------------------------------------------------------------------------

def _midranks(x):
    """Midranks (1-based, ties averaged) of a 1-d array."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = xs.size
    # boundaries of tie groups
    new = np.r_[True, xs[1:] != xs[:-1]]
    gid = np.cumsum(new) - 1
    starts = np.flatnonzero(new)
    ends = np.r_[starts[1:], n]
    mid = (starts + ends + 1) / 2.0          # average of 1-based ranks start+1..end
    r = np.empty(n)
    r[order] = mid[gid]
    return r


def placement_values(y, s):
    """DeLong placement values via midranks (Sun & Xu 2014).

    Returns (auc, v) where v[k] is V10 for a positive and V01 for a negative:
      V10(x_i) = #{neg < x_i} + 0.5 #{neg = x_i}  divided by n0
      V01(y_j) = #{pos > y_j} + 0.5 #{pos = y_j}  divided by n1
    """
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    r_all = _midranks(s)
    v = np.empty(s.size)
    r_pos = _midranks(s[y]); r_neg = _midranks(s[~y])
    v[y] = (r_all[y] - r_pos) / n0
    v[~y] = 1.0 - (r_all[~y] - r_neg) / n1
    auc = v[y].mean()
    return auc, v


def influence(y, s):
    """First-order influence contributions psi_k with sum_k psi_k = 0 and
    AUC_hat - AUC ~= sum_k psi_k  (Hoeffding projection of the U-statistic)."""
    y = np.asarray(y).astype(bool)
    auc, v = placement_values(y, s)
    n1, n0 = y.sum(), (~y).sum()
    psi = np.where(y, (v - auc) / n1, (v - auc) / n0)
    return auc, v, psi


# ----------------------------------------------------------------------------
# analytic variance estimators for Delta
# ----------------------------------------------------------------------------

def _timestamp_sums(d, t_idx, T_grid):
    D = np.zeros(T_grid)
    np.add.at(D, t_idx, d)
    return D


def analytic_tests(y, sa, sb, t_idx, T_grid, L):
    """Return dict method -> (delta, se) for delong, cluster_if, hac_if.

    t_idx : integer timestamp index on a regular grid 0..T_grid-1 (8h steps).
    L     : Bartlett bandwidth in grid steps for hac_if.
    """
    y = np.asarray(y).astype(bool)
    auc_a, va, pa = influence(y, sa)
    auc_b, vb, pb = influence(y, sb)
    delta = auc_a - auc_b
    n1, n0 = y.sum(), (~y).sum()
    # DeLong (1988): S10/n1 + S01/n0 applied to the difference of placement values
    dv = va - vb
    var_dl = dv[y].var(ddof=1) / n1 + dv[~y].var(ddof=1) / n0
    # cluster-robust (timestamps as clusters)
    d = pa - pb
    D = _timestamp_sums(d, t_idx, T_grid)
    occupied = np.zeros(T_grid, bool); occupied[t_idx] = True
    K = occupied.sum()
    g0 = np.dot(D, D)
    var_cl = g0 * K / (K - 1)
    # Driscoll-Kraay type HAC over the timestamp grid
    var_hac = g0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        var_hac += 2.0 * w * np.dot(D[l:], D[:-l])
    var_hac *= K / (K - 1)
    out = {"delong": (delta, np.sqrt(var_dl)),
           "cluster_if": (delta, np.sqrt(var_cl)),
           "hac_if": (delta, np.sqrt(max(var_hac, 0.0)))}
    return out


def z_summary(delta, se, alpha=0.05):
    z = delta / se
    p_one = 1.0 - norm.cdf(z)                 # H0: Delta <= 0 vs H1: Delta > 0
    q = norm.ppf(1 - alpha / 2)
    return p_one, delta - q * se, delta + q * se


# ----------------------------------------------------------------------------
# weighted AUC kernel (exact AUC of resampled data sets)
# ----------------------------------------------------------------------------

class SortedScore:
    """Pre-sorts one score vector once; afterwards the AUC for any matrix of
    non-negative observation weights W (B x n) costs O(B n)."""

    def __init__(self, y, s):
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, dtype=float)
        self.order = np.argsort(s, kind="mergesort")
        ss = s[self.order]
        new = np.r_[True, ss[1:] != ss[:-1]]
        self.starts = np.flatnonzero(new)
        self.ys = y[self.order].astype(float)
        self.has_ties = self.starts.size < s.size

    def auc(self, W):
        W = np.atleast_2d(W)
        Ws = W[:, self.order]
        Pw = Ws * self.ys
        Nw = Ws - Pw
        if self.has_ties:
            P = np.add.reduceat(Pw, self.starts, axis=1)
            N = np.add.reduceat(Nw, self.starts, axis=1)
        else:
            P, N = Pw, Nw
        cumN = np.cumsum(N, axis=1)
        num = (P * (cumN - 0.5 * N)).sum(axis=1)   # neg strictly below + half ties
        Ptot = P.sum(axis=1); Ntot = cumN[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / (Ptot * Ntot)


# ----------------------------------------------------------------------------
# bootstrap weight generators
# ----------------------------------------------------------------------------

def weights_iid(rng, n, B):
    W = np.empty((B, n))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def weights_cluster(rng, codes, K, B):
    """codes: cluster id (0..K-1) per observation. Draws clusters exactly as the
    original paired_boot_auc does: rng.integers(0, K, K) once per replication."""
    W = np.empty((B, codes.size))
    for b in range(B):
        cnt = np.bincount(rng.integers(0, K, K), minlength=K)
        W[b] = cnt[codes]
    return W


def weights_mbb(rng, t_idx, T_grid, L, B):
    """Moving-block bootstrap over the regular timestamp grid.
    ceil(T_grid/L) blocks of L consecutive grid steps, concatenated and truncated
    to T_grid steps; each observation inherits its timestamp's count."""
    nb = int(np.ceil(T_grid / L))
    W = np.empty((B, t_idx.size))
    ar = np.arange(L)
    for b in range(B):
        st = rng.integers(0, T_grid - L + 1, nb)
        idx = (st[:, None] + ar[None, :]).ravel()[:T_grid]
        cnt = np.bincount(idx, minlength=T_grid)
        W[b] = cnt[t_idx]
    return W


def boot_delta(y, sa, sb, W, chunk=100):
    """Bootstrap distribution of Delta for weight matrix W (B x n).
    Replications with a single class are dropped (as in the original code)."""
    A = SortedScore(y, sa); Bs = SortedScore(y, sb)
    yv = np.asarray(y).astype(float)
    out = []
    for i in range(0, W.shape[0], chunk):
        w = W[i:i + chunk]
        pos = w @ yv; neg = w.sum(1) - pos
        ok = (pos > 0) & (neg > 0)
        d = A.auc(w) - Bs.auc(w)
        out.append(d[ok])
    return np.concatenate(out)


def boot_summary(delta, diffs, alpha=0.05):
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_one = float((diffs <= 0).mean())
    return p_one, lo, hi


# ----------------------------------------------------------------------------
# reference implementation copied verbatim (logic) from funding_dispersion.py
# ----------------------------------------------------------------------------

def paired_boot_auc_reference(ya, pa, pb, codes, B=1000, seed=7):
    """Loop implementation of the original pipeline's paired_boot_auc on arrays
    already sorted by timestamp, with cluster codes in order of appearance."""
    from sklearn.metrics import roc_auc_score
    K = codes.max() + 1
    starts = np.searchsorted(codes, np.arange(K))
    ends = np.r_[starts[1:], len(codes)]
    rng = np.random.default_rng(seed); diffs = np.empty(B); kept = 0
    for i in range(B):
        pick = rng.integers(0, K, K)
        idx = np.concatenate([np.arange(starts[k], ends[k]) for k in pick])
        y, a, b = ya[idx], pa[idx], pb[idx]
        if np.unique(y).size < 2:
            continue
        diffs[kept] = roc_auc_score(y, a) - roc_auc_score(y, b); kept += 1
    diffs = diffs[:kept]
    point = float(roc_auc_score(ya, pa) - roc_auc_score(ya, pb))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean())


# ----------------------------------------------------------------------------
# convenience: run all six methods
# ----------------------------------------------------------------------------

METHODS = ["delong", "boot_iid", "cluster_if", "boot_cl", "hac_if", "boot_mbb"]


def all_methods(y, sa, sb, t_idx, T_grid, L, B, rng, chunk=100):
    """Returns dict method -> (delta, p_one_sided, ci_lo, ci_hi)."""
    y = np.asarray(y).astype(bool)
    res = {}
    an = analytic_tests(y, sa, sb, t_idx, T_grid, L)
    for m, (d, se) in an.items():
        res[m] = (d, *z_summary(d, se))
    delta = an["delong"][0]
    n = y.size
    codes = np.unique(t_idx, return_inverse=True)[1]
    K = codes.max() + 1
    for m, W in (("boot_iid", weights_iid(rng, n, B)),
                 ("boot_cl", weights_cluster(rng, codes, K, B)),
                 ("boot_mbb", weights_mbb(rng, t_idx, T_grid, L, B))):
        diffs = boot_delta(y, sa, sb, W, chunk)
        res[m] = (delta, *boot_summary(delta, diffs))
    return res
