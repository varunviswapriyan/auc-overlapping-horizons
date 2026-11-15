#!/usr/bin/env python3
"""
funding_dispersion.py  (v2: native 8h settlement grid)
=======================================================
Cross-exchange funding-rate dispersion as a short-horizon stress signal in crypto perpetuals.
Pipeline: fetch -> build (8h grid) -> models (expanding-window quarterly refits)
          -> evaluate (bootstrap, Holm, DM, MCS, stability, lead time) -> economics -> outputs

Run:  python3 funding_dispersion.py --stage all
Stages: fetch | build | models | evaluate | economics | outputs | all
Outputs land in ./fd_project/outputs/{tables,figures}.

Information timing: every feature at grid time E (00/08/16 UTC) uses only data from the settlement
window [E-8h, E).  Targets are measured strictly after E.  Venues settle at different clock times
(Binance/Phemex 00/08/16, KuCoin 04/12/20, Deribit hourly 8h-rate), all inside the window.
"""
import argparse, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(7)

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parent / "fd_project"
RAW = ROOT / "data" / "raw"; MANUAL = RAW / "manual"
OUT = ROOT / "outputs"; FIG = OUT / "figures"; TAB = OUT / "tables"
for p in (RAW, MANUAL, FIG, TAB): p.mkdir(parents=True, exist_ok=True)

ASSETS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
CORE = ["BTC", "ETH"]
EXCHANGES = ["binance", "kucoinfutures", "deribit", "phemex"]
PAGE_LIMIT = {"binance": 1000, "kucoinfutures": 200, "deribit": 500, "phemex": 1000}
MIN_EX = 3
START = "2021-01-01"
TEST_START = "2023-01-01"            # first out-of-sample quarter
PPY = 3 * 365                        # 8h periods per year
FEE_BP = 5                           # taker fee, basis points, per position change
STALE_LAG = 21                       # 7 days in 8h settlements (placebo)
DD_THRESHOLDS = [(8, 0.05, "dd5_8h"), (24, 0.05, "dd5_24h"), (24, 0.10, "dd10_24h"), (48, 0.10, "dd10_48h"), (168, 0.10, "dd10_7d")]
MAIN_CLS = "dd10_24h"
EPISODES = {                         # UTC dates, inclusive
    "May-2021 crash": ("2021-05-12", "2021-05-25"),
    "LUNA/UST": ("2022-05-07", "2022-05-16"),
    "3AC/Celsius": ("2022-06-10", "2022-06-20"),
    "FTX": ("2022-11-06", "2022-11-15"),
    "Mar-2023 USDC depeg": ("2023-03-09", "2023-03-13"),
    "Aug-2023 flash crash": ("2023-08-17", "2023-08-18"),
    "Aug-2024 unwind": ("2024-08-02", "2024-08-07"),
    "Oct-2025 cascade": ("2025-10-09", "2025-10-13"),
}
DISP_FAMILIES = {"sd": "Cross-venue SD", "rng": "Range", "scl": "Scaled SD", "sgn": "Sign disagreement"}
BASE_NUM = ["f_level", "f_mom", "n_ex", "lrv8", "lrv24", "lrv168", "r8", "r24", "r168", "fng"]
DISP = ["dsd", "dsd_max3", "drng", "drng_max3", "dscl", "dsgn"]
STALE = [c + "_l21" for c in DISP]

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ms(s): return int(pd.Timestamp(s, tz="UTC").timestamp() * 1000)
def save_table(df, name, float_fmt="%.4f"):
    df.to_csv(TAB / f"{name}.csv")
    try: (TAB / f"{name}.tex").write_text(df.to_latex(float_format=lambda x: float_fmt % x if isinstance(x, float) else str(x)))
    except Exception as e: log("latex skip", name, e)
def savefig(fig, name):
    import matplotlib.pyplot as plt
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)
def fwd(s, H, how):
    """Forward-looking window over the next H observations (strictly after t)."""
    r = s.shift(-1)[::-1].rolling(H, min_periods=H)
    return (r.min() if how == "min" else r.sum())[::-1]

# =============================================================================
# STAGE 1: FETCH   (Binance via bulk portal; KuCoin forward paging; Deribit windowed; Phemex forward)
# =============================================================================
VISION = "https://data.binance.vision/data"

def _ccxt(name, swap=True):
    import ccxt
    ex = getattr(ccxt, name)({"enableRateLimit": True, "options": {"defaultType": "swap" if swap else "spot"}})
    ex.load_markets(); return ex

def _months():
    return [str(p) for p in pd.period_range(START, pd.Timestamp.utcnow().tz_localize(None), freq="M")]

def _vision_csv(url):
    import requests, zipfile, io
    try: r = requests.get(url, timeout=60)
    except Exception as e: log("vision error", e); return None
    if r.status_code != 200: return None
    z = zipfile.ZipFile(io.BytesIO(r.content)); raw = z.read(z.namelist()[0]).decode()
    first = raw.splitlines()[0].split(",")[0]
    return pd.read_csv(io.StringIO(raw), header=0 if any(ch.isalpha() for ch in first) else None)

def _fix_ts(ts):
    ts = np.asarray(ts, dtype=np.int64); return np.where(ts > 10**14, ts // 1000, ts)

def fetch_funding_binance_vision(asset):
    rows = []
    for m in _months():
        df = _vision_csv(f"{VISION}/futures/um/monthly/fundingRate/{asset}USDT/{asset}USDT-fundingRate-{m}.zip")
        if df is None or len(df) == 0: continue
        cols = [str(c).lower() for c in df.columns]
        tcol = next((df.columns[i] for i, c in enumerate(cols) if "time" in c), df.columns[0])
        rcol = next((df.columns[i] for i, c in enumerate(cols) if "rate" in c and "interval" not in c), df.columns[-1])
        rows += list(zip(_fix_ts(df[tcol].values), df[rcol].astype(float).values))
    return rows

def _collect(batch, seen, rows):
    new = [b for b in (batch or []) if b.get("timestamp") and b["timestamp"] not in seen and b.get("fundingRate") is not None]
    for b in new: seen.add(b["timestamp"]); rows.append((int(b["timestamp"]), float(b["fundingRate"])))
    return new

def _paginate_forward(ex, sym, limit):
    since, seen, rows, it, now = ms(START), set(), [], 0, ex.milliseconds()
    while it < 4000:
        it += 1
        try: batch = ex.fetch_funding_rate_history(sym, since=since, limit=limit)
        except Exception as e:
            if "rate" in str(e).lower() and "limit" in str(e).lower(): time.sleep(5); continue
            log(f"[{ex.id}] forward error: {str(e)[:90]}"); break
        new = _collect(batch, seen, rows)
        if not new: break
        last = max(b["timestamp"] for b in new)
        if last <= since: break
        since = last + 1
        if last >= now - 8 * 3600 * 1000: break
    return rows

def _paginate_deribit(ex, sym, window_days=30):
    rows, seen, beg, now = [], set(), ms(START), ex.milliseconds()
    while beg < now:
        end = min(now, beg + window_days * 86400 * 1000)
        try: batch = ex.fetch_funding_rate_history(sym, since=beg, params={"until": end})
        except Exception as e: log(f"[deribit] error: {str(e)[:90]}"); break
        _collect(batch, seen, rows); beg = end + 1
    return rows

def fetch_funding(exname, asset):
    f = RAW / f"funding_{exname}_{asset}.parquet"; rows = []
    if f.exists():
        rows = pd.read_parquet(f)[["ts_ms", "rate"]].values.tolist()
    else:
        if exname == "binance": rows = fetch_funding_binance_vision(asset)
        else:
            try: ex = _ccxt(exname)
            except Exception as e: log(f"[{exname}] load_markets failed: {str(e)[:90]}"); return None
            sym = {"deribit": f"{asset}/USD:{asset}"}.get(exname, f"{asset}/USDT:USDT")
            if sym not in ex.markets: log(f"[{exname}] no market {sym}"); return None
            rows = _paginate_deribit(ex, sym) if exname == "deribit" else _paginate_forward(ex, sym, PAGE_LIMIT.get(exname, 100))
        if rows: pd.DataFrame(rows, columns=["ts_ms", "rate"]).to_parquet(f)
    mf = MANUAL / f"{exname}_{asset}.csv"
    if mf.exists(): rows += pd.read_csv(mf)[["ts_ms", "rate"]].astype(float).values.tolist()
    df = pd.DataFrame(rows, columns=["ts_ms", "rate"]).drop_duplicates("ts_ms").sort_values("ts_ms")
    df = df[df.ts_ms >= ms(START)]
    if len(df) < 50: return None
    df["ts"] = pd.to_datetime(df.ts_ms.astype(np.int64), unit="ms", utc=True)
    dt_h = df.ts.diff().dt.total_seconds().median() / 3600
    dt_h = float(np.clip(round(dt_h) if dt_h > 0.5 else 8, 1, 8))
    if exname == "deribit": dt_h = 8.0                     # Deribit prints the 8h rate hourly
    df["interval_h"] = dt_h; df["rate_ann"] = df.rate * (24 / dt_h) * 365
    df["exchange"], df["asset"] = exname, asset
    return df[["exchange", "asset", "ts", "rate", "interval_h", "rate_ann"]]

def fetch_ohlcv(asset):
    f = RAW / f"ohlcv_1h_{asset}.parquet"
    if f.exists(): return pd.read_parquet(f)
    rows = []
    for m in _months():
        df = _vision_csv(f"{VISION}/spot/monthly/klines/{asset}USDT/1h/{asset}USDT-1h-{m}.zip")
        if df is None or len(df) == 0: continue
        v = df.values[:, :6]
        rows += [(int(t), float(o), float(h), float(l), float(c), float(vol)) for t, o, h, l, c, vol in zip(_fix_ts(v[:, 0]), v[:, 1], v[:, 2], v[:, 3], v[:, 4], v[:, 5])]
    if not rows: log(f"no spot data for {asset}"); return None
    df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"]).drop_duplicates("ts_ms").sort_values("ts_ms")
    df["ts"] = pd.to_datetime(df.ts_ms.astype(np.int64), unit="ms", utc=True); df["asset"] = asset
    log(f"ohlcv {asset}: {df.ts.min().date()} -> {df.ts.max().date()} n={len(df)}"); df.to_parquet(f); return df

def fetch_fng():
    f = RAW / "fng.parquet"
    if f.exists(): return pd.read_parquet(f)
    import requests
    r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=60).json()["data"]
    df = pd.DataFrame(r); df["date"] = pd.to_datetime(df.timestamp.astype(int), unit="s", utc=True).dt.floor("D")
    df["fng"] = df.value.astype(float); df = df[["date", "fng"]].sort_values("date"); df.to_parquet(f); return df

def stage_fetch():
    frames, cov = [], []
    for ex in EXCHANGES:
        for a in ASSETS:
            d = fetch_funding(ex, a)
            if d is None: continue
            frames.append(d)
            cov.append({"exchange": ex, "asset": a, "first": d.ts.min(), "last": d.ts.max(), "n": len(d), "interval_h": d.interval_h.iloc[0], "mean_abs_rate_ann": float(d.rate_ann.abs().mean())})
            log(f"funding {ex:13s} {a:5s} {d.ts.min().date()} -> {d.ts.max().date()} n={len(d)} dt={d.interval_h.iloc[0]}h")
    pd.concat(frames).to_parquet(RAW / "funding_all.parquet"); pd.DataFrame(cov).to_csv(TAB / "T0_data_coverage.csv", index=False)
    px = pd.concat([x for x in (fetch_ohlcv(a) for a in ASSETS) if x is not None]); px.to_parquet(RAW / "ohlcv_all.parquet")
    fetch_fng(); log("fetch done")

# =============================================================================
# STAGE 2: BUILD 8h PANEL
# =============================================================================
def stage_build():
    fund = pd.read_parquet(RAW / "funding_all.parquet"); px = pd.read_parquet(RAW / "ohlcv_all.parquet"); fng = fetch_fng()
    fs = fng.set_index("date").fng
    out, venue_rows, persist_rows = [], [], []
    for a in ASSETS:
        fa = fund[fund.asset == a]
        if fa.empty: continue
        piv = fa.pivot_table(index=fa.ts.dt.floor("8h"), columns="exchange", values="rate_ann", aggfunc="mean")
        piv.index = piv.index + pd.Timedelta(hours=8)                 # information set = END of settlement window
        piv = piv[piv.notna().sum(1) >= MIN_EX]
        if piv.empty: continue
        lvl = piv.mean(1); sd = piv.std(1, ddof=0); rng = piv.max(1) - piv.min(1)
        nn = piv.notna().sum(1); sgn = 1 - np.maximum((piv > 0).sum(1), (piv < 0).sum(1)) / nn
        f = pd.DataFrame({"f_level": lvl, "f_binance": piv["binance"] if "binance" in piv else lvl, "n_ex": nn,
                          "dsd": np.log1p(sd), "drng": np.log1p(rng), "dscl": sd / (lvl.abs() + 0.01), "dsgn": sgn})
        f["dsd_max3"] = f.dsd.rolling(3).max(); f["drng_max3"] = f.drng.rolling(3).max()
        f["f_mom"] = f.f_level.rolling(21).mean() - f.f_level.rolling(90).mean()
        for c in DISP: f[c + "_l21"] = f[c].shift(STALE_LAG)         # stale placebo
        for v in piv.columns:                                       # leave-one-venue-out dispersion + venue contribution
            sub = piv.drop(columns=v); ok = sub.notna().sum(1) >= 2
            f[f"dsd_ex_{v}"] = np.log1p(sub.std(1, ddof=0).where(ok)); f[f"dsd_ex_{v}_max3"] = f[f"dsd_ex_{v}"].rolling(3).max()
            dev = (piv[v] - lvl).abs()
            venue_rows.append({"asset": a, "venue": v, "mean_abs_dev_from_mean": dev.mean(), "mean_rate_ann": piv[v].mean(), "corr_with_mean": piv[v].corr(lvl), "n_obs": int(piv[v].notna().sum())})
        rho = f.dsd.autocorr(1); persist_rows.append({"asset": a, "ar1_dsd": rho, "half_life_settlements": -np.log(2) / np.log(rho) if 0 < rho < 1 else np.nan})
        # prices: index = bar CLOSE time, so values at E are known at E
        p = px[px.asset == a].set_index("ts").sort_index()
        h = pd.DataFrame({"close": p.close, "low": p.low}); h.index = h.index + pd.Timedelta(hours=1)
        lc = np.log(h.close); sq = lc.diff() ** 2
        h["rv8"], h["rv24"], h["rv168"] = sq.rolling(8).sum(), sq.rolling(24).sum(), sq.rolling(168).sum()
        h["r8"], h["r24"], h["r168"] = lc.diff(8), lc.diff(24), lc.diff(168)
        for H, thr, name in DD_THRESHOLDS:
            fm = fwd(h.low, H, "min"); h[name] = ((fm / h.close - 1) <= -thr).astype(float).where(fm.notna())
        h["y_rv24"] = fwd(sq, 24, "sum"); h["r_next8"] = lc.shift(-8) - lc
        df = f.join(h.reindex(f.index), how="inner")
        df["fng"] = fs.reindex(df.index.floor("D")).values; df["fng"] = df.fng.ffill()
        df["lrv8"] = 0.5 * np.log((df.rv8 * PPY).clip(lower=1e-8)); df["lrv24"] = 0.5 * np.log((df.rv24 * 365).clip(lower=1e-8))
        df["lrv168"] = 0.5 * np.log((df.rv168 * 365 / 7).clip(lower=1e-8)); df["y_lrv24"] = 0.5 * np.log((df.y_rv24 * 365).clip(lower=1e-8))
        df["asset"] = a; df.index.name = "date"; out.append(df.reset_index())
    panel = pd.concat(out); panel["date"] = panel.date.dt.tz_convert(None)
    panel["tod"] = panel.date.dt.hour; panel["dow"] = panel.date.dt.dayofweek; panel["year"] = panel.date.dt.year
    panel["episode"] = ""
    for k, (s, e) in EPISODES.items():
        panel.loc[(panel.date >= pd.Timestamp(s)) & (panel.date < pd.Timestamp(e) + pd.Timedelta(days=1)), "episode"] = k
    panel["stress"] = (panel.episode != "").astype(int)
    panel.to_parquet(ROOT / "data" / "panel.parquet")
    # ---- descriptive tables
    save_table(panel.groupby("asset")[["f_level", "dsd", "drng", "dscl", "dsgn", "n_ex", "lrv24"]].agg(["mean", "std"]).round(4), "T1_descriptives")
    save_table(panel.groupby("asset")[[n for _, _, n in DD_THRESHOLDS]].mean().round(4), "T1b_event_rates")
    save_table(pd.DataFrame(venue_rows).set_index(["asset", "venue"]).round(4), "T1c_venue_contribution")
    save_table(pd.DataFrame(persist_rows).set_index("asset").round(3), "T1d_dispersion_persistence")
    # model-free decile sort: event rate of MAIN_CLS by within-asset dispersion decile
    q = panel.dropna(subset=["dsd", MAIN_CLS]).copy(); q["dec"] = q.groupby("asset").dsd.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    dec = q.groupby("dec")[MAIN_CLS].agg(["mean", "size"]).rename(columns={"mean": "event_rate", "size": "n"})
    p1, p0, n1, n0 = dec.event_rate.iloc[-1], dec.event_rate.iloc[0], dec.n.iloc[-1], dec.n.iloc[0]
    pp = (p1 * n1 + p0 * n0) / (n1 + n0); z = (p1 - p0) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    dec.loc["top_minus_bottom"] = [p1 - p0, np.nan]; dec.loc["z_two_prop"] = [z, np.nan]
    save_table(dec.round(4), "T1e_decile_sort_model_free")
    log("panel", panel.shape, "dates", panel.date.min(), panel.date.max())

# =============================================================================
# STAGE 3: MODELS
# =============================================================================
def design(panel):
    X = pd.concat([panel, pd.get_dummies(panel.tod, prefix="tod", drop_first=True).astype(float),
                   pd.get_dummies(panel.dow, prefix="dow", drop_first=True).astype(float),
                   pd.get_dummies(panel.asset, prefix="a", drop_first=True).astype(float)], axis=1)
    dums = [c for c in X.columns if c.startswith(("tod_", "dow_", "a_"))]
    return X, BASE_NUM + dums, BASE_NUM + DISP + dums, BASE_NUM + STALE + dums

def quarters(panel):
    for q in pd.period_range(TEST_START, panel.date.max(), freq="Q"):
        s, e = q.start_time, q.end_time.normalize() + pd.Timedelta(days=1)
        if (panel.date >= s).sum() > 0: yield s, e

def stage_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import lightgbm as lgb, shap
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    X, BASE, FULL, STALEF = design(panel)
    loo_venues = [c[len("dsd_ex_"):] for c in X.columns if c.startswith("dsd_ex_") and not c.endswith("_max3")]
    preds, imps, shaprows = [], [], []
    lgb_cls = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=7)
    def rec(te, tname, mname, p, qi):
        preds.append(pd.DataFrame({"date": te.date.values, "asset": te.asset.values, "target": tname, "model": mname, "p": p, "y": te[tname].values, "refit": qi}))
    for qi, (qs, qe) in enumerate(quarters(X)):
        log(f"refit {qi} test {qs.date()}..{qe.date()}")
        for H, thr, tname in DD_THRESHOLDS:
            need = FULL + STALEF + [tname]
            tr = X[X.date < qs].dropna(subset=need); te = X[(X.date >= qs) & (X.date < qe)].dropna(subset=need)
            if len(te) == 0 or tr[tname].nunique() < 2: continue
            for fsn, cols in (("BASE", BASE), ("FULL", FULL), ("STALE", STALEF)):
                lo = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(tr[cols], tr[tname])
                gb = lgb.LGBMClassifier(**lgb_cls).fit(tr[cols], tr[tname])
                rec(te, tname, f"logit_{fsn}", lo.predict_proba(te[cols])[:, 1], qi); rec(te, tname, f"lgbm_{fsn}", gb.predict_proba(te[cols])[:, 1], qi)
                if fsn == "FULL":
                    sv = shap.TreeExplainer(gb).shap_values(te[cols]); sv = sv[1] if isinstance(sv, list) else sv
                    imp = pd.Series(np.abs(sv).mean(0), index=cols)
                    imps.append(pd.DataFrame({"target": tname, "refit": qi, "feature": DISP, "imp": imp[DISP].values, "imp_total": imp.sum()}))
                    if tname == MAIN_CLS:
                        sh = pd.DataFrame(np.abs(sv)[:, [cols.index(c) for c in DISP]], columns=DISP); sh["date"], sh["asset"] = te.date.values, te.asset.values; shaprows.append(sh)
            if tname == MAIN_CLS:                                   # leave-one-venue-out robustness
                for v in loo_venues:
                    cols = BASE + [f"dsd_ex_{v}", f"dsd_ex_{v}_max3"]
                    trv, tev = tr.dropna(subset=cols), te.dropna(subset=cols)
                    if len(tev) == 0 or trv[tname].nunique() < 2: continue
                    gb = lgb.LGBMClassifier(**lgb_cls).fit(trv[cols], trv[tname]); rec(tev, tname, f"lgbm_LOO_{v}", gb.predict_proba(tev[cols])[:, 1], qi)
        # ---- 24h-ahead realised volatility
        need = FULL + ["y_lrv24"]
        tr = X[X.date < qs].dropna(subset=need); te = X[(X.date >= qs) & (X.date < qe)].dropna(subset=need)
        if len(te) == 0: continue
        for fsn, cols in (("BASE", BASE), ("FULL", FULL)):
            gb = lgb.LGBMRegressor(**lgb_cls).fit(tr[cols], tr.y_lrv24); rec(te, "y_lrv24", f"lgbmreg_{fsn}", gb.predict(te[cols]), qi)
        har_c = ["lrv8", "lrv24", "lrv168"]; hard_c = har_c + ["dsd", "dscl", "dsgn"]
        for a in X.asset.unique():
            tra, tea = tr[tr.asset == a], te[te.asset == a]
            if len(tra) < 300 or len(tea) == 0: continue
            for mname, cols in (("HAR", har_c), ("HAR_disp", hard_c)):
                A = np.c_[np.ones(len(tra)), tra[cols].values]; beta = np.linalg.lstsq(A, tra.y_lrv24.values, rcond=None)[0]
                rec(tea, "y_lrv24", mname, np.c_[np.ones(len(tea)), tea[cols].values] @ beta, qi)
            try:
                from arch import arch_model
                r = X[(X.asset == a) & (X.date < qe)].set_index("date").r8.dropna() * 100
                last_tr = r.index[r.index < qs][-1]
                res = arch_model(r, vol="GARCH", p=1, q=1, dist="t", mean="Constant").fit(last_obs=last_tr, disp="off")
                fc = res.forecast(horizon=3, start=last_tr, reindex=False).variance
                v24 = fc[["h.1", "h.2", "h.3"]].sum(1); v24 = v24[(v24.index >= qs) & (v24.index < qe)]
                lv = 0.5 * np.log((v24 / 1e4 * 365).clip(lower=1e-8))
                m = tea.set_index("date").join(lv.rename("p"), how="inner")
                preds.append(pd.DataFrame({"date": m.index.values, "asset": a, "target": "y_lrv24", "model": "GARCH", "p": m.p.values, "y": m.y_lrv24.values, "refit": qi}))
            except Exception as e: log(f"GARCH {a} skipped: {str(e)[:80]}")
    P = pd.concat(preds); P.to_parquet(OUT / "predictions.parquet")
    pd.concat(imps).to_parquet(OUT / "shap_importance_by_refit.parquet"); pd.concat(shaprows).to_parquet(OUT / "shap_rows_main.parquet")
    log("models done", P.shape)

# =============================================================================
# STAGE 4: EVALUATION
# =============================================================================
def dm_test(l1, l2):
    d = (l1 - l2).dropna().values; T = len(d)
    if T < 30: return np.nan, np.nan
    dbar = d.mean(); L = int(np.floor(4 * (T / 100) ** (2 / 9))); s = np.var(d)
    for k in range(1, L + 1): s += 2 * (1 - k / (L + 1)) * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    from scipy.stats import norm
    stat = dbar / np.sqrt(max(s, 1e-12) / T); return stat, 2 * (1 - norm.cdf(abs(stat)))

def mcs(losses, alpha=0.10, B=1000, block=10, seed=7):
    rng = np.random.default_rng(seed); L = losses.dropna(); T = len(L); included = list(L.columns); pv = {}; last_p = 0.0
    nb = int(np.ceil(T / block))
    boot_idx = np.array([np.concatenate([np.arange(s, s + block) for s in rng.integers(0, T - block + 1, nb)])[:T] for _ in range(B)])
    while len(included) > 1:
        M = L[included].values; d = M - M.mean(1, keepdims=True); dbar = d.mean(0); bm = d[boot_idx].mean(1)
        v = ((bm - dbar) ** 2).mean(0) + 1e-12; t = dbar / np.sqrt(v); Tmax = t.max(); Tb = ((bm - dbar) / np.sqrt(v)).max(1)
        p = max((Tb >= Tmax).mean(), last_p); worst = included[int(np.argmax(t))]; pv[worst] = p; last_p = p
        if p < alpha: included.remove(worst)
        else: break
    for m in included: pv[m] = max(pv.get(m, last_p), last_p)
    return pd.Series(pv).sort_values(ascending=False), included

def cls_metrics(df):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    if df.y.nunique() < 2 or len(df) < 100: return pd.Series({"n": len(df), "AUC": np.nan, "AP": np.nan, "Brier": np.nan, "base_rate": df.y.mean()})
    return pd.Series({"n": len(df), "AUC": roc_auc_score(df.y, df.p), "AP": average_precision_score(df.y, df.p), "Brier": brier_score_loss(df.y, df.p), "base_rate": df.y.mean()})

def paired_boot_auc(dfA, dfB, B=1000, seed=7):
    """Paired cluster bootstrap of AUC(A)-AUC(B). Clusters = 8h timestamps."""
    from sklearn.metrics import roc_auc_score
    m = dfA.merge(dfB, on=["date", "asset"], suffixes=("_a", "_b"))
    if len(m) < 100 or m.y_a.nunique() < 2: return np.nan, np.nan, np.nan, np.nan, len(m)
    m = m.sort_values("date").reset_index(drop=True)
    ya, pa, pb = m.y_a.to_numpy(), m.p_a.to_numpy(), m.p_b.to_numpy()
    # cluster index boundaries
    codes, _ = pd.factorize(m.date, sort=False)
    K = codes.max() + 1
    starts = np.searchsorted(codes, np.arange(K))
    ends = np.r_[starts[1:], len(codes)]
    rng = np.random.default_rng(seed); diffs = np.empty(B); kept = 0
    for i in range(B):
        pick = rng.integers(0, K, K)
        # list of cluster slices, concatenated once
        parts = [np.arange(starts[k], ends[k]) for k in pick]
        idx = np.concatenate(parts)
        y, pA, pB = ya[idx], pa[idx], pb[idx]
        if np.unique(y).size < 2: continue
        diffs[kept] = roc_auc_score(y, pA) - roc_auc_score(y, pB); kept += 1
    diffs = diffs[:kept]
    point = float(roc_auc_score(ya, pa) - roc_auc_score(ya, pb))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean()), len(m)

def stage_evaluate():
    from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
    from statsmodels.stats.multitest import multipletests
    P = pd.read_parquet(OUT / "predictions.parquet"); panel = pd.read_parquet(ROOT / "data" / "panel.parquet")
    P = P.merge(panel[["date", "asset", "stress", "episode", "year"]], on=["date", "asset"], how="left")
    P["group"] = np.where(P.asset.isin(CORE), "core", "alts")
    C = P[P.target != "y_lrv24"]; R = P[P.target == "y_lrv24"]
    # T2 metrics
    rows = []
    for (t, m), g in C.groupby(["target", "model"]):
        subs = [("all", g), ("core", g[g.group == "core"]), ("alts", g[g.group == "alts"])] + [(f"y{y}", g[g.year == y]) for y in sorted(g.year.unique())]
        for sub, gg in subs:
            r = cls_metrics(gg); r["target"], r["model"], r["subset"] = t, m, sub; rows.append(r)
    save_table(pd.DataFrame(rows).set_index(["target", "model", "subset"]).round(4), "T2_classification_metrics")
    # T3 incremental AUC: FULL-BASE and STALE-BASE, Holm across all tests
    rows = []
    for t in C.target.unique():
        for base in ("logit", "lgbm"):
            for alt in ("FULL", "STALE"):
                a, b = C[(C.target == t) & (C.model == f"{base}_{alt}")], C[(C.target == t) & (C.model == f"{base}_BASE")]
                d, lo, hi, p, n = paired_boot_auc(a, b)
                rows.append({"target": t, "model": base, "comparison": f"{alt}_minus_BASE", "dAUC": d, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p, "n": n})
    T3 = pd.DataFrame(rows)
    for cmp_ in ("FULL_minus_BASE", "STALE_minus_BASE"):
        msk = (T3.comparison == cmp_) & T3.p_one_sided.notna()
        if msk.sum(): T3.loc[msk, "p_holm"] = multipletests(T3.loc[msk, "p_one_sided"].clip(1e-4, 1), method="holm")[1]
    save_table(T3.set_index(["target", "model", "comparison"]).round(4), "T3_auc_gain_from_dispersion")
    # T3b by asset (main target, lgbm)
    rows = []
    for a_ in sorted(C.asset.unique()):
        g = C[(C.target == MAIN_CLS) & (C.asset == a_)]
        d, lo, hi, p, n = paired_boot_auc(g[g.model == "lgbm_FULL"], g[g.model == "lgbm_BASE"], B=200)
        rows.append({"asset": a_, "dAUC": d, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p, "n": n})
    save_table(pd.DataFrame(rows).set_index("asset").round(4), "T3b_auc_gain_by_asset")
    # T3c leave-one-venue-out
    rows = []
    for m in sorted(x for x in C.model.unique() if x.startswith("lgbm_LOO_")):
        g = C[C.target == MAIN_CLS]; d, lo, hi, p, n = paired_boot_auc(g[g.model == m], g[g.model == "lgbm_BASE"], B=200)
        rows.append({"excluded_venue": m.replace("lgbm_LOO_", ""), "dAUC_vs_BASE": d, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p, "n": n})
    save_table(pd.DataFrame(rows).set_index("excluded_venue").round(4), "T3c_leave_one_venue_out")
    # T4 volatility
    R = R.dropna(subset=["p", "y"]).copy(); R["mse"] = (R.p - R.y) ** 2
    ratio = np.exp(2 * R.y) / np.exp(2 * R.p); R["qlike"] = ratio - np.log(ratio) - 1
    T4 = R.groupby("model")[["mse", "qlike"]].mean(); T4["n"] = R.groupby("model").size()
    daily = {l: R.pivot_table(index="date", columns="model", values=l, aggfunc="mean") for l in ("mse", "qlike")}
    for l in ("mse", "qlike"):
        for m in T4.index: T4.loc[m, f"DM_vs_HAR_{l}"], T4.loc[m, f"DMp_vs_HAR_{l}"] = dm_test(daily[l][m], daily[l]["HAR"]) if "HAR" in daily[l] else (np.nan, np.nan)
        pv, inc = mcs(daily[l]); T4[f"MCS_p_{l}"] = pv.reindex(T4.index); T4[f"in_MCS90_{l}"] = T4.index.isin(inc)
    save_table(T4.round(4), "T4_volatility_forecasts"); save_table(R.groupby(["model", "stress"])[["mse", "qlike"]].mean().round(4), "T4b_volatility_by_regime")
    # T5 calibration
    g = C[(C.target == MAIN_CLS) & (C.model == "lgbm_FULL")].copy(); g["bin"] = pd.qcut(g.p.rank(method="first"), 10, labels=False)
    save_table(g.groupby("bin").agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("y", "size")).round(4), "T5_calibration_main")
    # T6 stability of constructions
    I = pd.read_parquet(OUT / "shap_importance_by_refit.parquet"); I = I[I.target == MAIN_CLS].copy()
    I["family"] = I.feature.str.extract(r"^d(sd|rng|scl|sgn)")[0]; I["share"] = I.imp / I.imp_total
    F = I.groupby(["refit", "family"]).share.sum().unstack("family"); fams = list(F.columns); rows = []
    try: fried = friedmanchisquare(*[F[c].values for c in fams])
    except Exception: fried = (np.nan, np.nan)
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            try: w = wilcoxon(F[fams[i]], F[fams[j]]); rows.append({"A": fams[i], "B": fams[j], "meanA": F[fams[i]].mean(), "meanB": F[fams[j]].mean(), "W": w.statistic, "p_raw": w.pvalue})
            except Exception: pass
    T6 = pd.DataFrame(rows)
    if len(T6): T6["p_holm"] = multipletests(T6.p_raw, method="holm")[1]
    save_table(T6.round(4), "T6_dispersion_stability_pairwise")
    ranks = F.rank(axis=1, ascending=False); rho = [spearmanr(ranks.iloc[k], ranks.iloc[k + 1])[0] for k in range(len(ranks) - 1)]
    T6b = pd.DataFrame({"mean_share": F.mean(), "sd_share": F.std(), "mean_rank": ranks.mean(), "rank_sd": ranks.std()}).round(4)
    T6b.loc["total_dispersion_share"] = [F.sum(1).mean(), F.sum(1).std(), np.nan, np.nan]
    T6b.loc["Friedman_chi2_p"] = [fried[0], fried[1], np.nan, np.nan]; T6b.loc["mean_consecutive_rank_rho"] = [np.nanmean(rho), np.nanstd(rho), np.nan, np.nan]
    save_table(T6b, "T6b_dispersion_stability_summary"); F.to_csv(TAB / "T6c_importance_share_by_refit.csv")
    # T7 regime importance
    S = pd.read_parquet(OUT / "shap_rows_main.parquet").merge(panel[["date", "asset", "stress", "episode"]], on=["date", "asset"])
    T7 = S.groupby("stress")[DISP].mean().T; T7.columns = ["calm", "stress"]; T7["ratio_stress_calm"] = T7.stress / T7.calm
    save_table(T7.round(5), "T7_regime_importance"); S.groupby("episode")[DISP].mean().round(5).to_csv(TAB / "T7b_importance_by_episode.csv")
    # T9 lead time around OOS episode starts (8h offsets), FULL vs BASE, plus percentile of the signal one period before
    rows, prow = [], []
    for mdl in ("lgbm_FULL", "lgbm_BASE"):
        g = C[(C.target == MAIN_CLS) & (C.model == mdl)]; allp = g.p.values
        for k, (s, e) in EPISODES.items():
            s = pd.Timestamp(s)
            if s < pd.Timestamp(TEST_START): continue
            for off in range(-30, 16):
                v = g[g.date == s + pd.Timedelta(hours=8 * off)].p.mean(); rows.append({"model": mdl, "episode": k, "offset_8h": off, "p": v})
            pre = g[(g.date >= s - pd.Timedelta(hours=24)) & (g.date < s)].p.mean()
            prow.append({"model": mdl, "episode": k, "mean_p_prior_24h": pre, "percentile_vs_all_oos": float((allp <= pre).mean()) if pd.notna(pre) else np.nan})
    pd.DataFrame(rows).to_csv(TAB / "T9_lead_time_long.csv", index=False)
    save_table(pd.DataFrame(prow).set_index(["episode", "model"]).round(4), "T9b_signal_percentile_before_episodes")
    log("evaluate done")

# =============================================================================
# STAGE 5: ECONOMICS (next-8h tail-hedge overlay; vol targeting)
# =============================================================================
def perf(ret, pos=None):
    ret = pd.Series(ret).dropna(); eq = (1 + ret).cumprod()
    out = {"ann_ret": ret.mean() * PPY, "ann_vol": ret.std() * np.sqrt(PPY), "sharpe": ret.mean() / (ret.std() + 1e-12) * np.sqrt(PPY),
           "max_dd": (eq / eq.cummax() - 1).min(), "worst_period": ret.min(), "cvar5": ret[ret <= ret.quantile(0.05)].mean(), "n_periods": len(ret)}
    if pos is not None: out["pct_time_on"] = float(np.mean(np.abs(pos) > 0)); out["switches"] = int((np.abs(np.diff(pos)) > 1e-9).sum())
    return pd.Series(out)

def stage_economics():
    P = pd.read_parquet(OUT / "predictions.parquet"); panel = pd.read_parquet(ROOT / "data" / "panel.parquet"); fee = FEE_BP / 1e4
    rows, curves = [], {}
    sigF = P[(P.target == MAIN_CLS) & (P.model == "lgbm_FULL")][["date", "asset", "p"]].rename(columns={"p": "p_full"})
    sigB = P[(P.target == MAIN_CLS) & (P.model == "lgbm_BASE")][["date", "asset", "p"]].rename(columns={"p": "p_base"})
    vol = P[P.target == "y_lrv24"].pivot_table(index=["date", "asset"], columns="model", values="p").reset_index()
    for a in ASSETS:
        d = panel[panel.asset == a].merge(sigF, on=["date", "asset"], how="left").merge(sigB, on=["date", "asset"], how="left").merge(vol, on=["date", "asset"], how="left")
        d = d[d.date >= pd.Timestamp(TEST_START)].sort_values("date").reset_index(drop=True).dropna(subset=["r_next8"])
        if len(d) < 200: continue
        rn = np.expm1(d.r_next8.values)
        strategies = {"long_always": np.ones(len(d))}
        for nm, col in (("hedge_FULL", "p_full"), ("hedge_BASE", "p_base")):
            thr = d[col].expanding(min_periods=90).quantile(0.90).shift(1)
            strategies[nm] = np.where(d[col].shift(1) >= thr, 0.0, 1.0)            # flat for the next 8h after a top-decile signal
        for name, pos in strategies.items():
            pos = np.asarray(pos, float); cost = np.abs(np.diff(np.r_[1.0, pos])) * fee; ret = pos * rn - cost
            r = perf(ret, pos); r["asset"], r["strategy"] = a, name; rows.append(r)
            if a in CORE: curves[(a, name)] = (d.date, (1 + pd.Series(ret)).cumprod().values)
        if a in CORE:
            for m in ("HAR", "HAR_disp", "GARCH", "lgbmreg_BASE", "lgbmreg_FULL"):
                if m not in d or d[m].isna().all(): continue
                w = np.minimum(2.0, 0.40 / np.exp(d[m])).shift(1).fillna(0).values; cost = np.abs(np.diff(np.r_[0.0, w])) * fee; ret = w * rn - cost
                r = perf(ret, w); r["asset"], r["strategy"] = a, f"voltarget_{m}"; rows.append(r); curves[(a, f"vt_{m}")] = (d.date, (1 + pd.Series(ret)).cumprod().values)
    E = pd.DataFrame(rows).set_index(["asset", "strategy"]).round(4); save_table(E, "T8_economic_value")
    agg = E.reset_index(); agg = agg[agg.strategy.str.startswith(("long", "hedge"))].groupby("strategy")[["ann_ret", "ann_vol", "sharpe", "max_dd", "worst_period", "cvar5"]].mean().round(4)
    save_table(agg, "T8b_hedge_overlay_pooled_mean"); pd.to_pickle(curves, OUT / "curves.pkl"); log("economics done")

# =============================================================================
# STAGE 6: FIGURES
# =============================================================================
def stage_outputs():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    panel = pd.read_parquet(ROOT / "data" / "panel.parquet"); P = pd.read_parquet(OUT / "predictions.parquet")
    def shade(ax):
        for k, (s, e) in EPISODES.items(): ax.axvspan(pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1), color="red", alpha=0.15)
    b = panel[panel.asset == "BTC"].set_index("date").resample("D").mean(numeric_only=True)
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(b.index, b.f_level * 100); ax[0].set_ylabel("Mean funding (% ann.)"); shade(ax[0])
    ax[1].plot(b.index, np.expm1(b.dsd) * 100); ax[1].set_ylabel("Cross-venue SD (% ann.)"); shade(ax[1])
    ax[2].plot(b.index, np.exp(b.lrv24) * 100); ax[2].set_ylabel("Realised vol, 24h (% ann.)"); shade(ax[2])
    savefig(fig, "F1_btc_dispersion_timeseries")
    C = P[P.target == MAIN_CLS]; fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for m in ["lgbm_BASE", "lgbm_FULL", "lgbm_STALE", "logit_BASE", "logit_FULL"]:
        g = C[C.model == m].dropna(subset=["p"])
        if len(g) < 100 or g.y.nunique() < 2: continue
        fpr, tpr, _ = roc_curve(g.y, g.p); pr, rc, _ = precision_recall_curve(g.y, g.p); st = dict(lw=1.8 if m.endswith("FULL") else 1, ls="-" if m.endswith("FULL") else "--")
        ax[0].plot(fpr, tpr, label=m, **st); ax[1].plot(rc, pr, label=m, **st)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[0].legend(fontsize=7); savefig(fig, "F2_roc_pr_main")
    T5 = pd.read_csv(TAB / "T5_calibration_main.csv"); fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(T5.p_mean, T5.y_rate, "o-"); ax.plot([0, 0.6], [0, 0.6], "k:"); ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); savefig(fig, "F3_calibration")
    F = pd.read_csv(TAB / "T6c_importance_share_by_refit.csv", index_col=0); fig, ax = plt.subplots(figsize=(8, 4))
    for c in F.columns: ax.plot(F.index, F[c], marker="o", label=DISP_FAMILIES.get(c, c))
    ax.set_xlabel("Refit (quarter index)"); ax.set_ylabel("SHAP importance share"); ax.legend(); savefig(fig, "F4_stability")
    T7 = pd.read_csv(TAB / "T7_regime_importance.csv", index_col=0); fig, ax = plt.subplots(figsize=(8, 4)); T7[["calm", "stress"]].plot.bar(ax=ax); ax.set_ylabel("Mean |SHAP|"); savefig(fig, "F5_regime_importance")
    curves = pd.read_pickle(OUT / "curves.pkl")
    for a in CORE:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for (aa, n), (dt, eq) in curves.items():
            if aa != a: continue
            (ax[0] if n.startswith(("long", "hedge")) else ax[1]).plot(dt, eq, label=n, lw=1.2)
        ax[0].set_xlabel("Date"); ax[0].set_ylabel("Cumulative return"); ax[1].set_xlabel("Date"); ax[1].set_ylabel("Cumulative return")
        for x in ax: x.legend(fontsize=7); shade(x)
        savefig(fig, f"F6_strategies_{a}")
    T3 = pd.read_csv(TAB / "T3_auc_gain_from_dispersion.csv"); T3 = T3[T3.model == "lgbm"]; fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(T3.target.nunique()); tg = list(T3.target.unique())
    for j, cmp_ in enumerate(("FULL_minus_BASE", "STALE_minus_BASE")):
        g = T3[T3.comparison == cmp_].set_index("target").reindex(tg)
        ax.bar(xs + (j - 0.5) * 0.35, g.dAUC, 0.35, yerr=[g.dAUC - g.ci_lo, g.ci_hi - g.dAUC], capsize=3, label=cmp_.replace("_", " "))
    ax.set_xticks(xs); ax.set_xticklabels(tg); ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("ΔAUC vs BASE (LightGBM)"); ax.legend(); savefig(fig, "F7_auc_gain")
    T9 = pd.read_csv(TAB / "T9_lead_time_long.csv")
    if len(T9):
        eps = list(T9.episode.unique()); fig, ax = plt.subplots(1, len(eps), figsize=(4.5 * len(eps), 4), squeeze=False)
        for i, k in enumerate(eps):
            for mdl, gg in T9[T9.episode == k].groupby("model"): ax[0, i].plot(gg.offset_8h, gg.p, marker=".", label=mdl)
            ax[0, i].axvline(0, color="k", ls=":"); ax[0, i].set_xlabel("8h periods from episode start"); ax[0, i].legend(fontsize=7)
        ax[0, 0].set_ylabel(f"P({MAIN_CLS})"); savefig(fig, "F8_lead_time")
    D = pd.read_csv(TAB / "T1e_decile_sort_model_free.csv", index_col=0); D = D[D.index.str.isnumeric()]; fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(D.index.astype(int), D.event_rate); ax.set_xlabel("Within-asset decile of cross-venue dispersion"); ax.set_ylabel(f"Realised rate of {MAIN_CLS}"); savefig(fig, "F9_decile_sort")
    (OUT / "MANIFEST.txt").write_text("\n".join(sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))); log("outputs done ->", OUT)
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); a = ap.parse_args()
    S = {"fetch": stage_fetch, "build": stage_build, "models": stage_models, "evaluate": stage_evaluate, "economics": stage_economics, "outputs": stage_outputs}
    for s in (["fetch", "build", "models", "evaluate", "economics", "outputs"] if a.stage == "all" else [a.stage]):
        if s == "fetch" and a.stage == "all" and (RAW / "funding_all.parquet").exists(): log("fetch cached, skipping"); continue
        log("=== stage", s); S[s]()