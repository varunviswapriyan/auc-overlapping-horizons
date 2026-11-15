# Replication package

**Overlapping forecast horizons overstate the significance of AUC comparisons in crypto perpetual futures**

Varun Viswapriyan, The Ohio State University.

This package regenerates every number, table and figure in the manuscript. All random numbers come from fixed seeds.

## Contents

```
replication/
  README.md                 this file
  requirements.txt          Python packages and the versions used
  code/
    auc_inference.py        the six inference procedures and the exact weighted bootstrap (numpy + scipy only)
    simulation.py           Monte Carlo study (Section 4)
    truth.py                population Delta-AUC for each simulation scenario
    application.py          re-analysis of the 20 comparisons (Section 5), equivalence check, timing
    make_outputs.py         builds manuscript tables (LaTeX) and figures from results/
    verify_reproduction.py  regenerates results and diffs them against the shipped files
  results/
    sim/                    scenarios.csv, truth.csv, reps_s00.csv ... reps_s23.csv (one row per replication x method)
    application/            A1 (all procedures, 20 comparisons), A2 (s.e. ratios), A3 (equivalence), A4 (timing)
  pipeline/
    funding_dispersion.py   original data and forecasting pipeline of the application
    fd_project/data/raw/    raw downloads (parquet), as retrieved
    fd_project/data/panel.parquet          assembled 8-hour panel, 47,313 rows
    fd_project/outputs/predictions.parquet out-of-sample predictions of all models (input to application.py)
    fd_project/outputs/tables, figures     the pipeline's own outputs
```

The manuscript itself is not part of this repository. `make_outputs.py` writes the manuscript's tables and figures to `../manuscript/tables` and `../manuscript/figures` (created if missing); `verify_reproduction.py` compares regenerated tables against that folder only if it exists.

## Software

Python 3.12 or later (the pinned numpy 2.5.3 requires Python >= 3.12). The results shipped here were produced with Python 3.13.18, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.1 and matplotlib 3.11.2 on Linux x86-64.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`auc_inference.py`, `simulation.py`, `truth.py` and `make_outputs.py` need only numpy, scipy, pandas and matplotlib. `application.py` and the pipeline also need scikit-learn and pyarrow (to read parquet). The pipeline's `models` stage additionally needs lightgbm and shap, the volatility models need arch, the evaluation stage needs statsmodels, and the `fetch` stage needs ccxt and requests.

NumPy's `Generator` streams are stable in practice but are not guaranteed to be identical across NumPy feature releases, so bit-for-bit reproduction of the simulation is guaranteed only with the pinned versions. Across platforms, floating-point summation order can produce differences of order 1e-16 in individual statistics.

## Reproducing the paper (run from `code/`)

| Step | Command | Time (2 cores) | Produces |
|---|---|---|---|
| 1. Application | `python application.py --pred ../pipeline/fd_project/outputs/predictions.parquet --out ../results/application` | ~3 min | Tables 3-4, Figure 2, the timing and equivalence numbers in Section 5 |
| 2. Simulation | `python simulation.py --reps 1000 --reps_alt 500 --B 499 --out ../results/sim` | ~1.5 h | `results/sim/reps_s*.csv` |
| 3. Population values | `python truth.py --out ../results/sim` | ~10 min | `results/sim/truth.csv` |
| 4. Tables and figures | `python make_outputs.py` | <1 min | `../manuscript/tables/*.tex`, `../manuscript/figures/*.pdf` |
| 5. Manuscript | in `../manuscript`: `pdflatex cssc_manuscript && bibtex cssc_manuscript && pdflatex cssc_manuscript && pdflatex cssc_manuscript` | | `cssc_manuscript.pdf` |

`simulation.py` skips any scenario whose result file already exists with the expected number of rows; delete `results/sim/reps_s*.csv` to recompute everything. Each replication `r` of scenario `s` uses `numpy.random.default_rng([20261113, s, r])`, so results do not depend on the number of worker processes.

## Verifying

```
python verify_reproduction.py              # application, scenarios 0 and 21, all LaTeX tables (~20 min)
python verify_reproduction.py --sim all    # every simulation scenario (~1.5 h)
python verify_reproduction.py --pipeline   # also re-runs the pipeline's build, evaluate and economics stages
```

The script prints PASS or FAIL for each check with the largest absolute difference. It checks that

* the weighted cluster bootstrap in `auc_inference.py` reproduces the pipeline's original loop implementation (`paired_boot_auc`) to machine precision, and reproduces the pipeline's table `T3_auc_gain_from_dispersion.csv` exactly at its reported precision;
* the regenerated application and simulation result files equal the shipped ones;
* the manuscript tables regenerated from the shipped results are identical to those in `../manuscript/tables`.

## The application pipeline

`pipeline/funding_dispersion.py` is the pipeline of the author's earlier analysis. It runs in stages:
`python funding_dispersion.py --stage {fetch|build|models|evaluate|economics|outputs|all}`.

* **Do not re-run `fetch`.** Exchange APIs change and histories are revised, so a new download would not reproduce the archived data. `--stage all` skips `fetch` automatically when `fd_project/data/raw/funding_all.parquet` exists. The raw folder also contains partial downloads from OKX, Gate, HTX, Bitget, MEXC and Kraken Futures that the pipeline does not use (only Binance, KuCoin Futures, Deribit and Phemex enter the panel).
* `build` reconstructs `panel.parquet` from the raw files.
* `models` refits all models and overwrites `predictions.parquet`. The logistic-regression predictions are deterministic. LightGBM was run with `random_state=7`, but LightGBM output can depend on its version and thread count, so a re-run may differ slightly from the shipped predictions. All inference in the manuscript takes the shipped `predictions.parquet` as its input data.
* `evaluate` and `economics` produce the pipeline's tables from the predictions and the panel.

What was verified independently while preparing this package: re-running `build` reproduced `panel.parquet` and tables T0-T1e exactly; refitting the logistic models reproduced their predictions to within 6e-10; and re-running `evaluate` and `economics` on the shipped predictions reproduced every pipeline table (largest difference 8e-25). The LightGBM, SHAP and GARCH steps of `models` could not be re-run in that environment and have not been independently re-executed.

Two cosmetic edits were made to the pipeline relative to the earlier version: the module docstring now states the feature window as `[E-8h, E)`, matching the code, and two unused lines in `paired_boot_auc` were removed. Neither changes any output.

## Data sources

Binance funding rates and hourly spot candles: https://data.binance.vision. KuCoin Futures, Deribit and Phemex funding histories: the exchanges' public REST APIs. Crypto Fear & Greed index: https://alternative.me/crypto/fear-and-greed-index/. The data are redistributed here only to allow replication.
