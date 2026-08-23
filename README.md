# GGWP: Deep Learning Pairs Trading System

GGWP (Gaussian-copula/Gumbel distribution With Pairs trading) is a deep learning-based
intraday pairs trading system for US large-cap equities. It replaces the discrete-action
reinforcement learning approaches common in prior pairs-trading literature with a
probabilistic deep learning model: a ResNet–Transformer network predicts the full
distribution of a pair's spread-reversal behavior, and a closed-form expected-profit
calculation over a **continuous** opening threshold selects the trading action — no
action-space discretization and no policy-gradient training required.

## Method Summary

1. **Pair selection & spread construction** — For every trading day, all candidate stock
   pairs from the stock pool are tested for cointegration using the Johansen test. Pairs
   that pass are assigned a cointegrating vector β = (β₁, β₂) that defines a mean-reverting
   spread `P(t) = β₁·ln(s₁(t)) + β₂·ln(s₂(t))`, computed over a *formation period* at the
   start of the trading day.
2. **Threshold prediction** — A ResNet–Transformer model observes the formation-period
   spread and each stock's own price return, and predicts:
   - the probability that the spread will revert to its mean during the trading period,
   - the parameters of Gumbel/Normal distributions over three threshold quantities
     (`Rtop`, `Top`, `Close` — the spread's peak deviation before reverting, its maximum
     deviation, and its deviation at market close), and
   - a Gaussian Copula correlation structure linking the three threshold distributions.
3. **Continuous threshold optimization** — Given the predicted distributions, the expected
   trading profit is maximized analytically over a continuous opening threshold `To`,
   rather than choosing from a fixed/discretized set of actions.
4. **Intraday execution** — The resulting threshold (plus a stop-loss, either fixed-offset
   or dynamic/std-dev-based) drives an intraday backtest simulation: positions open when
   the live spread crosses `To`, and close on mean-reversion, stop-loss, or forced closure
   at the end of the trading period (no overnight positions).

## Repository Structure

```
GGWP_US_LLTL/
├── src/                          # Core training/backtesting pipeline
│   ├── hyperparameters.py        #   Single source of truth for all config (paths, test cases, model/trading params)
│   ├── main.py                   #   Entry point: loads data, trains/tests the model, runs backtest
│   ├── model.py                  #   GGPTS model class (ResNet + Transformer, train/eval loop)
│   ├── module.py                 #   ResNet feature-extraction building blocks
│   ├── loss.py                   #   Loss functions (GaussCopGumLoss and variants)
│   ├── preprocess.py             #   Builds model input/label pickles from formation tables + minute prices
│   ├── dataloader.py             #   PyTorch Dataset/augmentation for training
│   ├── backtest.py               #   Backtest engine (threshold evaluation, P&L simulation, reporting)
│   ├── trade.py                  #   Low-level trade simulation and formation-table/price lookups
│   ├── utils.py                  #   Statistical helpers (Gumbel/Normal/Copula math, expected-profit optimization)
│   ├── MyTool.py                 #   Trading utilities (Johansen helpers, tax calc, position sizing)
│   └── GMM_cluster.py            #   Legacy clustering experiment (superseded by the DL approach)
│
├── create_formationtable/        # Johansen cointegration / formation-table generation
│   ├── mt.py                     #   Johansen test, VAR/VECM estimation, BIC model selection
│   ├── create_formationtable.py  #   Single-process formation table generator
│   └── multiprocessing/
│       └── Main_Check_parallel.py  # Parallelized, resumable production formation-table generator
│
├── data_cleaning/                # Iterative training-data cleaning loop
│   └── clean_data.py             #   Removes samples flagged as ill-conditioned during training
│
├── data/                         # Not tracked in git (see below) — populated at runtime
│   ├── full_data_AB/             #   Raw minute-level price data, one CSV per trading day
│   ├── formation_table/          #   Johansen test output per trading day (w1/w2, intercept, std, ...)
│   ├── preprocess_data/          #   Model-ready (input, label) pickles built by src/preprocess.py
│   ├── Train_ckpt/                #   Model checkpoints, one timestamped folder per training run
│   ├── Record/                   #   Backtest results/plots, one timestamped folder per run
│   └── inference_cache/          #   Cached GPU inference output for CPU-only backtest re-runs
│
└── GGWP_US_LLTL.ipynb            # Colab notebook: mounts Google Drive and orchestrates the full pipeline
```

> **Note on `data/`:** raw price data, formation tables, and trained models/pickles are
> multi-GB to multi-tens-of-GB and are intentionally excluded from this repository. Only
> the directory structure is preserved (via `.gitkeep`) so the expected layout is visible;
> the code populates it when run against the real dataset.

## Pipeline / Workflow

```
  full_data_AB/                 create_formationtable/
  (raw minute prices)   ──────▶ Johansen cointegration test  ──────▶ formation_table/
                                per pair, per trading day             (w1, w2, μ, σ, ...)
                                                                              │
                                                                              ▼
                                                                     src/preprocess.py
                                                                     builds (Norm_Spread,
                                                                     S1/S2_Return) inputs +
                                                                     (Revert, Rtop, Top,
                                                                     Close, Tax) labels
                                                                              │
                                                                              ▼
                                                                     preprocess_data/*.pickle
                                                                              │
                                                        ┌─────────────────────┴─────────────────────┐
                                                        ▼                                             │
                                              data_cleaning/clean_data.py                             │
                                              (iterative ill-conditioned-                              │
                                               sample removal, versioned                               │
                                               cleaned_data_v{n}.pickle)                                │
                                                        │                                             │
                                                        ▼                                             ▼
                                                  src/main.py  ──────────────────────────▶  src/backtest.py
                                                  train the GGPTS model                     intraday P&L simulation
                                                  (src/model.py, src/loss.py)                (src/trade.py)
```

1. **Formation table generation** (`create_formationtable/`) — for every trading day and
   every candidate stock pair, run VAR order selection, portmanteau/normality residual
   diagnostics, and the Johansen cointegration test; pairs that pass are written to a
   per-day formation table CSV with their cointegrating weights (`w1`, `w2`), equilibrium
   mean/std (`Johansen_intercept`, `Johansen_std`), and VECM model type. The
   `multiprocessing/Main_Check_parallel.py` variant parallelizes this across a full
   multi-year date range and supports resuming after interruption.
2. **Preprocessing** (`src/preprocess.py`) — for every (pair, day) in the formation
   tables, load the corresponding minute prices, normalize the formation-period spread by
   the pair's mean/std, and compute the training labels from the subsequent trading
   period. Output is a single pickle of (input, label) pairs indexed by date/stock pair.
3. **Data cleaning** (`data_cleaning/clean_data.py`) — training periodically flags
   ill-conditioned samples (e.g. near-singular copula correlation matrices); this script
   removes them from the training pickle, producing a new versioned `cleaned_data_v{n}.pickle`
   for the next training run.
4. **Training** (`src/main.py`, `src/model.py`) — the `GGPTS` model (ResNet feature
   extractor + Transformer encoder + five prediction heads) is trained against one of four
   loss variants (`GaussCopGumLoss` by default) using a temporal train/validation/test
   split defined in `hyperparameters.py`.
5. **Backtesting** (`src/backtest.py`, `src/trade.py`) — the trained model's predicted
   thresholds are used to simulate intraday trading on the test period, with configurable
   dynamic/fixed stop-loss, an optional 1.5σ fixed-threshold baseline for comparison, and
   summary/records output written to `data/Record/`.

## Model Architecture (`GGPTS`, `src/model.py`)

- **Feature extraction:** ResNet with 4 residual blocks (channel widths 3→128→256→512→1024)
  over the 3-channel (spread, stock-1 return, stock-2 return) formation-period time series.
- **Temporal encoding:** an 8-layer Transformer encoder (1024-dim, 4 attention heads) over
  the ResNet's output sequence.
- **Prediction heads:**
  - Predictor 1 — binary revert probability (`LogSoftmax`)
  - Predictors 2–4 — location/scale parameters for the `Rtop`/`Top`/`Close` threshold
    distributions
  - Predictor 5 *(optional, copula models only)* — 6 Gaussian Copula correlation
    parameters (`Tanh`) linking the three threshold distributions

## Configuration (`src/hyperparameters.py`)

All hyperparameters — data paths, the active train/validation/test date split
(`test_case`), model/training parameters, loss function choice, and trading-behavior
switches (dynamic vs. fixed stop-loss, open-probability threshold, baseline mode, etc.) —
are centralized in `src/hyperparameters.py` and imported from there throughout the
codebase, so there is a single place to reconfigure a run.

## Running

The project is designed to run on Google Colab with GPU, with `data/` mounted from Google
Drive (see `GGWP_US_LLTL.ipynb` for the full orchestration). The individual stages can also
be run directly:

```bash
# Generate formation tables (parallelized, resumable)
cd create_formationtable/multiprocessing
python Main_Check_parallel.py

# Build the model-ready training pickle (preprocess.py has no CLI entry point;
# call it from a notebook cell or a short script)
cd src
python -c "import preprocess, hyperparameters as hp; preprocess.create_preprocess_data(
    start_year=hp.start_year, start_month=hp.start_month,
    end_year=hp.end_year, end_month=hp.end_month,
    save_path=hp.preprocess_data_path)"

# Clean flagged samples out of the training set
cd data_cleaning
python clean_data.py

# Train + backtest (mode is set in hyperparameters.py)
cd src
python main.py
```
