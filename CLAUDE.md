# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GGWP_US_LLTL** is a deep learning-based pairs trading system for US large-cap long/short pairs. It uses a ResNet + Transformer architecture to predict pair spread reversal thresholds, with Gaussian Copula loss for correlation modeling. The primary execution environment is Google Colab with GPU (RTX PRO 6000 Blackwell), with data stored on Google Drive.

## Running the Project

```bash
# Full training + backtesting pipeline
cd src/
python main.py

# Data cleaning (run after identifying ill-conditioned samples)
cd data_cleaning/
python clean_data.py
```

The project is primarily orchestrated via `GGWP_US_LLTL.ipynb` (Google Colab notebook), which mounts Google Drive, runs training in loops, and manages the full pipeline.

## Configuration

All hyperparameters and test case definitions live in `src/hyperparameters.py`. Key settings:

- **Active test case**: Selected via `test_case` variable (5 cases defined; current default is full backtest 2018–2023)
- **Data split**: Train Oct 2018–Oct 2020 → Val Oct 2020–Oct 2021 → Test Oct 2021–Oct 2023
- **Optimizer**: Adam, lr=1e-4, batch_size=256, num_epochs=1000, early_stop_count=50
- **Loss function**: `GaussCopGumLoss` (default); options: `GumIndLoss`, `NormIndLoss`, `GaussCopNormLoss`
- **Stop-loss mode**: Dynamic (std-dev based, sigma=0.999) or fixed offset

## Architecture

### Model (`src/model.py` — `GGPTS` class)
- **Feature extraction**: ResNet with 4 residual blocks (channels 3→128→256→512→1024), defined in `src/module.py`
- **Temporal encoding**: Transformer encoder (8 layers, 1024 hidden dim, 4 attention heads)
- **Output heads**:
  - Predictor1: Binary revert probability (LogSoftmax)
  - Predictors 2–4: Log-scale regression for Rtop/Top/Close thresholds
  - Predictor5 (optional): 6 Gaussian Copula parameters (Tanh activation)

### Loss Functions (`src/loss.py`)
Four custom losses; `GaussCopGumLoss` is default and uses Gaussian Copula to model threshold correlations. The copula path requires Predictor5 in the model.

### Data Pipeline
- Raw data → `data_cleaning/clean_data.py` → versioned `.pickle` files (`cleaned_data_v*.pickle`)
- Preprocessed 10+ GB pickle files in `data/`
- `src/preprocess.py` loads and splits data temporally
- `src/dataloader.py` provides PyTorch Dataset with augmentation (noise, masking, flipping, pair-swapping)

### Backtesting
- `src/backtest.py`: Simulates pair trading with model predictions; handles entry/exit decisions
- `src/trade.py`: Trading engine tracking P&L, stop-loss hits, and metrics
- `src/MyTool.py`: Trading utilities (Johansen cointegration test, tax calculations, logging)
- Results logged to `data/Record/` with timestamps

## Data Cleaning Loop

The cleaning pipeline is iterative:
1. Training identifies ill-conditioned samples → logged to `data_cleaning/ill_conditioned_data_*.csv`
2. `clean_data.py` reads logs and produces a new versioned clean dataset
3. Repeat until no new ill-conditioned samples are found

## Key File Relationships

```
hyperparameters.py  ──►  main.py  ──►  model.py
                                  ──►  dataloader.py  ──►  preprocess.py
                                  ──►  loss.py
                                  ──►  backtest.py  ──►  trade.py
                                                    ──►  MyTool.py
```

## Notes

- The project has no test suite or linter configuration.
- `src/GMM_cluster.py` is legacy code (superseded by the deep learning approach).
- `torch.autograd.set_detect_anomaly(True)` is used during debugging; disable for performance.
- Pickle files are large (10–30 GB); data paths assume Google Drive mounting at `/content/drive/MyDrive/GGWP_US_LLTL/`.
