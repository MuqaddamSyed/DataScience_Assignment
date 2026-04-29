# Technical documentation
---

## 1. Overview

The system forecasts **weekly** beverage sales for **43 US states** over a configurable horizon (default **8 weeks**). It:

1. Loads and cleans an Excel source, resamples to a strict weekly index, and imputes gaps without future leakage.
2. Engineers causal features (lags, rolling stats, calendar, US holidays).
3. Trains five base models per state (ARIMA, SARIMA, XGBoost, LSTM, Prophet), builds a **top-two inverse–RMSE weighted ensemble**, and selects the best of six candidates by validation **RMSE**.
4. Persists artefacts under `models/` and exposes forecasts (with **95% intervals**) via **FastAPI** and optionally **Streamlit**.

---

## 2. Repository map

| Path | Role |
|------|------|
| `training_data/` | Raw Excel input |
| `config.yaml` | Single source of truth for paths, horizons, feature lists, model hyperparameters, API/CORS |
| `src/config_loader.py` | Loads YAML; resolves relative paths to **absolute paths under project root** |
| `src/preprocessing.py` | `load_raw`, `resample_state`, `build_state_series`, `run_preprocessing` |
| `src/feature_engineering.py` | Causal lags/rolling/calendar/holiday features; `train_val_split` |
| `src/train.py` | Per-state training; ensemble; model persistence |
| `src/evaluate.py` | RMSE, MAE, MAPE |
| `src/predict.py` | Load artefacts; multi-step forecast; confidence intervals |
| `src/cross_validation.py` | Expanding-window walk-forward splits; optional model comparison |
| `src/tuning.py` | Optuna tuning for XGBoost (representative states) |
| `run_training.py` | End-to-end training orchestration |
| `run_cv.py` | Walk-forward CV report |
| `run_tuning.py` | Writes tuned XGBoost params into `config.yaml` |
| `run_horizon_analysis.py` | Per-step error by horizon |
| `main.py` | FastAPI app; lifespan loads `AppState` |
| `api/` | Routes, Pydantic schemas, singleton dependencies |
| `dashboard/app.py` | Streamlit UI |
| `notebooks/01_eda.ipynb` | Exploratory analysis |
| `reports/` | Plots and CSV outputs from reporting scripts |
| `tests/` | Pytest suite |
| `.github/workflows/ci.yml` | CI: compile check, tests, smoke train |

---

## 3. Environment and setup

- **Python**: 3.9+ (README/CI may use 3.11 in GitHub Actions).
- **Virtualenv**: `make install` creates `venv/` and installs `requirements.txt`. Prophet needs **CmdStan** (`cmdstanpy.install_cmdstan()` during install).
- **macOS / XGBoost**: If `libxgboost` fails to load, install OpenMP (e.g. `brew install libomp`).
- **Reproducibility**: See `requirements.lock.txt` for a full frozen environment.

All paths in `config.yaml` under `paths` and `data.raw_path` are resolved relative to the repository root so training and the API behave the same from any working directory.

---

## 4. Data pipeline

### 4.1 Input

- Expected logical columns after normalisation: **date**, **state**, **sales** (see `preprocessing.py` for column-name mapping from the case-study Excel).

### 4.2 Weekly resampling

- Frequency: `config.yaml` → `data.resample_freq` (typically `"W"`).
- Aggregation: **sum** of sales per state per week.
- Missing weeks: forward-fill, then linear interpolation, then optional back-fill for leading NaNs; sales clamped at **≥ 0** where applicable.

### 4.3 Train / validation split

- Chronological: first **(1 − test_size)** of the series for training, last **test_size** (default 20%) for validation.
- **No random shuffle** — required for time series.

---

## 5. Feature engineering (no leakage)

For each row at time **t**, features use only information from **t−1 and earlier**:

- **Lags**: configurable list (e.g. 1, 2, 4, 8, 13, 26 weeks), aligned with weekly granularity (see [DECISIONS.md](DECISIONS.md) for day-lag mapping).
- **Rolling mean / std**: windows ending strictly in the past via `shift(1).rolling(...)`.
- **Calendar**: day of week, month, week of year, quarter, year.
- **Holiday**: US federal holidays overlapping the week (`holidays` library).

Regression tests in `tests/test_feature_engineering.py` assert causality for lags and rolling features.

---

## 6. Models and selection

### 6.1 Base models

| Model | Implementation notes |
|-------|----------------------|
| ARIMA | Grid over small `(p,d,q)` space; `statsmodels` |
| SARIMA | Seasonal period **52**; `SARIMAX` |
| XGBoost | All engineered features; params from `config.yaml` (tunable via `run_tuning.py`) |
| LSTM | `MinMaxScaler` **fit on train only**; sequence length from config; PyTorch |
| Prophet | Yearly + weekly seasonality; US holidays |

### 6.2 Ensemble

- Takes the **two** base models with **lowest validation RMSE** for that state.
- **Weights**: proportional to `1 / RMSE` (then normalised).
- Validation score for “ensemble” is computed by averaging those models’ validation predictions (see `train.py`).
- If ensemble **RMSE** is lowest among all six candidates, `best_model` is **`ensemble`** and `registry[state]["ensemble"]` stores `components` and `weights`.

### 6.3 Artefacts

Per state, files such as `{slug}_{model}.pkl|joblib` under `models/`. XGBoost may save a dict `{"model", "residual_std"}` for intervals. LSTM saves weights + scaler + `residual_std`, etc.

Global:

- `models/model_registry.joblib` — per-state best model, metrics, optional `ensemble` metadata.
- `models/state_series.joblib` — full weekly series per state (for API/dashboard without re-reading Excel).
- `models/training_report.json` — human-readable summary.

---

## 7. Forecasting and confidence intervals

- **Horizon**: `data.forecast_horizon` (default 8 weeks).
- **ARIMA / SARIMA**: `get_forecast` / `conf_int(alpha=0.05)`.
- **Prophet**: `yhat_lower`, `yhat_upper`.
- **XGBoost / LSTM**: residual-based bands: point ± **1.96 × σ_residual × √step** (conservative growth with horizon).
- **Ensemble**: weighted average of component **point** and **low** / **high** series.

Recursive **XGBoost** forecasting refits features step-by-step; **LSTM** unrolls one step at a time.

---

## 8. REST API

### 8.1 Startup

- `AppState` loads `model_registry.joblib` and `state_series.joblib` once in the FastAPI **lifespan** (`main.py`, `api/dependencies.py`).
- If files are missing, startup logs an error; endpoints may return errors until `run_training.py` has been run.

### 8.2 Endpoints (summary)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, counts of loaded states/models |
| GET | `/models` | All states: best model and validation metrics |
| GET | `/models?state={name}` | Single state (query validation applies) |
| GET | `/predict?state={name}` | Forecast for default horizon with `forecast_sales`, `forecast_low`, `forecast_high` |

OpenAPI: **`/docs`** (Swagger), **`/redoc`**.

### 8.3 Input validation

- `state`: length and regex constraints (see `api/routes.py`) to reduce injection and odd paths.

### 8.4 CORS

- Controlled by `config.yaml` → `api` (`cors_origins`, `cors_allow_credentials`). Using `allow_origins=["*"]` forces credentials off for browser safety.

---

## 9. Security notes (model loading)

`predict.load_best_model` enforces:

- Allow-listed **model names** only.
- Sanitised state slug; path must resolve **inside** `models/` (`relative_to` guard).

Treat `models/` as **trusted** — only the training pipeline should write there. Pickle/joblib deserialization is unsafe against hostile files.

---

## 10. Streamlit dashboard

- **Run**: `make dashboard` (sets a **project-local `HOME`** so `~/.streamlit` need not be writable — see [README.md](README.md)).
- **Behaviour**: Select state, optionally override model, adjust horizon; chart with history, validation, forecast, CI band; tables for forecast and metrics.

---

## 11. Makefile targets (reference)

| Target | Purpose |
|--------|---------|
| `install` | Create venv, install deps, CmdStan |
| `train` | Full training |
| `train-fast` | Three-state smoke train |
| `serve` | Uvicorn FastAPI |
| `test` | Pytest |
| `report` | Regenerate `reports/*.png` |
| `cv` | Walk-forward CV sample |
| `tune` | Optuna XGBoost tuning |
| `horizon` | Horizon error analysis |
| `dashboard` | Streamlit |
| `eda` | Execute EDA notebook |
| `docker` / `docker-run` | Container build/run |

---

## 12. Testing

- **44+ tests**: preprocessing, feature leakage, metrics, CV splitter, API (including CI bounds on `/predict`).
- **CI**: workflow runs tests and a one-state smoke train before re-running tests with artefacts present.

---

## 13. Troubleshooting

| Symptom | Likely cause | Action |
|---------|----------------|--------|
| API fails at startup | No `models/*.joblib` | Run `make train` or `train-fast` |
| XGBoost import error on Mac | Missing OpenMP | `brew install libomp` |
| Streamlit PermissionError on `~/.streamlit` | Sandboxed or read-only home | Use `make dashboard` (local `HOME`) |
| Prophet / CmdStan errors | CmdStan not installed | Re-run install step or `cmdstanpy.install_cmdstan()` |
| High MAPE | Short, noisy series | Expected; see [LIMITATIONS.md](LIMITATIONS.md) |

---

## 14. Related documents

| Document | Content |
|----------|---------|
| [README.md](README.md) | Quick start, results, assignment compliance |
| [DECISIONS.md](DECISIONS.md) | Design rationale |
| [LIMITATIONS.md](LIMITATIONS.md) | Known gaps and prioritised next steps |

---
