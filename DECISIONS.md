# Design Decisions

Concise rationale for the trade-offs taken in this project. Listed roughly in order of impact.

---

### 0. Mapping the assignment's lag specification to weekly granularity

The brief asks for lag features at `t-1, t-7, t-30`. These values conventionally mean *days*, but the source dataset is fundamentally **weekly**: dates are spaced 3–9 days apart and deduplicate to ~188 unique points per state over 5 years. There is no daily resolution to recover.

After resampling to a strict weekly grid (decision #1 below), the requested lag offsets translate to:

| Brief (days) | Weekly equivalent | Implemented as |
|---|---|---|
| `t-1`  | < 1 week — not representable in a weekly grid; closest is the previous week | `lag_1` (1 week back) |
| `t-7`  | exactly 1 week | `lag_1` |
| `t-30` | ~4.3 weeks | `lag_4` |

I also include `lag_2, lag_8, lag_13, lag_26` (2 weeks, 2 months, ~1 quarter, ~6 months) because these capture mid-range seasonality that the brief's three lags alone miss. All seven lags are listed in `config.yaml::features.lags` and built in `src/feature_engineering.py`.

**Trade-off considered**: I could have left the data daily and used the original `t-1, t-7, t-30` spec literally, but that would mean either (a) carrying NaN for the 4–5 days/week with no observations, or (b) introducing artificial daily values via interpolation that don't reflect reality. Weekly aggregation preserves real signal at the cost of losing day-level seasonality (which doesn't exist in the data anyway).

---

### 1. Resampling to a strict weekly grid

The raw data has irregular spacing (3–9 day gaps between observations). Lag-based features and statistical models like ARIMA/SARIMA assume a regular cadence. I resample each state's series to weekly (Sunday-ending) frequency with `sum()` aggregation. Missing weeks are filled with forward-fill + linear interpolation, both of which are causal (no future leakage).

Alternatives considered:
- Daily resampling → too sparse, would amplify noise.
- Keeping irregular spacing → would force me to drop ARIMA/SARIMA, and lag features become meaningless.

### 2. Per-state models, not a global model

I train and persist one model **per state** rather than one global multi-state model. Reasons: state series are very different in scale (Wyoming ≈ \$10M/week, California ≈ \$1B/week) and exhibit different seasonality. A global model would either underperform on small states or be biased by large ones. Per-state training also makes "best model varies by state" possible — the data shows Prophet wins for 38/43 states but LSTM is better for 3 noisy ones (Georgia, Michigan, North Carolina) and SARIMA wins for 2 (Mississippi, Nebraska).

Trade-off: more files on disk (215 model artefacts), longer training (~10 min), but each forecast is much more accurate.

### 3. Five-model bake-off, with selection by validation RMSE

I train ARIMA, SARIMA, XGBoost, LSTM, and Prophet for every state, evaluate them on a held-out chronological 20%, and select the one with the lowest RMSE. RMSE penalises large errors quadratically, which aligns with supply-chain cost: a single large forecast miss is more painful than many small ones.

Alternatives:
- AIC-based selection (only works for the same family of model).
- Ensembling the top-2 (better accuracy, more complexity, harder to debug — punted).

### 4. Strict zero-leakage feature engineering

Every lag/rolling feature is computed via `series.shift(k)` so feature[t] only uses values from t-1 or earlier. The rolling statistics use `.shift(1).rolling(window=w)` so the window itself never includes t. There is a dedicated test (`tests/test_feature_engineering.py::test_no_data_leakage_lag_features`) that asserts `lag_k[t] == sales[t-k]` for randomised samples — this is the most important test in the repo.

### 5. Recursive multi-step forecasting for ML models

XGBoost and LSTM are inherently single-step models. For 8-week horizons I unroll predictions: predict t+1, append it to the series, recompute features, predict t+2, etc. This accumulates error but is the only correct way to avoid leakage on an autoregressive feature set. ARIMA / SARIMA / Prophet handle multi-step natively.

### 6. Train-only scaler for LSTM

Initially the `MinMaxScaler` was fit on `train + val` combined, which leaks the validation min/max into training. The fix: fit on training data only. Validation values that exceed the training range are simply transformed slightly outside `[0, 1]`, which is realistic and matches what production inference would see.

### 7. Path-traversal hardening on model loader

`pickle.load` and `joblib.load` both deserialise arbitrary Python — equivalent to running the file's code. To mitigate the risk of an attacker placing a malicious file in `models/`:
- model name is checked against an allow-list,
- state is sanitised to a safe slug,
- the resolved path is asserted to remain inside `MODELS_DIR` via `Path.relative_to()`.

The `models/` directory is therefore part of the trust boundary — only the training pipeline should write there.

### 8. Configuration-driven, cwd-independent

All paths in `config.yaml` are resolved to absolute paths under the project root in `src/config_loader.py`. This means the API or training script can be launched from any working directory (Docker, cron, CI, etc.) without breaking. Before this fix, running `uvicorn main:app` from `/tmp` would silently fail to load models.

### 9. What I did NOT do (and why)

- **Hyper-parameter search per model** — out of scope for a 1-day exercise. Defaults plus a small ARIMA grid are reasonable; further tuning is the next obvious win.
- **Cross-validation / walk-forward** — single hold-out validation is enough to demonstrate the pattern; expanding-window CV would multiply training time by ~5×.
- **Confidence intervals on forecasts** — Prophet and ARIMA expose them natively; I'd surface them in the API as `forecast_low`/`forecast_high` if I had another hour.
- **Authentication, rate-limiting, observability** — not in scope for a take-home; would be required for a public deploy. Discussed in `LIMITATIONS.md`.
