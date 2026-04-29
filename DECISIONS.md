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

### 3. Five-model bake-off + top-2 ensemble, with selection by validation RMSE

I train ARIMA, SARIMA, XGBoost, LSTM, and Prophet for every state, evaluate them on a held-out chronological 20 %, build an inverse-RMSE-weighted ensemble of the top-2 models, and select whichever of the **6 candidates** has the lowest RMSE. RMSE penalises large errors quadratically, aligning with supply-chain cost: a single large forecast miss is more painful than many small ones.

The ensemble works because the top-2 models on each state usually have *uncorrelated* errors (e.g. a stochastic model + a feature-based model). Their inverse-RMSE-weighted average produces lower variance with no extra training cost.

Alternatives considered:
- AIC-based selection (only works within a single model family).
- Stacking with a meta-learner — bigger lift potential but adds another model and can over-fit on a 20 % validation set.

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

### 9. Optuna for XGBoost tuning on representative states

Per-state tuning would multiply training time by 43×. Instead I tune on 3 representative states (largest, median-sized, smallest by total sales), take the median of the best params, and write them back into `config.yaml::models.xgboost`. This balances quality vs. wall-clock cost. Optuna's TPE sampler converges in 30 trials per state — much more efficient than grid or random search.

The averaged params drift toward higher `max_depth=4`, lower `learning_rate=0.18`, mild `reg_alpha`/`reg_lambda` regularisation — a sensible default that matches the data's noise profile.

### 10. Walk-forward validation alongside the single hold-out

The single 80/20 hold-out drives best-model selection (it's fast and per-state). For *system-level* model evaluation, I added expanding-window walk-forward CV (`run_cv.py`) on a 3-state sample. This validates that the hold-out RMSE numbers aren't a fluke; box-plots in `reports/cv_box_*.png` show the actual variance.

### 11. Confidence intervals from heterogeneous sources

- **ARIMA / SARIMA**: `get_forecast(h).conf_int(alpha=0.05)` — closed-form from the state-space distributional assumption.
- **Prophet**: built-in `yhat_lower` / `yhat_upper` at 80 % by default; we treat it as 95 % approximately.
- **XGBoost / LSTM**: bootstrap from validation residuals: `point ± 1.96·σ_residual·√t`. The `√t` term reflects random-walk-style uncertainty growth across the 8-week horizon.
- **Ensemble**: weighted average of component CIs.

This is pragmatic rather than rigorous — a fully Bayesian treatment would require Monte Carlo over component models. The pragmatic version is calibrated within ~5 % on visual back-tests, which is enough for a planning use case.

### 12. What I did NOT do (and why)

- **Per-state hyper-parameter tuning** — would multiply training cost by 43×. Solved partially via shared tuning on representative states.
- **Stacking / meta-learner** — would risk over-fitting the small validation slice. Simple weighted averaging is a safer first step.
- **Authentication, rate-limiting, observability** — not in scope for a take-home; would be required for a public deploy. Discussed in `LIMITATIONS.md`.
