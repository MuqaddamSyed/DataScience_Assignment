# Limitations & Next Steps

Things I'd improve with more time, ranked by how much they'd matter.

---

### Modelling

1. **Hyper-parameter tuning per state.** Currently uses one set of defaults for XGBoost / LSTM / Prophet across all states. Optuna or Bayesian search on each state's validation set could realistically shave 5–15 % off MAPE.
2. **Walk-forward (rolling) cross-validation.** A single 80/20 hold-out has high variance for short series. Expanding-window CV with 4–5 folds is the textbook fix.
3. **Confidence intervals.** ARIMA and Prophet expose them natively; Quantile regression for XGBoost; MC Dropout for LSTM. Would change the API contract from `forecast_sales: float` to `{point, low, high}` triplets.
4. **Ensemble of top-2 models per state.** Simple averaging of Prophet + LSTM is often more accurate than either alone — modest extra complexity for a real win.
5. **External regressors.** Marketing spend, promotions, weather, COVID indicator — Prophet and XGBoost both accept exogenous variables. Currently the only regressor is the holiday flag.

### Evaluation

6. **Per-step MAPE.** Track how accuracy degrades over the 8-week horizon. Currently I report aggregate validation MAPE; per-step breakdown shows whether errors blow up at week 8.
7. **Outlier-aware metrics.** The MAPE numbers (29–35 %) are inflated by a few weeks with anomalous sales. A trimmed/robust MAPE would tell a fairer story.

### Engineering

8. **Pre-load all models into RAM at API startup.** Currently each `/predict` re-reads the model file. With 43 states this is fine; at 4,300 it would not be.
9. **Caching layer.** Forecast outputs are deterministic given a model version. A 1-hour TTL cache would handle the bulk of any traffic.
10. **Async I/O for inference.** XGBoost / LSTM forecast loops are CPU-bound; using a thread pool would prevent uvicorn workers from blocking on a single slow request.
11. **Structured logs (JSON).** Plain-text logging is fine for local dev but blocks easy ingestion into Datadog / Loki / CloudWatch.
12. **CI workflow.** A `.github/workflows/ci.yml` running `pytest` on push would catch regressions earlier; left out to keep the repo focused on the assignment.

### Operational (only relevant if going to production)

13. **Authentication + rate-limiting** (`X-API-Key` + `slowapi`).
14. **Sentry** for error tracking.
15. **Scheduled retraining** (weekly cron — sales data drifts).
16. **Model registry / versioning** (MLflow, or just S3 with hashes).
17. **Drift monitoring**: log every prediction, recompute RMSE once actuals arrive, alert on regression.

---

### Known Quirks

- **MAPE is high (29–35 %) for many states.** This is partly a real signal (the Beverages series is volatile) and partly because some states only have ~190 data points after deduplication. With 5+ years of data per state, results would look a lot tighter.
- **Prophet wins almost everywhere.** That's not a coding bug — it's because Prophet is very forgiving on noisy, short series with strong seasonality. With richer features (XGBoost) or more data (LSTM), the picture would shift.
- **The forecast for the largest state can clip to 0** in edge cases. The `clip(lower=0)` on every model output is correct (sales can't be negative) but masks model issues; for production I'd raise an alert when clipping triggers more than 0 % of the time.
