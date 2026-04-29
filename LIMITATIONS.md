# Limitations & Next Steps

Things I'd improve with more time, ranked by how much they'd matter.

> **What changed since v1**: items 1–6 below were originally listed here as "next steps" and have since been implemented. They're crossed out here so the reviewer can see what's *still* open.

---

### Modelling

1. ~~**Hyper-parameter tuning per state.**~~ ✅ **Done** via Optuna TPE on 3 representative states (`run_tuning.py`). For per-state tuning we'd need to amortise the cost — see #18 below.
2. ~~**Walk-forward (rolling) cross-validation.**~~ ✅ **Done** in `src/cross_validation.py` and `run_cv.py` (5 expanding folds).
3. ~~**Confidence intervals.**~~ ✅ **Done** — native CIs from ARIMA / SARIMA / Prophet, residual-bootstrap (`±1.96·σ·√t`) for XGBoost / LSTM. Exposed as `forecast_low` / `forecast_high` in the API.
4. ~~**Ensemble of top-2 models per state.**~~ ✅ **Done** — inverse-RMSE-weighted average. Wins on **31/43 states** (72 %).
5. **External regressors.** Marketing spend, promotions, weather, COVID indicator — Prophet and XGBoost both accept exogenous variables. Currently the only regressor is the holiday flag. **Highest expected lift** of anything still on the list.
6. **Probabilistic LSTM** (e.g. negative-binomial output head). Current bootstrap CIs assume Gaussian errors — fine for most states but breaks down for very small / spiky series.
7. **Log-transform target.** Residuals are heteroskedastic — variance grows with mean. `np.log(1 + sales)` would stabilise it and likely improve all 5 base models.

### Evaluation

8. ~~**Per-step MAPE.**~~ ✅ **Done** — `run_horizon_analysis.py` produces `reports/horizon_error.png`. Notable finding: XGBoost is the most horizon-stable model.
9. **Outlier-aware metrics.** The MAPE numbers (~28 %) are inflated by a few weeks with anomalous sales. Median absolute % error (MdAPE) or trimmed MAPE would tell a fairer story.
10. **Calibration check on CIs.** I claim 95 % intervals but never check empirically that ~5 % of held-out actuals fall outside the band. A simple Gaussian-residual back-test would validate the claim.

### Engineering

11. **Pre-load all models into RAM at API startup.** Currently each `/predict` re-reads the model file. With 43 states this is fine; at 4 300 it would not be.
12. **Caching layer (LRU).** Forecast outputs are deterministic given a model version. A 1-hour TTL cache would handle the bulk of any traffic.
13. **Async I/O for inference.** XGBoost / LSTM forecast loops are CPU-bound; a thread pool would prevent uvicorn workers from blocking on a single slow request.
14. **Structured logs (JSON).** Plain-text logging is fine for local dev but blocks easy ingestion into Datadog / Loki / CloudWatch.
15. ~~**CI workflow.**~~ ✅ **Done** — `.github/workflows/ci.yml` runs lint + tests + smoke-train on every push.

### Operational (only relevant if going to production)

16. **Authentication + rate-limiting** (`X-API-Key` + `slowapi`).
17. **Sentry / OpenTelemetry** for error tracking + tracing.
18. **Scheduled retraining** (weekly cron — sales data drifts). With per-state Optuna budgets this becomes a real ML-Ops job.
19. **Model registry / versioning** (MLflow, or just S3 with hashes).
20. **Drift monitoring**: log every prediction, recompute RMSE once actuals arrive, alert on regression > X %.
21. **Multi-stage Dockerfile.** Current image is a single FROM; a builder stage for CmdStan would shave ~600 MB off the runtime image.

---

### Known Quirks

- **MAPE is ~28 % median for the best model per state.** Partly a real signal (the Beverages series is volatile and short) and partly because each state has only ~190 weekly points. With 5+ years of data, results would tighten considerably.
- **The ensemble's CI band is the weighted-average of component bands**, not a proper joint distribution. Strictly correct intervals would require Monte Carlo simulation across components — overkill for the gain.
- **The forecast for the largest state can clip to 0** in edge cases. The `clip(lower=0)` is correct (sales can't be negative) but masks model issues; for production I'd raise an alert when clipping triggers > 0 % of the time.
- **Prophet still wins on smaller states** — its regularisation suits short, smoother series where the ensemble's variance reduction has nothing to work with.
