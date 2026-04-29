.PHONY: help install train serve test report docker clean

PYTHON := python3
VENV   := venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python
PORT   ?= 8000

help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:                    ## Create venv and install all dependencies
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PY) -c "import cmdstanpy; cmdstanpy.install_cmdstan()" || true
	@echo "Installed. Activate with: source $(VENV)/bin/activate"

train:                      ## Train all models (~10-15 min)
	$(PY) run_training.py

train-fast:                 ## Train only 3 states (smoke test, ~1 min)
	$(PY) run_training.py --states California Texas "New York"

serve:                      ## Start FastAPI on port $(PORT)
	$(VENV)/bin/uvicorn main:app --host 0.0.0.0 --port $(PORT) --reload

test:                       ## Run pytest suite
	$(VENV)/bin/pytest tests/ -v

report:                     ## Generate diagnostic plots into reports/
	$(PY) reports/generate_plots.py

cv:                         ## Walk-forward cross-validation on 3 sample states
	$(PY) run_cv.py

tune:                       ## Hyperparameter tuning for XGBoost
	$(PY) run_tuning.py --trials 30

horizon:                    ## Per-step error analysis
	$(PY) run_horizon_analysis.py

dashboard:                  ## Launch Streamlit dashboard
	$(VENV)/bin/streamlit run dashboard/app.py

eda:                        ## Re-execute the EDA notebook
	$(VENV)/bin/jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

docker:                     ## Build the Docker image
	docker build -t forecasting-api .

docker-run:                 ## Run the Docker image (port 8000)
	docker run -p 8000:8000 -v $(PWD)/models:/app/models forecasting-api

clean:                      ## Remove caches, logs, and temporary artifacts
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf logs/*.log

clean-models:               ## Remove all trained models (requires re-training)
	rm -rf models/*.pkl models/*.joblib models/*.json
