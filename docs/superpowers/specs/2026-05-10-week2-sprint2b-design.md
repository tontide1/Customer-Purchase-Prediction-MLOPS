# Week 2 Sprint 2b Training Design

**Goal:** Add the minimal offline training slice for Week 2: three native tree models, local MLflow tracking, Optuna smoke/target budgets, winner-only SHAP, and a fail-closed validation gate.

**Scope:** This spec covers only Sprint 2b. It does not include Kafka, Redis, FastAPI serving, Streamlit, online evaluation, or drift monitoring.

## Architecture

Sprint 2b consumes the Sprint 2a gold outputs and turns them into trainable model artifacts. The pipeline trains XGBoost, LightGBM, and Random Forest on `data/gold/train.parquet`, evaluates all candidates on `data/gold/val.parquet`, selects the winner by validation PR-AUC, then runs a final test evaluation on `data/gold/test.parquet`. Only the best model gets SHAP artifacts and the validation gate check.

MLflow becomes the experiment and registry backbone for this sprint. A dockerized tracking service stores runs, metrics, parameters, model artifacts, SHAP artifacts, and lineage metadata. Optuna is used per model with a tiny smoke budget for fast local iteration and the full target budget preserved in config. The validation gate is fail-closed by default and only allows manual override with explicit audit fields.

## Contract

The locked Sprint 2b contract is:

- Train exactly 3 native models: `XGBoost`, `LightGBM`, `Random Forest`
- Use validation PR-AUC to choose the best model
- Use 3 Optuna trials in smoke mode and 50 trials in target mode
- Apply model-specific imbalance handling:
  - XGBoost: `scale_pos_weight`
  - LightGBM: `is_unbalance=True` or `scale_pos_weight`
  - Random Forest: `class_weight='balanced'`
- Evaluate the selected best model on the test set only after validation selection
- Compute SHAP only for the best validation model
- Save the SHAP explainer pickle and summary artifact to MLflow
- Fail closed if MLflow registry lookup or production metric access fails
- First deployment auto-passes the validation gate
- Manual override requires `override_by`, `override_reason`, and `override_time`
- MLflow runs are grouped under one experiment, with one run per candidate model
- Log the validation comparison table (`model_name`, `validation_pr_auc`, `optimal_threshold`, `is_winner`) to MLflow
- MLflow is provided by docker-compose and uses a local SQLite backend for the sprint MVP
- `train` becomes a DVC stage after Sprint 2a artifacts exist
- Keep online-serving concerns out of this sprint

## Components

### 1. Training orchestration

`training/src/train.py` orchestrates the model runs. It loads the gold train/val/test files, builds a shared training matrix, and executes the three model candidates one by one so each candidate gets its own MLflow run.

The orchestrator is responsible for:

- reading `train.parquet`, `val.parquet`, and `test.parquet`
- initializing the shared MLflow experiment
- applying the correct imbalance setting per model
- launching Optuna search with the configured trial budget
- logging parameters, metrics, and the fitted model artifact
- comparing validation PR-AUC across the three runs
- triggering test evaluation only for the winner
- handing the winner to SHAP and validation gate logic

### 2. Model evaluation

`training/src/evaluate.py` computes the metrics needed for selection and reporting:

- PR-AUC
- F1
- Precision
- Recall
- confusion matrix
- optimal threshold from the precision-recall curve

The evaluation helper returns a metrics dictionary and the chosen threshold. The threshold is logged to MLflow and reused for the test-set report of the best model.

### 3. Validation gate

`training/src/model_validation.py` implements the gate contract. It reads the current Production model from MLflow, compares its PR-AUC to the new candidate, and returns a boolean pass/fail result.

The gate must behave as follows:

- if no Production model exists, return pass
- if the registry cannot be read, return fail unless manual override is explicitly enabled
- if the new model PR-AUC is below the configured minimum threshold, return fail
- if the new model PR-AUC is greater than or equal to Production, return pass
- if manual override is enabled, require all audit fields and log them to MLflow

### 4. Explainability

`training/src/explainability.py` generates SHAP artifacts only for the winning model. The implementation uses `shap.TreeExplainer`, writes a summary plot artifact, and serializes the explainer object so the later prediction API can load it without recomputing.

### 5. Lineage logging

`training/src/data_lineage.py` gathers lightweight metadata from the sprint inputs and logs it to MLflow. The goal is traceability, not a full data catalog.

Expected lineage fields include:

- `raw_input_manifest_hash`
- `raw_input_file_count`
- `window_start_utc`
- `window_end_utc`
- `row_count_gold_train`
- `row_count_gold_val`
- `row_count_gold_test`
- `dvc_data_revision`
- input and output artifact paths

### 6. MLflow service

`docker-compose.yml` gains a dedicated `mlflow` service for the sprint. The service exposes the MLflow UI on port `5000`, uses a local SQLite backend stored on a named volume, and persists artifacts on a named volume so repeated runs survive container restarts.

The sprint does not require a production-grade registry backend. The goal is a reproducible local tracking environment that is simple to run in CI and on a laptop.

### 7. Config and dependencies

`training/src/config.py` needs new settings for:

- MLflow tracking URI
- MLflow experiment name
- Optuna smoke trial count
- Optuna target trial count
- minimum PR-AUC threshold
- smoke-mode sample limit, if needed for faster local runs

`pyproject.toml` needs the training dependencies required by the sprint:

- `scikit-learn`
- `xgboost`
- `lightgbm`
- `mlflow`
- `shap`
- `optuna`
- `matplotlib`

### 8. DVC stage

`dvc.yaml` gains a `train` stage that depends on the gold outputs and the new training modules. The stage exists to preserve reproducibility of the offline training entrypoint, even though MLflow stores the model artifact itself.

## Data Flow

1. Sprint 2a produces `data/gold/train.parquet`, `data/gold/val.parquet`, and `data/gold/test.parquet`.
2. `train.py` loads the three gold files and starts one MLflow run per model.
3. Optuna searches the configured hyperparameter space with the smoke or target trial budget.
4. Each candidate logs validation metrics and a fitted model artifact.
5. The winner is selected by validation PR-AUC.
6. The winner is evaluated on the test set and logs the final report.
7. SHAP artifacts are generated only for the winner.
8. The validation gate compares the winner against Production in MLflow.
9. If the gate passes, the model is registered or promoted according to the existing MLflow flow.

## Error Handling

The sprint should fail closed on bad infrastructure or bad inputs.

- Missing gold files should fail the training entrypoint immediately.
- Empty train or validation data should fail fast.
- A model that cannot be fit should fail its own MLflow run and not block other candidates from running.
- If all candidates fail, the sprint must fail rather than promoting a partial result.
- If MLflow registry access fails during gate evaluation, the gate should fail closed unless manual override is explicitly enabled with all audit fields.
- SHAP generation should only run for the winner; if SHAP fails, the model can still be selected, but the failure must be visible in logs and MLflow artifacts.

## Testing

The tests should prove the contracts, not the implementation details.

- Train entrypoint uses exactly `XGBoost`, `LightGBM`, and `Random Forest`
- Smoke Optuna uses 3 trials while target config keeps 50
- Each model applies the correct imbalance parameter
- Validation PR-AUC selects the winner
- Threshold selection comes from the precision-recall curve
- Test-set evaluation runs only for the best model
- SHAP artifacts are logged only for the winner
- MLflow registry failures fail closed
- First deployment auto-passes the gate
- Manual override requires all audit fields
- DVC `train` stage points at the gold inputs and training modules

## Implementation Notes

- Keep the sprint local-first and dependency-light.
- Use the native MLflow flavors for model logging so later serving code can load the same artifacts consistently.
- Keep SHAP winner-only to avoid unnecessary artifact churn.
- Keep the MLflow backend simple for this sprint; do not add an external database.
- Do not add SMOTE experiments in this sprint; keep the imbalance strategy limited to the per-model settings in the contract.
- Do not introduce Kafka, Redis, FastAPI, dashboards, or online evaluation here.
