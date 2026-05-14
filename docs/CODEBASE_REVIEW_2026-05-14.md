# Codebase Review — 2026-05-14

**Scope:** `training/`, `shared/`, `dvc.yaml`, `pyproject.toml`, `AGENTS.md`

---

### 1. `dvc.yaml` — silver/gold stage dependencies are imprecise

- `silver` stage: `deps` lists `data/bronze/events.parquet` instead of `data/bronze/` (the directory). Changes to any parquet file in `data/bronze` would not trigger cache invalidation correctly.
- `gold` stage: `deps` lists files under `data/silver/` rather than the `data/silver/events.parquet` output of the `silver` stage. This breaks DAG dependency resolution.

### 2. `dvc.yaml` — silver output contract vs AGENTS.md inconsistency

- `dvc.yaml` has `silver` stage output as `data/silver/` (directory).
- AGENTS.md and source (`silver.py`) treat it as a single file: `data/silver/events.parquet`.
- Pick one contract and enforce it everywhere.

### 3. `training/src/data_lineage.py` — misleading field names

- Fields are named `raw_input_*` but actually hash the **gold** directory (or whatever `directory` is passed at call time).
- Rename fields to match what they actually measure (e.g., `gold_input_*` or `input_*`).

### 4. `training/src/train.py` — wrong MLflow model flavor for CatBoost

- Uses `mlflow.sklearn.log_model()` to log the CatBoost model.
- CatBoost has its own MLflow flavor (`mlflow.catboost.log_model()`) which preserves category handling, feature importances, and native serialization.
- Should use the CatBoost flavor when the model is a CatBoost classifier.

### 5. `training/src/train.py` — SHAP artifact generation never called

- `explainability.py` has `generate_shap_artifacts()` ready to write SHAP summary plots and artifacts.
- The orchestration in `train.py` never calls it. Winner-model SHAP artifacts are never produced.

### 6. `training/tests/test_train.py` — `sys.argv` mutation creates race conditions

- Tests directly mutate `sys.argv`, which is process-global.
- Parallel or ordered test runs can interfere with each other.
- Use a context manager or `testargs` fixture that saves/restores `sys.argv`.

### 7. `training/src/config.py` — `Config` class variables cause test interference

- `Config` stores state (e.g., `Config.env`) as class-level attributes.
- Tests that modify `Config.env` leak state across test cases.
- Use instance-based config or reset class state in fixtures.

### 8. `pyproject.toml` — overly strict version pins

- Pins like `pandas==2.2.3`, `scikit-learn==1.6.1` are exact, not range-based.
- This causes dependency resolution failures and blocks minor/patch updates.
- Prefer loose upper-bound pins (e.g., `pandas>=2.2,<3.0`) or `~=` compatible-release specifiers where appropriate.
