# Event Type, Average Precision, And Test Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve training signal and reporting by adding `event_type` to categorical features, adding `average_precision` to model metrics, and evaluating the selected winner once on `test.parquet`.

**Architecture:** Keep the current validation-based winner selection unchanged so the training loop stays stable. Extend the feature contract so `event_type` is treated like the other categorical inputs, extend metric computation so both PR-AUC style reporting and average precision are available, and add a post-selection test evaluation step that logs test metrics without using test data to choose the winner. Preserve the existing train/val/test input flow from `load_gold_data()`, but carry the test frame through `PreparedTrainingData` so the post-selection evaluation has a clean input.

**Tech Stack:** Python, pandas, scikit-learn, Optuna, CatBoost, LightGBM, XGBoost, MLflow, pytest, Ruff.

---

### Task 1: Add `event_type` to the categorical feature contract

**Files:**
- Modify: `training/src/categorical_features.py`
- Test: `training/tests/test_categorical_features.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_training_frame_includes_event_type():
    df = _build_gold_like_frame()
    df["event_type"] = ["view", "cart", "purchase", "view"]
    frame = prepare_training_frame(df)

    assert frame.categorical_columns == [
        "category_id",
        "category_code",
        "brand",
        "event_type",
    ]
    assert "event_type" in frame.features.columns
    assert str(frame.features["event_type"].dtype) == "object"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_categorical_features.py::test_prepare_training_frame_includes_event_type -v`
Expected: fail because `event_type` is not yet part of the categorical feature set.

- [ ] **Step 3: Write minimal implementation**

```python
CATEGORICAL_FEATURE_COLUMNS = [
    "category_id",
    "category_code",
    "brand",
    "event_type",
]

def prepare_training_frame(df: pd.DataFrame) -> TrainingFrame:
    required_columns = set(
        NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required training columns: {missing}")

    features = df[NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS].copy()
    features[NUMERIC_FEATURE_COLUMNS] = features[NUMERIC_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    target = df[TARGET_COLUMN].astype(int).copy()
    return TrainingFrame(
        features=features,
        target=target,
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_categorical_features.py::test_prepare_training_frame_includes_event_type -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/categorical_features.py training/tests/test_categorical_features.py training/tests/test_train.py
git commit -m "feat: add event type categorical feature"
```

### Task 2: Add `average_precision` to metric computation

**Files:**
- Modify: `training/src/evaluate.py`
- Test: `training/tests/test_evaluate.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compute_metrics_includes_average_precision(binary_predictions):
    y_true, y_pred = binary_predictions
    metrics, threshold = compute_metrics(y_true, y_pred)

    assert "average_precision" in metrics
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(y_true, y_pred)
    )
    assert metrics["optimal_threshold"] == threshold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_evaluate.py::test_compute_metrics_includes_average_precision -v`
Expected: fail because `compute_metrics()` does not return `average_precision` yet.

- [ ] **Step 3: Write minimal implementation**

```python
from sklearn.metrics import average_precision_score

def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple[dict, float]:
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    average_precision = average_precision_score(y_true, y_pred_proba)
    threshold = compute_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "pr_auc": float(pr_auc),
        "average_precision": float(average_precision),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "optimal_threshold": float(threshold),
    }

    return metrics, threshold
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_evaluate.py::test_compute_metrics_includes_average_precision -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/evaluate.py training/tests/test_evaluate.py training/tests/test_train.py
git commit -m "feat: add average precision metric"
```

### Task 3: Evaluate the selected winner on `test.parquet`

**Files:**
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_logs_test_metrics_for_selected_winner(gold_data, monkeypatch):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)

    seen_test_eval = {"called": False}

    def fake_train(*args, **kwargs):
        return _FakeModel(), {"pr_auc": 0.9, "average_precision": 0.9, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    def fake_eval_on_test(*args, **kwargs):
        seen_test_eval["called"] = True
        return {"pr_auc": 0.8, "average_precision": 0.8, "confusion_matrix": np.array([[1, 0], [0, 1]])}

    monkeypatch.setattr("training.src.train.train_catboost_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_lightgbm_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_xgboost_candidate", fake_train)
    monkeypatch.setattr("training.src.train.evaluate_winner_on_test", fake_eval_on_test)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "training.src.train",
            "--train",
            gold_data["train_path"],
            "--val",
            gold_data["val_path"],
            "--test",
            gold_data["test_path"],
            "--session-split-map",
            gold_data["split_map_path"],
            "--smoke-mode",
        ],
    )

    assert main() == 0
    assert seen_test_eval["called"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_main_logs_test_metrics_for_selected_winner -v`
Expected: fail because the winner is only evaluated on validation today.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class PreparedTrainingData:
    train_features: pd.DataFrame
    train_target: pd.Series
    val_features: pd.DataFrame
    val_target: pd.Series
    test_features: pd.DataFrame
    test_target: pd.Series
    categorical_columns: list[str]
    numeric_columns: list[str]
    categorical_artifacts: CategoricalEncodingArtifacts

def build_training_data(
    train_path: str,
    val_path: str,
    test_path: str,
) -> PreparedTrainingData:
    train_df, val_df, test_df = load_gold_data(train_path, val_path, test_path)

    train_frame = prepare_training_frame(train_df)
    val_frame = prepare_training_frame(val_df)
    test_frame = prepare_training_frame(test_df)

    categorical_artifacts = fit_categorical_encoders(train_frame.features)
    train_features = transform_with_categorical_contract(
        train_frame.features, categorical_artifacts
    )
    val_features = transform_with_categorical_contract(
        val_frame.features, categorical_artifacts
    )
    test_features = transform_with_categorical_contract(
        test_frame.features, categorical_artifacts
    )

    return PreparedTrainingData(
        train_features=train_features,
        train_target=train_frame.target,
        val_features=val_features,
        val_target=val_frame.target,
        test_features=test_features,
        test_target=test_frame.target,
        categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_artifacts=categorical_artifacts,
    )

def evaluate_winner_on_test(model, test_features: pd.DataFrame, test_target: pd.Series) -> dict:
    y_pred_proba = model.predict_proba(test_features)[:, 1]
    metrics, _ = compute_metrics(test_target, y_pred_proba)
    return metrics

# in main()
winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
test_metrics = evaluate_winner_on_test(
    winner_data["model"],
    data.test_features,
    data.test_target,
)
mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating))})
logger.info(
    "Test results for %s: PR-AUC=%.4f, average_precision=%.4f",
    winner_name,
    test_metrics["pr_auc"],
    test_metrics["average_precision"],
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_train.py::test_main_logs_test_metrics_for_selected_winner -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add training/src/train.py training/tests/test_train.py
git commit -m "feat: report winner metrics on test split"
```

### Task 4: Verify the updated training contract end to end

**Files:**
- Modify: none
- Test: `training/tests/test_categorical_features.py`
- Test: `training/tests/test_evaluate.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Run the focused tests**

Run:
```bash
pytest training/tests/test_categorical_features.py training/tests/test_evaluate.py training/tests/test_train.py -q
```
Expected: all tests pass.

- [ ] **Step 2: Run static checks on touched files**

Run:
```bash
ruff check training/src/categorical_features.py training/src/evaluate.py training/src/train.py training/tests/test_categorical_features.py training/tests/test_evaluate.py training/tests/test_train.py
```
Expected: no lint errors.

- [ ] **Step 3: Run a local smoke training command**

Run:
```bash
conda run -n MLOPS python -m training.src.train \
  --train data/gold/train.parquet \
  --val data/gold/val.parquet \
  --test data/gold/test.parquet \
  --session-split-map data/gold/session_split_map.parquet \
  --smoke-mode
```

Expected: training completes, winner is selected on validation, and the winner’s test metrics are logged separately with `test_` prefixes.

**Assumptions:**
- `test.parquet` is used only for final reporting, not for winner selection.
- Current one-day training data remains in place for testing; full-month data will be adopted later without changing the feature or metric interfaces.
- The existing `pr_auc` field remains for backward compatibility; `average_precision` is an additional metric, not a replacement.
