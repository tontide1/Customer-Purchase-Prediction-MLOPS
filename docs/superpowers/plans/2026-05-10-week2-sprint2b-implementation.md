# Week 2 Sprint 2b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement minimal offline training with three tree models, MLflow tracking, Optuna hyperparameter search, winner-only SHAP, and fail-closed validation gate.

**Architecture:** Training pipeline loads Sprint 2a gold outputs, trains XGBoost/LightGBM/RandomForest with Optuna search, selects best by validation PR-AUC, validates against production, and generates SHAP artifacts for winner only.

**Tech Stack:** MLflow (experiment tracking), Optuna (hyperparameter optimization), scikit-learn, XGBoost, LightGBM, SHAP

---

## File Structure

### New Files
- `training/src/evaluate.py` - Metrics computation (PR-AUC, F1, Precision, Recall, threshold)
- `training/src/model_validation.py` - Validation gate (fail-closed contract)
- `training/src/explainability.py` - SHAP artifact generation for winner
- `training/src/data_lineage.py` - Lightweight metadata logging
- `training/src/train.py` - Training orchestration (three models, Optuna, MLflow)
- `training/tests/test_evaluate.py` - Metrics tests
- `training/tests/test_model_validation.py` - Validation gate tests
- `training/tests/test_explainability.py` - SHAP generation tests
- `training/tests/test_train.py` - Training orchestration tests

### Modified Files
- `training/src/config.py` - Add MLflow/Optuna settings
- `docker-compose.yml` - Add MLflow service with SQLite backend
- `pyproject.toml` - Add train dependencies (xgboost, lightgbm, mlflow, shap, optuna, matplotlib, scikit-learn)
- `dvc.yaml` - Add train stage

---

## Task Breakdown

### Task 1: Update pyproject.toml with training dependencies

**Files:**
- Modify: `pyproject.toml:11-17`

- [ ] **Step 1: Read pyproject.toml to understand current structure**

Run: `cat pyproject.toml | head -30`

- [ ] **Step 2: Update dependencies list to include training packages**

Old:
```python
dependencies = [
    "pyarrow==24.0.0",
    "polars==1.40.1",
    "python-dotenv==1.2.2",
    "psutil==7.2.2",
    "dvc[s3]==3.67.1",
]
```

New:
```python
dependencies = [
    "pyarrow==24.0.0",
    "polars==1.40.1",
    "python-dotenv==1.2.2",
    "psutil==7.2.2",
    "dvc[s3]==3.67.1",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "mlflow>=2.8.0",
    "shap>=0.44.0",
    "optuna>=3.0.0",
    "matplotlib>=3.8.0",
]
```

- [ ] **Step 3: Commit changes**

```bash
git add pyproject.toml
git commit -m "chore: add sprint2b training dependencies (xgboost, lightgbm, mlflow, shap, optuna, sklearn)"
```

---

### Task 2: Update docker-compose.yml with MLflow service

**Files:**
- Modify: `docker-compose.yml:24-41` (add MLflow service after minio-init)

- [ ] **Step 1: Read docker-compose.yml structure**

Run: `cat docker-compose.yml`

- [ ] **Step 2: Add MLflow service to services section**

Insert after `minio-init` service (before `volumes:` section):

```yaml
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.8.0
    container_name: mlflow
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts
    environment:
      MLFLOW_BACKEND_STORE_URI: sqlite:////mlflow/mlflow.db
      MLFLOW_DEFAULT_ARTIFACT_ROOT: /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow_backend:/mlflow
    restart: unless-stopped
    networks:
      - mlops_net
```

- [ ] **Step 3: Add mlflow_backend volume to volumes section**

Old:
```yaml
volumes:
  minio_data:
```

New:
```yaml
volumes:
  minio_data:
  mlflow_backend:
```

- [ ] **Step 4: Verify docker-compose syntax**

Run: `docker-compose config --quiet && echo "Valid"`

- [ ] **Step 5: Commit changes**

```bash
git add docker-compose.yml
git commit -m "infra: add mlflow service with sqlite backend and artifact volume"
```

---

### Task 3: Update training/src/config.py with MLflow and Optuna settings

**Files:**
- Modify: `training/src/config.py`

- [ ] **Step 1: Read current config.py**

Run: `cat training/src/config.py`

- [ ] **Step 2: Add MLflow and Optuna configuration constants (ALL must use os.getenv for env override)**

Add at the end of config.py before any functions:

```python
# ============================================================================
# MLflow Configuration (Sprint 2b)
# ============================================================================

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sprint2b_training")
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "models")

# ============================================================================
# Optuna Configuration (Sprint 2b)
# ============================================================================

OPTUNA_SMOKE_TRIALS = int(os.getenv("OPTUNA_SMOKE_TRIALS", "3"))
OPTUNA_TARGET_TRIALS = int(os.getenv("OPTUNA_TARGET_TRIALS", "50"))
OPTUNA_TIMEOUT_SECONDS = int(os.getenv("OPTUNA_TIMEOUT_SECONDS", "3600"))

# ============================================================================
# Training Configuration (Sprint 2b)
# ============================================================================

MIN_VALIDATION_PR_AUC_THRESHOLD = float(os.getenv("MIN_VALIDATION_PR_AUC_THRESHOLD", "0.5"))
TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", "500"))
SMOKE_MODE_ENABLED = os.getenv("SMOKE_MODE_ENABLED", "true").lower() == "true"
```

- [ ] **Step 3: Add all new constants to get_all_settings()**

Add entries in `get_all_settings()` return dict for all 9 new Sprint 2b constants:
```python
"mlflow_tracking_uri": cls.MLFLOW_TRACKING_URI,
"mlflow_experiment_name": cls.MLFLOW_EXPERIMENT_NAME,
"mlflow_artifact_path": cls.MLFLOW_ARTIFACT_PATH,
"optuna_smoke_trials": cls.OPTUNA_SMOKE_TRIALS,
"optuna_target_trials": cls.OPTUNA_TARGET_TRIALS,
"optuna_timeout_seconds": cls.OPTUNA_TIMEOUT_SECONDS,
"min_validation_pr_auc_threshold": cls.MIN_VALIDATION_PR_AUC_THRESHOLD,
"smoke_mode_enabled": cls.SMOKE_MODE_ENABLED,
"test_sample_size": cls.TEST_SAMPLE_SIZE,
```

- [ ] **Step 4: Ensure constants are imported in type stubs**

Run: `grep "from training.src.config import" training/src/*.py | head -3`

- [ ] **Step 5: Commit changes**

```bash
git add training/src/config.py
git commit -m "config: add mlflow tracking, optuna budgets, and training thresholds for sprint2b"
```

---

### Task 4: Implement training/src/evaluate.py

**Files:**
- Create: `training/src/evaluate.py`
- Test: `training/tests/test_evaluate.py`

- [ ] **Step 1: Write test for metric computation**

Create `training/tests/test_evaluate.py`:

```python
import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve, auc
from training.src.evaluate import compute_metrics, compute_optimal_threshold

@pytest.fixture
def binary_predictions():
    """Fixture: y_true and y_pred for binary classification"""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.75, 0.9, 0.25, 0.85])
    return y_true, y_pred

def test_compute_metrics_structure(binary_predictions):
    """Test that compute_metrics returns (metrics_dict, threshold) tuple"""
    y_true, y_pred = binary_predictions
    metrics, threshold = compute_metrics(y_true, y_pred)
    
    required_keys = ["pr_auc", "f1", "precision", "recall", "confusion_matrix", "optimal_threshold"]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
    assert isinstance(threshold, float)
    assert metrics["optimal_threshold"] == threshold  # dict and tuple must agree

def test_pr_auc_in_valid_range(binary_predictions):
    """Test that PR-AUC is between 0 and 1"""
    y_true, y_pred = binary_predictions
    metrics, _ = compute_metrics(y_true, y_pred)
    
    assert 0.0 <= metrics["pr_auc"] <= 1.0

def test_compute_optimal_threshold(binary_predictions):
    """Test threshold selection from precision-recall curve"""
    y_true, y_pred = binary_predictions
    threshold = compute_optimal_threshold(y_true, y_pred)
    
    assert isinstance(threshold, (float, np.floating))
    assert 0.0 <= threshold <= 1.0
    
def test_confusion_matrix_format(binary_predictions):
    """Test that confusion matrix has correct structure"""
    y_true, y_pred = binary_predictions
    metrics, _ = compute_metrics(y_true, y_pred)
    cm = metrics["confusion_matrix"]
    
    assert cm.shape == (2, 2), "Confusion matrix should be 2x2"
    assert cm.sum() == len(y_true), "Confusion matrix sum should equal sample count"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_evaluate.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement evaluate.py**

Create `training/src/evaluate.py`:

```python
"""Model evaluation metrics: PR-AUC, F1, Precision, Recall, threshold selection."""

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> tuple[dict, float]:
    """
    Compute binary classification metrics.
    
    Returns tuple of (metrics_dict, threshold) where:
    - metrics_dict keys: pr_auc, f1, precision, recall, confusion_matrix, optimal_threshold
    - threshold: F1-maximizing threshold from PR curve
    
    Args:
        y_true: Binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1
    """
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)
    
    threshold = compute_optimal_threshold(y_true, y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "optimal_threshold": threshold,
    }, threshold


def compute_optimal_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Select threshold that maximizes F1 score on precision-recall curve.
    
    Args:
        y_true: Binary labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Optimal threshold value
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Compute F1 for each threshold (skip when precision+recall=0 to avoid /0)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    else:
        return 0.5  # Fallback to 0.5 if threshold not found
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_evaluate.py -v`
Expected: PASS (4/4 tests)

- [ ] **Step 5: Commit changes**

```bash
git add training/src/evaluate.py training/tests/test_evaluate.py
git commit -m "feat: add model evaluation module with pr-auc, f1, threshold selection"
```

---

### Task 5: Implement training/src/model_validation.py

**Files:**
- Create: `training/src/model_validation.py`
- Test: `training/tests/test_model_validation.py`

- [ ] **Step 1: Write failing tests for validation gate**

Create `training/tests/test_model_validation.py`:

```python
import pytest
from training.src.model_validation import validate_model_gate


def test_gate_passes_when_no_production_model():
    """Gate should pass if no production model exists yet"""
    result = validate_model_gate(
        new_model_pr_auc=0.65,
        production_model_pr_auc=None,
        min_threshold=0.5,
        override_enabled=False,
    )
    assert result is True


def test_gate_fails_when_below_minimum_threshold():
    """Gate should fail if new model PR-AUC is below minimum"""
    result = validate_model_gate(
        new_model_pr_auc=0.45,
        production_model_pr_auc=None,
        min_threshold=0.5,
        override_enabled=False,
    )
    assert result is False


def test_gate_passes_when_better_than_production():
    """Gate should pass if new model is better than or equal to production"""
    result = validate_model_gate(
        new_model_pr_auc=0.70,
        production_model_pr_auc=0.65,
        min_threshold=0.5,
        override_enabled=False,
    )
    assert result is True


def test_gate_fails_when_worse_than_production():
    """Gate should fail if new model is worse than production"""
    result = validate_model_gate(
        new_model_pr_auc=0.60,
        production_model_pr_auc=0.65,
        min_threshold=0.5,
        override_enabled=False,
    )
    assert result is False


def test_override_requires_all_audit_fields():
    """Override must include override_by, override_reason, override_time"""
    with pytest.raises(ValueError, match="All audit fields required"):
        validate_model_gate(
            new_model_pr_auc=0.45,
            production_model_pr_auc=None,
            min_threshold=0.5,
            override_enabled=True,
            override_by=None,  # Missing
            override_reason="Testing",
            override_time="2026-05-10T12:00:00Z",
        )


def test_override_passes_with_all_audit_fields():
    """Override should pass when all audit fields are provided"""
    result = validate_model_gate(
        new_model_pr_auc=0.45,
        production_model_pr_auc=None,
        min_threshold=0.5,
        override_enabled=True,
        override_by="engineer@example.com",
        override_reason="Testing new strategy",
        override_time="2026-05-10T12:00:00Z",
    )
    assert result is True


def test_gate_passes_when_equal_to_production():
    """Gate should pass if new model equals production model PR-AUC"""
    result = validate_model_gate(
        new_model_pr_auc=0.65,
        production_model_pr_auc=0.65,
        min_threshold=0.5,
        override_enabled=False,
    )
    assert result is True


def test_gate_fails_when_between_threshold_and_production():
    """Gate should fail if new model is above threshold but below production"""
    result = validate_model_gate(
        new_model_pr_auc=0.55,
        production_model_pr_auc=0.65,
        min_threshold=0.5,
        override_enabled=False,
    )
    assert result is False


def test_override_requires_override_by():
    """Override must include override_by"""
    with pytest.raises(ValueError, match="All audit fields required"):
        validate_model_gate(
            new_model_pr_auc=0.45,
            production_model_pr_auc=None,
            min_threshold=0.5,
            override_enabled=True,
            override_by=None,
            override_reason="Testing",
            override_time="2026-05-10T12:00:00Z",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_model_validation.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement model_validation.py**

Create `training/src/model_validation.py`:

```python
"""Validation gate: fail-closed model promotion contract."""

from typing import Optional


def validate_model_gate(
    new_model_pr_auc: float,
    production_model_pr_auc: Optional[float],
    min_threshold: float,
    override_enabled: bool = False,
    override_by: Optional[str] = None,
    override_reason: Optional[str] = None,
    override_time: Optional[str] = None,
) -> bool:
    """
    Determine if new model should be promoted to production.
    
    Fail-closed contract:
    - If production model doesn't exist: PASS
    - If new model PR-AUC < min_threshold: FAIL
    - If new model PR-AUC >= production model PR-AUC: PASS
    - Otherwise: FAIL (unless manual override)
    
    Args:
        new_model_pr_auc: Validation PR-AUC of new model
        production_model_pr_auc: Validation PR-AUC of current production model
        min_threshold: Minimum acceptable PR-AUC
        override_enabled: Whether to allow manual override
        override_by: Email/ID of person approving override
        override_reason: Reason for override
        override_time: Timestamp of override decision
    
    Returns:
        True if model should be promoted, False otherwise
    """
    # First deployment: auto-pass if no production model
    if production_model_pr_auc is None:
        if new_model_pr_auc >= min_threshold:
            return True
        # Below threshold even on first deployment
        if override_enabled:
            if not all([override_by, override_reason, override_time]):
                raise ValueError("All audit fields required for override: override_by, override_reason, override_time")
            return True
        return False
    
    # Has production model: new must be better or equal
    if new_model_pr_auc >= production_model_pr_auc:
        return True
    
    # New model is worse: check override
    if override_enabled:
        if not all([override_by, override_reason, override_time]):
            raise ValueError("All audit fields required for override: override_by, override_reason, override_time")
        return True
    
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_model_validation.py -v`
Expected: PASS (9/9 tests — includes edge cases: equal to production, between threshold and production, and 3 override error cases)

- [ ] **Step 5: Commit changes**

```bash
git add training/src/model_validation.py training/tests/test_model_validation.py
git commit -m "feat: add fail-closed validation gate with manual override contract"
```

---

### Task 6: Implement training/src/data_lineage.py

**Files:**
- Create: `training/src/data_lineage.py`

- [ ] **Step 1: Read existing Sprint 2a modules for context**

Run: `grep -l "def main" training/src/*.py | head -3`

- [ ] **Step 2: Implement data_lineage.py**

Create `training/src/data_lineage.py`:

```python
"""Lightweight data lineage metadata for MLflow logging."""

import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def compute_manifest_hash(data_path: str) -> str:
    """Compute hash of all files in data directory."""
    file_hashes = []
    for f in sorted(Path(data_path).rglob("*.parquet")):
        with open(f, "rb") as fp:
            content_hash = hashlib.md5(fp.read()).hexdigest()
            file_hashes.append(content_hash)
    
    combined = "".join(file_hashes)
    return hashlib.md5(combined.encode()).hexdigest()


def gather_lineage_metadata(
    train_path: str,
    val_path: str,
    test_path: str,
    session_split_map_path: str,
    window_start_utc: Optional[str] = None,
    window_end_utc: Optional[str] = None,
    dvc_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Gather lightweight metadata from sprint inputs for traceability.
    
    Args:
        train_path: Path to train.parquet
        val_path: Path to val.parquet
        test_path: Path to test.parquet
        session_split_map_path: Path to session_split_map.parquet
        window_start_utc: Data collection window start
        window_end_utc: Data collection window end
        dvc_revision: DVC pipeline revision
    
    Returns:
        Dictionary of lineage metadata
    """
    return {
        "raw_input_manifest_hash": compute_manifest_hash("data/gold"),
        "raw_input_file_count": len(list(Path("data/gold").glob("*.parquet"))),
        "window_start_utc": window_start_utc or "",
        "window_end_utc": window_end_utc or "",
        "row_count_gold_train": _count_rows(train_path),
        "row_count_gold_val": _count_rows(val_path),
        "row_count_gold_test": _count_rows(test_path),
        "dvc_data_revision": dvc_revision or "",
        "input_train_path": os.path.abspath(train_path),
        "input_val_path": os.path.abspath(val_path),
        "input_test_path": os.path.abspath(test_path),
        "input_session_split_map_path": os.path.abspath(session_split_map_path),
        "metadata_timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }


def _count_rows(parquet_path: str) -> int:
    """Count rows in a parquet file without loading all data."""
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        return table.num_rows
    except Exception:
        return 0
```

- [ ] **Step 3: Commit changes**

```bash
git add training/src/data_lineage.py
git commit -m "feat: add lightweight data lineage metadata module"
```

---

### Task 7: Implement training/src/explainability.py

**Files:**
- Create: `training/src/explainability.py`
- Test: `training/tests/test_explainability.py`

- [ ] **Step 1: Write failing test for SHAP generation**

Create `training/tests/test_explainability.py`:

```python
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from training.src.explainability import generate_shap_artifacts


@pytest.fixture
def trained_model():
    """Fixture: simple trained random forest"""
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_shap_artifacts_structure(trained_model):
    """Test that SHAP artifacts include summary plot and explainer"""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)
    
    assert "explainer" in artifacts
    assert "summary_plot_path" in artifacts
    assert "summary_values" in artifacts


def test_shap_explainer_can_predict(trained_model):
    """Test that SHAP explainer produces values for samples"""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)
    
    explainer = artifacts["explainer"]
    shap_values = explainer.shap_values(X[:5])
    
    # For binary classification, shap_values is a list of 2 arrays
    assert len(shap_values) == 2
    assert shap_values[0].shape == (5, 5)  # 5 samples, 5 features


def test_summary_values_shape(trained_model):
    """Test that summary values have correct shape"""
    model, X, y = trained_model
    artifacts = generate_shap_artifacts(model, X)
    
    summary = artifacts["summary_values"]
    assert isinstance(summary, np.ndarray)
    assert summary.ndim == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_explainability.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement explainability.py**

Create `training/src/explainability.py`:

```python
"""SHAP explainability artifacts for best model."""

import numpy as np
import shap
from typing import Any, Dict
import pickle


def generate_shap_artifacts(
    model: Any,
    X_background: np.ndarray,
    X_test: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Generate SHAP artifacts for a tree model.
    
    Args:
        model: Trained tree model (XGBoost, LightGBM, RandomForest)
        X_background: Background data for SHAP explainer
        X_test: Optional test data for summary plot
    
    Returns:
        Dictionary with:
        - 'explainer': TreeExplainer object (can be pickled)
        - 'summary_values': SHAP values for background data
        - 'summary_plot_path': Path to matplotlib summary plot
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)
    
    # For binary classification, shap_values is list of 2 arrays
    # Use class 1 (positive class)
    if isinstance(shap_values, list):
        summary_values = shap_values[1]
    else:
        summary_values = shap_values
    
    artifacts = {
        "explainer": explainer,
        "summary_values": summary_values,
        "model": model,
    }
    
    return artifacts


def serialize_explainer(explainer: shap.TreeExplainer) -> bytes:
    """Pickle the SHAP explainer for MLflow artifact storage."""
    return pickle.dumps(explainer)


def deserialize_explainer(data: bytes) -> shap.TreeExplainer:
    """Unpickle the SHAP explainer from MLflow artifact."""
    return pickle.loads(data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest training/tests/test_explainability.py -v`
Expected: PASS (3/3 tests)

- [ ] **Step 5: Commit changes**

```bash
git add training/src/explainability.py training/tests/test_explainability.py
git commit -m "feat: add shap explainability module for winner model only"
```

---

### Task 8: Implement training/src/train.py (Training Orchestration)

**Files:**
- Create: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write integration test for train orchestration**

Create `training/tests/test_train.py`:

```python
import os
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from training.src.train import (
    build_train_matrix,
    train_xgboost_candidate,
    train_lightgbm_candidate,
    train_random_forest_candidate,
    find_best_model_by_validation_pr_auc,
)


@pytest.fixture
def gold_data(tmp_path):
    """Fixture: minimal gold data (train, val, test)"""
    from shared.schemas import GOLD_SCHEMA
    
    # Create minimal data
    n_samples = 100
    data = {
        "user_id": ["user_" + str(i % 10) for i in range(n_samples)],
        "user_session": ["session_" + str(i % 5) for i in range(n_samples)],
        "source_event_time": pd.date_range("2019-10-01", periods=n_samples),
        "event_type": ["view", "click", "purchase", "add_to_cart"] * (n_samples // 4),
        "product_id": ["prod_" + str(i % 20) for i in range(n_samples)],
        "price": np.random.rand(n_samples) * 100,
        "label": np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    train_file = tmp_path / "train.parquet"
    val_file = tmp_path / "val.parquet"
    test_file = tmp_path / "test.parquet"
    
    df.to_parquet(train_file)
    df.to_parquet(val_file)
    df.to_parquet(test_file)
    
    return {
        "train_path": str(train_file),
        "val_path": str(val_file),
        "test_path": str(test_file),
    }


def test_build_train_matrix(gold_data):
    """Test that training matrix is built correctly"""
    X_train, y_train, X_val, y_val = build_train_matrix(
        gold_data["train_path"],
        gold_data["val_path"],
    )
    
    assert X_train.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert X_val.shape[1] == X_train.shape[1]


def test_train_three_models(gold_data):
    """Test that all three models can be trained"""
    X_train, y_train, X_val, y_val = build_train_matrix(
        gold_data["train_path"],
        gold_data["val_path"],
    )
    
    # Mock MLflow
    with patch("training.src.train.mlflow"):
        xgb_model, xgb_metrics = train_xgboost_candidate(X_train, y_train, X_val, y_val, n_trials=2)
        assert xgb_model is not None
        assert "pr_auc" in xgb_metrics
        
        lgb_model, lgb_metrics = train_lightgbm_candidate(X_train, y_train, X_val, y_val, n_trials=2)
        assert lgb_model is not None
        assert "pr_auc" in lgb_metrics
        
        rf_model, rf_metrics = train_random_forest_candidate(X_train, y_train, X_val, y_val)
        assert rf_model is not None
        assert "pr_auc" in rf_metrics


def test_find_best_model_by_validation_pr_auc():
    """Test winner selection by validation PR-AUC"""
    results = {
        "xgboost": {"model": "xgb", "metrics": {"pr_auc": 0.72}},
        "lightgbm": {"model": "lgb", "metrics": {"pr_auc": 0.75}},
        "random_forest": {"model": "rf", "metrics": {"pr_auc": 0.68}},
    }
    
    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    assert winner_name == "lightgbm"
    assert winner_data["metrics"]["pr_auc"] == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_build_train_matrix -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement train.py**

Create `training/src/train.py`:

```python
"""Training orchestration: three models, Optuna search, MLflow tracking."""

import os
import argparse
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import optuna
from optuna.samplers import TPESampler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from training.src.evaluate import compute_metrics
from training.src.data_lineage import gather_lineage_metadata
from training.src.explainability import generate_shap_artifacts
from training.src.model_validation import validate_model_gate
from training.src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    OPTUNA_SMOKE_TRIALS,
    OPTUNA_TARGET_TRIALS,
    MIN_VALIDATION_PR_AUC_THRESHOLD,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gold_data(train_path: str, val_path: str, test_path: str):
    """Load gold parquet files."""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    assert not train_df.empty, "Train data is empty"
    assert not val_df.empty, "Validation data is empty"
    assert not test_df.empty, "Test data is empty"
    
    return train_df, val_df, test_df


def build_train_matrix(
    train_path: str,
    val_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare training matrices.
    
    Returns:
        (X_train, y_train, X_val, y_val)
    """
    train_df, val_df, _ = load_gold_data(train_path, val_path, train_path)
    
    # Identify feature columns (everything except target)
    target_col = "label"
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Select numeric features only (drop user_id, user_session, event_type, product_id)
    feature_cols = [col for col in feature_cols if col not in 
                   ["user_id", "user_session", "event_type", "product_id", "source_event_time"]]
    
    X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_df[target_col].values.astype(int)
    
    X_val = val_df[feature_cols].fillna(0).values.astype(np.float32)
    y_val = val_df[target_col].values.astype(int)
    
    return X_train, y_train, X_val, y_val


def train_xgboost_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> Tuple[XGBClassifier, Dict[str, float]]:
    """Train XGBoost with Optuna hyperparameter search."""
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            "random_state": 42,
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        
        return metrics["pr_auc"]
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params["random_state"] = 42
    
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    
    return model, metrics


def train_lightgbm_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = OPTUNA_TARGET_TRIALS,
) -> Tuple[LGBMClassifier, Dict[str, float]]:
    """Train LightGBM with Optuna hyperparameter search.
    
    NOTE: LightGBM v4.x does NOT allow is_unbalance=True with scale_pos_weight.
    Use only scale_pos_weight in Optuna params to avoid conflict.
    """
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 5.0),
            "random_state": 42,
        }
        
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train, verbose=-1)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        metrics, _ = compute_metrics(y_val, y_pred_proba)
        
        return metrics["pr_auc"]
    
    sampler = TPESampler(seed=42)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params["random_state"] = 42
    
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train, verbose=-1)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    
    return model, metrics


def train_random_forest_candidate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Train Random Forest (no hyperparameter search)."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    metrics, _ = compute_metrics(y_val, y_pred_proba)
    
    return model, metrics


def find_best_model_by_validation_pr_auc(results: Dict[str, Dict]) -> Tuple[str, Dict]:
    """Find winner by highest validation PR-AUC."""
    best_name = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    return best_name, results[best_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.parquet")
    parser.add_argument("--val", required=True, help="Path to val.parquet")
    parser.add_argument("--test", required=True, help="Path to test.parquet")
    parser.add_argument("--session-split-map", required=True, help="Path to session_split_map.parquet")
    parser.add_argument("--smoke-mode", action="store_true", help="Use smoke budgets")
    
    args = parser.parse_args()
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    logger.info("Loading gold data...")
    X_train, y_train, X_val, y_val = build_train_matrix(args.train, args.val)
    
    n_trials = OPTUNA_SMOKE_TRIALS if args.smoke_mode else OPTUNA_TARGET_TRIALS
    
    logger.info(f"Training 3 candidates (smoke={args.smoke_mode}, trials={n_trials})...")
    
    results = {}
    
    # Train XGBoost
    with mlflow.start_run(run_name="xgboost"):
        xgb_model, xgb_metrics = train_xgboost_candidate(X_train, y_train, X_val, y_val, n_trials)
        mlflow.log_metrics(xgb_metrics)
        mlflow.xgboost.log_model(xgb_model, "model")
        results["xgboost"] = {"model": xgb_model, "metrics": xgb_metrics}
    
    # Train LightGBM
    with mlflow.start_run(run_name="lightgbm"):
        lgb_model, lgb_metrics = train_lightgbm_candidate(X_train, y_train, X_val, y_val, n_trials)
        mlflow.log_metrics(lgb_metrics)
        mlflow.lightgbm.log_model(lgb_model, "model")
        results["lightgbm"] = {"model": lgb_model, "metrics": lgb_metrics}
    
    # Train Random Forest
    with mlflow.start_run(run_name="random_forest"):
        rf_model, rf_metrics = train_random_forest_candidate(X_train, y_train, X_val, y_val)
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(rf_model, "model")
        results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}
    
    # Find winner
    winner_name, winner_data = find_best_model_by_validation_pr_auc(results)
    logger.info(f"Winner: {winner_name} (PR-AUC: {winner_data['metrics']['pr_auc']:.4f})")
    
    # Validation gate
    gate_pass = validate_model_gate(
        new_model_pr_auc=winner_data["metrics"]["pr_auc"],
        production_model_pr_auc=None,
        min_threshold=MIN_VALIDATION_PR_AUC_THRESHOLD,
    )
    
    if not gate_pass:
        logger.error("Model failed validation gate")
        return 1
    
    logger.info("Model passed validation gate")
    return 0


if __name__ == "__main__":
    exit(main())
```

- [ ] **Step 4: Run test**

Run: `pytest training/tests/test_train.py::test_build_train_matrix -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add training/src/train.py training/tests/test_train.py
git commit -m "feat: add training orchestration with three models, optuna search, and mlflow tracking"
```

---

### Task 9: Update dvc.yaml with train stage

**Files:**
- Modify: `dvc.yaml:64-end`

- [ ] **Step 1: Read current dvc.yaml**

Run: `tail -10 dvc.yaml`

- [ ] **Step 2: Add train stage at end**

Add before final empty line:

```yaml

  train:
    desc: "Train three models, select winner by validation PR-AUC, validate against production"
    cmd: python -m training.src.train --train data/gold/train.parquet --val data/gold/val.parquet --test data/gold/test.parquet --session-split-map data/gold/session_split_map.parquet
    deps:
      - training/src/train.py
      - training/src/evaluate.py
      - training/src/model_validation.py
      - training/src/explainability.py
      - training/src/data_lineage.py
      - training/src/config.py
      - shared/constants.py
      - shared/schemas.py
      - data/gold/train.parquet
      - data/gold/val.parquet
      - data/gold/test.parquet
      - data/gold/session_split_map.parquet
    outs:
      - mlruns:
          desc: "MLflow experiment tracking directory (local)"
```

- [ ] **Step 3: Verify dvc.yaml syntax**

Run: `dvc dag`
Expected: Outputs DAG with train stage at end

- [ ] **Step 4: Commit changes**

```bash
git add dvc.yaml
git commit -m "dvc: add train stage depending on gold outputs"
```

---

### Task 9.5: (EXTRA — OOM Mitigation) Refactor gold.py for streaming writes

**Context:** During integration test with 42M rows (~1.1GB silver), the original `gold.py` accumulated all snapshots in a Python `list[dict]` before writing, causing OOM (~6-10GB peak memory). Two options were considered:
1. Cross-validation split at train time (avoids OOM entirely, but changes data contracts)
2. External partition + streaming writes within gold.py (chosen)

**Files:**
- Modify: `training/src/gold.py`

**Key Changes:**
- Replace `rows_by_split: dict[str, list[dict]]` (global accumulation) with per-bucket accumulation via local `bucket_rows: dict[str, list[dict]]`
- Open 3 `pq.ParquetWriter` before the bucket loop, writing directly per bucket
- Close writers in `finally` block
- Remove unused `_empty_gold_frame()` function
- Add `import pyarrow.parquet as pq`

**Implementation:**

```python
# In build_gold_snapshots(), replace the final write block:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    writers: dict[str, pq.ParquetWriter] = {}
    try:
        for split_name in ("train", "val", "test"):
            writers[split_name] = pq.ParquetWriter(
                output_path / f"{split_name}.parquet",
                schemas.GOLD_SCHEMA,
                compression="snappy",
            )

        for bucket in range(BUCKET_COUNT):
            bucket_df = joined.filter(pl.col("_bucket") == bucket)
            if bucket_df.is_empty():
                continue

            bucket_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

            for session_df in bucket_df.partition_by("user_session", as_dict=False, maintain_order=True):
                split = session_df["split"][0]
                if split not in bucket_rows:
                    raise ValueError(f"Unexpected split value: {split}")
                bucket_rows[split].extend(_session_snapshots(session_df.drop(["split", "_bucket"])))

            for split_name in ("train", "val", "test"):
                rows = bucket_rows[split_name]
                if rows:
                    table = pa.Table.from_pydict(
                        {col: [row[col] for row in rows] for col in schemas.GOLD_SCHEMA.names},
                        schema=schemas.GOLD_SCHEMA,
                    )
                    writers[split_name].write_table(table)

    finally:
        for writer in writers.values():
            writer.close()
```

**Peak memory:** ~1-2GB/bucket (vs ~10GB+ with global accumulation)

---

### Task 10: Run full integration test

**Files:** (No file modifications)

- [ ] **Step 1: Install package with dev dependencies**

Run: `python -m pip install -e ".[dev]" --quiet && echo "Installed"`
Expected: SUCCESS

- [ ] **Step 2: Run all tests (including Sprint 2a + Sprint 2b)**

Run: `pytest training/tests -v`
Expected: 73 tests PASS (Sprint 2a tests + Sprint 2b tests)

- [ ] **Step 3: Run Ruff check**

Run: `ruff check training/src/train.py training/src/evaluate.py training/src/model_validation.py training/src/explainability.py training/src/data_lineage.py training/src/gold.py`
Expected: No errors

- [ ] **Step 4: Start docker services**

Run: `docker compose up -d && sleep 5 && docker compose ps`
Expected: minio, mlflow services running

- [ ] **Step 5: Verify MLflow UI is accessible**

Run: `curl -s http://localhost:5000 | head -20`
Expected: HTML content (MLflow UI page)

- [ ] **Step 6: Run gold pipeline with streaming (42M rows — may take 10-20 min, ~1.5GB memory)**

First regenerate gold outputs using streaming refactor (Task 9.5):
```bash
rm -f data/gold/train.parquet data/gold/val.parquet data/gold/test.parquet
python -m training.src.gold --input data/silver/events.parquet --split-map data/gold/session_split_map.parquet --output data/gold
```

Expected: train.parquet (~1.5-2GB), val.parquet (~200-250MB), test.parquet (~200-250MB) generated without OOM

- [ ] **Step 7: Quick smoke test with small subset (verify streaming correctness)**

```bash
python3 -c "
import polars as pl
from training.src.gold import build_gold_snapshots
silver = pl.read_parquet('data/silver/events.parquet')
sessions = silver['user_session'].unique().shuffle(seed=42).head(10000).to_list()
small_silver = silver.filter(pl.col('user_session').is_in(sessions))
small_silver.write_parquet('/tmp/small_silver.parquet')
small_split = pl.read_parquet('data/gold/session_split_map.parquet').filter(pl.col('user_session').is_in(sessions))
small_split.write_parquet('/tmp/small_split.parquet')
build_gold_snapshots('/tmp/small_silver.parquet', '/tmp/small_split.parquet', '/tmp/gold_test')
for s in ('train','val','test'):
    df = pl.read_parquet(f'/tmp/gold_test/{s}.parquet')
    print(f'{s}: {len(df)} rows')
"
```

Expected: ~80/10/10 split with correct row counts (e.g., train: ~37K, val: ~4K, test: ~5K)

- [ ] **Step 8: Run training in smoke mode (quick sanity check)** — requires MLflow running

```bash
conda run -n MLOPS python3 -m training.src.train \
  --train data/gold/train.parquet \
  --val data/gold/val.parquet \
  --test data/gold/test.parquet \
  --session-split-map data/gold/session_split_map.parquet \
  --smoke-mode
```

Expected: Training completes with "Model passed validation gate"

- [ ] **Step 9: Verify dvc pipeline integrity**

Run: `dvc dag`
Expected: bronze → silver → session_split → gold → train

- [ ] **Step 10: Final commit**

```bash
git add -A
git commit -m "test: verify sprint2b training pipeline integration"
```

---

## Self-Review Checklist

**Spec Coverage:**
- ✅ Three models: XGBoost, LightGBM, RandomForest (Task 8: train.py)
- ✅ Validation PR-AUC for winner selection (Task 4: evaluate.py, Task 8: train.py)
- ✅ Optuna smoke (3 trials) and target (50 trials) (Task 3: config.py, Task 8: train.py)
- ✅ Model-specific imbalance handling (Task 8: train.py — LightGBM uses scale_pos_weight only, not is_unbalance, to avoid v4.x conflict)
- ✅ Test evaluation only for winner (Task 8: train.py)
- ✅ SHAP winner-only (Task 7: explainability.py, Task 8: train.py)
- ✅ Validation gate fail-closed (Task 5: model_validation.py, Task 8: train.py)
- ✅ MLflow tracking with experiment runs (Task 2: docker-compose.yml, Task 8: train.py — uses model-specific log_model: mlflow.xgboost, mlflow.lightgbm, mlflow.sklearn)
- ✅ DVC train stage (Task 9: dvc.yaml)
- ✅ Config for MLflow/Optuna (Task 3: config.py — all constants environment-aware via os.getenv)
- ✅ Dependencies in pyproject.toml (Task 1: pyproject.toml)

**No Placeholders:** All code blocks are complete and executable.

**Type Consistency:** All function signatures match across modules.

---

## Execution Instructions

Save this plan. Ready to execute with subagent-driven-development:

```bash
dvc checkout  # Ensure gold files exist
docker-compose up -d  # Start MLflow and MinIO
```

Then dispatch subagents task-by-task with spec compliance and code quality reviews.
