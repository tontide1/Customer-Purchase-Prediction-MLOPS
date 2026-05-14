# Week 2 Categorical-Aware Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the Week 2 training pipeline to keep `category_id`, `category_code`, and `brand` as first-class categorical inputs while still training and comparing three models: `CatBoost`, `LightGBM`, and `XGBoost`.

**Architecture:** Keep the gold data contract unchanged, but stop collapsing every feature into one dense float matrix. Instead, build a shared training frame with explicit numeric and categorical columns, add a small preprocessing/adapter layer per model, and keep the rest of the orchestration the same: Optuna search, MLflow logging, winner selection by validation PR-AUC, SHAP for the winner, and the validation gate. Update repo docs in the same change so the executable contract and the written contract stay aligned.

**Tech Stack:** Python 3.11, pandas, scikit-learn, CatBoost, XGBoost, LightGBM, Optuna, MLflow, SHAP, pytest, DVC

---

## File Structure

### New Files
- `training/src/categorical_features.py` - shared categorical preprocessing helpers and column contract
- `training/tests/test_categorical_features.py` - preprocessing regression tests for nulls, unseen categories, and model-ready output

### Modified Files
- `training/src/train.py` - shared DataFrame contract, model adapters, and candidate roster
- `training/src/explainability.py` - SHAP support for CatBoost + categorical-aware tree models
- `training/src/config.py` - training constants if model roster or preprocessing settings need flags
- `pyproject.toml` - add `catboost`
- `dvc.yaml` - keep train stage command aligned with the new contract if args or artifacts change
- `README.md` - quick-start / current-state docs for the new model roster and categorical-aware input contract
- `AGENTS.md` - repo instructions and gotchas for categorical-aware training
- `docs/BLUEPRINT/01_OVERVIEW.md`
- `docs/BLUEPRINT/02_ARCHITECTURE.md`
- `docs/BLUEPRINT/03_FEATURES.md`
- `docs/BLUEPRINT/04_PIPELINES.md`
- `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- `docs/BLUEPRINT/07_TESTING.md`
- `docs/BLUEPRINT/09_EXPLAINABILITY.md`
- `docs/BLUEPRINT/11_DEMO.md`
- `docs/BLUEPRINT/12_ROADMAP.md`

---

## Task 1: Add CatBoost and lock the categorical-aware contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test for the new model roster**

Add a test that asserts the candidate set is exactly `{"catboost", "lightgbm", "xgboost"}` and that `build_train_matrix()` no longer attempts to coerce raw string columns to `float32`.

```python
from training.src.train import CANDIDATE_MODEL_NAMES


def test_candidate_models_are_catboost_lightgbm_xgboost():
    assert CANDIDATE_MODEL_NAMES == ("catboost", "lightgbm", "xgboost")
```

- [ ] **Step 2: Run the focused test to verify the current code fails**

Run: `pytest training/tests/test_train.py -q`
Expected: fail because the current code still includes `RandomForest` and still casts the whole matrix to `float32`.

- [ ] **Step 3: Implement the minimal contract update**

Update the candidate roster and add the CatBoost dependency:

```toml
dependencies = [
    "pyarrow==24.0.0",
    "polars==1.40.1",
    "python-dotenv==1.2.2",
    "psutil==7.2.2",
    "dvc[s3]==3.67.1",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "mlflow>=2.8.0",
    "shap>=0.44.0",
    "optuna>=3.0.0",
    "matplotlib>=3.8.0",
]
```

Keep `training/src/train.py` on a shared frame contract and expose a module-level constant such as `CANDIDATE_MODEL_NAMES = ("catboost", "lightgbm", "xgboost")` so the test can lock the roster without coupling to implementation internals.

- [ ] **Step 4: Re-run the focused test**

Run: `pytest training/tests/test_train.py -q`
Expected: pass once the roster and dependency contract match the new design.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml training/src/train.py training/tests/test_train.py
git commit -m "feat: switch sprint2b training to catboost lightgbm xgboost"
```

---

## Task 2: Introduce a shared categorical preprocessing contract

**Files:**
- Create: `training/src/categorical_features.py`
- Modify: `training/src/train.py`
- Test: `training/tests/test_categorical_features.py`

- [ ] **Step 1: Write the failing regression tests**

Add tests that use gold-like data containing string `category_id`, `category_code`, and `brand`, plus numeric engineered features.

```python
import pandas as pd
import pytest

from training.src.categorical_features import (
    CategoricalEncodingArtifacts,
    fit_categorical_encoders,
    prepare_training_frame,
    transform_with_categorical_contract,
)


@pytest.fixture
def gold_like_frame():
    return pd.DataFrame(
        {
            "total_views": [1, 2, 3],
            "total_carts": [0, 1, 1],
            "net_cart_count": [0, 1, 1],
            "cart_to_view_ratio": [0.0, 0.5, 0.33],
            "unique_categories": [1, 2, 2],
            "unique_products": [1, 2, 3],
            "session_duration_sec": [10.0, 20.0, 30.0],
            "price": [35.0, 50.0, 12.0],
            "category_id": ["1", "2", "3"],
            "category_code": ["a.b.c", "a.b.d", "a.b.e"],
            "brand": ["x", "y", "z"],
            "label": [0, 1, 0],
        }
    )


def test_prepare_training_frame_keeps_categorical_columns(gold_like_frame):
    frame = prepare_training_frame(gold_like_frame)
    assert list(frame.categorical_columns) == ["category_id", "category_code", "brand"]
    assert frame.numeric_columns == [
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "price",
    ]

def test_prepare_training_frame_handles_null_and_unseen_categories(gold_like_frame):
    train_df = gold_like_frame.copy()
    val_df = gold_like_frame.copy()
    val_df.loc[0, "brand"] = None
    val_df.loc[1, "category_code"] = "new.category"
    fitted = fit_categorical_encoders(train_df)
    transformed = transform_with_categorical_contract(val_df, fitted)
    assert transformed.isnull().sum().sum() == 0
```

- [ ] **Step 2: Run the tests to confirm the current float-cast path fails**

Run: `pytest training/tests/test_categorical_features.py -q`
Expected: fail because the current code still casts mixed dtype data into a numeric matrix too early.

- [ ] **Step 3: Implement the shared frame and encoder helpers**

Create a small preprocessing module that:

```python
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TrainingFrame:
    features: pd.DataFrame
    target: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]

@dataclass(frozen=True)
class CategoricalEncodingArtifacts:
    category_maps: dict[str, dict[str, int]]
    missing_token: str = "__MISSING__"
    unknown_token: str = "__UNK__"


def prepare_training_frame(df: pd.DataFrame) -> TrainingFrame:
    numeric_columns = [
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "price",
    ]
    categorical_columns = ["category_id", "category_code", "brand"]
    features = df[numeric_columns + categorical_columns].copy()
    target = df["label"].copy()
    return TrainingFrame(
        features=features,
        target=target,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def fit_categorical_encoders(train_df: pd.DataFrame) -> CategoricalEncodingArtifacts:
    category_maps: dict[str, dict[str, int]] = {}
    for column in ["category_id", "category_code", "brand"]:
        values = (
            train_df[column]
            .fillna("__MISSING__")
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        category_maps[column] = {
            "__MISSING__": 0,
            "__UNK__": 1,
            **{value: idx + 2 for idx, value in enumerate(values) if value not in {"__MISSING__", "__UNK__"}},
        }
    return CategoricalEncodingArtifacts(category_maps=category_maps)


def transform_with_categorical_contract(
    df: pd.DataFrame,
    artifacts: CategoricalEncodingArtifacts,
) -> pd.DataFrame:
    transformed = df.copy()
    for column, mapping in artifacts.category_maps.items():
        transformed[column] = (
            transformed[column]
            .fillna(artifacts.missing_token)
            .astype(str)
            .map(mapping)
            .fillna(mapping[artifacts.unknown_token])
            .astype("int64")
        )
    return transformed
```

The helper should:
- keep categorical columns as strings/category dtype
- fill nulls with a stable placeholder
- fit mappings on train only
- transform val/test using the train vocabulary
- leave numeric features as numeric columns in the same frame

- [ ] **Step 4: Re-run the tests**

Run: `pytest training/tests/test_categorical_features.py -q`
Expected: pass once the helper normalizes null/unseen categories and preserves the categorical columns.

- [ ] **Step 5: Commit**

```bash
git add training/src/categorical_features.py training/src/train.py training/tests/test_categorical_features.py
git commit -m "feat: add shared categorical preprocessing contract"
```

---

## Task 3: Add model adapters for CatBoost, LightGBM, and XGBoost

**Files:**
- Modify: `training/src/train.py`
- Modify: `training/src/explainability.py`
- Test: `training/tests/test_train.py`
- Test: `training/tests/test_explainability.py`

- [ ] **Step 1: Write adapter-level tests**

Add tests that verify each model can train from the shared frame contract and that the winner-only SHAP path works for all three model types.

```python
import pandas as pd
import pytest

@pytest.fixture
def shared_training_frame():
    train_df = pd.DataFrame(
        {
            "total_views": [1, 2, 3],
            "total_carts": [0, 1, 1],
            "net_cart_count": [0, 1, 1],
            "cart_to_view_ratio": [0.0, 0.5, 0.33],
            "unique_categories": [1, 2, 2],
            "unique_products": [1, 2, 3],
            "session_duration_sec": [10.0, 20.0, 30.0],
            "price": [35.0, 50.0, 12.0],
            "category_id": ["1", "2", "3"],
            "category_code": ["a.b.c", "a.b.d", "a.b.e"],
            "brand": ["x", "y", "z"],
            "label": [0, 1, 0],
        }
    )
    val_df = train_df.copy()
    return {"train": train_df, "val": val_df}


def test_catboost_adapter_trains_with_categorical_columns(shared_training_frame):
    model, metrics = train_catboost_candidate(
        shared_training_frame["train"],
        shared_training_frame["val"],
    )
    assert model is not None
    assert "pr_auc" in metrics

def test_shap_artifacts_support_catboost_and_tree_models(shared_training_frame):
    model, _ = train_catboost_candidate(
        shared_training_frame["train"],
        shared_training_frame["val"],
    )
    artifacts = generate_shap_artifacts(model, shared_training_frame["train"])
    assert artifacts["summary_plot_path"]
```

- [ ] **Step 2: Run the tests to confirm the current model stack is not ready**

Run: `pytest training/tests/test_train.py training/tests/test_explainability.py -q`
Expected: fail until CatBoost is wired in and RandomForest-specific assumptions are removed.

- [ ] **Step 3: Implement model-specific adapters**

Implement explicit adapters with one shared preprocessing output:

```python
def train_catboost_candidate(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> tuple[CatBoostClassifier, dict]:
    # categorical columns passed as names, no float-cast of raw strings

def train_lightgbm_candidate(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> tuple[LGBMClassifier, dict]:
    # mark categorical columns explicitly before fit

def train_xgboost_candidate(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> tuple[XGBClassifier, dict]:
    # use the categorical-aware input path supported by the installed XGBoost version
```

Use the same validation metrics helper and Optuna search wrapper pattern for each model, but keep the input preparation model-specific where necessary.

- [ ] **Step 4: Update SHAP handling for CatBoost**

Keep winner-only SHAP generation, but remove the RandomForest-specific branch and make sure CatBoost and the tree boosters all flow through a supported explainer path.

- [ ] **Step 5: Re-run the focused tests**

Run: `pytest training/tests/test_train.py training/tests/test_explainability.py -q`
Expected: pass for all three candidates and the winner SHAP path.

- [ ] **Step 6: Commit**

```bash
git add training/src/train.py training/src/explainability.py training/tests/test_train.py training/tests/test_explainability.py
git commit -m "feat: add catboost adapter and categorical-aware model training"
```

---

## Task 4: Update orchestration, logging, and validation behavior

**Files:**
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`
- Test: `training/tests/test_model_validation.py`
- Test: `training/tests/test_data_lineage.py`

- [ ] **Step 1: Add tests for end-to-end orchestration**

Verify that `main()` can train three candidates from gold inputs, select the winner by validation PR-AUC, and log the expected model metadata.

```python
import pandas as pd

def test_main_trains_three_categorical_models(tmp_path, monkeypatch):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "val.parquet"
    test_path = tmp_path / "test.parquet"
    split_map_path = tmp_path / "session_split_map.parquet"
    frame = pd.DataFrame(
        {
            "total_views": [1, 2],
            "total_carts": [0, 1],
            "net_cart_count": [0, 1],
            "cart_to_view_ratio": [0.0, 0.5],
            "unique_categories": [1, 2],
            "unique_products": [1, 2],
            "session_duration_sec": [10.0, 20.0],
            "price": [35.0, 50.0],
            "category_id": ["1", "2"],
            "category_code": ["a.b.c", "a.b.d"],
            "brand": ["x", "y"],
            "label": [0, 1],
        }
    )
    frame.to_parquet(train_path)
    frame.to_parquet(val_path)
    frame.to_parquet(test_path)
    pd.DataFrame(
        {
            "user_session": ["s1", "s2"],
            "session_start_time": ["2019-10-01 00:00:00", "2019-10-01 00:10:00"],
            "session_end_time": ["2019-10-01 00:01:00", "2019-10-01 00:11:00"],
            "split": ["train", "val"],
        }
    ).to_parquet(split_map_path)
    exit_code = main()
    assert exit_code == 0
```

- [ ] **Step 2: Run the tests and capture current failure mode**

Run: `pytest training/tests/test_model_validation.py training/tests/test_data_lineage.py training/tests/test_train.py -q`
Expected: current code fails until the new model roster and categorical-aware orchestration are in place.

- [ ] **Step 3: Wire the orchestration to the new candidates**

Keep the current execution flow, but replace the old model block with:
- CatBoost training run
- LightGBM training run
- XGBoost training run
- winner selection by validation PR-AUC
- fail-closed validation gate unchanged

Preserve `mlflow.log_metrics()` scalar handling and keep `confusion_matrix` logged separately as text/artifact data.

- [ ] **Step 4: Keep lineage and validation contracts stable**

Do not change the validation gate semantics or lineage hashing behavior unless a test demands it. The only behavioral change here is the candidate stack and the preprocessing contract feeding it.

- [ ] **Step 5: Re-run the orchestration tests**

Run: `pytest training/tests/test_model_validation.py training/tests/test_data_lineage.py training/tests/test_train.py -q`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add training/src/train.py training/tests/test_model_validation.py training/tests/test_data_lineage.py training/tests/test_train.py
git commit -m "feat: wire categorical-aware training orchestration"
```

---

## Task 5: Sync README, AGENTS, and blueprint docs

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/BLUEPRINT/01_OVERVIEW.md`
- Modify: `docs/BLUEPRINT/02_ARCHITECTURE.md`
- Modify: `docs/BLUEPRINT/03_FEATURES.md`
- Modify: `docs/BLUEPRINT/04_PIPELINES.md`
- Modify: `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- Modify: `docs/BLUEPRINT/07_TESTING.md`
- Modify: `docs/BLUEPRINT/09_EXPLAINABILITY.md`
- Modify: `docs/BLUEPRINT/11_DEMO.md`
- Modify: `docs/BLUEPRINT/12_ROADMAP.md`

- [ ] **Step 1: Write the doc change list before editing**

Update the docs to reflect:
- model roster: `CatBoost`, `LightGBM`, `XGBoost`
- categorical-aware training input contract
- the fact that `category_id`, `category_code`, and `brand` are preserved as first-class features
- SHAP behavior for the new model stack
- testing expectations for categorical-aware preprocessing and train orchestration

- [ ] **Step 2: Patch `README.md` and `AGENTS.md`**

In `README.md`, update the current-state and quick-start wording so new maintainers can see that:
- Week 2 training is categorical-aware
- the three candidate models are `CatBoost`, `LightGBM`, `XGBoost`
- gold parquet inputs keep categorical columns

In `AGENTS.md`, update the repo rules/gotchas so future agents know:
- not to collapse the training frame into `float32` too early
- `catboost` is now part of the runtime set
- `RandomForest` is no longer a candidate model
- categorical handling should go through the shared preprocessing contract, not ad hoc encoding

- [ ] **Step 3: Patch the blueprint docs**

Update the blueprint pages that mention the model roster or explainability:
- `01_OVERVIEW.md` and `02_ARCHITECTURE.md` for the high-level model stack
- `03_FEATURES.md` for categorical feature semantics
- `04_PIPELINES.md` for the training pipeline steps
- `05_PROJECT_STRUCTURE.md` for the repo layout and model experiment notes
- `07_TESTING.md` for the new preprocessing and model-adapter tests
- `09_EXPLAINABILITY.md` for SHAP support on the new model stack
- `11_DEMO.md` for demo text that names the correct models
- `12_ROADMAP.md` for roadmap entries that mention the training milestone

- [ ] **Step 4: Run a docs consistency pass**

Use `rg` to confirm there are no stale references to the old `RandomForest` roster in the synced docs unless they are explicitly describing history.

```bash
rg -n "RandomForest|XGBoost|LightGBM|CatBoost|category_id|category_code|brand" README.md AGENTS.md docs/BLUEPRINT
```

- [ ] **Step 5: Commit**

```bash
git add README.md AGENTS.md docs/BLUEPRINT/01_OVERVIEW.md docs/BLUEPRINT/02_ARCHITECTURE.md docs/BLUEPRINT/03_FEATURES.md docs/BLUEPRINT/04_PIPELINES.md docs/BLUEPRINT/05_PROJECT_STRUCTURE.md docs/BLUEPRINT/07_TESTING.md docs/BLUEPRINT/09_EXPLAINABILITY.md docs/BLUEPRINT/11_DEMO.md docs/BLUEPRINT/12_ROADMAP.md
git commit -m "docs: sync categorical-aware sprint2b training contract"
```

---

## Task 6: Final verification and cleanup

**Files:**
- No new code files
- Verification only

- [ ] **Step 1: Run the focused test set**

Run:

```bash
pytest training/tests/test_categorical_features.py training/tests/test_train.py training/tests/test_explainability.py training/tests/test_model_validation.py training/tests/test_data_lineage.py -q
```

Expected: pass.

- [ ] **Step 2: Run the repo-level checks that cover the touched surface**

Run:

```bash
ruff check .
pytest training/tests -q
dvc dag
```

Expected:
- Ruff passes
- training tests pass
- DVC graph still shows `bronze -> silver -> session_split -> gold -> train`

- [ ] **Step 3: Inspect the final diff for stale contract drift**

Verify the diff does not leave mixed messaging between code and docs:
- no stale `RandomForest` as the active Sprint 2b candidate
- no prose that says gold features were collapsed into one float matrix
- no docs that describe categorical columns as dropped from training entirely

- [ ] **Step 4: Final commit or PR handoff**

If the implementation is being landed locally, do one final commit after verification. If it is being handed off for PR creation, keep the worktree clean and note the verification commands in the handoff message.

---

## Assumptions

- `category_id`, `category_code`, and `brand` are intentionally part of the training signal for all three candidate models.
- The existing gold schema stays unchanged; this is a training/preprocessing refactor, not a data-contract refactor.
- The implementation should preserve the current CLI and DVC stage shape unless a test shows a necessary change.
- Documentation sync is part of the definition of done for this change.
