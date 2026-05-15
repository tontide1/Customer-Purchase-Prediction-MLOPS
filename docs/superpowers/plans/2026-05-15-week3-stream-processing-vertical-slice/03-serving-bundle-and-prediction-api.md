# Week 3 Serving Bundle And Prediction API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log a real MLflow serving bundle from training and expose authenticated predictions from Redis session state through FastAPI.

**Architecture:** This plan owns the serving boundary. Training logs the feature order, categorical maps, threshold, horizon, and model URI during the winner test-evaluation run; the API loads that bundle, assembles the same feature vector from Redis, and returns model or explicit fallback responses.

**Tech Stack:** Python 3.11, MLflow, pandas, FastAPI, Redis, Uvicorn, pytest, Docker.

---

### Task 7: Training Serving Bundle Artifacts

**Files:**
- Modify: `training/src/train.py`
- Test: `training/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Append to `training/tests/test_train.py`:

```python
def test_main_logs_serving_bundle_on_winner_test_run(gold_data, monkeypatch, tmp_path):
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("training.src.train.mlflow", fake_mlflow)
    monkeypatch.setattr("training.src.train.OPTUNA_SMOKE_TRIALS", 1)
    monkeypatch.chdir(tmp_path)
    _stub_shap_hooks(monkeypatch, Path(gold_data["train_path"]).with_name("shap.png"))

    def fake_train(*args, **kwargs):
        return _FakeModel(), {
            "pr_auc": 0.9,
            "average_precision": 0.9,
            "optimal_threshold": 0.42,
            "confusion_matrix": np.array([[1, 0], [0, 1]]),
        }

    monkeypatch.setattr("training.src.train.train_catboost_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_lightgbm_candidate", fake_train)
    monkeypatch.setattr("training.src.train.train_xgboost_candidate", fake_train)
    monkeypatch.setattr(
        "training.src.train.evaluate_winner_on_test",
        lambda *args, **kwargs: {
            "pr_auc": 0.8,
            "average_precision": 0.8,
            "confusion_matrix": np.array([[1, 0], [0, 1]]),
        },
    )

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
            "--device",
            "cpu",
            "--gpu-device-id",
            "0",
        ],
    )

    assert main() == 0
    test_run = fake_mlflow.runs[-1]
    logged_dicts = {args[1]: args[0] for args, _ in test_run["dicts"]}
    assert logged_dicts["serving/feature_column_order.json"] == [
        "total_views",
        "total_carts",
        "net_cart_count",
        "cart_to_view_ratio",
        "unique_categories",
        "unique_products",
        "session_duration_sec",
        "price",
        "category_id",
        "category_code",
        "brand",
        "event_type",
    ]
    assert logged_dicts["serving/threshold.json"] == {"optimal_threshold": 0.42}
    assert logged_dicts["serving/prediction_contract.json"] == {
        "prediction_horizon_minutes": 10,
        "response_contract_version": "v1",
    }
    assert logged_dicts["serving/categorical_encoding.json"]["missing_token"] == "__MISSING__"
    assert "category_maps" in logged_dicts["serving/categorical_encoding.json"]
    assert logged_dicts["serving/model_metadata.json"]["model_type"] == "catboost"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest training/tests/test_train.py::test_main_logs_serving_bundle_on_winner_test_run -q`

Expected: fail because no serving artifacts are logged.

- [ ] **Step 3: Add serving bundle helper**

In `training/src/train.py`, add near constants:

```python
RESPONSE_CONTRACT_VERSION = "v1"
SERVING_MODEL_ARTIFACT_PATH = "serving/model"
```

Add after `_log_model_artifact()`:

```python
def _log_serving_model_artifact(candidate_name: str, model) -> str:
    if candidate_name == "catboost":
        mlflow.catboost.log_model(model, SERVING_MODEL_ARTIFACT_PATH)
    else:
        mlflow.sklearn.log_model(model, SERVING_MODEL_ARTIFACT_PATH)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else ""
    return f"runs:/{run_id}/{SERVING_MODEL_ARTIFACT_PATH}"


def _log_serving_bundle(
    *,
    winner_name: str,
    winner_model,
    data: PreparedTrainingData,
    winner_threshold: float,
) -> None:
    model_uri = _log_serving_model_artifact(winner_name, winner_model)
    active_run = mlflow.active_run()
    run_id = active_run.info.run_id if active_run is not None else ""
    mlflow.log_dict(
        {
            "model_type": winner_name,
            "model_name": winner_name,
            "run_id": run_id,
            "model_uri": model_uri,
            "artifact_path": SERVING_MODEL_ARTIFACT_PATH,
        },
        "serving/model_metadata.json",
    )
    mlflow.log_dict(
        data.numeric_columns + data.categorical_columns,
        "serving/feature_column_order.json",
    )
    mlflow.log_dict(
        {
            "category_maps": data.categorical_artifacts.category_maps,
            "missing_token": data.categorical_artifacts.missing_token,
            "unknown_token": data.categorical_artifacts.unknown_token,
        },
        "serving/categorical_encoding.json",
    )
    mlflow.log_dict(
        {"optimal_threshold": float(winner_threshold)},
        "serving/threshold.json",
    )
    mlflow.log_dict(
        {
            "prediction_horizon_minutes": Config.PREDICTION_HORIZON_MINUTES,
            "response_contract_version": RESPONSE_CONTRACT_VERSION,
        },
        "serving/prediction_contract.json",
    )
```

Inside the existing winner test-evaluation run in `main()`, after `_log_winner_shap_artifacts(...)`, add:

```python
        _log_serving_bundle(
            winner_name=winner_name,
            winner_model=winner_data["model"],
            data=data,
            winner_threshold=winner_threshold,
        )
```

- [ ] **Step 4: Update fake MLflow support for active run**

In `training/tests/test_train.py`, update `_DummyRun.__enter__()`:

```python
    def __enter__(self):
        run = {
            "run_name": self.run_name,
            "metrics": [],
            "dicts": [],
            "texts": [],
        }
        self.mlflow.runs.append(run)
        self.mlflow._current_run = run
        self.info = SimpleNamespace(run_id=f"run-{len(self.mlflow.runs)}")
        self.mlflow._active_run = self
        return self
```

Update `_DummyRun.__exit__()`:

```python
    def __exit__(self, exc_type, exc, tb):
        self.mlflow._current_run = None
        self.mlflow._active_run = None
        return False
```

Update `_FakeMlflow.__init__()`:

```python
        self._active_run = None
```

Add `_FakeMlflow.active_run()`:

```python
    def active_run(self):
        return self._active_run
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest training/tests/test_train.py::test_main_logs_serving_bundle_on_winner_test_run -q`

Expected: pass.

- [ ] **Step 6: Run focused training tests**

Run: `pytest training/tests/test_train.py -q`

Expected: all `test_train.py` tests pass.

- [ ] **Step 7: Commit**

```bash
git add training/src/train.py training/tests/test_train.py
git commit -m "feat: log mlflow serving bundle"
```

### Task 8: Prediction API Bundle Loading And Feature Assembly

**Files:**
- Create: `services/prediction_api/__init__.py`
- Create: `services/prediction_api/bundle.py`
- Create: `services/prediction_api/features.py`
- Test: `services/tests/test_prediction_features.py`

- [ ] **Step 1: Write the failing tests**

Create `services/tests/test_prediction_features.py`:

```python
"""Tests for prediction API bundle and feature assembly."""

from __future__ import annotations

import pytest

from services.prediction_api.bundle import ServingBundle, validate_prediction_contract
from services.prediction_api.features import build_feature_row


class FakeRedis:
    def __init__(self):
        self.hashes = {
            "session:session-1": {
                "first_event_time": "2019-11-01T00:00:00",
                "last_event_time": "2019-11-01T00:02:00",
                "count_view": "2",
                "count_cart": "1",
                "count_remove_from_cart": "1",
                "latest_price": "0",
                "latest_category_id": "cat-id",
                "latest_category_code": "",
                "latest_brand": "",
                "latest_event_type": "cart",
            }
        }
        self.sets = {
            "session:session-1:products": {"p1", "p2"},
            "session:session-1:categories": {"cat-id"},
        }

    def hgetall(self, key):
        return self.hashes.get(key, {}).copy()

    def scard(self, key):
        return len(self.sets.get(key, set()))


def _bundle():
    return ServingBundle(
        model=None,
        model_uri="runs:/run-1/serving/model",
        model_version="run-1",
        feature_column_order=[
            "total_views",
            "total_carts",
            "net_cart_count",
            "cart_to_view_ratio",
            "unique_categories",
            "unique_products",
            "session_duration_sec",
            "price",
            "category_id",
            "category_code",
            "brand",
            "event_type",
        ],
        category_maps={
            "category_id": {"__MISSING__": 0, "__UNK__": 1, "cat-id": 2},
            "category_code": {"__MISSING__": 0, "__UNK__": 1},
            "brand": {"__MISSING__": 0, "__UNK__": 1},
            "event_type": {"__MISSING__": 0, "__UNK__": 1, "cart": 2},
        },
        missing_token="__MISSING__",
        unknown_token="__UNK__",
        threshold=0.5,
        prediction_horizon_minutes=10,
        response_contract_version="v1",
    )


def test_validate_prediction_contract_rejects_non_v1():
    with pytest.raises(ValueError, match="Unsupported response contract version"):
        validate_prediction_contract({"response_contract_version": "v2"})


def test_build_feature_row_matches_online_state_semantics():
    row = build_feature_row(FakeRedis(), "session-1", _bundle())

    assert row is not None
    assert row.iloc[0].to_dict() == {
        "total_views": 2,
        "total_carts": 1,
        "net_cart_count": 0,
        "cart_to_view_ratio": 0.5,
        "unique_categories": 1,
        "unique_products": 2,
        "session_duration_sec": 120.0,
        "price": 0.0,
        "category_id": 2,
        "category_code": 0,
        "brand": 0,
        "event_type": 2,
    }


def test_build_feature_row_returns_none_on_redis_miss():
    redis = FakeRedis()
    row = build_feature_row(redis, "missing-session", _bundle())

    assert row is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/tests/test_prediction_features.py -q`

Expected: fail with `ModuleNotFoundError` for `services.prediction_api`.

- [ ] **Step 3: Write bundle and feature implementation**

Create `services/prediction_api/__init__.py`:

```python
"""Prediction API service package."""
```

Create `services/prediction_api/bundle.py`:

```python
"""Serving bundle loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow

SUPPORTED_RESPONSE_CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class ServingBundle:
    model: Any
    model_uri: str
    model_version: str
    feature_column_order: list[str]
    category_maps: dict[str, dict[str, int]]
    missing_token: str
    unknown_token: str
    threshold: float
    prediction_horizon_minutes: int
    response_contract_version: str


def validate_prediction_contract(contract: dict[str, Any]) -> None:
    version = contract.get("response_contract_version")
    if version != SUPPORTED_RESPONSE_CONTRACT_VERSION:
        raise ValueError(f"Unsupported response contract version: {version}")


def load_serving_bundle(run_uri: str) -> ServingBundle:
    model_metadata = mlflow.artifacts.load_dict(f"{run_uri}/serving/model_metadata.json")
    feature_column_order = mlflow.artifacts.load_dict(f"{run_uri}/serving/feature_column_order.json")
    categorical_encoding = mlflow.artifacts.load_dict(f"{run_uri}/serving/categorical_encoding.json")
    threshold = mlflow.artifacts.load_dict(f"{run_uri}/serving/threshold.json")
    prediction_contract = mlflow.artifacts.load_dict(f"{run_uri}/serving/prediction_contract.json")
    validate_prediction_contract(prediction_contract)
    model = mlflow.pyfunc.load_model(model_metadata["model_uri"])
    return ServingBundle(
        model=model,
        model_uri=model_metadata["model_uri"],
        model_version=model_metadata["run_id"],
        feature_column_order=list(feature_column_order),
        category_maps=categorical_encoding["category_maps"],
        missing_token=categorical_encoding["missing_token"],
        unknown_token=categorical_encoding["unknown_token"],
        threshold=float(threshold["optimal_threshold"]),
        prediction_horizon_minutes=int(prediction_contract["prediction_horizon_minutes"]),
        response_contract_version=prediction_contract["response_contract_version"],
    )
```

Create `services/prediction_api/features.py`:

```python
"""Build model-ready feature rows from Redis online state."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from services.prediction_api.bundle import ServingBundle


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _encode(value: str, mapping: dict[str, int], *, missing_token: str, unknown_token: str) -> int:
    normalized = missing_token if value == "" else value
    return int(mapping.get(normalized, mapping[unknown_token]))


def build_feature_row(redis_client, user_session: str, bundle: ServingBundle) -> pd.DataFrame | None:
    hash_key = f"session:{user_session}"
    state = redis_client.hgetall(hash_key)
    if not state:
        return None

    first_time = dt.datetime.fromisoformat(_text(state["first_event_time"]))
    last_time = dt.datetime.fromisoformat(_text(state["last_event_time"]))
    total_views = int(_text(state.get("count_view", 0)))
    total_carts = int(_text(state.get("count_cart", 0)))
    total_removes = int(_text(state.get("count_remove_from_cart", 0)))
    cart_to_view_ratio = 0.0 if total_views == 0 else total_carts / total_views

    values = {
        "total_views": total_views,
        "total_carts": total_carts,
        "net_cart_count": total_carts - total_removes,
        "cart_to_view_ratio": cart_to_view_ratio,
        "unique_categories": redis_client.scard(f"{hash_key}:categories"),
        "unique_products": redis_client.scard(f"{hash_key}:products"),
        "session_duration_sec": (last_time - first_time).total_seconds(),
        "price": float(_text(state.get("latest_price", "0"))),
        "category_id": _encode(
            _text(state["latest_category_id"]),
            bundle.category_maps["category_id"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
        "category_code": _encode(
            _text(state.get("latest_category_code", "")),
            bundle.category_maps["category_code"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
        "brand": _encode(
            _text(state.get("latest_brand", "")),
            bundle.category_maps["brand"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
        "event_type": _encode(
            _text(state["latest_event_type"]),
            bundle.category_maps["event_type"],
            missing_token=bundle.missing_token,
            unknown_token=bundle.unknown_token,
        ),
    }
    return pd.DataFrame([{column: values[column] for column in bundle.feature_column_order}])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/tests/test_prediction_features.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add services/prediction_api/__init__.py services/prediction_api/bundle.py services/prediction_api/features.py services/tests/test_prediction_features.py
git commit -m "feat: assemble prediction features from redis"
```

### Task 9: Prediction API Endpoints And Fallbacks

**Files:**
- Create: `services/prediction_api/app.py`
- Create: `services/prediction_api/requirements.txt`
- Create: `services/prediction_api/Dockerfile`
- Test: `services/tests/test_prediction_api.py`

- [ ] **Step 1: Write the failing tests**

Create `services/tests/test_prediction_api.py`:

```python
"""Tests for prediction API endpoint contracts."""

from __future__ import annotations

from fastapi.testclient import TestClient

from services.prediction_api.app import create_app


class FakeRedisMiss:
    def hgetall(self, key):
        return {}


    def scard(self, key):
        return 0


def test_health_does_not_require_api_key():
    client = TestClient(create_app(api_key="secret", redis_client=FakeRedisMiss(), bundle_loader=lambda: None))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_requires_api_key():
    client = TestClient(create_app(api_key="secret", redis_client=FakeRedisMiss(), bundle_loader=lambda: None))

    response = client.get("/api/v1/predict/session-1")

    assert response.status_code == 401


def test_predict_rejects_invalid_session():
    client = TestClient(create_app(api_key="secret", redis_client=FakeRedisMiss(), bundle_loader=lambda: None))

    response = client.get("/api/v1/predict/bad session", headers={"X-API-Key": "secret"})

    assert response.status_code == 422


def test_predict_redis_miss_returns_explicit_fallback():
    client = TestClient(create_app(api_key="secret", redis_client=FakeRedisMiss(), bundle_loader=lambda: None))

    response = client.get("/api/v1/predict/session-1", headers={"X-API-Key": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["purchase_probability"] == 0.5
    assert body["prediction_horizon_minutes"] == 10
    assert body["prediction_mode"] == "fallback"
    assert body["fallback_reason"] == "redis_miss"
    assert body["cached"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest services/tests/test_prediction_api.py -q`

Expected: fail with `ModuleNotFoundError` for `services.prediction_api.app`.

- [ ] **Step 3: Write API implementation**

Create `services/prediction_api/app.py`:

```python
"""FastAPI prediction service for Week 3."""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Callable

import redis
from fastapi import Depends, FastAPI, Header, HTTPException

from services.prediction_api.bundle import ServingBundle, load_serving_bundle
from services.prediction_api.features import build_feature_row

SESSION_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+$")


def _unauthorized() -> HTTPException:
    return HTTPException(status_code=401, detail="Invalid API key")


def _prediction_time() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat()


def _fallback_response(reason: str, horizon_minutes: int = 10) -> dict:
    return {
        "purchase_probability": 0.5,
        "prediction_time": _prediction_time(),
        "prediction_horizon_minutes": horizon_minutes,
        "model_uri": None,
        "model_version": None,
        "prediction_mode": "fallback",
        "fallback_reason": reason,
        "cached": False,
    }


def create_app(
    *,
    api_key: str,
    redis_client,
    bundle_loader: Callable[[], ServingBundle | None],
) -> FastAPI:
    app = FastAPI()

    def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if x_api_key != api_key:
            raise _unauthorized()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/predict/{user_session}")
    def predict(user_session: str, _: None = Depends(require_api_key)) -> dict:
        if not SESSION_PATTERN.fullmatch(user_session):
            raise HTTPException(status_code=422, detail="Invalid user_session")

        try:
            bundle = bundle_loader()
        except Exception:
            return _fallback_response("model_unavailable")

        if bundle is None:
            horizon = 10
        else:
            horizon = bundle.prediction_horizon_minutes

        try:
            row = build_feature_row(redis_client, user_session, bundle) if bundle else None
        except Exception:
            return _fallback_response("model_unavailable", horizon_minutes=horizon)

        if row is None:
            return _fallback_response("redis_miss", horizon_minutes=horizon)

        try:
            proba = bundle.model.predict(row)
            score = float(proba[0])
        except Exception:
            return _fallback_response("model_unavailable", horizon_minutes=horizon)

        return {
            "purchase_probability": score,
            "prediction_time": _prediction_time(),
            "prediction_horizon_minutes": horizon,
            "model_uri": bundle.model_uri,
            "model_version": bundle.model_version,
            "prediction_mode": "model",
            "fallback_reason": None,
            "cached": False,
        }

    return app


def create_runtime_app() -> FastAPI:
    api_key = os.environ["API_KEY"]
    redis_client = redis.Redis.from_url(
        os.getenv("REDIS_URL", "redis://redis:6379/0"),
        decode_responses=True,
    )
    bundle_uri = os.environ["MLFLOW_SERVING_BUNDLE_URI"]
    return create_app(
        api_key=api_key,
        redis_client=redis_client,
        bundle_loader=lambda: load_serving_bundle(bundle_uri),
    )
```

Create `services/prediction_api/requirements.txt`:

```text
fastapi>=0.115.0
mlflow>=2.8.0
pandas>=2.2.0
redis>=5.0.0
uvicorn[standard]>=0.30.0
```

Create `services/prediction_api/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY training ./training
COPY services ./services
COPY services/prediction_api/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -e . && python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["uvicorn", "services.prediction_api.app:create_runtime_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest services/tests/test_prediction_api.py -q`

Expected: all tests pass.

- [ ] **Step 5: Add model-backed response test**

Append to `services/tests/test_prediction_api.py`:

```python
def test_predict_model_backed_response_shape():
    from services.prediction_api.bundle import ServingBundle

    class FakeRedisHit:
        def hgetall(self, key):
            return {
                "first_event_time": "2019-11-01T00:00:00",
                "last_event_time": "2019-11-01T00:01:00",
                "count_view": "1",
                "count_cart": "1",
                "count_remove_from_cart": "0",
                "latest_price": "12.5",
                "latest_category_id": "cat-id",
                "latest_category_code": "cat-code",
                "latest_brand": "brand-a",
                "latest_event_type": "cart",
            }

        def scard(self, key):
            return 1

    class FakeModel:
        def predict(self, row):
            return [0.73]

    bundle = ServingBundle(
        model=FakeModel(),
        model_uri="runs:/run-1/serving/model",
        model_version="run-1",
        feature_column_order=[
            "total_views",
            "total_carts",
            "net_cart_count",
            "cart_to_view_ratio",
            "unique_categories",
            "unique_products",
            "session_duration_sec",
            "price",
            "category_id",
            "category_code",
            "brand",
            "event_type",
        ],
        category_maps={
            "category_id": {"__MISSING__": 0, "__UNK__": 1, "cat-id": 2},
            "category_code": {"__MISSING__": 0, "__UNK__": 1, "cat-code": 2},
            "brand": {"__MISSING__": 0, "__UNK__": 1, "brand-a": 2},
            "event_type": {"__MISSING__": 0, "__UNK__": 1, "cart": 2},
        },
        missing_token="__MISSING__",
        unknown_token="__UNK__",
        threshold=0.5,
        prediction_horizon_minutes=10,
        response_contract_version="v1",
    )
    client = TestClient(
        create_app(
            api_key="secret",
            redis_client=FakeRedisHit(),
            bundle_loader=lambda: bundle,
        )
    )

    response = client.get("/api/v1/predict/session-1", headers={"X-API-Key": "secret"})

    assert response.status_code == 200
    body = response.json()
    assert body["purchase_probability"] == 0.73
    assert body["prediction_mode"] == "model"
    assert body["fallback_reason"] is None
    assert body["model_uri"] == "runs:/run-1/serving/model"
    assert body["model_version"] == "run-1"
    assert body["cached"] is False
```

- [ ] **Step 6: Run API tests**

Run: `pytest services/tests/test_prediction_api.py -q`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add services/prediction_api/app.py services/prediction_api/requirements.txt services/prediction_api/Dockerfile services/tests/test_prediction_api.py
git commit -m "feat: add authenticated prediction api"
```

