# 10. Testing Strategy

> **← Xem [6. Error Handling](06_ERROR_HANDLING.md)**  
> **→ Xem [8. Security](08_SECURITY.md)**

---

## 10.1. Unit Tests (Pytest)

* **Feature Engineering:** Test snapshot builder và từng function tạo feature (input → expected output), đảm bảo chỉ dùng past events tại thời điểm `t`.
* **Labeling Logic:** Test `purchase within next 10 minutes` được gán đúng cho cùng `user_session`.
* **Prediction API:** Test endpoint `/api/v1/predict/{user_session}` với mock Redis và mock model.
* **Explain API:** Test endpoint `/api/v1/explain/{user_session}` trả về đúng format SHAP response.
* **Schema Validation:** Test Pydantic schemas reject dữ liệu sai format.
* **Timestamp Semantics:** Test simulator/processor preserve `source_event_time` và gắn `replay_time` đúng cách.
* **Security:** Test request thiếu API Key → bị reject 401. Test rate limit → bị reject 429.
* **Event Deduplication:** Test `event_id` generation theo công thức canonical `hash(f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}")` và deduplicate logic. Gửi trùng event → verify không cập nhật state nhưng vẫn log audit.
* **Late Event Handling:** Test policy cho out-of-order events. Gửi event với `source_event_time` trễ hơn `source_last_event_time` - ngưỡng → verify không cập nhật online state, gửi vào `late_events`.
* **Manual Event Handling:** Test Streamlit manual events có `source = "manual"` và `source_event_time = replay_time_now`.
* **Exact Count Parity:** Test Redis Set counts khớp chính xác với offline exact counts cho `unique_products`, `unique_categories`.
* **Bronze Chunked Ingestion Contract:** Test raw source pool nhiều file vẫn được xử lý theo chunk, không yêu cầu load toàn bộ dataset vào một DataFrame duy nhất.
* **Bronze Row-Count Parity:** Test tổng số dòng valid/rejected sau bronze vẫn đúng khi input trải trên nhiều raw files.
* **Multi-File Timestamp Preservation:** Test nhiều raw files vẫn preserve đúng `source_event_time` sau bronze materialization.
* **Cross-Month Session Boundary:** Test session kéo qua ranh giới tháng vẫn được giữ nguyên một `user_session` logic ở downstream split stage.
* **Split Map Disjointness:** Test `session_split_map.parquet` luôn đảm bảo train/val/test disjoint theo `user_session`.
* **Window Isolation:** Test training window và replay/demo window không bị trộn dữ liệu.
* **Materialization Strategy Invariance:** Test thay đổi cách materialize bronze/silver không làm đổi exact counts và downstream labeling semantics.

---

## 10.2. Model Validation Tests

Test **Model Validation Gate** đảm bảo model kém không được deploy:

```python
# training/tests/test_model_validation.py
import pytest
from unittest.mock import MagicMock, patch
from training.src.model_validation import validate_model

class TestModelValidationGate:
    """Test Model Validation Gate logic."""

    def test_first_deployment_auto_passes(self):
        """No production model exists → gate should auto-pass."""
        with patch("training.src.model_validation.MlflowClient") as mock_client:
            mock_client().get_latest_versions.return_value = []
            assert validate_model(new_pr_auc=0.75, model_name="test-model") is True

    def test_new_model_beats_production(self):
        """New model has higher PR-AUC → gate should pass."""
        with patch("training.src.model_validation.MlflowClient") as mock_client:
            mock_version = MagicMock(run_id="run-123", version="1")
            mock_client().get_latest_versions.return_value = [mock_version]
            mock_run = MagicMock()
            mock_run.data.metrics = {"pr_auc": 0.80}
            mock_client().get_run.return_value = mock_run
            assert validate_model(new_pr_auc=0.85, model_name="test-model") is True

    def test_new_model_worse_than_production(self):
        """New model has lower PR-AUC → gate should FAIL."""
        with patch("training.src.model_validation.MlflowClient") as mock_client:
            mock_version = MagicMock(run_id="run-123", version="1")
            mock_client().get_latest_versions.return_value = [mock_version]
            mock_run = MagicMock()
            mock_run.data.metrics = {"pr_auc": 0.85}
            mock_client().get_run.return_value = mock_run
            assert validate_model(new_pr_auc=0.80, model_name="test-model") is False

    def test_below_minimum_threshold(self):
        """PR-AUC below absolute minimum → gate should FAIL regardless."""
        assert validate_model(new_pr_auc=0.5, model_name="test-model", min_threshold=0.7) is False

    def test_mlflow_unreachable_fails_closed_after_first_deployment(self):
        """MLflow connection error after first deployment → gate should FAIL (fail-closed)."""
        with patch("training.src.model_validation.MlflowClient") as mock_client:
            mock_client().get_latest_versions.side_effect = Exception("Connection refused")
            assert validate_model(new_pr_auc=0.80, model_name="test-model") is False

    def test_mlflow_unreachable_can_pass_with_manual_override(self):
        """Manual override can bypass fail-closed gate with explicit intent."""
        with patch("training.src.model_validation.MlflowClient") as mock_client:
            mock_client().get_latest_versions.side_effect = Exception("Connection refused")
            assert validate_model(
                new_pr_auc=0.80,
                model_name="test-model",
                manual_override=True,
                override_by="mlops-admin",
                override_reason="registry outage",
                override_time="2026-04-13T09:00:00Z"
            ) is True

    def test_manual_override_requires_audit_fields(self):
        """Manual override must include override_by, override_reason, override_time."""
        with patch("training.src.model_validation.MlflowClient") as mock_client:
            mock_client().get_latest_versions.side_effect = Exception("Connection refused")
            assert validate_model(
                new_pr_auc=0.80,
                model_name="test-model",
                manual_override=True,
                override_by="mlops-admin",
                override_reason="registry outage",
                override_time=None,
            ) is False

    def test_session_boundary_split_has_no_overlap(self):
        """A user_session must belong to exactly one split."""
        train_sessions = {"s1", "s2"}
        val_sessions = {"s3"}
        test_sessions = {"s4", "s5"}

        assert train_sessions.isdisjoint(val_sessions)
        assert train_sessions.isdisjoint(test_sessions)
        assert val_sessions.isdisjoint(test_sessions)
```

---

## 10.3. Model Hot-Reload Tests

Test **ModelLoader** đảm bảo hot-reload hoạt động đúng và thread-safe:

```python
# services/prediction-api/tests/test_model_loader.py
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from app.model_loader import ModelLoader

class TestModelLoader:
    """Test model hot-reload logic."""

    @patch("app.model_loader.mlflow")
    @patch("app.model_loader.MlflowClient")
    def test_initial_load_success(self, mock_client_cls, mock_mlflow):
        """ModelLoader should load model on initialization."""
        mock_version = MagicMock(version="1", run_id="run-1")
        mock_client_cls().get_latest_versions.return_value = [mock_version]
        mock_mlflow.pyfunc.load_model.return_value = MagicMock()
        mock_client_cls().download_artifacts.return_value = "/tmp/explainer.pkl"

        with patch("builtins.open"), patch("pickle.load", return_value=MagicMock()):
            loader = ModelLoader("test-model", "http://mlflow:5000", reload_interval=9999)
            assert loader.version == "1"
            assert loader.model is not None

    @patch("app.model_loader.mlflow")
    @patch("app.model_loader.MlflowClient")
    def test_no_production_model(self, mock_client_cls, mock_mlflow):
        """No production model → model should be None, no crash."""
        mock_client_cls().get_latest_versions.return_value = []
        loader = ModelLoader("test-model", "http://mlflow:5000", reload_interval=9999)
        assert loader.model is None
        assert loader.version == "unknown"

    @patch("app.model_loader.mlflow")
    @patch("app.model_loader.MlflowClient")
    def test_skip_reload_same_version(self, mock_client_cls, mock_mlflow):
        """Same model version → should skip reload."""
        mock_version = MagicMock(version="1", run_id="run-1")
        mock_client_cls().get_latest_versions.return_value = [mock_version]
        mock_mlflow.pyfunc.load_model.return_value = MagicMock()
        mock_client_cls().download_artifacts.return_value = "/tmp/explainer.pkl"

        with patch("builtins.open"), patch("pickle.load", return_value=MagicMock()):
            loader = ModelLoader("test-model", "http://mlflow:5000", reload_interval=9999)
            # Reset call count after initial load
            mock_mlflow.pyfunc.load_model.reset_mock()
            loader._load_latest_model()  # Same version → should skip
            mock_mlflow.pyfunc.load_model.assert_not_called()

    @patch("app.model_loader.mlflow")
    @patch("app.model_loader.MlflowClient")
    def test_load_failure_keeps_current_model(self, mock_client_cls, mock_mlflow):
        """Load failure → should keep current model, not crash."""
        mock_version = MagicMock(version="1", run_id="run-1")
        mock_client_cls().get_latest_versions.return_value = [mock_version]
        mock_mlflow.pyfunc.load_model.return_value = MagicMock()
        mock_client_cls().download_artifacts.return_value = "/tmp/explainer.pkl"

        with patch("builtins.open"), patch("pickle.load", return_value=MagicMock()):
            loader = ModelLoader("test-model", "http://mlflow:5000", reload_interval=9999)
            assert loader.version == "1"

        # Simulate failure on next reload
        mock_version_2 = MagicMock(version="2", run_id="run-2")
        mock_client_cls().get_latest_versions.return_value = [mock_version_2]
        mock_mlflow.pyfunc.load_model.side_effect = Exception("Network error")
        loader._load_latest_model()
        assert loader.version == "1"  # Still the old version
```

---

## 10.4. Prediction Cache Tests

Test **PredictionCache** đảm bảo cache hit/miss/invalidation hoạt động đúng:

```python
# services/prediction-api/tests/test_cache.py
import pytest
import json
from unittest.mock import MagicMock
from app.cache import PredictionCache

class TestPredictionCache:
    """Test prediction caching logic."""

    def setup_method(self):
        self.mock_redis = MagicMock()
        self.cache = PredictionCache(self.mock_redis, ttl=30)

    def test_cache_miss_returns_none(self):
        """Cache miss → should return None."""
        self.mock_redis.get.return_value = None
        result = self.cache.get("session_123")
        assert result is None
        self.mock_redis.get.assert_called_once_with("cache:predict:session:session_123")

    def test_cache_hit_returns_result_with_cached_flag(self):
        """Cache hit → should return result with cached=True."""
        cached_data = {"purchase_probability": 0.85, "model_version": "v1"}
        self.mock_redis.get.return_value = json.dumps(cached_data)
        result = self.cache.get("session_123")
        assert result["cached"] is True
        assert result["purchase_probability"] == 0.85

    def test_set_stores_with_ttl(self):
        """Cache set → should store with correct TTL."""
        prediction = {"purchase_probability": 0.85}
        self.cache.set("session_123", prediction)
        self.mock_redis.setex.assert_called_once_with(
            "cache:predict:session:session_123", 30, json.dumps(prediction)
        )

    def test_fallback_prediction_is_not_cached(self):
        """Fallback response must not be stored in cache."""
        fallback_prediction = {
            "purchase_probability": 0.5,
            "prediction_mode": "fallback",
            "fallback_reason": "redis_miss",
        }
        self.cache.set("session_123", fallback_prediction)
        self.mock_redis.setex.assert_not_called()

    def test_invalidate_deletes_key(self):
        """Cache invalidate → should delete the key."""
        self.cache.invalidate("session_123")
        self.mock_redis.delete.assert_called_once_with("cache:predict:session:session_123")

    def test_key_format(self):
        """Cache key → should follow `cache:predict:session:{user_session}` pattern."""
        assert self.cache._key("session_123") == "cache:predict:session:session_123"
        assert self.cache._key("DEMO_SESSION") == "cache:predict:session:DEMO_SESSION"
```

---

## 10.5. Integration Tests

* **Raw Pool -> Bronze Dataset:** Gửi nhiều raw files tháng vào pipeline, verify `event_time` được parse thành `source_event_time` và output được materialize vào `data/bronze/` dưới dạng dataset directory.
* **Bronze Dataset -> Silver Dataset:** Verify clean/null/invalid/sort logic tạo `data/silver/` đúng và deterministic khi input là bronze dataset nhiều partitions/files.
* **Silver -> Session Split:** Verify session index được build toàn cục trên training window và cùng một `user_session` chỉ nằm trong một split.
* **Cross-Month Session Split:** Verify session đi qua ranh giới tháng vẫn thuộc đúng một split duy nhất.
* **Silver -> Gold:** Verify snapshot dataset sinh đúng features + label 10 phút tới theo split assignment.
* **Window Isolation Contract:** Verify replay/demo artifacts không bị trộn với training artifacts.
* **Kafka → Processor → Redis:** Gửi event vào Kafka, verify feature của đúng `user_session` được cập nhật đúng trong Redis.
* **API → Redis → Model:** Gọi API theo `user_session`, verify response có đúng format, score hợp lệ, và có `prediction_time`.
* **Predict Fallback Contract:** Khi Redis miss/model unavailable, `/predict` vẫn trả 200 với `prediction_mode="fallback"` và `fallback_reason` hợp lệ.
* **Explain Unavailable Contract:** Khi explainer chưa load/lỗi, `/explain` phải trả HTTP 503 với `error_code="EXPLAINER_UNAVAILABLE"`.
* **Cache Invalidation E2E:** Gửi event mới → Verify `cache:predict:session:{user_session}` bị xóa → Gọi predict → Verify response có `cached: false`.
* **Fallback Non-Cache E2E:** Trigger fallback 2 lần liên tiếp cho cùng session → verify response vẫn `cached: false` và không có cache hit từ fallback response.
* **Fallback Metrics Filter:** Verify online evaluation chỉ tính các bản ghi `prediction_mode='model'`; fallback predictions bị loại khỏi PR-AUC/F1 batch.
* **Evaluation Mode Separation:** Verify metrics/logging được tách theo `evaluation_mode` (`demo_replay` vs `offline_backfill`), không bị merge vào cùng series.
* **Event Deduplication E2E:** Gửi cùng event 2 lần → Verify Redis state chỉ được cập nhật 1 lần, event thứ 2 vào audit log nhưng không thay đổi feature.
* **Late Event E2E:** Gửi event với `source_event_time` trễ hơn ngưỡng → Verify không cập nhật Redis features, gửi vào `late_events` topic.
* **Manual Event E2E:** Gửi event từ Streamlit → Verify `source = "manual"` được ghi đúng, state cập nhật bình thường.
* **Exact Count Parity E2E:** So sánh counts từ Redis Set (`SCARD`) với offline exact counts trên cùng session → verify khớp chính xác.

---

## 10.6. CI Pipeline (GitHub Actions)

> Example target-state CI workflow; illustrative, không đảm bảo chạy ngay trong current repository state.

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install ruff pytest pytest-cov
          pip install -r services/prediction-api/requirements.txt
          pip install -r training/requirements.txt
      - name: Lint
        run: ruff check .
      - name: Unit Tests (API)
        run: pytest services/prediction-api/tests/ -v --cov=app --cov-report=term-missing
      - name: Unit Tests (Training)
        run: pytest training/tests/ -v --cov=training/src --cov-report=term-missing
      - name: Check Coverage Threshold
        run: pytest services/prediction-api/tests/ training/tests/ --cov --cov-fail-under=70
```

**Test coverage target:** ≥ 70% cho cả `services/prediction-api/app/` và `training/src/`.
