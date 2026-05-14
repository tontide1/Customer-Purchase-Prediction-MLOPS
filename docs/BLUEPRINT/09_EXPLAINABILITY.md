# 9. Model Explainability

> **← Xem [8. Security](08_SECURITY.md)**
> **→ Xem [10. Performance](10_PERFORMANCE.md)**

> **Execution profile (local dev): `DEV_SMOKE`**
> - Train window (dev): `2019-10` -> `2019-10`
> - Replay window (dev): `2019-11` -> `2019-11`
> - Profile này chỉ để tăng tốc vòng lặp phát triển; canonical target-state windows trong blueprint vẫn giữ nguyên.

---

## 9.1. Tại sao cần Explainability?

Dự đoán "Session snapshot X có 85% khả năng mua trong 10 phút tới" là không đủ — cần giải thích **tại sao**. Điều này giúp:

* **Demo thuyết phục hơn:** Giảng viên thấy model không phải "hộp đen".
* **Debug model:** Phát hiện nếu model dựa vào feature không hợp lý.
* **Business value:** Trong thực tế, team marketing cần biết *tại sao* để hành động.

---

## 9.2. Giải pháp: SHAP (SHapley Additive exPlanations)

Sử dụng **`shap.TreeExplainer`** — được tối ưu cho **tất cả tree-based models** (CatBoost, LightGBM, XGBoost), **rất nhanh** (~1-5ms per prediction):

### Global Explanation (Tính 1 lần khi train)

* Chạy SHAP trên tập validation → Tạo **Feature Importance Bar Chart**.
* Lưu lên **MLflow artifacts** → Show trên Dashboard.
* Ví dụ output: `cart_to_view_ratio` đóng góp 35%, `session_duration` đóng góp 20%, ...

### Local Explanation (Tính real-time cho từng session snapshot)

* Endpoint `GET /api/v1/explain/{user_session}` trả về:

```json
{
  "user_session": "session_12345",
  "target_horizon_minutes": 10,
  "purchase_probability": 0.85,
  "prediction_time": "2026-04-12T10:30:05Z",
  "top_contributing_features": [
    {
      "feature": "cart_to_view_ratio",
      "value": 0.6,
      "shap_value": +0.32,
      "direction": "increases purchase probability"
    },
    {
      "feature": "session_duration_sec",
      "value": 420,
      "shap_value": +0.18,
      "direction": "increases purchase probability"
    },
    {
      "feature": "unique_products",
      "value": 2,
      "shap_value": -0.05,
      "direction": "decreases purchase probability"
    }
  ],
  "explanation_summary": "Session này có khả năng mua cao trong 10 phút tới vì: tỷ lệ cart/view cao (0.6), thời gian phiên đã kéo dài 7 phút và số sản phẩm đang tập trung."
}
```

---

## 9.3. Implementation Notes

* SHAP Explainer object được **pickle** và lưu lên MLflow cùng model.
* Khi FastAPI khởi động: load cả model + explainer vào memory. Sau đó **background thread** tự động poll MLflow mỗi 5 phút để hot-reload khi có model mới.
* `/predict` và `/explain` có failure contract khác nhau:
  * `/predict` có thể dùng degraded mode (fallback score) để giữ availability.
  * `/explain` không fallback thành prediction response; nếu explainer unavailable thì trả **HTTP 503**.

Ví dụ response khi `/explain` unavailable:

```json
{
  "error_code": "EXPLAINER_UNAVAILABLE",
  "message": "SHAP explainer is not loaded for the current production model.",
  "model_version": "v2"
}
```

### Model Hot-Reload (Zero-Downtime Model Update)

```python
# services/prediction-api/app/model_loader.py
import threading
import time
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

class ModelLoader:
    """Thread-safe model hot-reload from MLflow Registry."""

    def __init__(self, model_name: str, tracking_uri: str, reload_interval: int = 300):
        self.model_name = model_name
        self.reload_interval = reload_interval
        self._model = None
        self._explainer = None
        self._model_version: str | None = None
        self._lock = threading.Lock()
        self._client = MlflowClient(tracking_uri)

        # Initial load
        self._load_latest_model()

        # Start background polling thread
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _load_latest_model(self) -> None:
        """Load latest Production model from MLflow."""
        try:
            versions = self._client.get_latest_versions(
                self.model_name, stages=["Production"]
            )
            if not versions:
                logger.warning("No production model found in MLflow.")
                return

            latest = versions[0]
            if latest.version == self._model_version:
                return  # Already loaded

            logger.info(f"Loading model version {latest.version}...")
            model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/Production")

            # Load SHAP explainer from same run artifacts
            explainer_path = self._client.download_artifacts(
                latest.run_id, "shap_explainer.pkl"
            )
            import pickle
            with open(explainer_path, "rb") as f:
                explainer = pickle.load(f)

            # Thread-safe swap (atomic reference update)
            with self._lock:
                self._model = model
                self._explainer = explainer
                self._model_version = latest.version

            logger.info(f"✅ Model hot-reloaded: v{latest.version}")

        except Exception as e:
            logger.error(f"Model load failed: {e}. Keeping current model.")

    def _poll_loop(self) -> None:
        """Background loop to check for new model versions."""
        while True:
            time.sleep(self.reload_interval)
            self._load_latest_model()

    @property
    def model(self):
        with self._lock:
            return self._model

    @property
    def explainer(self):
        with self._lock:
            return self._explainer

    @property
    def version(self) -> str:
        with self._lock:
            return self._model_version or "unknown"
```

### Prediction Caching (Redis-based, Auto-Invalidation)

```python
# services/prediction-api/app/cache.py
import json
import redis
from loguru import logger

class PredictionCache:
    """Cache predictions in Redis with auto-invalidation."""

    def __init__(self, redis_client: redis.Redis, ttl: int = 30):
        self.redis = redis_client
        self.ttl = ttl

    def _key(self, user_session: str) -> str:
        return f"cache:predict:session:{user_session}"

    def get(self, user_session: str) -> dict | None:
        """Get cached prediction. Returns None on miss."""
        raw = self.redis.get(self._key(user_session))
        if raw:
            result = json.loads(raw)
            result["cached"] = True
            return result
        return None

    def set(self, user_session: str, prediction: dict) -> None:
        """Cache prediction with TTL (skip fallback responses)."""
        if prediction.get("prediction_mode") == "fallback":
            logger.info(
                f"Skip caching fallback prediction for session={user_session} "
                f"reason={prediction.get('fallback_reason')}"
            )
            return

        self.redis.setex(
            self._key(user_session), self.ttl, json.dumps(prediction)
        )

    def invalidate(self, user_session: str) -> None:
        """Delete cached prediction (called when features update)."""
        self.redis.delete(self._key(user_session))
```

> **Cache invalidation trong Stream Processor:** Khi Quix Streams ghi features mới cho session vào Redis, đồng thời gọi `DEL cache:predict:session:{user_session}` để đảm bảo lần predict tiếp theo dùng features mới nhất.

> **Fallback cache policy:** Response có `prediction_mode="fallback"` không được ghi vào cache, dù endpoint `/predict` vẫn trả HTTP 200.
