# 6. Chi tiết các Pipelines (Data Flow Spec)

> **← Xem [3. Features](03_FEATURES.md)**  
> **→ Xem [5. Project Structure](05_PROJECT_STRUCTURE.md)**

---

## Pipeline A: Training Pipeline (Offline — Chạy 1 lần hoặc khi retrain)

```
data/raw/<dataset-file>.csv → Bronze parse → Silver clean/sort → Session-boundary split → Gold snapshots
    → Train → Evaluate → SHAP Analysis
    → Validation Gate (fail-closed; first deploy auto-pass) → Register to MLflow
```

> **Artifact reproducibility:** `raw/bronze/silver/gold` artifacts được version bằng DVC; file thực nằm trên MinIO/S3 remote. Mọi lần train/retrain phải trace được về DVC revision.

**Chi tiết từng bước:**

1. **Load Raw:** Đọc từ `data/raw/<dataset-file>.csv` qua config `raw_data_path`, validate schema bằng Pydantic.
2. **Bronze Parse:** Parse raw CSV, rename `event_time -> source_event_time`, và ghi Parquet vào `data/bronze/`.
3. **Data Lineage Metadata:** Ngay sau bronze parse, ghi metadata của dataset lên MLflow để đảm bảo **reproducibility** và **traceability**:
    * `raw_file_md5`: MD5 checksum ở mức byte của file CSV gốc.
    * `logical_dataset_hash`: fingerprint của dataset sau khi load vào DataFrame (phát hiện thay đổi nội dung bản ghi).
    * `row_count`: Tổng số dòng trước/sau khi clean.
    * `date_range`: Khoảng thời gian (`min(source_event_time)` → `max(source_event_time)`) của dữ liệu gốc.
    * `data_source`: Đường dẫn raw file hoặc query PostgreSQL đã dùng.
    * Metadata này được log bằng `mlflow.log_params()` ngay đầu experiment run.

   ```python
    # training/src/data_lineage.py
     import hashlib
     import mlflow
     import pandas as pd

     def _file_md5(path: str) -> str:
         with open(path, "rb") as f:
             return hashlib.md5(f.read()).hexdigest()

     def _logical_dataset_hash(df: pd.DataFrame) -> str:
         return hashlib.md5(
             pd.util.hash_pandas_object(df, index=True).values.tobytes()
         ).hexdigest()

     def log_data_lineage(df: pd.DataFrame, data_source: str, raw_file_path: str) -> dict:
         """Log raw-file checksum + logical dataset fingerprint to MLflow."""
         lineage = {
             "data_source": data_source,
             "raw_file_md5": _file_md5(raw_file_path),
             "logical_dataset_hash": _logical_dataset_hash(df),
             "row_count_raw": len(df),
             "date_range_start": str(df["source_event_time"].min()),
             "date_range_end": str(df["source_event_time"].max()),
             "unique_users": df["user_id"].nunique(),
             "unique_sessions": df["user_session"].nunique(),
         }
         mlflow.log_params({f"data_{k}": v for k, v in lineage.items()})
         return lineage
    ```

4. **Silver Clean & Sort:**
    * Loại bỏ dòng thiếu `user_id`, `user_session`, `event_type`.
    * Loại bỏ dòng có `price <= 0` hoặc `price > 99th percentile`.
    * Xử lý null `category_code`: dùng `category_id` làm fallback để đảm bảo `unique_categories` không bị undercount (accessories không có `category_code`).
    * Ghi chú: `brand` null là expected — không loại bỏ dòng, dùng như signal cho `has_brand_info` (optional, xem mục 5.1).
    * Log số dòng bị loại và lý do.
    * Log `row_count_cleaned` lên MLflow sau khi clean.
    * Sort deterministic theo `user_session` + `source_event_time`.
    * Ghi Parquet vào `data/silver/`.
5. **Session Index & Split Assignment:**
    * Build session index từ silver layer với `user_session`, `session_start_time = min(source_event_time)`, `session_end_time = max(source_event_time)`.
    * Split **theo `user_session` boundary** ở mức session index, không split trên snapshot rows.
    * Assert một `user_session` chỉ thuộc đúng một split.
    * Persist split assignment để dùng lại khi generate gold snapshots.
6. **Gold Snapshot Generation & Feature Engineering:**
    * Sort events theo `source_event_time` trong từng `user_session`.
    * Tại mỗi thời điểm `t`, tạo 1 snapshot row chỉ dùng các event trước đó hoặc đúng tại `t`.
    * Tính các feature cumulative như mục 5.1 (bao gồm xử lý `remove_from_cart` events để tính `total_removes`, `net_cart_count`, `cart_remove_rate`).
    * Mục tiêu là để offline training nhìn thấy đúng loại state mà online inference sẽ nhận được ở Redis.
    * Ghi output theo split vào `data/gold/`.
7. **Labeling:** Với mỗi snapshot time `t`, label = `1` nếu cùng `user_session` có ít nhất 1 event `purchase` trong khoảng `(t, t + 10 phút]`, ngược lại `0`.
    * **Multiple purchases per session:** Dataset cho phép nhiều `purchase` events trong 1 session (= 1 đơn hàng duy nhất, nhiều items). Với target 10 phút tới, chỉ cần kiểm tra có tồn tại ít nhất 1 purchase trong horizon tương lai của snapshot.
8. **Handle Class Imbalance:** Tỷ lệ purchase thường chỉ ~2-5% trong eCommerce. Mỗi model có cách xử lý riêng:
   * **XGBoost:** Sử dụng `scale_pos_weight` (tỷ lệ negative/positive).
   * **LightGBM:** Sử dụng `is_unbalance=True` hoặc `scale_pos_weight`.
   * **Random Forest:** Sử dụng `class_weight='balanced'`.
   * So sánh với SMOTE trong experiment log.
9. **Train Multi-Model:** Train **3 models** song song, mỗi model có **experiment run riêng** trên MLflow:
   * **Model 1 — XGBoost:** Binary Classifier với hyperparameter tuning (Optuna, 50 trials).
   * **Model 2 — LightGBM:** Binary Classifier với hyperparameter tuning (Optuna, 50 trials). Thường nhanh hơn XGBoost trên large datasets.
   * **Model 3 — Random Forest:** Binary Classifier với hyperparameter tuning (Optuna, 50 trials). Ensemble learning kinh điển, ít bị overfit.
   * Mỗi model được log đầy đủ metrics (PR-AUC, F1, Precision, Recall) + hyperparameters lên MLflow.
   * **MLflow Experiment Tracking:** Sử dụng cùng 1 experiment name, mỗi model là 1 run → dễ dàng compare trên MLflow UI.
10. **Model Comparison & Auto-Selection:**
   * So sánh **PR-AUC** của 3 models trên **Validation Set**.
   * **Tự động chọn best model** (model có PR-AUC cao nhất) → Tiếp tục vào Validation Gate.
   * Log kết quả comparison (tên model, PR-AUC của từng model, model được chọn) lên MLflow.
   * **MLflow UI:** Giảng viên có thể nhìn thấy bảng so sánh 3 models ngay trên giao diện.
11. **Evaluate trên Test Set** (chỉ cho **best model**):
   * Metrics chính: **PR-AUC** cho target **purchase trong 10 phút tới** (phù hợp hơn ROC-AUC cho imbalanced data).
   * Metrics phụ: F1-Score, Precision, Recall, Confusion Matrix.
   * **Threshold tuning:** Chọn threshold tối ưu dựa trên Precision-Recall curve.
12. **SHAP Analysis (Model Explainability):**

* Tính **Global Feature Importance** bằng `shap.TreeExplainer` trên tập validation (tương thích với cả 3 tree-based models).
* Lưu SHAP summary plot (bar chart top features) lên MLflow artifacts.
* Lưu SHAP explainer object (pickle) để phục vụ real-time explanation qua API.

13. **Model Validation Gate (Quality Gate):**
    * Trước khi register, so sánh **PR-AUC của best model mới** với **model đang production** (lấy từ MLflow Model Registry, stage `Production`).
    * **Quy tắc:**
      * Nếu **chưa có model production** (lần train đầu tiên) → Tự động pass gate.
      * Nếu không truy cập được MLflow Model Registry hoặc không đọc được metrics model production → ❌ **Fail-closed** (không register model mới).
      * Nếu model mới **PR-AUC ≥ model cũ** → ✅ Pass → Tiến hành register.
      * Nếu model mới **PR-AUC < model cũ** → ❌ Fail → **Không register**, log warning, giữ model cũ.
    * Gate cũng kiểm tra **minimum threshold**: PR-AUC phải ≥ 0.7 (configurable) bất kể so sánh với model cũ.
    * Chỉ được bypass fail-closed bằng `manual_override=true` kèm audit fields bắt buộc (`override_by`, `override_reason`, `override_time`).
    * Kết quả gate (pass/fail, metrics comparison) được log lên MLflow.

    ```python
    # training/src/model_validation.py
    import mlflow
    from mlflow.tracking import MlflowClient
    from loguru import logger

    def validate_model(
        new_pr_auc: float,
        model_name: str,
        min_threshold: float = 0.7,
        manual_override: bool = False,
        override_by: str | None = None,
        override_reason: str | None = None,
        override_time: str | None = None,
    ) -> bool:
        """Model Validation Gate: fail-closed except first deployment."""
        client = MlflowClient()

        # Check minimum threshold
        if new_pr_auc < min_threshold:
            logger.warning(
                f"❌ Gate FAILED: PR-AUC {new_pr_auc:.4f} below minimum {min_threshold}"
            )
            mlflow.log_metric("validation_gate_passed", 0)
            return False

        # Get current production model
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if not prod_versions:
                logger.info("No production model found. First deployment — gate auto-passed.")
                mlflow.log_metric("validation_gate_passed", 1)
                return True

            prod_run = client.get_run(prod_versions[0].run_id)
            if "pr_auc" not in prod_run.data.metrics:
                raise RuntimeError("Production model missing pr_auc metric")

            prod_pr_auc = float(prod_run.data.metrics["pr_auc"])

            mlflow.log_metric("prod_model_pr_auc", prod_pr_auc)
            mlflow.log_metric("pr_auc_improvement", new_pr_auc - prod_pr_auc)

            if new_pr_auc >= prod_pr_auc:
                logger.info(
                    f"✅ Gate PASSED: new={new_pr_auc:.4f} >= prod={prod_pr_auc:.4f} "
                    f"(+{new_pr_auc - prod_pr_auc:.4f})"
                )
                mlflow.log_metric("validation_gate_passed", 1)
                return True
            else:
                logger.warning(
                    f"❌ Gate FAILED: new={new_pr_auc:.4f} < prod={prod_pr_auc:.4f} "
                    f"({new_pr_auc - prod_pr_auc:.4f})"
                )
                mlflow.log_metric("validation_gate_passed", 0)
                return False

        except Exception as e:
            logger.error(f"Gate fail-closed: cannot validate against production model: {e}")
            if manual_override:
                if not all([override_by, override_reason, override_time]):
                    logger.error(
                        "Manual override rejected: missing required audit fields "
                        "(override_by, override_reason, override_time)."
                    )
                    mlflow.log_metric("validation_gate_manual_override", 0)
                    mlflow.log_metric("validation_gate_passed", 0)
                    return False

                logger.warning(
                    "Manual override enabled. Bypassing fail-closed gate "
                    f"(override_by={override_by}, reason={override_reason}, override_time={override_time})."
                )
                mlflow.log_param("override_by", override_by)
                mlflow.log_param("override_reason", override_reason)
                mlflow.log_param("override_time", override_time)
                mlflow.log_metric("validation_gate_manual_override", 1)
                mlflow.log_metric("validation_gate_passed", 1)
                return True

            mlflow.log_metric("validation_gate_manual_override", 0)
            mlflow.log_metric("validation_gate_passed", 0)
            return False
    ```

14. **Register (Conditional):** Chỉ khi Validation Gate pass → Lưu best model, SHAP explainer, metrics, hyperparameters, model type (đánh dấu loại model: XGBoost/LightGBM/Random Forest), và data lineage lên **MLflow** → Transition model sang stage `Production`.

15. **DVC Artifact Commit & Push:**
    * Cập nhật `dvc.lock` sau khi materialize artifacts mới.
    * Push artifacts lên remote MinIO/S3 bằng `dvc push`.
    * Log `dvc_data_revision` (Git SHA hoặc DVC lock hash) vào MLflow run để liên kết model ↔ data revision.
    * **Rule:** Chỉ train khi required artifacts đã `dvc pull` thành công ở workspace hiện tại.

**Output artifacts của pipeline A:**
* `data/bronze/events.parquet`
* `data/silver/events.parquet`
* `data/gold/train_snapshots.parquet`
* `data/gold/val_snapshots.parquet`
* `data/gold/test_snapshots.parquet`
* `data/gold/session_split_map.parquet`
* `dvc.yaml`, `dvc.lock` (pipeline definition + frozen artifact revisions)

---

## Pipeline B: Real-time Inference Pipeline (Online — Chạy liên tục)

```
Simulator/User App → Kafka → Quix Streams → Redis + PostgreSQL → FastAPI → Dashboard
```

**Chi tiết từng bước:**

1. **Ingest:** `simulator.py` đọc CSV → Validate event → Preserve `source_event_time` → Gắn thêm `replay_time` → Gửi vào Kafka topic `raw_events`.
   * Events không hợp lệ → Log warning, skip (không gửi vào Kafka).
2. **Process:** Quix Streams worker:
   * Consume từ Kafka → Update session state incrementally theo `user_session` → Ghi Redis (real-time) + PostgreSQL (historical).
   * PostgreSQL lưu append-only events kèm `source_event_time` và `replay_time` để phục vụ audit, latency measurement, và retraining.
   * Nếu xử lý lỗi → Gửi event vào DLQ topic `failed_events`.
3. **Predict (with Caching):**
   * Dashboard/Client gọi `GET /api/v1/predict/{user_session}`.
   * **Cache check:** Kiểm tra Redis key `cache:predict:session:{user_session}`.
      * Nếu **cache hit** → Trả về cached result ngay lập tức (latency ~1-2ms).
      * Nếu **cache miss** → Lấy features từ Redis → Gọi model → Cache result (TTL 30s) → Trả về response.
   * Model được load từ MLflow khi startup và **hot-reload tự động** mỗi 5 phút (background thread kiểm tra model version mới trên MLflow Registry).
   * Response format:

      ```json
      {
        "user_session": "session_12345",
        "target_horizon_minutes": 10,
        "purchase_probability": 0.85,
        "model_version": "v1.2",
        "prediction_time": "2026-04-12T10:30:05Z",
        "prediction_mode": "model",
        "fallback_reason": null,
        "confidence": "high",
        "cached": false
      }
      ```

   * `model_version` giúp debug: xác định request đang được xử lý bởi model cũ hay mới sau khi retrain.
   * `cached` cho biết response là từ cache hay tính mới — hữu ích khi debug latency.
   * **Cache invalidation:** Khi Quix Streams ghi features mới cho session → Xóa `cache:predict:session:{user_session}` → Lần predict tiếp theo sẽ dùng features mới nhất.
   * **Fallback (`/predict` only):** Redis miss hoặc model error → Score `0.5`, `confidence: "low"`, `prediction_mode: "fallback"`, `fallback_reason`.
   * **Fallback caching policy:** Không cache fallback response.
4. **Explain:**
    * Dashboard/Client gọi `GET /api/v1/explain/{user_session}`.
    * FastAPI lấy features từ Redis → Tính SHAP values bằng explainer đã load (cùng hot-reload cycle với model) → Trả về top 3 features ảnh hưởng nhất.
    * Nếu explainer unavailable → endpoint `/explain` trả HTTP 503 với `error_code="EXPLAINER_UNAVAILABLE"`.
5. **Visualize:** Dashboard vẽ Score + Explanation lên biểu đồ real-time.

---

## Pipeline C: Model Retraining (Triggered — Khi cần thiết)

**Trigger conditions** (kiểm tra thủ công hoặc qua script scheduled):

* **Data Drift Detection:** Sử dụng **Population Stability Index (PSI)** để phát hiện distribution shift:
  * PSI < 0.1 → Không có drift đáng kể
  * 0.1 ≤ PSI < 0.2 → Drift nhẹ → Cảnh báo, theo dõi
  * PSI ≥ 0.2 → Drift nghiêm trọng → **Trigger retraining**
  * Tính PSI trên các features quan trọng: `cart_to_view_ratio`, `session_duration_sec`, `avg_price_viewed`
* **Concept Drift Detection:** So sánh prediction distribution giữa tuần hiện tại và tuần trước bằng **KL Divergence**:
  * KL Divergence > 0.1 → Cảnh báo
  * KL Divergence > 0.2 → **Trigger retraining**
* **Performance Degradation:** PR-AUC trên dữ liệu mới (từ PostgreSQL), với target **purchase trong 10 phút tới**, giảm dưới **0.7**.

**Script kiểm tra drift (chạy hàng ngày hoặc thủ công):**

```python
# training/src/drift_detection.py
from scipy.stats import entropy
import numpy as np

def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Calculate Population Stability Index."""
    expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)
    # Avoid division by zero
    expected_percents = np.clip(expected_percents, 0.001, None)
    actual_percents = np.clip(actual_percents, 0.001, None)
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi

def calculate_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate KL Divergence between two distributions."""
    return entropy(p, q)
```

**Quy trình retraining:**

> **Quy tắc quan trọng:** Retraining luôn đi qua **data lake pipeline** (raw → bronze → silver → gold). Không train trực tiếp từ PostgreSQL query. PostgreSQL chỉ cung cấp input events cho bước re-materialize.

1. Export dữ liệu mới từ PostgreSQL (7-14 ngày gần nhất).
2. **Re-materialize** dữ liệu mới qua pipeline:
   - Ghi vào `data/raw/` (giữ nguyên format như Kaggle source)
   - Chạy `bronze.py` → `data/bronze/`
   - Chạy `silver.py` → `data/silver/`
   - Chạy `session_split.py` + `gold.py` → `data/gold/` (**recompute split theo `session_start_time` trên retrain window 7-14 ngày**, không dùng split map cũ như mapping cứng)
3. Với outputs đã có trong `dvc.yaml`: chạy `dvc repro` rồi `dvc push` để lưu artifacts retrain window lên remote MinIO/S3.
4. Chỉ dùng `dvc add` cho artifacts ad-hoc chưa nằm trong pipeline definition.
5. Tính PSI và KL Divergence so với training data gốc.
6. Nếu vượt threshold → Chạy lại Pipeline A với dữ liệu gold mới (bao gồm log Data Lineage của data mới + `dvc_data_revision`).
7. **Model Validation Gate** tự động so sánh model mới vs model production hiện tại (xem Pipeline A, bước 10).
8. Nếu gate **pass** → Register model mới lên MLflow → Transition sang `Production` → FastAPI **tự động hot-reload** model mới trong vòng 5 phút (không cần restart).
9. Nếu gate **fail** → Giữ model cũ, log warning kèm metrics comparison → Grafana alert thông báo → Cần review thủ công.

**Lý do:** Đảm bảo mọi retrained model đều có lineage hoàn chỉnh qua data lake artifacts, có thể reproduce hoàn toàn từ source.

**Split policy cho retrain window:** `session_split_map.parquet` là artifact để reproducibility cho từng lần train/retrain; không phải source of truth để gán split cho các session mới trong tương lai.

---

## Pipeline D: Online Evaluation (Continuous Monitoring — Chạy liên tục)

**Mục đích:** Tính ground truth sau horizon 10 phút, đo accuracy/PR-AUC trên dữ liệu mới, phát hiện performance degradation.

**Timeline semantics:**
* **Demo replay mode:** Online evaluation dùng replay/serving timeline (`prediction_time`) để phản ánh behavior của hệ thống đang chạy.
* **Offline/backfill mode:** Label phải được tính theo source timeline semantics (`source_event_time`) để khớp với offline training contract.
* Hai mode không được trộn trong cùng một metric series.

```
PostgreSQL events (lưu prediction_id + prediction_time + purchase_probability)
    ↓
10 phút sau, kiểm tra: đó session có purchase event không?
    ↓
Join prediction ↔ outcome
    ↓
Tính accuracy, PR-AUC, confusion matrix trên micro-batches
    ↓
Ghi metrics lên Prometheus + MLflow
    ↓
Nếu PR-AUC < 0.7 → Trigger retrain alert
```

### Implementation Details

1. **Prediction Logging:** Khi FastAPI trả prediction, log bản ghi:
   ```json
    {
      "prediction_id": "pred_123456",
      "user_session": "session_xyz",
      "prediction_time": "2026-04-12T10:30:05Z",
      "purchase_probability": 0.85,
      "model_version": "v1.2",
      "prediction_mode": "model",
      "evaluation_mode": "demo_replay",
      "fallback_reason": null,
      "horizon_minutes": 10,
      "source_event_time": "2026-04-12T10:25:00Z"
    }
   ```
   → Lưu vào PostgreSQL table `predictions`.

2. **Background Job (chạy mỗi 15 phút):**
    - Query predictions từ 10 phút trước
    - **Demo replay mode:** Cho mỗi prediction, kiểm tra `user_session` có purchase event trong `(prediction_time, prediction_time + 10 phút]`?
    - **Offline/backfill mode:** Dùng source timeline (`source_event_time`) để tính label cùng semantics với offline snapshots.
    - Ghi outcome (0 hoặc 1) vào column `actual_purchase` trong `predictions` table.

3. **Accuracy Calculation (chạy mỗi 1 giờ):**
    - Aggregate predictions đã có ground truth từ 24 giờ trước với filter `prediction_mode = 'model'`
    - Tính PR-AUC, F1, Precision, Recall trên batch này
    - Ghi metrics theo từng mode lên Prometheus counter `model_accuracy_pr_auc{time_window='24h', evaluation_mode='<mode>'}`
    - Log lên MLflow theo mode (`mlflow.log_metric('online_pr_auc_24h_demo_replay', value)` hoặc `mlflow.log_metric('online_pr_auc_24h_offline_backfill', value)`)

   > **Rule:** Fallback predictions (`prediction_mode = 'fallback'`) không được dùng cho model-quality metrics (PR-AUC/F1/Precision/Recall).

4. **Degradation Alert:**
   - Nếu online PR-AUC < 0.7 trong 2 giờ liên tiếp → Trigger Grafana alert
   - Alert message: `"Model accuracy degraded: online PR-AUC = 0.65 (threshold 0.7). Consider retraining."`

### Schema: predictions table

```sql
CREATE TABLE predictions (
    prediction_id VARCHAR PRIMARY KEY,
    user_session VARCHAR,
    prediction_time TIMESTAMP,
    purchase_probability FLOAT,
    model_version VARCHAR,
    prediction_mode VARCHAR NOT NULL DEFAULT 'model',
    evaluation_mode VARCHAR NOT NULL,  -- demo_replay | offline_backfill
    fallback_reason VARCHAR NULL,
    horizon_minutes INT,
    source_event_time TIMESTAMP,
    
    -- Ground truth (filled 10 minutes later)
    actual_purchase INT DEFAULT NULL,  -- 0 or 1
    ground_truth_time TIMESTAMP DEFAULT NULL,
    
    -- Metrics
    is_correct_prediction BOOLEAN DEFAULT NULL,  -- TRUE if (pred >= 0.5) == actual_purchase
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_prediction_time ON predictions (prediction_time);
CREATE INDEX idx_predictions_user_session ON predictions (user_session);
CREATE INDEX idx_predictions_actual_purchase ON predictions (actual_purchase);
CREATE INDEX idx_predictions_prediction_mode ON predictions (prediction_mode);
CREATE INDEX idx_predictions_evaluation_mode ON predictions (evaluation_mode);
```

### Metrics to Track

| Metric | Frequency | Alert Threshold | Action |
|--------|-----------|-----------------|--------|
| **Online PR-AUC** | Hourly | < 0.7 (2 hours) | Trigger retrain |
| **Online F1-Score** | Hourly | < 0.5 (2 hours) | Warning alert |
| **Prediction Distribution** | Hourly | KL Divergence > 0.15 | Investigate drift |
| **Ground Truth Coverage** | Hourly | < 80% (late outcomes) | Check for prediction logging issues |
| **Fallback Rate** | Hourly | > 5% sustained | Investigate Redis/model availability |

> **Mode-separation rule:** Không aggregate metrics của `demo_replay` và `offline_backfill` vào cùng một series; mọi dashboard panel/alert phải filter theo `evaluation_mode`.
