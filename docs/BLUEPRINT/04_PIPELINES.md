# 6. Chi tiết các Pipelines (Data Flow Spec)

> **← Xem [3. Features](03_FEATURES.md)**  
> **→ Xem [5. Project Structure](05_PROJECT_STRUCTURE.md)**

---

## Pipeline A: Training Pipeline (Offline — Chạy 1 lần hoặc khi retrain)

```
data/raw/*.csv(.gz) -> Window selection -> Bronze ingestion (chunked, partitioned)
    -> Silver clean/sort (partition-aware) -> Session index -> Session-boundary split
    -> Gold snapshots -> Train -> Evaluate -> SHAP Analysis
    -> Validation Gate (fail-closed; first deploy auto-pass) -> Register to MLflow
```

> **Artifact reproducibility:** `raw/bronze/silver/gold` artifacts được version bằng DVC; file thực nằm trên MinIO/S3 remote. Mọi lần train/retrain phải trace được về DVC revision.

**Chi tiết từng bước:**

1. **Select Input Window:** Chọn usage window từ raw source pool trong `data/raw/` theo mục đích sử dụng.
   * Training pipeline mặc định dùng window `2019-10` -> `2020-02`.
   * Replay/demo pipeline dùng window `2020-03` -> `2020-04`.
   * Retraining pipeline dùng exported operational window từ PostgreSQL sau khi materialize lại vào `data/raw/`.

2. **Bronze Ingestion:**
   * Đọc từng file `.csv` hoặc `.csv.gz` trong raw window.
   * Ưu tiên đọc theo chunk để tránh gom toàn bộ dataset vào RAM.
   * Rename `event_time -> source_event_time`.
   * Validate schema cơ bản và `event_type`.
   * Ghi output vào `data/bronze/` dưới dạng parquet dataset có thể partition theo thời gian hoặc theo file nguồn.

3. **Data Lineage Metadata:**
   * Ghi metadata của input window lên MLflow để đảm bảo reproducibility và traceability.
   * Không assume chỉ có một raw file duy nhất.
   * Metadata nên gồm:
     * `raw_input_manifest_hash`
     * `raw_input_file_count`
     * `raw_input_files`
     * `window_start`
     * `window_end`
     * `row_count_raw`
     * `row_count_bronze`
     * `data_source_type` (`raw_pool` hoặc `postgres_export`)
   * Metadata này được log ngay đầu experiment run.

4. **Silver Clean & Sort:**
   * Đọc bronze dataset theo partition hoặc grouped window.
   * Loại bỏ dòng thiếu `user_id`, `user_session`, `event_type`.
   * Loại bỏ dòng có `price <= 0` hoặc invalid price theo contract clean hiện hành.
   * Xử lý `category_code` null bằng fallback phù hợp nếu feature contract cần exact category counts.
   * Sort deterministic theo semantics phục vụ downstream session indexing.
   * Ghi output vào `data/silver/` dưới dạng parquet dataset.

5. **Session Index & Split Assignment:**
   * Build session index toàn cục từ silver layer trên training window với:
     * `user_session`
     * `session_start_time = min(source_event_time)`
     * `session_end_time = max(source_event_time)`
   * Split train/val/test theo `session_start_time` của `user_session`.
   * Assert một `user_session` chỉ thuộc đúng một split.
   * Persist split assignment vào `data/gold/session_split_map.parquet` để reproducibility.

6. **Gold Snapshot Generation & Feature Engineering:**
   * Materialize snapshot rows từ silver dataset theo split map đã khóa.
   * Tại mỗi thời điểm `t`, snapshot chỉ dùng các event có `source_event_time <= t`.
   * Label = `1` nếu cùng `user_session` có ít nhất 1 `purchase` trong `(t, t + 10 phút]`, ngược lại `0`.
   * Ghi output gold theo split vào `data/gold/`.
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
* `data/bronze/`
* `data/silver/`
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

1. **Ingest:** `simulator.py` đọc replay window từ raw source pool trong `data/raw/` -> validate event -> preserve `source_event_time` -> gắn thêm `replay_time` -> gửi vào Kafka topic `raw_events`.
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

1. Export dữ liệu mới từ PostgreSQL theo retraining window vận hành.
2. Materialize dữ liệu export vào `data/raw/` theo format tương thích với raw contract.
3. Chạy bronze ingestion -> `data/bronze/`.
4. Chạy silver pipeline -> `data/silver/`.
5. Build session index + split map mới trên retrain window; không reuse split map cũ như mapping cứng cho session mới.
6. Materialize gold snapshots mới từ split map vừa tạo.
7. Với outputs đã có trong `dvc.yaml`: chạy `dvc repro` rồi `dvc push`.
8. Chỉ dùng `dvc add` cho artifacts ad-hoc chưa nằm trong pipeline definition.

**Lý do:** Đảm bảo mọi retrained model đều có lineage hoàn chỉnh qua data lake artifacts, có thể reproduce hoàn toàn từ source.

**Split policy cho retrain window:** `session_split_map.parquet` chỉ là reproducibility artifact của từng lần train/retrain; source of truth cho split assignment vẫn là session index được build lại từ silver layer của window hiện tại.

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
