# 12. Roadmap triển khai

> **← Xem [11. Demo](11_DEMO.md)**  
> **→ Xem [BLUEPRINT.md gốc](../../BLUEPRINT.md)**

| Tuần | Milestone | Deliverables |
| --- | --- | --- |
| **1** | Data Foundation | `data/raw/`, chunked + partitioned `data/bronze/`, partition-aware `data/silver/`, config paths theo directory semantics, timestamp contract (`source_event_time`, `replay_time`, `prediction_time`), DVC init + MinIO remote setup + bronze memory benchmark report |
| **2** | Training Pipeline | Global session index, session-boundary split, snapshot dataset builder, 10-minute horizon labeling, `data/gold/` artifacts, feature engineering, **multi-model training (XGBoost, LightGBM, Random Forest)**, model comparison & auto-selection, SHAP analysis, MLflow integration, **Data Lineage**, **Model Validation Gate** |
| **3** | Stream Processing | Quix Streams processor, session-scoped Redis feature store, Kafka topics, timestamp preservation, **cache invalidation logic** |
| **4** | Serving & API | FastAPI (predict + explain + health) theo `user_session`, security (API Key + rate limit), **Model Hot-Reload**, **Prediction Caching**, unit tests |
| **5** | Frontend & Dashboard | Streamlit User App + Admin Dashboard (tích hợp SHAP visualization) |
| **6** | Monitoring & CI | Prometheus + Grafana (latency panels + **6 alert rules + Webhook**), GitHub Actions (**pytest-cov ≥ 70%**), integration tests cho snapshot target và session-scoped serving |
| **7** | Polish & Demo | Demo script rehearsal (11 bước), documentation, edge case testing, bronze memory benchmark review + saved benchmark artifact |

---

## Execution Plan

### Docs Update Plan

**`BLUEPRINT.md`**
* Thêm summary ngắn về one data lake, multiple usage windows.
* Giữ wording tổng quan, không đi sâu vào implementation details của partitions.

**`docs/BLUEPRINT/01_OVERVIEW.md`**
* Mô tả raw source pool gồm 7 file tháng.
* Khóa training window, replay/demo window, và retraining flow như các usage windows khác nhau trên cùng data lake.
* Giữ canonical contract: `user_session`, `source_event_time`, `replay_time`, `prediction_time`.
* Khóa split policy downstream theo `session_start_time` và `user_session` boundary.

**`docs/BLUEPRINT/04_PIPELINES.md`**
* Mô tả rõ: raw source pool -> window selection -> bronze ingestion -> silver clean/sort -> session index -> split -> gold -> train.
* Bronze phải là chunked/partitioned ingestion.
* Silver phải là dataset-oriented processing, không assume single bronze file.
* Replay dùng raw window riêng và preserve `source_event_time`.

**`docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`**
* Cập nhật tree thư mục theo dataset directories cho bronze/silver.
* Cập nhật config example theo directory semantics.
* Thêm training window và replay window vào config example.

**`docs/BLUEPRINT/07_TESTING.md`**
* Thêm test cho chunked bronze ingestion contract.
* Thêm test cross-month session boundary.
* Thêm test disjoint split map và window isolation.

**Repo-wide contract scan**
* Chạy stale-contract scan để fail nếu còn reference path/setting cũ như `data/bronze/events.parquet`, `data/silver/events.parquet`, `raw_data_path`, `bronze_data_path`, `silver_data_path`.

### Code Module Plan

**`training/src/bronze.py`**
* Đọc raw source pool theo file/chunk.
* Rename `event_time -> source_event_time`.
* Validate schema cơ bản.
* Write output vào `data/bronze/` dưới dạng parquet dataset memory-safe.

**`training/src/silver.py`**
* Đọc bronze dataset theo partition/window.
* Clean null/invalid values.
* Sort theo semantics phục vụ session indexing.
* Write output vào `data/silver/` dưới dạng parquet dataset.

**`training/src/session_split.py`**
* Build global session index từ silver layer trên training window.
* Split theo `user_session` boundary bằng `session_start_time`.
* Persist `session_split_map.parquet`.

**`training/src/gold.py`**
* Materialize snapshot rows per split.
* Compute features at time `t`.
* Label horizon 10 phút tới.
* Write `train/val/test` gold Parquet.

**`training/src/train.py`**
* Load gold artifacts.
* Train models trên split đã khóa.
* Log metrics/validation gate/MLflow artifacts.
* Log `dvc_data_revision` để trace model ↔ data revision.

**`training/src/evaluate.py`**
* Evaluate PR-AUC/F1/Precision/Recall trên gold test.

**`training/src/data_lineage.py`**
* Log lineage cho raw/bronze/silver/gold artifacts.

**`training/src/drift_detection.py`**
* Tính drift trên silver/gold feature distributions.

**`training/src/explainability.py`**
* Persist SHAP artifacts sau train.

**`training/src/retrain.py`**
* Export events từ PostgreSQL (7-14 ngày gần nhất).
* **Re-materialize** qua bronze → silver → gold pipeline với lineage metadata.
* Chạy `dvc push` cho artifacts mới lên MinIO remote.
* Chạy training từ gold artifacts (không train trực tiếp từ PostgreSQL).
* Invoke Model Validation Gate → register nếu pass.

**`dvc.yaml`**
* Định nghĩa stages `bronze`, `silver`, `session_split`, `gold`, `train`.
* Bronze và silver outputs nên là directory-based artifacts.
* Track deps/outs để `dvc repro`, `dvc push`, `dvc pull` hoạt động với dataset lớn mà không yêu cầu load toàn bộ dữ liệu vào RAM.

**`docker-compose.yml`**
* Bổ sung service `minio` + `mc` init container.
* Expose ports 9000 (S3 API), 9001 (console) cho local demo.

**`.env.example`**
* Thêm biến `MINIO_*` và `AWS_*` cho DVC remote credentials.

**`services/simulator/simulator.py`**
* Read từ raw CSV.
* Emit `replay_time` while preserving `source_event_time`.
* Sinh deterministic `event_id` theo công thức canonical: `hash(f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}")`.
* Gắn `source` field: `"kaggle"` cho replay, `"manual"` cho Streamlit events.

**`services/stream-processor/processor.py`**
* Update Redis state by `user_session`.
* Maintain session-scoped invalidation.
* **Event deduplication:** Check `event_id` đã thấy chưa, bỏ qua nếu trùng (vẫn log audit).
* **Late event handling:** Kiểm tra `source_event_time` vs `source_last_event_time`, bỏ qua nếu trễ quá ngưỡng, gửi vào `late_events` topic.
* **Manual event handling:** Gắn source metadata, verify `source_event_time = replay_time_now` cho manual events.

**`services/prediction-api/app/main.py`**
* Serve by `user_session`.
* Return `prediction_time` và horizon metadata.
* Bổ sung `prediction_mode` (`model` | `fallback`) và `fallback_reason` (nullable).

**`services/prediction-api/app/config.py`**
* Load data path config, horizon config, API config, Redis config, và model reload interval.

**`services/prediction-api/app/security.py`**
* Validate API key và giữ rate limit theo session-based predict/explain endpoints.

**`services/prediction-api/app/cache.py`**
* Cache keys by session, not user.
* Không cache fallback responses.

**`training/src/online_evaluation.py`**
* Loại bản ghi `prediction_mode='fallback'` khỏi PR-AUC/F1/Precision/Recall batches.

**`services/prediction-api/app/model_loader.py`**
* Hot-reload model + explainer theo contract session-scoped.

**`services/prediction-api/app/routers/predict.py`**
* Fetch session state và trả horizon-aligned score.

**`services/prediction-api/app/routers/explain.py`**
* Explain predictions cho session snapshots.

### Verification Plan

1. Raw/bronze/silver/gold docs phải thống nhất.
2. Không file nào được hard-code `2019-Oct.csv` như source duy nhất.
3. Không phần nào còn assume `data/bronze/events.parquet` hoặc `data/silver/events.parquet` là contract chính thức.
4. Không split logic nào được mô tả ở snapshot boundary trước session boundary.
5. Tests phải assert session disjointness và timestamp preservation.
6. Cross-month session phải vẫn thuộc đúng một split duy nhất.
7. Event deduplication phải dùng deterministic `event_id`, test verify không update state khi trùng.
8. Late event policy phải được test: trễ quá ngưỡng thì không cập nhật online state.
9. Redis exact count (Set/SCARD) phải khớp với offline exact counts.
10. Bronze row-count parity phải giữ nguyên khi chuyển sang chunked/partitioned materialization.
11. `dvc repro` + `dvc push` thành công, artifacts có thể `dvc pull` lại trên máy mới.
12. Repo-wide stale-contract scan phải sạch, không còn reference contract cũ.
13. Bronze memory benchmark phải được lưu thành report/artifact cho raw source pool 7 file.
