# PROJECT BLUEPRINT: REAL-TIME COMMERCE INTENT SYSTEM

Hệ thống Phân tích & Dự đoán Ý định Mua hàng Thời gian thực

---

## ⚠️ Document Status

**Trạng thái:** `FREEZE-READY` ✅ (Kiến trúc mục tiêu locked & verified)

> **Lưu ý quan trọng:** Blueprint này mô tả **kiến trúc và thiết kế mục tiêu** cho hệ thống. Hiện tại, repository chứa:
> - ✅ Dataset gốc (`dataset/*.csv.gz`)
> - ✅ Exploratory Analysis (`notebook/eda.ipynb`)
> - ✅ **Week 1 Data Foundation:** Bronze/silver scripts, DVC pipeline scaffold (bronze/silver stages), schema contracts, basic tests
> - ⏳ Serving API, full MLOps orchestration, monitoring, CI/CD, integrated testing: **trong trang thái triển khai**
>
> Mục đích của Blueprint này là hướng dẫn chi tiết để **hoàn thiện** hệ thống đầy đủ theo kiến trúc được mô tả.
>
> Các code snippets trong tài liệu là **illustrative target-state examples**, có thể cần adaptation trước khi chạy được trong trạng thái repo hiện tại.
>
> **Roadmap chi tiết:** xem [12_ROADMAP.md](docs/BLUEPRINT/12_ROADMAP.md)

---

## Mục lục (Table of Contents)

Các section chi tiết được chia nhỏ trong thư mục [`docs/BLUEPRINT/`](docs/BLUEPRINT/):

| # | File | Mô tả |
|---|------|-------|
| 1 | [01_OVERVIEW.md](docs/BLUEPRINT/01_OVERVIEW.md) | Tổng quan (Overview) & Dữ liệu (Data Strategy) |
| 2 | [02_ARCHITECTURE.md](docs/BLUEPRINT/02_ARCHITECTURE.md) | Kiến trúc hệ thống (System Architecture) & Tech Stack |
| 3 | [03_FEATURES.md](docs/BLUEPRINT/03_FEATURES.md) | Feature Engineering (Offline & Real-time) |
| 4 | [04_PIPELINES.md](docs/BLUEPRINT/04_PIPELINES.md) | Training & Inference Pipelines |
| 5 | [05_PROJECT_STRUCTURE.md](docs/BLUEPRINT/05_PROJECT_STRUCTURE.md) | Cấu trúc dự án & Configuration |
| 6 | [06_ERROR_HANDLING.md](docs/BLUEPRINT/06_ERROR_HANDLING.md) | Error Handling & Logging |
| 7 | [07_TESTING.md](docs/BLUEPRINT/07_TESTING.md) | Testing Strategy |
| 8 | [08_SECURITY.md](docs/BLUEPRINT/08_SECURITY.md) | Security Considerations |
| 9 | [09_EXPLAINABILITY.md](docs/BLUEPRINT/09_EXPLAINABILITY.md) | Model Explainability (SHAP) |
| 10 | [10_PERFORMANCE.md](docs/BLUEPRINT/10_PERFORMANCE.md) | Performance Measurement |
| 11 | [11_DEMO.md](docs/BLUEPRINT/11_DEMO.md) | Kịch bản Demo (11 bước) |
| 12 | [12_ROADMAP.md](docs/BLUEPRINT/12_ROADMAP.md) | Roadmap triển khai |

---

## Tổng quan ngắn

Hệ thống **MLOps End-to-End** xử lý luồng dữ liệu clickstream để dự đoán
`P(purchase trong 10 phút tới | trạng thái của user_session tại thời điểm t)`
trong thời gian thực với độ trễ **< 1 giây**.

### Thành phần chính
- **Ingestion:** Apache Kafka + replay events giữ nguyên `source_event_time` + deterministic `event_id`
- **Processing:** Quix Streams với session-scoped incremental state, deduplication, late-event handling
- **Feature Store:** Redis (TTL 30 phút, exact counts via Set cho unique_products/categories)
- **Data Versioning & Artifacts:** DVC quản lý data artifacts (`raw/bronze/silver/gold` layers) + remote object storage S3-compatible (MinIO); training pipeline orchestrated qua dvc repro.
- **Model Registry & Orchestration:** MLflow quản lý model registry, experiment metrics, validation gate, re-materialization trigger cho retrain; DVC & MLflow tách rõ ranh giới — MLflow không quản lý data, DVC không quản lý model registry.
- **API:** FastAPI với session-scoped predict/explain, hot-reload + caching
- **Monitoring:** Prometheus + Grafana (6 alert rules)

### Điểm nổi bật

- **Train/Serve Alignment:** Offline training dùng snapshot theo thời điểm, cùng semantics với online inference: tại snapshot time `t`, feature chỉ dùng các event đã xảy ra đến `t`.
- **Serving & Evaluation Contracts:** `/predict` hỗ trợ graceful degradation khi model/feature-store unavailable; `/explain` trả HTTP 503 khi không khả dụng. Online evaluation tách rõ `demo_replay` (deterministic replay) vs `offline_backfill` (model-quality metric); không merge hai timeline.
- **One Data Lake, Multiple Usage Windows:** Raw source pool gồm 7 file tháng; training, replay/demo, và retraining chỉ là các cách sử dụng khác nhau trên cùng data lake, không phải 3 pipeline dữ liệu độc lập.
- **Data Lake Layers:** `raw/bronze/silver/gold` tách bạch ingest, chuẩn hóa schema, clean, session-aware split, và snapshot training.
- **Official Split Policy:** Train/val/test split được xác định downstream từ silver layer theo `session_start_time` (UTC-normalized) và `user_session` boundary, không split theo snapshot rows hoặc hard-code theo file tháng.
- **Training/Retraining Source of Truth:** Luôn đi qua data lake artifacts. PostgreSQL chỉ là operational store và nguồn export cho retraining trước khi re-materialize lại qua pipeline.
- **Memory-Safe Materialization:** Bronze và silver artifacts có thể được materialize dưới dạng chunked/partitioned parquet datasets để xử lý dataset lớn mà không cần gom toàn bộ vào RAM.
- **Exact Count Semantics:** Redis dùng Set thay vì HyperLogLog để đảm bảo train/serve parity chính xác cho `unique_products`, `unique_categories`.
- **Event Ordering Policy:** Deterministic `event_id`, deduplication, late-event handling, manual event metadata.
- **Multi-Model Experimentation:** Train và compare 3 models (XGBoost, LightGBM, Random Forest), auto-select best.
- **Closed-Loop MLOps:** Drift detection -> Retrain (re-materialize) -> Validation Gate -> Hot-Reload.
- **Real-time Explainability:** SHAP values cho từng prediction.
- **Zero-Downtime:** Model hot-reload mỗi 5 phút.
- **One-Command Deploy:** Docker Compose.

---

## Liên kết nhanh

- [Architecture Diagram](docs/BLUEPRINT/02_ARCHITECTURE.md#system-architecture)
- [Features](docs/BLUEPRINT/03_FEATURES.md)
- [Demo Script](docs/BLUEPRINT/11_DEMO.md)
- [Roadmap](docs/BLUEPRINT/12_ROADMAP.md)
