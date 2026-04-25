# 2. Kiến trúc Hệ thống (System Architecture)

> **← Xem [1. Overview](01_OVERVIEW.md)**  
> **→ Xem [3. Features](03_FEATURES.md)**

> **Execution profile (local dev): `DEV_SMOKE`**
> - Train window (dev): `2019-10` -> `2019-10`
> - Replay window (dev): `2019-11` -> `2019-11`
> - Profile này chỉ để tăng tốc vòng lặp phát triển; canonical target-state windows trong blueprint vẫn giữ nguyên.

Hệ thống hoạt động theo mô hình **Event-Driven Microservices**, chia thành 4 tầng:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     TẦNG 1: INGESTION (Đầu vào)                        │
│  ┌──────────────┐    ┌──────────────┐                                  │
│  │  Simulator    │    │  User App    │                                  │
│  │ (Replay Window)│    │  (Streamlit) │                                  │
│  └──────┬───────┘    └──────┬───────┘                                  │
│         └────────┬──────────┘                                          │
│                  ▼                                                      │
│         ┌──────────────┐                                               │
│         │ Apache Kafka │  (topic: raw_events)                          │
│         └──────┬───────┘                                               │
├────────────────┼───────────────────────────────────────────────────────┤
│                ▼          TẦNG 2: PROCESSING (Xử lý)                   │
│  ┌─────────────────────┐                                               │
│  │   Quix Streams       │  (Windowing + Aggregation)                   │
│  │   Stream Processor   │                                              │
│  └──────┬──────────────┘                                               │
│         │                                                              │
│         ├──────────────► ┌───────────┐  Real-time features             │
│         │                │   Redis   │  (TTL: 30 phút)                 │
│         │                └───────────┘                                  │
│         │                                                              │
│         └──────────────► ┌──────────────┐  Historical events           │
│                          │  PostgreSQL  │  (append-only log)            │
│                          └──────────────┘                              │
├────────────────────────────────────────────────────────────────────────┤
│                     TẦNG 3: SERVING (Phục vụ)                          │
│  ┌──────────────────────────────┐  ┌──────────┐  ┌──────────┐        │
│  │   FastAPI                           │  │  Redis   │  │  MLflow  │    │
│  │  GET /api/v1/predict/{user_session} │◄─│(features)│  │(registry)│    │
│  │  GET /api/v1/explain/{user_session} │  └──────────┘  └──────────┘    │
│  │  GET /health                 │                                     │
│  │  [API Key + Rate Limiting]   │                                     │
│  └──────┬───────────────────────┘                                     │
├─────────┼──────────────────────────────────────────────────────────────┤
│         ▼           TẦNG 4: MONITORING (Giám sát)                      │
│  ┌──────────────┐    ┌──────────────┐                                  │
│  │  Streamlit    │    │  Prometheus  │───► Grafana                     │
│  │  Dashboard    │    │  (metrics)   │    (system dashboard)          │
│  └──────────────┘    └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tầng 1: Ingestion & Simulation (Đầu vào)

* **Data Replayer (Simulator):** Đọc replay window `2019-Nov.csv.gz` từ replay/simulation raw source → Validate schema → Sinh deterministic `event_id` → Gắn `source=kaggle` → Gửi event vào Kafka topic `raw_events`.
  * **Event ID Generation (canonical):** `event_id = hash(f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}")` — deterministic để đảm bảo deduplication hoạt động đúng.
  * **Timestamp Contract:** Dữ liệu CSV giữ nguyên `source_event_time` để đảm bảo reproducibility. Simulator chỉ gắn thêm `replay_time` khi bắn event vào Kafka. Nếu stream processor cần xử lý theo processing time thì vẫn dùng clock hiện tại của worker, nhưng không được làm mất timestamp gốc.
  * **Usage Window Note:** Replay/demo là usage window `2019-11` và phải được giữ tách biệt với baseline training window `2019-10`.
* **User App (Streamlit):** Giao diện demo cho phép người demo chèn hành động thủ công vào luồng dữ liệu.
  * **Manual Event Contract:** Gắn `source = "manual"`, `source_event_time = replay_time_now` (không giả làm historical replay).
  * **Configurable:** Late event threshold (default 60 giây) có thể config qua `.env`.
* **Message Broker:** **Apache Kafka** (Docker) — Trung tâm phân phối event.

### Tầng 2: Processing & Storage (Xử lý)

* **Stream Processor:** **Quix Streams** (Python).
  * Logic: Kafka Consumer → Session-scoped state update → Derived feature calculation → Ghi Redis + PostgreSQL.
  * **Event deduplication:** Mỗi event có `event_id` được sinh deterministic tại ingestion theo công thức canonical `hash(f"{user_session}|{source_event_time}|{event_type}|{product_id}|{user_id}")`. Nếu `event_id` đã thấy → bỏ qua state update (vẫn log vào audit).
  * **Late event handling:** Nếu `source_event_time` < `source_last_event_time` - ngưỡng cho phép (default 60 giây) → không cập nhật online state, gửi vào `late_events` topic.
  * **Manual events (Streamlit):** Gắn `source = "manual"` và `source_event_time = replay_time_now`, không giả làm historical replay.
  * Stream framework có thể dùng windowing như cơ chế orchestration theo processing-time (ví dụ trigger/update cadence).
  * **Quan trọng:** Feature semantics chính thức là **session-scoped cumulative state đến snapshot time `t`**, không phải sliding-window features. Prediction target vẫn là purchase trong **10 phút tới** trên cùng `user_session` và train/serve alignment phải bám theo semantics cumulative này.
  * **Dead Letter Queue (DLQ):** Events xử lý lỗi được gửi vào Kafka topic `failed_events` để debug sau.

* **Feature Store (Real-time):** **Redis** (Docker).
  * Lưu trữ session state đã aggregate: `session:{user_session}:view_count`, `session:{user_session}:cart_count`, ...
  * **Exact count semantics:** Dùng Redis Set thay vì HyperLogLog để đảm bảo exact count cho `unique_categories`, `unique_products`. Điều này đảm bảo train/serve parity hoàn toàn chính xác.
  * **TTL: 30 phút** (configurable qua `.env`) — Tự động cleanup session state sau khi không còn event mới. Giá trị 30 phút được chọn vì dài hơn đáng kể so với session browsing trung bình, nhưng vẫn đủ chặt để tránh state rác tồn tại quá lâu.
  * **Key design:** Dùng `session:{user_session}` làm entity chính để train và serve cùng dự đoán trên một đối tượng. `user_id` vẫn có thể được lưu trong state như context phụ, nhưng không dùng để gộp nhiều session khác nhau vào cùng một prediction entity.

* **Historical Storage:** **PostgreSQL** (Docker).
  * Lưu trữ events đã xử lý dưới dạng append-only log, bao gồm cả `source_event_time` và `replay_time`.
  * Phục vụ phân tích sau (post-hoc analysis) và retraining model sau cửa sổ vận hành 7 ngày.

* **Artifact Storage (Data Lake Remote):** **MinIO** (S3-compatible object storage).
  * Lưu file artifacts lớn của `data/raw`, `data/bronze`, `data/silver`, `data/gold`.
  * Bronze/silver có thể materialize dưới dạng dataset directories partitioned/chunked để phục vụ memory-safe processing.
  * Là remote backend cho DVC (`dvc push`/`dvc pull`).

* **Data Versioning:** **DVC**.
  * Version metadata của artifacts trong Git (`.dvc`, `dvc.yaml`, `dvc.lock`).
  * Đảm bảo training/retraining có thể tái lập bằng cách checkout code revision + `dvc pull` đúng data revision.

* **Model Registry:** **MLflow** — Quản lý model versions, metrics, artifacts.

> **Scope boundary:** DVC + MinIO là source of truth cho data artifacts (`raw/bronze/silver/gold`). MLflow là source of truth cho model registry và experiment metrics.

### Tầng 3: Serving (Phục vụ)

* **Prediction Service:** **FastAPI**.
  * Endpoint `GET /api/v1/predict/{user_session}` — Trả về Purchase Probability cho **10 phút tới** + `model_version` + `prediction_time`.
  * Endpoint `GET /api/v1/explain/{user_session}` — Trả về SHAP-based explanation cho session snapshot hiện tại; nếu explainer unavailable thì trả HTTP 503 với `EXPLAINER_UNAVAILABLE`.
  * Endpoint `GET /health` — Health check cho monitoring (không cần versioning).
  * **API Versioning:** Prefix `/api/v1/` cho tất cả business endpoints — thể hiện tư duy mở rộng, dễ dàng thêm `/api/v2/` khi thay đổi response schema mà không break client cũ.
  * **Security:** API Key via header `X-API-Key` + Rate Limiting (xem mục 11).
  * **Fallback mechanism (`/predict`):** Nếu Redis miss/timeout hoặc model load fail → Trả về score mặc định `0.5`, `confidence: "low"`, `prediction_mode: "fallback"`, và `fallback_reason` (`redis_miss` hoặc `model_unavailable`).
  * **Fallback data policy:** Fallback predictions không được cache và không được đưa vào online model-quality metrics (PR-AUC/F1/Precision/Recall).
  * Input validation bằng **Pydantic** cho tất cả request/response.
  * **Model Hot-Reload:** Background thread poll MLflow Model Registry mỗi **5 phút** (configurable). Khi phát hiện model version mới ở stage `Production` → Tự động load model + SHAP explainer mới vào memory **mà không cần restart service**. Dùng **thread-safe swap** (load xong mới swap reference) để tránh downtime.
  * **Prediction Caching:** Cache prediction result trong Redis với key `cache:predict:session:{user_session}`, **TTL 30 giây** (configurable). Nếu cùng session gọi predict nhiều lần trong 30s → trả cache, giảm load lên model. Cache bị **invalidate tự động** khi Quix Streams cập nhật features mới cho session đó (ghi feature mới → xóa cache key).

### Tầng 4: Monitoring (Giám sát)

* **Admin Dashboard:** **Streamlit** — Hiển thị real-time metrics, user behavior, predictions.
* **System Monitoring:** **Prometheus + Grafana** (Docker).
  * **System metrics** (`system-overview.json`): Kafka consumer lag, API latency (p50/p95/p99), Redis memory usage, service health status.
  * **ML metrics** (`ml-metrics.json`): Prediction distribution over time, feature drift (PSI), accuracy metrics cho target **purchase trong 10 phút tới**, model version info.
  * **Docker Healthchecks:** Mọi service đều có healthcheck tích hợp → Grafana tự động hiển thị service health status.

* **Grafana Alerting:** Cảnh báo tự động khi hệ thống có vấn đề, gửi notification qua **Webhook** (hoặc Slack/Email nếu cấu hình):

  | Alert Rule | Condition | Severity | Action |
  | --- | --- | --- | --- |
  | **Kafka Consumer Lag** | Lag > 1000 messages trong 5 phút | 🔴 Critical | Notification + check stream-processor |
  | **API Latency Spike** | p95 > 500ms trong 3 phút | 🟠 Warning | Notification + check resource usage |
  | **API Error Rate** | Error rate > 5% trong 5 phút | 🔴 Critical | Notification + check logs |
  | **Feature Drift (PSI)** | PSI > 0.2 trên bất kỳ feature nào | 🟠 Warning | Notification + consider retraining |
  | **Model Staleness** | Model version không đổi > 7 ngày | 🟡 Info | Notification + review drift metrics |
  | **Redis Memory** | Memory usage > 80% | 🟠 Warning | Notification + check TTL configuration |

  * **Contact Point:** Cấu hình Grafana contact point dùng **Webhook** (gửi JSON payload đến endpoint tùy chỉnh). Trong môi trường demo, dùng webhook đến Streamlit Dashboard để hiển thị alert trực tiếp.
  * **Alert config** được provisioning tự động qua file `monitoring/grafana/provisioning/alerting.yml`.

### Docker Healthcheck Configuration

Mọi service trong `docker-compose.yml` đều được cấu hình **healthcheck** để:

* Tự động phát hiện service down/unhealthy.
* Grafana hiển thị trạng thái real-time (xanh/đỏ).
* Docker tự restart container nếu healthcheck fail liên tục.

```yaml
# docker-compose.yml (excerpt)
services:
  prediction-api:
    build: ./services/prediction-api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --save 60 1 --loglevel warning
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  kafka:
    image: confluentinc/cp-kafka:7.5.0
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER} -d $${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  mlflow:
    build: ./services/mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  minio-init:
    image: minio/mc:latest
    depends_on:
      - minio
    entrypoint: ["/bin/sh", "/init-bucket.sh"]
    restart: "no"
```

**Lợi ích:**

* `docker compose ps` hiển thị trạng thái `(healthy)` hoặc `(unhealthy)`.
* Prometheus scrape container health metrics → Grafana panel "Service Health Status".
* `depends_on` có thể dùng `condition: service_healthy` để đảm bảo thứ tự khởi động đúng.

---

# 3. Technology Stack

> **← Xem [2. Architecture](02_ARCHITECTURE.md)**  
> **→ Xem [3. Features](03_FEATURES.md)**

| Thành phần | Công nghệ | Vai trò cụ thể |
| --- | --- | --- |
| **Ingestion** | **Apache Kafka** | Message broker trung tâm, đảm bảo event ordering. |
| **Processing** | **Quix Streams** | Stream processing Python-native, hiệu năng cao, dễ debug. Xử lý deduplicate, late events, manual events. |
| **Feature Store** | **Redis** | Lưu trữ real-time features với TTL tự động. Dùng **Redis Set** cho exact counts (unique_products, unique_categories). |
| **Historical DB** | **PostgreSQL** | Persistent storage cho events và analytics. Chỉ là operational/audit store, không train trực tiếp. |
| **Object Storage** | **MinIO** (S3-compatible) | Remote storage cho artifacts lớn của data lake và DVC cache/object store. |
| **Data Versioning** | **DVC** | Version hóa artifacts `raw/bronze/silver/gold`, đảm bảo reproducibility theo revision. |
| **Model** | **XGBoost, LightGBM, Random Forest** | Multi-Model Experimentation — Train & compare 3 models, auto-select best cho Purchase Prediction. |
| **Explainability** | **SHAP** (TreeExplainer) | Giải thích prediction — tại sao model dự đoán user sẽ mua. |
| **MLOps** | **MLflow** | Model registry + experiment tracking + metrics/artifacts của run. |
| **Serving** | **FastAPI** | Prediction API + Explainability API. |
| **Security** | **slowapi** + API Key | Rate limiting + xác thực request đơn giản. |
| **Visualization** | **Streamlit** | User App (demo) + Admin Dashboard (monitoring). |
| **Monitoring** | **Prometheus + Grafana** | System metrics + performance benchmarking. |
| **Config** | **pydantic-settings** | Quản lý config qua environment variables. |
| **Testing** | **Pytest** | Unit tests + Integration tests. |
| **Infra** | **Docker Compose** | Orchestration toàn bộ hệ thống bằng 1 lệnh. |
| **CI** | **GitHub Actions** | Automated lint, test, build on every push. |
| **Orchestration** | **APScheduler / Airflow (planned)** | Scheduled drift detection, retraining trigger, model validation & promotion. |

---

# 4. Closed-Loop MLOps Orchestration

> **← Xem [4. Technology Stack](02_ARCHITECTURE.md)**  
> **→ Xem [3. Features](03_FEATURES.md)**

## 4.1. Retraining Trigger & Orchestration

**Trạng thái hiện tại:** Drift detection và retraining được chuẩn bị sẵn sàng để chạy thủ công hoặc qua scheduled job.

**Planned component (to-do):** Một scheduler service (APScheduler hoặc Airflow) để orchestrate quy trình retraining tự động.

### Retraining Workflow

```
┌──────────────────────────────────────────────────────────────┐
│          Scheduled Drift Detection Job (Hàng ngày)           │
│  (APScheduler / Airflow / cron task trong services)          │
└──────────────────┬───────────────────────────────────────────┘
                   ▼
         ┌─────────────────────────┐
         │  Calculate Drift        │
         │  - PSI (data features)  │
         │  - KL (predictions)     │
         │  - Performance degrade  │
         └──────────┬──────────────┘
                    ▼
         ┌──────────────────────┐
         │  Drift >= Threshold? │
         └──────┬───────────┬──────┘
                │ Yes       │ No
                ▼           ▼
         Trigger Retrain   Continue monitoring
                │
                ▼
    ┌──────────────────────────────┐
    │  Pipeline C: Retraining      │
    │  (từ 04_PIPELINES.md)        │
    │  - Export data từ PostgreSQL │
    │  - Re-materialize qua DLs    │
    │  - dvc push artifacts mới    │
    │  - Train multi-models        │
    └──────────┬───────────────────┘
               ▼
    ┌──────────────────────────────┐
    │  Model Validation Gate       │
    │  - Compare với production    │
    │  - PR-AUC >= threshold?      │
    └──────┬──────────────┬────────┘
           │ PASS         │ FAIL
           ▼              ▼
    Register model    Log warning
    Stage = Production (keep old model)
           │              │
           └──────┬───────┘
                  ▼
         ┌─────────────────────┐
         │  FastAPI Hot-Reload │
         │  (background thread)│
         │  Poll MLflow every  │
         │  5 min, detect new  │
         │  Production stage   │
         │  model → auto-load  │
         └─────────────────────┘
```

### Drift Thresholds (Configurable)

| Metric | Threshold | Action |
|--------|-----------|--------|
| **Data Drift (PSI)** | ≥ 0.2 | Trigger retrain |
| **Prediction Drift (KL)** | ≥ 0.2 | Trigger retrain |
| **Performance Degrade** | PR-AUC < 0.7 | Trigger retrain (optional warning-only) |

### Scheduler Implementation Options

**Option 1: APScheduler (Simple, lightweight)**
```python
# training/src/retrain_scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from training.src.drift_detection import check_drift_and_trigger_retrain

scheduler = BackgroundScheduler()
scheduler.add_job(check_drift_and_trigger_retrain, 'cron', hour=2, minute=0)
scheduler.start()  # Chạy retrain check mỗi ngày lúc 2:00 AM
```

**Option 2: Airflow DAG (Recommended for production)**
```python
# training/dags/retrain_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG('purchase_model_retrain', schedule_interval='0 2 * * *'):
    detect_drift = PythonOperator(
        task_id='detect_drift',
        python_callable=check_drift_and_trigger_retrain
    )
```

**Option 3: GitHub Actions Scheduled Workflow (Demo-friendly)**
```yaml
# .github/workflows/scheduled-retrain.yml
on:
  schedule:
    - cron: '0 2 * * *'  # Mỗi ngày 2:00 UTC

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check drift and trigger retrain
        run: python training/src/retrain_scheduler.py
```

---
