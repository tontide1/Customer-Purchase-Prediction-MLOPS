# 5. Cấu trúc Dự án (Project Structure)

> **← Xem [4. Pipelines](04_PIPELINES.md)**  
> **→ Xem [6. Error Handling](06_ERROR_HANDLING.md)**

> **Execution profile (local dev): `DEV_SMOKE`**
> - Train window (dev): `2019-10` -> `2019-10`
> - Replay window (dev): `2019-11` -> `2019-11`
> - Project contract hiện tại: `2019-Oct.csv.gz` cho baseline training, `2019-Nov.csv.gz` cho replay.

## 5.1. Repository Layout - TARGET STATE

Cấu trúc này mô tả **trạng thái hoàn chỉnh** của repository khi tất cả services, training, monitoring, CI/CD được triển khai đầy đủ:

> Các tree và snippets bên dưới là target-state illustration; có thể cần adaptation trước khi executable trong current repository.

```
REAL-TIME-ECOMMERCE-INTENT-SYSTEM/
├── docker-compose.yml              # Orchestration toàn bộ services (TO-DO)
├── .env.example                    # Template environment variables (TO-DO)
├── .dvc/                           # DVC internal metadata
├── .dvcignore                      # Ignore patterns for DVC scanning
├── dvc.yaml                        # DVC pipeline stages
├── dvc.lock                        # Frozen artifact revisions
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint + test (TO-DO)
│
├── services/                       # (TO-DO: hoàn thiện các services)
│   ├── simulator/                  # Data Replay service
│   │   ├── Dockerfile
│   │   ├── simulator.py
│   │   └── requirements.txt
│   │
│   ├── stream-processor/           # Quix Streams worker
│   │   ├── Dockerfile
│   │   ├── processor.py
│   │   └── requirements.txt
│   │
│   ├── prediction-api/             # FastAPI service
│   │   ├── Dockerfile
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py           # pydantic-settings
│   │   │   ├── schemas.py          # Request/Response models + internal logging schema (prediction_mode, fallback_reason, evaluation_mode)
│   │   │   ├── dependencies.py     # Redis, MLflow, SHAP clients
│   │   │   ├── model_loader.py     # Model hot-reload (background polling)
│   │   │   ├── cache.py            # Prediction caching logic
│   │   │   ├── security.py         # API Key validation + Rate Limiting
│   │   │   └── routers/
│   │   │       ├── predict.py
│   │   │       ├── explain.py      # SHAP explanation endpoint
│   │   │       └── health.py
│   │   ├── tests/
│   │   │   ├── test_predict.py
│   │   │   ├── test_explain.py
│   │   │   ├── test_health.py
│   │   │   ├── test_model_loader.py  # Hot-reload tests
│   │   │   └── test_cache.py         # Caching tests
│   │   └── requirements.txt
│   │
│   ├── dashboard/                  # Streamlit Admin Dashboard
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
│   │
│   └── user-app/                   # Streamlit User Demo
│       ├── Dockerfile
│       ├── app.py
│       └── requirements.txt
│
├── training/                       # (TO-DO: hoàn thiện training pipeline)
│   ├── src/
│   │   ├── train.py                # Main training script
│   │   ├── bronze.py               # Raw pool -> bronze parquet dataset
│   │   ├── silver.py               # Bronze dataset -> silver cleaned dataset
│   │   ├── session_split.py        # Silver -> session-boundary split map
│   │   ├── gold.py                 # Silver + split map -> gold snapshots
│   │   ├── features.py             # Feature engineering logic
│   │   ├── evaluate.py             # Evaluation metrics
│   │   ├── explainability.py       # SHAP analysis + save explainer
│   │   ├── model_validation.py     # Model Validation Gate (quality gate)
│   │   ├── data_lineage.py         # Data lineage metadata logging
│   │   ├── drift_detection.py      # PSI + KL Divergence drift detection
│   │   ├── online_evaluation.py    # Ground truth join + accuracy calc (exclude prediction_mode='fallback')
│   │   ├── retrain_scheduler.py    # Orchestrate retraining workflow
│   │   └── config.py               # Training hyperparameters
│   ├── tests/
│   │   ├── test_model_validation.py  # Validation gate tests
│   │   ├── test_data_lake.py         # Raw/bronze/silver/gold layer tests
│   │   └── test_online_eval.py       # Ground truth pipeline tests
│   └── requirements.txt
│
├── monitoring/                     # (TO-DO: hoàn thiện monitoring)
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       │   ├── system-overview.json    # System metrics (latency, throughput, health)
│       │   └── ml-metrics.json         # ML metrics (prediction distribution, drift, accuracy)
│       └── provisioning/
│           └── alerting.yml            # Alert rules + contact points
│
├── infra/                           # Infrastructure configs (TO-DO)
│   └── minio/
│       ├── init-bucket.sh           # Init bucket/policy for DVC remote
│       └── README.md
│
├── notebook/                       # (CURRENT STATE) Jupyter Notebooks
│   └── eda.ipynb                   # Exploratory Data Analysis
│
├── notebook-planned/               # (PLANNED) Thử nghiệm feature & models
│   ├── 02_feature_experiment.ipynb # Feature engineering experiments
│   └── 03_model_experiment.ipynb   # Multi-model comparison
│
├── docs/                           # Documentation
│   ├── BLUEPRINT/                  # Blueprint files (chi tiết từng phần)
│   │   ├── 01_OVERVIEW.md          # Overview & Data Strategy
│   │   ├── 02_ARCHITECTURE.md      # System Architecture & Tech Stack
│   │   ├── 03_FEATURES.md          # Feature Engineering
│   │   ├── 04_PIPELINES.md         # Training & Inference Pipelines
│   │   ├── 05_PROJECT_STRUCTURE.md # Project Structure & Config
│   │   ├── 06_ERROR_HANDLING.md    # Error Handling & Logging
│   │   ├── 07_TESTING.md           # Testing Strategy
│   │   ├── 08_SECURITY.md          # Security Considerations
│   │   ├── 09_EXPLAINABILITY.md    # Model Explainability
│   │   ├── 10_PERFORMANCE.md       # Performance Measurement
│   │   ├── 11_DEMO.md              # Demo Script
│   │   └── 12_ROADMAP.md           # Roadmap
│   └── BLUEPRINT.md                # (root) Overview & status
│
├── shared/                         # (TO-DO) Shared code across services
│   ├── schemas.py                  # Pydantic schemas (event validation)
│   └── constants.py                # Shared constants (topic names, Redis keys)
│
├── dataset/                        # (CURRENT STATE) Original datasets
│   ├── 2019-Oct.csv.gz
│   ├── 2019-Nov.csv.gz
│   ├── 2019-Dec.csv.gz
│   ├── 2020-Jan.csv.gz
│   ├── 2020-Feb.csv.gz
│   ├── 2020-Mar.csv.gz
│   └── 2020-Apr.csv.gz
│
└── data/                           # Data lake directory (gitignored)
    ├── raw/                        # Raw source pool (copy/symlink/materialized from dataset/)
    │   ├── 2019-Oct.csv.gz
    │   ├── 2019-Nov.csv.gz
    │   ├── 2019-Dec.csv.gz
    │   ├── 2020-Jan.csv.gz
    │   ├── 2020-Feb.csv.gz
    │   ├── 2020-Mar.csv.gz
    │   └── 2020-Apr.csv.gz
    ├── bronze/                     # Parsed parquet dataset, memory-safe materialization
    │   └── year=2019/
    │       └── month=10/
    │           └── part-000.parquet
    ├── silver/                     # Cleaned parquet dataset for downstream session indexing
    │   └── year=2019/
    │       └── month=10/
    │           └── part-000.parquet
    └── gold/                       # Snapshot training datasets + split artifacts
        ├── train_snapshots.parquet
        ├── val_snapshots.parquet
        ├── test_snapshots.parquet
        └── session_split_map.parquet
```

> **Lưu ý:** Partition tree bên trên chỉ là illustrative target-state. Có thể partition theo month, source file, hoặc grouped window; miễn là không đổi contract downstream theo `session_start_time` và `user_session`.

### Legend
- ✅ **CURRENT STATE**: Thành phần hiện đã có trong repo
- 🔄 **TO-DO**: Cần triển khai theo blueprint
- 📋 **PLANNED**: Được lên kế hoạch triển khai trong roadmap

---

## 5.2. Data Lake Paths - Tương ứng giữa Current và Target

| Purpose | Current Path | Target Path | Ghi chú |
|---------|--------------|------------|---------|
| **Raw Data Input** | `dataset/*.csv.gz` | `data/raw/` | `data/raw/` là raw source pool dùng cho training/replay/retraining materialization |
| **Analysis & EDA** | `notebook/eda.ipynb` | `notebook/` hoặc `notebook-planned/` | Tái sử dụng insights từ EDA hiện có |
| **Feature Experiments** | (không có) | `notebook-planned/02_feature_experiment.ipynb` | Cần tạo để experiment trước khi commit features |
| **Model Experiments** | (không có) | `notebook-planned/03_model_experiment.ipynb` | So sánh XGBoost vs LightGBM vs Random Forest |
| **Bronze Artifacts** | (không có) | `data/bronze/` | Dataset directory cho parquet bronze, có thể partition theo file/tháng |
| **Silver Artifacts** | (không có) | `data/silver/` | Dataset directory cho parquet silver, phục vụ session indexing |
| **Gold Artifacts** | (không có) | `data/gold/train_snapshots.parquet` | Output của `training/src/gold.py` |
| **Split Mapping** | (không có) | `data/gold/session_split_map.parquet` | Reproducibility artifact downstream; không phải source of truth cho session mới |

---

---

# 6. Configuration Management

> **← Xem [5. Project Structure](05_PROJECT_STRUCTURE.md)**  
> **→ Xem [6. Error Handling](06_ERROR_HANDLING.md)**

Tất cả config được quản lý qua **environment variables**, load bằng **pydantic-settings**.

**Nguyên tắc least privilege:** mỗi service chỉ load đúng secrets/config cần thiết cho vai trò của nó.

Ví dụ config cho **prediction-api** (không chứa DVC/MinIO credentials):

```python
# Ví dụ: services/prediction-api/app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_feature_ttl: int = 1800  # 30 phút

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_model_name: str = "purchase-predictor"

    # API
    api_default_score: float = 0.5
    api_score_threshold: float = 0.65
    prediction_horizon_minutes: int = 10
    exclude_fallback_from_quality_metrics: bool = True

    # Model Hot-Reload
    model_reload_interval: int = 300  # 5 phút (giây)

    # Prediction Caching
    prediction_cache_ttl: int = 30  # 30 giây

    # Security
    api_key: str = "dev-api-key-change-in-production"
    rate_limit: str = "60/minute"

    class Config:
        env_file = ".env"
```

Ví dụ config cho **training/data pipeline** (chứa DVC + MinIO):

```python
# Ví dụ: training/src/config.py
from pydantic_settings import BaseSettings

class TrainingSettings(BaseSettings):
    # Data Lake Paths
    raw_data_dir: str = "data/raw"
    bronze_data_dir: str = "data/bronze"
    silver_data_dir: str = "data/silver"
    gold_data_dir: str = "data/gold"

    # Usage Windows
    training_window_start: str = "2019-10-01T00:00:00Z"
    training_window_end: str = "2019-10-31T23:59:59Z"
    replay_window_start: str = "2019-11-01T00:00:00Z"
    replay_window_end: str = "2019-11-30T23:59:59Z"

    # Prediction Contract
    prediction_horizon_minutes: int = 10

    # DVC + Object Storage (MinIO/S3-compatible)
    dvc_remote_name: str = "minio"
    dvc_remote_url: str = "s3://mlops-artifacts"
    s3_endpoint_url: str = "http://minio:9000"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"

    class Config:
        env_file = ".env"
```

DEV_SMOKE local override (khuyến nghị khi iterate nhanh):

```env
TRAINING_WINDOW_START=2019-10
TRAINING_WINDOW_END=2019-10
REPLAY_WINDOW_START=2019-11
REPLAY_WINDOW_END=2019-11
```

**File `.env.example`** được commit vào repo, file `.env` thực tế được **gitignored**.

Các biến tối thiểu cho DVC + MinIO (dùng cho training/pipeline/infra, không cấp cho prediction-api runtime):

```env
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=http://minio:9000
MINIO_BUCKET=mlops-artifacts
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
```

### DVC Remote Config (target-state)

```bash
dvc remote add -d minio s3://mlops-artifacts
dvc remote modify minio endpointurl http://minio:9000
dvc remote modify minio access_key_id ${AWS_ACCESS_KEY_ID}
dvc remote modify minio secret_access_key ${AWS_SECRET_ACCESS_KEY}
dvc remote modify minio use_ssl false
```
