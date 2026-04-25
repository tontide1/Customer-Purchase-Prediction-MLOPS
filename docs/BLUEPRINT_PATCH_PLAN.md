# BLUEPRINT Patch Plan

## Mục tiêu

Chuẩn hóa lại BLUEPRINT để phản ánh đúng contract kiến trúc cần giữ và đúng hơn với source hiện có trong repo:

1. Giữ nguyên các nguyên tắc cốt lõi:
   - training/retraining source of truth là `raw -> bronze -> silver -> gold`
   - entity chính là `user_session`
   - prediction target là purchase trong 10 phút tới tại snapshot time `t`
   - split policy chính thức theo `user_session` boundary và `session_start_time`
   - train/serve alignment giữ semantics chỉ dùng event đến thời điểm `t`

2. Đổi wording và artifact contract từ mô hình single-file sang memory-safe dataset:
   - không còn mô tả `data/bronze/events.parquet` là contract chính
   - ưu tiên `data/bronze/` và `data/silver/` là dataset directories
   - bronze/silver wording phải hỗ trợ chunked + partitioned materialization

3. Tách rõ:
   - usage window
   - split policy
   - replay/demo window
   - retraining window

---

## Quyết định khóa trước khi patch

1. Raw source pool chính thức là 7 file tháng từ `2019-10` đến `2020-04`.
2. `data/raw/` là raw lake directory; có thể là copy/symlink/materialized view từ `dataset/*.csv.gz`.
3. `data/bronze/` và `data/silver/` là dataset directories, không khóa vào một file `events.parquet`.
4. Recommended partition layout:
   `data/bronze/year=2019/month=10/part-*.parquet`
   `data/silver/year=2019/month=10/part-*.parquet`
   - **⚠️ Partition layout chỉ là physical storage layout để materialize parquet dataset memory-safe; đây không phải processing boundary logic, split boundary, hay session-assignment boundary.**
   - **⚠️ Session index phải được build trên toàn bộ training window đã chọn, không được build/split độc lập theo từng year/month partition.**
5. Training window là Oct 2019 -> Feb 2020 (UTC-normalized).
6. Replay/demo window là Mar 2020 -> Apr 2020 (UTC-normalized).
7. Split train/val/test chỉ được xác định downstream từ silver layer theo `session_start_time` (UTC) và `user_session` boundary.
8. `session_split_map.parquet` là reproducibility artifact downstream từ silver, không phải source of truth cho session mới.
9. Retraining từ PostgreSQL phải export rồi re-materialize lại qua data lake.
10. Data lineage metadata phải lưu manifest dưới dạng artifact, không log toàn bộ danh sách files vào MLflow params.

---

## Phạm vi patch chính

1. `BLUEPRINT.md`
2. `docs/BLUEPRINT/01_OVERVIEW.md`
3. `docs/BLUEPRINT/04_PIPELINES.md`
4. `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
5. `docs/BLUEPRINT/07_TESTING.md`
6. `docs/BLUEPRINT/12_ROADMAP.md`

---

## Patch order

1. Sửa root summary ở `BLUEPRINT.md`
2. Sửa data strategy ở `01_OVERVIEW.md`
3. Sửa data flow contract ở `04_PIPELINES.md`
4. Sửa path/layout/config examples ở `05_PROJECT_STRUCTURE.md`
5. Sửa testing contract ở `07_TESTING.md`
6. Sửa roadmap và implementation priorities ở `12_ROADMAP.md`

---

## 1. Patch plan cho `BLUEPRINT.md`

### Vị trí cần sửa

1. `BLUEPRINT.md:45-49`
2. `BLUEPRINT.md:61-73`

### Hành động

1. Giữ phần mô tả mục tiêu tổng quát ở `45-49`.
2. Không thêm chi tiết implementation sâu vào root file.
3. Sửa block `Điểm nổi bật` để phản ánh:
   - one data lake, multiple usage windows
   - official split policy theo `user_session`
   - data lake là training/retraining source of truth
4. Không đưa path cứng kiểu `data/bronze/events.parquet` vào file này.

### Block nên thay ở phần `Điểm nổi bật`

Thay block hiện tại ở `61-69` bằng block sau:

```md
### Điểm nổi bật

- **Train/Serve Alignment:** Offline training dùng snapshot theo thời điểm, cùng semantics với online inference: tại snapshot time `t`, feature chỉ dùng các event đã xảy ra đến `t`.
- **One Data Lake, Multiple Usage Windows:** Raw source pool gồm 7 file tháng; training, replay/demo, và retraining chỉ là các cách sử dụng khác nhau trên cùng data lake, không phải 3 pipeline dữ liệu độc lập.
- **Data Lake Layers:** `raw/bronze/silver/gold` tách bạch ingest, chuẩn hóa schema, clean, session-aware split, và snapshot training.
- **Official Split Policy:** Train/val/test split được xác định downstream từ silver layer theo `session_start_time` và `user_session` boundary, không split theo snapshot rows hoặc hard-code theo file tháng.
- **Training/Retraining Source of Truth:** Luôn đi qua data lake artifacts. PostgreSQL chỉ là operational store và nguồn export cho retraining trước khi re-materialize lại qua pipeline.
- **Memory-Safe Materialization:** Bronze và silver artifacts có thể được materialize dưới dạng chunked/partitioned parquet datasets để xử lý dataset lớn mà không cần gom toàn bộ vào RAM.
- **Exact Count Semantics:** Redis dùng Set thay vì HyperLogLog để đảm bảo train/serve parity chính xác cho `unique_products`, `unique_categories`.
- **Event Ordering Policy:** Deterministic `event_id`, deduplication, late-event handling, manual event metadata.
- **Multi-Model Experimentation:** Train và compare 3 models (XGBoost, LightGBM, Random Forest), auto-select best.
- **Closed-Loop MLOps:** Drift detection -> Retrain (re-materialize) -> Validation Gate -> Hot-Reload.
- **Real-time Explainability:** SHAP values cho từng prediction.
- **Zero-Downtime:** Model hot-reload mỗi 5 phút.
- **One-Command Deploy:** Docker Compose.
```

---

## 2. Patch plan cho `docs/BLUEPRINT/01_OVERVIEW.md`

### Vị trí cần sửa

1. `01_OVERVIEW.md:32-40`
2. `01_OVERVIEW.md:42-54`
3. `01_OVERVIEW.md:55-76`

### Hành động

1. Sửa wording ở `2.1` để mô tả đúng raw source pool 7 file tháng.
2. Sửa `2.2` để bronze/silver là dataset directories, không ngầm ám chỉ single-file artifact.
3. Thay toàn bộ `2.3` từ "Dual Strategy" thành "Một Data Lake, Nhiều Mục Đích Sử Dụng".
4. Tách rõ:
   - training window
   - replay window
   - retraining flow
   - split policy downstream
5. Giữ nguyên các mục `2.5`, `2.6`, `2.7` vì contract ở đó đang đúng.

### Patch cụ thể cho `2.1. Nguồn dữ liệu`

Thay các dòng `38-40` bằng:

```md
* **Raw source pool:** Dataset hiện gồm 7 file tháng từ `2019-10` đến `2020-04`, đang lưu ở `dataset/*.csv.gz` và có thể được materialize vào `data/raw/` để chạy pipeline.
* **Các trường quan trọng:** `event_time`, `event_type`, `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`.
* **Quy ước timestamp trong hệ thống:** `event_time` từ Kaggle được preserve thành `source_event_time`; simulator thêm `replay_time`; FastAPI thêm `prediction_time` trong response.
```

### Patch cụ thể cho `2.2. Data Lake Layers`

Thay block `42-54` bằng:

```md
## 2.2. Data Lake Layers (Raw -> Bronze -> Silver -> Gold)

Luồng dữ liệu được chuẩn hóa theo 4 lớp bất biến:

* **`data/raw/`**: Raw source pool ở dạng CSV/CSV.GZ, giữ nguyên nội dung nguồn.
* **`data/bronze/`**: Parsed parquet artifacts sau khi rename `event_time -> source_event_time`; có thể được materialize theo chunk và partition để đảm bảo memory-safe processing.
* **`data/silver/`**: Cleaned parquet artifacts sau khi validate required fields, làm sạch dữ liệu, và sort theo semantics phục vụ downstream session indexing; có thể được materialize theo partition/window.
* **`data/gold/`**: Snapshot training datasets đã có features + label cho target 10 phút tới.

**Data versioning layer:** Các artifacts của 4 lớp trên được version bằng **DVC** và lưu file thực trên object storage S3-compatible (MinIO).

**Nguyên tắc:** split train/val/test được xác định downstream từ silver layer theo `user_session` boundary và `session_start_time`, trước khi sinh gold snapshots để tránh leakage.
```

### Patch cụ thể cho `2.3`

Đổi tiêu đề `2.3` và thay toàn bộ block `55-76` bằng:

```md
## 2.3. Một Data Lake, Nhiều Mục Đích Sử Dụng

Toàn bộ 7 file tháng được xem là **một raw source pool chung** trong data lake. Từ source pool này, hệ thống có 3 cách sử dụng dữ liệu khác nhau:

### Offline Training

* **Training window:** Chọn dữ liệu trong khoảng `2019-10` -> `2020-02`.
* **Input chuẩn:** Dùng `data/silver/` để build session index và split assignment, sau đó materialize `data/gold/`.
* **Đơn vị mẫu huấn luyện:** Không phải một dòng cho cả session đã kết thúc, mà là nhiều **snapshot rows** trên cùng `user_session`.
* **Quy tắc snapshot:** Tại mỗi thời điểm `t`, feature chỉ được tính từ các event có `source_event_time <= t`.
* **Quy tắc label:** `1` nếu cùng `user_session` có ít nhất 1 event `purchase` trong khoảng `(t, t + 10 phút]`, ngược lại `0`.
* **Split policy chính thức:** Build session index từ silver layer, sau đó split train/val/test theo `session_start_time` của `user_session`.
* **Ràng buộc:** Mỗi `user_session` chỉ được phép xuất hiện trong đúng một split. Không split theo snapshot rows và không hard-code theo file tháng.

### Online Replay / Demo

* **Replay window:** Chọn dữ liệu trong khoảng `2020-03` -> `2020-04`.
* **Nguồn replay:** Đọc từ raw source pool trong `data/raw/`.
* **Kỹ thuật:** Script replay đọc event, preserve `source_event_time`, gắn thêm `replay_time`, rồi gửi vào hệ thống theo thời gian thực.
* **Mục tiêu:** Mô phỏng behavior online thực tế mà không phá vỡ source timeline semantics của dữ liệu gốc.

### Retraining

* **Nguồn input:** Export events mới từ PostgreSQL theo một retraining window vận hành.
* **Yêu cầu bắt buộc:** Dữ liệu export phải được re-materialize lại qua `raw -> bronze -> silver -> gold` trước khi train.
* **Lý do:** Giữ reproducibility, lineage, và cùng semantics với offline training pipeline.
```

### Ghi chú wording phải giữ

Không dùng lại wording kiểu:
- "dataset được chia làm 2 phần"
- "Train = Oct-Dec, Val = Jan, Test = Feb"

---

## 3. Patch plan cho `docs/BLUEPRINT/04_PIPELINES.md`

### Vị trí cần sửa

1. `04_PIPELINES.md:10-14`
2. `04_PIPELINES.md:20-29`
3. `04_PIPELINES.md:30-59`
4. `04_PIPELINES.md:61-80`
5. `04_PIPELINES.md:214-220`
6. `04_PIPELINES.md:233-234`
7. `04_PIPELINES.md:314-321`
8. `04_PIPELINES.md:330`

### Hành động

1. Sửa pipeline ASCII để phản ánh raw source pool và session index.
2. Sửa step 1-6 để tách rõ:
   - window selection
   - chunked bronze ingestion
   - partition-aware silver
   - global session index
   - split assignment downstream
3. Sửa data lineage example vì hiện đang assume single raw file.
4. Sửa output artifacts từ single files sang dataset directories.
5. Sửa replay input wording ở Pipeline B.
6. Sửa retraining wording để không reuse split map cũ như mapping cứng.

### Patch cụ thể cho pipeline overview

Thay block `10-14` bằng:

```
data/raw/*.csv(.gz) -> Window selection -> Bronze ingestion (chunked, partitioned)
    -> Silver clean/sort (partition-aware) -> Session index -> Session-boundary split
    -> Gold snapshots -> Train -> Evaluate -> SHAP Analysis
    -> Validation Gate (fail-closed; first deploy auto-pass) -> Register to MLflow
```

### Patch cụ thể cho steps 1-6 của Pipeline A

Thay block `20-29` và `61-80` bằng wording sau:

```md
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
```

### Patch cụ thể cho data lineage example

Block `30-59` hiện dùng `raw_file_md5` và `raw_file_path`, không phù hợp với multi-file raw pool.

Cần sửa theo hướng sau:

1. Đổi `_file_md5(path)` thành helper cho manifest nhiều file hoặc hash của raw input manifest.
2. Đổi function signature từ:
   `log_data_lineage(df, data_source, raw_file_path)`
   thành dạng:
   `log_data_lineage(df, data_source_type, raw_input_files, window_start, window_end)`
3. Đổi các field log:
   - bỏ `raw_file_md5`
   - bỏ `data_source` nghĩa là một file đơn
   - thêm `raw_input_manifest_hash`
   - thêm `raw_input_file_count`
   - thêm `window_start`
   - thêm `window_end`

Không cần giữ exact sample code hiện tại; nên rewrite sample code để nó không khóa contract về một raw file duy nhất.

### Patch cụ thể cho output artifacts

Thay block `214-220` bằng:

```md
**Output artifacts của pipeline A:**
* `data/bronze/`
* `data/silver/`
* `data/gold/train_snapshots.parquet`
* `data/gold/val_snapshots.parquet`
* `data/gold/test_snapshots.parquet`
* `data/gold/session_split_map.parquet`
* `dvc.yaml`, `dvc.lock` (pipeline definition + frozen artifact revisions)
```

### Patch cụ thể cho Pipeline B

Sửa `233-234` từ wording "đọc CSV" sang wording raw pool + replay window:

```md
1. **Ingest:** `simulator.py` đọc replay window từ raw source pool trong `data/raw/` -> validate event -> preserve `source_event_time` -> gắn thêm `replay_time` -> gửi vào Kafka topic `raw_events`.
```

### Patch cụ thể cho retraining section

Sửa `314-321` thành:

```md
1. Export dữ liệu mới từ PostgreSQL theo retraining window vận hành.
2. Materialize dữ liệu export vào `data/raw/` theo format tương thích với raw contract.
3. Chạy bronze ingestion -> `data/bronze/`.
4. Chạy silver pipeline -> `data/silver/`.
5. Build session index + split map mới trên retrain window; không reuse split map cũ như mapping cứng cho session mới.
6. Materialize gold snapshots mới từ split map vừa tạo.
7. Với outputs đã có trong `dvc.yaml`: chạy `dvc repro` rồi `dvc push`.
8. Chỉ dùng `dvc add` cho artifacts ad-hoc chưa nằm trong pipeline definition.
```

### Patch cụ thể cho split policy note

Giữ ý ở `330`, nhưng sửa wording ngắn gọn hơn thành:

```md
**Split policy cho retrain window:** `session_split_map.parquet` chỉ là reproducibility artifact của từng lần train/retrain; source of truth cho split assignment vẫn là session index được build lại từ silver layer của window hiện tại.
```

---

## 4. Patch plan cho `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`

### Vị trí cần sửa

1. `05_PROJECT_STRUCTURE.md:139-147`
2. `05_PROJECT_STRUCTURE.md:157-169`
3. `05_PROJECT_STRUCTURE.md:221-240`

### Hành động

1. Sửa tree minh họa trong `data/` để thể hiện bronze/silver là dataset directories.
2. Sửa bảng path mapping để bỏ `events.parquet`.
3. Sửa config example theo directory semantics.
4. Bổ sung config windows cho training và replay.

### Patch cụ thể cho tree `data/`

Thay block `139-147` bằng:

```md
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

### Patch cụ thể cho bảng path mapping

Thay block `157-169` bằng:

```md
| Purpose | Current Path | Target Path | Ghi chú |
|---------|--------------|------------|---------|
| **Raw Data Input** | `dataset/*.csv.gz` | `data/raw/` | `data/raw/` là raw source pool dùng cho training/replay/retraining materialization |
| **Analysis & EDA** | `notebook/eda.ipynb` | `notebook/` hoặc `notebook-planned/` | Tái sử dụng insights từ EDA hiện có |
| **Feature Experiments** | (không có) | `notebook-planned/02_feature_experiment.ipynb` | Cần tạo để experiment trước khi commit features |
| **Model Experiments** | (không có) | `notebook-planned/03_model_experiment.ipynb` | So sánh XGBoost vs LightGBM vs Random Forest |
| **Bronze Artifacts** | (không có) | `data/bronze/` | Parquet dataset, có thể partition theo file/tháng |
| **Silver Artifacts** | (không có) | `data/silver/` | Cleaned parquet dataset cho session indexing |
| **Gold Artifacts** | (không có) | `data/gold/train_snapshots.parquet` | Output của `training/src/gold.py` |
| **Split Mapping** | (không có) | `data/gold/session_split_map.parquet` | Session-boundary split assignment để reproducibility |
```

### Patch cụ thể cho config example

Thay block `221-240` bằng:

```md
# Ví dụ: training/src/config.py
from pydantic_settings import BaseSettings

class TrainingSettings(BaseSettings):
    # Data Lake Paths
    raw_data_dir: str = "data/raw"
    bronze_data_dir: str = "data/bronze"
    silver_data_dir: str = "data/silver"
    gold_data_dir: str = "data/gold"

    # Usage Windows
    training_window_start: str = "2019-10-01"
    training_window_end: str = "2020-02-29 23:59:59"
    replay_window_start: str = "2020-03-01"
    replay_window_end: str = "2020-04-30 23:59:59"

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

### Wording note

Trong `05_PROJECT_STRUCTURE.md`, không dùng lại example:
- `raw_data_path: "data/raw/<dataset-file>.csv"`
- `bronze_data_path: "data/bronze/events.parquet"`
- `silver_data_path: "data/silver/events.parquet"`

---

## 5. Patch plan cho `docs/BLUEPRINT/07_TESTING.md`

### Vị trí cần sửa

1. `07_TESTING.md:8-21`
2. `07_TESTING.md:251-260`

### Hành động

1. Bổ sung unit test contract cho chunked bronze, multi-file raw pool, cross-month session boundary.
2. Bổ sung integration test contract cho dataset directories và window isolation.
3. Không viết test theo kiểu "assert no OOM" tuyệt đối trong unit tests.
4. Thay vào đó:
   - unit/integration test row-count parity
   - chunk/materialization behavior
   - benchmark memory usage như roadmap item riêng

### Patch cụ thể cho `10.1. Unit Tests`

Giữ các bullet hiện có và thêm các bullet sau ngay sau dòng `20`:

```md
* **Bronze Chunked Ingestion Contract:** Test raw source pool nhiều file vẫn được xử lý theo chunk, không yêu cầu load toàn bộ dataset vào một DataFrame duy nhất.
* **Bronze Row-Count Parity:** Test tổng số dòng valid/rejected sau bronze vẫn đúng khi input trải trên nhiều raw files.
* **Multi-File Timestamp Preservation:** Test nhiều raw files vẫn preserve đúng `source_event_time` sau bronze materialization.
* **Cross-Month Session Boundary:** Test session kéo qua ranh giới tháng vẫn được giữ nguyên một `user_session` logic ở downstream split stage.
* **Split Map Disjointness:** Test `session_split_map.parquet` luôn đảm bảo train/val/test disjoint theo `user_session`.
* **Window Isolation:** Test training window và replay/demo window không bị trộn dữ liệu.
* **Materialization Strategy Invariance:** Test thay đổi cách materialize bronze/silver không làm đổi exact counts và downstream labeling semantics.
```

### Patch cụ thể cho `10.5. Integration Tests`

Thay block `253-256` bằng:

```md
* **Raw Pool -> Bronze Dataset:** Gửi nhiều raw files tháng vào pipeline, verify `event_time` được parse thành `source_event_time` và output được materialize vào `data/bronze/` dưới dạng dataset directory.
* **Bronze Dataset -> Silver Dataset:** Verify clean/null/invalid/sort logic tạo `data/silver/` đúng và deterministic khi input là bronze dataset nhiều partitions/files.
* **Silver -> Session Split:** Verify session index được build toàn cục trên training window và cùng một `user_session` chỉ nằm trong một split.
* **Cross-Month Session Split:** Verify session đi qua ranh giới tháng vẫn thuộc đúng một split duy nhất.
* **Silver -> Gold:** Verify snapshot dataset sinh đúng features + label 10 phút tới theo split assignment.
* **Window Isolation Contract:** Verify replay/demo artifacts không bị trộn với training artifacts.
```

### Ghi chú cần thêm vào file

Thêm một câu ngắn ở cuối section testing:

```md
**Lưu ý:** Memory benchmark cho bronze step nên được theo dõi bằng benchmark/smoke test riêng hoặc CI metrics, không khóa bằng unit test cứng kiểu "không bao giờ OOM".
```

---

## 6. Patch plan cho `docs/BLUEPRINT/12_ROADMAP.md`

### Vị trí cần sửa

1. `12_ROADMAP.md:6-14`
2. `12_ROADMAP.md:22-43`
3. `12_ROADMAP.md:57-76`
4. `12_ROADMAP.md:103-105`
5. `12_ROADMAP.md:154-163`

### Hành động

1. Nâng độ cụ thể của milestone Week 1 và Week 2.
2. Sửa Docs Update Plan để phản ánh usage windows + split policy + dataset directories.
3. Sửa Code Module Plan cho bronze/silver.
4. Sửa `dvc.yaml` plan cho directory-based outputs.
5. Mở rộng Verification Plan với cross-month boundary và row parity.

### Patch cụ thể cho bảng tuần

Thay block `6-14` bằng:

```md
| Tuần | Milestone | Deliverables |
| --- | --- | --- |
| **1** | Data Foundation | `data/raw/`, chunked + partitioned `data/bronze/`, partition-aware `data/silver/`, config paths theo directory semantics, timestamp contract (`source_event_time`, `replay_time`, `prediction_time`), DVC init + MinIO remote setup |
| **2** | Training Pipeline | Global session index, session-boundary split, snapshot dataset builder, 10-minute horizon labeling, `data/gold/` artifacts, feature engineering, **multi-model training (XGBoost, LightGBM, Random Forest)**, model comparison & auto-selection, SHAP analysis, MLflow integration, **Data Lineage**, **Model Validation Gate** |
| **3** | Stream Processing | Quix Streams processor, session-scoped Redis feature store, Kafka topics, timestamp preservation, **cache invalidation logic** |
| **4** | Serving & API | FastAPI (predict + explain + health) theo `user_session`, security (API Key + rate limit), **Model Hot-Reload**, **Prediction Caching**, unit tests |
| **5** | Frontend & Dashboard | Streamlit User App + Admin Dashboard (tích hợp SHAP visualization) |
| **6** | Monitoring & CI | Prometheus + Grafana (latency panels + **6 alert rules + Webhook**), GitHub Actions (**pytest-cov ≥ 70%**), integration tests cho snapshot target và session-scoped serving |
| **7** | Polish & Demo | Demo script rehearsal (11 bước), documentation, edge case testing, bronze memory benchmark review |
```

### Patch cụ thể cho Docs Update Plan

Thay block `22-43` bằng:

```md
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
```

### Patch cụ thể cho Code Module Plan

Thay `57-65` bằng:

```md
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
```

### Patch cụ thể cho `dvc.yaml` plan

Thay `103-105` bằng:

```md
**`dvc.yaml`**
* Định nghĩa stages `bronze`, `silver`, `session_split`, `gold`, `train`.
* Bronze và silver outputs nên là directory-based artifacts.
* Track deps/outs để `dvc repro`, `dvc push`, `dvc pull` hoạt động với dataset lớn mà không yêu cầu load toàn bộ dữ liệu vào RAM.
```

### Patch cụ thể cho Verification Plan

Thay `154-163` bằng:

```md
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
```

---

## Ngoài phạm vi patch docs chính nhưng cần sync ngay sau đó

Các file dưới đây không nhất thiết phải nằm trong patch docs đầu tiên, nhưng phải được sync ngay sau khi contract mới được chốt:

1. `dvc.yaml`
2. `training/src/config.py`
3. `AGENTS.md`
4. `docs/WEEK1_SETUP.md`
5. `docs/WEEK1_IMPLEMENTATION_SUMMARY.md`
6. `docs/CHECKLIST/WEEK_1_DATA_FOUNDATION_CHECKLIST.md`

### Lý do

Các file này hiện vẫn còn contract cũ kiểu:
- `data/bronze/events.parquet`
- `data/silver/events.parquet`
- input raw theo một file đơn

Nếu không sync tiếp, blueprint docs mới sẽ lại lệch với executable/source-of-truth trong repo.

---

## Tiêu chí hoàn thành patch docs

Patch docs được xem là hoàn tất khi đạt đủ các điều kiện sau:

1. Không còn wording hard-code single-file artifact cho bronze/silver trong 6 file blueprint chính.
2. Training window, replay window, retraining window được mô tả như usage windows.
3. Split policy được mô tả downstream từ silver layer theo `session_start_time` và `user_session`.
4. Replay pipeline vẫn preserve `source_event_time`.
5. Retraining vẫn bắt buộc re-materialize qua data lake.
6. Testing và roadmap đã phản ánh chunked/partition-aware processing.
7. Root `BLUEPRINT.md` chỉ giữ summary, không bị quá chi tiết implementation.

---

## Recommended implementation sequence sau khi patch docs được duyệt

1. Patch 6 file docs blueprint chính.
2. Sync `dvc.yaml` và `training/src/config.py` theo directory semantics.
3. Đổi bronze output contract từ file đơn sang dataset directory.
4. Thiết kế lại silver theo dataset/partition-aware processing.
5. Bổ sung tests cho cross-month boundary, row parity, window isolation.
6. Sync `AGENTS.md` và Week 1 docs phụ.
