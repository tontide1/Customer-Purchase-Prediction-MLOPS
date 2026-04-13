# Week 1 Sprint Checklist: Data Foundation

Mục tiêu của tuần 1 là hoàn thành nền tảng dữ liệu tối thiểu nhưng chạy được thật, bám theo milestone `Data Foundation` trong `docs/BLUEPRINT/12_ROADMAP.md`.

Phạm vi tuần 1:

- Thiết lập `data/raw/`, `data/bronze/`, `data/silver/`
- Khóa timestamp contract: `source_event_time`, `replay_time`, `prediction_time`
- Thiết lập config path dùng chung cho training pipeline
- Hoàn thiện DVC + MinIO foundation để chuẩn bị cho các tuần sau

Ngoài phạm vi tuần 1:

- `session_split`
- `gold` snapshots
- `train`, `evaluate`, `MLflow`
- stream processing, API, dashboard

## 1. Sprint Outcomes

- [ ] Có cấu trúc thư mục dữ liệu rõ ràng cho `raw/bronze/silver/gold`
- [ ] Có shared contract cho schema và constants
- [ ] Có `training/src/config.py` để quản lý path và DVC/MinIO config
- [ ] Có `training/src/bronze.py` chạy được từ raw input sang bronze parquet
- [ ] Có `training/src/silver.py` chạy được từ bronze sang silver parquet
- [ ] `dvc.yaml` chỉ phản ánh các stage tuần 1 có thể chạy thật
- [ ] MinIO local bootstrap chạy được bằng `docker compose up -d`
- [ ] DVC remote có thể `push` và `pull` artifacts
- [ ] Docs blueprint liên quan đến tuần 1 được đồng bộ với implementation thực tế

## 2. Task Breakdown Theo File

### 2.1. Repository Scaffold

**Files/paths:**

- [ ] `data/raw/`
- [ ] `data/bronze/`
- [ ] `data/silver/`
- [ ] `data/gold/`
- [ ] `training/src/`
- [ ] `training/tests/`
- [ ] `shared/`

**Công việc:**

- [ ] Tạo các thư mục còn thiếu để phản ánh đúng data lake layout
- [ ] Bảo đảm `data/raw/` được dùng làm input làm việc chính, không đọc trực tiếp từ `dataset/` trong pipeline chính
- [ ] Thêm placeholder tối thiểu nếu cần để giữ cấu trúc repo rõ ràng

**Definition of done:**

- [ ] Repo có đủ khung thư mục để bắt đầu code tuần 1 mà không phải đoán cấu trúc

### 2.2. Shared Constants

**File:** `shared/constants.py`

**Công việc:**

- [ ] Định nghĩa tên các data layers: `raw`, `bronze`, `silver`, `gold`
- [ ] Định nghĩa tên artifact mặc định, ví dụ `events.parquet`
- [ ] Định nghĩa timestamp field names dùng chung: `event_time`, `source_event_time`, `replay_time`, `prediction_time`
- [ ] Định nghĩa allowed event types: `view`, `cart`, `remove_from_cart`, `purchase`

**Definition of done:**

- [ ] Các script tuần 1 không hard-code string lặp lại cho layer name hoặc timestamp name

### 2.3. Shared Schemas

**File:** `shared/schemas.py`

**Công việc:**

- [ ] Tạo raw event schema cho input CSV với field gốc `event_time`
- [ ] Tạo internal/bronze schema với field `source_event_time`
- [ ] Tách rõ field raw và internal, không trộn `event_time` với `source_event_time`
- [ ] Xác định field bắt buộc tối thiểu: `event_time`, `event_type`, `product_id`, `user_id`, `user_session`
- [ ] Chấp nhận field nullable hợp lệ như `category_code`, `brand`

**Definition of done:**

- [ ] Schema đủ rõ để dùng lại cho `bronze.py` và cho docs/testing tuần 1

### 2.4. Training Config

**File:** `training/src/config.py`

**Công việc:**

- [ ] Thêm path config: `raw_data_path`, `bronze_data_path`, `silver_data_path`, `gold_data_dir`
- [ ] Thêm `prediction_horizon_minutes = 10` để khóa contract sớm
- [ ] Thêm DVC/MinIO config: `dvc_remote_name`, `dvc_remote_url`, `s3_endpoint_url`
- [ ] Thêm `aws_access_key_id`, `aws_secret_access_key` nếu team chọn load qua settings
- [ ] Đảm bảo config dùng `.env` và có default phù hợp local demo

**Definition of done:**

- [ ] Mọi script tuần 1 đọc path qua config thay vì hard-code rải rác

### 2.5. Raw Intake Rule

**Files/paths:**

- [ ] `data/raw/`
- [ ] `dataset/*.csv.gz`
- [ ] tài liệu liên quan trong `docs/BLUEPRINT/`

**Công việc:**

- [ ] Chốt một cách chuẩn để populate `data/raw/` từ `dataset/*.csv.gz`
- [ ] Không hard-code `2019-Oct.csv` trong code hoặc docs
- [ ] Ghi rõ raw layer là immutable: không sửa nội dung record tại đây
- [ ] Xác định script tuần 1 sẽ đọc từ file đơn, nhiều file, hay cả thư mục

**Definition of done:**

- [ ] Có quy ước nhập dữ liệu vào `data/raw/` rõ ràng và nhất quán

### 2.6. Bronze Pipeline

**File:** `training/src/bronze.py`

**Công việc:**

- [ ] Nhận input từ config hoặc CLI argument
- [ ] Đọc raw CSV từ `data/raw/`
- [ ] Parse schema đầu vào
- [ ] Rename `event_time -> source_event_time`
- [ ] Validate `event_type` thuộc tập allowed values
- [ ] Ghi output sang `data/bronze/events.parquet`
- [ ] Log số dòng input/output và số dòng reject nếu có

**Definition of done:**

- [ ] Chạy được độc lập để tạo `data/bronze/events.parquet`
- [ ] Bronze output chứa `source_event_time`, không còn dùng `event_time` làm field chính nội bộ

### 2.7. Silver Pipeline

**File:** `training/src/silver.py`

**Công việc:**

- [ ] Đọc `data/bronze/events.parquet`
- [ ] Loại bỏ record thiếu field bắt buộc
- [ ] Loại bỏ `price <= 0`
- [ ] Xử lý outlier theo rule đã chốt cho tuần 1
- [ ] Sort deterministic theo `user_session` + `source_event_time`
- [ ] Chuẩn hóa field nullable nếu cần ở mức tối thiểu
- [ ] Ghi output sang `data/silver/events.parquet`
- [ ] Log row count trước và sau clean

**Definition of done:**

- [ ] Chạy được độc lập để tạo `data/silver/events.parquet`
- [ ] Silver output deterministic khi chạy lại cùng input

### 2.8. DVC Pipeline Definition

**File:** `dvc.yaml`

**Công việc:**

- [ ] Rà soát lại các stage đang khai báo
- [ ] Giữ lại hoặc điều chỉnh để chỉ còn stage tuần 1 chạy được thật: `bronze`, `silver`
- [ ] Khai báo đúng `deps` và `outs` cho từng stage
- [ ] Tránh để `dvc repro` phụ thuộc vào file code chưa tồn tại của tuần 2

**Definition of done:**

- [ ] `dvc repro` không fail vì stage tuần 2 chưa được implement

### 2.9. Environment Config

**File:** `.env.example`

**Công việc:**

- [ ] Kiểm tra đầy đủ `MINIO_*`
- [ ] Kiểm tra đầy đủ `AWS_*`
- [ ] Bổ sung biến còn thiếu nếu `training/src/config.py` yêu cầu
- [ ] Giữ phạm vi secrets đúng cho training/data pipeline, không trộn sang prediction API runtime contract

**Definition of done:**

- [ ] `.env.example` đủ để dựng MinIO và cấu hình DVC remote local

### 2.10. MinIO Bootstrap

**Files:**

- [ ] `docker-compose.yml`
- [ ] `infra/minio/init-bucket.sh`

**Công việc:**

- [ ] Verify service `minio` và `minio-init` khớp với env vars
- [ ] Verify bucket được tạo đúng theo `MINIO_BUCKET`
- [ ] Verify policy bucket là private
- [ ] Đảm bảo ports `9000` và `9001` được expose đúng cho local demo

**Definition of done:**

- [ ] `docker compose up -d` dựng được MinIO và bucket init thành công

### 2.11. Data Layer Tests

**Files:**

- [ ] `training/tests/test_data_lake.py`
- [ ] `docs/BLUEPRINT/07_TESTING.md`

**Công việc:**

- [ ] Thêm test raw -> bronze: `event_time` được chuyển thành `source_event_time`
- [ ] Thêm test bronze -> silver: clean/sort hoạt động đúng
- [ ] Thêm test deterministic ordering theo `user_session` + `source_event_time`
- [ ] Thêm test reject cho `event_type` không hợp lệ
- [ ] Thêm test timestamp contract được preserve đúng

**Definition of done:**

- [ ] Có test nền tảng bao phủ ít nhất luồng raw -> bronze -> silver

### 2.12. Blueprint Docs Sync

**Files:**

- [ ] `BLUEPRINT.md`
- [ ] `docs/BLUEPRINT/01_OVERVIEW.md`
- [ ] `docs/BLUEPRINT/04_PIPELINES.md`
- [ ] `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- [ ] `docs/BLUEPRINT/07_TESTING.md`

**Công việc:**

- [ ] Giữ wording nhất quán giữa `raw/bronze/silver/gold`
- [ ] Xác nhận split theo `user_session` boundary được mô tả là việc của tuần 2, không phải output tuần 1
- [ ] Xác nhận timestamp contract được nêu nhất quán
- [ ] Bỏ hard-code `2019-Oct.csv` nếu còn sót ở docs/code
- [ ] Đồng bộ project structure với những gì repo thật sự có hoặc sắp có ngay sau tuần 1

**Definition of done:**

- [ ] Docs không hứa hẹn sai về những gì tuần 1 đã hoàn thành

## 3. Verification Checklist

### 3.1. Functional Verification

- [ ] `docker compose up -d` chạy thành công cho MinIO scaffold
- [ ] `docker compose ps` hiển thị MinIO healthy
- [ ] Có thể truy cập MinIO console qua port `9001`
- [ ] `training/src/bronze.py` chạy thành công trên input mẫu
- [ ] `training/src/silver.py` chạy thành công trên bronze artifact
- [ ] `dvc repro` chạy được cho stage tuần 1
- [ ] `dvc push` đẩy được artifact lên MinIO
- [ ] `dvc pull` lấy lại được artifact trên workspace sạch

### 3.2. Contract Verification

- [ ] Raw layer vẫn dùng field gốc `event_time`
- [ ] Bronze/Silver layer dùng `source_event_time`
- [ ] Không có file nào hard-code `2019-Oct.csv`
- [ ] Không có stage tuần 2 làm vỡ pipeline tuần 1
- [ ] Config path được tập trung tại `training/src/config.py`

### 3.3. Data Quality Verification

- [ ] `event_type` ngoài tập cho phép bị reject
- [ ] Các record thiếu `user_session` hoặc `user_id` được xử lý theo rule đã chốt
- [ ] `price <= 0` không đi vào silver
- [ ] Silver được sort deterministic

## 4. Suggested Execution Order

- [ ] Bước 1: tạo scaffold thư mục và shared modules
- [ ] Bước 2: hoàn thiện `training/src/config.py`
- [ ] Bước 3: chốt raw intake rule cho `data/raw/`
- [ ] Bước 4: implement `training/src/bronze.py`
- [ ] Bước 5: implement `training/src/silver.py`
- [ ] Bước 6: cập nhật `dvc.yaml`
- [ ] Bước 7: verify MinIO + DVC remote
- [ ] Bước 8: thêm tests nền tảng
- [ ] Bước 9: sync docs blueprint liên quan

## 5. Exit Criteria Cho Sprint Tuần 1

- [ ] Repo có data foundation rõ ràng và chạy được tối thiểu end-to-end
- [ ] Artifacts `bronze` và `silver` có thể tái tạo bằng DVC
- [ ] Timestamp contract được khóa thống nhất giữa code và docs
- [ ] Sprint có thể bàn giao sang tuần 2 mà không cần sửa lại nền dữ liệu
