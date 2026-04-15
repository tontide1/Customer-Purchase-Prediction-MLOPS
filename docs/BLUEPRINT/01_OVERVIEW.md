# 1. Tổng quan (Overview)

> **← Xem [BLUEPRINT.md gốc](../BLUEPRINT.md)**  
> **→ Xem [2. Architecture](02_ARCHITECTURE.md)**

---

## 1.1. Mục tiêu

Xây dựng hệ thống **MLOps End-to-End** xử lý luồng dữ liệu clickstream để dự đoán
`P(purchase trong 10 phút tới | trạng thái của user_session tại thời điểm t)`
trong thời gian thực với độ trễ **< 1 giây**.

## 1.2. Phạm vi hệ thống

| Khía cạnh | Mô tả |
| --- | --- |
| **Input** | Clickstream events (view, cart, remove_from_cart, purchase) từ eCommerce |
| **Output** | Purchase Probability Score (0.0 → 1.0) cho mỗi `user_session` tại một snapshot time `t` |
| **Target** | Có purchase trong **10 phút tới** trên cùng `user_session` hay không |
| **Latency mục tiêu** | < 1 giây từ `replay_time` đến `prediction_time` |
| **Throughput mục tiêu** | ~500 events/giây (đủ cho mô phỏng và demo) |
| **Deployment** | Local + Docker Compose (toàn bộ hệ thống chạy bằng 1 lệnh) |

---

# 2. Dữ liệu (Data Strategy)

> **← Xem [1. Overview](01_OVERVIEW.md)**  
> **→ Xem [2. Architecture](02_ARCHITECTURE.md)**

## 2.1. Nguồn dữ liệu

* **Raw source pool:** Dataset hiện gồm 7 file tháng từ `2019-10` đến `2020-04`, đang lưu ở `dataset/*.csv.gz` và có thể được materialize vào `data/raw/` để chạy pipeline.
* **Các trường quan trọng:** `event_time`, `event_type`, `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`.
* **Quy ước timestamp trong hệ thống:** `event_time` từ Kaggle được preserve thành `source_event_time`; simulator thêm `replay_time`; FastAPI thêm `prediction_time` trong response.

## 2.2. Data Lake Layers (Raw -> Bronze -> Silver -> Gold)

Luồng dữ liệu được chuẩn hóa theo 4 lớp bất biến:

* **`data/raw/`**: Raw source pool ở dạng CSV/CSV.GZ, giữ nguyên nội dung nguồn.
* **`data/bronze/`**: Parsed parquet artifacts sau khi rename `event_time -> source_event_time`; có thể được materialize theo chunk và partition để đảm bảo memory-safe processing.
* **`data/silver/`**: Cleaned parquet artifacts sau khi validate required fields, làm sạch dữ liệu, và sort theo semantics phục vụ downstream session indexing; có thể được materialize theo partition/window.
* **`data/gold/`**: Snapshot training datasets đã có features + label cho target 10 phút tới.

**Data versioning layer:** Các artifacts của 4 lớp trên được version bằng **DVC** và lưu file thực trên object storage S3-compatible (MinIO).

**Nguyên tắc:** split train/val/test được xác định downstream từ silver layer theo `user_session` boundary và `session_start_time`, trước khi sinh gold snapshots để tránh leakage.

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
* **Mục đích:** Mô phỏng behavior online thực tế mà không phá vỡ source timeline semantics của dữ liệu gốc.

### Retraining

* **Nguồn input:** Export events mới từ PostgreSQL theo một retraining window vận hành.
* **Yêu cầu bắt buộc:** Dữ liệu export phải được re-materialize lại qua `raw -> bronze -> silver -> gold` trước khi train.
* **Lý do:** Giữ reproducibility, lineage, và cùng semantics với offline training pipeline.

## 2.4. Data Validation (Kiểm tra chất lượng)

Trước khi đưa vào pipeline, dữ liệu được validate bằng **Pydantic schemas**:

* Kiểm tra null values cho các trường bắt buộc (`user_id`, `user_session`, `event_type`, `event_time`).
* Kiểm tra `event_type` chỉ thuộc tập `{view, cart, remove_from_cart, purchase}`.
* Kiểm tra `price > 0` và loại bỏ outlier (price > 99th percentile).
* Log các dòng bị reject kèm lý do để debug.

## 2.5. Training & Retraining Source of Truth

> **Quy tắc quan trọng:** Training và retraining luôn đi qua **data lake pipeline** (`raw -> bronze -> silver -> gold`) đã được version bằng DVC. PostgreSQL chỉ là **operational/audit store**, không phải training source trực tiếp.

| Nguồn dữ liệu | Vai trò |
|---|---|
| `data/raw/` | Source of truth cho cả training và retraining |
| `data/bronze/` | Parsed data, schema validated |
| `data/silver/` | Cleaned, sorted, session-indexed |
| `data/gold/` | Snapshot training data với features + labels |
| PostgreSQL | **Chỉ** operational store, audit log, và monitoring. Dùng để thu thập events gần đây cho retraining **nhưng phải qua bước re-materialize** trước khi train. |

**Retraining flow:**
1. Export events mới từ PostgreSQL (7-14 ngày gần nhất)
2. **Materialize** dữ liệu đó qua `bronze -> silver -> gold` pipeline với lineage metadata
3. Với artifacts đã định nghĩa trong `dvc.yaml`: chạy `dvc repro` rồi `dvc push` để persist artifacts lên MinIO remote
4. Chỉ dùng `dvc add` cho artifacts ngoài pipeline definition (ad-hoc outputs)
5. Train/retrain model từ gold artifacts (không train trực tiếp từ PostgreSQL query)

**Lý do:** Đảm bảo reproducibility và lineage hoàn chỉnh — mọi model đều có thể trace về artifacts gốc.

## 2.6. Canonical Prediction Contract

Đây là định nghĩa chuẩn phải được dùng xuyên suốt toàn bộ blueprint:

* **Entity chính:** `user_session`
* **Context phụ:** `user_id` có thể được giữ lại làm feature hoặc metadata, nhưng không phải serving key chính
* **Snapshot time `t`:** feature chỉ dùng các event đã xảy ra đến thời điểm `t`
* **Prediction target:** có purchase trong **10 phút tới** trên cùng `user_session`
* **Timestamps:**
  * `source_event_time`: thời gian gốc của event trong dataset
  * `replay_time`: thời điểm simulator bắn event vào Kafka
  * `prediction_time`: thời điểm API trả score

## 2.7. Evaluation Timeline Semantics

Để tránh nhầm lẫn giữa contract offline và demo online, cần tách rõ hai timeline:

* **Source timeline (`source_event_time`)**: dùng cho offline labeling và backfill evaluation để giữ semantics đồng nhất với training snapshots.
* **Replay/serving timeline (`replay_time`, `prediction_time`)**: dùng cho live demo monitoring và latency behavior của hệ thống đang chạy.

**Quy tắc:**
* Online evaluation trong demo replay mode có thể dùng serving timeline.
* Khi đánh giá chất lượng model để so sánh với offline training contract, ground truth phải được tính theo source timeline semantics.
* Không trộn hai semantics trong cùng một metric series.
