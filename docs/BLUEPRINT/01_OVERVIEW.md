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

## 2.1. Nguồn dữ liệu (The Golden Dataset)

Sử dụng bộ dữ liệu thực tế: **[eCommerce Behavior Data from Multi Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)** (Tác giả: Michael Kechinov).

* **Đặc điểm:** Chứa hàng triệu dòng log hành vi (`view`, `cart`, `remove_from_cart`, `purchase`) của người dùng thực tế.
* **Các trường quan trọng:** `event_time`, `event_type`, `product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`, `user_session`.
* **Hiện tại:** Dataset được lưu trữ ở `dataset/*.csv.gz` (7 file từ 2019-10 đến 2020-04).
* **Quy ước timestamp trong hệ thống:** `event_time` từ Kaggle được preserve thành `source_event_time`; simulator thêm `replay_time`; FastAPI thêm `prediction_time` trong response.
* **Lưu ý về trường nullable:** `category_code` có thể null (thường bỏ qua cho accessories — dùng `category_id` làm fallback). `brand` có thể null với nhiều sản phẩm không có thông tin brand.

## 2.2. Data Lake Layers (Raw → Bronze → Silver → Gold)

Luồng dữ liệu được chuẩn hóa theo 4 lớp bất biến:

* **`data/raw/`**: CSV gốc (hoặc decompress từ `dataset/*.csv.gz`), giữ nguyên 100%, không sửa nội dung.
* **`data/bronze/`**: Parquet sau parse schema, rename `event_time -> source_event_time`.
* **`data/silver/`**: Parquet đã clean, sort theo `user_session` + `source_event_time`, chuẩn hóa null/outlier.
* **`data/gold/`**: Snapshot training dataset đã có features + label cho target 10 phút tới.

**Data versioning layer:** Các artifacts của 4 lớp trên được version bằng **DVC** và lưu file thực trên object storage S3-compatible (MinIO).

**Nguyên tắc:** split train/val/test được thực hiện ở mức `user_session` boundary trên lớp silver trước khi sinh gold snapshots để tránh leakage.

## 2.3. Chiến lược Sử dụng Dữ liệu "Kép" (Dual Strategy)

Dataset được chia làm **2 phần** phục vụ 2 mục đích riêng biệt:

### Chiến lược 1: Offline Training (Huấn luyện quá khứ)

* **Dữ liệu:** `data/silver/` làm input chuẩn, sau đó materialize `data/gold/` theo split.
* **Đơn vị mẫu huấn luyện:** Không phải 1 dòng cho cả session đã kết thúc, mà là nhiều **snapshot rows** trên cùng `user_session`.
* **Quy tắc snapshot:** Với mỗi thời điểm `t` trong session, feature chỉ được tính từ các event có `source_event_time <= t`.
* **Quy tắc label:** `1` nếu cùng `user_session` có ít nhất 1 event `purchase` trong khoảng `(t, t + 10 phút]`, ngược lại `0`.
* **Chia tập dữ liệu (Temporal Split):**
  * **Training set (70%):** Các `user_session` có `session_start_time` sớm nhất — để model học quy luật hành vi.
  * **Validation set (15%):** Các `user_session` tiếp theo — để tuning hyperparameters.
  * **Test set (15%):** Các `user_session` muộn nhất — để đánh giá model cuối cùng.
* **Lưu ý:** Split được thực hiện theo **`user_session` boundary**, không phải snapshot boundary. Mỗi `user_session` chỉ được phép xuất hiện trong đúng một split.

### Chiến lược 2: Online Simulation (Tái hiện thực tại)

* **Dữ liệu:** `data/raw/` làm nguồn bất biến cho replay.
* **Kỹ thuật:** **"Data Replay"** — Script Python đọc từng dòng dữ liệu, giữ nguyên `source_event_time`, gắn thêm `replay_time`, rồi gửi vào hệ thống theo thời gian thực (giả lập độ trễ giữa các event).
* **Lợi ích:** Dashboard hiển thị dữ liệu "có hồn", có quy luật thực tế thay vì dữ liệu random vô nghĩa.

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
