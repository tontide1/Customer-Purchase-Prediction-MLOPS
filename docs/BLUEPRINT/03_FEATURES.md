# 3. Feature Engineering

> **← Xem [2. Architecture](02_ARCHITECTURE.md)**  
> **→ Xem [4. Pipelines](04_PIPELINES.md)**

---

## 3.1. Offline Features (Training)

Training set được tạo dưới dạng **snapshot theo thời điểm** trên từng `user_session`, không phải 1 dòng tổng kết cho cả session đã kết thúc.

Với mỗi snapshot time `t`:

* Chỉ dùng các event trong cùng `user_session` có `source_event_time <= t`
* Mỗi dòng training đại diện cho trạng thái session tại đúng thời điểm `t`
* **Label:** `1` nếu cùng `user_session` có ít nhất 1 event `purchase` trong khoảng `(t, t + 10 phút]`, ngược lại `0`

Các feature chính:

| Feature | Mô tả | Ý nghĩa |
| --- | --- | --- |
| `total_views` | Tổng số lần `view` trong session **đến thời điểm `t`** | Mức độ quan tâm tổng thể |
| `total_carts` | Tổng số lần `cart` trong session **đến thời điểm `t`** | Ý định mua hàng (raw) |
| `total_removes` | Tổng số lần `remove_from_cart` trong session **đến thời điểm `t`** | Mức độ do dự / thay đổi ý kiến |
| `net_cart_count` | `total_carts - total_removes` | Số item thực sự còn trong giỏ |
| `cart_to_view_ratio` | `total_carts / total_views` | Tỷ lệ chuyển đổi — feature quan trọng nhất |
| `cart_remove_rate` | `total_removes / total_carts` (0 nếu `total_carts = 0`) | Tỷ lệ từ bỏ item đã thêm vào giỏ |
| `unique_categories` | Số category khác nhau đã xem **đến thời điểm `t`** (dùng `category_code` nếu có, fallback sang `category_id` khi null) | Browsing đa dạng hay tập trung |
| `unique_products` | Số product khác nhau đã xem **đến thời điểm `t`** | Đang so sánh hay đã chọn |
| `avg_price_viewed` | Giá trung bình sản phẩm đã xem **đến thời điểm `t`** | Phân khúc khách hàng |
| `session_duration_sec` | `t - source_session_start_time` | Session đang kéo dài bao lâu tại lúc dự đoán |
| `time_since_last_event_sec` | `t - source_previous_event_time` | Hành vi vội vã hay thong thả |

> **Quy tắc train/serve alignment:** Mọi feature offline ở trên phải có online equivalent được cập nhật incrementally theo cùng semantics. Không dùng bất kỳ thông tin "cuối session" nào trong training nếu online serving không nhìn thấy thông tin đó tại thời điểm `t`.

> **Optional feature — `has_brand_info` (cần verify qua EDA trước khi thêm):** `1` nếu session có ít nhất 1 sản phẩm có thông tin brand, else `0`. Xem `notebook-planned/02_feature_experiment.ipynb` để đánh giá correlation với purchase intent trước khi commit vào feature set chính thức.
>
> **Optional feature — `price_std` (cần verify qua EDA trước khi thêm):** Độ lệch chuẩn giá sản phẩm đã xem đến thời điểm `t`. Để tính online, cần lưu thêm `price_sum_sq` trong Redis (tổng bình phương giá), sau đó tính: `std = sqrt(price_sum_sq / view_count - avg_price_viewed²)`. Nếu EDA cho thấy feature importance cao thì thêm vào Redis keys.

---

## 3.2. Real-time Features (Inference)

Features được tính incrementally bởi Quix Streams, lưu vào Redis theo đúng entity `user_session`:

| Redis Key Pattern | Giá trị | Cập nhật khi |
| --- | --- | --- |
| `session:{user_session}:view_count` | int | Mỗi event `view` |
| `session:{user_session}:cart_count` | int | Mỗi event `cart` |
| `session:{user_session}:remove_count` | int | Mỗi event `remove_from_cart` |
| `session:{user_session}:unique_products` | Redis Set (`SADD`) | Mỗi event `view` — **exact count** |
| `session:{user_session}:unique_categories` | Redis Set (`SADD`) | Mỗi event `view` — **exact count** |
| `session:{user_session}:avg_price_viewed` | float (running avg) | Mỗi event `view` |
| `session:{user_session}:source_session_start_time` | timestamp | Event đầu tiên của session |
| `session:{user_session}:source_last_event_time` | timestamp | Mỗi event |
| `session:{user_session}:last_replay_time` | timestamp | Mỗi event |
| `session:{user_session}:user_id` | string | Event đầu tiên của session hoặc khi refresh context |

> **Exact count semantics:** `unique_products` và `unique_categories` dùng Redis Set (`SADD`) thay vì HyperLogLog để đảm bảo **exact count**. Lấy số lượng bằng `SCARD`. Điều này đảm bảo train/serve parity hoàn toàn chính xác.

> **Derived features tại inference time:** `net_cart_count`, `cart_remove_rate`, `cart_to_view_ratio`, `session_duration_sec`, và `time_since_last_event_sec` không nhất thiết phải lưu riêng trong Redis. Chúng có thể được tính tại thời điểm gọi API từ session state hiện tại.

> **Timestamp semantics:** Redis state luôn giữ riêng timeline gốc của dữ liệu (`source_session_start_time`, `source_last_event_time`) và timeline xử lý hiện tại (`last_replay_time`). Điều này giúp vừa serve real-time, vừa đo latency và debug một cách đúng nghĩa.
