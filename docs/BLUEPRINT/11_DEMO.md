# 14. Kịch bản Demo (The Demo Script)

> **← Xem [10. Performance](10_PERFORMANCE.md)**  
> **→ Xem [12. Roadmap](12_ROADMAP.md)**

Khi báo cáo với giảng viên, thực hiện theo trình tự:

---

## Bước 1: Khởi động hệ thống

* Show file `docker-compose.yml` và chạy: `docker compose up -d`.
* Mở **Grafana Dashboard** → Chỉ cho thấy tất cả services đều healthy (xanh).
* Giải thích: *"Toàn bộ hệ thống gồm 10 containers, khởi động bằng 1 lệnh duy nhất."*

---

## Bước 2: Demo Training Pipeline (MLOps — Multi-Model)

* Chạy `train.py` (hoặc load model đã train sẵn).
* Mở **MLflow UI** → Show experiment tracking:
  * **So sánh 3 models** (XGBoost, LightGBM, Random Forest) — bảng so sánh metrics ngay trên MLflow UI.
  * Show metrics từng model: PR-AUC, F1-Score, Confusion Matrix.
  * Highlight **best model được tự động chọn** và register lên Production.
  * Show model versioning (v1, v2, ...) và model type (XGBoost/LightGBM/Random Forest).
  * **Show SHAP Feature Importance chart** (artifact trên MLflow).
* Giải thích: *"Hệ thống tự động train 3 models, so sánh hiệu suất, và chọn model tốt nhất. Trong lần này, [model_name] đạt PR-AUC cao nhất. Feature quan trọng nhất là cart_to_view_ratio."*

---

## Bước 3: Bắt đầu Real-time Simulation

* Chạy `simulator.py`.
* Mở **Admin Dashboard**:
  * Chỉ cho giảng viên thấy các con số (View, Cart, Active Users) đang cập nhật real-time.
  * Show biểu đồ Purchase Probability trong **10 phút tới** phân phối theo thời gian.
* Mở **Grafana** → Show API latency: *"p95 = ~200ms, đạt target dưới 1 giây."*
* Giải thích: *"Đây là hành vi user thực tế từ bộ dữ liệu Kaggle đang được tái hiện với tốc độ 100 events/giây. Mỗi prediction là cho trạng thái hiện tại của một user_session, không phải hindsight sau khi session kết thúc."*

---

## Bước 4: Tương tác & Dự đoán

* Đóng vai một session mới (`DEMO_SESSION_001`) của user `DEMO_USER`.
* Vào **User App**, thực hiện kịch bản:
  1. **View** iPhone liên tục 5 lần → Dashboard: Score mua trong 10 phút tới tăng nhẹ (~30%).
  2. **Add to Cart** → Dashboard: Score nhảy vọt (~75%).
  3. **View thêm** 2 lần nữa → Dashboard: Score tiếp tục tăng (~90%).
* Giải thích: *"Hệ thống phát hiện pattern: xem nhiều + thêm vào giỏ = ý định mua cao trong 10 phút tới. Đây là kết hợp kiến thức quá khứ và hành vi hiện tại ở đúng thời điểm đang diễn ra."*

---

## Bước 5: Demo Explainability

* Gọi API: `GET /api/v1/explain/DEMO_SESSION_001` (có thể dùng Swagger UI hoặc Dashboard).
* Show response: *"Top 3 lý do model dự đoán 90% khả năng mua trong 10 phút tới: (1) cart_to_view_ratio = 0.14 → +25%, (2) total_views = 7 → +15%, (3) session_duration dài → +10%."*
* Giải thích: *"Model không phải hộp đen. Chúng ta giải thích được TẠI SAO model đưa ra dự đoán, nhờ SHAP values."*

---

## Bước 6: Demo Edge Cases

* **Case 1 — Window Shopping:** Session xem 20 sản phẩm khác nhau, không cart → Score mua trong 10 phút tới thấp (~15%).
  * Explain: `unique_products = 20` → giảm score, `cart_to_view_ratio = 0` → giảm mạnh.
* **Case 2 — Impulse Buy:** Session xem 1 sản phẩm, cart ngay → Score cao (~70%).
  * Explain: `cart_to_view_ratio = 1.0` → tăng mạnh score.
* Giải thích: *"Model không chỉ dựa vào view count, mà kết hợp nhiều features. SHAP giúp chúng ta hiểu logic bên trong."*

---

## Bước 7: Demo Security

* Gọi API **không có** API Key → Show HTTP 401 Unauthorized.
* Gọi API **có** API Key sai → Show HTTP 401 Unauthorized.
* Gọi API **có** API Key đúng → Show prediction thành công.
* Giải thích: *"API được bảo vệ bằng API Key và Rate Limiting. Mọi secret nằm trong .env, không hardcode."*

---

## Bước 8: Demo Prediction Caching

* Gọi `GET /api/v1/predict/DEMO_SESSION_001` **2 lần liên tiếp** (dùng Swagger UI hoặc curl).
* **Lần 1:** Response có `"cached": false` — prediction được tính mới từ model.
* **Lần 2:** Response có `"cached": true` — trả từ Redis cache, latency ~1-2ms.
* Mở **User App** → Thực hiện thêm 1 hành động (ví dụ: **View** thêm 1 sản phẩm).
* Gọi predict lần 3 → Response có `"cached": false` — cache đã bị invalidate vì features mới.
* Giải thích: *"Prediction Caching giúp giảm latency 50-100 lần cho repeated requests trên cùng session. Cache tự xóa khi session có hành vi mới — đảm bảo prediction luôn dựa trên features mới nhất."*

---

## Bước 9: Demo Model Hot-Reload

* Chạy lại `train.py` với dữ liệu mới (hoặc hyperparameter khác) → Model v2 được train.
* Mở **MLflow UI** → Show **Model Validation Gate** log:
  * `validation_gate_passed: 1`, `pr_auc_improvement: +0.02`.
  * Model v2 được transition sang stage `Production`.
* **Không restart** FastAPI service — đợi 5 phút (hoặc giảm `model_reload_interval` xuống 30s cho demo).
* Gọi `GET /api/v1/predict/DEMO_SESSION_001` → Kiểm tra `model_version` đã thay đổi từ `v1` sang `v2`.
* Giải thích: *"Model mới tự động được deploy mà không cần restart service. Background thread poll MLflow mỗi 5 phút. Trong production, điều này đảm bảo zero-downtime model update."*

---

## Bước 10: Demo Grafana Alerting

* Mở **Grafana** → Tab **Alerting** → Show 6 alert rules đã cấu hình.
* Nếu có thể: trigger thủ công bằng cách tạm dừng `stream-processor` → Kafka consumer lag tăng → Alert `Kafka Consumer Lag` chuyển sang trạng thái **Firing** (🔴).
* Restart `stream-processor` → Alert tự resolve → **Resolved** (✅).
* Show **Notification history** → Webhook payload đã được gửi.
* Giải thích: *"Grafana tự động phát hiện anomaly và gửi cảnh báo. Trong môi trường thực, webhook có thể kết nối với Slack, PagerDuty, hoặc Email."*

---

## Bước 11: Kết luận

* Tổng kết kiến trúc: Event-Driven → Stream Processing → Real-time Prediction + Explanation cho `user_session`.
* Nhấn mạnh **Multi-Model Experimentation + Closed-Loop MLOps** — điểm khác biệt:
  * **Multi-Model:** Train & compare 3 models (XGBoost, LightGBM, Random Forest), tự động chọn best model.
  * **MLflow:** Model versioning + experiment tracking + **Data Lineage** + **model comparison UI**.
  * **Model Validation Gate:** Tự động so sánh model mới vs cũ, chặn model kém.
  * **Model Hot-Reload:** Zero-downtime model update (background thread poll mỗi 5 phút).
  * **Prediction Caching:** Redis cache TTL 30s + auto-invalidation khi session features update.
  * **SHAP:** Model explainability (global + local) — tương thích với mọi tree-based model.
  * **Prometheus/Grafana:** Performance monitoring + **6 alert rules** + Webhook notification.
  * **GitHub Actions:** CI pipeline với **70% coverage threshold**.
  * **Docker Compose:** One-command deployment.
* *"Hệ thống không chỉ dùng 1 model cố định — mà tự động thử nghiệm 3 models, chọn best, và deploy cho bài toán: dự đoán khả năng purchase trong 10 phút tới từ trạng thái hiện tại của user_session. Nếu model degrade, hệ thống tự phát hiện drift (PSI/KL) → Grafana alert → Retrain 3 models → Validation Gate verify → Auto hot-reload. Đây là closed-loop MLOps."*
* *"Hệ thống hoạt động hoàn toàn trên local, không phụ thuộc cloud, và có thể mở rộng khi cần."*
