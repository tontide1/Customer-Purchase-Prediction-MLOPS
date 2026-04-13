# 13. Performance Measurement

> **← Xem [9. Explainability](09_EXPLAINABILITY.md)**  
> **→ Xem [11. Demo](11_DEMO.md)**

---

## 13.1. Cách đo Latency

Thay vì dùng load testing tools bên ngoài (Locust, k6), hệ thống **tự đo liên tục** thông qua Prometheus metrics đã tích hợp:

* **FastAPI middleware** tự động ghi lại request duration cho mọi endpoint.
* **Prometheus** scrape metrics mỗi 15 giây.
* **Grafana dashboard** hiển thị latency percentiles real-time.

```python
# services/prediction-api/app/main.py
from prometheus_fastapi_instrumentator import Instrumentator

# Tự động thu thập: request count, latency histogram, response size
Instrumentator().instrument(app).expose(app)
```

---

## 13.2. Metrics cần theo dõi

> **Contract note:** Model-quality metrics (PR-AUC/F1/Precision/Recall) chỉ tính trên predictions có `prediction_mode='model'`; fallback predictions bị loại khỏi các metric này.

> **Mode note:** Theo dõi tách riêng theo `evaluation_mode` (`demo_replay` vs `offline_backfill`), không trộn hai mode vào cùng một metric series.

| Metric | Target | Đo bằng |
| --- | --- | --- |
| **API Latency (p50)** | < 100ms | Prometheus histogram |
| **API Latency (p95)** | < 300ms | Prometheus histogram |
| **API Latency (p99)** | < 1000ms | Prometheus histogram |
| **End-to-end System Latency** | < 1 giây | Timestamp diff (`replay_time` → `prediction_time`) |
| **Data Freshness Gap** | Theo dõi, không hard SLO | Timestamp diff (`source_event_time` → `prediction_time`) |
| **Kafka Consumer Lag** | < 100 messages | Kafka JMX metrics |
| **Redis GET Latency** | < 5ms | Redis INFO stats |
| **Throughput** | > 500 events/s | Prometheus counter |
| **Fallback Rate** | < 5% sustained | Tỷ lệ `prediction_mode='fallback'` theo thời gian |

---

## 13.3. Cách verify target "< 1 giây" khi demo

1. Mở **Grafana Dashboard** → Panel "API Latency Percentiles".
2. Trong khi simulator đang chạy, chỉ cho giảng viên thấy:
   * p50 latency ~50-80ms
   * p95 latency ~150-250ms
   * p99 latency < 500ms
3. Nếu cần giải thích sâu hơn, phân biệt rõ:
   * **System latency:** từ lúc event được replay vào hệ thống đến lúc API trả score
   * **Data freshness gap:** khoảng cách giữa thời gian gốc của dữ liệu và thời điểm dự đoán khi replay dataset quá khứ
4. Giải thích: *"Prometheus thu thập metrics mỗi 15 giây. Đây là bằng chứng liên tục, không phải benchmark chạy 1 lần."*
