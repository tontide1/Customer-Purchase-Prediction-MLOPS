# 6. Error Handling & Logging

> **← Xem [5. Project Structure](05_PROJECT_STRUCTURE.md)**
> **→ Xem [7. Testing](07_TESTING.md)**

> **Execution profile (local dev): `DEV_SMOKE`**
> - Train window (dev): `2019-10` -> `2019-10`
> - Replay window (dev): `2019-11` -> `2019-11`
> - Profile này chỉ để tăng tốc vòng lặp phát triển; canonical target-state windows trong blueprint vẫn giữ nguyên.

---

## 6.1. Error Handling Strategy

> **Contract note:** Blueprint snippets ở file này là target-state contract, không phải code đã runnable ngay trong repo hiện tại.

| Component | Lỗi có thể xảy ra | Xử lý |
| --- | --- | --- |
| **Simulator** | CSV row invalid | Log warning, skip row, tiếp tục |
| **Kafka Producer** | Broker unreachable | Retry 3 lần (exponential backoff), sau đó log error |
| **Stream Processor** | Feature calculation error | Gửi event vào DLQ, log error, tiếp tục |
| **Redis** | Connection timeout | Retry 2 lần, nếu fail → API trả fallback score |
| **FastAPI `/predict`** | Model load fail | Trả score mặc định `0.5` với `confidence: "low"`, `prediction_mode: "fallback"`, `fallback_reason: "model_unavailable"` |
| **FastAPI `/predict`** | Redis miss hoặc Redis timeout | Trả score mặc định `0.5` với `prediction_mode: "fallback"`, `fallback_reason: "redis_miss"` |
| **FastAPI** | Invalid user_session | Return HTTP 404 kèm message rõ ràng |
| **FastAPI `/explain`** | SHAP explainer fail hoặc chưa load | Return HTTP 503 với error code `EXPLAINER_UNAVAILABLE` |
| **Model Hot-Reload** | MLflow unreachable during poll | Log warning, giữ model hiện tại, retry next cycle |
| **Prediction Cache** | Redis unavailable for cache ops | Skip cache, predict trực tiếp; fallback responses không được cache |
| **Grafana Alerting** | Webhook endpoint down | Grafana retry built-in (3 lần), log failed delivery |

**Canonical rule:** `/predict` có degraded mode; `/explain` không silent-degrade thành prediction response.

---

## 6.2. Structured Logging

Sử dụng **Loguru** với format JSON, thống nhất across all services:

```python
# Format: timestamp | service | level | message | extra_fields
{"time": "2024-01-15T10:30:00Z", "service": "stream-processor", "level": "INFO", "message": "Feature updated", "user_session": "session_12345", "user_id": "12345", "source_event_time": "2019-10-01T10:29:58Z", "replay_time": "2024-01-15T10:30:00Z", "view_count": 7}
```

---

## 6.3. Log Sanitization

> **Quy tắc quan trọng:** KHÔNG BAO GIỜ log thông tin nhạy cảm, dù đã dùng `.env`.

* **KHÔNG log:** API Key, Redis password, PostgreSQL password, raw `.env` values.
* **Được log:** `user_session`, `user_id`, `event_type`, `source_event_time`, `replay_time`, `prediction_time`, feature values, prediction score, model version.
* **Kiểm tra trước khi commit:** Review log output để đảm bảo không có secret bị leak vào file log hoặc stdout.
