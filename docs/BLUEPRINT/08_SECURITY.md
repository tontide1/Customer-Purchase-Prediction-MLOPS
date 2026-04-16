# 8. Security Considerations

> **← Xem [7. Testing](07_TESTING.md)**  
> **→ Xem [9. Explainability](09_EXPLAINABILITY.md)**

> **Execution profile (local dev): `DEV_SMOKE`**
> - Train window (dev): `2019-10` -> `2019-10`
> - Replay window (dev): `2020-03` -> `2020-03`
> - Profile này chỉ để tăng tốc vòng lặp phát triển; canonical target-state windows trong blueprint vẫn giữ nguyên.

> **Lưu ý:** Hệ thống chạy trên local (Docker Compose internal network), không expose ra internet.
> Các biện pháp security dưới đây ở mức phù hợp cho project demo, thể hiện nhận thức security mà không over-engineering.

---

## 8.1. API Key Authentication

Mọi request đến FastAPI phải kèm header `X-API-Key`. Đơn giản, hiệu quả, phù hợp cho internal services:

```python
# services/prediction-api/app/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != get_settings().api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key
```

* API Key được lưu trong `.env`, **KHÔNG** hardcode trong source code.
* Streamlit Dashboard và User App đọc API Key từ `.env` chung (vì cùng Docker Compose network).

---

## 8.2. Rate Limiting

Sử dụng **slowapi** để giới hạn số request, tránh abuse hoặc vòng lặp vô hạn từ client:

```python
# services/prediction-api/app/main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/api/v1/predict/{user_session}")
@limiter.limit("60/minute")
async def predict(user_session: str, request: Request):
    ...
```

* Mặc định: **60 requests/phút** per IP (configurable qua `.env`).
* Response khi vượt limit: HTTP 429 Too Many Requests.

---

## 8.3. Input Validation

Đã tích hợp xuyên suốt hệ thống (không phải section riêng — đây là best practice):

* **Pydantic schemas** validate tất cả API request/response.
* **Event schemas** validate data từ Kafka trước khi xử lý.
* **`user_session`** được sanitize: chỉ chấp nhận alphanumeric + underscore, tối đa 100 ký tự.
* **`user_id`** vẫn được validate trong event payload vì nó là context quan trọng cho analytics và future feature expansion, nhưng không còn là serving identifier chính.

---

## 8.4. Secrets Management

| Secret | Lưu trữ | Truy cập |
| --- | --- | --- |
| `API_KEY` | `.env` file (gitignored) | `pydantic-settings` |
| `REDIS_PASSWORD` | `.env` file (gitignored) | `pydantic-settings` |
| `POSTGRES_PASSWORD` | `.env` file (gitignored) | `pydantic-settings` |
| `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | `.env` file (gitignored) | Docker Compose env + init container |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (DVC remote) | `.env` file (gitignored) | DVC remote config |
| MLflow artifacts | Docker volume (local) | Internal network only |

**`.env.example`** chứa template với giá trị placeholder — commit vào repo để team member biết cần config gì.

**Logging rule:** Tuyệt đối không log giá trị credentials của MinIO/DVC (`MINIO_*`, `AWS_*`) ra stdout, application logs, hoặc notebook output.
