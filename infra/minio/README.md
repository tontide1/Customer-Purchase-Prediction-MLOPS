# MinIO bootstrap (target-state)

Thư mục này chứa script khởi tạo bucket MinIO dùng làm DVC remote object storage.

## Files

- `init-bucket.sh`: Script chạy trong service `minio-init` của `docker-compose.yml`.

## Mục tiêu

- Tạo bucket `mlops-artifacts` (hoặc giá trị từ `MINIO_BUCKET`).
- Đặt policy private cho bucket.
- Sẵn sàng cho `dvc push`/`dvc pull`.
- Lưu MLflow run artifacts dưới prefix `mlflow/` khi chạy local tracking server.

## Lưu ý

Đây là target-state scaffold cho local/demo environment, có thể cần điều chỉnh credentials, network, hoặc policy cho production.
