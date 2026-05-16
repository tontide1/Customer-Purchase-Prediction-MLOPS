FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY services ./services
COPY services/prediction_api/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -e . && python -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "services.prediction_api.app:create_runtime_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
