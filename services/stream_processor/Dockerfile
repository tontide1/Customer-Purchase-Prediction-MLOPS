FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY shared ./shared
COPY training ./training
COPY services ./services
COPY services/stream_processor/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -e . && python -m pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["python", "-m", "services.stream_processor.app"]
