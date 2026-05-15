# Week 3 Infra CI Docs And Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Week 3 services into local compose, add CI/static checks, update runbooks and blueprint docs, and run final verification.

**Architecture:** This plan owns the operational wrapper around the vertical slice. It keeps existing MinIO and MLflow behavior unchanged, adds explicit Redpanda topic creation plus Redis/PostgreSQL/API services, and updates CI and docs only after the service contracts exist.

**Tech Stack:** Docker Compose, Redpanda, Redis, PostgreSQL, FastAPI, pytest, Ruff, DVC.

---

### Task 10: Compose Infrastructure And Explicit Topic Creation

**Files:**
- Modify: `docker-compose.yml`
- Create: `infra/redpanda/init-topics.sh`
- Modify: `.env.example`
- Test: `services/tests/test_compose_contract.py`

- [ ] **Step 1: Write the failing contract test**

Create `services/tests/test_compose_contract.py`:

```python
"""Static compose contract tests for Week 3 services."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_compose_declares_week3_services_and_topics():
    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    for name in (
        "redpanda",
        "redpanda-init",
        "redis",
        "postgres",
        "simulator",
        "stream-processor",
        "prediction-api",
    ):
        assert name in services

    assert "infra/redpanda/init-topics.sh" in services["redpanda-init"]["command"]

    init_script = Path("infra/redpanda/init-topics.sh").read_text(encoding="utf-8")
    assert "raw_events" in init_script
    assert "late_events" in init_script
    assert "--partitions 3" in init_script
    assert "--replicas 1" in init_script
    assert "rpk cluster info" in init_script
    assert "rpk topic list" in init_script


def test_env_example_contains_week3_runtime_settings():
    content = Path(".env.example").read_text(encoding="utf-8")

    assert "API_KEY=local-dev-api-key" in content
    assert "SESSION_TTL_SECONDS=1800" in content
    assert "LATE_EVENT_THRESHOLD_SECONDS=60" in content
    assert "POSTGRES_DSN=postgresql://mlops:mlops@postgres:5432/mlops" in content
```

- [ ] **Step 2: Add PyYAML dev dependency**

Update `pyproject.toml`:

```toml
dev = [
    "pytest==9.0.3",
    "pytest-cov",
    "ruff",
    "mypy",
    "pre-commit",
    "PyYAML>=6.0.0",
]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest services/tests/test_compose_contract.py -q`

Expected: fail because Week 3 services are not in compose yet.

- [ ] **Step 4: Add topic init script**

Create `infra/redpanda/init-topics.sh`:

```sh
#!/bin/sh
set -eu

BROKERS="${KAFKA_BROKER:-redpanda:9092}"

for attempt in $(seq 1 30); do
  if rpk cluster info --brokers "$BROKERS" >/dev/null 2>&1; then
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    echo "Redpanda broker did not become ready" >&2
    exit 1
  fi
  sleep 2
done

rpk topic create raw_events --brokers "$BROKERS" --partitions 3 --replicas 1 || true
rpk topic create late_events --brokers "$BROKERS" --partitions 3 --replicas 1 || true

rpk topic list --brokers "$BROKERS" | grep -q '^raw_events'
rpk topic list --brokers "$BROKERS" | grep -q '^late_events'
```

- [ ] **Step 5: Extend compose**

In `docker-compose.yml`, keep existing `minio`, `minio-init`, and `mlflow` unchanged. Add these services under `services:`:

```yaml
  redpanda:
    image: redpandadata/redpanda:latest
    container_name: redpanda
    command:
      - redpanda
      - start
      - --overprovisioned
      - --smp
      - "1"
      - --memory
      - 512M
      - --reserve-memory
      - 0M
      - --node-id
      - "0"
      - --check=false
      - --kafka-addr
      - PLAINTEXT://0.0.0.0:9092,OUTSIDE://0.0.0.0:19092
      - --advertise-kafka-addr
      - PLAINTEXT://redpanda:9092,OUTSIDE://localhost:19092
    ports:
      - "19092:19092"
    networks:
      - mlops_net

  redpanda-init:
    image: redpandadata/redpanda:latest
    container_name: redpanda-init
    depends_on:
      - redpanda
    command: sh /infra/redpanda/init-topics.sh
    environment:
      KAFKA_BROKER: redpanda:9092
    volumes:
      - ./infra/redpanda/init-topics.sh:/infra/redpanda/init-topics.sh:ro
    restart: "no"
    networks:
      - mlops_net

  redis:
    image: redis:7-alpine
    container_name: redis
    ports:
      - "6379:6379"
    networks:
      - mlops_net

  postgres:
    image: postgres:16-alpine
    container_name: postgres
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-mlops}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlops}
      POSTGRES_DB: ${POSTGRES_DB:-mlops}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infra/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - mlops_net

  simulator:
    build:
      context: .
      dockerfile: services/simulator/Dockerfile
    image: iuh-final-simulator:local
    profiles: ["week3"]
    depends_on:
      - redpanda-init
    environment:
      KAFKA_BROKER: redpanda:9092
      RAW_EVENTS_TOPIC: raw_events
      SIMULATION_RAW_DATA_PATH: /app/data/simulation_raw/2019-Nov.csv.gz
      REPLAY_LIMIT: ${REPLAY_LIMIT:-1000}
    volumes:
      - ./data:/app/data:ro
    networks:
      - mlops_net

  stream-processor:
    build:
      context: .
      dockerfile: services/stream_processor/Dockerfile
    image: iuh-final-stream-processor:local
    depends_on:
      - redpanda-init
      - redis
      - postgres
    environment:
      KAFKA_BROKER: redpanda:9092
      RAW_EVENTS_TOPIC: raw_events
      LATE_EVENTS_TOPIC: late_events
      REDIS_URL: redis://redis:6379/0
      POSTGRES_DSN: postgresql://mlops:mlops@postgres:5432/mlops
      SESSION_TTL_SECONDS: ${SESSION_TTL_SECONDS:-1800}
      LATE_EVENT_THRESHOLD_SECONDS: ${LATE_EVENT_THRESHOLD_SECONDS:-60}
    networks:
      - mlops_net

  prediction-api:
    build:
      context: .
      dockerfile: services/prediction_api/Dockerfile
    image: iuh-final-prediction-api:local
    depends_on:
      - redis
      - mlflow
    environment:
      API_KEY: ${API_KEY:-local-dev-api-key}
      REDIS_URL: redis://redis:6379/0
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_SERVING_BUNDLE_URI: ${MLFLOW_SERVING_BUNDLE_URI:-runs:/replace-with-week3-smoke-run}
    ports:
      - "8080:8080"
    networks:
      - mlops_net
```

Add `postgres_data:` under `volumes:`.

- [ ] **Step 6: Extend `.env.example`**

Append:

```text
# ============================================================================
# Week 3 Online Services
# ============================================================================
KAFKA_BROKER=redpanda:9092
RAW_EVENTS_TOPIC=raw_events
LATE_EVENTS_TOPIC=late_events
REDIS_URL=redis://redis:6379/0
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops
POSTGRES_DB=mlops
POSTGRES_DSN=postgresql://mlops:mlops@postgres:5432/mlops
SESSION_TTL_SECONDS=1800
LATE_EVENT_THRESHOLD_SECONDS=60
REPLAY_LIMIT=1000
API_KEY=local-dev-api-key
MLFLOW_SERVING_BUNDLE_URI=runs:/replace-with-week3-smoke-run
```

- [ ] **Step 7: Run static checks**

Run: `pytest services/tests/test_compose_contract.py -q`

Expected: pass.

Run: `docker compose config >/tmp/iuh-week3-compose.yml`

Expected: command exits `0`.

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml infra/redpanda/init-topics.sh .env.example pyproject.toml services/tests/test_compose_contract.py
git commit -m "feat: add week3 compose services"
```

### Task 11: Compose Smoke Script

**Files:**
- Create: `scripts/week3_compose_smoke.py`
- Test: `services/tests/test_week3_smoke_script.py`

- [ ] **Step 1: Write the failing script test**

Create `services/tests/test_week3_smoke_script.py`:

```python
"""Tests for the Week 3 compose smoke helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.week3_compose_smoke import build_ci_fixture, require_real_serving_bundle_uri


def test_build_ci_fixture_contains_required_event_shapes(tmp_path):
    fixture = tmp_path / "2019-Nov.csv.gz"

    build_ci_fixture(fixture)

    content = fixture.read_bytes()
    assert content
    import pandas as pd

    frame = pd.read_csv(fixture)
    assert set(frame["user_session"]) == {"ci-session-1", "ci-session-2"}
    assert {"view", "cart", "remove_from_cart", "purchase"}.issubset(set(frame["event_type"]))
    assert frame["category_code"].isna().any() or frame["brand"].isna().any()


def test_require_real_serving_bundle_uri_rejects_missing_or_placeholder(monkeypatch):
    monkeypatch.delenv("MLFLOW_SERVING_BUNDLE_URI", raising=False)
    with pytest.raises(RuntimeError, match="MLFLOW_SERVING_BUNDLE_URI"):
        require_real_serving_bundle_uri()

    monkeypatch.setenv("MLFLOW_SERVING_BUNDLE_URI", "runs:/replace-with-week3-smoke-run")
    with pytest.raises(RuntimeError, match="MLFLOW_SERVING_BUNDLE_URI"):
        require_real_serving_bundle_uri()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest services/tests/test_week3_smoke_script.py -q`

Expected: fail with `ModuleNotFoundError` for `scripts.week3_compose_smoke`.

- [ ] **Step 3: Write smoke helper**

Create `scripts/week3_compose_smoke.py`:

```python
"""Small Week 3 compose smoke helper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import requests


def build_ci_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "event_time": "2019-11-01 00:00:00 UTC",
            "event_type": "view",
            "product_id": "p1",
            "category_id": "cat-1",
            "category_code": "cat.one",
            "brand": "brand-a",
            "price": 10.0,
            "user_id": "u1",
            "user_session": "ci-session-1",
        },
        {
            "event_time": "2019-11-01 00:01:00 UTC",
            "event_type": "cart",
            "product_id": "p1",
            "category_id": "cat-1",
            "category_code": None,
            "brand": None,
            "price": 10.0,
            "user_id": "u1",
            "user_session": "ci-session-1",
        },
        {
            "event_time": "2019-11-01 00:02:00 UTC",
            "event_type": "remove_from_cart",
            "product_id": "p1",
            "category_id": "cat-1",
            "category_code": "cat.one",
            "brand": "brand-a",
            "price": 10.0,
            "user_id": "u1",
            "user_session": "ci-session-1",
        },
        {
            "event_time": "2019-11-01 00:10:00 UTC",
            "event_type": "purchase",
            "product_id": "p2",
            "category_id": "cat-2",
            "category_code": "cat.two",
            "brand": "brand-b",
            "price": 20.0,
            "user_id": "u2",
            "user_session": "ci-session-2",
        },
        {
            "event_time": "2019-11-01 00:06:00 UTC",
            "event_type": "view",
            "product_id": "p3",
            "category_id": "cat-3",
            "category_code": "cat.three",
            "brand": "brand-c",
            "price": 30.0,
            "user_id": "u2",
            "user_session": "ci-session-2",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")


PLACEHOLDER_BUNDLE_URI = "runs:/replace-with-week3-smoke-run"


def require_real_serving_bundle_uri() -> None:
    uri = os.environ.get("MLFLOW_SERVING_BUNDLE_URI", "")
    if not uri or uri == PLACEHOLDER_BUNDLE_URI:
        raise RuntimeError("Full smoke requires MLFLOW_SERVING_BUNDLE_URI from a real smoke training run")


def run_command(command: list[str], *, input_text: str | None = None) -> None:
    subprocess.run(command, check=True, input=input_text, text=input_text is not None)


def inject_late_event() -> None:
    late_event = {
        "event_id": "ci-late-event-1",
        "user_session": "ci-session-1",
        "source_event_time": "2019-10-31T23:55:00",
        "event_type": "view",
        "product_id": "late-product",
        "category_id": "cat-late",
        "category_code": "cat.late",
        "brand": "brand-late",
        "price": 1.0,
        "user_id": "u1",
        "replay_time": "2026-05-15T09:00:00",
        "source": "replay",
    }
    run_command(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "redpanda",
            "rpk",
            "topic",
            "produce",
            "raw_events",
            "--brokers",
            "redpanda:9092",
            "--key",
            "ci-session-1",
        ],
        input_text=json.dumps(late_event) + "\n",
    )


def verify_late_event_routed() -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "redpanda",
            "rpk",
            "topic",
            "consume",
            "late_events",
            "--brokers",
            "redpanda:9092",
            "--num",
            "1",
            "--timeout",
            "10s",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if "ci-late-event-1" not in result.stdout:
        raise AssertionError(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="data/simulation_raw/2019-Nov.csv.gz")
    parser.add_argument("--api-key", default="local-dev-api-key")
    args = parser.parse_args()
    require_real_serving_bundle_uri()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        build_ci_fixture(fixture_path)

    run_command(["docker", "compose", "up", "-d", "--build", "redpanda", "redpanda-init", "redis", "postgres", "mlflow", "prediction-api", "stream-processor"])
    run_command(["docker", "compose", "run", "--rm", "simulator", "python", "-m", "services.simulator.app", "--limit", "5"])
    inject_late_event()
    verify_late_event_routed()
    response = requests.get(
        "http://localhost:8080/api/v1/predict/ci-session-1",
        headers={"X-API-Key": args.api_key},
        timeout=10,
    )
    response.raise_for_status()
    body = response.json()
    if body["prediction_mode"] != "model" or body["fallback_reason"] is not None:
        raise AssertionError(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add smoke dependencies**

Add `requests>=2.31.0` to the `dev` optional dependency list in `pyproject.toml`.

- [ ] **Step 5: Run script tests**

Run: `pytest services/tests/test_week3_smoke_script.py -q`

Expected: pass.

The smoke helper intentionally injects one late event directly to `raw_events` after normal replay has created accepted Redis state for `ci-session-1`. Do not depend on out-of-order fixture CSV input to prove late routing, because `iter_replay_events()` sorts events by `user_session` and `source_event_time`.

- [ ] **Step 6: Commit**

```bash
git add scripts/week3_compose_smoke.py services/tests/test_week3_smoke_script.py pyproject.toml
git commit -m "test: add week3 compose smoke helper"
```

### Task 12: CI Service Checks

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Update CI install step for service dependencies**

Modify `.github/workflows/ci.yml` after `Install project`:

```yaml
      - name: Install Week 3 service dependencies
        run: |
          python -m pip install -r services/simulator/requirements.txt
          python -m pip install -r services/stream_processor/requirements.txt
          python -m pip install -r services/prediction_api/requirements.txt
```

- [ ] **Step 2: Update test command**

Replace the test command:

```yaml
      - name: Run tests with coverage
        run: pytest training/tests services/tests -q --cov=training/src --cov=shared --cov=services --cov-report=term-missing
```

- [ ] **Step 3: Add compose config validation**

Add after tests:

```yaml
      - name: Validate Docker Compose graph
        run: docker compose config >/tmp/iuh-week3-compose.yml
```

- [ ] **Step 4: Run local CI-equivalent checks**

Run:

```bash
ruff check shared services training/src training/tests services/tests scripts/week3_compose_smoke.py
pytest training/tests services/tests -q
docker compose config >/tmp/iuh-week3-compose.yml
dvc dag
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: include week3 service checks"
```

### Task 13: README And Blueprint Sync

**Files:**
- Modify: `README.md`
- Modify: `docs/BLUEPRINT/01_OVERVIEW.md`
- Modify: `docs/BLUEPRINT/02_ARCHITECTURE.md`
- Modify: `docs/BLUEPRINT/04_PIPELINES.md`
- Modify: `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`
- Modify: `docs/BLUEPRINT/07_TESTING.md`

- [ ] **Step 1: Update README Week 3 section**

Add after the Week 2 training command in `README.md`:

```markdown
## Week 3 Online Replay Slice

Week 3 adds a local online path on top of the existing offline pipeline:

- `services/simulator` reads `data/simulation_raw/2019-Nov.csv.gz`, normalizes raw rows, and publishes bounded replay events to `raw_events`.
- `services/stream_processor` consumes `raw_events`, suppresses duplicate `event_id` values with Redis TTL keys, routes late events to `late_events`, updates Redis session state, invalidates prediction cache keys, and appends accepted replay events to PostgreSQL.
- `services/prediction_api` serves `GET /health` and authenticated `GET /api/v1/predict/{user_session}` using the MLflow serving bundle logged by a smoke training run.

Run the local online stack:

```bash
docker compose up -d --build redpanda redpanda-init redis postgres minio minio-init mlflow stream-processor prediction-api
```

Create a serving bundle with a smoke training run:

```bash
python -m training.src.train \
  --train data/gold/train.parquet \
  --val data/gold/val.parquet \
  --test data/gold/test.parquet \
  --session-split-map data/gold/session_split_map.parquet \
  --smoke-mode \
  --device cpu
```

Set `MLFLOW_SERVING_BUNDLE_URI` to the winner `{model}_test_evaluation` run URI. The placeholder `runs:/replace-with-week3-smoke-run` is only a compose default and must fail full smoke. Then replay a bounded batch:

```bash
docker compose run --rm simulator python -m services.simulator.app --limit 1000
```

Call the API:

```bash
curl -H "X-API-Key: ${API_KEY:-local-dev-api-key}" \
  http://localhost:8080/api/v1/predict/<user_session>
```

Run the Week 3 smoke helper:

```bash
python scripts/week3_compose_smoke.py
```
```

- [ ] **Step 2: Sync blueprint overview**

In `docs/BLUEPRINT/01_OVERVIEW.md`, add a concise Week 3 current-state note:

```markdown
### Week 3 Online Replay Slice

The implemented online slice uses Redpanda for `raw_events` and `late_events`, Redis for canonical session feature state, PostgreSQL `replay_events` as the accepted replay append log, and FastAPI for authenticated prediction. Explainability, hot reload, monitoring dashboards, and retraining orchestration remain outside the Week 3 slice.
```

- [ ] **Step 3: Sync architecture and pipeline docs**

In `docs/BLUEPRINT/02_ARCHITECTURE.md` and `docs/BLUEPRINT/04_PIPELINES.md`, add the same concrete flow:

```markdown
`data/simulation_raw/2019-Nov.csv.gz` -> simulator -> `raw_events` -> stream processor -> Redis `session:{user_session}` state + PostgreSQL `replay_events` append log -> prediction API.

Late events are routed to `late_events` and do not update Redis or PostgreSQL. Duplicate events are suppressed by Redis keys named `dedup:event:{event_id}` with a TTL at least as long as the session-state TTL.
```

- [ ] **Step 4: Sync project structure and testing docs**

In `docs/BLUEPRINT/05_PROJECT_STRUCTURE.md`, add:

```markdown
services/
  simulator/          Bounded November replay publisher
  stream_processor/   Quix Streams consumer, Redis state updates, PostgreSQL append
  prediction_api/     Authenticated model-backed prediction endpoint
infra/
  redpanda/           Local topic initialization
  postgres/           Replay append-log schema
```

In `docs/BLUEPRINT/07_TESTING.md`, add:

```markdown
Week 3 checks include service unit tests under `services/tests`, static compose validation with `docker compose config`, and the optional `scripts/week3_compose_smoke.py` local smoke run. Existing Week 1 and Week 2 checks remain required: `ruff check .`, `pytest training/tests -q`, and `dvc dag`.
```

- [ ] **Step 5: Run docs and static checks**

Run:

```bash
pytest services/tests/test_compose_contract.py services/tests/test_week3_smoke_script.py -q
ruff check README.md docs/BLUEPRINT/01_OVERVIEW.md docs/BLUEPRINT/02_ARCHITECTURE.md docs/BLUEPRINT/04_PIPELINES.md docs/BLUEPRINT/05_PROJECT_STRUCTURE.md docs/BLUEPRINT/07_TESTING.md
```

Expected: pytest passes. Ruff may report that Markdown files are skipped; it must not report Python lint errors.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/BLUEPRINT/01_OVERVIEW.md docs/BLUEPRINT/02_ARCHITECTURE.md docs/BLUEPRINT/04_PIPELINES.md docs/BLUEPRINT/05_PROJECT_STRUCTURE.md docs/BLUEPRINT/07_TESTING.md
git commit -m "docs: document week3 online replay slice"
```

### Task 14: Final Verification

**Files:**
- No source edits unless a verification command exposes a defect.

- [ ] **Step 1: Run focused service tests**

Run:

```bash
pytest services/tests -q
```

Expected: all service tests pass.

- [ ] **Step 2: Run existing training tests**

Run:

```bash
pytest training/tests -q
```

Expected: all training tests pass.

- [ ] **Step 3: Run lint**

Run:

```bash
ruff check .
```

Expected: no lint violations.

- [ ] **Step 4: Validate compose config**

Run:

```bash
docker compose config >/tmp/iuh-week3-compose.yml
```

Expected: command exits `0`.

- [ ] **Step 5: Validate DVC graph**

Run:

```bash
dvc dag
```

Expected: the existing `bronze -> silver -> session_split -> gold -> train` graph renders successfully.

- [ ] **Step 6: Optional local smoke**

Run only when Docker is available and the local MLflow serving bundle URI has been set:

```bash
export API_KEY=local-dev-api-key
export MLFLOW_SERVING_BUNDLE_URI=runs:/<winner-test-evaluation-run-id>
python scripts/week3_compose_smoke.py
```

Expected: script exits `0` and `GET /api/v1/predict/ci-session-1` returns HTTP `200`.
The response must be model-backed with `prediction_mode="model"` and `fallback_reason=null`; `redis_miss` and `model_unavailable` are valid endpoint fallbacks but full-smoke failures.

- [ ] **Step 7: Commit verification fixes**

If verification required code changes, commit only those touched files:

```bash
git status --short
git add <changed-week3-files>
git commit -m "fix: stabilize week3 vertical slice"
```
