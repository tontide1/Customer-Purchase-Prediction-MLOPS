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
    assert services["redpanda-init"]["entrypoint"] == ["/bin/sh", "-c"]

    init_script = Path("infra/redpanda/init-topics.sh").read_text(encoding="utf-8")
    assert "raw_events" in init_script
    assert "late_events" in init_script
    assert "--partitions 3" in init_script
    assert "--replicas 1" in init_script
    assert "rpk cluster info" in init_script
    assert "rpk topic list" in init_script
    assert "8080:8000" in services["prediction-api"]["ports"]
    assert "--allowed-hosts" in services["mlflow"]["command"]
    assert "mlflow,mlflow:5000,localhost,127.0.0.1" in services["mlflow"]["command"]


def test_env_example_contains_week3_runtime_settings():
    content = Path(".env.example").read_text(encoding="utf-8")

    assert "API_KEY=local-dev-api-key" in content
    assert "SESSION_TTL_SECONDS=1800" in content
    assert "LATE_EVENT_THRESHOLD_SECONDS=60" in content
    assert "POSTGRES_DSN=postgresql://mlops:mlops@postgres:5432/mlops" in content
