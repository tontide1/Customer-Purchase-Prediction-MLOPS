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
