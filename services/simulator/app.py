"""CLI for bounded November replay into the raw_events topic."""

from __future__ import annotations

import argparse
import logging
import os

from quixstreams import Application

from services.simulator.replay import iter_replay_events, publish_events

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.getenv("SIMULATION_RAW_DATA_PATH", "data/simulation_raw/2019-Nov.csv.gz"),
    )
    parser.add_argument("--limit", type=int, default=int(os.getenv("REPLAY_LIMIT", "1000")))
    parser.add_argument("--broker", default=os.getenv("KAFKA_BROKER", "redpanda:9092"))
    parser.add_argument("--topic", default=os.getenv("RAW_EVENTS_TOPIC", "raw_events"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    app = Application(broker_address=args.broker)
    app.topic(args.topic, value_serializer="json", key_serializer="str")
    events = iter_replay_events(args.input, limit=args.limit)
    with app.get_producer() as producer:
        count = publish_events(events, producer=producer, topic=args.topic)
    logger.info("Published %d replay events to %s", count, args.topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
