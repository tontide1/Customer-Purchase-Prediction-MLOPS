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
