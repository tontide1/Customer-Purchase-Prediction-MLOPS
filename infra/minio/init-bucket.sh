#!/bin/sh
set -e

MINIO_ALIAS="minio"

echo "Waiting for MinIO at ${MINIO_ENDPOINT} ..."
until mc alias set "${MINIO_ALIAS}" "${MINIO_ENDPOINT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"; do
  sleep 2
done

echo "Ensuring bucket '${MINIO_BUCKET}' exists ..."
mc mb --ignore-existing "${MINIO_ALIAS}/${MINIO_BUCKET}"

echo "Applying private policy to '${MINIO_BUCKET}' ..."
mc anonymous set none "${MINIO_ALIAS}/${MINIO_BUCKET}"

echo "MinIO bucket initialization complete."
