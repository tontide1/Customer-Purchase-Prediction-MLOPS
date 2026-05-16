"""FastAPI prediction API for the Week 3 serving bundle."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Path, status
from pydantic import BaseModel

from services.prediction_api.bundle import ServingBundle, load_serving_bundle
from services.prediction_api.features import build_feature_row

USER_SESSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


@dataclass(frozen=True)
class PredictionAPISettings:
    api_key: str
    redis_url: str
    mlflow_bundle_uri: str

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "PredictionAPISettings":
        source = os.environ if env is None else env
        bundle_uri = source.get("MLFLOW_SERVING_BUNDLE_URI", "")
        if not bundle_uri:
            bundle_uri = source.get("MLFLOW_BUNDLE_URI", "")
        return cls(
            api_key=source.get("API_KEY", ""),
            redis_url=source.get("REDIS_URL", "redis://redis:6379/0"),
            mlflow_bundle_uri=bundle_uri,
        )


class PredictionResponse(BaseModel):
    purchase_probability: float
    prediction_time: str
    prediction_horizon_minutes: int | None = None
    model_uri: str | None = None
    model_version: str | None = None
    prediction_mode: str
    fallback_reason: str | None = None
    cached: bool


def _default_redis_client_factory(redis_url: str):
    import redis

    return redis.Redis.from_url(redis_url, decode_responses=True)


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_user_session(user_session: str) -> str:
    import re

    if not re.fullmatch(USER_SESSION_PATTERN, user_session):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid user_session",
        )
    return user_session


def validate_api_key(expected_api_key: str, provided_api_key: str | None) -> None:
    if not expected_api_key or provided_api_key != expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "API-Key"},
        )


def _build_response(
    *,
    purchase_probability: float,
    prediction_mode: str,
    fallback_reason: str | None = None,
    bundle: ServingBundle | None = None,
) -> PredictionResponse:
    return PredictionResponse(
        purchase_probability=purchase_probability,
        prediction_time=_utc_now_iso(),
        prediction_horizon_minutes=(
            None if bundle is None else bundle.prediction_horizon_minutes
        ),
        model_uri=None if bundle is None else bundle.model_uri,
        model_version=None if bundle is None else bundle.model_version,
        prediction_mode=prediction_mode,
        fallback_reason=fallback_reason,
        cached=False,
    )


def _fallback_response(
    reason: str,
    *,
    bundle: ServingBundle | None = None,
) -> PredictionResponse:
    return _build_response(
        purchase_probability=0.5,
        prediction_mode="fallback",
        fallback_reason=reason,
        bundle=bundle,
    )


def _api_key_dependency(expected_api_key: str):
    def require_api_key(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        validate_api_key(expected_api_key, x_api_key)

    return require_api_key


def build_prediction_response(
    *,
    user_session: str,
    redis_client: Any,
    bundle_loader: Callable[[], ServingBundle],
) -> PredictionResponse:
    validate_user_session(user_session)

    hash_key = f"session:{user_session}"
    if not redis_client.hgetall(hash_key):
        return _fallback_response("redis_miss")

    try:
        bundle = bundle_loader()
    except Exception:
        return _fallback_response("model_unavailable")

    feature_row = build_feature_row(redis_client, user_session, bundle)
    if feature_row is None:
        return _fallback_response("redis_miss")

    try:
        probabilities = bundle.model.predict_proba(feature_row)
        purchase_probability = float(probabilities[:, 1][0])
    except Exception:
        return _fallback_response("model_unavailable", bundle=bundle)

    return _build_response(
        purchase_probability=purchase_probability,
        prediction_mode="model",
        bundle=bundle,
    )


def create_app(
    settings: PredictionAPISettings | None = None,
    *,
    redis_client_factory: Callable[[str], Any] | None = None,
    bundle_loader: Callable[[str], ServingBundle] | None = None,
) -> FastAPI:
    resolved_settings = settings or PredictionAPISettings.from_env()
    redis_factory = redis_client_factory or _default_redis_client_factory
    serving_bundle_loader = bundle_loader or load_serving_bundle

    app = FastAPI(title="Prediction API")

    @lru_cache(maxsize=1)
    def get_redis_client():
        return redis_factory(resolved_settings.redis_url)

    @lru_cache(maxsize=1)
    def get_serving_bundle():
        return serving_bundle_loader(resolved_settings.mlflow_bundle_uri)

    require_api_key = _api_key_dependency(resolved_settings.api_key)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/predict/{user_session}", response_model=PredictionResponse)
    def predict(
        user_session: str = Path(...),
        _: None = Depends(require_api_key),
    ) -> PredictionResponse:
        redis_client = get_redis_client()
        return build_prediction_response(
            user_session=user_session,
            redis_client=redis_client,
            bundle_loader=get_serving_bundle,
        )

    return app


app = create_app()
