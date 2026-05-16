"""Tests for the Week 3 compose smoke helper."""

from __future__ import annotations

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
