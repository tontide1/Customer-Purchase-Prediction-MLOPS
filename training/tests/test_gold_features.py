"""Tests for gold feature helpers."""

from training.src.features import (
    compute_cart_to_view_ratio,
    normalize_category_value,
)


def test_compute_cart_to_view_ratio_returns_zero_without_views() -> None:
    assert compute_cart_to_view_ratio(0, 3) == 0.0


def test_compute_cart_to_view_ratio_divides_counts() -> None:
    assert compute_cart_to_view_ratio(4, 2) == 0.5


def test_normalize_category_value_uses_category_code_when_present() -> None:
    assert normalize_category_value("cat-1", "cat-2") == "cat-1"


def test_normalize_category_value_falls_back_to_category_id_when_code_missing() -> None:
    assert normalize_category_value(None, "cat-2") == "cat-2"
