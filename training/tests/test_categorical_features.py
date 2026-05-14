"""Tests for categorical preprocessing helpers."""

from __future__ import annotations

import pandas as pd

from training.src.categorical_features import (
    CATEGORICAL_FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    fit_categorical_encoders,
    prepare_training_frame,
    transform_with_categorical_contract,
)


def _build_gold_like_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_views": [1, 2, 3, 4],
            "total_carts": [0, 1, 1, 2],
            "net_cart_count": [0, 1, 1, 2],
            "cart_to_view_ratio": [0.0, 0.5, 0.33, 0.5],
            "unique_categories": [1, 2, 2, 3],
            "unique_products": [1, 2, 3, 4],
            "session_duration_sec": [10.0, 20.0, 30.0, 40.0],
            "price": [35.0, 50.0, 12.0, 80.0],
            "category_id": ["cat-1", "cat-2", "cat-3", "cat-4"],
            "category_code": [
                "appliances.environment.vacuum",
                "electronics.smartphone",
                "furniture.living_room.sofa",
                "sports.trainers",
            ],
            "brand": ["brand-a", "brand-b", "brand-c", None],
            "label": [0, 1, 0, 1],
        }
    )


def test_prepare_training_frame_keeps_categorical_columns():
    frame = prepare_training_frame(_build_gold_like_frame())

    assert frame.numeric_columns == NUMERIC_FEATURE_COLUMNS
    assert frame.categorical_columns == CATEGORICAL_FEATURE_COLUMNS
    assert frame.features.columns.tolist() == NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    assert frame.target.tolist() == [0, 1, 0, 1]


def test_transform_with_categorical_contract_handles_null_and_unseen_categories():
    train_df = _build_gold_like_frame()
    val_df = _build_gold_like_frame()
    val_df.loc[0, "brand"] = "brand-new"
    val_df.loc[1, "category_code"] = None

    train_frame = prepare_training_frame(train_df)
    val_frame = prepare_training_frame(val_df)
    artifacts = fit_categorical_encoders(train_frame.features)
    transformed = transform_with_categorical_contract(val_frame.features, artifacts)

    assert transformed.isnull().sum().sum() == 0
    assert str(transformed["category_id"].dtype) == "category"
    assert str(transformed["category_code"].dtype) == "category"
    assert str(transformed["brand"].dtype) == "category"
    assert transformed.loc[0, "brand"] == "__UNK__"
    assert transformed.loc[1, "category_code"] == "__MISSING__"
