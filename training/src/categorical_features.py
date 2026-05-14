"""Shared categorical preprocessing for Week 2 training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

NUMERIC_FEATURE_COLUMNS = [
    "total_views",
    "total_carts",
    "net_cart_count",
    "cart_to_view_ratio",
    "unique_categories",
    "unique_products",
    "session_duration_sec",
    "price",
]

CATEGORICAL_FEATURE_COLUMNS = ["category_id", "category_code", "brand", "event_type"]
TARGET_COLUMN = "label"
MISSING_CATEGORY_TOKEN = "__MISSING__"
UNKNOWN_CATEGORY_TOKEN = "__UNK__"


@dataclass(frozen=True)
class TrainingFrame:
    """Prepared training frame with explicit column groups."""

    features: pd.DataFrame
    target: pd.Series
    numeric_columns: list[str]
    categorical_columns: list[str]


@dataclass(frozen=True)
class CategoricalEncodingArtifacts:
    """Train-fitted categorical vocabularies."""

    category_maps: dict[str, dict[str, int]]
    missing_token: str = MISSING_CATEGORY_TOKEN
    unknown_token: str = UNKNOWN_CATEGORY_TOKEN


def prepare_training_frame(df: pd.DataFrame) -> TrainingFrame:
    """Select the model-ready columns from a gold dataframe."""
    required_columns = set(NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required training columns: {missing}")

    features = df[NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS].copy()
    features[NUMERIC_FEATURE_COLUMNS] = features[NUMERIC_FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    target = df[TARGET_COLUMN].astype(int).copy()
    return TrainingFrame(
        features=features,
        target=target,
        numeric_columns=list(NUMERIC_FEATURE_COLUMNS),
        categorical_columns=list(CATEGORICAL_FEATURE_COLUMNS),
    )


def fit_categorical_encoders(train_df: pd.DataFrame) -> CategoricalEncodingArtifacts:
    """Build deterministic vocabularies from the training split only."""
    category_maps: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_FEATURE_COLUMNS:
        normalized = train_df[column].fillna(MISSING_CATEGORY_TOKEN).astype(str)
        ordered_values: list[str] = []
        seen: set[str] = set()
        for value in normalized.tolist():
            if value in {MISSING_CATEGORY_TOKEN, UNKNOWN_CATEGORY_TOKEN}:
                continue
            if value not in seen:
                seen.add(value)
                ordered_values.append(value)

        category_map = {
            MISSING_CATEGORY_TOKEN: 0,
            UNKNOWN_CATEGORY_TOKEN: 1,
        }
        category_map.update({value: index + 2 for index, value in enumerate(ordered_values)})
        category_maps[column] = category_map

    return CategoricalEncodingArtifacts(category_maps=category_maps)


def transform_with_categorical_contract(
    df: pd.DataFrame,
    artifacts: CategoricalEncodingArtifacts,
) -> pd.DataFrame:
    """Normalize null/unseen categories and keep categorical columns typed."""
    transformed = df.copy()
    for column, mapping in artifacts.category_maps.items():
        normalized = transformed[column].fillna(artifacts.missing_token).astype(str)
        normalized = normalized.where(normalized.isin(mapping), artifacts.unknown_token)
        transformed[column] = pd.Categorical(
            normalized,
            categories=list(mapping.keys()),
            ordered=False,
        )
    return transformed
