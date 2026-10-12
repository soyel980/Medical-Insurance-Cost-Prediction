"""Data preparation utilities used by the insurance prediction pipeline."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]
TARGET_COLUMN = "charges"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add the clinically motivated obese-smoker interaction feature.

    The class intentionally has no fitted state.  Keeping its module and class name
    stable also allows existing serialized models to be loaded safely.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureEngineer expects a pandas DataFrame.")

        missing = {"bmi", "smoker"}.difference(X.columns)
        if missing:
            raise ValueError(f"Missing columns required for feature engineering: {sorted(missing)}")

        transformed = X.copy()
        transformed["obese_smoker"] = (
            (transformed["bmi"] >= 30.0)
            & (transformed["smoker"].astype(str).str.strip().str.lower() == "yes")
        ).astype(int)
        return transformed


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the raw insurance dataset."""
    data = pd.read_csv(csv_path)
    missing = set(ALL_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data[ALL_COLUMNS].copy()


def split_data(
    data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create reproducible raw-feature train/test partitions."""
    return train_test_split(
        data[FEATURE_COLUMNS], data[TARGET_COLUMN], test_size=test_size, random_state=random_state
    )


def get_preprocessor(include_engineered: bool = True) -> ColumnTransformer:
    """Create the pipeline's categorical encoder and numeric passthrough."""
    numeric_features = ["age", "bmi", "children"]
    if include_engineered:
        numeric_features.append("obese_smoker")

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["sex", "smoker", "region"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
