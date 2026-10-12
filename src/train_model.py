"""Train and serialize the OLS insurance-cost prediction pipeline."""

from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import ClinicalFeatureEngineer, get_preprocessor, load_data, split_data


class InsurancePipeline(Pipeline):
    """Named Pipeline subclass retained for serialized-model compatibility."""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "insurance.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "insurance_model.joblib"


def build_pipeline() -> InsurancePipeline:
    """Construct the complete preprocessing and OLS regression pipeline."""
    return InsurancePipeline(
        steps=[
            ("feature_engineer", ClinicalFeatureEngineer()),
            ("preprocessor", get_preprocessor(include_engineered=True)),
            ("regressor", LinearRegression()),
        ]
    )


def train_and_save(
    data_path: str | Path = DEFAULT_DATA_PATH, model_path: str | Path = DEFAULT_MODEL_PATH
) -> InsurancePipeline:
    """Fit the model from the CSV and write a portable joblib artifact."""
    data = load_data(data_path)
    X_train, _, y_train, _ = split_data(data)
    model = build_pipeline().fit(X_train, y_train)

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


if __name__ == "__main__":
    # ``python -m src.train_model`` executes this file as ``__main__``.  Import
    # the canonical module before serializing so joblib records
    # ``src.train_model.InsurancePipeline`` rather than ``__main__``.
    from src.train_model import train_and_save as canonical_train_and_save

    canonical_train_and_save()
    print(f"Model saved to {DEFAULT_MODEL_PATH}")
