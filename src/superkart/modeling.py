"""Training, evaluation, and serialization for the sales regression models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from superkart.features import (
    CATEGORICAL_FEATURES, MODEL_FEATURE_COLUMNS, NUMERIC_FEATURES, RAW_FEATURE_COLUMNS,
    TARGET_COLUMN, engineer_features,
)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])


def regression_metrics(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    predictions = model.predict(features)
    count = len(target)
    r_squared = float(r2_score(target, predictions))
    feature_count = features.shape[1]
    adjusted_r_squared = (
        float(1 - (1 - r_squared) * (count - 1) / (count - feature_count - 1))
        if count > feature_count + 1 else float("nan")
    )
    return {
        "RMSE": float(np.sqrt(mean_squared_error(target, predictions))),
        "MAE": float(mean_absolute_error(target, predictions)),
        "R-squared": r_squared,
        "Adj. R-squared": adjusted_r_squared,
        "MAPE": float(mean_absolute_percentage_error(target, predictions)),
    }


def train_and_save(
    dataset_path: str | Path,
    artifact_path: str | Path,
    *,
    tune: bool = True,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare two ensemble regressors, tune Random Forest, and save the winner."""
    data = pd.read_csv(dataset_path)
    missing = sorted(set(RAW_FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    data = data.dropna(subset=[TARGET_COLUMN]).copy()
    features = engineer_features(data)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    valid = target.notna()
    features, target = features.loc[valid], target.loc[valid]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=random_state
    )
    models: dict[str, Any] = {
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("regressor", RandomForestRegressor(
                n_estimators=160, min_samples_leaf=2, random_state=random_state, n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=120, max_depth=2, learning_rate=0.06, random_state=random_state
            )),
        ]),
    }
    metrics: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        fitted[name] = model
        metrics[name] = regression_metrics(model, x_test, y_test)
    best_params: dict[str, Any] = {}
    if tune:
        search = GridSearchCV(
            Pipeline([
                ("preprocessor", make_preprocessor()),
                ("regressor", RandomForestRegressor(random_state=random_state, n_jobs=-1)),
            ]),
            {
                "regressor__n_estimators": [120, 200],
                "regressor__max_depth": [None, 16],
                "regressor__min_samples_leaf": [1, 2],
            },
            scoring="neg_root_mean_squared_error", cv=3, n_jobs=1, refit=True,
        )
        search.fit(x_train, y_train)
        fitted["Tuned Random Forest"] = search.best_estimator_
        metrics["Tuned Random Forest"] = regression_metrics(search.best_estimator_, x_test, y_test)
        best_params = search.best_params_
    winner_name = min(metrics, key=lambda name: metrics[name]["RMSE"])
    artifact = {
        "model": fitted[winner_name], "model_name": winner_name,
        "feature_columns": MODEL_FEATURE_COLUMNS, "metrics": metrics, "best_params": best_params,
    }
    output = Path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return artifact
