"""Command-line entry point for model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from superkart.modeling import train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and serialize the SuperKart sales model.")
    parser.add_argument("--data", default="data/SuperKart.csv", help="Path to training CSV")
    parser.add_argument("--output", default="backend_files/superkart_model.joblib", help="Artifact output path")
    parser.add_argument("--no-tune", action="store_true", help="Skip GridSearchCV tuning")
    arguments = parser.parse_args()
    artifact = train_and_save(arguments.data, arguments.output, tune=not arguments.no_tune)
    print(f"Selected model: {artifact['model_name']}")
    for name, scores in artifact["metrics"].items():
        print(f"{name}: RMSE={scores['RMSE']:.2f}, MAE={scores['MAE']:.2f}, R2={scores['R-squared']:.4f}")
    print(f"Saved artifact: {Path(arguments.output)}")


if __name__ == "__main__":
    main()
