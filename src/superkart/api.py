"""Flask API for single-row and CSV batch sales predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from superkart.features import validate_model_features


def create_app(model: Any | None = None) -> Flask:
    app = Flask(__name__)
    if model is None:
        model_path = Path(os.getenv("MODEL_PATH", "backend_files/superkart_model.joblib"))
        model = joblib.load(model_path)["model"]

    @app.get("/healthz")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @app.post("/v1/predict")
    def predict() -> tuple[Any, int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        try:
            features = validate_model_features(pd.DataFrame([payload]))
            prediction = float(model.predict(features)[0])
        except (ValueError, TypeError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"predicted_sales": prediction}), 200

    @app.post("/v1/predictbatch")
    def predict_batch() -> tuple[Any, int]:
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "Upload a CSV using the 'file' form field."}), 400
        try:
            batch = pd.read_csv(uploaded)
            features = validate_model_features(batch)
            predictions = model.predict(features)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
            return jsonify({"error": str(error)}), 400
        return jsonify({"predictions": [float(value) for value in predictions]}), 200

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
