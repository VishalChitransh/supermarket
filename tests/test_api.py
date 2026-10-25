import io

import pandas as pd

from superkart.api import create_app
from superkart.features import MODEL_FEATURE_COLUMNS


class ConstantModel:
    def predict(self, features):
        return [321.5] * len(features)


def sample_features():
    return {
        "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar",
        "Product_Allocated_Area": 0.027, "Product_MRP": 117.08, "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2",
        "Product_Id_char": "FD", "Store_Age_Years": 16,
        "Product_Type_Category": "Non Perishables",
    }


def test_online_prediction_and_health():
    client = create_app(ConstantModel()).test_client()
    assert client.get("/healthz").json == {"status": "ok"}
    response = client.post("/v1/predict", json=sample_features())
    assert response.status_code == 200
    assert response.json == {"predicted_sales": 321.5}


def test_batch_prediction_and_missing_feature_error():
    client = create_app(ConstantModel()).test_client()
    rows = pd.DataFrame([sample_features(), sample_features()], columns=MODEL_FEATURE_COLUMNS)
    upload = io.BytesIO(rows.to_csv(index=False).encode("utf-8"))
    response = client.post(
        "/v1/predictbatch", data={"file": (upload, "batch.csv")}, content_type="multipart/form-data"
    )
    bad_response = client.post("/v1/predict", json={"Product_MRP": 100})
    assert response.status_code == 200
    assert response.json == {"predictions": [321.5, 321.5]}
    assert bad_response.status_code == 400
