import pandas as pd
import pytest

from superkart.features import MODEL_FEATURE_COLUMNS, engineer_features, validate_model_features


def test_engineered_features_match_batch_schema():
    raw = pd.DataFrame([{
        "Product_Id": "FD6114", "Product_Weight": 12.66, "Product_Sugar_Content": "reg",
        "Product_Allocated_Area": 0.027, "Product_Type": "Frozen Foods", "Product_MRP": 117.08,
        "Store_Id": "OUT004", "Store_Establishment_Year": 2009, "Store_Size": "Medium",
        "Store_Location_City_Type": "Tier 2", "Store_Type": "Supermarket Type2",
    }])
    features = engineer_features(raw)
    assert list(features.columns) == MODEL_FEATURE_COLUMNS
    assert features.loc[0, "Product_Id_char"] == "FD"
    assert features.loc[0, "Store_Age_Years"] == 16
    assert features.loc[0, "Product_Type_Category"] == "Non Perishables"
    assert features.loc[0, "Product_Sugar_Content"] == "Regular"


def test_feature_validation_reports_missing_columns():
    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_model_features(pd.DataFrame([{"Product_MRP": 100}]))
