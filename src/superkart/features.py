"""Feature engineering shared by model training and online inference."""

from __future__ import annotations

import pandas as pd

REFERENCE_YEAR = 2025
TARGET_COLUMN = "Product_Store_Sales_Total"
RAW_FEATURE_COLUMNS = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area",
    "Product_Type", "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
    "Store_Location_City_Type", "Store_Type",
]
MODEL_FEATURE_COLUMNS = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
    "Store_Age_Years", "Product_Type_Category",
]
NUMERIC_FEATURES = [
    "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"
]
CATEGORICAL_FEATURES = [
    "Product_Sugar_Content", "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Product_Id_char", "Product_Type_Category",
]
PERISHABLE_TYPES = {
    "Dairy", "Meat", "Fruits and Vegetables", "Breads", "Breakfast", "Seafood"
}


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw source columns into the fixed model feature schema."""
    missing = sorted(set(RAW_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {', '.join(missing)}")
    result = data.copy()
    result["Product_Id_char"] = result["Product_Id"].astype("string").str.extract(
        r"^([A-Za-z]{2})", expand=False
    )
    result["Store_Age_Years"] = REFERENCE_YEAR - pd.to_numeric(
        result["Store_Establishment_Year"], errors="coerce"
    )
    sugar = result["Product_Sugar_Content"].astype("string").str.strip().str.lower()
    result["Product_Sugar_Content"] = sugar.replace({
        "reg": "Regular", "regular": "Regular", "low sugar": "Low Sugar", "no sugar": "No Sugar"
    })
    product_types = result["Product_Type"].astype("string").str.strip()
    result["Product_Type_Category"] = product_types.map(
        lambda value: "Perishables" if value in PERISHABLE_TYPES else "Non Perishables"
        if pd.notna(value) else pd.NA
    )
    return result[MODEL_FEATURE_COLUMNS].copy()


def validate_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and order the features accepted by the inference API."""
    missing = sorted(set(MODEL_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    result = data[MODEL_FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    sugar = result["Product_Sugar_Content"].astype("string").str.strip().str.lower()
    result["Product_Sugar_Content"] = sugar.replace({
        "reg": "Regular", "regular": "Regular", "low sugar": "Low Sugar", "no sugar": "No Sugar"
    })
    return result
