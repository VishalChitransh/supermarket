"""Streamlit interface for online SuperKart sales predictions."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:7860")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="wide")
st.title("SuperKart | Sales Forecast")
st.caption("Estimate product revenue for a store using product and outlet characteristics.")

with st.form("prediction"):
    left, right = st.columns(2)
    with left:
        product_weight = st.number_input("Product weight", min_value=0.0, value=12.66, step=0.1)
        product_area = st.number_input("Allocated display area", min_value=0.0, value=0.027, step=0.001, format="%.3f")
        product_mrp = st.number_input("Maximum retail price", min_value=0.0, value=117.08, step=1.0)
        sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        product_prefix = st.selectbox("Product ID prefix", ["FD", "DR", "NC"])
        category = st.selectbox("Product category", ["Perishables", "Non Perishables"])
    with right:
        store_size = st.selectbox("Store size", ["Small", "Medium", "High"])
        city_tier = st.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
        store_type = st.selectbox("Store type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"])
        store_age = st.number_input("Store age (years)", min_value=0, value=16, step=1)
    submitted = st.form_submit_button("Estimate sales", type="primary")

if submitted:
    payload = {
        "Product_Weight": product_weight, "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": product_area, "Product_MRP": product_mrp,
        "Store_Size": store_size, "Store_Location_City_Type": city_tier,
        "Store_Type": store_type, "Product_Id_char": product_prefix,
        "Store_Age_Years": store_age, "Product_Type_Category": category,
    }
    try:
        response = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=15)
        response.raise_for_status()
        st.metric("Predicted product-store sales", f"{response.json()['predicted_sales']:,.2f}")
    except requests.RequestException as error:
        st.error(f"Could not reach the prediction API at {API_URL}: {error}")

st.divider()
st.subheader("Batch forecast")
batch_file = st.file_uploader("Upload a CSV with the 10 model feature columns", type=["csv"])
if batch_file is not None and st.button("Predict uploaded rows"):
    try:
        batch = pd.read_csv(batch_file)
        response = requests.post(
            f"{API_URL}/v1/predictbatch",
            files={"file": (batch_file.name, batch_file.getvalue(), "text/csv")}, timeout=60,
        )
        response.raise_for_status()
        batch["Predicted_Product_Store_Sales_Total"] = response.json()["predictions"]
        st.dataframe(batch, use_container_width=True)
        st.download_button("Download predictions", batch.to_csv(index=False).encode("utf-8"),
                           file_name="superkart_sales_predictions.csv", mime="text/csv")
    except (requests.RequestException, ValueError) as error:
        st.error(f"Batch prediction failed: {error}")
